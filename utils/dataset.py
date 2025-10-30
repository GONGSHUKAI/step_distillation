from utils.lmdb import get_array_shape_from_lmdb, retrieve_row_from_lmdb
from torch.utils.data import Dataset
import numpy as np
import torch
import torchaudio
import lmdb
import json
from PIL import Image
import os
import torchvision.transforms.functional as TF
import pandas as pd
import cv2
import random
import math
from pathlib import Path
import decord
from torchvision.transforms.functional import resize
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

class OffsetDistributedSampler(DistributedSampler):
    def __init__(self, dataset, initial_step=0, gpu_num=4, **kwargs):
        super().__init__(dataset, **kwargs)
        if initial_step < len(dataset) // gpu_num:
            self.initial_step = initial_step
        else:
            self.initial_step = (
                (initial_step * gpu_num - len(dataset)) % len(dataset)
            ) // (gpu_num * gpu_num)
        self.first_time = True  # 标志位，表示是否是第一次加载

    def __iter__(self):
        # 获取原始索引
        indices = list(super().__iter__())

        # 如果是第一次加载，跳过前 initial_step 个索引
        if self.first_time and self.initial_step > 0:
            indices = indices[self.initial_step :]
            self.first_time = False  # 标志位设为 False，后续不再跳过

        return iter(indices)


class TextDataset(Dataset):
    def __init__(self, prompt_path, extended_prompt_path=None):
        with open(prompt_path, encoding="utf-8") as f:
            self.prompt_list = [line.rstrip() for line in f]

        if extended_prompt_path is not None:
            with open(extended_prompt_path, encoding="utf-8") as f:
                self.extended_prompt_list = [line.rstrip() for line in f]
            assert len(self.extended_prompt_list) == len(self.prompt_list)
        else:
            self.extended_prompt_list = None

    def __len__(self):
        return len(self.prompt_list)

    def __getitem__(self, idx):
        batch = {
            "prompts": self.prompt_list[idx],
            "idx": idx,
        }
        if self.extended_prompt_list is not None:
            batch["extended_prompts"] = self.extended_prompt_list[idx]
        return batch


class TextFolderDataset(Dataset):
    def __init__(self, data_path):
        self.texts = []
        for file in os.listdir(data_path):
            if file.endswith(".txt"):
                with open(os.path.join(data_path, file), "r") as f:
                    text = f.read().strip()
                    self.texts.append(text)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {"prompts": self.texts[idx], "idx": idx}


class ODERegressionLMDBDataset(Dataset):
    def __init__(self, data_path: str, max_pair: int = int(1e8)):
        self.env = lmdb.open(data_path, readonly=True,
                             lock=False, readahead=False, meminit=False)

        self.latents_shape = get_array_shape_from_lmdb(self.env, 'latents')
        self.max_pair = max_pair

    def __len__(self):
        return min(self.latents_shape[0], self.max_pair)

    def __getitem__(self, idx):
        """
        Outputs:
            - prompts: List of Strings
            - latents: Tensor of shape (num_denoising_steps, num_frames, num_channels, height, width). It is ordered from pure noise to clean image.
        """
        latents = retrieve_row_from_lmdb(
            self.env,
            "latents", np.float16, idx, shape=self.latents_shape[1:]
        )

        if len(latents.shape) == 4:
            latents = latents[None, ...]

        prompts = retrieve_row_from_lmdb(
            self.env,
            "prompts", str, idx
        )
        return {
            "prompts": prompts,
            "ode_latent": torch.tensor(latents, dtype=torch.float32)
        }


class ShardingLMDBDataset(Dataset):
    def __init__(self, data_path: str, max_pair: int = int(1e8)):
        self.envs = []
        self.index = []

        for fname in sorted(os.listdir(data_path)):
            path = os.path.join(data_path, fname)
            env = lmdb.open(path,
                            readonly=True,
                            lock=False,
                            readahead=False,
                            meminit=False)
            self.envs.append(env)

        self.latents_shape = [None] * len(self.envs)
        for shard_id, env in enumerate(self.envs):
            self.latents_shape[shard_id] = get_array_shape_from_lmdb(env, 'latents')
            for local_i in range(self.latents_shape[shard_id][0]):
                self.index.append((shard_id, local_i))

            # print("shard_id ", shard_id, " local_i ", local_i)

        self.max_pair = max_pair

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        """
            Outputs:
                - prompts: List of Strings
                - latents: Tensor of shape (num_denoising_steps, num_frames, num_channels, height, width). It is ordered from pure noise to clean image.
        """
        shard_id, local_idx = self.index[idx]

        latents = retrieve_row_from_lmdb(
            self.envs[shard_id],
            "latents", np.float16, local_idx,
            shape=self.latents_shape[shard_id][1:]
        )

        if len(latents.shape) == 4:
            latents = latents[None, ...]

        prompts = retrieve_row_from_lmdb(
            self.envs[shard_id],
            "prompts", str, local_idx
        )

        img = retrieve_row_from_lmdb(
            self.envs[shard_id],
            "img", np.uint8, local_idx,
            shape=(480, 832, 3)
        )
        img = Image.fromarray(img)
        img = TF.to_tensor(img).sub_(0.5).div_(0.5)

        return {
            "prompts": prompts,
            "ode_latent": torch.tensor(latents, dtype=torch.float32),
            "img": img
        }


class TextImagePairDataset(Dataset):
    def __init__(
        self,
        data_dir,
        transform=None,
        eval_first_n=-1,
        pad_to_multiple_of=None
    ):
        """
        Args:
            data_dir (str): Path to the directory containing:
                - target_crop_info_*.json (metadata file)
                - */ (subdirectory containing images with matching aspect ratio)
            transform (callable, optional): Optional transform to be applied on the image
        """
        self.transform = transform
        data_dir = Path(data_dir)

        # Find the metadata JSON file
        metadata_files = list(data_dir.glob('target_crop_info_*.json'))
        if not metadata_files:
            raise FileNotFoundError(f"No metadata file found in {data_dir}")
        if len(metadata_files) > 1:
            raise ValueError(f"Multiple metadata files found in {data_dir}")

        metadata_path = metadata_files[0]
        # Extract aspect ratio from metadata filename (e.g. target_crop_info_26-15.json -> 26-15)
        aspect_ratio = metadata_path.stem.split('_')[-1]

        # Use aspect ratio subfolder for images
        self.image_dir = data_dir / aspect_ratio
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")

        # Load metadata
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)

        eval_first_n = eval_first_n if eval_first_n != -1 else len(self.metadata)
        self.metadata = self.metadata[:eval_first_n]

        # Verify all images exist
        for item in self.metadata:
            image_path = self.image_dir / item['file_name']
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")

        self.dummy_prompt = "DUMMY PROMPT"
        self.pre_pad_len = len(self.metadata)
        if pad_to_multiple_of is not None and len(self.metadata) % pad_to_multiple_of != 0:
            # Duplicate the last entry
            self.metadata += [self.metadata[-1]] * (
                pad_to_multiple_of - len(self.metadata) % pad_to_multiple_of
            )

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        """
        Returns:
            dict: A dictionary containing:
                - image: PIL Image
                - caption: str
                - target_bbox: list of int [x1, y1, x2, y2]
                - target_ratio: str
                - type: str
                - origin_size: tuple of int (width, height)
        """
        item = self.metadata[idx]

        # Load image
        image_path = self.image_dir / item['file_name']
        image = Image.open(image_path).convert('RGB')

        # Apply transform if specified
        if self.transform:
            image = self.transform(image)

        return {
            'image': image,
            'prompts': item['caption'],
            'target_bbox': item['target_crop']['target_bbox'],
            'target_ratio': item['target_crop']['target_ratio'],
            'type': item['type'],
            'origin_size': (item['origin_width'], item['origin_height']),
            'idx': idx
        }


class ODERegressionCSVDataset(Dataset):
    def __init__(self, data_path: str, max_pair: int = int(1e8), num_frames=81, h=480, w=832):
        self.max_pair = max_pair
        self.data = pd.read_csv(data_path)
        self.data["text"] = self.data["text"].fillna("")
        self.log_file = "log/datasets_error_log.txt"
        self.num_frames = num_frames
        self.h = h
        self.w = w

    def __len__(self):
        return len(self.data)

    def _preprocess_video(self, sample) -> torch.Tensor:
        path = sample["path"]
        num_frames = sample["num_frames"]
        # if num_frames < self.num_frames:
        #     raise ValueError(f"Error: num_frames < {self.num_frames}")
        # frame_indices = list(range(self.num_frames))
        frame_indices = list(range(num_frames))

        if path.endswith(".mp4") or path.endswith(".mkv"):
            path = Path(path)
            video_reader = decord.VideoReader(uri=path.as_posix())
            frames = torch.tensor(
                video_reader.get_batch(frame_indices).asnumpy()
            ).float()  # [T, H, W, C]
            frames = frames.permute(0, 3, 1, 2).contiguous()  # [T, C, H, W]
        else:
            image_files = sorted(os.listdir(path))
            if not os.path.isdir(path) or not image_files:
                raise ValueError("Error: Invalid images path or no images found")
            frames = []
            for frame_index in frame_indices:
                frame_path = os.path.join(path, image_files[frame_index])
                frame = cv2.imread(frame_path, cv2.IMREAD_COLOR)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            frames = np.stack(frames)
            frames = torch.from_numpy(frames).float()
            frames = frames.permute(0, 3, 1, 2).contiguous()  # [T, C, H, W]
        orig_w, orig_h = frames.shape[3], frames.shape[2]
        max_pixels = self.h * self.w
        ori_pixels = orig_w * orig_h
        if ori_pixels <= max_pixels:
            target_w, target_h = orig_w, orig_h
        else:
            aspect_ratio = orig_w / orig_h
            target_h = int(math.sqrt(max_pixels / aspect_ratio))
            target_w = int(target_h * aspect_ratio)
        
        target_h = (target_h // 32) * 32
        target_w = (target_w // 32) * 32

        video_tensor = torch.stack([resize(frame, (target_h, target_w)) for frame in frames], dim=0)
        video_tensor = video_tensor.permute(1, 0, 2, 3) / 255.0
        video_tensor = video_tensor * 2 - 1
        return video_tensor

    def __getitem__(self, index):
        sample = self.data.iloc[index]
        try:
            video = self._preprocess_video(sample)
            return {
                "prompts": sample["text"],
                "video": video,
            }
        except Exception as e:
            with open(self.log_file, "a") as f:
                f.write(f"Error at index {index}: {str(e)}\n")
            print(f"Error at index {index}: {e}. Skipping this index.")
            return {
                "prompts": "",
                "video": torch.zeros((3, self.num_frames, self.h, self.w)),  # 占位符视频张量
            }

class OviCSVDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        num_frames: int = 121,
        h: int = 704,
        w: int = 1280,
        audio_sample_rate: int = 16000,
        audio_duration_secs: int = 5,
    ):
        """
        A dataset for loading (text, video, audio) triplets for Ovi model training.
        Args:
            data_path (str): Path to the CSV file. The CSV should contain 'text', 'video_path', and 'audio_path' columns.
            num_frames (int): The number of frames to sample from the video.
            h (int): Target height for video resizing.
            w (int): Target width for video resizing.
            audio_sample_rate (int): The target sample rate for audio.
            audio_duration_secs (int): The target duration for audio in seconds.
        """
        self.data = pd.read_csv(data_path)
        self.data["text"] = self.data["text"].fillna("")
        
        # --- Store configuration ---
        self.num_frames = num_frames
        self.h = h
        self.w = w
        self.audio_sample_rate = audio_sample_rate
        self.target_audio_length = math.ceil(self.audio_sample_rate * audio_duration_secs // 512) * 512  # Pad to nearest multiple of 512

        self.log_file = "log/ovi_dataset_error_log.txt"
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)


    def __len__(self):
        return len(self.data)

    def _preprocess_video(self, video_path: str) -> torch.Tensor:
        """Loads and preprocesses a video file."""
        path = Path(video_path)
        video_reader = decord.VideoReader(uri=path.as_posix(), num_threads=1)
        total_frames = (len(video_reader) - 1) // 4 * 4 + 1  # 4n+1
        frame_indices = list(range(total_frames))
        frames = torch.from_numpy(video_reader.get_batch(frame_indices).asnumpy()).float() # [T, H, W, C]
        frames = frames.permute(0, 3, 1, 2).contiguous()  # [T, C, H, W]

        orig_w, orig_h = frames.shape[3], frames.shape[2]
        max_pixels = self.h * self.w
        ori_pixels = orig_w * orig_h
        if ori_pixels <= max_pixels:
            target_w, target_h = orig_w, orig_h
        else:
            aspect_ratio = orig_w / orig_h
            target_h = int(math.sqrt(max_pixels / aspect_ratio))
            target_w = int(target_h * aspect_ratio)
        # Snap to multiple of 32 for model compatibility
        target_h = (target_h // 32) * 32
        target_w = (target_w // 32) * 32

        # Resize and normalize
        video_tensor = torch.stack([resize(frame, (target_h, target_w)) for frame in frames], dim=0)
        video_tensor = video_tensor.permute(1, 0, 2, 3)   # [C, T, H, W]
        video_tensor = (video_tensor / 255.0) * 2.0 - 1.0 # Normalize to [-1, 1]
        return video_tensor                               # Return shape [C, T, H, W]

    def _preprocess_audio(self, audio_path: str) -> torch.Tensor:
        waveform, sample_rate = torchaudio.load(audio_path)
        if sample_rate != self.audio_sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.audio_sample_rate)
            waveform = resampler(waveform)

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        if waveform.shape[1] > self.target_audio_length:
            waveform = waveform[:, :self.target_audio_length]
        else:
            waveform_len = waveform.shape[1]
            waveform_len = waveform_len // 512 * 512
            waveform = waveform[:, :waveform_len]
            
        return waveform.squeeze(0)  # Return shape [L]
    
    def __getitem__(self, index):
        sample = self.data.iloc[index]
        try:
            video = self._preprocess_video(sample["video_path"])
            audio = self._preprocess_audio(sample["audio_path"])
            
            return {
                "prompts": sample["text"],
                "video": video,
                "audio": audio,
            }
        except Exception as e:
            with open(self.log_file, "a") as f:
                f.write(f"Error at index {index} (video: {sample.get('video_path', 'N/A')}, audio: {sample.get('audio_path', 'N/A')}): {str(e)}\n")
            
            # Return a placeholder batch to avoid crashing the training
            return {
                "prompts": "placeholder prompt",
                "video": torch.zeros((3, self.num_frames, self.h, self.w)),
                "audio": torch.zeros((1, self.target_audio_length)),
            }
        
def cycle(dl):
    while True:
        for data in dl:
            yield data

def masks_like(tensor, zero=False, generator=None, p=0.2):
    # assert isinstance(tensor, list)
    out1 = [torch.ones(u.shape, dtype=u.dtype, device=u.device) for u in tensor]

    out2 = [torch.ones(u.shape, dtype=u.dtype, device=u.device) for u in tensor]

    if zero:
        if generator is not None:
            for u, v in zip(out1, out2):
                random_num = torch.rand(
                    1, generator=generator, device=generator.device).item()
                if random_num < p:
                    u[0, :] = torch.normal(
                        mean=-3.5,
                        std=0.5,
                        size=(1,),
                        device=u.device,
                        generator=generator).expand_as(u[0, :]).exp()
                    v[0, :] = torch.zeros_like(v[0, :])
                else:
                    u[0, :] = u[0, :]
                    v[0, :] = v[0, :]
        else:
            for u, v in zip(out1, out2):
                u[0, :] = torch.zeros_like(u[0, :])
                v[0, :] = torch.zeros_like(v[0, :])

    return out1, out2

if __name__ == '__main__':
    CSV_PATH = "/videogen/Wan2.2-TI2V-5B-Turbo/data/matrix_audio.csv"
    NUM_FRAMES = 121  # Use a number of frames that matches your CSV, e.g., 63
    TARGET_H = 704   # Example target resolution
    TARGET_W = 1280  # Example target resolution
    AUDIO_SAMPLE_RATE = 16000
    AUDIO_DURATION_SECS = 5
    # DataLoader parameters
    BATCH_SIZE = 1 # Use a batch size > 1 to test collation

    print("=" * 60)
    print("Testing OviCSVDataset and DataLoader Output...")
    print("=" * 60)

    # 1. Instantiate the dataset
    try:
        dataset = OviCSVDataset(
            data_path=CSV_PATH,
            num_frames=NUM_FRAMES,
            h=TARGET_H,
            w=TARGET_W,
            audio_sample_rate=AUDIO_SAMPLE_RATE,
            audio_duration_secs=AUDIO_DURATION_SECS,
        )
        print(f"✓ Dataset initialized successfully with {len(dataset)} samples.")
    except Exception as e:
        print(f"✗ Error initializing dataset: {e}")
        print("  Please ensure the CSV file exists at the specified path and is correctly formatted.")
        exit()

    # 2. Create a DataLoader
    data_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0 # Set to 0 for simple testing, can be increased for performance
    )
    print(f"✓ DataLoader created with batch size {BATCH_SIZE}.")

    # 3. Fetch and inspect one batch
    try:
        print("\nFetching one batch...")
        batch = next(iter(data_loader))
        print("✓ Batch fetched successfully.")

        # 4. Verify the contents and shapes
        print("\n--- Batch Content Verification ---")
        video_tensor = batch["video"]
        audio_tensor = batch["audio"]
        prompts = batch["prompts"]

        print(f"  - Prompts:")
        print(f"    - Type: {type(prompts)}")
        print(f"    - Length: {len(prompts)} (should match batch size of {BATCH_SIZE})")
        print(f"    - Example prompt: '{prompts[0][:80]}...'")

        print(f"\n  - Video Tensor:")
        print(f"    - Shape: {video_tensor.shape}")
        print(f"    - Dtype: {video_tensor.dtype}")
        print(f"    - Value Range: min={video_tensor.min():.2f}, max={video_tensor.max():.2f} (should be approx. [-1, 1])")
        
        # Assertions to confirm the shape is correct for the model VAE
        assert len(video_tensor.shape) == 5, f"Video tensor should have 5 dimensions, but got {len(video_tensor.shape)}"
        assert video_tensor.shape[0] == BATCH_SIZE, f"Video batch size is incorrect, expected {BATCH_SIZE}"
        assert video_tensor.shape[1] == 3, "Video should have 3 channels (RGB)"
        print("    - Shape is valid for VAE input.")

        print(f"\n  - Audio Tensor:")
        print(f"    - Shape: {audio_tensor.shape}")
        print(f"    - Dtype: {audio_tensor.dtype}")
        
        # Assertions to confirm the shape is correct
        assert len(audio_tensor.shape) == 2, f"Audio tensor should have 2 dimensions, but got {len(audio_tensor.shape)}"
        assert audio_tensor.shape[0] == BATCH_SIZE, f"Audio batch size is incorrect, expected {BATCH_SIZE}"
        print("    - Shape is valid for VAE input.")

        print("\n\033[92m✓ Batch shapes and dtypes are correct and match the requirements for the Ovi model.\033[0m")


    except StopIteration:
        print("✗ DataLoader is empty. This might happen if the CSV is empty or cannot be read.")
    except Exception as e:
        print(f"✗ An error occurred while fetching or inspecting the batch: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 60)
    print("Test completed.")
    print("=" * 60)