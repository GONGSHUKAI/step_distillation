import os
import json
import types
import torch
import torch.nn as nn
import torch.distributed as dist
from typing import List, Dict, Tuple, Optional, Union
import logging

from ovi.modules.fusion import FusionModel
from ovi.modules.t5 import umt5_xxl
from wan22.modules.vae2_2 import _video_vae as _video_vae_2_2
from ovi.modules.mmaudio.features_utils import FeaturesUtils
from ovi.modules.tokenizers import HuggingfaceTokenizer

from utils.scheduler import SchedulerInterface, FlowMatchScheduler
from safetensors.torch import load_file
import math

logger = logging.getLogger(__name__)

class OviTextEncoder(torch.nn.Module):
    def __init__(self, model_name: str = "Wan2.2-TI2V-5B") -> None:
        super().__init__()
        self.model_name = model_name
        
        logger.info("Initializing Ovi Text Encoder...") if dist.get_rank() == 0 else None
        self.text_encoder = umt5_xxl(
            encoder_only=True,
            return_tokenizer=False,
            dtype=torch.float32,
            device=torch.device('cpu')
        ).eval().requires_grad_(False)

        logger.info("Ovi Text Encoder initialized, loading model weights...") if dist.get_rank() == 0 else None
        self.text_encoder.load_state_dict(
            torch.load(f"/videogen/Ovi/ckpts/{self.model_name}/models_t5_umt5-xxl-enc-bf16.pth",
                       map_location='cpu', weights_only=False)
        )

        self.tokenizer = HuggingfaceTokenizer(
            name=f"/videogen/Ovi/ckpts/{self.model_name}/google/umt5-xxl", 
            seq_len=512, 
            clean='whitespace'
        )
        logger.info(f"Ovi Text Encoder weights and tokenizer loaded.") if dist.get_rank() == 0 else None
    
    @property
    def device(self):
        return torch.cuda.current_device()
    
    def forward(self, text_prompts: Union[str, List[str]]) -> dict:
        if isinstance(text_prompts, str):
            text_prompts = [text_prompts]
        
        ids, mask = self.tokenizer(
            text_prompts, return_mask=True, add_special_tokens=True)
        ids = ids.to(self.device)
        mask = mask.to(self.device)
        seq_lens = mask.gt(0).sum(dim=1).long()
        context = self.text_encoder(ids, mask)

        for u, v in zip(context, seq_lens):
            u[v:] = 0.0  # set padding to 0.0

        return {
            "prompt_embeds": context
        }


class OviVAEWrapper(torch.nn.Module):
    def __init__(
        self,
        z_dim_video: int = 48,
        c_dim_video: int = 160,
        video_vae_pth: str = "/videogen/Ovi/ckpts/Wan2.2-TI2V-5B/Wan2.2_VAE.pth",
        dim_mult: List[int] = [1, 2, 4, 4],
        temperal_downsample: List[bool] = [False, True, True],
        
        audio_mode: str = '16k',
        audio_tod_vae_ckpt: str = "/videogen/Ovi/ckpts/MMAudio/ext_weights/v1-16.pth",
        audio_bigvgan_ckpt: str = "/videogen/Ovi/ckpts/MMAudio/ext_weights/best_netG.pt",
    ):
        super().__init__()
        
        # ===== 视频VAE (Wan2.2) =====
        # 初始化时在CPU上 (与Wan2_2_VAEWrapper一致)
        self.mean = torch.tensor([
            -0.2289, -0.0052, -0.1323, -0.2339, -0.2799,  0.0174,  0.1838,  0.1557,
            -0.1382,  0.0542,  0.2813,  0.0891,  0.1570, -0.0098,  0.0375, -0.1825,
            -0.2246, -0.1207, -0.0698,  0.5109,  0.2665, -0.2108, -0.2158,  0.2502,
            -0.2055, -0.0322,  0.1109,  0.1567, -0.0729,  0.0899, -0.2799, -0.1230,
            -0.0313, -0.1649,  0.0117,  0.0723, -0.2839, -0.2083, -0.0520,  0.3748,
            0.0152,  0.1957,  0.1433, -0.2944,  0.3573, -0.0548, -0.1681, -0.0667
        ], dtype=torch.float32)

        self.std = torch.tensor([
            0.4765, 1.0364, 0.4514, 1.1677, 0.5313, 0.4990, 0.4818, 0.5013,
            0.8158, 1.0344, 0.5894, 1.0901, 0.6885, 0.6165, 0.8454, 0.4978,
            0.5759, 0.3523, 0.7135, 0.6804, 0.5833, 1.4146, 0.8986, 0.5659,
            0.7069, 0.5338, 0.4889, 0.4917, 0.4069, 0.4999, 0.6866, 0.4093,
            0.5709, 0.6065, 0.6415, 0.4944, 0.5726, 1.2042, 0.5458, 1.6887,
            0.3971, 1.0600, 0.3943, 0.5537, 0.5444, 0.4089, 0.7468, 0.7744
        ], dtype=torch.float32)

        self.video_dtype = torch.bfloat16
        
        logger.info("Initializing Wan2.2-VAE...") if dist.get_rank() == 0 else None
        self.video_vae = (
            _video_vae_2_2(
                pretrained_path=video_vae_pth,
                z_dim=z_dim_video,
                dim=c_dim_video,
                dim_mult=dim_mult,
                temperal_downsample=temperal_downsample,
            )
            .eval()
            .requires_grad_(False)
        )
        logger.info(f"Loaded Wan2.2-VAE weights from {video_vae_pth}") if dist.get_rank() == 0 else None
        
        # ===== 音频VAE (MMAudio) =====
        # 初始化时在CPU上
        logger.info("Initializing MMAudio VAE and Vocoder...") if dist.get_rank() == 0 else None
        self.audio_vae = FeaturesUtils(
            mode=audio_mode,
            need_vae_encoder=True,
            tod_vae_ckpt=audio_tod_vae_ckpt,
            bigvgan_vocoder_ckpt=audio_bigvgan_ckpt,
        )
        self.audio_vae.eval().requires_grad_(False)
        logger.info(f"Loaded MMAudio VAE weights from {audio_tod_vae_ckpt} and Vocoder weights from {audio_bigvgan_ckpt}") if dist.get_rank() == 0 else None
    
    # ===== 视频VAE接口 =====
    def encode(self, pixel):
        device, dtype = pixel[0].device, self.video_dtype
        scale = [self.mean.to(device=device, dtype=dtype),
                 1.0 / self.std.to(device=device, dtype=dtype)]
        output = [
            self.video_vae.encode(u.to(self.dtype).unsqueeze(0), scale).float().squeeze(0)
            for u in pixel
        ]
        return output
    
    def encode_video(self, pixel: torch.Tensor) -> torch.Tensor:
        # pixel: [batch_size, num_channels, num_frames, height, width]
        device, dtype = pixel.device, pixel.dtype
        scale = [self.mean.to(device=device, dtype=dtype),
                 1.0 / self.std.to(device=device, dtype=dtype)]

        output = [
            self.video_vae.encode(u.unsqueeze(0), scale).float().squeeze(0)
            for u in pixel
        ]
        output = torch.stack(output, dim=0)
        # from [batch_size, num_channels, num_frames, height, width]
        # to [batch_size, num_frames, num_channels, height, width]
        output = output.permute(0, 2, 1, 3, 4)
        return output

    def decode_video(self, latent: torch.Tensor, use_cache: bool = False) -> torch.Tensor:
        # from [batch_size, num_frames, num_channels, height, width]
        # to [batch_size, num_channels, num_frames, height, width]
        zs = latent.permute(0, 2, 1, 3, 4)
        if use_cache:
            assert latent.shape[0] == 1, "Batch size must be 1 when using cache"

        device, dtype = latent.device, latent.dtype
        scale = [self.mean.to(device=device, dtype=dtype),
                 1.0 / self.std.to(device=device, dtype=dtype)]

        if use_cache:
            decode_function = self.video_vae.cached_decode
        else:
            decode_function = self.video_vae.decode

        output = []
        for u in zs:
            output.append(decode_function(u.unsqueeze(0), scale).float().clamp_(-1, 1).squeeze(0))
        output = torch.stack(output, dim=0)
        # from [batch_size, num_channels, num_frames, height, width]
        # to [batch_size, num_frames, num_channels, height, width]
        output = output.permute(0, 2, 1, 3, 4)
        return output
    
    # ===== 音频VAE接口 =====
    
    def encode_audio(self, audio: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            latent = self.audio_vae.wrapped_encode(audio)
        return latent  # [B, L_latent, C_latent]
    
    def decode_audio(self, latent: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            audio = self.audio_vae.wrapped_decode(latent)  # [B, L]
        return audio


class OviFusionWrapper(torch.nn.Module):
    def __init__(
        self, 
        model_name: str = "Ovi",
        video_config_path: str = "ovi/configs/model/dit/video.json",
        audio_config_path: str = "ovi/configs/model/dit/audio.json",
        is_causal: bool = False,
        timestep_shift: float = 5.0,
        **kwargs
    ):
        super().__init__()
        self.model_name = model_name
        self.is_causal = is_causal
        
        with open(video_config_path) as f:
            self.video_config = json.load(f)
        with open(audio_config_path) as f:
            self.audio_config = json.load(f)

        logger.info(f"Initializing Ovi FusionModel: {self.video_config['num_layers']} video blocks and {self.audio_config['num_layers']} audio blocks...") if dist.get_rank() == 0 else None
        self.model = FusionModel(self.video_config, self.audio_config).to(dtype=torch.bfloat16, device=torch.device('cpu'))
        logger.info(f"Ovi FusionModel initialized, loading model weights...") if dist.get_rank() == 0 else None

        state_dict = load_file(
            f"/videogen/Ovi/ckpts/{self.model_name}/model.safetensors",
            device='cpu'
        )

        self.model.load_state_dict(state_dict)
        logger.info(f"Ovi weights loaded.") if dist.get_rank() == 0 else None
        self.model.eval()

        self.scheduler = FlowMatchScheduler(
            shift=timestep_shift,
            sigma_min=0.0,
            extra_one_step=True
        )
        self.scheduler.set_timesteps(1000, training=True)
        self.post_init()
    
    def enable_gradient_checkpointing(self):
        self.model.gradient_checkpointing = True
        
    def _convert_flow_pred_to_x0(self, flow_pred: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """
        将 flow matching 的预测 (velocity) 转换为 x0 预测。
        这个版本是正确的，并且可以通用地处理视频和音频。
        x0_pred = xt - sigma_t * flow_pred
        """
        # 使用更高精度以保证计算稳定
        original_dtype = flow_pred.dtype
        flow_pred_d = flow_pred.double()
        xt_d = xt.double()
        sigmas_d = self.scheduler.sigmas.to(xt.device, dtype=torch.double)
        timesteps_d = self.scheduler.timesteps.to(xt.device, dtype=torch.double)

        # 将多维时间步展平，使其与展平后的数据一一对应
        # timestep 输入形状可以是 [B] 或 [B, F] 或 [B, L]
        if timestep.dim() > 1:
            timestep_flat = timestep.flatten()
        else: # 如果是 [B]，需要扩展以匹配 xt 的 token 数量
            # shape of xt: video [B, F, C, H, W], audio [B, L, C]
            num_tokens_per_sample = xt.shape[1]    # F for video, L for audio
            timestep_flat = timestep.unsqueeze(1).repeat(1, num_tokens_per_sample).flatten()    # shape: [B * F] or [B * L]

        # 为每个 token 找到其对应的 sigma 值
        timestep_indices = torch.argmin(torch.abs(timesteps_d.unsqueeze(0) - timestep_flat.unsqueeze(1)), dim=1)
        sigma_t = sigmas_d[timestep_indices]
        
        # 将数据展平以便进行批处理
        xt_flat = xt_d.reshape(timestep_flat.shape[0], -1)  # video: [B * F, C * H * W], audio: [B * L, C]
        flow_pred_flat = flow_pred_d.reshape(timestep_flat.shape[0], -1)    # video: [B * F, C * H * W], audio: [B * L, C]

        # 调整 sigma_t 的形状以进行广播
        sigma_t = sigma_t.view(-1, *([1] * (xt_flat.dim() - 1)))    # video: [B * F, 1, 1, 1], audio: [B * L, 1]

        # 计算 x0 预测
        x0_pred_flat = xt_flat - sigma_t * flow_pred_flat
        
        # 恢复原始形状和数据类型
        x0_pred = x0_pred_flat.reshape(xt.shape).to(original_dtype)
        
        return x0_pred
    
    def forward(
        self,
        video_latent: torch.Tensor,
        audio_latent: torch.Tensor,
        timestep: torch.Tensor,
        conditional_dict: dict,
        
        wan22_image_latent: Optional[torch.Tensor] = None,
        mask2: Optional[torch.Tensor] = None,
        first_frame_is_clean: bool = False, # <--- 保留这个关键标志
        
        # 移除不再需要的 wan22_input_timestep
        **kwargs 
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        text_embeds = conditional_dict["prompt_embeds"]
        
        # 准备 FusionModel 的输入 (保持不变)
        num_frames, c, h, w = video_latent.shape[1:]        # video latent shape: (B, F, C, H, W)
        # ... (vid_seq_len, audio_seq_len 计算不变) ...
        _patch_size_h = self.model.video_model.patch_size[1]
        _patch_size_w = self.model.video_model.patch_size[2]
        vid_seq_len = num_frames * h * w // (_patch_size_h * _patch_size_w)
        audio_seq_len = audio_latent.shape[1]
        
        video_input = video_latent.squeeze(0).permute(1, 0, 2, 3)   # video input shape: [C, F, H, W]
        audio_input = audio_latent.squeeze(0)   # audio input shape: [L, C]
        text_input = text_embeds.squeeze(0)
        timestep = timestep[:, 0] if timestep.dim() > 1 else timestep
        # 调用底层 FusionModel，只传递简单的 timestep
        # logger.info(f"Calling FusionModel with timestep shape: {timestep.shape}, values: {timestep}") if dist.get_rank() == 0 else None
        # logger.info(f"Video input shape: {video_input.shape}, Audio input shape: {audio_input.shape}, Text input shape: {text_input.shape}") if dist.get_rank() == 0 else None
        flow_pred_video, flow_pred_audio = self.model(
            vid=[video_input],
            audio=[audio_input],
            t=timestep,  # <--- 只传递一个共享的、简单的 timestep
            vid_context=[text_input],
            audio_context=[text_input],
            vid_seq_len=vid_seq_len,
            audio_seq_len=audio_seq_len,
            first_frame_is_clean=first_frame_is_clean, # <--- 传递标志
        )
        
        # 恢复 batch 维度 (保持不变)
        flow_pred_video = flow_pred_video[0].permute(1, 0, 2, 3).unsqueeze(0)   # [1, F, C, H, W]
        flow_pred_audio = flow_pred_audio[0].unsqueeze(0)   # [1, L, C]
        
        # print(f"flow_pred_video shape: {flow_pred_video.shape}, flow_pred_audio shape: {flow_pred_audio.shape}")
        # print(f"timestep shape: {timestep.shape}, timestep values: {timestep}")
        # 将 flow 转换为 x0
        x0_pred_video = self._convert_flow_pred_to_x0(flow_pred_video, video_latent, timestep)
        x0_pred_audio = self._convert_flow_pred_to_x0(flow_pred_audio, audio_latent, timestep)

        # 在 x0 上应用图像注入逻辑
        if mask2 is not None and wan22_image_latent is not None:
            final_x0_video = (1. - mask2) * wan22_image_latent + mask2 * x0_pred_video
            final_x0_video = final_x0_video.to(video_latent.dtype)
        else:
            final_x0_video = x0_pred_video

        return final_x0_video, x0_pred_audio
    
    def get_scheduler(self) -> SchedulerInterface:
        scheduler = self.scheduler
        scheduler.convert_x0_to_noise = types.MethodType(
            SchedulerInterface.convert_x0_to_noise, scheduler)
        scheduler.convert_noise_to_x0 = types.MethodType(
            SchedulerInterface.convert_noise_to_x0, scheduler)
        scheduler.convert_velocity_to_x0 = types.MethodType(
            SchedulerInterface.convert_velocity_to_x0, scheduler)
        self.scheduler = scheduler
        return scheduler
    
    def post_init(self):
        self.get_scheduler()


if __name__ == "__main__":
    import torchaudio, imageio
    import decord, torchaudio
    import numpy as np
    from torchvision.transforms.functional import resize
    from torch.utils.data import DataLoader
    
    # PROMPT = "The video opens with a wide high-angle shot, looking down over a vast, arid desert landscape. In the immediate foreground, a large, dark brown rock formation partially obscures the view. As the camera pans slightly to the right, the rock formation moves out of the frame, revealing a barren desert valley with a highway running through it. A small, sparse town is visible on the right side of the highway. In the mid-ground, a large, enclosed dirt arena, possibly for a demolition derby, comes into full view. Several battered cars are scattered within the arena, surrounded by a low, yellow barrier. Beyond the arena, there's a large, open lot filled with numerous parked cars, resembling a junkyard or a used car lot. In the far background, a range of large, dark mountains stretches across the horizon under a hazy sky. The overall visual style is realistic."
    # VIDEO_PATH = "/videogen/audio_preprocess/matrix/video/1fa65cb31263f327a902_114_sdr_4.mp4"
    # AUDIO_PATH = "/videogen/audio_preprocess/matrix/audio/1fa65cb31263f327a902_114_sdr_4.wav"
    VIDEO_RECON_PATH = "/videogen/Wan2.2-TI2V-5B-Turbo/data/tmp/recon_video.mp4"
    AUDIO_RECON_PATH = "/videogen/Wan2.2-TI2V-5B-Turbo/data/tmp/recon_audio.wav"

    # def preprocess_video(video_path, max_pixels=704*1280):
    #     print(f"Preprocessing video from {video_path}...")
    #     video_reader = decord.VideoReader(uri=video_path, num_threads=1)
    #     num_frames = (len(video_reader) - 1) // 4 * 4 + 1  # 4n+1
    #     frame_indices = list(range(num_frames))
        
    #     frames = torch.from_numpy(video_reader.get_batch(frame_indices).asnumpy()).float()
    #     frames = frames.permute(0, 3, 1, 2) # T, C, H, W
    #     orig_h, orig_w = frames.shape[2], frames.shape[3]
    #     target_h, target_w = orig_h, orig_w
    #     if target_h * target_w > max_pixels:
    #         aspect_ratio = orig_w / orig_h
    #         target_h = int((max_pixels / aspect_ratio) ** 0.5)
    #         target_w = int(target_h * aspect_ratio)
            
    #     target_h = target_h // 32 * 32
    #     target_w = target_w // 32 * 32
    #     print(f"Resizing video from ({orig_h}, {orig_w}) to ({target_h}, {target_w}) to meet pixel limit.")
        
    #     resized_frames = torch.stack([resize(f, (target_h, target_w)) for f in frames], dim=0)
    #     video_tensor = resized_frames.permute(1, 0, 2, 3) # C, T, H, W
    #     video_tensor = (video_tensor / 255.0) * 2.0 - 1.0
    #     return video_tensor.unsqueeze(0) # Add batch dimension

    # def preprocess_audio(audio_path, sample_rate=16000, duration_secs=5):
    #     print(f"Preprocessing audio from {audio_path}...")
    #     target_len = math.ceil(sample_rate * duration_secs // 512) * 512
    #     waveform, sr = torchaudio.load(audio_path)
        
    #     if sr != sample_rate:
    #         resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)
    #         waveform = resampler(waveform)
            
    #     if waveform.shape[0] > 1:
    #         waveform = torch.mean(waveform, dim=0, keepdim=True)
            
    #     if waveform.shape[1] > target_len:
    #         waveform = waveform[:, :target_len]
    #     else:
    #         waveform_len = waveform.shape[1] // 512 * 512
    #         waveform = waveform[:, :waveform_len]
    #     return waveform
    
    def save_video(video_tensor: torch.Tensor, save_path: str, fps: int = 24):
        import imageio
        video_np = (video_tensor.clamp(-1,1)+1)/2*255          # → [0,255]
        video_np = video_np.squeeze(0).permute(0,2,3,1).cpu().numpy().astype(np.uint8)
        imageio.mimsave(save_path, video_np, fps=fps, codec='libx264')
        print(f"Saved video to {save_path}")

    def save_audio(audio_tensor: torch.Tensor, save_path: str, sample_rate: int = 16000):
        import torchaudio
        torchaudio.save(
            save_path,
            audio_tensor.clamp(-1, 1).cpu(),
            sample_rate=sample_rate
        )
        print(f"Saved audio to {save_path}")

    # try:
    #     video_tensor = preprocess_video(VIDEO_PATH)
    #     audio_tensor = preprocess_audio(AUDIO_PATH)
    #     print(f"\n✓ Data preprocessed:")
    #     print(f"  - Video tensor shape: {video_tensor.shape}, dtype: {video_tensor.dtype}, device: {video_tensor.device}")
    #     print(f"  - Audio tensor shape: {audio_tensor.shape}, dtype: {audio_tensor.dtype}, device: {audio_tensor.device}")
    # except Exception as e:
    #     print(f"✗ Error during data preprocessing: {e}")
    #     exit()

    from dataset import OviCSVDataset
    CSV_PATH = "/videogen/Wan2.2-TI2V-5B-Turbo/data/matrix_audio.csv"
    NUM_FRAMES = 121  # Use a number of frames that matches your CSV, e.g., 63
    TARGET_H = 704   # Example target resolution
    TARGET_W = 1280  # Example target resolution
    AUDIO_SAMPLE_RATE = 16000
    AUDIO_DURATION_SECS = 5
    # DataLoader parameters
    BATCH_SIZE = 1 # Use a batch size > 1 to test collation
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

    data_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0 # Set to 0 for simple testing, can be increased for performance
    )
    print(f"✓ DataLoader created with batch size {BATCH_SIZE}.")

    try:
        print("\nFetching one batch...")
        batch = next(iter(data_loader))
        print("✓ Batch fetched successfully.")

        # 4. Verify the contents and shapes
        print("\n--- Batch Content Verification ---")
        video_tensor = batch["video"]
        audio_tensor = batch["audio"]
        PROMPT = batch["prompts"]

        print(f"  - Prompts:")
        print(f"    - Type: {type(PROMPT)}")
        print(f"    - Length: {len(PROMPT)} (should match batch size of {BATCH_SIZE})")
        print(f"    - Example prompt: '{PROMPT[0][:80]}...'")

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


    # print("=" * 50)
    # print("Testing OviTextEncoder (CPU initialization)...")
    # print("=" * 50)
    # try:
    #     text_encoder = OviTextEncoder()
    #     print(f"✓ Text encoder initialized on: {next(text_encoder.parameters()).device}")
        
    #     # 移动到GPU (模拟FSDP包装前的操作)
    #     text_encoder = text_encoder.cuda()
    #     print(f"✓ Text encoder moved to: {next(text_encoder.parameters()).device}")
        
    #     # 测试forward
    #     prompts = PROMPT
    #     result = text_encoder(prompts)
    #     print(f"✓ prompt_embeds shape: {result['prompt_embeds'].shape}")
    #     print(f"✓ prompt_embeds device: {result['prompt_embeds'].device}")
    # except Exception as e:
    #     print(f"✗ Error: {e}")
    #     import traceback
    #     traceback.print_exc()

    # text_encoder = text_encoder.to(device=torch.device('cpu'))
    # del text_encoder 

    # print("\n" + "=" * 50)
    # print("Testing OviVAEWrapper (CPU initialization)...")
    # print("=" * 50)
    # try:
    #     vae = OviVAEWrapper()
    #     print(f"✓ VAE initialized")
    #     print(f"  Video VAE device: {next(vae.video_vae.parameters()).device}, dtype: {next(vae.video_vae.parameters()).dtype}")
    #     print(f"  Audio VAE device: {next(vae.audio_vae.parameters()).device}, dtype: {next(vae.audio_vae.parameters()).dtype}")
        
    #     # 移动到GPU (模拟Trainer中的操作)
    #     vae = vae.to(device=torch.device('cuda'))
    #     print(f"✓ VAE moved to GPU")
    #     print(f"  Video VAE device: {next(vae.video_vae.parameters()).device}, dtype: {next(vae.video_vae.parameters()).dtype}")
    #     print(f"  Audio VAE device: {next(vae.audio_vae.parameters()).device}, dtype: {next(vae.audio_vae.parameters()).dtype}")
        
    #     # 测试编解码
    #     video = video_tensor.cuda()
    #     video_latent = vae.encode_video(video)
    #     print(f"✓ Video latent shape: {video_latent.shape}, dtype: {video_latent.dtype}")
        
    #     video_recon = vae.decode_video(video_latent)
    #     print(f"✓ Video recon shape: {video_recon.shape}, dtype: {video_recon.dtype}")
        
    #     audio = audio_tensor.cuda()
    #     # audio = torch.randn(1, 16000 * 5).cuda()   # 使用随机音频进行测试
    #     audio_latent = vae.encode_audio(audio)
    #     print(f"✓ Audio latent shape: {audio_latent.shape}, dtype: {audio_latent.dtype}")
        
    #     audio_recon = vae.decode_audio(audio_latent)
    #     print(f"✓ Audio recon shape: {audio_recon.shape}, dtype: {audio_recon.dtype}")
    #     audio_recon = audio_recon.squeeze(0)

    #     # 保存视频、音频到/videogen/Wan2.2-TI2V-5B-Turbo/data/tmp
    #     os.makedirs("/videogen/Wan2.2-TI2V-5B-Turbo/data/tmp", exist_ok=True)
    #     save_video(video_recon, VIDEO_RECON_PATH, fps=24)
    #     save_audio(audio_recon, AUDIO_RECON_PATH, sample_rate=16000)
    #     print(f"✓ Reconstructed video and audio saved to /videogen/Wan2.2-TI2V-5B-Turbo/data/tmp/")
    # except Exception as e:
    #     print(f"✗ Error: {e}")
    #     import traceback
    #     traceback.print_exc()
    
    # vae = vae.to(device=torch.device('cpu'))
    # del vae

    print("\n" + "=" * 50)
    print("Testing OviFusionWrapper (CPU initialization)...")
    print("=" * 50)
    try:
        model = OviFusionWrapper()
        print(f"✓ Fusion model initialized")
        print(f"  Video model device: {next(model.model.video_model.parameters()).device}, dtype: {next(model.model.video_model.parameters()).dtype}")
        print(f"  Audio model device: {next(model.model.audio_model.parameters()).device}, dtype: {next(model.model.audio_model.parameters()).dtype}")
        
        # 移动到GPU (模拟FSDP包装前的操作)
        model = model.cuda()
        print(f"✓ Fusion model moved to GPU")
        
        # 测试forward
        # video_latent = video_latent.cuda().to(dtype=torch.bfloat16)           # standard video shape [B, F, C, H, W] = [B, 31, 48, 44, 80]
        video_latent = torch.randn(1, 31, 48, 44, 80).cuda().to(dtype=torch.bfloat16)  # 使用随机潜变量进行测试，形状为 [B, F, C, H, W]
        # audio_latent = audio_latent.permute(0, 2, 1).cuda().to(dtype=torch.bfloat16)                  # standard audio shape [B, L, D] = [B, 157, 20]
        audio_latent = torch.randn(1, 157, 20).cuda().to(dtype=torch.bfloat16)  # 使用随机潜变量进行测试，形状为 [B, L, D]
        timestep = torch.ones(1).long().cuda() * 500
        # text_embeds = result['prompt_embeds'].cuda().to(dtype=torch.bfloat16)
        text_embeds = torch.randn(1, 512, 4096).cuda().to(dtype=torch.bfloat16)  # 使用随机文本嵌入进行测试，形状为 [B, Seq_Len, D]
        conditional_dict = {"prompt_embeds": text_embeds}
        print(f"✓ Input video_latent shape: {video_latent.shape}, dtype: {video_latent.dtype}")
        print(f"✓ Input audio_latent shape: {audio_latent.shape}, dtype: {audio_latent.dtype}")
        print(f"✓ Input timestep: {timestep}")
        print(f"✓ Input text_embeds shape: {text_embeds.shape}, dtype: {text_embeds.dtype}")
        pred_video, pred_audio = model(
            video_latent=video_latent,
            audio_latent=audio_latent,
            timestep=timestep,
            conditional_dict=conditional_dict,
        )
        
        print(f"✓ Pred video shape: {pred_video.shape}, dtype: {pred_video.dtype}")
        print(f"✓ Pred audio shape: {pred_audio.shape}, dtype: {pred_audio.dtype}")
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
    
    del model

    print("\n" + "=" * 50)
    print("All tests completed!")
    print("=" * 50)