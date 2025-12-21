# FILE: ovi_causal_batch_inference.py

import argparse
import torch
import torch.nn.functional as F
import os
import csv
import math
import shutil
import subprocess
import logging
from tqdm import tqdm
from PIL import Image
from pathlib import Path
from omegaconf import OmegaConf
import torchvision.transforms.functional as TF
import soundfile as sf
import numpy as np
import decord
from diffusers.utils import export_to_video

from pipeline.ovi_causal_inference import OviCausalInferencePipeline
from utils.misc import set_seed

logging.basicConfig(level=logging.INFO, format="[%(filename)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _process_visual(file_path: str, w=1280, h=704) -> torch.Tensor:
    path = Path(file_path)
    ext = path.suffix.lower()
    
    if ext in {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}:
        try:
            img = Image.open(file_path).convert("RGB")
            pixel_values = TF.to_tensor(img)  # [3, H, W], [0, 1]
        except Exception as e:
            raise RuntimeError(f"Failed to load image {file_path}: {e}")
    else:
        try:
            video_reader = decord.VideoReader(uri=path.as_posix(), num_threads=1)
            frame = video_reader[0].asnumpy()  # [H, W, C]
            pixel_values = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0  # [3, H, W], [0, 1]
        except Exception as e:
            raise RuntimeError(f"Failed to load video first frame {file_path}: {e}")

    C, H_orig, W_orig = pixel_values.shape
    target_aspect = w / h
    orig_aspect = W_orig / H_orig

    if orig_aspect > target_aspect:
        crop_h = H_orig
        crop_w = int(H_orig * target_aspect)
        start_h = 0
        start_w = (W_orig - crop_w) // 2
    else:
        crop_w = W_orig
        crop_h = int(W_orig / target_aspect)
        start_w = 0
        start_h = (H_orig - crop_h) // 2
    
    pixel_values = pixel_values[:, start_h : start_h + crop_h, start_w : start_w + crop_w]
    pixel_values = F.interpolate(pixel_values.unsqueeze(0), size=(h, w), mode='bilinear', align_corners=False, antialias=True)
    
    pixel_values = (pixel_values - 0.5) * 2.0
    return pixel_values.contiguous()

def merge_audio_video(video_path, audio_path, output_path):
    command = [
        'ffmpeg', '-y',
        '-i', video_path,
        '-i', audio_path,
        '-c:v', 'copy',
        '-c:a', 'aac',
        '-shortest',
        output_path,
    ]
    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except Exception as e:
        logger.error(f"FFmpeg for video audio merging failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs/ovi_causal")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--debug_visual", action="store_true", help="for debugging")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)
    
    if not shutil.which('ffmpeg'):
        logger.error("ffmpeg not found.")

    config = OmegaConf.load(args.config_path)

    logger.info("Initializing Ovi Causal Pipeline...")
    pipeline = OviCausalInferencePipeline(config, device=device)
    logger.info(f"Loading causal weights from {args.checkpoint_path}.")
    state_dict = torch.load(args.checkpoint_path, map_location="cpu")
    if "generator_ema" in state_dict.keys():
        sd = state_dict["generator_ema"]
    elif "generator" in state_dict.keys():
        sd = state_dict["generator"]
    else:
        sd = state_dict
    
    logger.info(f"Loaded causal weights from {args.checkpoint_path}.")
    clean_sd = {}
    for k, v in sd.items():
        new_k = k.replace("_fsdp_wrapped_module.", "").replace("_checkpoint_wrapped_module.", "").replace("_orig_mod.", "")
        if new_k.startswith("model."): new_k = new_k[len("model."):]
        clean_sd[new_k] = v
    
    pipeline.generator.model.load_state_dict(clean_sd, strict=True)
    pipeline = pipeline.to(device=device, dtype=torch.bfloat16).eval()
    
    os.makedirs(args.output_dir, exist_ok=True)
    with open(args.csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        samples = list(reader)

    for idx, row in enumerate(tqdm(samples, desc="Processing samples")):
        prompt = row["prompt"]
        visual_path = row["image"]
        current_seed = int(row.get("seed", args.seed))
        set_seed(current_seed)
        processed_frame = _process_visual(visual_path, w=1280, h=704).to(device, dtype=torch.bfloat16)
        if args.debug_visual:
            debug_dir = os.path.join(args.output_dir, "debug_frames")
            os.makedirs(debug_dir, exist_ok=True)
            debug_img = ((processed_frame[0].permute(1, 2, 0).cpu().float() * 0.5 + 0.5) * 255).numpy().astype(np.uint8)
            Image.fromarray(debug_img).save(os.path.join(debug_dir, f"sample_{idx:03d}_input.png"))
            logger.info(f"Debug frame saved for sample {idx}")

        wan22_image_latent = pipeline.vae.encode_video(processed_frame.unsqueeze(2))
        video_noise = torch.randn((1, 32, 48, 704//16, 1280//16), device=device, dtype=torch.bfloat16)
        audio_noise = torch.randn((1, 160, 20), device=device, dtype=torch.bfloat16)

        logger.info(f"Generating sample {idx}: {prompt[:60]}...")
        video_out, audio_out = pipeline.inference(
            noise_video=video_noise,
            noise_audio=audio_noise,
            text_prompts=[prompt],
            wan22_image_latent=wan22_image_latent
        )

        base_name = f"result_{idx:03d}_s{current_seed}"
        temp_v = os.path.join(args.output_dir, f"{base_name}_v.mp4")
        temp_a = os.path.join(args.output_dir, f"{base_name}_a.wav")
        final_p = os.path.join(args.output_dir, f"{base_name}.mp4")

        video_np = video_out[0].permute(0, 2, 3, 1).cpu().float().numpy()
        export_to_video(video_np, temp_v, fps=24)

        audio_np = audio_out.squeeze().cpu().float().numpy()
        sf.write(temp_a, audio_np, config.audio_sample_rate)

        if merge_audio_video(temp_v, temp_a, final_p):
            os.remove(temp_v)
            os.remove(temp_a)
            logger.info(f"Final video merged: {final_p}")

if __name__ == "__main__":
    main()