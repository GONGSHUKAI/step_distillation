# FILE: ovi_fewstep_batch.py (FINAL, WITH MERGING)
from pipeline import OviFewstepInferencePipeline
import argparse
import csv
import math
import os
import shutil
import subprocess
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from diffusers.utils import export_to_video
from omegaconf import OmegaConf
import soundfile as sf
import numpy as np
import logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(filename)s] %(levelname)s: %(message)s"
)

def merge_audio_video(video_path, audio_path, output_path, fps=24):
    """
    使用 FFmpeg 将独立的视频和音频文件合并成一个文件。
    """
    print(f"🔀 开始合并音视频...")
    # 构建 FFmpeg 命令
    # -y: 覆盖已存在的文件
    # -i video_path: 输入视频文件
    # -i audio_path: 输入音频文件
    # -c:v copy: 直接复制视频流，不进行重编码，速度极快且无质量损失
    # -c:a aac: 将音频编码为AAC格式，这是MP4容器的标准
    # -shortest: 当最短的输入流结束时，完成输出
    command = [
        'ffmpeg',
        '-y',
        '-i', video_path,
        '-i', audio_path,
        '-c:v', 'copy',
        '-c:a', 'aac',
        '-shortest',
        output_path,
    ]

    try:
        # 使用subprocess执行命令，并隐藏ffmpeg的输出以保持界面整洁
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"✅ 音视频合并成功! 最终文件: {output_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ FFmpeg 合并失败。")
        print(f"   错误信息: {e.stderr.decode()}")
        return False
    except FileNotFoundError:
        print("❌ FFmpeg 命令未找到。请确保 FFmpeg 已安装并位于系统的 PATH 路径中。")
        return False


def save_audio(waveform: torch.Tensor, path: str, sample_rate: int):
    """保存音频张量到文件"""
    waveform_np = waveform.squeeze().cpu().float().numpy()
    sf.write(path, waveform_np, sample_rate)
    print(f"   - 临时音频已保存至: {path}")

def process_one(pipe: OviFewstepInferencePipeline, prompt, image_path, seed, idx, config):
    """处理CSV文件中的单行数据"""
    output_dir = config.output_dir
    os.makedirs(output_dir, exist_ok=True)
    base_filename = f"line_{idx:03d}_seed_{seed}"
    final_output_path = os.path.join(output_dir, f"{base_filename}_final.mp4")
    temp_video_path = os.path.join(output_dir, f"{base_filename}_temp_video.mp4")
    temp_audio_path = os.path.join(output_dir, f"{base_filename}_temp_audio.wav")

    # --- 图像预处理 ---
    target_w, target_h = config.video_h, config.video_w
    if image_path and os.path.exists(image_path):
        img = Image.open(image_path).convert("RGB")
        orig_w, orig_h = img.size
        max_pixels = 704 * 1280
        if orig_w * orig_h > max_pixels:
            aspect_ratio = orig_w / orig_h
            target_h = int(math.sqrt(max_pixels / aspect_ratio))
            target_w = int(target_h * aspect_ratio)
        else:
            target_w, target_h = orig_w, orig_h

        target_w = (target_w // 32) * 32
        target_h = (target_h // 32) * 32
        print(f"🖼️ 输入图像尺寸: ({orig_w}, {orig_h})，自动调整为: ({target_w}, {target_h})")

        img_resized = img.resize((target_w, target_h), Image.LANCZOS)
        img_tensor = TF.to_tensor(img_resized).sub_(0.5).div_(0.5).unsqueeze(1).to("cuda", dtype=torch.bfloat16)
        wan22_image_latent = pipe.vae.encode_video(img_tensor.unsqueeze(0))
    else:
        raise ValueError(f"TI2AV模式需要有效的图像路径，但 '{image_path}' 不存在。")

    # --- 生成视频和音频的初始噪声 ---
    generator = torch.Generator(device="cuda").manual_seed(seed)
    video_noise = torch.randn(
        (1, (config.video_num_frames - 1) // 4 + 1, 48, target_h // 16, target_w // 16), 
        generator=generator, 
        device="cuda", 
        dtype=torch.bfloat16
    )   # (1, lat_F, lat_C, lat_H, lat_W)
    audio_latent_len, audio_latent_dim = 157, 20
    audio_noise = torch.randn(
        (1, audio_latent_len, audio_latent_dim), generator=generator, device="cuda", dtype=torch.bfloat16
    )

    print("🚀 开始生成音视频...")
    # --- 调用推理流程 ---
    video_out, audio_out = pipe.inference(
        noise_video=video_noise,
        noise_audio=audio_noise,
        text_prompts=[prompt],
        wan22_image_latent=wan22_image_latent,
        video_guidance_scale=config.video_guidance_scale,
        audio_guidance_scale=config.audio_guidance_scale,
        video_negative_prompt=config.video_negative_prompt,
        audio_negative_prompt=config.audio_negative_prompt,
    )


    # --- 保存临时文件 ---
    print("   - 正在保存临时文件...")
    video_np = video_out[0].permute(0, 2, 3, 1).cpu().float().numpy()
    export_to_video(video_np, temp_video_path, fps=24)
    print(f"   - 临时视频已保存至: {temp_video_path}")
    save_audio(audio_out, temp_audio_path, sample_rate=config.audio_sample_rate)

    # --- 合并音视频 ---
    merge_success = merge_audio_video(temp_video_path, temp_audio_path, final_output_path)

    # --- 清理临时文件 ---
    if merge_success:
        print("   - 清理临时文件中...")
        os.remove(temp_video_path)
        os.remove(temp_audio_path)
        print("   - 清理完成。")

def main():
    # --- FFmpeg 依赖检查 ---
    if not shutil.which('ffmpeg'):
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!! 错误: 未找到 FFmpeg。                                     !!!")
        print("!!! 音视频合并功能需要 FFmpeg。请先安装它并确保其在系统PATH中。 !!!")
        print("!!! 例如: sudo apt update && sudo apt install ffmpeg          !!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        return

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--csv", type=str, required=True, help="用于批量推理的CSV文件路径")
    parser.add_argument("--h", type=int, default=704, help="视频的默认高度。在I2V模式下，会根据输入图片自动计算。")
    parser.add_argument("--w", type=int, default=1280, help="视频的默认宽度。在I2V模式下，会根据输入图片自动计算。")
    parser.add_argument("--output_dir", type=str, default="outputs", help="存放生成结果的目录")
    args = parser.parse_args()

    config = OmegaConf.load(args.config_path)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    pipe = OviFewstepInferencePipeline(config)

    print(f"Loading distilled checkpoint from: {args.checkpoint_path}")
    state_dict = torch.load(args.checkpoint_path, map_location="cpu")
    
    if "generator_ema" in state_dict:
        state_dict = state_dict["generator_ema"]
        print("Loaded 'generator_ema' weights.")
    elif "generator" in state_dict:
        state_dict = state_dict["generator"]
        print("Loaded 'generator' weights.")
    else:
        print("Loaded weights directly from checkpoint.")

    cleaned_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace("_fsdp_wrapped_module.", "").replace("_checkpoint_wrapped_module.", "").replace("_orig_mod.", "")
        if new_key.startswith("model."):
            new_key = new_key[len("model."):]
        cleaned_state_dict[new_key] = value

    missing, unexpected = pipe.generator.model.load_state_dict(cleaned_state_dict, strict=False)
    print(f"⚠️ 警告: 加载模型时缺少以下键: {missing}")
    print(f"⚠️ 警告: 加载模型时遇到意外的键: {unexpected}")
    
    pipe = pipe.to(device="cuda", dtype=torch.bfloat16).eval()

    os.makedirs(args.output_dir, exist_ok=True)
    with open(args.csv, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            prompt = row["prompt"]
            image_path = row.get("image")
            seed = int(row.get("seed", 42))
            
            print("\n" + "="*50)
            print(f"🎬 正在处理第 {idx+1} 行: {prompt[:80]}..., seed: {seed}")
            process_one(pipe, prompt, image_path, seed, idx + 1, config)
            print("="*50)

if __name__ == "__main__":
    main()