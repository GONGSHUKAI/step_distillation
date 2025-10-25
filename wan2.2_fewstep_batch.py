from pipeline import Wan22FewstepInferencePipeline
from diffusers.utils import export_to_video
from omegaconf import OmegaConf
import argparse
import torch
import os
import csv
import torchvision.transforms.functional as TF
from PIL import Image
import math

def process_one(prompt, image, seed, idx):
    os.makedirs("outputs", exist_ok=True)
    out_path = f"outputs/line_{idx:03d}_video.mp4"

    target_w, target_h = args.w, args.h
    wan22_image_latent = None

    if image is not None and os.path.exists(image):
        img = Image.open(image).convert("RGB")
        orig_w, orig_h = img.size
        max_pixels = 704 * 1280
        ori_pixels = orig_w * orig_h
        if ori_pixels <= max_pixels:
            target_w, target_h = orig_w, orig_h
        else:
            aspect_ratio = orig_w / orig_h
            target_h = int(math.sqrt(max_pixels / aspect_ratio))
            target_w = int(target_h * aspect_ratio)

        target_w = (target_w // 32) * 32
        target_h = (target_h // 32) * 32

        print(f"🖼️ 输入图像尺寸: ({orig_w}, {orig_h})，将自动调整为: ({target_w}, {target_h}) 以匹配长宽比并满足最大像素限制。")

        img_resized = img.resize((target_w, target_h), Image.LANCZOS)
        img_tensor = TF.to_tensor(img_resized).sub_(0.5).div_(0.5).to("cuda").unsqueeze(1).to(dtype=torch.bfloat16)
        wan22_image_latent = pipe.vae.encode_to_latent(img_tensor.unsqueeze(0))
    else:
        if image is not None:
            print(f"⚠️ 警告：找不到图像路径 '{image}'，将使用默认尺寸进行T2V生成。")
        print(f"ℹ️ 未提供输入图像或路径无效，使用默认尺寸: ({target_w}, {target_h})")


    # 使用目标尺寸生成视频
    video = (
        pipe.inference(
            noise=torch.randn(
                1,
                (args.num_frames - 1) // 4 + 1,
                48,
                target_h // 16, # 使用计算出的或默认的高度
                target_w // 16, # 使用计算出的或默认的宽度
                generator=torch.Generator(device="cuda").manual_seed(seed),
                dtype=torch.bfloat16,
                device="cuda",
            ),
            text_prompts=[prompt],
            wan22_image_latent=wan22_image_latent,
        )[0]
        .permute(0, 2, 3, 1)
        .cpu()
        .numpy()
    )
    export_to_video(video, out_path, fps=24)
    print(f"✅ 已生成：{out_path}")

# --- 主程序部分保持不变 ---

parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str)
parser.add_argument("--checkpoint_folder", type=str, default=None)
parser.add_argument("--seed", type=int, default=43)
# 更新了帮助信息以说明其作为默认值的作用
parser.add_argument("--h", type=int, default=704, help="视频的默认高度。在I2V模式下，会根据输入图片自动计算。")
parser.add_argument("--w", type=int, default=1280, help="视频的默认宽度。在I2V模式下，会根据输入图片自动计算。")
parser.add_argument("--num_frames", type=int, default=121)
parser.add_argument("--csv", type=str, default=None, help="用于批量推理的CSV文件路径")
args = parser.parse_args()
assert args.num_frames % 4 == 1, "num_frames必须是4的倍数加1"


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_grad_enabled(False)

config = OmegaConf.load(args.config_path)

pipe = Wan22FewstepInferencePipeline(config)
if args.checkpoint_folder is not None:
    model_path = os.path.join(args.checkpoint_folder, "model.pt")
    model_path2 = os.path.join(args.checkpoint_folder, "Wan2.2-TI2V-5B.pth")
    if os.path.exists(model_path):
        print(f"Loading checkpoint from: {model_path}, format .pt")
        state_dict = torch.load(model_path, map_location="cpu")
    elif os.path.exists(model_path2):
        print(f"Loading checkpoint from: {model_path2}, format .pth")
        state_dict = torch.load(model_path2, map_location="cpu")
    state_dict = state_dict["generator_ema"] if "generator_ema" in state_dict else state_dict
    
    new_state_dict = {}
    for key, value in state_dict.items():
        if not key.startswith("model"):
            key = "model." + key
        new_key = key.replace("_fsdp_wrapped_module.", "")
        new_key = new_key.replace("_checkpoint_wrapped_module.", "")
        new_key = new_key.replace("_orig_mod.", "")
        new_state_dict[new_key] = value
    m, u = pipe.generator.load_state_dict(new_state_dict, strict=False)
    assert len(u) == 0, f"Unexpected keys in state_dict: {u}"
pipe = pipe.to(device="cuda", dtype=torch.bfloat16)

if args.csv is not None:
    with open(args.csv, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            prompt = row["prompt"]
            # 确保image字段为空时传递None
            image = row["image"] if row.get("image") else None
            seed = int(row["seed"])
            print(f"🚀 正在处理第 {idx+1} 行：{prompt[:50]}...")
            process_one(prompt, image, seed, idx+1)