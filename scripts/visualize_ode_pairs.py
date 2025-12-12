import torch
import matplotlib.pyplot as plt
from wan.modules.vae import WanVAE
import warnings
warnings.filterwarnings("ignore")

# 1. 加载 VAE
vae = WanVAE(vae_pth="/cpfs01/gongshukai/weights/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth", device='cuda')

# 2. 加载 ODE pair
data = torch.load("/cpfs01/gongshukai/step_distillation/ode_init/mixkit_ode/00000.pt")
prompt, latents = list(data.items())[0]
print(prompt, latents.shape)

# 3. 解码成视频帧
with torch.no_grad():
    video = vae.decode(latents.squeeze(0).permute(0, 2, 1, 3, 4).to('cuda'))
    video = torch.stack(video, dim=0)          # [5, 3, 81, 480, 832]
    video = (video + 1) / 2                    # 0-1

sample_frame_interval = 20
video = video[:, :, ::sample_frame_interval]  # [5, 3, 8, 480, 832]
frames = video.shape[2]
noise_level = [1000.0, 937.5, 833.3, 625.0, 96.15]

# 获取单张图片的尺寸 (Height, Width)
img_h, img_w = video.shape[-2], video.shape[-1] # 480, 832
aspect_ratio = img_w / img_h # ~1.73

# 4. 画图
# -------------------- 修改开始 --------------------
# 设定一个基准高度 (例如 2.0 英寸)
unit_height = 2.0
# 根据图片比例计算对应的宽度，确保子图容器和图片比例一致
unit_width = unit_height * aspect_ratio 

# constrained_layout 必须设为 False，否则会强制添加空隙
fig, axes = plt.subplots(frames, len(noise_level),
                         figsize=(len(noise_level) * unit_width, frames * unit_height),
                         squeeze=False,
                         constrained_layout=False) # 关键：关闭自动布局

# 强制消除所有间距
fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
# -------------------- 修改结束 --------------------

for i in range(len(noise_level)):
    for j in range(frames):
        ax = axes[j, i]
        
        # 画图，aspect='auto' 可以强制图片填满格子，但因为我们上面计算了figsize，
        # 这里的 aspect='equal' (默认) 也能完美贴合。
        ax.imshow(video[i, :, j].permute(1, 2, 0).cpu().numpy())
        
        # 移除刻度和轴线
        ax.set_xticks([])
        ax.set_yticks([])
        
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1)
            spine.set_edgecolor('black')

        # ---- 文字 ----
        # 建议加一点 padding 避免字贴着最边缘
        ax.text(0.02, 0.98, f'F={j * sample_frame_interval}',
                transform=ax.transAxes,
                color='white', fontsize=10, va='top', ha='left',
                bbox=dict(boxstyle='square,pad=0.1', fc='black', ec='none', alpha=0.5))

        ax.text(0.5, 0.02, f't={noise_level[i]:.1f}',
                transform=ax.transAxes,
                color='white', fontsize=10, va='bottom', ha='center',
                bbox=dict(boxstyle='square,pad=0.1', fc='black', ec='none', alpha=0.5))

out_path = "/cpfs01/gongshukai/step_distillation/ode_init/ode_pairs_visualization.png"

# pad_inches=0 是最后的保障
plt.savefig(out_path, dpi=100, bbox_inches='tight', pad_inches=0)
print(" saved ->", out_path)