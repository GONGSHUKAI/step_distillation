# Wan2.2 TI2V-5B 步数蒸馏完整流程分析

## 目录
1. [整体架构概述](#1-整体架构概述)
2. [配置文件解析](#2-配置文件解析)
3. [训练入口与分布式设置](#3-训练入口与分布式设置)
4. [DMD核心算法](#4-dmd核心算法)
5. [反向模拟机制](#5-反向模拟机制)
6. [Wan2.2特殊处理](#6-wan22特殊处理)
7. [训练循环详解](#7-训练循环详解)
8. [推理管道](#8-推理管道)

---

## 1. 整体架构概述

### 1.1 核心思想
这个代码仓库实现了 **Distribution Matching Distillation (DMD)** 算法，将 Wan2.2 TI2V-5B 视频生成模型从多步（1000步）蒸馏到少步（4步）。

### 1.2 关键组件

```
训练流程架构:
┌─────────────────────────────────────────────────────────────────┐
│                        训练入口 (train.py)                        │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│              训练器 (Wan22ScoreDistillationTrainer)              │
│  - 初始化分布式环境                                                │
│  - 设置FSDP包装的模型                                              │
│  - 配置优化器和EMA                                                 │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                     DMD 模型 (dmd.py)                            │
│  ┌───────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │  Generator    │  │  Real Score  │  │  Fake Score  │         │
│  │  (Student)    │  │  (Teacher)   │  │  (Critic)    │         │
│  │  ✓ 可训练     │  │  ✗ 冻结      │  │  ✓ 可训练    │         │
│  └───────────────┘  └──────────────┘  └──────────────┘         │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│             反向模拟管道 (BidirectionalTrainingPipeline)           │
│  - 从噪声开始，逐步去噪生成样本                                      │
│  - 随机选择中间步骤进行梯度反向传播                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. 配置文件解析

### 2.1 `self_forcing_wan22_dmd.yaml` 关键参数

```yaml
# ============ 模型配置 ============
real_name: Wan2.2-TI2V-5B      # 教师模型（frozen）
fake_name: Wan2.2-TI2V-5B      # 判别器模型（trainable）
generator_name: Wan2.2-TI2V-5B # 学生模型（trainable）
generator_type: bidirectional   # 双向扩散（非因果）

# ============ 蒸馏步数配置 ============
denoising_step_list: [1000, 750, 500, 250]
# 原始含义：在离散步数索引 [1000, 750, 500, 250] 进行去噪
# warp_denoising_step=true 后，会被映射到连续时间步

warp_denoising_step: true
# 将离散索引映射到连续时间步
# 映射公式：timesteps[1000 - step_index]
# 例如：1000 -> timesteps[0], 750 -> timesteps[250], ...

# ============ 调度器配置 ============
num_train_timestep: 1000       # 总时间步数
timestep_shift: 5.0            # 时间步偏移（用于Flow Matching）
# shift公式：σ_shifted = shift * σ / (1 + (shift - 1) * σ)
# 效果：使噪声分布更集中在中间时间步

guidance_scale: 6.0            # CFG引导尺度（仅用于教师模型）
ts_schedule: false             # 是否使用时间步调度
# false：在 [min_score_timestep, num_train_timestep] 范围随机采样
# true：使用反向模拟得到的时间步范围

# ============ 损失配置 ============
denoising_loss_type: x0        # 损失类型：x0预测损失
# 选项：x0, noise, v, flow
distribution_loss: dmd         # 使用DMD算法

# ============ 训练配置 ============
trainer: score_distillation_wan22  # Wan2.2专用训练器
mixed_precision: true          # 使用bfloat16混合精度
sharding_strategy: full        # FSDP全分片策略
gradient_checkpointing: true   # 梯度检查点（节省显存）

lr: 5.0e-07                    # Generator学习率
lr_critic: 1.0e-07             # Critic学习率
batch_size: 1                  # 单GPU批次大小
total_batch_size: 64           # 全局批次大小（梯度累积）

dfake_gen_update_ratio: 5      # 每5步训练判别器，训练1次生成器
# 步骤模式：D, D, D, D, G, D, D, D, D, G, ...

# ============ EMA配置 ============
ema_weight: 0.99               # EMA衰减率
ema_start_step: 200            # 从第200步开始EMA

# ============ 视频配置（Wan2.2特定）============
image_or_video_shape: [1, 31, 48, 44, 80]
# [batch, frames, channels, height, width]
# 31帧 = 1帧图像 + 30帧视频（Wan2.2的特殊设计）
num_frames: 121                # 原始视频帧数（CSV数据集）
h: 704                         # 视频高度
w: 1280                        # 视频宽度
num_frame_per_block: 31        # 每个处理块的帧数
num_training_frames: 31        # 训练时使用的帧数

# ============ 其他 ============
negative_prompt: "色调艳丽，过曝，静态，细节模糊不清..."
```

### 2.2 参数详解

#### **时间步映射 (`warp_denoising_step`)**

原始配置：`denoising_step_list: [1000, 750, 500, 250]`

这些是**离散的步数索引**，范围是 [0, 1000]。

启用 `warp_denoising_step=true` 后，会进行映射：

```python
# 在 base.py 的 BaseModel.__init__ 中：
timesteps = torch.cat((self.scheduler.timesteps.cpu(), torch.tensor([0])))
self.denoising_step_list = timesteps[1000 - self.denoising_step_list]

# 假设 scheduler.timesteps 是连续的噪声水平 [σ_0, σ_1, ..., σ_999]
# 映射后：
# 1000 -> timesteps[0]   (最高噪声)
# 750  -> timesteps[250]
# 500  -> timesteps[500]
# 250  -> timesteps[750]
# 0    -> timesteps[1000] (无噪声)
```

**为什么需要这个映射？**
- Flow Matching 调度器使用**连续的 σ 值**（噪声水平）而非离散索引
- `timestep_shift=5.0` 会进一步调整 σ 的分布
- 映射确保训练时的时间步与推理时一致

#### **视频帧数配置**

Wan2.2 的特殊之处：

```
原始视频：121帧（从CSV数据集读取）
         ↓ 编码为latent
Latent视频：31帧（1 + 30）
         - 第1帧：图像latent（从首帧重新编码）
         - 后30帧：视频latent
```

这是 Wan2.2 的 **Text-to-Image-to-Video (TI2V)** 架构：
1. 首先生成高质量图像（1帧）
2. 然后将图像扩展为视频（+30帧）

---

## 3. 训练入口与分布式设置

### 3.1 `train.py` - 入口脚本

```python
# train.py 第45-46行
elif config.trainer == "score_distillation_wan22":
    trainer = Wan22ScoreDistillationTrainer(config)
trainer.train()
```

**作用：**
1. 加载配置文件（`self_forcing_wan22_dmd.yaml`）
2. 根据 `trainer: score_distillation_wan22` 实例化专用训练器
3. 调用 `trainer.train()` 开始训练

### 3.2 `wan22_distillation.py` - 训练器初始化

#### **3.2.1 分布式环境设置**

```python
# 第26-47行
torch.backends.cuda.matmul.allow_tf32 = True  # 启用TF32加速
torch.backends.cudnn.allow_tf32 = True

launch_distributed_job()  # 初始化分布式环境
global_rank = dist.get_rank()
self.world_size = dist.get_world_size()
self.dtype = torch.bfloat16  # 混合精度训练

# 设置随机种子（每个rank不同以增加数据多样性）
if config.seed == 0:
    random_seed = torch.randint(0, 10000000, (1,))
    dist.broadcast(random_seed, src=0)  # 从rank 0广播
    config.seed = random_seed.item()
set_seed(config.seed + global_rank)
```

**关键点：**
- 使用 `torch.distributed` 进行多GPU训练
- 每个GPU有不同的随机种子，但通过 `dist.broadcast` 同步关键随机决策
- TF32 加速浮点运算（精度略有损失但速度更快）

#### **3.2.2 模型初始化**

```python
# 第64-73行
if config.distribution_loss == "dmd":
    self.model = DMD(config, device=self.device)

# DMD模型包含3个组件：
# 1. Generator (Student): 需要训练的学生模型
# 2. Real Score (Teacher): 冻结的教师模型
# 3. Fake Score (Critic): 需要训练的判别器
```

#### **3.2.3 FSDP包装**

```python
# 第87-115行
self.model.generator = fsdp_wrap(
    self.model.generator,
    sharding_strategy=config.sharding_strategy,  # "full"
    mixed_precision=config.mixed_precision,      # True
    wrap_strategy=config.generator_fsdp_wrap_strategy  # "size"
)

self.model.real_score = fsdp_wrap(...)  # 教师模型也需要包装（即使冻结）
self.model.fake_score = fsdp_wrap(...)  # 判别器
self.model.text_encoder = fsdp_wrap(
    ...,
    cpu_offload=True  # Text encoder放到CPU节省显存
)
```

**FSDP (Fully Sharded Data Parallel) 原理：**

```
传统DDP：每个GPU持有完整模型副本
┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│   GPU 0     │  │   GPU 1     │  │   GPU 2     │
│ 完整模型 5B │  │ 完整模型 5B │  │ 完整模型 5B │
│ 显存: 20GB  │  │ 显存: 20GB  │  │ 显存: 20GB  │
└─────────────┘  └─────────────┘  └─────────────┘

FSDP：模型参数分片存储，使用时动态聚合
┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│   GPU 0     │  │   GPU 1     │  │   GPU 2     │
│ 参数片1/3   │  │ 参数片2/3   │  │ 参数片3/3   │
│ 显存: 7GB   │  │ 显存: 7GB   │  │ 显存: 7GB   │
└─────────────┘  └─────────────┘  └─────────────┘
     ↓                 ↓                 ↓
     └─────────────────┴─────────────────┘
              前向/反向传播时临时聚合
```

**`sharding_strategy="full"` 的含义：**
- 模型参数、梯度、优化器状态都分片
- 最大化显存节省，但通信开销最大
- 适合5B这种超大模型

#### **3.2.4 优化器设置**

```python
# 第121-135行
self.generator_optimizer = torch.optim.AdamW(
    [param for param in self.model.generator.parameters()
     if param.requires_grad],
    lr=5e-7,        # 较小的学习率
    betas=(0.0, 0.999),  # β1=0 (无动量)，β2=0.999
    weight_decay=config.weight_decay
)

self.critic_optimizer = torch.optim.AdamW(
    [param for param in self.model.fake_score.parameters()
     if param.requires_grad],
    lr=1e-7,        # 判别器学习率更小
    betas=(0.0, 0.999),
    weight_decay=config.weight_decay
)
```

**为什么 β1=0？**
- β1 控制动量（梯度的指数移动平均）
- 蒸馏训练中梯度方向变化较快，不适合使用动量
- β2 仍保留以平滑二阶矩估计

#### **3.2.5 EMA设置**

```python
# 第181-195行
self.ema_weight = 0.99
self.ema_start_step = 200
if (self.step >= self.ema_start_step):
    self.generator_ema = EMA_FSDP(self.model.generator, decay=0.99)
```

**EMA (Exponential Moving Average) 原理：**

```python
# 每次更新后：
ema_params = 0.99 * ema_params + 0.01 * current_params

# 效果：
# - 平滑参数更新，减少训练震荡
# - 推理时使用EMA参数，效果通常更好
# - 从第200步开始（避免早期不稳定影响EMA）
```

---

## 4. DMD核心算法

### 4.1 DMD算法概述

**DMD (Distribution Matching Distillation)** 的核心思想：

```
目标：让学生模型的输出分布匹配教师模型的输出分布

方法：
1. 使用教师模型（Real Score）定义"真实"分布
2. 使用判别器（Fake Score）估计学生模型的分布偏差
3. 通过梯度下降最小化分布差异
```

### 4.2 `dmd.py` - 核心实现

#### **4.2.1 模型初始化**

```python
# dmd.py 第9-53行
class DMD(SelfForcingModel):
    def __init__(self, args, device):
        super().__init__(args, device)  # 初始化3个模型（见base.py）
        
        # Wan2.2特定配置
        self.num_frame_per_block = 31  # 每次处理31帧
        self.num_training_frames = 31  # 训练时使用31帧
        
        # DMD超参数
        self.num_train_timestep = 1000
        self.min_step = 20    # 0.02 * 1000
        self.max_step = 980   # 0.98 * 1000
        self.real_guidance_scale = 6.0  # 教师模型CFG
        self.fake_guidance_scale = 0.0  # 判别器不使用CFG
        self.timestep_shift = 5.0
        
        # 启用梯度检查点（节省显存）
        if args.gradient_checkpointing:
            self.generator.enable_gradient_checkpointing()
            self.fake_score.enable_gradient_checkpointing()
```

**梯度检查点原理：**

```
不使用梯度检查点：
前向传播时保存所有中间激活值用于反向传播
显存占用 ∝ 层数 × batch_size × 激活值大小

使用梯度检查点：
前向传播时只保存部分检查点
反向传播时从检查点重新计算中间激活值
显存占用 ↓ 50-70%，计算时间 ↑ 20-30%
```

#### **4.2.2 KL梯度计算 (`_compute_kl_grad`)**

这是DMD算法的**核心**，实现论文中的公式 7：

```python
# dmd.py 第54-164行
def _compute_kl_grad(
    self, 
    noisy_image_or_video: torch.Tensor,      # [B, F, C, H, W]
    estimated_clean_image_or_video: torch.Tensor,  # [B, F, C, H, W]
    timestep: torch.Tensor,                  # [B, F]
    conditional_dict: dict,                  # 文本条件
    unconditional_dict: dict,                # 负提示词
    normalization: bool = True,
    wan22_image_latent = None,              # Wan2.2首帧latent
) -> Tuple[torch.Tensor, dict]:
```

**步骤1：Wan2.2特殊处理（第77-90行）**

```python
if "2.2" in self.generator.model_name:
    # 创建mask：第1帧=0, 其余帧=1
    mask1, mask2 = masks_like(noisy_image_or_video, zero=True)
    mask2 = torch.stack(mask2, dim=0)  # [1, 31, 48, 44, 80]
    
    # 混合：保持首帧为wan22_image_latent，其余帧为noisy
    noisy_image_or_video = (1. - mask2) * wan22_image_latent + mask2 * noisy_image_or_video
    
    # 构造时间步输入（Wan2.2需要特殊的序列格式）
    wan22_input_timestep = torch.tensor([timestep[0][0].item()])
    temp_ts = (mask2[:, :, 0, ::2, ::2] * wan22_input_timestep)
    temp_ts = temp_ts.reshape(temp_ts.shape[0], -1)
    # 填充到seq_len=27280
    temp_ts = torch.cat([temp_ts, temp_ts.new_ones(self.generator.seq_len - temp_ts.size(1)) * wan22_input_timestep], dim=1)
    wan22_input_timestep = temp_ts.to(dtype=torch.long)
```

**为什么需要mask？**
- Wan2.2的TI2V架构：首帧是图像，后续帧是视频
- 训练时首帧需要保持为干净的图像latent
- 只对后30帧添加噪声并进行去噪训练

**步骤2：计算判别器（Fake Score）预测（第92-119行）**

```python
# Fake Score条件预测
_, pred_fake_image_cond = self.fake_score(
    noisy_image_or_video=noisy_image_or_video,
    conditional_dict=conditional_dict,
    timestep=timestep,
    wan22_input_timestep=wan22_input_timestep,
    mask2=mask2,
    wan22_image_latent=wan22_image_latent,
)

# Fake Score无条件预测（如果使用CFG）
if self.fake_guidance_scale != 0.0:
    _, pred_fake_image_uncond = self.fake_score(
        noisy_image_or_video=noisy_image_or_video,
        conditional_dict=unconditional_dict,  # 负提示词
        ...
    )
    pred_fake_image = pred_fake_image_cond + (
        pred_fake_image_cond - pred_fake_image_uncond
    ) * self.fake_guidance_scale
else:
    pred_fake_image = pred_fake_image_cond  # 不使用CFG
```

**CFG (Classifier-Free Guidance) 原理：**

```
条件预测：pred_cond = model(x, condition)
无条件预测：pred_uncond = model(x, null_condition)
引导预测：pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)

效果：
- guidance_scale > 1: 增强条件的影响
- guidance_scale = 1: 标准条件预测
- guidance_scale = 0: 忽略条件
```

**步骤3：计算教师模型（Real Score）预测（第121-148行）**

```python
# Real Score条件预测
_, pred_real_image_cond = self.real_score(
    noisy_image_or_video=noisy_image_or_video,
    conditional_dict=conditional_dict,
    timestep=timestep,
    ...
)

# Real Score无条件预测
_, pred_real_image_uncond = self.real_score(
    noisy_image_or_video=noisy_image_or_video,
    conditional_dict=unconditional_dict,
    ...
)

# 应用CFG (guidance_scale=6.0)
pred_real_image = pred_real_image_cond + (
    pred_real_image_cond - pred_real_image_uncond
) * self.real_guidance_scale  # 6.0
```

**为什么教师用CFG而判别器不用？**
- 教师模型：需要高质量输出，CFG=6.0 增强文本条件
- 判别器：只需要估计分布差异，不需要CFG

**步骤4：计算DMD梯度（第150-159行）**

```python
# DMD论文公式7
grad = (pred_fake_image - pred_real_image)

# 梯度归一化（DMD论文公式8）
if normalization:
    p_real = (estimated_clean_image_or_video - pred_real_image)
    normalizer = torch.abs(p_real).mean(dim=[1, 2, 3, 4], keepdim=True)
    grad = grad / normalizer

grad = torch.nan_to_num(grad)  # 处理NaN
```

**归一化的作用：**
- 防止梯度爆炸/消失
- 使不同时间步的梯度尺度一致
- `normalizer` 基于教师模型的预测误差

#### **4.2.3 分布匹配损失 (`compute_distribution_matching_loss`)**

```python
# dmd.py 第166-238行
def compute_distribution_matching_loss(
    self,
    image_or_video: torch.Tensor,          # 生成器输出 [B, F, C, H, W]
    conditional_dict: dict,
    unconditional_dict: dict,
    gradient_mask: Optional[torch.Tensor] = None,
    denoised_timestep_from: int = 0,
    denoised_timestep_to: int = 0,
    ...
) -> Tuple[torch.Tensor, dict]:
```

**步骤1：随机采样时间步（第193-211行）**

```python
with torch.no_grad():  # 这部分不需要梯度
    # 确定时间步范围
    min_timestep = denoised_timestep_to if self.ts_schedule else self.min_score_timestep
    max_timestep = denoised_timestep_from if self.ts_schedule_max else self.num_train_timestep
    
    # 随机采样时间步（均匀分布）
    timestep = self._get_timestep(
        min_timestep, max_timestep,
        batch_size, num_frame,
        self.num_frame_per_block,
        uniform_timestep=True  # 所有帧使用相同时间步
    )
    
    # 应用timestep_shift
    if self.timestep_shift > 1:
        timestep = self.timestep_shift * (timestep / 1000) / \
                   (1 + (self.timestep_shift - 1) * (timestep / 1000)) * 1000
    
    timestep = timestep.clamp(self.min_step, self.max_step)  # [20, 980]
```

**`ts_schedule` 的作用：**
- `ts_schedule=False`（当前配置）：在整个范围 [20, 980] 随机采样
- `ts_schedule=True`：使用反向模拟的时间步范围
  - 例如：反向模拟在步骤500退出，则采样范围为 [250, 500]
  - 优点：训练时间步与推理时间步分布一致

**步骤2：添加噪声（第213-218行）**

```python
    noise = torch.randn_like(image_or_video)
    noisy_latent = self.scheduler.add_noise(
        image_or_video.flatten(0, 1),      # [B*F, C, H, W]
        noise.flatten(0, 1),
        timestep.flatten(0, 1)
    ).detach().unflatten(0, (batch_size, num_frame))
```

**噪声添加公式（Flow Matching）：**

```python
# scheduler.add_noise 的实现（scheduler.py 第159-176行）
sigma = self.sigmas[timestep_id]  # 根据timestep查找对应的σ
noisy_sample = (1 - sigma) * original_sample + sigma * noise

# Flow Matching 噪声调度：
# t=0:   σ=0.003   (几乎无噪声)
# t=500: σ=0.5     (中等噪声)
# t=1000: σ=0.997  (几乎全噪声)
```

**步骤3：计算KL梯度（第220-230行）**

```python
    # 计算判别器和教师模型的预测差异
    grad, dmd_log_dict = self._compute_kl_grad(
        noisy_image_or_video=noisy_latent,
        estimated_clean_image_or_video=image_or_video,  # 原始生成结果
        timestep=timestep,
        conditional_dict=conditional_dict,
        unconditional_dict=unconditional_dict,
        ...
    )
```

**步骤4：计算最终损失（第232-237行）**

```python
# DMD损失：最小化 (x - (x - grad))^2
if gradient_mask is not None:
    dmd_loss = 0.5 * F.mse_loss(
        image_or_video.double()[gradient_mask],
        (image_or_video.double() - grad.double()).detach()[gradient_mask],
        reduction="mean"
    )
else:
    dmd_loss = 0.5 * F.mse_loss(
        image_or_video.double(),
        (image_or_video.double() - grad.double()).detach(),
        reduction="mean"
    )
```

**损失函数解析：**

```
损失：L = 0.5 * ||x - (x - grad)||^2
    = 0.5 * ||grad||^2
    
其中：
x = 生成器输出
grad = pred_fake - pred_real (KL梯度)

目标：
- 最小化 grad 的模长
- 即：让 pred_fake ≈ pred_real
- 即：让判别器预测 ≈ 教师模型预测
- 即：让学生模型输出分布 ≈ 教师模型输出分布
```

**gradient_mask 的作用：**
- 只对生成的新帧计算损失
- 不对首帧（图像latent）或上下文帧计算损失
- 见 `base.py` 第190-198行

#### **4.2.4 生成器损失 (`generator_loss`)**

```python
# dmd.py 第240-289行
def generator_loss(
    self,
    image_or_video_shape,        # [1, 31, 48, 44, 80]
    conditional_dict: dict,
    unconditional_dict: dict,
    clean_latent: torch.Tensor,  # 实际为None（使用反向模拟）
    initial_latent: torch.Tensor = None,
    ...
) -> Tuple[torch.Tensor, dict]:
    
    # 步骤1：通过反向模拟生成fake样本
    pred_image, gradient_mask, denoised_timestep_from, denoised_timestep_to = \
        self._run_generator(
            image_or_video_shape=image_or_video_shape,
            conditional_dict=conditional_dict,
            initial_latent=initial_latent,
            ...
        )
    
    # 步骤2：计算DMD损失
    dmd_loss, dmd_log_dict = self.compute_distribution_matching_loss(
        image_or_video=pred_image,
        conditional_dict=conditional_dict,
        unconditional_dict=unconditional_dict,
        gradient_mask=gradient_mask,
        denoised_timestep_from=denoised_timestep_from,
        denoised_timestep_to=denoised_timestep_to,
        ...
    )
    
    return dmd_loss, dmd_log_dict
```

#### **4.2.5 判别器损失 (`critic_loss`)**

```python
# dmd.py 第291-410行
def critic_loss(
    self,
    image_or_video_shape,
    conditional_dict: dict,
    unconditional_dict: dict,
    ...
) -> Tuple[torch.Tensor, dict]:
    
    # 步骤1：生成fake样本（不计算梯度）
    with torch.no_grad():
        generated_image, _, denoised_timestep_from, denoised_timestep_to = \
            self._run_generator(
                image_or_video_shape=image_or_video_shape,
                conditional_dict=conditional_dict,
                ...
            )
    
    # 步骤2：随机采样时间步
    critic_timestep = self._get_timestep(...)
    if self.timestep_shift > 1:
        critic_timestep = ...  # 应用shift
    critic_timestep = critic_timestep.clamp(self.min_step, self.max_step)
    
    # 步骤3：添加噪声
    critic_noise = torch.randn_like(generated_image)
    noisy_generated_image = self.scheduler.add_noise(
        generated_image.flatten(0, 1),
        critic_noise.flatten(0, 1),
        critic_timestep.flatten(0, 1)
    ).unflatten(0, image_or_video_shape[:2])
    
    # 步骤4：Wan2.2特殊处理
    if "2.2" in self.generator.model_name:
        mask1, mask2 = masks_like(noisy_generated_image, zero=True)
        mask2 = torch.stack(mask2, dim=0)
        noisy_generated_image = (1. - mask2) * wan22_image_latent + mask2 * noisy_generated_image
        ...
    
    # 步骤5：判别器预测
    flow_pred, pred_fake_image = self.fake_score(
        noisy_image_or_video=noisy_generated_image,
        conditional_dict=conditional_dict,
        timestep=critic_timestep,
        ...
    )
    
    # 步骤6：计算去噪损失（x0预测）
    denoising_loss = self.denoising_loss_func(
        x=generated_image.flatten(0, 1),            # 真实的clean样本
        x_pred=pred_fake_image.flatten(0, 1),      # 判别器预测
        noise=critic_noise.flatten(0, 1),
        noise_pred=None,  # x0损失不需要
        alphas_cumprod=self.scheduler.alphas_cumprod,
        timestep=critic_timestep.flatten(0, 1),
        flow_pred=None,
    )
    
    return denoising_loss, critic_log_dict
```

**判别器训练目标：**
1. 输入：生成器生成的样本（加噪声）
2. 输出：预测clean样本
3. 损失：`||生成样本 - 预测clean样本||^2`
4. 目的：让判别器学会准确估计样本质量

**为什么判别器需要训练？**
- 判别器需要适应生成器的变化
- 生成器改进 → 生成样本分布变化 → 判别器需要重新学习

---

## 5. 反向模拟机制

### 5.1 反向模拟概述

**问题：** 如何在训练时生成样本？

**传统方法：** 从数据集读取真实样本
**DMD方法：** 从噪声开始，使用生成器逐步去噪

**为什么使用反向模拟？**
1. **数据无关性**：不需要大量训练数据（prompt即可）
2. **分布一致性**：训练时的采样过程 = 推理时的采样过程
3. **端到端**：整个流程可微分

### 5.2 `base.py::_run_generator` - 生成器运行逻辑

```python
# base.py 第117-201行
def _run_generator(
    self,
    image_or_video_shape,         # [1, 31, 48, 44, 80]
    conditional_dict: dict,
    initial_latent: torch.tensor = None,
    ...
) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
```

**步骤1：确定生成帧数（第148-165行）**

```python
# Wan2.2特定：31帧 = 1帧图像 + 30帧视频
latent_frames_num = 31  # for Wan2.2
min_num_frames = 31     # 最少生成31帧
max_num_frames = 31     # 最多生成31帧（配置中）

# 随机采样生成块数（这里固定为1块=31帧）
max_num_blocks = max_num_frames // self.num_frame_per_block  # 1
min_num_blocks = min_num_frames // self.num_frame_per_block  # 1
num_generated_blocks = torch.randint(min_num_blocks, max_num_blocks + 1, (1,))
dist.broadcast(num_generated_blocks, src=0)  # 同步到所有GPU
num_generated_frames = num_generated_blocks.item() * self.num_frame_per_block  # 31

noise_shape[1] = num_generated_frames
```

**为什么需要随机采样帧数？**
- 原始设计支持可变长度视频生成（21-81帧）
- Wan2.2简化为固定31帧
- 但代码保留了可扩展性

**步骤2：反向模拟（第167-174行）**

```python
pred_image_or_video, denoised_timestep_from, denoised_timestep_to = \
    self._consistency_backward_simulation(
        noise=torch.randn(noise_shape, device=self.device, dtype=self.dtype),
        clip_fea=clip_fea,
        y=y,
        wan22_image_latent=wan22_image_latent,
        **conditional_dict
    )
```

这会调用 `BidirectionalTrainingPipeline.inference_with_trajectory`

**步骤3：处理长视频（第176-188行）**

```python
# 如果生成超过31帧，需要重新编码首帧
if pred_image_or_video.shape[1] > latent_frames_num:  # 31
    with torch.no_grad():
        # 取前面的帧解码为像素
        latent_to_decode = pred_image_or_video[:, :-(latent_frames_num-1), ...]
        pixels = self.vae.decode_to_pixel(latent_to_decode)
        
        # 取最后一帧重新编码为图像latent
        frame = pixels[:, -1:, ...].to(self.dtype)
        frame = rearrange(frame, "b t c h w -> b c t h w")
        image_latent = self.vae.encode_to_latent(frame).to(self.dtype)
    
    # 拼接：新的图像latent + 后30帧视频latent
    pred_image_or_video_last_clip = torch.cat([
        image_latent, 
        pred_image_or_video[:, -(latent_frames_num-1):, ...]
    ], dim=1)
else:
    pred_image_or_video_last_clip = pred_image_or_video
```

**为什么需要这个操作？**
- Wan2.2的TI2V架构要求首帧是图像latent
- 长视频生成时，滑动窗口会生成多个31帧片段
- 每个片段的首帧需要重新编码以保持一致性

**步骤4：创建梯度掩码（第190-198行）**

```python
if num_generated_frames != min_num_frames:
    gradient_mask = torch.ones_like(pred_image_or_video_last_clip, dtype=torch.bool)
    if self.args.independent_first_frame:
        gradient_mask[:, :1] = False  # 不对首帧计算梯度
    else:
        gradient_mask[:, :self.num_frame_per_block] = False
else:
    gradient_mask = None
```

### 5.3 `bidirectional_training.py` - 反向模拟实现

```python
# bidirectional_training.py 第42-125行
def inference_with_trajectory(
    self, 
    noise: torch.Tensor,          # [B, 31, 48, 44, 80]
    clip_fea, y, 
    wan22_image_latent,
    **conditional_dict
) -> torch.Tensor:
```

**步骤1：随机选择退出步骤（第57行）**

```python
num_denoising_steps = len(self.denoising_step_list)  # 4步
exit_flags = self.generate_and_sync_list(num_denoising_steps, device=noise.device)
# exit_flags: 随机选择 [0, 1, 2, 3] 中的一个
```

**`generate_and_sync_list` 的作用（第25-40行）：**

```python
def generate_and_sync_list(self, num_denoising_steps, device):
    rank = dist.get_rank()
    
    if rank == 0:
        # Rank 0生成随机索引
        indices = torch.randint(0, num_denoising_steps, size=(1,), device=device)
    else:
        indices = torch.empty(1, dtype=torch.long, device=device)
    
    dist.broadcast(indices, src=0)  # 广播到所有GPU
    return indices.tolist()
```

**为什么需要随机退出？**
- 训练时只对**一个时间步**计算梯度
- 随机选择避免过拟合特定时间步
- 提高训练效率（不需要在每个时间步都反向传播）

**步骤2：去噪循环（第60-113行）**

```python
noisy_image_or_video = noise  # 初始为纯噪声

for index, current_timestep in enumerate(self.denoising_step_list):
    # 判断是否在当前步骤退出
    exit_flag = (index == exit_flags[0])
    
    # 构造时间步张量 [B, F]
    timestep = torch.ones(noise.shape[:2], device=noise.device, dtype=torch.int64) * current_timestep
    
    # Wan2.2特殊处理
    if "2.2" in self.generator.model_name:
        mask1, mask2 = masks_like(noisy_image_or_video, zero=True)
        mask2 = torch.stack(mask2, dim=0)
        
        # 混合首帧
        noisy_image_or_video = (1. - mask2) * wan22_image_latent + mask2 * noisy_image_or_video
        
        # 构造wan22_input_timestep
        wan22_input_timestep = torch.tensor([timestep[0][0].item()])
        temp_ts = (mask2[:, :, 0, ::2, ::2] * wan22_input_timestep)
        temp_ts = temp_ts.reshape(temp_ts.shape[0], -1)
        temp_ts = torch.cat([
            temp_ts, 
            temp_ts.new_ones(self.generator.seq_len - temp_ts.size(1)) * wan22_input_timestep
        ], dim=1)
        wan22_input_timestep = temp_ts.to(dtype=torch.long)
    
    if not exit_flag:
        # 非退出步骤：不计算梯度
        with torch.no_grad():
            _, denoised_pred = self.generator(
                noisy_image_or_video=noisy_image_or_video,
                conditional_dict=conditional_dict,
                timestep=timestep,
                wan22_input_timestep=wan22_input_timestep,
                mask2=mask2,
                wan22_image_latent=wan22_image_latent,
            )
            
            # 加噪声到下一个时间步
            next_timestep = self.denoising_step_list[index + 1]
            noisy_image_or_video = self.scheduler.add_noise(
                denoised_pred.flatten(0, 1),
                torch.randn_like(denoised_pred.flatten(0, 1)),
                next_timestep * torch.ones([batch_size * num_frames])
            ).unflatten(0, denoised_pred.shape[:2])
    else:
        # 退出步骤：计算梯度
        _, denoised_pred = self.generator(
            noisy_image_or_video=noisy_image_or_video,
            conditional_dict=conditional_dict,
            timestep=timestep,
            wan22_input_timestep=wan22_input_timestep,
            mask2=mask2,
            wan22_image_latent=wan22_image_latent,
        )
        break  # 退出循环
```

**去噪过程示意：**

```
假设 denoising_step_list = [1000, 750, 500, 250]
假设 exit_flags = [2] （在第2步退出）

步骤0: t=1000 (最高噪声)
├─ 生成器预测 x0
├─ 加噪声到 t=750
└─ no_grad (不计算梯度)

步骤1: t=750
├─ 生成器预测 x0
├─ 加噪声到 t=500
└─ no_grad (不计算梯度)

步骤2: t=500 ← 退出步骤
├─ 生成器预测 x0
└─ 计算梯度 (requires_grad=True)

步骤3: t=250 (跳过)
```

**步骤3：计算退出时间步（第115-123行）**

```python
if exit_flags[0] == len(self.denoising_step_list) - 1:
    # 最后一步退出
    denoised_timestep_to = 0
    denoised_timestep_from = 1000 - torch.argmin(
        (self.scheduler.timesteps.cuda() - self.denoising_step_list[exit_flags[0]].cuda()).abs()
    ).item()
else:
    # 中间步骤退出
    denoised_timestep_to = 1000 - torch.argmin(
        (self.scheduler.timesteps.cuda() - self.denoising_step_list[exit_flags[0] + 1].cuda()).abs()
    ).item()
    denoised_timestep_from = 1000 - torch.argmin(
        (self.scheduler.timesteps.cuda() - self.denoising_step_list[exit_flags[0]].cuda()).abs()
    ).item()

return denoised_pred, denoised_timestep_from, denoised_timestep_to
```

**时间步范围的用途：**
- 返回的 `[denoised_timestep_to, denoised_timestep_from]` 表示当前去噪的范围
- 例如：从 t=750 去噪到 t=500
  - `denoised_timestep_from` = 反向查找 750 对应的索引
  - `denoised_timestep_to` = 反向查找 500 对应的索引
- 这个范围可用于 `ts_schedule` 来限制判别器的时间步采样

---

## 6. Wan2.2特殊处理

### 6.1 Wan2.2架构概述

**Wan2.2 (Text-to-Image-to-Video)：**

```
输入文本 → 生成图像(1帧) → 扩展为视频(+30帧)
                ↓                    ↓
          Image Latent        Video Latent
           [1, 48, H, W]      [30, 48, H, W]
                └──────────────────┬─────────┘
                          Concatenate
                                 ↓
                        [31, 48, H, W]
```

### 6.2 Mask生成 (`dataset.py::masks_like`)

```python
# dataset.py 第312-329行
def masks_like(tensor, zero=False, generator=None, p=0.2):
    out1 = [torch.ones(u.shape, dtype=u.dtype, device=u.device) for u in tensor]
    out2 = [torch.ones(u.shape, dtype=u.dtype, device=u.device) for u in tensor]
    
    if zero:
        if generator is not None:
            # 训练时随机dropout（概率p=0.2）
            for u, v in zip(out1, out2):
                random_num = torch.rand(1, generator=generator, device=generator.device).item()
                if random_num < p:
                    u[0, :] = torch.normal(mean=-3.5, std=0.5, size=(1,)).exp()
                    v[0, :] = torch.zeros_like(v[0, :])
                else:
                    u[0, :] = u[0, :]
                    v[0, :] = v[0, :]
        else:
            # 推理时固定mask
            for u, v in zip(out1, out2):
                u[0, :] = torch.zeros_like(u[0, :])
                v[0, :] = torch.zeros_like(v[0, :])
    
    return out1, out2
```

**Mask含义：**
- `mask2[0, :, :, :]` = 0 （第1帧）
- `mask2[1:, :, :, :]` = 1 （后30帧）

**Dropout机制：**
- 训练时以20%概率将首帧mask设为随机小值（`exp(-3.5)≈0.03`）
- 目的：增强模型对首帧质量的鲁棒性
- 推理时mask固定为0（完全保留首帧）

### 6.3 Wan2.2时间步构造

**问题：** Wan2.2的Transformer需要每个位置都有时间步输入

```python
# bidirectional_training.py 第73-77行
wan22_input_timestep = torch.tensor([timestep[0][0].item()])  # 标量
temp_ts = (mask2[:, :, 0, ::2, ::2] * wan22_input_timestep)  # 广播到所有位置
temp_ts = temp_ts.reshape(temp_ts.shape[0], -1)  # 展平
temp_ts = torch.cat([
    temp_ts, 
    temp_ts.new_ones(self.generator.seq_len - temp_ts.size(1)) * wan22_input_timestep
], dim=1)  # 填充到seq_len=27280
wan22_input_timestep = temp_ts.to(dtype=torch.long)
```

**为什么需要这么复杂？**
- Wan2.2将视频展平为序列：[B, 27280]
- 27280 = 31帧 × 880个patch
- 每个patch位置需要一个时间步输入
- `mask2[:, :, 0, ::2, ::2]` 下采样到patch分辨率

### 6.4 Wan2.2混合逻辑

```python
# 在多处代码中出现
noisy_image_or_video = (1. - mask2) * wan22_image_latent + mask2 * noisy_image_or_video

# 等价于：
# frame 0:  noisy[0] = wan22_image_latent[0] (保持首帧)
# frame 1-30: noisy[1:] = noisy[1:] (正常加噪)
```

---

## 7. 训练循环详解

### 7.1 训练主循环 (`wan22_distillation.py::train`)

```python
# wan22_distillation.py 第386-466行
def train(self):
    start_step = self.step
    
    while True:
        TRAIN_GENERATOR = self.step % self.config.dfake_gen_update_ratio == 0
        # dfake_gen_update_ratio = 5
        # 步骤: D, D, D, D, G, D, D, D, D, G, ...
        
        # ========== 训练生成器 ==========
        if TRAIN_GENERATOR:
            self.generator_optimizer.zero_grad(set_to_none=True)
            
            batch = next(self.dataloader)
            generator_log_dict = self.fwdbwd_one_step(batch, train_generator=True)
            
            if not self.config.debug:
                self.generator_optimizer.step()
                if self.generator_ema is not None:
                    self.generator_ema.update(self.model.generator)
        
        # ========== 训练判别器 ==========
        self.critic_optimizer.zero_grad(set_to_none=True)
        
        batch = next(self.dataloader)
        critic_log_dict = self.fwdbwd_one_step(batch, train_generator=False)
        
        if not self.config.debug:
            self.critic_optimizer.step()
        
        # ========== 更新步数 ==========
        self.step += 1
        
        # ========== 创建EMA（延迟初始化）==========
        if (self.step >= self.ema_start_step) and \
           (self.generator_ema is None) and (self.ema_weight > 0):
            self.generator_ema = EMA_FSDP(self.model.generator, decay=self.ema_weight)
        
        # ========== 保存检查点 ==========
        if (not self.config.no_save) and (self.step % self.config.log_iters == 0):
            self.save()
        
        # ========== 日志记录 ==========
        if self.is_main_process:
            wandb_loss_dict = {...}
            wandb.log(wandb_loss_dict, step=self.step)
        
        # ========== 垃圾回收 ==========
        if self.step % self.config.gc_interval == 0:
            gc.collect()
            torch.cuda.empty_cache()
```

### 7.2 单步前向反向 (`fwdbwd_one_step`)

```python
# wan22_distillation.py 第266-354行
def fwdbwd_one_step(self, batch, train_generator):
    self.model.eval()  # 防止dropout等随机性
    
    # ========== 数据准备 ==========
    text_prompts = batch["prompts"]
    video_tensor = batch["video"].to(device=self.device, dtype=self.dtype)
    
    # 提取首帧并编码为image latent
    first_frame = video_tensor[:, :, :1, :, :]  # [B, C, 1, H, W]
    wan22_image_latent = self.model.vae.encode_to_latent(first_frame)
    
    batch_size = len(text_prompts)
    image_or_video_shape = [batch_size, 31, 48, 44, 80]
    
    # ========== 文本编码（缓存）==========
    with torch.no_grad():
        conditional_dict = self.model.text_encoder(text_prompts=text_prompts)
        
        if not getattr(self, "unconditional_dict", None):
            unconditional_dict = self.model.text_encoder(
                text_prompts=[self.config.negative_prompt] * batch_size
            )
            self.unconditional_dict = unconditional_dict  # 缓存负提示词embedding
        else:
            unconditional_dict = self.unconditional_dict
    
    # ========== 训练生成器 ==========
    if train_generator:
        generator_loss, generator_log_dict = self.model.generator_loss(
            image_or_video_shape=image_or_video_shape,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            clean_latent=None,  # 不使用真实数据
            wan22_image_latent=wan22_image_latent,
            ...
        )
        
        generator_loss.backward()
        generator_grad_norm = self.model.generator.clip_grad_norm_(
            self.max_grad_norm_generator  # 10.0
        )
        
        return {"generator_loss": generator_loss, "generator_grad_norm": generator_grad_norm}
    
    # ========== 训练判别器 ==========
    critic_loss, critic_log_dict = self.model.critic_loss(
        image_or_video_shape=image_or_video_shape,
        conditional_dict=conditional_dict,
        unconditional_dict=unconditional_dict,
        clean_latent=None,
        wan22_image_latent=wan22_image_latent,
        ...
    )
    
    critic_loss.backward()
    critic_grad_norm = self.model.fake_score.clip_grad_norm_(
        self.max_grad_norm_critic  # 10.0
    )
    
    return {"critic_loss": critic_loss, "critic_grad_norm": critic_grad_norm}
```

### 7.3 完整训练流程图

```
训练步骤 n:
┌─────────────────────────────────────────────────────────────┐
│ 1. 数据加载                                                   │
│    - 从CSV读取prompt和视频路径                                 │
│    - 加载视频并编码为latent                                    │
│    - 提取首帧并重新编码为wan22_image_latent                     │
└────────────────────────┬────────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. 文本编码（缓存）                                            │
│    - Text Encoder → conditional_dict                         │
│    - Negative Prompt → unconditional_dict (缓存)             │
└────────────────────────┬────────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ 3a. 生成器前向（每5步）                                         │
│    ┌──────────────────────────────────────────────────┐     │
│    │ 反向模拟：                                          │     │
│    │  - 从噪声开始                                       │     │
│    │  - 随机选择退出步骤（0-3）                          │     │
│    │  - 使用Generator逐步去噪                            │     │
│    │  - 在退出步骤计算梯度                               │     │
│    │  → pred_image [B, 31, 48, 44, 80]                 │     │
│    └──────────────────────────────────────────────────┘     │
│    ┌──────────────────────────────────────────────────┐     │
│    │ DMD损失计算：                                       │     │
│    │  - 随机采样时间步t                                  │     │
│    │  - 对pred_image加噪声 → noisy_latent              │     │
│    │  - Real Score预测：pred_real                       │     │
│    │  - Fake Score预测：pred_fake                       │     │
│    │  - KL梯度：grad = pred_fake - pred_real           │     │
│    │  - 损失：L = 0.5 * ||grad||^2                     │     │
│    └──────────────────────────────────────────────────┘     │
│    ┌──────────────────────────────────────────────────┐     │
│    │ 生成器反向：                                        │     │
│    │  - L.backward()                                    │     │
│    │  - clip_grad_norm_(10.0)                          │     │
│    │  - optimizer.step()                                │     │
│    │  - EMA更新 (如果步数 >= 200)                       │     │
│    └──────────────────────────────────────────────────┘     │
└────────────────────────┬────────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ 3b. 判别器前向（每步）                                          │
│    ┌──────────────────────────────────────────────────┐     │
│    │ 生成样本（no_grad）：                               │     │
│    │  - 反向模拟生成 generated_image                    │     │
│    └──────────────────────────────────────────────────┘     │
│    ┌──────────────────────────────────────────────────┐     │
│    │ 判别器损失计算：                                    │     │
│    │  - 随机采样时间步t                                  │     │
│    │  - 对generated_image加噪声 → noisy_generated      │     │
│    │  - Fake Score预测：pred_fake_image                │     │
│    │  - 去噪损失：L = ||generated_image - pred_fake||^2│     │
│    └──────────────────────────────────────────────────┘     │
│    ┌──────────────────────────────────────────────────┐     │
│    │ 判别器反向：                                        │     │
│    │  - L.backward()                                    │     │
│    │  - clip_grad_norm_(10.0)                          │     │
│    │  - optimizer.step()                                │     │
│    └──────────────────────────────────────────────────┘     │
└────────────────────────┬────────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. 日志与保存                                                 │
│    - 每步记录损失到W&B                                         │
│    - 每500步保存检查点                                         │
│    - 定期垃圾回收                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 8. 推理管道

### 8.1 推理入口 (`wan22_fewstep_inference.py`)

```python
# wan22_fewstep_inference.py 第11-50行
class Wan22FewstepInferencePipeline(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        
        # 初始化模型
        self.generator = WanDiffusionWrapper(model_name="Wan2.2-TI2V-5B", is_causal=False)
        self.text_encoder = WanTextEncoder(model_name="Wan2.2-TI2V-5B")
        self.vae = Wan2_2_VAEWrapper()
        
        # 设置去噪步数
        self.denoising_step_list = torch.tensor(args.denoising_step_list)
        self.scheduler = self.generator.get_scheduler()
        
        # 如果启用warp
        if args.warp_denoising_step:
            timesteps = torch.cat((self.scheduler.timesteps.cpu(), torch.tensor([0])))
            self.denoising_step_list = timesteps[1000 - self.denoising_step_list]
```

### 8.2 推理过程 (`inference`)

```python
# wan22_fewstep_inference.py 第52-101行
def inference(self, noise: torch.Tensor, text_prompts: List[str], 
              wan22_image_latent: torch.Tensor = None) -> torch.Tensor:
    
    # 文本编码
    conditional_dict = self.text_encoder(text_prompts=text_prompts)
    
    # 初始化
    noisy_image_or_video = noise
    
    # 创建mask
    if wan22_image_latent is not None:
        mask1, mask2 = masks_like(noisy_image_or_video, zero=True)
        mask2 = torch.stack(mask2, dim=0)
        noisy_image_or_video = (1. - mask2) * wan22_image_latent + mask2 * noisy_image_or_video
    
    # 去噪循环（推理时遍历所有步骤）
    for index, current_timestep in enumerate(self.denoising_step_list):
        
        # 构造wan22时间步
        wan22_input_timestep = ...
        
        # 生成器预测
        _, pred_image_or_video = self.generator(
            noisy_image_or_video=noisy_image_or_video,
            conditional_dict=conditional_dict,
            timestep=torch.ones(noise.shape[:2]) * current_timestep,
            wan22_input_timestep=wan22_input_timestep,
            mask2=mask2,
            wan22_image_latent=wan22_image_latent,
        )
        
        # 如果不是最后一步，加噪声到下一步
        if index < len(self.denoising_step_list) - 1:
            next_timestep = self.denoising_step_list[index + 1]
            noisy_image_or_video = self.scheduler.add_noise(
                pred_image_or_video.flatten(0, 1),
                torch.randn_like(pred_image_or_video.flatten(0, 1)),
                next_timestep.flatten(0, 1)
            ).unflatten(0, noise.shape[:2])
            
            # 混合首帧
            if wan22_image_latent is not None:
                noisy_image_or_video = (1. - mask2) * wan22_image_latent + mask2 * noisy_image_or_video
    
    # 解码为像素
    video = self.vae.decode_to_pixel(pred_image_or_video)
    video = (video * 0.5 + 0.5).clamp(0, 1)
    return video
```

### 8.3 推理与训练的区别

| 特性 | 训练 | 推理 |
|------|------|------|
| 退出步骤 | 随机选择一个 | 遍历所有步骤 |
| 梯度计算 | 只在退出步骤 | 不计算梯度 |
| 文本引导 | 使用/不使用CFG | 通常使用CFG |
| 首帧处理 | 从数据集编码 | 用户提供或随机 |

---

## 9. 关键设计决策总结

### 9.1 为什么使用DMD而非其他蒸馏方法？

**DMD的优势：**
1. **数据无关**：不需要大量训练数据，只需prompt
2. **分布匹配**：直接匹配输出分布，而非中间特征
3. **端到端**：整个流程可微分
4. **灵活性**：支持任意步数蒸馏（1步、2步、4步等）

**与其他方法对比：**
- **Progressive Distillation**：需要多轮蒸馏（2步→4步→8步...）
- **Consistency Models**：只能蒸馏到1步
- **Knowledge Distillation**：需要大量训练数据

### 9.2 为什么使用反向模拟？

**反向模拟的优势：**
1. **消除训练-推理差异**：训练时的采样过程 = 推理时的采样过程
2. **提高样本质量**：使用生成器自己的输出作为训练数据
3. **动态适应**：随着生成器改进，训练数据也改进

**传统方法的问题：**
- 使用真实数据：生成器可能过拟合数据集分布
- 使用随机噪声：与推理时的输入分布不一致

### 9.3 为什么需要判别器（Fake Score）？

**判别器的作用：**
1. **估计分布偏差**：`grad = pred_fake - pred_real`
2. **提供训练信号**：判别器越准确，生成器训练越有效
3. **适应生成器变化**：判别器需要持续训练以跟踪生成器

**对抗训练的稳定性：**
- 每5步训练1次生成器，训练5次判别器
- 防止判别器过快收敛（无法提供有效梯度）
- 防止生成器过快改进（判别器跟不上）

### 9.4 为什么Wan2.2使用TI2V架构？

**TI2V的优势：**
1. **首帧质量**：图像生成比视频生成更成熟，首帧质量更高
2. **时间一致性**：从高质量首帧扩展，保证视频的时间连贯性
3. **计算效率**：首帧只需生成一次，不需要重复去噪

**挑战：**
- 需要特殊的mask机制保持首帧干净
- 需要特殊的时间步构造（wan22_input_timestep）
- 编码/解码开销（长视频需要重新编码首帧）

---

## 10. 常见问题解答

### Q1: 如何修改蒸馏步数？

修改配置文件中的 `denoising_step_list`：

```yaml
# 4步蒸馏（当前配置）
denoising_step_list: [1000, 750, 500, 250]

# 2步蒸馏
denoising_step_list: [1000, 500]

# 8步蒸馏
denoising_step_list: [1000, 875, 750, 625, 500, 375, 250, 125]
```

**注意：** 步数越少，蒸馏越困难，需要更长的训练时间。

### Q2: 如何加载预训练权重？

在配置文件中添加：

```yaml
generator_ckpt: /path/to/checkpoint.pt
```

或者修改训练器的 `load` 方法，从指定路径加载检查点。

### Q3: 如何调整训练超参数？

常见超参数调整：

```yaml
# 学习率
lr: 5.0e-07              # 生成器学习率（较小更稳定）
lr_critic: 1.0e-07       # 判别器学习率（通常是生成器的1/5）

# 训练比例
dfake_gen_update_ratio: 5  # 判别器/生成器训练比例

# 梯度裁剪
max_grad_norm_generator: 10.0
max_grad_norm_critic: 10.0

# EMA
ema_weight: 0.99         # 越大越平滑
ema_start_step: 200      # 延迟开始避免早期不稳定

# CFG
guidance_scale: 6.0      # 教师模型引导强度
```

### Q4: 如何处理显存不足？

1. **启用梯度检查点**：`gradient_checkpointing: true`（已启用）
2. **减少批次大小**：`batch_size: 1`（已最小）
3. **使用FSDP**：`sharding_strategy: full`（已启用）
4. **CPU offload**：`text_encoder_cpu_offload: true`（已启用）
5. **减少生成帧数**：`num_training_frames: 31 → 21`

### Q5: 训练多久会收敛？

**经验值：**
- 4步蒸馏：约50,000步（根据数据集大小）
- 2步蒸馏：约100,000步
- 1步蒸馏：约200,000步

**监控指标：**
- `generator_loss` 下降并稳定
- `critic_loss` 保持在合理范围（不应为0或过大）
- `dmdtrain_gradient_norm` 逐渐减小

### Q6: 如何评估蒸馏效果？

1. **定性评估**：
   - 使用相同prompt生成视频
   - 对比教师模型（50步）和学生模型（4步）的输出
   - 评估视频质量、时间一致性、文本对齐度

2. **定量评估**：
   - FVD (Fréchet Video Distance)
   - CLIP Score（文本-视频对齐）
   - 时间一致性指标

3. **效率评估**：
   - 推理时间（应减少到原来的1/10）
   - 显存占用
   - 吞吐量（videos/sec）

---

## 附录：文件功能总结

### 核心文件

| 文件 | 功能 | 关键类/函数 |
|------|------|------------|
| `train.py` | 训练入口 | `main()` |
| `wan22_distillation.py` | 训练器 | `Trainer.__init__()`, `train()`, `fwdbwd_one_step()` |
| `dmd.py` | DMD算法 | `DMD`, `_compute_kl_grad()`, `generator_loss()`, `critic_loss()` |
| `base.py` | 基础模型 | `BaseModel`, `SelfForcingModel`, `_run_generator()` |
| `bidirectional_training.py` | 反向模拟 | `BidirectionalTrainingPipeline.inference_with_trajectory()` |
| `wan22_fewstep_inference.py` | 推理管道 | `Wan22FewstepInferencePipeline.inference()` |

### 工具文件

| 文件 | 功能 |
|------|------|
| `scheduler.py` | Flow Matching调度器 |
| `loss.py` | 损失函数定义 |
| `distributed.py` | FSDP分布式训练 |
| `dataset.py` | 数据加载（CSV格式） |
| `wan_wrapper.py` | Wan模型包装器 |

---

## 结语

这个代码仓库实现了一个完整的视频生成模型蒸馏系统，核心要点：

1. **DMD算法**：通过分布匹配实现步数蒸馏
2. **反向模拟**：使用生成器自己的输出作为训练数据
3. **Wan2.2特殊处理**：TI2V架构的mask和时间步构造
4. **分布式训练**：FSDP实现5B模型的高效训练
5. **对抗训练**：判别器提供准确的分布偏差估计

希望这份详细分析能帮助你完全理解整个蒸馏流程！
