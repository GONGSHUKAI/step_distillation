# Ovi模型四步蒸馏适配方案

## 一、背景分析

### 1.1 现有代码仓库特点
- **目标模型**: Wan2.2-TI2V-5B (视频生成模型)
- **架构**: 双向扩散 (bidirectional diffusion)
- **蒸馏方法**: DMD (Distribution Matching Distillation) + Self-Forcing训练
- **步数**: 4步 [1000, 750, 500, 250]
- **特殊机制**: TI2V架构,首帧为图像latent,后续为视频latent

### 1.2 Ovi模型特点
根据Ovi.pdf和项目知识库信息:
- **架构**: FusionModel,包含两个并行分支:
  - **视频分支**: Wan2.2-TI2V-5B结构 (使用预训练权重)
  - **音频分支**: Wan2.2-5B结构 (单独训练)
- **融合机制**: 通过cross-attention在每层进行音视频交互
- **VAE**: 
  - 视频: Wan2.2_VAE
  - 音频: MMAudio VAE
- **训练方式**: 音频单独训练,视频加载预训练,然后音视频联合训练

### 1.3 核心挑战
1. **模型架构差异**: Ovi是fusion模型,需要同时处理音频和视频两个模态
2. **VAE差异**: 需要支持两个不同的VAE (视频VAE和音频VAE)
3. **文本编码器**: 共享T5编码器,但需要处理不同的负提示词
4. **训练数据格式**: 需要同时提供音频和视频数据
5. **损失计算**: 需要同时计算音频和视频的蒸馏损失

---

## 二、整体适配策略

### 2.1 适配原则
✅ **复用为主,修改为辅**: 最大化利用现有的DMD框架和训练流程  
✅ **模块化设计**: 将Ovi特有的逻辑封装成独立模块  
✅ **向后兼容**: 保持与原Wan2.2蒸馏代码的兼容性  
✅ **渐进式开发**: 先实现单模态蒸馏,再扩展到联合蒸馏  

### 2.2 开发阶段划分

**Phase 1**: 视频分支单独蒸馏 (验证基础框架)  
**Phase 2**: 音频分支单独蒸馏 (扩展到音频模态)  
**Phase 3**: 音视频联合蒸馏 (实现完整的Ovi蒸馏)  

---

## 三、详细适配方案

### 3.1 文件结构规划

```
project/
├── ovi_wrapper.py              # 新增: Ovi模型封装
├── ovi_dmd.py                  # 新增: Ovi专用DMD实现
├── ovi_distillation.py         # 新增: Ovi蒸馏训练器
├── ovi_inference.py            # 新增: Ovi推理管道
├── ovi_config.yaml             # 新增: Ovi蒸馏配置文件
├── base.py                     # 修改: 扩展支持fusion模型
├── dmd.py                      # 可能需要小幅修改
├── wan_wrapper.py              # 参考,不修改
└── train.py                    # 小幅修改: 添加ovi trainer分支
```

---

## 四、核心模块设计

### 4.1 `ovi_wrapper.py` - Ovi模型封装

#### 4.1.1 OviTextEncoder (复用Wan的T5)
```python
class OviTextEncoder(torch.nn.Module):
    """
    复用WanTextEncoder,但需要处理音视频不同的负提示词
    """
    def __init__(self, model_name="Wan2.2-TI2V-5B"):
        super().__init__()
        # 复用WanTextEncoder的实现
        self.text_encoder = umt5_xxl(...)
        self.tokenizer = HuggingfaceTokenizer(...)
    
    def forward(self, prompts, video_neg_prompts, audio_neg_prompts):
        """
        返回:
        - prompt_embeds: [B, 512, 4096]
        - video_neg_embeds: [B, 512, 4096]  
        - audio_neg_embeds: [B, 512, 4096]
        """
        pass
```

#### 4.1.2 OviVAEWrapper (双VAE管理)
```python
class OviVAEWrapper(torch.nn.Module):
    """
    管理视频VAE和音频VAE
    """
    def __init__(self):
        super().__init__()
        self.video_vae = Wan2_2_VAEWrapper()
        self.audio_vae = MMAudioVAE()
    
    def encode_video(self, video):
        # [B, C, F, H, W] -> [B, F, C_latent, H_latent, W_latent]
        pass
    
    def encode_audio(self, audio):
        # [B, C, L] -> [B, L, C_latent]
        pass
    
    def decode_video(self, latent):
        pass
    
    def decode_audio(self, latent):
        pass
```

#### 4.1.3 OviFusionWrapper (核心封装)
```python
class OviFusionWrapper(torch.nn.Module):
    """
    封装FusionModel,提供统一的forward接口
    类似WanDiffusionWrapper的角色
    """
    def __init__(self, model_name="Ovi", is_causal=False, **kwargs):
        super().__init__()
        self.model_name = model_name
        
        # 加载fusion模型
        self.model = self._load_fusion_model()
        
        # 调度器 (复用Wan的FlowMatchScheduler)
        self.scheduler = FlowMatchScheduler(...)
    
    def _load_fusion_model(self):
        """
        加载FusionModel,参考ovi/utils/model_loading_utils.py
        """
        video_config = load_json("ovi/configs/model/dit/video.json")
        audio_config = load_json("ovi/configs/model/dit/audio.json")
        model = FusionModel(video_config, audio_config)
        
        # 加载预训练权重
        # 视频分支: 加载Wan2.2-TI2V-5B权重
        # 音频分支: 加载单独训练的音频权重
        # fusion参数: 从Ovi联合训练权重加载
        
        return model
    
    def forward(
        self,
        video_latent,      # [B, F, C, H, W]
        audio_latent,      # [B, L, C]
        timestep_video,    # [B, F]
        timestep_audio,    # [B, L]  
        text_embeds,       # [B, 512, 4096]
        **kwargs
    ):
        """
        统一的forward接口,调用FusionModel.forward
        返回:
        - pred_video: [B, F, C, H, W]
        - pred_audio: [B, L, C]
        """
        # 准备video_kwargs和audio_kwargs
        video_kwargs = self._prepare_video_kwargs(...)
        audio_kwargs = self._prepare_audio_kwargs(...)
        
        # 调用fusion模型
        pred_video, pred_audio = self.model(
            vid=video_latent,
            audio=audio_latent,
            vid_kwargs=video_kwargs,
            audio_kwargs=audio_kwargs
        )
        
        return pred_video, pred_audio
    
    def enable_gradient_checkpointing(self):
        self.model.gradient_checkpointing = True
```

---

### 4.2 `ovi_dmd.py` - Ovi专用DMD

继承自`dmd.py`的`DMD`类,扩展为支持双模态:

```python
class OviDMD(SelfForcingModel):
    """
    Ovi专用的DMD实现
    关键差异:
    1. 管理6个模型 (而不是3个):
       - video_generator, video_real_score, video_fake_score
       - audio_generator, audio_real_score, audio_fake_score
    2. 计算双模态的KL梯度
    3. 同时更新两个生成器
    """
    
    def __init__(self, args, device):
        # 不调用super().__init__,因为需要自定义初始化
        self.device = device
        self.args = args
        
        # 初始化模型
        self._initialize_ovi_models(args, device)
        
        # 初始化损失函数
        self.denoising_loss_func = get_denoising_loss(args.denoising_loss_type)()
        
        # Ovi特殊配置
        self.num_training_frames_video = 31
        self.num_training_frames_audio = getattr(args, "num_training_frames_audio", 512)
    
    def _initialize_ovi_models(self, args, device):
        """
        初始化6个模型:
        - Generator (Student): OviFusionWrapper (可训练)
        - Real Score (Teacher): OviFusionWrapper (冻结)
        - Fake Score (Critic): OviFusionWrapper (可训练)
        """
        self.generator = OviFusionWrapper(model_name="Ovi", **args.model_kwargs)
        self.generator.model.requires_grad_(True)
        
        self.real_score = OviFusionWrapper(model_name="Ovi")
        self.real_score.model.requires_grad_(False)
        
        self.fake_score = OviFusionWrapper(model_name="Ovi")
        self.fake_score.model.requires_grad_(True)
        
        # 共享的组件
        self.text_encoder = OviTextEncoder()
        self.text_encoder.requires_grad_(False)
        
        self.vae = OviVAEWrapper()
        self.vae.requires_grad_(False)
    
    def _compute_kl_grad(
        self,
        video_noisy_latent,           # [B, F, C, H, W]
        audio_noisy_latent,           # [B, L, C]
        video_estimated_clean,        # [B, F, C, H, W]
        audio_estimated_clean,        # [B, L, C]
        timestep_video,               # [B, F]
        timestep_audio,               # [B, L]
        conditional_dict,             # 包含文本嵌入等
        unconditional_dict,           # 负提示词嵌入
        normalization=True,
        wan22_image_latent=None,      # 视频首帧
    ):
        """
        计算双模态的KL梯度
        
        核心思路:
        1. 分别为视频和音频构造mask (类似Wan2.2的mask机制)
        2. 分别运行real_score和fake_score
        3. 分别计算video和audio的KL梯度
        4. 返回两个梯度用于更新生成器
        """
        
        # === 视频分支 ===
        # 1. Wan2.2特殊处理: 混合首帧和噪声帧
        if wan22_image_latent is not None:
            mask1_video, mask2_video = masks_like(video_noisy_latent, zero=True)
            mask2_video = torch.stack(mask2_video, dim=0)
            video_noisy_latent_mixed = (1. - mask2_video) * wan22_image_latent + \
                                       mask2_video * video_noisy_latent
        else:
            video_noisy_latent_mixed = video_noisy_latent
        
        # 2. 运行教师模型 (real_score)
        with torch.no_grad():
            pred_video_real, pred_audio_real = self.real_score(
                video_latent=video_noisy_latent_mixed,
                audio_latent=audio_noisy_latent,
                timestep_video=timestep_video,
                timestep_audio=timestep_audio,
                **conditional_dict
            )
        
        # 3. 运行判别器 (fake_score)
        pred_video_fake, pred_audio_fake = self.fake_score(
            video_latent=video_noisy_latent_mixed,
            audio_latent=audio_noisy_latent,
            timestep_video=timestep_video,
            timestep_audio=timestep_audio,
            **conditional_dict
        )
        
        # 4. 计算KL梯度
        # 4.1 视频部分
        video_kl_grad = pred_video_fake - pred_video_real
        if normalization:
            video_kl_grad = video_kl_grad / (video_kl_grad.std() + 1e-8)
        
        # 4.2 音频部分
        audio_kl_grad = pred_audio_fake - pred_audio_real
        if normalization:
            audio_kl_grad = audio_kl_grad / (audio_kl_grad.std() + 1e-8)
        
        # 5. 返回结果
        return {
            "video_kl_grad": video_kl_grad,
            "audio_kl_grad": audio_kl_grad,
            "video_pred_real": pred_video_real,
            "audio_pred_real": pred_audio_real,
            "video_pred_fake": pred_video_fake,
            "audio_pred_fake": pred_audio_fake,
        }
    
    def generator_loss(self, batch_data):
        """
        生成器损失
        
        步骤:
        1. 从batch_data提取视频和音频的真实数据
        2. 使用反向模拟生成训练样本 (self._run_ovi_generator)
        3. 计算KL梯度
        4. 计算去噪损失 (video + audio)
        5. 返回总损失
        """
        # 1. 提取数据
        video_clean = batch_data["video"]  # [B, F, C, H, W]
        audio_clean = batch_data["audio"]  # [B, L, C]
        prompts = batch_data["prompts"]
        
        # 2. 编码为latent
        with torch.no_grad():
            video_latent = self.vae.encode_video(video_clean)
            audio_latent = self.vae.encode_audio(audio_clean)
            
            # 文本编码
            text_dict = self.text_encoder(
                prompts=prompts,
                video_neg_prompts=batch_data["video_neg_prompts"],
                audio_neg_prompts=batch_data["audio_neg_prompts"]
            )
        
        # 3. 反向模拟生成样本
        generated_result = self._run_ovi_generator(
            video_shape=[B, F, C, H, W],
            audio_shape=[B, L, C],
            conditional_dict=text_dict,
            wan22_image_latent=video_latent[:, :1, ...] if wan22 else None
        )
        
        video_pred = generated_result["video_pred"]
        audio_pred = generated_result["audio_pred"]
        timestep_video = generated_result["timestep_video"]
        timestep_audio = generated_result["timestep_audio"]
        
        # 4. 计算KL梯度
        kl_result = self._compute_kl_grad(
            video_noisy_latent=...,
            audio_noisy_latent=...,
            video_estimated_clean=video_pred,
            audio_estimated_clean=audio_pred,
            timestep_video=timestep_video,
            timestep_audio=timestep_audio,
            conditional_dict=text_dict,
            unconditional_dict=...,
        )
        
        # 5. 计算去噪损失
        video_denoising_loss = self.denoising_loss_func(
            prediction=video_pred,
            target=video_latent,
            ...
        )
        
        audio_denoising_loss = self.denoising_loss_func(
            prediction=audio_pred,
            target=audio_latent,
            ...
        )
        
        # 6. 总损失
        total_loss = video_denoising_loss + audio_denoising_loss
        
        return {
            "loss": total_loss,
            "video_loss": video_denoising_loss,
            "audio_loss": audio_denoising_loss,
            ...
        }
    
    def critic_loss(self, batch_data):
        """
        判别器损失
        与generator_loss类似,但目标是训练fake_score
        """
        pass
```

---

### 4.3 `ovi_distillation.py` - Ovi蒸馏训练器

继承自`wan22_distillation.py`的`Wan22ScoreDistillationTrainer`:

```python
class OviScoreDistillationTrainer:
    """
    Ovi专用蒸馏训练器
    
    关键差异:
    1. 数据加载: 需要同时加载视频和音频数据
    2. FSDP包装: 需要分别包装video和audio分支
    3. 优化器: 可以为视频和音频设置不同的学习率
    4. 日志: 分别记录视频和音频的指标
    """
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda")
        
        # 1. 初始化分布式环境 (复用)
        launch_distributed_job()
        
        # 2. 初始化数据集
        self.train_dataset = OviDataset(
            video_csv=config.video_csv_path,
            audio_csv=config.audio_csv_path,
            video_dir=config.video_dir,
            audio_dir=config.audio_dir,
        )
        
        # 3. 初始化模型
        self.model = OviDMD(config, device=self.device)
        
        # 4. FSDP包装
        self._wrap_with_fsdp()
        
        # 5. 优化器
        self._setup_optimizers()
        
        # 6. EMA
        self._setup_ema()
    
    def _wrap_with_fsdp(self):
        """
        FSDP包装策略:
        
        选项1 (推荐): 分别包装video和audio分支
        - self.model.generator.model.video_model -> FSDP
        - self.model.generator.model.audio_model -> FSDP
        - 优点: 更灵活,可以设置不同的wrap策略
        
        选项2: 整体包装fusion模型
        - self.model.generator -> FSDP
        - 优点: 简单
        - 缺点: 可能不够灵活
        """
        # 选项1实现
        self.model.generator.model.video_model = fsdp_wrap(
            self.model.generator.model.video_model,
            sharding_strategy=self.config.sharding_strategy,
            mixed_precision=self.config.mixed_precision,
            wrap_strategy=self.config.video_fsdp_wrap_strategy
        )
        
        self.model.generator.model.audio_model = fsdp_wrap(
            self.model.generator.model.audio_model,
            sharding_strategy=self.config.sharding_strategy,
            mixed_precision=self.config.mixed_precision,
            wrap_strategy=self.config.audio_fsdp_wrap_strategy
        )
        
        # 类似地包装real_score和fake_score
        ...
    
    def _setup_optimizers(self):
        """
        优化器配置
        
        可选策略:
        1. 统一优化器: 所有参数使用相同学习率
        2. 分组优化器: video和audio使用不同学习率
        """
        # 策略2实现 (推荐)
        video_params = list(self.model.generator.model.video_model.parameters())
        audio_params = list(self.model.generator.model.audio_model.parameters())
        
        self.optimizer_generator = torch.optim.AdamW([
            {"params": video_params, "lr": self.config.lr_video},
            {"params": audio_params, "lr": self.config.lr_audio},
        ], betas=(self.config.beta1, self.config.beta2))
        
        # 类似地设置critic优化器
        ...
    
    def train(self):
        """
        训练主循环
        复用wan22_distillation.py的逻辑,但需要:
        1. 处理双模态数据
        2. 分别计算video和audio的loss
        3. 分别记录video和audio的指标
        """
        for step in range(self.step, self.config.total_steps):
            # 1. 获取batch数据
            batch = next(self.train_dataloader)
            batch = {
                "video": batch["video"].to(self.device),
                "audio": batch["audio"].to(self.device),
                "prompts": batch["prompts"],
                "video_neg_prompts": batch.get("video_neg_prompts", [""]*len(batch["prompts"])),
                "audio_neg_prompts": batch.get("audio_neg_prompts", [""]*len(batch["prompts"])),
            }
            
            # 2. 训练判别器 (每5步)
            if step % self.config.dfake_gen_update_ratio != 0:
                loss_dict = self.fwdbwd_one_step(
                    batch_data=batch,
                    update_generator=False
                )
            
            # 3. 训练生成器 (每1步)
            else:
                loss_dict = self.fwdbwd_one_step(
                    batch_data=batch,
                    update_generator=True
                )
            
            # 4. 日志
            if step % self.config.log_iters == 0:
                self._log_metrics(loss_dict, step)
            
            # 5. 保存checkpoint
            if step % self.config.save_iters == 0:
                self._save_checkpoint(step)
    
    def fwdbwd_one_step(self, batch_data, update_generator=False):
        """
        前向+反向一步
        
        与wan22_distillation.py的实现类似,但需要:
        1. 调用self.model.generator_loss或self.model.critic_loss
        2. 处理双模态的梯度
        3. 返回双模态的loss字典
        """
        if update_generator:
            # 训练生成器
            loss_dict = self.model.generator_loss(batch_data)
            loss = loss_dict["loss"]
            
            self.optimizer_generator.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(
                self.model.generator.parameters(),
                self.config.max_grad_norm_generator
            )
            
            self.optimizer_generator.step()
            
            # EMA更新
            if step >= self.config.ema_start_step:
                self.ema.update()
        
        else:
            # 训练判别器
            loss_dict = self.model.critic_loss(batch_data)
            loss = loss_dict["loss"]
            
            self.optimizer_critic.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(
                self.model.fake_score.parameters(),
                self.config.max_grad_norm_critic
            )
            
            self.optimizer_critic.step()
        
        return loss_dict
```

---

### 4.4 `ovi_config.yaml` - Ovi蒸馏配置文件

```yaml
# ============ 模型配置 ============
model_name: Ovi
generator_type: bidirectional  # fusion模型不是causal的

# ============ 权重加载 ============
ovi_pretrained_ckpt: /path/to/ovi_joint_trained.pt
# 包含:
# - video_model: Wan2.2-TI2V-5B weights
# - audio_model: 单独训练的音频权重
# - fusion layers: k_fusion, v_fusion等

# ============ 蒸馏步数配置 ============
denoising_step_list: [1000, 750, 500, 250]
warp_denoising_step: true
num_train_timestep: 1000
timestep_shift: 5.0

# 指导尺度 (分别设置)
video_guidance_scale: 6.0
audio_guidance_scale: 6.0

# ============ 训练配置 ============
trainer: score_distillation_ovi  # 新增
distribution_loss: dmd

mixed_precision: true
sharding_strategy: full
gradient_checkpointing: true

# 学习率 (可以分开设置)
lr_video: 5.0e-07
lr_audio: 5.0e-07
lr_critic: 1.0e-07

beta1: 0.0
beta2: 0.999

batch_size: 1
total_batch_size: 64
ema_weight: 0.99
ema_start_step: 200

dfake_gen_update_ratio: 5

# ============ 数据配置 ============
video_csv_path: /path/to/video_prompts.csv
audio_csv_path: /path/to/audio_prompts.csv  # 可以与video相同
video_dir: /path/to/videos
audio_dir: /path/to/audios

# ============ 视频配置 ============
num_frames_video: 121
h: 704
w: 1280
num_training_frames_video: 31
num_frame_per_block: 31

# ============ 音频配置 ============
audio_sample_rate: 16000
audio_duration: 8.0  # 秒
num_training_frames_audio: 512  # latent长度

# ============ FSDP配置 ============
video_fsdp_wrap_strategy: size
audio_fsdp_wrap_strategy: size
fake_score_fsdp_wrap_strategy: size
real_score_fsdp_wrap_strategy: size
text_encoder_fsdp_wrap_strategy: size
text_encoder_cpu_offload: true

# ============ 负提示词 ============
video_negative_prompt: "色调艳丽,过曝,静态,细节模糊不清..."
audio_negative_prompt: "noisy, distorted, low quality..."

# ============ 其他 ============
seed: 0
log_iters: 500
save_iters: 5000
wandb_project: ovi-distill
```

---

### 4.5 数据集实现 `ovi_dataset.py`

```python
class OviDataset(torch.utils.data.Dataset):
    """
    Ovi数据集
    
    需要提供:
    1. 视频数据 (从video_csv和video_dir加载)
    2. 音频数据 (从audio_csv和audio_dir加载)
    3. 文本提示词
    
    策略:
    - 可以使用相同的prompt同时生成视频和音频
    - 也可以分别提供video_prompt和audio_prompt
    """
    
    def __init__(
        self,
        video_csv,
        audio_csv,
        video_dir,
        audio_dir,
        video_transform=None,
        audio_transform=None,
    ):
        self.video_data = pd.read_csv(video_csv)
        self.audio_data = pd.read_csv(audio_csv)
        
        # 假设两个CSV有相同的行数和prompt
        assert len(self.video_data) == len(self.audio_data)
        
        self.video_dir = video_dir
        self.audio_dir = audio_dir
        self.video_transform = video_transform
        self.audio_transform = audio_transform
    
    def __len__(self):
        return len(self.video_data)
    
    def __getitem__(self, idx):
        # 1. 加载视频
        video_path = os.path.join(self.video_dir, self.video_data.iloc[idx]["video_file"])
        video = self._load_video(video_path)  # [C, F, H, W]
        
        # 2. 加载音频
        audio_path = os.path.join(self.audio_dir, self.audio_data.iloc[idx]["audio_file"])
        audio = self._load_audio(audio_path)  # [C, L]
        
        # 3. 获取提示词
        prompt = self.video_data.iloc[idx]["prompt"]
        
        return {
            "video": video,
            "audio": audio,
            "prompt": prompt,
        }
```

---

### 4.6 修改 `train.py`

```python
# train.py

if config.trainer == "score_distillation_wan22":
    trainer = Wan22ScoreDistillationTrainer(config)
elif config.trainer == "score_distillation_ovi":  # 新增
    from ovi_distillation import OviScoreDistillationTrainer
    trainer = OviScoreDistillationTrainer(config)
else:
    raise ValueError(f"Unknown trainer: {config.trainer}")

trainer.train()
```

---

## 五、关键技术细节

### 5.1 权重加载策略

**问题**: Ovi模型的权重如何加载?

**方案**:
1. **视频分支**: 加载Wan2.2-TI2V-5B的预训练权重
   ```python
   video_state_dict = torch.load("Wan2.2-TI2V-5B.pth")
   model.generator.model.video_model.load_state_dict(video_state_dict, strict=False)
   ```

2. **音频分支**: 加载单独训练的音频权重
   ```python
   audio_state_dict = torch.load("audio_pretrained.pth")
   model.generator.model.audio_model.load_state_dict(audio_state_dict, strict=False)
   ```

3. **Fusion层**: 加载联合训练的权重
   ```python
   fusion_state_dict = torch.load("ovi_joint_trained.pth")
   # 只加载包含"fusion"的参数
   fusion_params = {k: v for k, v in fusion_state_dict.items() if "fusion" in k}
   model.generator.model.load_state_dict(fusion_params, strict=False)
   ```

4. **教师模型和判别器**: 
   - Real Score: 完整加载Ovi联合训练权重 (冻结)
   - Fake Score: 从Real Score复制初始化 (可训练)

### 5.2 反向模拟机制

**问题**: 如何为双模态实现反向模拟?

**方案**: 扩展`BidirectionalTrainingPipeline`为`OviBidirectionalTrainingPipeline`

```python
class OviBidirectionalTrainingPipeline:
    """
    Ovi专用的反向模拟管道
    
    核心差异:
    1. 同时模拟视频和音频的噪声到清晰过程
    2. 在每步去噪中,video和audio相互作为条件 (通过fusion)
    """
    
    def inference_with_trajectory(
        self,
        video_noise,     # [B, F, C, H, W]
        audio_noise,     # [B, L, C]
        text_embeds,
        wan22_image_latent=None,
    ):
        """
        从噪声开始,逐步去噪,返回中间轨迹
        
        步骤:
        1. 初始化: video_noise和audio_noise
        2. 遍历denoising_step_list [1000, 750, 500, 250]
        3. 每步调用generator进行去噪
        4. 随机选择一个中间步作为训练起点
        5. 返回轨迹和训练时间步
        """
        trajectory_video = [video_noise]
        trajectory_audio = [audio_noise]
        
        for i, t in enumerate(self.denoising_step_list):
            timestep_video = torch.full((B, F), t, device=device)
            timestep_audio = torch.full((B, L), t, device=device)
            
            # 去噪
            with torch.no_grad():
                pred_video, pred_audio = self.generator(
                    video_latent=trajectory_video[-1],
                    audio_latent=trajectory_audio[-1],
                    timestep_video=timestep_video,
                    timestep_audio=timestep_audio,
                    text_embeds=text_embeds,
                )
            
            trajectory_video.append(pred_video)
            trajectory_audio.append(pred_audio)
        
        # 随机选择训练起点
        train_step_idx = torch.randint(0, len(self.denoising_step_list), (1,)).item()
        
        return {
            "video_pred": trajectory_video[train_step_idx + 1],
            "audio_pred": trajectory_audio[train_step_idx + 1],
            "timestep_from": self.denoising_step_list[train_step_idx],
            "timestep_to": self.denoising_step_list[train_step_idx + 1],
        }
```

### 5.3 Wan2.2首帧处理

**问题**: 视频分支需要特殊处理首帧 (image latent),如何在fusion模型中实现?

**方案**:
1. **编码阶段**: 
   ```python
   # 对于Wan2.2 TI2V,首帧需要单独编码为image latent
   first_frame = video[:, :1, ...]  # [B, 1, C, H, W]
   rest_frames = video[:, 1:, ...]  # [B, F-1, C, H, W]
   
   image_latent = vae.encode_image(first_frame)  # [B, 1, C_l, H_l, W_l]
   video_latent = vae.encode_video(rest_frames)  # [B, F-1, C_l, H_l, W_l]
   
   full_latent = torch.cat([image_latent, video_latent], dim=1)  # [B, F, C_l, H_l, W_l]
   ```

2. **训练阶段**: 使用mask保持首帧不变
   ```python
   # 在_compute_kl_grad中
   mask = torch.zeros_like(video_latent)
   mask[:, 0, ...] = 1.0  # 首帧mask为1,表示保持原值
   
   video_latent_mixed = mask * wan22_image_latent + (1 - mask) * video_noisy_latent
   ```

### 5.4 音频和视频的时间步对齐

**问题**: 音频和视频的时间维度不同 (视频31帧 vs 音频512帧),如何对齐?

**解决方案**:
1. **独立时间步**: 视频和音频使用各自的时间步
   ```python
   timestep_video = torch.randint(min_t, max_t, (B, 31))
   timestep_audio = torch.randint(min_t, max_t, (B, 512))
   ```

2. **共享时间步** (简化版): 使用相同的标量时间步
   ```python
   t = torch.randint(min_t, max_t, (B,))
   timestep_video = t.unsqueeze(1).repeat(1, 31)
   timestep_audio = t.unsqueeze(1).repeat(1, 512)
   ```

推荐使用**方案1**,因为音视频的噪声水平可以独立控制。

---

## 六、开发和测试计划

### Phase 1: 基础框架搭建 (1-2周)
- [ ] 实现`ovi_wrapper.py` (OviFusionWrapper, OviVAEWrapper, OviTextEncoder)
- [ ] 实现`ovi_dataset.py`
- [ ] 实现`ovi_config.yaml`
- [ ] 测试模型加载和前向传播

**验证点**:
```python
# 测试代码
model = OviFusionWrapper(model_name="Ovi")
video_latent = torch.randn(1, 31, 16, 44, 80)
audio_latent = torch.randn(1, 512, 20)
timestep = torch.ones(1, 31) * 500

pred_video, pred_audio = model(
    video_latent=video_latent,
    audio_latent=audio_latent,
    timestep_video=timestep,
    timestep_audio=timestep,
    text_embeds=text_embeds,
)

assert pred_video.shape == video_latent.shape
assert pred_audio.shape == audio_latent.shape
print("✅ Phase 1 passed!")
```

### Phase 2: DMD核心实现 (2-3周)
- [ ] 实现`ovi_dmd.py` (OviDMD类)
- [ ] 实现`_compute_kl_grad`方法
- [ ] 实现`generator_loss`和`critic_loss`方法
- [ ] 实现`OviBidirectionalTrainingPipeline`

**验证点**:
```python
# 测试KL梯度计算
dmd_model = OviDMD(config, device)
batch = {
    "video": torch.randn(1, 121, 3, 704, 1280),
    "audio": torch.randn(1, 16000*8),
    "prompts": ["test prompt"],
}

loss_dict = dmd_model.generator_loss(batch)
assert "loss" in loss_dict
assert "video_loss" in loss_dict
assert "audio_loss" in loss_dict
print("✅ Phase 2 passed!")
```

### Phase 3: 训练器实现 (2-3周)
- [ ] 实现`ovi_distillation.py` (OviScoreDistillationTrainer)
- [ ] 实现FSDP包装逻辑
- [ ] 实现训练循环
- [ ] 实现checkpoint保存/加载
- [ ] 实现wandb日志

**验证点**:
```python
# 测试训练一个step
trainer = OviScoreDistillationTrainer(config)
loss_dict = trainer.fwdbwd_one_step(batch, update_generator=True)
assert loss_dict["loss"] < float("inf")
print("✅ Phase 3 passed!")
```

### Phase 4: 端到端训练 (3-4周)
- [ ] 准备小规模数据集 (100个样本)
- [ ] 运行端到端训练 (1000步)
- [ ] 验证loss下降
- [ ] 生成样本质量检查

**验证点**:
- Loss曲线平滑下降
- 生成的视频和音频质量可接受
- 没有NaN或Inf

### Phase 5: 大规模训练和调优 (4-6周)
- [ ] 扩展到完整数据集
- [ ] 超参数调优
- [ ] 多机多卡训练
- [ ] 定期评估和checkpoint选择

---

## 七、潜在风险和解决方案

### 风险1: 显存不足
**现象**: OOM错误

**解决方案**:
1. 启用梯度检查点 (`gradient_checkpointing: true`)
2. 减小batch size到1
3. 使用FSDP的CPU offload (`text_encoder_cpu_offload: true`)
4. 减少训练帧数 (`num_training_frames_video: 21`)
5. 使用混合精度训练 (`mixed_precision: true`)

### 风险2: 训练不稳定
**现象**: Loss震荡或NaN

**解决方案**:
1. 降低学习率 (`lr: 1e-7`)
2. 增加梯度裁剪 (`max_grad_norm: 5.0`)
3. 延迟EMA启动 (`ema_start_step: 500`)
4. 检查数据归一化
5. 使用warmup schedule

### 风险3: 音视频不同步
**现象**: 生成的音频和视频内容不匹配

**解决方案**:
1. 增强fusion层的训练 (降低fusion参数的初始化scale)
2. 使用更强的cross-attention
3. 增加训练步数
4. 使用paired数据集 (确保音视频对齐)

### 风险4: 步数蒸馏失败
**现象**: 4步生成质量远低于50步

**解决方案**:
1. 增加训练步数 (从50K到100K+)
2. 调整判别器训练频率 (`dfake_gen_update_ratio`)
3. 增强教师模型的guidance scale
4. 使用更好的初始化 (从2步蒸馏模型warm start)

---

## 八、总结

本方案通过以下策略实现Ovi模型的四步蒸馏:

✅ **最小修改原则**: 复用现有DMD框架,只在必要处扩展  
✅ **模块化设计**: 将Ovi特有逻辑封装在独立模块中  
✅ **渐进式开发**: 分阶段实现,每个阶段都有明确的验证点  
✅ **详细文档**: 每个模块都有清晰的设计说明和代码示例  

**关键文件**:
1. `ovi_wrapper.py`: Ovi模型封装 (核心)
2. `ovi_dmd.py`: 双模态DMD实现 (核心)
3. `ovi_distillation.py`: Ovi蒸馏训练器 (核心)
4. `ovi_config.yaml`: 配置文件
5. `ovi_dataset.py`: 数据集

**预期工作量**: 6-10周 (包括开发、测试、调优)

**下一步**: 
1. 确认Ovi预训练权重的路径和格式
2. 准备音视频配对数据集
3. 开始Phase 1开发

---

## 九、常见问题FAQ

### Q1: 为什么不直接修改wan22_distillation.py?
**A**: 虽然可以修改,但会导致代码耦合度高,难以维护。独立实现可以:
- 保持原代码不变,便于对比
- 更容易测试和调试
- 未来支持更多模型时更灵活

### Q2: Ovi的fusion层需要蒸馏吗?
**A**: 是的。Fusion层是Ovi的核心创新,它们在联合训练时学到了音视频的对应关系。蒸馏时需要保持这些fusion层的能力。

### Q3: 可以先只蒸馏视频分支吗?
**A**: 可以,这是推荐的开发策略:
1. Phase 1: 只蒸馏视频分支 (类似Wan2.2)
2. Phase 2: 只蒸馏音频分支
3. Phase 3: 联合蒸馏 (启用fusion)

### Q4: 需要多少GPU?
**A**: 
- 最低: 8x A100 80GB (batch_size=1, gradient_accumulation)
- 推荐: 64x A100 80GB (total_batch_size=64)
- Fusion模型参数量约10B,显存需求较高

### Q5: 蒸馏需要多久?
**A**: 
- 50K steps (约5-7天,64 GPUs)
- 100K steps (约10-14天,64 GPUs)
- 取决于数据集大小和GPU数量

---

## 十、参考资料

1. **DMD论文**: [Distribution Matching Distillation](https://arxiv.org/abs/2405.14867)
2. **Self-Forcing论文**: 项目中的`selfforcing.pdf`
3. **Wan2.2文档**: 项目中的`wan22_distillation_analysis.md`
4. **Ovi论文**: 项目中的`Ovi.pdf`
5. **FSDP教程**: [PyTorch FSDP Documentation](https://pytorch.org/docs/stable/fsdp.html)

---

**最后更新**: 2025-10-23  
**作者**: Claude  
**状态**: 初稿待审核
