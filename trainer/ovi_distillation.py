# FILE: trainer/ovi_distillation.py (A new file, adapted from wan22_distillation.py)

import gc
import logging
import torch
import torch.distributed as dist
import wandb
import time
import os
from omegaconf import OmegaConf

# --- OVI IMPORTS ---
from model.ovi_dmd import OviDMD # MODIFIED: Import OviDMD
from utils.dataset import OviCSVDataset, cycle, OffsetDistributedSampler # MODIFIED: You need a new dataset class
from utils.distributed import EMA_FSDP, fsdp_wrap, fsdp_state_dict, launch_distributed_job
from utils.misc import set_seed, merge_dict_list

logger = logging.getLogger(__name__)

class Trainer: # MODIFIED: Renamed class
    def __init__(self, config):
        self.config = config
        self.step = 0

        # --- Step 1: Distributed Environment Setup (No changes needed) ---
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        launch_distributed_job()
        global_rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.is_main_process = global_rank == 0
        logger.info("Setting up the distributed environment...") if self.is_main_process else None
        self.dtype = torch.bfloat16 if config.mixed_precision else torch.float32
        self.device = torch.cuda.current_device()
        
        self.disable_wandb = config.disable_wandb
        logger.info(f"Using wandb: {not self.disable_wandb}") if self.is_main_process else None
        
        if config.seed == 0:
            random_seed = torch.randint(0, 10000000, (1,), device=self.device)
            dist.broadcast(random_seed, src=0)
            config.seed = random_seed.item()
        set_seed(config.seed + global_rank)
        if self.is_main_process and not self.disable_wandb:
            wandb.login(host=config.wandb_host, key=config.wandb_key)
            wandb.init(config=OmegaConf.to_container(config, resolve=True), name=config.config_name, project=config.wandb_project, dir=config.wandb_save_dir)
        self.output_path = config.logdir

        self.debug_distributed_training()
        logger.info(f"Finished setting up the distributed environment, world size: {self.world_size}") if self.is_main_process else None

        # --- Step 2: Initialize the OVI model and optimizer ---
        logger.info(f"Initializing the {config.distribution_loss} distillation model...") if self.is_main_process else None
        if config.distribution_loss == "dmd":
            self.model = OviDMD(config, device=self.device) # MODIFIED: Use OviDMD
        else:
            raise ValueError("Ovi trainer currently only supports 'dmd' loss")
        logger.info(f"Finished initializing the distillation model.") if self.is_main_process else None

        # ... (Resuming, FSDP wrapping, optimizer setup is mostly the same)
        # Note: The wrappers inside OviDMD's __init__ are already Ovi-specific
        pretrained_ckpt_path, self.step = self.load(self.output_path)
        if pretrained_ckpt_path is not None:
            logger.info(f"Loading checkpoint from {pretrained_ckpt_path} at step {self.step}") if self.is_main_process else None
            state_dict = torch.load(pretrained_ckpt_path, map_location="cpu")
            self.model.generator.load_state_dict(state_dict["generator"], strict=True)
            self.model.fake_score.load_state_dict(state_dict["critic"], strict=True)
        else:
            logger.info("No checkpoint found, training from scratch.") if self.is_main_process else None

        self.fake_score_state_dict_cpu = self.model.fake_score.state_dict()

        # FSDP Wrapping
        logger.info("Wrapping model components with FSDP...") if self.is_main_process else None
        logger.info(f"Before FSDP, model architecture: {self.model.generator}") if self.is_main_process else None
        orig_student = sum(p.numel() for p in self.model.generator.parameters() if p.requires_grad)
        logger.info(f"Before FSDP, student parameters: {orig_student/1e9:.2f}B") if self.is_main_process else None
        self.model.generator = fsdp_wrap(
            self.model.generator,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.generator_fsdp_wrap_strategy
        )
        logger.info(f"After FSDP, model architecture: {self.model.generator}") if self.is_main_process else None
        fsdp_student = sum(p.numel() for p in self.model.generator.parameters() if p.requires_grad)
        logger.info(f"After FSDP, generator parameters: {fsdp_student/1e9:.2f}B") if self.is_main_process else None

        orig_teacher = sum(p.numel() for p in self.model.real_score.parameters() if p.requires_grad)
        logger.info(f"Before FSDP, teacher parameters: {orig_teacher/1e9:.2f}B") if self.is_main_process else None
        self.model.real_score = fsdp_wrap(
            self.model.real_score,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.real_score_fsdp_wrap_strategy
        )
        fsdp_teacher = sum(p.numel() for p in self.model.real_score.parameters() if p.requires_grad)
        logger.info(f"After FSDP, teacher parameters: {fsdp_teacher/1e9:.2f}B") if self.is_main_process else None

        orig_critic = sum(p.numel() for p in self.model.fake_score.parameters() if p.requires_grad)
        logger.info(f"Before FSDP, critic parameters: {orig_critic/1e9:.2f}B") if self.is_main_process else None
        self.model.fake_score = fsdp_wrap(
            self.model.fake_score,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.fake_score_fsdp_wrap_strategy
        )
        fsdp_critic = sum(p.numel() for p in self.model.fake_score.parameters() if p.requires_grad)
        logger.info(f"After FSDP, critic parameters: {fsdp_critic/1e9:.2f}B") if self.is_main_process else None

        orig_text = sum(p.numel() for p in self.model.text_encoder.parameters() if p.requires_grad)
        logger.info(f"Before FSDP, text encoder parameters: {orig_text/1e9:.2f}B") if self.is_main_process else None
        self.model.text_encoder = fsdp_wrap(
            self.model.text_encoder,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.text_encoder_fsdp_wrap_strategy,
            cpu_offload=getattr(config, "text_encoder_cpu_offload", False)
        )
        fsdp_text = sum(p.numel() for p in self.model.text_encoder.parameters() if p.requires_grad)
        logger.info(f"After FSDP, text encoder parameters: {fsdp_text/1e9:.2f}B") if self.is_main_process else None
        
        orig_vae = sum(p.numel() for p in self.model.vae.parameters() if p.requires_grad)
        logger.info(f"VAE parameters: {orig_vae/1e9:.2f}B") if self.is_main_process else None
        self.model.vae = self.model.vae.to(device=self.device, dtype=self.dtype)
        logger.info("Finished wrapping model components with FSDP.") if self.is_main_process else None
        logger.info(f"GPU memory after FSDP wrapping: {torch.cuda.memory_allocated(self.device)/1e9:.2f} GB") if self.is_main_process else None

        # Optimizers
        logger.info("Setting up optimizers...") if self.is_main_process else None
        self.generator_optimizer = torch.optim.AdamW(
            [p for p in self.model.generator.parameters() if p.requires_grad], 
            lr=config.lr, 
            betas=(config.beta1, config.beta2), 
            weight_decay=config.weight_decay
        )
        self.critic_optimizer = torch.optim.AdamW(
            [p for p in self.model.fake_score.parameters() if p.requires_grad], 
            lr=config.lr_critic, 
            betas=(config.beta1_critic, config.beta2_critic), 
            weight_decay=config.weight_decay
        )
        logger.info("Finished setting up optimizers.") if self.is_main_process else None

        # --- Step 3: Initialize the OVI dataloader ---
        # MODIFIED: Use a dataset that returns both video and audio
        logger.info("Setting up dataset and dataloader...") if self.is_main_process else None
        dataset = OviCSVDataset(
            config.data_path,
            num_frames=config.num_frames,
            h=config.h,
            w=config.w,
            audio_sample_rate=config.audio_sample_rate,
            audio_duration_secs=config.audio_duration_secs,
        )
        sampler = OffsetDistributedSampler(dataset, initial_step=self.step, gpu_num=self.world_size, shuffle=False, drop_last=True)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, sampler=sampler, num_workers=8)
        self.dataloader = cycle(dataloader)
        logger.info(f"Finished setting up dataset and dataloader, dataset size: {len(dataset)}.") if self.is_main_process else None
        
        # --- Step 4: (EMA setup, checkpoint loading, etc. are the same as before) ---
        logger.info("Setting up EMA parameters...") if self.is_main_process else None
        rename_param = (
            lambda name: name.replace("_fsdp_wrapped_module.", "")
            .replace("_checkpoint_wrapped_module.", "")
            .replace("_orig_mod.", "")
        )
        self.name_to_trainable_params = {}
        for n, p in self.model.generator.named_parameters():
            if not p.requires_grad:
                continue

            renamed_n = rename_param(n)
            self.name_to_trainable_params[renamed_n] = p
        self.ema_weight = config.get("ema_weight", -1.0)
        self.ema_start_step = config.get("ema_start_step", 0)
        self.generator_ema = None

        if (self.ema_weight > 0.0) and (self.step >= self.ema_start_step):
            logger.info(f"Setting up EMA with weight {self.ema_weight}") if dist.get_rank() == 0 else None
            self.generator_ema = EMA_FSDP(self.model.generator, decay=self.ema_weight)
            
            # Load EMA state dict if available in checkpoint
            if pretrained_ckpt_path is not None:
                checkpoint_state_dict = torch.load(pretrained_ckpt_path, map_location="cpu")
                if "generator_ema" in checkpoint_state_dict:
                    logger.info("Loading generator_ema from checkpoint")
                    self.generator_ema.load_state_dict(checkpoint_state_dict["generator_ema"])
                else:
                    logger.info("No generator_ema found in checkpoint, starting fresh EMA")
        logger.info("Finished setting up EMA parameters.") if self.is_main_process else None
        
        # --- Step 5 (If resuming) Load the model and optimizer, lr_scheduler, ema's statedicts
        if getattr(config, "generator_ckpt", False):
            logger.info(f"Loading pretrained generator from {config.generator_ckpt}") if dist.get_rank() == 0 else None
            state_dict = torch.load(config.generator_ckpt, map_location="cpu")
            if "generator" in state_dict:
                state_dict = state_dict["generator"]
            elif "model" in state_dict:
                state_dict = state_dict["model"]
            self.model.generator.load_state_dict(
                state_dict, strict=True
            )
        
        self.max_grad_norm_generator = getattr(config, "max_grad_norm_generator", 10.0)
        self.max_grad_norm_critic = getattr(config, "max_grad_norm_critic", 10.0)
        self.previous_time = None

    def debug_distributed_training(self):
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        hostname = os.uname()[1] 

        logger.info(
            f"[DIAGNOSTIC] Hostname: {hostname}, "
            f"Global Rank: {rank}, "
            f"World Size: {world_size}, "
            f"Master Addr: {os.environ.get('MASTER_ADDR')}, "
            f"Node Rank: {os.environ.get('NODE_RANK')}"
        )
        
        if dist.is_initialized():
            dist.barrier()

    def load(self, out_path):
        # 1. Find latest checkpoint folder (ranked by step)
        if not os.path.exists(out_path): 
            return None, 0
        ckpt_folders = [f for f in os.listdir(out_path) if f.startswith("checkpoint_model_")]
        if not ckpt_folders: 
            return None, 0
        
        def extract_step(folder_name):
            import re
            match = re.search(r"checkpoint_model_(\d+)", folder_name)
            return int(match.group(1)) if match else -1
        latest_ckpt_folder = sorted(ckpt_folders, key=extract_step)[-1]
        
        # 2. read model.pt and step
        model_path = os.path.join(out_path, latest_ckpt_folder, "model.pt")
        step = extract_step(latest_ckpt_folder)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"{model_path} not found")
        return model_path, step

    def save(self):
        logger.info("Start gathering distributed model states...")
        generator_state_dict = fsdp_state_dict(
            self.model.generator)
        critic_state_dict = fsdp_state_dict(
            self.model.fake_score)

        if (self.ema_weight > 0.0) and (self.ema_start_step < self.step):
            state_dict = {
                "generator": generator_state_dict,
                "critic": critic_state_dict,
                "generator_ema": self.generator_ema.state_dict(),
            }
        else:
            state_dict = {
                "generator": generator_state_dict,
                "critic": critic_state_dict,
            }

        if self.is_main_process:
            os.makedirs(os.path.join(self.output_path,
                        f"checkpoint_model_{self.step:06d}"), exist_ok=True)
            torch.save(state_dict, os.path.join(self.output_path,
                       f"checkpoint_model_{self.step:06d}", "model.pt"))
            logger.info("Model saved to", os.path.join(self.output_path, f"checkpoint_model_{self.step:06d}", "model.pt"))

    def fwdbwd_one_step(self, batch, train_generator):
        # --- HEAVILY MODIFIED FOR OVI ---
        self.model.eval()

        if self.step % 20 == 0:
            torch.cuda.empty_cache()

        # Step 1: Get batch of (text, video, audio)
        text_prompts = batch["prompts"]
        video_tensor = batch["video"].to(device=self.device, dtype=self.dtype)      # shape: [B, C, F, H, W]
        audio_tensor = batch["audio"].to(device=self.device, dtype=self.dtype)      # shape: [B, L]

        # Step 2: Encode inputs to latents
        with torch.no_grad():
            # Video: get clean latent and special first-frame latent
            first_frame = video_tensor[:, :, :1, :, :]
            wan22_image_latent = self.model.vae.encode_video(first_frame) # shape: [B=1, F=1, C=48, H//16, W//16]
            
            # Text encoding
            conditional_dict = self.model.text_encoder(text_prompts=text_prompts)
            if not getattr(self, "unconditional_dict", None):
                self.unconditional_dict_v = self.model.text_encoder(text_prompts=[self.config.video_negative_prompt] * len(text_prompts))
                self.unconditional_dict_a = self.model.text_encoder(text_prompts=[self.config.audio_negative_prompt] * len(text_prompts))
                unconditional_dict = {k: v.detach() for k, v in self.unconditional_dict_v.items()}
                unconditional_dict.update({k: v.detach() for k, v in self.unconditional_dict_a.items()})
                self.unconditional_dict = unconditional_dict  # cache the unconditional_dict
            unconditional_dict = self.unconditional_dict
        
        # Define latent shapes from config
        batch_size = len(text_prompts)
        # video_latent_shape = self.config.video_latent_shape     # [1, 31, 48, 44, 80]
        # audio_latent_shape = self.config.audio_latent_shape     # [1, 157, 20]
        _, _, _, H, W = video_tensor.shape
        video_latent_shape = [1, 31, 48, H // 16, W // 16]  # Modify height and width according to wan22_image_latent
        audio_latent_shape = [1, 157, 20]                   # Audio latent shape based on audio length

        video_latent_shape[0] = batch_size
        audio_latent_shape[0] = batch_size
        latent_shapes = (video_latent_shape, audio_latent_shape)

        # Step 3: Call generator or critic loss
        if train_generator:
            generator_loss, generator_log_dict = self.model.generator_loss(
                latent_shapes=latent_shapes,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict,
                wan22_image_latent=wan22_image_latent,
            )
            torch.cuda.empty_cache()
            generator_loss.backward()
            generator_grad_norm = self.model.generator.clip_grad_norm_(self.max_grad_norm_generator)
            generator_log_dict.update({"generator_loss": generator_loss, 
                                       "generator_grad_norm": generator_grad_norm})
            return generator_log_dict
        else:
            generator_log_dict = {}
        
        # Step 4: Store gradients for the critic (if training the critic)
        critic_loss, critic_log_dict = self.model.critic_loss(
            latent_shapes=latent_shapes,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            wan22_image_latent=wan22_image_latent,
        )
        critic_loss.backward()
        critic_grad_norm = self.model.fake_score.clip_grad_norm_(self.max_grad_norm_critic)
        critic_log_dict.update({"critic_loss": critic_loss, 
                                "critic_grad_norm": critic_grad_norm})
        return critic_log_dict

    def train(self):
        # --- MODIFIED FOR OVI LOGGING ---
        start_step = self.step
        while True:
            if self.is_main_process:
                print(f"training step {self.step} ...")
            TRAIN_GENERATOR = self.step % self.config.dfake_gen_update_ratio == 0

            # Train Generator
            if TRAIN_GENERATOR:
                self.generator_optimizer.zero_grad(set_to_none=True)
                batch = next(self.dataloader)
                generator_log_dict = self.fwdbwd_one_step(batch, train_generator=True)
                if not self.config.debug:
                    self.generator_optimizer.step()
                    if self.generator_ema is not None: 
                        self.generator_ema.update(self.model.generator)
            
            # Train Critic
            self.critic_optimizer.zero_grad(set_to_none=True)
            batch = next(self.dataloader)
            critic_log_dict = self.fwdbwd_one_step(batch, train_generator=False)
            if not self.config.debug:
                self.critic_optimizer.step()

            self.step += 1
            
            # EMA creation
            if (self.step >= self.ema_start_step) and (self.generator_ema is None) and (self.ema_weight > 0):
                self.generator_ema = EMA_FSDP(self.model.generator, decay=self.ema_weight)

            # Save model
            if (not self.config.no_save) and (self.step - start_step) > 0 and (self.step % self.config.log_iters == 0):
                torch.cuda.empty_cache()
                self.save()
                torch.cuda.empty_cache()

            # Logging
            if self.is_main_process:
                wandb_log = {}
                if TRAIN_GENERATOR:
                    wandb_log.update({
                        "Loss/Generator": generator_log_dict["generator_loss"].mean().item(),
                        "Loss/Generator_Video": generator_log_dict["dmd_loss_video"].mean().item(),
                        "Loss/Generator_Audio": generator_log_dict["dmd_loss_audio"].mean().item(),
                        "GradNorm/Generator": generator_log_dict["generator_grad_norm"].mean().item(),
                        "GradNorm/DMD_Video": generator_log_dict["dmdtrain_gradient_norm_video"].mean().item(),
                        "GradNorm/DMD_Audio": generator_log_dict["dmdtrain_gradient_norm_audio"].mean().item(),
                    })
                
                wandb_log.update({
                    "Loss/Critic": critic_log_dict["critic_loss"].mean().item(),
                    "Loss/Critic_Video": critic_log_dict["critic_loss_video"].mean().item(),
                    "Loss/Critic_Audio": critic_log_dict["critic_loss_audio"].mean().item(),
                    "GradNorm/Critic": critic_log_dict["critic_grad_norm"].mean().item(),
                })
                if not self.disable_wandb:
                    wandb.log(wandb_log, step=self.step)

            # Garbage Collection
            if self.step % self.config.get("gc_interval", 20) == 0:
                gc.collect()
                torch.cuda.empty_cache()
            
            if self.is_main_process:
                current_time = time.time()
                if self.previous_time is None:
                    self.previous_time = current_time
                else:
                    if not self.disable_wandb:
                        wandb.log({"per iteration time": current_time - self.previous_time}, step=self.step)
                    self.previous_time = current_time