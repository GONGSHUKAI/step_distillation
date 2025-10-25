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

logger = logging.getLogger()

class OviScoreDistillationTrainer: # MODIFIED: Renamed class
    def __init__(self, config):
        self.config = config
        self.step = 0

        # --- Step 1: Distributed Environment Setup (No changes needed) ---
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        launch_distributed_job()
        global_rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.dtype = torch.bfloat16 if config.mixed_precision else torch.float32
        self.device = torch.cuda.current_device()
        self.is_main_process = global_rank == 0
        self.disable_wandb = config.disable_wandb
        # ... (rest of setup, seeds, wandb init is the same)
        if config.seed == 0:
            random_seed = torch.randint(0, 10000000, (1,), device=self.device)
            dist.broadcast(random_seed, src=0)
            config.seed = random_seed.item()
        set_seed(config.seed + global_rank)
        if self.is_main_process and not self.disable_wandb:
            wandb.login(host=config.wandb_host, key=config.wandb_key)
            wandb.init(config=OmegaConf.to_container(config, resolve=True), name=config.config_name, project=config.wandb_project, dir=config.wandb_save_dir)
        self.output_path = config.logdir


        # --- Step 2: Initialize the OVI model and optimizer ---
        if config.distribution_loss == "dmd":
            self.model = OviDMD(config, device=self.device) # MODIFIED: Use OviDMD
        else:
            raise ValueError("Ovi trainer currently only supports 'dmd' loss")
        
        # ... (Resuming, FSDP wrapping, optimizer setup is mostly the same)
        # Note: The wrappers inside OviDMD's __init__ are already Ovi-specific
        pretrained_ckpt_path, self.step = self.load(self.output_path)
        if pretrained_ckpt_path is not None:
             if self.is_main_process: print(f"Loading checkpoint from {pretrained_ckpt_path} at step {self.step}")
             state_dict = torch.load(pretrained_ckpt_path, map_location="cpu")
             self.model.generator.load_state_dict(state_dict["generator"], strict=True)
             self.model.fake_score.load_state_dict(state_dict["critic"], strict=True)
        
        # FSDP Wrapping
        self.model.generator = fsdp_wrap(self.model.generator, **config.get("fsdp_kwargs", {}))
        self.model.real_score = fsdp_wrap(self.model.real_score, **config.get("fsdp_kwargs", {}))
        self.model.fake_score = fsdp_wrap(self.model.fake_score, **config.get("fsdp_kwargs", {}))
        self.model.text_encoder = fsdp_wrap(self.model.text_encoder, cpu_offload=config.text_encoder_cpu_offload, **config.get("fsdp_kwargs", {}))
        self.model.vae = self.model.vae.to(device=self.device, dtype=self.dtype) # VAE to device

        # Optimizers
        self.generator_optimizer = torch.optim.AdamW([p for p in self.model.generator.parameters() if p.requires_grad], lr=config.lr, betas=(config.beta1, config.beta2), weight_decay=config.weight_decay)
        self.critic_optimizer = torch.optim.AdamW([p for p in self.model.fake_score.parameters() if p.requires_grad], lr=config.lr_critic, betas=(config.beta1_critic, config.beta2_critic), weight_decay=config.weight_decay)


        # --- Step 3: Initialize the OVI dataloader ---
        # MODIFIED: Use a dataset that returns both video and audio
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
        
        # --- (EMA setup, checkpoint loading, etc. are the same as before) ---
        self.ema_weight = config.get("ema_weight", -1.0)
        self.ema_start_step = config.get("ema_start_step", 0)
        self.generator_ema = None
        # ... (rest of the __init__ is the same)
        self.max_grad_norm_generator = getattr(config, "max_grad_norm_generator", 10.0)
        self.max_grad_norm_critic = getattr(config, "max_grad_norm_critic", 10.0)

    def load(self, out_path):
        # ... (This method can remain the same)
        if not os.path.exists(out_path): return None, 0
        ckpt_folders = [f for f in os.listdir(out_path) if f.startswith("checkpoint_model_")]
        if not ckpt_folders: return None, 0
        def extract_step(folder_name):
            import re
            match = re.search(r"checkpoint_model_(\d+)", folder_name)
            return int(match.group(1)) if match else -1
        latest_ckpt_folder = sorted(ckpt_folders, key=extract_step)[-1]
        model_path = os.path.join(out_path, latest_ckpt_folder, "model.pt")
        step = extract_step(latest_ckpt_folder)
        return model_path, step

    def save(self):
        # ... (This method can remain the same, as it saves the FSDP wrapped models)
        generator_state_dict = fsdp_state_dict(self.model.generator)
        critic_state_dict = fsdp_state_dict(self.model.fake_score)
        state_dict = {"generator": generator_state_dict, "critic": critic_state_dict}
        if (self.ema_weight > 0.0) and (self.ema_start_step < self.step):
            state_dict["generator_ema"] = self.generator_ema.state_dict()
        if self.is_main_process:
            save_dir = os.path.join(self.output_path, f"checkpoint_model_{self.step:06d}")
            os.makedirs(save_dir, exist_ok=True)
            torch.save(state_dict, os.path.join(save_dir, "model.pt"))

    def fwdbwd_one_step(self, batch, train_generator):
        # --- HEAVILY MODIFIED FOR OVI ---
        self.model.eval()

        # Step 1: Get batch of (text, video, audio)
        text_prompts = batch["prompts"]
        video_tensor = batch["video"].to(device=self.device, dtype=self.dtype)
        audio_tensor = batch["audio"].to(device=self.device, dtype=self.dtype) # NEW

        # Step 2: Encode inputs to latents
        with torch.no_grad():
            # Video: get clean latent and special first-frame latent
            first_frame = video_tensor[:, :, :1, :, :]
            wan22_image_latent = self.model.vae.video_vae.encode_to_latent(first_frame)
            
            # Audio: get clean latent
            # The OviVAEWrapper should have a method for this
            clean_audio_latent = self.model.vae.audio_vae.encode_to_latent(audio_tensor)
            
            # Text encoding
            conditional_dict = self.model.text_encoder(text_prompts=text_prompts)
            if not getattr(self, "unconditional_dict", None):
                self.unconditional_dict = self.model.text_encoder(text_prompts=[self.config.negative_prompt] * len(text_prompts))
            unconditional_dict = self.unconditional_dict
        
        # Define latent shapes from config
        batch_size = len(text_prompts)
        video_latent_shape = self.config.video_latent_shape
        audio_latent_shape = self.config.audio_latent_shape
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
            generator_loss.backward()
            generator_grad_norm = self.model.generator.clip_grad_norm_(self.max_grad_norm_generator)
            generator_log_dict.update({"generator_loss": generator_loss, "generator_grad_norm": generator_grad_norm})
            return generator_log_dict
        else:
            critic_loss, critic_log_dict = self.model.critic_loss(
                latent_shapes=latent_shapes,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict,
                wan22_image_latent=wan22_image_latent,
            )
            critic_loss.backward()
            critic_grad_norm = self.model.fake_score.clip_grad_norm_(self.max_grad_norm_critic)
            critic_log_dict.update({"critic_loss": critic_loss, "critic_grad_norm": critic_grad_norm})
            return critic_log_dict

    def train(self):
        # --- MODIFIED FOR OVI LOGGING ---
        start_step = self.step
        while True:
            TRAIN_GENERATOR = self.step % self.config.dfake_gen_update_ratio == 0

            # Train Generator
            if TRAIN_GENERATOR:
                self.generator_optimizer.zero_grad(set_to_none=True)
                batch = next(self.dataloader)
                generator_log_dict = self.fwdbwd_one_step(batch, train_generator=True)
                if not self.config.debug:
                    self.generator_optimizer.step()
                    if self.generator_ema is not None: self.generator_ema.update(self.model.generator)
            
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
            if (not self.config.no_save) and (self.step % self.config.log_iters == 0):
                self.save()

            # Logging
            if self.is_main_process and not self.disable_wandb:
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
                wandb.log(wandb_log, step=self.step)

            # Garbage Collection
            if self.step % self.config.get("gc_interval", 20) == 0:
                gc.collect()
                torch.cuda.empty_cache()