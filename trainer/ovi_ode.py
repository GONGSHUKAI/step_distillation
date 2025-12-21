# FILE: trainer/ovi_ode.py

import gc
import logging
from utils.dataset import OviODERegressionDataset, cycle
from model.ovi_ode_regression import OviODERegression # <--- Will define this next
from collections import defaultdict
from utils.misc import set_seed, merge_dict_list
from utils.ovi_wrapper import remap_ovi_state_dict_for_refactored
import torch.distributed as dist
from omegaconf import OmegaConf
import torch
import wandb
import time
import os

from utils.distributed import barrier, fsdp_wrap, fsdp_state_dict, launch_distributed_job
from ovi.modules.causal_ovi import CausalFusionAttentionBlock # For FSDP wrapping

logger = logging.getLogger(__name__)

class Trainer:
    def __init__(self, config):
        self.config = config
        self.step = 0
        
        # --- Step 1: Distributed Setup ---
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        launch_distributed_job()
        global_rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.is_main_process = dist.get_rank() == 0
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
        
        if self.is_main_process and not config.disable_wandb:
            wandb.login(host=config.wandb_host, key=config.wandb_key)
            wandb.init(
                config=OmegaConf.to_container(config, resolve=True),
                name=config.config_name,
                mode="online",
                entity=config.wandb_entity,
                project=config.wandb_project,
                dir=config.wandb_save_dir
            )
            
        self.output_path = config.logdir

        # --- Step 2: Model Init ---
        logger.info(f"Initializing Ovi ODE Regression Model...") if self.is_main_process else None
        assert config.distribution_loss == "ode", "Only ODE loss is supported for ODE training"
        self.model = OviODERegression(config, device=self.device)
        logger.info(f"Finished initializing the distillation model.") if self.is_main_process else None
        
        pretrained_ckpt_path, self.step = self.load(self.output_path)
        if pretrained_ckpt_path is not None:
            logger.info(f"Loading checkpoint from {pretrained_ckpt_path} at step {self.step}") if self.is_main_process else None
            state_dict = torch.load(pretrained_ckpt_path, map_location="cpu")
            self.model.generator.load_state_dict(state_dict["generator"], strict=True)
            logger.info(f"Checkpoint at step {self.step} loaded") if self.is_main_process else None
            # Free memory
            del state_dict  
            gc.collect()
            torch.cuda.empty_cache()
        else:
            logger.info("No checkpoint found, training from scratch.") if self.is_main_process else None

        # FSDP Wrapping
        logger.info("Wrapping generator with FSDP...") if self.is_main_process else None
        # Ovi generator structure: self.model.generator.model (FusionModel) -> fusion_blocks
        # We wrap the FusionAttentionBlock
        orig_student = sum(p.numel() for p in self.model.generator.parameters() if p.requires_grad)
        logger.info(f"Before FSDP, generator parameters: {orig_student/1e9:.2f}B") if self.is_main_process else None
        self.model.generator = fsdp_wrap(
            self.model.generator,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.generator_fsdp_wrap_strategy,
            transformer_module=(CausalFusionAttentionBlock, ) # <--- Important!
        )
        fsdp_student = sum(p.numel() for p in self.model.generator.parameters() if p.requires_grad)
        logger.info(f"After FSDP, generator parameters: {fsdp_student/1e9:.2f}B") if self.is_main_process else None
        
        # Text Encoder Wrapping
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

        # Optimizer
        logger.info("Setting up optimizers...") if self.is_main_process else None
        self.generator_optimizer = torch.optim.AdamW(
            [p for p in self.model.generator.parameters() if p.requires_grad],
            lr=config.lr,
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay
        )
        logger.info("Finished setting up optimizers.") if self.is_main_process else None

        # --- Step 3: Dataset ---
        logger.info(f"Setting up dataset and dataloader...") if self.is_main_process else None
        dataset = OviODERegressionDataset(config.data_path)
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True, drop_last=True)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=config.batch_size, sampler=sampler, num_workers=4
        )
        self.dataloader = cycle(dataloader)
        logger.info(f"Finished setting up dataset and dataloader, dataset class name: {dataset.__class__.__name__}, size: {len(dataset)}, batch size: {config.batch_size}") if self.is_main_process else None

        self.max_grad_norm = 10.0
        self.previous_time = None

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
        logger.info("Start gathering distributed model states...") if self.is_main_process else None
        state_dict = {"generator": fsdp_state_dict(self.model.generator)}
        
        if self.is_main_process:
            os.makedirs(os.path.join(self.output_path, f"checkpoint_model_{self.step:06d}"), exist_ok=True)
            torch.save(state_dict, os.path.join(self.output_path, f"checkpoint_model_{self.step:06d}", "model.pt"))
            logger.info(f"Model saved to {os.path.join(self.output_path, f'checkpoint_model_{self.step:06d}', 'model.pt')}")


    def train_one_step(self):
        self.model.eval() # Ovi is trained in Eval mode (no dropout)
        
        batch = next(self.dataloader)
        
        # [B, 5, 32, 48, H, W]
        video_ode_latent = batch["video_ode_latent"].to(self.device, self.dtype)
        # [B, 5, 160, 20]
        audio_ode_latent = batch["audio_ode_latent"].to(self.device, self.dtype)
        prompts = batch["prompts"]

        with torch.no_grad():
            conditional_dict = self.model.text_encoder(prompts)
        
        # Compute Loss
        loss, log_dict = self.model.generator_loss(
            video_ode_latent=video_ode_latent,
            audio_ode_latent=audio_ode_latent,
            conditional_dict=conditional_dict
        )
        
        # Optimization
        self.generator_optimizer.zero_grad()
        loss.backward()
        grad_norm = self.model.generator.clip_grad_norm_(self.max_grad_norm)
        self.generator_optimizer.step()
        
        # Logging
        if self.is_main_process:
            wandb_log = {
                "loss": loss.item(),
                "grad_norm": grad_norm.item(),
                "video_loss": log_dict["loss_video"].item(),
                "audio_loss": log_dict["loss_audio"].item()
            }
            if not self.config.disable_wandb:
                wandb.log(wandb_log, step=self.step)
            
            logger.info(f"Step {self.step}: ODE Loss={loss.item():.4f}, Video ODE Loss={wandb_log['video_loss']:.4f}, Audio ODE Loss={wandb_log['audio_loss']:.4f}, Grad Norm={wandb_log['grad_norm']}")

    def train(self):
        start_step = self.step
        while True:
            self.train_one_step()
            self.step += 1

            if self.step > 0 and self.step % self.config.log_iters == 0:
                self.save()
                torch.cuda.empty_cache()

            if self.is_main_process:
                current_time = time.time()
                if self.previous_time is None:
                    self.previous_time = current_time
                else:
                    if not self.disable_wandb:
                        wandb.log({"per iteration time": current_time - self.previous_time}, step=self.step)
                    self.previous_time = current_time

            if self.step % 100 == 0:
                gc.collect()
                torch.cuda.empty_cache()