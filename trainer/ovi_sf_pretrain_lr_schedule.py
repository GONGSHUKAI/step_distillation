# FILE: trainer/ovi_ode.py

import gc
import logging
from utils.dataset import OviODERegressionDataset, cycle, process_visual
from model.ovi_sf_regression import OviSelfForcingRegression # <--- Will define this next
from collections import defaultdict
from utils.misc import set_seed, merge_dict_list
from utils.ovi_wrapper import remap_ovi_state_dict_for_refactored
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from omegaconf import OmegaConf
import torch
import wandb
import time
import os
from tqdm import tqdm
from utils.distributed import barrier, fsdp_wrap, fsdp_state_dict, launch_distributed_job, fsdp_optim_state_dict
from ovi.modules.causal_ovi import CausalFusionAttentionBlock # For FSDP wrapping
from ovi.utils.io_utils import save_video
import csv
import tempfile
import numpy as np
from transformers import get_cosine_schedule_with_warmup, get_constant_schedule_with_warmup

logger = logging.getLogger(__name__)

# Helper function to extract step from checkpoint folder name
def extract_step(folder_name):
    import re
    match = re.search(r"checkpoint_model_(\d+)", folder_name)
    return int(match.group(1)) if match else -1


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
        self.debug_distributed_training()
        # === NEW: Gradient Accumulation Setup ===
        self.gradient_accumulation_steps = getattr(config, "gradient_accumulation_steps", 1)
        if self.gradient_accumulation_steps > 1 and self.is_main_process:
            logger.info(f"Using gradient accumulation with {self.gradient_accumulation_steps} steps.")

        self.video_log_iter = self.config.video_log_iters
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
        logger.info(f"Initializing Ovi Self Forcing Regression Model...") if self.is_main_process else None
        assert config.distribution_loss == "ode", "Only ODE loss is supported for ODE training"
        self.model = OviSelfForcingRegression(config, device=self.device)
        logger.info(f"Finished initializing the Self Forcing Regression model.") if self.is_main_process else None
        
        # Load checkpoint (model weights only, optimizer will be loaded after FSDP wrapping)
        pretrained_ckpt_path, self.step = self.load(self.output_path)
        state_dict = None
        if pretrained_ckpt_path is not None:
            logger.info(f"Loading checkpoint from {pretrained_ckpt_path} at step {self.step}") if self.is_main_process else None
            state_dict = torch.load(pretrained_ckpt_path, map_location="cpu")
            self.model.generator.load_state_dict(state_dict["generator"], strict=True)
            logger.info(f"Checkpoint at step {self.step} loaded") if self.is_main_process else None
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

        self.model.vae = self.model.vae.to(device=self.device, dtype=self.dtype)
        
        # Optimizer
        logger.info("Setting up optimizers...") if self.is_main_process else None
        self.generator_optimizer = torch.optim.AdamW(
            [p for p in self.model.generator.parameters() if p.requires_grad],
            lr=config.lr,
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay
        )
        logger.info("Finished setting up optimizers.") if self.is_main_process else None
        
        self.lr_schedule_type = getattr(config, "lr_schedule", "constant")
        self.warmup_steps = getattr(config, "warmup_steps", 0)
        if self.lr_schedule_type == "cosine":
            self.max_train_steps = getattr(config, "max_train_steps", 30000)
            logger.info(f"Using Cosine LR Schedule with warmup={self.warmup_steps}, max_steps={self.max_train_steps}") if self.is_main_process else None
            self.lr_scheduler = get_cosine_schedule_with_warmup(
                optimizer=self.generator_optimizer,
                num_warmup_steps=self.warmup_steps,
                num_training_steps=self.max_train_steps
            )
        else:
            logger.info(f"Using Constant LR Schedule with warmup={self.warmup_steps}") if self.is_main_process else None
            self.lr_scheduler = get_constant_schedule_with_warmup(
                optimizer=self.generator_optimizer,
                num_warmup_steps=self.warmup_steps
            )
            
        # Load optimizer state if resuming (must be done AFTER FSDP wrapping and optimizer creation)
        if pretrained_ckpt_path is not None and state_dict is not None:
            if state_dict.get("generator_optimizer", None) is not None:
                logger.info("Loading generator optimizer state from checkpoint") if self.is_main_process else None
                self.generator_optimizer.load_state_dict(
                    FSDP.optim_state_dict_to_load(
                        self.model.generator,
                        self.generator_optimizer,
                        state_dict["generator_optimizer"]
                    )
                )
                logger.info("Generator optimizer state loaded") if self.is_main_process else None
            else:
                logger.info("No generator_optimizer found in checkpoint, starting with fresh optimizer state") if self.is_main_process else None

            if "lr_scheduler" in state_dict:
                logger.info("Loading lr_scheduler state from checkpoint") if self.is_main_process else None
                self.lr_scheduler.load_state_dict(state_dict["lr_scheduler"])
            else:
                if self.step > 0:
                    logger.info(f"No lr_scheduler in checkpoint. Manually stepping scheduler {self.step} times to sync.") if self.is_main_process else None
                    for _ in range(self.step): self.lr_scheduler.step()

        # Free memory after loading
        del state_dict
        gc.collect()
        torch.cuda.empty_cache()

        # --- Step 3: Dataset ---
        logger.info(f"Setting up dataset and dataloader...") if self.is_main_process else None
        dataset = OviODERegressionDataset(config.data_path)
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True, drop_last=True)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=config.batch_size, sampler=sampler, num_workers=4, drop_last=True
        )
        self.dataloader = cycle(dataloader)
        logger.info(f"Finished setting up dataset and dataloader, dataset class name: {dataset.__class__.__name__}, size: {len(dataset)}, batch size: {config.batch_size}") if self.is_main_process else None

        self.max_grad_norm = 10.0
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
        
        latest_ckpt_folder = sorted(ckpt_folders, key=extract_step)[-1]
        
        # 2. Read checkpoint.pt and step (changed from model.pt to checkpoint.pt for consistency)
        # Try checkpoint.pt first, fallback to model.pt for backward compatibility
        model_path = os.path.join(out_path, latest_ckpt_folder, "checkpoint.pt")
        if not os.path.exists(model_path):
            model_path = os.path.join(out_path, latest_ckpt_folder, "model.pt")
        
        step = extract_step(latest_ckpt_folder)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Neither checkpoint.pt nor model.pt found in {os.path.join(out_path, latest_ckpt_folder)}")
        return model_path, step
    
    def save(self):
        os.makedirs(self.output_path, exist_ok=True)
        
        # Optional: Clean up old checkpoints (similar to ovi_distillation.py)
        checkpoint_folders = os.listdir(self.output_path)
        checkpoint_infos = [
            (checkpoint_folder, extract_step(checkpoint_folder))
            for checkpoint_folder in checkpoint_folders if checkpoint_folder.startswith("checkpoint_model_")
        ]
        # Sort by step number
        checkpoint_infos = sorted(checkpoint_infos, key=lambda x: x[1], reverse=True)
        # Filter out checkpoints to keep
        checkpoint_infos = [info for info in checkpoint_infos if info[1] not in self.config.get("checkpoints_to_keep", [])]

        # NOTE: maybe ensure safer deletion since we are all using root
        # if (
        #     (len(checkpoint_infos) >= self.config.get("num_keep_checkpoints", float("inf"))) # since we are also going to save one, we remove one when reaching the number
        #     and (dist.get_rank() == 0)
        # ): # a very large number, lol
        #     # remove the oldest info
        #     os.removedirs(os.path.join(self.output_path, checkpoint_infos[-1][0]))

        logger.info("Start gathering distributed model states...") if self.is_main_process else None
        
        # Gather generator state dict
        generator_state_dict = fsdp_state_dict(self.model.generator)
        
        # Prepare complete state dictionary
        state_dict = {
            "generator": generator_state_dict,
            "generator_optimizer": fsdp_optim_state_dict(
                self.model.generator,
                self.generator_optimizer
            ),
            "step": self.step,
            "lr_scheduler": self.lr_scheduler.state_dict()
        }
        
        if self.is_main_process:
            # Create checkpoint directory
            checkpoint_dir = os.path.join(self.output_path, f"checkpoint_model_{self.step:06d}")
            os.makedirs(checkpoint_dir, exist_ok=True)
            logger.info("Finished gathering distributed model states.")
            
            # Save model and optimizer states (using checkpoint.pt for consistency)
            model_path = os.path.join(checkpoint_dir, "checkpoint.pt")
            torch.save(state_dict, model_path)
            logger.info(f"Model and optimizer saved to {model_path}")

    def _load_eval_csv(self, csv_path: str, max_len=5):
        eval_data = []
        if not os.path.exists(csv_path): return eval_data
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for row in reader:
                if len(row) >= 2:
                    eval_data.append({
                        "prompt": row[0],
                        "image_path": row[1],
                        "seed": int(row[2]) if len(row) > 2 else 0  # default seed is 0
                    })
        eval_data = eval_data[:max_len] if len(eval_data) > max_len else eval_data
        return eval_data

    @torch.no_grad()
    def run_eval(self) -> dict:
        eval_csv_path = getattr(self.config, "eval_csv", None)
        if eval_csv_path is None or not os.path.exists(eval_csv_path):
            if self.is_main_process:
                logger.warning(f"Eval CSV not found: {eval_csv_path}")
            return {}
        
        eval_data = self._load_eval_csv(eval_csv_path)
        if len(eval_data) == 0:
            if self.is_main_process:
                logger.warning("Eval CSV is empty")
            return {}
        
        eval_videos = []
        
        for idx, sample in tqdm(enumerate(eval_data), total=len(eval_data), disable=(dist.get_rank()!=0), desc="Running ODE Eval"):
            prompt = sample["prompt"]
            image_path = sample["image_path"]
            # set_seed(sample["seed"])

            image_exists = image_path and os.path.exists(image_path)
            if image_exists:
                processed_frame = process_visual(image_path, w=1280, h=704) # 对应配置中的尺寸
                processed_frame = processed_frame.to(self.device, dtype=self.dtype)
                wan22_image_latent = self.model.vae.encode_video(processed_frame.unsqueeze(2))
                
                conditional_dict = self.model.text_encoder(text_prompts=[prompt])
                conditional_dict = {
                    "video_prompt_embeds": conditional_dict["prompt_embeds"].detach(),
                    "audio_prompt_embeds": conditional_dict["prompt_embeds"].detach(),
                }
                
                video_latent_shape = self.config.video_latent_shape
                audio_latent_shape = self.config.audio_latent_shape
                noise_video = torch.randn(1, *video_latent_shape, device=self.device, dtype=self.dtype)
                noise_audio = torch.randn(1, *audio_latent_shape, device=self.device, dtype=self.dtype)
                noises = (noise_video, noise_audio)
                
                video_latent, audio_latent = self.model.full_inference(
                    noises=noises,
                    wan22_image_latent=wan22_image_latent,
                    **conditional_dict
                )
                
                video = self.model.vae.decode_video(video_latent)
                audio = self.model.vae.decode_audio(audio_latent.transpose(1, 2))

                if self.is_main_process:
                    video = ((video + 1) / 2 * 255).clip(0, 255)
                    video_np = video.squeeze(0).permute(1, 0, 2, 3).cpu().float().numpy().astype(np.uint8)
                    audio_np = audio.squeeze(0).cpu().float().numpy().flatten()
                    
                    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
                        save_video(f.name, video_np, audio_np)
                        eval_videos.append({"path": f.name, "idx": idx})
                
                # All ranks clean up
                torch.cuda.empty_cache()
            else: 
                # Image not found - all ranks skip but still sync
                logger.warning(f"Image not found: {image_path}, skipping sample {idx}") if self.is_main_process else None
            
            if dist.is_initialized():
                dist.barrier()
        
        wandb_log = {}
        if self.is_main_process:
            for item in eval_videos:
                wandb_log[f"Evaluation/Sample_{item['idx']}"] = wandb.Video(item["path"], format="mp4")
        
        return wandb_log


    def fwdbwd_one_step(self, batch, log_video=False):
        """Forward and backward for one step (no optimizer step)."""
        self.model.eval()  # Ovi is trained in Eval mode (no dropout)
        
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
            conditional_dict=conditional_dict,
            log_video=log_video
        )
        
        if self.gradient_accumulation_steps > 1:
            loss = loss / self.gradient_accumulation_steps
        
        loss.backward()
        
        log_dict.update({
            "loss": loss * self.gradient_accumulation_steps,  # Log unscaled loss
        })
        return log_dict

    def train(self):
        start_step = self.step
        while True:
            if self.video_log_iter == 0:
                LOG_VIDEO = False
            else:
                LOG_VIDEO = self.step % self.video_log_iter == 0
            self.generator_optimizer.zero_grad()
            
            for accum_step in tqdm(range(self.gradient_accumulation_steps), total=len(range(self.gradient_accumulation_steps)), desc="Gradient accumulating for ODE pretraining", leave=False, disable=(dist.get_rank()!=0)):
                batch = next(self.dataloader)
                log_dict = self.fwdbwd_one_step(batch, LOG_VIDEO)
            
            # Optimizer step after accumulation
            grad_norm = self.model.generator.clip_grad_norm_(self.max_grad_norm)
            self.generator_optimizer.step()
            self.lr_scheduler.step()

            self.step += 1

            # Logging (use the last step's log_dict)
            if LOG_VIDEO: 
                logger.info(f"Step {self.step}: Running on-the-fly evaluation...") if self.is_main_process else None
                try:
                    eval_wandb_log = self.run_eval()
                    if self.is_main_process and not self.config.disable_wandb:
                        wandb.log(eval_wandb_log, step=self.step)
                except Exception as e:
                    logger.error(f"Eval failed at step {self.step}: {e}") if self.is_main_process else None

            if self.is_main_process:
                wandb_log = {
                    "loss": log_dict["loss"].item(),
                    "grad_norm": grad_norm.item(),
                    "video_loss": log_dict["loss_video"].item(),
                    "audio_loss": log_dict["loss_audio"].item(),
                    "lr": self.lr_scheduler.get_last_lr()[0],
                }
                if LOG_VIDEO: 
                    wandb_log.update({
                        "Visualization/Generated_Video_Audio": wandb.Video(log_dict['generated_video_audio'], format="mp4"),
                        "Visualization/Ground_Truth_Video_Audio": wandb.Video(log_dict['gt_video_audio'], format="mp4"),
                    })
                if not self.config.disable_wandb:
                    wandb.log(wandb_log, step=self.step)
                
                logger.info(f"Step {self.step}: ODE Loss={wandb_log['loss']:.4f}, Video ODE Loss={wandb_log['video_loss']:.4f}, Audio ODE Loss={wandb_log['audio_loss']:.4f}, Grad Norm={wandb_log['grad_norm']:.4f}, LR={wandb_log['lr']:.6f}")

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