# FILE: trainer/ovi_distillation.py (A new file, adapted from wan22_distillation.py)

import gc
import logging
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import wandb
import time
import os
from omegaconf import OmegaConf
import csv
from PIL import Image
import torchvision.transforms as T
import numpy as np
import tempfile
from tqdm import tqdm

# --- OVI IMPORTS ---
from model.ovi_dmd import OviDMD
from utils.dataset import OviCSVDataset, OviCSVImageVideoDataset, cycle, OffsetDistributedSampler, masks_like, process_visual
from utils.distributed import EMA_FSDP, fsdp_wrap, fsdp_state_dict, launch_distributed_job, fsdp_optim_state_dict
from utils.misc import set_seed, merge_dict_list
from ovi.modules.ovi import FusionAttentionBlock
from ovi.modules.causal_ovi import CausalFusionAttentionBlock
from ovi.utils.io_utils import save_video

logger = logging.getLogger(__name__)

# Some helper functions
def extract_step(folder_name):
    import re
    match = re.search(r"checkpoint_model_(\d+)", folder_name)
    return int(match.group(1)) if match else -1


class Trainer: # MODIFIED: Renamed class
    def __init__(self, config):
        self.config = config
        self.step = 0

        # --- Step 1: Distributed Environment Setup ---
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
        
        # === NEW: Gradient Accumulation Setup ===
        self.gradient_accumulation_steps = getattr(config, "gradient_accumulation_steps", 1)
        if self.gradient_accumulation_steps > 1 and self.is_main_process:
            logger.info(f"Using gradient accumulation with {self.gradient_accumulation_steps} steps.")
        
        # === NEW: Critic Warmup Setup ===
        self.critic_warmup = getattr(config, "critic_warmup", 0)
        if self.critic_warmup > 0 and self.is_main_process:
            logger.info(f"Critic warmup enabled for {self.critic_warmup} steps.")

        # # === NEW: Eval CSV Setup ===
        # self.eval_csv = getattr(config, "eval_csv", None)
        # self.eval_data = None
        # if self.eval_csv is not None and os.path.exists(self.eval_csv):
        #     self.eval_data = self._load_eval_csv(self.eval_csv)
        #     if self.is_main_process:
        #         logger.info(f"Loaded {len(self.eval_data)} samples from eval_csv: {self.eval_csv}")       

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

        # --- Step 3: (If resuming) Load the model and optimizer, lr_scheduler, ema's statedicts ---
        if getattr(config, "generator_ckpt", False):
            logger.info(f"Loading pretrained generator from {config.generator_ckpt}") if dist.get_rank() == 0 else None
            generator_state_dict = torch.load(config.generator_ckpt, map_location="cpu")
            if "generator" in generator_state_dict:
                generator_state_dict = generator_state_dict["generator"]
            elif "model" in generator_state_dict:
                generator_state_dict = generator_state_dict["model"]
            self.model.generator.load_state_dict(
                generator_state_dict, strict=True
            )
            del generator_state_dict
            gc.collect()
            torch.cuda.empty_cache()

        # --- Step 4: Load checkpoint if resuming ---
        pretrained_ckpt_path, self.step = self.load(self.output_path)
        if pretrained_ckpt_path is not None:
            logger.info(f"Loading checkpoint from {pretrained_ckpt_path} at step {self.step}") if self.is_main_process else None
            state_dict = torch.load(pretrained_ckpt_path, map_location="cpu")
            logger.info(f"Loaded: {state_dict.keys()=} on {dist.get_rank()}")
            self.model.generator.load_state_dict(state_dict["generator"], strict=True)
            self.model.fake_score.load_state_dict(state_dict["critic"], strict=True)
        else:
            logger.info("No checkpoint found, training from scratch.") if self.is_main_process else None
            state_dict = None

        # --- Step 5: FSDP Wrapping (assumed done in OviDMD.__init__ or here) ---
        logger.info("Wrapping model components with FSDP...") if self.is_main_process else None
        logger.info(f"Before FSDP, model architecture: {self.model.generator}") if self.is_main_process else None
        orig_student = sum(p.numel() for p in self.model.generator.parameters() if p.requires_grad)
        logger.info(f"Before FSDP, student parameters: {orig_student/1e9:.2f}B") if self.is_main_process else None
        transformer_module = (
            (CausalFusionAttentionBlock, )
            if config.generator_type == "causal"
            else (FusionAttentionBlock, )
        )
        self.model.generator = fsdp_wrap(
            self.model.generator,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.generator_fsdp_wrap_strategy,
            transformer_module=transformer_module
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
            wrap_strategy=config.real_score_fsdp_wrap_strategy,
            transformer_module=(FusionAttentionBlock, )
        )
        fsdp_teacher = sum(p.numel() for p in self.model.real_score.parameters() if p.requires_grad)
        logger.info(f"After FSDP, teacher parameters: {fsdp_teacher/1e9:.2f}B") if self.is_main_process else None

        orig_critic = sum(p.numel() for p in self.model.fake_score.parameters() if p.requires_grad)
        logger.info(f"Before FSDP, critic parameters: {orig_critic/1e9:.2f}B") if self.is_main_process else None
        self.model.fake_score = fsdp_wrap(
            self.model.fake_score,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.fake_score_fsdp_wrap_strategy,
            transformer_module=(FusionAttentionBlock, )
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

        # --- Step 6: Setup optimizers ---
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

        # load optimizer states if resuming
        if pretrained_ckpt_path is not None:
            if state_dict.get("generator_optimizer", None) is not None:
                logger.info("Loading generator optimizer state from checkpoint") if self.is_main_process else None
                self.generator_optimizer.load_state_dict(
                    FSDP.optim_state_dict_to_load(
                        self.model.generator,
                        self.generator_optimizer,
                        state_dict["generator_optimizer"]
                    )
                )
            if state_dict.get("critic_optimizer", None) is not None:
                logger.info("Loading critic optimizer state from checkpoint") if self.is_main_process else None
                self.critic_optimizer.load_state_dict(
                    FSDP.optim_state_dict_to_load(
                        self.model.fake_score,
                        self.critic_optimizer,
                        state_dict["critic_optimizer"]
                    )
                )

        # --- Step 7: Setup dataloader ---
        logger.info(f"Setting up dataset and dataloader...") if self.is_main_process else None
        dataset = OviCSVImageVideoDataset(
            config.data_path,
            num_frames=config.num_frames,
            h=config.h,
            w=config.w,
            # Audio params are just for consistency, not used for loading
            audio_sample_rate=config.audio_sample_rate,
            audio_duration_secs=config.audio_duration_secs,
        )
        sampler = OffsetDistributedSampler(dataset, initial_step=self.step, gpu_num=self.world_size, shuffle=False, drop_last=True)
        dataloader = torch.utils.data.DataLoader(dataset, 
                                                 batch_size=config.batch_size, sampler=sampler, 
                                                 num_workers=4, 
                                                 prefetch_factor=2, 
                                                 pin_memory=True,
                                                 persistent_workers=True,
                                                 drop_last=True)
        self.dataloader = cycle(dataloader)
        logger.info(f"Finished setting up dataset and dataloader, dataset class name: {dataset.__class__.__name__}, size: {len(dataset)}, batch size: {config.batch_size}") if self.is_main_process else None

        # --- Step 8: Setup EMA ---
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
                if state_dict.get("generator_ema", None) is not None:
                    logger.info("Loading generator_ema from checkpoint") if self.is_main_process else None
                    self.generator_ema.load_state_dict(state_dict["generator_ema"])
                else:
                    logger.info("No generator_ema found in checkpoint, starting fresh EMA")
        logger.info("Finished setting up EMA parameters.") if self.is_main_process else None

        self.max_grad_norm_generator = getattr(config, "max_grad_norm_generator", 10.0)
        self.max_grad_norm_critic = getattr(config, "max_grad_norm_critic", 10.0)
        self.previous_time = None
        del state_dict
        gc.collect()
        torch.cuda.empty_cache()

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
        
        # 2. Read model.pt and step
        model_path = os.path.join(out_path, latest_ckpt_folder, "checkpoint.pt")
        step = extract_step(latest_ckpt_folder)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"{model_path} not found")
        return model_path, step
    
    def save(self):
        os.makedirs(self.output_path, exist_ok=True)
        checkpoint_folders = os.listdir(self.output_path)
        checkpoint_infos = [
            (checkpoint_folder, extract_step(checkpoint_folder))
            for checkpoint_folder in checkpoint_folders if checkpoint_folder.startswith("checkpoint_model_")
        ]
        # Sort by step number
        checkpoint_infos = sorted(checkpoint_infos, key=lambda x: x[1], reverse=True)

        # filter out to keep checkpoint steps
        checkpoint_infos = [info for info in checkpoint_infos if info[1] not in self.config.get("checkpoints_to_keep", [])]

        # NOTE: maybe ensure safer deletion since we are all using root
        # if (
        #     (len(checkpoint_infos) >= self.config.get("num_keep_checkpoints", float("inf"))) # since we are also going to save one, we remove one when reaching the number
        #     and (dist.get_rank() == 0)
        # ): # a very large number, lol
        #     # remove the oldest info
        #     os.removedirs(os.path.join(self.output_path, checkpoint_infos[-1][0]))

        generator_state_dict = fsdp_state_dict(self.model.generator)
        critic_state_dict = fsdp_state_dict(self.model.fake_score)
        state_dict = {}
        # Prepare model state dictionary
        state_dict.update({
            "generator": generator_state_dict,
            "critic": critic_state_dict,
        })
        optimizer_state_dict = {
            "generator_optimizer": fsdp_optim_state_dict(
                self.model.generator, 
                self.generator_optimizer
            ),
            "critic_optimizer": fsdp_optim_state_dict(
                self.model.fake_score, 
                self.critic_optimizer
            ),
            "step": self.step,
        }
        state_dict.update(optimizer_state_dict)
        if (self.ema_weight > 0.0) and (self.ema_start_step < self.step):
            state_dict.update({
                "generator_ema": self.generator_ema.state_dict(),
            })
        if dist.get_rank() == 0:
            # Create checkpoint directory
            checkpoint_dir = os.path.join(self.output_path, f"checkpoint_model_{self.step:06d}")
            os.makedirs(checkpoint_dir, exist_ok=True)
            logger.info("Finished gathering distributed model states.")
            # Save model states
            model_path = os.path.join(checkpoint_dir, "checkpoint.pt")
            torch.save(state_dict, model_path)
            logger.info(f"Model saved to {model_path}")
        
    def fwdbwd_one_step(self, batch, train_generator, log_video=False):
        # --- HEAVILY MODIFIED FOR OVI ---
        self.model.eval()

        if self.step % 20 == 0:
            torch.cuda.empty_cache()

        # Step 1: Get batch of (text, video, audio)
        text_prompts = batch["prompts"]
        first_frame = batch["first_frame"].to(device=self.device, dtype=self.dtype, non_blocking=True)    # shape: [B, 3, 1, H, W]
        # audio_tensor = batch["audio"].to(device=self.device, dtype=self.dtype)      # shape: [B, L]

        # Step 2: Encode inputs to latents
        with torch.no_grad():
            # Video: get clean latent and special first-frame latent
            # first_frame = video_tensor[:, :, :1, :, :]
            wan22_image_latent = self.model.vae.encode_video(first_frame) # shape: [B=1, F=1, C=48, H//16, W//16]
            
            # Text encoding
            conditional_dict = self.model.text_encoder(text_prompts=text_prompts)
            conditional_dict = {
                "video_prompt_embeds": conditional_dict["prompt_embeds"].detach(),
                "audio_prompt_embeds": conditional_dict["prompt_embeds"].detach(),
            }
            if not getattr(self, "unconditional_dict", None):
                vid_neg_prompt_embed = self.model.text_encoder(text_prompts=[self.config.video_negative_prompt] * len(text_prompts))
                aud_neg_prompt_embed = self.model.text_encoder(text_prompts=[self.config.audio_negative_prompt] * len(text_prompts))
                self.unconditional_dict = {
                    "video_prompt_embeds": vid_neg_prompt_embed["prompt_embeds"].detach(),
                    "audio_prompt_embeds": aud_neg_prompt_embed["prompt_embeds"].detach(),
                }
            unconditional_dict = self.unconditional_dict
        
        # Define latent shapes from config
        batch_size = len(text_prompts)
        video_latent_shape = self.config.video_latent_shape     # [31, 48, 44, 80] or [32, 48, 44, 80]
        audio_latent_shape = self.config.audio_latent_shape     # [157, 20] or [160, 20]
        # _, _, _, H, W = first_frame.shape # NOTE: currently using hard coded latent shapes from config
        latent_shapes = (
            [batch_size, *video_latent_shape], 
            [batch_size, *audio_latent_shape]
        )

        # Step 3: Call generator or critic loss
        if train_generator:
            generator_loss, generator_log_dict = self.model.generator_loss(
                latent_shapes=latent_shapes,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict,
                wan22_image_latent=wan22_image_latent,
                log_video=log_video,
            )
            torch.cuda.empty_cache()
            
            # === MODIFIED: Scale loss for gradient accumulation ===
            if self.gradient_accumulation_steps > 1:
                generator_loss = generator_loss / self.gradient_accumulation_steps
            
            generator_loss.backward()
            generator_grad_norm = self.model.generator.clip_grad_norm_(self.max_grad_norm_generator)
            generator_log_dict.update({
                "generator_loss": generator_loss * self.gradient_accumulation_steps,  # Log unscaled loss
                "generator_grad_norm": generator_grad_norm}
            )
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
        
        # === MODIFIED: Scale loss for gradient accumulation ===
        if self.gradient_accumulation_steps > 1:
            critic_loss = critic_loss / self.gradient_accumulation_steps
            
        critic_loss.backward()
        critic_grad_norm = self.model.fake_score.clip_grad_norm_(self.max_grad_norm_critic)
        critic_log_dict.update({
            "critic_loss": critic_loss * self.gradient_accumulation_steps,  # Log unscaled loss
            "critic_grad_norm": critic_grad_norm}
        )
        return critic_log_dict
    
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
        
        for idx, sample in tqdm(enumerate(eval_data), total=len(eval_data), leave=False, disable=(dist.get_rank()!=0), desc=f"Running Full Inference Eval"):
            prompt = sample["prompt"]
            image_path = sample["image_path"]
            # set_seed(sample["seed"])
            
            # Check if image exists (all ranks need to agree on whether to skip)
            image_exists = image_path and os.path.exists(image_path)
            if image_exists:
                # === All ranks execute inference ===
                # logger.info(f"Running eval on sample {idx + 1}/{len(eval_data)}: {prompt[:50]}...") if self.is_main_process else None
                # 1. Encode first frame
                processed_frame = process_visual(image_path, w=1280, h=704)
                processed_frame = processed_frame.to(self.device, dtype=self.dtype)
                wan22_image_latent = self.model.vae.encode_video(processed_frame.unsqueeze(2))
                
                # 2. Encode text
                conditional_dict = self.model.text_encoder(text_prompts=[prompt])
                conditional_dict = {
                    "video_prompt_embeds": conditional_dict["prompt_embeds"].detach(),
                    "audio_prompt_embeds": conditional_dict["prompt_embeds"].detach(),
                }
                
                # 3. Generate noise and run inference
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
                
                # 4. Decode video and audio
                video = self.model.vae.decode_video(video_latent)
                audio = self.model.vae.decode_audio(audio_latent.transpose(1, 2))
            
                # === Only rank 0 saves results ===
                if self.is_main_process:
                    video = ((video + 1) / 2 * 255).clip(0, 255)
                    video_np = video.squeeze(0).permute(1, 0, 2, 3).cpu().float().numpy().astype(np.uint8)
                    audio_np = audio.squeeze(0).cpu().float().numpy().flatten()
                    
                    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
                        save_video(f.name, video_np, audio_np)
                        eval_videos.append({"path": f.name, "prompt": prompt[:50], "idx": idx})
                
                # All ranks clean up
                torch.cuda.empty_cache()
            
            else:
                # Image not found - all ranks skip but still sync
                logger.warning(f"Image not found: {image_path}, skipping sample {idx}") if self.is_main_process else None
            
            # Sync all ranks after each sample
            if dist.is_initialized():
                dist.barrier()
        
        # === Only rank 0 prepares wandb log ===
        wandb_log = {}
        if self.is_main_process:
            for item in eval_videos:
                key = f"Evaluation/Sample_{item['idx']}"
                wandb_log[key] = wandb.Video(item["path"], format="mp4")
        
        return wandb_log

    def train(self):
        # --- MODIFIED FOR OVI LOGGING ---
        start_step = self.step
        
        # Critic Warmup Phase (Optional)
        if self.critic_warmup > 0 and self.step == 0:
            logger.info(f"Starting critic warmup for {self.critic_warmup} steps...") if self.is_main_process else None
            
            for warmup_step in range(self.critic_warmup):
                self.critic_optimizer.zero_grad(set_to_none=True)
                
                # Accumulate gradients for critic warmup (support gradient accumulation)
                for accum_step in tqdm(range(self.gradient_accumulation_steps), total=len(range(self.gradient_accumulation_steps)), desc="Gradient accumulating for critic", leave=False, disable=(dist.get_rank()!=0)):
                    batch = next(self.dataloader)
                    critic_log_dict = self.fwdbwd_one_step(batch, train_generator=False)
                
                self.critic_optimizer.step()
                
                logger.info(f"Critic warmup step {warmup_step + 1}/{self.critic_warmup}, Critic Loss: {critic_log_dict['critic_loss'].mean().item():.4f}, Critic Video Loss: {critic_log_dict['critic_loss_video'].mean().item():.4f}, Critic Audio Loss: {critic_log_dict['critic_loss_audio'].mean().item():.4f}, GradNorm: {critic_log_dict['critic_grad_norm'].mean().item():.4f}") if self.is_main_process else None

                if warmup_step % self.config.gc_interval == 0:
                    gc.collect()
                    torch.cuda.empty_cache()
            
            logger.info("Critic warmup completed.") if self.is_main_process else None

        while True:
            TRAIN_GENERATOR = self.step % self.config.dfake_gen_update_ratio == 0
            LOG_VIDEO = (
                TRAIN_GENERATOR
                and (self.step % self.config.video_log_iters == 0)
            )
            # Train Generator
            if TRAIN_GENERATOR:
                self.generator_optimizer.zero_grad(set_to_none=True)
                for accum_step in tqdm(range(self.gradient_accumulation_steps), total=len(range(self.gradient_accumulation_steps)), desc="Gradient accumulating for generator", leave=False, disable=(dist.get_rank()!=0)):
                    batch = next(self.dataloader)
                    log_video_this_step = LOG_VIDEO and (accum_step == self.gradient_accumulation_steps - 1)
                    generator_log_dict = self.fwdbwd_one_step(
                        batch,
                        train_generator=True,
                        log_video=log_video_this_step
                    )
                
                if not self.config.debug:
                    self.generator_optimizer.step()
                    if self.generator_ema is not None: 
                        self.generator_ema.update(self.model.generator)
            
            # Train Critic
            self.critic_optimizer.zero_grad(set_to_none=True)
            for accum_step in tqdm(range(self.gradient_accumulation_steps), total=len(range(self.gradient_accumulation_steps)), desc="Gradient accumulating for critic", leave=False, disable=(dist.get_rank()!=0)):
                batch = next(self.dataloader)
                critic_log_dict = self.fwdbwd_one_step(batch, train_generator=False)
            if not self.config.debug:
                self.critic_optimizer.step()
                
            self.step += 1
            
            # EMA creation
            if (self.step >= self.ema_start_step) and (self.generator_ema is None) and (self.ema_weight > 0):
                self.generator_ema = EMA_FSDP(self.model.generator, decay=self.ema_weight)

            # Save model
            if (
                (not self.config.no_save)
                and (self.step - start_step) > 0
                and (self.step % self.config.log_iters == 0)
            ):
                torch.cuda.empty_cache()
                self.save()
                torch.cuda.empty_cache()

            # Logging
            eval_log = {}
            if TRAIN_GENERATOR and LOG_VIDEO:
                logger.info(f"Step {self.step}: Running on-the-fly evaluation...") if self.is_main_process else None
                try:
                    eval_log = self.run_eval()
                except Exception as e:
                    logger.warning(f"Error during eval: {e}") if self.is_main_process else None

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
                    if LOG_VIDEO:
                        wandb_log.update({
                            "Visualization/Generated_Video_Audio": wandb.Video(generator_log_dict['generated_video_audio'], format="mp4"),
                        })
                        wandb_log.update(eval_log)
                    logger.info(f"Step {self.step}: Generator Loss: {generator_log_dict['generator_loss'].mean().item():.4f}, Video DMD Loss: {generator_log_dict['dmd_loss_video'].mean().item():.4f}, Audio DMD Loss: {generator_log_dict['dmd_loss_audio'].mean().item():.4f}, GradNorm: {generator_log_dict['generator_grad_norm'].mean().item():.4f}, DMD Video GradNorm: {generator_log_dict['dmdtrain_gradient_norm_video'].mean().item():.4f}, DMD Audio GradNorm: {generator_log_dict['dmdtrain_gradient_norm_audio'].mean().item():.4f}")
                
                wandb_log.update({
                    "Loss/Critic": critic_log_dict["critic_loss"].mean().item(),
                    "Loss/Critic_Video": critic_log_dict["critic_loss_video"].mean().item(),
                    "Loss/Critic_Audio": critic_log_dict["critic_loss_audio"].mean().item(),
                    "GradNorm/Critic": critic_log_dict["critic_grad_norm"].mean().item(),
                })
                logger.info(f"Step {self.step}: Critic Loss: {critic_log_dict['critic_loss'].mean().item():.4f}, Critic Video Loss: {critic_log_dict['critic_loss_video'].mean().item():.4f}, Critic Audio Loss: {critic_log_dict['critic_loss_audio'].mean().item():.4f}, GradNorm: {critic_log_dict['critic_grad_norm'].mean().item():.4f}")
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