import gc
import logging

from utils.dataset import ODERegressionCSVDataset, cycle, OffsetDistributedSampler
from utils.distributed import EMA_FSDP, fsdp_wrap, fsdp_state_dict, launch_distributed_job
from utils.misc import (
    set_seed,
    merge_dict_list
)
import torch.distributed as dist
from omegaconf import OmegaConf
from model import CausVid, DMD, SiD
import torch
import wandb
import time
import os
import logging

logger = logging.getLogger(__name__)

class Trainer:
    def __init__(self, config):
        self.config = config
        self.step = 0

        # Step 1: Initialize the distributed training environment (rank, seed, dtype, logging etc.)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        launch_distributed_job()
        global_rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.is_main_process = global_rank == 0
        logger.info("Setting up the distributed environment...") if self.is_main_process else None
        self.dtype = torch.bfloat16 if config.mixed_precision else torch.float32
        self.device = torch.cuda.current_device()
        self.causal = config.causal
        self.disable_wandb = config.disable_wandb
        logger.info(f"Using wandb: {not self.disable_wandb}") if self.is_main_process else None

        # use a random seed for the training
        if config.seed == 0:
            random_seed = torch.randint(0, 10000000, (1,), device=self.device)
            dist.broadcast(random_seed, src=0)
            config.seed = random_seed.item()

        set_seed(config.seed + global_rank)

        if self.is_main_process and not self.disable_wandb:
            wandb.login(host=config.wandb_host, key=config.wandb_key)
            wandb.init(
                config=OmegaConf.to_container(config, resolve=True),
                name=config.config_name,
                mode="online",
                # entity=config.wandb_entity,
                project=config.wandb_project,
                dir=config.wandb_save_dir
            )

        self.output_path = config.logdir
        self.debug_distributed_training()
        logger.info(f"Finished setting up the distributed environment, world size: {self.world_size}") if self.is_main_process else None

        # Step 2: Initialize the model and optimizer
        logger.info(f"Initializing the {config.distribution_loss} distillation model...") if self.is_main_process else None
        if config.distribution_loss == "causvid":
            self.model = CausVid(config, device=self.device)
        elif config.distribution_loss == "dmd":
            self.model = DMD(config, device=self.device)
        elif config.distribution_loss == "sid":
            self.model = SiD(config, device=self.device)
        else:
            raise ValueError("Invalid distribution matching loss")
        logger.info(f"Finished initializing the distillation model.") if self.is_main_process else None

        # Resume Training from Latest Checkpoint
        pretrained_ckpt_path, self.step = self.load(self.output_path)
        if pretrained_ckpt_path is not None:
            logger.info(f"Loading checkpoint from {pretrained_ckpt_path} at step {self.step}") if self.is_main_process else None
            state_dict = torch.load(pretrained_ckpt_path, map_location="cpu")
            self.model.generator.load_state_dict(state_dict["generator"], strict=True)
            self.model.fake_score.load_state_dict(state_dict["critic"], strict=True)

        # Save pretrained model state_dicts to CPU
        self.fake_score_state_dict_cpu = self.model.fake_score.state_dict()

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
        self.model.vae = self.model.vae.to(
            device=self.device, dtype=torch.bfloat16 if config.mixed_precision else torch.float32)
        logger.info("Finished wrapping model components with FSDP.") if self.is_main_process else None
        logger.info(f"GPU memory after FSDP wrapping: {torch.cuda.memory_allocated(self.device)/1e9:.2f} GB") if self.is_main_process else None

        logger.info("Setting up optimizers...") if self.is_main_process else None
        self.generator_optimizer = torch.optim.AdamW(
            [param for param in self.model.generator.parameters()
             if param.requires_grad],
            lr=config.lr,
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay
        )

        self.critic_optimizer = torch.optim.AdamW(
            [param for param in self.model.fake_score.parameters()
             if param.requires_grad],
            lr=config.lr_critic if hasattr(config, "lr_critic") else config.lr,
            betas=(config.beta1_critic, config.beta2_critic),
            weight_decay=config.weight_decay
        )
        logger.info("Finished setting up optimizers.") if self.is_main_process else None

        # Step 3: Initialize the dataloader
        logger.info("Setting up dataset and dataloader...") if self.is_main_process else None
        dataset = ODERegressionCSVDataset(
            config.data_path, 
            max_pair=int(1e8), 
            num_frames=config.num_frames,
            h=config.h,
            w=config.w,
        )
            
        sampler = OffsetDistributedSampler(
            dataset,
            initial_step=self.step,
            gpu_num=self.world_size,
            shuffle=False,
            drop_last=True,
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config.batch_size,
            sampler=sampler,
            num_workers=8)

        if dist.get_rank() == 0:
            print("DATASET SIZE %d" % len(dataset))
        self.dataloader = cycle(dataloader)
        logger.info(f"Finished setting up dataset and dataloader, dataset size: {len(dataset)}.") if self.is_main_process else None

        ##############################################################################################################
        # 6. Set up EMA parameter containers
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
                    logger.info("Loading generator_ema from checkpoint") if dist.get_rank() == 0 else None
                    self.generator_ema.load_state_dict(checkpoint_state_dict["generator_ema"])
                else:
                    logger.info("No generator_ema found in checkpoint, starting fresh EMA")
        logger.info("Finished setting up EMA parameters.") if self.is_main_process else None
        ##############################################################################################################
        # 7. (If resuming) Load the model and optimizer, lr_scheduler, ema's statedicts
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

        ##############################################################################################################

        # Let's delete EMA params for early steps to save some computes at training and inference
        # if self.step < config.ema_start_step:
        #     self.generator_ema = None

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
        ckpt_folders.sort(key=extract_step)
        latest_ckpt_folder = ckpt_folders[-1]

        # 2. read model.pt and step
        model_path = os.path.join(out_path, latest_ckpt_folder, "model.pt")
        step = extract_step(latest_ckpt_folder)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"{model_path} not found")
        return model_path, step
    
    def save(self):
        print("Start gathering distributed model states...")
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
            print("Model saved to", os.path.join(self.output_path,
                  f"checkpoint_model_{self.step:06d}", "model.pt"))

    def fwdbwd_one_step(self, batch, train_generator):
        self.model.eval()  # prevent any randomness (e.g. dropout)

        if self.step % 20 == 0:
            torch.cuda.empty_cache()

        # Step 1: Get the next batch of text prompts
        text_prompts = batch["prompts"]
        video_tensor = batch["video"].to(device=self.device, dtype=self.dtype)
        first_frame = video_tensor[:, :, :1, :, :]
        wan22_image_latent = self.model.vae.encode_to_latent(first_frame) # torch.Size([1, 1, 48, 44, 80])
        clean_latent = None
        image_latent = None

        batch_size = len(text_prompts)
        # image_or_video_shape = list(self.config.image_or_video_shape)
        # image_or_video_shape[0] = batch_size
        _, _, _, H, W = video_tensor.shape
        image_or_video_shape = [batch_size, 31, 48, H//16, W//16]

        # Step 2: Extract the conditional infos
        with torch.no_grad():
            conditional_dict = self.model.text_encoder(
                text_prompts=text_prompts)

            if not getattr(self, "unconditional_dict", None):
                unconditional_dict = self.model.text_encoder(
                    text_prompts=[self.config.negative_prompt] * batch_size)
                unconditional_dict = {k: v.detach()
                                      for k, v in unconditional_dict.items()}
                self.unconditional_dict = unconditional_dict  # cache the unconditional_dict
            else:
                unconditional_dict = self.unconditional_dict

            if self.config.i2v:
                img = batch["img"].to(self.device).squeeze(0)
                clip_fea = self.model.image_encoder(img)
                y = self.model.vae.run_vae_encoder(img)
            else:
                clip_fea = None
                y = None

        # Step 3: Store gradients for the generator (if training the generator)
        if train_generator:
            generator_loss, generator_log_dict = self.model.generator_loss(
                image_or_video_shape=image_or_video_shape,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict,
                clean_latent=clean_latent,
                initial_latent=image_latent if self.config.i2v else None,
                clip_fea=clip_fea,
                y=y,
                wan22_image_latent=wan22_image_latent,
            )

            torch.cuda.empty_cache()

            generator_loss.backward()
            generator_grad_norm = self.model.generator.clip_grad_norm_(
                self.max_grad_norm_generator)

            generator_log_dict.update({"generator_loss": generator_loss,
                                       "generator_grad_norm": generator_grad_norm})

            return generator_log_dict
        else:
            generator_log_dict = {}

        # Step 4: Store gradients for the critic (if training the critic)
        critic_loss, critic_log_dict = self.model.critic_loss(
            image_or_video_shape=image_or_video_shape,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            clean_latent=clean_latent,
            initial_latent=image_latent if self.config.i2v else None,
            clip_fea=clip_fea,
            y=y,
            wan22_image_latent=wan22_image_latent,
        )

        critic_loss.backward()
        critic_grad_norm = self.model.fake_score.clip_grad_norm_(
            self.max_grad_norm_critic)

        critic_log_dict.update({"critic_loss": critic_loss,
                                "critic_grad_norm": critic_grad_norm})

        return critic_log_dict

    def generate_video(self, pipeline, prompts, image=None):
        batch_size = len(prompts)
        if image is not None:
            image = image.squeeze(0).unsqueeze(0).unsqueeze(2).to(device="cuda", dtype=torch.bfloat16)

            # Encode the input image as the first latent
            initial_latent = pipeline.vae.encode_to_latent(image).to(device="cuda", dtype=torch.bfloat16)
            initial_latent = initial_latent.repeat(batch_size, 1, 1, 1, 1)
            sampled_noise = torch.randn(
                [batch_size, self.model.num_training_frames - 1, 16, 60, 104],
                device="cuda",
                dtype=self.dtype
            )
        else:
            initial_latent = None
            sampled_noise = torch.randn(
                [batch_size, self.model.num_training_frames, 16, 60, 104],
                device="cuda",
                dtype=self.dtype
            )

        video, _ = pipeline.inference(
            noise=sampled_noise,
            text_prompts=prompts,
            return_latents=True,
            initial_latent=initial_latent
        )
        current_video = video.permute(0, 1, 3, 4, 2).cpu().numpy() * 255.0
        return current_video

    def train(self):
        start_step = self.step

        while True:
            if self.is_main_process:
                print(f"training step {self.step} ...")
            TRAIN_GENERATOR = self.step % self.config.dfake_gen_update_ratio == 0

            # Train the generator
            if TRAIN_GENERATOR:
                self.generator_optimizer.zero_grad(set_to_none=True)
                extras_list = []
                batch = next(self.dataloader)
                extra = self.fwdbwd_one_step(batch, True)
                extras_list.append(extra)
                generator_log_dict = merge_dict_list(extras_list)
                if not self.config.debug:
                    self.generator_optimizer.step()
                    if self.generator_ema is not None:
                        self.generator_ema.update(self.model.generator)

            # Train the critic
            self.critic_optimizer.zero_grad(set_to_none=True)
            extras_list = []
            batch = next(self.dataloader)
            extra = self.fwdbwd_one_step(batch, False)
            extras_list.append(extra)
            critic_log_dict = merge_dict_list(extras_list)
            if not self.config.debug:
                self.critic_optimizer.step()

            # Increment the step since we finished gradient update
            self.step += 1

            # Create EMA params (if not already created)
            if (self.step >= self.ema_start_step) and \
                    (self.generator_ema is None) and (self.ema_weight > 0):
                self.generator_ema = EMA_FSDP(self.model.generator, decay=self.ema_weight)

            # Save the model
            if (not self.config.no_save) and (self.step - start_step) > 0 and self.step % self.config.log_iters == 0:
                torch.cuda.empty_cache()
                self.save()
                torch.cuda.empty_cache()

            # Logging
            if self.is_main_process:
                wandb_loss_dict = {}
                if TRAIN_GENERATOR:
                    wandb_loss_dict.update(
                        {
                            "generator_loss": generator_log_dict["generator_loss"].mean().item(),
                            "generator_grad_norm": generator_log_dict["generator_grad_norm"].mean().item(),
                            "dmdtrain_gradient_norm": generator_log_dict["dmdtrain_gradient_norm"].mean().item()
                        }
                    )

                wandb_loss_dict.update(
                    {
                        "critic_loss": critic_log_dict["critic_loss"].mean().item(),
                        "critic_grad_norm": critic_log_dict["critic_grad_norm"].mean().item()
                    }
                )

                if not self.disable_wandb:
                    wandb.log(wandb_loss_dict, step=self.step)

            if self.step % self.config.gc_interval == 0:
                if dist.get_rank() == 0:
                    print("DistGarbageCollector: Running GC.")
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
