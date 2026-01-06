import gc
import logging
from utils.dataset import ODERegressionLMDBDataset, cycle
from model import ODERegression
from collections import defaultdict
from utils.misc import (
    set_seed
)
import torch.distributed as dist
from omegaconf import OmegaConf
import torch
import wandb
import time
import os

from utils.distributed import barrier, fsdp_wrap, fsdp_state_dict, launch_distributed_job
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
                entity=config.wandb_entity,
                project=config.wandb_project,
                dir=config.wandb_save_dir
            )

        self.output_path = config.logdir

        # Step 2: Initialize the model and optimizer
        logger.info(f"Initializing the {config.distribution_loss} distillation model...") if self.is_main_process else None
        assert config.distribution_loss == "ode", "Only ODE loss is supported for ODE training"
        self.model = ODERegression(config, device=self.device)
        logger.info(f"Finished initializing the distillation model.") if self.is_main_process else None

        logger.info("Wrapping model components with FSDP...") if self.is_main_process else None
        # logger.info(f"Before FSDP, model architecture: {self.model.generator}") if self.is_main_process else None
        orig_student = sum(p.numel() for p in self.model.generator.parameters() if p.requires_grad)
        logger.info(f"Before FSDP, generator parameters: {orig_student/1e9:.2f}B") if self.is_main_process else None
        self.model.generator = fsdp_wrap(
            self.model.generator,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.generator_fsdp_wrap_strategy
        )
        # logger.info(f"After FSDP, generator architecture: {self.model.generator}") if self.is_main_process else None
        fsdp_student = sum(p.numel() for p in self.model.generator.parameters() if p.requires_grad)
        logger.info(f"After FSDP, generator parameters: {fsdp_student/1e9:.2f}B") if self.is_main_process else None

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

        if not config.no_visualize or config.load_raw_video:
            logger.info("Using bfloat16 for VAE") if self.is_main_process else None
            self.model.vae = self.model.vae.to(
                device=self.device, dtype=torch.bfloat16 if config.mixed_precision else torch.float32)

        logger.info("Setting up optimizers...") if self.is_main_process else None
        self.generator_optimizer = torch.optim.AdamW(
            [param for param in self.model.generator.parameters()
             if param.requires_grad],
            lr=config.lr,
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay
        )
        logger.info("Finished setting up optimizers.") if self.is_main_process else None

        # Step 3: Initialize the dataloader
        logger.info(f"Setting up dataset and dataloader...") if self.is_main_process else None
        dataset = ODERegressionLMDBDataset(config.data_path, max_pair=getattr(config, "max_pair", int(1e8)))
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True, drop_last=True)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, sampler=sampler, num_workers=8)
        total_batch_size = getattr(config, "total_batch_size", None)
        if total_batch_size is not None:
            assert total_batch_size == config.batch_size * self.world_size, "Gradient accumulation is not supported for ODE training"
        self.dataloader = cycle(dataloader)
        logger.info(f"Finished setting up dataset and dataloader, dataset class name: {dataset.__class__.__name__}, size: {len(dataset)}, batch size: {config.batch_size}") if self.is_main_process else None

        self.step = 0

        # ##############################################################################################################
        # # 7. (If resuming) Load the model and optimizer, lr_scheduler, ema's statedicts
        # if getattr(config, "generator_ckpt", False):
        #     logger.info(f"Loading pretrained generator from {config.generator_ckpt}") if self.is_main_process else None
        #     state_dict = torch.load(config.generator_ckpt, map_location="cpu")[
        #         'generator']
        #     self.model.generator.load_state_dict(
        #         state_dict, strict=True
        #     )

        # ##############################################################################################################

        self.max_grad_norm = 10.0
        self.previous_time = None

    def save(self):
        logger.info("Start gathering distributed model states...") if self.is_main_process else None
        generator_state_dict = fsdp_state_dict(
            self.model.generator)
        state_dict = {
            "generator": generator_state_dict
        }

        if self.is_main_process:
            os.makedirs(os.path.join(self.output_path,
                        f"checkpoint_model_{self.step:06d}"), exist_ok=True)
            torch.save(state_dict, os.path.join(self.output_path,
                       f"checkpoint_model_{self.step:06d}", "model.pt"))
            logger.info(f"Model saved to {os.path.join(self.output_path, f'checkpoint_model_{self.step:06d}', 'model.pt')}")

    def train_one_step(self):
        VISUALIZE = self.step % 100 == 0
        # 在 ODERegression 的初始化里，generator（Student）是被设为 requires_grad_(True) 的。虽然是 eval 模式，但在 PyTorch 中，eval 不影响反向传播，只影响 Dropout/BN 层。这在回归任务（Regression）中很常见，为了保证输出的确定性。
        self.model.eval()  # prevent any randomness (e.g. dropout)

        # Step 1: Get the next batch of text prompts
        batch = next(self.dataloader)
        text_prompts = batch["prompts"]
        # ode_latent: 这个张量的形状是 [B, 5, 21, 16, 60, 104]（假设 Batch=B），它包含 5 个时间点的数据（对应之前的 [1000, 938, 833, 625, 0]）。
        ode_latent = batch["ode_latent"].to(device=self.device, dtype=self.dtype)

        # Step 2: Extract the conditional infos
        with torch.no_grad():
            conditional_dict = self.model.text_encoder(text_prompts=text_prompts)

        # Step 3: Train the generator
        # 此时我们拿到了 generator_loss（标量）和 log_dict（包含未归约的 Loss 和每个样本对应的时间步）。
        generator_loss, log_dict = self.model.generator_loss(
            ode_latent=ode_latent,
            conditional_dict=conditional_dict
        )

        unnormalized_loss = log_dict["unnormalized_loss"]   # [Batch_Size]
        timestep = log_dict["timestep"]                     # [Batch_Size]

        if self.world_size > 1:
            # 假设你有 8 张卡 (Rank 0-7)。Rank 0 这一轮随机抽到了 t=1000 的数据。Rank 1 这一轮随机抽到了 t=250 的数据。为了看清楚模型在每个时间步表现如何，需要把所有卡上的 Loss 和 Timestep 汇总到一起统计。
            gathered_unnormalized_loss = torch.zeros(
                [self.world_size, *unnormalized_loss.shape],
                dtype=unnormalized_loss.dtype, device=self.device)
            gathered_timestep = torch.zeros(
                [self.world_size, *timestep.shape],
                dtype=timestep.dtype, device=self.device)

            dist.all_gather_into_tensor(
                gathered_unnormalized_loss, unnormalized_loss)
            dist.all_gather_into_tensor(gathered_timestep, timestep)
        else:
            gathered_unnormalized_loss = unnormalized_loss
            gathered_timestep = timestep

        loss_breakdown = defaultdict(list)
        stats = {}

        # 遍历收集到的所有样本
        for index, t in enumerate(timestep):
            # 1. 对时间步进行“归桶” (Bucketizing)
            # int(t.item()) // 250 * 250 的意思是：将时间步向下取整到 250 的倍数。
            # 1000 -> 1000，938  -> 938 // 250 = 3 -> 3 * 250 = 750 (注意：这里逻辑是为了把相近的时间归类，实际代码逻辑可能是为了把离散的时间步归类为字符串 Key)
            loss_breakdown[str(int(t.item()) // 250 * 250)].append(
                unnormalized_loss[index].item())

        # 2. 计算每个桶的平均 Loss
        # loss_at_time_1000: 极高噪声下的去噪能力（通常最难，Loss 最大）。
        # loss_at_time_750: 高噪声下的能力。
        # loss_at_time_250: 低噪声下的能力（微调细节）。
        for key_t in loss_breakdown.keys():
            stats["loss_at_time_" + key_t] = sum(loss_breakdown[key_t]) / \
                len(loss_breakdown[key_t])

        self.generator_optimizer.zero_grad()
        generator_loss.backward()
        generator_grad_norm = self.model.generator.clip_grad_norm_(
            self.max_grad_norm)
        self.generator_optimizer.step()

        # Step 4: Visualization
        if VISUALIZE and not self.config.no_visualize and not self.config.disable_wandb and self.is_main_process:
            # Visualize the input, output, and ground truth
            input = log_dict["input"]
            output = log_dict["output"]
            ground_truth = ode_latent[:, -1]

            input_video = self.model.vae.decode_to_pixel(input)
            output_video = self.model.vae.decode_to_pixel(output)
            ground_truth_video = self.model.vae.decode_to_pixel(ground_truth)
            input_video = 255.0 * (input_video.cpu().numpy() * 0.5 + 0.5)
            output_video = 255.0 * (output_video.cpu().numpy() * 0.5 + 0.5)
            ground_truth_video = 255.0 * (ground_truth_video.cpu().numpy() * 0.5 + 0.5)

            # Visualize the input, output, and ground truth
            wandb.log({
                "input": wandb.Video(input_video, caption="Input", fps=16, format="mp4"),
                "output": wandb.Video(output_video, caption="Output", fps=16, format="mp4"),
                "ground_truth": wandb.Video(ground_truth_video, caption="Ground Truth", fps=16, format="mp4"),
            }, step=self.step)

        if self.is_main_process:
            log_str = f"Step {self.step}: Loss: {generator_loss.item():.4f}, GradNorm: {generator_grad_norm.item():.4f}"
            for k, v in stats.items():
                log_str += f", {k}: {v:.4f}"
            logger.info(log_str)

        if self.is_main_process and not self.disable_wandb:
            wandb_loss_dict = {
                "generator_loss": generator_loss.item(),
                "generator_grad_norm": generator_grad_norm.item(),
                **stats
            }
            wandb.log(wandb_loss_dict, step=self.step)


        if self.step % self.config.gc_interval == 0 and self.step > 0:
            logger.info("DistGarbageCollector: Running GC.") if self.is_main_process else None
            gc.collect()

    def train(self):
        while True:
            self.train_one_step()
            if (not self.config.no_save) and self.step % self.config.log_iters == 0 and self.step > 0:
                self.save()
                torch.cuda.empty_cache()

            barrier()
            if self.is_main_process:
                current_time = time.time()
                if self.previous_time is None:
                    self.previous_time = current_time
                else:
                    if not self.disable_wandb:
                        wandb.log({"per iteration time": current_time - self.previous_time}, step=self.step)
                    self.previous_time = current_time

            self.step += 1
