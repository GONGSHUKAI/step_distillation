import torch.nn.functional as F
from typing import Tuple
import torch

from model.base import BaseModel
from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper
import logging
logger = logging.getLogger(__name__)

class ODERegression(BaseModel):
    def __init__(self, args, device):
        """
        Initialize the ODERegression module.
        This class is self-contained and compute generator losses
        in the forward pass given precomputed ode solution pairs.
        This class supports the ode regression loss for both causal and bidirectional models.
        See Sec 4.3 of CausVid https://arxiv.org/abs/2412.07772 for details
        """
        super().__init__(args, device)

        # Step 1: Initialize all models

        self.generator = WanDiffusionWrapper(**getattr(args, "model_kwargs", {}), is_causal=True)
        self.generator.model.requires_grad_(True)
        if getattr(args, "generator_ckpt", False):
            logger.info(f"Loading pretrained generator from {args.generator_ckpt}") if torch.distributed.get_rank() == 0 else None
            state_dict = torch.load(args.generator_ckpt, map_location="cpu")['generator']
            self.generator.load_state_dict(state_dict, strict=True)

        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)

        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block

        self.independent_first_frame = getattr(args, "independent_first_frame", False)
        if self.independent_first_frame:
            self.generator.model.independent_first_frame = True
        if args.gradient_checkpointing:
            self.generator.enable_gradient_checkpointing()

        # Step 2: Initialize all hyperparameters
        self.timestep_shift = getattr(args, "timestep_shift", 1.0)
        self.denoising_step_list = self.denoising_step_list.to(self.device)
        logger.info(f"Using denoising step list: {self.denoising_step_list} with time shift {self.timestep_shift}") if torch.distributed.get_rank() == 0 else None

    def _initialize_models(self, args, device):
        self.generator = WanDiffusionWrapper(**getattr(args, "model_kwargs", {}), is_causal=True)
        self.generator.model.requires_grad_(True)

        self.text_encoder = WanTextEncoder()
        self.text_encoder.requires_grad_(False)

        self.vae = WanVAEWrapper()
        self.vae.requires_grad_(False)

        self.scheduler = self.generator.get_scheduler()
        self.scheduler.timesteps = self.scheduler.timesteps.to(device)

    @torch.no_grad()
    def _prepare_generator_input(self, ode_latent: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Given a tensor containing the whole ODE sampling trajectories,
        randomly choose an intermediate timestep and return the latent as well as the corresponding timestep.
        Input:
            - ode_latent: a tensor containing the whole ODE sampling trajectories [batch_size, num_denoising_steps, num_frames, num_channels, height, width].
        Output:
            - noisy_input: a tensor containing the selected latent [batch_size, num_frames, num_channels, height, width].
            - timestep: a tensor containing the corresponding timestep [batch_size].
        """
        # ode_latent的形状是 [B, 5, 21, 16, 40, 104]，它包含了从高噪声到低噪声的整个生成轨迹（ODE Trajectory）。
        # num_denoising_steps在这个阶段应该是5, 对应[1000, 750, 500, 250, 0]，warp过后应该是[1000, 937.5, 833.3, 625, 96.2] (96.2其实就是干净的)
        # ode_latent[:, -1] 就是最终的干净视频 x0 （虽然标的时间步为96.2，但实则对应t=0），也就是 Ground Truth。
        batch_size, num_denoising_steps, num_frames, num_channels, height, width = ode_latent.shape

        # index 是一个形状为 [batch_size, num_frames] = [B, 21] 的 Tensor。
        # 它为每一帧随机选择一个从 0 到 3 的索引。
        # 例如，假设 batch_size=2, num_frames=21, num_frame_per_block=3，则index可以是
        # T2V index: [[0, 0, 0, 1, 1, 1, 2, 2, 2, ..., 3, 3, 3], [0, 0, 0, 1, 1, 1, 2, 2, 2, ..., 3, 3, 3]]
        # I2V index: [[0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 1 ,1 ,1, 1]]
        
        # Ovi I2AV video index[0, 1, 1, 1, 2, 2, 2]
        # Ovi      audio index[0, 0, 0, 0, 0, 1, 1, ..., 1 (15个1)]
        index = self._get_timestep(
            0,
            len(self.denoising_step_list),
            batch_size,
            num_frames,
            self.num_frame_per_block,
            uniform_timestep=False
        )   # index shape: [batch_size, num_frames] = [2, 21]

        # index[:, 0] = len(self.denoising_step_list) - 1：这行代码强制将第 0 帧（第一帧）的索引设为最大值（即指向 ODE 轨迹的最后一个元素，也就是 t=0 的干净 Latent）。
        # 意义: 在 I2V 任务中，第一帧是参考图，是已知的、清晰的。所以无论其他帧处于什么噪声水平，第一帧必须始终保持为“干净状态”。这模拟了 I2V 推理时第一帧作为 Condition 的情况。
        if self.args.i2v:
            index[:, 0] = len(self.denoising_step_list) - 1 # [1000, 750, 500, 250, 0]
            # This ensures that the first frame is always the clean latent (t=0).
            # len(self.denoising_step_list) - 1 = 3, so index[:, 0] = 3

        # 提取 Latent (torch.gather): 根据生成的 index，从 ode_latent 中取出对应的 Latent。
        # 结果 noisy_input 的形状是 [batch_size, num_frames, num_channels, height, width] = [B, 21, 16, 40, 104]。
        # 这实际上是一个混合了不同噪声水平的视频。有的帧可能在t=1000，有的在 t=250，而 I2V 的第一帧固定在 t=0。
        noisy_input = torch.gather(
            ode_latent, dim=1,
            index=index.reshape(batch_size, 1, num_frames, 1, 1, 1).expand(
                -1, -1, -1, num_channels, height, width).to(self.device)
        ).squeeze(1)

        # 从 self.denoising_step_list 中取出对应的时间步数值。
        # index: [[0, 0, 0, 1, 1, 1, 2, 2, 2, ..., 3, 3, 3], [0, 0, 0, 1, 1, 1, 2, 2, 2, ..., 3, 3, 3]]
        # 则 timestep 的形状是 [batch_size, num_frames] = [B, 21]。
        # 例如，假设 denoising_step_list = [1000, 937.5, 833.3, 625, 96.2]，则 timestep 可以是
        # timestep: [[1000, 1000, 1000, 937.5, 937.5, 937.5, 833.3, 833.3, 833.3, ..., 625, 625, 625], [1000, 1000, 1000, 937.5, 937.5, 937.5, 833.3, 833.3, 833.3, ..., 625, 625, 625]]
        timestep = self.denoising_step_list[index].to(self.device)

        # if self.extra_noise_step > 0:
        #     random_timestep = torch.randint(0, self.extra_noise_step, [
        #                                     batch_size, num_frames], device=self.device, dtype=torch.long)
        #     perturbed_noisy_input = self.scheduler.add_noise(
        #         noisy_input.flatten(0, 1),
        #         torch.randn_like(noisy_input.flatten(0, 1)),
        #         random_timestep.flatten(0, 1)
        #     ).detach().unflatten(0, (batch_size, num_frames)).type_as(noisy_input)

        #     noisy_input[timestep == 0] = perturbed_noisy_input[timestep == 0]

        return noisy_input, timestep

    def generator_loss(self, ode_latent: torch.Tensor, conditional_dict: dict) -> Tuple[torch.Tensor, dict]:
        """
        Generate image/videos from noisy latents and compute the ODE regression loss.
        Input:
            - ode_latent: a tensor containing the ODE latents [batch_size, num_denoising_steps, num_frames, num_channels, height, width].
            They are ordered from most noisy to clean latents.
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
        Output:
            - loss: a scalar tensor representing the generator loss.
            - log_dict: a dictionary containing additional information for loss timestep breakdown.
        """
        # Step 1: Run generator on noisy latents
        target_latent = ode_latent[:, -1]   # 取出第5个元素，即 t=0 的 Clean Latent，形状是 [B, 21, 16, 60, 104]（假设 Batch=B）。

        # noisy_input：混合了不同噪声水平的视频。有的帧可能在t=1000，有的在 t=250，而 I2V 的第一帧固定在 t=0，形状为 [B, 21, 16, 60, 104]。
        # timestep：对应的时间步，形状是 [B, 21]。
        noisy_input, timestep = self._prepare_generator_input(
            ode_latent=ode_latent)

        # generator（Causal的）输入 noisy_input 和 timestep，生成预测的图像或视频 pred_image_or_video。
        # pred_image_or_video 的形状是 [B, 21, 16, 60, 104]
        # I2V 的 Mask: 如果配置了 independent_first_frame=True（通常 I2V 会开启），那么 CausalWanModel 会使用 _prepare_blockwise_causal_attn_mask_i2v，确保所有帧都能看到第一帧（参考帧）。
        _, pred_image_or_video = self.generator(
            noisy_image_or_video=noisy_input,
            conditional_dict=conditional_dict,
            timestep=timestep
        )

        # Step 2: Compute the regression loss
        # mask = timestep != 0: 创建一个 Mask，只计算那些非零时间步（即真正加了噪、需要去噪）的帧的 Loss。
        # 对于 I2V，第一帧的 timestep 是 0，所以第一帧的 Loss 会被自动忽略。这是合理的，因为第一帧是作为 Condition 输入的，不需要预测它（或者说预测它是平凡的）。
        mask = timestep != 0

        # 计算 Student 的预测值 x0 与ground truth x0之间的均方误差。
        loss = F.mse_loss(pred_image_or_video[mask], target_latent[mask], reduction="mean")

        log_dict = {
            "unnormalized_loss": F.mse_loss(pred_image_or_video, target_latent, reduction='none').mean(dim=[1, 2, 3, 4]).detach(),
            "timestep": timestep.float().mean(dim=1).detach(),
            "input": noisy_input.detach(),
            "output": pred_image_or_video.detach(),
        }

        return loss, log_dict
