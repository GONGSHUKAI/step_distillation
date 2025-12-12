from utils.wan_wrapper import WanDiffusionWrapper
from utils.scheduler import SchedulerInterface
from typing import List, Optional
import torch
import torch.distributed as dist


class SelfForcingTrainingPipeline:
    def __init__(self,
                 model_name: str,
                 denoising_step_list: List[int],
                 scheduler: SchedulerInterface,
                 generator: WanDiffusionWrapper,
                 num_frame_per_block=3,
                 independent_first_frame: bool = False,
                 same_step_across_blocks: bool = False,
                 last_step_only: bool = False,
                 num_max_frames: int = 21,
                 context_noise: int = 0,
                 **kwargs):
        super().__init__()
        self.model_name = model_name
        self.scheduler = scheduler
        self.generator = generator
        self.denoising_step_list = denoising_step_list
        if self.denoising_step_list[-1] == 0:
            self.denoising_step_list = self.denoising_step_list[:-1]  # remove the zero timestep for inference

        # Wan specific hyperparameters
        self.num_transformer_blocks = 40 if "14B" in model_name else 30
        self.frame_seq_length = 1560
        self.num_frame_per_block = num_frame_per_block
        self.context_noise = context_noise
        self.i2v = False

        self.kv_cache1 = None
        self.kv_cache2 = None
        self.crossattn_cache = None

        self.independent_first_frame = independent_first_frame
        self.same_step_across_blocks = same_step_across_blocks
        self.last_step_only = last_step_only
        self.kv_cache_size = num_max_frames * self.frame_seq_length

    def generate_and_sync_list(self, num_blocks, num_denoising_steps, device):
        rank = dist.get_rank() if dist.is_initialized() else 0

        if rank == 0:
            # Generate random indices
            indices = torch.randint(
                low=0,
                high=num_denoising_steps,
                size=(num_blocks,),
                device=device
            )
            if self.last_step_only:
                indices = torch.ones_like(indices) * (num_denoising_steps - 1)
        else:
            indices = torch.empty(num_blocks, dtype=torch.long, device=device)

        dist.broadcast(indices, src=0)  # Broadcast the random indices to all ranks
        return indices.tolist()

    def inference_with_trajectory(
            self,
            noise: torch.Tensor,
            clip_fea: Optional[torch.Tensor] = None,
            y: Optional[torch.Tensor] = None,
            initial_latent: Optional[torch.Tensor] = None,
            return_sim_step: bool = False,
            **conditional_dict
    ) -> torch.Tensor:
        # noise: [1, 21, 16, 60, 104] 的纯高斯噪声，这只是一个纯高斯噪声的大容器。
        batch_size, num_frames, num_channels, height, width = noise.shape

        # self.independent_first_frame: False (T2V 模式下，通常所有 Block 一视同仁，都是 3 帧)。
        # self.num_frame_per_block: 3。
        # initial_latent: None (T2V 没有参考图 Latent，或者参考图通过 Text/CLIP embedding 传入，不占据时间轴)。
        # noise: 传入的纯高斯噪声。

        # 1. 如果模型配置不要求第一帧独立（即所有帧都按标准 Block Size 划分），
        # 2. 或者如果要求第一帧独立（比如 I2V），但是你已经把第一帧通过 initial_latent 传进来了。这意味着 noise 里不包含那个特殊的第 1 帧，noise 里全是后续的普通帧 (1+30)。
        if not self.independent_first_frame or (self.independent_first_frame and initial_latent is not None):
            # 那么输入的帧数必须能被 Block 大小整除。（例如21//3=7）
            assert num_frames % self.num_frame_per_block == 0
            # 这样一来要生成的 Block 数量就是 num_frames // self.num_frame_per_block。
            num_blocks = num_frames // self.num_frame_per_block
            
        # 这表示我们要生成视频，且模型要求第一帧单独生成（Block Size=1），后续帧按 Block Size=3 生成。
        else:
            # 对于Wan2.1-I2V, 21帧被划分为[1,4,4,4,4]
            # 对于Wan2.2-TI2V, 31帧被划分为[1,3,3,3,3,3,3,3,3,3,3]
            assert (num_frames - 1) % self.num_frame_per_block == 0
            num_blocks = (num_frames - 1) // self.num_frame_per_block

        # 检查有没有外部传入的初始 Latent（Reference Frame）。
        # T2V 训练中，initial_latent 是 None
        num_input_frames = initial_latent.shape[1] if initial_latent is not None else 0 
        # 输出长度 = 需要生成的帧数 (noise 长度) + 已经给定的帧数 (initial_latent 长度)。
        # 这里num_output_frames = 21 + 0 = 21 (T2V 模式下，初始 Latent 不占用时间轴)。
        num_output_frames = num_frames + num_input_frames  # add the initial latent frames

        # output 是一个全零张量，形状 [1, 21, 16, 60, 104]。这将用于存储生成的视频。
        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=noise.device,
            dtype=noise.dtype
        )

        # Step 1: Initialize KV cache to all zeros
        # 1. 初始化 KV Cache (用于存之前 Block 生成的 Key/Value)
        self._initialize_kv_cache(
            batch_size=batch_size, dtype=noise.dtype, device=noise.device
        )
        self._initialize_crossattn_cache(
            batch_size=batch_size, dtype=noise.dtype, device=noise.device
        )
        # if self.kv_cache1 is None:
        #     self._initialize_kv_cache(
        #         batch_size=batch_size,
        #         dtype=noise.dtype,
        #         device=noise.device,
        #     )
        #     self._initialize_crossattn_cache(
        #         batch_size=batch_size,
        #         dtype=noise.dtype,
        #         device=noise.device
        #     )
        # else:
        #     # reset cross attn cache
        #     for block_index in range(self.num_transformer_blocks):
        #         self.crossattn_cache[block_index]["is_init"] = False
        #     # reset kv cache
        #     for block_index in range(len(self.kv_cache1)):
        #         self.kv_cache1[block_index]["global_end_index"] = torch.tensor(
        #             [0], dtype=torch.long, device=noise.device)
        #         self.kv_cache1[block_index]["local_end_index"] = torch.tensor(
        #             [0], dtype=torch.long, device=noise.device)

        # Step 2: Cache context feature
        current_start_frame = 0
        # 如果给了initial_latent，进入如下分支
        # 它做了这件事：把参考图（首帧）“强行写入”模型的显存记忆（KV Cache）中，让后续生成的帧能够“看见”它。
        if initial_latent is not None:
            # 创建一个形状为 [B,1]=[1,1] 的张量，值全为 0。因为 initial_latent 是参考图，它是干净的 (Clean)，没有加噪。我们要告诉模型，这个参考图的时间步是 0。
            timestep = torch.ones([batch_size, 1], device=noise.device, dtype=torch.int64) * 0
            # Assume num_input_frames is 1 + self.num_frame_per_block * num_input_blocks
            
            # 把参考图复制到最终输出视频 output 的第 0 帧位置。
            # 数值: output 的 [0:1] 切片被填满。
            # 目的: 最终生成的视频必须包含这张参考图。
            output[:, :1] = initial_latent
            with torch.no_grad():
                # Generator 空跑 (Pre-filling Cache)
                self.generator(
                    noisy_image_or_video=initial_latent,     # 输入：干净的首帧 [1, 1, 16, 60, 104]
                    conditional_dict=conditional_dict,       # 文本prompt
                    timestep=timestep * 0,                   # 明确告诉它是 t=0
                    kv_cache=self.kv_cache1,                 # 【关键】传入全零的 Cache
                    crossattn_cache=self.crossattn_cache,    # 传入 CrossAttn Cache
                    current_start=current_start_frame * self.frame_seq_length   # 0 * 1560 = 0
                )
            #  帧指针加 1, 紧接着的 for 循环（自回归生成）将从 noise 的第 1 帧开始切片（而不是第 0 帧），并且它的 RoPE 将从 start_frame=1 开始计算。
            current_start_frame += 1

        # Step 3: Temporal denoising loop
        # all_num_frames = [3, 3, 3, 3, 3, 3, 3] (共7个)
        all_num_frames = [self.num_frame_per_block] * num_blocks    
        if self.independent_first_frame and initial_latent is None:
            # 如果第一帧独立生成，那么第一个 Block 只生成 1 帧，后续 Block 生成 3 帧。
            # 那么变为 [1, 3, 3, 3, 3, 3, 3, 3]。
            all_num_frames = [1] + all_num_frames
        num_denoising_steps = len(self.denoising_step_list)
        # 随机采样每个 Block 的梯度截断点 (Exit Flags)，这里num_denoising_steps = 4
        #  假设 exit_flags = [1, 2, 0, 3, 1, 1, 2]。这意味着 Block 0 会在第 1 步去噪时保留梯度，Block 1 在第 2 步保留，以此类推。这是为了省显存。
        exit_flags = self.generate_and_sync_list(len(all_num_frames), num_denoising_steps, device=noise.device)
        start_gradient_frame_index = num_output_frames - 21

        # 循环每个 Block (0 到 6)，现在来模拟一遍
        for block_index, current_num_frames in enumerate(all_num_frames):
            # 取出当前 Block 对应的噪声切片，不妨设现在生成Block 0 (Frame 0 到 2)
            # noisy_input 取自 noise[:, 0:3]，形状：[1, 3, 16, 60, 104]
            # current_start_frame = 0，current_start_token = 0。
            noisy_input = noise[
                :, current_start_frame - num_input_frames:current_start_frame + current_num_frames - num_input_frames]

            # 3.1 空间去噪循环 (Spatial Denoising Loop)
            # 例如 steps = [1000, 750, 500, 250] (Few-step diffusion) 或 warp过后 [1000, 937.5, 866.3, 625]
            for index, current_timestep in enumerate(self.denoising_step_list): # 不妨假设 t=1000 -> 750：
                if self.same_step_across_blocks:
                    exit_flag = (index == exit_flags[0])
                else:
                    exit_flag = (index == exit_flags[block_index])  # Only backprop at the randomly selected timestep (consistent across all ranks)

                # 这里的 timestep 是当前 Block 的时间步，形状为 [batch_size, current_num_frames] = [1, 3]
                timestep = torch.ones(
                    [batch_size, current_num_frames],
                    device=noise.device,
                    dtype=torch.int64) * current_timestep

                if not exit_flag:   # 如果还没到最后一步，不带着梯度去噪
                    with torch.no_grad():
                        _, denoised_pred = self.generator(
                            noisy_image_or_video=noisy_input,
                            conditional_dict=conditional_dict,
                            timestep=timestep,
                            kv_cache=self.kv_cache1,
                            crossattn_cache=self.crossattn_cache,
                            current_start=current_start_frame * self.frame_seq_length
                        )
                        next_timestep = self.denoising_step_list[index + 1]
                        noisy_input = self.scheduler.add_noise(
                            denoised_pred.flatten(0, 1),
                            torch.randn_like(denoised_pred.flatten(0, 1)),
                            next_timestep * torch.ones(
                                [batch_size * current_num_frames], device=noise.device, dtype=torch.long)
                        ).unflatten(0, denoised_pred.shape[:2])
                else:   
                    # 如果到了最后一步，
                    # for getting real output
                    # with torch.set_grad_enabled(current_start_frame >= start_gradient_frame_index):
                    if current_start_frame < start_gradient_frame_index:
                        # 如果当前 Block 还没到 start_gradient_frame_index，
                        with torch.no_grad():
                            # 这里的 generator 调用非常关键：
                            # kv_cache1 中存储了 *之前所有 Block* (由模型自己生成的) 的 KV。
                            # 模型只能看到过去自己生成的，看不到未来的 GT (因为根本没给GT)。
                            _, denoised_pred = self.generator(
                                noisy_image_or_video=noisy_input,
                                conditional_dict=conditional_dict,
                                timestep=timestep,
                                kv_cache=self.kv_cache1,
                                crossattn_cache=self.crossattn_cache,
                                current_start=current_start_frame * self.frame_seq_length
                            )
                    else:
                        # 这里的 generator 调用非常关键：
                        # kv_cache1 中存储了 *之前所有 Block* (由模型自己生成的) 的 KV。
                        # 模型只能看到过去自己生成的，看不到未来的 GT (因为根本没给GT)。
                        _, denoised_pred = self.generator(
                            noisy_image_or_video=noisy_input,
                            conditional_dict=conditional_dict,
                            timestep=timestep,
                            kv_cache=self.kv_cache1,
                            crossattn_cache=self.crossattn_cache,
                            current_start=current_start_frame * self.frame_seq_length
                        )
                    break

            # 3.2 记录当前 Block 最终生成的 latent
            output[:, current_start_frame:current_start_frame + current_num_frames] = denoised_pred

            # 3.3 更新 KV Cache
            # 我们在去噪过程中 Cache 存的是中间态（比如 t=750 的特征）。为了让下一个 Block (Block 1) 看到最干净的历史，我们需要用最终生成的 denoised_pred (视为 t=0) 再跑一遍模型。
            # 用生成好的 denoised_pred (即 x_0 估计) 再次过一遍模型
            # 这次是为了把生成的特征存入 self.kv_cache1，供下一个 Block 使用
            context_timestep = torch.ones_like(timestep) * self.context_noise   # 通常 t=0
            # add context noise
            denoised_pred = self.scheduler.add_noise(
                denoised_pred.flatten(0, 1),
                torch.randn_like(denoised_pred.flatten(0, 1)),
                # context_timestep * torch.ones([batch_size * current_num_frames], device=noise.device, dtype=torch.long)
                context_timestep.flatten(0, 1)
            ).unflatten(0, denoised_pred.shape[:2])
            with torch.no_grad():
                # 这玩意儿虽然没有返回值，但是在 self.generator 内部会更新 self.kv_cache1。使得Cache的第一个block的3帧都是绝对干净的
                self.generator(
                    noisy_image_or_video=denoised_pred,     # 刚才生成的 [1, 3, 16, 60, 104] 的 denoised_pred
                    conditional_dict=conditional_dict,      
                    timestep=context_timestep,              # context_timestep总是设置为t=0
                    kv_cache=self.kv_cache1,                # 写入 Cache
                    crossattn_cache=self.crossattn_cache,   # text embedding的 Cache
                    current_start=current_start_frame * self.frame_seq_length   # current_start = current_start_frame * 1560 (如果是第一个block，那么 current_start = 0，换言之会覆盖kv_cache的0:4680部分)
                )

            # Step 3.4: update the start and end frame indices
            # current_start_frame = 0 增加 current_num_frames = 3，挪动到下一个 Block 的起始位置
            current_start_frame += current_num_frames

        # Step 3.5: Return the denoised timestep
        if not self.same_step_across_blocks:
            denoised_timestep_from, denoised_timestep_to = None, None
        elif exit_flags[0] == len(self.denoising_step_list) - 1:
            denoised_timestep_to = 0
            denoised_timestep_from = 1000 - torch.argmin(
                (self.scheduler.timesteps.cuda() - self.denoising_step_list[exit_flags[0]].cuda()).abs(), dim=0).item()
        else:
            denoised_timestep_to = 1000 - torch.argmin(
                (self.scheduler.timesteps.cuda() - self.denoising_step_list[exit_flags[0] + 1].cuda()).abs(), dim=0).item()
            denoised_timestep_from = 1000 - torch.argmin(
                (self.scheduler.timesteps.cuda() - self.denoising_step_list[exit_flags[0]].cuda()).abs(), dim=0).item()

        if return_sim_step:
            return output, denoised_timestep_from, denoised_timestep_to, exit_flags[0] + 1

        # 返回完整生成的视频 output (由7个自回归生成的 Block 拼接而成)
        # 计算图:
        # output[:, 0:3]: 依赖于 Block 0 的计算图。
        # output[:, 3:6]: 依赖于 Block 1 的计算图，且通过 Cache 连接到 Block 0。
        # ... 以此类推
        return output, denoised_timestep_from, denoised_timestep_to

    def _initialize_kv_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU KV cache for the Wan model.
        """
        kv_cache1 = []

        for _ in range(self.num_transformer_blocks):
            # 模型有 30 层 Transformer Block，所以创建一个长度为 30 的列表 kv_cache1。
            # 列表中的每个元素（每层）是一个字典，包含：
            # "k": 形状 [1, 32760, 12, 128] (全零，显存预分配)。
            # "v": 形状 [1, 32760, 12, 128] (全零)。
            # "global_end_index": tensor([0]) (指针，指示当前 Cache 写到了哪里)。
            # "local_end_index": tensor([0])。
            kv_cache1.append({
                "k": torch.zeros([batch_size, self.kv_cache_size, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, self.kv_cache_size, 12, 128], dtype=dtype, device=device),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device)
            })

        self.kv_cache1 = kv_cache1  # always store the clean cache

    def _initialize_crossattn_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU cross-attention cache for the Wan model.
        """
        crossattn_cache = []

        for _ in range(self.num_transformer_blocks):
            # 模型有 30 层 Transformer Block，所以创建一个长度为 30 的列表 crossattn_cache。
            # 列表中的每个元素（每层）是一个字典，包含：
            # "k": 形状 [1, 512, 12, 128] (全零，显存预分配)。
            # "v": 形状 [1, 512, 12, 128] (全零)。
            # "is_init": False (标志，指示当前 Cache 是否已初始化)。
            # 注意：这里的 512 是一个固定值，表示每个 Block 的 cross-attention 特征长度（T5XXL 的text embedding长度）。
            crossattn_cache.append({
                "k": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                "is_init": False
            })
        self.crossattn_cache = crossattn_cache
