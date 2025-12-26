from typing import List, Tuple
import torch

from utils.ovi_wrapper import OviFusionWrapper
from utils.scheduler import SchedulerInterface
import torch.distributed as dist
from utils.dataset import masks_like
import logging
logger = logging.getLogger(__name__)

class OviSelfForcingTrainingPipeline(torch.nn.Module):
    def __init__(
        self,
        model_name: str,
        denoising_step_list: List[int],
        scheduler: SchedulerInterface,
        generator: OviFusionWrapper, 
        # TODO: need to configure causal ovi specific configuration, including causal related (such as block sizes for video/audio)
        # TODO: need to configure self forcing related parameters, take self_forcing_training.py as reference
        num_blocks: int,
        vid_block_size: int,
        aud_block_size: int,
        num_training_frames_video: int,
        num_training_frames_audio: int,
        start_gradient_frame_index_video: int,
        context_noise: float,
        last_step_only: bool,
        same_step_accross_blocks: bool,
    ):
        super().__init__()
        self.model_name = model_name
        self.scheduler = scheduler
        self.generator = generator
        self.denoising_step_list = denoising_step_list  # after time shift, [1000, 750, 500, 250]->[995, 745, 495, 245]
        if self.denoising_step_list[-1] == 0:   # eliminate 0 if exists
            self.denoising_step_list = self.denoising_step_list[:-1]

        # TODO: need to configure causal ovi specific configuration, including causal related (such as block sizes for video/audio)
        self.num_blocks = num_blocks
        self.vid_block_size = vid_block_size
        self.aud_block_size = aud_block_size
        self.num_training_frames_video = num_training_frames_video
        self.num_training_frames_audio = num_training_frames_audio

        # TODO: need to figure out the usage of these arguments
        self.start_gradient_frame_index_video = start_gradient_frame_index_video
        self.context_noise = context_noise
        self.last_step_only = last_step_only


        self.kv_cache_list = None

        # TODO: hard coding for now, need to make it configurable later
        self.tokens_per_vid_frame = 880     # 44*80/2/2
        self.tokens_per_aud_frame = 1
        self.num_layers = self.generator.model.num_blocks   # 30 layers for Ovi / Wan2.2 5B
        self.same_step_accross_blocks = same_step_accross_blocks


    def _initialize_kv_cache_list(self, batch_size, h_latent, w_latent, dtype):
        if self.kv_cache_list is None:
            device = torch.device('cuda')
            max_vid_tokens = 32 * self.tokens_per_vid_frame  # 32*880=28160
            max_aud_tokens = 160 * self.tokens_per_aud_frame
            
            head_dim_v = self.generator.model.video_config['dim'] // self.generator.model.video_config['num_heads'] # 3072//24=128
            head_dim_a = self.generator.model.audio_config['dim'] // self.generator.model.audio_config['num_heads'] # 3072//24=128
            num_heads_v = self.generator.model.video_config['num_heads']    # 24
            num_heads_a = self.generator.model.audio_config['num_heads']    # 24

            cache_list = []
            for _ in range(self.num_layers):
                layer_cache = {}
                def create_buf(length, n_heads, d_head):
                    return {
                        "k": torch.zeros(batch_size, length, n_heads, d_head, device=device, dtype=dtype),
                        "v": torch.zeros(batch_size, length, n_heads, d_head, device=device, dtype=dtype),
                        "global_end_index": torch.zeros(1, device=device, dtype=torch.long),
                        "local_end_index": torch.zeros(1, device=device, dtype=torch.long),
                    }
                
                layer_cache['vid_self'] = create_buf(max_vid_tokens, num_heads_v, head_dim_v)
                layer_cache['aud_self'] = create_buf(max_aud_tokens, num_heads_a, head_dim_a)
                layer_cache['vid_fusion'] = create_buf(max_aud_tokens, num_heads_v, head_dim_v)
                layer_cache['aud_fusion'] = create_buf(max_vid_tokens, num_heads_a, head_dim_a)
                layer_cache['vid_text'] = {"k": None, "v": None, "is_init": False}
                layer_cache['aud_text'] = {"k": None, "v": None, "is_init": False}
                cache_list.append(layer_cache)
            
            self.kv_cache_list = cache_list
            logger.info(f"KV Cache initialized.")
        else:
            for layer_cache in self.kv_cache_list:
                # reset selfattn cache
                for key in ['vid_self', 'aud_self', 'vid_fusion', 'aud_fusion']:
                    layer_cache[key]['global_end_index'].zero_()
                    layer_cache[key]['local_end_index'].zero_()
                
                # reset crossattn cache
                layer_cache['vid_text']['is_init'] = False
                layer_cache['aud_text']['is_init'] = False
            
            logger.info(f"KV Cache Indices Reset.")


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
        noises: Tuple[torch.Tensor, torch.Tensor],
        wan22_image_latent: torch.Tensor,
        **conditional_dict
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], int, int]:
        """
        MODIFIED FOR OVI: Perform inference on a tuple of (video_noise, audio_noise).
        The entire data flow now handles tuples for both modalities.
        """
        # These noise are nothing but a large container, no use
        noise_video, noise_audio = noises
        device = noise_video.device     
        dtype = noise_video.dtype       # torch.bfloat16 if args.mixed_precision
        # video_noise shape: (B, F, C, H, W) = (B, 31, 48, H_real//16, W_real//16)
        # audio_noise shape: (B, L, D) = (B, 157, 20)
        num_denoising_steps = len(self.denoising_step_list) # here equals 4
        num_blocks = self.num_blocks   # here equals 4
        start_gradient_frame_index_video = self.start_gradient_frame_index_video
        
        B, F_total, C_v, H_v, W_v = noise_video.shape
        _, L_total, C_a = noise_audio.shape
        dtype = noise_video.dtype
        self._initialize_kv_cache_list(B, H_v, W_v, dtype)

        out_video_latents, out_audio_latents = torch.zeros_like(noise_video), torch.zeros_like(noise_audio)

        _, mask2 = masks_like(noise_video[:, :self.vid_block_size], zero=True) # NOTE: ermu2001: masking on the causal block
        mask2 = torch.stack(mask2, dim=0).to(device, dtype=dtype) # [B, 4, 48, H, W]
        
        current_start_vid, current_start_audio = 0, 0
        all_num_frames_video = self.num_blocks * [self.vid_block_size]
        all_num_frames_audio = self.num_blocks * [self.aud_block_size]
        
        exit_flags = self.generate_and_sync_list(num_blocks, num_denoising_steps, device=device)    # Random exit step is shared between both branches, exit at 0-3

        start_gradient_frame_index_video = self.start_gradient_frame_index_video # TODO: figure out what is this for??

        # 4. Blockwise inference loop
        for block_idx in range(self.num_blocks):
            # logger.info(f"Processing Block {block_idx+1}/{self.num_blocks}...")
            curr_noise_v = noise_video[:, block_idx * self.vid_block_size : (block_idx + 1) * self.vid_block_size]
            curr_noise_a = noise_audio[:, block_idx * self.aud_block_size : (block_idx + 1) * self.aud_block_size]

            # Step 4.1: Denoising loop within each block
            curr_latent_v = curr_noise_v    # shape: [B, F_block=4, 48, H, W]
            curr_latent_a = curr_noise_a    # shape: [B, L_block=20, 20]

            for i, t_val in enumerate(self.denoising_step_list):
                is_first_block = (block_idx == 0)
                if is_first_block:
                    curr_latent_v = (1. - mask2) * wan22_image_latent + mask2 * curr_latent_v   # substitute latent first frame for latent ref frame
                    curr_latent_v = curr_latent_v.to(dtype)
                # Inference with kv cache
                
                if self.same_step_accross_blocks:
                    exit_flag = (i == exit_flags[0])
                else:
                    exit_flag = (i == exit_flags[block_idx])

                t_v = torch.full((B, self.vid_block_size), t_val.item(), dtype=t_val.dtype, device=curr_latent_v.device)    # [B, F_block] = [B, 4]
                t_a = torch.full((B, self.aud_block_size), t_val.item(), dtype=t_val.dtype, device=curr_latent_a.device)    # [B, L_block] = [B, 20]
                if not exit_flag:
                    with torch.no_grad():
                        x0_v, x0_a, _, _ = self.generator(
                            video_latent=curr_latent_v,
                            audio_latent=curr_latent_a,
                            timestep_v=t_v,
                            timestep_a=t_a,
                            conditional_dict=conditional_dict,
                            kv_cache_list=self.kv_cache_list,
                            current_start_vid=current_start_vid,
                            current_start_audio=current_start_audio,
                            wan22_image_latent=wan22_image_latent if is_first_block else None,
                            mask2=mask2 if is_first_block else None,
                            first_frame_is_clean=is_first_block
                        )
                        # TODO: figure out why this "if" here, if not needed, skip.
                        if i < len(self.denoising_step_list) - 1:
                            next_t = self.denoising_step_list[i+1]
                            next_ts_video = torch.full((x0_v.shape[0], x0_v.shape[1]), next_t.item(), device=x0_v.device, dtype=torch.long)
                            next_ts_audio = torch.full((x0_a.shape[0], x0_a.shape[1]), next_t.item(), device=x0_a.device, dtype=torch.long)
                            curr_latent_v = self.scheduler.add_noise(
                                x0_v.flatten(0, 1), 
                                torch.randn_like(x0_v.flatten(0, 1)), 
                                next_ts_video.flatten(0, 1)
                            ).unflatten(0, x0_v.shape[:2])
                            
                            curr_latent_a = self.scheduler.add_noise(
                                x0_a.flatten(0, 1), 
                                torch.randn_like(x0_a.flatten(0, 1)), 
                                next_ts_audio.flatten(0, 1)
                            ).unflatten(0, x0_a.shape[:2])
                else:
                    # NOTE: ermu2001: this implementation is extremely ugly, please fix once POC
                    if current_start_vid < start_gradient_frame_index_video:
                        # 如果当前 Block 还没到 start_gradient_frame_index，
                        with torch.no_grad():
                            # 这里的 generator 调用非常关键：
                            # kv_cache1 中存储了 *之前所有 Block* (由模型自己生成的) 的 KV。
                            # 模型只能看到过去自己生成的，看不到未来的 GT (因为根本没给GT)。
                            # _, denoised_pred = self.generator(
                            #     noisy_image_or_video=noisy_input,
                            #     conditional_dict=conditional_dict,
                            #     timestep=timestep,
                            #     kv_cache=self.kv_cache1,
                            #     crossattn_cache=self.crossattn_cache,
                            #     current_start=current_start_vid * self.frame_seq_length
                            # )
                            x0_v, x0_a, _, _ = self.generator(
                                video_latent=curr_latent_v,
                                audio_latent=curr_latent_a,
                                timestep_v=t_v,
                                timestep_a=t_a,
                                conditional_dict=conditional_dict,
                                kv_cache_list=self.kv_cache_list,
                                current_start_vid=current_start_vid,
                                current_start_audio=current_start_audio,
                                wan22_image_latent=wan22_image_latent if is_first_block else None,
                                mask2=mask2 if is_first_block else None,
                                first_frame_is_clean=is_first_block
                            )
                    else:
                        # 这里的 generator 调用非常关键：
                        # kv_cache1 中存储了 *之前所有 Block* (由模型自己生成的) 的 KV。
                        # 模型只能看到过去自己生成的，看不到未来的 GT (因为根本没给GT)。
                        # _, denoised_pred = self.generator(
                        #     noisy_image_or_video=noisy_input,
                        #     conditional_dict=conditional_dict,
                        #     timestep=timestep,
                        #     kv_cache=self.kv_cache1,
                        #     crossattn_cache=self.crossattn_cache,
                        #     current_start=current_start_frame * self.frame_seq_length
                        # )
                        x0_v, x0_a, _, _ = self.generator(
                            video_latent=curr_latent_v,
                            audio_latent=curr_latent_a,
                            timestep_v=t_v,
                            timestep_a=t_a,
                            conditional_dict=conditional_dict,
                            kv_cache_list=self.kv_cache_list,
                            current_start_vid=current_start_vid,
                            current_start_audio=current_start_audio,
                            wan22_image_latent=wan22_image_latent if is_first_block else None,
                            mask2=mask2 if is_first_block else None,
                            first_frame_is_clean=is_first_block
                        )
                    break # NOTE: ermu2001: then this is a bit wierd since sampled exit flag not necessary is the "last step". The self forcing is bahaving on top of various noise levels denoised predictions.


            # Step 4.2: Fill in x0 for this block into the final output
            out_video_latents[:, block_idx * self.vid_block_size : (block_idx + 1) * self.vid_block_size] = x0_v
            out_audio_latents[:, block_idx * self.aud_block_size : (block_idx + 1) * self.aud_block_size] = x0_a
            
            # # TODO: not yet done debugging, if want to could use:
            # self.save_video_debug(x0_v, x0_a, block_idx)

            # Step 4.3: Update KV Cache with clean latent (i.e. latent denoised in the final timestep).
            with torch.no_grad():
                t_context_v = torch.ones((B, self.vid_block_size), device=x0_v.device) * self.context_noise
                t_context_a = torch.ones((B, self.aud_block_size), device=x0_a.device) * self.context_noise

                if is_first_block:
                    x0_v = (1. - mask2) * wan22_image_latent + mask2 * x0_v
                    x0_v = x0_v.to(dtype)

                self.generator(
                    video_latent=x0_v,
                    audio_latent=x0_a,
                    timestep_v=t_context_v,
                    timestep_a=t_context_a,
                    conditional_dict=conditional_dict,
                    kv_cache_list=self.kv_cache_list,
                    current_start_vid=current_start_vid,
                    current_start_audio=current_start_audio,

                    wan22_image_latent=wan22_image_latent if is_first_block else None,
                    mask2=mask2 if is_first_block else None,
                    first_frame_is_clean=is_first_block
                )
            
            # Update counter
            current_start_vid += (self.vid_block_size * self.tokens_per_vid_frame)
            current_start_audio += (self.aud_block_size * self.tokens_per_aud_frame)

        # 5. Denoised output: crop back to 31 video frames and 157 audio frames to match original latent size of Ovi
        final_v_lat = out_video_latents[:, :31]
        final_a_lat = out_audio_latents[:, :157] # NOTE: for self forcing, need to return same shape as input noise (at least order of dimensions should be the same)
        denoised_preds = (final_v_lat, final_a_lat)

        # TODO: figure out what these two are for, need to checkout self_forcing_training.py, for corresponding steps
        denoised_timestep_from = self.denoising_step_list[0]
        denoised_timestep_to = self.denoising_step_list[-1]

        return denoised_preds, denoised_timestep_from, denoised_timestep_to