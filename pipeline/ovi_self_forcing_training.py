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
        num_frame_per_block_video: int,
        num_frame_per_block_audio: int,        
        num_training_frames_video: int,
        num_training_frames_audio: int,
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
        self.num_frame_per_block_video = num_frame_per_block_video
        self.num_frame_per_block_audio = num_frame_per_block_audio
        self.num_training_frames_video = num_training_frames_video
        self.num_training_frames_audio = num_training_frames_audio

    def _initialize_kv_cache_list(self, batch_size, h_latent, w_latent, dtype):
        if self.kv_cache_list is None:
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
                        "k": torch.zeros(batch_size, length, n_heads, d_head, device=self.device, dtype=dtype),
                        "v": torch.zeros(batch_size, length, n_heads, d_head, device=self.device, dtype=dtype),
                        "global_end_index": torch.zeros(1, device=self.device, dtype=torch.long),
                        "local_end_index": torch.zeros(1, device=self.device, dtype=torch.long),
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
        video_noise, audio_noise = noises
        device = video_noise.device     
        dtype = video_noise.dtype       # torch.bfloat16 if args.mixed_precision
        # video_noise shape: (B, F, C, H, W) = (B, 31, 48, H_real//16, W_real//16)
        # audio_noise shape: (B, L, D) = (B, 157, 20)
        num_denoising_steps = len(self.denoising_step_list) # here equals 4
        B, F_total, C_v, H_v, W_v = video_noise.shape
        _, L_total, C_a = audio_noise.shape

        # TODO: implement when correctly configured causal ovi related parameters
        gen_block_num = self.generate_and_sync_list(..., device=device)
        exit_flags = self.generate_and_sync_list(num_denoising_steps, device=device)    # Random exit step is shared between both branches, exit at 0-3

        # TODO: implement when correctly configured causal ovi related parameters
        # The file is copied from bidirectional, it is very important that the code here to align with inference code and ensure consistent of implementation.
        # Need to adapt bahaviors from pipeline/self_forcing_training.py
        ...
        denoised_preds, denoised_timestep_from, denoised_timestep_to = None, None, None
        return denoised_preds, denoised_timestep_from, denoised_timestep_to