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
    ):
        super().__init__()
        self.model_name = model_name
        self.scheduler = scheduler
        self.generator = generator
        self.denoising_step_list = denoising_step_list  # after time shift, [1000, 750, 500, 250]->[995, 745, 495, 245]
        if self.denoising_step_list[-1] == 0:   # eliminate 0 if exists
            self.denoising_step_list = self.denoising_step_list[:-1]

        # TODO: need to configure causal ovi specific configuration, including causal related (such as block sizes for video/audio)
        ...
    def generate_and_sync_list(self, num_denoising_steps, device):
        # TODO: implement when correctly configured causal ovi related parameters
        # add block specific logic, take self_forcing_training.py as reference
        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank == 0:
            indices = torch.randint(low=0, high=num_denoising_steps, size=(1,), device=device)  # a randint index between 0 and 3, shape (1,)
        else:
            indices = torch.empty(1, dtype=torch.long, device=device)
        dist.broadcast(indices, src=0)
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

        # TODO: implement when correctly configured causal ovi related parameters
        gen_block_num = self.generate_and_sync_list(..., device=device)
        exit_flags = self.generate_and_sync_list(num_denoising_steps, device=device)    # Random exit step is shared between both branches, exit at 0-3

        # TODO: implement when correctly configured causal ovi related parameters
        # The file is copied from bidirectional, it is very important that the code here to align with inference code and ensure consistent of implementation.
        # Need to adapt bahaviors from pipeline/self_forcing_training.py
        ...
        denoised_preds, denoised_timestep_from, denoised_timestep_to = None, None, None
        return denoised_preds, denoised_timestep_from, denoised_timestep_to