from typing import List, Tuple
import torch

from utils.ovi_wrapper import OviFusionWrapper
from utils.scheduler import SchedulerInterface
import torch.distributed as dist
from utils.dataset import masks_like

class OviBidirectionalTrainingPipeline(torch.nn.Module):
    def __init__(
        self,
        model_name: str,
        denoising_step_list: List[int],
        scheduler: SchedulerInterface,
        generator: OviFusionWrapper, 
    ):
        super().__init__()
        self.model_name = model_name
        self.scheduler = scheduler
        self.generator = generator
        self.denoising_step_list = denoising_step_list
        if self.denoising_step_list[-1] == 0:
            self.denoising_step_list = self.denoising_step_list[:-1]

    def generate_and_sync_list(self, num_denoising_steps, device):
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
        video_noise, audio_noise = noises
        device = video_noise.device     
        dtype = video_noise.dtype       # torch.bfloat16 if args.mixed_precision
        # video_noise shape: (B, F, C, H, W) = (B, 31, 48, H_real//16, W_real//16)
        # audio_noise shape: (B, L, D) = (B, 157, 20)

        num_denoising_steps = len(self.denoising_step_list) # here equals 4
        exit_flags = self.generate_and_sync_list(num_denoising_steps, device=device)    # Random exit step is shared between both branches, exit at 0-3
        
        # Initial point is a tuple of noisy latents
        noisy_latents = (video_noise, audio_noise)
        # Denoising loop
        for index, current_timestep in enumerate(self.denoising_step_list):
            exit_flag = (index == exit_flags[0])
            noisy_video, noisy_audio = noisy_latents
            # timestep = torch.ones(noisy_video.shape[:2], device=device, dtype=torch.int64) * current_timestep   # shape: (B, F)
            timestep = torch.ones(noisy_video.shape[0], device=device, dtype=torch.long) * current_timestep   # shape: (B,)

            if "Ovi" in self.generator.model_name and wan22_image_latent is not None:
                mask1, mask2 = masks_like(noisy_video, zero=True)       # shape of mask2: (B, F, C, H, W)
                mask2 = torch.stack(mask2, dim=0)
                noisy_video = (1. - mask2) * wan22_image_latent + mask2 * noisy_video
                noisy_video = noisy_video.to(device, dtype=dtype)
                first_frame_is_clean = True
            else:
                mask2 = None
                first_frame_is_clean = False

            # Call the generator (OviDiffusionWrapper)
            if not exit_flag:
                with torch.no_grad():
                    pred_video, pred_audio = self.generator(
                        video_latent=noisy_video,   # shape: (B, F, C, H, W)
                        audio_latent=noisy_audio,   # shape: (B, L, D)
                        conditional_dict=conditional_dict,
                        timestep=timestep,
                        mask2=mask2,
                        wan22_image_latent=wan22_image_latent,
                        first_frame_is_clean=first_frame_is_clean,
                    )
                    # Add noise for the next step (logic remains the same, but inputs might be different)
                    next_timestep_val = self.denoising_step_list[index + 1]
                    next_ts = torch.full((pred_video.shape[0],), next_timestep_val, dtype=torch.long, device=device)
                    
                    next_noisy_video = self.scheduler.add_noise(pred_video, torch.randn_like(pred_video), next_ts)
                    next_noisy_audio = self.scheduler.add_noise(pred_audio, torch.randn_like(pred_audio), next_ts)
                    noisy_latents = (next_noisy_video, next_noisy_audio)
            else:
                # This is the exit step, compute with gradients
                pred_video, pred_audio = self.generator(
                    video_latent=noisy_video,
                    audio_latent=noisy_audio,
                    conditional_dict=conditional_dict,
                    timestep=timestep,
                    mask2=mask2,
                    wan22_image_latent=wan22_image_latent,
                    first_frame_is_clean=first_frame_is_clean,
                )
                denoised_preds = (pred_video, pred_audio)
                break

        # This part calculates the timestep range for logging/scheduling, remains the same.
        if exit_flags[0] == len(self.denoising_step_list) - 1:
            denoised_timestep_to = 0
            denoised_timestep_from = 1000 - torch.argmin(
                (self.scheduler.timesteps.to(device) - self.denoising_step_list[exit_flags[0]].to(device)).abs(), dim=0).item()
        else:
            denoised_timestep_to = 1000 - torch.argmin(
                (self.scheduler.timesteps.to(device) - self.denoising_step_list[exit_flags[0] + 1].to(device)).abs(), dim=0).item()
            denoised_timestep_from = 1000 - torch.argmin(
                (self.scheduler.timesteps.to(device) - self.denoising_step_list[exit_flags[0]].to(device)).abs(), dim=0).item()

        return denoised_preds, denoised_timestep_from, denoised_timestep_to