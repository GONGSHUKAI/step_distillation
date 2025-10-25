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
            indices = torch.randint(low=0, high=num_denoising_steps, size=(1,), device=device)
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
        num_denoising_steps = len(self.denoising_step_list)
        exit_flags = self.generate_and_sync_list(num_denoising_steps, device=device)    # Random exit step is shared between both branches
        device = video_noise.device
        dtype = video_noise.dtype
        
        # Initial point is a tuple of noisy latents
        noisy_latents = (video_noise, audio_noise)
        # Denoising loop
        for index, current_timestep in enumerate(self.denoising_step_list):
            exit_flag = (index == exit_flags[0])
            
            noisy_video, noisy_audio = noisy_latents
            
            # Timestep is primarily shaped for video [B, F]
            timestep = torch.ones(noisy_video.shape[:2], device=device, dtype=torch.int64) * current_timestep

            # --- Wan2.2 video-specific processing ---
            # This part ONLY applies to the video latent.
            if "Ovi" in self.generator.model_name and wan22_image_latent is not None:
                mask1, mask2 = masks_like(noisy_video, zero=True)
                mask2 = torch.stack(mask2, dim=0)
                noisy_video = (1. - mask2) * wan22_image_latent + mask2 * noisy_video
                noisy_video = noisy_video.to(device, dtype=dtype)

                # Construct special timestep format for the video model
                wan22_input_timestep = torch.tensor([timestep[0][0].item()], device=device, dtype=torch.long)
                temp_ts = (mask2[:, :, 0, ::2, ::2] * wan22_input_timestep.float())
                temp_ts = temp_ts.reshape(temp_ts.shape[0], -1)
                temp_ts = torch.cat([temp_ts, temp_ts.new_ones(temp_ts.shape[0], self.generator.seq_len - temp_ts.size(1)) * wan22_input_timestep.float()], dim=1)
                wan22_input_timestep = temp_ts.to(device, dtype=torch.long)
            else:
                mask2 = None
                wan22_input_timestep = None
            
            # Re-package latents after video-specific modifications
            current_noisy_latents = (noisy_video, noisy_audio)

            # Call the generator (OviDiffusionWrapper)
            if not exit_flag:
                with torch.no_grad():
                    # Generator returns a tuple of predictions
                    _, denoised_preds = self.generator(
                        noisy_latents=current_noisy_latents,
                        conditional_dict=conditional_dict,
                        timestep=timestep,
                        wan22_input_timestep=wan22_input_timestep,
                        mask2=mask2,
                        wan22_image_latent=wan22_image_latent,
                    )
                    pred_video, pred_audio = denoised_preds

                    # Add noise to the next timestep for both branches
                    next_timestep_val = self.denoising_step_list[index + 1]
                    
                    # --- Video Branch Noise Addition ---
                    next_timestep_video = next_timestep_val * torch.ones(pred_video.shape[:2], dtype=torch.long, device=device)
                    next_noisy_video = self.scheduler.add_noise(
                        pred_video.flatten(0, 1),
                        torch.randn_like(pred_video.flatten(0, 1)),
                        next_timestep_video.flatten(0, 1)
                    ).unflatten(0, pred_video.shape[:2])
                    
                    # --- Audio Branch Noise Addition ---
                    next_timestep_audio_flat = next_timestep_val * torch.ones(pred_audio.numel() // pred_audio.shape[-1], dtype=torch.long, device=device)
                    next_noisy_audio = self.scheduler.add_noise(
                        pred_audio.flatten(0, 1),
                        torch.randn_like(pred_audio.flatten(0, 1)),
                        next_timestep_audio_flat
                    ).unflatten(0, pred_audio.shape[:2])

                    noisy_latents = (next_noisy_video, next_noisy_audio)
            else:
                # This is the exit step, compute with gradients
                _, denoised_preds = self.generator(
                    noisy_latents=current_noisy_latents,
                    conditional_dict=conditional_dict,
                    timestep=timestep,
                    wan22_input_timestep=wan22_input_timestep,
                    mask2=mask2,
                    wan22_image_latent=wan22_image_latent,
                )
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