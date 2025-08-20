from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, Wan2_2_VAEWrapper
from typing import List
import torch
from tqdm import tqdm
from utils.dataset import masks_like

class Wan22FewstepInferencePipeline(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.timestep_shift = getattr(args, "timestep_shift", 1.0)
        # Step 1: Initialize all models
        self.generator_model_name = getattr(args, "generator_name", args.model_name)
        self.generator = WanDiffusionWrapper(model_name=self.generator_model_name,is_causal=False)
        self.text_encoder = WanTextEncoder(model_name=self.generator_model_name)
        self.vae = Wan2_2_VAEWrapper()

        # Step 2: Initialize all bidirectional wan hyperparmeters
        self.denoising_step_list = torch.tensor(args.denoising_step_list, dtype=torch.long)
        self.scheduler = self.generator.get_scheduler()
        if args.warp_denoising_step:
            timesteps = torch.cat((self.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32)))
            self.denoising_step_list = timesteps[1000 - self.denoising_step_list]

    def inference(self, noise: torch.Tensor, text_prompts: List[str], wan22_image_latent : torch.Tensor = None) -> torch.Tensor:
        """
        Perform inference on the given noise and text prompts.
        Inputs:
            noise (torch.Tensor): The input noise tensor of shape
                (batch_size, num_frames, num_channels, height, width).
            text_prompts (List[str]): The list of text prompts.
        Outputs:
            video (torch.Tensor): The generated video tensor of shape
                (batch_size, num_frames, num_channels, height, width). It is normalized to be in the range [0, 1].
        """
        conditional_dict = self.text_encoder(
            text_prompts=text_prompts
        )

        # initial point
        noisy_image_or_video = noise


        if wan22_image_latent is not None:
            mask1, mask2 = masks_like(noisy_image_or_video, zero=True)
            mask2 = torch.stack(mask2, dim=0)
            noisy_image_or_video = (1. - mask2) * wan22_image_latent + mask2 * noisy_image_or_video
            noisy_image_or_video = noisy_image_or_video.to(noise.device, dtype=noise.dtype)
        else:
            mask1, mask2 = masks_like(noisy_image_or_video, zero=False)
            mask2 = torch.stack(mask2, dim=0)

        progress_bar = tqdm(
            enumerate(self.denoising_step_list), 
            total=len(self.denoising_step_list),
            desc="Denoising Steps",
            unit="step"
        )
        
        for index, current_timestep in progress_bar:

            wan22_input_timestep = torch.tensor([current_timestep.item()], device=noise.device, dtype=noise.dtype)
            temp_ts = (mask2[:, :, 0, ::2, ::2] * wan22_input_timestep) # torch.Size([1])
            temp_ts = temp_ts.reshape(temp_ts.shape[0], -1) # torch.Size([1, 15004])
            temp_ts = torch.cat([temp_ts, temp_ts.new_ones(temp_ts.shape[0], self.generator.seq_len - temp_ts.size(1)) * wan22_input_timestep.unsqueeze(-1)], dim=1)
            wan22_input_timestep = temp_ts.to(noise.device, dtype=torch.long)

            _, pred_image_or_video = self.generator(
                noisy_image_or_video=noisy_image_or_video,
                conditional_dict=conditional_dict,
                timestep=torch.ones(noise.shape[:2], dtype=torch.long, device=noise.device) * current_timestep,
                wan22_input_timestep=wan22_input_timestep,
                mask2=mask2,
                wan22_image_latent=wan22_image_latent,
            )

            if index < len(self.denoising_step_list) - 1:
                next_timestep = self.denoising_step_list[index + 1] * torch.ones(
                    noise.shape[:2], dtype=torch.long, device=noise.device)

                noisy_image_or_video = self.scheduler.add_noise(
                    pred_image_or_video.flatten(0, 1),
                    torch.randn_like(pred_image_or_video.flatten(0, 1)),
                    next_timestep.flatten(0, 1)
                ).unflatten(0, noise.shape[:2])
                if wan22_image_latent is not None:
                    # Apply the mask to the noisy image or video
                    noisy_image_or_video = (1. - mask2) * wan22_image_latent + mask2 * noisy_image_or_video
                    noisy_image_or_video = noisy_image_or_video.to(noise.device, dtype=noise.dtype)

        video = self.vae.decode_to_pixel(pred_image_or_video)
        video = (video * 0.5 + 0.5).clamp(0, 1)
        return video