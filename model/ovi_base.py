# base_ovi.py (A new or modified file for Ovi's base models)

from typing import Tuple
from einops import rearrange
from torch import nn
import torch.distributed as dist
import torch

# Assuming your Ovi wrappers and pipelines are defined elsewhere
from pipeline import OviBidirectionalTrainingPipeline # You need to create this
from utils.loss import get_denoising_loss
from utils.ovi_wrapper import OviFusionWrapper, OviTextEncoder, OviVAEWrapper # Use your Ovi wrappers

class OviBaseModel(nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self.is_causal = False
        self._initialize_models(args, device)

        self.device = device
        self.args = args
        self.dtype = torch.bfloat16 if args.mixed_precision else torch.float32
        if hasattr(args, "denoising_step_list"):
            self.denoising_step_list = torch.tensor(args.denoising_step_list, dtype=torch.long)
            if args.warp_denoising_step:
                timesteps = torch.cat((self.generator.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32)))
                self.denoising_step_list = timesteps[1000 - self.denoising_step_list]

    def _initialize_models(self, args, device):
        self.real_model_name = getattr(args, "real_name", "Ovi")        # the teacher model
        self.fake_model_name = getattr(args, "fake_name", "Ovi")        # the critic model
        self.generator_name = getattr(args, "generator_name", "Ovi")    # the student model

        self.generator = OviFusionWrapper(
            **getattr(args, "model_kwargs", {}),
            model_name=self.generator_name,
            is_causal=self.is_causal
        )
        self.generator.model.requires_grad_(True)

        self.real_score = OviFusionWrapper(model_name=self.real_model_name, is_causal=False)
        self.real_score.model.requires_grad_(False)

        self.fake_score = OviFusionWrapper(model_name=self.fake_model_name, is_causal=False)
        self.fake_score.model.requires_grad_(True)

        self.text_encoder = OviTextEncoder()
        self.text_encoder.requires_grad_(False)
        
        # Ovi has a unified VAE wrapper for both video and audio
        self.vae = OviVAEWrapper()
        self.vae.requires_grad_(False)

        self.scheduler = self.generator.get_scheduler()
        self.scheduler.timesteps = self.scheduler.timesteps.to(device)

    def _get_timestep(
            self,
            min_timestep: int,
            max_timestep: int,
            batch_size: int,
            num_frame: int, # num_frame is specific to video
            num_frame_per_block: int,
            uniform_timestep: bool = False
    ) -> torch.Tensor:
        """
        This function is video-specific. For Ovi, timestep is usually uniform
        across the whole sample (both video and audio). The original implementation is fine
        as long as `uniform_timestep=True` is used, which it is in dmd.py.
        """
        # No changes needed here if uniform_timestep=True is always used for Ovi.
        if uniform_timestep:
            timestep = torch.randint(
                min_timestep,
                max_timestep,
                [batch_size, 1],
                device=self.device,
                dtype=torch.long
            ).repeat(1, num_frame)
            return timestep
        else:
            raise NotImplementedError("Non-uniform timestep is not supported for Ovi distillation.")

class OviSelfForcingModel(OviBaseModel):
    def __init__(self, args, device):
        super().__init__(args, device)
        self.denoising_loss_func = get_denoising_loss(args.denoising_loss_type)()

    def _run_generator(
        self,
        latent_shapes: Tuple[list, list],
        conditional_dict: dict,
        wan22_image_latent: torch.Tensor = None,
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor, int, int]:
        """
        MODIFIED FOR OVI: Handles both video and audio branches.
        Generates latents for both modalities using backward simulation.
        Inputs:
            - latent_shapes: A tuple of two lists, each specifying the shape of video and audio latents. [B, F, C, H, W] for video and [B, L] for audio.
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
            - wan22_image_latent: a tensor with shape [B, 1, C, H, W] or None, used for Wan2.2 video part only.
        """
        video_shape, audio_shape = latent_shapes
        # Step 1: Sample noise and backward simulate the generator's input
        assert getattr(self.args, "backward_simulation", True), "Backward simulation needs to be enabled"
        video_noise_shape = video_shape.copy()
        audio_noise_shape = audio_shape.copy()
        
        # --- Frame Sampling Logic (from original, applies ONLY to video) ---
        # NOTE: This assumes the audio length is tied to the video length.
        latent_frames_num = 31 # Wan2.2 video part specific
        min_num_frames = latent_frames_num  # 31
        max_num_frames = self.num_training_frames  # 31
        
        assert max_num_frames % self.num_frame_per_block == 0 
        assert min_num_frames % self.num_frame_per_block == 0
        
        max_num_blocks = max_num_frames // self.num_frame_per_block # 31//31=1
        min_num_blocks = min_num_frames // self.num_frame_per_block # 31//31=1
        num_generated_blocks = torch.randint(min_num_blocks, max_num_blocks + 1, (1,), device=self.device)  # num_generated_blocks in [1, 2), which is always 1 here
        dist.broadcast(num_generated_blocks, src=0)
        num_generated_blocks = num_generated_blocks.item()
        num_generated_frames = num_generated_blocks * self.num_frame_per_block  # num_generated_frames is always 31 here
        
        video_noise_shape[1] = num_generated_frames
        
        # --- Create noise tensors ---
        video_noise = torch.randn(video_noise_shape, device=self.device, dtype=self.dtype)
        audio_noise = torch.randn(audio_noise_shape, device=self.device, dtype=self.dtype)

        # --- Run Backward Simulation for BOTH branches ---
        # The pipeline must be adapted to handle a tuple of noises and return a tuple of latents.
        pred_latents, denoised_timestep_from, denoised_timestep_to = self._consistency_backward_simulation(
            noises=(video_noise, audio_noise),
            wan22_image_latent=wan22_image_latent,
            **conditional_dict
        )
        pred_video, pred_audio = pred_latents

        # # --- Post-processing (applies ONLY to video) ---
        # # The logic for re-encoding the first frame for long videos is kept.
        # if pred_video.shape[1] > latent_frames_num: # i.e., num_generated_frames > 31, which will not happen here
        #     with torch.no_grad():
        #         latent_to_decode = pred_video[:, :-(latent_frames_num-1), ...]
        #         pixels = self.vae.decode_video(latent_to_decode) # Use the video part of the VAE
        #         frame = pixels[:, -1:, ...].to(self.dtype)
        #         frame = rearrange(frame, "b t c h w -> b c t h w")
        #         image_latent = self.vae.video_vae.encode_to_latent(frame).to(self.dtype)
            
        #     pred_video_last_clip = torch.cat([image_latent, pred_video[:, -(latent_frames_num-1):, ...]], dim=1)
        # else:
        #     pred_video_last_clip = pred_video

        # # --- Gradient Mask (applies ONLY to video) ---
        # if num_generated_frames != min_num_frames:  # 31 != 31, which will not happen here
        #     gradient_mask = torch.ones_like(pred_video_last_clip, dtype=torch.bool)
        #     # Do not compute loss on the context frames (first block)
        #     gradient_mask[:, :self.num_frame_per_block] = False
        # else:
        #     gradient_mask = None
        gradient_mask = None
        final_pred_latents = (pred_video.to(self.dtype), pred_audio.to(self.dtype))
        
        return final_pred_latents, gradient_mask, denoised_timestep_from, denoised_timestep_to

    def _consistency_backward_simulation(
        self,
        noises: Tuple[torch.Tensor, torch.Tensor],
        wan22_image_latent: torch.Tensor,
        **conditional_dict: dict
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], int, int]:
        """
        MODIFIED FOR OVI: The pipeline now takes a tuple of noises.
        """
        if self.inference_pipeline is None:
            self._initialize_inference_pipeline()

        return self.inference_pipeline.inference_with_trajectory(
            noises=noises, wan22_image_latent=wan22_image_latent, **conditional_dict
        )

    def _initialize_inference_pipeline(self):
        """
        MODIFIED FOR OVI: Initialize a pipeline that supports dual branches.
        """
        if self.is_causal:
            raise NotImplementedError("Causal models are not supported for Ovi.")
        else:
            # You must create this new pipeline class.
            self.inference_pipeline = OviBidirectionalTrainingPipeline(
                model_name=self.generator_name,
                denoising_step_list=self.denoising_step_list,
                scheduler=self.scheduler,
                generator=self.generator, # Pass the OviFusionWrapper
            )