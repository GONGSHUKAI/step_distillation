import torch
import torch.nn.functional as F
from typing import Optional, Tuple, Union

from model.ovi_base import OviSelfForcingModel
from utils.dataset import masks_like
import logging
logger = logging.getLogger(__name__)

class OviDMD(OviSelfForcingModel):
    def __init__(self, args, device):
        """
        Initialize the OviDMD (Distribution Matching Distillation) module for Audio-Video models.
        This class is adapted to handle dual-branch (video and audio) models.
        It computes generator and fake score losses for both modalities in the forward pass.
        """
        super().__init__(args, device)
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 31)
        self.num_training_frames = getattr(args, "num_training_frames", 31)
        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block
        if args.gradient_checkpointing:
            self.generator.enable_gradient_checkpointing()
            self.fake_score.enable_gradient_checkpointing()

        # this will be init later with fsdp-wrapped modules
        # TODO: self.inference_pipeline = None need to be realized later!
        self.inference_pipeline = None

        # DMD-specific parameters
        self.num_train_timestep = args.num_train_timestep
        self.min_step = int(0.02 * self.num_train_timestep)
        self.max_step = int(0.98 * self.num_train_timestep)
        self.real_guidance_scale = args.guidance_scale
        self.fake_guidance_scale = 0.0
        self.timestep_shift = getattr(args, "timestep_shift", 1.0)
        self.ts_schedule = getattr(args, "ts_schedule", True)
        self.ts_schedule_max = getattr(args, "ts_schedule_max", False)
        self.min_score_timestep = getattr(args, "min_score_timestep", 0)

        if getattr(self.scheduler, "alphas_cumprod", None) is not None:
            self.scheduler.alphas_cumprod = self.scheduler.alphas_cumprod.to(device)
        else:
            self.scheduler.alphas_cumprod = None

    def _compute_kl_grad(
        self,
        noisy_latents: Tuple[torch.Tensor, torch.Tensor],
        estimated_clean_latents: Tuple[torch.Tensor, torch.Tensor],
        timestep: torch.Tensor,
        conditional_dict: dict,
        unconditional_dict: dict,
        normalization: bool = True,
        wan22_image_latent: Optional[torch.Tensor] = None,
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], dict]:
        """
        Compute the KL grad for both video and audio branches.
        """
        noisy_video, noisy_audio = noisy_latents
        estimated_clean_video, estimated_clean_audio = estimated_clean_latents

        # --- Wan2.2 Specific Pre-processing for VIDEO branch ONLY ---
        if "Ovi" in self.generator.model_name and wan22_image_latent is not None:
            # Create mask and mix the first frame latent for the video
            mask1, mask2 = masks_like(noisy_video, zero=True)
            mask2 = torch.stack(mask2, dim=0)
            noisy_video = (1. - mask2) * wan22_image_latent + mask2 * noisy_video
            noisy_video = noisy_video.to(self.device, dtype=self.dtype)
            first_frame_is_clean = True
        else:
            mask2 = None
            first_frame_is_clean = False

        # --- Step 1: Compute the Fake Score (from Critic) for both branches ---
        pred_fake_video_cond, pred_fake_audio_cond = self.fake_score(
            video_latent=noisy_video,
            audio_latent=noisy_audio,
            conditional_dict=conditional_dict,
            timestep=timestep,
            mask2=mask2,
            wan22_image_latent=wan22_image_latent,
            first_frame_is_clean=first_frame_is_clean,
        )
        pred_fake_video = pred_fake_video_cond
        pred_fake_audio = pred_fake_audio_cond

        # --- Step 2: Compute the Real Score (from Teacher) for both branches ---
        pred_real_video_cond, pred_real_audio_cond = self.real_score(
            video_latent=noisy_video,
            audio_latent=noisy_audio,
            conditional_dict=conditional_dict,
            timestep=timestep,
            mask2=mask2,
            wan22_image_latent=wan22_image_latent,
            first_frame_is_clean=first_frame_is_clean,
        )
        pred_real_video_uncond, pred_real_audio_uncond = self.real_score(
            video_latent=noisy_video,
            audio_latent=noisy_audio,
            conditional_dict=unconditional_dict,
            timestep=timestep,
            mask2=mask2,
            wan22_image_latent=wan22_image_latent,
            first_frame_is_clean=first_frame_is_clean,
        )

        # Apply CFG for the teacher model
        pred_real_video = pred_real_video_cond + (pred_real_video_cond - pred_real_video_uncond) * self.real_guidance_scale
        pred_real_audio = pred_real_audio_cond + (pred_real_audio_cond - pred_real_audio_uncond) * self.real_guidance_scale

        # --- Step 3: Compute the DMD gradient for each branch ---
        grad_video = (pred_fake_video - pred_real_video)
        grad_audio = (pred_fake_audio - pred_real_audio)

        if normalization:
            # --- Video Branch Normalization ---
            p_real_video = (estimated_clean_video - pred_real_video)
            normalizer_video = torch.abs(p_real_video).mean(dim=[1, 2, 3, 4], keepdim=True)
            grad_video = grad_video / normalizer_video

            # --- Audio Branch Normalization ---
            # Note the different dimensions for mean calculation
            p_real_audio = (estimated_clean_audio - pred_real_audio)
            normalizer_audio = torch.abs(p_real_audio).mean(dim=[1, 2], keepdim=True)
            grad_audio = grad_audio / normalizer_audio

        grad_video = torch.nan_to_num(grad_video)
        grad_audio = torch.nan_to_num(grad_audio)
        
        # Combine gradients into a tuple
        grads = (grad_video, grad_audio)

        log_dict = {
            "dmdtrain_gradient_norm_video": torch.mean(torch.abs(grad_video)).detach(),
            "dmdtrain_gradient_norm_audio": torch.mean(torch.abs(grad_audio)).detach(),
            "timestep": timestep.detach()
        }
        return grads, log_dict

    def compute_distribution_matching_loss(
        self,
        latents: Tuple[torch.Tensor, torch.Tensor],
        conditional_dict: dict,
        unconditional_dict: dict,
        gradient_mask: Optional[torch.Tensor] = None,
        denoised_timestep_from: int = 0,
        denoised_timestep_to: int = 0,
        wan22_image_latent: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute the combined DMD loss for both video and audio branches.
        """
        video_latent, audio_latent = latents
        batch_size, num_frame = video_latent.shape[:2]

        with torch.no_grad():
            # Step 1: Sample timestep (shared for both branches) and add noise
            min_timestep = denoised_timestep_to if self.ts_schedule and denoised_timestep_to is not None else self.min_score_timestep
            max_timestep = denoised_timestep_from if self.ts_schedule_max and denoised_timestep_from is not None else self.num_train_timestep
            timestep = self._get_timestep(min_timestep, max_timestep, batch_size, num_frame, self.num_frame_per_block, uniform_timestep=True)

            if self.timestep_shift > 1:
                timestep = self.timestep_shift * (timestep / 1000) / (1 + (self.timestep_shift - 1) * (timestep / 1000)) * 1000
            timestep = timestep.clamp(self.min_step, self.max_step)

            # Add noise separately to each branch
            video_noise = torch.randn_like(video_latent)
            noisy_video = self.scheduler.add_noise(video_latent.flatten(0, 1), video_noise.flatten(0, 1), timestep.flatten(0, 1)).detach().unflatten(0, (batch_size, num_frame))

            audio_noise = torch.randn_like(audio_latent)
            # Timestep for audio needs to match the shape [B*L]
            audio_timestep_flat = timestep[:,0].unsqueeze(1).repeat(1, audio_latent.shape[1]).flatten()
            noisy_audio = self.scheduler.add_noise(audio_latent.flatten(0, 1), audio_noise.flatten(0, 1), audio_timestep_flat).detach().unflatten(0, (batch_size, audio_latent.shape[1]))

            # Step 2: Compute the KL grad for both branches
            grads, dmd_log_dict = self._compute_kl_grad(
                noisy_latents=(noisy_video, noisy_audio),
                estimated_clean_latents=(video_latent, audio_latent),
                timestep=timestep,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict,
                wan22_image_latent=wan22_image_latent,
            )
            grad_video, grad_audio = grads

        # Step 3: Calculate MSE loss for each branch and sum them up
        # --- Video Loss ---
        if gradient_mask is not None:
            # gradient_mask applies only to video frames
            dmd_loss_video = 0.5 * F.mse_loss(video_latent.double()[gradient_mask], (video_latent.double() - grad_video.double()).detach()[gradient_mask], reduction="mean")
        else:
            dmd_loss_video = 0.5 * F.mse_loss(video_latent.double(), (video_latent.double() - grad_video.double()).detach(), reduction="mean")
        
        # --- Audio Loss ---
        dmd_loss_audio = 0.5 * F.mse_loss(audio_latent.double(), (audio_latent.double() - grad_audio.double()).detach(), reduction="mean")

        # --- Combine Losses ---
        # A simple sum is used here. You could introduce a weighting factor if needed.
        total_dmd_loss = dmd_loss_video + dmd_loss_audio
        dmd_log_dict['dmd_loss_video'] = dmd_loss_video.detach()
        dmd_log_dict['dmd_loss_audio'] = dmd_loss_audio.detach()
        
        return total_dmd_loss, dmd_log_dict

    def generator_loss(
        self,
        latent_shapes: Tuple[list, list],
        conditional_dict: dict,
        unconditional_dict: dict,
        wan22_image_latent: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Run the dual-branch generator and compute the combined DMD loss.
        """
        # Step 1: Unroll generator via backward simulation to obtain fake video and audio latents
        pred_latents, gradient_mask, denoised_timestep_from, denoised_timestep_to = self._run_generator(
            latent_shapes=latent_shapes,
            conditional_dict=conditional_dict,
            wan22_image_latent=wan22_image_latent,
        )

        # Step 2: Compute the combined DMD loss on the generated latents
        dmd_loss, dmd_log_dict = self.compute_distribution_matching_loss(
            latents=pred_latents,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            gradient_mask=gradient_mask, # Applies only to video
            denoised_timestep_from=denoised_timestep_from,
            denoised_timestep_to=denoised_timestep_to,
            wan22_image_latent=wan22_image_latent,
        )

        del pred_latents, gradient_mask, denoised_timestep_from, denoised_timestep_to
        return dmd_loss, dmd_log_dict

    def critic_loss(
        self,
        latent_shapes: Tuple[list, list],
        conditional_dict: dict,
        unconditional_dict: dict,
        wan22_image_latent: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Train the critic using samples generated by the dual-branch generator.
        """
        video_shape, audio_shape = latent_shapes
        batch_size = video_shape[0]
        num_video_frames = video_shape[1]

        # Step 1: Generate samples from the generator without gradients
        with torch.no_grad():
            generated_latents, _, denoised_timestep_from, denoised_timestep_to = self._run_generator(
                latent_shapes=latent_shapes,
                conditional_dict=conditional_dict,
                wan22_image_latent=wan22_image_latent,
            )
        generated_video, generated_audio = generated_latents

        # Step 2: Sample a timestep and add noise to the generated samples
        min_timestep = denoised_timestep_to if self.ts_schedule and denoised_timestep_to is not None else self.min_score_timestep
        max_timestep = denoised_timestep_from if self.ts_schedule_max and denoised_timestep_from is not None else self.num_train_timestep
        critic_timestep = self._get_timestep(min_timestep, max_timestep, batch_size, num_video_frames, self.num_frame_per_block, uniform_timestep=True)

        if self.timestep_shift > 1:
            critic_timestep = self.timestep_shift * (critic_timestep / 1000) / (1 + (self.timestep_shift - 1) * (critic_timestep / 1000)) * 1000
        critic_timestep = critic_timestep.clamp(self.min_step, self.max_step)

        # Add noise separately to each branch
        critic_noise_video = torch.randn_like(generated_video)
        noisy_generated_video = self.scheduler.add_noise(generated_video.flatten(0, 1), critic_noise_video.flatten(0, 1), critic_timestep.flatten(0, 1)).unflatten(0, (batch_size, num_video_frames))

        critic_noise_audio = torch.randn_like(generated_audio)
        audio_timestep_flat = critic_timestep[:,0].unsqueeze(1).repeat(1, generated_audio.shape[1]).flatten()
        noisy_generated_audio = self.scheduler.add_noise(generated_audio.flatten(0, 1), critic_noise_audio.flatten(0, 1), audio_timestep_flat).unflatten(0, (batch_size, generated_audio.shape[1]))

        # Step 3: Apply Wan2.2 specific processing for the VIDEO branch ONLY
        if "2.2" in self.generator.model_name and wan22_image_latent is not None:
            mask1, mask2 = masks_like(noisy_generated_video, zero=True)
            mask2 = torch.stack(mask2, dim=0)
            noisy_generated_video = (1. - mask2) * wan22_image_latent + mask2 * noisy_generated_video
            noisy_generated_video = noisy_generated_video.to(self.device, dtype=self.dtype)
            first_frame_is_clean = True
        else:
            mask2 = None
            first_frame_is_clean = False
        
        # Step 4: Get predictions from the critic (fake_score)
        pred_fake_video, pred_fake_audio = self.fake_score(
            video_latent=noisy_generated_video,
            audio_latent=noisy_generated_audio,
            conditional_dict=conditional_dict,
            timestep=critic_timestep,
            mask2=mask2,
            wan22_image_latent=wan22_image_latent,
        )

        # Step 5: Compute the denoising loss for the critic on both branches
        # generated_video, pred_fake_video, critic_noise_video: shape [B, F, C, H, W]
        # generated_audio, pred_fake_audio, critic_noise_audio: shape [B, L, D]
        # critic_timestep shape: [B, F]
        # audio_timestep_flat shape: [B*L], (already flattened)

        # logger.info(f"generated_video shape: {generated_video.shape}, pred_fake_video shape: {pred_fake_video.shape}, critic_noise_video shape: {critic_noise_video.shape}")
        # logger.info(f"generated_audio shape: {generated_audio.shape}, pred_fake_audio shape: {pred_fake_audio.shape}, critic_noise_audio shape: {critic_noise_audio.shape}")
        # logger.info
        # logger.info(f"critic_timestep shape: {critic_timestep.shape}, audio_timestep_flat shape: {audio_timestep_flat.shape}")

        # --- Video Critic Loss ---
        denoising_loss_video = self.denoising_loss_func(
            x=generated_video.flatten(0, 1),
            x_pred=pred_fake_video.flatten(0, 1),
            noise=critic_noise_video.flatten(0, 1),
            # parameters not used
            noise_pred = None,
            timestep=critic_timestep.flatten(0, 1),
            alphas_cumprod=self.scheduler.alphas_cumprod
        )
        
        # --- Audio Critic Loss ---
        denoising_loss_audio = self.denoising_loss_func(
            x=generated_audio.flatten(0, 1),
            x_pred=pred_fake_audio.flatten(0, 1),
            noise=critic_noise_audio.flatten(0, 1),
            # parameters not used
            noise_pred = None, 
            timestep=audio_timestep_flat,
            alphas_cumprod=self.scheduler.alphas_cumprod
        )

        # --- Combine Losses ---
        total_denoising_loss = denoising_loss_video + denoising_loss_audio
        
        critic_log_dict = {
            "critic_loss_video": denoising_loss_video.detach(),
            "critic_loss_audio": denoising_loss_audio.detach(),
            "critic_timestep": critic_timestep.detach()
        }

        return total_denoising_loss, critic_log_dict