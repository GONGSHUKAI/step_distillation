# FILE: model/ovi_self_forcing_regression.py
from typing import Tuple, Optional
import tempfile
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from utils.ovi_wrapper import OviFusionWrapper, OviTextEncoder, OviVAEWrapper
from utils.dataset import OviODERegressionDataset, cycle, masks_like
from pipeline import OviSelfForcingTrainingPipeline
import numpy as np
from model.ovi_base import OviBaseModel
import logging
from ovi.utils.io_utils import save_video

logger = logging.getLogger(__name__)


class OviSelfForcingRegression(OviBaseModel):
    def __init__(self, args, device):
        super().__init__(args, device)
        self.args = args
        self.device = device

        if self.denoising_step_list[-1] == 0:
            self.denoising_step_list = self.denoising_step_list[:-1]
        logger.info(f"Self-Forcing Regression denoising step list: {self.denoising_step_list}") if not dist.is_initialized() or dist.get_rank() == 0 else None

        self.num_frame_per_block_vid = 4
        self.num_frame_per_block_aud = 20
        self.num_blocks = 8
        self.num_aud_frame_per_vid = self.num_frame_per_block_aud // self.num_frame_per_block_vid
        
        self.num_training_frames_video = getattr(args, 'num_training_frames_video', 32)
        self.num_training_frames_audio = getattr(args, 'num_training_frames_audio', 160)
        self.start_gradient_frame_index_video = getattr(args, 'start_gradient_frame_index_video', 0)
        self.context_noise = getattr(args, 'context_noise', 0.0)
        self.last_step_only = getattr(args, 'last_step_only', False)  
        self.same_step_accross_blocks = getattr(args, 'same_step_accross_blocks', False)

        self.inference_pipeline = None

    def _initialize_models(self, args, device):
        self.generator_name = getattr(args, "generator_name", "Ovi")
        self.generator_path = getattr(args, "generator_path", None)

        self.generator = OviFusionWrapper(
            **getattr(args, "model_kwargs", {}),
            model_name=self.generator_name,
            model_path=self.generator_path,
            is_causal=self.is_causal
        )
        self.generator.model.requires_grad_(True)

        self.text_encoder = OviTextEncoder()
        self.text_encoder.requires_grad_(False)

        self.vae = OviVAEWrapper()
        self.vae.requires_grad_(False)

        self.scheduler = self.generator.get_scheduler()
        self.scheduler.timesteps = self.scheduler.timesteps.to(device)

    def _initialize_inference_pipeline(self):
        if self.inference_pipeline is None:
            self.inference_pipeline = OviSelfForcingTrainingPipeline(
                model_name=self.generator_name,
                denoising_step_list=self.denoising_step_list,
                scheduler=self.scheduler,
                generator=self.generator,
                num_blocks=self.num_blocks,
                vid_block_size=self.num_frame_per_block_vid,
                aud_block_size=self.num_frame_per_block_aud,
                num_training_frames_video=self.num_training_frames_video,
                num_training_frames_audio=self.num_training_frames_audio,
                start_gradient_frame_index_video=self.start_gradient_frame_index_video,
                context_noise=self.context_noise,
                last_step_only=self.last_step_only,
                same_step_accross_blocks=self.same_step_accross_blocks,
            )

    def _prepare_noises_from_ode_latent(
        self, 
        video_ode_latent: torch.Tensor, 
        audio_ode_latent: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        noisy_video = video_ode_latent[:, 0]
        noisy_audio = audio_ode_latent[:, 0]
        clean_video = video_ode_latent[:, -1]
        wan22_image_latent = clean_video[:, 0:1, :, :, :]
        
        return noisy_video, noisy_audio, wan22_image_latent

    def generator_loss(
        self, 
        video_ode_latent: torch.Tensor, 
        audio_ode_latent: torch.Tensor, 
        conditional_dict: dict,
        log_video: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, dict]:
        if self.inference_pipeline is None:
            self._initialize_inference_pipeline()
        
        target_video_latent = video_ode_latent[:, -1]  # [B, 32, C, H, W]
        target_audio_latent = audio_ode_latent[:, -1]  # [B, 160, D]
        noise_video, noise_audio, wan22_image_latent = self._prepare_noises_from_ode_latent(
            video_ode_latent, audio_ode_latent
        )
        
        B = video_ode_latent.shape[0]
        
        cond_expanded = {
            "video_prompt_embeds": conditional_dict["prompt_embeds"],
            "audio_prompt_embeds": conditional_dict["prompt_embeds"]
        }
        
        noise_video = noise_video.to(self.device, dtype=self.dtype)
        noise_audio = noise_audio.to(self.device, dtype=self.dtype)
        wan22_image_latent = wan22_image_latent.to(self.device, dtype=self.dtype)
        
        noises = (noise_video, noise_audio)
        
        denoised_preds, _, _ = self.inference_pipeline.inference_with_trajectory(
            noises=noises,
            wan22_image_latent=wan22_image_latent,
            **cond_expanded
        )
        
        pred_video, pred_audio = denoised_preds  # [B, 31, C, H, W], [B, 157, D]
        target_video = target_video_latent[:, :31]  # [B, 31, C, H, W]
        target_audio = target_audio_latent[:, :157]  # [B, 157, D]
        log_dict = {}
        if log_video and dist.get_rank() == 0:
            logger.info("Logging video and audio latents from generator rollout...")
            with torch.no_grad():
                device, dtype = 'cuda', torch.bfloat16
                video = self.vae.decode_video(pred_video[:1].to(device, dtype)) # [B, C, F, H, W]
                audio = self.vae.decode_audio(pred_audio[:1].transpose(1, 2).to(device, dtype)) # [B, L]
                video = ((video + 1) / 2 * 255).clip(0, 255)
                video_np = video.squeeze(0).permute(1, 0, 2, 3).cpu().float().numpy().astype(np.uint8)
                audio_np = audio.squeeze(0).cpu().float().numpy().flatten()

                gt_video = self.vae.decode_video(target_video[:1].to(device, dtype)) # [B, C, F, H, W]
                gt_audio = self.vae.decode_audio(target_audio[:1].transpose(1, 2).to(device, dtype)) # [B, L]
                gt_video = ((gt_video + 1) / 2 * 255).clip(0, 255)
                gt_video_np = gt_video.squeeze(0).permute(1, 0, 2, 3).cpu().float().numpy().astype(np.uint8)
                gt_audio_np = gt_audio.squeeze(0).cpu().float().numpy().flatten()

                # create a temp file and save to it
                with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f1:
                    save_video(
                        output_path=f1.name,
                        video_numpy=video_np,
                        audio_numpy=audio_np
                    )
                with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f2:
                    save_video(
                        output_path=f2.name,
                        video_numpy=gt_video_np,
                        audio_numpy=gt_audio_np
                    )
                log_dict['generated_video_audio'] = f1.name
                log_dict['gt_video_audio'] = f2.name
        
        mask_v = torch.ones(B, 31, 1, 1, 1, device=self.device, dtype=torch.bool)
        mask_v[:, 0] = False  # Exclude first frame (clean reference)
        mask_v = mask_v.expand_as(pred_video)
        
        mask_a = torch.ones(B, 157, 1, device=self.device, dtype=torch.bool)
        mask_a = mask_a.expand_as(pred_audio)
        
        loss_v = F.mse_loss(pred_video[mask_v], target_video[mask_v], reduction="mean")
        loss_a = F.mse_loss(pred_audio[mask_a], target_audio[mask_a], reduction="mean")
        
        total_loss = 0.85 * loss_v + 0.15 * loss_a
        log_dict.update({
            "loss_video": loss_v.detach(),
            "loss_audio": loss_a.detach()
        })
        return total_loss, log_dict

    @torch.no_grad()
    def full_inference(
        self, 
        noises: Tuple[torch.Tensor, torch.Tensor], 
        wan22_image_latent: torch.Tensor, 
        **conditional_dict
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.inference_pipeline is None:
            self._initialize_inference_pipeline()
        
        return self.inference_pipeline.full_inference(
            noises=noises,
            wan22_image_latent=wan22_image_latent,
            **conditional_dict
        )


class OviSelfForcingRegressionDebug(OviSelfForcingRegression):
    def generator_loss(
        self, 
        video_ode_latent: torch.Tensor, 
        audio_ode_latent: torch.Tensor, 
        conditional_dict: dict
    ) -> Tuple[torch.Tensor, dict]:        
        if self.inference_pipeline is None:
            self._initialize_inference_pipeline()
        
        target_video_latent = video_ode_latent[:, -1]
        target_audio_latent = audio_ode_latent[:, -1]
        
        noise_video, noise_audio, wan22_image_latent = self._prepare_noises_from_ode_latent(
            video_ode_latent, audio_ode_latent
        )
        
        B = video_ode_latent.shape[0]
        
        cond_expanded = {
            "video_prompt_embeds": conditional_dict["prompt_embeds"],
            "audio_prompt_embeds": conditional_dict["prompt_embeds"]
        }
        
        noise_video = noise_video.to(self.device, dtype=self.dtype)
        noise_audio = noise_audio.to(self.device, dtype=self.dtype)
        wan22_image_latent = wan22_image_latent.to(self.device, dtype=self.dtype)
        
        # Debug: Save noisy input
        if dist.get_rank() == 0:
            print(f"[Debug] noise_video shape: {noise_video.shape}")
            print(f"[Debug] noise_audio shape: {noise_audio.shape}")
            print(f"[Debug] wan22_image_latent shape: {wan22_image_latent.shape}")
        
        noises = (noise_video, noise_audio)
        
        denoised_preds, denoised_from, denoised_to = self.inference_pipeline.inference_with_trajectory(
            noises=noises,
            wan22_image_latent=wan22_image_latent,
            **cond_expanded
        )
        
        pred_video, pred_audio = denoised_preds
        
        if dist.get_rank() == 0:
            print(f"[Debug] pred_video shape: {pred_video.shape}")
            print(f"[Debug] pred_audio shape: {pred_audio.shape}")
            # print(f"[Debug] denoised from timestep: {denoised_from} to: {denoised_to}")
        
        target_video = target_video_latent[:, :31]
        target_audio = target_audio_latent[:, :157]
        
        mask_v = torch.ones(B, 31, 1, 1, 1, device=self.device, dtype=torch.bool)
        mask_v[:, 0] = False
        mask_v = mask_v.expand_as(pred_video)
        
        mask_a = torch.ones(B, 157, 1, device=self.device, dtype=torch.bool)
        mask_a = mask_a.expand_as(pred_audio)
        
        loss_v = F.mse_loss(pred_video[mask_v], target_video[mask_v], reduction="mean")
        loss_a = F.mse_loss(pred_audio[mask_a], target_audio[mask_a], reduction="mean")
        
        if dist.get_rank() == 0:
            print(f"[Debug] loss_video: {loss_v.item():.6f}")
            print(f"[Debug] loss_audio: {loss_a.item():.6f}")
        
        total_loss = 0.85 * loss_v + 0.15 * loss_a
        
        return total_loss, {
            "loss_video": loss_v.detach(),
            "loss_audio": loss_a.detach()
        }
    
    @torch.no_grad()
    def save_debug_videos(
        self, 
        pred_video: torch.Tensor, 
        pred_audio: torch.Tensor,
        target_video: torch.Tensor,
        target_audio: torch.Tensor,
        prefix: str = "debug"
    ):
        if dist.get_rank() != 0:
            return
        
        # Decode and save predicted
        self.vae.to(self.device, self.dtype)
        
        pred_v_decoded = self.vae.decode_video(pred_video[:1])
        pred_a_decoded = self.vae.decode_audio(pred_audio[:1].transpose(1, 2))
        pred_v_decoded = ((pred_v_decoded + 1) / 2 * 255).clip(0, 255)
        
        pred_v_np = pred_v_decoded.squeeze(0).permute(1, 0, 2, 3).cpu().float().numpy().astype(np.uint8)
        pred_a_np = pred_a_decoded.squeeze(0).cpu().float().numpy().flatten()
        
        save_video(
            output_path=f"{prefix}_pred.mp4",
            video_numpy=pred_v_np,
            audio_numpy=pred_a_np
        )
        
        # Decode and save target
        target_v_decoded = self.vae.decode_video(target_video[:1])
        target_a_decoded = self.vae.decode_audio(target_audio[:1].transpose(1, 2))
        target_v_decoded = ((target_v_decoded + 1) / 2 * 255).clip(0, 255)
        
        target_v_np = target_v_decoded.squeeze(0).permute(1, 0, 2, 3).cpu().float().numpy().astype(np.uint8)
        target_a_np = target_a_decoded.squeeze(0).cpu().float().numpy().flatten()
        
        save_video(
            output_path=f"{prefix}_target.mp4",
            video_numpy=target_v_np,
            audio_numpy=target_a_np
        )
        
        print(f"[Debug] Saved {prefix}_pred.mp4 and {prefix}_target.mp4")


if __name__ == "__main__":
    """
    Unit test for OviSelfForcingRegression
    
    Run with:
        PYTHONPATH=. python model/ovi_self_forcing_regression.py
    """
    import sys
    import os
    
    logging.basicConfig(
        level=logging.INFO,
        format="[%(filename)s] %(levelname)s: %(message)s"
    )

    if not dist.is_initialized():
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12346"
        dist.init_process_group(
            backend="nccl" if torch.cuda.is_available() else "gloo", 
            rank=0, 
            world_size=1
        )

    print("="*80)
    print("Running Unit Test for OviSelfForcingRegression")
    print("="*80)

    class TestArgs:
        def __init__(self):
            self.model_name = "Ovi"
            self.generator_name = "Ovi"
            self.generator_type = "causal"
            self.generator_path = "/cpfs01/gongshukai/step_distillation/logs/ovi_ode_init_1229/checkpoint_model_010000/checkpoint.pt"

            self.model_kwargs = {
                "timestep_shift": 5.0
            }

            self.denoising_step_list = [1000, 750, 500, 250]
            self.timestep_shift = 5.0
            
            self.gradient_checkpointing = True
            self.mixed_precision = True
            self.warp_denoising_step = True
            
            # Self-forcing specific
            self.vid_block_size = 4
            self.aud_block_size = 20
            self.num_training_frames_video = 32
            self.num_training_frames_audio = 160
            self.start_gradient_frame_index_video = 0
            self.context_noise = 0.0
            self.last_step_only = False
            self.same_step_accross_blocks = False

    args = TestArgs()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if args.mixed_precision else torch.float32

    # Load test data
    dataset = OviODERegressionDataset(
        data_path="/cpfs01/gongshukai/step_distillation/data/ode_pairs_overfit"
    )
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True, drop_last=True)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, sampler=sampler, num_workers=0
    )
    dataloader = cycle(dataloader)
    batch = next(dataloader)
    
    video_ode_latent = batch["video_ode_latent"].to(device, dtype)
    audio_ode_latent = batch["audio_ode_latent"].to(device, dtype)
    prompts = batch["prompts"]
    
    print(f"Prompts: {prompts}")
    print(f"video_ode_latent shape: {video_ode_latent.shape}")
    print(f"audio_ode_latent shape: {audio_ode_latent.shape}")
    print(f"Device: {device}, Dtype: {dtype}")

    # Initialize model
    try:
        print("\n[Init] Initializing OviSelfForcingRegression...")
        model = OviSelfForcingRegressionDebug(args, device).to(device)
        print(f"[Init] Model initialized successfully.")
    except Exception as e:
        print(f"\n[Error] Model Initialization Failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Get text embeddings
    with torch.no_grad():
        conditional_dict = model.text_encoder(prompts)

    # Test forward pass
    print("\n[Test] Running Forward Pass & Loss Computation...")
    try:
        with torch.amp.autocast('cuda', dtype=dtype):
            total_loss, log_dict = model.generator_loss(
                video_ode_latent, 
                audio_ode_latent, 
                conditional_dict
            )
            
        print(f"  > Total Loss: {total_loss.item():.6f}")
        print(f"  > Video Loss: {log_dict['loss_video'].item():.6f}")
        print(f"  > Audio Loss: {log_dict['loss_audio'].item():.6f}")
        
        print("  > Backward pass check...")
        total_loss.backward()
        
        grad_found = False
        for name, param in model.generator.named_parameters():
            if param.requires_grad and param.grad is not None:
                grad_found = True
                break
        
        if grad_found:
            print("  ✅ Gradients computed successfully.")
        else:
            print("  ❌ No gradients found! Check requires_grad status.")

    except Exception as e:
        print(f"  ❌ Forward/Backward Failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*80)
    print("Test Completed.")
    print("="*80)