# FILE: model/ovi_ode_regression.py
from typing import Tuple
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

class OviODERegression(OviBaseModel):
    def __init__(self, args, device):
        super().__init__(args, device)
        self.args = args
        self.device = device
        if 0 not in self.denoising_step_list:
            self.denoising_step_list = torch.cat((self.denoising_step_list, torch.tensor([0], dtype=torch.float32)))
            logger.info(f"ODE Pretrain denoising step list: {self.denoising_step_list}") if not dist.is_initialized() or dist.get_rank() == 0 else None
        self.num_frame_per_block_vid = 4
        self.num_frame_per_block_aud = 20
        self.num_blocks = 8
        self.num_aud_frame_per_vid = self.num_frame_per_block_aud // self.num_frame_per_block_vid
        self.inference_pipeline = None

    def _initialize_models(self, args, device):
        self.generator_name = getattr(args, "generator_name", "Ovi")    # the student model
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

    def _get_aligned_timestep_indices(self, B):
        block_indices = torch.randint(
            0, 
            len(self.denoising_step_list), 
            (B, self.num_blocks)
        )
        
        index_video = block_indices.unsqueeze(-1).repeat(1, 1, self.num_frame_per_block_vid).flatten(1, 2)
        index_audio = block_indices.unsqueeze(-1).repeat(1, 1, self.num_frame_per_block_aud).flatten(1, 2)
        return index_video, index_audio
    
    @torch.no_grad()
    def _prepare_generator_input(self, video_ode_latent: torch.Tensor, audio_ode_latent: torch.Tensor):
        B, num_steps, F, C, H, W = video_ode_latent.shape   # [B, 5, 32, 48, H, W]
        _, _, L, D = audio_ode_latent.shape # [B, 5, 160, 20]
        index_video, index_audio = self._get_aligned_timestep_indices(B)    # [B, 32], [B, 160]
        noisy_video_input = torch.gather(
            video_ode_latent, dim=1, index=index_video.reshape(B, 1, F, 1, 1, 1).expand(
                -1, -1, -1, C, H, W).to(self.device)
        ).squeeze(1)

        noisy_audio_input = torch.gather(
            audio_ode_latent, dim=1, index=index_audio.reshape(B, 1, L, 1).expand(
                -1, -1, -1, D).to(self.device)
        ).squeeze(1)
        timestep_v = self.denoising_step_list[index_video].to(self.device)
        timestep_a = self.denoising_step_list[index_audio].to(self.device)
        
        return noisy_video_input, noisy_audio_input, timestep_v, timestep_a
        
    def generator_loss(self, video_ode_latent, audio_ode_latent, conditional_dict):
        """
        video_ode_latent: [B, 5, 32, 48, H, W]
        audio_ode_latent: [B, 5, 160, 20]
        """
        target_video_latent = video_ode_latent[:, -1] # [B, 32, 48, H, W]
        target_audio_latent = audio_ode_latent[:, -1] # [B, 160, 20]
        wan22_image_latent = target_video_latent[:, 0:1, :, :, :] # [B, 1, 48, H, W]
        noisy_video_input, noisy_audio_input, timestep_v, timestep_a =  self._prepare_generator_input(video_ode_latent, audio_ode_latent)
        B = video_ode_latent.shape[0]
        
        cond_expanded = {
            "video_prompt_embeds": conditional_dict["prompt_embeds"],
            "audio_prompt_embeds": conditional_dict["prompt_embeds"]
        }

        mask1, mask2 = masks_like(noisy_video_input, zero=True)
        mask2 = torch.stack(mask2, dim=0)
        noisy_video_input = (1. - mask2) * wan22_image_latent + mask2 * noisy_video_input
        noisy_video_input = noisy_video_input.to(self.device, dtype=self.dtype)

        x0_video, x0_audio, _, _ = self.generator(
            video_latent=noisy_video_input,
            audio_latent=noisy_audio_input,
            timestep_v=timestep_v,
            timestep_a=timestep_a,
            conditional_dict=cond_expanded,
            
            wan22_image_latent=wan22_image_latent,
            mask2=mask2,
            first_frame_is_clean=True,
        )

        mask_pad_v = torch.cat((torch.ones([B, 31]), torch.zeros([B, 1])), dim=1).to(self.device)
        mask_pad_a = torch.cat((torch.ones([B, 157]), torch.zeros([B, 3])), dim=1).to(self.device)
        mask_t_v = (timestep_v != 0)  # [B, 32]
        mask_t_a = (timestep_a != 0)  # [B, 160]

        mask_v = (mask_pad_v * mask_t_v).view(B, 32, 1, 1, 1).expand_as(x0_video).bool()
        mask_a = (mask_pad_a * mask_t_a).view(B, 160, 1).expand_as(x0_audio).bool()

        loss_v = F.mse_loss(x0_video[mask_v], target_video_latent[mask_v], reduction="mean")
        loss_a = F.mse_loss(x0_audio[mask_a], target_audio_latent[mask_a], reduction="mean")
        
        total_loss = 0.85 * loss_v + 0.15 * loss_a
        
        return total_loss, {
            "loss_video": loss_v.detach(),
            "loss_audio": loss_a.detach()
        }
    
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
                num_training_frames_video=32,
                num_training_frames_audio=160,
                start_gradient_frame_index_video=0,
                context_noise=0.0,
                last_step_only=False,
                same_step_accross_blocks=False,
            )

    @torch.no_grad()
    def full_inference(self, noises, wan22_image_latent, **conditional_dict):
        if self.inference_pipeline is None:
            self._initialize_inference_pipeline()
        
        return self.inference_pipeline.full_inference(
            noises=noises,
            wan22_image_latent=wan22_image_latent,
            **conditional_dict
        )
    
class OviODERegressionDebug(OviBaseModel):
    def __init__(self, args, device):
        super().__init__(args, device)
        self.args = args
        self.device = device
        if 0 not in self.denoising_step_list:
            self.denoising_step_list = torch.cat((self.denoising_step_list, torch.tensor([0], dtype=torch.float32)))
            logger.info(f"ODE Pretrain denoising step list: {self.denoising_step_list}") if not dist.is_initialized() or dist.get_rank() == 0 else None
        self.num_frame_per_block_vid = 4
        self.num_frame_per_block_aud = 20
        self.num_blocks = 8
        self.num_aud_frame_per_vid = self.num_frame_per_block_aud // self.num_frame_per_block_vid

    def _initialize_models(self, args, device):
        self.generator_name = getattr(args, "generator_name", "Ovi")    # the student model
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

    def _get_aligned_timestep_indices(self, B):
        block_indices = torch.randint(
            0, 
            len(self.denoising_step_list), 
            (B, self.num_blocks)
        )
        
        index_video = block_indices.unsqueeze(-1).repeat(1, 1, self.num_frame_per_block_vid).flatten(1, 2)
        index_audio = block_indices.unsqueeze(-1).repeat(1, 1, self.num_frame_per_block_aud).flatten(1, 2)
        return index_video, index_audio
    
    @torch.no_grad()
    def _prepare_generator_input(self, video_ode_latent: torch.Tensor, audio_ode_latent: torch.Tensor):
        B, num_steps, F, C, H, W = video_ode_latent.shape   # [B, 5, 32, 48, H, W]
        _, _, L, D = audio_ode_latent.shape # [B, 5, 160, 20]
        index_video, index_audio = self._get_aligned_timestep_indices(B)    # [B, 32], [B, 160]
        noisy_video_input = torch.gather(
            video_ode_latent, dim=1, index=index_video.reshape(B, 1, F, 1, 1, 1).expand(
                -1, -1, -1, C, H, W).to(self.device)
        ).squeeze(1)

        noisy_audio_input = torch.gather(
            audio_ode_latent, dim=1, index=index_audio.reshape(B, 1, L, 1).expand(
                -1, -1, -1, D).to(self.device)
        ).squeeze(1)
        timestep_v = self.denoising_step_list[index_video].to(self.device)
        timestep_a = self.denoising_step_list[index_audio].to(self.device)
        
        return noisy_video_input, noisy_audio_input, timestep_v, timestep_a
        
    def generator_loss(self, video_ode_latent, audio_ode_latent, conditional_dict):
        """
        video_ode_latent: [B, 5, 32, 48, H, W]
        audio_ode_latent: [B, 5, 160, 20]
        """
        target_video_latent = video_ode_latent[:, -1] # [B, 32, 48, H, W]
        target_audio_latent = audio_ode_latent[:, -1] # [B, 160, 20]
        wan22_image_latent = target_video_latent[:, 0:1, :, :, :] # [B, 1, 48, H, W]
        noisy_video_input, noisy_audio_input, timestep_v, timestep_a =  self._prepare_generator_input(video_ode_latent, audio_ode_latent)
        B = video_ode_latent.shape[0]
        
        cond_expanded = {
            "video_prompt_embeds": conditional_dict["prompt_embeds"],
            "audio_prompt_embeds": conditional_dict["prompt_embeds"]
        }

        mask1, mask2 = masks_like(noisy_video_input, zero=True)
        mask2 = torch.stack(mask2, dim=0)
        noisy_video_input = (1. - mask2) * wan22_image_latent + mask2 * noisy_video_input
        noisy_video_input = noisy_video_input.to(self.device, dtype=self.dtype)
        print(f"In OviODERegression, timestep_v shape: {timestep_v.shape}, timestep_v: {timestep_v}")
        print(f"In OviODERegression, timestep_a shape: {timestep_a.shape}, timestep_a: {timestep_a}")
        
        with torch.no_grad():
            print(f"saving noisy input for debugging.")
            noisy_video_lat = noisy_video_input[0, :-1].unsqueeze(0).to(device, dtype)
            noisy_audio_lat = noisy_audio_input[0, :-3].unsqueeze(0).transpose(1, 2).to(device, dtype)

            print(f"noisy shape: {noisy_video_lat.shape, noisy_audio_lat.shape}")

            model.vae.to(device, dtype)
            noisy_video = model.vae.decode_video(noisy_video_lat) # [B, C, F, H, W]
            noisy_audio = model.vae.decode_audio(noisy_audio_lat) # [B, L]
            noisy_video = ((noisy_video + 1) / 2 * 255).clip(0, 255)


            noisy_video_np = noisy_video.squeeze(0).permute(1, 0, 2, 3).cpu().float().numpy().astype(np.uint8)
            noisy_audio_np = noisy_audio.squeeze(0).cpu().float().numpy().flatten()

            save_video(
                output_path="noisy_video.mp4",
                video_numpy=noisy_video_np,
                audio_numpy=noisy_audio_np
            )

        x0_video, x0_audio, _, _ = self.generator(
            video_latent=noisy_video_input,
            audio_latent=noisy_audio_input,
            timestep_v=timestep_v,        # for debugging
            timestep_a=timestep_a,
            conditional_dict=cond_expanded,
            
            wan22_image_latent=wan22_image_latent,
            mask2=mask2,
            first_frame_is_clean=True,
        )

        mask_pad_v = torch.cat((torch.ones([B, 31]), torch.zeros([B, 1])), dim=1).to(self.device)
        mask_pad_a = torch.cat((torch.ones([B, 157]), torch.zeros([B, 3])), dim=1).to(self.device)
        mask_t_v = (timestep_v != 0)  # [B, 32]
        mask_t_a = (timestep_a != 0)  # [B, 160]

        mask_v = (mask_pad_v * mask_t_v).view(B, 32, 1, 1, 1).expand_as(x0_video).bool()
        mask_a = (mask_pad_a * mask_t_a).view(B, 160, 1).expand_as(x0_audio).bool()
        # mask_v_more = torch.cat((torch.zeros([B, 1]), torch.ones([B, 30]), torch.zeros([B, 1])), dim=1).view(B, 32, 1, 1, 1).expand_as(x0_video).bool() 
        with torch.no_grad():
            print(f"saving ground truth and model prediction for debugging.")
            clean_video_lat = target_video_latent[0, :-1].unsqueeze(0).to(device, dtype)
            clean_audio_lat = target_audio_latent[0, :-3].unsqueeze(0).transpose(1, 2).to(device, dtype)

            pred_video_lat = x0_video[0, :-1].unsqueeze(0).to(device, dtype)
            pred_audio_lat = x0_audio[0, :-3].unsqueeze(0).transpose(1, 2).to(device, dtype)

            print(f"clean shape: {clean_video_lat.shape, clean_audio_lat.shape}, pred shape: {pred_video_lat.shape, pred_audio_lat.shape}")

            model.vae.to(device, dtype)
            clean_video = model.vae.decode_video(clean_video_lat) # [B, C, F, H, W]
            clean_audio = model.vae.decode_audio(clean_audio_lat) # [B, L]
            clean_video = ((clean_video + 1) / 2 * 255).clip(0, 255)

            pred_video = model.vae.decode_video(pred_video_lat) # [B, C, F, H, W]
            pred_audio = model.vae.decode_audio(pred_audio_lat) # [B, L]
            pred_video = ((pred_video + 1) / 2 * 255).clip(0, 255)

            video_np = clean_video.squeeze(0).permute(1, 0, 2, 3).cpu().float().numpy().astype(np.uint8)
            audio_np = clean_audio.squeeze(0).cpu().float().numpy().flatten()

            pred_video_np = pred_video.squeeze(0).permute(1, 0, 2, 3).cpu().float().numpy().astype(np.uint8)
            pred_audio_np = pred_audio.squeeze(0).cpu().float().numpy().flatten()

            save_video(
                output_path="clean_video.mp4",
                video_numpy=video_np,
                audio_numpy=audio_np
            )

            save_video(
                output_path="pred_video.mp4",
                video_numpy=pred_video_np,
                audio_numpy=pred_audio_np
            )
        # print(f"{x0_video[mask_v].shape=}, {x0_audio[mask_a].shape=}")
        # x0_video[mask_v].shape: torch.Size([10475520]) = 2 * 31 * 48 * 44 * 80
        # x0_audio[mask_a].shape: torch.Size([6280]) = 2 * 157 * 20
        loss_v = F.mse_loss(x0_video[mask_v], target_video_latent[mask_v], reduction="mean")
        loss_a = F.mse_loss(x0_audio[mask_a], target_audio_latent[mask_a], reduction="mean")
        
        # loss_v_less = F.mse_loss(x0_video[mask_pad_v.view(B, 32, 1, 1, 1).expand_as(x0_video).bool()], target_video_latent[mask_pad_v.view(B, 32, 1, 1, 1).expand_as(x0_video).bool()], reduction="mean")
        # loss_a_less = F.mse_loss(x0_audio[mask_pad_a.view(B, 160, 1).expand_as(x0_audio).bool()], target_audio_latent[mask_pad_a.view(B, 160, 1).expand_as(x0_audio).bool()], reduction="mean")
        # print(f"Loss video without eliminating timestep=0: {loss_v_less=}, {loss_a_less=}")
        total_loss = 0.85 * loss_v + 0.15 * loss_a
        
        return total_loss, {
            "loss_video": loss_v.detach(),
            "loss_audio": loss_a.detach()
        }

if __name__ == "__main__":
    """
    PYTHONPATH=. python model/ovi_ode_regression.py 
    """
    import sys
    import torch.distributed as dist
    import os
    logging.basicConfig(
        level=logging.INFO,
        format="[%(filename)s] %(levelname)s: %(message)s"
    )

    if not dist.is_initialized():
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12345"
        dist.init_process_group(
            backend="nccl" if torch.cuda.is_available() else "gloo", 
            rank=0, 
            world_size=1
        )

    print("="*80)
    print("Running Unit Test for OviODERegression")
    print("="*80)

    class RealArgs:
        def __init__(self):
            self.model_name = "Ovi"
            self.generator_name = "Ovi"
            self.generator_type = "causal"
            self.generator_path = "/cpfs01/gongshukai/step_distillation/logs/ovi_ode_init/checkpoint_model_005000/model.pt"

            self.model_kwargs = {
                "timestep_shift": 5.0
            }

            self.denoising_step_list = [1000, 750, 500, 250]
            self.timestep_shift = 5.0
            
            self.gradient_checkpointing = True
            self.mixed_precision = True
            self.warp_denoising_step = True

    args = RealArgs()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if args.mixed_precision else torch.float32

    
    dataset = OviODERegressionDataset(data_path="/cpfs01/gongshukai/step_distillation/data/ode_pairs_debug")
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True, drop_last=True)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=2, sampler=sampler, num_workers=4
    )
    dataloader = cycle(dataloader)
    batch = next(dataloader)
    video_ode_latent = batch["video_ode_latent"].to(device, dtype)
    audio_ode_latent = batch["audio_ode_latent"].to(device, dtype)
    prompts = batch["prompts"]
    print(prompts)

    print(f"video_ode_latent shape: {video_ode_latent.shape}, audio_ode_latent shape: {audio_ode_latent.shape}")
    print(f"Device: {device}, Dtype: {dtype}")

    try:
        print("\n[Init] Initializing OviODERegression...")
        model = OviODERegressionDebug(args, device).to(device)
        print(f"[Init] Model initialized successfully.")
    except Exception as e:
        print(f"\n[Error] Model Initialization Failed: {e}")
        sys.exit(1)

    # Video: [B, 5, 32, 48, H, W]
    # Audio: [B, 5, 160, 20]
    B = 2
    H_latent, W_latent = 44, 80
    with torch.no_grad():
        conditional_dict = model.text_encoder(prompts)
    
    # # Examine clean latent
    # with torch.no_grad():
    #     clean_video_lat = video_ode_latent[0, -1, :31, :, :, :].unsqueeze(0).to(device, dtype)
    #     clean_audio_lat = audio_ode_latent[0, -1, :157, :].unsqueeze(0).transpose(1, 2).to(device, dtype)
    #     noisy_video_lat = video_ode_latent[0, -2, :31, :, :, :].unsqueeze(0).to(device, dtype)
    #     noisy_audio_lat = audio_ode_latent[0, -2, :157, :].unsqueeze(0).transpose(1, 2).to(device, dtype)

    #     print(clean_video_lat.shape, clean_audio_lat.shape)
    #     model.vae.to(device, dtype)
    #     out_video = model.vae.decode_video(clean_video_lat) # [B, C, F, H, W]
    #     out_audio = model.vae.decode_audio(clean_audio_lat) # [B, L]
    #     out_video = ((out_video + 1) / 2 * 255).clip(0, 255)

    #     out_noisy_video = model.vae.decode_video(noisy_video_lat) # [B, C, F, H, W]
    #     out_noisy_audio = model.vae.decode_audio(noisy_audio_lat) # [B, L]
    #     out_noisy_video = ((out_noisy_video + 1) / 2 * 255).clip(0, 255)

    #     video_np = out_video.squeeze(0).permute(1, 0, 2, 3).cpu().float().numpy().astype(np.uint8)
    #     audio_np = out_audio.squeeze(0).cpu().float().numpy().flatten()

    #     video_noisy_np = out_noisy_video.squeeze(0).permute(1, 0, 2, 3).cpu().float().numpy().astype(np.uint8)
    #     audio_noisy_np = out_noisy_audio.squeeze(0).cpu().float().numpy().flatten()

    #     save_video(
    #         output_path="example_video.mp4",
    #         video_numpy=video_np,
    #         audio_numpy=audio_np
    #     )

    #     save_video(
    #         output_path="example_video_noisy.mp4",
    #         video_numpy=video_noisy_np,
    #         audio_numpy=audio_noisy_np
    #     )


    print("\n[Test 1] Verifying Block Alignment & Timestep Gathering...")
    try:
        vid_in, aud_in, t_v, t_a = model._prepare_generator_input(video_ode_latent, audio_ode_latent)
        
        print(f"  > Video Input Shape: {vid_in.shape} (Expected: {B, 32, 48, H_latent, W_latent})")
        print(f"  > Audio Input Shape: {aud_in.shape} (Expected: {B, 160, 20})")
        print(f"  > Timestep Video Shape: {t_v.shape} (Expected: {B, 32})")
        print(f"  > Timestep Audio Shape: {t_a.shape} (Expected: {B, 160})")
        
        # Block alignment
        # Block 0 Video: frames 0-3
        # Block 0 Audio: tokens 0-19
        t_v_b0 = t_v[0, 0:4]
        t_a_b0 = t_a[0, 0:20]
        t_v_b1 = t_v[0, 4:8]
        t_a_b1 = t_a[0, 20:40]
        t_v_b2 = t_v[0, 8:12]
        t_a_b2 = t_a[0, 40:60]
        
        print(f"  > Block 0 Video Timesteps: {t_v_b0.tolist()}")
        print(f"  > Block 0 Audio Timesteps: {t_a_b0.tolist()}")
        print(f"  > Block 1 Video Timesteps: {t_v_b1.tolist()}") 
        print(f"  > Block 1 Audio Timesteps: {t_a_b1.tolist()}") 
        print(f"  > Block 2 Video Timesteps: {t_v_b2.tolist()}") 
        print(f"  > Block 2 Audio Timesteps: {t_a_b2.tolist()}") 

        
        assert len(torch.unique(t_v_b0)) == 1, "Video frames in Block 0 have different timesteps!"
        assert len(torch.unique(t_a_b0)) == 1, "Audio tokens in Block 0 have different timesteps!"
        
        assert t_v_b0[0].item() == t_a_b0[0].item(), "Video and Audio Block 0 are NOT aligned!"
        
        print("  ✅ Alignment Check Passed.")
        
    except Exception as e:
        print(f"  ❌ Alignment Check Failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n[Test 2] Running Forward Pass & Loss Computation...")
    
    try:
        with torch.amp.autocast('cuda', dtype=dtype):
            total_loss, log_dict = model.generator_loss(
                video_ode_latent, 
                audio_ode_latent, 
                conditional_dict
            )
            
        print(f"  > Total Loss: {total_loss.item()}")
        print(f"  > Video Loss: {log_dict['loss_video'].item()}")
        print(f"  > Audio Loss: {log_dict['loss_audio'].item()}")
        print("  > Backward pass check...")
        total_loss.backward()
        
        grad_found = False
        for name, param in model.generator.named_parameters():
            if param.requires_grad and param.grad is not None:
                grad_found = True
                # print(f"    Gradient found in: {name}")
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