# FILE: ovi_fewstep_inference.py
from typing import List, Tuple
import torch
from tqdm import tqdm
from utils.dataset import masks_like
from utils.ovi_wrapper_inference import OviFusionWrapper, OviTextEncoder, OviVAEWrapper
from ovi.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
import logging
logger = logging.getLogger(__name__)

class OviFewstepInferencePipeline(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.timestep_shift = getattr(args, "timestep_shift", 1.0)

        logger.info("Initializing Ovi models...")
        self.generator = OviFusionWrapper(
            model_name=args.model_name,
            timestep_shift=self.timestep_shift,
            **getattr(args, "model_kwargs", {})
        )
        logger.info(f"Generator model loaded: {args.model_name}")
        self.text_encoder = OviTextEncoder()
        logger.info("Text encoder model loaded")
        self.vae = OviVAEWrapper()
        logger.info("VAE model loaded")
        logger.info("Models initialized.")

        logger.info("Setting up denoising steps...")
        self.scheduler = self.generator.get_scheduler()
        if getattr(args, "inference_steps", None):
            # 模式一：为原始模型设置推理采样器
            self.inference_mode = "original_cfg"
            logger.info(f"Initialized for ORIGINAL model inference with {args.inference_steps} steps.")
        elif getattr(args, "denoising_step_list", None):
            # 模式二：为蒸馏模型设置采样器
            self.inference_mode = "distilled"
            self.scheduler = self.generator.get_scheduler() # Use the FlowMatchScheduler
            self.denoising_step_list = torch.tensor(args.denoising_step_list, dtype=torch.long)
            if args.warp_denoising_step:
                timesteps = torch.cat((self.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32)))
                self.denoising_step_list = timesteps[1000 - self.denoising_step_list]
            logger.info(f"Initialized for DISTILLED model inference with fixed steps: {self.denoising_step_list}")
        else:
            raise ValueError("Config must provide either 'inference_steps' (for original model) or 'denoising_step_list' (for distilled model)")

        
    
    @torch.inference_mode()
    def inference(
        self,
        noise_video: torch.Tensor,
        noise_audio: torch.Tensor,
        text_prompts: List[str],
        wan22_image_latent: torch.Tensor,
        # +++ START NEW PARAMS +++
        video_guidance_scale: float = 4.0,
        audio_guidance_scale: float = 3.0,
        video_negative_prompt: str = "",
        audio_negative_prompt: str = ""
        # +++ END NEW PARAMS +++
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        if self.inference_mode == "original_cfg":
            return self._inference_original_cfg(
                noise_video, noise_audio, text_prompts, wan22_image_latent,
                video_guidance_scale, audio_guidance_scale,
                video_negative_prompt, audio_negative_prompt
            )
        else:
            return self._inference_distill(
                noise_video, noise_audio, text_prompts, wan22_image_latent
            )
        
    def _inference_original_cfg(
        self,
        noise_video: torch.Tensor,
        noise_audio: torch.Tensor,
        text_prompts: List[str],
        wan22_image_latent: torch.Tensor,
        video_guidance_scale: float,
        audio_guidance_scale: float,
        video_negative_prompt: str,
        audio_negative_prompt: str,
        slg_layer: int = 11,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        device = noise_video.device
        dtype = noise_video.dtype
        batch_size = len(text_prompts)

        scheduler_video = FlowUniPCMultistepScheduler(
            num_train_timesteps=1000,
            shift=1,
            use_dynamic_shifting=False
        )
        scheduler_audio = FlowUniPCMultistepScheduler(
            num_train_timesteps=1000,
            shift=1,
            use_dynamic_shifting=False
        )
        scheduler_video.set_timesteps(self.args.inference_steps, device='cpu', shift=self.timestep_shift)
        scheduler_audio.set_timesteps(self.args.inference_steps, device='cpu', shift=self.timestep_shift)

        timesteps_v = scheduler_video.timesteps
        timesteps_a = scheduler_audio.timesteps
        
        prompts_for_encoding = text_prompts + [video_negative_prompt] * batch_size + [audio_negative_prompt] * batch_size
        all_embeds = self.text_encoder(text_prompts=prompts_for_encoding)["prompt_embeds"]
        
        pos_embeds = all_embeds[:batch_size]
        vid_neg_embeds = all_embeds[batch_size:batch_size*2]
        aud_neg_embeds = all_embeds[batch_size*2:]

        # --- Prepare latents and image mask ---
        noisy_video = noise_video
        noisy_audio = noise_audio
        mask1, mask2 = masks_like(noisy_video, zero=True)
        mask2 = torch.stack(mask2, dim=0)
        
        # Initial image latent injection
        noisy_video = (1. - mask2) * wan22_image_latent + mask2 * noisy_video
        noisy_video = noisy_video.to(device, dtype=dtype)

        progress_bar = tqdm(zip(timesteps_v, timesteps_a), total=len(timesteps_v), desc="Denoising Steps (CFG Mode)")
        
        for t_v, t_a in progress_bar:
            timestep = torch.full((batch_size,), t_v.item(), device=device, dtype=torch.long)
            
            # --- CFG Step 2: Double forward pass ---
            # Unconditional Pass (Negative Prompts)
            _, _, pred_vid_neg, pred_aud_neg = self.generator(
                video_latent=noisy_video,
                audio_latent=noisy_audio,
                timestep=timestep,
                conditional_dict={"video_prompt_embeds": vid_neg_embeds, "audio_prompt_embeds": aud_neg_embeds},
                wan22_image_latent=wan22_image_latent,
                mask2=mask2,
                first_frame_is_clean=True,
                slg_layer=slg_layer
            )

            # Conditional Pass (Positive Prompts)
            _, _, pred_vid_pos, pred_aud_pos = self.generator(
                video_latent=noisy_video,
                audio_latent=noisy_audio,
                timestep=timestep,
                conditional_dict={"video_prompt_embeds": pos_embeds, "audio_prompt_embeds": pos_embeds},
                wan22_image_latent=wan22_image_latent,
                mask2=mask2,
                first_frame_is_clean=True,
            )

            # --- CFG Step 3: Combine predictions ---
            pred_video_guided = pred_vid_neg + video_guidance_scale * (pred_vid_pos - pred_vid_neg)
            pred_audio_guided = pred_aud_neg + audio_guidance_scale * (pred_aud_pos - pred_aud_neg)

            # --- Scheduler Step: Denoise for one step ---
            # NOTE: The scheduler's step function expects a batch dimension.
            # shape of pred_video_guided: (B, F, C, H, W), pred_audio_guided: (B, L, D)
            # shape of noisy_video: (B, F, C, H, W), noisy_audio: (B, L, D)
            pred_video_guided = pred_video_guided.permute(0, 2, 1, 3, 4)  # (B, C, F, H, W)
            noisy_video = noisy_video.permute(0, 2, 1, 3, 4)  # (B, C, F, H, W)

            noisy_video = scheduler_video.step(pred_video_guided, t_v, noisy_video, return_dict=False)[0]
            noisy_audio = scheduler_audio.step(pred_audio_guided, t_a, noisy_audio, return_dict=False)[0]

            noisy_video = noisy_video.permute(0, 2, 1, 3, 4)  # Back to (B, F, C, H, W)
            # Re-apply the image latent at each step
            noisy_video = (1. - mask2) * wan22_image_latent + mask2 * noisy_video
            noisy_video = noisy_video.to(device, dtype=dtype)

        # Final prediction is the last state of noisy_video/audio, which should be clean
        pred_video_x0 = noisy_video
        pred_audio_x0 = noisy_audio
        
        # --- Decode ---
        pred_audio_x0 = pred_audio_x0.transpose(1, 2)
        video = self.vae.decode_video(pred_video_x0)
        audio = self.vae.decode_audio(pred_audio_x0)
        video = (video * 0.5 + 0.5).clamp(0, 1)
        return video, audio

    def _inference_distill(
        self,
        noise_video: torch.Tensor,
        noise_audio: torch.Tensor,
        text_prompts: List[str],
        wan22_image_latent: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = noise_video.device
        dtype = noise_video.dtype
        conditional_dict = self.text_encoder(text_prompts=text_prompts)
        noisy_video = noise_video
        noisy_audio = noise_audio
        mask1, mask2 = masks_like(noisy_video, zero=True)
        mask2 = torch.stack(mask2, dim=0)
        noisy_video = (1. - mask2) * wan22_image_latent + mask2 * noisy_video
        noisy_video = noisy_video.to(device, dtype=dtype)

        progress_bar = tqdm(
            enumerate(self.denoising_step_list),
            total=len(self.denoising_step_list),
            desc="Denoising Steps",
        )

        for index, current_timestep in progress_bar:
            timestep = torch.full((noise_video.shape[0],), current_timestep.item(), device=device, dtype=torch.long)
            # logger.info(f"Current timestep: {timestep[0].item()} at step {index+1}/{len(self.denoising_step_list)}")
            pred_video_x0, pred_audio_x0, _, _ = self.generator(
                video_latent=noisy_video,
                audio_latent=noisy_audio,
                timestep=timestep,
                conditional_dict=conditional_dict,
                wan22_image_latent=wan22_image_latent,
                mask2=mask2,
                first_frame_is_clean=True,
            )

            if index < len(self.denoising_step_list) - 1:
                next_timestep_val = self.denoising_step_list[index + 1]
                next_ts_video = torch.full((pred_video_x0.shape[0], pred_video_x0.shape[1]), next_timestep_val.item(), dtype=torch.long, device=device)
                next_ts_audio = torch.full((pred_audio_x0.shape[0], pred_audio_x0.shape[1]), next_timestep_val.item(), dtype=torch.long, device=device)
                # logger.info(f"Next timestep for video: {next_ts_video[0,0].item()}, Next timestep for audio: {next_ts_audio[0,0].item()}")
                noisy_video = self.scheduler.add_noise(
                    pred_video_x0.flatten(0,1), torch.randn_like(pred_video_x0.flatten(0,1)), next_ts_video.flatten(0,1)
                ).unflatten(0, pred_video_x0.shape[:2])
                noisy_audio = self.scheduler.add_noise(
                    pred_audio_x0.flatten(0,1), torch.randn_like(pred_audio_x0.flatten(0,1)), next_ts_audio.flatten(0,1)
                ).unflatten(0, pred_audio_x0.shape[:2])
                noisy_video = (1. - mask2) * wan22_image_latent + mask2 * noisy_video
                noisy_video = noisy_video.to(device, dtype=dtype)

        pred_audio_x0 = pred_audio_x0.transpose(1, 2) # (1, L, D) -> (1, D, L)
        video = self.vae.decode_video(pred_video_x0)
        audio = self.vae.decode_audio(pred_audio_x0)
        video = (video * 0.5 + 0.5).clamp(0, 1)
        return video, audio