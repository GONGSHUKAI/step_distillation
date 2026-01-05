# FILE: pipeline/ovi_causal_inference.py

import math
import torch
import torch.nn as nn
from typing import List, Optional, Tuple
from tqdm import tqdm
import numpy as np
from utils.ovi_wrapper import OviFusionWrapper, OviTextEncoder, OviVAEWrapper
from utils.dataset import masks_like
import logging
import torch.distributed as dist
from ovi.utils.io_utils import save_video

logger = logging.getLogger(__name__)

class OviCausalInferencePipeline(torch.nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self.args = args
        self.device = device

        self.generator = OviFusionWrapper(
            **getattr(args, "model_kwargs", {}), 
            is_causal=True
        )
        self.text_encoder = OviTextEncoder()
        self.vae = OviVAEWrapper()
        
        self.scheduler = self.generator.get_scheduler()
        self.denoising_step_list = torch.tensor(args.denoising_step_list, dtype=torch.long)
        if args.warp_denoising_step:
            timesteps = torch.cat((self.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32)))
            self.denoising_step_list = timesteps[1000 - self.denoising_step_list]
            logger.info(f"Warped denoising step list: {self.denoising_step_list}")
        
        # Ovi specific model architecture
        self.num_layers = self.generator.model.num_blocks   # 30 layers for Ovi / Wan2.2 5B
        self.num_blocks = 8 # 8 blocks
        self.vid_block_size = 4 # 4 video frames per video block
        self.aud_block_size = 20 # 20 audio frames per audio block
        
        self.tokens_per_vid_frame = 880     # 44*80/2/2
        self.tokens_per_aud_frame = 1
        self.kv_cache_list = None

        logger.info(f"Ovi Causal Inference initialized with {self.num_blocks} video and audio blocks.")

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


    @torch.inference_mode()
    def inference(
        self,
        noise_video: torch.Tensor, # [B, 32, 48, H, W]
        noise_audio: torch.Tensor, # [B, 160, 20]
        text_prompts: List[str],
        wan22_image_latent: torch.Tensor, # [B, 1, 48, H, W]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        B, F_total, C_v, H_v, W_v = noise_video.shape
        _, L_total, C_a = noise_audio.shape
        dtype = noise_video.dtype
        
        cond_dict = self.text_encoder(text_prompts=text_prompts)
        cond_expanded = {
            "video_prompt_embeds": cond_dict["prompt_embeds"],
            "audio_prompt_embeds": cond_dict["prompt_embeds"],
        }
        
        self._initialize_kv_cache_list(B, H_v, W_v, dtype)
        
        out_video_latents = torch.zeros_like(noise_video)
        out_audio_latents = torch.zeros_like(noise_audio)
        
        # mask2 is used for first frame injection, first block has shape: [B, 4, 48, H, W]
        _, mask2 = masks_like(noise_video[:, :self.vid_block_size], zero=True)
        mask2 = torch.stack(mask2, dim=0).to(self.device, dtype=dtype) # [B, 4, 48, H, W]

        current_start_vid = 0
        current_start_aud = 0

        # 4. Blockwise inference loop
        for block_idx in tqdm(range(self.num_blocks), total=len(range(self.num_blocks)), desc="Blockwise generation"):
            logger.info(f"Processing Block {block_idx+1}/{self.num_blocks}...")
            
            curr_noise_v = noise_video[:, block_idx * self.vid_block_size : (block_idx + 1) * self.vid_block_size]
            curr_noise_a = noise_audio[:, block_idx * self.aud_block_size : (block_idx + 1) * self.aud_block_size]
            
            # Step 4.1: Denoising loop within each block
            curr_latent_v = curr_noise_v    # shape: [B, F_block=4, 48, H, W]
            curr_latent_a = curr_noise_a    # shape: [B, L_block=20, 20]
            
            for i, t_val in tqdm(enumerate(self.denoising_step_list), total=len(self.denoising_step_list), desc="Denoising Steps"):
                t_v = torch.full((B, self.vid_block_size), t_val.item(), device=self.device)    # [B, F_block] = [B, 4]
                t_a = torch.full((B, self.aud_block_size), t_val.item(), device=self.device)    # [B, L_block] = [B, 20]
                
                is_first_block = (block_idx == 0)
                if is_first_block:
                    curr_latent_v = (1. - mask2) * wan22_image_latent + mask2 * curr_latent_v   # substitute latent first frame for latent ref frame
                    curr_latent_v = curr_latent_v.to(dtype)
                # Inference with kv cache
                x0_v, x0_a, _, _ = self.generator(
                    video_latent=curr_latent_v,
                    audio_latent=curr_latent_a,
                    timestep_v=t_v,
                    timestep_a=t_a,
                    conditional_dict=cond_expanded,
                    kv_cache_list=self.kv_cache_list,
                    current_start_vid=current_start_vid,
                    current_start_audio=current_start_aud,
                    wan22_image_latent=wan22_image_latent if is_first_block else None,
                    mask2=mask2 if is_first_block else None,
                    first_frame_is_clean=is_first_block
                )
                
                if i < len(self.denoising_step_list) - 1:
                    next_t = self.denoising_step_list[i+1]
                    next_ts_video = torch.full((x0_v.shape[0], x0_v.shape[1]), next_t.item(), device=self.device, dtype=torch.long)
                    next_ts_audio = torch.full((x0_a.shape[0], x0_a.shape[1]), next_t.item(), device=self.device, dtype=torch.long)
                    curr_latent_v = self.scheduler.add_noise(
                        x0_v.flatten(0, 1), 
                        torch.randn_like(x0_v.flatten(0, 1)), 
                        next_ts_video.flatten(0, 1)
                    ).unflatten(0, x0_v.shape[:2])
                    
                    curr_latent_a = self.scheduler.add_noise(
                        x0_a.flatten(0, 1), 
                        torch.randn_like(x0_a.flatten(0, 1)), 
                        next_ts_audio.flatten(0, 1)
                    ).unflatten(0, x0_a.shape[:2])
            
            # Step 4.2: Fill in x0 for this block into the final output
            out_video_latents[:, block_idx * self.vid_block_size : (block_idx + 1) * self.vid_block_size] = x0_v
            out_audio_latents[:, block_idx * self.aud_block_size : (block_idx + 1) * self.aud_block_size] = x0_a
            
            # self.save_video_debug(x0_v, x0_a, block_idx)

            # Step 4.3: Update KV Cache with clean latent (i.e. latent denoised in the final timestep).
            with torch.no_grad():
                t_zero_v = torch.zeros((B, self.vid_block_size), device=self.device)
                t_zero_a = torch.zeros((B, self.aud_block_size), device=self.device)

                if is_first_block:
                    x0_v = (1. - mask2) * wan22_image_latent + mask2 * x0_v
                    x0_v = x0_v.to(dtype)

                self.generator(
                    video_latent=x0_v,
                    audio_latent=x0_a,
                    timestep_v=t_zero_v,
                    timestep_a=t_zero_a,
                    conditional_dict=cond_expanded,
                    kv_cache_list=self.kv_cache_list,
                    current_start_vid=current_start_vid,
                    current_start_audio=current_start_aud,

                    wan22_image_latent=wan22_image_latent if is_first_block else None,
                    mask2=mask2 if is_first_block else None,
                    first_frame_is_clean=is_first_block
                )
            
            # Update counter
            current_start_vid += (self.vid_block_size * self.tokens_per_vid_frame)
            current_start_aud += (self.aud_block_size * self.tokens_per_aud_frame)

        # 5. Denoised output: crop back to 31 video frames and 157 audio frames to match original latent size of Ovi
        final_v_lat = out_video_latents[:, :31]
        final_a_lat = out_audio_latents[:, :157].transpose(1, 2) # [B, D, L] for MMAudio VAE
        
        video = self.vae.decode_video(final_v_lat)
        audio = self.vae.decode_audio(final_a_lat)
        
        video = (video * 0.5 + 0.5).clamp(0, 1)
        return video, audio
    

    def save_video_debug(self, pred_video, pred_audio, block_idx):
        with torch.no_grad():
            print(f"saving model prediction for debugging.")
            print(f"pred shape: {pred_video.shape, pred_audio.shape}")
            # pred_video = pred_video[:, :31]
            # pred_audio = pred_audio[:, :157]
            pred_video = self.vae.decode_video(pred_video)
            pred_audio = self.vae.decode_audio(pred_audio.transpose(1, 2))
            pred_video = ((pred_video + 1) / 2 * 255).clip(0, 255)
            pred_video_np = pred_video.squeeze(0).permute(1, 0, 2, 3).cpu().float().numpy().astype(np.uint8)
            pred_audio_np = pred_audio.squeeze(0).cpu().float().numpy().flatten()
            print(f"decoded shape: {pred_video_np.shape}, {pred_audio_np.shape}")
            save_video(
                output_path=f"pred_video_block_idx_{block_idx}.mp4",
                video_numpy=pred_video_np,
                audio_numpy=pred_audio_np
            )