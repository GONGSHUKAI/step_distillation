import os
import json
import types
import torch
import torch.nn as nn
import torch.distributed as dist
from typing import List, Dict, Tuple, Optional, Union
import logging
import random

# from ovi.modules.fusion import FusionModel
from ovi.modules.ovi import FusionModel
from ovi.modules.causal_ovi import CausalFusionModel
from ovi.modules.t5 import umt5_xxl
from wan22.modules.vae2_2 import _video_vae as _video_vae_2_2
from ovi.modules.mmaudio.features_utils import FeaturesUtils
from ovi.modules.tokenizers import HuggingfaceTokenizer

from utils.scheduler import SchedulerInterface, FlowMatchScheduler
# from utils.sde_util import sde_step_with_logprob
from utils.dataset import masks_like
from safetensors.torch import load_file
import math
from tqdm import tqdm

logger = logging.getLogger(__name__)

class OviTextEncoder(torch.nn.Module):
    def __init__(self, model_name: str = "Wan2.2-TI2V-5B") -> None:
        super().__init__()
        self.model_name = model_name
        self.is_main_process = not dist.is_initialized() or dist.get_rank() == 0
        logger.info("Initializing Ovi Text Encoder...") if self.is_main_process else None
        self.text_encoder = umt5_xxl(
            encoder_only=True,
            return_tokenizer=False,
            dtype=torch.bfloat16,   # dtype=torch.float32
            device=torch.device('cpu')
        ).eval().requires_grad_(False)

        logger.info("Ovi Text Encoder initialized, loading model weights...") if self.is_main_process else None
        self.text_encoder.load_state_dict(
            torch.load(f"/cpfs01/gongshukai/weights/Ovi/{self.model_name}/models_t5_umt5-xxl-enc-bf16.pth",
                       map_location='cpu', weights_only=False)
        )

        self.tokenizer = HuggingfaceTokenizer(
            name=f"/cpfs01/gongshukai/weights/Ovi/{self.model_name}/google/umt5-xxl", 
            seq_len=512, 
            clean='whitespace'
        )
        logger.info(f"Ovi Text Encoder weights and tokenizer loaded.") if self.is_main_process else None
    
    @property
    def device(self):
        return torch.cuda.current_device()
    
    def forward(self, text_prompts: Union[str, List[str]]) -> dict:
        if isinstance(text_prompts, str):
            text_prompts = [text_prompts]
        
        ids, mask = self.tokenizer(
            text_prompts, return_mask=True, add_special_tokens=True)
        ids = ids.to(self.device)
        mask = mask.to(self.device)
        seq_lens = mask.gt(0).sum(dim=1).long()
        context = self.text_encoder(ids, mask)

        for u, v in zip(context, seq_lens):
            u[v:] = 0.0  # set padding to 0.0

        return {
            "prompt_embeds": context
        }


class OviVAEWrapper(torch.nn.Module):
    def __init__(
        self,
        z_dim_video: int = 48,
        c_dim_video: int = 160,
        video_vae_pth: str = "/cpfs01/gongshukai/weights/Ovi/Wan2.2-TI2V-5B/Wan2.2_VAE.pth",
        dim_mult: List[int] = [1, 2, 4, 4],
        temperal_downsample: List[bool] = [False, True, True],
        
        audio_mode: str = '16k',
        audio_tod_vae_ckpt: str = "/cpfs01/gongshukai/weights/Ovi/MMAudio/ext_weights/v1-16.pth",
        audio_bigvgan_ckpt: str = "/cpfs01/gongshukai/weights/Ovi/MMAudio/ext_weights/best_netG.pt",
    ):
        super().__init__()
        self.is_main_process = not dist.is_initialized() or dist.get_rank() == 0
        # ===== 视频VAE (Wan2.2) =====
        # 初始化时在CPU上 (与Wan2_2_VAEWrapper一致)
        self.mean = torch.tensor([
            -0.2289, -0.0052, -0.1323, -0.2339, -0.2799,  0.0174,  0.1838,  0.1557,
            -0.1382,  0.0542,  0.2813,  0.0891,  0.1570, -0.0098,  0.0375, -0.1825,
            -0.2246, -0.1207, -0.0698,  0.5109,  0.2665, -0.2108, -0.2158,  0.2502,
            -0.2055, -0.0322,  0.1109,  0.1567, -0.0729,  0.0899, -0.2799, -0.1230,
            -0.0313, -0.1649,  0.0117,  0.0723, -0.2839, -0.2083, -0.0520,  0.3748,
            0.0152,  0.1957,  0.1433, -0.2944,  0.3573, -0.0548, -0.1681, -0.0667
        ], dtype=torch.float32)

        self.std = torch.tensor([
            0.4765, 1.0364, 0.4514, 1.1677, 0.5313, 0.4990, 0.4818, 0.5013,
            0.8158, 1.0344, 0.5894, 1.0901, 0.6885, 0.6165, 0.8454, 0.4978,
            0.5759, 0.3523, 0.7135, 0.6804, 0.5833, 1.4146, 0.8986, 0.5659,
            0.7069, 0.5338, 0.4889, 0.4917, 0.4069, 0.4999, 0.6866, 0.4093,
            0.5709, 0.6065, 0.6415, 0.4944, 0.5726, 1.2042, 0.5458, 1.6887,
            0.3971, 1.0600, 0.3943, 0.5537, 0.5444, 0.4089, 0.7468, 0.7744
        ], dtype=torch.float32)

        self.video_dtype = torch.bfloat16
        
        logger.info("Initializing Wan2.2-VAE...") if self.is_main_process else None
        self.video_vae = (
            _video_vae_2_2(
                pretrained_path=video_vae_pth,
                z_dim=z_dim_video,
                dim=c_dim_video,
                dim_mult=dim_mult,
                temperal_downsample=temperal_downsample,
            )
            .eval()
            .requires_grad_(False)
        )
        logger.info(f"Loaded Wan2.2-VAE weights from {video_vae_pth}") if self.is_main_process else None
        
        # ===== 音频VAE (MMAudio) =====
        # 初始化时在CPU上
        logger.info("Initializing MMAudio VAE and Vocoder...") if self.is_main_process else None
        self.audio_vae = FeaturesUtils(
            mode=audio_mode,
            need_vae_encoder=True,
            tod_vae_ckpt=audio_tod_vae_ckpt,
            bigvgan_vocoder_ckpt=audio_bigvgan_ckpt,
        )
        self.audio_vae.eval().requires_grad_(False)
        logger.info(f"Loaded MMAudio VAE weights from {audio_tod_vae_ckpt} and Vocoder weights from {audio_bigvgan_ckpt}") if self.is_main_process else None
    
    # ===== 视频VAE接口 =====
    def encode(self, pixel):
        device, dtype = pixel[0].device, self.video_dtype
        scale = [self.mean.to(device=device, dtype=dtype),
                 1.0 / self.std.to(device=device, dtype=dtype)]
        output = [
            self.video_vae.encode(u.to(self.dtype).unsqueeze(0), scale).float().squeeze(0)
            for u in pixel
        ]
        return output
    
    def encode_video(self, pixel: torch.Tensor) -> torch.Tensor:
        # pixel: [batch_size, num_channels, num_frames, height, width]
        device, dtype = pixel.device, pixel.dtype
        scale = [self.mean.to(device=device, dtype=dtype),
                 1.0 / self.std.to(device=device, dtype=dtype)]

        output = [
            self.video_vae.encode(u.unsqueeze(0), scale).float().squeeze(0)
            for u in pixel
        ]
        output = torch.stack(output, dim=0)
        # from [batch_size, num_channels, num_frames, height, width]
        # to [batch_size, num_frames, num_channels, height, width]
        output = output.permute(0, 2, 1, 3, 4)
        return output

    def decode_video(self, latent: torch.Tensor, use_cache: bool = False) -> torch.Tensor:
        # from [batch_size, num_frames, num_channels, height, width]
        # to [batch_size, num_channels, num_frames, height, width]
        zs = latent.permute(0, 2, 1, 3, 4)
        if use_cache:
            assert latent.shape[0] == 1, "Batch size must be 1 when using cache"

        device, dtype = latent.device, latent.dtype
        scale = [self.mean.to(device=device, dtype=dtype),
                 1.0 / self.std.to(device=device, dtype=dtype)]

        if use_cache:
            decode_function = self.video_vae.cached_decode
        else:
            decode_function = self.video_vae.decode

        output = []
        for u in zs:
            output.append(decode_function(u.unsqueeze(0), scale).float().clamp_(-1, 1).squeeze(0))
        output = torch.stack(output, dim=0)
        # from [batch_size, num_channels, num_frames, height, width]
        # to [batch_size, num_frames, num_channels, height, width]
        output = output.permute(0, 2, 1, 3, 4)
        return output
    
    # ===== 音频VAE接口 =====
    
    def encode_audio(self, audio: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            latent = self.audio_vae.wrapped_encode(audio)
        return latent  # [B, L_latent, C_latent]
    
    def decode_audio(self, latent: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            audio = self.audio_vae.wrapped_decode(latent)  # [B, L]
        return audio


class OviFusionWrapper(torch.nn.Module):
    def __init__(
        self, 
        model_name: str = "Ovi",
        model_path: Optional[str] = None,
        video_config_path: str = "ovi/configs/model/dit/video.json",
        audio_config_path: str = "ovi/configs/model/dit/audio.json",
        is_causal: bool = False,
        timestep_shift: float = 5.0,
        **kwargs
    ):
        super().__init__()
        self.model_name = model_name
        self.is_causal = is_causal
        self.is_main_process = not dist.is_initialized() or dist.get_rank() == 0

        with open(video_config_path) as f:
            self.video_config = json.load(f)
        with open(audio_config_path) as f:
            self.audio_config = json.load(f)

        if is_causal:
            logger.info(f"Initializing CausalFusionModel: {self.video_config['num_layers']} video blocks and {self.audio_config['num_layers']} audio blocks...") if self.is_main_process else None
            self.model = CausalFusionModel(self.video_config, self.audio_config).to(dtype=torch.bfloat16, device=torch.device('cpu'))
        else:
            logger.info(f"Initializing FusionModel: {self.video_config['num_layers']} video blocks and {self.audio_config['num_layers']} audio blocks...") if self.is_main_process else None
            self.model = FusionModel(self.video_config, self.audio_config).to(dtype=torch.bfloat16, device=torch.device('cpu'))

        if model_path is not None:
            logger.info(f"Ovi FusionModel initialized, loading model weights from {model_path}...") if self.is_main_process else None
            if model_path.endswith(".pt"):
                original_state_dict = torch.load(model_path, map_location='cpu')
                original_state_dict = original_state_dict["generator_ema"] if "generator_ema" in original_state_dict.keys() else original_state_dict["generator"]
            else:
                original_state_dict = load_file(model_path, device='cpu')
        else:
            model_path = f"/cpfs01/gongshukai/weights/Ovi/{self.model_name}/model_960x960.safetensors"
            logger.info(f"Ovi FusionModel initialized, loading model weights from {model_path}...") if self.is_main_process else None
            original_state_dict = load_file(model_path, device='cpu')
        remapped_state_dict = remap_ovi_state_dict_for_refactored(original_state_dict)

        missing_keys, unexpected_keys = self.model.load_state_dict(remapped_state_dict, strict=False)
        if missing_keys: 
            logger.warning(f"Ovi weights loading: Missing keys: {missing_keys}")
        if unexpected_keys: 
            logger.warning(f"Ovi weights loading: Unexpected keys: {unexpected_keys}")
        
        logger.info(f"Ovi weights loaded into refactored model.") if self.is_main_process else None
        self.model.eval()

        self.scheduler = FlowMatchScheduler(
            shift=timestep_shift,
            sigma_min=0.0,
            extra_one_step=True
        )
        self.scheduler.set_timesteps(1000, training=True)
        self.post_init()
    
    def enable_gradient_checkpointing(self):
        self.model.gradient_checkpointing = True
        
    def _convert_flow_pred_to_x0(self, flow_pred: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """
        将 flow matching 的预测 (velocity) 转换为 x0 预测。
        这个版本是正确的，并且可以通用地处理视频和音频。
        x0_pred = xt - sigma_t * flow_pred
        """
        # 使用更高精度以保证计算稳定
        original_dtype = flow_pred.dtype
        flow_pred_d = flow_pred.double()
        xt_d = xt.double()
        sigmas_d = self.scheduler.sigmas.to(xt.device, dtype=torch.double)
        timesteps_d = self.scheduler.timesteps.to(xt.device, dtype=torch.double)

        # 将多维时间步展平，使其与展平后的数据一一对应
        # timestep 输入形状可以是 [B] 或 [B, F] 或 [B, L]
        if timestep.dim() > 1:
            timestep_flat = timestep.flatten()
        else: # 如果是 [B]，需要扩展以匹配 xt 的 token 数量
            # shape of xt: video [B, F, C, H, W], audio [B, L, C]
            num_tokens_per_sample = xt.shape[1]    # F for video, L for audio
            timestep_flat = timestep.unsqueeze(1).repeat(1, num_tokens_per_sample).flatten()    # shape: [B * F] or [B * L]

        # 为每个 token 找到其对应的 sigma 值
        timestep_indices = torch.argmin(torch.abs(timesteps_d.unsqueeze(0) - timestep_flat.unsqueeze(1)), dim=1)
        sigma_t = sigmas_d[timestep_indices]
        
        # 将数据展平以便进行批处理
        xt_flat = xt_d.reshape(timestep_flat.shape[0], -1)  # video: [B * F, C * H * W], audio: [B * L, C]
        flow_pred_flat = flow_pred_d.reshape(timestep_flat.shape[0], -1)    # video: [B * F, C * H * W], audio: [B * L, C]

        # 调整 sigma_t 的形状以进行广播
        sigma_t = sigma_t.view(-1, *([1] * (xt_flat.dim() - 1)))    # video: [B * F, 1, 1, 1], audio: [B * L, 1]

        # 计算 x0 预测
        x0_pred_flat = xt_flat - sigma_t * flow_pred_flat
        
        # 恢复原始形状和数据类型
        x0_pred = x0_pred_flat.reshape(xt.shape).to(original_dtype)
        
        return x0_pred
    
    def forward(
        self,
        video_latent: Optional[torch.Tensor] = None,
        audio_latent: Optional[torch.Tensor] = None,
        timestep: Optional[torch.Tensor] = None,
        conditional_dict: Optional[dict] = None,
        
        wan22_image_latent: Optional[torch.Tensor] = None,
        mask2: Optional[torch.Tensor] = None,
        first_frame_is_clean: bool = False,
        slg_layer: Optional[int] = False,
        timestep_v: Optional[torch.Tensor] = None,
        timestep_a: Optional[torch.Tensor] = None,

        kv_cache_list: Optional[List] = None,
        current_start_vid: Optional[int] = None,
        current_start_audio: Optional[int] = None,

        **kwargs 
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        video_prompt_embeds = conditional_dict["video_prompt_embeds"] # [B, S, D]
        audio_prompt_embeds = conditional_dict["audio_prompt_embeds"] # [B, S, D]
        
        # --- 1. 准备输入：处理 Batch 维度 ---
        
        # Video Latent: [B, F, C, H, W] -> 目标: List of [C, F, H, W]
        # 这里的 permute 把 C 放到第 1 维: [B, C, F, H, W]
        # unbind(0) 把 Batch 拆成列表
        video_input_list = list(video_latent.permute(0, 2, 1, 3, 4).unbind(0))
        
        # Audio Latent: [B, L, C] -> 目标: List of [L, C]
        audio_input_list = list(audio_latent.unbind(0))
        
        # Context: [B, S, D] -> List of [S, D]
        video_context_list = list(video_prompt_embeds.unbind(0))
        audio_context_list = list(audio_prompt_embeds.unbind(0))
        
        # Timestep: [B] (保持原样，FusionModel 会处理)
        # 如果传入的是 [B, F] 这种，只取第一个维度
        

        # 计算 seq_len (用于 RoPE 等)
        num_frames, c, h, w = video_latent.shape[1:]
        _patch_size_h = self.model.video_patch_size[1]
        _patch_size_w = self.model.video_patch_size[2]
        vid_seq_len = num_frames * h * w // (_patch_size_h * _patch_size_w)
        audio_seq_len = audio_latent.shape[1]
        
        # --- 2. 调用 FusionModel ---
        # FusionModel 接受 List[Tensor] 作为输入
        if self.is_causal:
            assert timestep_v is not None and timestep_a is not None
            if kv_cache_list is not None:
                assert current_start_vid is not None and current_start_audio is not None
                flow_pred_video_list, flow_pred_audio_list = self.model(
                    vid=video_input_list,
                    audio=audio_input_list,
                    t_vid=timestep_v,
                    t_aud=timestep_a,
                    vid_context=video_context_list,
                    audio_context=audio_context_list,
                    vid_seq_len=vid_seq_len,
                    audio_seq_len=audio_seq_len,
                    kv_cache_list=kv_cache_list,
                    current_start_vid=current_start_vid,
                    current_start_audio=current_start_audio,
                    first_frame_is_clean=first_frame_is_clean,
                    slg_layer=slg_layer
                )
            else:
                flow_pred_video_list, flow_pred_audio_list = self.model(
                    vid=video_input_list,
                    audio=audio_input_list,
                    t_vid=timestep_v,
                    t_aud=timestep_a,
                    vid_context=video_context_list,
                    audio_context=audio_context_list,
                    vid_seq_len=vid_seq_len,
                    audio_seq_len=audio_seq_len,
                    first_frame_is_clean=first_frame_is_clean,
                    slg_layer=slg_layer
                )
            flow_pred_video = torch.stack(flow_pred_video_list).permute(0, 2, 1, 3, 4)
            flow_pred_audio = torch.stack(flow_pred_audio_list)
            x0_pred_video = self._convert_flow_pred_to_x0(flow_pred_video, video_latent, timestep_v)
            x0_pred_audio = self._convert_flow_pred_to_x0(flow_pred_audio, audio_latent, timestep_a)

        else:
            ts_input = timestep[:, 0] if timestep.dim() > 1 else timestep
            flow_pred_video_list, flow_pred_audio_list = self.model(
                vid=video_input_list,
                audio=audio_input_list,
                t=ts_input,
                vid_context=video_context_list,
                audio_context=audio_context_list,
                vid_seq_len=vid_seq_len,
                audio_seq_len=audio_seq_len,
                first_frame_is_clean=first_frame_is_clean,
                slg_layer=slg_layer
            )
            flow_pred_video = torch.stack(flow_pred_video_list).permute(0, 2, 1, 3, 4)
            flow_pred_audio = torch.stack(flow_pred_audio_list)
            x0_pred_video = self._convert_flow_pred_to_x0(flow_pred_video, video_latent, ts_input)
            x0_pred_audio = self._convert_flow_pred_to_x0(flow_pred_audio, audio_latent, ts_input)

        # --- 3. Mask 处理 ---
        if mask2 is not None and wan22_image_latent is not None:
            # mask2 shape: [B, F, C, H, W] (需要确保 masks_like 返回了正确的 batch size)
            # wan22_image_latent shape: [B, 1, C, H, W]
            final_x0_video = (1. - mask2) * wan22_image_latent + mask2 * x0_pred_video
            final_x0_video = final_x0_video.to(video_latent.dtype)
        else:
            final_x0_video = x0_pred_video

        return final_x0_video, x0_pred_audio, flow_pred_video, flow_pred_audio
    
    def get_scheduler(self) -> SchedulerInterface:
        scheduler = self.scheduler
        scheduler.convert_x0_to_noise = types.MethodType(
            SchedulerInterface.convert_x0_to_noise, scheduler)
        scheduler.convert_noise_to_x0 = types.MethodType(
            SchedulerInterface.convert_noise_to_x0, scheduler)
        scheduler.convert_velocity_to_x0 = types.MethodType(
            SchedulerInterface.convert_velocity_to_x0, scheduler)
        self.scheduler = scheduler
        return scheduler
    
    def post_init(self):
        self.get_scheduler()

def remap_ovi_state_dict_for_refactored(state_dict):
    """
    将预训练的 state_dict 键转换为新的扁平化 FusionModel 结构。
    """
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k
        # 规则 1: video_model.blocks.X.* -> fusion_blocks.X.vid_block.*
        if k.startswith("video_model.blocks."):
            parts = k.split('.')
            block_idx = parts[2]
            remaining_path = ".".join(parts[3:])
            new_key = f"fusion_blocks.{block_idx}.vid_block.{remaining_path}"
        # 规则 2: audio_model.blocks.X.* -> fusion_blocks.X.audio_block.*
        elif k.startswith("audio_model.blocks."):
            parts = k.split('.')
            block_idx = parts[2]
            remaining_path = ".".join(parts[3:])
            new_key = f"fusion_blocks.{block_idx}.audio_block.{remaining_path}"
        # 规则 3: video_model.* -> video_*
        elif k.startswith("video_model."):
            new_key = k.replace("video_model.", "video_")
        # 规则 4: audio_model.* -> audio_*
        elif k.startswith("audio_model."):
            new_key = k.replace("audio_model.", "audio_")
        
        new_key = new_key.replace("_fsdp_wrapped_module.", "")\
                         .replace("_checkpoint_wrapped_module.", "")\
                         .replace("_orig_mod.", "")
        if new_key.startswith("model."):
            new_key = new_key[len("model."):]
        
        new_state_dict[new_key] = v
    return new_state_dict