import os
import json
import types
import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Optional, Union
import logging

from ovi.modules.fusion import FusionModel
from ovi.modules.t5 import umt5_xxl
from wan22.modules.vae2_2 import _video_vae as _video_vae_2_2
from ovi.modules.mmaudio.features_utils import FeaturesUtils
from ovi.modules.tokenizers import HuggingfaceTokenizer

from utils.scheduler import SchedulerInterface, FlowMatchScheduler
from safetensors.torch import load_file

class OviTextEncoder(torch.nn.Module):
    def __init__(self, model_name: str = "Wan2.2-TI2V-5B") -> None:
        super().__init__()
        self.model_name = model_name
        
        self.text_encoder = umt5_xxl(
            encoder_only=True,
            return_tokenizer=False,
            dtype=torch.float32,
            device=torch.device('cpu')
        ).eval().requires_grad_(False)

        self.text_encoder.load_state_dict(
            torch.load(f"/videogen/Ovi/ckpts/{self.model_name}/models_t5_umt5-xxl-enc-bf16.pth",
                       map_location='cpu', weights_only=False)
        )

        self.tokenizer = HuggingfaceTokenizer(
            name=f"/videogen/Ovi/ckpts/{self.model_name}/google/umt5-xxl", 
            seq_len=512, 
            clean='whitespace'
        )
        logging.info(f"Loaded Ovi Text Encoder weights and tokenizer")
    
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
        video_vae_pth: str = "/videogen/Ovi/ckpts/Wan2.2-TI2V-5B/Wan2.2_VAE.pth",
        dim_mult: List[int] = [1, 2, 4, 4],
        temperal_downsample: List[bool] = [False, True, True],
        
        audio_mode: str = '16k',
        audio_tod_vae_ckpt: str = "/videogen/Ovi/ckpts/MMAudio/ext_weights/v1-16.pth",
        audio_bigvgan_ckpt: str = "/videogen/Ovi/ckpts/MMAudio/ext_weights/best_netG.pt",
    ):
        super().__init__()
        
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
        logging.info(f"Loaded Wan2.2-VAE weights from {video_vae_pth}")
        
        # ===== 音频VAE (MMAudio) =====
        # 初始化时在CPU上
        self.audio_vae = FeaturesUtils(
            mode=audio_mode,
            need_vae_encoder=True,
            tod_vae_ckpt=audio_tod_vae_ckpt,
            bigvgan_vocoder_ckpt=audio_bigvgan_ckpt,
        )
        self.audio_vae.eval().requires_grad_(False)
        logging.info(f"Loaded MMAudio VAE weights from {audio_tod_vae_ckpt} and Vocoder weights from {audio_bigvgan_ckpt}")
    
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
        video_config_path: str = "ovi/configs/model/dit/video.json",
        audio_config_path: str = "ovi/configs/model/dit/audio.json",
        is_causal: bool = False,
        timestep_shift: float = 5.0,
        **kwargs
    ):
        super().__init__()
        self.model_name = model_name
        self.is_causal = is_causal
        
        with open(video_config_path) as f:
            self.video_config = json.load(f)
        with open(audio_config_path) as f:
            self.audio_config = json.load(f)

        self.model = FusionModel(self.video_config, self.audio_config).to(dtype=torch.bfloat16, device=torch.device('cpu'))
        logging.info(f"Initialized FusionModel with {self.video_config['num_layers']} layers of video and audio blocks.")

        state_dict = load_file(
            f"/videogen/Ovi/ckpts/{self.model_name}/model.safetensors",
            device='cpu'
        )
        self.model.load_state_dict(state_dict)
        logging.info(f"Loaded Ovi weights")
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
    
    def forward(
        self,
        video_latent: torch.Tensor,         # [B, F, C, H, W]
        audio_latent: torch.Tensor,         # [B, L, C]
        timestep: torch.Tensor,             # [B] 共享时间步
        conditional_dict: dict,
        video_neg_embeds: Optional[torch.Tensor] = None,
        audio_neg_embeds: Optional[torch.Tensor] = None,
        clip_fea: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
        first_frame_is_clean: bool = False,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        text_embeds = conditional_dict["prompt_embeds"]
        batch_size = video_latent.shape[0]
        
        num_frames, c, h, w = video_latent.shape[1:]
        _patch_size_h = self.model.video_model.patch_size[1]
        _patch_size_w = self.model.video_model.patch_size[2]
        vid_seq_len = num_frames * h * w // (_patch_size_h * _patch_size_w)
        audio_seq_len = audio_latent.shape[1]  # L
        
        # video: [B, F, C, H, W] -> [C, F, H, W]
        # audio: [B, L, C] -> [L, C]
        video_input = video_latent.squeeze(0).permute(1, 0, 2, 3)
        audio_input = audio_latent.squeeze(0)
        text_input = text_embeds.squeeze(0)
        
        pred_video, pred_audio = self.model(
            vid=[video_input],
            audio=[audio_input],
            t=timestep,
            vid_context=[text_input],
            audio_context=[text_input],
            vid_seq_len=vid_seq_len,
            audio_seq_len=audio_seq_len,
            clip_fea=clip_fea,
            y=y,
            first_frame_is_clean=first_frame_is_clean,
        )
        
        pred_video = pred_video[0].permute(1, 0, 2, 3).unsqueeze(0)  # [C,F,H,W] -> [B,F,C,H,W]
        pred_audio = pred_audio[0].unsqueeze(0)  # [L,C] -> [B,L,C]
        
        return pred_video, pred_audio
    
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


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # print("=" * 50)
    # print("Testing OviTextEncoder (CPU initialization)...")
    # print("=" * 50)
    # try:
    #     text_encoder = OviTextEncoder()
    #     print(f"✓ Text encoder initialized on: {next(text_encoder.parameters()).device}")
        
    #     # 移动到GPU (模拟FSDP包装前的操作)
    #     text_encoder = text_encoder.cuda()
    #     print(f"✓ Text encoder moved to: {next(text_encoder.parameters()).device}")
        
    #     # 测试forward
    #     prompts = "A cat playing piano"
    #     result = text_encoder(prompts)
    #     print(f"✓ prompt_embeds shape: {result['prompt_embeds'].shape}")
    #     print(f"✓ prompt_embeds device: {result['prompt_embeds'].device}")
    # except Exception as e:
    #     print(f"✗ Error: {e}")
    #     import traceback
    #     traceback.print_exc()

    # del text_encoder 

    # print("\n" + "=" * 50)
    # print("Testing OviVAEWrapper (CPU initialization)...")
    # print("=" * 50)
    # try:
    #     vae = OviVAEWrapper()
    #     print(f"✓ VAE initialized")
    #     print(f"  Video VAE device: {next(vae.video_vae.parameters()).device}, dtype: {next(vae.video_vae.parameters()).dtype}")
    #     print(f"  Audio VAE device: {next(vae.audio_vae.parameters()).device}, dtype: {next(vae.audio_vae.parameters()).dtype}")
        
    #     # 移动到GPU (模拟Trainer中的操作)
    #     vae = vae.to(device=torch.device('cuda'))
    #     print(f"✓ VAE moved to GPU")
    #     print(f"  Video VAE device: {next(vae.video_vae.parameters()).device}, dtype: {next(vae.video_vae.parameters()).dtype}")
    #     print(f"  Audio VAE device: {next(vae.audio_vae.parameters()).device}, dtype: {next(vae.audio_vae.parameters()).dtype}")
        
    #     # 测试编解码
    #     video = torch.randn(1, 3, 121, 704, 1280).cuda()
    #     video_latent = vae.encode_video(video)
    #     print(f"✓ Video latent shape: {video_latent.shape}, dtype: {video_latent.dtype}")
        
    #     video_recon = vae.decode_video(video_latent)
    #     print(f"✓ Video recon shape: {video_recon.shape}, dtype: {video_recon.dtype}")
        
    #     audio = torch.randn(1, 128000).cuda()
    #     audio_latent = vae.encode_audio(audio)
    #     print(f"✓ Audio latent shape: {audio_latent.shape}, dtype: {audio_latent.dtype}")
        
    #     audio_recon = vae.decode_audio(audio_latent)
    #     print(f"✓ Audio recon shape: {audio_recon.shape}, dtype: {audio_recon.dtype}")
    # except Exception as e:
    #     print(f"✗ Error: {e}")
    #     import traceback
    #     traceback.print_exc()
    
    # del vae

    print("\n" + "=" * 50)
    print("Testing OviFusionWrapper (CPU initialization)...")
    print("=" * 50)
    try:
        model = OviFusionWrapper()
        print(f"✓ Fusion model initialized")
        print(f"  Video model device: {next(model.model.video_model.parameters()).device}, dtype: {next(model.model.video_model.parameters()).dtype}")
        print(f"  Audio model device: {next(model.model.audio_model.parameters()).device}, dtype: {next(model.model.audio_model.parameters()).dtype}")
        
        # 移动到GPU (模拟FSDP包装前的操作)
        model = model.cuda()
        print(f"✓ Fusion model moved to GPU")
        
        # 测试forward
        video_latent = torch.randn(1, 31, 48, 44, 80).cuda().to(dtype=torch.bfloat16)           # standard video shape [B, F, C, H, W] = [B, 31, 48, 44, 80]
        audio_latent = torch.randn(1, 157, 20).cuda().to(dtype=torch.bfloat16)                  # standard audio shape [B, L, D] = [B, 157, 20]
        timestep = torch.ones(1).long().cuda() * 500
        text_embeds = torch.randn(1, 512, 4096).cuda().to(dtype=torch.bfloat16)
        conditional_dict = {"prompt_embeds": text_embeds}
        print(f"✓ Input video_latent shape: {video_latent.shape}, dtype: {video_latent.dtype}")
        print(f"✓ Input audio_latent shape: {audio_latent.shape}, dtype: {audio_latent.dtype}")
        print(f"✓ Input timestep: {timestep}")
        print(f"✓ Input text_embeds shape: {text_embeds.shape}, dtype: {text_embeds.dtype}")
        pred_video, pred_audio = model(
            video_latent=video_latent,
            audio_latent=audio_latent,
            timestep=timestep,
            conditional_dict=conditional_dict,
        )
        
        print(f"✓ Pred video shape: {pred_video.shape}, dtype: {pred_video.dtype}")
        print(f"✓ Pred audio shape: {pred_audio.shape}, dtype: {pred_audio.dtype}")
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
    
    del model

    print("\n" + "=" * 50)
    print("All tests completed!")
    print("=" * 50)