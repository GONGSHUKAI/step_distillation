import os
import json
import torch
import matplotlib.pyplot as plt
import numpy as np
import traceback
import gc

from ovi.modules.causal_ovi import CausalFusionModel

def unit_test_attn_mask(model: CausalFusionModel, device):
    print("\n--- Test (i): Visualizing Attention Masks ---")
    
    # Inputs: 32 Video Frames, 160 Audio Tokens
    F_vid = 32
    L_aud = 160
    
    vid_shape = [model.video_config['in_dim'], F_vid, 44, 80] 
    aud_shape = [L_aud, model.audio_config['in_dim']]
    
    model._prepare_masks(
        device, vid_shape, aud_shape, 
        local_attn_size=model.video_config.get('local_attn_size', -1), 
        sink_size=model.video_config.get('sink_size', 0)
    )
    print("Masks prepared successfully.")
    
    if hasattr(model, 'vid_cross_mask') and model.vid_cross_mask is not None:
        print(f"Video Cross Mask generated: {model.vid_cross_mask}")
    else:
        print("Error: Mask generation failed.")

def unit_test_forward_train(model: CausalFusionModel, device):
    print("\n--- Test (ii): _forward_train ---")
    model.train()
    
    # 强制开启 GC
    model.gradient_checkpointing = True
    gc.collect()
    torch.cuda.empty_cache()
    
    B = 2
    F = 32
    L_aud = 160
    
    C_vid = model.video_config['in_dim']
    H = 44
    W = 80
    C_aud = model.audio_config['in_dim']
    
    # Inputs
    vid_input_list = [
        torch.randn(C_vid, F, H, W, device=device, dtype=torch.bfloat16) 
        for _ in range(B)
    ]
    aud_input_list = [
        torch.randn(L_aud, C_aud, device=device, dtype=torch.bfloat16)
        for _ in range(B)
    ]
    
    # [UPDATED] Construct Blockwise Timesteps
    # 假设 Block 数为 8 (32/4 = 8, 160/20 = 8)
    num_blocks = 8
    vid_frames_per_block = 4
    aud_tokens_per_block = 20
    
    # 随机生成每个 Block 的时间步索引 (0~3) -> 映射到真实时间步 (e.g. 1000, 750...)
    # 这里直接生成时间步数值模拟
    denoising_step_list = torch.tensor([1000, 937.5, 833.3, 625, 0], device=device)
    def _get_aligned_timestep_indices(B):
        block_indices = torch.randint(
            0, 
            len(denoising_step_list), 
            (B, num_blocks), 
            device=device
        )
        
        index_video = block_indices.unsqueeze(-1).repeat(1, 1, vid_frames_per_block).flatten(1, 2)
        index_audio = block_indices.unsqueeze(-1).repeat(1, 1, aud_tokens_per_block).flatten(1, 2)
        return index_video, index_audio
    
    index_v, index_a = _get_aligned_timestep_indices(B)

    t_vid = denoising_step_list[index_v].to(device)
    t_aud = denoising_step_list[index_a].to(device)
    
    
    print(f"t_vid shape: {t_vid.shape}, t_aud shape: {t_aud.shape}")
    print(f"t_vid: {t_vid}, t_aud: {t_aud}")
    text_dim = 4096
    vid_context_list = [
        torch.randn(model.video_config['text_len'], text_dim, device=device, dtype=torch.bfloat16)
        for _ in range(B)
    ]
    aud_context_list = [
        torch.randn(model.audio_config['text_len'], text_dim, device=device, dtype=torch.bfloat16)
        for _ in range(B)
    ]
    
    vid_seq_len = 32*44*80//2//2
    aud_seq_len = 160
    
    try:
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            # [UPDATED] Pass t_vid and t_aud instead of t
            vid_out, aud_out = model(
                vid=vid_input_list,
                audio=aud_input_list,
                t_vid=t_vid,  # <--- NEW
                t_aud=t_aud,  # <--- NEW
                vid_context=vid_context_list,
                audio_context=aud_context_list,
                vid_seq_len=vid_seq_len,
                audio_seq_len=aud_seq_len,
                first_frame_is_clean=True
            )
        print(f"Forward Train Success!")
        print(f"Video Output Type: {type(vid_out)}")
        if isinstance(vid_out, list):
            print(f"Video Sample 0 Shape: {vid_out[0].shape}") 
        else: 
            print(f"Video Output Shape: {vid_out.shape}")
            
        print(f"Audio Output Type: {type(aud_out)}")
        if isinstance(aud_out, list):
            print(f"Audio Sample 0 Shape: {aud_out[0].shape}") 
        else:
            print(f"Audio Output Shape: {aud_out.shape}")
    except Exception as e:
        print(f"Forward Train Failed: {e}")
        traceback.print_exc()
        
    del vid_input_list, aud_input_list, vid_context_list, aud_context_list, vid_out, aud_out
    torch.cuda.empty_cache()

def unit_test_forward_inference(model: CausalFusionModel, device):
    print("\n" + "="*60)
    print("--- Test (iii): _forward_inference (With KV Cache & List Inputs) ---")
    print("="*60)
    
    model.eval()
    B = 2
    
    # Block Size Inference
    F_block = 4
    L_aud_block = 20
    
    C_vid = model.video_config['in_dim']
    H_latent = 44 
    W_latent = 80 
    C_aud = model.audio_config['in_dim']
    
    # [UPDATED] Timesteps for Inference
    # Usually inference uses a single global timestep per step (e.g. t=1000)
    # But now model expects [B, F] and [B, L]
    t_val = 100.0
    
    # Step 1: Block 0
    # Video: 4 frames -> t_vid: [B, 4]
    # Audio: 20 tokens -> t_aud: [B, 20]
    t_vid_step1 = torch.full((B, F_block), t_val, device=device).float()
    t_aud_step1 = torch.full((B, L_aud_block), t_val, device=device).float()
    
    vid_context_list = [torch.randn(model.video_config['text_len'], model.text_dim, device=device, dtype=torch.bfloat16) for _ in range(B)]
    aud_context_list = [torch.randn(model.audio_config['text_len'], model.text_dim, device=device, dtype=torch.bfloat16) for _ in range(B)]

    vid_seq_len = 27280
    aud_seq_len = 160

    def init_cache(model, batch_size, device, dtype):
        cache_list = []
        pixels_per_frame = H_latent * W_latent
        patch_area = model.video_config['patch_size'][1] * model.video_config['patch_size'][2]
        tokens_per_vid_frame = pixels_per_frame // patch_area
        
        max_vid_frames = 32 
        max_aud_len = 160   
        
        max_vid_tokens = max_vid_frames * tokens_per_vid_frame
        max_aud_tokens = max_aud_len 
        
        head_dim_v = model.video_config['dim'] // model.video_config['num_heads']
        head_dim_a = model.audio_config['dim'] // model.audio_config['num_heads']
        
        for _ in range(model.num_blocks):
            layer_cache = {}
            def create_buffer(max_len, num_heads, head_dim):
                return {
                    "k": torch.zeros(batch_size, max_len, num_heads, head_dim, device=device, dtype=dtype),
                    "v": torch.zeros(batch_size, max_len, num_heads, head_dim, device=device, dtype=dtype),
                    "global_end_index": torch.zeros(1, device=device, dtype=torch.long),
                    "local_end_index": torch.zeros(1, device=device, dtype=torch.long),
                }
            def create_cross_buffer():
                return {"k": None, "v": None, "is_init": False}

            layer_cache['vid_self'] = create_buffer(max_vid_tokens, model.video_config['num_heads'], head_dim_v)
            layer_cache['aud_self'] = create_buffer(max_aud_tokens, model.audio_config['num_heads'], head_dim_a)
            layer_cache['vid_fusion'] = create_buffer(max_aud_tokens, model.video_config['num_heads'], head_dim_v)
            layer_cache['aud_fusion'] = create_buffer(max_vid_tokens, model.audio_config['num_heads'], head_dim_a)
            layer_cache['vid_text'] = create_cross_buffer()
            layer_cache['aud_text'] = create_cross_buffer()
            cache_list.append(layer_cache)
        return cache_list, tokens_per_vid_frame

    kv_cache_list, tokens_per_vid_frame = init_cache(model, B, device, torch.bfloat16)
    print(f"Cache initialized. Tokens per video frame: {tokens_per_vid_frame}")

    # Step 1: Block 0
    print("\n>>> Step 1: Infer Block 0 (4 frames, 20 tokens)")
    vid_step1_list = [torch.randn(C_vid, F_block, H_latent, W_latent, device=device, dtype=torch.bfloat16) for _ in range(B)]
    aud_step1_list = [torch.randn(L_aud_block, C_aud, device=device, dtype=torch.bfloat16) for _ in range(B)]
    
    try:
        with torch.no_grad():
            # [UPDATED] Pass t_vid and t_aud
            v_out1, a_out1 = model(
                vid=vid_step1_list, 
                audio=aud_step1_list, 
                t_vid=t_vid_step1, 
                t_aud=t_aud_step1,
                vid_context=vid_context_list, 
                audio_context=aud_context_list,
                vid_seq_len=vid_seq_len, 
                audio_seq_len=aud_seq_len,
                kv_cache_list=kv_cache_list,
                current_start_vid=0, 
                current_start_audio=0,
                first_frame_is_clean=True 
            )
        print(f"[Success] Step 1 Output Shapes: Video={v_out1[0].shape}, Audio={a_out1[0].shape}")
    except Exception as e:
        print(f"[Failed] Step 1: {e}")
        traceback.print_exc()
        return

    # Step 2: Block 1
    print("\n>>> Step 2: Infer Block 1 (4 frames, 20 tokens)")
    current_start_vid = F_block * tokens_per_vid_frame
    current_start_audio = L_aud_block
    vid_step2_list = [torch.randn(C_vid, F_block, H_latent, W_latent, device=device, dtype=torch.bfloat16) for _ in range(B)]
    aud_step2_list = [torch.randn(L_aud_block, C_aud, device=device, dtype=torch.bfloat16) for _ in range(B)]
    
    # Timesteps for Step 2
    t_val2 = 833.3
    t_vid_step2 = torch.full((B, F_block), t_val2, device=device).float()
    t_aud_step2 = torch.full((B, L_aud_block), t_val2, device=device).float()

    try:
        with torch.no_grad():
            v_out2, a_out2 = model(
                vid=vid_step2_list, 
                audio=aud_step2_list, 
                t_vid=t_vid_step2,
                t_aud=t_aud_step2,
                vid_context=vid_context_list, 
                audio_context=aud_context_list,
                vid_seq_len=vid_seq_len, 
                audio_seq_len=aud_seq_len,
                kv_cache_list=kv_cache_list,
                current_start_vid=current_start_vid, 
                current_start_audio=current_start_audio,
                first_frame_is_clean=False
            )
        print(f"[Success] Step 2 Output Shapes: Video={v_out2[0].shape}, Audio={a_out2[0].shape}")
    except Exception as e:
        print(f"[Failed] Step 2: {e}")
        traceback.print_exc()

def unit_test_load_weight(ckpt_path, model: CausalFusionModel):
    print("\n--- Test (iv): Loading Weights ---")
    if os.path.exists(ckpt_path):
        try:
            print(f"Loading checkpoint from {ckpt_path}...")
            state_dict = torch.load(ckpt_path, map_location='cpu')
            if "generator_ema" in state_dict.keys():
                state_dict = state_dict["generator_ema"]
            
            cleaned_state_dict = {}
            for key, value in state_dict.items():
                new_key = key.replace("_fsdp_wrapped_module.", "")\
                             .replace("_checkpoint_wrapped_module.", "")\
                             .replace("_orig_mod.", "")
                if new_key.startswith("model."):
                    new_key = new_key[len("model."):]
                cleaned_state_dict[new_key] = value
            
            # v2 model handles key mapping internally, strict=True is fine
            missing, unexpected = model.load_state_dict(cleaned_state_dict, strict=True)
            print("Weights loaded.")
            if len(missing) > 0: print(f"Missing keys: {missing}")
            if len(unexpected) > 0: print(f"Unexpected keys: {unexpected}")
                
        except Exception as e:
            print(f"Failed to load weights: {e}")
            traceback.print_exc()
    else:
        print(f"Checkpoint file not found at {ckpt_path}. Skipping load test.")
        
    print("\n" + "="*80)
    print("All Tests Completed.")
    print("="*80)

if __name__ == "__main__":
    print("="*80)
    print("Running Unit Tests for CausalFusionModel")
    print("="*80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on device: {device}")

    # 模拟路径
    video_config_path = "/cpfs01/gongshukai/step_distillation/ovi/configs/model/dit/video.json"
    audio_config_path = "/cpfs01/gongshukai/step_distillation/ovi/configs/model/dit/audio.json"
    ckpt_path = "/cpfs01/gongshukai/step_distillation/logs/legacy/OviDMD/model_ema.pt"

    def load_config(path):
        if os.path.exists(path):
            print(f"Loading config from {path}")
            with open(path, 'r') as f:
                return json.load(f)
        else:
            raise RuntimeError(f"config path {path} not found")

    video_config = load_config(video_config_path)
    audio_config = load_config(audio_config_path)

    # 初始化模型
    model = CausalFusionModel(video_config, audio_config).to(device).bfloat16()
    model.eval()
    print("Model initialized successfully.")

    unit_test_attn_mask(model, device)
    unit_test_forward_train(model, device) 
    unit_test_forward_inference(model, device)
    unit_test_load_weight(ckpt_path, model)