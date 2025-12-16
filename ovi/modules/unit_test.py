"""
PYTHONPATH=. python -m ovi.modules.unit_test
"""

from ovi.modules.causal_ovi import CausalFusionModel
import logging, os
logging.basicConfig(
    level=logging.INFO,
    format="[%(filename)s] %(levelname)s: %(message)s"
)

import torch
def unit_test_attn_mask(model: CausalFusionModel, device):
    print("\n--- Test (i): _prepare_masks ---")
    # the input of Ovi:
    # video: List[torch.Tensor], shape [torch.Tensor([48, 31, 44, 80]), torch.Tensor([48, 31, 44, 80]), ...]
    # audio: List[torch.Tensor], shape [torch.Tensor([155, 20]), torch.Tensor([155, 20])]
    vid_shape = [48, 31, 44, 80]
    aud_shape = [155, 20]
    model._prepare_masks(device, vid_shape, aud_shape, local_attn_size=-1, sink_size=0)

def unit_test_forward_train(model: CausalFusionModel):
    print("\n--- Test (ii): _forward_train ---")
    model.train()
    
    B = 2
    F = 31
    C_vid = 48
    H = 44
    W = 80
    L_aud = 155
    C_aud = 20
    
    # [修正] 构造 List[Tensor]
    # 每个 Video 样本: [C_vid, F, H, W] -> (48, 31, 44, 80)
    # 这一步很重要：Ovi/Wan 习惯上单个样本是 C First 的
    vid_input_list = [
        torch.randn(C_vid, F, H, W, device=device, dtype=torch.bfloat16) 
        for _ in range(B)
    ]
    
    # 每个 Audio 样本: [L_aud, C_aud] -> (155, 20) 
    # 或者 (C_aud, L_aud) 取决于 Conv1d。
    # 你的模型里 Audio Embed 是 ChannelLastConv1d(in_dim, ...)，这意味着输入应该是 (L, C)
    aud_input_list = [
        torch.randn(L_aud, C_aud, device=device, dtype=torch.bfloat16)
        for _ in range(B)
    ]
    
    t = torch.tensor([10, 20], device=device).float()
    
    # Context 依然是 List[Tensor] 或者 Stacked Tensor 都可以，
    # 但模型内部 prepare 函数有处理 context 的逻辑：
    # context = text_embedding(torch.stack([...])) 
    # 这意味着 context 也最好是 List[Tensor]，每个 Shape [L, D]
    text_dim = 4096
    vid_context_list = [
        torch.randn(video_config['text_len'], text_dim, device=device, dtype=torch.bfloat16)
        for _ in range(B)
    ]
    aud_context_list = [
        torch.randn(audio_config['text_len'], text_dim, device=device, dtype=torch.bfloat16)
        for _ in range(B)
    ]
    
    vid_seq_len = 27280 
    aud_seq_len = 155
    
    try:
        # 传入 List
        vid_out, aud_out = model(
            vid=vid_input_list,
            audio=aud_input_list,
            t=t,
            vid_context=vid_context_list,
            audio_context=aud_context_list,
            vid_seq_len=vid_seq_len,
            audio_seq_len=aud_seq_len,
            first_frame_is_clean=True
        )
        print(f"Forward Train Success!")
        # 输出通常会被 stack 回 Tensor: [B, C, F, H, W] 或类似
        print(f"Video Output: {type(vid_out)}, shape: {vid_out[0].shape}") 
        print(f"Audio Output: {type(aud_out)}, shape: {aud_out[0].shape}")
    except Exception as e:
        print(f"Forward Train Failed: {e}")
        import traceback
        traceback.print_exc()

def unit_test_forward_inference(model: CausalFusionModel, device):
    print("\n" + "="*60)
    print("--- Test (iii): _forward_inference (With KV Cache & List Inputs) ---")
    print("="*60)
    
    model.eval()
    
    # ------------------------------------------------------------------
    # 1. Setup Dimensions & Data
    # ------------------------------------------------------------------
    B = 2
    
    # Video Config
    F_vid_step1 = 1  # First frame (Reference)
    F_vid_step2 = 3  # Next block
    C_vid = model.video_config['in_dim'] # 48
    H = 44 * 2 # Latent H (88) -> Patch H=44. Let's use input pixel size or latent size?
               # Assuming input to model is LATENT after VAE encode.
               # If patch_size is (1,2,2), input H/W should be divisible by 2.
    H_latent = 44 # Example
    W_latent = 80 # Example
    
    # Audio Config
    L_aud_step1 = 5
    L_aud_step2 = 15
    C_aud = model.audio_config['in_dim'] # Usually 16 or similar
    
    # Timesteps: [B]
    t = torch.tensor([100.0, 50.0], device=device).to(torch.bfloat16) # Batch size 2
    
    # Context (List[Tensor]): [Seq_Len, Dim]
    vid_context_list = [
        torch.randn(model.video_config['text_len'], model.text_dim, device=device, dtype=torch.bfloat16)
        for _ in range(B)
    ]
    aud_context_list = [
        torch.randn(model.audio_config['text_len'], model.text_dim, device=device, dtype=torch.bfloat16)
        for _ in range(B)
    ]

    # Max Sequence Lengths (for Positional Encoding limits)
    vid_seq_len = 27280 
    aud_seq_len = 155

    # ------------------------------------------------------------------
    # 2. Initialize KV Cache Function
    # ------------------------------------------------------------------
    def init_cache(model, batch_size, device, dtype):
        cache_list = []
        
        # Calculate tokens per frame for Video to estimate buffer size
        # Patch size (1, 2, 2) -> 2*2=4 pixels per token
        pixels_per_frame = H_latent * W_latent
        patch_area = model.video_config['patch_size'][1] * model.video_config['patch_size'][2]
        tokens_per_vid_frame = pixels_per_frame // patch_area # 44*80/4 = 880
        
        max_vid_frames = 31 # From config usually
        max_aud_len = 157   # From config usually
        
        # Buffer sizes
        max_vid_tokens = max_vid_frames * tokens_per_vid_frame
        max_aud_tokens = max_aud_len # Audio maps 1-to-1 usually if not patched temporally
        
        # Head dimensions
        head_dim_v = model.video_config['dim'] // model.video_config['num_heads']
        head_dim_a = model.audio_config['dim'] // model.audio_config['num_heads']
        
        for _ in range(model.num_blocks):
            layer_cache = {}
            
            # Helper to create one K/V buffer [B, Max_Len, H, D]
            def create_buffer(max_len, num_heads, head_dim):
                return {
                    "k": torch.zeros(batch_size, max_len, num_heads, head_dim, device=device, dtype=dtype),
                    "v": torch.zeros(batch_size, max_len, num_heads, head_dim, device=device, dtype=dtype),
                    "global_end_index": torch.zeros(1, device=device, dtype=torch.long),
                    "local_end_index": torch.zeros(1, device=device, dtype=torch.long),
                }
            
            # Helper for CrossAttn Cache (Text)
            def create_cross_buffer():
                return {
                    "k": None, "v": None, "is_init": False
                }

            # --- Self Attention Caches ---
            layer_cache['vid_self'] = create_buffer(max_vid_tokens, model.video_config['num_heads'], head_dim_v)
            layer_cache['aud_self'] = create_buffer(max_aud_tokens, model.audio_config['num_heads'], head_dim_a)
            
            # --- Fusion Caches (Cross Stream) ---
            # NOTE on Dimensions:
            # - vid_fusion: Video block attends to Audio. Key/Value source is AUDIO. So dim is head_dim_v (query dim), but length is max_aud_tokens.
            # - aud_fusion: Audio block attends to Video. Key/Value source is VIDEO. So dim is head_dim_a (query dim), but length is max_vid_tokens.
            # (Assuming K_fusion/V_fusion project target to source's head dimension structure)
            
            layer_cache['vid_fusion'] = create_buffer(max_aud_tokens, model.video_config['num_heads'], head_dim_v)
            layer_cache['aud_fusion'] = create_buffer(max_vid_tokens, model.audio_config['num_heads'], head_dim_a)
            
            # --- Text Cross Caches ---
            layer_cache['vid_text'] = create_cross_buffer()
            layer_cache['aud_text'] = create_cross_buffer()
            
            cache_list.append(layer_cache)
        return cache_list, tokens_per_vid_frame

    # Init Cache
    kv_cache_list, tokens_per_vid_frame = init_cache(model, B, device, torch.bfloat16)
    print(f"Cache initialized. Tokens per video frame: {tokens_per_vid_frame}")

    # ------------------------------------------------------------------
    # 3. Step 1: First Frame (Prefill)
    # ------------------------------------------------------------------
    print("\n>>> Step 1: Prefill (Video: 1 frame, Audio: 5 tokens)")
    
    # Input: List[Tensor]. 
    # Video Tensor: [C, F, H, W]
    vid_step1_list = [
        torch.randn(C_vid, F_vid_step1, H_latent, W_latent, device=device, dtype=torch.bfloat16)
        for _ in range(B)
    ]
    # Audio Tensor: [L, C] (Channel Last)
    aud_step1_list = [
        torch.randn(L_aud_step1, C_aud, device=device, dtype=torch.bfloat16)
        for _ in range(B)
    ]
    
    try:
        with torch.no_grad():
            v_out1, a_out1 = model(
                vid=vid_step1_list, 
                audio=aud_step1_list, 
                t=t,
                vid_context=vid_context_list, 
                audio_context=aud_context_list,
                vid_seq_len=vid_seq_len, 
                audio_seq_len=aud_seq_len,
                kv_cache_list=kv_cache_list,
                current_start_vid=0, 
                current_start_audio=0,
                first_frame_is_clean=True # I2V typically assumes 1st frame is clean condition
            )
        # Expected Output: [B, C_out, F, H, W] (Video)
        print(f"[Success] Step 1 Output Shapes: Video={v_out1[0].shape}, Audio={a_out1[0].shape}")
        
    except Exception as e:
        print(f"[Failed] Step 1: {e}")
        import traceback
        traceback.print_exc()
        return

    # ------------------------------------------------------------------
    # 4. Step 2: Next Block (Decode)
    # ------------------------------------------------------------------
    print("\n>>> Step 2: Decode (Video: 3 frames, Audio: 15 tokens)")
    
    # Update start indices based on previous step lengths
    # Video: 1 frame * 880 tokens
    # Audio: 5 tokens
    current_start_vid = F_vid_step1 * tokens_per_vid_frame
    current_start_audio = L_aud_step1
    
    # Input for Step 2
    vid_step2_list = [
        torch.randn(C_vid, F_vid_step2, H_latent, W_latent, device=device, dtype=torch.bfloat16)
        for _ in range(B)
    ]
    aud_step2_list = [
        torch.randn(L_aud_step2, C_aud, device=device, dtype=torch.bfloat16)
        for _ in range(B)
    ]
    # print(kv_cache_list)
    try:
        with torch.no_grad():
            v_out2, a_out2 = model(
                vid=vid_step2_list, 
                audio=aud_step2_list, 
                t=t,
                vid_context=vid_context_list, 
                audio_context=aud_context_list,
                vid_seq_len=vid_seq_len, 
                audio_seq_len=aud_seq_len,
                kv_cache_list=kv_cache_list, # Reuse same cache
                current_start_vid=current_start_vid, 
                current_start_audio=current_start_audio,
                first_frame_is_clean=False # Usually False for generated frames
            )
        print(f"[Success] Step 2 Output Shapes: Video={v_out2[0].shape}, Audio={a_out2[0].shape}")
        
    except Exception as e:
        print(f"[Failed] Step 2: {e}")
        import traceback
        traceback.print_exc()

    # ------------------------------------------------------------------
    # 5. Verify Cache State (Optional)
    # ------------------------------------------------------------------
    print("\n>>> Verifying Cache Updates")
    # Check first block's vid_self cache
    cache_global_idx = kv_cache_list[0]['vid_self']['global_end_index'].item()
    expected_vid_tokens = (F_vid_step1 + F_vid_step2) * tokens_per_vid_frame
    print(f"Global End Index (Vid Self): {cache_global_idx} (Expected: {expected_vid_tokens})")
    
    cache_aud_idx = kv_cache_list[0]['aud_self']['global_end_index'].item()
    expected_aud_tokens = L_aud_step1 + L_aud_step2
    print(f"Global End Index (Aud Self): {cache_aud_idx} (Expected: {expected_aud_tokens})")

    if cache_global_idx == expected_vid_tokens and cache_aud_idx == expected_aud_tokens:
        print(">>> Cache updated correctly!")
    else:
        print(">>> Cache index mismatch!")

def unit_test_load_weight(ckpt_path, model: CausalFusionModel):
    # -------------------------------------------------------------------------
    # (iv) Load Weights
    # -------------------------------------------------------------------------
    print("\n--- Test (iv): Loading Weights ---")
    print(model)
    if os.path.exists(ckpt_path):
        try:
            print(f"Loading checkpoint from {ckpt_path}...")
            state_dict = torch.load(ckpt_path, map_location='cpu')
            if "generator_ema" in state_dict.keys():
                state_dict = state_dict["generator_ema"]
            
            # for k, v in state_dict.items():
            #     print(k, v.shape)
                
            cleaned_state_dict = {}
            for key, value in state_dict.items():
                new_key = key.replace("_fsdp_wrapped_module.", "")\
                             .replace("_checkpoint_wrapped_module.", "")\
                             .replace("_orig_mod.", "")
                if new_key.startswith("model."):
                    new_key = new_key[len("model."):]
                cleaned_state_dict[new_key] = value
            # print(cleaned_state_dict.keys())

            missing, unexpected = model.load_state_dict(cleaned_state_dict, strict=True)
            print("Weights loaded.")
            print(f"Missing keys (Expected if configs differ): {missing}")
            print(f"Unexpected keys: {unexpected}")
                
        except Exception as e:
            print(f"Failed to load weights: {e}")
    else:
        print(f"Checkpoint file not found at {ckpt_path}. Skipping load test.")
        
    print("\n" + "="*80)
    print("All Tests Completed.")
    print("="*80)


    
if __name__ == "__main__":
    import os
    import json
    import torch
    import matplotlib.pyplot as plt
    import numpy as np

    print("="*80)
    print("Running Unit Tests for CausalFusionModel")
    print("="*80)

    # -------------------------------------------------------------------------
    # 0. Setup & Config Loading
    # -------------------------------------------------------------------------
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
    model.eval() # 默认 Eval，部分测试切换 Train
    print("Model initialized successfully.")

    unit_test_attn_mask(model, device)
    unit_test_forward_train(model)
    unit_test_forward_inference(model, device)
    unit_test_load_weight(ckpt_path, model)