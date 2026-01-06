"""
PYTHONPATH=. python -m ovi.modules.unit_test
"""

import os
import json
import torch
import matplotlib.pyplot as plt
import numpy as np
import traceback
import gc

from ovi.modules.debug_ovi import CausalFusionModel, FusionModel
from ovi.modules import debug_ovi

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
    print("--- Test (iii): _forward_inference (With KV Cache & Logic Verification) ---")
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
    
    t_val = 100.0
    t_vid_step = torch.full((B, F_block), t_val, device=device).float()
    t_aud_step = torch.full((B, L_aud_block), t_val, device=device).float()
    
    vid_context_list = [torch.randn(model.video_config['text_len'], model.text_dim, device=device, dtype=torch.bfloat16) for _ in range(B)]
    aud_context_list = [torch.randn(model.audio_config['text_len'], model.text_dim, device=device, dtype=torch.bfloat16) for _ in range(B)]

    vid_seq_len = 28160 # 32 frames
    aud_seq_len = 160

    # 模拟 KV Cache 初始化
    def init_cache(model, batch_size, device, dtype):
        cache_list = []
        pixels_per_frame = H_latent * W_latent
        patch_area = model.video_config['patch_size'][1] * model.video_config['patch_size'][2]
        tokens_per_vid_frame = pixels_per_frame // patch_area
        
        max_vid_tokens = 32 * tokens_per_vid_frame
        max_aud_tokens = 160 
        
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

    # =========================================================================
    # Step 1: Infer Block 0 (Append Mode)
    # =========================================================================
    print("\n>>> Step 1: Infer Block 0 (Time T=1000)")
    vid_step1_list = [torch.randn(C_vid, F_block, H_latent, W_latent, device=device, dtype=torch.bfloat16) for _ in range(B)]
    aud_step1_list = [torch.randn(L_aud_block, C_aud, device=device, dtype=torch.bfloat16) for _ in range(B)]
    
    try:
        with torch.no_grad():
            v_out1, a_out1 = model(
                vid=vid_step1_list, audio=aud_step1_list, 
                t_vid=t_vid_step, t_aud=t_aud_step,
                vid_context=vid_context_list, audio_context=aud_context_list,
                vid_seq_len=vid_seq_len, audio_seq_len=aud_seq_len,
                kv_cache_list=kv_cache_list,
                current_start_vid=0, current_start_audio=0,
                first_frame_is_clean=True 
            )
        print(f"[Success] Output: {v_out1[0].shape}")
        
        # Check Cache Pointers
        g_end = kv_cache_list[0]['vid_self']['global_end_index'].item()
        l_end = kv_cache_list[0]['vid_self']['local_end_index'].item()
        print(f"  > Cache Indices after Step 1: Global={g_end}, Local={l_end}")
        expected_len = F_block * tokens_per_vid_frame
        assert g_end == expected_len, f"Expected Global End {expected_len}, got {g_end}"
        
    except Exception as e:
        print(f"[Failed] Step 1: {e}")
        traceback.print_exc()
        return

    # =========================================================================
    # Step 1.5: Infer Block 0 AGAIN (Overwrite/Denoising Mode)
    # 这就是导致之前 RuntimeError 的场景
    # =========================================================================
    print("\n>>> Step 1.5: Infer Block 0 Again (Time T=750) -> Testing Overwrite Logic")
    # Simulate a denoising step: Input size same, start position same
    vid_step1_5_list = [torch.randn(C_vid, F_block, H_latent, W_latent, device=device, dtype=torch.bfloat16) for _ in range(B)]
    
    try:
        with torch.no_grad():
            v_out1_5, _ = model(
                vid=vid_step1_5_list, audio=aud_step1_list, 
                t_vid=t_vid_step, t_aud=t_aud_step, # Timestep value doesn't matter for shape
                vid_context=vid_context_list, audio_context=aud_context_list,
                vid_seq_len=vid_seq_len, audio_seq_len=aud_seq_len,
                kv_cache_list=kv_cache_list,
                current_start_vid=0, current_start_audio=0, # SAME start as Step 1
                first_frame_is_clean=True 
            )
        
        g_end_new = kv_cache_list[0]['vid_self']['global_end_index'].item()
        l_end_new = kv_cache_list[0]['vid_self']['local_end_index'].item()
        print(f"  > Cache Indices after Step 1.5: Global={g_end_new}, Local={l_end_new}")
        
        # 关键验证：指针不应增加
        assert g_end_new == g_end, f"Error: Global index grew from {g_end} to {g_end_new} during overwrite!"
        print("  [PASS] Cache correctly overwritten without growing.")
        
    except Exception as e:
        print(f"[Failed] Step 1.5 (Overwrite): {e}")
        traceback.print_exc()
        return

    # =========================================================================
    # Step 2: Infer Block 1 (Append Mode)
    # =========================================================================
    print("\n>>> Step 2: Infer Block 1 (Append Mode)")
    current_start_vid = F_block * tokens_per_vid_frame
    current_start_audio = L_aud_block
    vid_step2_list = [torch.randn(C_vid, F_block, H_latent, W_latent, device=device, dtype=torch.bfloat16) for _ in range(B)]
    aud_step2_list = [torch.randn(L_aud_block, C_aud, device=device, dtype=torch.bfloat16) for _ in range(B)]
    
    try:
        with torch.no_grad():
            v_out2, a_out2 = model(
                vid=vid_step2_list, audio=aud_step2_list, 
                t_vid=t_vid_step, t_aud=t_aud_step,
                vid_context=vid_context_list, audio_context=aud_context_list,
                vid_seq_len=vid_seq_len, audio_seq_len=aud_seq_len,
                kv_cache_list=kv_cache_list,
                current_start_vid=current_start_vid, current_start_audio=current_start_audio,
                first_frame_is_clean=False
            )
        g_end_2 = kv_cache_list[0]['vid_self']['global_end_index'].item()
        print(f"  > Cache Indices after Step 2: Global={g_end_2}")
        
        expected_len_2 = current_start_vid + (F_block * tokens_per_vid_frame)
        assert g_end_2 == expected_len_2, f"Expected {expected_len_2}, got {g_end_2}"
        print(f"  [PASS] Cache correctly appended.")
        
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

def compare_traces():
    print("\n" + "=" * 80)
    print("🔬 DETAILED LAYER-WISE COMPARISON (Block 0)")
    print("=" * 80)
    
    bi_trace = debug_ovi.DEBUG_TRACER.get('bi', {})
    causal_trace = debug_ovi.DEBUG_TRACER.get('causal', {})
    
    if not bi_trace or not causal_trace:
        print("⚠️  Traces are empty. Make sure trace() is called inside models.")
        return

    def sort_key(k):
        if k.startswith("prepare_"):
            emb_order = [
                'prepare_x_vid', 'prepare_x_aud',
                'prepare_e_vid', 'prepare_e_aud',
                'prepare_e0_vid', 'prepare_e0_aud',
                'prepare_context_vid', 'prepare_context_aud',
                'prepare_freqs_vid', 'prepare_freqs_aud', # Freqs 可能会有 shape 差异，注意
                'prepare_grid_sizes_vid', 'prepare_seq_lens_vid'
            ]
            if k in emb_order:
                return -1000 + emb_order.index(k)
            return -500 # 其他 prepare

        # 2. Block 层 (b0, b1...)
        if k.startswith('b'):
            parts = k.split('.')
            try:
                block_idx = int(parts[0][1:])
            except ValueError:
                return 99999
            
            layer_name = parts[1]
            
            # === 更新后的详细 Layer Order ===
            layer_order = [
                'vid_in', 'aud_in',     # Input
                'vid_e_chunks', 'audio_e_chunks',   # Modulation Factors
                'aud_norm1', 'audio_y', 'audio_self_out', # Audio Self-Attention
                'vid_norm1', 'vid_y', 'vid_self_out', # Video Self-Attention
                'og_audio', # Cross-Attention Prep
                
                # Audio Cross-Attention (Text + Vid)
                'aud_text_crossattn_proj_q', # Query Projection
                'aud_text_crossattn',        # Output of FlashAttn(Q, K_text, V_text)
                'aud_fusion_out',            # Final Cross Output (Text + Vid mixed)
                
                # Video Cross-Attention (Text + Aud)
                'vid_text_crossattn_proj_q',
                'vid_text_crossattn',
                'vid_fusion_out',
                
                # FFN / Block Output
                'aud_final_fusion_out',
                'vid_final_fusion_out'
            ]
            
            order = layer_order.index(layer_name) if layer_name in layer_order else 99
            return block_idx * 1000 + order
            
        return 99999

    common_keys = sorted(list(set(bi_trace.keys()) & set(causal_trace.keys())), key=sort_key)
    first_failure = None
    print(f"{'STATUS':<4} | {'LAYER NAME':<30} | {'SUM DIFF':<12} | {'NORM DIFF':<12} | {'SHAPE'}")
    print("-" * 85)

    for key in common_keys:
        t_bi = bi_trace[key].float()
        t_causal = causal_trace[key].float()
        
        if t_bi.shape != t_causal.shape:
            if t_bi.numel() == t_causal.numel():
                t_bi = t_bi.flatten()
                t_causal = t_causal.flatten()
            else:
                print(f"❌   | {key:<30} | SHAPE MISMATCH: {tuple(t_bi.shape)} vs {tuple(t_causal.shape)}")
                if first_failure is None: first_failure = key
                continue

        sum_diff = (t_bi - t_causal).abs().sum().item()
        norm_df = torch.norm(t_bi - t_causal).item()
        threshold = 1e-2

        status = "✅" if sum_diff < threshold else "❌"
        
        print(f"{status:<4} | {key:<30} | {sum_diff:.6f}     | {norm_df:.6f}     | {tuple(t_bi.shape)}")
        
        if norm_df > threshold and first_failure is None:
            first_failure = key

    print("-" * 85)
    if first_failure:
        print(f"\n🚨 First divergence detected at: {first_failure}")
        print("   -> Check inputs to this layer and operations immediately preceding it.")
    else:
        print("\n✨ All layers matched within tolerance.")


def unit_test_compute_diff(ckpt_path, causal_model: CausalFusionModel, bi_model: FusionModel):
    print("\n" + "=" * 80)
    print("UNIT TEST: Comparing First Block Inference (Block 0 Only)")
    print("Logic: Causal Inference (Block 0, Empty Cache) == Bidirectional Inference (Short Sequence)")
    print("Config: Video=4 frames, Audio=20 frames")
    print("=" * 80)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16
    
    # 1. Load Weights
    print(f"[1] Loading weights from {ckpt_path}...")
    state_dict = torch.load(ckpt_path, map_location="cpu")
    if "generator_ema" in state_dict: 
        sd = state_dict["generator_ema"]
    elif "generator" in state_dict: 
        sd = state_dict["generator"]
    else: 
        sd = state_dict
    
    def clean_state_dict(sd):
        clean_sd = {}
        for key, value in sd.items():
            new_key = key.replace("_fsdp_wrapped_module.", "")\
                            .replace("_checkpoint_wrapped_module.", "")\
                            .replace("_orig_mod.", "")
            if new_key.startswith("model."):
                new_key = new_key[len("model."):]
            
            # 这里可能需要根据 debug_ovi.py 中的命名调整映射
            # 假设 debug_ovi.py 里的类结构需要以下映射:
            if new_key.startswith("video_model.blocks."):
                parts = new_key.split('.')
                new_key = f"fusion_blocks.{parts[2]}.vid_block." + ".".join(parts[3:])
            elif new_key.startswith("audio_model.blocks."):
                parts = new_key.split('.')
                new_key = f"fusion_blocks.{parts[2]}.audio_block." + ".".join(parts[3:])
            elif new_key.startswith("video_model."):
                new_key = new_key.replace("video_model.", "video_")
            elif new_key.startswith("audio_model."):
                new_key = new_key.replace("audio_model.", "audio_")
                
            clean_sd[new_key] = value
        return clean_sd
        
    clean_sd = clean_state_dict(sd)

    print(f"[2] Loading weights into models...")
    missing_bi, _ = bi_model.load_state_dict(clean_sd, strict=False) # 改为 False 以容忍部分不匹配，如 freqs
    missing_causal, _ = causal_model.load_state_dict(clean_sd, strict=False)
    
    bi_model.to(device, dtype).eval()
    causal_model.to(device, dtype).eval()

    # 2. Prepare Inputs (First Block Only)
    # Video: 4 frames
    # Audio: 20 frames
    B = 1
    F_vid = 4
    L_aud = 20
    C_vid = 48
    C_aud = 20
    H, W = 44, 80
    
    torch.manual_seed(42) # 固定随机种子
    
    # 输入 Latents
    vid_latent = torch.randn(B, C_vid, F_vid, H, W, device=device, dtype=dtype)
    aud_latent = torch.randn(B, L_aud, C_aud, device=device, dtype=dtype)
    
    # Timestep (scalar t=500)
    t_val = 500
    
    # Context (Text Embeddings)
    # 假设文本长度 512, 维度 4096
    vid_context = torch.randn(B, 512, 4096, device=device, dtype=dtype)
    aud_context = torch.randn(B, 512, 4096, device=device, dtype=dtype)
    
    # 准备 List 格式输入 (FusionModel 需要 List)
    bi_vid_in = [v for v in vid_latent] # List of [C, 4, H, W]
    bi_aud_in = [a for a in aud_latent] # List of [20, C]
    bi_vid_ctx = [c for c in vid_context]
    bi_aud_ctx = [c for c in aud_context]
    
    # 计算 seq_len
    _patch_h, _patch_w = bi_model.video_patch_size[1], bi_model.video_patch_size[2]
    vid_seq_len = F_vid * H * W // (_patch_h * _patch_w) # 4 * 880 = 3520
    aud_seq_len = L_aud # 20

    # 3. Run FusionModel (Bidirectional)
    print(f"[3] Running FusionModel (Bidirectional)...")
    
    # === [HOOK] Start Tracing Bi-Model ===
    debug_ovi.CURRENT_MODEL_TYPE = 'bi'
    debug_ovi.DEBUG_TRACER['bi'] = {}
    
    t_scalar = torch.tensor([t_val] * B, device=device, dtype=torch.long)
    
    with torch.no_grad():
        bi_vid_out_list, bi_aud_out_list = bi_model(
            vid=bi_vid_in,
            audio=bi_aud_in,
            t=t_scalar,
            vid_context=bi_vid_ctx,
            audio_context=bi_aud_ctx,
            vid_seq_len=vid_seq_len,
            audio_seq_len=aud_seq_len
        )
        bi_vid_out = torch.stack(bi_vid_out_list) # [B, C, 4, H, W]
        bi_aud_out = torch.stack(bi_aud_out_list) # [B, 20, C]

    # 4. Run CausalFusionModel (Inference Path)
    print(f"[4] Running CausalFusionModel (_forward_inference)...")
    
    # === [HOOK] Start Tracing Causal-Model ===
    debug_ovi.CURRENT_MODEL_TYPE = 'causal'
    debug_ovi.DEBUG_TRACER['causal'] = {}

    # 4.1 Initialize Empty KV Cache
    num_layers = causal_model.num_blocks
    vid_tokens_per_frame = (H * W) // (_patch_h * _patch_w) # 880
    max_vid_tokens = 32 * vid_tokens_per_frame 
    max_aud_tokens = 160
    
    head_dim_v = causal_model.video_config['dim'] // causal_model.video_config['num_heads']
    head_dim_a = causal_model.audio_config['dim'] // causal_model.audio_config['num_heads']
    num_heads_v = causal_model.video_config['num_heads']
    num_heads_a = causal_model.audio_config['num_heads']
    
    kv_cache_list = []
    for _ in range(num_layers):
        layer_cache = {}
        def create_buf(length, n_heads, d_head):
            return {
                "k": torch.zeros(B, length, n_heads, d_head, device=device, dtype=dtype),
                "v": torch.zeros(B, length, n_heads, d_head, device=device, dtype=dtype),
                "global_end_index": torch.zeros(1, device=device, dtype=torch.long),
                "local_end_index": torch.zeros(1, device=device, dtype=torch.long),
            }
        layer_cache['vid_self'] = create_buf(max_vid_tokens, num_heads_v, head_dim_v)
        layer_cache['aud_self'] = create_buf(max_aud_tokens, num_heads_a, head_dim_a)
        layer_cache['vid_fusion'] = create_buf(max_aud_tokens, num_heads_v, head_dim_v)
        layer_cache['aud_fusion'] = create_buf(max_vid_tokens, num_heads_a, head_dim_a)
        layer_cache['vid_text'] = {"k": None, "v": None, "is_init": False}
        layer_cache['aud_text'] = {"k": None, "v": None, "is_init": False}
        kv_cache_list.append(layer_cache)

    # 4.2 Prepare Inference Timesteps [B, 4]
    t_vid_block = torch.full((B, F_vid), t_val, device=device, dtype=torch.long) 
    t_aud_block = torch.full((B, L_aud), t_val, device=device, dtype=torch.long)

    with torch.no_grad():
        causal_vid_out_list, causal_aud_out_list = causal_model(
            vid=bi_vid_in,      
            audio=bi_aud_in,    
            t_vid=t_vid_block,
            t_aud=t_aud_block,
            vid_context=bi_vid_ctx,
            audio_context=bi_aud_ctx,
            vid_seq_len=vid_seq_len, 
            audio_seq_len=aud_seq_len, 
            
            kv_cache_list=kv_cache_list,
            current_start_vid=0,         
            current_start_audio=0,       
            first_frame_is_clean=False   
        )

        causal_vid_out = torch.stack(causal_vid_out_list) 
        causal_aud_out = torch.stack(causal_aud_out_list) 

    # 5. Compare Results (Layer-wise)
    compare_traces()

    # 6. Compare Final Results
    print(f"\n[6] Comparing Final Outputs...")

    diff_v = (bi_vid_out - causal_vid_out).abs()
    max_diff_v, mean_diff_v, sum_diff_v, norm_diff_v = diff_v.max().item(), diff_v.mean().item(), diff_v.sum().item(), diff_v.norm().item()
    print(f"    Video Output | MaxDiff: {max_diff_v:.6f} | MeanDiff: {mean_diff_v:.6f}")

    diff_a = (bi_aud_out - causal_aud_out).abs()
    max_diff_a, mean_diff_a, sum_diff_a, norm_diff_a = diff_a.max().item(), diff_a.mean().item(), diff_a.sum().item(), diff_a.norm().item()
    print(f"    Audio Output | MaxDiff: {max_diff_a:.6f} | MeanDiff: {mean_diff_a:.6f}")

    if max_diff_v < 0.1 and max_diff_a < 0.1:
        print("\n✅ SUCCESS: Causal Model matches Bidirectional Model.")
    else:
        print("\n❌ FAILURE: Final outputs mismatch.")


if __name__ == "__main__":
    print("="*80)
    print("Running Unit Tests for CausalFusionModel")
    print("="*80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on device: {device}")

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

    model = CausalFusionModel(video_config, audio_config).to(device).bfloat16()
    model.eval()
    print("CausalFusionModel initialized successfully.")

    model2 = FusionModel(video_config, audio_config).to(device).bfloat16()
    model2.eval()
    print("FusionModel initialized successfully.")

    # unit_test_attn_mask(model, device)
    # unit_test_forward_train(model, device) 
    # unit_test_forward_inference(model, device)
    # unit_test_load_weight(ckpt_path, model)
    unit_test_compute_diff(ckpt_path, causal_model=model, bi_model=model2)