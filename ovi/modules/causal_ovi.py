# FILE: ovi/modules/causal_ovi.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import torch.distributed as dist

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin

from ovi.modules.model import (
    WanRMSNorm, WanLayerNorm, Head, ChannelLastConv1d, ConvMLP, 
    rope_params, sinusoidal_embedding_1d, rope_apply, 
    ModulationAdd
)
# Using 'attention' from ovi.modules.attention
from ovi.modules.attention import attention, flash_attention

# Flex Attention imports
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
import logging

logger = logging.getLogger(__name__)

# Compile flex_attention for performance
flex_attention = torch.compile(
    flex_attention, dynamic=False, mode="max-autotune-no-cudagraphs"
)

def gradient_checkpointing(module: nn.Module, *args, enabled: bool, **kwargs):
    # if enabled:
    #     return checkpoint(module, *args, use_reentrant=False, **kwargs)
    # else:
    #     return module(*args, **kwargs)
    return checkpoint(module, *args, use_reentrant=False, **kwargs)

# ========================================================================================
# RoPE Utilities (保持不变)
# ========================================================================================

def causal_rope_apply_1d(x, grid_sizes, freqs, start_index=0):
    n, c = x.size(2), x.size(3) // 2
    c_rope = freqs.shape[1] 
    output = []
    for i, (l,) in enumerate(grid_sizes.tolist()):
        seq_len = l
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(seq_len, n, -1, 2))
        freqs_i = freqs[start_index : start_index + seq_len].unsqueeze(1)
        x_i_rope = x_i[:, :, :c_rope] * freqs_i
        x_i_passthrough = x_i[:, :, c_rope:]
        x_i = torch.cat([x_i_rope, x_i_passthrough], dim=2)
        x_i = torch.view_as_real(x_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])
        output.append(x_i)
    return torch.stack(output).type_as(x)

def causal_rope_apply_3d(x, grid_sizes, freqs, start_frame_index=0):
    n, c = x.size(2), x.size(3) // 2
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(seq_len, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][start_frame_index : start_frame_index + f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ], dim=-1).reshape(seq_len, 1, -1)
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])
        output.append(x_i)
    return torch.stack(output).type_as(x)

def causal_rope_apply(x, grid_sizes, freqs, start_index=0, tokens_per_frame=None):
    x_ndim = grid_sizes.shape[-1]
    if x_ndim == 3:
        assert tokens_per_frame is not None and tokens_per_frame > 0
        start_frame = start_index // tokens_per_frame
        return causal_rope_apply_3d(x, grid_sizes, freqs, start_frame_index=start_frame)
    else:
        return causal_rope_apply_1d(x, grid_sizes, freqs, start_index=start_index)

# ========================================================================================
# Core Calculation Logic (Stateless)
# ========================================================================================

def causal_fusion_logic(
    # Weights container (the cross_attn block where layers are injected)
    module_container, 
    # Inputs
    q_src, target_seq, 
    # Props
    local_attn_size, sink_size,
    # Context
    block_mask=None, fusion_cache=None,
    grid_sizes_src=None, freqs_src=None,
    grid_sizes_target=None, freqs_target=None,
    current_start=0, cache_start=0
):
    """
    执行 Fusion Attention 的具体计算逻辑。
    参数 module_container 应当包含 k_fusion, v_fusion, pre_attn_norm_fusion, norm_k_fusion 层。
    """
    b, s_q, n, d = q_src.shape
    
    # 1. Projection (使用注入的权重)
    target_seq_norm = module_container.pre_attn_norm_fusion(target_seq)
    k = module_container.norm_k_fusion(module_container.k_fusion(target_seq_norm)).view(b, -1, n, d)
    v = module_container.v_fusion(target_seq_norm).view(b, -1, n, d)
    
    # Helper
    def get_tokens_per_frame(g): return math.prod(g[0][1:]).item() if g.shape[-1] == 3 else 1
    tpf_src = get_tokens_per_frame(grid_sizes_src)
    tpf_target = get_tokens_per_frame(grid_sizes_target)
    
    # Max attention size calculation
    # base_tokens = 880 # Just for reference in calculation
    current_max_attn_size = 27280 if local_attn_size == -1 else local_attn_size * tpf_target

    if fusion_cache is not None:
        # --- INFERENCE MODE ---
        q_src = causal_rope_apply(q_src, grid_sizes_src, freqs_src, start_index=current_start, tokens_per_frame=tpf_src)
        k = causal_rope_apply(k, grid_sizes_target, freqs_target, start_index=cache_start, tokens_per_frame=tpf_target)
        
        real_sink_tokens = sink_size * tpf_target
        kv_cache_size = fusion_cache["k"].shape[1]
        num_new_tokens = k.shape[1]

        current_end = fusion_cache["global_end_index"].item() + num_new_tokens
        
        # Cache Rolling Logic (Sink + Window)
        if local_attn_size != -1 and (current_end > fusion_cache["global_end_index"].item()) and \
           (num_new_tokens + fusion_cache["local_end_index"].item() > kv_cache_size):
            
            num_evicted = num_new_tokens + fusion_cache["local_end_index"].item() - kv_cache_size
            num_rolled = fusion_cache["local_end_index"].item() - num_evicted - real_sink_tokens
            
            if num_rolled > 0:
                fusion_cache["k"][:, real_sink_tokens : real_sink_tokens + num_rolled] = \
                    fusion_cache["k"][:, real_sink_tokens + num_evicted : real_sink_tokens + num_evicted + num_rolled].clone()
                fusion_cache["v"][:, real_sink_tokens : real_sink_tokens + num_rolled] = \
                    fusion_cache["v"][:, real_sink_tokens + num_evicted : real_sink_tokens + num_evicted + num_rolled].clone()
            
            local_write_start = fusion_cache["local_end_index"].item() - num_evicted
            local_write_end = local_write_start + num_new_tokens
            
            fusion_cache["k"][:, local_write_start : local_write_end] = k
            fusion_cache["v"][:, local_write_start : local_write_end] = v
            local_end_index = local_write_end
        else:
            local_write_start = fusion_cache["local_end_index"].item()
            local_write_end = local_write_start + num_new_tokens
            
            fusion_cache["k"][:, local_write_start : local_write_end] = k
            fusion_cache["v"][:, local_write_start : local_write_end] = v
            local_end_index = local_write_end

        fusion_cache["global_end_index"].fill_(current_end)
        fusion_cache["local_end_index"].fill_(local_end_index)
        
        att_start = max(0, local_end_index - current_max_attn_size)
        k_view = fusion_cache["k"][:, :local_end_index] # View valid part
        v_view = fusion_cache["v"][:, :local_end_index]
        x = attention(q_src, k_view, v_view)
        
    else:
        # --- TRAINING MODE ---
        q_src = rope_apply(q_src, grid_sizes_src, freqs_src)
        k = rope_apply(k, grid_sizes_target, freqs_target)
        
        s_kv = k.shape[1]
        pad_len_q = math.ceil(s_q / 128) * 128 - s_q
        pad_len_kv = math.ceil(s_kv / 128) * 128 - s_kv
        
        def pad_tensor(t, pad):
            if pad > 0: return torch.cat([t, torch.zeros(b, pad, n, d, device=t.device, dtype=t.dtype)], dim=1)
            return t

        q_pad = pad_tensor(q_src, pad_len_q)
        k_pad = pad_tensor(k, pad_len_kv)
        v_pad = pad_tensor(v, pad_len_kv)
        
        x = flex_attention(
            query=q_pad.transpose(2, 1),
            key=k_pad.transpose(2, 1),
            value=v_pad.transpose(2, 1),
            block_mask=block_mask
        )
        x = x.transpose(2, 1)
        
        if pad_len_q > 0: x = x[:, :s_q]
            
    x = x.flatten(2)
    return x

# ========================================================================================
# Attention Modules
# ========================================================================================

class CausalWanSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, local_attn_size=-1, sink_size=0, qk_norm=True, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qk_norm = qk_norm
        self.eps = eps
        
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size
        self.max_attention_size = 27280 if local_attn_size == -1 else local_attn_size * 880 

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, seq_lens, grid_sizes, freqs, block_mask=None, kv_cache=None, current_start=0, cache_start=None):
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        if cache_start is None: cache_start = current_start
        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n, d)

        if grid_sizes.shape[-1] == 3:
            tokens_per_frame = math.prod(grid_sizes[0][1:]).item()
        else:
            tokens_per_frame = 1

        if kv_cache is not None:
            # Inference
            q_rope = causal_rope_apply(q, grid_sizes, freqs, start_index=current_start, tokens_per_frame=tokens_per_frame)
            k_rope = causal_rope_apply(k, grid_sizes, freqs, start_index=current_start, tokens_per_frame=tokens_per_frame)
            
            sink_tokens = self.sink_size * tokens_per_frame
            kv_cache_size = kv_cache["k"].shape[1]
            num_new_tokens = k_rope.shape[1]
            current_end = kv_cache["global_end_index"].item() + num_new_tokens
            
            if self.local_attn_size != -1 and (current_end > kv_cache["global_end_index"].item()) and \
               (num_new_tokens + kv_cache["local_end_index"].item() > kv_cache_size):
                num_evicted = num_new_tokens + kv_cache["local_end_index"].item() - kv_cache_size
                num_rolled = kv_cache["local_end_index"].item() - num_evicted - sink_tokens
                if num_rolled > 0:
                    kv_cache["k"][:, sink_tokens : sink_tokens + num_rolled] = kv_cache["k"][:, sink_tokens + num_evicted : sink_tokens + num_evicted + num_rolled].clone()
                    kv_cache["v"][:, sink_tokens : sink_tokens + num_rolled] = kv_cache["v"][:, sink_tokens + num_evicted : sink_tokens + num_evicted + num_rolled].clone()
                local_write_start = kv_cache["local_end_index"].item() - num_evicted
                local_write_end = local_write_start + num_new_tokens
                kv_cache["k"][:, local_write_start : local_write_end] = k_rope
                kv_cache["v"][:, local_write_start : local_write_end] = v
                local_end_index = local_write_end
            else:
                local_write_start = kv_cache["local_end_index"].item()
                local_write_end = local_write_start + num_new_tokens
                kv_cache["k"][:, local_write_start : local_write_end] = k_rope
                kv_cache["v"][:, local_write_start : local_write_end] = v
                local_end_index = local_write_end

            kv_cache["global_end_index"].fill_(current_end)
            kv_cache["local_end_index"].fill_(local_end_index)
            
            att_start = max(0, local_end_index - self.max_attention_size)
            k_view = kv_cache["k"][:, att_start:local_end_index]
            v_view = kv_cache["v"][:, att_start:local_end_index]
            x = attention(q_rope, k_view, v_view)
        else:
            # Training
            q = rope_apply(q, grid_sizes, freqs)
            k = rope_apply(k, grid_sizes, freqs)
            padded_length = math.ceil(s / 128) * 128 - s
            def pad_tensor(t):
                if padded_length > 0: return torch.cat([t, torch.zeros(b, padded_length, n, d, device=t.device, dtype=t.dtype)], dim=1)
                return t
            q_pad, k_pad, v_pad = pad_tensor(q), pad_tensor(k), pad_tensor(v)
            x = flex_attention(query=q_pad.transpose(2, 1), key=k_pad.transpose(2, 1), value=v_pad.transpose(2, 1), block_mask=block_mask)
            x = x.transpose(2, 1)
            if padded_length > 0: x = x[:, :s]

        x = x.flatten(2)
        return self.o(x)


class CausalWanT2VCrossAttention(CausalWanSelfAttention):
    def forward(self, x, context, context_lens, crossattn_cache=None):
        b, n, d = x.size(0), self.num_heads, self.head_dim
        q = self.norm_q(self.q(x)).view(b, -1, n, d)

        if crossattn_cache is not None:
            if not crossattn_cache["is_init"]:
                crossattn_cache["is_init"] = True
                k = self.norm_k(self.k(context)).view(b, -1, n, d)
                v = self.v(context).view(b, -1, n, d)
                crossattn_cache["k"] = k
                crossattn_cache["v"] = v
            else:
                k = crossattn_cache["k"]
                v = crossattn_cache["v"]
        else:
            k = self.norm_k(self.k(context)).view(b, -1, n, d)
            v = self.v(context).view(b, -1, n, d)

        x_out = flash_attention(q, k, v, k_lens=context_lens)
        x_out = x_out.flatten(2)
        # Note: We return raw Attention Output and Q (Projected)
        # We do NOT run self.o(x) here because it will be done after summing fusion attn
        return x_out, q
    
class CausalWanAttentionBlock(nn.Module):
    def __init__(self, cross_attn_type, dim, ffn_dim, num_heads, local_attn_size=-1, sink_size=0, qk_norm=True, cross_attn_norm=False, eps=1e-6, additional_emb_length=None):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.local_attn_size = local_attn_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = CausalWanSelfAttention(dim, num_heads, local_attn_size, sink_size, qk_norm, eps)
        self.norm3 = WanLayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = CausalWanT2VCrossAttention(dim, num_heads, (-1, -1), qk_norm, eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'), nn.Linear(ffn_dim, dim))
        self.modulation = ModulationAdd(dim, 6)

    def forward(self, *args, **kwargs):
        pass 


class CausalFusionAttentionBlock(nn.Module):
    def __init__(self, vid_block: CausalWanAttentionBlock, audio_block: CausalWanAttentionBlock):
        super().__init__()
        self.vid_block = vid_block
        self.audio_block = audio_block
        
        # Configuration for Fusion Logic (Saved as dicts)
        # Note the cross-wiring:
        # Vid Block Fusion -> Queries Audio KV -> Needs Audio Config
        self.vid_fusion_props = {
            'local_attn_size': audio_block.self_attn.local_attn_size,
            'sink_size': audio_block.self_attn.sink_size
        }
        # Audio Block Fusion -> Queries Video KV -> Needs Video Config
        self.aud_fusion_props = {
            'local_attn_size': vid_block.self_attn.local_attn_size,
            'sink_size': vid_block.self_attn.sink_size
        }

    def forward(
        self, 
        vid, audio, 
        vid_self_cache=None, audio_self_cache=None, 
        vid_fusion_cache=None, audio_fusion_cache=None, 
        vid_text_cache=None, audio_text_cache=None, 
        vid_self_mask=None, audio_self_mask=None,
        vid_cross_mask=None, audio_cross_mask=None,
        current_start=0, cache_start=None, 
        current_start_vid=0, current_start_audio=0, 
        vid_context=None, vid_context_lens=None,
        audio_context=None, audio_context_lens=None,
        **kwargs
    ):
        cs_vid = current_start_vid
        cs_aud = current_start_audio
        
        vid_e_chunks = kwargs['vid_e'].chunk(6, dim=2)
        audio_e_chunks = kwargs['audio_e'].chunk(6, dim=2)
        
        # === 1. Audio Self ===
        audio_norm = self.audio_block.norm1(audio).bfloat16() * (1 + audio_e_chunks[1].squeeze(2)) + audio_e_chunks[0].squeeze(2)
        audio_y = self.audio_block.self_attn(
            audio_norm, 
            seq_lens=kwargs['audio_seq_lens'], grid_sizes=kwargs['audio_grid_sizes'], freqs=kwargs['audio_freqs'],
            block_mask=audio_self_mask, kv_cache=audio_self_cache, current_start=cs_aud, cache_start=cache_start
        )
        audio = audio + audio_y * audio_e_chunks[2].squeeze(2)

        # === 2. Video Self ===
        vid_norm = self.vid_block.norm1(vid).bfloat16() * (1 + vid_e_chunks[1].squeeze(2)) + vid_e_chunks[0].squeeze(2)
        vid_y = self.vid_block.self_attn(
            vid_norm,
            seq_lens=kwargs['vid_seq_lens'], grid_sizes=kwargs['vid_grid_sizes'], freqs=kwargs['vid_freqs'],
            block_mask=vid_self_mask, kv_cache=vid_self_cache, current_start=cs_vid, cache_start=cache_start
        )
        vid = vid + vid_y * vid_e_chunks[2].squeeze(2)

        # === 3. Cross Attention ===
        og_audio = audio
        
        # --- Audio Block Cross (Text + Video) ---
        audio_text_unproj, audio_q_proj = self.audio_block.cross_attn(
            self.audio_block.norm3(audio), context=audio_context, context_lens=audio_context_lens, crossattn_cache=audio_text_cache 
        )
        # Fusion: Audio Q queries Video KV
        # Weights are in self.audio_block.cross_attn (injected)
        audio_vid_unproj = causal_fusion_logic(
            module_container=self.audio_block.cross_attn,
            q_src=audio_q_proj, target_seq=vid,
            local_attn_size=self.aud_fusion_props['local_attn_size'],
            sink_size=self.aud_fusion_props['sink_size'],
            block_mask=audio_cross_mask, fusion_cache=audio_fusion_cache,
            grid_sizes_src=kwargs['audio_grid_sizes'], freqs_src=kwargs['audio_freqs'],
            grid_sizes_target=kwargs['vid_grid_sizes'], freqs_target=kwargs['vid_freqs'],
            current_start=cs_aud, cache_start=cs_vid
        )
        audio_cross_out = self.audio_block.cross_attn.o(audio_text_unproj + audio_vid_unproj)

        # --- Video Block Cross (Text + Audio) ---
        vid_text_unproj, vid_q_proj = self.vid_block.cross_attn(
            self.vid_block.norm3(vid), context=vid_context, context_lens=vid_context_lens, crossattn_cache=vid_text_cache
        )
        # Fusion: Video Q queries Audio KV
        # Weights are in self.vid_block.cross_attn (injected)
        vid_audio_unproj = causal_fusion_logic(
            module_container=self.vid_block.cross_attn,
            q_src=vid_q_proj, target_seq=og_audio, 
            local_attn_size=self.vid_fusion_props['local_attn_size'],
            sink_size=self.vid_fusion_props['sink_size'],
            block_mask=vid_cross_mask, fusion_cache=vid_fusion_cache,
            grid_sizes_src=kwargs['vid_grid_sizes'], freqs_src=kwargs['vid_freqs'],
            grid_sizes_target=kwargs['audio_grid_sizes'], freqs_target=kwargs['audio_freqs'],
            current_start=cs_vid, cache_start=cs_aud
        )
        vid_cross_out = self.vid_block.cross_attn.o(vid_text_unproj + vid_audio_unproj)

        # FFNs
        audio = audio + audio_cross_out
        audio_ffn = self.audio_block.ffn(
            self.audio_block.norm2(audio).bfloat16() * (1 + audio_e_chunks[4].squeeze(2)) + audio_e_chunks[3].squeeze(2)
        )
        audio = audio + audio_ffn * audio_e_chunks[5].squeeze(2)
        
        vid = vid + vid_cross_out
        vid_ffn = self.vid_block.ffn(
            self.vid_block.norm2(vid).bfloat16() * (1 + vid_e_chunks[4].squeeze(2)) + vid_e_chunks[3].squeeze(2)
        )
        vid = vid + vid_ffn * vid_e_chunks[5].squeeze(2)

        return vid, audio


# ========================================================================================
# Causal Fusion Model (Main)
# ========================================================================================

class CausalFusionModel(ModelMixin, ConfigMixin):
    def __init__(self, video_config, audio_config):
        super().__init__()
        self.video_config = video_config
        self.audio_config = audio_config
        
        vc = video_config
        self.video_patch_size = vc['patch_size']
        self.text_dim = 4096

        self.video_patch_embedding = nn.Conv3d(vc['in_dim'], vc['dim'], kernel_size=vc['patch_size'], stride=vc['patch_size'])
        self.video_text_embedding = nn.Sequential(nn.Linear(self.text_dim, vc['dim']), nn.GELU(approximate='tanh'), nn.Linear(vc['dim'], vc['dim']))
        self.video_time_embedding = nn.Sequential(nn.Linear(vc['freq_dim'], vc['dim']), nn.SiLU(), nn.Linear(vc['dim'], vc['dim']))
        self.video_time_projection = nn.Sequential(nn.SiLU(), nn.Linear(vc['dim'], vc['dim'] * 6))
        self.video_head = Head(vc['dim'], vc['out_dim'], vc['patch_size'], vc['eps'])
        self.sink_size_v = video_config.get('sink_size', 0)

        ac = audio_config
        self.audio_patch_size = ac['patch_size']
        self.audio_patch_embedding = nn.Sequential(ChannelLastConv1d(ac['in_dim'], ac['dim'], kernel_size=7, padding=3), nn.SiLU(), ConvMLP(ac['dim'], ac['dim'] * 4, kernel_size=7, padding=3))
        self.audio_text_embedding = nn.Sequential(nn.Linear(self.text_dim, ac['dim']), nn.GELU(approximate='tanh'), nn.Linear(ac['dim'], ac['dim']))
        self.audio_time_embedding = nn.Sequential(nn.Linear(ac['freq_dim'], ac['dim']), nn.SiLU(), nn.Linear(ac['dim'], ac['dim']))
        self.audio_time_projection = nn.Sequential(nn.SiLU(), nn.Linear(ac['dim'], ac['dim'] * 6))
        self.audio_head = Head(ac['dim'], ac['out_dim'], ac['patch_size'], ac['eps'])
        self.sink_size_a = audio_config.get('sink_size', 0)

        self.num_blocks = vc['num_layers']
        
        self.fusion_blocks = nn.ModuleList()
        for i in range(self.num_blocks):
            vid_blk = CausalWanAttentionBlock(
                vc['model_type'], vc['dim'], vc['ffn_dim'], vc['num_heads'],
                local_attn_size=vc.get('local_attn_size', -1), sink_size=self.sink_size_v,
                qk_norm=vc['qk_norm'], cross_attn_norm=vc['cross_attn_norm'], eps=vc['eps'],
                additional_emb_length=vc.get('additional_emb_length'),
            )
            aud_blk = CausalWanAttentionBlock(
                ac['model_type'], ac['dim'], ac['ffn_dim'], ac['num_heads'],
                local_attn_size=ac.get('local_attn_size', -1), sink_size=self.sink_size_a,
                qk_norm=ac['qk_norm'], cross_attn_norm=ac['cross_attn_norm'], eps=ac['eps'],
                additional_emb_length=ac.get('additional_emb_length'),
            )
            self.fusion_blocks.append(CausalFusionAttentionBlock(vid_blk, aud_blk))
        
        # Inject Fusion Weights (Matches Ovi.py structure)
        self.inject_cross_attention_kv_projections()
        
        self.set_rope_params()
        self.gradient_checkpointing = vc['gradient_checkpointing'] and ac['gradient_checkpointing']
        logger.info(f"Initialized CausalFusionModel with gradient checkpointing = {self.gradient_checkpointing}")
        
        self.vid_self_mask = None
        self.audio_self_mask = None
        self.vid_cross_mask = None
        self.audio_cross_mask = None

        self.independent_first_frame = True 
        self.num_frame_per_block_vid = 3
        self.num_frame_per_block_aud = 15
        self.num_aud_frame_per_vid = self.num_frame_per_block_aud // self.num_frame_per_block_vid

    def inject_cross_attention_kv_projections(self):
        for fusion_block in self.fusion_blocks:
            vid_block = fusion_block.vid_block
            audio_block = fusion_block.audio_block
            
            # Inject weights into Video Block's Cross Attn (for querying Audio)
            vid_block.cross_attn.k_fusion = nn.Linear(vid_block.dim, vid_block.dim)
            vid_block.cross_attn.v_fusion = nn.Linear(vid_block.dim, vid_block.dim)
            vid_block.cross_attn.pre_attn_norm_fusion = WanLayerNorm(vid_block.dim, elementwise_affine=True)
            vid_block.cross_attn.norm_k_fusion = WanRMSNorm(vid_block.dim, eps=1e-6) if vid_block.qk_norm else nn.Identity()
            
            # Inject weights into Audio Block's Cross Attn (for querying Video)
            audio_block.cross_attn.k_fusion = nn.Linear(audio_block.dim, audio_block.dim)
            audio_block.cross_attn.v_fusion = nn.Linear(audio_block.dim, audio_block.dim)
            audio_block.cross_attn.pre_attn_norm_fusion = WanLayerNorm(audio_block.dim, elementwise_affine=True)
            audio_block.cross_attn.norm_k_fusion = WanRMSNorm(audio_block.dim, eps=1e-6) if audio_block.qk_norm else nn.Identity()

    def set_rope_params(self):
        vc = self.video_config
        d = vc['dim'] // vc['num_heads']
        self.video_freqs = torch.cat([rope_params(1024, d - 4 * (d // 6)), rope_params(1024, 2 * (d // 6)), rope_params(1024, 2 * (d // 6))], dim=1)
        ac = self.audio_config
        d = ac['dim'] // ac['num_heads']
        self.audio_freqs = rope_params(1024, d - 4 * (d // 6), freqs_scaling=ac['temporal_rope_scaling_factor'])

    def init_weights(self):
        nn.init.zeros_(self.video_head.head.weight)
        nn.init.zeros_(self.audio_head.head.weight)
        # Init injected fusion weights (Xavier)
        for name, mod in self.named_modules():
            if ("k_fusion" in name or "v_fusion" in name) and isinstance(mod, nn.Linear):
                nn.init.xavier_uniform_(mod.weight)
                if mod.bias is not None: nn.init.zeros_(mod.bias)
            if "fusion" in name and isinstance(mod, nn.Linear):
                # Scale down fusion layers slightly as in Ovi
                with torch.no_grad(): mod.weight.div_(10.0)

    @staticmethod
    def _get_block_indices(total_len, block_sizes, device):
        block_indices = torch.zeros(total_len, device=device, dtype=torch.long)
        current_idx = 0
        for i, size in enumerate(block_sizes):
            block_indices[current_idx : current_idx + size] = i
            current_idx += size
        return block_indices

    def _prepare_masks(self, device, vid_shape, audio_shape, local_attn_size=-1, sink_size=0):
        # Implementation is identical to causal_ovi.py
        C_vid, F_vid, H_vid, W_vid = vid_shape
        L_aud, D_aud = audio_shape
        
        vid_tokens_per_frame = (H_vid * W_vid) // (self.video_config['patch_size'][1] * self.video_config['patch_size'][2])
        
        vid_block_structure = []
        if self.independent_first_frame:
            vid_block_structure.append(1 * vid_tokens_per_frame)
            remaining = F_vid - 1
        else:
            remaining = F_vid
        while remaining > 0:
            c = min(self.num_frame_per_block_vid, remaining)
            vid_block_structure.append(c * vid_tokens_per_frame)
            remaining -= c
        total_vid_tokens = sum(vid_block_structure)
        
        aud_block_structure = []
        num_vid_blocks = len(vid_block_structure)
        current_aud = 0
        for i in range(num_vid_blocks):
            size = 5 if i == 0 else 15
            aud_block_structure.append(size)
            current_aud += size
        diff = L_aud - current_aud  
        if diff != 0:   
            aud_block_structure[-1] += diff
            if aud_block_structure[-1] < 0: aud_block_structure = [L_aud]
        total_aud_tokens = sum(aud_block_structure)
        
        vid_ends = torch.zeros(total_vid_tokens, device=device, dtype=torch.long)   
        aud_ends = torch.zeros(total_aud_tokens, device=device, dtype=torch.long)   
        def fill_ends(structure, out_ends):
            cum = 0
            for sz in structure:
                out_ends[cum : cum + sz] = cum + sz
                cum += sz
        fill_ends(vid_block_structure, vid_ends)    
        fill_ends(aud_block_structure, aud_ends)    
        
        map_vid_to_aud = torch.zeros(total_vid_tokens, device=device, dtype=torch.long) 
        map_aud_to_vid = torch.zeros(total_aud_tokens, device=device, dtype=torch.long) 
        v_ptr, a_ptr = 0, 0
        for v_sz, a_sz in zip(vid_block_structure, aud_block_structure):
            v_next, a_next = v_ptr + v_sz, a_ptr + a_sz
            map_vid_to_aud[v_ptr : v_next] = a_next 
            map_aud_to_vid[a_ptr : a_next] = v_next 
            v_ptr, a_ptr = v_next, a_next

        pad_vid = math.ceil(total_vid_tokens / 128) * 128 - total_vid_tokens    
        pad_aud = math.ceil(total_aud_tokens / 128) * 128 - total_aud_tokens    
        def extend(t, pad, val=0): 
            if pad > 0: return torch.cat([t, torch.full((pad,), val, device=device, dtype=t.dtype)])
            return t
        vid_ends = extend(vid_ends, pad_vid)    
        aud_ends = extend(aud_ends, pad_aud)    
        map_vid_to_aud = extend(map_vid_to_aud, pad_vid)    
        map_aud_to_vid = extend(map_aud_to_vid, pad_aud)    
        
        # Config reading
        window_size_v = local_attn_size * vid_tokens_per_frame if local_attn_size != -1 else -1
        window_size_a = (local_attn_size * self.num_aud_frame_per_vid) * 1 if local_attn_size !=-1 else -1
        sink_size_v = self.sink_size_v * vid_tokens_per_frame
        sink_size_a = self.sink_size_a * 1

        def self_mask(ends, total_len, window_size, sink_size):
            def fn(b, h, q, k):
                if window_size != -1:
                    return (q < total_len) & (k < total_len) & ((k <= ends[q]) & (k >= (ends[q] - window_size)) | (k <= sink_size))
                else:
                    return (q < total_len) & (k < total_len) & ((k <= ends[q]))
            return fn
            
        def cross_mask(mapper, total_q, total_k, window_size, sink_size):
            def fn(b, h, q, k):
                if window_size != -1:
                    return (q < total_q) & (k < total_k) & ((k <= mapper[q]) & (k >= (mapper[q] - window_size)) | (k <= sink_size))
                else:
                    return (q < total_q) & (k < total_k) & (k <= mapper[q])
            return fn
            
        self.vid_self_mask = create_block_mask(self_mask(vid_ends, total_vid_tokens, window_size_v, sink_size_v), B=None, H=None, Q_LEN=total_vid_tokens+pad_vid, KV_LEN=total_vid_tokens+pad_vid, _compile=False, device=device)
        self.audio_self_mask = create_block_mask(self_mask(aud_ends, total_aud_tokens, window_size_a, sink_size_a), B=None, H=None, Q_LEN=total_aud_tokens+pad_aud, KV_LEN=total_aud_tokens+pad_aud, _compile=False, device=device)
        self.vid_cross_mask = create_block_mask(cross_mask(map_vid_to_aud, total_vid_tokens, total_aud_tokens, window_size_a, sink_size_a), B=None, H=None, Q_LEN=total_vid_tokens+pad_vid, KV_LEN=total_aud_tokens+pad_aud, _compile=False, device=device)
        self.audio_cross_mask = create_block_mask(cross_mask(map_aud_to_vid, total_aud_tokens, total_vid_tokens, window_size_v, sink_size_v), B=None, H=None, Q_LEN=total_aud_tokens+pad_aud, KV_LEN=total_vid_tokens+pad_vid, _compile=False, device=device)

        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f"Video self attention mask: {self.vid_self_mask}")
            print(f"Audio self attention mask: {self.audio_self_mask}")
            print(f"Video cross attention mask: {self.vid_cross_mask}")
            print(f"Audio cross attention mask: {self.audio_cross_mask}")
            
            import imageio
            import numpy as np
            import cv2
            from torch.nn.attention.flex_attention import create_mask

            # 1. Video Self Mask (修正：使用 vid 参数)
            # shape: [27392, 27392]
            vid_self_dense = create_mask(
                self_mask(vid_ends, total_vid_tokens, window_size_v, sink_size_v), # <--- 改为 vid
                B=None, H=None, 
                Q_LEN=total_vid_tokens+pad_vid,   # <--- 改为 vid
                KV_LEN=total_vid_tokens+pad_vid,  # <--- 改为 vid
                _compile=False, device=device
            )
            # 建议使用 INTER_NEAREST 保持二值化清晰度
            vid_self_save = cv2.resize(vid_self_dense[0, 0].cpu().float().numpy(), (1024, 1024), interpolation=cv2.INTER_NEAREST)
            imageio.imwrite(f"vid_self_mask.jpg", np.uint8(255. * vid_self_save))

            # 2. Audio Self Mask (原本就是对的)
            aud_self_dense = create_mask(
                self_mask(aud_ends, total_aud_tokens, window_size_a, sink_size_a), 
                B=None, H=None, 
                Q_LEN=total_aud_tokens+pad_aud, 
                KV_LEN=total_aud_tokens+pad_aud, 
                _compile=False, device=device
            )
            aud_self_save = cv2.resize(aud_self_dense[0, 0].cpu().float().numpy(), (1024, 1024), interpolation=cv2.INTER_NEAREST)
            imageio.imwrite(f"aud_self_mask.jpg", np.uint8(255. * aud_self_save))
            
            # 3. Video Cross Mask (原本是对的，Vid 查询 Aud)
            vid_cross_dense = create_mask(
                cross_mask(map_vid_to_aud, total_vid_tokens, total_aud_tokens, window_size_a, sink_size_a), 
                B=None, H=None, 
                Q_LEN=total_vid_tokens+pad_vid, 
                KV_LEN=total_aud_tokens+pad_aud, 
                _compile=False, device=device
            )
            vid_cross_save = cv2.resize(vid_cross_dense[0, 0].cpu().float().numpy(), (1024, 1024), interpolation=cv2.INTER_NEAREST)
            imageio.imwrite(f"vid_cross_mask.jpg", np.uint8(255. * vid_cross_save))
            
            # 4. Audio Cross Mask (修正：Aud 查询 Vid，需要用 map_aud_to_vid)
            aud_cross_dense = create_mask(
                cross_mask(map_aud_to_vid, total_aud_tokens, total_vid_tokens, window_size_v, sink_size_v), # <--- 改为 map_aud_to_vid
                B=None, H=None, 
                Q_LEN=total_aud_tokens+pad_aud,   # <--- Aud 是 Query
                KV_LEN=total_vid_tokens+pad_vid,  # <--- Vid 是 Key
                _compile=False, device=device
            )
            aud_cross_save = cv2.resize(aud_cross_dense[0, 0].cpu().float().numpy(), (1024, 1024), interpolation=cv2.INTER_NEAREST)
            imageio.imwrite(f"aud_cross_mask.jpg", np.uint8(255. * aud_cross_save))
    # -------------------------------------------------------------------------
    # Wrappers
    # -------------------------------------------------------------------------
    def prepare_transformer_block_kwargs(self, x, t, context, seq_len, is_video, first_frame_is_clean):
        # ... (Identical to causal_ovi.py)
        if is_video:
            patch_embedding, text_embedding = self.video_patch_embedding, self.video_text_embedding
            text_len, freqs, time_embedding, time_projection, dim = self.video_config['text_len'], self.video_freqs, self.video_time_embedding, self.video_time_projection, self.video_config['dim']
        else:
            patch_embedding, text_embedding = self.audio_patch_embedding, self.audio_text_embedding
            text_len, freqs, time_embedding, time_projection, dim = self.audio_config['text_len'], self.audio_freqs, self.audio_time_embedding, self.audio_time_projection, self.audio_config['dim']
        
        device = x[0].device
        if freqs.device != device:
            if is_video: self.video_freqs = freqs.to(device)
            else: self.audio_freqs = freqs.to(device)
            freqs = freqs.to(device)

        x = [patch_embedding(u.unsqueeze(0)) for u in x]
        
        if is_video:
            grid_sizes = torch.stack([torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
            x = [u.flatten(2).transpose(1, 2) for u in x]
        else:
            grid_sizes = torch.stack([torch.tensor(u.shape[1:2], dtype=torch.long) for u in x])

        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat(x)
        cur_seq_len = x.shape[1] 
        
        if t.dim() == 1:
            if first_frame_is_clean:
                t = torch.ones((t.size(0), cur_seq_len), device=t.device, dtype=t.dtype) * t.unsqueeze(1)
                _first_images_seq_len = grid_sizes[:, 1:].prod(-1)
                for i in range(t.size(0)): t[i, :_first_images_seq_len[i]] = 0
            else:
                t = t.unsqueeze(1).expand(t.size(0), cur_seq_len)
            
        bt = t.size(0)
        t_flat = t.flatten()
        f_dim = self.video_config['freq_dim'] if is_video else self.audio_config['freq_dim']
        e = time_embedding(sinusoidal_embedding_1d(f_dim, t_flat).type_as(x).unflatten(0, (bt, cur_seq_len)))
        e0 = time_projection(e).unflatten(2, (6, dim))
        
        context = text_embedding(torch.stack([torch.cat([u, u.new_zeros(text_len - u.size(0), u.size(1))]) for u in context]))
        
        return x, e, dict(e=e0, seq_lens=seq_lens, grid_sizes=grid_sizes, freqs=freqs, context=context)

    def post_transformer_block_out(self, x, grid_sizes, e, is_video):
        head = self.video_head if is_video else self.audio_head
        x = head(x, e)
        if not is_video:
            x = [u[:gs[0]] for u, gs in zip(x, grid_sizes)]
        else:
            c, patch_size = self.video_config['out_dim'], self.video_config['patch_size']
            out = []
            for u, v in zip(x, grid_sizes.tolist()):
                u = u[:math.prod(v)].view(*v, *patch_size, c)
                u = torch.einsum('fhwpqrc->cfphqwr', u)
                u = u.reshape(c, *[i * j for i, j in zip(v, patch_size)])
                out.append(u)
            x = out
        return [u.bfloat16() for u in x]

    def _forward_train(self, vid, audio, t, vid_context, audio_context, vid_seq_len, audio_seq_len, 
                       first_frame_is_clean=False, slg_layer=False, **kwargs):
        if self.vid_self_mask is None:
            self._prepare_masks(vid[0].device, vid.shape, audio.shape)
            
        vid_inp, vid_e_base, vid_kwargs = self.prepare_transformer_block_kwargs(
            x=vid, t=t, context=vid_context, seq_len=vid_seq_len, is_video=True, first_frame_is_clean=first_frame_is_clean
        )

        audio_inp, audio_e_base, audio_kwargs = self.prepare_transformer_block_kwargs(
            x=audio, t=t, context=audio_context, seq_len=audio_seq_len, is_video=False, first_frame_is_clean=False
        )
        
        all_kwargs = {
            'vid_e': vid_kwargs['e'], 'vid_seq_lens': vid_kwargs['seq_lens'], 
            'vid_grid_sizes': vid_kwargs['grid_sizes'], 'vid_freqs': vid_kwargs['freqs'], 
            'vid_context': vid_kwargs['context'],
            'audio_e': audio_kwargs['e'], 'audio_seq_lens': audio_kwargs['seq_lens'], 
            'audio_grid_sizes': audio_kwargs['grid_sizes'], 'audio_freqs': audio_kwargs['freqs'],
            'audio_context': audio_kwargs['context'],
            'vid_self_mask': self.vid_self_mask, 'audio_self_mask': self.audio_self_mask,
            'vid_cross_mask': self.vid_cross_mask, 'audio_cross_mask': self.audio_cross_mask
        }

        vid_h, audio_h = vid_inp, audio_inp
        for block in self.fusion_blocks:
            if slg_layer > 0 and i == slg_layer: continue
            vid_h, audio_h = gradient_checkpointing(
                block, 
                vid_h, 
                audio_h, 
                enabled=self.training and self.gradient_checkpointing,
                **all_kwargs
            )
                
        vid_out = self.post_transformer_block_out(vid_h, vid_kwargs['grid_sizes'], vid_e_base, True)
        audio_out = self.post_transformer_block_out(audio_h, audio_kwargs['grid_sizes'], audio_e_base, False)
        return vid_out, audio_out 

    def _forward_inference(self, vid, audio, t, vid_context, audio_context, vid_seq_len, audio_seq_len,
                           kv_cache_list=None, current_start_vid=0, current_start_audio=0,
                           first_frame_is_clean=False, slg_layer=False, **kwargs):
        vid_inp, vid_e_base, vid_kwargs = self.prepare_transformer_block_kwargs(
            x=vid, t=t, context=vid_context, seq_len=vid_seq_len, is_video=True, first_frame_is_clean=first_frame_is_clean
        )

        audio_inp, audio_e_base, audio_kwargs = self.prepare_transformer_block_kwargs(
            x=audio, t=t, context=audio_context, seq_len=audio_seq_len, is_video=False, first_frame_is_clean=False
        )
        
        all_kwargs = {
            'vid_e': vid_kwargs['e'], 'vid_seq_lens': vid_kwargs['seq_lens'], 
            'vid_grid_sizes': vid_kwargs['grid_sizes'], 'vid_freqs': vid_kwargs['freqs'], 
            'vid_context': vid_kwargs['context'],
            'audio_e': audio_kwargs['e'], 'audio_seq_lens': audio_kwargs['seq_lens'], 
            'audio_grid_sizes': audio_kwargs['grid_sizes'], 'audio_freqs': audio_kwargs['freqs'],
            'audio_context': audio_kwargs['context'],
            'current_start_vid': current_start_vid,
            'current_start_audio': current_start_audio
        }

        vid_h, audio_h = vid_inp, audio_inp
        for i, block in enumerate(self.fusion_blocks):
            if slg_layer > 0 and i == slg_layer: continue
            caches = kv_cache_list[i]
            vid_h, audio_h = gradient_checkpointing(
                block, 
                vid_h, 
                audio_h, 
                enabled=self.gradient_checkpointing,
                vid_self_cache=caches['vid_self'], audio_self_cache=caches['aud_self'],
                vid_fusion_cache=caches['vid_fusion'], audio_fusion_cache=caches['aud_fusion'],
                vid_text_cache=caches['vid_text'], audio_text_cache=caches['aud_text'],
                **all_kwargs
            )
            
        vid_out = self.post_transformer_block_out(vid_h, vid_kwargs['grid_sizes'], vid_e_base, True)
        audio_out = self.post_transformer_block_out(audio_h, audio_kwargs['grid_sizes'], audio_e_base, False)
        return vid_out, audio_out

    def forward(self, *args, **kwargs):
        if kwargs.get('kv_cache_list', None) is not None:
            return self._forward_inference(*args, **kwargs)
        else:
            return self._forward_train(*args, **kwargs)

    def load_state_dict(self, state_dict, strict=True):
        """
        Load standard Ovi weights. 
        Since we structure the model exactly like Ovi (injecting fusion into cross_attn),
        standard FSDP clean-up is sufficient. No key remapping needed.
        """
        new_state_dict = {}
        for key, value in state_dict.items():
            # Standard cleanup
            new_key = key.replace("_fsdp_wrapped_module.", "")\
                         .replace("_checkpoint_wrapped_module.", "")\
                         .replace("_orig_mod.", "")
            if new_key.startswith("model."):
                new_key = new_key[len("model."):]
            
            # Map legacy separate model keys to fusion_blocks
            if new_key.startswith("video_model.blocks."):
                parts = new_key.split('.')
                block_idx = parts[2]
                remaining = ".".join(parts[3:])
                new_key = f"fusion_blocks.{block_idx}.vid_block.{remaining}"
            elif new_key.startswith("audio_model.blocks."):
                parts = new_key.split('.')
                block_idx = parts[2]
                remaining = ".".join(parts[3:])
                new_key = f"fusion_blocks.{block_idx}.audio_block.{remaining}"
            elif new_key.startswith("video_model."):
                new_key = new_key.replace("video_model.", "video_")
            elif new_key.startswith("audio_model."):
                new_key = new_key.replace("audio_model.", "audio_")
                
            new_state_dict[new_key] = value

        return super().load_state_dict(new_state_dict, strict=strict)