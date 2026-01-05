# FILE: debug_ovi.py
# VERSION: Instrumented for Debugging

import math
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
import torch.amp as amp
import torch.distributed as dist
from diffusers.configuration_utils import ConfigMixin
from diffusers.models.modeling_utils import ModelMixin
import torch.nn.functional as F
import logging

from ovi.modules.model import (
    WanModel, WanAttentionBlock, WanLayerNorm, WanRMSNorm, rope_apply, 
    sinusoidal_embedding_1d, Head, ChannelLastConv1d, ConvMLP, ModulationAdd, MLPProj
)
from ovi.modules.attention import attention, flash_attention
from ovi.distributed_comms.communications import all_gather, all_to_all_4D
from ovi.distributed_comms.parallel_states import nccl_info, get_sequence_parallel_state
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

logger = logging.getLogger(__name__)

# ==============================================================================
# [DEBUG TOOL] Global Tracer
# ==============================================================================
DEBUG_TRACER = {} 
CURRENT_MODEL_TYPE = None # 'bi' or 'causal'

def trace(name, tensor):
    if CURRENT_MODEL_TYPE is None: return
    if tensor is None: return
    
    if isinstance(tensor, (list, tuple)):
        # List[C, F, H, W] -> torch.tensor([B, C, F, H, W])
        # 或者 chunk 返回的 tuple -> stacked tensor
        try:
            tensor = torch.stack(list(tensor))
        except:
            # 如果无法 stack (形状不一致等)，为了不中断流程，暂时忽略或取第一个
            # 这里为了安全起见，如果不能 stack 就不存了，或者你可以选择打印个 warning
            print(f"Warning: fail to deal with {name}.")
            return 
            
    key = name
    if CURRENT_MODEL_TYPE not in DEBUG_TRACER:
        DEBUG_TRACER[CURRENT_MODEL_TYPE] = {}
        
    # Clone & Detach to save state
    DEBUG_TRACER[CURRENT_MODEL_TYPE][key] = tensor.detach().clone()

# ==============================================================================

flex_attention = torch.compile(
    flex_attention, dynamic=False, mode="max-autotune-no-cudagraphs"
)

def gradient_checkpointing(module: nn.Module, *args, enabled: bool, **kwargs):
    return module(*args, **kwargs)

# ========================================================================================
# RoPE Utilities (Causal)
# ========================================================================================
@amp.autocast('cuda', enabled=False)
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

@amp.autocast('cuda', enabled=False)
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

@amp.autocast('cuda', enabled=False)
def causal_rope_apply(x, grid_sizes, freqs, start_index=0, tokens_per_frame=None):
    x_ndim = grid_sizes.shape[-1]
    if x_ndim == 3:
        assert tokens_per_frame is not None and tokens_per_frame > 0
        start_frame = start_index // tokens_per_frame
        return causal_rope_apply_3d(x, grid_sizes, freqs, start_frame_index=start_frame)
    else:
        return causal_rope_apply_1d(x, grid_sizes, freqs, start_index=start_index)

# ========================================================================================
# FusionAttentionBlock (Bi-Directional, with TRACE)
# ========================================================================================
class FusionAttentionBlock(nn.Module):
    def __init__(self, vid_block: WanAttentionBlock, audio_block: WanAttentionBlock):
        super().__init__()
        self.vid_block = vid_block
        self.audio_block = audio_block

    def single_fusion_cross_attention_ffn_forward(self, attn_block, src_seq, src_grid_sizes, src_freqs, target_seq, target_seq_lens, target_grid_sizes, target_freqs, context, context_lens, src_e, use_sp, sp_size=None, sp_rank=None, block_idx=None, type=None):
        cross_attn_output = self.single_fusion_cross_attention_forward(attn_block.cross_attn, attn_block.norm3(src_seq), src_grid_sizes=src_grid_sizes, src_freqs=src_freqs, target_seq=target_seq, target_seq_lens=target_seq_lens, target_grid_sizes=target_grid_sizes, target_freqs=target_freqs, context=context, context_lens=context_lens, use_sp=use_sp, sp_size=sp_size, sp_rank=sp_rank, block_idx=block_idx, type=type)
        trace(f"b{block_idx}.{type}_fusion_out", cross_attn_output)
        src_seq = src_seq + cross_attn_output
        y = attn_block.ffn(attn_block.norm2(src_seq).bfloat16() * (1 + src_e[4].squeeze(2)) + src_e[3].squeeze(2))
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            src_seq = src_seq + y * src_e[5].squeeze(2)
        return src_seq

    def single_fusion_cross_attention_forward(self, cross_attn_block, src_seq, src_grid_sizes, src_freqs, target_seq, target_seq_lens, target_grid_sizes, target_freqs, context, context_lens, use_sp, sp_size=None, sp_rank=None, block_idx=None, type=None):
        b, n, d = src_seq.size(0), cross_attn_block.num_heads, cross_attn_block.head_dim
        q, k, v, k_img, v_img = (None,) * 5
        if hasattr(cross_attn_block, "k_img"):
            q, k, v, k_img, v_img = cross_attn_block.qkv_fn(src_seq, context)
        else:
            q, k, v = cross_attn_block.qkv_fn(src_seq, context)
        # Assuming NO SP for Unit Test
        x = flash_attention(q, k, v, k_lens=context_lens)
        trace(f"b{block_idx}.{type}_text_crossattn", x)
        trace(f"b{block_idx}.{type}_text_crossattn_proj_q", q)
        if k_img is not None:
            img_x = flash_attention(q, k_img, v_img, k_lens=None)
            x = x + img_x
        target_seq = cross_attn_block.pre_attn_norm_fusion(target_seq)
        k_target = cross_attn_block.norm_k_fusion(cross_attn_block.k_fusion(target_seq)).view(b, -1, n, d)
        v_target = cross_attn_block.v_fusion(target_seq).view(b, -1, n, d)
        q = rope_apply(q, src_grid_sizes, src_freqs)
        k_target = rope_apply(k_target, target_grid_sizes, target_freqs)
        target_x = flash_attention(q, k_target, v_target, k_lens=target_seq_lens)
        x = x + target_x
        x = x.flatten(2)
        return cross_attn_block.o(x)

    def forward(self, vid, audio, block_idx=0, **all_kwargs): # Added block_idx
        vid_e, vid_seq_lens, vid_grid_sizes, vid_freqs, vid_context, vid_context_lens = \
            all_kwargs['vid_e'], all_kwargs['vid_seq_lens'], all_kwargs['vid_grid_sizes'], all_kwargs['vid_freqs'], all_kwargs['vid_context'], all_kwargs['vid_context_lens']
        audio_e, audio_seq_lens, audio_grid_sizes, audio_freqs, audio_context, audio_context_lens = \
            all_kwargs['audio_e'], all_kwargs['audio_seq_lens'], all_kwargs['audio_grid_sizes'], all_kwargs['audio_freqs'], all_kwargs['audio_context'], all_kwargs['audio_context_lens']
        use_sp = all_kwargs['use_sp']
        sp_size = all_kwargs.get('sp_size')
        sp_rank = all_kwargs.get('sp_rank')

        # [TRACE] INPUT
        trace(f"b{block_idx}.vid_in", vid)
        trace(f"b{block_idx}.aud_in", audio)

        # 1. Audio Self
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            audio_e_chunks = self.audio_block.modulation(audio_e).chunk(6, dim=2)
        trace(f"b{block_idx}.audio_e_chunks", audio_e_chunks)
        aud_norm1 = self.audio_block.norm1(audio).bfloat16() * (1 + audio_e_chunks[1].squeeze(2)) + audio_e_chunks[0].squeeze(2)
        trace(f"b{block_idx}.aud_norm1", aud_norm1)

        audio_y = self.audio_block.self_attn(aud_norm1, audio_seq_lens, audio_grid_sizes, audio_freqs)
        trace(f"b{block_idx}.audio_y", audio_y)

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            audio = audio + audio_y * audio_e_chunks[2].squeeze(2)
        trace(f"b{block_idx}.audio_self_out", audio)
        # 2. Video Self
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            vid_e_chunks = self.vid_block.modulation(vid_e).chunk(6, dim=2)
        trace(f"b{block_idx}.vid_e_chunks", vid_e_chunks)

        vid_norm1 = self.vid_block.norm1(vid).bfloat16() * (1 + vid_e_chunks[1].squeeze(2)) + vid_e_chunks[0].squeeze(2)
        trace(f"b{block_idx}.vid_norm1", vid_norm1)

        vid_y = self.vid_block.self_attn(vid_norm1, vid_seq_lens, vid_grid_sizes, vid_freqs)
        trace(f"b{block_idx}.vid_y", vid_y)

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            vid = vid + vid_y * vid_e_chunks[2].squeeze(2)

        trace(f"b{block_idx}.vid_self_out", vid)

        # 3. Cross
        og_audio = audio
        trace(f"b{block_idx}.og_audio", og_audio)

        audio = self.single_fusion_cross_attention_ffn_forward(self.audio_block, audio, audio_grid_sizes, audio_freqs, vid, vid_seq_lens, vid_grid_sizes, vid_freqs, audio_context, audio_context_lens, audio_e_chunks, use_sp, sp_size, sp_rank, block_idx, "aud")
        
        # [TRACE] Output before return
        trace(f"b{block_idx}.aud_final_fusion_out", audio)

        vid = self.single_fusion_cross_attention_ffn_forward(self.vid_block, vid, vid_grid_sizes, vid_freqs, og_audio, audio_seq_lens, audio_grid_sizes, audio_freqs, vid_context, vid_context_lens, vid_e_chunks, use_sp, sp_size, sp_rank, block_idx, "vid")
        trace(f"b{block_idx}.vid_final_fusion_out", vid)

        return vid, audio

# ========================================================================================
# FusionModel (Bi-Directional)
# ========================================================================================
class FusionModel(nn.Module):
    # ... (__init__, prepare_transformer_block_kwargs, post_transformer_block_out, merge_kwargs, inject_cross_attention_kv_projections, init_weights, set_rope_params 保持你原来的代码不变 ...)
    # 仅修改 forward 循环
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

        ac = audio_config
        self.audio_patch_size = ac['patch_size']
        self.audio_patch_embedding = nn.Sequential(ChannelLastConv1d(ac['in_dim'], ac['dim'], kernel_size=7, padding=3), nn.SiLU(), ConvMLP(ac['dim'], ac['dim'] * 4, kernel_size=7, padding=3))
        self.audio_text_embedding = nn.Sequential(nn.Linear(self.text_dim, ac['dim']), nn.GELU(approximate='tanh'), nn.Linear(ac['dim'], ac['dim']))
        self.audio_time_embedding = nn.Sequential(nn.Linear(ac['freq_dim'], ac['dim']), nn.SiLU(), nn.Linear(ac['dim'], ac['dim']))
        self.audio_time_projection = nn.Sequential(nn.SiLU(), nn.Linear(ac['dim'], ac['dim'] * 6))
        self.audio_head = Head(ac['dim'], ac['out_dim'], ac['patch_size'], ac['eps'])

        self.num_blocks = vc['num_layers']
        assert self.num_blocks == ac['num_layers']

        video_blocks = nn.ModuleList([
            WanAttentionBlock(vc['model_type'], vc['dim'], vc['ffn_dim'], vc['num_heads'], vc['window_size'], vc['qk_norm'], vc['cross_attn_norm'], vc['eps'], vc.get('additional_emb_length'))
            for _ in range(self.num_blocks)
        ])
        audio_blocks = nn.ModuleList([
            WanAttentionBlock(ac['model_type'], ac['dim'], ac['ffn_dim'], ac['num_heads'], ac['window_size'], ac['qk_norm'], ac['cross_attn_norm'], ac['eps'], ac.get('additional_emb_length'))
            for _ in range(self.num_blocks)
        ])

        self.fusion_blocks = nn.ModuleList([
            FusionAttentionBlock(video_blocks[i], audio_blocks[i])
            for i in range(self.num_blocks)
        ])
        
        self.use_sp = get_sequence_parallel_state()
        if self.use_sp:
            self.sp_size = nccl_info.sp_size
            self.sp_rank = nccl_info.rank_within_group

        self.inject_cross_attention_kv_projections()
        # self.init_weights()
        self.set_rope_params()
        self.gradient_checkpointing = vc['gradient_checkpointing'] if 'gradient_checkpointing' in vc else False

    def prepare_transformer_block_kwargs(self, x, t, context, seq_len, clip_fea, y, first_frame_is_clean, is_video):
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

        if y: x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]
        x = [patch_embedding(u.unsqueeze(0)) for u in x]
        
        if is_video:
            grid_sizes = torch.stack([torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
            x = [u.flatten(2).transpose(1, 2) for u in x]
        else:
            grid_sizes = torch.stack([torch.tensor(u.shape[1:2], dtype=torch.long) for u in x])

        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat([torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1) for u in x])
        
        if t.dim() == 1:
            if first_frame_is_clean:
                t = torch.ones((t.size(0), seq_len), device=t.device, dtype=t.dtype) * t.unsqueeze(1)
                _first_images_seq_len = grid_sizes[:, 1:].prod(-1)
                for i in range(t.size(0)): t[i, :_first_images_seq_len[i]] = 0
            else:
                t = t.unsqueeze(1).expand(t.size(0), seq_len)
        
        with amp.autocast('cuda', dtype=torch.bfloat16):
            bt = t.size(0)
            t = t.flatten()
            e = time_embedding(sinusoidal_embedding_1d(self.video_config['freq_dim'], t).unflatten(0, (bt, seq_len)).bfloat16())
            e0 = time_projection(e).unflatten(2, (6, dim))
            
        context = text_embedding(torch.stack([torch.cat([u, u.new_zeros(text_len - u.size(0), u.size(1))]) for u in context]))
        kwargs = dict(e=e0, seq_lens=seq_lens, grid_sizes=grid_sizes, freqs=freqs, context=context, context_lens=None)
        
        # [TRACE] Time Embeddings
        name = "vid" if is_video else "aud"
        trace(f"prepare_x_{name}", x)
        trace(f"prepare_e_{name}", e)
        trace(f"prepare_e0_{name}", e0)
        trace(f"prepare_seq_lens_{name}", seq_lens)
        trace(f"prepare_grid_sizes_{name}", grid_sizes)
        trace(f"prepare_freqs_{name}", freqs)
        trace(f"prepare_context_{name}", context)
        
        return x, e, kwargs

    def post_transformer_block_out(self, x, grid_sizes, e, is_video):
        head = self.video_head if is_video else self.audio_head
        x = head(x, e)
        if self.use_sp: x = all_gather(x, dim=1)
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

    def merge_kwargs(self, vid_kwargs, audio_kwargs):
        merged_kwargs = {'vid_e': vid_kwargs['e'], 'vid_seq_lens': vid_kwargs['seq_lens'], 'vid_grid_sizes': vid_kwargs['grid_sizes'], 'vid_freqs': vid_kwargs['freqs'], 'vid_context': vid_kwargs['context'], 'vid_context_lens': vid_kwargs['context_lens'], 'audio_e': audio_kwargs['e'], 'audio_seq_lens': audio_kwargs['seq_lens'], 'audio_grid_sizes': audio_kwargs['grid_sizes'], 'audio_freqs': audio_kwargs['freqs'], 'audio_context': audio_kwargs['context'], 'audio_context_lens': audio_kwargs['context_lens'], 'use_sp': self.use_sp}
        if self.use_sp: merged_kwargs.update({'sp_size': self.sp_size, 'sp_rank': self.sp_rank})
        return merged_kwargs

    def inject_cross_attention_kv_projections(self):
        for fusion_block in self.fusion_blocks:
            vid_block = fusion_block.vid_block
            vid_block.cross_attn.k_fusion = nn.Linear(vid_block.dim, vid_block.dim)
            vid_block.cross_attn.v_fusion = nn.Linear(vid_block.dim, vid_block.dim)
            vid_block.cross_attn.pre_attn_norm_fusion = WanLayerNorm(vid_block.dim, elementwise_affine=True)
            vid_block.cross_attn.norm_k_fusion = WanRMSNorm(vid_block.dim, eps=1e-6) if vid_block.qk_norm else nn.Identity()
            audio_block = fusion_block.audio_block
            audio_block.cross_attn.k_fusion = nn.Linear(audio_block.dim, audio_block.dim)
            audio_block.cross_attn.v_fusion = nn.Linear(audio_block.dim, audio_block.dim)
            audio_block.cross_attn.pre_attn_norm_fusion = WanLayerNorm(audio_block.dim, elementwise_affine=True)
            audio_block.cross_attn.norm_k_fusion = WanRMSNorm(audio_block.dim, eps=1e-6) if audio_block.qk_norm else nn.Identity()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)
        nn.init.xavier_uniform_(self.video_patch_embedding.weight.flatten(1))
        for m in self.video_text_embedding.modules():
            if isinstance(m, nn.Linear): nn.init.normal_(m.weight, std=.02)
        for m in self.video_time_embedding.modules():
            if isinstance(m, nn.Linear): nn.init.normal_(m.weight, std=.02)
        nn.init.zeros_(self.video_head.head.weight)
        for m in self.audio_text_embedding.modules():
            if isinstance(m, nn.Linear): nn.init.normal_(m.weight, std=.02)
        for m in self.audio_time_embedding.modules():
            if isinstance(m, nn.Linear): nn.init.normal_(m.weight, std=.02)
        nn.init.zeros_(self.audio_head.head.weight)
        for name, mod in self.named_modules():
            if "fusion" in name and isinstance(mod, nn.Linear):
                with torch.no_grad(): mod.weight.div_(10.0)

    def set_rope_params(self):
        from ovi.modules.model import rope_params 
        vc = self.video_config
        d = vc['dim'] // vc['num_heads']
        self.video_freqs = torch.cat([rope_params(1024, d - 4 * (d // 6)), rope_params(1024, 2 * (d // 6)), rope_params(1024, 2 * (d // 6))], dim=1)
        ac = self.audio_config
        d = ac['dim'] // ac['num_heads']
        self.audio_freqs = rope_params(1024, d - 4 * (d // 6), freqs_scaling=ac['temporal_rope_scaling_factor'])

    def forward(self, vid, audio, t, vid_context, audio_context, vid_seq_len, audio_seq_len,
                clip_fea=None, clip_fea_audio=None, y=None, first_frame_is_clean=False, slg_layer=False):
        vid, vid_e, vid_kwargs = self.prepare_transformer_block_kwargs(x=vid, t=t, context=vid_context, seq_len=vid_seq_len, clip_fea=clip_fea, y=y, first_frame_is_clean=first_frame_is_clean, is_video=True)
        audio, audio_e, audio_kwargs = self.prepare_transformer_block_kwargs(x=audio, t=t, context=audio_context, seq_len=audio_seq_len, clip_fea=clip_fea_audio, y=None, first_frame_is_clean=False, is_video=False)
        all_kwargs = self.merge_kwargs(vid_kwargs, audio_kwargs)
        for i, fusion_block in enumerate(self.fusion_blocks):
            if slg_layer > 0 and i == slg_layer: continue
            vid, audio = fusion_block(vid, audio, block_idx=i, **all_kwargs) # Pass block_idx
        
        vid = self.post_transformer_block_out(vid, vid_kwargs['grid_sizes'], vid_e, is_video=True)
        audio = self.post_transformer_block_out(audio, audio_kwargs['grid_sizes'], audio_e, is_video=False)
        return vid, audio

# ========================================================================================
# Causal Fusion Logic & Attention (With TRACE)
# ========================================================================================

def causal_fusion_logic(
    module_container, q_src, target_seq, 
    local_attn_size, sink_size,
    block_mask=None, fusion_cache=None,
    grid_sizes_src=None, freqs_src=None,
    grid_sizes_target=None, freqs_target=None,
    current_start=0, cache_start=0
):
    b, s_q, n, d = q_src.shape
    
    # 1. Projection (Weights from module_container)
    target_seq_norm = module_container.pre_attn_norm_fusion(target_seq)
    k = module_container.norm_k_fusion(module_container.k_fusion(target_seq_norm)).view(b, -1, n, d)
    v = module_container.v_fusion(target_seq_norm).view(b, -1, n, d)
    
    def get_tokens_per_frame(g): return math.prod(g[0][1:]).item() if g.shape[-1] == 3 else 1
    tpf_src = get_tokens_per_frame(grid_sizes_src)
    tpf_target = get_tokens_per_frame(grid_sizes_target)
    
    current_max_attn_size = 28160 if local_attn_size == -1 else local_attn_size * tpf_target

    if fusion_cache is not None:
        # Inference (Rolling Cache)
        q_src = causal_rope_apply(q_src, grid_sizes_src, freqs_src, start_index=current_start, tokens_per_frame=tpf_src)
        k = causal_rope_apply(k, grid_sizes_target, freqs_target, start_index=cache_start, tokens_per_frame=tpf_target)
        
        real_sink_tokens = sink_size * tpf_target
        kv_cache_size = fusion_cache["k"].shape[1]
        num_new_tokens = k.shape[1]
        current_end = cache_start + num_new_tokens
        
        # Only check for eviction if we are strictly moving forward (Append mode)
        if local_attn_size != -1 and (current_end > fusion_cache["global_end_index"].item()) and \
           (num_new_tokens + fusion_cache["local_end_index"].item() > kv_cache_size):
            # Calculate the number of new tokens added in this step
            # Shift existing cache content left to discard oldest tokens
            num_evicted_tokens = num_new_tokens + fusion_cache["local_end_index"].item() - kv_cache_size
            num_rolled_tokens = fusion_cache["local_end_index"].item() - num_evicted_tokens - real_sink_tokens
            # Clone to avoid overlapping memory error
            fusion_cache["k"][:, real_sink_tokens : real_sink_tokens + num_rolled_tokens] = \
                fusion_cache["k"][:, real_sink_tokens + num_evicted_tokens : real_sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
            fusion_cache["v"][:, real_sink_tokens : real_sink_tokens + num_rolled_tokens] = \
                fusion_cache["v"][:, real_sink_tokens + num_evicted_tokens : real_sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
            
            # Insert the new keys/values at the end
            # Overwrite (去噪): current_end == global_end_index。跳过 Eviction 判断。利用公式 local_end = local_end + (current_end - global_end) 计算出的偏移量为 0，从而实现原地覆盖。
            # Append (新Block): current_end > global_end_index。进入 Eviction 判断。
            local_end_index = fusion_cache["local_end_index"].item() + current_end - fusion_cache["global_end_index"].item() - num_evicted_tokens
            local_start_index = local_end_index - num_new_tokens
            
            fusion_cache["k"][:, local_start_index : local_end_index] = k.detach()
            fusion_cache["v"][:, local_start_index : local_end_index] = v.detach()

        else:
            # print(f'{current_start=}, {cache_start=}')
            # print(f'{fusion_cache["local_end_index"].item()=}, {current_end=}, {fusion_cache["global_end_index"].item()=}')

            # Assign new keys/values directly up to current_end
            # Overwrite (去噪): current_end == global_end_index, 利用公式 local_end = local_end + (current_end - global_end) 计算出的偏移量为 0，从而实现原地覆盖。
            # Append (新Block): current_end > global_end_index
            local_end_index = fusion_cache["local_end_index"].item() + current_end - fusion_cache["global_end_index"].item()
            local_start_index = local_end_index - num_new_tokens
            
            fusion_cache["k"][:, local_start_index : local_end_index] = k.detach()
            fusion_cache["v"][:, local_start_index : local_end_index] = v.detach()

        fusion_cache["global_end_index"].fill_(current_end)
        fusion_cache["local_end_index"].fill_(local_end_index)
        
        att_start = max(0, local_end_index - current_max_attn_size)
        # NOTE: shukai added on Dec 27, 2025. KV Cache, no matter what, should have no gradient
        # k_view = fusion_cache["k"][:, att_start:local_end_index]
        # v_view = fusion_cache["v"][:, att_start:local_end_index]
        k_view = torch.cat([fusion_cache["k"][:, att_start:local_start_index], k], dim=1)   # NOTE: shukai: att_start:local_start_index: no gradient, local_start_index:local_end_index: with gradient
        v_view = torch.cat([fusion_cache["v"][:, att_start:local_start_index], v], dim=1)   # NOTE: shukai: att_start:local_start_index: no gradient, local_start_index:local_end_index: with gradient
        # NOTE: ermu2001: debugging
        # if torch.distributed.get_rank() == 0:
        #     print(f"At causal fusion logic: {local_start_index=}, {local_end_index=}")
        #     print(f"{k_view.shape=}, {k_view.requires_grad=}, {v_view.shape=}, {v_view.requires_grad=}, {q_src.shape=}, {q_src.requires_grad=}")
        #     print(f'{fusion_cache["k"].shape=}, {fusion_cache["v"].shape=}')
        #     print(f'{fusion_cache["k"].requires_grad=}, {fusion_cache["v"].requires_grad=}')
        x = attention(q_src, k_view, v_view)
    else:
        # Training
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
        x = flex_attention(query=q_pad.transpose(2, 1), key=k_pad.transpose(2, 1), value=v_pad.transpose(2, 1), block_mask=block_mask)
        x = x.transpose(2, 1)
        if pad_len_q > 0: x = x[:, :s_q]
            
    x = x.flatten(2)
    return x

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
        self.max_attention_size = 28160 if local_attn_size == -1 else local_attn_size * 880 

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
            tokens_per_frame = 880
        else:
            tokens_per_frame = 1

        if kv_cache is not None:
            q_rope = causal_rope_apply(q, grid_sizes, freqs, start_index=current_start, tokens_per_frame=tokens_per_frame)
            k_rope = causal_rope_apply(k, grid_sizes, freqs, start_index=current_start, tokens_per_frame=tokens_per_frame)
            
            sink_tokens = self.sink_size * tokens_per_frame
            kv_cache_size = kv_cache["k"].shape[1]
            num_new_tokens = k_rope.shape[1]
            current_end = current_start + num_new_tokens
            
            if self.local_attn_size != -1 and (current_end > kv_cache["global_end_index"].item()) and \
               (num_new_tokens + kv_cache["local_end_index"].item() > kv_cache_size):
                
                num_evicted_tokens = num_new_tokens + kv_cache["local_end_index"].item() - kv_cache_size
                num_rolled_tokens = kv_cache["local_end_index"].item() - num_evicted_tokens - sink_tokens
                
                kv_cache["k"][:, sink_tokens : sink_tokens + num_rolled_tokens] = \
                    kv_cache["k"][:, sink_tokens + num_evicted_tokens : sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
                kv_cache["v"][:, sink_tokens : sink_tokens + num_rolled_tokens] = \
                    kv_cache["v"][:, sink_tokens + num_evicted_tokens : sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
                
                # Insert the new keys/values at the end
                local_end_index = kv_cache["local_end_index"].item() + current_end - \
                    kv_cache["global_end_index"].item() - num_evicted_tokens
                local_start_index = local_end_index - num_new_tokens
                
                kv_cache["k"][:, local_start_index : local_end_index] = k_rope.detach()
                kv_cache["v"][:, local_start_index : local_end_index] = v.detach()
            else:
                # Assign new keys/values directly up to current_end
                local_end_index = kv_cache["local_end_index"].item() + current_end - kv_cache["global_end_index"].item()
                local_start_index = local_end_index - num_new_tokens
                
                kv_cache["k"][:, local_start_index : local_end_index] = k_rope.detach()
                kv_cache["v"][:, local_start_index : local_end_index] = v.detach()

            kv_cache["global_end_index"].fill_(current_end)
            kv_cache["local_end_index"].fill_(local_end_index)
            
            att_start = max(0, local_end_index - self.max_attention_size)
            # NOTE: ermu2001: debugging
            # if torch.distributed.get_rank() == 0:
            #     print(f"At self attention: {local_start_index=}, {local_end_index=}")
            #     print(f"{k_rope.shape=}, {k_rope.requires_grad=}, {v.shape=}, {v.requires_grad=}, {q_rope.shape=}, {q_rope.requires_grad=}")
            #     print(f"{kv_cache['k'].shape=}, {kv_cache['v'].shape=}")
            #     print(f"{kv_cache['k'].requires_grad=}, {kv_cache['v'].requires_grad=}")
            # NOTE: shukai added on Dec 27, 2025. KV Cache, no matter what, should have no gradient
            x = attention(
                q_rope, 
                # kv_cache["k"][:, att_start:local_end_index], 
                # kv_cache["v"][:, att_start:local_end_index]
                torch.cat([kv_cache["k"][:, att_start:local_start_index], k_rope], dim=1), # NOTE: shukai: att_start:local_start_index: no gradient, local_start_index:local_end_index: with gradient
                torch.cat([kv_cache["v"][:, att_start:local_start_index], v], dim=1),      # NOTE: shukai: att_start:local_start_index: no gradient, local_start_index:local_end_index: with gradient
            )
        else:
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
        return x_out, q

class CausalWanAttentionBlock(nn.Module):
    def __init__(self, cross_attn_type, dim, ffn_dim, num_heads, local_attn_size=-1, sink_size=0, qk_norm=True, cross_attn_norm=False, eps=1e-6, additional_emb_length=None):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.qk_norm = qk_norm 
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

# ========================================================================================
# CausalFusionAttentionBlock (Causal-Model, with TRACE)
# ========================================================================================
class CausalFusionAttentionBlock(nn.Module):
    def __init__(self, vid_block: CausalWanAttentionBlock, audio_block: CausalWanAttentionBlock):
        super().__init__()
        self.vid_block = vid_block
        self.audio_block = audio_block
        
        self.vid_fusion_props = {
            'local_attn_size': audio_block.self_attn.local_attn_size,
            'sink_size': audio_block.self_attn.sink_size
        }
        self.aud_fusion_props = {
            'local_attn_size': vid_block.self_attn.local_attn_size,
            'sink_size': vid_block.self_attn.sink_size
        }

    def forward(
        self, 
        vid, audio, block_idx=0, # Added block_idx
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
        
        # [TRACE] INPUT
        trace(f"b{block_idx}.vid_in", vid)
        trace(f"b{block_idx}.aud_in", audio)

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            # 错误：注释里面没用modulation调制
            # vid_e_chunks = kwargs['vid_e'].chunk(6, dim=2)
            # audio_e_chunks = kwargs['audio_e'].chunk(6, dim=2)
            vid_e_chunks = self.vid_block.modulation(kwargs['vid_e']).chunk(6, dim=2)
            audio_e_chunks = self.audio_block.modulation(kwargs['audio_e']).chunk(6, dim=2)
        
        trace(f"b{block_idx}.vid_e_chunks", vid_e_chunks)
        trace(f"b{block_idx}.audio_e_chunks", audio_e_chunks)
        # === 1. Audio Self ===
        audio_norm = self.audio_block.norm1(audio).bfloat16() * (1 + audio_e_chunks[1].squeeze(2)) + audio_e_chunks[0].squeeze(2)
        trace(f"b{block_idx}.aud_norm1", audio_norm)

        audio_y = self.audio_block.self_attn(
            audio_norm, 
            seq_lens=kwargs['audio_seq_lens'], grid_sizes=kwargs['audio_grid_sizes'], freqs=kwargs['audio_freqs'],
            block_mask=audio_self_mask, kv_cache=audio_self_cache, current_start=cs_aud, cache_start=cache_start
        )
        trace(f"b{block_idx}.audio_y", audio_y)
        
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            audio = audio + audio_y * audio_e_chunks[2].squeeze(2)
        trace(f"b{block_idx}.audio_self_out", audio)

        # === 2. Video Self ===
        vid_norm = self.vid_block.norm1(vid).bfloat16() * (1 + vid_e_chunks[1].squeeze(2)) + vid_e_chunks[0].squeeze(2)
        trace(f"b{block_idx}.vid_norm1", vid_norm)

        vid_y = self.vid_block.self_attn(
            vid_norm,
            seq_lens=kwargs['vid_seq_lens'], grid_sizes=kwargs['vid_grid_sizes'], freqs=kwargs['vid_freqs'],
            block_mask=vid_self_mask, kv_cache=vid_self_cache, current_start=cs_vid, cache_start=cache_start
        )
        trace(f"b{block_idx}.vid_y", vid_y)

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            vid = vid + vid_y * vid_e_chunks[2].squeeze(2)
        trace(f"b{block_idx}.vid_self_out", vid)

        # === 3. Cross Attention ===
        og_audio = audio
        trace(f"b{block_idx}.og_audio", og_audio)

        # --- Audio Block Cross ---
        audio_norm_cross = self.audio_block.norm3(audio)
        audio_text_unproj, audio_q_proj = self.audio_block.cross_attn(
            audio_norm_cross, context=audio_context, context_lens=audio_context_lens, crossattn_cache=audio_text_cache 
        )
        trace(f"b{block_idx}.aud_text_crossattn", audio_text_unproj)
        trace(f"b{block_idx}.aud_text_crossattn_proj_q", audio_q_proj)

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
        trace(f"b{block_idx}.aud_fusion_out", audio_cross_out)

        # --- Video Block Cross ---
        vid_norm_cross = self.vid_block.norm3(vid)
        vid_text_unproj, vid_q_proj = self.vid_block.cross_attn(
            vid_norm_cross, context=vid_context, context_lens=vid_context_lens, crossattn_cache=vid_text_cache
        )
        trace(f"b{block_idx}.vid_text_crossattn", vid_text_unproj)
        trace(f"b{block_idx}.vid_text_crossattn_proj_q", vid_q_proj)

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
        trace(f"b{block_idx}.vid_fusion_out", vid_cross_out)

        # FFNs
        audio = audio + audio_cross_out
        audio_ffn = self.audio_block.ffn(
            self.audio_block.norm2(audio).bfloat16() * (1 + audio_e_chunks[4].squeeze(2)) + audio_e_chunks[3].squeeze(2)
        )
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            audio = audio + audio_ffn * audio_e_chunks[5].squeeze(2)
        trace(f"b{block_idx}.aud_final_fusion_out", audio)
        
        vid = vid + vid_cross_out
        vid_ffn = self.vid_block.ffn(
            self.vid_block.norm2(vid).bfloat16() * (1 + vid_e_chunks[4].squeeze(2)) + vid_e_chunks[3].squeeze(2)
        )
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            vid = vid + vid_ffn * vid_e_chunks[5].squeeze(2)
        trace(f"b{block_idx}.vid_final_fusion_out", vid)

        return vid, audio

# ========================================================================================
# CausalFusionModel (Causal-Model)
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
        
        self.inject_cross_attention_kv_projections()
        
        self.set_rope_params()
        self.gradient_checkpointing = vc.get('gradient_checkpointing', False) and ac.get('gradient_checkpointing', False)
        logger.info(f"Initialized CausalFusionModel with gradient checkpointing = {self.gradient_checkpointing}") if not dist.is_initialized() or dist.get_rank() == 0 else None
        
        self.vid_tokens_per_frame = 880
        self.aud_tokens_per_frame = 1

        self.vid_self_mask = None
        self.audio_self_mask = None
        self.vid_cross_mask = None
        self.audio_cross_mask = None

        self.independent_first_frame = False 
        self.num_frame_per_block_vid = 4 
        self.num_frame_per_block_aud = 20
        self.num_aud_frame_per_vid = self.num_frame_per_block_aud // self.num_frame_per_block_vid

    def inject_cross_attention_kv_projections(self):
        for fusion_block in self.fusion_blocks:
            vid_block = fusion_block.vid_block
            audio_block = fusion_block.audio_block
            
            vid_block.cross_attn.k_fusion = nn.Linear(vid_block.dim, vid_block.dim)
            vid_block.cross_attn.v_fusion = nn.Linear(vid_block.dim, vid_block.dim)
            vid_block.cross_attn.pre_attn_norm_fusion = WanLayerNorm(vid_block.dim, elementwise_affine=True)
            vid_block.cross_attn.norm_k_fusion = WanRMSNorm(vid_block.dim, eps=1e-6) if vid_block.qk_norm else nn.Identity()
            
            audio_block.cross_attn.k_fusion = nn.Linear(audio_block.dim, audio_block.dim)
            audio_block.cross_attn.v_fusion = nn.Linear(audio_block.dim, audio_block.dim)
            audio_block.cross_attn.pre_attn_norm_fusion = WanLayerNorm(audio_block.dim, elementwise_affine=True)
            audio_block.cross_attn.norm_k_fusion = WanRMSNorm(audio_block.dim, eps=1e-6) if audio_block.qk_norm else nn.Identity()

    def set_rope_params(self):
        from ovi.modules.model import rope_params 
        vc = self.video_config
        d = vc['dim'] // vc['num_heads']
        self.video_freqs = torch.cat([rope_params(1024, d - 4 * (d // 6)), rope_params(1024, 2 * (d // 6)), rope_params(1024, 2 * (d // 6))], dim=1)
        ac = self.audio_config
        d = ac['dim'] // ac['num_heads']
        self.audio_freqs = rope_params(1024, d - 4 * (d // 6), freqs_scaling=ac['temporal_rope_scaling_factor'])

    def init_weights(self):
        nn.init.zeros_(self.video_head.head.weight)
        nn.init.zeros_(self.audio_head.head.weight)
        for name, mod in self.named_modules():
            if ("k_fusion" in name or "v_fusion" in name) and isinstance(mod, nn.Linear):
                nn.init.xavier_uniform_(mod.weight)
                if mod.bias is not None: nn.init.zeros_(mod.bias)
            if "fusion" in name and isinstance(mod, nn.Linear):
                with torch.no_grad(): mod.weight.div_(10.0)

    @staticmethod
    def _get_block_indices(total_len, block_sizes, device):
        block_indices = torch.zeros(total_len, device=device, dtype=torch.long)
        current_idx = 0
        for i, size in enumerate(block_sizes):
            block_indices[current_idx : current_idx + size] = i
            current_idx += size
        return block_indices

    def prepare_transformer_block_kwargs(self, x, t, context, seq_len, is_video, first_frame_is_clean, clip_fea=None, y=None, proj_layer=None):
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

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

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
        else:
            tokens_per_frame = self.vid_tokens_per_frame if is_video else self.aud_tokens_per_frame
            t = t.unsqueeze(-1).repeat(1, 1, tokens_per_frame).flatten(1, 2)
            if first_frame_is_clean and is_video:
                t[:, :tokens_per_frame] = 0 
        
        with amp.autocast('cuda', dtype=torch.bfloat16):
            bt = t.size(0)
            t_flat = t.flatten()
            f_dim = self.video_config['freq_dim'] if is_video else self.audio_config['freq_dim']
            e = time_embedding(sinusoidal_embedding_1d(f_dim, t_flat).type_as(x).unflatten(0, (bt, cur_seq_len)))
            e0 = time_projection(e).unflatten(2, (6, dim))
        
        context = text_embedding(torch.stack([torch.cat([u, u.new_zeros(text_len - u.size(0), u.size(1))]) for u in context]))
        if clip_fea is not None and proj_layer is not None:
            context_clip = proj_layer(clip_fea)
            context = torch.concat([context_clip, context], dim=1)

        # [TRACE] Time Embeddings
        name = "vid" if is_video else "aud"
        trace(f"prepare_x_{name}", x)
        trace(f"prepare_e_{name}", e)
        trace(f"prepare_e0_{name}", e0)
        trace(f"prepare_seq_lens_{name}", seq_lens)
        trace(f"prepare_grid_sizes_{name}", grid_sizes)
        trace(f"prepare_freqs_{name}", freqs)
        trace(f"prepare_context_{name}", context)

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

    def _forward_inference(self, vid, audio, t_vid, t_aud, vid_context, audio_context, vid_seq_len, audio_seq_len,
                           kv_cache_list=None, current_start_vid=0, current_start_audio=0, 
                           clip_fea=None, clip_fea_audio=None, y=None, first_frame_is_clean=False, slg_layer=False, **kwargs):
        
        vid_inp, vid_e_base, vid_kwargs = self.prepare_transformer_block_kwargs(
            vid, t_vid, vid_context, vid_seq_len, True, first_frame_is_clean,
            clip_fea=clip_fea, y=y, proj_layer=getattr(self, 'video_img_emb', None)
        )
        audio_inp, audio_e_base, audio_kwargs = self.prepare_transformer_block_kwargs(
            audio, t_aud, audio_context, audio_seq_len, False, False,
            clip_fea=clip_fea_audio, y=None, proj_layer=None
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

            kwargs = dict(
                block_idx=i,
                vid_self_cache=caches['vid_self'],
                audio_self_cache=caches['aud_self'],
                vid_fusion_cache=caches['vid_fusion'],
                audio_fusion_cache=caches['aud_fusion'],
                # NOTE: 
                # vid_text_cache=caches['vid_text'],
                # audio_text_cache=caches['aud_text'],
            )
            # if torch.distributed.get_rank() == 0:
            #     print(f"{torch.is_grad_enabled()=}")
            if torch.is_grad_enabled():
                vid_h, audio_h = gradient_checkpointing(
                    block, 
                    vid_h, 
                    audio_h, 
                    enabled=self.gradient_checkpointing,
                    **kwargs,
                    **all_kwargs
                )
                # vid_h, audio_h = block(vid_h, audio_h, **kwargs, **all_kwargs)
            else:
                all_kwargs.update(dict(
                    # vid_fusion_cache=caches['vid_fusion'],
                    # audio_fusion_cache=caches['aud_fusion'],
                    vid_text_cache=caches['vid_text'],
                    audio_text_cache=caches['aud_text'],
                ))
                vid_h, audio_h = block(vid_h, audio_h, **kwargs, **all_kwargs)

            
        vid_out = self.post_transformer_block_out(vid_h, vid_kwargs['grid_sizes'], vid_e_base, True)
        audio_out = self.post_transformer_block_out(audio_h, audio_kwargs['grid_sizes'], audio_e_base, False)
        return vid_out, audio_out

    def forward(self, *args, **kwargs):
        # Only support inference path for debugging context
        return self._forward_inference(*args, **kwargs)

    def load_state_dict(self, state_dict, strict=True):
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key.replace("_fsdp_wrapped_module.", "")\
                         .replace("_checkpoint_wrapped_module.", "")\
                         .replace("_orig_mod.", "")
            if new_key.startswith("model."):
                new_key = new_key[len("model."):]
            
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