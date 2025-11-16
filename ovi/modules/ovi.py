# FILE: /videogen/Wan2.2-TI2V-5B-Turbo/ovi/modules/fusion.py
# VERSION: Final Optimized FSDP-Native Refactor

import math
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
import torch.amp as amp
import torch.distributed as dist
from ovi.modules.model import (
    WanModel, WanAttentionBlock, WanLayerNorm, WanRMSNorm, rope_apply, 
    sinusoidal_embedding_1d, Head, ChannelLastConv1d, ConvMLP
)
from ovi.modules.attention import flash_attention
from ovi.distributed_comms.communications import all_gather, all_to_all_4D
from ovi.distributed_comms.parallel_states import nccl_info, get_sequence_parallel_state
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)

def gradient_checkpointing(module: nn.Module, *args, enabled: bool, **kwargs):
    # if enabled:
    #     return checkpoint(module, *args, use_reentrant=False, **kwargs)
    # else:
    #     return module(*args, **kwargs)
    return checkpoint(module, *args, use_reentrant=False, **kwargs)
        
# ========================================================================================
# FusionAttentionBlock: 保持不变
# ========================================================================================
class FusionAttentionBlock(nn.Module):
    def __init__(self, vid_block: WanAttentionBlock, audio_block: WanAttentionBlock):
        super().__init__()
        self.vid_block = vid_block
        self.audio_block = audio_block

    # ... (这个类的所有内部方法都保持不变，此处省略以保持简洁)
    def single_fusion_cross_attention_ffn_forward(self, attn_block, src_seq, src_grid_sizes, src_freqs, target_seq, target_seq_lens, target_grid_sizes, target_freqs, context, context_lens, src_e, use_sp, sp_size=None, sp_rank=None):
        cross_attn_output = self.single_fusion_cross_attention_forward(attn_block.cross_attn, attn_block.norm3(src_seq), src_grid_sizes=src_grid_sizes, src_freqs=src_freqs, target_seq=target_seq, target_seq_lens=target_seq_lens, target_grid_sizes=target_grid_sizes, target_freqs=target_freqs, context=context, context_lens=context_lens, use_sp=use_sp, sp_size=sp_size, sp_rank=sp_rank)
        src_seq = src_seq + cross_attn_output
        y = attn_block.ffn(attn_block.norm2(src_seq).bfloat16() * (1 + src_e[4].squeeze(2)) + src_e[3].squeeze(2))
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            src_seq = src_seq + y * src_e[5].squeeze(2)
        return src_seq

    def single_fusion_cross_attention_forward(self, cross_attn_block, src_seq, src_grid_sizes, src_freqs, target_seq, target_seq_lens, target_grid_sizes, target_freqs, context, context_lens, use_sp, sp_size=None, sp_rank=None):
        b, n, d = src_seq.size(0), cross_attn_block.num_heads, cross_attn_block.head_dim
        q, k, v, k_img, v_img = (None,) * 5
        if hasattr(cross_attn_block, "k_img"):
            q, k, v, k_img, v_img = cross_attn_block.qkv_fn(src_seq, context)
        else:
            q, k, v = cross_attn_block.qkv_fn(src_seq, context)
        if use_sp:
            q = all_to_all_4D(q, scatter_dim=2, gather_dim=1)
            k = torch.chunk(k, sp_size, dim=2)[sp_rank]
            v = torch.chunk(v, sp_size, dim=2)[sp_rank]
            if k_img is not None: k_img = torch.chunk(k_img, sp_size, dim=2)[sp_rank]
            if v_img is not None: v_img = torch.chunk(v_img, sp_size, dim=2)[sp_rank]
        x = flash_attention(q, k, v, k_lens=context_lens)
        if k_img is not None:
            img_x = flash_attention(q, k_img, v_img, k_lens=None)
            x = x + img_x
        target_seq = cross_attn_block.pre_attn_norm_fusion(target_seq)
        k_target = cross_attn_block.norm_k_fusion(cross_attn_block.k_fusion(target_seq)).view(b, -1, n, d)
        v_target = cross_attn_block.v_fusion(target_seq).view(b, -1, n, d)
        if use_sp:
            k_target = all_to_all_4D(k_target, scatter_dim=2, gather_dim=1)
            v_target = all_to_all_4D(v_target, scatter_dim=2, gather_dim=1)
        q = rope_apply(q, src_grid_sizes, src_freqs)
        k_target = rope_apply(k_target, target_grid_sizes, target_freqs)
        target_x = flash_attention(q, k_target, v_target, k_lens=target_seq_lens)
        x = x + target_x
        if use_sp:
            x = all_to_all_4D(x, scatter_dim=1, gather_dim=2)
        x = x.flatten(2)
        return cross_attn_block.o(x)

    def forward(self, vid, audio, **all_kwargs):
        vid_e, vid_seq_lens, vid_grid_sizes, vid_freqs, vid_context, vid_context_lens = \
            all_kwargs['vid_e'], all_kwargs['vid_seq_lens'], all_kwargs['vid_grid_sizes'], all_kwargs['vid_freqs'], all_kwargs['vid_context'], all_kwargs['vid_context_lens']
        audio_e, audio_seq_lens, audio_grid_sizes, audio_freqs, audio_context, audio_context_lens = \
            all_kwargs['audio_e'], all_kwargs['audio_seq_lens'], all_kwargs['audio_grid_sizes'], all_kwargs['audio_freqs'], all_kwargs['audio_context'], all_kwargs['audio_context_lens']
        use_sp = all_kwargs['use_sp']
        sp_size = all_kwargs.get('sp_size')
        sp_rank = all_kwargs.get('sp_rank')

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            audio_e_chunks = self.audio_block.modulation(audio_e).chunk(6, dim=2)
        audio_y = self.audio_block.self_attn(self.audio_block.norm1(audio).bfloat16() * (1 + audio_e_chunks[1].squeeze(2)) + audio_e_chunks[0].squeeze(2), audio_seq_lens, audio_grid_sizes, audio_freqs)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            audio = audio + audio_y * audio_e_chunks[2].squeeze(2)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            vid_e_chunks = self.vid_block.modulation(vid_e).chunk(6, dim=2)
        vid_y = self.vid_block.self_attn(self.vid_block.norm1(vid).bfloat16() * (1 + vid_e_chunks[1].squeeze(2)) + vid_e_chunks[0].squeeze(2), vid_seq_lens, vid_grid_sizes, vid_freqs)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            vid = vid + vid_y * vid_e_chunks[2].squeeze(2)

        og_audio = audio
        audio = self.single_fusion_cross_attention_ffn_forward(self.audio_block, audio, audio_grid_sizes, audio_freqs, vid, vid_seq_lens, vid_grid_sizes, vid_freqs, audio_context, audio_context_lens, audio_e_chunks, use_sp, sp_size, sp_rank)
        vid = self.single_fusion_cross_attention_ffn_forward(self.vid_block, vid, vid_grid_sizes, vid_freqs, og_audio, audio_seq_lens, audio_grid_sizes, audio_freqs, vid_context, vid_context_lens, vid_e_chunks, use_sp, sp_size, sp_rank)
        return vid, audio

# ========================================================================================
# FusionModel: 最终优化版，直接构建，一次性加载
# ========================================================================================
class FusionModel(nn.Module):
    def __init__(self, video_config, audio_config):
        super().__init__()
        
        self.video_config = video_config
        self.audio_config = audio_config
        
        # --- 视频组件直接初始化 ---
        vc = video_config
        self.video_patch_size = vc['patch_size']
        self.text_dim = 4096

        self.video_patch_embedding = nn.Conv3d(vc['in_dim'], vc['dim'], kernel_size=vc['patch_size'], stride=vc['patch_size'])
        self.video_text_embedding = nn.Sequential(nn.Linear(self.text_dim, vc['dim']), nn.GELU(approximate='tanh'), nn.Linear(vc['dim'], vc['dim']))
        self.video_time_embedding = nn.Sequential(nn.Linear(vc['freq_dim'], vc['dim']), nn.SiLU(), nn.Linear(vc['dim'], vc['dim']))
        self.video_time_projection = nn.Sequential(nn.SiLU(), nn.Linear(vc['dim'], vc['dim'] * 6))
        self.video_head = Head(vc['dim'], vc['out_dim'], vc['patch_size'], vc['eps'])

        # --- 音频组件直接初始化 ---
        ac = audio_config
        self.audio_patch_size = ac['patch_size']
        self.audio_patch_embedding = nn.Sequential(ChannelLastConv1d(ac['in_dim'], ac['dim'], kernel_size=7, padding=3), nn.SiLU(), ConvMLP(ac['dim'], ac['dim'] * 4, kernel_size=7, padding=3))
        self.audio_text_embedding = nn.Sequential(nn.Linear(self.text_dim, ac['dim']), nn.GELU(approximate='tanh'), nn.Linear(ac['dim'], ac['dim']))
        self.audio_time_embedding = nn.Sequential(nn.Linear(ac['freq_dim'], ac['dim']), nn.SiLU(), nn.Linear(ac['dim'], ac['dim']))
        self.audio_time_projection = nn.Sequential(nn.SiLU(), nn.Linear(ac['dim'], ac['dim'] * 6))
        self.audio_head = Head(ac['dim'], ac['out_dim'], ac['patch_size'], ac['eps'])

        # --- 构建融合 Block 列表 ---
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
        self.init_weights()
        self.set_rope_params()

        self.gradient_checkpointing = vc['gradient_checkpointing'] if 'gradient_checkpointing' in vc else False
        logger.info(f"Using gradient checkpointing: {(self.gradient_checkpointing and self.training)}") if dist.get_rank() == 0 else None
        
    # --- 以下方法是从 WanModel 复制并适配的 ---
    def prepare_transformer_block_kwargs(self, x, t, context, seq_len, clip_fea, y, first_frame_is_clean, is_video):
        # ... (此方法保持不变，此处省略)
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
        
        if self.use_sp:
            pad_size = (-x.shape[1]) % self.sp_size
            if pad_size > 0:
                x, e, e0 = [F.pad(t, (0, 0, 0, pad_size)) if t.dim() == 3 else F.pad(t, (0,0,0,0,0,pad_size)) for t in [x, e, e0]]
            x, e, e0 = [torch.chunk(t, self.sp_size, dim=1)[self.sp_rank] for t in [x, e, e0]]
            
        context = text_embedding(torch.stack([torch.cat([u, u.new_zeros(text_len - u.size(0), u.size(1))]) for u in context]))
        kwargs = dict(e=e0, seq_lens=seq_lens, grid_sizes=grid_sizes, freqs=freqs, context=context, context_lens=None)
        return x, e, kwargs

    def post_transformer_block_out(self, x, grid_sizes, e, is_video):
        # ... (此方法保持不变，此处省略)
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

    def forward(self, vid, audio, t, vid_context, audio_context, vid_seq_len, audio_seq_len,
                clip_fea=None, clip_fea_audio=None, y=None, first_frame_is_clean=False, slg_layer=False):
        # ... (此方法保持不变，此处省略)
        vid, vid_e, vid_kwargs = self.prepare_transformer_block_kwargs(x=vid, t=t, context=vid_context, seq_len=vid_seq_len, clip_fea=clip_fea, y=y, first_frame_is_clean=first_frame_is_clean, is_video=True)
        audio, audio_e, audio_kwargs = self.prepare_transformer_block_kwargs(x=audio, t=t, context=audio_context, seq_len=audio_seq_len, clip_fea=clip_fea_audio, y=None, first_frame_is_clean=False, is_video=False)
        all_kwargs = self.merge_kwargs(vid_kwargs, audio_kwargs)
        for i, fusion_block in enumerate(self.fusion_blocks):
            if slg_layer > 0 and i == slg_layer: continue
            # vid, audio = fusion_block(vid, audio, all_kwargs)
            vid, audio = gradient_checkpointing(
                enabled = (self.training and self.gradient_checkpointing),
                module = fusion_block,
                vid = vid,
                audio = audio,
                **all_kwargs
            )
        vid = self.post_transformer_block_out(vid, vid_kwargs['grid_sizes'], vid_e, is_video=True)
        audio = self.post_transformer_block_out(audio, audio_kwargs['grid_sizes'], audio_e, is_video=False)
        return vid, audio
        
    def merge_kwargs(self, vid_kwargs, audio_kwargs):
        # ... (此方法保持不变，此处省略)
        merged_kwargs = {'vid_e': vid_kwargs['e'], 'vid_seq_lens': vid_kwargs['seq_lens'], 'vid_grid_sizes': vid_kwargs['grid_sizes'], 'vid_freqs': vid_kwargs['freqs'], 'vid_context': vid_kwargs['context'], 'vid_context_lens': vid_kwargs['context_lens'], 'audio_e': audio_kwargs['e'], 'audio_seq_lens': audio_kwargs['seq_lens'], 'audio_grid_sizes': audio_kwargs['grid_sizes'], 'audio_freqs': audio_kwargs['freqs'], 'audio_context': audio_kwargs['context'], 'audio_context_lens': audio_kwargs['context_lens'], 'use_sp': self.use_sp}
        if self.use_sp: merged_kwargs.update({'sp_size': self.sp_size, 'sp_rank': self.sp_rank})
        return merged_kwargs

    def inject_cross_attention_kv_projections(self):
        # ... (此方法保持不变，此处省略)
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
        # ... (此方法保持不变，此处省略)
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
        # 需要从原始 WanModel 的静态方法或逻辑中获取 rope_params
        from ovi.modules.model import rope_params # 确保可以访问到
        vc = self.video_config
        d = vc['dim'] // vc['num_heads']
        self.video_freqs = torch.cat([rope_params(1024, d - 4 * (d // 6)), rope_params(1024, 2 * (d // 6)), rope_params(1024, 2 * (d // 6))], dim=1)
        ac = self.audio_config
        d = ac['dim'] // ac['num_heads']
        self.audio_freqs = rope_params(1024, d - 4 * (d // 6), freqs_scaling=ac['temporal_rope_scaling_factor'])