from wan.modules.attention import attention
from wan.modules.model import (
    WanRMSNorm,
    rope_apply,
    WanLayerNorm,
    WAN_CROSSATTENTION_CLASSES,
    rope_params,
    MLPProj,
    sinusoidal_embedding_1d
)
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
from diffusers.configuration_utils import ConfigMixin, register_to_config
from torch.nn.attention.flex_attention import BlockMask
from diffusers.models.modeling_utils import ModelMixin
import torch.nn as nn
import torch
import math
import torch.distributed as dist

# wan 1.3B model has a weird channel / head configurations and require max-autotune to work with flexattention
# see https://github.com/pytorch/pytorch/issues/133254
# change to default for other models
flex_attention = torch.compile(
    flex_attention, dynamic=False, mode="max-autotune-no-cudagraphs")


def causal_rope_apply(x, grid_sizes, freqs, start_frame=0):
    """
    与 `model.py` 的区别及意义：
    - 原版 (`rope_apply`)：假设输入 `x` 就是完整的视频，时间维度从 t=0 开始。
    - 新版 (`causal_rope_apply`)：增加了一个参数 `start_frame`。在流式推理（Autoregressive Inference）中，每次输入模型的一小块（Chunk）可能对应视频的第 10-12 帧。如果直接用原版 RoPE，模型会以为这是第 0-2 帧。必须通过 `start_frame=10` 让 RoPE 取出对应第 10 帧的位置编码，这样模型才知道“哦，我现在是在生成视频的中段”。
    """
    # 视频[B, 21, 16, 60, 104] -> [B, 21*60*104/2/2, 12, 128] = [B, 32760, 12, 128]
    # 对于每个block里面的3帧：x形状为 [Batch, Length, Heads, Dim] = [1, 4680, 12, 128]  (3帧数据)
    # 假设 start_frame: 3 (因为 Block 1 从第3帧开始)
    n, c = x.size(2), x.size(3) // 2

    # split freqs, 计算频率 split与原版一致
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []

    # 在 inference 时，grid_sizes 是当前 chunk 的大小 [f=3, h=30, w=52] 
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
            seq_len, n, -1, 2))
        
        # 关键区别在这里！
        # 原版 model.py：freqs 直接从 0 取到 f (总帧数)
        # 这里：从 freqs[0] 中切片，切片范围是 [start_frame : start_frame + f]

        # 例如我们需要取出 [3, 4, 5] 帧对应的编码，而不是 [0, 1, 2] 帧的。
        # start_frame=3, f=3 -> 切片 [3:6]
        freqs_i = torch.cat([
            freqs[0][start_frame:start_frame + f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ],
            dim=-1).reshape(seq_len, 1, -1)

        # apply rotary embedding
        # ... (应用旋转位置编码) ...
        # 这样，虽然输入张量 x 是从 index 0 开始的，但它携带的位置信息是 "我是第3-5帧"
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).type_as(x)


class CausalWanSelfAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 local_attn_size=-1,
                 sink_size=0,
                 qk_norm=True,
                 eps=1e-6):
        # 新增参数：
        # local_attn_size: 限制注意力窗口大小（滑动窗口），用于长视频生成，防止显存爆炸。
        # sink_size: "注意力汇聚点"，保留最初的几帧 Token 不被挤出 Cache，保证生成稳定性（类似 StreamingLLM）。

        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size
        self.qk_norm = qk_norm
        self.eps = eps
        self.max_attention_size = 32760 if local_attn_size == -1 else local_attn_size * 1560

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(
        self,
        x,
        seq_lens,
        grid_sizes,
        freqs,
        block_mask,
        kv_cache=None,
        current_start=0,
        cache_start=None
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
            block_mask (BlockMask)，这里的 block_mask 是外部传入的“因果掩码”
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        if cache_start is None:
            cache_start = current_start

        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        # q, k, v shape: [1, s, 12, 128]
        q, k, v = qkv_fn(x)

        # ================= 分支 1：训练模式 (无 KV Cache) =================
        # Self-forcing 的 Rollout 阶段走的是 kv_cache is not None 的推理分支；
        # Self-forcing 的梯度回传阶段（计算 Loss）走的是 kv_cache is None 且 is_tf=False 的分支（配合 block_mask）

        if kv_cache is None:
            # if it is teacher forcing training? 
            # 当且仅当输入的序列长度 s 等于 seq_lens[0] * 2 时。才会触发is_tf分支
            # 如果 s == 32760 * 2 = 66520，说明输入是 [Clean || Noisy] 拼接
            # 在Teacher Forcing 基线训练中，数据加载器会把“Clean GT 视频”和“Noisy 视频”拼接到一起输入模型。
            # 这里不会触发
            is_tf = (s == seq_lens[0].item() * 2)   
            # --- 子分支 A1: Teacher Forcing (Self-forcing 不走这里) ---
            if is_tf:
                # 这一段逻辑是为了复现论文中 "TF" baseline。
                # 逻辑：
                # 1. 把输入切两半：前半截是 Clean GT，后半截是 Noisy Input。
                q_chunk = torch.chunk(q, 2, dim=1)
                k_chunk = torch.chunk(k, 2, dim=1)
                roped_query = []
                roped_key = []
                # rope should be same for clean and noisy parts
                # 2. 独立 RoPE
                # 这一点至关重要！后半截 Noisy 视频也是从第0帧开始的。
                # 必须分别对它们做 RoPE，让它们都认为自己是第 0-20 帧。
                # 如果不做切分直接 RoPE，后半截会被认为是第 21-41 帧，那就错了。
                for ii in range(2):
                    rq = rope_apply(q_chunk[ii], grid_sizes, freqs).type_as(v)
                    rk = rope_apply(k_chunk[ii], grid_sizes, freqs).type_as(v)
                    roped_query.append(rq)
                    roped_key.append(rk)

                # 3. 拼回去 -> [1, 65520, 12, 128]
                roped_query = torch.cat(roped_query, dim=1)
                roped_key = torch.cat(roped_key, dim=1)

                padded_length = math.ceil(q.shape[1] / 128) * 128 - q.shape[1]
                padded_roped_query = torch.cat(
                    [roped_query,
                     torch.zeros([q.shape[0], padded_length, q.shape[2], q.shape[3]],
                                 device=q.device, dtype=v.dtype)],
                    dim=1
                )

                padded_roped_key = torch.cat(
                    [roped_key, torch.zeros([k.shape[0], padded_length, k.shape[2], k.shape[3]],
                                            device=k.device, dtype=v.dtype)],
                    dim=1
                )

                padded_v = torch.cat(
                    [v, torch.zeros([v.shape[0], padded_length, v.shape[2], v.shape[3]],
                                    device=v.device, dtype=v.dtype)],
                    dim=1
                )
                
                # 4. FlexAttention + 特殊 Mask
                # Mask 逻辑：Noisy Frame T 只能看 Clean Frame 0...T (且不能看 Clean Future)
                # 这样训练出来的模型学会：根据 Clean 的历史预测当前。
                x = flex_attention(
                    query=padded_roped_query.transpose(2, 1),
                    key=padded_roped_key.transpose(2, 1),
                    value=padded_v.transpose(2, 1),
                    block_mask=block_mask
                )[:, :, :-padded_length].transpose(2, 1)

                # 5. 截取
                # 输出通常只需要后半截（Noisy部分的去噪结果）

            # --- 子分支 A2: Self-forcing Backward / ODE Pretrain (走这里) ---
            else:
                # 场景：Self-forcing 计算 Loss 时
                # 输入 x: [1, 32760, 1536] (由 Rollout 生成的一整条视频)
                # 1. 标准 RoPE
                # 直接对这 32760 个 Token 加上 0-20 帧的位置编码
                roped_query = rope_apply(q, grid_sizes, freqs).type_as(v)
                roped_key = rope_apply(k, grid_sizes, freqs).type_as(v)

                padded_length = math.ceil(q.shape[1] / 128) * 128 - q.shape[1]
                padded_roped_query = torch.cat(
                    [roped_query,
                     torch.zeros([q.shape[0], padded_length, q.shape[2], q.shape[3]],
                                 device=q.device, dtype=v.dtype)],
                    dim=1
                )

                padded_roped_key = torch.cat(
                    [roped_key, torch.zeros([k.shape[0], padded_length, k.shape[2], k.shape[3]],
                                            device=k.device, dtype=v.dtype)],
                    dim=1
                )

                padded_v = torch.cat(
                    [v, torch.zeros([v.shape[0], padded_length, v.shape[2], v.shape[3]],
                                    device=v.device, dtype=v.dtype)],
                    dim=1
                )

                # 2. FlexAttention + Block Mask
                # Mask 逻辑 (_prepare_blockwise_causal_attn_mask):
                # Frame 0-2 (Block 0) -> 只能看 Block 0
                # Frame 3-5 (Block 1) -> 只能看 Block 0, 1
                # ...
                # Frame 18-20 (Block 6) -> 能看 Block 0...6

                # 为什么这叫 "Self-forcing 训练"？
                # 因为虽然是一次性并行计算，但 Mask 保证了梯度回传时，
                # Block 1 的 Loss 只会更新它对 Block 0 的依赖权重，
                # 而不会依赖 Block 2（因为看不见）。
                x = flex_attention(
                    query=padded_roped_query.transpose(2, 1),
                    key=padded_roped_key.transpose(2, 1),
                    value=padded_v.transpose(2, 1),
                    block_mask=block_mask
                )[:, :, :-padded_length].transpose(2, 1)
        # ==========================================
        # 分支 B: 推理 / Self-forcing Rollout (kv_cache is NOT None)
        # ==========================================
        else:
            # 场景：Self-forcing 在做前向生成，正在处理 Block 1 (Frame 3-5)
            # x shape: [1, 4680, 1536] (仅包含 Block 1 的 Tokens)
            # current_start: 4680 (之前 Block 0 已经有 4680 个 Token 了)

            frame_seqlen = math.prod(grid_sizes[0][1:]).item()  # 1560

            # 计算当前 Chunk 属于第几帧： 4680 // 1560 = 3 (第3帧)
            current_start_frame = current_start // frame_seqlen

            # 1. Causal RoPE
            # 使用 start_frame=3，让这 4680 个 Token 获得第 3-5 帧的位置编码
            # roped_query: [1, 4680, 12, 128]
            roped_query = causal_rope_apply(
                q, grid_sizes, freqs, start_frame=current_start_frame).type_as(v)
            roped_key = causal_rope_apply(
                k, grid_sizes, freqs, start_frame=current_start_frame).type_as(v)
            
            # 2. KV Cache 更新
            # cache["k"] 是预先分配好的大张量 [1, 32760, 12, 128]
            # global_end_index: 4680 (Cache 里目前有效数据是 Block 0)
            current_end = current_start + roped_query.shape[1]  # 4680 + 4680 = 9360
            sink_tokens = self.sink_size * frame_seqlen # 0 * 1560 = 0
            # If we are using local attention and the current KV cache size is larger than the local attention size, we need to truncate the KV cache

            # --- Rolling KV Cache 逻辑 (处理缓存更新与滚动) ---
            # 1. 判断是否需要滚动 (Cache满了)
            kv_cache_size = kv_cache["k"].shape[1]  # 
            num_new_tokens = roped_query.shape[1]   # 
            # 如果开启了 local attention 且 cache 不够放了
            if self.local_attn_size != -1 and (current_end > kv_cache["global_end_index"].item()) and (
                    num_new_tokens + kv_cache["local_end_index"].item() > kv_cache_size):
                # 执行 Cache 滚动 (Rolling/Eviction)
                # 保留 sink_tokens (开头部分)，把中间旧的挤出去，腾出空间给新 token

                # Calculate the number of new tokens added in this step
                # Shift existing cache content left to discard oldest tokens
                # Clone the source slice to avoid overlapping memory error
                num_evicted_tokens = num_new_tokens + kv_cache["local_end_index"].item() - kv_cache_size
                num_rolled_tokens = kv_cache["local_end_index"].item() - num_evicted_tokens - sink_tokens
                kv_cache["k"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                    kv_cache["k"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
                kv_cache["v"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                    kv_cache["v"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
                # Insert the new keys/values at the end
                # 计算写入位置
                # local_end_index = 4680 + 9360 - 4680 = 9360
                # local_start_index = 9360 - 4680 = 4680
                local_end_index = kv_cache["local_end_index"].item() + current_end - \
                    kv_cache["global_end_index"].item() - num_evicted_tokens
                local_start_index = local_end_index - num_new_tokens
                # 写入显存：把当前 Block 1 的 K, V 填入 Cache 的 [4680:9360] 位置 
                kv_cache["k"][:, local_start_index:local_end_index] = roped_key
                kv_cache["v"][:, local_start_index:local_end_index] = v
            else:
                # Assign new keys/values directly up to current_end
                # 显存足够，直接追加 (Append)
                local_end_index = kv_cache["local_end_index"].item() + current_end - kv_cache["global_end_index"].item()
                local_start_index = local_end_index - num_new_tokens
                kv_cache["k"][:, local_start_index:local_end_index] = roped_key
                kv_cache["v"][:, local_start_index:local_end_index] = v

             # 3. 计算 Attention
            # Query: roped_query (Block 1, 4680 tokens)
            # Key/Value: 取 Cache 的 [0:9360] (Block 0 + Block 1)
            # 这样 Block 1 就能看到 Block 0 的历史了

            x = attention(
                roped_query,
                # max(0, 9360 - 32760) = 0
                kv_cache["k"][:, max(0, local_end_index - self.max_attention_size):local_end_index],    # kv_cache["k"][:, 0:9360] 是 Block 0 + Block 1 的 K
                kv_cache["v"][:, max(0, local_end_index - self.max_attention_size):local_end_index]     # kv_cache["v"][:, 0:9360] 是 Block 0 + Block 1 的 V
            )

            # 更新 Cache 指针
            kv_cache["global_end_index"].fill_(current_end)
            kv_cache["local_end_index"].fill_(local_end_index)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class CausalWanAttentionBlock(nn.Module):

    def __init__(self,
                 cross_attn_type,
                 dim,
                 ffn_dim,
                 num_heads,
                 local_attn_size=-1,
                 sink_size=0,
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.local_attn_size = local_attn_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = CausalWanSelfAttention(dim, num_heads, local_attn_size, sink_size, qk_norm, eps)
        self.norm3 = WanLayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](dim,
                                                                      num_heads,
                                                                      (-1, -1),
                                                                      qk_norm,
                                                                      eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
        block_mask,             # 新增：传给 SelfAttn 用于训练 
        kv_cache=None,          # 新增：传给 SelfAttn 用于推理
        crossattn_cache=None,   # 新增：传给 CrossAttn 用于推理
        current_start=0,        # 新增：传给 SelfAttn 用于 RoPE
        cache_start=None
    ):
        # 原版：forward(x, e, seq_lens, ...)
        # 新版：forward(x, e, ..., block_mask, kv_cache=None, current_start=0, ...)，它将 CausalWanModel 传入的 Mask 和 Cache 信息，透传给内部的 self_attn。

        # Cache 的来源：
        # 创建：在 pipeline/self_forcing_training.py 的 inference_with_trajectory 函数中，调用 self._initialize_kv_cache 和 self._initialize_crossattn_cache 创建全零列表。
        # 传入：pipeline -> WanDiffusionWrapper -> CausalWanModel。
        # 分发：CausalWanModel._forward_inference 循环层数 block_index，把 kv_cache[block_index] 塞给第 block_index 个 Block。

        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, F, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        num_frames, frame_seqlen = e.shape[1], x.shape[1] // e.shape[1]
        # assert e.dtype == torch.float32
        # with amp.autocast(dtype=torch.float32):
        e = (self.modulation.unsqueeze(1) + e).chunk(6, dim=2)
        # assert e[0].dtype == torch.float32

        # self-attention
        y = self.self_attn(
            (self.norm1(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * (1 + e[1]) + e[0]).flatten(1, 2),
            seq_lens, grid_sizes,
            freqs, block_mask, kv_cache, current_start, cache_start)

        # with amp.autocast(dtype=torch.float32):
        x = x + (y.unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * e[2]).flatten(1, 2)

        # cross-attention & ffn function
        def cross_attn_ffn(x, context, context_lens, e, crossattn_cache=None):
            # 这里的 cross_attn 也支持 cache
            # 为什么 Cross Attn 需要 Cache？
            # 因为 Context (Text Embedding) 是不变的。
            # Block 0 算过的 Text Key/Value，Block 1 可以直接复用，不用重算。
            x = x + self.cross_attn(self.norm3(x), context,
                                    context_lens, crossattn_cache=crossattn_cache)
            y = self.ffn(
                (self.norm2(x).unflatten(dim=1, sizes=(num_frames,
                 frame_seqlen)) * (1 + e[4]) + e[3]).flatten(1, 2)
            )
            # with amp.autocast(dtype=torch.float32):
            x = x + (y.unflatten(dim=1, sizes=(num_frames,
                     frame_seqlen)) * e[5]).flatten(1, 2)
            return x

        x = cross_attn_ffn(x, context, context_lens, e, crossattn_cache)
        return x


class CausalHead(nn.Module):

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, F, 1, C]
        """
        # 在 Self-forcing 训练中，我们可能希望 Block 0 处于 t=0 (已生成完毕)，而 Block 1 处于 t=500 (正在生成)。这就是 Mixed Timesteps
        # x: [B=1, L_total=32760, Dim=1536] (训练时是一整条)
        # e: [B=1, F=21, 1, Dim=1536] (时间嵌入)
        # 注意 e 的形状！它有 F=21 个时间步，每一帧对应一个 t。
        # 在 Blockwise 训练中，Block 0 的帧对应 t=0 的嵌入，Block 1 的帧对应 t=500 的嵌入。

        # assert e.dtype == torch.float32
        # with amp.autocast(dtype=torch.float32):

        # 1. 计算维度，num_frames = 21, frame_seqlen = 32760 // 21 = 1560
        num_frames, frame_seqlen = e.shape[1], x.shape[1] // e.shape[1] 

        # 2. 调制参数 (AdaLN Zero 的一部分)
        # self.modulation: [1, 2, 1536] -> [1, 1, 2, 1536]
        # e: [1, 21, 1, 1536]
        # 相加广播 -> [1, 21, 2, 1536] -> chunk -> e[0], e[1] 均为 [1, 21, 1, 1536]
        e = (self.modulation.unsqueeze(1) + e).chunk(2, dim=2)

        # 3. 关键的 Reshape
        # 原版 Head 无法处理 "每一帧的时间步不一样" 这种情况，因为原版假定 e 是 [B, D]。
        # 这里必须把 x 拆开，跟 e 对齐。
        # x: [1, 32760, 1536] -> norm -> unflatten -> [1, 21, 1560, 1536]

        # 4. 广播乘法
        # x_reshaped: [1, 21, 1560, 1536]
        # e[1]:       [1, 21,    1, 1536]
        # 这样，第 i 帧的所有 1560 个 Token，都会乘以第 i 帧对应的时间嵌入 e[1][:, i, ...]
        x = (self.head(self.norm(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * (1 + e[1]) + e[0]))

        # 输出 x: [1, 21, 1560, Out_Dim]
        return x


class CausalWanModel(ModelMixin, ConfigMixin):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """

    ignore_for_config = [
        'patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim'
    ]
    _no_split_modules = ['WanAttentionBlock']
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(self,
                 model_type='t2v',
                 patch_size=(1, 2, 2),
                 text_len=512,
                 in_dim=16,
                 dim=2048,
                 ffn_dim=8192,
                 freq_dim=256,
                 text_dim=4096,
                 out_dim=16,
                 num_heads=16,
                 num_layers=32,
                 local_attn_size=-1,
                 sink_size=0,
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6):
        r"""
        Initialize the diffusion model backbone.

        Args:
            model_type (`str`, *optional*, defaults to 't2v'):
                Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video)
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 16):
                Input video channels (C_in)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks
            local_attn_size (`int`, *optional*, defaults to -1):
                Window size for temporal local attention (-1 indicates global attention)
            sink_size (`int`, *optional*, defaults to 0):
                Size of the attention sink, we keep the first `sink_size` frames unchanged when rolling the KV cache
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to False):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
        """

        super().__init__()

        assert model_type in ['t2v', 'i2v']
        self.model_type = model_type

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.local_attn_size = local_attn_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim))

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks
        cross_attn_type = 't2v_cross_attn' if model_type == 't2v' else 'i2v_cross_attn'
        self.blocks = nn.ModuleList([
            CausalWanAttentionBlock(cross_attn_type, dim, ffn_dim, num_heads,
                                    local_attn_size, sink_size, qk_norm, cross_attn_norm, eps)
            for _ in range(num_layers)
        ])

        # head
        self.head = CausalHead(dim, out_dim, patch_size, eps)

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = torch.cat([
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6))
        ],
            dim=1)

        if model_type == 'i2v':
            self.img_emb = MLPProj(1280, dim)

        # initialize weights
        self.init_weights()

        self.gradient_checkpointing = False

        self.block_mask = None

        self.num_frame_per_block = 1
        self.independent_first_frame = False

    def _set_gradient_checkpointing(self, module, value=False):
        self.gradient_checkpointing = value

    @staticmethod
    def _prepare_blockwise_causal_attn_mask(
        device: torch.device | str, num_frames: int = 21,
        frame_seqlen: int = 1560, num_frame_per_block=1, local_attn_size=-1
    ) -> BlockMask:
        # frame_seqlen = 1560是因为1560 = 60 * 104 / 2 / 2，表示一帧需要的token数量
        # 输入：21 帧 (Latent Space)。Block 划分：num_frame_per_block=3，所以有7个 Block。
        # Attention 行为：
        # Block 0 (Frame 0-2): 只能看 Block 0。内部是双向的（Frame 0 可以看 Frame 2）。
        # Block 1 (Frame 3-5): 可以看 Block 0 和 Block 1。
        # ...
        # Block 6 (Frame 18-20): 可以看所有 Block。
        # 训练目标：使用 ODERegression (Flow Matching Loss)，但在上述 Mask 限制下，迫使模型学会“根据过去预测未来（Block）”。

        """
        we will divide the token sequence into the following format
        [1 latent frame] [1 latent frame] ... [1 latent frame]
        We use flexattention to construct the attention mask
        """
        total_length = num_frames * frame_seqlen     # 21*1560 = 32760

        # we do right padding to get to a multiple of 128
        padded_length = math.ceil(total_length / 128) * 128 - total_length  # padded_length = 32768 - 32760 = 8

        ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)   # 形状 [32768]，用于标记每个位置能看到的"最远"位置

        # Block-wise causal mask will attend to all elements that are before the end of the current chunk
        # 核心逻辑：定义每个 token 所属的 block 结束位置
        # Frame indices 按照 block size 跳跃
        frame_indices = torch.arange(
            start=0,
            end=total_length,
            step=frame_seqlen * num_frame_per_block,    # 步长 = 单帧token数 * block大小(3) = 4680
            device=device 
        )   # 形如 [0, 4680, 9360, 14040, 18720, 23400, 28080]，表示每个 block 的起始位置

        # 填充 ends 数组，标记每个位置能看到的"最远"位置
        for tmp in frame_indices:   # 遍历0，4680，9360，...
            # 当前 block 内的所有 token，其可视范围截止到当前 block 的末尾
            ends[tmp:tmp + frame_seqlen * num_frame_per_block] = tmp + frame_seqlen * num_frame_per_block   # 标记每个 token 所属的 block 结束位置，例如ends[0:4680] = 4680, ends[4680:9360] = 9360, ...

        def attention_mask(b, h, q_idx, kv_idx):
            if local_attn_size == -1:
                # local_attn_size == -1 表示全局注意力
                # 1. kv_idx < ends[q_idx]: 如果 Query 在 Block N，ends[q_idx] 就是 Block N 的结尾。这意味着它只能看到 Block 0 到 Block N 的所有 Key/Value。它看不到 Block N+1 及以后的内容。
                # 2. (q_idx == kv_idx): 自身对角线
                
                return (kv_idx < ends[q_idx]) | (q_idx == kv_idx)   # 这样做出来的attention mask是blockwise因果的
            else:
                return ((kv_idx < ends[q_idx]) & (kv_idx >= (ends[q_idx] - local_attn_size * frame_seqlen))) | (q_idx == kv_idx)
            # return ((kv_idx < total_length) & (q_idx < total_length))  | (q_idx == kv_idx) # bidirectional mask

        # 创建 FlexAttention 的 BlockMask
        block_mask = create_block_mask(attention_mask, B=None, H=None, Q_LEN=total_length + padded_length,
                                       KV_LEN=total_length + padded_length, _compile=False, device=device)

        import torch.distributed as dist
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(
                f" cache a block wise causal mask with block size of {num_frame_per_block} frames")
            print(block_mask)

        # import imageio
        # import numpy as np
        # from torch.nn.attention.flex_attention import create_mask

        # mask = create_mask(attention_mask, B=None, H=None, Q_LEN=total_length +
        #                    padded_length, KV_LEN=total_length + padded_length, device=device)
        # import cv2
        # mask = cv2.resize(mask[0, 0].cpu().float().numpy(), (1024, 1024))
        # imageio.imwrite("mask_%d.jpg" % (0), np.uint8(255. * mask))

        return block_mask

    @staticmethod
    def _prepare_teacher_forcing_mask(
        device: torch.device | str, num_frames: int = 21,
        frame_seqlen: int = 1560, num_frame_per_block=1
    ) -> BlockMask:
        """
        we will divide the token sequence into the following format
        [1 latent frame] [1 latent frame] ... [1 latent frame]
        We use flexattention to construct the attention mask
        """
        # debug
        DEBUG = False
        if DEBUG:
            num_frames = 9
            frame_seqlen = 256

        total_length = num_frames * frame_seqlen * 2

        # we do right padding to get to a multiple of 128
        padded_length = math.ceil(total_length / 128) * 128 - total_length

        clean_ends = num_frames * frame_seqlen
        # for clean context frames, we can construct their flex attention mask based on a [start, end] interval
        context_ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        # for noisy frames, we need two intervals to construct the flex attention mask [context_start, context_end] [noisy_start, noisy_end]
        noise_context_starts = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        noise_context_ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        noise_noise_starts = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        noise_noise_ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)

        # Block-wise causal mask will attend to all elements that are before the end of the current chunk
        attention_block_size = frame_seqlen * num_frame_per_block
        frame_indices = torch.arange(
            start=0,
            end=num_frames * frame_seqlen,
            step=attention_block_size,
            device=device, dtype=torch.long
        )

        # attention for clean context frames
        for start in frame_indices:
            context_ends[start:start + attention_block_size] = start + attention_block_size

        noisy_image_start_list = torch.arange(
            num_frames * frame_seqlen, total_length,
            step=attention_block_size,
            device=device, dtype=torch.long
        )
        noisy_image_end_list = noisy_image_start_list + attention_block_size

        # attention for noisy frames
        for block_index, (start, end) in enumerate(zip(noisy_image_start_list, noisy_image_end_list)):
            # attend to noisy tokens within the same block
            noise_noise_starts[start:end] = start
            noise_noise_ends[start:end] = end
            # attend to context tokens in previous blocks
            # noise_context_starts[start:end] = 0
            noise_context_ends[start:end] = block_index * attention_block_size

        def attention_mask(b, h, q_idx, kv_idx):
            # first design the mask for clean frames
            clean_mask = (q_idx < clean_ends) & (kv_idx < context_ends[q_idx])
            # then design the mask for noisy frames
            # noisy frames will attend to all clean preceeding clean frames + itself
            C1 = (kv_idx < noise_noise_ends[q_idx]) & (kv_idx >= noise_noise_starts[q_idx])
            C2 = (kv_idx < noise_context_ends[q_idx]) & (kv_idx >= noise_context_starts[q_idx])
            noise_mask = (q_idx >= clean_ends) & (C1 | C2)

            eye_mask = q_idx == kv_idx
            return eye_mask | clean_mask | noise_mask

        block_mask = create_block_mask(attention_mask, B=None, H=None, Q_LEN=total_length + padded_length,
                                       KV_LEN=total_length + padded_length, _compile=False, device=device)

        if DEBUG:
            print(block_mask)
            import imageio
            import numpy as np
            from torch.nn.attention.flex_attention import create_mask

            mask = create_mask(attention_mask, B=None, H=None, Q_LEN=total_length +
                               padded_length, KV_LEN=total_length + padded_length, device=device)
            import cv2
            mask = cv2.resize(mask[0, 0].cpu().float().numpy(), (1024, 1024))
            imageio.imwrite("mask_%d.jpg" % (0), np.uint8(255. * mask))

        return block_mask

    @staticmethod
    def _prepare_blockwise_causal_attn_mask_i2v(
        device: torch.device | str, num_frames: int = 21,
        frame_seqlen: int = 1560, num_frame_per_block=4, local_attn_size=-1
    ) -> BlockMask:
        """
        we will divide the token sequence into the following format
        [1 latent frame] [N latent frame] ... [N latent frame]
        The first frame is separated out to support I2V generation
        We use flexattention to construct the attention mask
        """
        

        total_length = num_frames * frame_seqlen

        # we do right padding to get to a multiple of 128
        padded_length = math.ceil(total_length / 128) * 128 - total_length

        ends = torch.zeros(total_length + padded_length,
                           device=device, dtype=torch.long)

        # special handling for the first frame
        # 关键点：特殊处理第一帧
        # frame_seqlen 是第一帧的 Token 数
        # ends[:frame_seqlen] = frame_seqlen 表示：
        # 第一帧内部的所有 Token，都可以看到第一帧的所有 Token (全可见)。

        ends[:frame_seqlen] = frame_seqlen

        # Block-wise causal mask will attend to all elements that are before the end of the current chunk
        # 后续帧的处理：
        # frame_indices 从 start = frame_seqlen 表示后续帧的处理从第二帧开始
        frame_indices = torch.arange(
            start=frame_seqlen,
            end=total_length,
            step=frame_seqlen * num_frame_per_block,
            device=device
        )

        for idx, tmp in enumerate(frame_indices):
            ends[tmp:tmp + frame_seqlen * num_frame_per_block] = tmp + \
                frame_seqlen * num_frame_per_block

        def attention_mask(b, h, q_idx, kv_idx):
            # kv_idx < ends[q_idx]: 
            # 后面帧的 Query 查表得到 ends，这个 ends 肯定包含了 frame_seqlen (第一帧的范围)
            # 所以后面所有帧都能看到第一帧。
            if local_attn_size == -1:
                return (kv_idx < ends[q_idx]) | (q_idx == kv_idx)
            else:
                return ((kv_idx < ends[q_idx]) & (kv_idx >= (ends[q_idx] - local_attn_size * frame_seqlen))) | \
                    (q_idx == kv_idx)

        block_mask = create_block_mask(attention_mask, B=None, H=None, Q_LEN=total_length + padded_length,
                                       KV_LEN=total_length + padded_length, _compile=False, device=device)

        if not dist.is_initialized() or dist.get_rank() == 0:
            print(
                f" cache a block wise causal mask with block size of {num_frame_per_block} frames")
            print(block_mask)

        # import imageio
        # import numpy as np
        # from torch.nn.attention.flex_attention import create_mask

        # mask = create_mask(attention_mask, B=None, H=None, Q_LEN=total_length +
        #                    padded_length, KV_LEN=total_length + padded_length, device=device)
        # import cv2
        # mask = cv2.resize(mask[0, 0].cpu().float().numpy(), (1024, 1024))
        # imageio.imwrite("mask_%d.jpg" % (0), np.uint8(255. * mask))

        return block_mask

    # Training Mode (_forward_train)：并行计算，利用 Mask 实现“伪”自回归（用于 ODE Pretrain 和 Self-forcing 的 Loss 计算）。
    # Inference Mode (_forward_inference)：串行计算，利用 KV Cache 实现“真”自回归（用于 Self-forcing 的 Rollout 生成和最终推理）。

    # Self-forcing训练时backward simulation (Rollout)的时候走 _forward_inference。输入是一小块 x (Chunk) 和 kv_cache。
    # x shape: [B, L_total, C] = [B, 3*1560, 1536] = [B, 4680, 1536] (3帧，每帧1560个token)
    # t shape: [B, Chunk_Frames] = [B, 3] (3帧的时间步)
    def _forward_inference(
        self,
        x,
        t,
        context,
        seq_len,
        clip_fea=None,
        y=None,
        kv_cache: dict = None,          # 必须有
        crossattn_cache: dict = None,   # 必须有
        current_start: int = 0,         # 当前 Chunk 在全局视频中的起始 Token 索引
        cache_start: int = 0
    ):
        r"""
        Run the diffusion model with kv caching.
        See Algorithm 2 of CausVid paper https://arxiv.org/abs/2412.07772 for details.
        This function will be run for num_frame times.
        Process the latent frames one by one (1560 tokens each)

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """

        if self.model_type == 'i2v':
            assert clip_fea is not None and y is not None
        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat(x)
        """
        torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                      dim=1) for u in x
        ])
        """

        # time embeddings
        # with amp.autocast(dtype=torch.float32):

        # 3. Time Embedding
        # t 是 [B, Chunk_Frames]，即当前 Chunk 每帧对应的时间步
        # Self-forcing 中，Block 0 可能是 0，Block 1 可能是 500
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t.flatten()).type_as(x))
        e0 = self.time_projection(e).unflatten(
            1, (6, self.dim)).unflatten(dim=0, sizes=t.shape)
        # assert e.dtype == torch.float32 and e0.dtype == torch.float32

        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))

        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
            context = torch.concat([context_clip, context], dim=1)

        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
            block_mask=self.block_mask
        )

        def create_custom_forward(module):
            def custom_forward(*inputs, **kwargs):
                return module(*inputs, **kwargs)
            return custom_forward

        for block_index, block in enumerate(self.blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                # 关键：从列表中取出【当前层】的 Cache 字典
                # 传入 current_start 用于 RoPE 对齐
                kwargs.update(
                    {
                        "kv_cache": kv_cache[block_index],
                        "current_start": current_start,
                        "cache_start": cache_start
                    }
                )
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x, **kwargs,
                    use_reentrant=False,
                )
            else:
                # 关键：从列表中取出【当前层】的 Cache 字典
                # 传入 current_start 用于 RoPE 对齐
                kwargs.update(
                    {
                        "kv_cache": kv_cache[block_index],
                        "crossattn_cache": crossattn_cache[block_index],
                        "current_start": current_start,
                        "cache_start": cache_start
                    }
                )
                # 调用 Block，Block 内部调用 CausalWanSelfAttention
                # 注意：这里没有传 block_mask，因为用 Cache 实现了物理上的因果
                x = block(x, **kwargs)

        # head
        x = self.head(x, e.unflatten(dim=0, sizes=t.shape).unsqueeze(2))
        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return torch.stack(x)

    # ODE Pretrain 时走 _forward_train 路径。输入是完整的 [B, C, 21, H, W]。
    def _forward_train(
        self,
        x,
        t,
        context,
        seq_len,
        clean_x=None,   # 如果是 Teacher Forcing，这里会传入 Clean GT
        aug_t=None,     # Teacher Forcing 时 Clean 部分的时间步
        clip_fea=None,  # I2V 参数
        y=None,
    ):
        r"""
        Forward pass through the diffusion model

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """
        if self.model_type == 'i2v':
            assert clip_fea is not None and y is not None
        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        # Construct blockwise causal attn mask
        # 1. 准备 Mask (如果还没缓存)
        if self.block_mask is None:
            if clean_x is not None:
                # Teacher Forcing 模式
                if self.independent_first_frame:
                    raise NotImplementedError()
                else:
                    self.block_mask = self._prepare_teacher_forcing_mask(
                        device, num_frames=x.shape[2],
                        frame_seqlen=x.shape[-2] * x.shape[-1] // (self.patch_size[1] * self.patch_size[2]),
                        num_frame_per_block=self.num_frame_per_block
                    )
            else:
                # ODE / Self-forcing 模式
                # 根据是否是 I2V (independent_first_frame) 选择不同的 Mask 生成器
                if self.independent_first_frame:
                    self.block_mask = self._prepare_blockwise_causal_attn_mask_i2v(
                        device, num_frames=x.shape[2],
                        frame_seqlen=x.shape[-2] * x.shape[-1] // (self.patch_size[1] * self.patch_size[2]),
                        num_frame_per_block=self.num_frame_per_block,
                        local_attn_size=self.local_attn_size
                    )
                else:
                    self.block_mask = self._prepare_blockwise_causal_attn_mask(
                        device, num_frames=x.shape[2],
                        frame_seqlen=x.shape[-2] * x.shape[-1] // (self.patch_size[1] * self.patch_size[2]),
                        num_frame_per_block=self.num_frame_per_block,
                        local_attn_size=self.local_attn_size
                    )
        # 2. 处理 I2V 的 Concat 输入 (如果是 Wan2.1-I2V那种Concat 模式)
        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]

        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]

        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_lens[0] - u.size(1), u.size(2))],
                      dim=1) for u in x
        ])

        # time embeddings
        # with amp.autocast(dtype=torch.float32):

        # 4. Time Embedding
        # 这里的 t 是 [B, 21]，被 flatten 成 [B*21]
        # e 最终形状: [B, 21, 1, Dim] (经过 reshape)
        # 这里生成了每帧独立的时间嵌入。第 0 帧有 t=0 的嵌入，第 1 帧有 t=1000 的嵌入，第4帧有 t=937.5 的嵌入，等等...
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t.flatten()).type_as(x))
        e0 = self.time_projection(e).unflatten(
            1, (6, self.dim)).unflatten(dim=0, sizes=t.shape)
        # assert e.dtype == torch.float32 and e0.dtype == torch.float32

        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))

        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
            context = torch.concat([context_clip, context], dim=1)

        # 6. 处理 Teacher Forcing 的拼接 (如果有 clean_x)
        if clean_x is not None:
            # 将 Clean Latent 也 Embedding，拼接到 x 前面
            # 构造 [Clean || Noisy] 的长序列
            clean_x = [self.patch_embedding(u.unsqueeze(0)) for u in clean_x]
            clean_x = [u.flatten(2).transpose(1, 2) for u in clean_x]

            seq_lens_clean = torch.tensor([u.size(1) for u in clean_x], dtype=torch.long)
            assert seq_lens_clean.max() <= seq_len
            clean_x = torch.cat([
                torch.cat([u, u.new_zeros(1, seq_lens_clean[0] - u.size(1), u.size(2))], dim=1) for u in clean_x
            ])

            x = torch.cat([clean_x, x], dim=1)
            if aug_t is None:
                aug_t = torch.zeros_like(t)
            e_clean = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, aug_t.flatten()).type_as(x))
            e0_clean = self.time_projection(e_clean).unflatten(
                1, (6, self.dim)).unflatten(dim=0, sizes=t.shape)
            # 拼接对应的时间嵌入 (Clean 部分通常 t=0 或 aug_t)
            e0 = torch.cat([e0_clean, e0], dim=1)

        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
            block_mask=self.block_mask)

        def create_custom_forward(module):
            def custom_forward(*inputs, **kwargs):
                return module(*inputs, **kwargs)
            return custom_forward

        for block in self.blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x, **kwargs,
                    use_reentrant=False,
                )
            else:
                # 传入 block_mask，在 Attention 内部实现因果遮蔽
                x = block(x, **kwargs)

        # 8. 如果是 Teacher Forcing，只保留后半部分 (Noisy 部分) 的输出
        if clean_x is not None:
            x = x[:, x.shape[1] // 2:]

        # 9. 输出 Head
        # 传入 e (时间嵌入)，CausalHead 会处理广播
        x = self.head(x, e.unflatten(dim=0, sizes=t.shape).unsqueeze(2))

        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return torch.stack(x)

    def forward(
        self,
        *args,
        **kwargs
    ):
        # 判据：是否传入了 kv_cache？
        # 如果有 cache，说明在做流式生成 (Rollout / Inference) -> 走 _forward_inference
        # 如果没 cache，说明在做并行训练 (Training) -> 走 _forward_train
        if kwargs.get('kv_cache', None) is not None:
            return self._forward_inference(*args, **kwargs)
        else:
            return self._forward_train(*args, **kwargs)

    def unpatchify(self, x, grid_sizes):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[Tensor]:
                Reconstructed video tensors with shape [C_out, F, H / 8, W / 8]
        """

        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

    def init_weights(self):
        r"""
        Initialize model parameters using Xavier initialization.
        """

        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)

        # init output layer
        nn.init.zeros_(self.head.head.weight)
