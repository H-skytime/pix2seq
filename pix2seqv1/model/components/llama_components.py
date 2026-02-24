import torch
import torch.nn as nn
from torch.nn import functional as F

from model.components.attention import MultiHeadAttention


class RoPEMultiHeadAttention(MultiHeadAttention):
    """MultiHeadAttention with Rotary Position Embeddings."""

    def __init__(
        self,
        embedding_dim,
        num_heads,
        is_causal=False,
        bias=False,
        dropout=0.0,
        q_max_seq_len=8192,  # Query 最大序列长度
        kv_max_seq_len=None,  # KV 最大序列长度
        q_rope_base=500000.0,  # Default Llama 3 RoPE theta base
        k_rope_base=None,  # Default Llama 3 RoPE theta base
    ):
        # 合法性校验: kv_max_seq_len和k_rope_base必须同时设置 / None
        if (kv_max_seq_len is None) ^ (k_rope_base is None):
            raise ValueError(
                "Both kv_max_seq_len and k_rope_base must be set together or both be None."
            )

        super().__init__(embedding_dim, num_heads, is_causal, bias, dropout)

        # 保存 RoPE 核心参数
        self.q_max_seq_len = q_max_seq_len
        self.kv_max_seq_len = kv_max_seq_len or q_max_seq_len
        self.q_rope_base = q_rope_base
        self.k_rope_base = k_rope_base or q_rope_base

        # 预计算 Query 的 RoPE 余弦/正弦值
        q_cos, q_sin = precompute_rope_params(
            head_dim=self.head_dim, context_length=q_max_seq_len, theta_base=q_rope_base
        )

        if q_max_seq_len != self.kv_max_seq_len:
            k_cos, k_sin = precompute_rope_params(
                head_dim=self.head_dim,
                context_length=self.kv_max_seq_len,
                theta_base=self.k_rope_base,
            )  # KV序列长度不同时单独预计算
        else:
            k_cos, k_sin = q_cos, q_sin

        # 注册为非可训练缓冲区
        self.register_buffer("q_cos", q_cos.clone())
        self.register_buffer("q_sin", q_sin.clone())
        self.register_buffer("k_cos", k_cos.clone())
        self.register_buffer("k_sin", k_sin.clone())

    def _compute_qkv(
        self,
        x_for_q,
        x_for_kv,
        batch_size,
        kv_batch_size,
        embed_dim,
        kv_embed_dim,
        use_cache=False,
    ):
        """ Override to add RoPE to Q/K. """

        # 1. 调用父类获取原始 Q/K/V, [B, num_head, L, head_dim]
        q, k, v = super()._compute_qkv(
            x_for_q, x_for_kv, batch_size, kv_batch_size, embed_dim, kv_embed_dim
        )

        # 2. 确定 RoPE 的起始位置
        if use_cache and self.kv_cache is not None:
            start_pos = int(self.kv_cache.cache_pos)  # 推理阶段：从缓存位置开始
        else:
            start_pos = 0  # 训练阶段：从0开始

        # 3. 生成当前序列的位置索引
        seq_len_q = q.size(2)
        seq_len_k = k.size(2)
        positions_q = torch.arange(start_pos, start_pos + seq_len_q, device=q.device)
        positions_k = torch.arange(start_pos, start_pos + seq_len_k, device=k.device)

        # 4. 应用RoPE编码到Q/K (V不需要位置编码)
        q = compute_rope(q, self.q_cos[positions_q], self.q_sin[positions_q])
        k = compute_rope(k, self.k_cos[positions_k], self.k_sin[positions_k])

        return q, k, v


class SwiGLUFFN(nn.Module):
    """Llama's SwiGLU feed-forward network"""

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)  # 映射到隐藏层
        self.w2 = nn.Linear(dim, hidden_dim, bias=False)  # 门控分支
        self.w3 = nn.Linear(hidden_dim, dim, bias=False)  # 映射回原维度
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.silu(self.w1(x)) * self.w2(x)
        x = self.dropout(x)
        return self.w3(x)


class RMSNorm(nn.Module):
    def __init__(self, emb_dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.emb_dim = emb_dim
        self.weight = nn.Parameter(torch.ones(emb_dim)).float()

    def forward(self, x):
        # 1. 计算最后一维（嵌入维度）的平方均值
        means = x.pow(2).mean(dim=-1, keepdim=True)
        # 2. 归一化：x / sqrt(平方均值 + eps)
        x_normed = x * torch.rsqrt(means + self.eps)
        # 3. 缩放 + 数据类型对齐
        return (x_normed * self.weight).to(dtype=x.dtype)


def precompute_rope_params(
    head_dim, theta_base=500_000, context_length=8192, freq_config=None
):
    """
    head_dim: 每个注意力头的维度; theta_base: RoPE基值;
    context_length: 最大序列长度; freq_config: 频率调整配置(None);
    """
    assert head_dim % 2 == 0, "Embedding dimension must be even"

    # 1. 计算逆频率 - Compute the inverse frequencies
    inv_freq = 1.0 / (
        theta_base ** (torch.arange(0, head_dim, 2)[: (head_dim // 2)].float() / head_dim)
    )

    # 频率调整 - Frequency adjustments
    if freq_config is not None:
        # 计算高低频波长阈值，对不同频率分量缩放，适配更长序列
        # calculate wavelengths for frequency bands
        low_freq_wavelen = (
            freq_config["original_context_length"] / freq_config["low_freq_factor"]
        )
        high_freq_wavelen = (
            freq_config["original_context_length"] / freq_config["high_freq_factor"]
        )

        # convert frequencies to wavelengths - 频率转波长
        wavelen = 2 * torch.pi / inv_freq

        # handle low frequency components
        inv_freq_llama = torch.where(
            wavelen > low_freq_wavelen, inv_freq / freq_config["factor"], inv_freq
        )

        # smooth transition between frequency bands
        smooth_factor = (
            freq_config["original_context_length"] / wavelen
            - freq_config["low_freq_factor"]
        ) / (freq_config["high_freq_factor"] - freq_config["low_freq_factor"])

        smoothed_inv_freq = (1 - smooth_factor) * (
            inv_freq / freq_config["factor"]
        ) + smooth_factor * inv_freq

        # Apply smoothing to medium frequencies
        is_medium_freq = (wavelen <= low_freq_wavelen) & (wavelen >= high_freq_wavelen)
        inv_freq_llama = torch.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)
        inv_freq = inv_freq_llama

    # 2. 生成位置索引
    positions = torch.arange(context_length)

    # 3. 计算每个位置、每个维度的旋转角度
    angles = (
        positions[:, None] * inv_freq[None, :]
    )  # Shape: (context_length, head_dim // 2)

    # 4. 扩展angles匹配head_dim 
    angles = torch.cat([angles, angles], dim=1)  # Shape: (context_length, head_dim)

    # 5. 预计算cos/sin
    cos = torch.cos(angles)
    sin = torch.sin(angles)

    return cos, sin


def compute_rope(x, cos, sin):
    # x: (batch_size, num_heads, seq_len, head_dim)
    batch_size, num_heads, seq_len, head_dim = x.shape
    assert head_dim % 2 == 0, "Head dimension must be even"

    # 1. 将x分成前后两部分
    x1 = x[..., : head_dim // 2]  # First half
    x2 = x[..., head_dim // 2 :]  # Second half

    # 2. 调整cos/sin的形状 - 仅取当前序列长度
    cos = cos[:seq_len, :].unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, seq_len, head_dim)
    sin = sin[:seq_len, :].unsqueeze(0).unsqueeze(0)

    # 3. 旋转变换 - Apply the rotary transformation
    rotated = torch.cat((-x2, x1), dim=-1)
    x_rotated = (x * cos) + (rotated * sin)  # 

    return x_rotated.to(dtype=x.dtype)


def estimate_rope_theta(max_context_length, embedding_dim):
    return 10000 * (max_context_length / 512) ** (64 / embedding_dim)


class LlamaEncoderBlock(nn.Module):
    """Base Llama block with self-attention"""

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        hidden_dim: int,
        context_length: int = 8192,
        rope_base: float = 500000.0,
        dropout: float = 0.0,
        bias: bool = False,
        is_causal: bool = False,
    ):
        super().__init__()

        # self.self_attn = RoPEMultiHeadAttention(
        #     embedding_dim=embedding_dim,
        #     num_heads=num_heads,
        #     q_max_seq_len=context_length,
        #     q_rope_base=rope_base,
        #     dropout=dropout,
        #     bias=bias,
        #     is_causal=is_causal,
        # )
        ## 修改为 MHA 版本
        self.self_attn = MultiHeadAttention(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            bias=bias,
            is_causal=is_causal,
        )

        self.ff = SwiGLUFFN(dim=embedding_dim, hidden_dim=hidden_dim, dropout=dropout)

        self.norm1 = RMSNorm(embedding_dim)
        self.norm2 = RMSNorm(embedding_dim)

    def forward(self, x, padding_mask=None):
        # Self-attention block with residual
        shortcut = x
        x = self.norm1(x)
        x = self.self_attn(x, padding_mask=padding_mask)
        x = x + shortcut

        # FFN block with residual
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = x + shortcut

        return x


class LlamaDecoderBlock(LlamaEncoderBlock):
    """Llama block with self-attention and cross-attention"""

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        hidden_dim: int,
        q_max_len: int = 8192,
        q_rope_base: float = 500000.0,
        kv_max_len: int = None,
        k_rope_base: float = None,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        # Initialize parent with causal masking
        super().__init__(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            context_length=q_max_len,
            rope_base=q_rope_base,
            dropout=dropout,
            bias=bias,
            is_causal=True,  # Decoder always uses causal mask
        )

        # Add cross attention components
        # self.cross_attn = RoPEMultiHeadAttention(
        #     embedding_dim=embedding_dim,
        #     num_heads=num_heads,
        #     q_max_seq_len=q_max_len,
        #     kv_max_seq_len=kv_max_len,
        #     q_rope_base=q_rope_base,
        #     k_rope_base=k_rope_base,
        #     dropout=dropout,
        #     bias=bias,
        #     is_causal=False,  # Cross attention is never causal
        # )
        ## 修改为 MHA 版本
        self.cross_attn = MultiHeadAttention(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            is_causal=False, # Cross Attention 不是因果的
            bias=bias,
            dropout=dropout,
        )


        self.norm_cross = RMSNorm(embedding_dim)

    def create_cache(self, max_seq_len: int = 1024):
        """Create a cache for self attention if it doesn't exist."""
        if self.training:
            raise RuntimeError("Cache should not be created during training")
        self.self_attn.create_cache(max_seq_len)

    def reset_cache(self):
        """Reset the cache if it exists."""
        if hasattr(self.self_attn, "kv_cache") and self.self_attn.kv_cache is not None:
            self.self_attn.kv_cache.reset()

    def forward(
        self,
        x,
        encoder_input,
        padding_mask=None,
        encoder_padding_mask=None,
        use_cache=False,
    ):
        # Self attention (with optional caching)
        shortcut = x
        x = self.norm1(x)
        x = self.self_attn(x, padding_mask=padding_mask, use_cache=use_cache)
        x = x + shortcut

        # Cross attention with encoder output
        shortcut = x
        x = self.norm_cross(x)
        x = self.cross_attn(
            x, encoder_input=encoder_input, padding_mask=encoder_padding_mask
        )
        x = x + shortcut

        # FFN
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = x + shortcut

        return x
