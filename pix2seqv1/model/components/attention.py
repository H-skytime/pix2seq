import math
import torch
import torch.nn as nn
from model.inference import SingleLayerKVCache
from torch.nn import functional as F


class MultiHeadAttention(nn.Module):
    def __init__(
        self, embedding_dim, num_heads, is_causal=False, bias=False, dropout=0.0
    ):
        super().__init__()
        assert embedding_dim % num_heads == 0

        self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        self.head_dim = embedding_dim // num_heads

        # 线性投影层 - separate query, key, value projections for all heads
        self.q_proj = nn.Linear(embedding_dim, embedding_dim, bias=bias)
        self.k_proj = nn.Linear(embedding_dim, embedding_dim, bias=bias)
        self.v_proj = nn.Linear(embedding_dim, embedding_dim, bias=bias)
        # 输出投影层 - output projection
        self.out_proj = nn.Linear(embedding_dim, embedding_dim, bias=bias)
        # Dropout层 - regularization
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        self.dropout = dropout
        self.is_causal = is_causal  # 标记是否为因果注意力

        # 初始化缓存为 None, 在 inference 时才会创建
        self.kv_cache = None

    def create_cache(self, max_seq_len: int = 1024):
        """Create a new KV cache for this layer.
        Should only be called during inference mode."""

        if self.training:
            raise RuntimeError("KV cache should not be created during training")
        # 实例化缓存对象
        self.kv_cache = SingleLayerKVCache(
            max_seq_len=max_seq_len,
            n_heads=self.num_heads,
            head_dim=self.head_dim,  # self.n_embd // self.n_head
        )
        # 将缓存移动到与"模型参数"相同的设备上
        self.kv_cache = self.kv_cache.to(next(self.parameters()).device)
        return self.kv_cache

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
        # 1. 线性投影 [Batch, SeqLen, Dim]
        # calculate query, key, values for all heads in batch using separate projections
        q = self.q_proj(x_for_q)
        k = self.k_proj(x_for_kv)
        v = self.v_proj(x_for_kv)

        # 2. 维度重塑与转置
        q = q.view(
            batch_size, 
            x_for_q.size(1), 
            self.num_heads, 
            embed_dim // self.num_heads
        ).transpose(1, 2)  # [batch_dim, n_head, target_seq_len, embed_dim/n_head]
        k = k.view(
            kv_batch_size,
            x_for_kv.size(1),
            self.num_heads,
            kv_embed_dim // self.num_heads,
        ).transpose(1, 2)  # [batch_dim, n_head, seq_len, embed_dim/n_head]
        v = v.view(
            kv_batch_size,
            x_for_kv.size(1),
            self.num_heads,
            kv_embed_dim // self.num_heads,
        ).transpose(1, 2)  # [batch_dim, n_head, seq_len, embed_dim/n_head]

        return q, k, v

    def _calculate_attn_mask(
        self,
        target_seq_len,
        seq_len,
        padding_mask,
        use_cache,
        encoder_input,
        batch_size,
        device,
    ):
        attn_mask = None
        should_use_causal = self.is_causal and encoder_input is None

        if padding_mask is not None:
            # Convert padding_mask to correct shape: [batch_dim, seq_len] -> [batch_dim, ..., target_seq_len, seq_len]

            if encoder_input is not None:
                # Cross attention:
                # padding_mask is [batch_dim, seq_len] for memory sequence
                # Need [batch_dim, n_head, target_seq_len, seq_len] since query and key lengths are different
                padding_mask = padding_mask.unsqueeze(1).unsqueeze(
                    1
                )  # [batch_dim, 1, 1, seq_len]
                padding_mask = padding_mask.expand(
                    batch_size, self.num_heads, target_seq_len, seq_len
                )  # [batch_dim, n_head, target_seq_len, seq_len]
                attn_mask = ~padding_mask

            else:
                # Self attention:
                if should_use_causal:
                    # Combine causal and padding masks
                    causal_mask = torch.tril(
                        torch.ones(
                            target_seq_len,
                            target_seq_len,
                            dtype=torch.bool,
                            device=padding_mask.device,
                        )
                    )
                    # padding_mask is [batch_dim, target_seq_len] for target sequence
                    # Need [B, H, target_seq_len, target_seq_len] since query and key lengths are the same
                    padding_mask = padding_mask.unsqueeze(1).unsqueeze(
                        1
                    )  # [batch_dim, 1, 1, target_seq_len]
                    padding_mask = padding_mask.expand(
                        batch_size, self.num_heads, target_seq_len, target_seq_len
                    )  # [batch_dim, n_head, target_seq_len, target_seq_len]
                    combined_mask = (
                        causal_mask.view(1, 1, target_seq_len, target_seq_len)
                        & ~padding_mask
                    )
                    attn_mask = combined_mask
                    should_use_causal = (
                        False  # We've incorporated causality into the mask
                    )
                else:
                    # just padding mask
                    # padding_mask is [batch_dim, target_seq_len] for target sequence
                    # Need [B, H, target_seq_len, target_seq_len] since query and key lengths are the same
                    padding_mask = padding_mask.unsqueeze(1).unsqueeze(
                        1
                    )  # [batch_dim, 1, 1, target_seq_len]
                    padding_mask = padding_mask.expand(
                        batch_size, self.num_heads, target_seq_len, target_seq_len
                    )  # [batch_dim, n_head, target_seq_len, target_seq_len]
                    attn_mask = ~padding_mask

        elif use_cache and should_use_causal:
            # self attention has issues with causal mask when length of q less than k/v (e.g. when using cache)
            # https://github.com/pytorch/pytorch/issues/144858
            # overcome this by generating our own causal mask
            attn_mask = torch.tril(
                torch.ones(seq_len, seq_len, dtype=torch.bool, device=device)
            )[-target_seq_len:, :]
            should_use_causal = False

        return attn_mask, should_use_causal

    def forward(self, x, encoder_input=None, padding_mask=None, use_cache=False):
        
        # batch size, sequence length, embedding dimensionality (n_embd)
        batch_size, seq_len, embed_dim = (x.size())

        # Never use cache during training
        use_cache = use_cache and not self.training and self.is_causal

        # Handle key/value inputs
        key_value_input = encoder_input if encoder_input is not None else x
        kv_batch_size, seq_len, kv_embed_dim = (
            key_value_input.size()
        )  # seq_len is source sequence length

        if use_cache:
            # 如果使用缓存, 只需要处理最后一个 Token (Query)
            # but allow full sequence input for API consistency
            last_token = x[:, -1:, :]
            x_for_kv = (
                last_token if encoder_input is None else key_value_input
            )  # Only compute K,V for last token
            x_for_q = last_token  # Only need Q for last token

        else:
            x_for_kv = key_value_input
            x_for_q = x

        # 计算 Q, K, V
        q, k, v = self._compute_qkv(
            x_for_q,
            x_for_kv,
            batch_size,
            kv_batch_size,
            embed_dim,
            kv_embed_dim,
            use_cache=use_cache,
        )

        # KV Cache 更新逻辑
        if use_cache and self.kv_cache is not None:
            if self.kv_cache.cache_pos == 0:
                # First step, just cache the k/v
                (k, v) = self.kv_cache.update(k, v)

            else:
                # Get cached states
                k, v = self.kv_cache.update(k[:, :, -1:], v[:, :, -1:])

        attn_mask = None
        should_use_causal = self.is_causal and encoder_input is None

        target_seq_len = q.size(2)  # Target sequence length
        seq_len = k.size(2)  # Source sequence length

        # 计算掩码 #
        attn_mask, should_use_causal = self._calculate_attn_mask(
            target_seq_len,
            seq_len,
            padding_mask,
            use_cache,
            encoder_input,
            batch_size,
            k.device,
        )

        # 核心注意力计算
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0,
            is_causal=should_use_causal,  # Will only be True if we have no mask at all
        )

        # 重新组合多头结果: [B, Heads, L, D_h] -> [B, L, Heads, D_h] -> [B, L, D]
        y = (y.transpose(1, 2).contiguous().view(batch_size, x_for_q.size(1), embed_dim))

        # output projection
        y = self.resid_dropout(self.out_proj(y))
        return y



class DeformableAttention(nn.Module):
    """ 多尺度局部稀疏交叉注意力模块 """

    def __init__(
        self, embedding_dim, num_heads, is_causal=False, bias=False, dropout=0.0, 
        num_points=4, num_levels=3
    ):
        super().__init__()
        assert embedding_dim % num_heads == 0
        
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads  # 注意力头数，并行建模不同注意力模式
        self.head_dim = embedding_dim // num_heads

        self.num_points = num_points  # 每个头为每个Query采样的点数 (稀疏度控制)
        self.num_levels = num_levels  # 支持的特征金字塔层数

        # 1. Query 投影
        self.q_proj = nn.Linear(embedding_dim, embedding_dim, bias=bias)

        # 2. 参考点预测 (Reference Point Prediction)
        # 输入 Query, 预测 (x, y) 坐标, 代表该 Query 关注的宏观中心
        self.reference_point_head = nn.Linear(embedding_dim, 2, bias=bias)

        num_samples = num_heads * num_levels * num_points  # 采样点总数
        # 3. 采样偏移预测 (Sampling Offset Prediction)
        # 输入 Query, 预测该 Query 所有头、所有层、所有采样点的 (x, y) 偏移量
        self.sampling_offsets = nn.Linear(embedding_dim, num_samples * 2, bias=bias)
        # 4. 注意力权重预测 (Attention Weights)
        # 输入 Query, 直接预测每个采样点的权重
        self.attention_weights = nn.Linear(embedding_dim, num_samples, bias=bias)
        
        # 5. Value 投影
        self.v_proj = nn.Linear(embedding_dim, embedding_dim, bias=bias)
        # 6. 输出投影
        self.out_proj = nn.Linear(embedding_dim, embedding_dim, bias=bias)
        
        self.dropout = nn.Dropout(dropout)

        # 执行初始化
        self.kv_cache = None
        self._reset_parameters()

    def create_cache(self, max_seq_len: int = 1024):
        """Create a new KV cache for this layer.
        与 MultiHeadAttention 一致 """
        if self.training:
            raise RuntimeError("KV cache should not be created during training")
        self.kv_cache = SingleLayerKVCache(
            max_seq_len=max_seq_len,
            n_heads=self.num_heads,
            head_dim=self.head_dim,
        )
        self.kv_cache = self.kv_cache.to(next(self.parameters()).device)
        return self.kv_cache

    def _reset_parameters(self):

        # --- 1. 初始化参考点预测头 ---
        # 初始化为 0 -> Sigmoid(0) = 0.5
        # 保证训练初期, 所有 Query 都默认关注图像正中心, 提供稳定的梯度起点
        nn.init.constant_(self.reference_point_head.weight, 0.)
        if self.reference_point_head.bias is not None:
            nn.init.constant_(self.reference_point_head.bias, 0.)

        # --- 2. 初始化采样偏移 ---
        # 初始化偏移量为 0，防止训练初期采样点跳变
        nn.init.constant_(self.sampling_offsets.weight, 0.)

        # 生成初始的扩散图案 (星星状分布)
        # 不同 heads 对应不同扩散方向, 不同 levels/points 对应不同扩散半径 # 
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (
            grid_init / grid_init.abs().max(-1, keepdim=True)[0]
        ).view(self.num_heads, 1, 1, 2).repeat(1, self.num_levels*self.num_points, 1, 1)
        for i in range(self.num_levels * self.num_points):
            grid_init[:, i, :, :] *= i + 1  # 不同采样点给予不同的扩散半径
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))

        # --- 3. 初始化注意力权重 ---
        # 初始化为0, softmax后权重均匀, 代表训练初期公平关注各采样点
        nn.init.constant_(self.attention_weights.weight, 0.)
        if self.attention_weights.bias is not None:
            nn.init.constant_(self.attention_weights.bias, 0.)

    def forward(
        self, x, encoder_input, padding_mask=None, use_cache=False,
        spatial_shapes=None, level_start_index=None
    ):
        """
        x: Decoder Query [Batch, SeqLen_Q, Dim]
        encoder_input: Encoder Feature Maps [Batch, SeqLen_V, Dim]
        """
        if use_cache:
            x = x[:, -1:, :]

        batch_size, len_q, dim = x.shape
        _, len_v, _ = encoder_input.shape

        # --- 1. 准备 Query ---
        query = self.q_proj(x)  # :[B, Lq, D]

        # --- 2. 预测参考点 ---
        reference_points = self.reference_point_head(query)  # :[B, Lq, 2]
        reference_points = reference_points.sigmoid()  # 归一化

        # 扩展维度以匹配采样点数量 - 所有采样点共享同一个参考中心，在此基础上做偏移
        reference_points = reference_points.view(
            batch_size, len_q, 1, 1, 1, 2
        ).expand(
            batch_size, len_q, self.num_heads, self.num_levels, self.num_points, 2
        )  # :[B, Lq, Heads, Levels, Points, 2]
        
        # --- 3. 预测采样偏移 ---
        offsets = self.sampling_offsets(query).view(
            batch_size, len_q, self.num_heads, self.num_levels, self.num_points, 2
        )  # [B, Lq, Heads*Levels*Points*2] -> [B, Lq, Heads, Levels, Points, 2]
        
        # --- 4. 最终采样位置 = 参考点 + 偏移量 ---
        sampling_locations = reference_points + offsets

        # --- 5. 预测注意力权重 ---
        attn_weights = self.attention_weights(query).view(
            batch_size, len_q, self.num_heads, self.num_levels * self.num_points
        )
        # 所有尺度采样点统一做 softmax
        attn_weights = F.softmax(attn_weights, dim=-1).view(
            batch_size, len_q, self.num_heads, self.num_levels, self.num_points, 1
        )  # :[B, Lq, Heads, Levels, Points, 1]

        # --- 6.  Value 投影 ---
        value = self.v_proj(encoder_input)
        value = value.view(batch_size, len_v, self.num_heads, self.head_dim)

        # --- 7. 分层局部稀疏采样 ---
        input_dtype = value.dtype 
        output = torch.zeros(batch_size, len_q, self.num_heads, self.head_dim, device=value.device, dtype=torch.float32)

        # 划分不同尺度数据
        split_sizes = [h*w for h, w in spatial_shapes]
        value_list = torch.split(value, split_sizes, dim=1)
        
        # 遍历不同尺度
        for level, (h, w) in enumerate(spatial_shapes):
            level_value = value_list[level].permute(0, 2, 3, 1).reshape(batch_size * self.num_heads, self.head_dim, h, w)
            level_sampling_grid = sampling_locations[:, :, :, level, :, :].permute(0, 2, 1, 3, 4).flatten(0, 1)
            level_sampling_grid = level_sampling_grid * 2 - 1 # 归一化到 [-1, 1]
            
            # 双线性插值采样
            sampled_value = F.grid_sample(
                level_value.to(torch.float32),          # 输入特征图: [B*Heads, HeadDim, H, W]
                level_sampling_grid.to(torch.float32),  # 采样网格：[B*Heads, Lq, Points, 2]
                mode='bilinear',                        # 双线性插值：采样点不在像素中心时，取周围4个点的加权平均，特征更平滑
                padding_mode='zeros',                   # 采样点超出特征图时，填充0
                align_corners=False
            )

            # --- 8. 加权聚合 ---
            # [B*Heads, HeadDim, Lq, Points] -> [B, Heads, HeadDim, Lq, Points] -> [B, Lq, Heads, Points, HeadDim]
            sampled_value = sampled_value.view(
                batch_size, self.num_heads, self.head_dim, len_q, self.num_points
            ).permute(0, 3, 1, 4, 2)
            level_weights = attn_weights[:, :, :, level, :]
            # 加权求和(sum over Points) :[B, Lq, Heads, HeadDim]
            output += (sampled_value * level_weights).sum(dim=3)
        
        # 拼接多头: [B, Lq, Heads * HeadDim]
        output = output.to(input_dtype).flatten(2)
        
        return self.out_proj(output)