"""PyTorch dataset implementation for Pix2Seq object detection and segmentation.

This module provides a modular implementation of the Pix2Seq data processing pipeline,
with support for different base datasets and customizable augmentation strategies.

Key components:
- Base dataset interface for different detection datasets
- Image preprocessing and normalization
- Image and box augmentation pipelines
- Token sequence generation for training
"""

from enum import Enum
from typing import Iterable, List, Optional, Tuple, Union

import torch
import numpy as np


class LabelCorruptionStrategy(Enum):
    # 标签扰动策略枚举
    NONE = "none"  # 保留原始标签 不扰动
    RANDOM = "rand_cls"  # 50%原始标签 + 50%随机有效类别
    RANDOM_AND_FAKE = "rand_n_fake_cls"  # 50%原始 + 25%随机 + 25%伪造


class TokenProcessor:
    """ 把检测框, 类别和分割掩码转成 Pix2Seq 的 token 序列

    Sequence format (Detection):
    [DET] [y1 x1 y2 x2 c1] [y1 x1 y2 x2 c2] ... [EOS] [PAD]

    Sequence format (Segmentation):
    [SEG] [y1 x1 y2 x2 c1] [p1_y p1_x ... pn_y pn_x] ... [EOS]
    (其中 Box 和 Class 作为 Prompt, 权重为 0; Polygon 作为 Target, 权重为 1)

    Args:
        coord_vocab_shift: Starting index for coordinate tokens (e.g. 1000)
        quantization_bins: Number of bins for coordinate quantization
        noise_bbox_weight: Weight for noise box tokens in training
        eos_token_weight: Weight for end-of-sequence token
        max_seq_len: Maximum allowed sequence length
        num_classes: Number of classes in dataset
    """

    def __init__(
        self,
        quantization_bins: int,
        noise_bbox_weight: float,
        eos_token_weight: float,
        max_seq_len: int,
        num_classes: int,
        num_special_tokens=10,
        corrupt_class_labels: bool = False,
        corruption_strategy: LabelCorruptionStrategy = LabelCorruptionStrategy.NONE,
        verbose=True
    ):
        self.quantization_bins = quantization_bins      # 坐标量化精度
        self.noise_bbox_weight = noise_bbox_weight      # 噪声框token损失权重
        self.bos_eos_token_weight = eos_token_weight    # BOS/EOS token损失权重
        self.max_seq_len = max_seq_len                  # 最大序列长度
        self.num_classes = num_classes                  # 目标类别数
        self._corrupt_class_labels = corrupt_class_labels  # 扰动/增强类别标签
        self._corruption_strategy = corruption_strategy    # 扰动/增强策略

        # 定义 token ID
        self.PADDING_TOKEN = 0
        self.BOS_TOKEN = 1
        self.EOS_TOKEN = 2
        
        # [新增] 任务 Token
        self.DET_TASK_TOKEN = 3  # 检测任务起始符
        self.SEG_TASK_TOKEN = 4  # 分割任务起始符
        
        self.BASE_VOCAB_SHIFT = num_special_tokens  # 类别token起始ID
        self.FAKE_CLASS_TOKEN = self.BASE_VOCAB_SHIFT + self.num_classes  # 伪造类别ID
        self.coord_vocab_shift = self.FAKE_CLASS_TOKEN + 1  # 坐标token起始ID
        self.max_coord_token = self.coord_vocab_shift + self.quantization_bins - 1 # 坐标token最大ID

        # 词汇表大小
        self.vocab_size = self.max_coord_token + 1

        # 验证参数合法性
        self._validate_init_params()
        # 打印 token range
        if verbose:
            self.log_token_ranges()

    def _validate_init_params(self):
        """Validate initialization parameters and vocab space."""

        total_vocab_size = self.vocab_size  #
        min_vocab_size = (
            self.FAKE_CLASS_TOKEN + 1
        )  # Must have room for at least 1 coordinate

        if total_vocab_size < min_vocab_size:
            raise ValueError(
                f"Total vocab size ({total_vocab_size}) must be > "
                f"minimum required size ({min_vocab_size})"
            )

        if self.num_classes < 1:
            raise ValueError(f"num_classes must be positive, got {self.num_classes}")

        if self.quantization_bins < 100:
            raise ValueError(
                "quantization_bins should be at least 100 for reasonable precision"
            )

        # Validate sequence length (5 tokens per box + EOS)
        min_seq_len = 6  # One box (5) + EOS (1)
        if self.max_seq_len < min_seq_len:
            raise ValueError(
                f"max_seq_len must be at least {min_seq_len}, got {self.max_seq_len}"
            )

    def log_token_ranges(self):
        """Log token range information for debugging."""
        print("\nToken Processor initialized with:")
        print(f"  Special tokens: 0-{self.BASE_VOCAB_SHIFT - 1}")
        print(f"    Padding token: {self.PADDING_TOKEN}")
        print(f"    BOS token: {self.BOS_TOKEN}")
        print(f"    EOS token: {self.EOS_TOKEN}")
        print(f"    DET Task token: {self.DET_TASK_TOKEN}")
        print(f"    SEG Task token: {self.SEG_TASK_TOKEN}")
        print(f"  Base vocab shift: {self.BASE_VOCAB_SHIFT}")
        print(f"  Class tokens: {self.BASE_VOCAB_SHIFT}-{self.BASE_VOCAB_SHIFT + self.num_classes - 1}")
        print(f"  FAKE_CLASS_TOKEN: {self.FAKE_CLASS_TOKEN}")
        print(f"  Coordinate tokens: {self.coord_vocab_shift}-{self.max_coord_token}")
        print(f"  Total vocab size: {self.vocab_size}\n")

    def quantize(self, boxes: torch.Tensor):
        """Quantize normalized box coordinates to integer tokens. - 坐标量化"""

        # Scale coordinates to quantization range [0, bins-1]
        boxes = torch.round(boxes * (self.quantization_bins - 1))
        # Clamp to valid range
        boxes = torch.clamp(boxes, 0, self.quantization_bins - 1)
        # Shift to coordinate vocabulary range
        boxes = boxes + self.coord_vocab_shift
        return boxes.long()

    def dequantize(self, tokens: torch.Tensor):
        """Convert coordinate tokens back to normalized coordinates. - 坐标反量化"""

        # Remove coordinate vocabulary shift
        tokens = tokens - self.coord_vocab_shift
        # Scale back to [0, 1] range
        tokens = torch.clamp(tokens, 0, self.quantization_bins - 1)
        # Convert back to [0,1] normalized coordinates
        return tokens.float() / (self.quantization_bins - 1)

    # ===============
    # = 目标检测任务 =
    # ===============
    def corrupt_class_labels(
        self, labels: torch.Tensor, padding_mask: torch.Tensor
    ):
        """Corrupt class labels according to specified strategy. - 根据指定策略扰动类别标签 

        For all strategies, we first decide whether to keep original labels,
        Then for labels we'll corrupt, we apply the strategy's noise type:  
        - NONE: Keep all labels unchanged - 保持所有标签不变
        - RANDOM: Replace with random valid classes - 替换为随机有效类别
        - RANDOM_AND_FAKE: Equal split between random classes and fake token - 随机类别和伪标记的均等分割"""

        if (
            self._corruption_strategy == LabelCorruptionStrategy.NONE
            or not self._corrupt_class_labels
        ):
            return labels

        batch_size, num_labels = labels.shape
        valid_tokens = ~padding_mask  # 有效标签掩码

        keep_mask = (
            torch.rand(batch_size, num_labels, device=labels.device) < 0.5
        ) & valid_tokens

        corrupted = labels.clone()
        rand_cls = torch.randint(
            self.BASE_VOCAB_SHIFT, self.BASE_VOCAB_SHIFT + self.num_classes, (batch_size, num_labels), device=labels.device
        )

        if self._corruption_strategy == LabelCorruptionStrategy.RANDOM:
            corrupted = torch.where(valid_tokens & ~keep_mask, rand_cls, corrupted)
        elif self._corruption_strategy == LabelCorruptionStrategy.RANDOM_AND_FAKE:
            noise_mask = torch.rand(batch_size, num_labels, device=labels.device) < 0.5
            tokens_to_corrupt = valid_tokens & ~keep_mask
            corrupted = torch.where(tokens_to_corrupt & noise_mask, rand_cls, corrupted)
            fake_cls = torch.full_like(
                labels, self.FAKE_CLASS_TOKEN, device=labels.device
            )
            corrupted = torch.where(
                tokens_to_corrupt & ~noise_mask, fake_cls, corrupted
            )

        return corrupted

    def build_sequences(
        self, boxes: torch.Tensor, labels: torch.Tensor
    ):
        """Build token sequences for Detection training. - 构建检测任务序列"""

        batch_size, num_boxes = boxes.shape[:2]

        # 1. 框格式转换: XYXY -> YXYX
        boxes = boxes[..., [1, 0, 3, 2]]

        # 2. 修正无效框：比如 ymax≤ymin 时，加小偏移保证合法
        ymin, xmin, ymax, xmax = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
        invalid_ymax = ymax <= ymin
        invalid_xmax = xmax <= xmin

        if invalid_ymax.any() or invalid_xmax.any():
            # Correction factor
            correction_factor = 0.01
            # Apply correction where necessary
            ymax += invalid_ymax * (ymin - ymax + correction_factor)
            xmax += invalid_xmax * (xmin - xmax + correction_factor)
            # Update boxes with corrected values
            boxes = torch.stack([ymin, xmin, ymax, xmax], dim=-1)

        # 3. 量化坐标
        boxes = self.quantize(boxes)  # [B,N,4]

        # 4. 处理填充
        is_padding = labels == -1
        # Process boxes
        boxes = torch.where(
            is_padding.unsqueeze(-1), torch.full_like(boxes, self.PADDING_TOKEN), boxes
        )
        # Process labels
        target_labels = labels + self.BASE_VOCAB_SHIFT
        target_labels = torch.where(
            is_padding, torch.full_like(target_labels, self.PADDING_TOKEN), target_labels,
        )

        # 5. 生成输入标签与目标标签
        if self._corrupt_class_labels:
            input_labels = self.corrupt_class_labels(
                labels + self.BASE_VOCAB_SHIFT, is_padding
            )
            input_labels = torch.where(
                is_padding, torch.full_like(input_labels, self.PADDING_TOKEN), input_labels,
            )
        else:
            input_labels = target_labels

        # 6. 拼接基础序列 (Box + Class)
        target_seq = torch.cat([boxes, target_labels.unsqueeze(-1)], dim=-1)  # [B,N,5]
        input_seq = torch.cat([boxes, input_labels.unsqueeze(-1)], dim=-1)  # [B,N,5]

        # 7. 计算 Token 权重
        is_fake = target_labels == self.FAKE_CLASS_TOKEN
        
        # 7.1 框坐标权重
        bbox_weights = torch.where(
            is_padding.unsqueeze(-1),
            torch.zeros_like(boxes, dtype=torch.float32),
            torch.where(
                is_fake.unsqueeze(-1),
                torch.full_like(boxes, self.noise_bbox_weight, dtype=torch.float32),
                torch.ones_like(boxes, dtype=torch.float32),
            ),
        )

        # 7.2 类别权重
        label_weights = torch.where(
            is_padding,
            torch.zeros_like(labels, dtype=torch.float32),
            torch.ones_like(labels, dtype=torch.float32),
        )
        # 7.3 合并权重
        token_weights = torch.cat(
            [bbox_weights, label_weights.unsqueeze(-1)], dim=-1
        )  # [B, N, 5]

        """ 对于真实对象（来自真实数据+抖动）：正确学习坐标和类别（所有权重=1.0）
        对于假对象（来自移动/随机生成）：学习将它们分类为“假”，但不要浪费精力学习坐标精度（坐标权重=0.0，类别权重=1.0）
        对于填充：完全忽略（所有权重=0.0） """

        # 8. 展平序列
        target_seq = target_seq.reshape(batch_size, -1)
        input_seq = input_seq.reshape(batch_size, -1)
        token_weights = token_weights.reshape(batch_size, -1)

        # 9. 添加 EOS token
        first_non_padding_idx = (target_seq == self.PADDING_TOKEN).float().argmax(dim=1) 
        target_seq[torch.arange(batch_size), first_non_padding_idx] = self.EOS_TOKEN
        input_seq[torch.arange(batch_size), first_non_padding_idx] = self.EOS_TOKEN
        token_weights[torch.arange(batch_size), first_non_padding_idx] = self.bos_eos_token_weight

        # 10. 添加 Task Token (DET) 作为起始
        task_token = torch.full((batch_size, 1), self.DET_TASK_TOKEN, device=boxes.device).long()
        # Task Token 的权重设为 0（作为 Prompt）或保留 bos_eos_token_weight
        task_weight = torch.zeros((batch_size, 1), device=boxes.device, dtype=torch.float32)

        target_seq = torch.cat([task_token, target_seq], dim=1)
        input_seq = torch.cat([task_token, input_seq], dim=1)
        token_weights = torch.cat([task_weight, token_weights], dim=1)

        # [输入序列, 目标序列, token损失权重]
        return input_seq, target_seq, token_weights  

    # ===============
    # = 实例分割任务 =
    # ===============
    def sample_polygon(self, polygon, num_points=128):
        """
        [新增] 将多边形均匀重采样为固定数量的点。
        Args:
            polygon: List[float] or np.ndarray, [x1, y1, x2, y2, ...] 格式
            num_points: 采样的点数
        Returns:
            sampled_pts: np.ndarray [num_points, 2] -> (y, x) 格式
        """
        # 1. 转换为 Nx2 数组
        pts = np.array(polygon).reshape(-1, 2)
        if len(pts) < 3: # 多边形至少有三个顶点
            return np.zeros((num_points, 2))

        # 2. 计算每段线段的长度
        pts = np.concatenate([pts, pts[:1]], axis=0)  # 闭合多边形
        diff = pts[1:] - pts[:-1]  # 计算相邻点坐标差
        distances = np.linalg.norm(diff, axis=1)  # 计算各边欧式长度
        perimeter = distances.sum()
        
        if perimeter == 0:
            return np.zeros((num_points, 2))

        # 3. 累积距离
        cum_dist = np.concatenate([[0], np.cumsum(distances)])
        
        # 4. 生成均匀采样的目标距离
        target_dists = np.linspace(0, perimeter, num_points + 1)[:-1] 
        
        # 5. 插值计算新坐标 (保持原始数据的 x, y 顺序)
        new_x = np.interp(target_dists, cum_dist, pts[:, 0])
        new_y = np.interp(target_dists, cum_dist, pts[:, 1])
        
        # Pix2Seq 通常使用 (y, x) 顺序，这里进行交换
        sampled_pts = np.stack([new_y, new_x], axis=1) 
        return sampled_pts

    def build_segmentation_sequences(
        self, 
        boxes: torch.Tensor, 
        labels: torch.Tensor, 
        polygons: List[List[Union[List[float], np.ndarray]]], 
        num_points: int = 128,
        prompt_mode: bool = True
    ):
        """
        [新增] 构建分割任务序列
    
        Args:
            prompt_mode (bool): 
                True - Box 和 Class 作为提示 (Weight=0), 仅预测 Polygon (Weight=1)
                False - 全部预测 (Weight=1)
        
        Sequence: [SEG] [Box] [Class] [Poly] ... [EOS]
        """
        batch_size = boxes.shape[0]
        device = boxes.device
        
        # 1. Box 预处理 (XYXY -> YXYX -> Quantize)
        boxes = boxes[..., [1, 0, 3, 2]]
        boxes = self.quantize(boxes)
        
        batch_seqs, batch_weights = [], []
        poly_len = num_points * 2
        
        for b in range(batch_size):
            # 筛选当前样本的有效物体索引
            valid_indices = (labels[b] != -1).nonzero().squeeze(-1)
            seq_list, w_list = [], []
            
            for idx in valid_indices:
                # A. Box [4] (Prompt)
                box = boxes[b, idx]
                w_box = torch.zeros(4, device=device) if prompt_mode else torch.ones(4, device=device)
                
                # B. Class [1] (Prompt)
                cls = (labels[b, idx] + self.BASE_VOCAB_SHIFT).unsqueeze(0)
                w_cls = torch.zeros(1, device=device) if prompt_mode else torch.ones(1, device=device)
                
                # C. Polygon [2N] (Target)
                if b < len(polygons) and idx < len(polygons[b]):
                    raw_poly = polygons[b][idx]  # 当前物体的polygon
                    sampled = self.sample_polygon(raw_poly, num_points)
                    sampled = torch.from_numpy(sampled).float().to(device).clamp(0, 1)
                    poly = self.quantize(sampled).flatten()
                    w_poly = torch.ones(poly_len, device=device)
                else:
                    poly = torch.full((poly_len,), self.PADDING_TOKEN, device=device).long()
                    w_poly = torch.zeros(poly_len, device=device)
                
                # 拼接单个物体: [Box, Class, Poly]
                seq_list.append(torch.cat([box, cls, poly]))
                w_list.append(torch.cat([w_box, w_cls, w_poly]))
            
            # 拼接该样本的所有物体
            if seq_list:
                seq = torch.cat(seq_list)
                w = torch.cat(w_list)
            else:
                seq = torch.tensor([], device=device).long()
                w = torch.tensor([], device=device)
            
            # 添加 EOS
            seq = torch.cat([seq, torch.tensor([self.EOS_TOKEN], device=device)])
            w = torch.cat([w, torch.tensor([self.bos_eos_token_weight], device=device)])
            
            # 添加 SEG Task Token (Prompt)
            task = torch.tensor([self.SEG_TASK_TOKEN], device=device)
            w_task = torch.zeros(1, device=device)
            
            full_seq = torch.cat([task, seq])
            full_w = torch.cat([w_task, w])
            
            # Pad / Clip
            if len(full_seq) > self.max_seq_len:
                full_seq = full_seq[:self.max_seq_len]
                full_w = full_w[:self.max_seq_len]
            else:
                pad_len = self.max_seq_len - len(full_seq)
                full_seq = torch.cat([full_seq, torch.full((pad_len,), self.PADDING_TOKEN, device=device)])
                full_w = torch.cat([full_w, torch.zeros(pad_len, device=device)])
                
            batch_seqs.append(full_seq)
            batch_weights.append(full_w)
            
        final_seq = torch.stack(batch_seqs)
        final_w = torch.stack(batch_weights)
        
        # 输入和目标暂时一致（Collator层会做shift）
        return final_seq, final_seq, final_w 



    ## TODO 预测过程
    def decode_tokens(
        self,
        tokens: torch.Tensor,  # [B,S]
        token_scores: Optional[torch.Tensor] = None,  # [B,S]
        num_poly_points: int = 128 # 采样点数
    ):
        """Decode token sequences back to boxes, class labels, and polygons."""
        batch_size = len(tokens)
        
        # 识别任务类型
        first_token = tokens[0, 0].item()
        if tokens[0, 0] in [self.DET_TASK_TOKEN, self.SEG_TASK_TOKEN, self.BOS_TOKEN]:
            tokens = tokens[:, 1:]  # Remove Task Token
            
        if first_token == self.SEG_TASK_TOKEN:
            return self._decode_segmentation(tokens, batch_size, num_poly_points)
        else:
            return self._decode_detection(tokens, batch_size, token_scores)

    def _decode_detection(self, tokens, batch_size, token_scores):
        
        # 1. 标记出 [PAD] 或 [EOS] 的位置
        is_end = (tokens == self.PADDING_TOKEN) | (tokens == self.EOS_TOKEN)
        end_indices = []

        for b in range(batch_size):
            try:
                # Find first padding/EOS token
                end_idx = is_end[b].nonzero()[0].item()
                # Ensure 5-token groups
                end_idx = (end_idx // 5) * 5
            except IndexError:
                # No padding/EOS found, use all tokens and ensure 5-token groups
                end_idx = (tokens.size(1) // 5) * 5
            end_indices.append(end_idx)

        # 2. 截断到 batch 中最长的有效长度
        max_len = max(end_indices)
        if max_len == 0:
            return [torch.tensor([])]*batch_size, [torch.tensor([])]*batch_size, [None]*batch_size, None

        tokens = tokens[:, :max_len].reshape(batch_size, -1, 5)  # [B, N, 5]

        # 3. 拆分 coordinates 和 class labels
        coord_tokens = tokens[..., :4]
        class_tokens = tokens[..., 4]

        # 4. 坐标反量化
        boxes = self.dequantize(coord_tokens)
        boxes = boxes[..., [1, 0, 3, 2]] # YXYX -> XYXY

        # 5. 类别映射
        labels = class_tokens - self.BASE_VOCAB_SHIFT
        
        # 6. 置信度计算
        scores = None
        if token_scores is not None:
            # token_scores 是模型在生成类别词时的原始预测值 [B, num_objects, Vocab_Size] - only for class tokens
            num_objects = token_scores.size(1)
            num_valid_objects = min(num_objects, class_tokens.size(1))

            # 仅保留有效物体的 类别token和置信度score
            valid_class_tokens = class_tokens[:, :num_valid_objects]  # [B,num_valid]
            valid_scores = token_scores[:, :num_valid_objects]  # [B,num_valid,V]

            # 对所有词汇进行 Softmax，得到归一化的概率 [0-1]
            class_probs = torch.softmax(valid_scores, dim=-1)  # [B,num_valid,V]

            # 使用 torch.gather 从词表概率中，根据模型选中的类别 ID，精准提取出该类别的概率值
            scores = torch.gather(
                class_probs,
                dim=-1,
                index=valid_class_tokens.unsqueeze(-1),  # [B,num_valid,1]
            ).squeeze(-1)  # [B,num_valid]

            # 如果 scores 的长度因某些原因（如生成截断）短于 boxes，补 0
            if scores.size(1) < boxes.size(1):
                padding = torch.zeros(
                    (batch_size, boxes.size(1) - scores.size(1)),
                    device=scores.device,
                    dtype=scores.dtype,
                )
                scores = torch.cat([scores, padding], dim=1)  # [B,N]
        
        # 将 Batch Tensor 拆分为列表格式
        boxes_list = [boxes[i] for i in range(batch_size)]
        labels_list = [labels[i] for i in range(batch_size)]
        scores_list = [scores[i] if scores is not None else None for i in range(batch_size)]
        
        polys_list = [None] * batch_size
        return boxes_list, labels_list, scores_list, polys_list
    
    def _decode_segmentation(self, tokens, batch_size, num_points):
        # 分割序列结构: [Box(4), Class(1), Poly(2*N)] = 5 + 2N
        stride = 5 + 2 * num_points
        
        boxes_list, labels_list, polys_list = [], [], []
        
        for b in range(batch_size):
            seq = tokens[b]
            eos_idx = (seq == self.EOS_TOKEN).nonzero()
            if len(eos_idx) > 0:
                seq = seq[:eos_idx[0].item()]
            
            # 截断到 stride 的倍数
            num_objs = len(seq) // stride
            if num_objs == 0:
                boxes_list.append(torch.tensor([]))
                labels_list.append(torch.tensor([]))
                polys_list.append([])
                continue
                
            seq = seq[:num_objs * stride].reshape(num_objs, stride)
            
            # 解析
            box_toks = seq[:, :4]
            cls_toks = seq[:, 4]
            poly_toks = seq[:, 5:]
            
            # 反量化 Box
            boxes = self.dequantize(box_toks)
            boxes = boxes[..., [1, 0, 3, 2]] # XYXY
            
            # 解析 Label
            labels = cls_toks - self.BASE_VOCAB_SHIFT
            
            # 反量化 Polygon (注意：这里不需要反YXYX，因为Poly本来就是YX顺序存的，或者看sample_polygon的实现)
            # sample_polygon 返回的是 (y, x)，quantize 后也是 (y, x)
            # 如果需要 XY 格式，dequantize 后需要交换最后一维
            polys = self.dequantize(poly_toks).reshape(num_objs, num_points, 2)
            # YX -> XY
            polys = polys[..., [1, 0]] 
            
            boxes_list.append(boxes)
            labels_list.append(labels)
            polys_list.append(polys)
            
        # 分割任务 Score: TODO
        scores_list = [None] * batch_size
        
        return boxes_list, labels_list, scores_list, polys_list
    
    # @property
    # def vocab_size(self):
    #     return self.max_coord_token + 1

    # @property
    # def max_coord_token(self):
    #     return self.coord_vocab_shift + self.quantization_bins - 1

    def post_process_sequences(
        self,
        sequences: torch.Tensor,
        class_logits: Optional[torch.Tensor] = None,
        confidence_threshold: float = 0.05,
    ):
        """Post-process and filter invalid boxes from decoded sequences."""
        # 将token序列解码为 `检测框、类别、置信度、实例框`
        boxes_list, labels_list, scores_list, polys_list = self.decode_tokens(
            sequences, class_logits
        )

        # 遍历 Batch 中的每一张图片及其对应的预测结果
        filtered_boxes, filtered_labels, filtered_scores, filtered_polys = [], [], [], []

        for i, (boxes, labels, scores, polys) in enumerate(zip(boxes_list, labels_list, scores_list, polys_list)):
            if len(boxes) == 0:
                filtered_boxes.append(boxes)
                filtered_labels.append(labels)
                filtered_scores.append(scores if scores is not None else None)
                filtered_polys.append(torch.tensor([]) if polys is not None else None)
                continue

            # Create validity mask for coordinate constraints
            valid_mask = torch.ones(len(boxes), dtype=torch.bool, device=boxes.device)
            valid_mask &= boxes[:, 2] > boxes[:, 0]  # ymax > ymin
            valid_mask &= boxes[:, 3] > boxes[:, 1]  # xmax > xmin
            valid_mask &= torch.all((boxes >= 0) & (boxes <= 1), dim=1)

            if scores is not None:
                valid_mask &= scores > confidence_threshold
                filtered_scores.append(scores[valid_mask])
            else:
                filtered_scores.append(None)

            filtered_boxes.append(boxes[valid_mask])
            filtered_labels.append(labels[valid_mask])

            # [新增] 同步过滤 Polygons
            if polys is not None and len(polys) > 0:
                # polys 的形状通常是 [N, num_points, 2]; valid_mask 的形状是 [N]
                filtered_polys.append(polys[valid_mask])
            else:
                filtered_polys.append(None)

        return filtered_boxes, filtered_labels, filtered_scores, filtered_polys