from typing import List, Tuple, Dict, Optional, Union

import torch
import torch.nn.functional as F


class VideoTokenProcessor:
    """
    Video Tokenizer / 序列处理器（解耦版：DET / KP / ACT）

    ========= 词表布局 =========
      0 : PAD
      1 : BOS
      2 : EOS
      3 : DET_TASK_TOKEN
      4 : KP_TASK_TOKEN
      5 : ACT_TASK_TOKEN
      6 : INVISIBLE_KP_TOKEN
      7 : PROMPT_PAD_TOKEN           # 用于无效 prompt / 缺人占位（配合 attention mask）
      10- (10+num_classes-1) : 检测类别（COCO 类别）
      90 : FAKE_CLASS_TOKEN
      91- (91+num_actions-1) : 动作类别 token（如 fall/slap/kick/push/normal）
      101- (101+quantization_bins-1) : 坐标 bins（默认 1000）
      1101- (1101+num_frame_bins-1)  : 帧号 token（默认 50，可选）

    ========= DET（tube） =========
      target:
        [DET]
        [tube_1: (y1,x1,y2,x2)*T + class]
        [tube_2: (y1,x1,y2,x2)*T + class]
        ...
        [EOS]
        [PAD...]

      注：输入 tube_boxes 为 XYXY(0~1)，序列内部存为 YXYX 的坐标 token。

    ========= KP（单人 tube，逐人展开训练） =========
      target:
        [KP] 
        [frame_0 bbox (y1,x1,y2,x2)]   # prompt, weight=0
        [frame_0 keypoints (y,x)*K]    # predict, weight=1 (invisible -> 0)
        ...
        [frame_{T-1} bbox]             # prompt, weight=0
        [frame_{T-1} keypoints]        # predict, weight=1 (invisible -> 0)
        [EOS]
        [PAD...]

      - bbox 为 prompt（不计入 loss）
      - keypoints 不可见：用 INVISIBLE_KP_TOKEN 填充 y/x 两个位置（loss 权重=0）
      - 无效 bbox 帧：用 PROMPT_PAD_TOKEN 占位（prompt 区不计 loss）

    ========= ACT（GT-KP 提示 + 预测 action） =========
      target:
        [ACT]
        # frame-major：每帧按 person0, person1... 依次铺开
        [frame_0 person0 keypoints (y,x)*K]
        [frame_0 person1 keypoints (y,x)*K]      # 若单人，用 PROMPT_PAD_TOKEN 整段占位
        [frame_1 person0 keypoints]
        [frame_1 person1 keypoints]
        ...
        [frame_{T-1} person0 keypoints]
        [frame_{T-1} person1 keypoints]
        [ACTION_TOKEN]                            # 预测段（one token）
        [EOS]
        [PAD...]

      - keypoints 全部为 prompt（不计入 loss）
      - action token 计入 loss
      - EOS 计入 loss（bos_eos_token_weight）
    """

    def __init__(
        self,
        max_seq_len: int,
        *,
        num_classes: int = 80,
        num_actions: int = 5,
        num_keypoints: int = 16,
        quantization_bins: int = 1000,
        num_frame_bins: int = 50,
        tube_frames: int = 8,
        bos_eos_token_weight: float = 1.0,
        verbose: bool = True,
    ):
        self.max_seq_len = int(max_seq_len)
        self.num_classes = int(num_classes)
        self.num_actions = int(num_actions)
        self.num_keypoints = int(num_keypoints)
        self.quantization_bins = int(quantization_bins)
        self.num_frame_bins = int(num_frame_bins)
        self.tube_frames = int(tube_frames)
        self.bos_eos_token_weight = float(bos_eos_token_weight)

        # ========== 特殊 token（按你最新定义） ==========
        self.PADDING_TOKEN = 0
        self.BOS_TOKEN = 1
        self.EOS_TOKEN = 2
        self.DET_TASK_TOKEN = 3
        self.KP_TASK_TOKEN = 4
        self.ACT_TASK_TOKEN = 5
        self.INVISIBLE_KP_TOKEN = 6
        self.PROMPT_PAD_TOKEN = 7

        # ========== 类别 token ==========
        self.BASE_CLASS_SHIFT = 10
        self.FAKE_CLASS_TOKEN = 90

        # ========== 动作 token ==========
        self.ACTION_BASE_SHIFT = 91
        # 0..(num_actions-1) -> ACTION_BASE_SHIFT..ACTION_BASE_SHIFT+num_actions-1
        self.action_id_to_token: Dict[int, int] = {
            i: self.ACTION_BASE_SHIFT + i for i in range(self.num_actions)
        }
        self.action_token_to_id: Dict[int, int] = {
            v: k for k, v in self.action_id_to_token.items()
        }

        # ========== 坐标 / 帧号 token ==========
        self.coord_vocab_shift = 101
        self.frame_vocab_shift = 1101

        self._validate_layout()

        if verbose:
            self._log_token_ranges()

    # ------------------------------------------------------------------
    # 基本属性
    # ------------------------------------------------------------------
    @property
    def vocab_size(self):
        return self.frame_vocab_shift + self.num_frame_bins  # 1101 + 50 = 1151

    # ------------------------------------------------------------------
    # 布局合法性检查
    # ------------------------------------------------------------------
    def _validate_layout(self):
        if not (1 <= self.num_classes <= 80):
            raise ValueError(f"num_classes must be in [1,80], got {self.num_classes}")
        if self.quantization_bins != 1000:
            raise ValueError(
                f"quantization_bins must be 1000 to match [101-1100], got {self.quantization_bins}"
            )
        if self.num_actions < 1:
            raise ValueError(f"num_actions must be >= 1, got {self.num_actions}")
        if self.tube_frames <= 0:
            raise ValueError(f"tube_frames must be > 0, got {self.tube_frames}")
        if self.num_keypoints <= 0:
            raise ValueError(f"num_keypoints must be > 0, got {self.num_keypoints}")

        # 最小长度约束（给出明确报错，避免 silent truncation）
        # DET：至少 1 tube + EOS
        det_min = 1 + (4 * self.tube_frames + 1) + 1

        # KP：bbox(T*4) + kp(T*K*2) + EOS
        kp_min = 1 + (self.tube_frames * 4) + (self.tube_frames * self.num_keypoints * 2) + 1

        # ACT：按最大 2 人估计（最保守），prompt(T*2*K*2) + action + EOS
        act_min = 1 + (self.tube_frames * 2 * self.num_keypoints * 2) + 2

        min_need = max(det_min, kp_min, act_min)
        if self.max_seq_len < min_need:
            raise ValueError(
                f"max_seq_len too small: max_seq_len={self.max_seq_len}, "
                f"need at least {min_need} (det_min={det_min}, kp_min={kp_min}, act_min(2p)={act_min})."
            )

    def _log_token_ranges(self):
        print("\n[VideoTokenProcessor] Layout (Decoupled):")
        print(
            f"  PAD/BOS/EOS/DET/KP/ACT/INV_KP/PROMPT_PAD = "
            f"{self.PADDING_TOKEN},{self.BOS_TOKEN},{self.EOS_TOKEN},"
            f"{self.DET_TASK_TOKEN},{self.KP_TASK_TOKEN},{self.ACT_TASK_TOKEN},"
            f"{self.INVISIBLE_KP_TOKEN},{self.PROMPT_PAD_TOKEN}"
        )
        print(
            f"  Class tokens: {self.BASE_CLASS_SHIFT} - "
            f"{self.BASE_CLASS_SHIFT + self.num_classes - 1}"
        )
        print(f"  FAKE_CLASS_TOKEN: {self.FAKE_CLASS_TOKEN}")
        print(
            f"  Action tokens: {self.ACTION_BASE_SHIFT} - "
            f"{self.ACTION_BASE_SHIFT + self.num_actions - 1}"
        )
        print(
            f"  Coord tokens: {self.coord_vocab_shift} - "
            f"{self.coord_vocab_shift + self.quantization_bins - 1}"
        )
        print(
            f"  Frame tokens: {self.frame_vocab_shift} - "
            f"{self.frame_vocab_shift + self.num_frame_bins - 1}"
        )
        print(f"  Vocab size: {self.vocab_size}")
        print(f"  Tube frames (T): {self.tube_frames}, Keypoints (K): {self.num_keypoints}\n")

    # ------------------------------------------------------------------
    # 通用 pad/clip
    # ------------------------------------------------------------------
    @staticmethod
    def _pad_or_clip_to(t: torch.Tensor, length: int, pad_value):
        if t.size(1) == length:
            return t
        if t.size(1) > length:
            return t[:, :length]
        B = t.size(0)
        pad = torch.full(
            (B, length - t.size(1)),
            pad_value,
            dtype=t.dtype,
            device=t.device,
        )
        return torch.cat([t, pad], dim=1)

    # ------------------------------------------------------------------
    # 坐标 / 帧号 编码 & 反编码
    # ------------------------------------------------------------------
    def quantize_coords(self, x: torch.Tensor):
        x = torch.round(x * (self.quantization_bins - 1))
        x = torch.clamp(x, 0, self.quantization_bins - 1)
        x = x + self.coord_vocab_shift
        return x.long()

    def dequantize_coords(self, tokens: torch.Tensor):
        x = tokens - self.coord_vocab_shift
        x = torch.clamp(x, 0, self.quantization_bins - 1)
        return x.float() / (self.quantization_bins - 1)

    def encode_frame_ids(self, frame_ids: torch.Tensor):
        return (frame_ids.long() + self.frame_vocab_shift).clamp(
            self.frame_vocab_shift,
            self.frame_vocab_shift + self.num_frame_bins - 1,
        )

    # ================================================================
    # DET（保持 tube 版本，已同步 token id）
    # ================================================================
    def build_detection_sequences(
        self,
        tube_boxes: torch.Tensor,   # [B, N, T, 4] XYXY (0~1)
        class_ids: torch.Tensor,    # [B, N] 0..num_classes-1, padding=-1
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = tube_boxes.device
        B, N, T, C = tube_boxes.shape
        assert C == 4, f"tube_boxes last dim must be 4 (XYXY), got {C}"
        assert T == self.tube_frames, f"tube_boxes T={T} mismatch tube_frames={self.tube_frames}"

        class_ids = class_ids.to(device)
        is_padding = (class_ids == -1)  # [B,N]

        # XYXY -> YXYX，再量化
        boxes_yxyx = tube_boxes[..., [1, 0, 3, 2]]  # [B,N,T,4]
        q_tube = self.quantize_coords(boxes_yxyx)   # [B,N,T,4]

        # padding tube -> PAD
        pad_mask = is_padding.unsqueeze(-1).unsqueeze(-1)  # [B,N,1,1]
        q_tube = torch.where(
            pad_mask,
            torch.full_like(q_tube, self.PADDING_TOKEN, dtype=torch.long),
            q_tube,
        )

        # class token
        cls_tok = class_ids + self.BASE_CLASS_SHIFT
        cls_tok = torch.where(
            is_padding,
            torch.full_like(cls_tok, self.PADDING_TOKEN, dtype=torch.long),
            cls_tok,
        )

        coords_flat = q_tube.view(B, N, T * 4)  # [B,N,4T]
        obj_tokens = torch.cat([coords_flat, cls_tok.unsqueeze(-1)], dim=-1)  # [B,N,4T+1]

        chunk_len = 4 * T + 1

        obj_weights = torch.where(
            is_padding.unsqueeze(-1),
            torch.zeros_like(obj_tokens, dtype=torch.float32),
            torch.ones_like(obj_tokens, dtype=torch.float32),
        )

        body_tokens = obj_tokens.reshape(B, -1).long()
        body_weights = obj_weights.reshape(B, -1).float()

        valid_obj = (~is_padding).sum(dim=1)           # [B]
        valid_tok = (valid_obj * chunk_len).long()     # [B]

        max_body = int(valid_tok.max().item() + 1)     # +1 for EOS
        max_body = max(1, max_body)
        max_body = min(max_body, self.max_seq_len - 1) # reserve 1 for DET token

        body_tokens = self._pad_or_clip_to(body_tokens, max_body, self.PADDING_TOKEN)
        body_weights = self._pad_or_clip_to(body_weights, max_body, 0.0)

        rows = torch.arange(B, device=device)
        eos_pos = torch.clamp(valid_tok, max=body_tokens.size(1) - 1)
        body_tokens[rows, eos_pos] = self.EOS_TOKEN

        eos_w = torch.zeros_like(body_weights, dtype=torch.float32)
        eos_w[rows, eos_pos] = self.bos_eos_token_weight
        body_weights = torch.where(body_tokens == self.EOS_TOKEN, eos_w, body_weights)

        det_tok = torch.full((B, 1), self.DET_TASK_TOKEN, dtype=torch.long, device=device)
        det_w = torch.zeros((B, 1), dtype=torch.float32, device=device)

        target_seq_full = torch.cat([det_tok, body_tokens], dim=1)
        token_w_full = torch.cat([det_w, body_weights], dim=1)

        S = self.max_seq_len
        target_seq_full = self._pad_or_clip_to(target_seq_full, S, self.PADDING_TOKEN)
        token_w_full = self._pad_or_clip_to(token_w_full, S, 0.0)

        det_input_seq = target_seq_full.clone()
        det_input_seq[:, 1:] = target_seq_full[:, :-1]
        det_input_seq[:, 0] = self.DET_TASK_TOKEN
        token_w_full[:, 0] = 0.0

        return det_input_seq, target_seq_full, token_w_full

    def decode_detect_sequences_from_tokens(
        self,
        seq: torch.Tensor,                    # [S]
        tube_frames: Optional[int] = None,
        confidence_threshold: float = 0.0,
        class_logits: Optional[torch.Tensor] = None,  # [S,V] or None
    ):
        device = seq.device
        tokens = seq.tolist()
        S = len(tokens)

        T = tube_frames if tube_frames is not None else self.tube_frames
        chunk_len = 4 * T + 1

        PAD = self.PADDING_TOKEN
        EOS = self.EOS_TOKEN
        coord_shift = self.coord_vocab_shift
        Q = self.quantization_bins
        base_shift = self.BASE_CLASS_SHIFT
        num_classes = self.num_classes
        fake_token = getattr(self, "FAKE_CLASS_TOKEN", None)

        start_idx = 1
        try:
            det_pos = tokens.index(self.DET_TASK_TOKEN)
            if det_pos + chunk_len <= S:
                start_idx = det_pos + 1
        except ValueError:
            pass

        def _tok2coord_float(t: int) -> float:
            v = t - coord_shift
            v = max(0, min(Q - 1, v))
            return float(v) / float(Q - 1)

        use_logits = class_logits is not None
        if use_logits:
            assert class_logits.dim() == 2 and class_logits.size(0) == S, \
                f"class_logits shape mismatch: got {class_logits.shape}, expected ({S}, V)"

        tube_boxes: List[List[List[float]]] = []
        labels: List[int] = []
        scores: List[float] = []

        i = start_idx
        while i + chunk_len - 1 < S:
            t0 = tokens[i]
            if t0 == PAD or t0 == EOS:
                break

            tube_chunk = tokens[i: i + chunk_len]
            coord_tokens = tube_chunk[: 4 * T]
            t_cls = tube_chunk[4 * T]

            if any((t < coord_shift) or (t >= coord_shift + Q) for t in coord_tokens):
                i += chunk_len
                continue

            cls_id = t_cls - base_shift
            if cls_id < 0 or cls_id >= num_classes:
                i += chunk_len
                continue

            frame_boxes: List[List[float]] = []
            for f in range(T):
                t_y1 = coord_tokens[4 * f + 0]
                t_x1 = coord_tokens[4 * f + 1]
                t_y2 = coord_tokens[4 * f + 2]
                t_x2 = coord_tokens[4 * f + 3]
                y1 = _tok2coord_float(t_y1)
                x1 = _tok2coord_float(t_x1)
                y2 = _tok2coord_float(t_y2)
                x2 = _tok2coord_float(t_x2)
                frame_boxes.append([x1, y1, x2, y2])  # XYXY

            if use_logits:
                cls_pos = i + 4 * T
                if 0 <= cls_pos < class_logits.size(0):
                    logit_vec = class_logits[cls_pos]
                    if fake_token is not None and 0 <= fake_token < logit_vec.size(0):
                        logit_vec = logit_vec.clone()
                        logit_vec[fake_token] = -1e9
                    prob_vec = F.softmax(logit_vec, dim=-1)
                    score = float(prob_vec[t_cls].item()) if (0 <= t_cls < prob_vec.size(0)) else 0.0
                else:
                    score = 0.0
            else:
                score = 1.0

            if score < confidence_threshold:
                i += chunk_len
                continue

            tube_boxes.append(frame_boxes)
            labels.append(cls_id)
            scores.append(score)
            i += chunk_len

        if len(tube_boxes) == 0:
            return (
                torch.empty((0, T, 4), device=device, dtype=torch.float32),
                torch.empty((0,), device=device, dtype=torch.long),
                torch.empty((0,), device=device, dtype=torch.float32),
            )

        tube_boxes_t = torch.tensor(tube_boxes, device=device, dtype=torch.float32)
        labels_t = torch.tensor(labels, device=device, dtype=torch.long)
        scores_t = torch.tensor(scores, device=device, dtype=torch.float32)
        return tube_boxes_t, labels_t, scores_t

    def post_process_sequences(
        self,
        sequences: torch.Tensor,          # [B,S]
        class_logits: Optional[torch.Tensor] = None,  # [B,S,V] or None
        confidence_threshold: float = 0.0,
        tube_frames: Optional[int] = None,
    ):
        T = tube_frames if tube_frames is not None else self.tube_frames
        B, S = sequences.shape
        use_logits = class_logits is not None
        if use_logits:
            assert class_logits.shape[0] == B and class_logits.shape[1] == S, \
                f"class_logits shape mismatch: {class_logits.shape} vs sequences {sequences.shape}"

        batch_tubes, batch_labels, batch_scores = [], [], []
        for b in range(B):
            seq_b = sequences[b]
            logits_b = class_logits[b] if use_logits else None
            tubes_b, labels_b, scores_b = self.decode_detect_sequences_from_tokens(
                seq_b,
                tube_frames=T,
                confidence_threshold=confidence_threshold,
                class_logits=logits_b,
            )
            batch_tubes.append(tubes_b)
            batch_labels.append(labels_b)
            batch_scores.append(scores_b)
        return batch_tubes, batch_labels, batch_scores

    # ================================================================
    # KP（新增：bbox prompt + kp 预测）
    # ================================================================
    def build_kp_sequences(
        self,
        tube_boxes: torch.Tensor,              # [B, T, 4] or [B,1,T,4]  XYXY(0~1) 作为 prompt
        keypoints_traj: torch.Tensor,          # [B, T, K, 3]
        keypoints_visible_traj: torch.Tensor,  # [B, T, K] (bool/0-1)
        *,
        xy_order: str = "xy",
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = tube_boxes.device

        # 允许 [B,1,T,4]
        if tube_boxes.dim() == 4:
            assert tube_boxes.size(1) == 1, f"KP expects single person tube, got tube_boxes shape {tube_boxes.shape}"
            tube_boxes = tube_boxes[:, 0]

        B, T, C = tube_boxes.shape
        B2, T2, K, Ck = keypoints_traj.shape
        assert B == B2 and T == T2, "tube_boxes / keypoints_traj shape mismatch"
        assert C == 4, f"tube_boxes last dim must be 4, got {C}"
        assert Ck == 3, f"keypoints_traj last dim must be 3, got {Ck}"
        assert K == self.num_keypoints, f"num_keypoints mismatch: expected {self.num_keypoints}, got {K}"
        assert T == self.tube_frames, f"T mismatch: got {T}, expected {self.tube_frames}"

        # ---- bbox prompt：无效帧 -> PROMPT_PAD ----
        x1, y1, x2, y2 = tube_boxes[..., 0], tube_boxes[..., 1], tube_boxes[..., 2], tube_boxes[..., 3]
        finite = torch.isfinite(tube_boxes).all(dim=-1)
        bbox_valid = finite & (x2 > x1) & (y2 > y1)  # [B,T]

        # 序列内部存 YXYX（与原实现保持一致）
        boxes_yxyx = tube_boxes[..., [1, 0, 3, 2]].clamp(0.0, 1.0)  # [B,T,4]
        bbox_tok = self.quantize_coords(boxes_yxyx)                 # [B,T,4] long
        bbox_tok = torch.where(
            bbox_valid.unsqueeze(-1),
            bbox_tok,
            torch.full_like(bbox_tok, self.PROMPT_PAD_TOKEN),
        )  # [B,T,4]

        # ---- kp（预测段）：不可见 -> INVISIBLE（对应 y/x 两个位置）；loss 权重=0 ----
        vis = keypoints_visible_traj.to(device).bool()  # [B,T,K]

        if xy_order == "xy":
            kps_x = keypoints_traj[..., 0]
            kps_y = keypoints_traj[..., 1]
        elif xy_order == "yx":
            kps_y = keypoints_traj[..., 0]
            kps_x = keypoints_traj[..., 1]
        else:
            raise ValueError(f"xy_order must be 'xy' or 'yx', got {xy_order}")

        kps_y_tok = self.quantize_coords(kps_y.clamp(0.0, 1.0))  # [B,T,K] long
        kps_x_tok = self.quantize_coords(kps_x.clamp(0.0, 1.0))  # [B,T,K] long

        inv = torch.full_like(kps_y_tok, self.INVISIBLE_KP_TOKEN)
        kps_y_tok = torch.where(vis, kps_y_tok, inv)
        kps_x_tok = torch.where(vis, kps_x_tok, inv)

        # (y,x) 成对展开
        kp_pair = torch.stack([kps_y_tok, kps_x_tok], dim=-1)  # [B,T,K,2]
        kp_per_frame = kp_pair.reshape(B, T, K * 2)            # [B,T,2K]

        # ---- 逐帧交错组装：[bbox(4)] + [kp(2K)] ----
        frame_block = 4 + K * 2
        interleaved = torch.cat([bbox_tok, kp_per_frame], dim=-1)  # [B,T,4+2K]
        interleaved_flat = interleaved.reshape(B, T * frame_block) # [B, T*(4+2K)]

        # ---- 组装 target ----
        needed_len = 1 + T * frame_block + 1  # KP + T*(bbox+kp) + EOS
        if needed_len > self.max_seq_len:
            raise ValueError(
                f"max_seq_len={self.max_seq_len} too small for KP seq needed_len={needed_len} "
                f"(T={T}, K={K}, frame_block={frame_block})."
            )

        tgt = torch.full((B, self.max_seq_len), self.PADDING_TOKEN, dtype=torch.long, device=device)
        tgt[:, 0] = self.KP_TASK_TOKEN

        region_start = 1
        region_end = region_start + T * frame_block
        eos_pos = region_end

        tgt[:, region_start:region_end] = interleaved_flat
        tgt[:, eos_pos] = self.EOS_TOKEN

        # ---- 权重：bbox prompt=0；kp=1（INVISIBLE/PAD/PROMPT_PAD 置 0）；EOS=bos_eos_token_weight ----
        w = torch.zeros((B, self.max_seq_len), dtype=torch.float32, device=device)

        # region 内：只给 kp token 位置赋 1（bbox 位置保持 0）
        # 对 region 内每个位置 i（从 0 开始），如果 (i % frame_block) >= 4 则为 kp 区
        pos = torch.arange(region_start, region_end, device=device, dtype=torch.long)  # [T*frame_block]
        within = (pos - region_start) % frame_block
        kp_pos_mask = (within >= 4).float().view(1, -1)  # [1, T*frame_block]
        w[:, region_start:region_end] = kp_pos_mask

        # 不可见关键点 / 各类占位 token：权重清零（bbox 区本来就是 0，这里乘一下不影响）
        region_tokens = tgt[:, region_start:region_end]
        drop = (
            (region_tokens == self.INVISIBLE_KP_TOKEN)
            | (region_tokens == self.PADDING_TOKEN)
            | (region_tokens == self.PROMPT_PAD_TOKEN)
        )
        w[:, region_start:region_end] = w[:, region_start:region_end] * (~drop).float()

        # EOS 权重（可保持你原来的 bos_eos_token_weight）
        w[:, eos_pos] = self.bos_eos_token_weight
        w[:, 0] = 0.0

        # ---- 输入序列：右移一位（teacher-forcing 并行版）----
        inp = tgt.clone()
        inp[:, 1:] = tgt[:, :-1]
        inp[:, 0] = self.KP_TASK_TOKEN

        return inp, tgt, w

    def decode_kp_sequence_from_tokens(
        self,
        seq: torch.Tensor,                 # [S]
        tube_frames: Optional[int] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        返回：
        bbox_xyxy:   [T,4] or None
        keypoints:   [T,K,3] (x,y,vis) or None

        适配交错序列：
        [KP] + T * ([bbox 4] + [kp 2K]) + [EOS] (+ PAD...)
        """
        # ---- 取 token 列表 ----
        if not isinstance(seq, torch.Tensor):
            return None, None
        tokens = seq.tolist()
        S = len(tokens)

        # ---- 特殊 token（做 None 安全）----
        PAD = int(getattr(self, "PADDING_TOKEN", 0))
        EOS = int(getattr(self, "EOS_TOKEN", 2))
        KP_TASK = int(getattr(self, "KP_TASK_TOKEN", -1))

        PROMPT_PAD = getattr(self, "PROMPT_PAD_TOKEN", None)
        PROMPT_PAD = int(PROMPT_PAD) if PROMPT_PAD is not None else None

        INVISIBLE = getattr(self, "INVISIBLE_KP_TOKEN", None)
        INVISIBLE = int(INVISIBLE) if INVISIBLE is not None else None

        # ---- 找到 KP 起点 ----
        try:
            kp_pos = tokens.index(KP_TASK)
        except ValueError:
            return None, None

        T = int(tube_frames if tube_frames is not None else self.tube_frames)
        K = int(self.num_keypoints)
        frame_block = 4 + K * 2

        # ---- 找到 EOS（没有就当到序列末尾）----
        try:
            eos_pos = tokens.index(EOS, kp_pos + 1)
        except ValueError:
            eos_pos = S

        start = kp_pos + 1
        seg = tokens[start:eos_pos]  # 期望长度 T*frame_block
        expected = T * frame_block

        # 不足则 PAD 补齐（早停也能解析）
        if len(seg) < expected:
            seg = seg + [PAD] * (expected - len(seg))
        seg = seg[:expected]

        seg_t = torch.tensor(seg, dtype=torch.long)  # CPU tensor: [T*frame_block]
        seg_t = seg_t.view(T, frame_block)

        bbox_bt = seg_t[:, :4]    # [T,4]
        kp_bt = seg_t[:, 4:]      # [T,2K]

        coord_shift = int(self.coord_vocab_shift)
        Q = int(self.quantization_bins)

        def _tok_ok(t: torch.Tensor) -> torch.Tensor:
            return (t >= coord_shift) & (t < coord_shift + Q)

        # ===================== decode bbox =====================
        ok_bbox = _tok_ok(bbox_bt) & (bbox_bt != PAD)
        if PROMPT_PAD is not None:
            ok_bbox = ok_bbox & (bbox_bt != PROMPT_PAD)

        bbox_safe = torch.where(ok_bbox, bbox_bt, torch.full_like(bbox_bt, coord_shift))

        # 这里你原注释写的是 (y1,x1,y2,x2)；若你的数据实际是 (x1,y1,x2,y2)，请在这里交换解码顺序
        y1 = self.dequantize_coords(bbox_safe[:, 0])
        x1 = self.dequantize_coords(bbox_safe[:, 1])
        y2 = self.dequantize_coords(bbox_safe[:, 2])
        x2 = self.dequantize_coords(bbox_safe[:, 3])

        # 规范化：确保 x1<=x2, y1<=y2
        x_min = torch.minimum(x1, x2)
        x_max = torch.maximum(x1, x2)
        y_min = torch.minimum(y1, y2)
        y_max = torch.maximum(y1, y2)

        bbox_xyxy = torch.stack([x_min, y_min, x_max, y_max], dim=-1)  # [T,4]

        # 可选：若你的坐标理论范围是 [0,1]，建议 clamp 一下（避免越界扰动下游）
        bbox_xyxy = bbox_xyxy.clamp_(0.0, 1.0)

        # ===================== decode keypoints =====================
        kp_tok = kp_bt.view(T, K, 2)  # [T,K,2]
        y_tok = kp_tok[..., 0].reshape(-1)  # [T*K]
        x_tok = kp_tok[..., 1].reshape(-1)  # [T*K]

        def _is_invis(t: torch.Tensor) -> torch.Tensor:
            m = (t == PAD)
            if PROMPT_PAD is not None:
                m = m | (t == PROMPT_PAD)
            if INVISIBLE is not None:
                m = m | (t == INVISIBLE)
            return m

        y_ok = _tok_ok(y_tok) & (~_is_invis(y_tok))
        x_ok = _tok_ok(x_tok) & (~_is_invis(x_tok))
        vis = (y_ok & x_ok).view(T, K).float()

        y_tok_safe = torch.where(y_ok, y_tok, torch.full_like(y_tok, coord_shift))
        x_tok_safe = torch.where(x_ok, x_tok, torch.full_like(x_tok, coord_shift))

        y = self.dequantize_coords(y_tok_safe).view(T, K)
        x = self.dequantize_coords(x_tok_safe).view(T, K)

        kps_xyv = torch.stack([x, y, vis], dim=-1)  # [T,K,3]
        return bbox_xyxy, kps_xyv

    def post_process_kp_sequences(
        self,
        sequences: torch.Tensor,             # [B,S]
        tube_frames: Optional[int] = None,
    ) -> Tuple[List[Optional[torch.Tensor]], List[Optional[torch.Tensor]]]:
        B, _ = sequences.shape
        bboxes_list: List[Optional[torch.Tensor]] = []
        kps_list: List[Optional[torch.Tensor]] = []
        for b in range(B):
            bbox_b, kps_b = self.decode_kp_sequence_from_tokens(sequences[b], tube_frames=tube_frames)
            bboxes_list.append(bbox_b)
            kps_list.append(kps_b)
        return bboxes_list, kps_list


    # ================================================================
    # ACT（解耦后：GT-KP 提示 + 预测 action）
    # ================================================================
    def build_action_sequences(
        self,
        keypoints_traj: torch.Tensor,          # [B, P, T, K, 3]
        keypoints_visible_traj: torch.Tensor,  # [B, P, T, K]
        action_ids: torch.Tensor,              # [B]
        *,
        persons_per_clip: int = 2,             # 固定结构：建议 1 或 2；单人则 person1 用 PROMPT_PAD 占位
        xy_order: str = "xy",
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = keypoints_traj.device
        B, P, T, K, Ck = keypoints_traj.shape
        assert Ck == 3, f"keypoints_traj last dim must be 3, got {Ck}"
        assert K == self.num_keypoints, f"num_keypoints mismatch: expected {self.num_keypoints}, got {K}"
        assert T == self.tube_frames, f"T mismatch: got {T}, expected {self.tube_frames}"
        assert persons_per_clip in (1, 2), "persons_per_clip must be 1 or 2"

        P_target = persons_per_clip
        if P < P_target:
            raise ValueError(f"keypoints_traj P={P} < persons_per_clip={P_target}. Please pad in collator.")

        keypoints_traj = keypoints_traj[:, :P_target]
        keypoints_visible_traj = keypoints_visible_traj[:, :P_target].to(device).bool()

        # person_valid：任意帧任意点可见则认为该人存在；否则视作缺人 -> PROMPT_PAD 整段
        person_valid = keypoints_visible_traj.any(dim=(-1, -2))         # [B,P]
        person_invalid = (~person_valid).unsqueeze(-1).unsqueeze(-1)    # [B,P,1,1]

        if xy_order == "xy":
            kps_x = keypoints_traj[..., 0]   # [B,P,T,K]
            kps_y = keypoints_traj[..., 1]
        elif xy_order == "yx":
            kps_y = keypoints_traj[..., 0]
            kps_x = keypoints_traj[..., 1]
        else:
            raise ValueError(f"xy_order must be 'xy' or 'yx', got {xy_order}")

        vis = keypoints_visible_traj  # [B,P,T,K]

        kps_y_tok = self.quantize_coords(kps_y.clamp(0.0, 1.0))  # [B,P,T,K]
        kps_x_tok = self.quantize_coords(kps_x.clamp(0.0, 1.0))  # [B,P,T,K]

        inv = torch.full_like(kps_y_tok, self.INVISIBLE_KP_TOKEN)
        kps_y_tok = torch.where(vis, kps_y_tok, inv)
        kps_x_tok = torch.where(vis, kps_x_tok, inv)

        # 缺人：整段 prompt_pad（配合 attention mask）
        pp = torch.full_like(kps_y_tok, self.PROMPT_PAD_TOKEN)
        kps_y_tok = torch.where(person_invalid, pp, kps_y_tok)
        kps_x_tok = torch.where(person_invalid, pp, kps_x_tok)

        # (y,x) interleave -> [B,P,T,K,2]
        kp_pair = torch.stack([kps_y_tok, kps_x_tok], dim=-1)

        # 关键：frame-major 展平： [B,P,T,K,2] -> [B,T,P,K,2] -> [B, T*P*K*2]
        kp_pair = kp_pair.permute(0, 2, 1, 3, 4).contiguous()
        kp_flat = kp_pair.view(B, T * P_target * K * 2)

        # action token
        if ((action_ids < 0) | (action_ids >= self.num_actions)).any():
            bad = action_ids[(action_ids < 0) | (action_ids >= self.num_actions)][0].item()
            raise ValueError(f"action_id must be in [0,{self.num_actions-1}], got {bad}")

        lut = torch.tensor(
            [self.action_id_to_token[i] for i in range(self.num_actions)],
            device=device,
            dtype=torch.long,
        )
        action_tok = lut[action_ids.long()]  # [B]

        prompt_len = T * P_target * K * 2
        needed_len = 1 + prompt_len + 2  # ACT + prompt + action + EOS
        if needed_len > self.max_seq_len:
            raise ValueError(
                f"max_seq_len={self.max_seq_len} too small for ACT seq needed_len={needed_len} "
                f"(P={P_target}, T={T}, K={K})."
            )

        tgt = torch.full((B, self.max_seq_len), self.PADDING_TOKEN, dtype=torch.long, device=device)
        tgt[:, 0] = self.ACT_TASK_TOKEN

        prompt_start = 1
        act_pos = prompt_start + prompt_len
        eos_pos = act_pos + 1

        tgt[:, prompt_start:act_pos] = kp_flat
        tgt[:, act_pos] = action_tok
        tgt[:, eos_pos] = self.EOS_TOKEN

        # 权重：prompt=0；action=1；EOS=bos_eos_token_weight
        w = torch.zeros((B, self.max_seq_len), dtype=torch.float32, device=device)
        w[:, act_pos] = 1.0
        w[:, eos_pos] = self.bos_eos_token_weight
        w[:, 0] = 0.0

        inp = tgt.clone()
        inp[:, 1:] = tgt[:, :-1]
        inp[:, 0] = self.ACT_TASK_TOKEN

        return inp, tgt, w

    def decode_action_sequence_from_tokens(
        self,
        seq: torch.Tensor,
        *,
        persons_per_clip: int = 2,
        tube_frames: Optional[int] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[int]]:
        """
        解码 ACT 序列（frame-major prompt + action）：
          返回：
            prompt_keypoints: [P_eff, T, K, 3] (x,y,vis) 或 None
            action_id: int 或 None
        """
        tokens = seq.tolist()
        S = len(tokens)

        try:
            act_pos0 = tokens.index(self.ACT_TASK_TOKEN)
        except ValueError:
            return None, None

        T = tube_frames if tube_frames is not None else self.tube_frames
        K = self.num_keypoints
        P_req = max(1, min(2, persons_per_clip))

        try:
            eos_pos = tokens.index(self.EOS_TOKEN, act_pos0 + 1)
        except ValueError:
            eos_pos = S

        # 找 action token（从 EOS 前往前找第一个合法 action token）
        action_token = None
        action_idx = None
        valid_action_tokens = set(self.action_token_to_id.keys())
        for i in range(eos_pos - 1, act_pos0, -1):
            t = tokens[i]
            if t in valid_action_tokens:
                action_token = t
                action_idx = i
                break
        action_id = self.action_token_to_id.get(action_token, None) if action_token is not None else None
        if action_idx is None:
            return None, action_id

        prompt_tokens = tokens[act_pos0 + 1: action_idx]
        prompt_len = T * P_req * K * 2
        if len(prompt_tokens) < prompt_len:
            return None, action_id
        prompt_tokens = prompt_tokens[:prompt_len]

        # 还原 frame-major：[T,P,K,2] -> [P,T,K,2]
        pt = torch.tensor(prompt_tokens, dtype=torch.long).view(T, P_req, K, 2)
        pt = pt.permute(1, 0, 2, 3).contiguous()  # [P,T,K,2]
        y_tok = pt[..., 0]  # [P,T,K]
        x_tok = pt[..., 1]

        coord_shift = self.coord_vocab_shift
        Q = self.quantization_bins

        def _tok_ok(t: torch.Tensor) -> torch.Tensor:
            return (t >= coord_shift) & (t < coord_shift + Q)

        def _is_invis(t: torch.Tensor) -> torch.Tensor:
            return (t == self.INVISIBLE_KP_TOKEN) | (t == self.PROMPT_PAD_TOKEN) | (t == self.PADDING_TOKEN)

        y_ok = _tok_ok(y_tok) & (~_is_invis(y_tok))
        x_ok = _tok_ok(x_tok) & (~_is_invis(x_tok))
        vis = (y_ok & x_ok).float()

        y_tok_safe = torch.where(y_ok, y_tok, torch.full_like(y_tok, coord_shift))
        x_tok_safe = torch.where(x_ok, x_tok, torch.full_like(x_tok, coord_shift))

        y = self.dequantize_coords(y_tok_safe)
        x = self.dequantize_coords(x_tok_safe)

        kps_xyv = torch.stack([x, y, vis], dim=-1)  # [P,T,K,3]

        # P_eff：若整个人全是 PROMPT_PAD/不可见，可按需裁掉；这里保守返回 P_req
        return kps_xyv, action_id

    def post_process_action_sequences(
        self,
        sequences: torch.Tensor,          # [B,S]
        *,
        persons_per_clip: int = 2,
        tube_frames: Optional[int] = None,
    ) -> Tuple[List[Optional[torch.Tensor]], List[Optional[int]]]:
        B, _ = sequences.shape
        kps_list: List[Optional[torch.Tensor]] = []
        act_ids: List[Optional[int]] = []
        for b in range(B):
            kps_b, act_id_b = self.decode_action_sequence_from_tokens(
                sequences[b],
                persons_per_clip=persons_per_clip,
                tube_frames=tube_frames,
            )
            kps_list.append(kps_b)
            act_ids.append(act_id_b)
        return kps_list, act_ids
