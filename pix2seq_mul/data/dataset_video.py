# -*- coding: utf-8 -*-
"""
PyTorch dataset implementation for Pix2Seq-Video (detection + keypoints).

该模块负责：
- 从视频基础数据集（如 NTU VideoClipDataset_kp_d）取出一个 clip（T 帧）
- 进行时空一致的图像增强（VideoAugmentor_kp_d）
- 进行 Tube 级别的时序增强（TubeAugmentor_kp_d：随机打乱实例顺序）
- 将 numpy 数据整理为 PyTorch tensor，供后续 Tokenizer / Collator 使用

输入 sample 格式（与 NTUVideoClipDatasetFromAvi_kp_d / VideoClipDataset_kp_d 对齐）：
    sample = {
        "images": images,                  # [T,H,W,3]  uint8, RGB
        "boxes": boxes_list,               # 长度 T， 第 t 项 [N_t,4] (xyxy，像素 or 归一化)
        "class_ids": class_ids_list,       # 长度 T， 第 t 项 [N_t]
        "keypoints": keypoints_list,       # 长度 T， 第 t 项 [N_t,K,3]
        "keypoints_visible": keypoints_vis_list,  # 长度 T， 第 t 项 [N_t,K]
        "person_ids": person_ids_list,     # 长度 T， 第 t 项 [N_t]
        "object_ids": object_ids_list,     # 可选，长度 T， 第 t 项 [N_t]，静态物体 ID（人时可为 -1）
        "video_id": video_id: str,
        "frame_ids": frame_indices: List[int],
        "image_ids": image_ids_center: List[int],
        "image_hw": (H, W),
        "frame_valid_mask": frame_valid_mask,  # [T]
        "action_id": ...,
        "action_name": ...,
    }

输出格式（供后续 Pix2Seq-video Tokenizer/Collator 使用）：
    {
        "images": FloatTensor [T,3,H,W]  (0~1)
        "boxes": List[T]  每项 FloatTensor [N_t,4]
        "class_ids": List[T]  每项 LongTensor [N_t]
        "keypoints": List[T]  每项 FloatTensor [N_t,K,3]
        "keypoints_visible": List[T]  每项 FloatTensor [N_t,K]
        "person_ids": List[T]  每项 LongTensor [N_t]
        "object_ids": List[T] or None
        "video_id": str
        "frame_ids": LongTensor [T]
        "image_ids": LongTensor [T]
        "image_hw": LongTensor [2]      # (H, W)（增强后）
        "frame_valid_mask": FloatTensor [T]
        "action_id": int or None
        "action_name": str or None
        "num_instances_per_frame": LongTensor [T]  # 每帧实例数
    }
"""

from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from data.augmentations_video import VideoAugmentor_kp_d, TubeAugmentor_kp_d



class Pix2SeqVideoCollator_kp_d:
    """
    Pix2Seq Video Collator (Decoupled DET / KP / ACT)

    输出给 Trainer 的 batch keys（与你 trainer_video_decoupled.py 对齐）：

    Common:
      images:            [B, T, 3, H, W]
      frame_ids:         [B, T]
      image_hw:          [B, 2]
      video_id:          List[Any] (len=B)

    DET:
      det_input_seq:           [B, S]
      det_target_seq:          [B, S]
      det_token_weights:       [B, S]
      det_input_padding_mask:  [B, S]
      det_target_padding_mask: [B, S]  (可选但推荐保留，便于 debug)

    KP (expanded per-person):
      kp_images:               [Bk, T, 3, H, W]   (Bk can be 0)
      kp_parent_index:         [Bk]
      kp_person_pid:           [Bk]
      kp_input_seq:            [Bk, S]
      kp_target_seq:           [Bk, S]
      kp_token_weights:        [Bk, S]
      kp_input_padding_mask:   [Bk, S]
      kp_frame_ids:            [Bk, T] (可选)

    ACT:
      action_id:               [B] or None
      act_keypoints_traj:      [B, P, T, K, 3] (prompt only)
      act_keypoints_visible_traj: [B, P, T, K]
      act_input_seq:           [B, S] or None
      act_target_seq:          [B, S] or None
      act_token_weights:       [B, S] or None
      act_input_padding_mask:  [B, S] or None
      act_prefix_seq:          [B, 1+T*P*K*2] or None
      act_prefix_padding_mask: [B, 1+T*P*K*2] or None
      act_prefix_len:          int
    """

    def __init__(
        self,
        *,
        token_processor,
        persons_per_clip: int = 2,
        kp_max_persons_per_clip: int = 16,
        corrupt_and_randomise: bool = False,
    ):
        self.token_processor = token_processor
        self.persons_per_clip = int(persons_per_clip)
        self.kp_max_persons_per_clip = int(kp_max_persons_per_clip)
        self.corrupt_and_randomise = bool(corrupt_and_randomise)

        self._is_training = True

        # 常用 token id
        self._pad_id = int(getattr(self.token_processor, "PADDING_TOKEN", 0))
        self._prompt_pad_id = int(getattr(self.token_processor, "PROMPT_PAD_TOKEN", -1))
        self._invisible_id = int(getattr(self.token_processor, "INVISIBLE_KP_TOKEN", -1))

    def set_mode(self, is_training: bool = True):
        self._is_training = bool(is_training)

    # ---------------------- padding mask（PAD + PROMPT_PAD） ---------------------- #
    def _make_padding_mask(self, seq: Tensor) -> Tensor:
        if seq.numel() == 0:
            return torch.empty_like(seq, dtype=torch.bool)
        mask = (seq == self._pad_id)
        if self._prompt_pad_id >= 0:
            mask = mask | (seq == self._prompt_pad_id)
        return mask

    # ---------------------- 根治：统一修复 seq / weights / masks 一致性 ---------------------- #
    def _sanitize_seq_triplet(
        self,
        name: str,
        input_seq: Tensor,
        target_seq: Tensor,
        token_weights: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        强制保证：
          - padding_mask(input/target) 的位置 token_weights == 0
          - INVISIBLE 位置 token_weights == 0（与 trainer 的防御逻辑对齐）
          - 返回 input_mask / target_mask 便于 debug

        注意：不把 PROMPT_PAD 改写成 PAD（避免影响 prefix_len 推断/生成器行为）。
        """
        # dtype 统一
        input_seq = input_seq.to(dtype=torch.long)
        target_seq = target_seq.to(dtype=torch.long)
        token_weights = token_weights.to(dtype=torch.float32)

        in_mask = self._make_padding_mask(input_seq)
        tgt_mask = self._make_padding_mask(target_seq)

        # 任意一侧认为是 padding，都不应该贡献 loss
        union_mask = in_mask | tgt_mask

        # 强制 padding 区域权重为 0
        token_weights = token_weights.masked_fill(union_mask, 0.0)

        # 强制 INVISIBLE 区域权重为 0（避免误监督）
        if self._invisible_id >= 0:
            token_weights = token_weights.masked_fill(target_seq == self._invisible_id, 0.0)

        # 额外保险：PROMPT_PAD 永远不监督
        if self._prompt_pad_id >= 0:
            token_weights = token_weights.masked_fill(target_seq == self._prompt_pad_id, 0.0)

        return input_seq, target_seq, token_weights, in_mask, tgt_mask

    # ---------------------- 检测：构造 tube ---------------------- #
    def _build_tubes_from_sample(
        self,
        sample: Dict[str, Any],
        device: torch.device,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        boxes_list: Optional[List[Any]] = sample.get("boxes", None)
        class_ids_list: Optional[List[Any]] = sample.get("class_ids", None)
        person_ids_list: Optional[List[Any]] = sample.get("person_ids", None)
        object_ids_list: Optional[List[Any]] = sample.get("object_ids", None)

        # 无 boxes：直接返回空
        if boxes_list is None:
            images = sample.get("images", None)
            if isinstance(images, Tensor):
                T = images.shape[0]
            else:
                T = 0
            tube_boxes = torch.zeros((0, T, 4), dtype=torch.float32, device=device)
            tube_class_ids = torch.zeros((0,), dtype=torch.long, device=device)
            tube_valid_mask = torch.zeros((0, T), dtype=torch.bool, device=device)
            return tube_boxes, tube_class_ids, tube_valid_mask

        T = len(boxes_list)

        if class_ids_list is None:
            tube_boxes = torch.zeros((0, T, 4), dtype=torch.float32, device=device)
            tube_class_ids = torch.zeros((0,), dtype=torch.long, device=device)
            tube_valid_mask = torch.zeros((0, T), dtype=torch.bool, device=device)
            return tube_boxes, tube_class_ids, tube_valid_mask

        assert len(class_ids_list) == T
        if person_ids_list is not None:
            assert len(person_ids_list) == T
        if object_ids_list is not None:
            assert len(object_ids_list) == T

        # 1) tube_key -> tube_idx
        tube_key_to_idx: Dict[Tuple[Any, ...], int] = {}
        tube_class_ids_list: List[int] = []
        next_tube_idx = 0

        for t in range(T):
            boxes_t = boxes_list[t]
            if boxes_t is None:
                continue
            b_np = np.asarray(boxes_t)
            if b_np.size == 0:
                continue

            N_t = b_np.shape[0]
            class_ids_t = np.asarray(class_ids_list[t]) if class_ids_list[t] is not None else None
            person_ids_t = (
                np.asarray(person_ids_list[t])
                if person_ids_list is not None and person_ids_list[t] is not None
                else None
            )
            object_ids_t = (
                np.asarray(object_ids_list[t])
                if object_ids_list is not None and object_ids_list[t] is not None
                else None
            )

            for i in range(N_t):
                pid = int(person_ids_t[i]) if person_ids_t is not None else -1
                oid = int(object_ids_t[i]) if object_ids_t is not None else -1

                if pid >= 0:
                    tube_key: Tuple[Any, ...] = ("p", pid)
                elif oid >= 0:
                    tube_key = ("o", oid)
                else:
                    tube_key = ("a", t, i)

                if tube_key not in tube_key_to_idx:
                    tube_key_to_idx[tube_key] = next_tube_idx
                    tube_class_ids_list.append(int(class_ids_t[i]) if class_ids_t is not None else -1)
                    next_tube_idx += 1

        N_tubes = next_tube_idx
        if N_tubes == 0:
            tube_boxes = torch.zeros((0, T, 4), dtype=torch.float32, device=device)
            tube_class_ids = torch.zeros((0,), dtype=torch.long, device=device)
            tube_valid_mask = torch.zeros((0, T), dtype=torch.bool, device=device)
            return tube_boxes, tube_class_ids, tube_valid_mask

        # 2) 填充 tube
        tube_boxes = torch.zeros((N_tubes, T, 4), dtype=torch.float32, device=device)
        tube_valid_mask = torch.zeros((N_tubes, T), dtype=torch.bool, device=device)

        for t in range(T):
            boxes_t = boxes_list[t]
            if boxes_t is None:
                continue
            b_np = np.asarray(boxes_t)
            if b_np.size == 0:
                continue

            boxes_t_tensor = torch.as_tensor(b_np, dtype=torch.float32, device=device)
            N_t = boxes_t_tensor.size(0)

            person_ids_t = (
                np.asarray(person_ids_list[t])
                if person_ids_list is not None and person_ids_list[t] is not None
                else None
            )
            object_ids_t = (
                np.asarray(object_ids_list[t])
                if object_ids_list is not None and object_ids_list[t] is not None
                else None
            )

            for i in range(N_t):
                pid = int(person_ids_t[i]) if person_ids_t is not None else -1
                oid = int(object_ids_t[i]) if object_ids_t is not None else -1

                if pid >= 0:
                    tube_key = ("p", pid)
                elif oid >= 0:
                    tube_key = ("o", oid)
                else:
                    tube_key = ("a", t, i)

                tube_idx = tube_key_to_idx[tube_key]
                tube_boxes[tube_idx, t] = boxes_t_tensor[i]
                tube_valid_mask[tube_idx, t] = True

        tube_class_ids = torch.as_tensor(tube_class_ids_list, dtype=torch.long, device=device)
        return tube_boxes, tube_class_ids, tube_valid_mask

    def _fill_tube_gaps_(
        self,
        tube_boxes: Tensor,       # [N,T,4]
        tube_valid_mask: Tensor,  # [N,T]
        tube_class_ids: Tensor,   # [N]
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        将 tube 中间缺帧用最近邻 bbox 补齐，并把 mask 置 True（只对本来有出现过的 tube 生效）。
        """
        if tube_boxes.numel() == 0:
            return tube_boxes, tube_valid_mask, tube_class_ids

        N, T, _ = tube_boxes.shape
        for n in range(N):
            m = tube_valid_mask[n]
            if not bool(m.any()):
                continue  # 全空（通常是 padding tube），不补

            # 前向填充
            last = None
            for t in range(T):
                if bool(m[t]):
                    last = tube_boxes[n, t].clone()
                else:
                    if last is not None:
                        tube_boxes[n, t] = last
                        m[t] = True

            # 后向补齐（处理前段缺失）
            last = None
            for t in range(T - 1, -1, -1):
                if bool(m[t]):
                    last = tube_boxes[n, t].clone()
                else:
                    if last is not None:
                        tube_boxes[n, t] = last
                        m[t] = True

            tube_valid_mask[n] = m

        return tube_boxes, tube_valid_mask, tube_class_ids

    # ---------------------- 人体：对齐 bbox + kp 轨迹（供 KP/ACT 共用） ---------------------- #
    def _build_human_trajectories_from_sample(
        self,
        sample: Dict[str, Any],
        device: torch.device,
    ) -> Tuple[List[int], Tensor, Tensor, Tensor]:
        """
        返回所有 person（按可见关键点总数降序）：
            pids_sorted : List[int]
            boxes_traj  : [P,T,4]
            kps_traj    : [P,T,K,3]
            vis_traj    : [P,T,K]
        """
        boxes_list = sample.get("boxes", None)
        keypoints_list = sample.get("keypoints", None)
        keypoints_vis_list = sample.get("keypoints_visible", None)
        person_ids_list = sample.get("person_ids", None)
        class_ids_list = sample.get("class_ids", None)

        images = sample.get("images", None)
        if isinstance(images, Tensor):
            T = int(images.shape[0])
        else:
            T = len(keypoints_list) if keypoints_list is not None else 0

        K = int(getattr(self.token_processor, "num_keypoints", 0))
        if K <= 0 or T <= 0:
            return (
                [],
                torch.zeros((0, T, 4), device=device, dtype=torch.float32),
                torch.zeros((0, T, K, 3), device=device, dtype=torch.float32),
                torch.zeros((0, T, K), device=device, dtype=torch.bool),
            )

        # 缺关键字段：直接空
        if (
            boxes_list is None
            or keypoints_list is None
            or keypoints_vis_list is None
            or person_ids_list is None
            or class_ids_list is None
        ):
            return (
                [],
                torch.zeros((0, T, 4), device=device, dtype=torch.float32),
                torch.zeros((0, T, K, 3), device=device, dtype=torch.float32),
                torch.zeros((0, T, K), device=device, dtype=torch.bool),
            )

        pid_to_boxes: Dict[int, Tensor] = {}
        pid_to_kps: Dict[int, Tensor] = {}
        pid_to_vis: Dict[int, Tensor] = {}

        for t in range(T):
            boxes_t_raw = boxes_list[t] if t < len(boxes_list) else None
            kps_t_raw = keypoints_list[t] if t < len(keypoints_list) else None
            vis_t_raw = keypoints_vis_list[t] if t < len(keypoints_vis_list) else None
            pids_t_raw = person_ids_list[t] if t < len(person_ids_list) else None
            cls_t_raw = class_ids_list[t] if t < len(class_ids_list) else None

            if boxes_t_raw is None or kps_t_raw is None or vis_t_raw is None or pids_t_raw is None or cls_t_raw is None:
                continue

            boxes_t = torch.as_tensor(boxes_t_raw, device=device, dtype=torch.float32)  # [N,4]
            kps_t = torch.as_tensor(kps_t_raw, device=device, dtype=torch.float32)      # [N,K,3]
            vis_t = torch.as_tensor(vis_t_raw, device=device, dtype=torch.bool)         # [N,K]
            pids_t = torch.as_tensor(pids_t_raw, device=device, dtype=torch.long)       # [N]
            cls_t = torch.as_tensor(cls_t_raw, device=device, dtype=torch.long)         # [N]

            if kps_t.numel() == 0:
                continue
            if kps_t.dim() != 3 or kps_t.size(1) != K:
                raise ValueError(f"Keypoints K mismatch: expect {K}, got {tuple(kps_t.shape)}")

            N = kps_t.size(0)
            if boxes_t.size(0) != N or vis_t.size(0) != N or pids_t.size(0) != N or cls_t.size(0) != N:
                raise ValueError("boxes/keypoints/vis/pids/class_ids count mismatch at frame t")

            for i in range(N):
                pid = int(pids_t[i].item())
                if pid < 0:
                    continue
                # 只考虑 person（约定 class_id==0）
                if int(cls_t[i].item()) != 0:
                    continue

                if pid not in pid_to_boxes:
                    pid_to_boxes[pid] = torch.zeros((T, 4), dtype=torch.float32, device=device)
                    pid_to_kps[pid] = torch.zeros((T, K, 3), dtype=torch.float32, device=device)
                    pid_to_vis[pid] = torch.zeros((T, K), dtype=torch.bool, device=device)

                pid_to_boxes[pid][t] = boxes_t[i]
                pid_to_kps[pid][t] = kps_t[i]
                pid_to_vis[pid][t] = vis_t[i]

        if not pid_to_kps:
            return (
                [],
                torch.zeros((0, T, 4), device=device, dtype=torch.float32),
                torch.zeros((0, T, K, 3), device=device, dtype=torch.float32),
                torch.zeros((0, T, K), device=device, dtype=torch.bool),
            )

        pid_list = list(pid_to_kps.keys())
        pid_scores = [(pid, int(pid_to_vis[pid].sum().item())) for pid in pid_list]
        pid_scores.sort(key=lambda x: x[1], reverse=True)

        pids_sorted = [pid for pid, _ in pid_scores]
        boxes_traj = torch.stack([pid_to_boxes[pid] for pid in pids_sorted], dim=0)  # [P,T,4]
        kps_traj = torch.stack([pid_to_kps[pid] for pid in pids_sorted], dim=0)      # [P,T,K,3]
        vis_traj = torch.stack([pid_to_vis[pid] for pid in pids_sorted], dim=0)      # [P,T,K]

        return pids_sorted, boxes_traj, kps_traj, vis_traj

    # ---------------------- 根治：ACT 缺人 slot 的 prompt 强制 PROMPT_PAD ---------------------- #
    def _force_act_prompt_pad_for_missing_persons(
        self,
        act_input_seq: Tensor,        # [B,S]
        act_target_seq: Tensor,       # [B,S]
        act_token_weights: Tensor,    # [B,S]
        persons_eff_list: List[int],  # len=B, 每个 clip 实际有效人数（<= P_target）
        T: int,
        P_target: int,
        K: int,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        假定 ACT target 的 prompt layout 为：
            [ACT] + prompt(T*P*K*2) + action_token + EOS + PAD...
        对缺失 person 的 prompt span：
            全部写 PROMPT_PAD_TOKEN，并确保 weights=0

        ★注意：本版本采用“前填充”策略：
            若 P_eff < P_target，则缺失 persons 对应的 slot 为 [0, P_pad)，其中 P_pad = P_target - P_eff
        """
        if self._prompt_pad_id < 0:
            return act_input_seq, act_target_seq, act_token_weights

        B, S = act_target_seq.shape
        prompt_len = T * P_target * K * 2
        prompt_start = 1
        prompt_end = min(1 + prompt_len, S)

        span_per_person_per_frame = K * 2  # 每帧每人 token 数

        for b in range(B):
            P_eff = int(persons_eff_list[b])
            if P_eff <= 0:
                P_pad = P_target
            else:
                P_pad = max(P_target - P_eff, 0)

            if P_pad <= 0:
                continue

            # 缺失 persons: [0, P_pad)
            for p in range(0, P_pad):
                for t in range(T):
                    offset = (t * P_target + p) * span_per_person_per_frame
                    s0 = prompt_start + offset
                    s1 = min(s0 + span_per_person_per_frame, prompt_end)
                    if s0 >= prompt_end:
                        continue

                    act_input_seq[b, s0:s1] = self._prompt_pad_id
                    act_target_seq[b, s0:s1] = self._prompt_pad_id
                    act_token_weights[b, s0:s1] = 0.0

        return act_input_seq, act_target_seq, act_token_weights

    # ---------------------- 主入口 ---------------------- #
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        B = len(batch)
        assert B > 0

        # device 推断
        if isinstance(batch[0].get("images", None), Tensor):
            device = batch[0]["images"].device
        else:
            device = torch.device("cpu")

        # ===== 1) 基本视频信息 =====
        videos = torch.stack([x["images"] for x in batch], dim=0)   # [B,T,3,H,W]
        T = int(videos.shape[1])

        frame_ids = torch.stack(
            [
                (x["frame_ids"].long() if isinstance(x["frame_ids"], Tensor)
                else torch.as_tensor(x["frame_ids"], dtype=torch.long, device=device))
                for x in batch
            ],
            dim=0,
        )  # [B,T]

        image_hw = torch.stack(
            [torch.tensor(list(x["image_hw"]), dtype=torch.long, device=device) for x in batch],
            dim=0,
        )  # [B,2]

        video_ids = [x["video_id"] for x in batch]

        batch_data: Dict[str, Any] = {
            "images": videos,
            "frame_ids": frame_ids,
            "image_hw": image_hw,
            "video_id": video_ids,
        }

        # ===== 2) DET：tube + build_detection_sequences =====
        tube_list: List[Tensor] = []
        cls_list: List[Tensor] = []
        tube_valid_masks: List[Tensor] = []

        for x in batch:
            tube_boxes_i, class_ids_i, tube_valid_mask_i = self._build_tubes_from_sample(x, device=device)
            tube_boxes_i, tube_valid_mask_i, class_ids_i = self._fill_tube_gaps_(
                tube_boxes_i, tube_valid_mask_i, class_ids_i
            )

            # 训练时可随机打乱 tube 顺序
            if self._is_training and tube_boxes_i.shape[0] > 0:
                idx = torch.randperm(tube_boxes_i.shape[0], device=device)
                tube_boxes_i = tube_boxes_i[idx]
                class_ids_i = class_ids_i[idx]
                tube_valid_mask_i = tube_valid_mask_i[idx]

            tube_list.append(tube_boxes_i)
            cls_list.append(class_ids_i)
            tube_valid_masks.append(tube_valid_mask_i)

        num_tubes = [t.shape[0] for t in tube_list]
        N_max = max(num_tubes) if num_tubes else 0

        if N_max == 0:
            tube_boxes_batch = torch.empty((B, 0, T, 4), device=device, dtype=torch.float32)
            class_ids_batch = torch.empty((B, 0), device=device, dtype=torch.long)
            tube_valid_mask_batch = torch.empty((B, 0, T), device=device, dtype=torch.bool)
        else:
            padded_tubes: List[Tensor] = []
            padded_cls: List[Tensor] = []
            padded_masks: List[Tensor] = []
            for tubes_i, cls_i, mask_i in zip(tube_list, cls_list, tube_valid_masks):
                N_i = tubes_i.shape[0]
                if N_i < N_max:
                    pad_tubes = torch.zeros((N_max - N_i, T, 4), dtype=tubes_i.dtype, device=device)
                    pad_cls = torch.full((N_max - N_i,), -1, dtype=cls_i.dtype, device=device)
                    pad_mask = torch.zeros((N_max - N_i, T), dtype=mask_i.dtype, device=device)
                    tubes_i = torch.cat([tubes_i, pad_tubes], dim=0)
                    cls_i = torch.cat([cls_i, pad_cls], dim=0)
                    mask_i = torch.cat([mask_i, pad_mask], dim=0)
                padded_tubes.append(tubes_i)
                padded_cls.append(cls_i)
                padded_masks.append(mask_i)

            tube_boxes_batch = torch.stack(padded_tubes, dim=0)       # [B,N_max,T,4]
            class_ids_batch = torch.stack(padded_cls, dim=0)          # [B,N_max]
            tube_valid_mask_batch = torch.stack(padded_masks, dim=0)  # [B,N_max,T]

        batch_data["tube_boxes"] = tube_boxes_batch
        batch_data["class_ids"] = class_ids_batch
        batch_data["tube_valid_mask"] = tube_valid_mask_batch

        det_input_seq, det_target_seq, det_token_weights = self.token_processor.build_detection_sequences(
            tube_boxes=tube_boxes_batch,
            class_ids=class_ids_batch,
        )

        det_input_seq, det_target_seq, det_token_weights, det_in_mask, det_tgt_mask = self._sanitize_seq_triplet(
            "DET", det_input_seq, det_target_seq, det_token_weights
        )

        batch_data.update(
            {
                "det_input_seq": det_input_seq,
                "det_target_seq": det_target_seq,
                "det_token_weights": det_token_weights,
                "det_input_padding_mask": det_in_mask,
                "det_target_padding_mask": det_tgt_mask,
            }
        )

        # ===== 3) KP：逐人展开 build_kp_sequences =====
        has_kp = all(k in batch[0] for k in ["keypoints", "keypoints_visible", "person_ids", "class_ids", "boxes"])

        if has_kp:
            kp_parent_b: List[int] = []
            kp_person_pid: List[int] = []
            kp_tube_boxes_1p: List[Tensor] = []
            kp_kps_1p: List[Tensor] = []
            kp_vis_1p: List[Tensor] = []
            kp_images: List[Tensor] = []
            kp_frame_ids: List[Tensor] = []

            for b_idx, x in enumerate(batch):
                pids_sorted, boxes_traj, kps_traj, vis_traj = self._build_human_trajectories_from_sample(
                    x, device=device
                )
                if len(pids_sorted) == 0:
                    continue

                pids_sorted = pids_sorted[: self.kp_max_persons_per_clip]
                P = len(pids_sorted)

                for p_local in range(P):
                    kp_parent_b.append(b_idx)
                    kp_person_pid.append(int(pids_sorted[p_local]))
                    kp_tube_boxes_1p.append(boxes_traj[p_local])
                    kp_kps_1p.append(kps_traj[p_local])
                    kp_vis_1p.append(vis_traj[p_local])
                    kp_images.append(videos[b_idx])
                    kp_frame_ids.append(frame_ids[b_idx])

            Bk = len(kp_parent_b)
            if Bk == 0:
                batch_data.update(
                    {
                        "kp_images": videos[:0],
                        "kp_parent_index": torch.empty((0,), device=device, dtype=torch.long),
                        "kp_person_pid": torch.empty((0,), device=device, dtype=torch.long),
                        "kp_input_seq": torch.empty((0, int(self.token_processor.max_seq_len)), device=device, dtype=torch.long),
                        "kp_target_seq": torch.empty((0, int(self.token_processor.max_seq_len)), device=device, dtype=torch.long),
                        "kp_token_weights": torch.empty((0, int(self.token_processor.max_seq_len)), device=device, dtype=torch.float32),
                        "kp_input_padding_mask": torch.empty((0, int(self.token_processor.max_seq_len)), device=device, dtype=torch.bool),

                        # ★新增：方案三交错prompt辅助输出
                        "kp_bbox_prompt_seq": torch.empty((0, T, 4), device=device, dtype=torch.long),
                        "kp_force_prompt_mask": torch.empty((0, int(self.token_processor.max_seq_len)), device=device, dtype=torch.bool),
                    }
                )
            else:
                kp_images_t = torch.stack(kp_images, dim=0)
                kp_frame_ids_t = torch.stack(kp_frame_ids, dim=0)

                kp_boxes_t = torch.stack(kp_tube_boxes_1p, dim=0)  # [Bk,T,4]
                kp_kps_t = torch.stack(kp_kps_1p, dim=0)          # [Bk,T,K,3]
                kp_vis_t = torch.stack(kp_vis_1p, dim=0)          # [Bk,T,K]

                kp_input_seq, kp_target_seq, kp_token_weights = self.token_processor.build_kp_sequences(
                    tube_boxes=kp_boxes_t,
                    keypoints_traj=kp_kps_t,
                    keypoints_visible_traj=kp_vis_t,
                    xy_order="xy",
                )

                kp_input_seq, kp_target_seq, kp_token_weights, kp_in_mask, kp_tgt_mask = self._sanitize_seq_triplet(
                    "KP", kp_input_seq, kp_target_seq, kp_token_weights
                )

                # -------------------------
                # ✅ 方案三(KP交错序列)下的 prefix / prompt 辅助信息
                #   target layout:
                #     [KP]
                #     [f0 bbox(4)] [f0 kp(2K)]
                #     [f1 bbox(4)] [f1 kp(2K)]
                #     ...
                #     [f{T-1} bbox(4)] [f{T-1} kp(2K)]
                #     [EOS]
                # -------------------------
                K = int(getattr(self.token_processor, "num_keypoints", 0))
                frame_block = 4 + K * 2  # 每帧: bbox(4) + kp(2K)

                region_start = 1
                region_end = region_start + T * frame_block
                eos_pos = region_end

                # 全帧 bbox prompt 单独抽取：[Bk,T,4]
                kp_region = kp_target_seq[:, region_start:region_end].contiguous()  # [Bk, T*frame_block]
                kp_region = kp_region.view(Bk, T, frame_block)                      # [Bk, T, 4+2K]
                kp_bbox_prompt_seq = kp_region[:, :, :4].contiguous()               # [Bk, T, 4]

                # 提示位 mask（KP token + 每帧bbox 4 token + EOS）
                kp_force_prompt_mask = torch.zeros_like(kp_target_seq, dtype=torch.bool)  # [Bk,S]
                kp_force_prompt_mask[:, 0] = True
                for t in range(T):
                    base = region_start + t * frame_block
                    kp_force_prompt_mask[:, base: base + 4] = True
                if eos_pos < kp_force_prompt_mask.size(1):
                    kp_force_prompt_mask[:, eos_pos] = True

                batch_data.update(
                    {
                        "kp_images": kp_images_t,
                        "kp_frame_ids": kp_frame_ids_t,
                        "kp_parent_index": torch.tensor(kp_parent_b, device=device, dtype=torch.long),
                        "kp_person_pid": torch.tensor(kp_person_pid, device=device, dtype=torch.long),
                        "kp_input_seq": kp_input_seq,
                        "kp_target_seq": kp_target_seq,
                        "kp_token_weights": kp_token_weights,
                        "kp_input_padding_mask": kp_in_mask,
                        "kp_target_padding_mask": kp_tgt_mask,

                        # ★新增：方案三交错prompt辅助输出（不影响训练）
                        "kp_bbox_prompt_seq": kp_bbox_prompt_seq,       # [Bk,T,4]
                        "kp_force_prompt_mask": kp_force_prompt_mask,   # [Bk,S]
                    }
                )
        else:
            batch_data["kp_images"] = None  # 明确：该 batch 没有 KP 监督

        # ===== 4) ACT：GT-KP prompt -> build_action_sequences =====
        has_action = ("action_id" in batch[0]) and has_kp
        if has_action:
            action_ids: List[int] = []
            act_kps_list: List[Tensor] = []
            act_vis_list: List[Tensor] = []
            persons_eff_list: List[int] = []  # 每个 clip 实际人数（<= P_target）

            P_target = self.persons_per_clip
            K = int(getattr(self.token_processor, "num_keypoints", 0))

            # --------- 构造 ACT 的 KP prompt（前填充缺人 slot）---------
            for x in batch:
                aid = x["action_id"]
                if isinstance(aid, Tensor):
                    aid = int(aid.item())
                action_ids.append(int(aid))

                pids_sorted, boxes_traj, kps_traj, vis_traj = self._build_human_trajectories_from_sample(
                    x, device=device
                )

                if len(pids_sorted) == 0:
                    P_eff = 0
                    kps_i = torch.zeros((P_target, T, K, 3), device=device, dtype=torch.float32)
                    vis_i = torch.zeros((P_target, T, K), device=device, dtype=torch.bool)
                else:
                    P_eff = min(P_target, int(kps_traj.size(0)))
                    kps_real = kps_traj[:P_eff]
                    vis_real = vis_traj[:P_eff]

                    P_pad = P_target - P_eff
                    if P_pad > 0:
                        pad_kps = torch.zeros((P_pad, T, K, 3), device=device, dtype=torch.float32)
                        pad_vis = torch.zeros((P_pad, T, K), device=device, dtype=torch.bool)
                        # ★前填充：pad 在前，真实人在后
                        kps_i = torch.cat([pad_kps, kps_real], dim=0)
                        vis_i = torch.cat([pad_vis, vis_real], dim=0)
                    else:
                        kps_i = kps_real
                        vis_i = vis_real

                persons_eff_list.append(P_eff)
                act_kps_list.append(kps_i)
                act_vis_list.append(vis_i)

            act_keypoints_traj = torch.stack(act_kps_list, dim=0)          # [B,P,T,K,3]
            act_keypoints_visible_traj = torch.stack(act_vis_list, dim=0)  # [B,P,T,K]
            action_ids_tensor = torch.tensor(action_ids, device=device, dtype=torch.long)

            act_input_seq, act_target_seq, act_token_weights = self.token_processor.build_action_sequences(
                keypoints_traj=act_keypoints_traj,
                keypoints_visible_traj=act_keypoints_visible_traj,
                action_ids=action_ids_tensor,
                persons_per_clip=self.persons_per_clip,
                xy_order="xy",
            )

            # 根治：缺人 slot 的 prompt span 强制 PROMPT_PAD，并把 weights 清零（适配前填充）
            act_input_seq, act_target_seq, act_token_weights = self._force_act_prompt_pad_for_missing_persons(
                act_input_seq=act_input_seq,
                act_target_seq=act_target_seq,
                act_token_weights=act_token_weights,
                persons_eff_list=persons_eff_list,
                T=T,
                P_target=P_target,
                K=K,
            )

            # 统一对齐 mask/weights（PAD/PP/INVISIBLE 都不会被监督）
            act_input_seq, act_target_seq, act_token_weights, act_in_mask, act_tgt_mask = self._sanitize_seq_triplet(
                "ACT", act_input_seq, act_target_seq, act_token_weights
            )

            # ACT prefix = [ACT] + kp_prompt（T*P*K*2）
            prompt_len = T * P_target * K * 2
            act_prefix_len = 1 + prompt_len

            # --------------------- 动作 token 监督：从 target 扫描得到位置 ---------------------
            B2, S2 = act_target_seq.shape
            rows = torch.arange(B2, device=device)

            # 用“真实可见关键点”判断该 clip 是否有有效人（与你 debug 的 vis_cnt 一致）
            vis_sum = act_keypoints_visible_traj.to(torch.bool).sum(dim=(2, 3))  # [B,P]
            has_any_person = (vis_sum.sum(dim=1) > 0)                            # [B]

            # action token 范围（优先走 token_processor 暴露的字段，兼容旧实现）
            a0 = int(getattr(self.token_processor, "action_vocab_shift",
                    getattr(self.token_processor, "ACTION_TOKEN_START", -1)))
            if a0 < 0:
                fake = int(getattr(self.token_processor, "FAKE_CLASS_TOKEN", 90))
                a0 = fake + 1
            a1 = a0 + int(getattr(self.token_processor, "num_actions", 5))

            is_act = (act_target_seq >= a0) & (act_target_seq < a1)   # [B,S]
            action_cnt = is_act.sum(dim=1)
            action_pos = is_act.float().argmax(dim=1)                 # [B]

            valid_action = has_any_person & (action_cnt > 0) & (~act_tgt_mask[rows, action_pos])
            act_token_weights[rows[valid_action], action_pos[valid_action]] = 1.0

            # EOS 监督（同样从 target 扫描）
            eos_id = int(getattr(self.token_processor, "EOS_TOKEN", 2))
            eos_w = float(getattr(self.token_processor, "bos_eos_token_weight", 0.1))

            is_eos = (act_target_seq == eos_id)
            eos_cnt = is_eos.sum(dim=1)
            eos_pos = is_eos.float().argmax(dim=1)

            valid_eos = has_any_person & (eos_cnt > 0) & (~act_tgt_mask[rows, eos_pos])
            act_token_weights[rows[valid_eos], eos_pos[valid_eos]] = eos_w
            # -------------------------------------------------------------------------

            # 最终保险：mask=True 的位置权重必须为 0（保持与你的断言一致）
            act_token_weights = act_token_weights.masked_fill(act_in_mask, 0.0)
            act_token_weights = act_token_weights.masked_fill(act_tgt_mask, 0.0)

            act_input_padding_mask = act_in_mask

            act_prefix_seq = act_target_seq[:, :act_prefix_len].contiguous()
            act_prefix_padding_mask = self._make_padding_mask(act_prefix_seq)

            batch_data.update(
                {
                    "action_id": action_ids_tensor,
                    "act_keypoints_traj": act_keypoints_traj,
                    "act_keypoints_visible_traj": act_keypoints_visible_traj,
                    "act_input_seq": act_input_seq,
                    "act_target_seq": act_target_seq,
                    "act_token_weights": act_token_weights,
                    "act_input_padding_mask": act_input_padding_mask,
                    "act_target_padding_mask": act_tgt_mask,
                    "act_prefix_seq": act_prefix_seq,
                    "act_prefix_padding_mask": act_prefix_padding_mask,
                    "act_prefix_len": act_prefix_len,
                }
            )
        else:
            batch_data["action_id"] = None
            batch_data["act_input_seq"] = None
            batch_data["act_target_seq"] = None
            batch_data["act_token_weights"] = None
            batch_data["act_input_padding_mask"] = None
            batch_data["act_target_padding_mask"] = None
            batch_data["act_prefix_seq"] = None
            batch_data["act_prefix_padding_mask"] = None
            batch_data["act_prefix_len"] = 0

        return batch_data




class Pix2SeqVideoDataset_kp_d(Dataset):
    """
    Pix2Seq-Video 版数据集整合器：
      - 封装任意“视频 clip 数据集”（如 VideoClipDataset_kp_d / NTUVideoClipDatasetFromAvi_kp_d）
      - 统一做 Video 级图像增强 + Tube 级时序增强
      - 输出结构化的 tensor，供 Pix2Seq-video 的 Tokenizer / Collator 使用
    """

    def __init__(
        self,
        base_dataset,                    # 视频基础数据集，__getitem__ 返回 sample dict
        num_classes: int,
        training: bool = True,
        max_num_objects_per_frame: Optional[int] = 100,
        image_size: int = 640,
        jitter_scale: Tuple[float, float] = (0.3, 2.0),
        color_jitter_strength: float = 0.4,
        use_video_augmentation: bool = True,
        use_tube_augmentation: bool = True,
    ):
        """
        Args:
            base_dataset:  视频基础数据集（已经完成标注融合 / ID 分配等）
            num_classes:   真实检测类别数（不含 fake 类），主要为后续 Tokenizer 使用
            training:      训练 / 测试 标志；控制是否启用随机增强
            max_num_objects_per_frame: 每帧最多保留多少个实例，None 表示不过滤
            image_size:    VideoAugmentor_kp_d 中使用的目标尺寸
            jitter_scale:  VideoAugmentor_kp_d 的缩放抖动范围
            color_jitter_strength: 颜色抖动强度
            use_video_augmentation: 是否启用图像级视频增强
            use_tube_augmentation:  是否启用 Tube 级时序增强（随机打乱实例顺序）
        """
        super().__init__()

        self.base_dataset = base_dataset
        self.num_classes = num_classes
        self.training = training
        self.max_num_objects_per_frame = max_num_objects_per_frame
        self.categories = base_dataset.categories

        # 图像级视频增强（时空一致）
        self.video_augmentor: Optional[VideoAugmentor_kp_d] = None
        if use_video_augmentation:
            self.video_augmentor = VideoAugmentor_kp_d(
                image_size=image_size,
                jitter_scale=jitter_scale,
                color_jitter_strength=color_jitter_strength,
                training=training,          # True: 随机缩放/裁剪/翻转；False: 仅缩放+pad
                normalize_boxes=True,       # 输出归一化 bbox，方便 Tokenizer 使用
            )

        # Tube 级增强：只在训练模式下启用
        self.tube_augmentor: Optional[TubeAugmentor_kp_d] = None
        if use_tube_augmentation:
            self.tube_augmentor = TubeAugmentor_kp_d(
                shuffle_prob=1.0 if training else 0.0
            )

    # -------------------------------------------------------------
    # Dataset 接口
    # -------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        # 1. 从基础数据集取出一个 clip（T 帧）
        sample = self.base_dataset[index]

        # 2. 图像级视频增强（训练 & 测试都调用；内部根据 training 决定是否随机）
        if self.video_augmentor is not None:
            sample = self.video_augmentor(sample)

        # 3. Tube 级增强：只在训练时启用（随机打乱实例顺序，但保持 ID 对应关系）
        if self.training and self.tube_augmentor is not None:
            sample = self.tube_augmentor(sample)

        # 4. 将 numpy / list 转成 torch tensor，并可选限制每帧实例数
        images_np: np.ndarray = sample["images"]   # [T,H,W,3] uint8
        T, H, W, _ = images_np.shape

        # 4.1 图像 [T,3,H,W]，归一化到 [0,1]
        images = torch.from_numpy(images_np.astype(np.float32) / 255.0)  # [T,H,W,3]
        images = images.permute(0, 3, 1, 2).contiguous()                 # [T,3,H,W]

        # 4.2 将逐帧的 boxes / class_ids / keypoints 等转为 List[Tensor]
        boxes_list_np: List[np.ndarray] = sample["boxes"]
        class_ids_list_np: List[np.ndarray] = sample["class_ids"]
        keypoints_list_np: List[np.ndarray] = sample["keypoints"]
        keypoints_vis_list_np: List[np.ndarray] = sample["keypoints_visible"]
        person_ids_list_np: List[np.ndarray] = sample["person_ids"]
        object_ids_list_np: Optional[List[np.ndarray]] = sample.get("object_ids", None)

        boxes_list: List[torch.Tensor] = []
        class_ids_list: List[torch.Tensor] = []
        keypoints_list: List[torch.Tensor] = []
        keypoints_vis_list: List[torch.Tensor] = []
        person_ids_list: List[torch.Tensor] = []
        object_ids_list: Optional[List[torch.Tensor]] = [] if object_ids_list_np is not None else None

        num_instances_per_frame: List[int] = []

        for t in range(T):
            boxes_t = torch.as_tensor(boxes_list_np[t], dtype=torch.float32)         # [N_t,4]
            cls_t = torch.as_tensor(class_ids_list_np[t], dtype=torch.long)         # [N_t]
            kps_t = torch.as_tensor(keypoints_list_np[t], dtype=torch.float32)      # [N_t,K,3]
            kps_vis_t = torch.as_tensor(keypoints_vis_list_np[t], dtype=torch.float32)  # [N_t,K]
            pid_t = torch.as_tensor(person_ids_list_np[t], dtype=torch.long)        # [N_t]

            if object_ids_list_np is not None:
                oid_t = torch.as_tensor(object_ids_list_np[t], dtype=torch.long)    # [N_t]
            else:
                oid_t = None

            N_t = boxes_t.shape[0]

            # 4.3 可选：限制每帧实例数
            if (self.max_num_objects_per_frame is not None) and (N_t > self.max_num_objects_per_frame):
                if self.training:
                    keep_idx = torch.randperm(N_t)[: self.max_num_objects_per_frame]
                else:
                    keep_idx = torch.arange(self.max_num_objects_per_frame)

                boxes_t = boxes_t[keep_idx]
                cls_t = cls_t[keep_idx]
                kps_t = kps_t[keep_idx]
                kps_vis_t = kps_vis_t[keep_idx]
                pid_t = pid_t[keep_idx]
                if oid_t is not None:
                    oid_t = oid_t[keep_idx]

            num_instances_per_frame.append(boxes_t.shape[0])

            boxes_list.append(boxes_t)
            class_ids_list.append(cls_t)
            keypoints_list.append(kps_t)
            keypoints_vis_list.append(kps_vis_t)
            person_ids_list.append(pid_t)
            if object_ids_list is not None:
                object_ids_list.append(oid_t)

        # 5. 其他元信息
        frame_ids = torch.as_tensor(sample["frame_ids"], dtype=torch.long)              # [T]
        image_ids = torch.as_tensor(sample["image_ids"], dtype=torch.long)              # [T]
        image_hw = torch.as_tensor(sample["image_hw"], dtype=torch.long)                # (H,W)
        frame_valid_mask = torch.as_tensor(sample["frame_valid_mask"], dtype=torch.float32)  # [T]
        num_instances_per_frame = torch.as_tensor(num_instances_per_frame, dtype=torch.long)  # [T]

        # 动作信息：直接透传基础数据集的字段
        action_id = sample.get("action_id", None)              # 0~3 或 None
        raw_action_id = sample.get("raw_action_id", None)      # 41/50/51/52 或 None
        action_name = sample.get("action_name", None)          # "fall"/"slap"/"kick"/"push" 或 None
        video_id = sample.get("video_id", f"vid_{index}")

        # 6. 整理输出字典
        out: Dict[str, Any] = {
            "images": images,                           # [T,3,H,W]
            "boxes": boxes_list,                        # 长度 T，每项 [N_t,4]
            "class_ids": class_ids_list,                # 长度 T，每项 [N_t]
            "keypoints": keypoints_list,                # 长度 T，每项 [N_t,K,3]
            "keypoints_visible": keypoints_vis_list,    # 长度 T，每项 [N_t,K]
            "person_ids": person_ids_list,              # 长度 T，每项 [N_t]
            "video_id": video_id,
            "frame_ids": frame_ids,                     # [T]
            "image_ids": image_ids,                     # [T]
            "image_hw": image_hw,                       # (H,W)
            "frame_valid_mask": frame_valid_mask,       # [T]
            "num_instances_per_frame": num_instances_per_frame,  # [T]

            # 动作相关
            "action_id": action_id,                     # 0~3
            "raw_action_id": raw_action_id,             # 41/50/51/52
            "action_name": action_name,                 # 字符串
        }

        if object_ids_list is not None:
            out["object_ids"] = object_ids_list         # 长度 T，每项 [N_t]

        return out


