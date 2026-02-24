# evaluation/detect_token_accuracy_evaluator.py
from typing import Dict, Optional, Tuple

import torch


class DetectTokenAccuracyEvaluator:
    """
    视频 tube-based 目标检测 token 级评估器。

    序列格式约定：
        [DET,
         tube0_bbox_yx_yx...(4*T tokens), tube0_class,
         tube1_bbox_yx_yx...(4*T tokens), tube1_class,
         ...
         EOS,
         PAD...]

    参数
    ----
    pad_token_id : int
        padding 的 token id
    eos_token_id : int
        EOS 的 token id
    tokens_per_tube : Optional[int]
        每条 tube 的 token 数（不含 DET / EOS / PAD）。
        对当前设计为：tokens_per_tube = 4 * tube_frames + 1
        - 若为 None，则只计算整体 token 精度和 EOS 位置精度；
        - 若不为 None，则额外计算 tube 粒度的指标（bbox/class/完整 tube / tube 数量）。
    """

    def __init__(
        self,
        pad_token_id: int = 0,
        eos_token_id: int = 2,
        tokens_per_tube: Optional[int] = None,
    ):
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.tokens_per_tube = tokens_per_tube
        self.reset_metrics()

    # ========== 公共状态管理 ==========

    def reset_metrics(self):
        """重置一个 epoch 内的累计指标。"""
        self.accumulated_metrics: Dict[str, list] = {}

    def _zero_counts(self, device):
        z = torch.zeros((), device=device, dtype=torch.long)
        return z, z

    def _get_correct_and_total(
        self,
        pred_tokens: torch.Tensor,
        target_seq: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        correct = (pred_tokens == target_seq) & mask
        total = mask
        return correct.sum(), total.sum()

    # ========== 单样本级：基础 token 精度 ==========

    def _compute_token_accuracy_single(
        self,
        pred_tokens: torch.Tensor,
        target_seq: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        c, t = self._get_correct_and_total(pred_tokens, target_seq, valid_mask)
        return {"det_token": (c, t)}

    # ========== 单样本级：tube 内 bbox / class 精度（需 tokens_per_tube） ==========

    def _compute_bbox_and_class_accuracy_single(
        self,
        pred_tokens: torch.Tensor,
        target_seq: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """
        假设：
            idx 0 为 DET_TASK_TOKEN，不参与 tube 分组
            从 idx=1 开始到 EOS 之前，为若干条 tube 串联：
                tube_k: [4T bbox token, 1 class token]，长度 = tokens_per_tube
        """
        if self.tokens_per_tube is None:
            device = target_seq.device
            zc, zt = self._zero_counts(device)
            return {
                "det_bbox_token": (zc, zt),
                "det_class_token": (zc, zt),
            }

        device = target_seq.device
        seq_len = target_seq.size(0)
        idxs = torch.arange(seq_len, device=device)

        eos_pos = (target_seq == self.eos_token_id).nonzero(as_tuple=True)[0]
        eos_idx = eos_pos[0].item() if len(eos_pos) > 0 else seq_len

        # tube 区域：跳过 idx=0 的 DET，只看 [1, eos_idx) 区间
        tube_start = 1
        tube_end = max(tube_start, eos_idx)
        tube_region_len = max(0, tube_end - tube_start)

        num_tubes = tube_region_len // self.tokens_per_tube
        if num_tubes == 0:
            zc, zt = self._zero_counts(device)
            return {
                "det_bbox_token": (zc, zt),
                "det_class_token": (zc, zt),
            }

        body_end = tube_start + num_tubes * self.tokens_per_tube

        tube_region_mask = (
            (idxs >= tube_start) & (idxs < body_end) & valid_mask
        )
        rel_pos = (idxs - tube_start) % self.tokens_per_tube

        # tube 内最后一个 token 作为 class，其余 token 视为 bbox
        bbox_mask = tube_region_mask & (rel_pos < (self.tokens_per_tube - 1))
        class_mask = tube_region_mask & (rel_pos == (self.tokens_per_tube - 1))

        c_bbox, t_bbox = self._get_correct_and_total(
            pred_tokens, target_seq, bbox_mask
        )
        c_cls, t_cls = self._get_correct_and_total(
            pred_tokens, target_seq, class_mask
        )

        return {
            "det_bbox_token": (c_bbox, t_bbox),
            "det_class_token": (c_cls, t_cls),
        }

    # ========== 单样本级：整条 tube 精度（需 tokens_per_tube） ==========

    def _compute_tube_instance_accuracy_single(
        self,
        pred_tokens: torch.Tensor,
        target_seq: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """
        一条 tube 定义为 tokens_per_tube 个连续 token：
            [4T bbox tokens, 1 class token]

        tube 正确：在该 tube 的所有有效位置（valid_mask==1）上，预测与标签完全一致。
        """
        if self.tokens_per_tube is None:
            device = target_seq.device
            zc, zt = self._zero_counts(device)
            return {"det_tube": (zc, zt)}

        device = target_seq.device
        seq_len = target_seq.size(0)

        eos_pos = (target_seq == self.eos_token_id).nonzero(as_tuple=True)[0]
        eos_idx = eos_pos[0].item() if len(eos_pos) > 0 else seq_len

        tube_start = 1
        tube_end = max(tube_start, eos_idx)
        tube_region_len = max(0, tube_end - tube_start)
        num_tubes = tube_region_len // self.tokens_per_tube

        if num_tubes == 0:
            zc, zt = self._zero_counts(device)
            return {"det_tube": (zc, zt)}

        body_end = tube_start + num_tubes * self.tokens_per_tube

        body_pred = pred_tokens[tube_start:body_end].view(num_tubes, self.tokens_per_tube)
        body_tgt = target_seq[tube_start:body_end].view(num_tubes, self.tokens_per_tube)
        body_mask = valid_mask[tube_start:body_end].view(num_tubes, self.tokens_per_tube)

        # tube 正确：所有需要比较的位置都相等
        group_correct = ((body_pred == body_tgt) | (~body_mask)).all(dim=1)
        group_valid = body_mask.any(dim=1)

        c = group_correct.sum()
        t = group_valid.sum()
        return {"det_tube": (c, t)}

    # ========== 单样本级：序列级指标（EOS & tube 数量） ==========

    def _compute_sequence_metrics_single(
        self,
        pred_tokens: torch.Tensor,
        target_seq: torch.Tensor,
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """
        序列级指标：
        - det_sequence_num_tubes: 预测 tube 数是否等于目标 tube 数
        - det_sequence_eos_pos : 预测 EOS 位置是否与目标一致
        """
        device = target_seq.device
        seq_len = target_seq.size(0)

        tgt_eos_pos = (target_seq == self.eos_token_id).nonzero(as_tuple=True)[0]
        tgt_eos_idx = tgt_eos_pos[0].item() if len(tgt_eos_pos) > 0 else seq_len

        pred_eos_pos = (pred_tokens == self.eos_token_id).nonzero(as_tuple=True)[0]
        pred_eos_idx = pred_eos_pos[0].item() if len(pred_eos_pos) > 0 else seq_len

        one = torch.ones((), device=device, dtype=torch.long)

        # EOS 位置是否一致
        eos_correct = torch.tensor(
            [
                len(pred_eos_pos) > 0
                and len(tgt_eos_pos) > 0
                and pred_eos_pos[0].item() == tgt_eos_pos[0].item()
            ],
            device=device,
            dtype=torch.long,
        )
        metrics = {
            "det_sequence_eos_pos": (eos_correct, one),
        }

        # 若 tokens_per_tube 未指定，则不做 tube 数判断
        if self.tokens_per_tube is None:
            return metrics

        # 目标 tube 数
        tube_start = 1
        tube_end_tgt = max(tube_start, tgt_eos_idx)
        tube_len_tgt = max(0, tube_end_tgt - tube_start)
        tgt_tubes = tube_len_tgt // self.tokens_per_tube

        # 预测 tube 数（按预测 EOS 截断）
        tube_end_pred = max(tube_start, pred_eos_idx)
        tube_len_pred = max(0, tube_end_pred - tube_start)
        pred_tubes = tube_len_pred // self.tokens_per_tube

        inst_correct = torch.tensor(
            [pred_tubes == tgt_tubes],
            device=device,
            dtype=torch.long,
        )

        metrics["det_sequence_num_tubes"] = (inst_correct, one)
        return metrics

    # ========== 多样本 / batch 汇总 ==========

    def gather_and_normalize_metrics(
        self,
        raw_metrics: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
        gather_fn=None,
    ) -> Dict[str, float]:
        """
        将 (correct, total) 聚合到所有进程，并归一化为 0~1 精度。
        同时累加到 epoch 级指标。
        """
        out: Dict[str, float] = {}
        for name, (c, t) in raw_metrics.items():
            local = torch.stack([c, t])
            if gather_fn is not None:
                gathered = gather_fn(local)
                correct = gathered[0::2].sum()
                total = gathered[1::2].sum()
            else:
                correct, total = local
            acc = (correct / total.clamp(min=1)).item()
            out[name] = acc

            if name not in self.accumulated_metrics:
                self.accumulated_metrics[name] = []
            self.accumulated_metrics[name].append(acc)
        return out

    def compute_batch_metrics(
        self,
        logits: torch.Tensor,         # [B, S, V]
        target_seq: torch.Tensor,     # [B, S]
        token_weights: Optional[torch.Tensor] = None,  # [B, S]
        gather_fn=None,
    ) -> Dict[str, float]:
        """
        外部接口：对一个 batch 计算各种精度指标。
        """
        device = target_seq.device
        B, S = target_seq.shape

        pred_tokens = torch.argmax(logits, dim=-1)  # [B, S]

        raw: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}

        for b in range(B):
            tgt_b = target_seq[b]      # [S]
            pred_b = pred_tokens[b]    # [S]
            if token_weights is not None:
                w_b = token_weights[b]
                valid_mask_b = (tgt_b != self.pad_token_id) & (w_b > 0)
            else:
                valid_mask_b = (tgt_b != self.pad_token_id)

            # 单样本所有指标
            metrics_b: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
            metrics_b.update(
                self._compute_token_accuracy_single(pred_b, tgt_b, valid_mask_b)
            )
            metrics_b.update(
                self._compute_bbox_and_class_accuracy_single(pred_b, tgt_b, valid_mask_b)
            )
            metrics_b.update(
                self._compute_tube_instance_accuracy_single(pred_b, tgt_b, valid_mask_b)
            )
            metrics_b.update(
                self._compute_sequence_metrics_single(pred_b, tgt_b)
            )

            # 累加到 batch 级
            for name, (c_b, t_b) in metrics_b.items():
                if name not in raw:
                    raw[name] = (c_b, t_b)
                else:
                    c, t = raw[name]
                    raw[name] = (c + c_b, t + t_b)

        if gather_fn is not None:
            return self.gather_and_normalize_metrics(raw, gather_fn)
        else:
            # 单机直接归一化
            out: Dict[str, float] = {}
            for name, (c, t) in raw.items():
                out[name] = (c / t.clamp(min=1)).item()
            return out

    def get_epoch_metrics(self) -> Dict[str, float]:
        """返回当前 epoch 累计的平均精度。"""
        return {
            name: sum(vals) / len(vals)
            for name, vals in self.accumulated_metrics.items()
        }
