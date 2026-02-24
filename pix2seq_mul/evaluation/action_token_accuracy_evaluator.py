from typing import Dict, Optional, Tuple

import torch


class ActionTokenAccuracyEvaluator:
    """
    ACT 任务 token 级评估器（只评 “(GT keypoints)->action_id” 的预测段）：

    期望序列（双人统一，单人用 PROMPT_PAD 占位并在 attention 中屏蔽）：
        [ACT_TASK_TOKEN]
        [frame_0 p0 keypoints]
        [frame_0 p1 keypoints or PROMPT_PAD...]
        ...
        [frame_{T-1} p0 keypoints]
        [frame_{T-1} p1 keypoints or PROMPT_PAD...]
        [ACTION_CLASS_TOKEN]   # supervised (token_weights>0)
        [EOS]                  # supervised (token_weights>0)
        [PAD...]

    评估口径：
      - 只统计 supervised positions：token_weights>0 且 target != PAD
      - class_token：supervised 中 target != EOS 的位置（应只有 action token）
      - eos_token：supervised 中 target == EOS 的位置
      - sequence_eos_pos：预测序列与目标序列的“首个 EOS 位置”是否一致（结构性指标）

    注意：不再评估 prompt 的关键点 token（它们应由 token_weights=0 排除）。
    """

    def __init__(
        self,
        pad_token_id: int = 0,
        eos_token_id: int = 2,
        prompt_pad_token_id: Optional[int] = None,
    ):
        self.pad_token_id = int(pad_token_id)
        self.eos_token_id = int(eos_token_id)
        self.prompt_pad_token_id = int(prompt_pad_token_id) if prompt_pad_token_id is not None else None
        self.reset_metrics()

    # ------------------------ epoch state ------------------------

    def reset_metrics(self):
        self._epoch_correct: Dict[str, int] = {}
        self._epoch_total: Dict[str, int] = {}

    def _accumulate_epoch(self, name: str, correct: torch.Tensor, total: torch.Tensor):
        c = int(correct.item())
        t = int(total.item())
        self._epoch_correct[name] = self._epoch_correct.get(name, 0) + c
        self._epoch_total[name] = self._epoch_total.get(name, 0) + t

    def get_epoch_metrics(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for name in self._epoch_total.keys():
            t = max(int(self._epoch_total.get(name, 0)), 1)
            c = int(self._epoch_correct.get(name, 0))
            out[name] = float(c) / float(t)
        return out

    # ------------------------ distributed gather helpers ------------------------

    @staticmethod
    def _gather_counts(
        correct: torch.Tensor,
        total: torch.Tensor,
        gather_fn=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if gather_fn is None:
            return correct, total

        local = torch.stack([correct, total])  # [2]
        gathered = gather_fn(local)

        if gathered is None:
            return correct, total

        if gathered.dim() == 2 and gathered.size(-1) == 2:
            c = gathered[:, 0].sum()
            t = gathered[:, 1].sum()
            return c, t

        c = gathered[0::2].sum()
        t = gathered[1::2].sum()
        return c, t

    # ------------------------ metrics core ------------------------

    def compute_batch_metrics(
        self,
        logits: torch.Tensor,                 # [B, S, V]
        target_seq: torch.Tensor,             # [B, S]
        token_weights: Optional[torch.Tensor] = None,  # [B, S] supervised positions > 0
        gather_fn=None,
    ) -> Dict[str, float]:
        device = target_seq.device
        pred_tokens = torch.argmax(logits, dim=-1)  # [B, S]

        valid = (target_seq != self.pad_token_id)
        if token_weights is not None:
            valid = valid & (token_weights > 0)

        # 防御：prompt_pad 不应被纳入评估（理论上 token_weights=0 已排除）
        if self.prompt_pad_token_id is not None:
            valid = valid & (target_seq != int(self.prompt_pad_token_id))

        # 1) supervised_token accuracy (action + eos)
        sup_correct = ((pred_tokens == target_seq) & valid).sum()
        sup_total = valid.sum()
        sup_correct, sup_total = self._gather_counts(sup_correct, sup_total, gather_fn)
        self._accumulate_epoch("supervised_token", sup_correct, sup_total)

        # 2) class_token accuracy (supervised non-EOS position -> action token)
        cls_mask = valid & (target_seq != self.eos_token_id)
        cls_correct = ((pred_tokens == target_seq) & cls_mask).sum()
        cls_total = cls_mask.sum()
        cls_correct, cls_total = self._gather_counts(cls_correct, cls_total, gather_fn)
        self._accumulate_epoch("class_token", cls_correct, cls_total)

        # 3) eos_token accuracy
        eos_mask = valid & (target_seq == self.eos_token_id)
        eos_correct = ((pred_tokens == target_seq) & eos_mask).sum()
        eos_total = eos_mask.sum()
        eos_correct, eos_total = self._gather_counts(eos_correct, eos_total, gather_fn)
        self._accumulate_epoch("eos_token", eos_correct, eos_total)

        # 4) sequence_eos_pos
        B = target_seq.size(0)

        def first_eos_idx(x: torch.Tensor) -> torch.Tensor:
            eos_mask_local = (x == self.eos_token_id)
            has = eos_mask_local.any(dim=1)
            idx = eos_mask_local.float().argmax(dim=1)
            idx = torch.where(has, idx, torch.full_like(idx, -1))
            return idx

        tgt_pos = first_eos_idx(target_seq)
        pred_pos = first_eos_idx(pred_tokens)
        seq_ok = ((tgt_pos == pred_pos) & (tgt_pos >= 0) & (pred_pos >= 0)).sum()
        seq_total = torch.tensor(B, device=device, dtype=torch.long)
        seq_ok, seq_total = self._gather_counts(seq_ok, seq_total, gather_fn)
        self._accumulate_epoch("sequence_eos_pos", seq_ok, seq_total)

        def safe_div(c: torch.Tensor, t: torch.Tensor) -> float:
            return float((c / t.clamp(min=1)).item())

        return {
            "supervised_token": safe_div(sup_correct, sup_total),
            "class_token": safe_div(cls_correct, cls_total),
            "eos_token": safe_div(eos_correct, eos_total),
            "sequence_eos_pos": safe_div(seq_ok, seq_total),
        }
