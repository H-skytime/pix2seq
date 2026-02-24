# evaluation/video_tube_map.py

from typing import Dict, List, Any, Tuple, Optional
import torch


def _compute_box_iou_2d(box1: torch.Tensor, box2: torch.Tensor):
    """
    单帧 2D bbox IoU，box1/box2: [4] = [y1, x1, y2, x2] (或 [x1,y1,x2,y2]，只要一致即可)
    这里按 [y1, x1, y2, x2] 解释。
    """
    y1_1, x1_1, y2_1, x2_1 = box1
    y1_2, x1_2, y2_2, x2_2 = box2

    inter_y1 = torch.max(y1_1, y1_2)
    inter_x1 = torch.max(x1_1, x1_2)
    inter_y2 = torch.min(y2_1, y2_2)
    inter_x2 = torch.min(x2_1, x2_2)

    inter_h = torch.clamp(inter_y2 - inter_y1, min=0)
    inter_w = torch.clamp(inter_x2 - inter_x1, min=0)
    inter_area = inter_h * inter_w

    area1 = torch.clamp(y2_1 - y1_1, min=0) * torch.clamp(x2_1 - x1_1, min=0)
    area2 = torch.clamp(y2_2 - y1_2, min=0) * torch.clamp(x2_2 - x1_2, min=0)

    union = area1 + area2 - inter_area
    iou = inter_area / torch.clamp(union, min=1e-6)
    return iou


def _compute_tube_iou(
    gt_boxes: torch.Tensor,  # [T,4]
    pred_boxes: torch.Tensor,  # [T,4]
):
    """
    tube IoU 定义：对每一帧做 2D IoU，然后在 T 帧上做平均（或者 sum / T）。

    假设 gt 和 pred 的 T 相同；如果不同，你可以改成按 min(T_gt, T_pred) 对齐。
    """
    assert gt_boxes.dim() == 2 and gt_boxes.size(1) == 4
    assert pred_boxes.dim() == 2 and pred_boxes.size(1) == 4

    T = min(gt_boxes.size(0), pred_boxes.size(0))
    if T == 0:
        return torch.zeros((), dtype=torch.float32)

    ious = []
    for t in range(T):
        ious.append(_compute_box_iou_2d(gt_boxes[t], pred_boxes[t]))
    ious = torch.stack(ious, dim=0)  # [T]

    return ious.mean()  # 平均 IoU 作为 tube IoU


class VideoTubeMeanAveragePrecision:
    """
    简化版 Tube mAP 评估器：

    - 按 category_id 分组；
    - 对每个 IoU 阈值 (0.5:0.95) 计算 AP；
    - 最终对所有类别 / 所有 IoU 阈值取均值，得到一个 scalar mAP。

    只考虑 "annotation" / "prediction" 粒度为「一条 tube」。
    """

    def __init__(
        self,
        iou_thresholds: Optional[List[float]] = None,
    ):
        if iou_thresholds is None:
            # 仿 COCO：0.5:0.95 step=0.05
            self.iou_thresholds = [0.5 + 0.05 * i for i in range(10)]
        else:
            self.iou_thresholds = list(iou_thresholds)

    # ------------------------------------------------
    # 主入口
    # ------------------------------------------------
    def compute(
        self,
        targets_json: Dict[str, Any],
        predictions_json: List[Dict[str, Any]],
    ) -> float:
        """
        Args:
            targets_json:
                含有 "annotations" 列表，每个元素形如：
                    {
                        "id": int,
                        "video_id": int,
                        "category_id": int,
                        "boxes": List[List[float]],  # [T,4]
                    }

            predictions_json:
                列表，每个元素形如：
                    {
                        "video_id": int,
                        "category_id": int,
                        "score": float,
                        "boxes": List[List[float]],  # [T,4]
                    }

        Returns:
            mAP: float
        """
        # 1) 按类别组织 GT / Pred
        gt_by_cat: Dict[int, List[Dict[str, Any]]] = {}
        pred_by_cat: Dict[int, List[Dict[str, Any]]] = {}

        for ann in targets_json.get("annotations", []):
            cid = int(ann["category_id"])
            gt_by_cat.setdefault(cid, []).append(ann)

        for pred in predictions_json:
            cid = int(pred["category_id"])
            pred_by_cat.setdefault(cid, []).append(pred)

        # 2) 对每个 category、每个 IoU 阈值计算 AP，然后取平均
        aps: List[float] = []
        for cid, gt_list in gt_by_cat.items():
            preds_c = pred_by_cat.get(cid, [])
            if len(gt_list) == 0:
                continue

            # 按 IoU 阈值遍历
            for thr in self.iou_thresholds:
                ap_c_t = self._compute_ap_for_category_iou(
                    gt_list, preds_c, iou_thr=thr
                )
                aps.append(ap_c_t)

        if len(aps) == 0:
            return 0.0

        return float(sum(aps) / len(aps))

    # ------------------------------------------------
    # 单个类别、单个 IoU 阈值的 AP（VOC-style 简化版）
    # ------------------------------------------------
    def _compute_ap_for_category_iou(
        self,
        gt_list: List[Dict[str, Any]],
        pred_list: List[Dict[str, Any]],
        iou_thr: float,
    ) -> float:
        """
        对某个类别、某个 IoU 阈值，按照「tube 级别」计算 AP。

        统一在 (video_id) 维度上做匹配：
          - 不同视频之间的 tube 不互相匹配；
          - 同一视频内，pred tube 和 gt tube 做贪心匹配（最大 IoU 且 > iou_thr）。
        """
        if len(gt_list) == 0:
            return 0.0

        # 1) 把 GT 按 video_id 分组
        gts_by_video: Dict[int, List[Dict[str, Any]]] = {}
        for g in gt_list:
            vid = str(g["video_id"])
            gts_by_video.setdefault(vid, []).append(g)

        # 2) Pred 按分数降序排序
        preds = sorted(pred_list, key=lambda x: float(x["score"]), reverse=True)

        if len(preds) == 0:
            return 0.0

        tp = torch.zeros(len(preds), dtype=torch.float32)
        fp = torch.zeros(len(preds), dtype=torch.float32)

        # 给每条 GT tube 一个 matched 标记
        matched = {}
        for g in gt_list:
            matched[g["id"]] = False

        # 3) 遍历所有预测，做贪心匹配
        for i, p in enumerate(preds):
            vid = str(p["video_id"])
            cand_gts = gts_by_video.get(vid, [])
            if len(cand_gts) == 0:
                fp[i] = 1.0
                continue

            # 预测 tube boxes
            pred_boxes = torch.tensor(p["boxes"], dtype=torch.float32)  # [T,4]

            best_iou = 0.0
            best_gt_idx = -1
            best_gt_id = None

            for g in cand_gts:
                g_id = g["id"]
                if matched[g_id]:
                    continue
                gt_boxes = torch.tensor(g["boxes"], dtype=torch.float32)  # [T,4]
                iou = _compute_tube_iou(gt_boxes, pred_boxes)
                if iou.item() > best_iou:
                    best_iou = iou.item()
                    best_gt_idx = g_id
                    best_gt_id = g_id

            if best_gt_id is not None and best_iou >= iou_thr:
                tp[i] = 1.0
                matched[best_gt_id] = True
            else:
                fp[i] = 1.0

        # 4) 按经典 PR 曲线算 AP（VOC 11-point 简化版）
        cum_tp = torch.cumsum(tp, dim=0)
        cum_fp = torch.cumsum(fp, dim=0)

        num_gt = len(gt_list)
        if num_gt == 0:
            return 0.0

        recall = cum_tp / (num_gt + 1e-6)
        precision = cum_tp / torch.clamp(cum_tp + cum_fp, min=1e-6)

        # VOC-2007 11-point AP
        ap = 0.0
        for r in torch.linspace(0, 1, steps=11):
            mask = recall >= r
            if mask.any():
                p_max = precision[mask].max().item()
            else:
                p_max = 0.0
            ap += p_max / 11.0

        return float(ap)


class ActionTrajectoryMetricsEvaluator:
    """
    ACT 任务：基于 (GT keypoints) -> action_id 的 clip 级动作识别评估器。

    序列格式约定：
        [ACT,
         kp_tokens(prompt; token_weights=0),
         act_token (supervised; token_weights>0),
         EOS       (supervised; token_weights>0),
         PAD...]

    输出指标（clip 级）：
        - act_traj_top1    : 动作类别 top-1 精度
        - act_traj_topK    : 动作类别 top-K 精度
        - act_traj_eos_pos : EOS 位置是否预测正确（首个 EOS 位置一致）
    """

    def __init__(
        self,
        pad_token_id: int = 0,
        eos_token_id: int = 2,
        topk: int = 3,
    ):
        self.pad_token_id = int(pad_token_id)
        self.eos_token_id = int(eos_token_id)
        self.topk = int(topk)
        self.reset_metrics()

    # ---------------- epoch state (global counts) ----------------

    def reset_metrics(self):
        self._epoch_correct: Dict[str, int] = {}
        self._epoch_total: Dict[str, int] = {}

    def _accum(self, name: str, correct: torch.Tensor, total: torch.Tensor):
        c = int(correct.item())
        t = int(total.item())
        self._epoch_correct[name] = self._epoch_correct.get(name, 0) + c
        self._epoch_total[name] = self._epoch_total.get(name, 0) + t

    def get_epoch_metrics(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for name, t in self._epoch_total.items():
            denom = max(int(t), 1)
            out[name] = float(self._epoch_correct.get(name, 0)) / float(denom)
        return out

    # ---------------- distributed gather ----------------

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
            return gathered[:, 0].sum(), gathered[:, 1].sum()

        return gathered[0::2].sum(), gathered[1::2].sum()

    # ---------------- helpers ----------------

    def _zero(self, device):
        z = torch.zeros((), device=device, dtype=torch.long)
        return z, z

    def _first_eos_idx_1d(self, seq_1d: torch.Tensor) -> int:
        pos = (seq_1d == self.eos_token_id).nonzero(as_tuple=True)[0]
        if len(pos) == 0:
            return -1
        return int(pos[0].item())

    # ---------------- per-sample computation ----------------

    def _compute_one_sample(
        self,
        logits: torch.Tensor,                 # [S,V]
        target_seq: torch.Tensor,             # [S]
        token_weights: Optional[torch.Tensor] = None,  # [S] or None
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        device = target_seq.device
        S, V = logits.shape

        tgt_eos_idx = self._first_eos_idx_1d(target_seq)
        if tgt_eos_idx <= 0:
            c, t = self._zero(device)
            return {
                "act_traj_top1": (c, t),
                "act_traj_topK": (c, t),
                "act_traj_eos_pos": (c, t),
            }

        act_idx = tgt_eos_idx - 1
        if act_idx < 0 or act_idx >= S:
            c, t = self._zero(device)
            return {
                "act_traj_top1": (c, t),
                "act_traj_topK": (c, t),
                "act_traj_eos_pos": (c, t),
            }

        tgt_act_token = target_seq[act_idx]
        if tgt_act_token == self.pad_token_id:
            c, t = self._zero(device)
            return {
                "act_traj_top1": (c, t),
                "act_traj_topK": (c, t),
                "act_traj_eos_pos": (c, t),
            }

        # 关键：ACT 的监督策略要求只在 act_idx / eos_idx 上监督
        if token_weights is not None and token_weights.numel() == S:
            if token_weights[act_idx] <= 0:
                c, t = self._zero(device)
                return {
                    "act_traj_top1": (c, t),
                    "act_traj_topK": (c, t),
                    "act_traj_eos_pos": (c, t),
                }

        # ---- top1 / topK at act_idx ----
        probs = torch.softmax(logits[act_idx], dim=-1)  # [V]

        pred_top1 = int(torch.argmax(probs).item())
        correct_top1 = int(pred_top1 == int(tgt_act_token.item()))
        c1 = torch.tensor([correct_top1], device=device, dtype=torch.long)
        t1 = torch.tensor([1], device=device, dtype=torch.long)

        k = min(self.topk, V)
        _, topk_idx = torch.topk(probs, k=k, dim=-1)
        correct_topk = int(int(tgt_act_token.item()) in topk_idx.tolist())
        ck = torch.tensor([correct_topk], device=device, dtype=torch.long)
        tk = torch.tensor([1], device=device, dtype=torch.long)

        # ---- EOS position correctness (first EOS) ----
        pred_tokens = torch.argmax(logits, dim=-1)  # [S]
        pred_eos_pos = (pred_tokens == self.eos_token_id).nonzero(as_tuple=True)[0]
        eos_correct = int(len(pred_eos_pos) > 0 and int(pred_eos_pos[0].item()) == tgt_eos_idx)
        ce = torch.tensor([eos_correct], device=device, dtype=torch.long)
        te = torch.tensor([1], device=device, dtype=torch.long)

        return {
            "act_traj_top1": (c1, t1),
            "act_traj_topK": (ck, tk),
            "act_traj_eos_pos": (ce, te),
        }

    # ---------------- batch computation ----------------

    def compute_batch_metrics(
        self,
        logits: torch.Tensor,                 # [B,S,V]
        target_seq: torch.Tensor,             # [B,S]
        token_weights: Optional[torch.Tensor] = None,  # [B,S] or None
        gather_fn=None,
    ) -> Dict[str, float]:
        B, S, V = logits.shape
        raw: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}

        for b in range(B):
            w_b = token_weights[b] if token_weights is not None else None
            mb = self._compute_one_sample(logits[b], target_seq[b], w_b)
            for name, (c_b, t_b) in mb.items():
                if name not in raw:
                    raw[name] = (c_b, t_b)
                else:
                    c, t = raw[name]
                    raw[name] = (c + c_b, t + t_b)

        # gather + normalize + accumulate epoch
        out: Dict[str, float] = {}
        for name, (c, t) in raw.items():
            c_g, t_g = self._gather_counts(c, t, gather_fn)
            self._accum(name, c_g, t_g)
            out[name] = float((c_g / t_g.clamp(min=1)).item())
        return out

class KeypointTrajectoryMetricsEvaluator:
    """
    KP 任务：bbox -> keypoints 的 tube 级（单人展开样本）评估器（对齐 KeypointTokenAccuracyEvaluator 口径）。

    期望序列格式（单人 tube）：
        [KP_TASK_TOKEN]
        [frame_0 bbox] ... [frame_{T-1} bbox]              # prompt (token_weights=0)
        [frame_0 keypoints] ... [frame_{T-1} keypoints]    # supervised (token_weights>0)
        [EOS]                                              # supervised (token_weights>0)
        [PAD...]

    输出指标（tube级）：
      - kp_traj_sup_token : 监督段整体 token 准确率（keypoints + EOS）
      - kp_traj_kp_token  : 关键点 token 准确率（监督段，且排除 EOS；并可忽略 INVISIBLE/PROMPT_PAD）
      - kp_traj_eos_pos   : EOS 位置是否预测正确（从监督段起点之后找首个 EOS 位置是否一致）
    """

    def __init__(
        self,
        pad_token_id: int = 0,
        eos_token_id: int = 2,
        prompt_pad_token_id: Optional[int] = None,
        invisible_kp_token_id: Optional[int] = 3,
    ):
        self.pad_token_id = int(pad_token_id)
        self.eos_token_id = int(eos_token_id)
        self.prompt_pad_token_id = int(prompt_pad_token_id) if prompt_pad_token_id is not None else None
        self.invisible_kp_token_id = int(invisible_kp_token_id) if invisible_kp_token_id is not None else None
        self.reset_metrics()

    # ---------------- epoch state (global counts) ----------------

    def reset_metrics(self):
        self._epoch_correct: Dict[str, int] = {}
        self._epoch_total: Dict[str, int] = {}

    def _accum(self, name: str, correct: torch.Tensor, total: torch.Tensor):
        c = int(correct.item())
        t = int(total.item())
        self._epoch_correct[name] = self._epoch_correct.get(name, 0) + c
        self._epoch_total[name] = self._epoch_total.get(name, 0) + t

    def get_epoch_metrics(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for name, t in self._epoch_total.items():
            denom = max(int(t), 1)
            out[name] = float(self._epoch_correct.get(name, 0)) / float(denom)
        return out

    # ---------------- distributed gather ----------------

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

        # common shapes: [world,2] or [2*world]
        if gathered.dim() == 2 and gathered.size(-1) == 2:
            return gathered[:, 0].sum(), gathered[:, 1].sum()

        return gathered[0::2].sum(), gathered[1::2].sum()

    # ---------------- eos position helpers ----------------

    @staticmethod
    def _first_true_idx(mask_2d: torch.Tensor) -> torch.Tensor:
        """
        mask_2d: [B,S] bool
        return: [B] first True idx, if none -> -1
        """
        has = mask_2d.any(dim=1)
        idx = mask_2d.float().argmax(dim=1)
        idx = torch.where(has, idx, torch.full_like(idx, -1))
        return idx

    def _first_eos_after_start(
        self,
        seq: torch.Tensor,          # [B,S]
        start_idx: torch.Tensor,    # [B]
    ) -> torch.Tensor:
        B, S = seq.shape
        pos = torch.arange(S, device=seq.device).view(1, S).expand(B, S)
        after_start = pos >= start_idx.view(B, 1)
        eos_mask = (seq == self.eos_token_id) & after_start
        return self._first_true_idx(eos_mask)

    # ---------------- core computation ----------------

    def compute_batch_metrics(
        self,
        logits: torch.Tensor,                 # [Bk, S, V]
        target_seq: torch.Tensor,             # [Bk, S]
        token_weights: Optional[torch.Tensor] = None,  # [Bk, S] or None
        gather_fn=None,
    ) -> Dict[str, float]:
        """
        对齐 KeypointTokenAccuracyEvaluator：
          - valid = (target!=PAD) & (token_weights>0) & (target!=PROMPT_PAD) & (target!=INVISIBLE)
          - kp_traj_kp_token: valid & (target!=EOS)
          - eos_pos: 从监督段起点之后找首个 EOS 的位置是否一致
        """
        pred_tokens = torch.argmax(logits, dim=-1)  # [Bk,S]
        device = target_seq.device
        B, S = target_seq.shape

        # ---------------- supervised mask（与训练/KeypointTokenAccuracyEvaluator 对齐） ----------------
        valid = (target_seq != self.pad_token_id)

        if token_weights is not None:
            valid = valid & (token_weights > 0)

        if self.prompt_pad_token_id is not None:
            valid = valid & (target_seq != self.prompt_pad_token_id)

        if self.invisible_kp_token_id is not None:
            valid = valid & (target_seq != self.invisible_kp_token_id)

        # ---- 1) supervised token acc (keypoints + EOS) ----
        sup_correct = ((pred_tokens == target_seq) & valid).sum()
        sup_total = valid.sum()
        sup_correct, sup_total = self._gather_counts(sup_correct, sup_total, gather_fn)
        self._accum("kp_traj_sup_token", sup_correct, sup_total)

        # ---- 2) keypoint token acc (supervised but not EOS) ----
        kp_mask = valid & (target_seq != self.eos_token_id)
        kp_correct = ((pred_tokens == target_seq) & kp_mask).sum()
        kp_total = kp_mask.sum()
        kp_correct, kp_total = self._gather_counts(kp_correct, kp_total, gather_fn)
        self._accum("kp_traj_kp_token", kp_correct, kp_total)

        # ---- 3) EOS position correctness (structure; search after supervised start) ----
        if token_weights is not None:
            sup_pos_mask = (token_weights > 0) & (target_seq != self.pad_token_id)

            if self.prompt_pad_token_id is not None:
                sup_pos_mask = sup_pos_mask & (target_seq != self.prompt_pad_token_id)
            if self.invisible_kp_token_id is not None:
                sup_pos_mask = sup_pos_mask & (target_seq != self.invisible_kp_token_id)

            has_sup = sup_pos_mask.any(dim=1)
            start_idx = sup_pos_mask.float().argmax(dim=1)
            start_idx = torch.where(has_sup, start_idx, torch.zeros_like(start_idx))
        else:
            start_idx = torch.zeros((B,), device=device, dtype=torch.long)

        tgt_pos = self._first_eos_after_start(target_seq, start_idx)
        pred_pos = self._first_eos_after_start(pred_tokens, start_idx)

        ok = ((tgt_pos == pred_pos) & (tgt_pos >= 0)).sum()
        tot = torch.tensor(int(B), device=device, dtype=torch.long)
        ok, tot = self._gather_counts(ok, tot, gather_fn)
        self._accum("kp_traj_eos_pos", ok, tot)

        def safe_div(c: torch.Tensor, t: torch.Tensor) -> float:
            return float((c / t.clamp(min=1)).item())

        return {
            "kp_traj_sup_token": safe_div(sup_correct, sup_total),
            "kp_traj_kp_token": safe_div(kp_correct, kp_total),
            "kp_traj_eos_pos": safe_div(ok, tot),
        }
