import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

import pandas as pd
import torch
from pytorch_accelerated.callbacks import TrainerCallback

from evaluation.video_evaluator import VideoTubeMeanAveragePrecision, ActionTrajectoryMetricsEvaluator, KeypointTrajectoryMetricsEvaluator


COCO80_TO_COCO91_MAP = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28,
                        31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
                        56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84,
                        85, 86, 87, 88, 89, 90]


# 类型转换：将连续的1-80类别对应回稀疏的类别序号
def coco80_to_coco91_lookup():
    return {i: v for i, v in enumerate(COCO80_TO_COCO91_MAP)}


class ConvertPredictionClassesCallback(TrainerCallback):
    """
    评估前对模型输出的预测结果做“类别 id / 动作 id”转换。

    支持两个来源：
      1) 检测预测（用于 mAP）：
         - batch_output["predictions"] : [N, 7]
           约定为 [xmin, ymin, xmax, ymax, score, class_id, image_id]
           → 将 class_id 从「连续 0..num_classes-1」映射到数据集真实 id
             （默认 COCO 80 → COCO 91）

      2) 动作预测（视频动作任务）：
         - batch_output["action_predictions"] : [M, K]
           约定最后一列为 action_id（其它列可以是 video_id / score / 起止帧等）
           → 将 action_id 从「连续 0..num_actions-1」映射到数据集真实动作 id

    映射字典说明：
      - det_class_mapping: Dict[int, int]
            例：0 -> 1, 1 -> 2, ...
            默认为 coco80_to_coco91_lookup()
      - action_class_mapping: Dict[int, int]
            例：0 -> 43 (fall), 1 -> 12 (slap), ...
            若为 None，则不对 action_id 做任何修改。
    """

    def __init__(
        self, det_class_mapping=None, 
        action_class_mapping=None,
        map_detection=True, 
        map_actions=True,
    ):
        if det_class_mapping is None:
            det_class_mapping = coco80_to_coco91_lookup()
        self.det_class_mapping = det_class_mapping
        self.action_class_mapping = action_class_mapping
        self.map_detection = map_detection
        self.map_actions = map_actions

    # ----------------- Trainer 回调接口 ----------------- #
    def on_eval_step_end(self, trainer, batch, batch_output, **kwargs):
        # -------- map DET: tube_predictions --------
        if self.map_detection and "tube_predictions" in batch_output:
            tube_preds = batch_output["tube_predictions"]
            if isinstance(tube_preds, dict):
                tube_preds = [tube_preds]
            if isinstance(tube_preds, list):
                for p in tube_preds:
                    if not isinstance(p, dict):
                        continue
                    cid = p.get("category_id", None)
                    if cid is None:
                        continue
                    cid_i = int(cid)
                    if cid_i >= 0:
                        p["category_id"] = int(self.det_class_mapping.get(cid_i, cid_i))
            batch_output["tube_predictions"] = tube_preds

        # -------- map ACT: pred_action_ids --------
        if self.map_actions and (self.action_class_mapping is not None) and ("pred_action_ids" in batch_output):
            ids = batch_output["pred_action_ids"]
            if isinstance(ids, torch.Tensor):
                ids_m = ids.clone()
                for i in range(ids_m.numel()):
                    v = int(ids_m[i].item())
                    if v >= 0:
                        ids_m[i] = ids_m.new_tensor(self.action_class_mapping.get(v, v))
                batch_output["pred_action_ids"] = ids_m
            elif isinstance(ids, (list, tuple)):
                out = []
                for v in ids:
                    if v is None:
                        out.append(v)
                        continue
                    vi = int(v)
                    out.append(self.action_class_mapping.get(vi, vi) if vi >= 0 else vi)
                batch_output["pred_action_ids"] = out

class CalculateVideoTubeMeanAveragePrecisionCallback(TrainerCallback):
    """
    Tube-based mAP evaluator:
    - 自动从 val_dataset 生成 tube GT
    - 从 batch_output 读取 tube_predictions
    - 计算 tube-mAP
    - ✅ 保证每个 epoch 一定记录 metric_name（默认 0.0），避免 SaveBestModelCallback 报错
    - ✅ 打印指标日志
    """

    def __init__(
        self,
        val_dataset,
        iou_thresholds=None,
        save_predictions_output_dir_path=None,
        verbose=False,
        metric_name="tube_map",
        default_metric_value: float = 0.0,  # ✅ 新增：无预测/异常时的兜底值
    ):
        self.val_dataset = val_dataset
        self.verbose = bool(verbose)
        self.metric_name = str(metric_name)
        self.default_metric_value = float(default_metric_value)

        self.save_predictions_path = (
            Path(save_predictions_output_dir_path)
            if save_predictions_output_dir_path is not None
            else None
        )

        # ---- 构造 GT：封装成 {"annotations": [...] } ----
        annotations = self.build_tube_gt_from_dataset(val_dataset)
        self.targets_json = {"annotations": annotations}

        self.evaluator = VideoTubeMeanAveragePrecision(iou_thresholds)
        self.eval_predictions: List[Dict[str, Any]] = []

    # ============================================================
    #   构造 tube 级 Ground Truth
    # ============================================================
    def build_tube_gt_from_dataset(self, dataset):
        """
        从 val_dataset 构造 tube 级 GT，用于 tube-mAP 评估。

        返回 list[dict]，每个元素：
        {
            "id": int,
            "video_id": int/str,
            "category_id": int,
            "boxes": [[x1,y1,x2,y2], ...]
        }
        """
        tube_targets = []
        ann_id = 1

        for idx in range(len(dataset)):
            sample = dataset[idx]

            # video_id：尽量与预测侧保持一致
            if "video_id" in sample:
                video_id = sample["video_id"]
            else:
                video_id = idx

            # ===== 情况一：已有 tube 级 GT =====
            if "tube_boxes" in sample:
                tube_boxes = sample["tube_boxes"]
                tube_labels = sample.get("tube_class_ids", sample.get("class_ids", None))
                tube_valid_mask = sample.get("tube_valid_mask", None)

                if tube_labels is None:
                    raise ValueError("tube_boxes 存在，但没有找到 tube_class_ids / class_ids")

                if isinstance(tube_boxes, list):
                    tube_boxes = torch.as_tensor(tube_boxes, dtype=torch.float32)
                if isinstance(tube_labels, list):
                    tube_labels = torch.as_tensor(tube_labels, dtype=torch.long)

                if tube_boxes.dim() != 3:
                    raise ValueError(
                        f"tube_boxes 期望 3 维 [N,T,4] 或 [T,N,4]，当前 shape={tube_boxes.shape}"
                    )

                # 统一为 [N,T,4]
                if tube_boxes.size(0) == tube_labels.shape[0]:
                    tubes_NT4 = tube_boxes
                elif tube_boxes.size(1) == tube_labels.shape[0]:
                    tubes_NT4 = tube_boxes.permute(1, 0, 2)
                else:
                    raise ValueError(
                        f"tube_boxes 的前两维和 tube_labels 对不上，"
                        f"tube_boxes.shape={tube_boxes.shape}, tube_labels.shape={tube_labels.shape}"
                    )

                # tube_valid_mask：[N,T]
                if tube_valid_mask is not None:
                    if isinstance(tube_valid_mask, list):
                        tube_valid_mask = torch.as_tensor(tube_valid_mask, dtype=torch.bool)
                    if tube_valid_mask.shape != tubes_NT4.shape[:2]:
                        raise ValueError(
                            f"tube_valid_mask 形状不匹配，tube_valid_mask.shape={tube_valid_mask.shape}, "
                            f"tube_boxes.shape[:2]={tubes_NT4.shape[:2]}"
                        )

                for n in range(tubes_NT4.size(0)):
                    boxes_T4 = tubes_NT4[n]
                    if tube_valid_mask is not None:
                        mask = tube_valid_mask[n]
                        if mask.dtype != torch.bool:
                            mask = mask > 0
                        boxes_T4 = boxes_T4[mask]

                    if boxes_T4.numel() == 0:
                        continue

                    tube_targets.append(
                        {
                            "id": ann_id,
                            "video_id": video_id,
                            "category_id": int(tube_labels[n].item()),
                            "boxes": boxes_T4.tolist(),
                        }
                    )
                    ann_id += 1

            # ===== 情况二：只有帧级 GT（退路） =====
            else:
                if "boxes" not in sample or "class_ids" not in sample:
                    raise ValueError("既没有 tube_boxes，也没有帧级 boxes/class_ids，无法构造 tube GT。")

                boxes_per_frame = sample["boxes"]       # List[T] each [N_t,4]
                labels_per_frame = sample["class_ids"]  # List[T] each [N_t]

                T = len(boxes_per_frame)
                for t in range(T):
                    boxes_t = (
                        torch.as_tensor(boxes_per_frame[t], dtype=torch.float32)
                        if isinstance(boxes_per_frame[t], list)
                        else boxes_per_frame[t]
                    )
                    if boxes_t.numel() == 0:
                        continue

                    labels_t = (
                        torch.as_tensor(labels_per_frame[t], dtype=torch.long)
                        if isinstance(labels_per_frame[t], list)
                        else labels_per_frame[t]
                    )

                    Nt = boxes_t.shape[0]
                    for i in range(Nt):
                        tube_targets.append(
                            {
                                "id": ann_id,
                                "video_id": video_id,
                                "category_id": int(labels_t[i].item()),
                                "boxes": [boxes_t[i].tolist()],  # 单帧 tube
                            }
                        )
                        ann_id += 1

        return tube_targets

    # ============================================================
    #   lifecycle
    # ============================================================
    def on_training_run_start(self, trainer, **kwargs):
        """
        ✅ 关键兜底：训练一开始先写入一次默认 tube_map，
        这样即使首个 epoch 评估异常，SaveBestModelCallback 也不会因为“没有任何记录”而崩。
        """
        try:
            trainer.run_history.update_metric(self.metric_name, float(self.default_metric_value))
        except Exception as e:
            logging.warning(f"[DET][tube-mAP] failed to init metric '{self.metric_name}': {e}")

    def on_eval_epoch_start(self, trainer, **kwargs):
        self.eval_predictions = []

    def on_eval_step_end(self, trainer, batch, batch_output, **kwargs):
        preds = batch_output.get("tube_predictions", None)
        if preds is None:
            return
        if isinstance(preds, list) and len(preds) > 0:
            self.eval_predictions.extend(preds)

    # ============================================================
    #   helpers
    # ============================================================
    def _get_epoch(self, trainer) -> int:
        if hasattr(trainer, "run_history") and hasattr(trainer.run_history, "current_epoch"):
            try:
                return int(trainer.run_history.current_epoch)
            except Exception:
                return -1
        return -1

    def _evaluate(self) -> Dict[str, float]:
        """
        兼容不同 evaluator API，返回 dict[str,float]。
        """
        ev = self.evaluator

        if hasattr(ev, "evaluate") and callable(getattr(ev, "evaluate")):
            out = ev.evaluate(self.targets_json, self.eval_predictions)
        elif callable(ev):
            out = ev(self.targets_json, self.eval_predictions)
        elif hasattr(ev, "update") and hasattr(ev, "compute"):
            if hasattr(ev, "reset"):
                ev.reset()
            ev.update(self.eval_predictions, self.targets_json)
            out = ev.compute()
        else:
            raise RuntimeError("VideoTubeMeanAveragePrecision evaluator has no usable API.")

        if out is None:
            return {}
        if isinstance(out, (int, float)):
            return {self.metric_name: float(out)}
        if isinstance(out, dict):
            norm: Dict[str, float] = {}
            for k, v in out.items():
                if isinstance(v, (int, float)):
                    norm[str(k)] = float(v)
                elif torch.is_tensor(v) and v.numel() == 1:
                    norm[str(k)] = float(v.item())
            return norm
        return {}

    def _select_main_value(self, metrics: Dict[str, float]) -> float:
        """
        选取主指标：优先 metric_name，其次常见 key，否则取第一个。
        """
        if self.metric_name in metrics:
            return float(metrics[self.metric_name])

        for cand in ("map", "mAP", "tube_map", "map_50", "map50", "map_75", "map75"):
            if cand in metrics:
                return float(metrics[cand])

        return float(next(iter(metrics.values()))) if len(metrics) > 0 else float(self.default_metric_value)

    # ============================================================
    #   eval end: record + log + optional save
    # ============================================================
    def on_eval_epoch_end(self, trainer, **kwargs):
        epoch = self._get_epoch(trainer)

        # 默认先写一个兜底值：保证本 epoch 一定有记录
        main_val = float(self.default_metric_value)
        metrics: Dict[str, float] = {}

        try:
            # 计算指标
            metrics = self._evaluate()

            # 如果评估返回空（例如没有任何 prediction），保持默认值
            if isinstance(metrics, dict) and len(metrics) > 0:
                main_val = self._select_main_value(metrics)

                # 也把其它指标写入 run_history，便于看 map50/map75 等
                for k, v in metrics.items():
                    trainer.run_history.update_metric(str(k), float(v))

        except Exception as e:
            # 评估异常：保留默认值，并给出明确日志
            logging.warning(
                f"[DET][tube-mAP] epoch={epoch} evaluation failed, use default={self.default_metric_value:.6f}. error={e}"
            )

        # ✅ 最关键：无论如何都写入主指标 metric_name，避免 SaveBestModelCallback 报错
        trainer.run_history.update_metric(self.metric_name, float(main_val))

        # ✅ 日志打印（你要求的“指标准确率”打印；数值仍为 mAP）
        if len(metrics) > 0:
            detail_str = ", ".join([f"{k}={v:.6f}" for k, v in metrics.items()])
            logging.info(f"[DET][tube-mAP] epoch={epoch} {detail_str}")
        else:
            logging.info(f"[DET][tube-mAP] epoch={epoch} {self.metric_name}={main_val:.6f} (no valid metrics/preds)")

        # 可选：保存预测
        if self.save_predictions_path is not None:
            try:
                self.save_predictions_path.mkdir(parents=True, exist_ok=True)
                out_path = self.save_predictions_path / f"tube_predictions_epoch_{epoch}.json"
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(self.eval_predictions, f, ensure_ascii=False)
                if self.verbose:
                    logging.info(f"[DET][tube-mAP] saved predictions to {str(out_path)}")
            except Exception as e:
                logging.warning(f"[DET][tube-mAP] save predictions failed: {e}")

class CalculateKeypointTrajectoryMetricsCallback(TrainerCallback):
    """
    KP 任务：tube级 / token级评估回调（与训练监督口径对齐，使用 KeypointTrajectoryMetricsEvaluator）

    读取字段（优先 batch_output，其次 batch）：
      - kp_logits        : [Bk, S, V]
      - kp_target_seq    : [Bk, S]
      - kp_token_weights : [Bk, S]  (prompt=0; keypoints+EOS=1)

    写入 run_history：
      - train_kp_traj_sup_token_accuracy / eval_kp_traj_sup_token_accuracy
      - train_kp_traj_kp_token_accuracy  / eval_kp_traj_kp_token_accuracy
      - train_kp_traj_eos_pos_accuracy   / eval_kp_traj_eos_pos_accuracy
      - train_epoch_kp_traj_*_accuracy   / eval_epoch_kp_traj_*_accuracy
    """

    def __init__(
        self,
        pad_token_id: int = 0,
        eos_token_id: int = 2,
        prompt_pad_token: Optional[int] = 7,
        invisible_kp_token: Optional[int] = 6,
    ):
        # ✅ 改为使用你刚确认的 KP 指标评估器
        self.evaluator = KeypointTrajectoryMetricsEvaluator(
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            prompt_pad_token_id=prompt_pad_token,
            invisible_kp_token_id=invisible_kp_token,
        )

    def _update_run_history(self, trainer, metrics: Dict[str, float], prefix: str):
        for name, value in metrics.items():
            trainer.run_history.update_metric(f"{prefix}_{name}_accuracy", float(value))

    def on_train_epoch_start(self, trainer, **kwargs):
        self.evaluator.reset_metrics()

    def on_eval_epoch_start(self, trainer, **kwargs):
        self.evaluator.reset_metrics()

    @staticmethod
    def _get_targets_and_weights(batch: Dict, batch_output: Dict):
        target_seq = batch_output.get("kp_target_seq", batch.get("kp_target_seq", None))
        token_weights = batch_output.get("kp_token_weights", batch.get("kp_token_weights", None))
        return target_seq, token_weights

    def on_train_step_end(self, trainer, batch, batch_output, **kwargs):
        logits = batch_output.get("kp_logits", None)
        if logits is None:
            return

        target_seq, token_weights = self._get_targets_and_weights(batch, batch_output)
        if target_seq is None:
            return

        metrics = self.evaluator.compute_batch_metrics(
            logits=logits,
            target_seq=target_seq,
            token_weights=token_weights,
            gather_fn=getattr(trainer, "gather", None),
        )
        self._update_run_history(trainer, metrics, "train")

    def on_eval_step_end(self, trainer, batch, batch_output, **kwargs):
        logits = batch_output.get("kp_logits", None)
        if logits is None:
            return

        target_seq, token_weights = self._get_targets_and_weights(batch, batch_output)
        if target_seq is None:
            return

        metrics = self.evaluator.compute_batch_metrics(
            logits=logits,
            target_seq=target_seq,
            token_weights=token_weights,
            gather_fn=getattr(trainer, "gather", None),
        )
        self._update_run_history(trainer, metrics, "eval")

    def on_train_epoch_end(self, trainer, **kwargs):
        epoch_metrics = self.evaluator.get_epoch_metrics()
        for name, value in epoch_metrics.items():
            trainer.run_history.update_metric(f"train_epoch_{name}_accuracy", float(value))

    def on_eval_epoch_end(self, trainer, **kwargs):
        epoch_metrics = self.evaluator.get_epoch_metrics()
        for name, value in epoch_metrics.items():
            trainer.run_history.update_metric(f"eval_epoch_{name}_accuracy", float(value))

class CalculateActionTrajectoryMetricsCallback(TrainerCallback):
    """
    ACT 任务 clip 级动作识别评估回调（只评 GT-KP -> action_id）。
    """

    def __init__(
        self,
        pad_token_id: int = 0,
        eos_token_id: int = 2,
        topk: int = 3,
        metric_prefix: str = "act_traj",
    ):
        # metric_prefix 保留，但不强依赖（指标名本身已含 act_traj_ 前缀）
        self.evaluator = ActionTrajectoryMetricsEvaluator(
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            topk=topk,
        )
        self.metric_prefix = metric_prefix

    def _update_run_history(self, trainer, metrics: Dict[str, float], prefix: str):
        for name, value in metrics.items():
            trainer.run_history.update_metric(f"{prefix}_{name}_accuracy", value)

    def on_train_epoch_start(self, trainer, **kwargs):
        self.evaluator.reset_metrics()

    def on_eval_epoch_start(self, trainer, **kwargs):
        self.evaluator.reset_metrics()

    def _get_targets_and_weights(self, batch, batch_output):
        target_seq = batch_output.get("act_target_seq", batch.get("act_target_seq", None))
        token_weights = batch_output.get("act_token_weights", batch.get("act_token_weights", None))
        return target_seq, token_weights

    def on_train_step_end(self, trainer, batch, batch_output, **kwargs):
        if "act_logits" not in batch_output:
            return

        logits = batch_output["act_logits"]  # [B,S,V]
        target_seq, token_weights = self._get_targets_and_weights(batch, batch_output)
        if target_seq is None:
            return

        metrics = self.evaluator.compute_batch_metrics(
            logits=logits,
            target_seq=target_seq,
            token_weights=token_weights,
            gather_fn=getattr(trainer, "gather", None),
        )
        self._update_run_history(trainer, metrics, "train")

    def on_eval_step_end(self, trainer, batch, batch_output, **kwargs):
        if "act_logits" not in batch_output:
            return

        logits = batch_output["act_logits"]
        target_seq, token_weights = self._get_targets_and_weights(batch, batch_output)
        if target_seq is None:
            return

        metrics = self.evaluator.compute_batch_metrics(
            logits=logits,
            target_seq=target_seq,
            token_weights=token_weights,
            gather_fn=getattr(trainer, "gather", None),
        )
        self._update_run_history(trainer, metrics, "eval")

    def on_train_epoch_end(self, trainer, **kwargs):
        epoch_metrics = self.evaluator.get_epoch_metrics()
        for name, value in epoch_metrics.items():
            trainer.run_history.update_metric(f"train_epoch_{name}_accuracy", value)

    def on_eval_epoch_end(self, trainer, **kwargs):
        epoch_metrics = self.evaluator.get_epoch_metrics()
        for name, value in epoch_metrics.items():
            trainer.run_history.update_metric(f"eval_epoch_{name}_accuracy", value)


