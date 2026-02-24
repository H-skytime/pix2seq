import json
from pathlib import Path

import pandas as pd
import torch
from evaluation.coco_evaluator import (
    CLASS_ID_COL,
    IMAGE_ID_COL,
    SCORE_COL,
    XMAX_COL,
    XMIN_COL,
    YMAX_COL,
    YMIN_COL,
    COCOMeanAveragePrecision,
)
from pytorch_accelerated.callbacks import TrainerCallback

from data.base_dataset import coco80_to_coco91_lookup


class ConvertPredictionClassesCallback(TrainerCallback):
    """Callback to convert prediction class IDs from COCO-80 to COCO-91 format.

    The COCO dataset uses two different category ID systems:
    - COCO-80: A contiguous format with IDs 0-79 commonly used by detection models
    - COCO-91: The original format with IDs 1-91 (with some IDs skipped) used in evaluation

    This callback converts model predictions from COCO-80 to COCO-91 format during
    evaluation to ensure correct mean average precision (MAP) calculation.
    """

    def __init__(self):
        """Initialize the callback with COCO-80 to COCO-91 mapping."""
        self.lookup = coco80_to_coco91_lookup()

    def on_eval_step_end(self, trainer, batch, batch_output, **kwargs):
        """Convert prediction class IDs after each evaluation step.
        - 在验证(evaluation)阶段每一个批次(step)结束时触发, 执行类别ID转换

        Args:
            trainer: The Pix2SeqTrainer instance
            batch: Current batch of data
            batch_output: Dictionary containing model outputs including predictions
                The predictions tensor has shape [N, 7] with columns:
                [xmin, ymin, xmax, ymax, score, class_id, image_id]
            kwargs: Additional arguments passed by the trainer

        Note:
            Modifies batch_output["predictions"] in-place by converting the class_id
            column from COCO-80 to COCO-91 format.
        """
        predictions = batch_output["predictions"]
        # Check if predictions is empty or 1D
        if predictions.numel() == 0 or predictions.dim() == 1:
            return

        try:
            coco_80_class_ids = predictions[:, -2]
            coco_91_class_ids = torch.as_tensor(
                [self.lookup[int(c)] for c in coco_80_class_ids],
                device=predictions.device,
                dtype=predictions.dtype,
            )
            # modify batch output inplace
            batch_output["predictions"][:, -2] = coco_91_class_ids
        except Exception as e:
            print(f"Error processing predictions of shape {predictions.shape}: {e}")
            return


class CalculateMeanAveragePrecisionCallback(TrainerCallback):
    """
    A callback which accumulates predictions made during an epoch and uses these to calculate the Mean Average Precision
    from the given targets.
    一个回调函数, 用于累加一个训练 epoch 内产生的预测结果
    并利用这些结果基于给定的目标标注计算 Mean Average Precision

    .. Note:: If using distributed training or evaluation, this callback assumes that predictions have been gathered
    from all processes during the evaluation step of the main training loop.
    """

    def __init__(
        self,
        targets_json,
        iou_threshold=None,
        save_predictions_output_dir_path=None,
        verbose=False,
    ):
        """
        :param targets_json: a COCO-formatted dictionary with the keys "images", "categories" and "annotations"
        :param iou_threshold: If set, the IoU threshold at which mAP will be calculated. Otherwise, the COCO default range of IoU thresholds will be used.
        :param save_predictions_output_dir_path: If provided, the path to which the accumulated predictions will be saved, in coco json format.
        :param verbose: If True, display the output provided by pycocotools, containing the average precision and recall across a range of box sizes.
        """
        # 初始化COCO评估器
        self.evaluator = COCOMeanAveragePrecision(iou_threshold)
        self.targets_json = targets_json  # 存储真实标签的JSON对象
        self.verbose = verbose  # 是否打印详细的评估日志
        self.save_predictions_path = (
            Path(save_predictions_output_dir_path)
            if save_predictions_output_dir_path is not None
            else None
        )  # 预测结果保存路径

        # 初始化用于存储当前 epoch 预测结果的列表
        self.eval_predictions = []
        # 初始化用于追踪已处理图片ID的集合，防止重复计算
        self.image_ids = set()

    def on_eval_step_end(self, trainer, batch, batch_output, **kwargs): 
        # - 分步累加
        predictions = batch_output["predictions"]  # 提取当前批次的模型预测结果
        if len(predictions) > 0:
            self._update(predictions)

    def on_eval_epoch_end(self, trainer, **kwargs):
        # - 集中计算
        preds_df = pd.DataFrame(
            self.eval_predictions,
            columns=[
                XMIN_COL,
                YMIN_COL,
                XMAX_COL,
                YMAX_COL,
                SCORE_COL,
                CLASS_ID_COL,
                IMAGE_ID_COL,
            ],
        )  # 将累计的列表转换为 Pandas Dataframe

        # 将 DataFrame 转换为符合 COCO 标准的 JSON 格式
        predictions_json = self.evaluator.create_predictions_coco_json_from_df(preds_df)
        # 将预测结果保存到本地文件（可选）
        self._save_predictions(trainer, predictions_json, preds_df)

        # 如果是主进程且开启了 verbose，则打印详细评估报告
        if self.verbose and trainer.run_config.is_local_process_zero:
            self.evaluator.verbose = True

        # 调用 evaluator, 对比真实标签和预测结果, 计算 mAP
        map_ = self.evaluator.compute(self.targets_json, predictions_json)
        # 将计算出的 mAP 更新到 Trainer 的运行历史中, 供日志显示或模型保存参考
        trainer.run_history.update_metric("map", map_)

        # 重置状态, 清空当前 epoch 的数据, 为下一轮做准备
        self._reset()

    @classmethod
    def create_from_targets_df(
        cls,
        targets_df,
        image_ids,
        iou_threshold=None,
        save_predictions_output_dir_path=None,
        verbose=False,
    ):
        """
        Create an instance of :class:`CalculateMeanAveragePrecisionCallback` from a dataframe containing the ground
        truth targets and a collections of all image ids in the dataset.

        :param targets_df: DF w/ cols: ["image_id", "xmin", "ymin", "xmax", "ymax", "class_id"]
        :param image_ids: A collection of all image ids in the dataset, including those without annotations.
        :param iou_threshold:  If set, the IoU threshold at which mAP will be calculated. Otherwise, the COCO default range of IoU thresholds will be used.
        :param save_predictions_output_dir_path: If provided, the path to which the accumulated predictions will be saved, in coco json format.
        :param verbose: If True, display the output provided by pycocotools, containing the average precision and recall across a range of box sizes.
        :return: An instance of :class:`CalculateMeanAveragePrecisionCallback`
        """

        targets_json = COCOMeanAveragePrecision.create_targets_coco_json_from_df(
            targets_df, image_ids
        )

        return cls(
            targets_json=targets_json,
            iou_threshold=iou_threshold,
            save_predictions_output_dir_path=save_predictions_output_dir_path,
            verbose=verbose,
        )

    def _remove_seen(self, labels):
        """
        去除已经处理过的图片ID, 这在分布式训练（数据无法整除时）中非常重要,
        防止同一张图片的预测结果被计算多次
        """
        image_ids = labels[:, -1].tolist()

        # 创建掩码, 标记哪些 ID 是本轮epoch已经见过的
        seen_id_mask = torch.as_tensor(
            [False if idx not in self.image_ids else True for idx in image_ids]
        )

        if seen_id_mask.all():
            # no update required as all ids already seen this pass
            return []
        elif seen_id_mask.any():  # at least one True
            # remove predictions for images already seen this pass
            labels = labels[~seen_id_mask]

        return labels

    def _update(self, predictions):
        # 过滤重复图片的预测结果
        filtered_predictions = self._remove_seen(predictions)

        if len(filtered_predictions) > 0:
            # 将过滤后的预测结果添加到总列表中
            self.eval_predictions.extend(filtered_predictions.tolist())
            # 更新已处理图片 ID 集合
            updated_ids = filtered_predictions[:, -1].unique().tolist()
            self.image_ids.update(updated_ids)

    def _reset(self):
        # 清空状态
        self.image_ids = set()
        self.eval_predictions = []

    def _save_predictions(self, trainer, predictions_json, preds_df):
        if (
            self.save_predictions_path is not None
            and trainer.run_config.is_world_process_zero
        ):
            if len(predictions_json) > 0:
                with open(self.save_predictions_path / "predictions.json", "w") as f:
                    json.dump(predictions_json, f)
                preds_df.to_csv(
                    self.save_predictions_path / "predictions.csv", index=False
                )
