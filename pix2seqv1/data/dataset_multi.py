"""PyTorch dataset implementation for Pix2Seq object detection and segmentation."""

from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from data.augmentations import BBoxAugmentation, ImageAugmentor
from data.base_dataset import COCOBaseDataset
from pix2seqv1.src.pix2seq.data.tokenizer import TokenProcessor


class Pix2SeqDataset(Dataset):
    """PyTorch dataset for Pix2Seq object detection and segmentation."""

    def __init__(
        self,
        base_dataset: COCOBaseDataset,
        num_classes: int,
        training: bool = True,
        max_num_objects: int = 100,
        image_size: int = 640,
        jitter_scale: Tuple[float, float] = (0.3, 2.0), # 尺寸抖动幅度
        color_jitter_strength: float = 0.4,             # 颜色抖动强度
        task: str = "detection",                        # 任务类型
        seg_instances: int = 3,                         # 最大分割实例
    ):
        super().__init__()
        self.base_dataset = base_dataset
        self.num_classes = num_classes
        self.categories = base_dataset.get_categories()
        self.training = training
        self.max_instances = max_num_objects
        self.image_size = image_size  #
        self.task = task  # 
        self.seg_instances = seg_instances  # 

        # [新增] 根据任务类型智能配置 augmentor
        # 分割任务因为处理复杂, 暂时关闭复杂几何变换(Flip/Crop)以保证对齐
        simple_geometry = (task == "segmentation")

        # 初始化图像增强组件
        self.augmentor = ImageAugmentor(
            image_size=image_size,
            jitter_scale=jitter_scale if not simple_geometry else (1.0, 1.0),
            color_jitter_strength=color_jitter_strength,
            training=training,
            enable_replay=False,
            simple_geometry=simple_geometry
        )
        # 初始化边界框增强组件
        self.bbox_augmentor = BBoxAugmentation(num_classes=num_classes)

    def __getitem__(self, idx: int):
        # 1. 从 COCOBaseDataset 解包数据
        image, boxes, labels, raw_polygons, image_id, orig_size = self.base_dataset[idx]
        orig_h, orig_w = orig_size

        # 2. 图像增强 (returns uint8 image and normalized boxes)
        image, boxes, labels, unpadded_size = self.augmentor(
            image, boxes, labels, normalize_boxes=True
        )

        # 转换 Tensor张量
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.long)

        # 3. [新增] polygons 处理与归一化
        processed_polygons = []
        if self.task == "segmentation":
            # 计算缩放与填充参数 (假设 augmentor 执行了 LongestMaxSize + CenterPadding)
            scale = self.image_size / max(orig_h, orig_w)
            new_h, new_w  = int(orig_h * scale), int(orig_w * scale)
            pad_h, pad_w = (self.image_size - new_h) / 2, (self.image_size - new_w) / 2

            for poly in raw_polygons:
                if isinstance(poly, list) and len(poly) > 0:
                    # 针对 “一个物体由多个不连通多边形组成”, 取最大连通域
                    p = max(poly, key=len) if isinstance(poly[0], list) else poly
                    p = np.array(p, dtype=np.float32)
                    
                    # 坐标变换 x_new = x * scale + pad
                    p[0::2] = p[0::2] * scale + pad_w
                    p[1::2] = p[1::2] * scale + pad_h
                    # 归一化
                    p /= self.image_size
                    p = np.clip(p, 0.0, 1.0)
                    processed_polygons.append(p)
                else:
                    processed_polygons.append(np.array([]))

        # 4. 限制边界框数量
        if len(boxes) > self.max_instances:
            # Track occurrence of truncation for monitoring
            print(
                f"Image {image_id} has {len(boxes)} boxes, exceeding max_instances={self.max_instances}. "
                f"Will {'randomly sample' if self.training else 'truncate to first'} {self.max_instances} boxes."
            )

            if self.training:
                # During training, randomly sample max_instances boxes - 随机采样
                indices = torch.randperm(len(boxes))[: self.max_instances]
            else:
                # During evaluation, take first max_instances boxes to ensure deterministic behavior - 顺序采样
                indices = torch.arange(self.max_instances)
            
            boxes = boxes[indices]
            labels = labels[indices]
            # 同步处理 Polygons
            if self.task == "segmentation" and processed_polygons:  
                processed_polygons = [processed_polygons[i] for i in indices.tolist()]

        # 5. [新增] 限制分割实例数量 (Top-K Area)
        if self.task == "segmentation" and len(processed_polygons) > self.seg_instances:
            # 计算 Box 的面积
            widths = boxes[:, 2] - boxes[:, 0]
            heights = boxes[:, 3] - boxes[:, 1]
            areas = widths * heights
            
            # 找到面积最大的前 seg_instances 个索引
            k = min(len(areas), self.seg_instances)
            _, topk_indices = torch.topk(areas, k)
            keep_indices_set = set(topk_indices.tolist())
            
            # 清空非 Top-K 的 Polygon (不删除 Box, 只把 Mask 设空)
            for i in range(len(processed_polygons)):
                if i not in keep_indices_set:
                    processed_polygons[i] = np.array([])
        
        # 6. 边界框增强 (仅检测任务)
        if self.training and self.task == "detection":
            n_noise = max(0, self.max_instances - len(boxes))  # Ensure non-negative
            boxes, labels = self.bbox_augmentor.augment_bbox(
                boxes, labels, max_jitter=0.05, n_noise_bbox=n_noise, mix_rate=0.5,
            )

        num_boxes = len(boxes)

        # Normalize image to [0,1] and convert to CHW format
        image = torch.from_numpy(image.astype(np.float32) / 255.0)
        image = image.permute(2, 0, 1)  # HWC -> CHW

        output = {
            "task_type": self.task,  # [新增] 输出任务类型，供 Collator 路由使用
            "image": image,  # [3,H,W] float32 normalized
            "boxes": boxes,  # [max_instances,4] float32 normalized XYXY
            "labels": labels,  # [max_instances] long
            "image_id": image_id,  # int
            "orig_image_size": torch.tensor(orig_size),  # [2] long -原始图片尺寸
            "unpadded_image_size": torch.tensor(unpadded_size),  # [2] long -转换图片尺寸
            "num_boxes": torch.tensor(len(boxes)),  # int - actual number before padding
        }
        if self.task == "segmentation":
            output["polygons"] = processed_polygons

        return output

    def __len__(self):
        return len(self.base_dataset)


class Pix2SeqCollator:
    def __init__(
        self, token_processor: TokenProcessor, corrupt_and_randomise: bool = False
    ):
        self.token_processor = token_processor
        self._is_training = False
        self.corrupt_and_randomise = corrupt_and_randomise

        # [新增] 任务分发字典: 将 task_type 字符串映射到具体的处理函数
        self.dispatch_map = {
            "detection": self._collate_detection,
            "segmentation": self._collate_segmentation,
        }

    @property  # Add property to ensure consistent state
    def is_training(self):
        return self._is_training

    def set_mode(self, is_training=True):
        if self.corrupt_and_randomise:
            self._is_training = is_training
            self.token_processor._corrupt_class_labels = is_training
            self.token_processor.random_order = is_training

    def __call__(self, batch: List[Dict[str, torch.Tensor]]):

        # 1. 当前任务类型
        if not batch:
            raise ValueError("Received empty batch")
        task_type = batch[0].get("task_type", "detection")  # 默认为 detection

        # 2. 处理通用字段
        batch_data = {
            "task_type": task_type, 
            "image": torch.stack([x["image"] for x in batch]),  # [B,3,H,W] 堆叠batch图片
            "image_id": torch.tensor([x["image_id"] for x in batch]),  # batch图片ID
            "orig_image_sizes": torch.stack([x["orig_image_size"] for x in batch]),  # batch原始图片尺寸
            "unpadded_image_size": torch.stack(  
                [x["unpadded_image_size"] for x in batch]
            ),  # batch未填充图片尺寸
            "num_boxes": torch.tensor([x["num_boxes"] for x in batch]),  # batch每个样本的真实边界框数量
        }
        
        # 3. 动态路由分发
        if task_type in self.dispatch_map:
            handler = self.dispatch_map[task_type]
            return handler(batch, batch_data)
        else:
            raise ValueError(
                f"Pix2SeqCollator received unknown task type: '{task_type}'. "
                f"Supported tasks: {list(self.dispatch_map.keys())}"
            )

    def _collate_detection(self, batch, batch_data):
        # Find max boxes in current batch
        max_boxes = max(x["num_boxes"].item() for x in batch) + 1  # Add 1 for EOS token

        # Pad boxes and labels
        padded_boxes, padded_labels = [], []

        for item in batch:
            num_boxes = item["num_boxes"].item()
            boxes = item["boxes"]  # [N,4]
            labels = item["labels"]  # [N]

            # Random ordering if training - 随机对象排序
            if self._is_training:
                idx = torch.randperm(num_boxes, device=boxes.device)
                boxes = boxes[idx]
                labels = labels[idx]

            if num_boxes < max_boxes:
                # 填充边界框：生成 [max_boxes - num_boxes, 4] 的填充张量，填充值为-1
                pad_boxes = torch.full(
                    (max_boxes - num_boxes, 4), -1, dtype=boxes.dtype, device=boxes.device,
                )
                # 填充标签
                pad_labels = torch.full(
                    (max_boxes - num_boxes,), -1, dtype=labels.dtype, device=labels.device,
                )

                boxes = torch.cat([boxes, pad_boxes], dim=0)  # [max_boxes, 4]
                labels = torch.cat([labels, pad_labels], dim=0)  # [max_boxes]

            padded_boxes.append(boxes)
            padded_labels.append(labels)

        # Stack padded boxes and labels - 堆叠padding的边界框与标签
        batch_data.update(
            {
                "boxes": torch.stack(padded_boxes),  # [B,max_boxes,4]
                "labels": torch.stack(padded_labels),  # [B,max_boxes]
            }
        )

        # Generate sequences using token processor - 构建模型输入/目标序列
        input_seq, target_seq, token_weights = self.token_processor.build_sequences(
            boxes=batch_data["boxes"],  # [B,max_boxes,4]
            labels=batch_data["labels"],  # [B,max_boxes]
        )

        return self._finalize_batch(batch_data, input_seq, target_seq, token_weights)

    def _collate_segmentation(self, batch, batch_data):
        boxes_list = []
        labels_list = []
        polygons_list = []

        max_objs = max(len(x["boxes"]) for x in batch)

        for item in batch:
            n = len(item["boxes"])
            boxes = item["boxes"]
            labels = item["labels"]
            
            if n < max_objs:
                pad_b = torch.full((max_objs - n, 4), -1.0, device=boxes.device)
                pad_l = torch.full((max_objs - n,), -1, dtype=torch.long, device=boxes.device)
                boxes = torch.cat([boxes, pad_b])
                labels = torch.cat([labels, pad_l])
            
            boxes_list.append(boxes)
            labels_list.append(labels)
            polygons_list.append(item["polygons"]) 

        batch_boxes = torch.stack(boxes_list)
        batch_labels = torch.stack(labels_list)

        input_seq, target_seq, weights = self.token_processor.build_segmentation_sequences(
            boxes=batch_boxes,
            labels=batch_labels,
            polygons=polygons_list,
            prompt_mode=True
        )

        return self._finalize_batch(batch_data, input_seq, target_seq, weights)


    def _finalize_batch(self, batch_data, input_seq, target_seq, token_weights):
        """Finalize batch by handling sequence shifting and mask creation."""

        # 1. Shift Target Sequence (错位预测)
        shifted_target_seq = torch.cat(
            [
                target_seq[:, 1:],
                torch.full_like(target_seq[:, :1], self.token_processor.PADDING_TOKEN),
            ],
            dim=1,
        )

        # 2. Shift Token Weights (权重同步移位)
        shifted_token_weights = torch.cat(
            [
                token_weights[:, 1:],
                torch.zeros_like(token_weights[:, :1]), # 移位后补 0
            ],
            dim=1
        )

        # 3. Generate Masks
        input_padding_mask = input_seq == self.token_processor.PADDING_TOKEN
        target_padding_mask = shifted_target_seq == self.token_processor.PADDING_TOKEN

        batch_data.update({
            "input_seq": input_seq,
            "target_seq": shifted_target_seq,
            "token_weights": shifted_token_weights,
            "input_padding_mask": input_padding_mask,
            "target_padding_mask": target_padding_mask,
        })
        return batch_data