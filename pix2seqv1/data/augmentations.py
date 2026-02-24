from typing import Tuple, Union

import albumentations as A
import cv2
import numpy as np
import torch


class ImageAugmentor:
    """Implements Pix2Seq image augmentation pipeline using albumentations."""

    def __init__(
        self,
        image_size: int = 640,
        jitter_scale: Tuple[float, float] = (0.3, 2.0),
        color_jitter_strength: float = 0.4,
        training: bool = True,
        enable_replay: bool = False,  # Add replay flag
    ):
        self.image_size = image_size
        self.jitter_scale = jitter_scale
        self.color_jitter_strength = color_jitter_strength

        self.training = training
        self.enable_replay = enable_replay
        self.BACKGROUND_VALUE = int(0.3 * 255)
        compose_class = A.ReplayCompose if enable_replay else A.Compose

        # Common bbox params
        bbox_params = A.BboxParams(
            format="pascal_voc",
            min_visibility=0.1,
            label_fields=["labels"],
        ) # pascal_voc: [x_min,y_min,x_max,y_max]; coco: [x_min,y_min,width,weight]

        # Calculate min scale to ensure crop will fit
        scale_limit = (jitter_scale[0], jitter_scale[1]) # 
        if training:
            # Training transforms
            self.transform = compose_class(
                [
                    # Color augmentation - 颜色增强，A.OneOf包裹，二选一执行
                    A.OneOf(
                        [
                            # 颜色抖动，同时调整亮度、对比度、饱和度、色调
                            A.ColorJitter(
                                brightness=color_jitter_strength,
                                contrast=color_jitter_strength,
                                saturation=color_jitter_strength,
                                hue=0.2 * color_jitter_strength,
                                p=0.8,  # 执行概率
                            ),
                            # 转为灰度图
                            A.ToGray(p=0.2),
                        ],
                        p=0.8,  # 二选一增强的总执行概率
                    ),

                    # Geometric augmentations - 几何增强，按顺序执行
                    # 最长边自适应缩放
                    A.LongestMaxSize(max_size=image_size),
                    # 填充为正方形
                    A.PadIfNeeded(
                        min_height=image_size,
                        min_width=image_size,
                        border_mode=cv2.BORDER_CONSTANT,  # 常量填充
                        value=[self.BACKGROUND_VALUE],    # 填充背景色
                    ),
                    # 随机缩放
                    # A.RandomScale(scale_limit=scale_limit, p=1.0),
                    A.RandomScale(scale_limit=0.2, p=1.0), # 
                    # - 避免缩放后尺寸小于裁剪尺寸
                    A.Resize(
                        height=image_size,
                        width=image_size,
                        interpolation=cv2.INTER_LINEAR,
                        always_apply=True
                    ),
                    # 随机尺寸裁剪
                    A.RandomSizedCrop(
                        min_max_height=(int(image_size * 0.8), image_size),  # 裁剪的正方形区域范围
                        height=image_size,
                        width=image_size,
                        w2h_ratio=1.0,
                        p=1.0,
                    ),
                    # 随机水平翻转
                    A.HorizontalFlip(p=0.5),
                ],
                bbox_params=bbox_params,
            )
        else:
            # Eval transforms
            self.transform = compose_class(
                [
                    A.LongestMaxSize(max_size=image_size),
                    A.PadIfNeeded(
                        min_height=image_size,
                        min_width=image_size,
                        border_mode=cv2.BORDER_CONSTANT,
                        value=[self.BACKGROUND_VALUE],
                    ),
                ],
                bbox_params=bbox_params,
            )

    def __call__(
        self,
        image: np.ndarray,  # [H,W,3] in uint8 format
        boxes: np.ndarray,  # [N,4] in absolute pixel coordinates
        labels: np.ndarray,  # [N]
        normalize_boxes: bool = True,
    ):
        """Apply transforms to image and boxes."""
        h, w = image.shape[:2]

        # 确保原始输入坐标不越界
        if len(boxes) > 0:
            boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, w)
            boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, h)

        transformed = self.transform(image=image, bboxes=boxes, labels=labels)

        image = transformed["image"]
        boxes = np.array(transformed["bboxes"])
        labels = np.array(transformed["labels"], dtype=labels.dtype)

        # Calculate scale factor for unpadded size
        scale = self.image_size / max(h, w)
        unpadded_size = (int(h * scale), int(w * scale))  # 缩放且未填充尺寸

        # 边界框尺寸归一化
        if normalize_boxes and len(boxes) > 0:
            boxes = boxes.astype(np.float32)
            boxes[:, [0, 2]] /= image.shape[1]  # normalize x
            boxes[:, [1, 3]] /= image.shape[0]  # normalize y

        return image, boxes, labels, unpadded_size


class BBoxAugmentation:
    """Implements bounding box augmentation strategies from Pix2Seq paper.

    Creates three types of boxes during training:
    1. Positive examples: Original boxes with small jitter
    2. Hard negatives: Original boxes shifted to wrong locations
    3. Random negatives: Randomly generated boxes

    All negative examples (2,3) are assigned the fake class token.
    """

    def __init__(self, num_classes: int):
        self.num_classes = num_classes

        print(
            f"BBoxAugmentation initialized with:"
            f"\n  - num_classes: {num_classes}"
            f"\n  - Will generate fake labels as: {num_classes}"
        )  # Should be 80 for COCO

    def jitter_bbox(
        self,
        bbox: torch.Tensor,
        max_range: float = 0.05,
        truncation: bool = True,
    ):
        """Applies small jitter to create positive examples. - 生成 Positive examples
        - 对原始真实框添加小幅位置抖动，生成“接近真实目标的样本”，提升模型对目标微小变化的鲁棒性"""
        n = len(bbox)
        # heights = bbox[:, 2] - bbox[:, 0]
        # widths = bbox[:, 3] - bbox[:, 1]
        # sizes = torch.stack([heights, widths, heights, widths], -1)
        widths = bbox[:, 2] - bbox[:, 0]  # 计算每个框的宽度（BBOX:XYXY）
        heights = bbox[:, 3] - bbox[:, 1]  # 计算每个框的高度
        sizes = torch.stack([widths, heights, widths, heights], -1)

        # Simple truncated normal distribution for jitter
        noise_rate = torch.randn(n, 4) * (max_range / 2.0)  # 生成正态分布噪声，缩放为max_range/2的幅度
        noise_rate = torch.clamp(noise_rate, -max_range, max_range)  # 限制截断

        bbox = bbox + sizes * noise_rate
        if truncation:
            bbox = torch.clamp(bbox, 0.0, 1.0)  # 保证边界框不超出图片范围
        return bbox

    def shift_bbox(self, bbox: torch.Tensor, truncation: bool = True):
        """Creates hard negative examples by shifting boxes to wrong locations. - 生成 Hard negatives
        - 将原始真实框平移到错误位置（保持框尺寸不变），生成“形状合理但位置错误的样本”，提升模型对目标位置的判别能力

        Maintains original box dimensions to create plausible but incorrect boxes.
        """
        n = len(bbox)
        # heights = bbox[:, 2] - bbox[:, 0]
        # widths = bbox[:, 3] - bbox[:, 1]
        widths = bbox[:, 2] - bbox[:, 0]
        heights = bbox[:, 3] - bbox[:, 1]

        # Random new centers
        cy = torch.rand(n, 1)  # 生成 [N,1] 的随机y轴中心坐标（0~1 之间）
        cx = torch.rand(n, 1)

        shifted = torch.cat(
            [
                cx - widths.unsqueeze(1) / 2,  # xmin
                cy - heights.unsqueeze(1) / 2,  # ymin
                cx + widths.unsqueeze(1) / 2,  # xmax
                cy + heights.unsqueeze(1) / 2,  # ymax
            ],
            -1,
        )

        if truncation:
            shifted = torch.clamp(shifted, 0.0, 1.0)
        return shifted

    def random_bbox(
        self,
        n: int,
        max_size: float = 1.0,
        truncation: bool = True,
        return_labels: bool = False,
    ):
        """Creates random negative examples. - 生成 Random negatives
        - 随机生成边界框，模拟背景中的干扰区域，增加样本多样性，让模型能够区分真实目标和随机背景

        -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]"""
        # Random centers
        cy = torch.rand(n, 1)  # 随机y轴中心
        cx = torch.rand(n, 1)

        # Random dimensions
        h = torch.randn(n, 1) * max_size / 2  # 随机高度
        w = torch.randn(n, 1) * max_size / 2

        bbox = torch.cat(
            [
                cx - torch.abs(w) / 2,
                cy - torch.abs(h) / 2,
                cx + torch.abs(w) / 2,
                cy + torch.abs(h) / 2,
            ],
            -1,
        )

        if truncation:
            bbox = torch.clamp(bbox, 0.0, 1.0)

        if return_labels:
            fake_labels = torch.full((n,), self.num_classes, dtype=torch.long)
            return bbox, fake_labels

        return bbox

    def augment_bbox(
        self,
        bbox: torch.Tensor,  # [N,4] normalized XYXY
        bbox_label: torch.Tensor,  # [N] class indices
        max_jitter: float = 0.05,
        n_noise_bbox: int = 0,
        mix_rate: float = 0.5,  # Added mix_rate parameter
    ):
        """Augment bbox with noise following TF implementation.
        整合Positives examples, Hard negatives and Random negatives, 生成最终框增强数据

        Args:
            bbox: Original boxes in XYXY format - 初始边界框
            bbox_label: Original class labels - 初始类别标签
            max_jitter: Maximum jitter for positive examples - Positive examples最大抖动幅度 
            n_noise_bbox: Number of noise boxes to add - 添加的噪声框数量
            mix_rate: Probability of mixing noise boxes with real boxes - 打乱正例框与负例框的概率阈值

        -> Tuple[torch.Tensor, torch.Tensor]
        """
        n = len(bbox)

        # Ensure valid n_noise_bbox
        n_noise_bbox = max(0, n_noise_bbox)  # Prevent negative

        if n > 0:
            # Small jitter for real boxes but keep original labels
            bbox = self.jitter_bbox(bbox, max_range=max_jitter)  # 生成 Positive examples 

            if n_noise_bbox > 0:
                # 随机分配dup框数量
                dup_bbox_size = torch.randint(0, n_noise_bbox + 1, (1,)).item()
            else:
                dup_bbox_size = 0
        else:
            dup_bbox_size = n_noise_bbox  # 无初始边界框，所有噪声框都暂定dup框

        bad_bbox_size = n_noise_bbox - dup_bbox_size  # bad框数量，包含Hard negatives和Random negatives

        # ===== Create bad boxes - 生成bad框 =====
        if bad_bbox_size > 0:
            if n > 0:
                # Generate both shifted and random boxes
                indices = torch.randint(0, n, (bad_bbox_size,))  # 随机选择bad框数量的真实框索引
                bad_bbox_shift = self.shift_bbox(bbox[indices])  # 生成 Hard negatives
                bad_bbox_random = self.random_bbox(bad_bbox_size)  # 生成 Random negatives

                # Randomly mix shifted and random boxes
                # 生成混合掩码：50%概率为1，50%概率为0
                mix_mask = (torch.rand(bad_bbox_size, 1) < 0.5).float()
                # 按掩码混合两类框：mask=1时取Hard negatives，mask=0时取随机Random negatives
                bad_bbox = mix_mask * bad_bbox_shift + (1 - mix_mask) * bad_bbox_random
            else:
                # If no real boxes, just use random
                # 无初始边界框时，仅生成Random negatives
                bad_bbox = self.random_bbox(bad_bbox_size)

            # Assign fake class labels
            # 为Bad框分配假类别标签
            bad_label = torch.full(
                (bad_bbox_size,),  # 形状[bad_bbox_size]，与Bad框数量匹配
                self.num_classes,  # 假标签值=类别总数（如COCO80类则为80），区别于真实类别[0,79]
                dtype=bbox_label.dtype,
                device=bbox.device,
            )
        else:
            # 无需生成Bad框时，创建空的框和标签
            bad_bbox = bbox.new_zeros((0, 4))
            bad_label = bbox_label.new_zeros(0)

        # ===== Create duplicate boxes if we have real boxes - 生成dup框 =====
        if dup_bbox_size > 0 and n > 0:
            dup_indices = torch.randperm(n)[:dup_bbox_size]
            dup_bbox = self.shift_bbox(bbox[dup_indices])  # 生成dup框：对选中的原始框进行平移（本质是Hard negatives的一种）
            dup_label = torch.full_like(bbox_label[dup_indices], self.num_classes)  # 为dup框分配假标签（与bad框标签一致）
        else:
            # 无需生成dup框时，创建空的框和标签
            dup_bbox = bbox.new_zeros((0, 4))
            dup_label = bbox_label.new_zeros(0)

        # Combine positive and negative examples
        if torch.rand(1) < mix_rate and n > 0: 
            # 模式1：打乱混合模式
            # Mix and shuffle everything together
            noise_bbox = torch.cat([bad_bbox, dup_bbox])  # 合并两类负例框：Bad框（Hard neg + Random neg） + dup框（特殊Hard neg）
            noise_label = torch.cat([bad_label, dup_label])  # 合并两类负例框的标签：Bad标签 + dup标签

            bbox_new = torch.cat([bbox, noise_bbox])
            label_new = torch.cat([bbox_label, noise_label])

            # Random shuffle to mix real and noise boxes
            perm = torch.randperm(len(bbox_new))
            bbox_new = bbox_new[perm]
            label_new = label_new[perm]
        else:
            # 模式2：顺序拼接模式
            # Simply append noise boxes
            noise_bbox = torch.cat([bad_bbox, dup_bbox])
            noise_label = torch.cat([bad_label, dup_label])

            bbox_new = torch.cat([bbox, noise_bbox]) if n > 0 else noise_bbox
            label_new = torch.cat([bbox_label, noise_label]) if n > 0 else noise_label

        return bbox_new, label_new

    def _validate_outputs(self, bbox_new, label_new, n_orig, n_noise_bbox):
        """Add validation to check box augmentation is working correctly."""
        n_total = len(bbox_new)
        n_fake = (label_new == self.num_classes).sum().item()
        n_real = n_total - n_fake

        print("\nBox Augmentation Stats:")
        print(f"Original boxes: {n_orig}")
        print(f"Total boxes: {n_total}")
        print(f"Real boxes: {n_real}")
        print(f"Fake boxes: {n_fake}")
        print(f"Target noise boxes: {n_noise_bbox}")

        # Validate box coordinates
        if len(bbox_new) > 0:
            print("\nBox Coordinates Stats:")
            print(f"Min coords: {bbox_new.min().item():.3f}")
            print(f"Max coords: {bbox_new.max().item():.3f}")
            print(f"Mean coords: {bbox_new.mean().item():.3f}")
