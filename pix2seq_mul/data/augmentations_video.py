"""
Video / Tube augmentors for detection + keypoints on video clips.

This module provides:
1) VideoAugmentorKpD: clip-level augmentation (same geometric/color transform replayed on all frames)
2) TubeAugmentorKpD: instance-order shuffle per frame (keeps tube IDs aligned)

Design goals:
- Deterministic multi-frame geometry via Albumentations ReplayCompose
- Synchronized update for bboxes / labels / keypoints / visibility
- Keep person_ids / object_ids aligned with filtered/reordered instances
"""

from __future__ import annotations

import inspect
import logging
from typing import Any, Dict, List, Optional, Tuple

import albumentations as A
import cv2
import numpy as np


LOGGER = logging.getLogger(__name__)


class VideoAugmentorKpD:
    """
    对整段视频 clip（T 帧）执行“统一增强”（同一组随机参数复用到所有帧）。

    核心特性
    - 使用 Albumentations ReplayCompose：第 1 帧采样随机增强参数，后续帧 replay 同一参数。
    - 同步更新每帧的 boxes / class_ids / keypoints / keypoints_visible。
    - person_ids / object_ids 不参与几何变换，但会随 bbox 的筛选/重排保持对齐。

    预期输入 sample（与常见 COCO-as-video/NTU 风格兼容）
        sample = {
            "images": np.ndarray,              # [T,H,W,3], RGB, uint8
            "boxes": List[np.ndarray],         # len=T, each [N_t,4] xyxy 像素坐标
            "class_ids": List[np.ndarray],     # len=T, each [N_t]
            "keypoints": List[np.ndarray],     # len=T, each [N_t,K,3]
            "keypoints_visible": List[np.ndarray],  # len=T, each [N_t,K]
            "person_ids": List[np.ndarray],    # 可选，len=T, each [N_t]
            "object_ids": List[np.ndarray],    # 可选，len=T, each [N_t]
            "frame_valid_mask": np.ndarray,    # 可选，[T]
            ...
        }

    输出
    - 结构与输入一致；images/boxes/keypoints 等字段被增强；
    - 若 normalize_boxes=True，则 boxes 与 keypoints 归一化到 [0,1]（相对于增强后图像尺寸）。
    """

    # 经验背景灰（pad 用），保持为模块常量方便统一修改
    _DEFAULT_BACKGROUND_VALUE = int(0.3 * 255)

    def __init__(
        self,
        image_size: int = 640,
        jitter_scale: Tuple[float, float] = (0.3, 2.0),
        color_jitter_strength: float = 0.4,
        training: bool = True,
        normalize_boxes: bool = True,
        kp_flip_pairs: Optional[List[Tuple[int, int]]] = None,
    ):
        """
        Args:
            image_size: 输出目标尺寸（增强后为 image_size x image_size）
            jitter_scale: RandomScale 缩放范围（min_scale, max_scale）
            color_jitter_strength: 颜色扰动强度（ColorJitter）
            training: True=训练增强（随机）；False=仅 resize+pad（确定性）
            normalize_boxes: True=输出 boxes/keypoints 归一化到 [0,1]
            kp_flip_pairs: 水平翻转后左右关键点交换对（用于语义对齐）
        """
        if image_size <= 0:
            raise ValueError("image_size must be positive.")
        if jitter_scale[0] <= 0 or jitter_scale[1] <= 0 or jitter_scale[0] > jitter_scale[1]:
            raise ValueError("jitter_scale must be positive and satisfy min<=max.")
        if not (0.0 <= color_jitter_strength):
            raise ValueError("color_jitter_strength must be non-negative.")

        self.image_size = int(image_size)
        self.jitter_scale = (float(jitter_scale[0]), float(jitter_scale[1]))
        self.color_jitter_strength = float(color_jitter_strength)
        self.training = bool(training)
        self.normalize_boxes = bool(normalize_boxes)

        # 默认 U-16 左右关节交换对（可按你的关键点定义覆盖）
        if kp_flip_pairs is None:
            self.kp_flip_pairs: List[Tuple[int, int]] = [
                (4, 5),    # Shoulder
                (6, 7),    # Elbow
                (8, 9),    # Wrist / Hand
                (10, 11),  # Hip
                (12, 13),  # Knee
                (14, 15),  # Ankle
            ]
        else:
            self.kp_flip_pairs = [(int(a), int(b)) for a, b in kp_flip_pairs]

        self.background_value = self._DEFAULT_BACKGROUND_VALUE

        self._transform = self._build_transform()

    # -------------------------------------------------------------------------
    # Transform builders (Albumentations version compatibility)
    # -------------------------------------------------------------------------
    @staticmethod
    def _pad_to_square(img_size: int, bg: int) -> A.BasicTransform:
        """
        构造 PadIfNeeded：兼容 albumentations 1.x / 2.x 的参数差异。
        """
        sig = inspect.signature(A.PadIfNeeded.__init__)
        if "border_value" in sig.parameters:  # albumentations >= 2.x
            return A.PadIfNeeded(
                min_height=img_size,
                min_width=img_size,
                border_mode=cv2.BORDER_CONSTANT,
                position="center",
                border_value=bg,
            )

        # albumentations 1.x
        return A.PadIfNeeded(
            min_height=img_size,
            min_width=img_size,
            border_mode=cv2.BORDER_CONSTANT,
            position="center",
            fill=bg,
        )

    @staticmethod
    def _random_square_crop(img_size: int) -> A.BasicTransform:
        """
        构造“尽量方形”的随机裁剪变换，兼容不同版本的 RandomResizedCrop / RandomSizedCrop。
        """
        if hasattr(A, "RandomResizedCrop"):
            # albumentations 在不同版本中参数名可能是 size 或 height/width
            try:
                return A.RandomResizedCrop(
                    size=(img_size, img_size),
                    scale=(0.8, 1.0),
                    ratio=(0.999, 1.001),
                    p=1.0,
                )
            except Exception:
                return A.RandomResizedCrop(
                    height=img_size,
                    width=img_size,
                    scale=(0.8, 1.0),
                    ratio=(0.999, 1.001),
                    p=1.0,
                )

        # 旧版本兼容
        return A.RandomSizedCrop(
            min_max_height=(int(img_size * 0.8), img_size),
            height=img_size,
            width=img_size,
            w2h_ratio=1.0,
            p=1.0,
        )

    def _build_transform(self) -> A.ReplayCompose:
        """
        构建 ReplayCompose pipeline（训练/测试两种模式）。
        """
        bbox_params = A.BboxParams(
            format="pascal_voc",  # xyxy: [x_min, y_min, x_max, y_max]
            min_visibility=0.1,
            label_fields=["labels", "bbox_ids"],
        )
        keypoint_params = A.KeypointParams(
            format="xy",            # 仅 (x,y)，可见性由外部字段维护
            remove_invisible=False,  # 保留出界点，后续我们会显式处理可见性与裁剪
        )

        scale_limit = (self.jitter_scale[0] - 1.0, self.jitter_scale[1] - 1.0)

        if self.training:
            transforms: List[A.BasicTransform] = [
                A.OneOf(
                    [
                        A.ColorJitter(
                            brightness=self.color_jitter_strength,
                            contrast=self.color_jitter_strength,
                            saturation=self.color_jitter_strength,
                            hue=0.2 * self.color_jitter_strength,
                            p=0.8,
                        ),
                        A.ToGray(p=0.2),
                    ],
                    p=0.8,
                ),
                A.LongestMaxSize(max_size=self.image_size),
                self._pad_to_square(self.image_size, self.background_value),
                A.RandomScale(scale_limit=scale_limit, p=1.0),
                self._random_square_crop(self.image_size),
                A.HorizontalFlip(p=0.5),
            ]
        else:
            transforms = [
                A.LongestMaxSize(max_size=self.image_size),
                self._pad_to_square(self.image_size, self.background_value),
            ]

        return A.ReplayCompose(
            transforms,
            bbox_params=bbox_params,
            keypoint_params=keypoint_params,
        )

    # -------------------------------------------------------------------------
    # Replay helpers
    # -------------------------------------------------------------------------
    @staticmethod
    def _replay_has_applied_transform(replay: Optional[Dict[str, Any]], *, name: str) -> bool:
        """
        递归检查 ReplayCompose 的 replay 字典中，指定 transform 是否 applied=True。

        Args:
            replay: ReplayCompose 产出的 replay 字典
            name:  目标 transform 的类名（例如 "HorizontalFlip"）
        """
        if not replay:
            return False

        def _walk(node: Any) -> bool:
            if isinstance(node, dict):
                cls_full = str(node.get("__class_fullname__", ""))
                cls_short = cls_full.split(".")[-1] if cls_full else ""
                if bool(node.get("applied", False)) and (cls_short == name or cls_full.endswith(name)):
                    return True

                children = node.get("transforms")
                if children is not None:
                    return _walk(children)
                return False

            if isinstance(node, (list, tuple)):
                return any(_walk(x) for x in node)

            return False

        return _walk(replay.get("transforms", replay))

    def _swap_kp_pairs_inplace(self, kps: np.ndarray, kps_vis: np.ndarray) -> None:
        """
        水平翻转后交换左右关节索引（语义对齐）。

        注意：就地修改 kps 与 kps_vis。
        """
        if kps.size == 0 or kps_vis.size == 0:
            return

        num_joints = int(kps.shape[1])
        for left, right in self.kp_flip_pairs:
            if 0 <= left < num_joints and 0 <= right < num_joints and left != right:
                kps[:, [left, right], :] = kps[:, [right, left], :]
                kps_vis[:, [left, right]] = kps_vis[:, [right, left]]

    # -------------------------------------------------------------------------
    # Single-frame augmentation
    # -------------------------------------------------------------------------
    def _augment_one_frame(
        self,
        image: np.ndarray,
        boxes: Optional[np.ndarray],
        labels: Optional[np.ndarray],
        keypoints: Optional[np.ndarray],
        keypoints_visible: Optional[np.ndarray],
        person_ids: Optional[np.ndarray],
        object_ids: Optional[np.ndarray],
        replay: Optional[Dict[str, Any]],
        normalize_boxes: bool,
    ):
        """
        对单帧做增强（支持 replay 复用），并保持 boxes/labels/kps/ids 对齐。

        说明
        - Albumentations 的 keypoints 以 list[(x,y)] 形式输入输出；
        - 我们额外维护 kp_meta 以将 flat keypoints 还原回 [N,K,3]；
        - 裁剪/缩放后，若关键点出界，则置 visible=0（避免错误监督）。
        """
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("image must be [H,W,3].")

        orig_h, orig_w = image.shape[:2]

        boxes_arr = np.asarray(boxes, dtype=np.float32) if boxes is not None else np.zeros((0, 4), np.float32)
        labels_arr = np.asarray(labels, dtype=np.int64) if labels is not None else np.zeros((0,), np.int64)

        num_instances = int(boxes_arr.shape[0])
        bbox_ids = list(range(num_instances))

        # -------- flatten keypoints: [N,K,3] -> flat list[(x,y)] + meta --------
        flat_kps: List[Tuple[float, float]] = []
        kp_meta: List[Tuple[int, int, float]] = []  # (orig_inst, joint_idx, vis)

        num_joints = 0
        if keypoints is not None and keypoints.ndim == 3 and keypoints.shape[0] > 0:
            keypoints_arr = np.asarray(keypoints, dtype=np.float32)
            num_joints = int(keypoints_arr.shape[1])
            for i in range(int(keypoints_arr.shape[0])):
                for j in range(num_joints):
                    x, y, v = keypoints_arr[i, j]
                    flat_kps.append((float(x), float(y)))

                    if keypoints_visible is not None and np.asarray(keypoints_visible).shape[0] == keypoints_arr.shape[0]:
                        vis = float(np.asarray(keypoints_visible)[i, j])
                    else:
                        vis = 1.0 if float(v) > 0 else 0.0

                    kp_meta.append((i, j, vis))
        else:
            # 无关键点：保持 num_joints=0，后续返回空结构
            num_joints = 0

        # -------- Albumentations call (sample or replay) --------
        if replay is None:
            transformed = self._transform(
                image=image,
                bboxes=boxes_arr.tolist(),
                labels=labels_arr.tolist(),
                bbox_ids=bbox_ids,
                keypoints=flat_kps,
            )
            replay_out = transformed.get("replay")
        else:
            transformed = A.ReplayCompose.replay(
                replay,
                image=image,
                bboxes=boxes_arr.tolist(),
                labels=labels_arr.tolist(),
                bbox_ids=bbox_ids,
                keypoints=flat_kps,
            )
            replay_out = replay

        image_t = transformed["image"]
        boxes_out = np.asarray(transformed["bboxes"], dtype=np.float32)          # [N',4]
        labels_out = np.asarray(transformed["labels"], dtype=labels_arr.dtype)   # [N']
        bbox_ids_out = np.asarray(transformed["bbox_ids"], dtype=np.int64)       # [N']
        flat_kps_out = transformed["keypoints"]                                  # len == len(flat_kps)

        # 是否发生水平翻转（用于关键点左右语义交换）
        did_hflip = self._replay_has_applied_transform(replay_out, name="HorizontalFlip")

        # pad 前大小（与你原始逻辑保持一致：按 LongestMaxSize 的 scale 计算）
        scale = self.image_size / float(max(orig_h, orig_w))
        unpadded_size = (int(round(orig_h * scale)), int(round(orig_w * scale)))

        # -------- normalize boxes to [0,1] --------
        h_t, w_t = image_t.shape[:2]
        if normalize_boxes and boxes_out.shape[0] > 0:
            boxes_out[:, [0, 2]] /= float(w_t)
            boxes_out[:, [1, 3]] /= float(h_t)

        # -------- restore keypoints to [N',K,3] and visibility --------
        kept_n = int(boxes_out.shape[0])
        if num_joints > 0:
            kps_restored = np.zeros((kept_n, num_joints, 3), dtype=np.float32)
            kps_vis_restored = np.zeros((kept_n, num_joints), dtype=np.float32)

            # 原实例 idx -> 新实例 idx
            orig2new = {int(orig_idx): new_i for new_i, orig_idx in enumerate(bbox_ids_out.tolist())}

            # flat_kps_out 与 kp_meta 一一对应
            for (x_t, y_t), (orig_inst, joint_idx, vis_old) in zip(flat_kps_out, kp_meta):
                if orig_inst not in orig2new:
                    continue

                new_inst = orig2new[orig_inst]

                raw_x = float(x_t)
                raw_y = float(y_t)
                finite_xy = np.isfinite(raw_x) and np.isfinite(raw_y)
                in_frame = finite_xy and (0.0 <= raw_x < float(w_t)) and (0.0 <= raw_y < float(h_t))
                vis_new = 1.0 if (vis_old > 0 and in_frame) else 0.0

                # 坐标安全裁剪：即使不可见，也裁剪到合法范围，避免 NaN/Inf 传播
                if not finite_xy:
                    raw_x, raw_y = 0.0, 0.0

                x_clip = float(np.clip(raw_x, 0.0, float(w_t - 1)))
                y_clip = float(np.clip(raw_y, 0.0, float(h_t - 1)))

                kps_restored[new_inst, joint_idx, 0] = x_clip
                kps_restored[new_inst, joint_idx, 1] = y_clip
                kps_restored[new_inst, joint_idx, 2] = 2.0 if vis_new > 0 else 0.0
                kps_vis_restored[new_inst, joint_idx] = vis_new

            if normalize_boxes and kept_n > 0:
                kps_restored[..., 0] /= float(w_t)
                kps_restored[..., 1] /= float(h_t)

            # 翻转后左右关节语义交换
            if did_hflip and kept_n > 0 and self.kp_flip_pairs:
                self._swap_kp_pairs_inplace(kps_restored, kps_vis_restored)
        else:
            kps_restored = np.zeros((kept_n, 0, 3), dtype=np.float32)
            kps_vis_restored = np.zeros((kept_n, 0), dtype=np.float32)

        # -------- keep ids aligned with bbox filtering/reordering --------
        def _sync_ids(ids_arr: Optional[np.ndarray]) -> np.ndarray:
            """
            将 ids 按 bbox_ids_out 同步筛选/重排；若缺失或长度不匹配则填 -1。
            """
            if ids_arr is None:
                return np.full((kept_n,), -1, dtype=np.int64)

            ids_np = np.asarray(ids_arr, dtype=np.int64)
            if ids_np.shape[0] != num_instances:
                return np.full((kept_n,), -1, dtype=np.int64)

            if kept_n == 0:
                return np.zeros((0,), dtype=np.int64)

            return ids_np[bbox_ids_out]

        person_ids_t = _sync_ids(person_ids)
        object_ids_t = _sync_ids(object_ids)

        return (
            image_t,
            boxes_out,
            labels_out,
            kps_restored,
            kps_vis_restored,
            person_ids_t,
            object_ids_t,
            replay_out,
            unpadded_size,
        )

    # -------------------------------------------------------------------------
    # Clip-level augmentation
    # -------------------------------------------------------------------------
    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        对 clip sample 执行统一增强。

        Args:
            sample: dataset __getitem__ 返回的字典（见类注释）

        Notes:
            - 本函数不会原地改写输入 sample；会浅拷贝并替换相关字段。
            - 若输入字段缺失或维度异常，将抛出异常以便尽早暴露数据问题。
        """
        images = sample["images"]
        boxes_list = sample["boxes"]
        class_ids_list = sample["class_ids"]
        keypoints_list = sample["keypoints"]
        keypoints_vis_list = sample["keypoints_visible"]

        person_ids_list = sample.get("person_ids")
        object_ids_list = sample.get("object_ids")

        if not isinstance(images, np.ndarray) or images.ndim != 4:
            raise ValueError('"images" must be a numpy array with shape [T,H,W,3].')

        t_frames = int(images.shape[0])

        # 输出容器
        images_out: List[np.ndarray] = []
        boxes_out: List[np.ndarray] = []
        class_ids_out: List[np.ndarray] = []
        kps_out: List[np.ndarray] = []
        kps_vis_out: List[np.ndarray] = []
        person_ids_out: List[np.ndarray] = []
        object_ids_out: List[np.ndarray] = []

        replay: Optional[Dict[str, Any]] = None
        last_unpadded_size: Optional[Tuple[int, int]] = None

        for t in range(t_frames):
            person_ids_t = person_ids_list[t] if person_ids_list is not None else None
            object_ids_t = object_ids_list[t] if object_ids_list is not None else None

            (
                img_aug,
                boxes_aug,
                labels_aug,
                kps_aug,
                kps_vis_aug,
                person_ids_aug,
                object_ids_aug,
                replay,
                last_unpadded_size,
            ) = self._augment_one_frame(
                image=images[t],
                boxes=boxes_list[t],
                labels=class_ids_list[t],
                keypoints=keypoints_list[t],
                keypoints_visible=keypoints_vis_list[t],
                person_ids=person_ids_t,
                object_ids=object_ids_t,
                replay=replay,
                normalize_boxes=self.normalize_boxes,
            )

            images_out.append(img_aug)
            boxes_out.append(boxes_aug)
            class_ids_out.append(labels_aug)
            kps_out.append(kps_aug)
            kps_vis_out.append(kps_vis_aug)
            person_ids_out.append(person_ids_aug)
            object_ids_out.append(object_ids_aug)

        images_out_np = np.stack(images_out, axis=0)
        h_new, w_new = images_out_np.shape[1:3]

        new_sample = dict(sample)  # 浅拷贝，避免改写原 sample 的引用
        new_sample["images"] = images_out_np
        new_sample["boxes"] = boxes_out
        new_sample["class_ids"] = class_ids_out
        new_sample["keypoints"] = kps_out
        new_sample["keypoints_visible"] = kps_vis_out
        new_sample["person_ids"] = person_ids_out
        new_sample["object_ids"] = object_ids_out
        new_sample["image_hw"] = (h_new, w_new)
        new_sample["unpadded_size"] = last_unpadded_size  # 可选信息：pad 前大小

        return new_sample


class TubeAugmentorKpD:
    """
    Tube 级“实例顺序增强”（训练阶段常用）：对 clip 的每一帧随机打乱实例维度顺序。

    目的
    - 防止模型学习“实例在数组中的固定排序规律”
    - 保持 tube / ID 的跨帧关联依赖 person_ids / object_ids，而不是下标位置

    作用字段（逐帧一致打乱）
    - boxes[t]
    - class_ids[t]
    - keypoints[t]
    - keypoints_visible[t]
    - person_ids[t]
    - object_ids[t]

    参数
        shuffle_prob:       每帧执行打乱的概率（0~1）
        min_instances:      至少多少个实例才考虑打乱（避免 N=0/1 无意义）
        only_valid_frames:  若 sample 提供 frame_valid_mask，则仅对有效帧增强
        seed:               可选随机种子（便于可复现）
    """

    def __init__(
        self,
        shuffle_prob: float = 0.5,
        min_instances: int = 2,
        only_valid_frames: bool = True,
        seed: Optional[int] = None,
        logger: Optional[logging.Logger] = None,
    ):
        if not (0.0 <= shuffle_prob <= 1.0):
            raise ValueError("shuffle_prob must be in [0, 1].")

        self.shuffle_prob = float(shuffle_prob)
        self.min_instances = max(1, int(min_instances))
        self.only_valid_frames = bool(only_valid_frames)
        self.rng = np.random.default_rng(seed)

        self._logger = logger if logger is not None else LOGGER
        self._logger.info(
            "Initialized TubeAugmentorKpD(shuffle_prob=%.3f, min_instances=%d, only_valid_frames=%s, seed=%s)",
            self.shuffle_prob,
            self.min_instances,
            self.only_valid_frames,
            str(seed),
        )

    def _augment_one_frame(
        self,
        boxes_t: Any,
        class_ids_t: Any = None,
        keypoints_t: Any = None,
        keypoints_visible_t: Any = None,
        person_ids_t: Any = None,
        object_ids_t: Any = None,
        frame_valid: bool = True,
    ):
        """
        对单帧做实例维度打乱：
        - 生成同一个 index（bbox_ids_out），对所有字段同时索引；
        - 若字段长度与 boxes 不匹配，则保持原样（以容忍异常数据）。
        """
        if boxes_t is None:
            return (
                boxes_t,
                class_ids_t,
                keypoints_t,
                keypoints_visible_t,
                person_ids_t,
                object_ids_t,
            )

        boxes_np = np.asarray(boxes_t)
        num_instances = int(boxes_np.shape[0])

        # 帧无效或无实例：原样返回
        if num_instances == 0 or (self.only_valid_frames and not frame_valid):
            return (
                boxes_np,
                class_ids_t,
                keypoints_t,
                keypoints_visible_t,
                person_ids_t,
                object_ids_t,
            )

        do_shuffle = (num_instances >= self.min_instances) and (self.rng.random() <= self.shuffle_prob)
        if do_shuffle:
            idx = self.rng.permutation(num_instances)
        else:
            idx = np.arange(num_instances, dtype=np.int64)

        def _reorder(arr: Any) -> Any:
            """
            使用同一 idx 对字段进行重排；若 arr 为 None 或长度不匹配则返回原值。
            """
            if arr is None:
                return None
            arr_np = np.asarray(arr)
            if arr_np.shape[0] != num_instances:
                return arr
            return arr_np[idx]

        boxes_out = boxes_np[idx]
        class_ids_out = _reorder(class_ids_t)
        keypoints_out = _reorder(keypoints_t)
        keypoints_visible_out = _reorder(keypoints_visible_t)
        person_ids_out = _reorder(person_ids_t)
        object_ids_out = _reorder(object_ids_t)

        return (
            boxes_out,
            class_ids_out,
            keypoints_out,
            keypoints_visible_out,
            person_ids_out,
            object_ids_out,
        )

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        对 sample（clip）执行逐帧实例维度的重排增强。

        约定
        - 本函数会“就地改写” sample 的相关字段引用（与数据增强常见做法一致）。
        - 若 sample 中缺失 boxes，则直接返回 sample。
        """
        boxes_list = sample.get("boxes")
        if boxes_list is None:
            return sample

        t_frames = len(boxes_list)

        frame_valid_mask: Optional[np.ndarray] = sample.get("frame_valid_mask")
        class_ids_list = sample.get("class_ids")
        keypoints_list = sample.get("keypoints")
        keypoints_visible_list = sample.get("keypoints_visible")
        person_ids_list = sample.get("person_ids")
        object_ids_list = sample.get("object_ids")

        new_boxes_list: List[Any] = [None] * t_frames
        new_class_ids_list: Optional[List[Any]] = [None] * t_frames if class_ids_list is not None else None
        new_keypoints_list: Optional[List[Any]] = [None] * t_frames if keypoints_list is not None else None
        new_keypoints_visible_list: Optional[List[Any]] = [None] * t_frames if keypoints_visible_list is not None else None
        new_person_ids_list: Optional[List[Any]] = [None] * t_frames if person_ids_list is not None else None
        new_object_ids_list: Optional[List[Any]] = [None] * t_frames if object_ids_list is not None else None

        for t in range(t_frames):
            frame_valid = True
            if frame_valid_mask is not None and t < len(frame_valid_mask):
                frame_valid = bool(frame_valid_mask[t] > 0)

            class_ids_t = class_ids_list[t] if class_ids_list is not None and t < len(class_ids_list) else None
            keypoints_t = keypoints_list[t] if keypoints_list is not None and t < len(keypoints_list) else None
            keypoints_visible_t = (
                keypoints_visible_list[t]
                if keypoints_visible_list is not None and t < len(keypoints_visible_list)
                else None
            )
            person_ids_t = person_ids_list[t] if person_ids_list is not None and t < len(person_ids_list) else None
            object_ids_t = object_ids_list[t] if object_ids_list is not None and t < len(object_ids_list) else None

            (
                boxes_out,
                class_ids_out,
                keypoints_out,
                keypoints_visible_out,
                person_ids_out,
                object_ids_out,
            ) = self._augment_one_frame(
                boxes_t=boxes_list[t],
                class_ids_t=class_ids_t,
                keypoints_t=keypoints_t,
                keypoints_visible_t=keypoints_visible_t,
                person_ids_t=person_ids_t,
                object_ids_t=object_ids_t,
                frame_valid=frame_valid,
            )

            new_boxes_list[t] = boxes_out
            if new_class_ids_list is not None:
                new_class_ids_list[t] = class_ids_out
            if new_keypoints_list is not None:
                new_keypoints_list[t] = keypoints_out
            if new_keypoints_visible_list is not None:
                new_keypoints_visible_list[t] = keypoints_visible_out
            if new_person_ids_list is not None:
                new_person_ids_list[t] = person_ids_out
            if new_object_ids_list is not None:
                new_object_ids_list[t] = object_ids_out

        sample["boxes"] = new_boxes_list
        if new_class_ids_list is not None:
            sample["class_ids"] = new_class_ids_list
        if new_keypoints_list is not None:
            sample["keypoints"] = new_keypoints_list
        if new_keypoints_visible_list is not None:
            sample["keypoints_visible"] = new_keypoints_visible_list
        if new_person_ids_list is not None:
            sample["person_ids"] = new_person_ids_list
        if new_object_ids_list is not None:
            sample["object_ids"] = new_object_ids_list

        return sample


# -------------------------------------------------------------------------
# Backward-compatible aliases (if your existing code imports old names)
# -------------------------------------------------------------------------
VideoAugmentor_kp_d = VideoAugmentorKpD
TubeAugmentor_kp_d = TubeAugmentorKpD
