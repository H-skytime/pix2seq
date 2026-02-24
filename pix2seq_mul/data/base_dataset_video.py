import os
import json
import math
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.datasets.coco import CocoDetection
import torchvision
import cv2  # pip install opencv-python

try:
    from pycocotools.coco import COCO
except ImportError:
    COCO = None


"""
统一说明一下 sample 的结构（NTU 与 COCO-包装成视频 都保持一致）：

sample = {
    "images": images,                  # [T, H, W, 3] uint8，RGB
    "boxes": boxes_list,               # 长度 T；第 t 个是 [N_t, 4]，xyxy
    "class_ids": class_ids_list,       # 长度 T；第 t 个是 [N_t]，类别 id（0-79）
    "keypoints": keypoints_list,       # 长度 T；第 t 个是 [N_t, K, 3]
    "keypoints_visible": keypoints_vis_list,  # 长度 T；第 t 个是 [N_t, K]

    # ★ 跨帧 ID（用于跟踪、视频建模）
    #   - 对于 NTU：
    #       person_ids: 人类实例的跨帧 ID，来自 _person_global_id，非人目标为 -1
    #       object_ids: 非人实例的跨帧 ID，来自 _object_global_id，人类为 -1
    #   - 对于 COCO：
    #       person_ids / object_ids 均为 -1（只为对齐结构，COCO 没有跨帧 ID）
    "person_ids": person_ids_list,     # 长度 T；第 t 个是 [N_t]
    "object_ids": object_ids_list,     # 长度 T；第 t 个是 [N_t]

    "video_id": video_id,              # NTU：SxxxCxxxPxxxRxxxAxxx；COCO：img_123456
    "frame_ids": frame_indices,        # NTU：原视频帧号；COCO：0..T-1
    "image_ids": image_ids_center,     # 标注 JSON 中 images.id
    "image_hw": image_hw,              # (H, W)
    "frame_valid_mask": frame_valid_mask,  # [T]，1 表示真实帧，0 表示 padding 帧

    "action_id": video_meta.get("action_id", None),
    "action_name": video_meta.get("action_name", None),
}
"""


# ===================== COCO 91 -> COCO 80 映射 =====================
COCO80_TO_COCO91_MAP = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
    35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
    46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
    56, 57, 58, 59, 60, 61, 62, 63, 64, 65,
    67, 70, 72, 73, 74, 75, 76, 77, 78, 79,
    80, 81, 82, 84, 85, 86, 87, 88, 89, 90
]


def coco91_to_coco80_lookup():
    return {v: i for i, v in enumerate(COCO80_TO_COCO91_MAP)}


# ===================================================================
#   1. NTU 视频数据集：从 avi + merged JSON 读取，支持 person_ids + object_ids
# ===================================================================
class VideoClipDataset_kp_d(Dataset):
    """
    读取 NTU 格式：
        root/
          ├── annotations/
          │     ├── S001C001P001R001A043_merged.json
          └── videos/
                ├── S001C001P001R001A043_rgb.avi

    抽帧逻辑：
      - 从 json 的 videos[0].length 取视频总帧数 L（例如 75）
      - 把 [0, L) 均分为 clip_len 段，每段取中间帧 center_idx
      - 对每个 center_idx，额外看 center_idx±neighbor_window 帧的标注，做“查漏补缺” + NMS
      - 按 center_idx 从 avi 中解码图像帧

    输出（每条样本就是一个视频 clip，长度 clip_len，默认 8）：
      - images: [T,H,W,3] uint8
      - boxes:  List[T]，每项 [N_t,4]，xyxy 绝对坐标
      - class_ids: List[T]，每项 [N_t]，类别 0-79（COCO80 映射）
      - keypoints: List[T]，每项 [N_t,K,3]（对融合得到的 person 也解码 keypoints_u16；v 作为深度保留）
      - keypoints_visible: List[T]，每项 [N_t,K]（不再用 v 判可见；若 keypoints_u16 合法则默认全 1）
      - person_ids: List[T]，每项 [N_t]，人类实例的跨帧 ID（_person_global_id），其他类别为 -1
      - object_ids: List[T]，每项 [N_t]，非人实例的跨帧 ID（_object_global_id），人类为 -1
    """

    def __init__(
        self,
        root_dir: str,
        clip_len: int = 8,
        filter_empty: bool = True,
        min_area: Optional[float] = None,
        tfms: Optional[Any] = None,
        use_neighbor_fusion: bool = True,
        neighbor_window: int = 1,
        nms_iou_threshold: float = 0.5,
    ):
        super().__init__()

        self.root_dir = root_dir
        self.ann_dir = os.path.join(root_dir, "annotations")
        self.video_dir = os.path.join(root_dir, "videos")

        self.clip_len = clip_len
        self.filter_empty = filter_empty
        self.min_area = min_area
        self.tfms = tfms

        # 邻帧补标 + NMS
        self.use_neighbor_fusion = use_neighbor_fusion
        self.neighbor_window = neighbor_window
        self.nms_iou_threshold = nms_iou_threshold

        # 目标类别映射：COCO91 -> COCO80 (0-79)
        self.coco91_to_coco80: Dict[int, int] = coco91_to_coco80_lookup()
        self.valid_category_ids = set(self.coco91_to_coco80.keys())
        # 类别名字（0-79）
        self.categories: Dict[int, str] = {}

        # ★ 动作类别映射：NTU 原始 action_id -> 压缩后的 0-3
        self.action_id_to_idx: Dict[int, int] = {
            43: 0,
            50: 1,
            51: 2,
            52: 3,
        }
        self.action_idx_to_name: Dict[int, str] = {
            0: "fall",
            1: "slap",
            2: "kick",
            3: "push",
        }

        # U-16 关键点数量（从 categories 里读取）
        self.num_keypoints: int = 0

        # clip 列表（每个 json 对应一个 clip）
        self.clips: List[Dict[str, Any]] = []

        self._build_index()

    # ---------------------------------------------------------------
    #   索引构建：按 videos[0].length 抽中心帧
    # ---------------------------------------------------------------
    def _build_index(self):
        ann_files = [f for f in os.listdir(self.ann_dir) if f.endswith(".json")]
        ann_files.sort()

        first_categories = None

        for ann_file in ann_files:
            ann_path = os.path.join(self.ann_dir, ann_file)

            with open(ann_path, "r") as f:
                dataset = json.load(f)

            images = dataset.get("images", [])
            annotations = dataset.get("annotations", [])
            videos = dataset.get("videos", [])
            categories = dataset.get("categories", [])

            # 第一次读取 categories，确定 U-16 关键点数
            if first_categories is None:
                first_categories = categories
                person_cat = next(
                    (c for c in categories if c.get("id") == 1 and "keypoints_u16" in c),
                    None,
                )
                if person_cat is not None:
                    self.num_keypoints = len(person_cat["keypoints_u16"])
                else:
                    self.num_keypoints = 0

                # 构建 COCO80 的类别名字（0-79）
                max_coco80 = max(self.coco91_to_coco80.values())
                num_classes = max_coco80 + 1  # 一般是 80

                names_80 = [""] * num_classes
                for cat in categories:
                    cid_91 = cat.get("id")
                    name = cat.get("name", "")
                    if cid_91 in self.coco91_to_coco80:
                        cid_80 = self.coco91_to_coco80[cid_91]
                        if 0 <= cid_80 < num_classes:
                            if not names_80[cid_80] and name:
                                names_80[cid_80] = name

                for i in range(num_classes):
                    if not names_80[i]:
                        names_80[i] = f"class_{i}"

                self.categories = {i: names_80[i] for i in range(num_classes)}

            # image_id -> image_info
            images_dict: Dict[int, Dict] = {img["id"]: img for img in images}

            # image_id -> anns
            anns_by_image: Dict[int, List[Dict]] = {}
            for ann in annotations:
                img_id = ann["image_id"]
                anns_by_image.setdefault(img_id, []).append(ann)

            # 找视频路径
            base = ann_file.replace("_merged.json", "")
            video_path = os.path.join(self.video_dir, base + "_rgb.avi")
            if not os.path.exists(video_path):
                alt = os.path.join(self.video_dir, base + ".avi")
                if os.path.exists(alt):
                    video_path = alt
                else:
                    print(f"[WARN] video not found for {ann_file}, skip.")
                    continue

            # video 元信息
            if len(videos) > 0:
                video_id = videos[0].get("id", base)
                video_meta = videos[0]
            else:
                video_id = base
                video_meta = {}

            # 视频总帧数 L
            length = video_meta.get("length", None)
            if length is None:
                frame_ids_all = [int(img.get("_frame_id", 0)) for img in images]
                if len(frame_ids_all) == 0:
                    print(f"[WARN] no frames in {ann_file}, skip.")
                    continue
                length = max(frame_ids_all) + 1
            L = int(length)
            if L <= 0:
                continue

            # frame_id -> image_id
            frame_to_image: Dict[int, int] = {}
            for img in images:
                img_id = img["id"]
                frame_id = int(img.get("_frame_id", 0))
                frame_to_image[frame_id] = img_id

            # 判断视频是否有至少一个有效标注
            if self.filter_empty and (not self._video_has_valid_ann(anns_by_image)):
                continue

            # 把 [0,L) 均分为 clip_len 段，取每段中间帧
            bounds = np.linspace(0, L, self.clip_len + 1, dtype=int)
            frame_indices: List[int] = []
            image_ids: List[int] = []

            existing_frame_ids_sorted = sorted(frame_to_image.keys())

            for i in range(self.clip_len):
                start = bounds[i]
                end = bounds[i + 1]
                if end <= start:
                    mid = min(start, L - 1)
                else:
                    mid = (start + end - 1) // 2
                    if mid >= L:
                        mid = L - 1

                if mid in frame_to_image:
                    img_id = frame_to_image[mid]
                else:
                    nearest_frame = min(existing_frame_ids_sorted, key=lambda x: abs(x - mid))
                    img_id = frame_to_image[nearest_frame]

                frame_indices.append(mid)
                image_ids.append(img_id)

            self.clips.append(
                {
                    "video_id": video_id,
                    "video_path": video_path,
                    "frame_indices": frame_indices,
                    "image_ids": image_ids,
                    "frame_to_image": frame_to_image,
                    "video_length": L,
                    "images": images_dict,
                    "anns_by_image": anns_by_image,
                    "video_meta": video_meta,
                }
            )

        print(f"[INFO] Built {len(self.clips)} video clips from {len(ann_files)} json files.")

    # ---------------------------------------------------------------
    #   工具函数
    # ---------------------------------------------------------------
    def _video_has_valid_ann(self, anns_by_image):
        for _img_id, anns in anns_by_image.items():
            for ann in anns:
                cat_id = ann.get("category_id")
                bbox = ann.get("bbox")
                if cat_id is None or bbox is None:
                    continue
                if cat_id not in self.valid_category_ids:
                    continue
                if not self._is_valid_box(bbox):
                    continue
                return True
        return False

    def _is_valid_box(self, bbox: List[float]):
        if not bbox or len(bbox) != 4:
            return False
        x, y, w, h = bbox
        if w <= 0 or h <= 0:
            return False
        if self.min_area is not None and (w * h) < self.min_area:
            return False
        return True

    def _decode_keypoints(
        self,
        kp_list: Optional[List[float]],
        image_w: int,
        image_h: int,
    ):
        """
        约定：
        - keypoints_u16: [x, y, v] * K
        - 这里 v 视为“深度/附加信息”，不再用于可见性判定
        - 可见性策略：
            * kp_list 合法（长度==3K） -> vis 全 1
            * kp_list 缺失/长度不对 -> vis 全 0
        """
        K = int(self.num_keypoints)

        # 保持返回形状稳定：即使 K==0 也返回 (0,3) 和 (0,)
        if K <= 0:
            return (
                np.zeros((0, 3), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
            )

        if kp_list is None or len(kp_list) != 3 * K:
            kps = np.zeros((K, 3), dtype=np.float32)
            vis = np.zeros((K,), dtype=np.float32)
            return kps, vis

        kps = np.array(kp_list, dtype=np.float32).reshape(K, 3)

        # clip x/y 到图像范围，v 不动
        kps[:, 0] = np.clip(kps[:, 0], 0, image_w - 1)
        kps[:, 1] = np.clip(kps[:, 1], 0, image_h - 1)

        # 不再使用 v 判可见：默认全部可见
        vis = np.ones((K,), dtype=np.float32)
        return kps, vis

    def _gather_and_fuse_annotations_for_frame(
        self,
        center_frame_idx: int,
        frame_to_image: Dict[int, int],
        anns_by_image: Dict[int, List[Dict]],
        video_length: int,
        image_w: int,
        image_h: int,
    ):
        """
        对某个中心帧 index：
          - 取中心帧及左右 neighbor_window 帧的标注
          - 对 bbox 做“查漏补缺” + NMS
          - 对 person：无论来自中心帧还是补帧，只要 ann 含 keypoints_u16，就解码并认为有效
          - keypoints_visible：不再依赖 v；kp_list 合法即全 1，否则全 0
        """
        offsets = [0]
        if self.use_neighbor_fusion:
            for d in range(1, self.neighbor_window + 1):
                offsets.extend([-d, d])

        existing_frames = sorted(frame_to_image.keys())
        used_img_ids = set()

        candidates_boxes: List[np.ndarray] = []
        candidates_labels: List[int] = []
        candidates_scores: List[float] = []
        candidates_kps: List[np.ndarray] = []
        candidates_vis: List[np.ndarray] = []
        candidates_person_ids: List[int] = []
        candidates_object_ids: List[int] = []

        for off in offsets:
            idx = center_frame_idx + off
            if idx < 0 or idx >= video_length:
                continue

            # 找最近的有标注帧
            if idx in frame_to_image:
                fid = idx
            else:
                if not existing_frames:
                    continue
                fid = min(existing_frames, key=lambda x: abs(x - idx))

            img_id_n = frame_to_image[fid]
            if img_id_n in used_img_ids:
                continue
            used_img_ids.add(img_id_n)

            anns = anns_by_image.get(img_id_n, [])
            is_center = (off == 0)
            base_score = 1.0 if is_center else 0.9

            for ann in anns:
                bbox = ann.get("bbox")
                cat_id = ann.get("category_id")
                if bbox is None or cat_id is None:
                    continue
                if cat_id not in self.valid_category_ids:
                    continue
                if not self._is_valid_box(bbox):
                    continue

                # xywh -> xyxy
                box_xyxy = torchvision.ops.box_convert(
                    torch.as_tensor(bbox, dtype=torch.float32),
                    "xywh", "xyxy",
                ).numpy()

                mapped_cls = self.coco91_to_coco80[cat_id]

                # person_global_id / object_global_id
                if cat_id == 1:
                    person_gid = int(ann.get("_person_global_id", -1))
                    object_gid = -1
                else:
                    person_gid = -1
                    object_gid = int(ann.get("_object_global_id", -1))

                # 关键点：对 person，不再限制 is_center；只要字段存在就解码
                if cat_id == 1:
                    kp_list = ann.get("keypoints_u16")
                    kps, vis = self._decode_keypoints(
                        kp_list, image_w=image_w, image_h=image_h
                    )
                else:
                    K = int(self.num_keypoints)
                    kps = np.zeros((K, 3), dtype=np.float32)
                    vis = np.zeros((K,), dtype=np.float32)

                candidates_boxes.append(box_xyxy)
                candidates_labels.append(mapped_cls)
                candidates_scores.append(base_score)
                candidates_kps.append(kps)
                candidates_vis.append(vis)
                candidates_person_ids.append(person_gid)
                candidates_object_ids.append(object_gid)

        # 没有候选
        if len(candidates_boxes) == 0:
            boxes_arr = np.zeros((0, 4), dtype=np.float32)
            classes_arr = np.zeros((0,), dtype=np.int64)
            kps_arr = np.zeros((0, self.num_keypoints, 3), dtype=np.float32)
            kps_vis_arr = np.zeros((0, self.num_keypoints), dtype=np.float32)
            person_ids_arr = np.zeros((0,), dtype=np.int64)
            object_ids_arr = np.zeros((0,), dtype=np.int64)
            return (
                boxes_arr, classes_arr,
                kps_arr, kps_vis_arr,
                person_ids_arr, object_ids_arr
            )

        boxes_np = np.stack(candidates_boxes, axis=0).astype(np.float32)   # [M,4]
        labels_np = np.array(candidates_labels, dtype=np.int64)            # [M]
        scores_np = np.array(candidates_scores, dtype=np.float32)          # [M]
        kps_np = np.stack(candidates_kps, axis=0).astype(np.float32)       # [M,K,3]
        vis_np = np.stack(candidates_vis, axis=0).astype(np.float32)       # [M,K]
        person_ids_np = np.array(candidates_person_ids, dtype=np.int64)    # [M]
        object_ids_np = np.array(candidates_object_ids, dtype=np.int64)    # [M]

        # 按类别做 NMS
        keep_indices: List[int] = []
        unique_classes = np.unique(labels_np)
        for cls in unique_classes:
            cls_mask = (labels_np == cls)
            idxs = np.where(cls_mask)[0]
            if len(idxs) == 0:
                continue

            boxes_t = torch.from_numpy(boxes_np[idxs])
            scores_t = torch.from_numpy(scores_np[idxs])
            keep_rel = torchvision.ops.nms(boxes_t, scores_t, self.nms_iou_threshold)
            keep_global = idxs[keep_rel.numpy()]
            keep_indices.extend(list(keep_global))

        if len(keep_indices) == 0:
            boxes_arr = np.zeros((0, 4), dtype=np.float32)
            classes_arr = np.zeros((0,), dtype=np.int64)
            kps_arr = np.zeros((0, self.num_keypoints, 3), dtype=np.float32)
            kps_vis_arr = np.zeros((0, self.num_keypoints), dtype=np.float32)
            person_ids_arr = np.zeros((0,), dtype=np.int64)
            object_ids_arr = np.zeros((0,), dtype=np.int64)
            return (
                boxes_arr, classes_arr,
                kps_arr, kps_vis_arr,
                person_ids_arr, object_ids_arr
            )

        keep_indices = sorted(set(keep_indices))
        boxes_arr = boxes_np[keep_indices]
        classes_arr = labels_np[keep_indices]
        kps_arr = kps_np[keep_indices]
        kps_vis_arr = vis_np[keep_indices]
        person_ids_arr = person_ids_np[keep_indices]
        object_ids_arr = object_ids_np[keep_indices]

        # （可选）按 person_ids / object_ids 排序：人会优先被排到前面
        if boxes_arr.shape[0] > 0:
            sort_keys = np.stack([person_ids_arr, object_ids_arr], axis=1)  # [N,2]
            sort_idx = np.lexsort(sort_keys.T)  # 先按 object_ids，再按 person_ids
            boxes_arr = boxes_arr[sort_idx]
            classes_arr = classes_arr[sort_idx]
            kps_arr = kps_arr[sort_idx]
            kps_vis_arr = kps_vis_arr[sort_idx]
            person_ids_arr = person_ids_arr[sort_idx]
            object_ids_arr = object_ids_arr[sort_idx]

        return (
            boxes_arr, classes_arr,
            kps_arr, kps_vis_arr,
            person_ids_arr, object_ids_arr
        )

    # ---------------------------------------------------------------
    #   Dataset 接口
    # ---------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        clip = self.clips[index]
        video_path = clip["video_path"]
        video_id = clip["video_id"]
        frame_indices = clip["frame_indices"]      # 中心帧索引（长度=clip_len）
        image_ids_center = clip["image_ids"]
        frame_to_image = clip["frame_to_image"]
        video_length = clip["video_length"]
        anns_by_image = clip["anns_by_image"]
        video_meta = clip["video_meta"]

        if len(frame_indices) == 0:
            raise RuntimeError(
                f"[VideoClipDataset_kp_d] clip {index} has EMPTY frame_indices. "
                f"video_id={video_id}, video_path={video_path}"
            )

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(
                f"[VideoClipDataset_kp_d] Cannot open video: {video_path} "
                f"(clip index={index}, video_id={video_id})"
            )

        # 预取视频属性尺寸（有时不可靠，但可作为兜底）
        prop_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        prop_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        prop_w = int(prop_w) if isinstance(prop_w, (int, float)) and prop_w > 0 and not math.isnan(prop_w) else -1
        prop_h = int(prop_h) if isinstance(prop_h, (int, float)) and prop_h > 0 and not math.isnan(prop_h) else -1

        # frame_count 软读取（可能为 0 / NaN / 不准）
        fc = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        frame_count = int(fc) if isinstance(fc, (int, float)) and fc > 0 and not math.isnan(fc) else -1

        images_list: List[Optional[np.ndarray]] = []
        boxes_list: List[np.ndarray] = []
        class_ids_list: List[np.ndarray] = []
        keypoints_list: List[np.ndarray] = []
        keypoints_vis_list: List[np.ndarray] = []
        person_ids_list: List[np.ndarray] = []
        object_ids_list: List[np.ndarray] = []
        valid_mask_list: List[float] = []

        target_H: Optional[int] = None
        target_W: Optional[int] = None

        def _empty_ann():
            boxes_arr = np.zeros((0, 4), dtype=np.float32)
            classes_arr = np.zeros((0,), dtype=np.int64)
            kps_arr = np.zeros((0, self.num_keypoints, 3), dtype=np.float32)
            kps_vis_arr = np.zeros((0, self.num_keypoints), dtype=np.float32)
            person_ids_arr = np.zeros((0,), dtype=np.int64)
            object_ids_arr = np.zeros((0,), dtype=np.int64)
            return boxes_arr, classes_arr, kps_arr, kps_vis_arr, person_ids_arr, object_ids_arr

        try:
            for center_idx, _img_id_center in zip(frame_indices, image_ids_center):
                decoded = False

                if center_idx < 0:
                    print(
                        f"[WARN] clip {index} video_id={video_id}: "
                        f"center_idx={center_idx} < 0 for video {video_path}, will pad."
                    )
                else:
                    if frame_count > 0 and center_idx >= frame_count:
                        print(
                            f"[WARN] clip {index} video_id={video_id}: "
                            f"center_idx={center_idx} >= CAP_PROP_FRAME_COUNT={frame_count}, "
                            f"but CAP_PROP_FRAME_COUNT may be inaccurate; will still try to read."
                        )

                    cap.set(cv2.CAP_PROP_POS_FRAMES, center_idx)
                    ok, frame_bgr = cap.read()

                    if ok and frame_bgr is not None:
                        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

                        # 原始尺寸（标注像素坐标系）
                        orig_h, orig_w = frame_rgb.shape[:2]

                        # 统一输出尺寸（以第一帧成功解码的尺寸为准；否则用 prop_h/prop_w 兜底）
                        if target_H is None or target_W is None:
                            target_H, target_W = orig_h, orig_w
                        scale_x = float(target_W) / float(orig_w)
                        scale_y = float(target_H) / float(orig_h)

                        if (orig_h, orig_w) != (target_H, target_W):
                            print(
                                f"[WARN] clip {index} video_id={video_id}: "
                                f"frame size changed from {(target_H, target_W)} to {(orig_h, orig_w)}. "
                                f"resize to {(target_H, target_W)}."
                            )
                            frame_rgb = cv2.resize(frame_rgb, (target_W, target_H), interpolation=cv2.INTER_LINEAR)

                        # 收集并融合标注：必须传“原始尺寸”，因为 JSON bbox/kp 都是原始像素坐标
                        boxes_arr, classes_arr, kps_arr, kps_vis_arr, person_ids_arr, object_ids_arr = \
                            self._gather_and_fuse_annotations_for_frame(
                                center_frame_idx=center_idx,
                                frame_to_image=frame_to_image,
                                anns_by_image=anns_by_image,
                                video_length=video_length,
                                image_w=orig_w,
                                image_h=orig_h,
                            )

                        # 若发生 resize：缩放 bbox / keypoints 到 target 尺寸坐标系
                        if (scale_x != 1.0) or (scale_y != 1.0):
                            if boxes_arr is not None and boxes_arr.shape[0] > 0:
                                boxes_arr = boxes_arr.astype(np.float32, copy=True)
                                boxes_arr[:, [0, 2]] *= scale_x
                                boxes_arr[:, [1, 3]] *= scale_y
                                boxes_arr[:, 0] = np.clip(boxes_arr[:, 0], 0, target_W - 1)
                                boxes_arr[:, 2] = np.clip(boxes_arr[:, 2], 0, target_W - 1)
                                boxes_arr[:, 1] = np.clip(boxes_arr[:, 1], 0, target_H - 1)
                                boxes_arr[:, 3] = np.clip(boxes_arr[:, 3], 0, target_H - 1)

                            if kps_arr is not None and kps_arr.shape[0] > 0 and kps_arr.ndim == 3:
                                kps_arr = kps_arr.astype(np.float32, copy=True)
                                kps_arr[..., 0] *= scale_x
                                kps_arr[..., 1] *= scale_y
                                kps_arr[..., 0] = np.clip(kps_arr[..., 0], 0, target_W - 1)
                                kps_arr[..., 1] = np.clip(kps_arr[..., 1], 0, target_H - 1)

                        images_list.append(frame_rgb)
                        boxes_list.append(boxes_arr)
                        class_ids_list.append(classes_arr)
                        keypoints_list.append(kps_arr)
                        keypoints_vis_list.append(kps_vis_arr)
                        person_ids_list.append(person_ids_arr)
                        object_ids_list.append(object_ids_arr)
                        valid_mask_list.append(1.0)
                        decoded = True
                    else:
                        print(
                            f"[WARN] clip {index} video_id={video_id}: "
                            f"failed to read frame {center_idx} from {video_path}, will pad."
                        )

                if not decoded:
                    if target_H is None or target_W is None:
                        if prop_h > 0 and prop_w > 0:
                            target_H, target_W = prop_h, prop_w

                    images_list.append(None)
                    boxes_arr, classes_arr, kps_arr, kps_vis_arr, person_ids_arr, object_ids_arr = _empty_ann()
                    boxes_list.append(boxes_arr)
                    class_ids_list.append(classes_arr)
                    keypoints_list.append(kps_arr)
                    keypoints_vis_list.append(kps_vis_arr)
                    person_ids_list.append(person_ids_arr)
                    object_ids_list.append(object_ids_arr)
                    valid_mask_list.append(0.0)

        finally:
            cap.release()

        if (target_H is None) or (target_W is None) or all(v == 0.0 for v in valid_mask_list):
            raise RuntimeError(
                f"[VideoClipDataset_kp_d] No frames decoded for clip {index}: "
                f"video_id={video_id}, video_path={video_path}, "
                f"frame_indices={frame_indices}, frame_count={frame_count}"
            )

        for i in range(len(images_list)):
            if images_list[i] is None:
                images_list[i] = np.zeros((target_H, target_W, 3), dtype=np.uint8)

        images = np.stack(images_list, axis=0)  # [T,H,W,3]
        image_hw = (target_H, target_W)
        frame_valid_mask = np.asarray(valid_mask_list, dtype=np.float32)

        # ==== 计算 action_id / action_name ====
        raw_action_id = video_meta.get("action_id", None)
        if raw_action_id is not None:
            try:
                raw_action_id_int = int(raw_action_id)
            except Exception:
                raw_action_id_int = None
        else:
            raw_action_id_int = None

        if raw_action_id_int is not None and raw_action_id_int in self.action_id_to_idx:
            action_id_4 = self.action_id_to_idx[raw_action_id_int]
            action_name = self.action_idx_to_name.get(action_id_4, None)
        else:
            action_id_4 = None
            action_name = None

        # 计算 T（clip 帧数）
        if torch.is_tensor(frame_indices):
            T = int(frame_indices.shape[-1])
            frame_ids_rel = torch.arange(T, device=frame_indices.device, dtype=torch.long)
        elif isinstance(frame_indices, np.ndarray):
            T = int(frame_indices.shape[-1])
            frame_ids_rel = np.arange(T, dtype=np.int64)
        else:
            T = len(frame_indices)
            frame_ids_rel = list(range(T))

        sample = {
            "images": images,
            "boxes": boxes_list,
            "class_ids": class_ids_list,
            "keypoints": keypoints_list,
            "keypoints_visible": keypoints_vis_list,
            "person_ids": person_ids_list,
            "object_ids": object_ids_list,
            "video_id": video_id,
            "frame_ids": frame_indices,
            "frame_ids_rel": frame_ids_rel,
            "image_ids": image_ids_center,
            "image_hw": image_hw,
            "frame_valid_mask": frame_valid_mask,

            "action_id": action_id_4,
            "raw_action_id": raw_action_id_int,
            "action_name": action_name,
        }
        return sample
