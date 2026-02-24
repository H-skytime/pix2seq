# infer_video_make_mp4.py
import os
import re
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union

import numpy as np
import torch
from torch.utils.data import DataLoader

# 重要：matplotlib 必须在 pyplot 之前设置 Agg（无显示环境也可保存/渲染）
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import cv2

from func_to_script import load_config_from_yaml, script

from model.model import Pix2SeqModel
from model.modelv2 import LlamaPix2Seq

from data.base_dataset_video import VideoClipDataset_kp_d
from data.dataset_video import Pix2SeqVideoDataset_kp_d, Pix2SeqVideoCollator_kp_d
from data.tokenizer_video import VideoTokenProcessor

from model.inference_kp_d import TaskSequenceGenerator


FILE_PATH = Path(__file__).resolve().parent


# ============================================================
# 你补充的画图文件核心（只保留“本脚本需要用到的部分”）
# ============================================================

BASE_VOCAB_SHIFT = 10


def _get_category_name(category_names, label_id):
    """
    统一从 category_names 中取出类别名，兼容多种格式:
      1) dict: {0: "person", 1: "bicycle", ...}
      2) list[str]: ["person", "bicycle", ...]
      3) list[dict]: [{"id":0,"name":"person"}, ...]
    """
    if category_names is None:
        return str(label_id)

    if isinstance(category_names, dict):
        cat = category_names.get(label_id, None)
        if cat is None:
            return str(label_id)
        if isinstance(cat, dict):
            return str(cat.get("name", label_id))
        return str(cat)

    if isinstance(category_names, (list, tuple)):
        if not (0 <= label_id < len(category_names)):
            return str(label_id)
        cat = category_names[label_id]
        if isinstance(cat, dict):
            return str(cat.get("name", label_id))
        return str(cat)

    return str(label_id)


# ===================== U-16 骨架定义 =====================
U16_SKELETON: List[Tuple[int, int]] = [
    (0, 1), (1, 2), (2, 3),
    (1, 4), (1, 5),
    (4, 6), (6, 8),
    (5, 7), (7, 9),
    (3, 10), (3, 11),
    (10, 12), (12, 14),
    (11, 13), (13, 15),
]


def show_image_with_boxes(
    image: Union[np.ndarray, torch.Tensor],
    boxes: Union[np.ndarray, torch.Tensor] = None,
    labels: Union[np.ndarray, torch.Tensor] = None,
    title: Optional[str] = None,
    category_names: Optional[Dict] = None,
    ax: Optional[plt.Axes] = None,
    normalized_boxes: bool = False,
    box_color: Optional[str] = None,
    label_prefix: Optional[str] = None,
    real_noise_coloring: bool = False,
    instance_ids: Union[np.ndarray, torch.Tensor, None] = None,  # 同一 tube 同色
) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(12, 12))

    def to_numpy(x):
        if x is None:
            return None
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return x

    image = to_numpy(image)
    boxes = to_numpy(boxes)
    labels = to_numpy(labels)
    instance_ids = to_numpy(instance_ids)

    if image.shape[0] == 3:  # CHW
        image = np.transpose(image, (1, 2, 0))
    if image.dtype in [np.float32, np.float64] and image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)

    ax.imshow(image)

    num_classes = 80
    num_colors = 50
    colors_for_labels = plt.cm.rainbow(np.linspace(0, 1, num_classes))
    colors_for_instances = plt.cm.tab20(np.linspace(0, 1, num_colors))

    if boxes is not None and len(boxes) > 0:
        height, width = image.shape[:2]
        boxes = boxes.copy()

        if normalized_boxes:
            boxes[:, [0, 2]] *= width
            boxes[:, [1, 3]] *= height

        for i, (xmin, ymin, xmax, ymax) in enumerate(boxes):
            if xmax <= xmin or ymax <= ymin:
                continue

            # 颜色选择逻辑
            if box_color is not None:
                color = box_color
            elif real_noise_coloring and labels is not None:
                label_id = int(labels[i])
                is_real = label_id != num_classes
                color = "green" if is_real else "red"
            elif instance_ids is not None:
                inst_id = int(instance_ids[i]) if i < len(instance_ids) else 0
                color = colors_for_instances[inst_id % num_colors]
            elif labels is not None:
                label_id = int(labels[i])
                color = colors_for_labels[label_id % num_classes]
            else:
                color = "red"

            rect = plt.Rectangle(
                (xmin, ymin),
                xmax - xmin,
                ymax - ymin,
                fill=False,
                linewidth=2,
                edgecolor=color,
            )
            ax.add_patch(rect)

            # 文本标签
            if labels is not None:
                label_id = int(labels[i])
                if real_noise_coloring:
                    label_text = "Real" if label_id < num_classes else "Fake"
                elif category_names and (label_id in category_names or isinstance(category_names, (list, tuple))):
                    label_text = _get_category_name(category_names, int(label_id))
                else:
                    label_text = f"Class {label_id}"
                if label_prefix:
                    label_text = f"{label_prefix} {label_text}"
            else:
                label_text = f"{label_prefix or ''} Box {i}"

            ax.text(
                xmin,
                ymin - 2,
                label_text.strip(),
                color=color,
                fontsize=10,
                bbox=dict(facecolor="white", alpha=0.8),
            )

    if title:
        ax.set_title(title, fontsize=12, pad=10)
    ax.axis("off")
    return ax


def show_image_with_keypoints(
    image: Union[np.ndarray, torch.Tensor],
    keypoints: Union[np.ndarray, torch.Tensor],
    *,
    normalized: bool = True,
    num_keypoints: int = 16,
    draw_skeleton: bool = True,
    ax: Optional[plt.Axes] = None,
    point_size: int = 16,
    point_color: str = "yellow",
    line_color: str = "yellow",
    vis_threshold: float = 0.5,
):
    """
    U-16 可视化（支持 v 可见性）：
      keypoints:
        - [K,2]/[K,3] 或 [N,K,2]/[N,K,3]
      若第三维存在，视为 (x,y,v)，严格使用 v 控制点/线绘制。
    """
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(6, 6))

    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    if isinstance(keypoints, torch.Tensor):
        keypoints = keypoints.detach().cpu().numpy()

    if image.ndim == 3 and image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))

    if image.dtype in (np.float32, np.float64) and image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)

    ax.imshow(image)
    h, w = image.shape[:2]

    if keypoints.ndim == 2:
        keypoints = keypoints[None, ...]
    elif keypoints.ndim != 3:
        raise ValueError(f"keypoints 维度应为 2 或 3，当前为 {keypoints.shape}")

    C = keypoints.shape[-1]
    if C < 2:
        raise ValueError(f"keypoints 最后一维 < 2，形状为 {keypoints.shape}")

    keypoints_xy = keypoints[..., :2]

    if C >= 3:
        keypoints_v = keypoints[..., 2]
        visible_mask = keypoints_v > vis_threshold
    else:
        visible_mask = np.ones(keypoints_xy.shape[:-1], dtype=bool)

    for person_xy, person_vis in zip(keypoints_xy, visible_mask):
        person_xy = person_xy[:num_keypoints]
        person_vis = person_vis[:num_keypoints]

        finite = np.isfinite(person_xy).all(axis=-1)
        person_vis = person_vis & finite

        if normalized:
            in_range = (
                (person_xy[:, 0] >= 0.0) & (person_xy[:, 0] <= 1.0) &
                (person_xy[:, 1] >= 0.0) & (person_xy[:, 1] <= 1.0)
            )
            person_vis = person_vis & in_range

        if person_vis.sum() == 0:
            continue

        if normalized:
            kps_px = person_xy.copy()
            kps_px[:, 0] *= w
            kps_px[:, 1] *= h
        else:
            kps_px = person_xy

        idxs = np.where(person_vis)[0]
        for i in idxs:
            x, y = kps_px[i]
            ax.scatter(
                [x], [y],
                s=point_size,
                c=point_color,
                edgecolors="black",
                linewidths=0.5,
                zorder=5,
            )

        if draw_skeleton:
            for (p1, p2) in U16_SKELETON:
                if p1 >= len(kps_px) or p2 >= len(kps_px):
                    continue
                if not (person_vis[p1] and person_vis[p2]):
                    continue
                x1, y1 = kps_px[p1]
                x2, y2 = kps_px[p2]
                ax.plot([x1, x2], [y1, y2], color=line_color, linewidth=1.5, zorder=4)

    ax.axis("off")
    return ax


# ============================================================
# run.py 里已有的 build_token_processor / build_model
# ============================================================

def build_token_processor(config):
    tube_frames = getattr(config.data, "tube_frames", 8)
    max_tubes = getattr(config.data, "max_tubes_per_clip", 30)
    num_keypoints = getattr(config.data, "num_keypoints", 16)
    persons_per_clip = getattr(config.data, "persons_per_clip", 2)

    det_tokens_per_tube = tube_frames * 4 + 1
    det_total_seq_len = max_tubes * det_tokens_per_tube + 2

    act_total_seq_len = persons_per_clip * tube_frames * (4 + 2 * num_keypoints) + 3
    max_seq_len = max(det_total_seq_len, act_total_seq_len)

    return VideoTokenProcessor(
        max_seq_len=max_seq_len,
        num_classes=config.data.num_classes,
        num_actions=getattr(config.data, "num_actions", 5),
        num_keypoints=num_keypoints,
        quantization_bins=config.tokenization.quantization_bins,
        num_frame_bins=tube_frames,
        tube_frames=tube_frames,
        bos_eos_token_weight=config.tokenization.eos_token_weight,
        verbose=False,
    )


def build_model(config, token_processor):
    model_cls = Pix2SeqModel if not config.model.llama_model else LlamaPix2Seq
    model = model_cls(
        image_size=config.data.image_size,
        patch_size=config.model.patch_size,
        num_encoder_layers=config.model.num_encoder_layers,
        num_decoder_layers=config.model.num_decoder_layers,
        embedding_dim=config.model.d_model,
        num_heads=config.model.nhead,
        dim_feedforward=config.model.dim_feedforward,
        dropout=config.model.dropout,
        drop_path=config.model.drop_path,
        shared_decoder_embedding=config.model.shared_decoder_embedding,
        decoder_output_bias=config.model.decoder_output_bias,
        eos_token_id=token_processor.EOS_TOKEN,
        bos_token_id=token_processor.BOS_TOKEN,
        coord_vocab_shift=token_processor.coord_vocab_shift,
        base_vocab_shift=token_processor.BASE_CLASS_SHIFT,
        num_quantization_bins=token_processor.quantization_bins,
        max_seq_len=token_processor.max_seq_len,
        token_processor=token_processor,
    )
    return model


def _extract_state_dict(ckpt: Any) -> Dict[str, torch.Tensor]:
    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt and isinstance(ckpt["model_state_dict"], dict):
            return ckpt["model_state_dict"]
        if "model" in ckpt and isinstance(ckpt["model"], dict):
            return ckpt["model"]
        if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            return ckpt["state_dict"]
        if all(isinstance(k, str) for k in ckpt.keys()):
            return ckpt
    raise ValueError("Checkpoint format not recognized (cannot extract model state_dict).")


# ============================================================
# Tube NMS（与你 run.py 保持一致）
# ============================================================

def _box_iou_xyxy(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    ax1, ay1, ax2, ay2 = a.unbind(-1)
    bx1, by1, bx2, by2 = b.unbind(-1)

    inter_x1 = torch.maximum(ax1, bx1)
    inter_y1 = torch.maximum(ay1, by1)
    inter_x2 = torch.minimum(ax2, bx2)
    inter_y2 = torch.minimum(ay2, by2)

    inter_w = torch.clamp(inter_x2 - inter_x1, min=0.0)
    inter_h = torch.clamp(inter_y2 - inter_y1, min=0.0)
    inter = inter_w * inter_h

    area_a = torch.clamp(ax2 - ax1, min=0.0) * torch.clamp(ay2 - ay1, min=0.0)
    area_b = torch.clamp(bx2 - bx1, min=0.0) * torch.clamp(by2 - by1, min=0.0)
    union = area_a + area_b - inter
    return inter / (union + eps)


def _tube_duplicate_ratio(tube_a: torch.Tensor, tube_b: torch.Tensor, iou_thr: float = 0.8):
    assert tube_a.ndim == 2 and tube_b.ndim == 2 and tube_a.shape == tube_b.shape
    T = tube_a.shape[0]
    cnt = 0
    for t in range(T):
        iou = _box_iou_xyxy(tube_a[t], tube_b[t]).item()
        if iou >= iou_thr:
            cnt += 1
    return cnt / max(T, 1)


def tube_nms_by_duplicate_ratio(
    tube_boxes: torch.Tensor,
    tube_labels: torch.Tensor,
    tube_scores: torch.Tensor,
    *,
    dup_iou_thr: float = 0.8,
    dup_ratio_thr: float = 0.8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[int]]:
    if tube_boxes.numel() == 0:
        return tube_boxes, tube_labels, tube_scores, []

    device = tube_boxes.device
    order = torch.argsort(tube_scores, descending=True)
    kept: List[int] = []

    for idx in order.tolist():
        lab = int(tube_labels[idx].item())
        keep_flag = True
        for kept_idx in kept:
            if int(tube_labels[kept_idx].item()) != lab:
                continue
            dup_ratio = _tube_duplicate_ratio(tube_boxes[idx], tube_boxes[kept_idx], iou_thr=dup_iou_thr)
            if dup_ratio >= dup_ratio_thr:
                keep_flag = False
                break
        if keep_flag:
            kept.append(idx)

    kept_tensor = torch.tensor(kept, dtype=torch.long, device=device)
    return (
        tube_boxes.index_select(0, kept_tensor),
        tube_labels.index_select(0, kept_tensor),
        tube_scores.index_select(0, kept_tensor),
        kept,
    )


# ============================================================
# 新增：将“8帧可视化结果”写成 mp4 的工具函数
# ============================================================

def _safe_stem(s: str) -> str:
    s = str(s)
    s = s.replace(os.sep, "_")
    s = re.sub(r"[^0-9a-zA-Z_\-\.]+", "_", s)
    return s[:200] if len(s) > 200 else s


def _time_stretch_indices(T: int, fps: int, target_seconds: float) -> List[int]:
    total_frames = max(1, int(round(float(target_seconds) * float(fps))))
    if total_frames <= T:
        return list(range(total_frames))
    idx = np.floor(np.linspace(0, T, total_frames, endpoint=False)).astype(np.int64)
    idx = np.clip(idx, 0, T - 1).tolist()
    return idx


def _mpl_fig_to_bgr(fig: plt.Figure) -> np.ndarray:
    """
    Matplotlib Figure -> OpenCV BGR ndarray
    兼容不同 matplotlib 版本：
      - 优先使用 buffer_rgba()（较新版本）
      - 其次使用 tostring_rgb()
      - 再次使用 tostring_argb()（你当前环境可用）
    """
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()

    # 1) 新版本：buffer_rgba -> RGBA
    if hasattr(fig.canvas, "buffer_rgba"):
        rgba = np.asarray(fig.canvas.buffer_rgba())
        rgba = np.ascontiguousarray(rgba)  # cv2 需要连续内存
        bgr = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
        return bgr

    # 2) 旧版本：tostring_rgb -> RGB
    if hasattr(fig.canvas, "tostring_rgb"):
        rgb = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
        rgb = np.ascontiguousarray(rgb)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        return bgr

    # 3) 你当前环境：tostring_argb -> ARGB，需要转成 RGBA
    if hasattr(fig.canvas, "tostring_argb"):
        argb = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8).reshape(h, w, 4)
        argb = np.ascontiguousarray(argb)
        rgba = argb[:, :, [1, 2, 3, 0]]  # ARGB -> RGBA
        bgr = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
        return bgr

    raise AttributeError("FigureCanvas does not support RGB/ARGB extraction methods.")



def _try_make_videowriter(path: str, fps: int, size_wh: Tuple[int, int]) -> cv2.VideoWriter:
    W, H = size_wh
    candidates = ["mp4v", "avc1", "H264"]
    for c in candidates:
        fourcc = cv2.VideoWriter_fourcc(*c)
        vw = cv2.VideoWriter(path, fourcc, float(fps), (W, H))
        if vw.isOpened():
            return vw
    return cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (W, H))


def write_mp4_from_frames(
    out_path: str,
    frames_bgr: List[np.ndarray],
    *,
    fps: int,
    target_seconds: float,
):
    if len(frames_bgr) == 0:
        return
    H, W = frames_bgr[0].shape[:2]
    idxs = _time_stretch_indices(T=len(frames_bgr), fps=fps, target_seconds=target_seconds)

    vw = _try_make_videowriter(out_path, fps=fps, size_wh=(W, H))
    if not vw.isOpened():
        raise RuntimeError(f"cv2.VideoWriter open failed for: {out_path}")

    for i in idxs:
        frame = frames_bgr[i]
        if frame.shape[:2] != (H, W):
            frame = cv2.resize(frame, (W, H), interpolation=cv2.INTER_LINEAR)
        vw.write(frame)

    vw.release()


# ============================================================
# 新增：基于 tube_predictions 构造逐帧 boxes/labels/scores/instance_ids（复用你callback逻辑）
# ============================================================

def _group_tube_predictions_by_video(
    tube_predictions: Optional[Union[List[Dict[str, Any]], Dict[str, Any]]]
) -> Dict[Any, List[Dict[str, Any]]]:
    pred_by_vid: Dict[Any, List[Dict[str, Any]]] = {}
    if tube_predictions is None:
        return pred_by_vid
    if isinstance(tube_predictions, dict):
        tube_predictions = [tube_predictions]
    for p in tube_predictions:
        vid = p.get("video_id", None)
        if vid is None:
            continue
        pred_by_vid.setdefault(str(vid), []).append(p)
    return pred_by_vid


def _build_framewise_preds_for_clip(
    T: int,
    clip_video_id: Any,
    pred_by_vid: Dict[Any, List[Dict[str, Any]]],
):
    frame_boxes:  List[List[np.ndarray]] = [[] for _ in range(T)]
    frame_labels: List[List[int]]       = [[] for _ in range(T)]
    frame_scores: List[List[float]]     = [[] for _ in range(T)]
    frame_inst:   List[List[int]]       = [[] for _ in range(T)]

    tubes = pred_by_vid.get(str(clip_video_id), [])
    for tube_idx, tube in enumerate(tubes):
        if "boxes" not in tube:
            continue
        cid = int(tube.get("category_id", -1))
        score = float(tube.get("score", 0.0))
        boxes = np.asarray(tube["boxes"], dtype=np.float32)  # [T_pred,4]
        if boxes.ndim != 2 or boxes.shape[1] != 4:
            continue
        T_pred = boxes.shape[0]
        T_use = min(T, T_pred)

        for t in range(T_use):
            frame_boxes[t].append(boxes[t])
            frame_labels[t].append(cid)
            frame_scores[t].append(score)
            frame_inst[t].append(tube_idx)

    out_boxes, out_labels, out_scores, out_inst = [], [], [], []
    for t in range(T):
        if len(frame_boxes[t]) == 0:
            out_boxes.append(torch.empty((0, 4), dtype=torch.float32))
            out_labels.append(torch.empty((0,), dtype=torch.long))
            out_scores.append(torch.empty((0,), dtype=torch.float32))
            out_inst.append(torch.empty((0,), dtype=torch.long))
        else:
            out_boxes.append(torch.from_numpy(np.stack(frame_boxes[t], axis=0)))
            out_labels.append(torch.from_numpy(np.asarray(frame_labels[t], dtype=np.int64)))
            out_scores.append(torch.from_numpy(np.asarray(frame_scores[t], dtype=np.float32)))
            out_inst.append(torch.from_numpy(np.asarray(frame_inst[t], dtype=np.int64)))

    return out_boxes, out_labels, out_scores, out_inst


# ============================================================
# 新增：GT / Pred keypoints 逐帧提取（与你 callback 逻辑对齐）
# ============================================================

def _safe_get_listlike(x, b: int):
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x[b]
    if isinstance(x, (list, tuple)):
        return x[b]
    return x


def _extract_gt_keypoints_per_frame(
    b: int,
    T: int,
    batch: Dict[str, Any],
    num_keypoints: int,
    tube_valid_mask_b: Optional[torch.Tensor],
) -> List[Optional[torch.Tensor]]:
    gt_kps_per_frame: List[Optional[torch.Tensor]] = [None] * T

    tube_keypoints = batch.get("tube_keypoints", None)          # [B,N,T,K,2]
    act_keypoints_traj = batch.get("act_keypoints_traj", None)  # [B,P,T,K,C]

    if isinstance(tube_keypoints, torch.Tensor):
        kps_b = tube_keypoints[b]  # [N,T,K,2]
        if kps_b.dim() == 4 and kps_b.size(-1) >= 2:
            N, T_kp, K, _ = kps_b.shape
            T_use = min(T, T_kp)
            for t in range(T_use):
                if tube_valid_mask_b is not None and tube_valid_mask_b.dim() == 2:
                    valid_idx = [n for n in range(N) if bool(tube_valid_mask_b[n, t].item())]
                else:
                    valid_idx = list(range(N))
                if len(valid_idx) == 0:
                    gt_kps_per_frame[t] = torch.empty((0, num_keypoints, 2), dtype=torch.float32)
                else:
                    gt_kps_per_frame[t] = kps_b[valid_idx, t, :num_keypoints, :2]
            return gt_kps_per_frame

    if isinstance(act_keypoints_traj, torch.Tensor):
        kps_b = act_keypoints_traj[b]  # [P,T,K,C]
        if kps_b.dim() == 4 and kps_b.size(-1) >= 2:
            P, T_kp, K, C = kps_b.shape
            T_use = min(T, T_kp)
            for t in range(T_use):
                gt_kps_per_frame[t] = kps_b[:, t, :num_keypoints, :2]
            return gt_kps_per_frame

    return gt_kps_per_frame


def _extract_pred_kp_keypoints_per_frame(
    b: int,
    T: int,
    batch: Dict[str, Any],
    batch_output: Dict[str, Any],
    num_keypoints: int,
) -> List[Optional[torch.Tensor]]:
    pred_kps_per_frame: List[Optional[torch.Tensor]] = [None] * T

    # A) clip 级直接输出
    for key in ("pred_kp_keypoints", "kp_pred_keypoints", "pred_keypoints", "kp_keypoints_pred"):
        if key in batch_output and isinstance(batch_output[key], torch.Tensor):
            k = batch_output[key]
            if k.dim() == 5:
                kb = k[b]
                if kb.dim() != 4:
                    break
                if kb.shape[1] == T:
                    P, T_kp, K, C = kb.shape
                    T_use = min(T, T_kp)
                    for t in range(T_use):
                        pred_kps_per_frame[t] = kb[:, t, :num_keypoints, :2]
                    return pred_kps_per_frame
                if kb.shape[0] == T:
                    T_kp, P, K, C = kb.shape
                    T_use = min(T, T_kp)
                    for t in range(T_use):
                        pred_kps_per_frame[t] = kb[t, :, :num_keypoints, :2]
                    return pred_kps_per_frame

            if k.dim() == 4:
                kb = k[b]
                if kb.shape[0] == T:
                    T_kp, K, C = kb.shape
                    T_use = min(T, T_kp)
                    for t in range(T_use):
                        pred_kps_per_frame[t] = kb[t, :num_keypoints, :2].unsqueeze(0)
                    return pred_kps_per_frame

    # B) 扁平化输出
    flat = None
    for key in ("kp_pred_keypoints_flat", "pred_kp_keypoints_flat", "kp_keypoints_pred_flat"):
        if key in batch_output and isinstance(batch_output[key], torch.Tensor):
            flat = batch_output[key]
            break

    if flat is None or flat.dim() < 3:
        return pred_kps_per_frame
    if flat.dim() != 4:
        return pred_kps_per_frame
    if flat.size(-1) < 2:
        return pred_kps_per_frame

    row_to_b = None
    for key in ("kp_row_to_clip_index", "kp_row_to_bidx", "kp_row_to_batch_index"):
        if key in batch_output and isinstance(batch_output[key], torch.Tensor):
            row_to_b = batch_output[key]
            break
    if row_to_b is None:
        return pred_kps_per_frame

    row_to_pid = None
    for key in ("kp_row_to_person_index", "kp_row_to_pid", "kp_row_to_person_id"):
        if key in batch_output and isinstance(batch_output[key], torch.Tensor):
            row_to_pid = batch_output[key]
            break

    row_to_b = row_to_b.to(flat.device).long()
    select_rows = (row_to_b == int(b)).nonzero(as_tuple=True)[0]
    if select_rows.numel() == 0:
        return pred_kps_per_frame

    pid_to_row: Dict[int, int] = {}
    if row_to_pid is not None:
        row_to_pid = row_to_pid.to(flat.device).long()
        for r in select_rows.tolist():
            pid = int(row_to_pid[r].item())
            if pid not in pid_to_row:
                pid_to_row[pid] = r
    else:
        pid_to_row[0] = int(select_rows[0].item())

    pid_sorted = sorted(pid_to_row.keys())
    rows_sorted = [pid_to_row[pid] for pid in pid_sorted]
    rows_t = torch.tensor(rows_sorted, device=flat.device, dtype=torch.long)

    flat_sel = flat.index_select(0, rows_t)  # [P,T,K,C]
    if flat_sel.dim() != 4 or flat_sel.size(1) <= 0:
        return pred_kps_per_frame

    P, T_kp, K, C = flat_sel.shape
    T_use = min(T, T_kp)
    for t in range(T_use):
        pred_kps_per_frame[t] = flat_sel[:, t, :num_keypoints, :2]
    return pred_kps_per_frame


# ============================================================
# 新增：单帧渲染（Matplotlib -> ndarray(BGR)），用于写视频
# ============================================================

def render_one_frame_to_bgr(
    *,
    image_t: torch.Tensor,  # [3,H,W] or [H,W,3] 皆可（show_image函数已兼容）
    boxes_t: torch.Tensor,  # [N,4] normalized xyxy
    labels_t: torch.Tensor, # [N]
    scores_t: torch.Tensor, # [N]
    inst_ids_t: torch.Tensor,  # [N] tube id (same tube same color)
    keypoints_t: Optional[torch.Tensor],  # [P,K,2] 或 None
    category_names: Optional[Dict],
    conf_thr: float,
    num_keypoints: int,
    title_text: str,
    vis_threshold: float = 0.5,
) -> np.ndarray:
    # 置信度过滤（与原 callback 一致：先过滤再画）
    if isinstance(scores_t, torch.Tensor) and scores_t.numel() > 0:
        m = scores_t > float(conf_thr)
        boxes_t = boxes_t[m]
        labels_t = labels_t[m]
        inst_ids_t = inst_ids_t[m]
    else:
        boxes_t = torch.empty((0, 4), dtype=torch.float32)
        labels_t = torch.empty((0,), dtype=torch.long)
        inst_ids_t = torch.empty((0,), dtype=torch.long)

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    # 画 Pred boxes（同tube同色：instance_ids）
    show_image_with_boxes(
        image=image_t,
        boxes=boxes_t,
        labels=labels_t,
        title=None,
        category_names=category_names,
        ax=ax,
        normalized_boxes=True,
        instance_ids=inst_ids_t,
    )

    # 画 keypoints（叠在同一个 ax 上）
    if isinstance(keypoints_t, torch.Tensor) and keypoints_t.numel() > 0:
        # show_image_with_keypoints 内部会再次 imshow（默认 zorder 更低），不会破坏 box patch
        show_image_with_keypoints(
            image=image_t,
            keypoints=keypoints_t,
            normalized=True,
            num_keypoints=num_keypoints,
            draw_skeleton=True,
            ax=ax,
            vis_threshold=float(vis_threshold),
        )

    # 左上角文字（版本说明 + t）
    ax.text(
        5, 20, title_text,
        fontsize=12,
        color="white",
        bbox=dict(facecolor="black", alpha=0.6, pad=2),
    )

    plt.tight_layout(pad=0.1)
    bgr = _mpl_fig_to_bgr(fig)
    plt.close(fig)
    return bgr


# ============================================================
# 主推理：生成两版 mp4（PredBox+PredKP / PredBox+GTKP）
# ============================================================

@script
def infer(
    video_dir: str = "./input",
    checkpoint: str = "./outputs/2025-12-28_21-34-02/final_model.pt",
    config_file: str = "overfit_eval.yaml",
    output_dir: str = "./output_infer",
    device: str = "cuda",
    # tube 去重阈值
    tube_dup_iou_thr: float = 0.5,
    tube_dup_ratio_thr: float = 0.8,
    # 可视化阈值
    confidence_threshold: float = 0.8,
    # 视频参数：复制/拉伸帧用
    video_fps: int = 15,
    target_seconds: float = 3.0,
    # keypoints 可见性阈值（与你 show_image_with_keypoints 一致）
    kp_vis_threshold: float = 0.5,
):
    device = torch.device(device)

    # 1) config & token processor
    config = load_config_from_yaml((FILE_PATH / "config") / config_file)
    token_processor = build_token_processor(config)

    # 2) dataset
    base_dataset = VideoClipDataset_kp_d(
        root_dir=video_dir,
        clip_len=getattr(config.data, "tube_frames", 8),
        filter_empty=False,
        use_neighbor_fusion=True,
        neighbor_window=4,
    )

    dataset = Pix2SeqVideoDataset_kp_d(
        base_dataset=base_dataset,
        num_classes=config.data.num_classes,
        training=False,
        max_num_objects_per_frame=config.data.max_tubes_per_clip,
        image_size=config.data.image_size,
        jitter_scale=config.data.jitter_scale,
        color_jitter_strength=config.data.color_jitter_strength,
        use_video_augmentation=True,
        use_tube_augmentation=False,
    )

    collator = Pix2SeqVideoCollator_kp_d(
        token_processor=token_processor,
        persons_per_clip=getattr(config.data, "persons_per_clip", 2),
        corrupt_and_randomise=False,
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        collate_fn=collator,
    )

    # 3) model
    model = build_model(config, token_processor).to(device)
    ckpt = torch.load(checkpoint, map_location="cpu")
    sd = _extract_state_dict(ckpt)

    if any(k.startswith("module.") for k in sd.keys()):
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}

    missing, unexpected = model.load_state_dict(sd, strict=False)
    if len(missing) > 0 or len(unexpected) > 0:
        print(f"[CKPT] missing={len(missing)} unexpected={len(unexpected)}")
        print("[CKPT] missing sample:", missing[:20])
        print("[CKPT] unexpected sample:", unexpected[:20])

    model.eval()

    # 4) generator
    generator = TaskSequenceGenerator(
        token_processor=token_processor,
        temperature=config.generation.temperature,
        top_k=config.generation.top_k,
        top_p=config.generation.top_p,
        max_seq_len=token_processor.max_seq_len,
        persons_per_clip=getattr(config.data, "persons_per_clip", 2),
    )

    os.makedirs(output_dir, exist_ok=True)
    category_names = getattr(dataset, "categories", None)

    # 5) inference loop
    with torch.no_grad():
        for idx, batch in enumerate(loader):
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)

            videos = batch["images"]  # [B,T,3,H,W]
            B, T_full = videos.shape[:2]
            T = min(int(T_full), int(token_processor.tube_frames))

            # clip id
            clip_vid = batch.get("video_id", None)
            clip_vid = str(clip_vid[0]) if clip_vid is not None else f"clip{idx:03d}"
            clip_stem = _safe_stem(clip_vid) + f"_clip{idx:03d}"

            # ----------------
            # DET inference
            # ----------------
            det_seq, det_logits = generator.generate(
                model=model,
                images=videos,
                task_token_id=token_processor.DET_TASK_TOKEN,
            )

            tube_boxes, tube_labels, tube_scores = token_processor.post_process_sequences(
                det_seq,
                class_logits=det_logits,
                confidence_threshold=0.0,
                tube_frames=token_processor.tube_frames,
            )

            tb0 = tube_boxes[0]
            tl0 = tube_labels[0]
            ts0 = tube_scores[0]
            if not isinstance(tb0, torch.Tensor):
                tb0 = torch.tensor(tb0, device=device)
            if not isinstance(tl0, torch.Tensor):
                tl0 = torch.tensor(tl0, device=device)
            if not isinstance(ts0, torch.Tensor):
                ts0 = torch.tensor(ts0, device=device)

            before_n = int(tb0.shape[0])
            tb0_nms, tl0_nms, ts0_nms, kept_idx = tube_nms_by_duplicate_ratio(
                tb0, tl0, ts0,
                dup_iou_thr=tube_dup_iou_thr,
                dup_ratio_thr=tube_dup_ratio_thr,
            )
            after_n = int(tb0_nms.shape[0])
            if before_n != after_n:
                print(f"[Infer][DET-NMS] tubes: {before_n} -> {after_n} (dup_iou_thr={tube_dup_iou_thr}, dup_ratio_thr={tube_dup_ratio_thr})")

            # 构造 run.py 里同格式 tube_predictions（后续逐帧拆分完全复用 callback 的逻辑）
            batch_output: Dict[str, Any] = {
                "tube_predictions": [
                    {
                        "video_id": str(clip_vid),
                        "category_id": int(tl0_nms[i].item()),
                        "score": float(ts0_nms[i].item()),
                        "boxes": tb0_nms[i].detach().cpu().tolist(),  # [T,4]
                    }
                    for i in range(after_n)
                ]
            }

            # ----------------
            # KP inference（与你 run.py 一致：flat + 映射）
            # ----------------
            if "kp_images" in batch:
                kp_seq, _ = generator.generate(
                    model=model,
                    images=batch["kp_images"],
                    task_token_id=token_processor.KP_TASK_TOKEN,
                    forced_tokens=batch["kp_target_seq"],
                    forced_mask=batch["kp_token_weights"] <= 0,
                    frame_ids=batch.get("kp_frame_ids"),
                )
                _, kp_kps = token_processor.post_process_kp_sequences(
                    kp_seq, tube_frames=token_processor.tube_frames
                )
                batch_output["kp_pred_keypoints_flat"] = torch.stack(kp_kps, dim=0)
                batch_output["kp_row_to_clip_index"] = batch["kp_parent_index"]
                batch_output["kp_row_to_person_index"] = batch["kp_person_pid"]

            # ----------------
            # 逐帧 Pred boxes（含 instance_ids= tube_idx）
            # ----------------
            pred_by_vid = _group_tube_predictions_by_video(batch_output.get("tube_predictions", None))
            pred_boxes_pf, pred_labels_pf, pred_scores_pf, pred_inst_pf = _build_framewise_preds_for_clip(
                T=T, clip_video_id=str(clip_vid), pred_by_vid=pred_by_vid
            )

            # ----------------
            # 逐帧 GT / Pred keypoints（与你 callback 对齐）
            # ----------------
            tube_valid_mask = batch.get("tube_valid_mask", None)  # [B,N,T]
            tube_valid_mask_b = tube_valid_mask[0] if isinstance(tube_valid_mask, torch.Tensor) else None

            gt_kps_pf = _extract_gt_keypoints_per_frame(
                b=0, T=T, batch=batch, num_keypoints=token_processor.num_keypoints, tube_valid_mask_b=tube_valid_mask_b
            )
            pred_kps_pf = _extract_pred_kp_keypoints_per_frame(
                b=0, T=T, batch=batch, batch_output=batch_output, num_keypoints=token_processor.num_keypoints
            )

            # ----------------
            # 渲染两套帧序列（matplotlib -> BGR）
            # ----------------
            frames_v1: List[np.ndarray] = []  # PredBox + PredKP
            frames_v2: List[np.ndarray] = []  # PredBox + GTKP

            clip_images = videos[0, :T].detach().cpu()  # [T,3,H,W]

            for t in range(T):
                img_t = clip_images[t]

                # v1: PredBox + PredKP
                kps_pred_t = pred_kps_pf[t] if (t < len(pred_kps_pf)) else None
                frame1 = render_one_frame_to_bgr(
                    image_t=img_t,
                    boxes_t=pred_boxes_pf[t],
                    labels_t=pred_labels_pf[t],
                    scores_t=pred_scores_pf[t],
                    inst_ids_t=pred_inst_pf[t],
                    keypoints_t=kps_pred_t,
                    category_names=category_names,
                    conf_thr=float(confidence_threshold),
                    num_keypoints=token_processor.num_keypoints,
                    title_text=f"t={t}  PredBox + PredKP",
                    vis_threshold=float(kp_vis_threshold),
                )
                frames_v1.append(frame1)

                # v2: PredBox + GTKP
                kps_gt_t = gt_kps_pf[t] if (t < len(gt_kps_pf)) else None
                frame2 = render_one_frame_to_bgr(
                    image_t=img_t,
                    boxes_t=pred_boxes_pf[t],
                    labels_t=pred_labels_pf[t],
                    scores_t=pred_scores_pf[t],
                    inst_ids_t=pred_inst_pf[t],
                    keypoints_t=kps_gt_t,
                    category_names=category_names,
                    conf_thr=float(confidence_threshold),
                    num_keypoints=token_processor.num_keypoints,
                    title_text=f"t={t}  PredBox + GTKP",
                    vis_threshold=float(kp_vis_threshold),
                )
                frames_v2.append(frame2)

            # ----------------
            # 写两版 mp4（自动复制/拉伸帧）
            # ----------------
            out1 = str(Path(output_dir) / f"{clip_stem}_predbox_predkp.mp4")
            out2 = str(Path(output_dir) / f"{clip_stem}_predbox_gtkp.mp4")

            write_mp4_from_frames(out1, frames_v1, fps=int(video_fps), target_seconds=float(target_seconds))
            write_mp4_from_frames(out2, frames_v2, fps=int(video_fps), target_seconds=float(target_seconds))

            print(f"[Infer] processed clip {idx+1}/{len(dataset)}")
            print(f"[MP4] saved:\n  {out1}\n  {out2}")


if __name__ == "__main__":
    infer()
