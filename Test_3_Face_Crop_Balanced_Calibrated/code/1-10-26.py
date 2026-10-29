"""
1-10-26: Degradation-aware deepfake detector - face crops, 128x128, balanced sampling,
validation-selected thresholds, dataset-wise / robustness / cross-dataset evaluation
=========================================================================================

Builds on 30-9-26test.py (kept unchanged as the 96x96 full-frame baseline = Experiment A).

Why this update (findings from the 30-9-26 run, run_30-9-26/test):
  * Stage 2 overall accuracy ~0.89 hid a failure on FF++: 199/200 FF++ real videos were
    predicted fake, FF++ AUC ~0.58 (BioDeepAV AUC ~0.99).
  * Inputs differed systematically between datasets: BioDeepAV videos are already tight face
    crops (real 208x192, fake 512x512) while FF++ are full scenes (640x480 .. 1920x1080), so a
    full-frame 96x96 resize shows FF++ faces at ~20 px. FF++ is also 1 real : 6 fake, and only
    BioDeepAV has audio. These are POTENTIAL dataset shortcuts (tested in the diagnostics
    section, not assumed).

What changed (Experiment B):
  * Face detection (OpenCV YuNet, landmarks) -> square face crop with margin -> optional eye
    alignment -> resize to IMG_SIZE (default 128, configurable). Frames without a detection
    reuse the nearest detected frame's box; videos with no detection fall back to the old
    full-frame resize. Per-video and per-dataset detection statistics are logged.
  * Degraded videos are cropped with their SOURCE video's face boxes (default), so original
    and degraded crops are pixel-aligned (needed for a fair Grad-CAM stability comparison).
  * Dataset x class balanced sampling (BioDeepAV real/fake, FF++ real/fake each ~25%), with
    an oversampling cap. Loss class-weighting is only used when sampling is not balanced.
  * Model selection on the validation set by MACRO per-dataset AUC (a model that fails on one
    dataset can no longer win on overall AUC).
  * Validation-only threshold optimisation (balanced accuracy, F1 tie-break) for each stage;
    threshold + full config saved inside the checkpoint. Test set is never used for selection.
  * Reports: dataset-wise (Combined / BioDeepAV / FF++) x condition (original / degraded)
    metrics, confusion matrices, robustness drops + prediction flips, degradation x severity
    table, threshold sensitivity (analysis only), cross-dataset experiments, shortcut
    diagnostics, 96 vs 128 comparison (when the 30-9-26 checkpoints/caches exist),
    Grad-CAM stability on the frozen models, final_experiment_summary.json.
  * Inference API for the next (XAI) phase: DeepfakeDetector(checkpoint).predict_video(path).

Leakage: identity groups (YouTube id for BioDeepAV real, FF++ id pairs via union-find, DFD actor
pair) are assigned to ONE split; degraded versions always inherit their source video's split.

Usage:
  python 1-10-26.py --mode selftest                                   # smoke tests (no training)
  python 1-10-26.py --max-per-folder 30 --epochs1 1 --epochs2 1 --cross-epochs 1 --work-dir run_1-10-26_smoke
  python 1-10-26.py                                                   # full: stage1 -> stage2 -> evaluation -> cross -> diagnostics -> XAI
  python 1-10-26.py --mode stage1 | stage2 | evaluate | threshold | cross
  python 1-10-26.py --mode predict --video path\\to\\video.mp4          # frozen final model
"""

from __future__ import annotations

import os
import re
import sys
import json
import time
import math
import random
import hashlib
import threading
import warnings
import argparse
import subprocess
import urllib.request
from collections import deque, defaultdict
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import cv2
from scipy.io import wavfile
from scipy.stats import spearmanr
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix

warnings.filterwarnings("ignore")
cv2.setNumThreads(1)
try:
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
except Exception:
    pass

# =====================================================================================
# CENTRAL CONFIGURATION (every value can be overridden from the command line)
# =====================================================================================
IMG_SIZE = 128
NUM_FRAMES = 16          # frames cached per video (unchanged)
CLIP_LEN = 8             # frames per model clip (unchanged)
BATCH_SIZE = 32          # unchanged
LR_STAGE1 = 1e-3         # unchanged
LR_STAGE2 = 3e-4         # unchanged
EPOCHS_STAGE1 = 8        # unchanged
EPOCHS_STAGE2 = 8        # unchanged
RANDOM_SEED = 42
DEFAULT_THRESHOLD = 0.50
OPTIMIZE_THRESHOLD = True
THRESHOLD_METRIC = "balanced_accuracy"
THRESHOLD_GRID = (0.01, 0.99, 0.01)
SENSITIVITY_THRESHOLDS = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
FACE_CROP_ENABLED = True
FACE_ALIGNMENT_ENABLED = False
FACE_MARGIN = 0.20       # crop side = max(face w, h) * (1 + 2 * margin)
FACE_SCORE_THRESHOLD = 0.6
BALANCED_SAMPLING = True
DATASET_BALANCED = True  # True -> "dataset_class" cells, False -> "class" only
MAX_OVERSAMPLE = 4.0     # cap on how often a sampling cell may be oversampled per epoch

YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
DEGRADATION_TYPES = ["noise", "blur", "brightness", "compression", "saturation", "contrast", "motion_blur", "frame_drop"]
SEVERITIES = ["mild", "medium", "severe"]
FRAME_DROP_LEVELS = ["5pct", "10pct", "30pct"]
SEVERITY_LEVEL = {"mild": 0, "medium": 1, "severe": 2, "5pct": 0, "10pct": 1, "30pct": 2}
SEVERITY_NAME = {"mild": "mild", "medium": "medium", "severe": "severe", "5pct": "mild", "10pct": "medium", "30pct": "severe"}
FFPP_FOLDERS = ["original", "Deepfakes", "Face2Face", "FaceShifter", "FaceSwap", "NeuralTextures", "DeepFakeDetection"]
NO_WINDOW = subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0
METRIC_KEYS = ["accuracy", "balanced_accuracy", "precision", "recall", "specificity", "f1", "auc"]


# =====================================================================================
# UTILITIES & THERMAL GUARD (from 30-9-26)
# =====================================================================================
def format_time(seconds):
    if seconds is None or np.isnan(seconds) or seconds < 0:
        return "--:--"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h > 0 else f"{m:02d}:{s:02d}"


def progress_bar(current, total, bar_len=20):
    frac = min(1.0, max(0.0, current / max(1, total)))
    filled = int(frac * bar_len)
    bar = ("=" * filled + (">" if filled < bar_len else "")).ljust(bar_len, ".")
    return f"[{bar}] {frac * 100:5.1f}%"


_last_gpu_temp_check = 0
_cached_gpu_temp = None


def get_gpu_temp(ttl=1.5):
    global _last_gpu_temp_check, _cached_gpu_temp
    now = time.time()
    if now - _last_gpu_temp_check < ttl and _cached_gpu_temp is not None:
        return _cached_gpu_temp
    if not torch.cuda.is_available():
        return None
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"], creationflags=NO_WINDOW, timeout=1.5
        )
        _cached_gpu_temp = int(out.decode().strip().split()[0])
        _last_gpu_temp_check = now
        return _cached_gpu_temp
    except Exception:
        return _cached_gpu_temp


def check_gpu_thermal_guard(max_temp, pause_sec):
    curr_temp = get_gpu_temp(ttl=0.0)
    if curr_temp is None or curr_temp < max_temp:
        return 0
    print(f"\n\n{'!' * 78}\n [THERMAL WARNING] GPU at {curr_temp}C (threshold {max_temp}C) - pausing {pause_sec // 60} min\n{'!' * 78}\n")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    t_pause_start = time.time()
    while True:
        rem = pause_sec - (time.time() - t_pause_start)
        if rem <= 0:
            break
        t_now = get_gpu_temp(ttl=0.0)
        sys.stdout.write(f"\r  [Cooling Down] Remaining: {format_time(rem)} | GPU: {t_now if t_now is not None else 'N/A'}C   ")
        sys.stdout.flush()
        time.sleep(1.0)
    sys.stdout.write("\r  [Cooling Down Complete] Resuming!                                   \n\n")
    return int(time.time() - t_pause_start)


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seeds every RNG. Full cuDNN determinism is optional because it slows convolutions;
    without it, runs are reproducible up to small GPU non-determinism (documented in the report)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.benchmark = True


def to_jsonable(o):
    if isinstance(o, dict):
        return {str(k): to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_jsonable(v) for v in o]
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if (np.isnan(o) or np.isinf(o)) else float(o)
    if isinstance(o, np.ndarray):
        return to_jsonable(o.tolist())
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(obj), f, indent=2)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# =====================================================================================
# CLI
# =====================================================================================
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="1-10-26: face-crop 128x128 balanced two-stage deepfake detector + evaluation")
    p.add_argument("--mode", default="full",
                   choices=["full", "stage1", "stage2", "evaluate", "threshold", "cross", "selftest", "predict"],
                   help="full = stage1 -> stage2 -> evaluate (+cross, diagnostics, XAI); evaluate/threshold use saved checkpoints")
    p.add_argument("--datasets", default="bio,ffpp", help="Comma list: bio, ffpp")
    p.add_argument("--bio-root", default=r"E:\Datasets\BioDeepAV")
    p.add_argument("--ffpp-root", default=r"E:\Datasets\FaceForensics++_C23")
    p.add_argument("--ffpp-folders", default=",".join(FFPP_FOLDERS))
    p.add_argument("--extra-manifests", nargs="*", default=[])
    p.add_argument("--max-per-folder", type=int, default=None, help="Random subset per folder (smoke tests)")
    p.add_argument("--test-ratio", type=float, default=0.20)
    p.add_argument("--val-ratio", type=float, default=0.10)
    # model / input
    p.add_argument("--img-size", type=int, default=IMG_SIZE)
    p.add_argument("--frames", type=int, default=NUM_FRAMES)
    p.add_argument("--clip-len", type=int, default=CLIP_LEN)
    p.add_argument("--modality", choices=["av", "video"], default="av", help="'av' (audio-visual, as before) or 'video' (audio masked)")
    p.add_argument("--audio-dropout", type=float, default=0.2)
    # face preprocessing
    p.add_argument("--no-face-crop", action="store_true", help="Disable face cropping (old full-frame resize)")
    p.add_argument("--face-align", action="store_true", default=FACE_ALIGNMENT_ENABLED, help="Rotate crops so the eyes are horizontal")
    p.add_argument("--face-margin", type=float, default=FACE_MARGIN)
    p.add_argument("--face-score", type=float, default=FACE_SCORE_THRESHOLD)
    p.add_argument("--face-model", default=None, help="YuNet .onnx (default: <script dir>/models/face_detection_yunet_2023mar.onnx, auto-downloaded)")
    p.add_argument("--deg-face-boxes", choices=["source", "self"], default="source",
                   help="'source': crop degraded videos with their source video's face boxes (aligned crops); 'self': detect on degraded frames")
    # sampling
    p.add_argument("--sampling", choices=["dataset_class", "class", "none"],
                   default=("dataset_class" if DATASET_BALANCED else "class") if BALANCED_SAMPLING else "none")
    p.add_argument("--max-oversample", type=float, default=MAX_OVERSAMPLE)
    p.add_argument("--epoch-size", type=int, default=None, help="Samples per epoch (default: number of training items)")
    # training
    p.add_argument("--epochs1", type=int, default=EPOCHS_STAGE1)
    p.add_argument("--epochs2", type=int, default=EPOCHS_STAGE2)
    p.add_argument("--cross-epochs", type=int, default=None, help="Epochs for cross-dataset models (default: --epochs1)")
    p.add_argument("--lr1", type=float, default=LR_STAGE1)
    p.add_argument("--lr2", type=float, default=LR_STAGE2)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--patience", type=int, default=0, help="Early stopping patience in epochs (0 = off)")
    p.add_argument("--synth-per-video", type=int, default=3)
    # thresholds
    p.add_argument("--no-optimize-threshold", action="store_true", help=f"Use the fixed {DEFAULT_THRESHOLD} threshold")
    p.add_argument("--threshold-metric", default=THRESHOLD_METRIC, choices=["balanced_accuracy", "f1", "youden"])
    # experiments
    p.add_argument("--skip-cross", action="store_true")
    p.add_argument("--skip-xai", action="store_true")
    p.add_argument("--xai-samples", type=int, default=150)
    p.add_argument("--baseline-dir", default=None, help="30-9-26 run dir with 96x96 checkpoints/caches (default: ./run_30-9-26)")
    p.add_argument("--baseline-extract", action="store_true", help="Extract 96x96 full-frame caches if not found (for the 96 vs 128 comparison)")
    p.add_argument("--compare-runs", nargs="*", default=[], help="Other 1-10-26 run dirs whose metrics_long.csv to add to resolution_comparison.csv")
    p.add_argument("--show-random", type=int, default=20)
    # misc
    p.add_argument("--video", default=None, help="--mode predict: video to classify")
    p.add_argument("--checkpoint", default=None, help="--mode predict: checkpoint (default: <work-dir>/results/final/final_model.pt)")
    p.add_argument("--work-dir", default=None, help="Default: ./run_1-10-26")
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--loader-threads", type=int, default=3)
    p.add_argument("--gpu-max-temp", type=int, default=72)
    p.add_argument("--gpu-pause", type=int, default=300)
    p.add_argument("--seed", type=int, default=RANDOM_SEED)
    p.add_argument("--deterministic", action="store_true", help="cuDNN deterministic mode (slower)")
    a = p.parse_args(argv)
    a.face_crop = not a.no_face_crop
    a.optimize_threshold = OPTIMIZE_THRESHOLD and not a.no_optimize_threshold
    a.cross_epochs = a.cross_epochs if a.cross_epochs is not None else a.epochs1
    return a


# =====================================================================================
# ARCHITECTURE (unchanged; resolution-agnostic thanks to adaptive pooling)
# =====================================================================================
def block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1), nn.ReLU(inplace=True), nn.BatchNorm2d(cout),
        nn.Conv2d(cout, cout, 3, padding=1), nn.ReLU(inplace=True), nn.BatchNorm2d(cout),
        nn.MaxPool2d(2),
    )


class VideoCNN(nn.Module):
    def __init__(self, width=(16, 32, 64, 128)):
        super().__init__()
        ch = (3,) + tuple(width)
        self.features = nn.Sequential(*[block(ch[i], ch[i + 1]) for i in range(len(width))])
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):  # [B, T, 3, H, W]
        B, T, C, H, W = x.shape
        f = self.pool(self.features(x.reshape(B * T, C, H, W))).flatten(1).view(B, T, -1)
        return torch.cat([f.mean(1), f.amax(1)], dim=1)  # [B, 256]


class AudioCNN(nn.Module):
    def __init__(self, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(128, out_dim, 3, padding=1), nn.BatchNorm2d(out_dim), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

    def forward(self, x):
        return self.net(x).flatten(1)


class AudioVisualDeepFakeCNN(nn.Module):
    """Same network as 30-9-26 (state dicts are interchangeable). a_mask=0 zeroes the audio features."""

    def __init__(self):
        super().__init__()
        self.video_net = VideoCNN(width=(16, 32, 64, 128))
        self.audio_net = AudioCNN(out_dim=128)
        self.fusion_head = nn.Sequential(nn.Linear(256 + 128, 128), nn.ReLU(inplace=True), nn.Dropout(0.5), nn.Linear(128, 2))

    def forward(self, x_v, x_a=None, mode="av", a_mask=None):
        ref = x_v if x_v is not None else x_a
        B, dev = ref.size(0), ref.device
        f_v = self.video_net(x_v) if mode in ("av", "v_only") else torch.zeros(B, 256, device=dev)
        f_a = self.audio_net(x_a) if mode in ("av", "a_only") else torch.zeros(B, 128, device=dev)
        if mode == "av" and a_mask is not None:
            f_a = f_a * a_mask.view(-1, 1).to(f_a.dtype)
        return self.fusion_head(torch.cat([f_v, f_a], dim=1))


def to_video_tensor(arr: np.ndarray, device) -> torch.Tensor:
    """uint8 RGB [B,T,H,W,3] -> float [B,T,3,H,W] in [-1,1] (normalisation unchanged)."""
    t = torch.from_numpy(np.ascontiguousarray(arr)).permute(0, 1, 4, 2, 3).float().div_(127.5).sub_(1.0)
    return t.to(device, non_blocking=True)


# =====================================================================================
# FACE PREPROCESSING
# =====================================================================================
def ensure_face_model(path: Optional[str], script_dir: Path) -> Optional[Path]:
    p = Path(path) if path else script_dir / "models" / "face_detection_yunet_2023mar.onnx"
    if p.exists():
        return p
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        print(f"  Downloading YuNet face detector -> {p}")
        urllib.request.urlretrieve(YUNET_URL, p)
        return p
    except Exception as e:
        print(f"  [WARN] Could not download the YuNet face model ({e}); face cropping falls back to full frames.")
        return None


class FaceDetector:
    """Thread-safe wrapper around OpenCV YuNet (box + 5 landmarks). One detector per thread."""

    def __init__(self, model_path: Optional[Path], score_threshold: float = FACE_SCORE_THRESHOLD, max_side: int = 640):
        self.model_path = model_path
        self.score_threshold = score_threshold
        self.max_side = max_side
        self.backend = "yunet" if (model_path is not None and Path(model_path).exists() and hasattr(cv2, "FaceDetectorYN")) else "none"
        self._local = threading.local()

    def _detector(self):
        d = getattr(self._local, "det", None)
        if d is None:
            d = cv2.FaceDetectorYN.create(str(self.model_path), "", (320, 320), self.score_threshold, 0.3, 50)
            self._local.det = d
        return d

    def detect(self, frame_bgr: np.ndarray) -> Optional[Dict[str, Any]]:
        """Returns the most prominent face {bbox: (x,y,w,h), landmarks: (5,2) or None, score} or None."""
        if self.backend != "yunet":
            return None
        H, W = frame_bgr.shape[:2]
        scale = min(1.0, self.max_side / max(H, W))
        img = cv2.resize(frame_bgr, (int(W * scale), int(H * scale)), interpolation=cv2.INTER_AREA) if scale < 1 else frame_bgr
        det = self._detector()
        det.setInputSize((img.shape[1], img.shape[0]))
        try:
            _, faces = det.detect(img)
        except Exception:
            return None
        if faces is None or len(faces) == 0:
            return None
        faces = faces[(faces[:, 2] >= 16) & (faces[:, 3] >= 16)]
        if len(faces) == 0:
            return None
        best = faces[np.argmax(faces[:, 2] * faces[:, 3] * faces[:, 14])]
        return dict(bbox=best[:4] / scale, landmarks=best[4:14].reshape(5, 2) / scale, score=float(best[14]))


def detect_face(frame_bgr: np.ndarray, detector: Optional[FaceDetector]) -> Optional[Dict[str, Any]]:
    return detector.detect(frame_bgr) if detector is not None else None


def face_box_relative(bbox, W: int, H: int, margin: float) -> np.ndarray:
    """Square crop box around a face as (cx/W, cy/H, side/min(W,H)); resolution independent, so the
    same box can be applied to a degraded copy of the video with a different resolution."""
    x, y, w, h = [float(v) for v in bbox]
    side = min(max(w, h) * (1.0 + 2.0 * margin), float(min(W, H)))
    return np.array([(x + w / 2) / W, (y + h / 2) / H, side / min(W, H)], dtype=np.float32)


def eye_angle(landmarks: Optional[np.ndarray]) -> float:
    """YuNet landmarks: 0 = right eye (image left), 1 = left eye (image right)."""
    if landmarks is None:
        return 0.0
    (rx, ry), (lx, ly) = landmarks[0], landmarks[1]
    return float(np.degrees(np.arctan2(ly - ry, lx - rx)))


def align_face(frame: np.ndarray, center: Tuple[float, float], angle: float, side: float, size: int) -> np.ndarray:
    """Rotate around the face centre so the eyes are horizontal, crop `side` px and resize, in one warp."""
    cx, cy = center
    M = cv2.getRotationMatrix2D((cx, cy), angle, size / side)
    M[0, 2] += size / 2.0 - cx
    M[1, 2] += size / 2.0 - cy
    return cv2.warpAffine(frame, M, (size, size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def crop_face(frame: np.ndarray, box_rel: np.ndarray, size: int, angle: float = 0.0) -> np.ndarray:
    """Crop the square box (kept inside the frame, no padding so no dataset-specific borders) and resize."""
    H, W = frame.shape[:2]
    side = float(np.clip(box_rel[2] * min(W, H), 8, min(W, H)))
    cx, cy = float(box_rel[0] * W), float(box_rel[1] * H)
    cx = float(np.clip(cx, side / 2, W - side / 2))
    cy = float(np.clip(cy, side / 2, H - side / 2))
    if abs(angle) > 1.0:
        return align_face(frame, (cx, cy), angle, side, size)
    s = int(round(side))
    x0, y0 = int(round(cx - side / 2)), int(round(cy - side / 2))
    x0, y0 = max(0, min(x0, W - s)), max(0, min(y0, H - s))
    patch = frame[y0 : y0 + s, x0 : x0 + s]
    return cv2.resize(patch, (size, size), interpolation=cv2.INTER_AREA if s > size else cv2.INTER_CUBIC)


def preprocess_frame(frame: np.ndarray, size: int, box_rel: Optional[np.ndarray] = None, angle: float = 0.0) -> np.ndarray:
    """Face crop when a box is available, otherwise the previous full-frame resize (fallback)."""
    if box_rel is None:
        return cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)
    return crop_face(frame, box_rel, size, angle)


def read_sampled_frames(path, n_frames: int, max_side: int = 960):
    """n_frames uniformly spaced BGR frames (same sampling as 30-9-26) + video metadata."""
    cap = cv2.VideoCapture(str(path))
    meta = dict(width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                fps=float(cap.get(cv2.CAP_PROP_FPS) or 0.0), total_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))

    def shrink(fr):
        h, w = fr.shape[:2]
        sc = max_side / max(h, w)
        return cv2.resize(fr, (int(w * sc), int(h * sc)), interpolation=cv2.INTER_AREA) if sc < 1 else fr

    frames = {}
    total = meta["total_frames"]
    if total > 0:
        want = set(np.linspace(0, total - 1, n_frames).round().astype(int).tolist())
        last, i = max(want), 0
        while i <= last:
            if not cap.grab():
                break
            if i in want:
                ok, fr = cap.retrieve()
                if ok:
                    frames[i] = shrink(fr)
            i += 1
    if not frames:  # broken frame count -> sequential decode
        cap.release()
        cap = cv2.VideoCapture(str(path))
        tmp = []
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            tmp.append(shrink(fr))
        if not tmp:
            cap.release()
            return None, meta
        frames = {j: tmp[j] for j in np.linspace(0, len(tmp) - 1, n_frames).round().astype(int)}
    cap.release()
    keys = sorted(frames)
    want_list = np.linspace(0, keys[-1], n_frames).round().astype(int)
    return [frames[min(keys, key=lambda k: abs(k - t))] for t in want_list], meta


def smooth_boxes(boxes: np.ndarray) -> np.ndarray:
    """3-frame moving average of crop centre/size to reduce jitter."""
    if len(boxes) < 3:
        return boxes
    pad = np.concatenate([boxes[:1], boxes, boxes[-1:]])
    return ((pad[:-2] + pad[1:-1] + pad[2:]) / 3.0).astype(np.float32)


def extract_face_clip(path, n_frames: int, size: int, detector: Optional[FaceDetector], margin: float,
                      align: bool, ref: Optional[Tuple[np.ndarray, np.ndarray, bool]] = None):
    """Video -> frames -> face detection -> crop (+ optional alignment) -> resize -> RGB uint8 [T,S,S,3].
    ref = (boxes [T,3], angles [T], fallback) reuses known boxes (degraded copies of a source video).
    Returns (clip or None, info)."""
    info = dict(frames=0, detected=0, failed=0, detection_rate=float("nan"), fallback=True, face_rel_size=float("nan"),
                boxes=np.zeros((n_frames, 3), np.float32), angles=np.zeros(n_frames, np.float32), det=np.zeros(n_frames, np.int8),
                width=0, height=0, fps=0.0, total_frames=0, error="")
    try:
        frames, meta = read_sampled_frames(path, n_frames)
        info.update(meta)
        if frames is None:
            info["error"] = "no frames decoded"
            return None, info
        h, w = frames[0].shape[:2]
        T = len(frames)
        info["frames"] = T
        if ref is not None:
            boxes, angles, fallback = ref
            info["detected"], info["failed"] = -1, -1  # boxes inherited, no detection run
        else:
            dets = [detect_face(f, detector) for f in frames]
            detected = np.array([d is not None for d in dets])
            info["detected"], info["failed"] = int(detected.sum()), int(T - detected.sum())
            info["detection_rate"] = float(detected.mean())
            info["det"] = detected.astype(np.int8)
            fallback = not detected.any()
            boxes = np.zeros((T, 3), np.float32)
            angles = np.zeros(T, np.float32)
            if not fallback:
                idx_det = np.where(detected)[0]
                for t in range(T):  # frames without a detection borrow the nearest detected frame's box
                    d = dets[t] if detected[t] else dets[idx_det[np.argmin(np.abs(idx_det - t))]]
                    boxes[t] = face_box_relative(d["bbox"], w, h, margin)
                    angles[t] = eye_angle(d["landmarks"]) if align else 0.0
                boxes = smooth_boxes(boxes)
                info["face_rel_size"] = float(boxes[:, 2].mean())
        info["boxes"], info["angles"], info["fallback"] = boxes, angles, bool(fallback)
        crops = [preprocess_frame(f, size, None if fallback else boxes[t], float(angles[t])) for t, f in enumerate(frames)]
        return np.stack(crops)[..., ::-1].copy(), info  # BGR -> RGB
    except Exception as e:
        info["error"] = str(e)
        return None, info


def extract_video_clip(path, n_frames=16, size=96):
    """30-9-26 full-frame extraction (kept for the 96x96 baseline caches)."""
    frames, _ = read_sampled_frames(path, n_frames, max_side=100000)
    if frames is None:
        return None
    return np.stack([cv2.resize(f, (size, size), interpolation=cv2.INTER_AREA) for f in frames])[..., ::-1].copy()


# =====================================================================================
# AUDIO (from 30-9-26)
# =====================================================================================
def extract_audio_spectrogram(wav_candidates, video_path, n_samples=48000):
    """Returns (spectrogram [1,128,128] float32, has_audio bool)."""
    audio = None
    for wav_path in wav_candidates:
        if not Path(wav_path).exists():
            continue
        try:
            sr, raw = wavfile.read(wav_path)
            if raw.ndim > 1:
                raw = raw[:, 0]
            audio = raw.astype(np.float32) / 32768.0 if raw.dtype == np.int16 else raw.astype(np.float32)
            if sr != 16000 and len(audio) > 0:
                from scipy.signal import resample
                audio = resample(audio, int(len(audio) * 16000 / sr)).astype(np.float32)
            break
        except Exception:
            audio = None
    if audio is None or len(audio) == 0:
        try:
            cmd = ["ffmpeg", "-v", "error", "-i", str(video_path), "-vn", "-ac", "1", "-ar", "16000", "-f", "s16le", "-"]
            raw_bytes = subprocess.check_output(cmd, creationflags=NO_WINDOW, timeout=20, stderr=subprocess.DEVNULL)
            audio = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        except Exception:
            audio = None
    if audio is None or len(audio) == 0 or float(np.abs(audio).max()) < 1e-6:
        return np.zeros((1, 128, 128), dtype=np.float32), False
    audio = np.tile(audio, int(np.ceil(n_samples / len(audio))))[:n_samples] if len(audio) < n_samples else audio[:n_samples]
    with torch.no_grad():
        t = torch.from_numpy(np.ascontiguousarray(audio))
        stft = torch.stft(t, n_fft=512, hop_length=256, win_length=512, window=torch.hann_window(512), return_complex=True)
        mag = torch.log1p(stft.abs())
        spec = torch.nn.functional.interpolate(mag[None, None], size=(128, 128), mode="bilinear", align_corners=False)
        spec = ((spec - spec.mean()) / (spec.std() + 1e-6)).clamp(-3.0, 3.0) / 3.0
        return spec.squeeze(0).numpy().astype(np.float32), True


# =====================================================================================
# SYNTHETIC DEGRADATION (from 30-9-26)
# =====================================================================================
def random_degradation_spec(rng, exclude=()):
    choices = [d for d in DEGRADATION_TYPES if d not in exclude] or DEGRADATION_TYPES
    deg = choices[int(rng.integers(len(choices)))]
    levels = FRAME_DROP_LEVELS if deg == "frame_drop" else SEVERITIES
    return deg, levels[int(rng.integers(len(levels)))]


def synth_degrade(clip, deg, sev, rng):
    x = np.array(clip, dtype=np.uint8, copy=True)
    lvl = SEVERITY_LEVEL.get(sev, 1)
    if deg == "noise":
        x = np.clip(x.astype(np.float32) + rng.normal(0, (5.0, 10.0, 18.0)[lvl], x.shape), 0, 255).astype(np.uint8)
    elif deg == "blur":
        x = np.stack([cv2.GaussianBlur(f, (0, 0), (0.8, 1.4, 2.2)[lvl]) for f in x])
    elif deg == "brightness":
        d = (25, 45, 70)[lvl] * (1 if rng.random() < 0.5 else -1)
        x = np.clip(x.astype(np.int16) + d, 0, 255).astype(np.uint8)
    elif deg == "contrast":
        a = (0.75, 0.55, 0.35)[lvl] if rng.random() < 0.7 else (1.25, 1.45, 1.7)[lvl]
        m = float(x.mean())
        x = np.clip((x.astype(np.float32) - m) * a + m, 0, 255).astype(np.uint8)
    elif deg == "saturation":
        s = (0.6, 0.3, 0.05)[lvl] if rng.random() < 0.7 else (1.5, 2.0, 2.5)[lvl]
        out = []
        for f in x:
            hsv = cv2.cvtColor(f, cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[..., 1] = np.clip(hsv[..., 1] * s, 0, 255)
            out.append(cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB))
        x = np.stack(out)
    elif deg == "compression":
        out = []
        for f in x:
            ok, enc = cv2.imencode(".jpg", cv2.cvtColor(f, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, (45, 25, 10)[lvl]])
            out.append(cv2.cvtColor(cv2.imdecode(enc, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB) if ok else f)
        x = np.stack(out)
    elif deg == "motion_blur":
        k = (3, 5, 9)[lvl]
        kernel = np.zeros((k, k), dtype=np.float32)
        direction = int(rng.integers(3))
        if direction == 0:
            kernel[k // 2, :] = 1.0
        elif direction == 1:
            kernel[:, k // 2] = 1.0
        else:
            np.fill_diagonal(kernel, 1.0)
        kernel /= kernel.sum()
        x = np.stack([cv2.filter2D(f, -1, kernel) for f in x])
    elif deg == "frame_drop":
        T = len(x)
        drop = rng.random(T) < (0.05, 0.10, 0.30)[lvl]
        drop[0] = False
        if T > 1 and not drop.any():
            drop[int(rng.integers(1, T))] = True
        for t in range(1, T):
            if drop[t]:
                x[t] = x[t - 1]
    return x


# =====================================================================================
# DATASET INDEXING & LEAKAGE-FREE SPLITS (from 30-9-26; split is identical for the same seed)
# =====================================================================================
def yt_group(name):
    mm = re.match(r"^(.*)_\d{4}_S\d+_E\d+", name)
    return mm.group(1) if mm else name


class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, a):
        self.parent.setdefault(a, a)
        while self.parent[a] != a:
            self.parent[a] = self.parent[self.parent[a]]
            a = self.parent[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def ffpp_identity_tokens(stem, subset):
    if subset == "DeepFakeDetection":
        mm = re.match(r"^(\d+)_(\d+)__", stem)
        return [f"dfd:{mm.group(1)}_{mm.group(2)}"] if mm else [f"dfd:{stem}"]
    if subset == "original":
        return [f"ff:{stem}"]
    parts = stem.split("_")
    return [f"ff:{p}" for p in parts[:2]] if len(parts) >= 2 else [f"ff:{stem}"]


def discover_sources(args, rng):
    datasets = {d.strip().lower() for d in args.datasets.split(",") if d.strip()}
    folders = []
    if "bio" in datasets:
        bio = Path(args.bio_root)
        folders += [("BioDeepAV", "real", 0, bio / "real" / "videos"), ("BioDeepAV", "fake", 1, bio / "fake" / "videos")]
    if "ffpp" in datasets:
        ff = Path(args.ffpp_root)
        for sub in [s.strip() for s in args.ffpp_folders.split(",") if s.strip()]:
            folders.append(("FF++", sub, 0 if sub.lower() == "original" else 1, ff / sub))
    sources = []
    for ds, sub, label, folder in folders:
        if not folder.exists():
            print(f"  [WARN] Missing folder, skipped: {folder}")
            continue
        files = sorted(folder.glob("*.mp4"))
        if args.max_per_folder and len(files) > args.max_per_folder:
            files = sorted(rng.sample(files, args.max_per_folder))
        for p in files:
            sources.append(dict(key=f"{ds}/{sub}/{p.name}", dataset=ds, subset=sub, label=label, path=p, folder=folder,
                                wavs=[folder / (p.stem + ".wav"), folder.parent / "audios" / (p.stem + ".wav")]))
    uf = UnionFind()
    for s in sources:
        if s["dataset"] == "FF++":
            toks = ffpp_identity_tokens(s["path"].stem, s["subset"])
            for t in toks[1:]:
                uf.union(toks[0], t)
    for s in sources:
        if s["dataset"] == "FF++":
            s["group"] = "FF++|" + uf.find(ffpp_identity_tokens(s["path"].stem, s["subset"])[0])
        elif s["label"] == 0:
            s["group"] = "bio_real|" + yt_group(s["path"].name)
        else:
            s["group"] = "bio_fake|" + s["path"].stem
    return sources, folders


def discover_degraded(sources, folders, extra_manifests):
    lookup = {(str(s["folder"]).lower(), s["path"].name): i for i, s in enumerate(sources)}
    manifest_paths = [f / "degraded_output" / "manifest.json" for _, _, _, f in folders] + [Path(m) for m in extra_manifests]
    degraded, seen, summary = [], set(), []
    for m_path in manifest_paths:
        if not m_path.exists() or str(m_path).lower() in seen:
            continue
        seen.add(str(m_path).lower())
        deg_base = m_path.parent
        src_folder = str(deg_base.parent).lower()
        with open(m_path, "r", encoding="utf-8") as f:
            m_data = json.load(f)
        n_linked = n_missing = 0
        for item in m_data.get("files", []):
            si = lookup.get((src_folder, Path(item["source"]).name))
            if si is None:
                continue
            for out in item.get("outputs", []):
                p = deg_base / out["output"]
                if not p.exists():
                    n_missing += 1
                    continue
                degraded.append(dict(src=si, path=p, degradation=out.get("degradation", "unknown"),
                                     severity=out.get("severity", out.get("level", "")), set=out.get("randomSet", "")))
                n_linked += 1
        summary.append(dict(manifest=str(m_path), linked_outputs=n_linked, missing_files=n_missing))
    return degraded, summary


def make_splits(sources, test_ratio, val_ratio, seed):
    """Grouped split by identity group (unchanged from 30-9-26, so the test set is identical)."""
    groups = defaultdict(list)
    for i, s in enumerate(sources):
        groups[s["group"]].append(i)
    strata = defaultdict(list)
    for g, members in groups.items():
        labels = {sources[i]["label"] for i in members}
        ds = sources[members[0]]["dataset"]
        strata[(ds, labels.pop()) if len(labels) == 1 else (ds, "mixed")].append(g)
    rng = random.Random(seed)
    split = np.empty(len(sources), dtype=object)
    for key in sorted(strata, key=str):
        gs = sorted(strata[key])
        rng.shuffle(gs)
        n = sum(len(groups[g]) for g in gs)
        test_t, val_t = int(round(n * test_ratio)), int(round(n * val_ratio))
        targets = {"test": test_t, "val": val_t, "train": n - test_t - val_t}
        filled = {"test": 0, "val": 0, "train": 0}
        for g in sorted(gs, key=lambda g: -len(groups[g])):
            name = max(targets, key=lambda k: (targets[k] - filled[k]) / max(1, targets[k]))
            filled[name] += len(groups[g])
            for i in groups[g]:
                split[i] = name
    return split


# =====================================================================================
# CACHES (memory-mapped, resumable, reusable across runs)
# =====================================================================================
class ClipCache:
    """uint8 [N,T,S,S,3] memmap + per-item status (0 pending, 1 ok, 2 failed) + optional face info."""

    INFO_FIELDS = ("boxes", "angles", "det", "meta")

    def __init__(self, name, key_src, n_items, n_frames, size, cache_dir: Path, search_dirs: Sequence[Path] = (), with_info=False):
        self.key = hashlib.md5(key_src.encode("utf-8")).hexdigest()[:10]
        self.fname = f"cache_{name}_T{n_frames}_S{size}_n{n_items}_{self.key}"
        self.name, self.n, self.T, self.S, self.with_info = name, n_items, n_frames, size, with_info
        self.cache_dir, self.search_dirs = cache_dir, [Path(d) for d in search_dirs]
        self.arr, self.status, self.info = None, None, None

    def _paths(self, d: Path):
        return d / f"{self.fname}.npy", d / f"{self.fname}.status.npy", d / f"{self.fname}.info.npz"

    def _empty_info(self):
        return dict(boxes=np.zeros((self.n, self.T, 3), np.float32), angles=np.zeros((self.n, self.T), np.float32),
                    det=np.full((self.n, self.T), -1, np.int8), meta=np.full((self.n, 8), np.nan, np.float32))

    def find_complete(self) -> bool:
        for d in [self.cache_dir] + self.search_dirs:
            a, s, i = self._paths(d)
            if a.exists() and s.exists() and (not self.with_info or i.exists()):
                status = np.load(s)
                if len(status) == self.n and (status != 0).all():
                    self.arr, self.status = np.load(a, mmap_mode="r"), status
                    if self.with_info:
                        z = np.load(i)
                        self.info = {k: z[k] for k in self.INFO_FIELDS}
                    print(f"  -> Found cached {self.name.upper()}: {a} ({int((status == 1).sum())} ok, {int((status == 2).sum())} failed)")
                    return True
        return False

    def build(self, work_fn: Callable[[int], Tuple[Optional[np.ndarray], Optional[dict]]], n_workers: int, label: str = ""):
        """work_fn(i) -> (clip, info). Resumes a partially built cache in cache_dir."""
        if self.find_complete():
            return self
        a_path, s_path, i_path = self._paths(self.cache_dir)
        shape = (self.n, self.T, self.S, self.S, 3)
        if a_path.exists() and s_path.exists():
            status = np.load(s_path)
            arr = np.lib.format.open_memmap(a_path, mode="r+")
            if self.with_info and i_path.exists():
                z = np.load(i_path)
                info = {k: z[k] for k in z.files}
            else:
                info = self._empty_info()
        else:
            status = np.zeros(self.n, dtype=np.int8)
            arr = np.lib.format.open_memmap(a_path, mode="w+", dtype=np.uint8, shape=shape)
            info = self._empty_info()
        todo = np.where(status == 0)[0]
        print(f"\n[Extracting {label or self.name.upper()}] {len(todo)} of {self.n} videos remaining | {n_workers} threads -> {a_path.name}")

        def save():
            arr.flush()
            np.save(s_path, status)
            if self.with_info:
                np.savez(i_path, **info)

        t0 = time.time()
        last_print = last_save = 0
        done = 0
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            futures = {ex.submit(work_fn, int(i)): int(i) for i in todo}
            for fut in as_completed(futures):
                i = futures[fut]
                try:
                    clip, inf = fut.result()
                except Exception as e:
                    clip, inf = None, None
                if clip is not None and clip.shape == shape[1:]:
                    arr[i] = clip
                    status[i] = 1
                else:
                    arr[i] = 0
                    status[i] = 2
                if self.with_info and inf is not None:
                    T = min(self.T, len(inf["boxes"]))
                    info["boxes"][i, :T] = inf["boxes"][:T]
                    info["angles"][i, :T] = inf["angles"][:T]
                    info["det"][i, :T] = inf["det"][:T] if inf["detected"] >= 0 else -1
                    info["meta"][i] = [inf["width"], inf["height"], inf["fps"], inf["total_frames"], float(inf["fallback"]),
                                       inf["face_rel_size"], inf["detected"], inf["frames"]]
                done += 1
                now = time.time()
                if now - last_save >= 30:
                    save()
                    last_save = now
                if now - last_print >= 0.2 or done == len(todo):
                    rate = done / max(0.001, now - t0)
                    sys.stdout.write(f"\r  {self.name:16s}: {progress_bar(done, len(todo))} ({done}/{len(todo)}) | {rate:5.1f} vid/s | "
                                     f"ETA {format_time((len(todo) - done) / max(0.001, rate))}   ")
                    sys.stdout.flush()
                    last_print = now
        sys.stdout.write("\n")
        save()
        del arr
        self.arr, self.status = np.load(a_path, mmap_mode="r"), status
        self.info = info if self.with_info else None
        print(f"  Saved {a_path.name} ({a_path.stat().st_size / 1e9:.2f} GB, {int((status == 2).sum())} failed)")
        return self

    @property
    def ok(self) -> np.ndarray:
        return self.status == 1


def build_audio_cache(sources, cache_dir: Path, search_dirs, n_workers):
    """float32 [N,1,128,128] spectrograms + has_audio (same file format/key as 30-9-26, so it is reused)."""
    key = hashlib.md5("\n".join(str(s["path"]) for s in sources).encode("utf-8")).hexdigest()[:10]
    fname = f"cache_audio_n{len(sources)}_{key}"
    for d in [cache_dir] + [Path(x) for x in search_dirs]:
        a, s = d / f"{fname}.npy", d / f"{fname}.status.npy"
        if a.exists() and s.exists():
            status = np.load(s)
            if (status != 0).all():
                print(f"  -> Found cached audio spectrograms: {a}")
                return np.load(a, mmap_mode="r"), status == 1
    a_path, s_path = cache_dir / f"{fname}.npy", cache_dir / f"{fname}.status.npy"
    if a_path.exists() and s_path.exists():
        status, arr = np.load(s_path), np.lib.format.open_memmap(a_path, mode="r+")
    else:
        status = np.zeros(len(sources), dtype=np.int8)
        arr = np.lib.format.open_memmap(a_path, mode="w+", dtype=np.float32, shape=(len(sources), 1, 128, 128))
    todo = np.where(status == 0)[0]
    print(f"  -> Extracting audio spectrograms for {len(todo)} videos ({n_workers} threads)...")
    t0, done, last = time.time(), 0, 0
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futures = {ex.submit(extract_audio_spectrogram, sources[i]["wavs"], sources[i]["path"]): int(i) for i in todo}
        for fut in as_completed(futures):
            i = futures[fut]
            try:
                spec, has = fut.result()
            except Exception:
                spec, has = np.zeros((1, 128, 128), np.float32), False
            arr[i] = spec
            status[i] = 1 if has else 3
            done += 1
            if time.time() - last > 0.2 or done == len(todo):
                sys.stdout.write(f"\r  audio: {progress_bar(done, len(todo))} ({done}/{len(todo)})   ")
                sys.stdout.flush()
                last = time.time()
    sys.stdout.write("\n")
    arr.flush()
    np.save(s_path, status)
    del arr
    return np.load(a_path, mmap_mode="r"), status == 1


# =====================================================================================
# METRICS, THRESHOLDS
# =====================================================================================
def compute_metrics(y_true, y_prob, threshold: float = DEFAULT_THRESHOLD) -> Dict[str, Any]:
    """All classification metrics at a FIXED threshold. Undefined values are NaN (never errors)."""
    y = np.asarray(y_true).astype(int)
    p = np.asarray(y_prob, dtype=np.float64)
    keep = ~np.isnan(p)
    y, p = y[keep], p[keep]
    out = dict(n=int(len(y)), n_real=int((y == 0).sum()), n_fake=int((y == 1).sum()), threshold=float(threshold))
    if len(y) == 0:
        out.update({k: float("nan") for k in METRIC_KEYS}, tn=0, fp=0, fn=0, tp=0)
        return out
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = (int(v) for v in confusion_matrix(y, pred, labels=[0, 1]).ravel())
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")  # TN / (TN + FP), safe denominator
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (not np.isnan(recall) and precision + recall > 0) else 0.0
    bal = (recall + specificity) / 2 if not (np.isnan(recall) or np.isnan(specificity)) else float("nan")
    auc = float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")
    out.update(accuracy=float((pred == y).mean()), balanced_accuracy=float(bal), precision=float(precision), recall=float(recall),
               specificity=float(specificity), f1=float(f1), auc=auc, tn=tn, fp=fp, fn=fn, tp=tp)
    return out


def threshold_table(y_true, y_prob, thresholds: Sequence[float]) -> pd.DataFrame:
    rows = []
    for t in thresholds:
        m = compute_metrics(y_true, y_prob, float(t))
        rows.append({"threshold": round(float(t), 4), **{k: m[k] for k in ["accuracy", "balanced_accuracy", "precision", "recall", "specificity", "f1"]}})
    return pd.DataFrame(rows)


def find_optimal_threshold(y_true, y_prob, metric: str = "balanced_accuracy",
                           grid: Tuple[float, float, float] = THRESHOLD_GRID) -> Dict[str, float]:
    """Searches thresholds on VALIDATION predictions only. Primary metric (default balanced accuracy),
    ties broken by F1, then by closeness to 0.5."""
    lo, hi, step = grid
    thresholds = np.round(np.arange(lo, hi + step / 2, step), 4)
    tab = threshold_table(y_true, y_prob, thresholds)
    if metric == "youden":
        tab["youden"] = tab["recall"] + tab["specificity"] - 1
    key = metric if metric in tab.columns else "balanced_accuracy"
    tab = tab.assign(_k=tab[key].fillna(-1), _f1=tab["f1"].fillna(-1), _d=-(tab["threshold"] - 0.5).abs())
    best = tab.sort_values(["_k", "_f1", "_d"], ascending=False).iloc[0]
    return {"threshold": float(best["threshold"]), "metric": key,
            **{k: float(best[k]) for k in ["balanced_accuracy", "accuracy", "precision", "recall", "specificity", "f1"]}}


def log_loss(y, p):
    p = np.clip(np.asarray(p, np.float64), 1e-7, 1 - 1e-7)
    y = np.asarray(y)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))) if len(y) else float("nan")


def nan0(v, default=0.5):
    return default if (v is None or np.isnan(v) or np.isinf(v)) else v


# =====================================================================================
# CHECKPOINTS
# =====================================================================================
def save_checkpoint(path: Path, model, optimizer, epoch, best_val_metric, threshold, stage, config: dict, extra: dict = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "epoch": epoch,
        "best_val_metric": best_val_metric,
        "threshold": threshold,
        "img_size": config.get("img_size"),
        "stage": stage,
        "seed": config.get("seed"),
        "config": config,
        "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    if extra:
        ckpt.update(extra)
    torch.save(ckpt, path)


def load_checkpoint(path: Path, device) -> Tuple[nn.Module, dict]:
    """Loads 1-10-26 checkpoints (dict with metadata) and 30-9-26 checkpoints (raw state_dict, 96x96, no threshold)."""
    obj = torch.load(path, map_location=device, weights_only=False)
    model = AudioVisualDeepFakeCNN().to(device)
    if isinstance(obj, dict) and "model_state_dict" in obj:
        model.load_state_dict(obj["model_state_dict"])
        meta = {k: v for k, v in obj.items() if k not in ("model_state_dict", "optimizer_state_dict")}
    else:
        model.load_state_dict(obj)
        meta = dict(threshold=None, img_size=96, stage=None, config=dict(img_size=96, face_crop=False, modality="av"), legacy=True)
    model.eval()
    return model, meta


# =====================================================================================
# GRAD-CAM (same target layer as 30-9-26; resolution independent)
# =====================================================================================
def grad_cam(model: nn.Module, video: torch.Tensor, audio: Optional[torch.Tensor] = None, a_mask: Optional[torch.Tensor] = None,
             target_class: int = 1, threshold: float = DEFAULT_THRESHOLD) -> Dict[str, np.ndarray]:
    """video: [B,T,3,H,W] normalised. Returns spatial CAM [B,H,W] (temporal mean, min-max normalised),
    temporal CAM [B,T] (per-frame CAM mass, sums to 1), frame CAMs [B,T,H,W], p_fake [B], prediction [B]."""
    store = {}

    def fwd_hook(_m, _i, o):
        store["a"] = o
        o.register_hook(lambda g: store.__setitem__("g", g))

    B, T, _, H, W = video.shape
    if audio is None:
        audio = torch.zeros(B, 1, 128, 128, device=video.device)
        a_mask = torch.zeros(B, device=video.device)
    h = model.video_net.features[-1].register_forward_hook(fwd_hook)
    try:
        model.eval()
        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            logits = model(video, audio, mode="av", a_mask=a_mask)
            logits[:, target_class].sum().backward()
    finally:
        h.remove()
    A, G = store["a"].detach(), store["g"].detach()
    cam = torch.relu((G.mean((2, 3), keepdim=True) * A).sum(1))  # [B*T, h, w]
    cam = torch.nn.functional.interpolate(cam[:, None], size=(H, W), mode="bilinear", align_corners=False)[:, 0].view(B, T, H, W)
    temporal = cam.flatten(2).sum(-1)
    temporal = temporal / (temporal.sum(1, keepdim=True) + 1e-8)
    spatial = cam.mean(1).flatten(1)
    mn, mx = spatial.min(1, keepdim=True)[0], spatial.max(1, keepdim=True)[0]
    spatial = ((spatial - mn) / (mx - mn + 1e-8)).view(B, H, W)
    prob = torch.softmax(logits.detach().float(), 1)[:, 1]
    model.zero_grad(set_to_none=True)
    return dict(spatial_cam=spatial.cpu().numpy(), temporal_cam=temporal.cpu().numpy(), frame_cams=cam.cpu().numpy(),
                probability_fake=prob.cpu().numpy(), prediction=(prob >= threshold).int().cpu().numpy())


def cam_metrics(a, b, top=0.2):
    r = float(np.corrcoef(a.ravel(), b.ravel())[0, 1]) if a.std() > 1e-8 and b.std() > 1e-8 else float("nan")
    ta, tb = a >= np.quantile(a, 1 - top), b >= np.quantile(b, 1 - top)
    iou = float((ta & tb).sum() / max(1, (ta | tb).sum()))
    yy, xx = np.mgrid[0 : a.shape[0], 0 : a.shape[1]]

    def com(c):
        w = c.sum() + 1e-8
        return np.array([(yy * c).sum() / w, (xx * c).sum() / w])

    return r, iou, float(np.linalg.norm(com(a) - com(b)) / a.shape[0])


# =====================================================================================
# INFERENCE API FOR THE NEXT (XAI) PHASE
# =====================================================================================
class DeepfakeDetector:
    """Frozen detector: model + threshold + preprocessing restored from the checkpoint.

        det = DeepfakeDetector("run_1-10-26/results/final/final_model.pt")
        result = det.predict_video("video.mp4")
    """

    def __init__(self, checkpoint: str, device: Optional[str] = None, face_model: Optional[str] = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model, self.meta = load_checkpoint(Path(checkpoint), self.device)
        cfg = self.meta.get("config", {})
        self.img_size = int(self.meta.get("img_size") or cfg.get("img_size", IMG_SIZE))
        self.n_frames = int(cfg.get("frames", NUM_FRAMES))
        self.clip_len = int(cfg.get("clip_len", CLIP_LEN))
        self.modality = cfg.get("modality", "av")
        self.face_crop = bool(cfg.get("face_crop", False))
        self.margin = float(cfg.get("face_margin", FACE_MARGIN))
        self.align = bool(cfg.get("face_align", False))
        self.threshold = float(self.meta["threshold"]) if self.meta.get("threshold") is not None else DEFAULT_THRESHOLD
        self.stage = self.meta.get("stage")
        fm = ensure_face_model(face_model, Path(__file__).resolve().parent) if self.face_crop else None
        self.detector = FaceDetector(fm, float(cfg.get("face_score", FACE_SCORE_THRESHOLD))) if self.face_crop else None

    @torch.no_grad()
    def predict_video(self, video_path: str, wav_path: Optional[str] = None) -> Dict[str, Any]:
        vp = Path(video_path)
        if self.face_crop:
            clip, info = extract_face_clip(vp, self.n_frames, self.img_size, self.detector, self.margin, self.align)
        else:
            clip, info = extract_video_clip(vp, self.n_frames, self.img_size), {}
        if clip is None:
            raise RuntimeError(f"Could not decode video: {video_path}")
        wavs = [Path(wav_path)] if wav_path else [vp.with_suffix(".wav"), vp.parent.parent / "audios" / (vp.stem + ".wav")]
        spec, has_audio = extract_audio_spectrogram(wavs, vp) if self.modality == "av" else (np.zeros((1, 128, 128), np.float32), False)
        xa = torch.from_numpy(spec[None]).to(self.device)
        am = torch.tensor([1.0 if has_audio else 0.0], device=self.device)
        stride = max(1, self.n_frames // self.clip_len)
        p = 0.0
        for v in range(stride):
            xv = to_video_tensor(clip[None, v::stride][:, : self.clip_len], self.device)
            p += float(torch.softmax(self.model(xv, xa, mode="av", a_mask=am).float(), 1)[0, 1])
        prob = p / stride
        return {"probability_fake": prob, "prediction": int(prob >= self.threshold), "label": "FAKE" if prob >= self.threshold else "REAL",
                "threshold": self.threshold, "model_stage": self.stage, "img_size": self.img_size, "has_audio": bool(has_audio),
                "face_detection_rate": info.get("detection_rate"), "face_fallback": info.get("fallback")}


# =====================================================================================
# EXPERIMENT
# =====================================================================================
class View:
    """One preprocessing variant of the data (e.g. 128 face crops, or 96 full frames for the baseline)."""

    def __init__(self, name, img_size, clean: ClipCache, deg: Optional[ClipCache], face_crop: bool):
        self.name, self.img_size, self.clean, self.deg, self.face_crop = name, img_size, clean, deg, face_crop


class Experiment:
    def __init__(self, args):
        self.args = args
        self.script_dir = Path(__file__).resolve().parent
        self.work_dir = Path(args.work_dir) if args.work_dir else self.script_dir / "run_1-10-26"
        self.res = self.work_dir / "results"
        self.cache_dir = self.work_dir / "cache"
        self.baseline_dir = Path(args.baseline_dir) if args.baseline_dir else self.script_dir / "run_30-9-26"
        for d in [self.cache_dir] + [self.res / s / sub for s in ("stage1", "stage2") for sub in
                                     ("checkpoints", "metrics", "predictions", "confusion_matrices", "threshold")] + \
                 [self.res / d for d in ("robustness", "cross_dataset", "resolution_comparison", "plots", "diagnostics", "xai", "final", "logs")]:
            d.mkdir(parents=True, exist_ok=True)
        set_seed(args.seed, args.deterministic)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_workers = args.workers or (os.cpu_count() or 8)
        self.T, self.L = args.frames, args.clip_len
        self.stride = max(1, self.T // self.L)
        self.ckpt = {1: self.res / "stage1" / "checkpoints" / "stage1_best.pt", 2: self.res / "stage2" / "checkpoints" / "stage2_best.pt"}
        self.status = defaultdict(lambda: "Not Completed")

    # ------------------------------------------------------------------ config
    def config(self) -> dict:
        a = self.args
        return dict(img_size=a.img_size, frames=a.frames, clip_len=a.clip_len, modality=a.modality, audio_dropout=a.audio_dropout,
                    face_crop=a.face_crop, face_align=a.face_align, face_margin=a.face_margin, face_score=a.face_score,
                    face_backend=self.face_detector.backend if a.face_crop else "none", deg_face_boxes=a.deg_face_boxes,
                    sampling=a.sampling, max_oversample=a.max_oversample, epoch_size=a.epoch_size, batch_size=a.batch_size,
                    lr1=a.lr1, lr2=a.lr2, epochs1=a.epochs1, epochs2=a.epochs2, patience=a.patience, synth_per_video=a.synth_per_video,
                    threshold_metric=a.threshold_metric, optimize_threshold=a.optimize_threshold, threshold_grid=THRESHOLD_GRID,
                    seed=a.seed, deterministic=a.deterministic, datasets=a.datasets, ffpp_folders=a.ffpp_folders,
                    max_per_folder=a.max_per_folder, test_ratio=a.test_ratio, val_ratio=a.val_ratio,
                    model_selection="macro per-dataset validation AUC")

    # ------------------------------------------------------------------ data
    def prepare_data(self):
        a = self.args
        print("=" * 96)
        print(" 1-10-26: FACE-CROP / BALANCED / THRESHOLD-CALIBRATED DEGRADATION-AWARE DEEPFAKE DETECTOR")
        print(f"  * Mode {a.mode.upper()} | device {self.device}" + (f" ({torch.cuda.get_device_name(0)})" if self.device.type == "cuda" else ""))
        print(f"  * IMG_SIZE {a.img_size} | face crop {a.face_crop} (align {a.face_align}, margin {a.face_margin}) | sampling {a.sampling} | modality {a.modality}")
        print(f"  * Work dir {self.work_dir}")
        print("=" * 96)
        self.face_detector = FaceDetector(ensure_face_model(a.face_model, self.script_dir) if a.face_crop else None, a.face_score)
        if a.face_crop and self.face_detector.backend == "none":
            print("  [WARN] No face detector available -> every video uses the full-frame fallback.")
        write_json(self.work_dir / "experiment_config.json", self.config())

        self.sources, self.folders = discover_sources(a, random.Random(a.seed))
        if not self.sources:
            raise RuntimeError("No source videos found (check --bio-root / --ffpp-root / --datasets).")
        self.degraded, self.manifest_summary = discover_degraded(self.sources, self.folders, a.extra_manifests)
        split = make_splits(self.sources, a.test_ratio, a.val_ratio, a.seed)
        self.labels = np.array([s["label"] for s in self.sources], dtype=np.int64)
        self.ds_of = np.array([s["dataset"] for s in self.sources])
        self.subset_of = np.array([f"{s['dataset']}/{s['subset']}" for s in self.sources])
        self.deg_src = np.array([d["src"] for d in self.degraded], dtype=np.int64)

        print("\n--- Caches ---")
        src_paths = [str(s["path"]) for s in self.sources]
        deg_paths = [str(d["path"]) for d in self.degraded]
        search = [self.baseline_dir / "cache"]
        if a.face_crop:
            face_key = f"|face|m{a.face_margin}|a{int(a.face_align)}|s{a.face_score}|{self.face_detector.backend}"
            clean = ClipCache("faceclean", "\n".join(src_paths) + f"|{self.T}|{a.img_size}" + face_key, len(src_paths), self.T, a.img_size,
                              self.cache_dir, search, with_info=True)
            det = self.face_detector
            clean.build(lambda i: extract_face_clip(self.sources[i]["path"], self.T, a.img_size, det, a.face_margin, a.face_align),
                        self.n_workers, "ORIGINAL VIDEOS (face crop)")
            deg = None
            if self.degraded:
                deg = ClipCache("facedeg", "\n".join(deg_paths) + f"|{self.T}|{a.img_size}" + face_key + f"|boxes-{a.deg_face_boxes}",
                                len(deg_paths), self.T, a.img_size, self.cache_dir, search, with_info=True)

                def deg_work(j):
                    s = int(self.deg_src[j])
                    if not clean.ok[s]:
                        return None, None
                    if a.deg_face_boxes == "source":  # same crop region as the source video -> aligned original/degraded pairs
                        ref = (clean.info["boxes"][s], clean.info["angles"][s], bool(clean.info["meta"][s, 4] > 0.5))
                        return extract_face_clip(self.degraded[j]["path"], self.T, a.img_size, None, a.face_margin, a.face_align, ref=ref)
                    return extract_face_clip(self.degraded[j]["path"], self.T, a.img_size, det, a.face_margin, a.face_align)

                deg.build(deg_work, self.n_workers, "DEGRADED VIDEOS (face crop)")
        else:
            clean = ClipCache("clean", "\n".join(src_paths) + f"|{self.T}|{a.img_size}", len(src_paths), self.T, a.img_size, self.cache_dir, search)
            clean.build(lambda i: (extract_video_clip(self.sources[i]["path"], self.T, a.img_size), None), self.n_workers)
            deg = None
            if self.degraded:
                deg = ClipCache("degraded", "\n".join(deg_paths) + f"|{self.T}|{a.img_size}", len(deg_paths), self.T, a.img_size, self.cache_dir, search)
                deg.build(lambda j: (extract_video_clip(self.degraded[j]["path"], self.T, a.img_size), None), self.n_workers)
        self.view = View(f"{a.img_size}{'_face' if a.face_crop else '_full'}", a.img_size, clean, deg, a.face_crop)
        print("\n--- Audio ---")
        self.audio, self.has_audio = build_audio_cache(self.sources, self.cache_dir, search, self.n_workers)

        # Degraded versions inherit their source's split (leakage rule); failed extractions are dropped.
        self.split = np.where(clean.ok, split, "dropped")
        deg_ok = deg.ok if deg is not None else np.zeros(0, bool)
        self.src_to_deg = defaultdict(list)
        for j in range(len(self.degraded)):
            if deg_ok[j] and clean.ok[self.deg_src[j]]:
                self.src_to_deg[int(self.deg_src[j])].append(j)
        self.idx = {k: np.where(self.split == k)[0] for k in ("train", "val", "test")}
        write_json(self.work_dir / "split_1-10-26.json", {k: [self.sources[i]["key"] for i in v] for k, v in self.idx.items()})
        self.verify_leakage()
        self.report_inventory()
        if a.face_crop:
            self.report_face_stats()
        self.build_eval_sets()

    def verify_leakage(self):
        """Rule 2: no identity group and no source/degraded pair crosses train/val/test."""
        grp_splits = defaultdict(set)
        for s, sp in zip(self.sources, self.split):
            if sp != "dropped":
                grp_splits[s["group"]].add(sp)
        bad_groups = sum(len(v) > 1 for v in grp_splits.values())
        bad_pairs = 0  # degraded items are indexed through their source, so they cannot leave its split; verified explicitly:
        for s, js in self.src_to_deg.items():
            for j in js:
                bad_pairs += int(self.split[self.deg_src[j]] != self.split[s])
        ff_ids = defaultdict(set)
        for s, sp in zip(self.sources, self.split):
            if s["dataset"] == "FF++" and sp != "dropped":
                for t in ffpp_identity_tokens(s["path"].stem, s["subset"]):
                    ff_ids[t].add(sp)
        bad_ids = sum(len(v) > 1 for v in ff_ids.values())
        self.leakage = dict(groups=len(grp_splits), groups_in_multiple_splits=bad_groups, degraded_pairs_crossing_splits=bad_pairs,
                            ffpp_identities_in_multiple_splits=bad_ids)
        print(f"\n[Leakage check] groups {len(grp_splits):,} | groups crossing splits {bad_groups} | "
              f"degraded/source pairs crossing splits {bad_pairs} | FF++ identities crossing splits {bad_ids}")
        if bad_groups or bad_pairs or bad_ids:
            raise RuntimeError(f"Data leakage detected: {self.leakage}")

    def report_inventory(self):
        rows = []
        for ds, sub, label, folder in self.folders:
            idx = [i for i, s in enumerate(self.sources) if s["folder"] == folder]
            if not idx:
                continue
            rows.append({"Folder": f"{ds}/{sub}", "Label": "REAL" if label == 0 else "FAKE", "Original": len(idx),
                         "Degraded": sum(len(self.src_to_deg[i]) for i in idx), "Audio": int(self.has_audio[idx].sum()),
                         **{k.capitalize(): int(sum(self.split[i] == k for i in idx)) for k in ("train", "val", "test", "dropped")}})
        self.df_inv = pd.DataFrame(rows).set_index("Folder")
        print("\n" + "=" * 96 + "\n [DATASET INVENTORY]\n" + "=" * 96)
        print(self.df_inv.to_string())
        for m in self.manifest_summary:
            print(f"  * {m['manifest']} -> {m['linked_outputs']:,} degraded videos ({m['missing_files']} missing)")
        print(f"  * split: train {len(self.idx['train']):,} | val {len(self.idx['val']):,} | test {len(self.idx['test']):,}")

    def report_face_stats(self):
        info = self.view.clean.info
        meta = info["meta"]
        rows = []
        for i, s in enumerate(self.sources):
            frames, detected = meta[i, 7], meta[i, 6]
            rows.append(dict(video=s["key"], dataset=s["dataset"], subset=s["subset"], frames=frames, detected_faces=detected,
                             failed=frames - detected, detection_rate=detected / frames if frames > 0 else np.nan,
                             fallback_full_frame=bool(meta[i, 4] > 0.5), face_crop_rel_size=meta[i, 5],
                             width=meta[i, 0], height=meta[i, 1], fps=meta[i, 2], extraction_ok=bool(self.view.clean.ok[i])))
        df = pd.DataFrame(rows)
        df.to_csv(self.res / "logs" / "face_detection_per_video.csv", index=False)
        agg = df.groupby(["dataset", "subset"]).agg(videos=("video", "count"), frames=("frames", "sum"), detected=("detected_faces", "sum"),
                                                     failed=("failed", "sum"), fallback_videos=("fallback_full_frame", "sum"),
                                                     mean_crop_rel_size=("face_crop_rel_size", "mean"))
        agg["detection_rate_%"] = 100 * agg["detected"] / agg["frames"].clip(lower=1)
        tot = df.agg(dict(frames="sum", detected_faces="sum", failed="sum", fallback_full_frame="sum"))
        agg.to_csv(self.res / "logs" / "face_detection_summary.csv")
        self.face_summary = dict(per_subset=agg.reset_index().to_dict(orient="records"),
                                 total_frames=float(tot["frames"]), detected=float(tot["detected_faces"]),
                                 detection_rate=float(tot["detected_faces"] / max(1, tot["frames"])),
                                 fallback_videos=int(tot["fallback_full_frame"]))
        print("\n" + "=" * 96 + "\n [FACE DETECTION STATISTICS (original videos)]\n" + "=" * 96)
        print(agg.round(3).to_string())
        print(f"  * Overall: {int(tot['detected_faces']):,}/{int(tot['frames']):,} frames with a face "
              f"({100 * self.face_summary['detection_rate']:.2f}%), {int(tot['fallback_full_frame'])} videos fell back to full frames")
        ex = df[df.extraction_ok].sample(min(3, len(df)), random_state=self.args.seed)
        for _, r in ex.iterrows():
            print(f"    Video ID: {r.video} | Frames: {int(r.frames)} | Detected faces: {int(r.detected_faces)} | "
                  f"Failed: {int(r.failed)} | Detection rate: {100 * r.detection_rate:.2f}%")

    # ------------------------------------------------------------------ items
    # item = (kind, idx, spec): ("clean", src, None) | ("deg", deg_j, None) | ("synth", src, (type, severity, seed) or None)
    def item_src(self, it) -> int:
        return int(self.deg_src[it[1]]) if it[0] == "deg" else int(it[1])

    def item_degradation(self, it):
        if it[0] == "clean":
            return "original", ""
        if it[0] == "deg":
            d = self.degraded[it[1]]
            return d["degradation"], d["severity"]
        return it[2][0], it[2][1]

    def item_name(self, it):
        if it[0] == "deg":
            return self.degraded[it[1]]["path"].name
        base = self.sources[it[1]]["path"].name
        return base if it[0] == "clean" else f"{base} [synthetic {it[2][0]}_{it[2][1]}]"

    def get_clip(self, view: View, it, sel=None):
        if it[0] == "deg":
            c = view.deg.arr[it[1]] if sel is None else view.deg.arr[it[1], sel]
            return np.asarray(c)
        c = np.asarray(view.clean.arr[it[1]] if sel is None else view.clean.arr[it[1], sel])
        if it[0] == "synth":
            if it[2] is None:
                raise ValueError("synth item without spec")
            c = synth_degrade(c, it[2][0], it[2][1], np.random.default_rng(it[2][2]))
        return c

    def all_degraded_items(self, idx_list, seed_offset):
        items = []
        for s in idx_list:
            s = int(s)
            if self.src_to_deg[s]:
                items += [("deg", j, None) for j in self.src_to_deg[s]]
            elif self.args.synth_per_video > 0:
                rng_s = np.random.default_rng(self.args.seed * 100003 + seed_offset + s)
                used = []
                for _ in range(self.args.synth_per_video):
                    deg, sev = random_degradation_spec(rng_s, exclude=used)
                    used.append(deg)
                    items.append(("synth", s, (deg, sev, int(rng_s.integers(2**31)))))
        return items

    def random_pick(self, items, idx_list, seed):
        by_src = defaultdict(list)
        for pos, it in enumerate(items):
            by_src[self.item_src(it)].append(pos)
        rng = random.Random(seed)
        return np.array([rng.choice(by_src[int(s)]) if by_src[int(s)] else -1 for s in idx_list], dtype=np.int64)

    def build_eval_sets(self):
        """Fixed evaluation inputs (same degraded versions for every model, so comparisons are paired)."""
        a = self.args
        self.E = {}
        for split, off in (("val", 7), ("test", 2026)):
            idx = self.idx[split]
            all_deg = self.all_degraded_items(idx, off)
            pick = self.random_pick(all_deg, idx, a.seed + off)
            self.E[split] = dict(idx=idx, clean=[("clean", int(s), None) for s in idx], deg_all=all_deg, pick=pick)

    # ------------------------------------------------------------------ prediction
    def audio_inputs(self, src, modality, force_mask=None):
        xa = torch.from_numpy(np.asarray(self.audio[src])).to(self.device)
        if modality == "video" or force_mask == 0:
            am = np.zeros(len(src), np.float32)
        else:
            am = self.has_audio[src].astype(np.float32)
        return xa, torch.from_numpy(am).to(self.device)

    @torch.no_grad()
    def predict(self, model, view: View, items, modality="av", force_mask=None, bs=32) -> np.ndarray:
        model.eval()
        out = []
        for i in range(0, len(items), bs):
            chunk = items[i : i + bs]
            clips = np.stack([self.get_clip(view, it) for it in chunk])
            src = np.array([self.item_src(it) for it in chunk])
            xa, am = self.audio_inputs(src, modality, force_mask)
            p = 0
            for v in range(self.stride):
                logits = model(to_video_tensor(clips[:, v :: self.stride][:, : self.L], self.device), xa, mode="av", a_mask=am)
                p = p + torch.softmax(logits.float(), 1)[:, 1].cpu().numpy()
            out.append(p / self.stride)
        return np.concatenate(out) if out else np.zeros(0)

    def predict_sets(self, model, view, modality, splits=("val", "test")) -> dict:
        """Predictions on original + every degraded version, for val and test (cached per model)."""
        P = {}
        for split in splits:
            e = self.E[split]
            pc = self.predict(model, view, e["clean"], modality)
            pa = self.predict(model, view, e["deg_all"], modality)
            pd_ = np.where(e["pick"] >= 0, pa[np.maximum(e["pick"], 0)] if len(pa) else np.nan, np.nan)
            P[split] = dict(clean=pc, deg_all=pa, deg=pd_)
        return P

    # ------------------------------------------------------------------ sampling
    def sampling_weights(self, items, strategy) -> Tuple[np.ndarray, pd.DataFrame]:
        """Per-item sampling probabilities. dataset_class: every (dataset, label) cell gets an equal share;
        class: real/fake 50/50; none: uniform. A cell is never oversampled more than --max-oversample times
        its natural size per epoch (the freed share is redistributed to the other cells)."""
        src = np.array([self.item_src(it) for it in items])
        lab, ds = self.labels[src], self.ds_of[src]
        cells = [f"{d}|{'real' if l == 0 else 'fake'}" for d, l in zip(ds, lab)] if strategy == "dataset_class" else \
                [("real" if l == 0 else "fake") for l in lab] if strategy == "class" else ["all"] * len(items)
        cells = np.array(cells)
        names, counts = np.unique(cells, return_counts=True)
        n = len(items)
        epoch = self.args.epoch_size or n
        share = {c: 1.0 / len(names) for c in names}
        if strategy != "none":
            cap = {c: self.args.max_oversample * cnt / epoch for c, cnt in zip(names, counts)}
            for _ in range(10):  # water-filling with the oversampling cap
                over = {c for c in names if share[c] > cap[c] + 1e-12}
                if not over:
                    break
                free = sum(share[c] - cap[c] for c in over)
                for c in over:
                    share[c] = cap[c]
                rest = [c for c in names if c not in over and share[c] < cap[c]]
                tot = sum(share[c] for c in rest)
                for c in rest:
                    share[c] += free * share[c] / tot if tot > 0 else 0
        else:
            share = {c: cnt / n for c, cnt in zip(names, counts)}
        cnt_map = dict(zip(names, counts))
        w = np.array([share[c] / cnt_map[c] for c in cells])
        w = w / w.sum()
        dist = pd.DataFrame([{"cell": c, "items": int(cnt_map[c]), "natural_share_%": 100 * cnt_map[c] / n,
                              "sampled_share_%": 100 * share[c], "expected_samples_per_epoch": int(round(share[c] * epoch)),
                              "oversampling_x": share[c] * epoch / cnt_map[c]} for c in names])
        return w, dist

    # ------------------------------------------------------------------ training
    def make_batch(self, view, batch_items, seed, modality):
        a = self.args
        rng = np.random.default_rng(seed)
        B, S = len(batch_items), view.img_size
        frames = np.empty((B, self.L, S, S, 3), dtype=np.uint8)
        src = np.empty(B, dtype=np.int64)
        for b, it in enumerate(batch_items):
            sel = np.sort(rng.choice(self.T, self.L, replace=False))
            if it[0] == "deg":
                clip = np.asarray(view.deg.arr[it[1], sel])
            else:
                clip = np.asarray(view.clean.arr[it[1], sel])
                if it[0] == "synth":
                    deg, sev = random_degradation_spec(rng) if it[2] is None else it[2][:2]
                    clip = synth_degrade(clip, deg, sev, rng)
            if rng.random() < 0.5:
                clip = clip[:, :, ::-1, :]
            pad_px = max(2, S // 24)  # same relative jitter as 30-9-26 (4 px at 96)
            pad = np.pad(clip, ((0, 0), (pad_px, pad_px), (pad_px, pad_px), (0, 0)), mode="edge")
            dy, dx = rng.integers(0, 2 * pad_px + 1, 2)
            frames[b] = pad[:, dy : dy + S, dx : dx + S]
            src[b] = self.item_src(it)
        audio = np.asarray(self.audio[src]).astype(np.float32)
        if modality == "video":
            a_mask = np.zeros(B, np.float32)
        else:
            a_mask = (self.has_audio[src] & (rng.random(B) >= a.audio_dropout)).astype(np.float32)
        return frames, audio, a_mask, self.labels[src]

    def evaluate_val(self, model, view, val_sets, modality) -> dict:
        """Validation metrics. Score = mean over val sets of the MACRO per-dataset AUC."""
        row, scores = {}, []
        for vname, items in val_sets:
            if not items:
                continue
            src = np.array([self.item_src(it) for it in items])
            y, ds = self.labels[src], self.ds_of[src]
            p = self.predict(model, view, items, modality)
            m = compute_metrics(y, p, DEFAULT_THRESHOLD)
            row[f"val_{vname}_loss"] = log_loss(y, p)
            row[f"val_{vname}_auc"] = m["auc"]
            row[f"val_{vname}_bacc"] = m["balanced_accuracy"]
            ds_aucs = []
            for d in np.unique(ds):
                md = compute_metrics(y[ds == d], p[ds == d])
                row[f"val_{vname}_{d}_auc"] = md["auc"]
                if not np.isnan(md["auc"]):
                    ds_aucs.append(md["auc"])
            row[f"val_{vname}_macro_auc"] = float(np.mean(ds_aucs)) if ds_aucs else m["auc"]
            scores.append(nan0(row[f"val_{vname}_macro_auc"]))
        row["score"] = float(np.mean(scores)) if scores else float("nan")
        return row

    def train_stage(self, stage, title, model, view, train_items, epochs, lr, ckpt_path, val_sets, modality, sampling,
                    log_name=None, cfg_override: Optional[dict] = None) -> dict:
        a = self.args
        bs = a.batch_size
        weights, dist = self.sampling_weights(train_items, sampling)
        epoch_size = a.epoch_size or len(train_items)
        steps = int(math.ceil(epoch_size / bs))
        total_steps = max(1, epochs * steps)
        log_name = log_name or f"stage{stage}"
        print("\n" + "=" * 96 + f"\n {title}\n" + "=" * 96)
        kinds = pd.Series([it[0] for it in train_items]).value_counts().to_dict()
        print(f"  * Training items {len(train_items):,} ({', '.join(f'{k}={v:,}' for k, v in kinds.items())}) | "
              f"samples/epoch {epoch_size:,} | {steps} steps x {epochs} epochs | max lr {lr} | modality {modality}")
        print(f"  * Sampling strategy: {sampling}  (balanced sampling replaces loss class-weighting)")
        print("  * Training distribution (expected per epoch):")
        print("    " + dist.round(2).to_string(index=False).replace("\n", "\n    "))
        dist.to_csv(self.res / "logs" / f"sampling_distribution_{log_name}.csv", index=False)

        if sampling == "none":  # keep the 30-9-26 behaviour: class-weighted loss
            y_items = self.labels[[self.item_src(it) for it in train_items]]
            counts = np.bincount(y_items, minlength=2).astype(np.float64)
            criterion = nn.CrossEntropyLoss(weight=torch.tensor(counts.sum() / (2 * np.maximum(counts, 1)), dtype=torch.float32).to(self.device))
        else:
            criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, total_steps=total_steps)
        rng = np.random.default_rng(a.seed + 1000 * stage + len(log_name))
        history, best = [], dict(score=-1.0, epoch=-1)
        t_start = time.time()
        overall = 0
        loader = ThreadPoolExecutor(max_workers=max(1, a.loader_threads))
        prefetch = max(2, a.loader_threads * 2)
        realised = defaultdict(int)
        bad_epochs = 0

        for epoch in range(1, epochs + 1):
            model.train()
            order = rng.choice(len(train_items), size=epoch_size, replace=True, p=weights) if sampling != "none" else \
                rng.permutation(len(train_items))[:epoch_size]
            if epoch == 1:
                for k in order:
                    s = self.item_src(train_items[k])
                    realised[f"{self.ds_of[s]}|{'real' if self.labels[s] == 0 else 'fake'}"] += 1
            seeds = rng.integers(0, 2**31, size=steps)
            batches = [order[s * bs : (s + 1) * bs] for s in range(steps)]
            queue, nxt = deque(), 0

            def submit():
                nonlocal nxt
                if nxt < steps:
                    queue.append(loader.submit(self.make_batch, view, [train_items[k] for k in batches[nxt]], int(seeds[nxt]), modality))
                    nxt += 1

            for _ in range(prefetch):
                submit()
            tot_loss, n_loss, t0, last = 0.0, 0, time.time(), 0
            for s in range(steps):
                frames, audio, a_mask, yb = queue.popleft().result()
                submit()
                overall += 1
                if len(yb) < 2:
                    continue
                logits = model(to_video_tensor(frames, self.device), torch.from_numpy(audio).to(self.device), mode="av",
                               a_mask=torch.from_numpy(a_mask).to(self.device))
                loss = criterion(logits, torch.from_numpy(yb).to(self.device))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if scheduler.last_epoch < total_steps - 1:
                    scheduler.step()
                tot_loss += loss.item()
                n_loss += 1
                now = time.time()
                if now - last >= 0.2 or s + 1 == steps:
                    gt = get_gpu_temp()
                    if gt is not None and gt >= a.gpu_max_temp:
                        paused = check_gpu_thermal_guard(a.gpu_max_temp, a.gpu_pause)
                        t_start += paused
                        t0 += paused
                        now = time.time()
                    ep_eta = (steps - s - 1) / max(1e-3, (s + 1) / max(1e-3, now - t0))
                    tot_eta = (total_steps - overall) / max(1e-3, overall / max(1e-3, now - t_start))
                    sys.stdout.write(f"\r[{log_name}] Epoch {epoch:2d}/{epochs} {progress_bar(s + 1, steps, 16)} "
                                     f"loss {tot_loss / max(1, n_loss):.4f} | GPU {gt if gt is not None else 'N/A'}C | "
                                     f"ep ETA {format_time(ep_eta)} | total ETA {format_time(tot_eta)}   ")
                    sys.stdout.flush()
                    last = now
            if epoch == 1:
                tot_r = sum(realised.values())
                print("\n  * Realised epoch-1 distribution: " + ", ".join(f"{k} {v} ({100 * v / tot_r:.1f}%)" for k, v in sorted(realised.items())))

            row = dict(stage=log_name, epoch=epoch, train_loss=tot_loss / max(1, n_loss), lr=float(scheduler.get_last_lr()[0]))
            row.update(self.evaluate_val(model, view, val_sets, modality))
            history.append(row)
            improved = row["score"] > best["score"]
            if improved:
                best = dict(score=row["score"], epoch=epoch, row=row)
                save_checkpoint(ckpt_path, model, optimizer, epoch, row["score"], None, stage, {**self.config(), **(cfg_override or {})},
                                extra=dict(model_selection_metric="macro per-dataset val AUC", run_name=log_name))
                bad_epochs = 0
            else:
                bad_epochs += 1
            v0 = val_sets[0][0]
            ds_str = " ".join(f"{k.split('_')[-2]}={row[k]:.3f}" for k in row if k.startswith(f"val_{v0}_") and k.endswith("_auc")
                              and k not in (f"val_{v0}_auc", f"val_{v0}_macro_auc"))
            extra = "".join(f" | val {vn} AUC {row.get(f'val_{vn}_auc', float('nan')):.4f}" for vn, _ in val_sets[1:])
            sys.stdout.write(f"\r[{log_name}] Epoch {epoch:2d}/{epochs} | train loss {row['train_loss']:.4f} | "
                             f"val loss {row.get(f'val_{v0}_loss', float('nan')):.4f} | val AUC {row.get(f'val_{v0}_auc', float('nan')):.4f} "
                             f"| val BAcc@0.5 {row.get(f'val_{v0}_bacc', float('nan')):.4f} | per-dataset AUC [{ds_str}]{extra} | "
                             f"select {row['score']:.4f} | lr {row['lr']:.2e} | {format_time(time.time() - t0)}"
                             f"{'  <-- BEST' if improved else ''}          \n")
            sys.stdout.flush()
            if a.patience and bad_epochs >= a.patience:
                print(f"  Early stopping: no improvement for {a.patience} epochs.")
                break
        loader.shutdown(wait=True)
        train_time = time.time() - t_start
        pd.DataFrame(history).to_csv(self.res / "logs" / f"history_{log_name}.csv", index=False)
        print("-" * 96 + f"\n{title} done in {format_time(train_time)} | best epoch {best['epoch']} | selection score {best['score']:.4f}")
        return dict(history=history, best=best, train_time=train_time, sampling_distribution=dist.to_dict(orient="records"))

    # ------------------------------------------------------------------ thresholds
    def select_threshold(self, P_val, idx_val, use_degraded: bool) -> dict:
        """Validation-only threshold (Rule 1). Stage 1: original val videos; Stage 2: original + degraded val videos."""
        y = self.labels[idx_val]
        if use_degraded:
            e = self.E["val"]
            y_deg = self.labels[[self.item_src(it) for it in e["deg_all"]]]
            yy, pp = np.concatenate([y, y_deg]), np.concatenate([P_val["clean"], P_val["deg_all"]])
        else:
            yy, pp = y, P_val["clean"]
        if not self.args.optimize_threshold:
            m = compute_metrics(yy, pp, DEFAULT_THRESHOLD)
            return {"threshold": DEFAULT_THRESHOLD, "metric": "fixed", **{k: m[k] for k in ["balanced_accuracy", "accuracy", "precision", "recall", "specificity", "f1"]}}
        res = find_optimal_threshold(yy, pp, self.args.threshold_metric)
        res["n_val"] = int(len(yy))
        # diagnostic only: what each dataset alone would prefer (NOT used)
        ds = np.concatenate([self.ds_of[idx_val], self.ds_of[[self.item_src(it) for it in self.E["val"]["deg_all"]]]]) if use_degraded else self.ds_of[idx_val]
        res["per_dataset_optimal_diagnostic_only"] = {str(d): find_optimal_threshold(yy[ds == d], pp[ds == d], self.args.threshold_metric)["threshold"]
                                                      for d in np.unique(ds)}
        return res

    def finalize_stage(self, stage: int, model, title: str) -> dict:
        """Load best checkpoint, select the threshold on validation, re-save checkpoint with the threshold (frozen)."""
        model, meta = load_checkpoint(self.ckpt[stage], self.device)
        P = self.predict_sets(model, self.view, self.args.modality, splits=("val",))
        thr = self.select_threshold(P["val"], self.idx["val"], use_degraded=(stage == 2))
        obj = torch.load(self.ckpt[stage], map_location="cpu", weights_only=False)
        obj["threshold"] = thr["threshold"]
        obj["threshold_selection"] = thr
        torch.save(obj, self.ckpt[stage])
        write_json(self.res / f"stage{stage}" / "threshold" / "threshold.json",
                   {"model_path": str(self.ckpt[stage]), "img_size": self.args.img_size, "threshold": thr["threshold"],
                    "dataset": "combined" if len(set(self.ds_of)) > 1 else self.ds_of[0], "stage": stage,
                    "selected_on": "validation (original" + (" + degraded" if stage == 2 else "") + ")", "selection": thr})
        threshold_table(self.labels[self.idx["val"]], P["val"]["clean"], np.round(np.arange(0.01, 1.0, 0.01), 2)).to_csv(
            self.res / f"stage{stage}" / "threshold" / "validation_threshold_curve.csv", index=False)
        print(f"\n  Stage {stage} Threshold = {thr['threshold']:.3f}  (val {thr['metric']} {thr.get(thr['metric'], thr['balanced_accuracy']):.4f}, "
              f"F1 {thr['f1']:.4f}; per-dataset optima, diagnostic only: {thr.get('per_dataset_optimal_diagnostic_only')})")
        best_row = meta.get("best_val_metric")
        print(f"  Best validation selection score {best_row:.4f} at epoch {meta.get('epoch')}")
        return thr

    # ------------------------------------------------------------------ stages
    def run_stage1(self):
        a = self.args
        model = AudioVisualDeepFakeCNN().to(self.device)
        print(f"\nModel AudioVisualDeepFakeCNN | parameters {sum(p.numel() for p in model.parameters()):,} | input {a.img_size}x{a.img_size}")
        items = [("clean", int(s), None) for s in self.idx["train"]]
        self.stage_info = getattr(self, "stage_info", {})
        self.stage_info[1] = self.train_stage(1, "STAGE 1: ORIGINAL VIDEOS (face crop, balanced)", model, self.view, items, a.epochs1, a.lr1,
                                              self.ckpt[1], [("original", self.E["val"]["clean"])], a.modality, a.sampling)
        self.stage_info[1]["threshold"] = self.finalize_stage(1, model, "Stage 1")
        self.status["stage1"] = "Completed"

    def run_stage2(self):
        a = self.args
        if not self.ckpt[1].exists():
            raise FileNotFoundError(f"Stage 1 checkpoint missing: {self.ckpt[1]}")
        model, _ = load_checkpoint(self.ckpt[1], self.device)
        print(f"\n[Stage 2 initialised from] {self.ckpt[1]}")
        items = [("clean", int(s), None) for s in self.idx["train"]]
        for s in self.idx["train"]:
            s = int(s)
            items += [("deg", j, None) for j in self.src_to_deg[s]] if self.src_to_deg[s] else [("synth", s, None)] * a.synth_per_video
        e = self.E["val"]
        val_deg = [e["deg_all"][p] for p in e["pick"] if p >= 0]
        self.stage_info = getattr(self, "stage_info", {})
        self.stage_info[2] = self.train_stage(2, "STAGE 2: DEGRADED + ORIGINAL VIDEOS (fine-tune Stage 1)", model, self.view, items, a.epochs2,
                                              a.lr2, self.ckpt[2], [("original", e["clean"]), ("degraded", val_deg)], a.modality, a.sampling)
        self.stage_info[2]["threshold"] = self.finalize_stage(2, model, "Stage 2")
        self.status["stage2"] = "Completed"

    def rethreshold(self):
        """--mode threshold: recompute validation thresholds for saved checkpoints."""
        for st in (1, 2):
            if self.ckpt[st].exists():
                self.finalize_stage(st, None, f"Stage {st}")

    # ------------------------------------------------------------------ evaluation
    def metric_rows(self, exp, P, thr, idx, meta_cols, split="test", thr_source="validation") -> List[dict]:
        """Rows for Combined / each dataset x original / degraded (random version) / all degraded versions."""
        e = self.E[split]
        y = self.labels[idx]
        ds = self.ds_of[idx]
        hv = e["pick"] >= 0
        src_all = np.array([self.item_src(it) for it in e["deg_all"]], dtype=np.int64)
        rows = []
        for dname in ["Combined"] + sorted(set(ds)):
            sel = np.ones(len(idx), bool) if dname == "Combined" else ds == dname
            sel_all = np.ones(len(src_all), bool) if dname == "Combined" else (self.ds_of[src_all] == dname if len(src_all) else np.zeros(0, bool))
            for cond, yy, pp in (("original", y[sel], P[split]["clean"][sel]),
                                 ("degraded", y[sel & hv], P[split]["deg"][sel & hv]),
                                 ("degraded_all_versions", self.labels[src_all][sel_all] if len(src_all) else [], P[split]["deg_all"][sel_all] if len(src_all) else [])):
                m = compute_metrics(yy, pp, thr)
                rows.append({**meta_cols, "experiment": exp, "split": split, "dataset": dname, "condition": cond, "threshold_source": thr_source, **m})
        return rows

    def evaluate(self):
        """Frozen models + frozen validation thresholds -> test (Rule 1, Section 33)."""
        a = self.args
        print("\n" + "=" * 96 + "\n FINAL TEST: FROZEN MODELS + FROZEN VALIDATION THRESHOLDS\n" + "=" * 96)
        self.models = {}
        for st in (1, 2):
            if self.ckpt[st].exists():
                m, meta = load_checkpoint(self.ckpt[st], self.device)
                if meta.get("threshold") is None:
                    raise RuntimeError(f"Stage {st} checkpoint has no validation threshold; run --mode threshold first.")
                self.models[f"stage{st}"] = dict(model=m, meta=meta, threshold=float(meta["threshold"]), view=self.view, modality=a.modality,
                                                 resolution=a.img_size, face_crop=a.face_crop, sampling=a.sampling, stage=st)
        if not self.models:
            raise FileNotFoundError("No checkpoints to evaluate.")
        self.add_baseline_models()

        all_rows = []
        self.P = {}
        for tag, M in self.models.items():
            print(f"  predicting [{tag}] ...")
            P = self.predict_sets(M["model"], M["view"], M["modality"])
            if M.get("needs_threshold"):  # 96x96 baseline had no threshold: select it on validation the same way
                M["threshold"] = self.select_threshold(P["val"], self.idx["val"], use_degraded=(M["stage"] == 2))["threshold"]
            self.P[tag] = P
            meta_cols = dict(model=tag, resolution=M["resolution"], face_crop=M["face_crop"], sampling=M["sampling"], stage=M["stage"])
            all_rows += self.metric_rows(tag, P, M["threshold"], self.idx["test"], meta_cols)
            all_rows += self.metric_rows(tag, P, DEFAULT_THRESHOLD, self.idx["test"], meta_cols, thr_source="fixed_0.5")
            all_rows += self.metric_rows(tag, P, M["threshold"], self.idx["val"], meta_cols, split="val")
        self.df_long = pd.DataFrame(all_rows)
        self.df_long.to_csv(self.res / "metrics_long.csv", index=False)
        cols = ["model", "resolution", "face_crop", "sampling", "stage", "split", "dataset", "condition", "threshold_source", "threshold",
                "n", "n_real", "n_fake"] + METRIC_KEYS + ["tn", "fp", "fn", "tp"]
        for st in (1, 2):
            sub = self.df_long[(self.df_long.model == f"stage{st}")]
            if len(sub):
                sub[cols].to_csv(self.res / f"stage{st}" / "metrics" / f"stage{st}_metrics.csv", index=False)
        self.save_predictions()
        self.print_main_tables()
        self.plot_confusions()
        self.plot_rocs()
        self.robustness()
        self.degradation_table()
        self.threshold_sensitivity()
        self.resolution_comparison()
        self.status["evaluation"] = "Completed"

    def add_baseline_models(self):
        """Experiment A: 30-9-26 96x96 full-frame checkpoints evaluated on the SAME split/degraded versions."""
        a = self.args
        b1, b2 = self.baseline_dir / "best_stage1_original.pt", self.baseline_dir / "best_stage2_degraded.pt"
        if not (b1.exists() or b2.exists()):
            print(f"  [96x96 baseline] no checkpoints in {self.baseline_dir} -> skipped")
            return
        old_split = self.baseline_dir / "split_30-9-26.json"
        same_split = False
        if old_split.exists():
            old = json.load(open(old_split, encoding="utf-8"))
            same_split = sorted(old.get("test", [])) == sorted(self.sources[i]["key"] for i in self.idx["test"])
        if not same_split:
            print("  [96x96 baseline] split differs from this run (or unknown) -> skipped to avoid an invalid comparison")
            return
        src_paths = [str(s["path"]) for s in self.sources]
        deg_paths = [str(d["path"]) for d in self.degraded]
        search = [self.baseline_dir / "cache"]
        c96 = ClipCache("clean", "\n".join(src_paths) + f"|{self.T}|96", len(src_paths), self.T, 96, self.cache_dir, search)
        d96 = ClipCache("degraded", "\n".join(deg_paths) + f"|{self.T}|96", len(deg_paths), self.T, 96, self.cache_dir, search) if deg_paths else None
        if not c96.find_complete() or (d96 is not None and not d96.find_complete()):
            if not a.baseline_extract:
                print("  [96x96 baseline] full-frame 96x96 caches not found -> skipped (use --baseline-extract to build them)")
                return
            c96.build(lambda i: (extract_video_clip(self.sources[i]["path"], self.T, 96), None), self.n_workers)
            if d96 is not None:
                d96.build(lambda j: (extract_video_clip(self.degraded[j]["path"], self.T, 96), None), self.n_workers)
        # extractions that failed in either view are dropped from the baseline only if they differ
        if not np.array_equal(c96.ok, self.view.clean.ok):
            print("  [96x96 baseline] note: extraction failures differ between the 96 and 128 caches")
        v96 = View("96_full", 96, c96, d96, False)
        for st, pth in ((1, b1), (2, b2)):
            if pth.exists():
                m, meta = load_checkpoint(pth, self.device)
                self.models[f"baseline96_stage{st}"] = dict(model=m, meta=meta, threshold=None, needs_threshold=True, view=v96,
                                                            modality="av", resolution=96, face_crop=False, sampling="none (class-weighted loss)", stage=st)
        print("  [96x96 baseline] 30-9-26 checkpoints added (threshold re-selected on validation; fixed 0.5 rows also reported)")

    def save_predictions(self):
        e = self.E["test"]
        for tag, M in self.models.items():
            P = self.P[tag]
            thr = M["threshold"]
            df = pd.DataFrame({"video": [self.sources[s]["key"] for s in e["idx"]], "dataset": self.ds_of[e["idx"]],
                               "subset": self.subset_of[e["idx"]], "true_label": self.labels[e["idx"]], "threshold": thr,
                               "p_fake_original": P["test"]["clean"], "pred_original": (P["test"]["clean"] >= thr).astype(int),
                               "degraded_version": [self.item_name(e["deg_all"][p]) if p >= 0 else "" for p in e["pick"]],
                               "degradation": [self.item_degradation(e["deg_all"][p])[0] if p >= 0 else "" for p in e["pick"]],
                               "severity": [self.item_degradation(e["deg_all"][p])[1] if p >= 0 else "" for p in e["pick"]],
                               "p_fake_degraded": P["test"]["deg"],
                               "pred_degraded": np.where(e["pick"] >= 0, (np.nan_to_num(P["test"]["deg"]) >= thr).astype(int), -1)})
            out = self.res / (f"stage{M['stage']}" if not tag.startswith("baseline") else "resolution_comparison") / \
                ("predictions" if not tag.startswith("baseline") else "")
            out.mkdir(parents=True, exist_ok=True)
            df.to_csv(out / f"test_predictions_{tag}.csv", index=False)
            src_all = [self.item_src(it) for it in e["deg_all"]]
            pd.DataFrame({"video": [self.sources[s]["key"] for s in src_all], "dataset": self.ds_of[src_all] if src_all else [],
                          "degraded_version": [self.item_name(it) for it in e["deg_all"]],
                          "origin": ["manifest" if it[0] == "deg" else "synthetic" for it in e["deg_all"]],
                          "degradation": [self.item_degradation(it)[0] for it in e["deg_all"]],
                          "severity": [self.item_degradation(it)[1] for it in e["deg_all"]],
                          "true_label": self.labels[src_all] if src_all else [], "p_fake": P["test"]["deg_all"]}).to_csv(
                out / f"test_predictions_all_degraded_{tag}.csv", index=False)

    def print_main_tables(self):
        d = self.df_long[(self.df_long.split == "test") & (self.df_long.threshold_source == "validation")
                         & (self.df_long.condition.isin(["original", "degraded"]))]
        show = d[["model", "dataset", "condition", "threshold", "n", "accuracy", "balanced_accuracy", "precision", "recall", "specificity", "f1", "auc", "tn", "fp", "fn", "tp"]]
        print("\n" + "=" * 96 + "\n TEST METRICS PER MODEL x DATASET x CONDITION (validation thresholds)\n" + "=" * 96)
        with pd.option_context("display.width", 250, "display.max_columns", 30):
            print(show.round(4).to_string(index=False))

    def plot_confusions(self):
        d = self.df_long[(self.df_long.split == "test") & (self.df_long.threshold_source == "validation")
                         & (self.df_long.condition.isin(["original", "degraded"]))]
        for tag in d.model.unique():
            sub = d[d.model == tag]
            dsets = list(dict.fromkeys(sub.dataset))
            fig, axes = plt.subplots(len(dsets), 2, figsize=(9, 3.6 * len(dsets)))
            axes = np.atleast_2d(axes)
            for r, dn in enumerate(dsets):
                for c, cond in enumerate(("original", "degraded")):
                    row = sub[(sub.dataset == dn) & (sub.condition == cond)]
                    ax = axes[r, c]
                    if not len(row):
                        ax.axis("off")
                        continue
                    row = row.iloc[0]
                    cm = np.array([[row.tn, row.fp], [row.fn, row.tp]])
                    ax.imshow(cm, cmap="Blues")
                    for (i, j), v in np.ndenumerate(cm):
                        ax.text(j, i, f"{int(v):,}", ha="center", va="center", fontsize=12, fontweight="bold",
                                color="white" if cm[i, j] > cm.max() / 2 else "black")
                    ax.set_xticks([0, 1], ["Pred Real", "Pred Fake"])
                    ax.set_yticks([0, 1], ["Actual Real", "Actual Fake"])
                    ax.set_title(f"{dn} - {cond} (thr {row.threshold:.2f})\nBAcc {row.balanced_accuracy:.3f} AUC {row.auc:.3f}", fontsize=9)
            fig.suptitle(f"{tag}: confusion matrices (test)", fontweight="bold")
            plt.tight_layout()
            out = self.res / (f"stage{tag[-1]}" if tag.startswith("stage") else "resolution_comparison")
            out = out / "confusion_matrices" if tag.startswith("stage") else out
            plt.savefig(out / f"confusion_{tag}.png", dpi=120)
            plt.close()

    def plot_rocs(self):
        e = self.E["test"]
        y = self.labels[e["idx"]]
        ds = self.ds_of[e["idx"]]
        hv = e["pick"] >= 0
        dsets = ["Combined"] + sorted(set(ds))
        fig, axes = plt.subplots(1, len(dsets), figsize=(5.2 * len(dsets), 4.8))
        axes = np.atleast_1d(axes)
        for ax, dn in zip(axes, dsets):
            sel = np.ones(len(y), bool) if dn == "Combined" else ds == dn
            for tag in self.models:
                for cond, p, s2, ls in (("orig", self.P[tag]["test"]["clean"], sel, "-"), ("deg", self.P[tag]["test"]["deg"], sel & hv, "--")):
                    if len(np.unique(y[s2])) < 2:
                        continue
                    fpr, tpr, _ = roc_curve(y[s2], p[s2])
                    ax.plot(fpr, tpr, ls, lw=1.6, label=f"{tag} {cond} ({roc_auc_score(y[s2], p[s2]):.3f})")
            ax.plot([0, 1], [0, 1], ":", color="gray")
            ax.set_title(f"ROC - {dn}", fontweight="bold")
            ax.set_xlabel("FPR")
            ax.set_ylabel("TPR")
            ax.legend(fontsize=7, loc="lower right")
            ax.grid(alpha=0.4, linestyle="--")
        plt.tight_layout()
        plt.savefig(self.res / "plots" / "roc_per_dataset.png", dpi=120)
        plt.close()

    def robustness(self):
        """Original vs degraded (same test videos, frozen thresholds): metric drops + prediction flips."""
        e = self.E["test"]
        idx, hv = e["idx"], e["pick"] >= 0
        y, ds = self.labels[idx], self.ds_of[idx]
        rows, flips, stab = [], [], []
        for tag, M in self.models.items():
            thr = M["threshold"]
            pc, pdg = self.P[tag]["test"]["clean"], self.P[tag]["test"]["deg"]
            for dn in ["Combined"] + sorted(set(ds)):
                sel = (np.ones(len(y), bool) if dn == "Combined" else ds == dn) & hv
                mo, md = compute_metrics(y[sel], pc[sel], thr), compute_metrics(y[sel], pdg[sel], thr)
                flip = float(np.mean((pc[sel] >= thr) != (pdg[sel] >= thr))) if sel.any() else float("nan")
                rows.append(dict(model=tag, dataset=dn, threshold=thr, n=int(sel.sum()),
                                 original_auc=mo["auc"], degraded_auc=md["auc"], auc_drop=mo["auc"] - md["auc"],
                                 original_accuracy=mo["accuracy"], degraded_accuracy=md["accuracy"], accuracy_drop=mo["accuracy"] - md["accuracy"],
                                 original_balanced_accuracy=mo["balanced_accuracy"], degraded_balanced_accuracy=md["balanced_accuracy"],
                                 balanced_accuracy_drop=mo["balanced_accuracy"] - md["balanced_accuracy"],
                                 original_f1=mo["f1"], degraded_f1=md["f1"], f1_drop=mo["f1"] - md["f1"],
                                 prediction_flip_rate_pct=100 * flip, mean_abs_prob_shift=float(np.abs(pc[sel] - pdg[sel]).mean()) if sel.any() else np.nan))
            degs = np.array([self.item_degradation(e["deg_all"][p])[0] if p >= 0 else "" for p in e["pick"]])
            for i in np.where(hv)[0]:
                flips.append(dict(model=tag, video=self.sources[idx[i]]["key"], dataset=ds[i], true_label=int(y[i]), degradation=degs[i],
                                  p_original=pc[i], p_degraded=pdg[i], pred_original=int(pc[i] >= thr), pred_degraded=int(pdg[i] >= thr),
                                  flipped=bool((pc[i] >= thr) != (pdg[i] >= thr))))
        self.df_rob = pd.DataFrame(rows)
        self.df_rob.to_csv(self.res / "robustness" / "robustness_summary.csv", index=False)
        df_f = pd.DataFrame(flips)
        df_f.to_csv(self.res / "robustness" / "prediction_flips.csv", index=False)
        if len(df_f):
            df_f["abs_prob_shift"] = (df_f.p_original - df_f.p_degraded).abs()
            st = df_f.groupby(["model", "dataset", "degradation"]).agg(n=("flipped", "size"), flip_rate_pct=("flipped", "mean"),
                                                                        mean_abs_prob_shift=("abs_prob_shift", "mean")).reset_index()
            st["flip_rate_pct"] *= 100
            st.to_csv(self.res / "prediction_stability.csv", index=False)
        print("\n" + "=" * 96 + "\n ROBUSTNESS: ORIGINAL vs DEGRADED (same videos, frozen thresholds)\n" + "=" * 96)
        print(self.df_rob[["model", "dataset", "n", "original_auc", "degraded_auc", "auc_drop", "balanced_accuracy_drop", "f1_drop",
                           "prediction_flip_rate_pct"]].round(4).to_string(index=False))
        self.status["robustness"] = "Completed"

    def degradation_table(self):
        """Degradation x severity: AUC, BAcc, F1, flip rate vs the same video's original prediction."""
        e = self.E["test"]
        pos_of_src = {int(s): k for k, s in enumerate(e["idx"])}
        src_all = np.array([self.item_src(it) for it in e["deg_all"]], dtype=np.int64)
        if not len(src_all):
            return
        degs = np.array([self.item_degradation(it)[0] for it in e["deg_all"]])
        sevs_raw = np.array([self.item_degradation(it)[1] for it in e["deg_all"]])
        sevs = np.array([SEVERITY_NAME.get(s, s) for s in sevs_raw])
        origin = np.array(["manifest" if it[0] == "deg" else "synthetic" for it in e["deg_all"]])
        y = self.labels[src_all]
        ds = self.ds_of[src_all]
        rows = []
        for tag, M in self.models.items():
            thr = M["threshold"]
            pa = self.P[tag]["test"]["deg_all"]
            pc = self.P[tag]["test"]["clean"][[pos_of_src[int(s)] for s in src_all]]
            for dn in ["Combined"] + sorted(set(ds)):
                for dg in sorted(set(degs)):
                    for sv in ["mild", "medium", "severe"]:
                        sel = (degs == dg) & (sevs == sv) & ((ds == dn) if dn != "Combined" else True)
                        if not sel.any():
                            continue
                        m = compute_metrics(y[sel], pa[sel], thr)
                        rows.append(dict(model=tag, dataset=dn, degradation=dg, severity=sv, n=int(sel.sum()),
                                         manifest_share=float((origin[sel] == "manifest").mean()), auc=m["auc"],
                                         balanced_accuracy=m["balanced_accuracy"], f1=m["f1"], accuracy=m["accuracy"],
                                         prediction_flip_rate_pct=100 * float(np.mean((pa[sel] >= thr) != (pc[sel] >= thr)))))
        self.df_deg = pd.DataFrame(rows)
        self.df_deg.to_csv(self.res / "robustness" / "degradation_metrics.csv", index=False)
        self.df_deg.to_csv(self.res / "degradation_metrics.csv", index=False)
        main = [t for t in ("stage1", "stage2") if t in self.models]
        if main:
            piv = self.df_deg[(self.df_deg.dataset == "Combined") & self.df_deg.model.isin(main)].pivot_table(
                index=["degradation", "severity"], columns="model", values=["auc", "balanced_accuracy", "f1", "prediction_flip_rate_pct"])
            print("\n" + "=" * 96 + "\n DEGRADATION x SEVERITY (Combined test set)\n" + "=" * 96)
            print(piv.round(3).to_string())
            fig, ax = plt.subplots(figsize=(13, 5))
            pv = self.df_deg[(self.df_deg.dataset == "Combined") & self.df_deg.model.isin(main)].pivot_table(
                index=["degradation", "severity"], columns="model", values="auc")
            pv.plot(kind="bar", ax=ax)
            ax.set_ylim(0, 1.05)
            ax.set_ylabel("AUC")
            ax.set_title("AUC per degradation x severity (test, Combined)", fontweight="bold")
            ax.grid(axis="y", alpha=0.4, linestyle="--")
            plt.tight_layout()
            plt.savefig(self.res / "plots" / "auc_per_degradation_severity.png", dpi=120)
            plt.close()

    def threshold_sensitivity(self):
        """Analysis only: metrics at 0.30..0.70 on val and test. The selected threshold always comes from validation."""
        rows = []
        for tag, M in self.models.items():
            for split in ("val", "test"):
                e = self.E[split]
                idx, hv = e["idx"], e["pick"] >= 0
                y, ds = self.labels[idx], self.ds_of[idx]
                for dn in ["Combined"] + sorted(set(ds)):
                    sel = np.ones(len(y), bool) if dn == "Combined" else ds == dn
                    for cond, s2, p in (("original", sel, self.P[tag][split]["clean"]), ("degraded", sel & hv, self.P[tag][split]["deg"])):
                        t = threshold_table(y[s2], p[s2], SENSITIVITY_THRESHOLDS)
                        t.insert(0, "condition", cond)
                        t.insert(0, "dataset", dn)
                        t.insert(0, "split", split)
                        t.insert(0, "model", tag)
                        t["selected_threshold"] = M["threshold"]
                        t["used_for_selection"] = False
                        rows.append(t)
        df = pd.concat(rows, ignore_index=True)
        df.to_csv(self.res / "threshold_analysis.csv", index=False)
        main = [t for t in ("stage1", "stage2") if t in self.models]
        fig, axes = plt.subplots(1, 4, figsize=(18, 4.2))
        for ax, met in zip(axes, ["balanced_accuracy", "f1", "recall", "specificity"]):
            for tag in main:
                for cond, ls in (("original", "-"), ("degraded", "--")):
                    d = df[(df.model == tag) & (df.split == "test") & (df.dataset == "Combined") & (df.condition == cond)]
                    ax.plot(d.threshold, d[met], ls, marker="o", ms=3, label=f"{tag} {cond}")
                ax.axvline(self.models[tag]["threshold"], color="gray", lw=0.8, linestyle=":")
            ax.set_title(f"Threshold vs {met}", fontsize=10)
            ax.set_xlabel("threshold")
            ax.grid(alpha=0.4, linestyle="--")
        axes[0].legend(fontsize=7)
        fig.suptitle("Threshold sensitivity on TEST (analysis only - thresholds were selected on validation, dotted lines)", fontsize=10)
        plt.tight_layout()
        plt.savefig(self.res / "plots" / "threshold_sensitivity.png", dpi=120)
        plt.close()

    def resolution_comparison(self):
        """96x96 (30-9-26) vs this run. NOTE: the baseline also differs in face cropping and sampling, so
        differences cannot be attributed to resolution alone; run ablations (--no-face-crop / --img-size 96)
        into separate work dirs and pass them with --compare-runs to isolate each factor."""
        d = self.df_long[(self.df_long.split == "test") & (self.df_long.threshold_source == "validation")
                         & self.df_long.condition.isin(["original", "degraded"])].copy()
        frames = [d]
        for rd in self.args.compare_runs:
            f = Path(rd) / "results" / "metrics_long.csv"
            if f.exists():
                o = pd.read_csv(f)
                o = o[(o.split == "test") & (o.threshold_source == "validation") & o.condition.isin(["original", "degraded"]) & o.model.isin(["stage1", "stage2"])]
                o["model"] = Path(rd).name + ":" + o["model"]
                frames.append(o)
        d = pd.concat(frames, ignore_index=True)
        d["confounded_with"] = np.where(d.model.str.startswith("baseline96"), "full frame (no face crop) + class-weighted loss (no balanced sampling)", "")
        cols = ["model", "resolution", "face_crop", "sampling", "stage", "dataset", "condition", "threshold"] + METRIC_KEYS + ["confounded_with"]
        d[cols].to_csv(self.res / "resolution_comparison" / "resolution_comparison.csv", index=False)
        self.df_res = d[cols]
        if any(m.startswith("baseline96") for m in self.models):
            piv = d[d.condition == "original"].pivot_table(index=["dataset"], columns=["resolution", "stage"], values="balanced_accuracy")
            print("\n" + "=" * 96 + "\n 96 vs 128 (balanced accuracy, original test videos; baseline is confounded - see CSV)\n" + "=" * 96)
            print(piv.round(4).to_string())

    # ------------------------------------------------------------------ cross-dataset
    def cross_dataset(self):
        """Train on one dataset, test on the other (video-only, Stage-1 protocol, own-validation threshold).
        Combined training rows come from the main Stage 1 / Stage 2 models."""
        a = self.args
        dsets = sorted(set(self.ds_of))
        if len(dsets) < 2:
            print("  [cross-dataset] needs both datasets -> skipped")
            self.status["cross"] = "Not Completed (single dataset)"
            return
        out = self.res / "cross_dataset"
        rows = []
        for train_ds in dsets:
            tag = f"train_{train_ds.replace('+', 'p')}"
            ck = out / f"{tag}.pt"
            tr = [int(s) for s in self.idx["train"] if self.ds_of[s] == train_ds]
            va = [("clean", int(s), None) for s in self.idx["val"] if self.ds_of[s] == train_ds]
            if not ck.exists():
                model = AudioVisualDeepFakeCNN().to(self.device)
                self.train_stage(1, f"CROSS-DATASET: TRAIN ON {train_ds} ONLY (video-only, original videos)", model, self.view,
                                 [("clean", s, None) for s in tr], a.cross_epochs, a.lr1, ck, [("original", va)], "video", "class",
                                 log_name=tag, cfg_override=dict(modality="video", sampling="class", train_dataset=train_ds))
            model, meta = load_checkpoint(ck, self.device)
            pv = self.predict(model, self.view, va, "video")
            thr = find_optimal_threshold(self.labels[[it[1] for it in va]], pv, a.threshold_metric)["threshold"] if a.optimize_threshold else DEFAULT_THRESHOLD
            obj = torch.load(ck, map_location="cpu", weights_only=False)
            obj["threshold"] = thr
            torch.save(obj, ck)
            P = self.predict_sets(model, self.view, "video", splits=("test",))
            for r in self.metric_rows(tag, P, thr, self.idx["test"], dict(model=tag, resolution=a.img_size, face_crop=a.face_crop, sampling="class", stage=1)):
                if r["dataset"] == "Combined" or r["condition"] == "degraded_all_versions":
                    continue
                r["train_dataset"], r["test_dataset"] = train_ds, r["dataset"]
                r["setting"] = "in-domain" if r["dataset"] == train_ds else "cross-dataset"
                r["modality"] = "video"
                rows.append(r)
        for tag in ("stage1", "stage2"):  # combined training (main models)
            if tag in self.models:
                for r in self.df_long[(self.df_long.model == tag) & (self.df_long.split == "test") & (self.df_long.threshold_source == "validation")
                                      & self.df_long.condition.isin(["original", "degraded"]) & (self.df_long.dataset != "Combined")].to_dict(orient="records"):
                    r["train_dataset"], r["test_dataset"] = "BioDeepAV+FF++", r["dataset"]
                    r["setting"] = f"combined ({tag})"
                    r["modality"] = a.modality
                    rows.append(r)
        df = pd.DataFrame(rows)
        cols = ["setting", "train_dataset", "test_dataset", "model", "modality", "condition", "threshold"] + METRIC_KEYS + ["n", "tn", "fp", "fn", "tp"]
        df[cols].to_csv(out / "cross_dataset_metrics.csv", index=False)
        df[cols].to_csv(self.res / "cross_dataset_metrics.csv", index=False)
        self.df_cross = df[cols]
        print("\n" + "=" * 96 + "\n CROSS-DATASET EVALUATION\n" + "=" * 96)
        print(self.df_cross[["setting", "train_dataset", "test_dataset", "condition", "threshold", "auc", "balanced_accuracy", "f1"]].round(4).to_string(index=False))
        self.status["cross"] = "Completed"

    # ------------------------------------------------------------------ shortcut diagnostics
    def shortcut_diagnostics(self):
        """Does the prediction track nuisance properties (dataset, folder, audio, resolution, face size,
        frame statistics) more than the label? Results are hypotheses ('potential shortcut'), not proof."""
        tag = "stage2" if "stage2" in self.models else "stage1" if "stage1" in self.models else None
        if tag is None:
            return
        M = self.models[tag]
        e = self.E["test"]
        idx = e["idx"]
        p = self.P[tag]["test"]["clean"]
        thr = M["threshold"]
        clips = self.view.clean.arr
        bright, sharp = [], []
        for s in idx:
            c = np.asarray(clips[s, :: max(1, self.T // 4)])
            g = c.mean(-1)
            bright.append(float(g.mean()))
            sharp.append(float(np.mean([cv2.Laplacian(f.astype(np.float32), cv2.CV_32F).var() for f in g])))
        meta = self.view.clean.info["meta"][idx] if self.view.clean.info is not None else np.full((len(idx), 8), np.nan)
        df = pd.DataFrame(dict(video=[self.sources[s]["key"] for s in idx], dataset=self.ds_of[idx], subset=self.subset_of[idx],
                               label=self.labels[idx], has_audio=self.has_audio[idx].astype(int), width=meta[:, 0], height=meta[:, 1],
                               fps=meta[:, 2], total_frames=meta[:, 3], face_fallback=meta[:, 4], face_crop_rel_size=meta[:, 5],
                               face_detection_rate=meta[:, 6] / np.maximum(meta[:, 7], 1), crop_brightness=bright, crop_sharpness=sharp,
                               p_fake=p, pred=(p >= thr).astype(int)))
        df["resolution_px"] = df.width * df.height
        df.to_csv(self.res / "diagnostics" / "shortcut_features_test.csv", index=False)
        summary = df.groupby(["dataset", "subset", "label"]).agg(n=("video", "count"), pred_fake_rate=("pred", "mean"), mean_p_fake=("p_fake", "mean"),
                                                                  has_audio=("has_audio", "mean"), width=("width", "median"), height=("height", "median"),
                                                                  face_crop_rel_size=("face_crop_rel_size", "mean"), sharpness=("crop_sharpness", "mean"),
                                                                  brightness=("crop_brightness", "mean"), fallback=("face_fallback", "mean")).reset_index()
        summary.to_csv(self.res / "diagnostics" / "prediction_by_group.csv", index=False)
        feats = ["has_audio", "resolution_px", "fps", "total_frames", "face_crop_rel_size", "face_detection_rate", "crop_brightness", "crop_sharpness"]
        rows = []
        for scope in ["Combined"] + sorted(set(df.dataset)):
            d = df if scope == "Combined" else df[df.dataset == scope]
            for f in feats:
                x = d[f].astype(float)
                ok = ~x.isna()
                if ok.sum() < 10 or x[ok].nunique() < 2:
                    continue
                auc_label = roc_auc_score(d.label[ok], x[ok]) if d.label[ok].nunique() > 1 else np.nan
                auc_pred = roc_auc_score(d.pred[ok], x[ok]) if d.pred[ok].nunique() > 1 else np.nan
                within = []
                for lab in (0, 1):  # does p_fake follow the feature WITHIN a class? (label held fixed)
                    dd = d[ok & (d.label == lab)]
                    if len(dd) >= 10 and dd[f].nunique() > 1:
                        within.append(spearmanr(dd[f], dd.p_fake).correlation)
                sep = abs(nan0(auc_label) - 0.5) if not np.isnan(auc_label) else 0
                rho = np.nanmax(np.abs(within)) if within else np.nan
                flag = ""
                if sep >= 0.3:
                    flag = "potential shortcut: feature nearly separates the classes (label-confounded)"
                if not np.isnan(rho) and rho >= 0.3:
                    flag = (flag + "; " if flag else "") + "potential shortcut: prediction follows the feature within a class"
                rows.append(dict(scope=scope, feature=f, auc_feature_vs_label=auc_label, auc_feature_vs_prediction=auc_pred,
                                 spearman_feature_vs_pfake_within_real=within[0] if len(within) > 0 else np.nan,
                                 spearman_feature_vs_pfake_within_fake=within[1] if len(within) > 1 else np.nan, note=flag))
        df_s = pd.DataFrame(rows)
        df_s.to_csv(self.res / "diagnostics" / "shortcut_feature_tests.csv", index=False)
        # Audio-availability test: BioDeepAV test videos with audio, predicted with audio vs audio masked
        audio_test = {}
        sel = np.where(self.has_audio[idx])[0]
        if len(sel) and self.args.modality == "av":
            items = [e["clean"][i] for i in sel]
            p_mask = self.predict(M["model"], self.view, items, "av", force_mask=0)
            p_with = p[sel]
            audio_test = dict(n=int(len(sel)), mean_p_fake_with_audio=float(p_with.mean()), mean_p_fake_audio_masked=float(p_mask.mean()),
                              flip_rate_pct=float(100 * np.mean((p_with >= thr) != (p_mask >= thr))),
                              auc_with_audio=compute_metrics(self.labels[idx][sel], p_with, thr)["auc"],
                              auc_audio_masked=compute_metrics(self.labels[idx][sel], p_mask, thr)["auc"],
                              note="large shift when audio is removed = decision depends on audio presence (potential shortcut, since only BioDeepAV has audio)")
        self.diagnostics = dict(model=tag, by_group=summary.to_dict(orient="records"), feature_tests=df_s.to_dict(orient="records"), audio_masking_test=audio_test,
                                caveat="Correlations indicate POTENTIAL shortcuts only; they are not proof of shortcut learning.")
        write_json(self.res / "diagnostics" / "shortcut_diagnostics.json", self.diagnostics)
        print("\n" + "=" * 96 + f"\n SHORTCUT DIAGNOSTICS ({tag}, test originals) - potential shortcuts only, not proof\n" + "=" * 96)
        print(summary[["dataset", "subset", "label", "n", "pred_fake_rate", "has_audio", "width", "height", "face_crop_rel_size", "sharpness"]].round(3).to_string(index=False))
        flagged = df_s[df_s.note != ""]
        if len(flagged):
            print("\n  Flagged features:")
            print(flagged[["scope", "feature", "auc_feature_vs_label", "spearman_feature_vs_pfake_within_real", "spearman_feature_vs_pfake_within_fake", "note"]].round(3).to_string(index=False))
        if audio_test:
            print(f"\n  Audio masking (BioDeepAV, n={audio_test['n']}): mean p_fake {audio_test['mean_p_fake_with_audio']:.3f} -> "
                  f"{audio_test['mean_p_fake_audio_masked']:.3f}, flips {audio_test['flip_rate_pct']:.1f}%, "
                  f"AUC {audio_test['auc_with_audio']:.3f} -> {audio_test['auc_audio_masked']:.3f}")

    # ------------------------------------------------------------------ Grad-CAM stability (kept from 30-9-26)
    def xai_stability(self):
        a = self.args
        e = self.E["test"]
        hv = np.where(e["pick"] >= 0)[0]
        main = [t for t in ("stage1", "stage2") if t in self.models]
        if a.skip_xai or a.xai_samples <= 0 or not len(hv) or not main:
            return
        rng = np.random.default_rng(a.seed + 5)
        pos = np.sort(rng.choice(hv, size=min(a.xai_samples, len(hv)), replace=False))
        print("\n" + "=" * 96 + f"\n GRAD-CAM STABILITY ON FROZEN MODELS ({len(pos)} test videos, aligned original/degraded crops)\n" + "=" * 96)
        rows, examples = [], {}
        for tag in main:
            M = self.models[tag]
            for k in range(0, len(pos), 16):
                pp = pos[k : k + 16]
                ci = [e["clean"][i] for i in pp]
                di = [e["deg_all"][e["pick"][i]] for i in pp]
                src = np.array([self.item_src(it) for it in ci])
                cc = np.stack([self.get_clip(self.view, it)[:: self.stride][: self.L] for it in ci])
                dc = np.stack([self.get_clip(self.view, it)[:: self.stride][: self.L] for it in di])
                xa, am = self.audio_inputs(src, M["modality"])
                gc = grad_cam(M["model"], to_video_tensor(cc, self.device), xa, am, 1, M["threshold"])
                gd = grad_cam(M["model"], to_video_tensor(dc, self.device), xa, am, 1, M["threshold"])
                for b, i in enumerate(pp):
                    r_, iou_, sh_ = cam_metrics(gc["spatial_cam"][b], gd["spatial_cam"][b])
                    tc = float(np.corrcoef(gc["temporal_cam"][b], gd["temporal_cam"][b])[0, 1]) if gc["temporal_cam"][b].std() > 1e-8 and gd["temporal_cam"][b].std() > 1e-8 else np.nan
                    deg = self.item_degradation(di[b])
                    rows.append(dict(model=tag, video=self.sources[src[b]]["key"], dataset=self.ds_of[src[b]], true_label=int(self.labels[src[b]]),
                                     degradation=deg[0], severity=deg[1], threshold=M["threshold"],
                                     p_fake_original=float(gc["probability_fake"][b]), p_fake_degraded=float(gd["probability_fake"][b]),
                                     prediction_stable=bool(gc["prediction"][b] == gd["prediction"][b]), pearson_r=r_, top20_iou=iou_,
                                     center_shift=sh_, temporal_cam_r=tc, explanation_unstable=bool(iou_ < 0.5)))
                    if tag == main[-1] and len(examples) < 6:
                        examples[i] = (cc[b, self.L // 2], gc["spatial_cam"][b], dc[b, self.L // 2], gd["spatial_cam"][b],
                                       gc["probability_fake"][b], gd["probability_fake"][b], r_, iou_, deg[0])
        df = pd.DataFrame(rows)
        df.to_csv(self.res / "xai" / "gradcam_stability.csv", index=False)
        summ = {}
        for tag in main:
            d = df[df.model == tag]
            st = d[d.prediction_stable]
            summ[tag] = dict(n=len(d), mean_pearson_r=float(d.pearson_r.mean()), mean_top20_iou=float(d.top20_iou.mean()),
                             mean_center_shift=float(d.center_shift.mean()), mean_temporal_cam_r=float(d.temporal_cam_r.mean()),
                             prediction_stable_rate=float(d.prediction_stable.mean()),
                             unstable_explanation_given_stable_prediction=float(st.explanation_unstable.mean()) if len(st) else np.nan,
                             per_dataset=d.groupby("dataset")[["pearson_r", "top20_iou", "center_shift"]].mean().round(4).to_dict(orient="index"),
                             per_degradation=d.groupby("degradation")[["pearson_r", "top20_iou", "center_shift"]].mean().round(4).to_dict(orient="index"))
            s = summ[tag]
            print(f"  [{tag}] r {s['mean_pearson_r']:.3f} | top-20% IoU {s['mean_top20_iou']:.3f} | centre shift {s['mean_center_shift']:.3f} | "
                  f"prediction stable {100 * s['prediction_stable_rate']:.1f}% | explanation changed while prediction stable "
                  f"{100 * nan0(s['unstable_explanation_given_stable_prediction'], np.nan):.1f}%")
        write_json(self.res / "xai" / "gradcam_stability_summary.json", summ)
        self.xai_summary = summ
        if examples:
            fig, axes = plt.subplots(len(examples), 4, figsize=(10, 2.6 * len(examples)))
            axes = np.atleast_2d(axes)
            for r, (i, (fc, cc_, fd, cd, pc, pdg, r_, iou_, dg)) in enumerate(examples.items()):
                axes[r, 0].imshow(fc); axes[r, 0].set_title(f"original p={pc:.2f}", fontsize=8)
                axes[r, 1].imshow(fc); axes[r, 1].imshow(cc_, cmap="jet", alpha=0.45); axes[r, 1].set_title("Grad-CAM original", fontsize=8)
                axes[r, 2].imshow(fd); axes[r, 2].set_title(f"{dg} p={pdg:.2f}", fontsize=8)
                axes[r, 3].imshow(fd); axes[r, 3].imshow(cd, cmap="jet", alpha=0.45); axes[r, 3].set_title(f"Grad-CAM degraded r={r_:.2f} IoU={iou_:.2f}", fontsize=8)
                for ax in axes[r]:
                    ax.axis("off")
            plt.tight_layout()
            plt.savefig(self.res / "xai" / "gradcam_examples.png", dpi=120)
            plt.close()
        self.status["gradcam"] = "Compatible"

    # ------------------------------------------------------------------ freeze + summary
    def freeze_final(self):
        """Final = Stage 2 (else Stage 1). Copy + hash so the XAI phase can verify the frozen model is unchanged."""
        st = 2 if self.ckpt[2].exists() else 1 if self.ckpt[1].exists() else None
        if st is None:
            return None
        dst = self.res / "final" / "final_model.pt"
        obj = torch.load(self.ckpt[st], map_location="cpu", weights_only=False)
        obj["frozen"] = True
        torch.save(obj, dst)
        info = {"model_path": str(dst), "source_checkpoint": str(self.ckpt[st]), "stage": st, "img_size": obj.get("img_size"),
                "threshold": obj.get("threshold"), "sha256": file_sha256(dst), "frozen_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "rule": "Do not retrain or change this model/threshold based on XAI results."}
        write_json(self.res / "final" / "FROZEN.json", info)
        return info

    def final_summary(self, frozen):
        a = self.args
        get = lambda t: (self.models[t]["threshold"] if hasattr(self, "models") and t in self.models else None)

        def stage_metrics(tag):
            if not hasattr(self, "df_long"):
                return {}
            d = self.df_long[(self.df_long.model == tag) & (self.df_long.split == "test") & (self.df_long.threshold_source == "validation")]
            return {f"{r.dataset}|{r.condition}": {k: r[k] for k in METRIC_KEYS + ["n", "tn", "fp", "fn", "tp"]} for _, r in d.iterrows()}

        summary = {
            "image_size": a.img_size, "face_crop": a.face_crop, "face_alignment": a.face_align, "sampling": a.sampling, "modality": a.modality,
            "seed": a.seed, "deterministic": a.deterministic,
            "stage1": {"threshold": get("stage1"), "metrics": stage_metrics("stage1"), "status": self.status["stage1"]},
            "stage2": {"threshold": get("stage2"), "metrics": stage_metrics("stage2"), "status": self.status["stage2"]},
            "datasets": self.df_inv.reset_index().to_dict(orient="records"),
            "leakage_check": self.leakage,
            "face_detection": getattr(self, "face_summary", None),
            "degradations": self.df_deg.to_dict(orient="records") if hasattr(self, "df_deg") else {},
            "robustness": self.df_rob.to_dict(orient="records") if hasattr(self, "df_rob") else {},
            "cross_dataset": self.df_cross.to_dict(orient="records") if hasattr(self, "df_cross") else {},
            "resolution_comparison": self.df_res.to_dict(orient="records") if hasattr(self, "df_res") else {},
            "shortcut_diagnostics": getattr(self, "diagnostics", {}),
            "gradcam_stability": getattr(self, "xai_summary", {}),
            "frozen_model": frozen,
        }
        write_json(self.res / "final_experiment_summary.json", summary)

        def ds_line(tag, dn):
            if not hasattr(self, "df_long") or tag not in self.models:
                return "n/a"
            d = self.df_long[(self.df_long.model == tag) & (self.df_long.split == "test") & (self.df_long.threshold_source == "validation")
                             & (self.df_long.dataset == dn) & (self.df_long.condition == "original")]
            return f"AUC = {d.auc.iloc[0]:.3f}\nBalanced Accuracy = {d.balanced_accuracy.iloc[0]:.3f}" if len(d) else "n/a"

        fin = "stage2" if "stage2" in getattr(self, "models", {}) else "stage1"
        thr_s = lambda v: f"{v:.3f}" if v is not None else "n/a"
        print("\n" + "=" * 40 + "\nDEEPFAKE DETECTION UPDATE COMPLETE\n" + "=" * 40)
        print(f"\nInput Resolution:\n{a.img_size} x {a.img_size}\n\nFace Cropping:\n{'Enabled' if a.face_crop else 'Disabled'}"
              f"\n\nDataset Balancing:\n{'Enabled (' + a.sampling + ')' if a.sampling != 'none' else 'Disabled'}"
              f"\n\nStage 1:\n{self.status['stage1']}\n\nStage 1 Threshold:\n{thr_s(get('stage1'))}"
              f"\n\nStage 2:\n{self.status['stage2']}\n\nStage 2 Threshold:\n{thr_s(get('stage2'))}"
              f"\n\nBioDeepAV ({fin}, original test):\n{ds_line(fin, 'BioDeepAV')}\n\nFF++ ({fin}, original test):\n{ds_line(fin, 'FF++')}"
              f"\n\nCross Dataset:\n{self.status['cross']}\n\nRobustness:\n{self.status['robustness']}\n\nGrad-CAM:\n{self.status['gradcam']}")
        print("\n" + "=" * 40 + "\nNEXT RESEARCH PHASE\n" + "=" * 40)
        print("\nFreeze final detector\n        |\nXAI implementation\n        |\nGrad-CAM stability\n        |\nTemporal XAI\n        |\nAdditional XAI methods")
        print("=" * 40)
        print(f"\nResults: {self.res}")
        if frozen:
            print(f"Frozen model: {frozen['model_path']} (threshold {thr_s(frozen['threshold'])}, sha256 {frozen['sha256'][:12]}...)")


# =====================================================================================
# SELF TEST (smoke tests, no training)
# =====================================================================================
def selftest(args):
    script_dir = Path(__file__).resolve().parent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}{(' - ' + detail) if detail else ''}")

    print("=" * 80 + "\n SELF TEST\n" + "=" * 80)
    fm = ensure_face_model(args.face_model, script_dir)
    det = FaceDetector(fm, args.face_score)
    check("face detector available", det.backend == "yunet", det.backend)
    sources, folders = discover_sources(argparse.Namespace(**{**vars(args), "max_per_folder": 1}), random.Random(args.seed))
    for s in sources:
        clip, info = extract_face_clip(s["path"], args.frames, args.img_size, det, args.face_margin, args.face_align)
        check(f"video load + face crop {s['key'][:60]}", clip is not None and clip.shape == (args.frames, args.img_size, args.img_size, 3),
              f"shape {None if clip is None else clip.shape} | frames {info['frames']} detected {info['detected']} failed {info['failed']} "
              f"rate {100 * nan0(info['detection_rate'], 0):.1f}% fallback {info['fallback']} | src {info['width']}x{info['height']}")
        if clip is not None and s is sources[0]:
            cv2.imwrite(str(script_dir / "selftest_face_crop.jpg"), cv2.cvtColor(np.concatenate(list(clip[:4]), 1), cv2.COLOR_RGB2BGR))
    # fallback path: blank video frame -> no face -> full-frame resize
    blank = np.zeros((240, 320, 3), np.uint8)
    check("no-face fallback", detect_face(blank, det) is None and preprocess_frame(blank, args.img_size).shape == (args.img_size, args.img_size, 3))

    model = AudioVisualDeepFakeCNN().to(device)
    x = torch.randn(2, args.clip_len, 3, args.img_size, args.img_size, device=device)
    out = model(x, torch.randn(2, 1, 128, 128, device=device), a_mask=torch.tensor([1.0, 0.0], device=device))
    check(f"forward {args.img_size}x{args.img_size}", tuple(out.shape) == (2, 2), str(tuple(out.shape)))
    g = grad_cam(model, x, threshold=0.5)
    check("Grad-CAM output", g["spatial_cam"].shape == (2, args.img_size, args.img_size) and g["temporal_cam"].shape == (2, args.clip_len)
          and np.allclose(g["temporal_cam"].sum(1), 1, atol=1e-4), f"spatial {g['spatial_cam'].shape} temporal {g['temporal_cam'].shape}")

    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 400)
    p = np.clip(0.3 * y + rng.normal(0.4, 0.2, 400), 0, 1)
    thr = find_optimal_threshold(y, p)
    m = compute_metrics(y, p, thr["threshold"])
    from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score
    pred = (p >= thr["threshold"]).astype(int)
    check("threshold in [0,1]", 0 <= thr["threshold"] <= 1, f"{thr['threshold']:.2f}")
    check("metrics match sklearn", np.isclose(m["balanced_accuracy"], balanced_accuracy_score(y, pred)) and np.isclose(m["f1"], f1_score(y, pred))
          and np.isclose(m["precision"], precision_score(y, pred)) and np.isclose(m["recall"], recall_score(y, pred)))
    check("specificity = TN/(TN+FP)", np.isclose(m["specificity"], m["tn"] / (m["tn"] + m["fp"])))
    check("empty / single-class safe", np.isnan(compute_metrics([], []).get("auc")) and np.isnan(compute_metrics([1, 1], [0.2, 0.9])["specificity"]))

    tmp = script_dir / "selftest_ckpt.pt"
    cfg = dict(img_size=args.img_size, frames=args.frames, clip_len=args.clip_len, face_crop=True, face_margin=args.face_margin,
               face_align=args.face_align, modality="av", seed=args.seed)
    save_checkpoint(tmp, model, optim.AdamW(model.parameters()), 1, 0.9, 0.43, 2, cfg)
    m2, meta = load_checkpoint(tmp, device)
    same = all(torch.equal(a_, b_) for a_, b_ in zip(model.state_dict().values(), m2.state_dict().values()))
    check("checkpoint save/reload", same and meta["threshold"] == 0.43 and meta["img_size"] == args.img_size and meta["stage"] == 2)
    if sources:
        det_api = DeepfakeDetector(str(tmp))
        r = det_api.predict_video(str(sources[0]["path"]))
        check("predict_video API", set(r) >= {"probability_fake", "prediction", "threshold", "model_stage", "img_size"} and r["img_size"] == args.img_size,
              json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}))
    tmp.unlink(missing_ok=True)
    legacy = script_dir / "run_30-9-26" / "best_stage2_degraded.pt"
    if legacy.exists():
        _, lm = load_checkpoint(legacy, device)
        check("legacy 96x96 checkpoint loads", lm.get("legacy") and lm["img_size"] == 96)
    print("=" * 80 + f"\n SELF TEST {'PASSED' if ok else 'FAILED'}\n" + "=" * 80)
    return ok


# =====================================================================================
# MAIN
# =====================================================================================
def main(argv=None):
    args = parse_args(argv)
    if args.mode == "selftest":
        sys.exit(0 if selftest(args) else 1)
    if args.mode == "predict":
        if not args.video:
            raise SystemExit("--mode predict needs --video")
        wd = Path(args.work_dir) if args.work_dir else Path(__file__).resolve().parent / "run_1-10-26"
        ck = args.checkpoint or str(wd / "results" / "final" / "final_model.pt")
        print(json.dumps(DeepfakeDetector(ck, face_model=args.face_model).predict_video(args.video), indent=2))
        return

    exp = Experiment(args)
    exp.prepare_data()
    if args.mode in ("full", "stage1"):
        exp.run_stage1()
    if args.mode in ("full", "stage2"):
        exp.run_stage2()
    if args.mode == "threshold":
        exp.rethreshold()
    if args.mode in ("full", "stage1", "stage2", "evaluate", "threshold"):
        for st in (1, 2):
            if exp.ckpt[st].exists():
                exp.status[f"stage{st}"] = "Completed"
        exp.evaluate()
        exp.shortcut_diagnostics()
    if args.mode == "cross" or (args.mode == "full" and not args.skip_cross):
        if args.mode == "cross":
            exp.evaluate()
        exp.cross_dataset()
    if args.mode in ("full", "evaluate"):
        exp.xai_stability()
    frozen = exp.freeze_final() if args.mode in ("full", "stage2", "evaluate", "threshold") else None
    exp.final_summary(frozen)


if __name__ == "__main__":
    main()
