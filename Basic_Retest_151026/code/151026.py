"""
151026.py - REAL / FAKE VIDEO DETECTOR WITH EXPLAINABLE AI (BioDeepAV + FaceForensics++ C23)
=============================================================================================

Goal: put ANY video in, get "REAL" or "FAKE" (with a calibrated probability) plus an explanation of WHY.

This script is a fresh, standalone experiment. It does not read anything from earlier tests (no old caches,
splits, checkpoints or degraded videos). It uses ONLY the original real and fake videos:

  BioDeepAV            E:\\Datasets\\BioDeepAV\\real\\videos  (REAL)   E:\\Datasets\\BioDeepAV\\fake\\videos (FAKE)
  FaceForensics++ C23  original (REAL); Deepfakes, Face2Face, FaceShifter, FaceSwap, NeuralTextures, DeepFakeDetection (FAKE)

Design choices (why this is the best model for "any video")
  * Pretrained backbone: EfficientNet-B0 (ImageNet weights, torchvision) - the standard strong choice for face-forgery
    detection on FF++; far better generalisation than a small CNN trained from scratch.
  * Video, not audio: FF++ has no audio and audio presence was a shortcut, so the detector uses the face video only.
  * Face-centred: YuNet face detector (Haar cascade fallback), 16 frames per video, square crop with margin, temporal
    box smoothing. Identical code is used for training, evaluation and prediction.
  * Resolution shortcut removed: every face is resized to a fixed effective resolution (112 px "bottleneck") before the
    network sees it, so the model cannot use video resolution (BioDeepAV real 168 px vs fake 512 px).
  * Temporal model: per-frame EfficientNet features -> 1-layer temporal transformer -> attention pooling. The attention
    weights say WHICH frames mattered; a per-frame auxiliary loss keeps every frame informative.
  * Robust training: joint training on both datasets with dataset x class balanced sampling, real-world augmentation
    (JPEG, blur, down/up-scaling, colour jitter, random erasing, flips, crops), label smoothing, AMP, EMA weights,
    warm-up + cosine learning rate, early model selection on validation macro-AUC over both datasets.
  * No leakage: videos are grouped by identity (FF++ sequence ids, DFD actor ids, BioDeepAV YouTube ids) AND by
    duplicate content (exact file hash, near-identical face fingerprints) before the 70 / 15 / 15 split.
  * Calibrated decision: temperature scaling + a decision threshold chosen on validation (mean balanced accuracy over
    the two datasets). Predictions close to the threshold are reported as "uncertain".
  * Several seeds can be trained; the one with the best validation score becomes best_model.pt.

Explainable AI (XAI)
  * Grad-CAM++ on the last EfficientNet block: where in each face the evidence for the decision is.
  * Temporal attention: which frames drove the decision.
  * Occlusion sensitivity (model-agnostic check, prediction mode): how much the probability drops when each face
    region is hidden.
  * Faithfulness test on the test set: hiding the top-20 % Grad-CAM region must change the prediction more than hiding
    a random 20 % region - if not, the explanation is not trusted. Reported in the evaluation report.

Usage (Windows, from E:\\Code)
  python 151026.py --phase selftest                 # quick unit tests (no data)
  python 151026.py                                  # full pipeline: inventory -> faces -> split -> train -> evaluate -> xai -> report
  python 151026.py --predict "D:\\any\\video.mp4"     # REAL / FAKE + explanation for one video (or a folder of videos)
  python 151026.py --phase gui                      # small window: pick a video, see the verdict and heat-maps
  python 151026.py --smoke                          # tiny end-to-end run (40 videos per folder) in a separate folder
  Every phase is resumable. Options: --seeds 42,123 (train several, keep the best), --img 160 (less disk), --epochs 12
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import re
import shutil
import sys
import threading
import time
import urllib.request
import warnings
from collections import Counter, defaultdict, deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve  # noqa: E402

try:
    import torchvision
    from torchvision.models import efficientnet_b0
except ImportError:  # pragma: no cover
    torchvision = None

warnings.filterwarnings("ignore")
cv2.setNumThreads(1)
try:
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
except Exception:
    pass

# =====================================================================================
# CONFIGURATION
# =====================================================================================
SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent
DEFAULT_ROOT = SCRIPT_DIR / "run_151026"
SMOKE_ROOT = SCRIPT_DIR / "run_151026_smoke"
FACE_MODEL = SCRIPT_DIR / "models" / "face_detection_yunet_2023mar.onnx"
YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"

SOURCES = [  # (dataset, subset, label, folder)   label 0 = REAL, 1 = FAKE
    ("BioDeepAV", "real", 0, r"E:\Datasets\BioDeepAV\real\videos"),
    ("BioDeepAV", "fake", 1, r"E:\Datasets\BioDeepAV\fake\videos"),
    ("FF++", "original", 0, r"E:\Datasets\FaceForensics++_C23\original"),
    ("FF++", "Deepfakes", 1, r"E:\Datasets\FaceForensics++_C23\Deepfakes"),
    ("FF++", "Face2Face", 1, r"E:\Datasets\FaceForensics++_C23\Face2Face"),
    ("FF++", "FaceShifter", 1, r"E:\Datasets\FaceForensics++_C23\FaceShifter"),
    ("FF++", "FaceSwap", 1, r"E:\Datasets\FaceForensics++_C23\FaceSwap"),
    ("FF++", "NeuralTextures", 1, r"E:\Datasets\FaceForensics++_C23\NeuralTextures"),
    ("FF++", "DeepFakeDetection", 1, r"E:\Datasets\FaceForensics++_C23\DeepFakeDetection"),
]
VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".mpg", ".mpeg", ".wmv", ".flv"}
DATASETS = ["BioDeepAV", "FF++"]

N_FRAMES = 16             # frames sampled per video (uniformly over the whole video)
CLIP_FRAMES = 8           # frames per training clip (evaluation / prediction use all 16)
IMG = 192                 # network input size
FACE_PX = 112             # effective face resolution (resolution-shortcut bottleneck)
FACE_MARGIN = 0.30
FACE_SCORE = 0.6
DECODE_MAX_SIDE = 1280
SPLIT = (0.70, 0.15, 0.15)
SPLIT_SEED = 151026
SEEDS = "42"
EPOCHS = 12
BATCH_CLIPS = 8           # 8 clips x 8 frames = 64 face images per step
ACCUM = 2
LR_BACKBONE, LR_HEAD, WEIGHT_DECAY = 1e-4, 1e-3, 1e-4
WARMUP_FRAC = 0.08
LABEL_SMOOTH = 0.05
FRAME_LOSS_W = 0.3
EMA_DECAY = 0.998
MAX_OVERSAMPLE = 4.0
PATIENCE = 4
UNCERTAIN_BAND = 0.10     # |p - threshold| < band -> "uncertain"
NEARDUP_MEAN, NEARDUP_MAX = 4.0, 8
XAI_TOP = 0.20
FAITH_N = 200
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
METRIC_KEYS = ["auc", "accuracy", "balanced_accuracy", "precision", "recall", "specificity", "f1"]


# =====================================================================================
# UTILITIES
# =====================================================================================
def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def fmt_time(s) -> str:
    if s is None or not np.isfinite(s) or s < 0:
        return "--:--"
    m, s = divmod(int(s), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def bar(cur, tot, n=20) -> str:
    f = min(1.0, max(0.0, cur / max(1, tot)))
    k = int(f * n)
    return f"[{('=' * k + ('>' if k < n else '')).ljust(n, '.')}] {100 * f:5.1f}%"


def to_jsonable(o):
    if isinstance(o, dict):
        return {str(k): to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [to_jsonable(v) for v in o]
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if (np.isnan(o) or np.isinf(o)) else float(o)
    if isinstance(o, np.ndarray):
        return to_jsonable(o.tolist())
    return o


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(obj), f, indent=2)
    os.replace(tmp, path)


def read_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def atomic_save(obj, path: Path):
    tmp = path.with_name(path.name + ".writing")
    with open(tmp, "wb") as f:
        torch.save(obj, f)
        f.flush()
        os.fsync(f.fileno())
    # Windows: AV / indexer / sync clients can briefly lock the target -> retry instead of crashing
    for i in range(30):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if i == 29:
                raise
            time.sleep(min(0.5 * (i + 1), 5.0))


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def quick_hash(path: str, chunk: int = 2 << 20) -> str:
    """size + first/last 2 MB (exact-duplicate screen; full hashing of ~11k videos is not needed for that)."""
    h = hashlib.sha256()
    sz = os.path.getsize(path)
    h.update(str(sz).encode())
    with open(path, "rb") as f:
        h.update(f.read(chunk))
        if sz > 2 * chunk:
            f.seek(-chunk, 2)
            h.update(f.read(chunk))
    return h.hexdigest()


def md_table(df: pd.DataFrame, nd: int = 3, max_rows: int = 80) -> str:
    if df is None or not len(df):
        return "_(no rows)_"
    d = df.head(max_rows)

    def c(v):
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            return "n/a"
        if isinstance(v, (float, np.floating)):
            return f"{v:.{nd}f}"
        return str(v).replace("|", "/")

    out = ["| " + " | ".join(map(str, d.columns)) + " |", "|" + "---|" * len(d.columns)]
    out += ["| " + " | ".join(c(v) for v in r) + " |" for r in d.itertuples(index=False)]
    return "\n".join(out)


def imap_bounded(fn: Callable, items, workers: int, ahead: Optional[int] = None):
    ahead = max(1, ahead or 2 * workers)
    it = iter(items)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        pending = {}
        for x in it:
            pending[ex.submit(fn, x)] = x
            if len(pending) >= ahead:
                break
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for fut in done:
                x = pending.pop(fut)
                nxt = next(it, None)
                if nxt is not None:
                    pending[ex.submit(fn, nxt)] = nxt
                yield x, fut


class Tee:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(path, "a", encoding="utf-8")
        self.out = sys.stdout

    def write(self, s):
        self.out.write(s)
        self.f.write(s)
        if "\n" in s or "\r" in s:
            self.out.flush()
            self.f.flush()
        return len(s)

    def flush(self):
        self.out.flush()
        self.f.flush()


def lower_priority():
    try:
        if sys.platform == "win32":
            import ctypes
            ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x00004000)
        else:
            os.nice(5)
    except Exception:
        pass


class Paths:
    def __init__(self, root: Path):
        self.root = Path(root)
        for d in ("data", "cache", "splits", "checkpoints", "logs", "metrics", "figures", "reports", "xai", "predictions"):
            setattr(self, d, self.root / d)
        self.inventory = self.data / "inventory.csv"
        self.split_csv = self.splits / "split.csv"
        self.best = self.checkpoints / "best_model.pt"

    def make(self):
        for d in ("data", "cache", "splits", "checkpoints", "logs", "metrics", "figures", "reports", "xai", "predictions"):
            getattr(self, d).mkdir(parents=True, exist_ok=True)


# =====================================================================================
# FACE PIPELINE (identical for training, evaluation and prediction)
# =====================================================================================
def ensure_face_model() -> Optional[Path]:
    if FACE_MODEL.exists():
        return FACE_MODEL
    try:
        FACE_MODEL.parent.mkdir(parents=True, exist_ok=True)
        print(f"  downloading YuNet face detector -> {FACE_MODEL}")
        urllib.request.urlretrieve(YUNET_URL, FACE_MODEL)
        return FACE_MODEL
    except Exception as e:
        print(f"  [warn] YuNet unavailable ({type(e).__name__}); using OpenCV Haar cascade")
        return None


class FaceDetector:
    """Thread-safe: YuNet if available, else Haar cascade. Returns the largest, most confident face (x, y, w, h)."""

    def __init__(self, model_path: Optional[Path], score: float = FACE_SCORE, max_side: int = 640, quiet: bool = False):
        self.model_path, self.score, self.max_side = model_path, score, max_side
        self.backend = ("yunet" if (model_path and Path(model_path).exists() and hasattr(cv2, "FaceDetectorYN"))
                        else "haar" if hasattr(cv2, "CascadeClassifier") else "none")
        if self.backend == "none" and not quiet:
            print("  [warn] no face detector available (YuNet model missing and no Haar cascade) - centre crops are used")
        self._local = threading.local()

    def detect(self, bgr):
        H, W = bgr.shape[:2]
        sc = min(1.0, self.max_side / max(H, W))
        img = cv2.resize(bgr, (int(W * sc), int(H * sc)), interpolation=cv2.INTER_AREA) if sc < 1 else bgr
        if self.backend == "yunet":
            d = getattr(self._local, "det", None)
            if d is None:
                d = cv2.FaceDetectorYN.create(str(self.model_path), "", (320, 320), self.score, 0.3, 50)
                self._local.det = d
            d.setInputSize((img.shape[1], img.shape[0]))
            try:
                _, faces = d.detect(img)
            except Exception:
                return None
            if faces is None or not len(faces):
                return None
            faces = faces[(faces[:, 2] >= 16) & (faces[:, 3] >= 16)]
            if not len(faces):
                return None
            b = faces[np.argmax(faces[:, 2] * faces[:, 3] * faces[:, 14])]
            return np.array(b[:4], np.float32) / sc
        if self.backend == "none":
            return None
        d = getattr(self._local, "haar", None)
        if d is None:
            try:
                d = cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml"))
            except Exception:              # OpenCV 5 builds without the legacy cascade module
                self.backend = "none"
                return None
            self._local.haar = d
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = d.detectMultiScale(g, 1.1, 5, minSize=(24, 24))
        if faces is None or not len(faces):
            return None
        b = faces[np.argmax(faces[:, 2] * faces[:, 3])]
        return np.array(b, np.float32) / sc


def open_video(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
    return cap


def read_frames(path, n: int = N_FRAMES, max_side: int = DECODE_MAX_SIDE):
    """n frames uniformly spaced over the video (BGR) + metadata. Never keeps the whole video in memory."""
    cap = open_video(path)
    meta = dict(width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                fps=float(cap.get(cv2.CAP_PROP_FPS) or 0.0), total_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))

    def shrink(fr):
        h, w = fr.shape[:2]
        s = max_side / max(h, w)
        return cv2.resize(fr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else fr

    frames = {}
    total = meta["total_frames"]
    if total > 0:
        want = set(np.linspace(0, total - 1, n).round().astype(int).tolist())
        last, i = max(want), 0
        while i <= last:
            if not cap.grab():
                break
            if i in want:
                ok, fr = cap.retrieve()
                if ok:
                    frames[i] = shrink(fr)
            i += 1
    cap.release()
    if not frames:
        cap = open_video(path)
        count = 0
        while cap.grab():
            count += 1
        cap.release()
        if not count:
            return None, meta
        want = set(np.linspace(0, count - 1, n).round().astype(int).tolist())
        cap = open_video(path)
        for j in range(max(want) + 1):
            ok, fr = cap.read()
            if not ok:
                break
            if j in want:
                frames[j] = shrink(fr)
        cap.release()
        if not frames:
            return None, meta
    keys = sorted(frames)
    pick = np.linspace(0, keys[-1], n).round().astype(int)
    return [frames[min(keys, key=lambda k: abs(k - t))] for t in pick], meta


def square_box(b, W, H, margin):
    x, y, w, h = b
    cx, cy = x + w / 2, y + h / 2
    s = max(w, h) * (1 + 2 * margin)
    s = min(s, W, H)
    x0 = float(np.clip(cx - s / 2, 0, W - s))
    y0 = float(np.clip(cy - s / 2, 0, H - s))
    return np.array([x0, y0, s], np.float32)


def face_bottleneck(patch, img: int, face_px: int):
    p = cv2.resize(patch, (face_px, face_px), interpolation=cv2.INTER_AREA if patch.shape[0] > face_px else cv2.INTER_LINEAR)
    return cv2.resize(p, (img, img), interpolation=cv2.INTER_LINEAR)


def extract_faces(path, detector: FaceDetector, n: int = N_FRAMES, img: int = IMG, face_px: int = FACE_PX,
                  margin: float = FACE_MARGIN, keep_frames: bool = False):
    """video -> n RGB face crops uint8 [n,img,img,3] + info (boxes in frame pixels, detection rate, fallback)."""
    info = dict(ok=False, detected=0, detection_rate=0.0, fallback=True, native_face_px=float("nan"), error="")
    try:
        frames, meta = read_frames(path, n)
        info.update(meta)
        if frames is None:
            info["error"] = "no decodable frames"
            return None, info
        H, W = frames[0].shape[:2]
        dets = [detector.detect(f) for f in frames]
        hit = np.array([d is not None for d in dets])
        info["detected"], info["detection_rate"] = int(hit.sum()), float(hit.mean())
        boxes = np.zeros((n, 3), np.float32)
        if hit.any():
            idx = np.where(hit)[0]
            for t in range(n):
                d = dets[t] if hit[t] else dets[idx[np.argmin(np.abs(idx - t))]]
                boxes[t] = square_box(d, W, H, margin)
            sm = boxes.copy()                                   # temporal smoothing (3-frame median)
            for t in range(n):
                sm[t] = np.median(boxes[max(0, t - 1):t + 2], axis=0)
            boxes = sm
            info["fallback"] = False
            info["native_face_px"] = float(np.median([d[2] for d in dets if d is not None]) * (info["width"] / W if W and info["width"] else 1.0))
        else:
            s = float(min(W, H))
            boxes[:] = [(W - s) / 2, (H - s) / 2, s]
        crops = []
        for t, f in enumerate(frames):
            x0, y0, s = boxes[t]
            x0, y0, s = int(round(x0)), int(round(y0)), max(8, int(round(s)))
            patch = f[y0:y0 + s, x0:x0 + s]
            if patch.size == 0:
                patch = f
            crops.append(face_bottleneck(patch, img, face_px)[..., ::-1])
        info["ok"] = True
        info["boxes"] = boxes
        info["frame_hw"] = (H, W)
        if keep_frames:
            info["frames"] = frames
        return np.ascontiguousarray(np.stack(crops)), info
    except Exception as e:
        info["error"] = f"{type(e).__name__}: {e}"
        return None, info


# =====================================================================================
# INVENTORY + FACE CACHE
# =====================================================================================
def yt_id(stem: str) -> Optional[str]:
    m = re.match(r"^([A-Za-z0-9_-]{11})_\d{4}", stem)
    return m.group(1) if m else None


def identity_tokens(ds: str, subset: str, label: int, stem: str) -> List[str]:
    if ds == "BioDeepAV":
        if label == 0:
            y = yt_id(stem)
            return [f"bio_yt:{y}"] if y else [f"bio_file:{stem}"]
        return [f"bio_fake:{stem}"]
    if subset == "DeepFakeDetection":
        m = re.match(r"^(\d+)_(\d+)__", stem)
        return [f"dfd_actor:{m.group(1)}", f"dfd_actor:{m.group(2)}"] if m else [f"dfd_file:{stem}"]
    if subset == "original":
        return [f"ff:{stem.split('_')[0]}"] if stem.split("_")[0].isdigit() else [f"ff_file:{stem}"]
    p = stem.split("_")
    if len(p) >= 2 and p[0].isdigit() and p[1].isdigit():
        return [f"ff:{p[0]}", f"ff:{p[1]}"]
    return [f"ff_file:{stem}"]


def phase_inventory(a, P: Paths) -> pd.DataFrame:
    print("\n" + "=" * 90 + "\n PHASE INVENTORY: original real / fake videos only\n" + "=" * 90)
    rows = []
    for ds, sub, lab, folder in a.sources:
        fp = Path(folder)
        if not fp.exists():
            print(f"  [warn] missing folder {fp}")
            continue
        files = sorted(p for p in fp.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXT)
        if a.max_per_folder:
            rng = np.random.default_rng(SPLIT_SEED)
            files = [files[i] for i in sorted(rng.choice(len(files), min(len(files), a.max_per_folder), replace=False))]
        for f in files:
            rows.append(dict(dataset=ds, subset=sub, label=lab, path=str(f), stem=f.stem, file_size=f.stat().st_size))
        print(f"  {ds:10s} {sub:18s} {'FAKE' if lab else 'REAL'}  {len(files):6,d} videos  ({fp})")
    inv = pd.DataFrame(rows)
    if not len(inv):
        raise SystemExit("[STOP] no videos found - check the dataset folders")
    print(f"  hashing {len(inv):,} files (exact-duplicate screen) ...")
    hashes = [None] * len(inv)
    for i, fut in imap_bounded(lambda i: quick_hash(inv.path.iloc[i]), list(range(len(inv))), a.io_workers):
        hashes[i] = fut.result()
    inv["quick_hash"] = hashes
    inv["tokens"] = [";".join(identity_tokens(r.dataset, r.subset, r.label, r.stem)) for r in inv.itertuples()]
    inv["idx"] = np.arange(len(inv))
    inv.to_csv(P.inventory, index=False)
    print(f"  inventory: {len(inv):,} videos | REAL {int((inv.label == 0).sum()):,} | FAKE {int((inv.label == 1).sum()):,} -> {P.inventory}")
    return inv


class FaceCache:
    def __init__(self, P: Paths, inv: pd.DataFrame, img: int, face_px: int):
        key = sha256_text("\n".join(f"{p}|{s}" for p, s in zip(inv.path, inv.file_size)) + f"|{N_FRAMES}|{img}|{face_px}|{FACE_MARGIN}")
        self.n, self.img = len(inv), img
        stem = f"faces_n{self.n}_T{N_FRAMES}_S{img}_F{face_px}_{key[:12]}"
        self.arr_p, self.st_p, self.info_p = (P.cache / f"{stem}{s}" for s in (".npy", ".status.npy", ".info.csv"))
        self.arr = None

    def complete(self) -> bool:
        return self.arr_p.exists() and self.st_p.exists() and bool((np.load(self.st_p) != 0).all())

    def load(self):
        self.arr = np.load(self.arr_p, mmap_mode="r")
        self.status = np.load(self.st_p)
        self.info = pd.read_csv(self.info_p)
        return self


def phase_faces(a, P: Paths, inv: pd.DataFrame) -> FaceCache:
    print("\n" + "=" * 90 + f"\n PHASE FACES: {N_FRAMES} face crops per video ({a.img}px, effective face {a.face_px}px)\n" + "=" * 90)
    C = FaceCache(P, inv, a.img, a.face_px)
    if C.complete():
        print(f"  complete cache found: {C.arr_p.name}")
        return C.load()
    shape = (C.n, N_FRAMES, a.img, a.img, 3)
    need = int(np.prod(shape))
    if not C.arr_p.exists() and need > shutil.disk_usage(P.cache).free - 2e9:
        raise SystemExit(f"[STOP] face cache needs {need / 1e9:.1f} GB on {P.cache}; use --img 160 or free space")
    arr = np.lib.format.open_memmap(C.arr_p, mode="r+" if C.arr_p.exists() else "w+", dtype=np.uint8, shape=shape)
    status = np.load(C.st_p) if C.st_p.exists() else np.zeros(C.n, np.int8)
    info = pd.read_csv(C.info_p).to_dict("records") if C.info_p.exists() else [dict(idx=i) for i in range(C.n)]
    det = FaceDetector(ensure_face_model())
    print(f"  face detector: {det.backend} | {C.n:,} videos | cache {need / 1e9:.1f} GB -> {C.arr_p}")
    todo = [int(i) for i in np.where(status == 0)[0]]

    def work(i):
        return extract_faces(inv.path.iloc[i], det, N_FRAMES, a.img, a.face_px)

    def save():
        arr.flush()
        np.save(C.st_p, status)
        pd.DataFrame(info).to_csv(C.info_p, index=False)

    t0, done, last_s, last_p = time.time(), 0, time.time(), 0
    for i, fut in imap_bounded(work, todo, a.workers):
        clip, inf = fut.result()
        if clip is not None:
            arr[i] = clip
            status[i] = 1
        else:
            status[i] = 2
        info[i] = dict(idx=i, ok=bool(clip is not None), detection_rate=inf.get("detection_rate", 0.0), fallback=inf.get("fallback", True),
                       native_face_px=inf.get("native_face_px", np.nan), width=inf.get("width", 0), height=inf.get("height", 0),
                       fps=inf.get("fps", 0.0), total_frames=inf.get("total_frames", 0), error=str(inf.get("error", ""))[:120])
        done += 1
        t = time.time()
        if t - last_s > 60:
            save()
            last_s = t
        if t - last_p > 0.5 or done == len(todo):
            r = done / max(1e-3, t - t0)
            sys.stdout.write(f"\r  {bar(done, len(todo))} {done:,}/{len(todo):,} | {r:4.1f} vid/s | ETA {fmt_time((len(todo) - done) / max(r, 1e-3))}   ")
            sys.stdout.flush()
            last_p = t
    sys.stdout.write("\n")
    save()
    del arr
    C.load()
    ok = C.info.ok.astype(bool)
    print(f"  extracted {int(ok.sum()):,} | failed {int((~ok).sum()):,} | mean face detection rate {C.info.detection_rate[ok].mean():.3f} | "
          f"no face found in {int(C.info.fallback[ok].astype(bool).sum()):,} videos (centre crop)")
    return C


# =====================================================================================
# GROUPING + SPLIT (identity + duplicate aware)
# =====================================================================================
class UnionFind:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def dhash64(gray: np.ndarray) -> np.uint64:
    s = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = (s[:, 1:] > s[:, :-1]).flatten()
    return np.uint64(int("".join("1" if b else "0" for b in bits), 2))


POP8 = np.array([bin(i).count("1") for i in range(256)], np.uint8)


def fingerprints(C: FaceCache) -> Tuple[np.ndarray, np.ndarray]:
    H = np.zeros((C.n, 4), np.uint64)
    valid = np.zeros(C.n, bool)
    for i in range(C.n):
        if C.status[i] != 1:
            continue
        fr = C.arr[i, [0, 5, 10, 15]]
        g = fr.mean(-1).astype(np.float32)
        if g.std() < 4:
            continue
        H[i] = [dhash64(x) for x in g]
        valid[i] = True
    return H, valid


def near_duplicates(H: np.ndarray, valid: np.ndarray, chunk: int = 256) -> List[Tuple[int, int]]:
    idx = np.where(valid)[0]
    Hv = H[idx]
    out = []
    for s in range(0, len(idx), chunk):
        x = Hv[s:s + chunk]
        xor = (x[:, None, :] ^ Hv[None, :, :]).view(np.uint8).reshape(len(x), len(Hv), 4, 8)
        d = POP8[xor].sum(-1).astype(np.int16)                # [c, N, 4] Hamming per frame
        ok = (d.mean(-1) <= NEARDUP_MEAN) & (d.max(-1) <= NEARDUP_MAX)
        ii, jj = np.where(ok)
        for i, j in zip(ii + s, jj):
            if i < j:
                out.append((int(idx[i]), int(idx[j])))
    return out


def make_split(groups: Dict[int, List[int]], inv: pd.DataFrame, seed: int, ratios=SPLIT) -> Dict[int, str]:
    """Greedy, stratified by (dataset, label): biggest groups first, each to the split with the most room."""
    names = ["train", "val", "test"]
    keys = sorted(set(zip(inv.dataset, inv.label)))
    total = Counter(zip(inv.dataset, inv.label))
    target = {s: {k: max(1e-9, r * total[k]) for k in keys} for s, r in zip(names, ratios)}
    cur = {s: Counter() for s in names}
    rng = np.random.default_rng(seed)
    order = list(groups)
    rng.shuffle(order)
    order.sort(key=lambda g: -len(groups[g]))
    out = {}
    for g in order:
        cnt = Counter(zip(inv.dataset.values[groups[g]], inv.label.values[groups[g]]))
        score = {s: sum(c * (cur[s][k] + c) / target[s][k] for k, c in cnt.items()) for s in names}
        s = min(names, key=lambda x: score[x])
        out[g] = s
        cur[s].update(cnt)
    return out


def phase_split(a, P: Paths, inv: pd.DataFrame, C: FaceCache) -> pd.DataFrame:
    print("\n" + "=" * 90 + "\n PHASE SPLIT: identity + duplicate grouping, 70 / 15 / 15\n" + "=" * 90)
    if P.split_csv.exists():
        sp = pd.read_csv(P.split_csv)
        if len(sp) == len(inv) and (sp.path.values == inv.path.values).all():
            print(f"  existing split found -> {P.split_csv}")
            return sp
    uf = UnionFind(len(inv))
    first = {}
    for i, toks in enumerate(inv.tokens):
        for t in toks.split(";"):
            if t in first:
                uf.union(i, first[t])
            else:
                first[t] = i
    n_exact = 0
    for _, g in inv.groupby("quick_hash"):
        ids = g.idx.values
        for j in ids[1:]:
            uf.union(int(ids[0]), int(j))
            n_exact += 1
    print("  face fingerprints + near-duplicate screen ...")
    H, valid = fingerprints(C)
    nd = near_duplicates(H, valid)
    for i, j in nd:
        uf.union(i, j)
    root = np.array([uf.find(i) for i in range(len(inv))])
    ok = C.status == 1
    groups = defaultdict(list)
    for i in np.where(ok)[0]:
        groups[int(root[i])].append(int(i))
    assign = make_split(dict(groups), inv, SPLIT_SEED)
    sp = inv.copy()
    sp["group"] = [f"G{r}" for r in root]
    sp["split"] = [assign.get(int(r), "excluded") if ok[i] else "excluded" for i, r in enumerate(root)]
    sp.to_csv(P.split_csv, index=False)
    cross = sp[sp.split != "excluded"].groupby("group").split.nunique().max()
    print(f"  {len(groups):,} groups | exact duplicates merged {n_exact} | near-duplicate pairs merged {len(nd):,} | "
          f"groups in >1 split: {int(cross > 1)} (must be 0)")
    tab = sp[sp.split != "excluded"].pivot_table(index=["dataset", "subset"], columns="split", values="idx", aggfunc="count", fill_value=0)
    print(tab.to_string())
    for ds in DATASETS:
        for s in ("val", "test"):
            for lab in (0, 1):
                n = int(((sp.dataset == ds) & (sp.split == s) & (sp.label == lab)).sum())
                if 0 < n < 20 and not a.smoke:
                    print(f"  [warn] only {n} {'fake' if lab else 'real'} {ds} videos in {s}")
    write_json(P.splits / "split_summary.json", dict(created=now(), groups=len(groups), exact_duplicates_merged=n_exact,
                                                    near_duplicate_pairs=len(nd), groups_crossing_splits=int(cross > 1),
                                                    counts=sp.groupby(["dataset", "subset", "split"]).size().to_dict()))
    return sp


# =====================================================================================
# MODEL
# =====================================================================================
class DeepfakeDetector(nn.Module):
    """EfficientNet-B0 frame encoder -> temporal transformer -> attention pooling -> P(fake)."""

    def __init__(self, pretrained: bool = True, d: int = 256, max_t: int = N_FRAMES):
        super().__init__()
        if torchvision is None:
            raise SystemExit("[STOP] torchvision is required: pip install torchvision")
        weights = None
        if pretrained:
            try:
                weights = torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1
            except Exception:
                weights = None
        try:
            net = efficientnet_b0(weights=weights)
            self.pretrained = weights is not None
        except Exception as e:
            print(f"  [warn] ImageNet weights could not be loaded ({type(e).__name__}) - training from scratch (weaker)")
            net = efficientnet_b0(weights=None)
            self.pretrained = False
        self.backbone = net.features                     # [N,1280,h,w]
        self.proj = nn.Sequential(nn.Linear(1280, d), nn.LayerNorm(d), nn.GELU(), nn.Dropout(0.2))
        self.pos = nn.Embedding(max_t, d)
        self.temporal = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 4, 2 * d, 0.1, activation="gelu", batch_first=True,
                                                                         norm_first=True), 1, enable_nested_tensor=False)
        self.att = nn.Linear(d, 1)
        self.frame_head = nn.Linear(d, 1)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Dropout(0.3), nn.Linear(d, 1))

    @property
    def cam_layer(self):
        return self.backbone[-1]

    def forward(self, x, t_idx=None):
        """x [B,T,3,H,W] normalised; t_idx [B,T] frame positions (0..15). Returns dict of logits and attention."""
        B, T = x.shape[:2]
        f = self.backbone(x.flatten(0, 1))
        h = self.proj(f.mean((2, 3))).view(B, T, -1)
        if t_idx is None:
            t_idx = torch.arange(T, device=x.device).expand(B, T)
        h = self.temporal(h + self.pos(t_idx))
        w = torch.softmax(self.att(h).squeeze(-1).float(), 1)
        z = (w.unsqueeze(-1) * h.float()).sum(1)
        return dict(logit=self.head(z.to(h.dtype)).squeeze(-1).float(), frame_logit=self.frame_head(h).squeeze(-1).float(), attn=w)


def normalise(clips_u8: np.ndarray, device) -> torch.Tensor:
    """uint8 [B,T,H,W,3] RGB -> float [B,T,3,H,W] ImageNet-normalised."""
    x = torch.from_numpy(np.ascontiguousarray(clips_u8)).to(device, non_blocking=True).float().div_(255.0)
    x = (x - torch.tensor(MEAN, device=device)) / torch.tensor(STD, device=device)
    return x.permute(0, 1, 4, 2, 3).contiguous()


class EMA:
    def __init__(self, model, decay):
        import copy
        self.decay = decay
        self.model = copy.deepcopy(model).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for e, p in zip(self.model.parameters(), model.parameters()):
            e.mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)
        for e, b in zip(self.model.buffers(), model.buffers()):
            e.copy_(b)


# =====================================================================================
# AUGMENTATION (real-world quality changes; same parameters for every frame of a clip)
# =====================================================================================
def augment_clip(clip: np.ndarray, rng) -> np.ndarray:
    T, S = clip.shape[:2]
    out = clip
    sc = rng.uniform(0.80, 1.0)
    cs = int(S * sc)
    y0, x0 = rng.integers(0, S - cs + 1, 2)
    flip = rng.random() < 0.5
    bright, contrast, sat = rng.uniform(-0.15, 0.15), rng.uniform(0.85, 1.15), rng.uniform(0.8, 1.2)
    jpeg = int(rng.integers(40, 96)) if rng.random() < 0.5 else None
    blur = float(rng.uniform(0.4, 1.4)) if rng.random() < 0.2 else None
    down = float(rng.uniform(0.45, 0.9)) if rng.random() < 0.3 else None
    erase = None
    if rng.random() < 0.25:
        ew, eh = int(S * rng.uniform(0.15, 0.4)), int(S * rng.uniform(0.15, 0.4))
        erase = (int(rng.integers(0, S - eh)), int(rng.integers(0, S - ew)), eh, ew)
    res = np.empty_like(out)
    for t in range(T):
        f = out[t, y0:y0 + cs, x0:x0 + cs]
        f = cv2.resize(f, (S, S), interpolation=cv2.INTER_LINEAR)
        if flip:
            f = f[:, ::-1]
        if down:
            k = max(16, int(S * down))
            f = cv2.resize(cv2.resize(f, (k, k), interpolation=cv2.INTER_AREA), (S, S), interpolation=cv2.INTER_LINEAR)
        if blur:
            f = cv2.GaussianBlur(f, (0, 0), blur)
        g = f.astype(np.float32)
        gray = g.mean(-1, keepdims=True)
        g = gray + (g - gray) * sat
        g = (g - 128.0) * contrast + 128.0 + 255.0 * bright
        f = np.clip(g, 0, 255).astype(np.uint8)
        if jpeg:
            ok, enc = cv2.imencode(".jpg", f[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, jpeg])
            if ok:
                f = cv2.imdecode(enc, cv2.IMREAD_COLOR)[..., ::-1]
        if erase:
            ey, ex, eh, ew = erase
            f = f.copy()
            f[ey:ey + eh, ex:ex + ew] = rng.integers(0, 256, 3, dtype=np.uint8)
        res[t] = f
    return res


# =====================================================================================
# METRICS + CALIBRATION
# =====================================================================================
def compute_metrics(y, p, thr) -> dict:
    y = np.asarray(y).astype(int)
    p = np.asarray(p, float)
    out = dict(n=len(y), n_real=int((y == 0).sum()), n_fake=int((y == 1).sum()), threshold=float(thr))
    if not len(y):
        return {**out, **{k: np.nan for k in METRIC_KEYS}, "tn": 0, "fp": 0, "fn": 0, "tp": 0}
    pr = (p >= thr).astype(int)
    tn, fp, fn, tp = (int(v) for v in confusion_matrix(y, pr, labels=[0, 1]).ravel())
    rec = tp / (tp + fn) if tp + fn else np.nan
    spec = tn / (tn + fp) if tn + fp else np.nan
    prec = tp / (tp + fp) if tp + fp else np.nan
    f1 = 2 * prec * rec / (prec + rec) if np.isfinite(prec) and np.isfinite(rec) and prec + rec > 0 else 0.0
    auc = roc_auc_score(y, p) if len(np.unique(y)) > 1 else np.nan
    bacc = np.nanmean([rec, spec]) if np.isfinite(rec) or np.isfinite(spec) else np.nan
    return {**out, "auc": auc, "accuracy": float((pr == y).mean()), "balanced_accuracy": bacc, "precision": prec, "recall": rec,
            "specificity": spec, "f1": f1, "tn": tn, "fp": fp, "fn": fn, "tp": tp}


def fit_temperature(logits, y) -> float:
    x, y = np.asarray(logits, float), np.asarray(y, float)
    grid = np.exp(np.linspace(np.log(0.05), np.log(20), 241))
    nll = [np.mean(np.logaddexp(0, x / t) - y * x / t) for t in grid]
    return float(grid[int(np.argmin(nll))])


def choose_threshold(y, p, ds) -> float:
    """Maximise the MEAN balanced accuracy over datasets (each dataset counts equally)."""
    best, bt = -1, 0.5
    for t in np.round(np.arange(0.05, 0.951, 0.01), 2):
        vals = []
        for d in np.unique(ds):
            m = ds == d
            yy, pp = y[m], p[m] >= t
            r = pp[yy == 1].mean() if (yy == 1).any() else np.nan
            s = (~pp[yy == 0]).mean() if (yy == 0).any() else np.nan
            vals.append(np.nanmean([r, s]))
        v = np.nanmean(vals) - 1e-4 * abs(t - 0.5)
        if v > best:
            best, bt = v, float(t)
    return bt


def ece(y, p, bins=15) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    b = np.minimum((p * bins).astype(int), bins - 1)
    return float(sum((b == k).sum() * abs(p[b == k].mean() - y[b == k].mean()) for k in range(bins) if (b == k).any()) / max(1, len(y)))


def sigmoid(x):
    return 1 / (1 + np.exp(-np.asarray(x, float)))


# =====================================================================================
# TRAINING
# =====================================================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


class Data:
    def __init__(self, sp: pd.DataFrame, C: FaceCache):
        self.sp, self.C = sp, C
        self.y = sp.label.values.astype(np.int64)
        self.ds = sp.dataset.values
        self.cell = np.array([f"{d}|{l}" for d, l in zip(sp.dataset, sp.label)])

    def idx(self, split, dataset=None):
        m = self.sp.split.values == split
        if dataset:
            m &= self.ds == dataset
        return np.where(m)[0]


def balanced_weights(cells: np.ndarray, epoch_size: int) -> np.ndarray:
    names, counts = np.unique(cells, return_counts=True)
    share = {c: 1.0 / len(names) for c in names}
    cap = {c: MAX_OVERSAMPLE * n / epoch_size for c, n in zip(names, counts)}
    for _ in range(10):
        over = [c for c in names if share[c] > cap[c] + 1e-12]
        if not over:
            break
        free = sum(share[c] - cap[c] for c in over)
        for c in over:
            share[c] = cap[c]
        rest = [c for c in names if c not in over]
        tot = sum(share[c] for c in rest)
        for c in rest:
            share[c] += free * share[c] / tot if tot else 0
    cnt = dict(zip(names, counts))
    w = np.array([share[c] / cnt[c] for c in cells])
    return w / w.sum()


class Trainer:
    def __init__(self, a, P: Paths, D: Data, seed: int):
        self.a, self.P, self.D, self.seed = a, P, D, seed
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.amp = self.dev.type == "cuda" and not a.no_amp
        self.ck_last = P.checkpoints / f"seed{seed}_last.pt"
        self.ck_best = P.checkpoints / f"seed{seed}_best.pt"

    def batch(self, items, seed, train=True):
        rng = np.random.default_rng(seed)
        clips, tix = [], []
        for i in items:
            if train:
                sel = np.sort(rng.choice(N_FRAMES, CLIP_FRAMES, replace=False))
                clips.append(augment_clip(np.asarray(self.D.C.arr[i, sel]), rng))
            else:
                sel = np.arange(N_FRAMES)
                clips.append(np.asarray(self.D.C.arr[i]))
            tix.append(sel)
        return np.stack(clips), np.stack(tix), self.D.y[items]

    @torch.no_grad()
    def predict(self, model, idx: np.ndarray, bs: int = 8) -> Tuple[np.ndarray, np.ndarray]:
        model.eval()
        lo, att = [], []
        for s in range(0, len(idx), bs):
            x, t, _ = self.batch(idx[s:s + bs], 0, train=False)
            with torch.autocast("cuda", dtype=torch.float16, enabled=self.amp):
                o = model(normalise(x, self.dev), torch.from_numpy(t).to(self.dev))
            lo.append(o["logit"].float().cpu().numpy())
            att.append(o["attn"].float().cpu().numpy())
        return (np.concatenate(lo), np.concatenate(att)) if lo else (np.zeros(0), np.zeros((0, N_FRAMES)))

    def validate(self, model) -> dict:
        va = self.D.idx("val")
        lg, _ = self.predict(model, va)
        row = {}
        for d in DATASETS:
            m = self.D.ds[va] == d
            if m.any() and len(np.unique(self.D.y[va][m])) > 1:
                row[f"val_auc_{d}"] = roc_auc_score(self.D.y[va][m], lg[m])
        row["val_score"] = float(np.mean([v for k, v in row.items() if k.startswith("val_auc_")])) if row else np.nan
        return row

    def train(self) -> Path:
        a, D = self.a, self.D
        if self.ck_best.exists() and torch.load(self.ck_best, map_location="cpu", weights_only=False).get("final"):
            print(f"  [resume] seed {self.seed} already trained -> {self.ck_best}")
            return self.ck_best
        set_seed(self.seed)
        model = DeepfakeDetector(pretrained=not a.no_pretrained).to(self.dev)
        print(f"  model: EfficientNet-B0 ({'ImageNet' if model.pretrained else 'random'} init) + temporal transformer | "
              f"{sum(p.numel() for p in model.parameters()):,} parameters | device {self.dev} | AMP {self.amp}")
        bb = list(model.backbone.parameters())
        other = [p for n, p in model.named_parameters() if not n.startswith("backbone.")]
        opt = torch.optim.AdamW([dict(params=bb, lr=a.lr_backbone), dict(params=other, lr=a.lr_head)], weight_decay=WEIGHT_DECAY)
        tr = D.idx("train")
        steps = int(math.ceil(len(tr) / (a.batch * a.accum))) * a.accum
        if a.max_steps:
            steps = min(steps, a.max_steps * a.accum)
        total_upd = max(1, steps // a.accum * a.epochs)
        warm = max(1, int(WARMUP_FRAC * total_upd))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda u: (u + 1) / warm if u < warm else
                                                  0.5 * (1 + math.cos(math.pi * min(1.0, (u - warm) / max(1, total_upd - warm)))))
        scaler = torch.amp.GradScaler("cuda", enabled=self.amp)
        ema = EMA(model, a.ema_decay)
        hist, best, start_ep, updates, bad = [], dict(score=-1.0, epoch=0), 1, 0, 0
        if self.ck_last.exists():
            st = torch.load(self.ck_last, map_location=self.dev, weights_only=False)
            model.load_state_dict(st["model"])
            ema.model.load_state_dict(st["ema"])
            opt.load_state_dict(st["opt"])
            sched.load_state_dict(st["sched"])
            scaler.load_state_dict(st["scaler"])
            hist, best, start_ep, updates, bad = st["hist"], st["best"], st["epoch"] + 1, st["updates"], st["bad"]
            print(f"  [resume] seed {self.seed}: continuing at epoch {start_ep}")
        w = balanced_weights(D.cell[tr], steps * a.batch)
        bce = nn.BCEWithLogitsLoss()
        loader = ThreadPoolExecutor(max_workers=max(1, a.loader_threads))
        print(f"  train {len(tr):,} videos (dataset x class balanced) | val {len(D.idx('val')):,} | {steps} steps x {a.epochs} epochs | "
              f"{a.batch} clips x {CLIP_FRAMES} frames x accum {a.accum}")
        t_run = time.time()
        for ep in range(start_ep, a.epochs + 1):
            if bad >= PATIENCE:
                print(f"  early stop: no validation improvement for {PATIENCE} epochs")
                break
            model.train()
            rng = np.random.default_rng([self.seed, ep])
            draws = rng.choice(tr, size=steps * a.batch, replace=True, p=w)
            seeds = rng.integers(0, 2 ** 31, steps)
            q, nxt = deque(), 0

            def submit():
                nonlocal nxt
                if nxt < steps:
                    q.append(loader.submit(self.batch, draws[nxt * a.batch:(nxt + 1) * a.batch], int(seeds[nxt])))
                    nxt += 1

            for _ in range(2 * a.loader_threads + 1):
                submit()
            tot, t0, last = 0.0, time.time(), 0
            opt.zero_grad(set_to_none=True)
            for st in range(steps):
                x, tix, y = q.popleft().result()
                submit()
                ys = torch.from_numpy(y).float().to(self.dev) * (1 - a.label_smooth) + 0.5 * a.label_smooth
                with torch.autocast("cuda", dtype=torch.float16, enabled=self.amp):
                    o = model(normalise(x, self.dev), torch.from_numpy(tix).to(self.dev))
                    loss = bce(o["logit"], ys) + FRAME_LOSS_W * bce(o["frame_logit"], ys[:, None].expand_as(o["frame_logit"]))
                scaler.scale(loss / a.accum).backward()
                if (st + 1) % a.accum == 0:
                    scaler.unscale_(opt)
                    nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    scaler.step(opt)
                    scaler.update()
                    opt.zero_grad(set_to_none=True)
                    if updates < total_upd - 1:
                        sched.step()
                    updates += 1
                    ema.update(model)
                tot += float(loss.item())
                if time.time() - last > 0.5 or st + 1 == steps:
                    el = time.time() - t0
                    sys.stdout.write(f"\r  [seed {self.seed}] ep {ep}/{a.epochs} {bar(st + 1, steps, 16)} loss {tot / (st + 1):.4f} | "
                                     f"epoch ETA {fmt_time(el / (st + 1) * (steps - st - 1))}   ")
                    sys.stdout.flush()
                    last = time.time()
            row = dict(epoch=ep, train_loss=tot / steps, epoch_time_s=time.time() - t0, **self.validate(ema.model))
            hist.append(row)
            better = np.isfinite(row["val_score"]) and row["val_score"] > best["score"]
            if better:
                best, bad = dict(score=row["val_score"], epoch=ep), 0
                atomic_save(dict(model=ema.model.state_dict(), epoch=ep, val=row), self.ck_best.with_suffix(".tmp"))
            else:
                bad += 1
            print(f"\r  [seed {self.seed}] ep {ep}/{a.epochs} | loss {row['train_loss']:.4f} | "
                  + " | ".join(f"{k.replace('val_auc_', 'val AUC ')} {v:.4f}" for k, v in row.items() if k.startswith("val_auc_"))
                  + f" | score {row['val_score']:.4f}{'  <- best' if better else ''} | {fmt_time(row['epoch_time_s'])}" + " " * 10)
            atomic_save(dict(model=model.state_dict(), ema=ema.model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(),
                             scaler=scaler.state_dict(), hist=hist, best=best, epoch=ep, updates=updates, bad=bad), self.ck_last)
        loader.shutdown(wait=True)
        pd.DataFrame(hist).to_csv(self.P.logs / f"history_seed{self.seed}.csv", index=False)
        tmp = self.ck_best.with_suffix(".tmp")
        if not tmp.exists():
            raise RuntimeError("no valid validation score - nothing saved")
        model.load_state_dict(torch.load(tmp, map_location=self.dev, weights_only=False)["model"])
        # calibration + threshold on validation
        va = D.idx("val")
        lg, _ = self.predict(model, va)
        T = fit_temperature(lg, D.y[va])
        p = sigmoid(lg / T)
        thr = choose_threshold(D.y[va], p, D.ds[va])
        cfg = dict(img=self.a.img, face_px=self.a.face_px, n_frames=N_FRAMES, margin=FACE_MARGIN, mean=MEAN.tolist(), std=STD.tolist(),
                   backbone="efficientnet_b0", pretrained=bool(model.pretrained))
        atomic_save(dict(model=model.state_dict(), temperature=T, threshold=thr, best_epoch=best["epoch"], val_score=best["score"],
                         val_ece_raw=ece(D.y[va], sigmoid(lg)), val_ece=ece(D.y[va], p), history=hist, config=cfg, seed=self.seed,
                         script=SCRIPT_PATH.name, saved_at=now(), final=True), self.ck_best)
        tmp.unlink(missing_ok=True)
        print(f"  seed {self.seed}: best epoch {best['epoch']} | val macro-AUC {best['score']:.4f} | temperature {T:.2f} | threshold {thr:.2f} "
              f"| {fmt_time(time.time() - t_run)}")
        return self.ck_best


def load_model(path: Path, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    obj = torch.load(path, map_location=device, weights_only=False)
    m = DeepfakeDetector(pretrained=False).to(device)
    m.load_state_dict(obj["model"])
    m.eval()
    return m, obj


def phase_train(a, P: Paths, D: Data):
    print("\n" + "=" * 90 + f"\n PHASE TRAIN: seeds {a.seed_list} (the best validation score becomes best_model.pt)\n" + "=" * 90)
    res = []
    for s in a.seed_list:
        ck = Trainer(a, P, D, s).train()
        obj = torch.load(ck, map_location="cpu", weights_only=False)
        res.append((obj["val_score"], s, ck))
    res.sort(reverse=True)
    score, seed, ck = res[0]
    shutil.copyfile(ck, P.best)
    write_json(P.reports / "model_selection.json", dict(chosen_seed=seed, val_macro_auc=score,
                                                        candidates=[dict(seed=s, val_macro_auc=v) for v, s, _ in res], selected_at=now()))
    print(f"  best model: seed {seed} (val macro-AUC {score:.4f}) -> {P.best}")


# =====================================================================================
# EXPLAINABILITY
# =====================================================================================
def gradcam_pp(model, x: torch.Tensor, t_idx: torch.Tensor, target: int = 1):
    """Grad-CAM++ for the video logit (target 1 = evidence for FAKE, 0 = evidence for REAL).
    Returns cam [B,T,H,W] in [0,1] (normalised per video), probability-free logit [B], attention [B,T]."""
    store = {}

    def hook(_m, _i, o):
        store["a"] = o
        o.register_hook(lambda g: store.__setitem__("g", g))

    h = model.cam_layer.register_forward_hook(hook)
    try:
        model.eval()
        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            o = model(x, t_idx)
            sign = 1.0 if target == 1 else -1.0
            (sign * o["logit"]).sum().backward()
    finally:
        h.remove()
    A, G = store["a"].detach().float(), store["g"].detach().float()        # [B*T,C,h,w]
    g2, g3 = G ** 2, G ** 3
    alpha = g2 / (2 * g2 + (A.sum((2, 3), keepdim=True) * g3) + 1e-8)
    wts = (alpha * F.relu(G)).sum((2, 3), keepdim=True)
    cam = F.relu((wts * A).sum(1))
    B, T = x.shape[:2]
    cam = F.interpolate(cam[:, None], size=x.shape[-2:], mode="bilinear", align_corners=False)[:, 0].view(B, T, *x.shape[-2:])
    cam = cam / (cam.flatten(1).amax(1).view(B, 1, 1, 1) + 1e-8)
    model.zero_grad(set_to_none=True)
    return cam.cpu().numpy(), o["logit"].detach().float().cpu().numpy(), o["attn"].detach().float().cpu().numpy()


def occlusion_map(model, clip_u8: np.ndarray, device, patch: int = 32, T_temp: float = 1.0) -> np.ndarray:
    """Model-agnostic sensitivity: drop in P(fake) when a patch (same place in all frames) is grey-filled. [H,W]."""
    S = clip_u8.shape[1]
    with torch.no_grad():
        base = sigmoid(model(normalise(clip_u8[None], device))["logit"].float().cpu().numpy() / T_temp)[0]
        grid = list(range(0, S - patch + 1, patch))
        heat = np.zeros((S, S), np.float32)
        cnt = np.zeros((S, S), np.float32)
        batch, locs = [], []
        for y in grid:
            for x in grid:
                c = clip_u8.copy()
                c[:, y:y + patch, x:x + patch] = 128
                batch.append(c)
                locs.append((y, x))
                if len(batch) == 6 or (y, x) == (grid[-1], grid[-1]):
                    p = sigmoid(model(normalise(np.stack(batch), device))["logit"].float().cpu().numpy() / T_temp)
                    for (yy, xx), pp in zip(locs, p):
                        heat[yy:yy + patch, xx:xx + patch] += base - pp
                        cnt[yy:yy + patch, xx:xx + patch] += 1
                    batch, locs = [], []
    return heat / np.maximum(cnt, 1), float(base)


def overlay(img_rgb: np.ndarray, cam: np.ndarray, alpha=0.45) -> np.ndarray:
    hm = cv2.applyColorMap(np.uint8(255 * np.clip(cam, 0, 1)), cv2.COLORMAP_JET)[..., ::-1]
    return np.uint8(np.clip((1 - alpha) * img_rgb + alpha * hm, 0, 255))


def blur_mask(clip_u8: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Replace masked pixels (mask [T,H,W] bool) by a strongly blurred version (removes local evidence)."""
    out = clip_u8.copy()
    for t in range(len(clip_u8)):
        b = cv2.GaussianBlur(clip_u8[t], (0, 0), 8)
        out[t][mask[t]] = b[mask[t]]
    return out


def save_xai_figure(path: Path, crops: np.ndarray, cam: np.ndarray, attn: np.ndarray, p: float, verdict: str, title: str,
                    occ: Optional[np.ndarray] = None):
    T = len(crops)
    show = np.linspace(0, T - 1, 8).round().astype(int)
    rows = 3 if occ is None else 3
    fig = plt.figure(figsize=(16, 6.4))
    gs = fig.add_gridspec(rows, 8, height_ratios=[1, 1, 0.55])
    for k, t in enumerate(show):
        ax = fig.add_subplot(gs[0, k])
        ax.imshow(crops[t])
        ax.set_title(f"frame {t}", fontsize=8)
        ax.axis("off")
        ax = fig.add_subplot(gs[1, k])
        ax.imshow(overlay(crops[t], cam[t]))
        ax.set_title(f"Grad-CAM++ (att {attn[t]:.2f})", fontsize=7)
        ax.axis("off")
    ax = fig.add_subplot(gs[2, :5] if occ is not None else gs[2, :])
    ax.bar(range(T), attn, color=["#c0392b" if t in set(show[np.argsort(-attn[show])[:2]]) else "#2a78d6" for t in range(T)])
    ax.set_xlabel("frame")
    ax.set_ylabel("temporal attention")
    ax.set_xticks(range(T))
    if occ is not None:
        ax2 = fig.add_subplot(gs[2, 6:])
        mid = crops[T // 2]
        o = occ / (np.abs(occ).max() + 1e-8)
        ax2.imshow(overlay(mid, np.clip(o, 0, 1)))
        ax2.set_title("occlusion: where hiding the face lowers P(fake)", fontsize=7)
        ax2.axis("off")
    col = "#c0392b" if verdict.startswith("FAKE") else ("#1e8449" if verdict.startswith("REAL") else "#b7950b")
    fig.suptitle(f"{title}\n{verdict}   P(fake) = {p:.3f}", fontsize=12, color=col)
    plt.tight_layout()
    plt.savefig(path, dpi=110)
    plt.close(fig)


# =====================================================================================
# EVALUATION
# =====================================================================================
def phase_evaluate(a, P: Paths, D: Data):
    print("\n" + "=" * 90 + "\n PHASE EVALUATE: held-out test videos of both datasets\n" + "=" * 90)
    tr = Trainer(a, P, D, 0)
    model, obj = load_model(P.best, tr.dev)
    T, thr = obj["temperature"], obj["threshold"]
    rows = []
    for split in ("val", "test"):
        idx = D.idx(split)
        lg, att = tr.predict(model, idx)
        for i, l, w in zip(idx, lg, att):
            r = D.sp.iloc[i]
            rows.append(dict(split=split, idx=int(i), path=r.path, dataset=r.dataset, subset=r.subset, label=int(r.label), group=r.group,
                             logit=float(l), p_fake=float(sigmoid(l / T)), top_frame=int(np.argmax(w))))
    pred = pd.DataFrame(rows)
    pred["pred"] = (pred.p_fake >= thr).astype(int)
    pred["verdict"] = np.where(np.abs(pred.p_fake - thr) < UNCERTAIN_BAND, "uncertain", np.where(pred.pred == 1, "FAKE", "REAL"))
    pred.to_csv(P.metrics / "predictions.csv", index=False)
    te = pred[pred.split == "test"]
    met = []
    for name, g in [("ALL (both datasets)", te)] + [(d, te[te.dataset == d]) for d in DATASETS if (te.dataset == d).any()]:
        m = compute_metrics(g.label, g.p_fake, thr)
        m05 = compute_metrics(g.label, g.p_fake, 0.5)
        met.append(dict(test_set=name, **{k: m[k] for k in ["n", "n_real", "n_fake", "auc", "accuracy", "balanced_accuracy", "precision",
                                                            "recall", "specificity", "f1", "tn", "fp", "fn", "tp"]},
                        accuracy_at_0_5=m05["accuracy"], ece=ece(g.label, g.p_fake), uncertain_pct=100 * (g.verdict == "uncertain").mean()))
    met = pd.DataFrame(met)
    met.to_csv(P.metrics / "test_metrics.csv", index=False)
    sub = te.groupby(["dataset", "subset"]).apply(lambda g: pd.Series(dict(
        n=len(g), true_label="FAKE" if g.label.iloc[0] else "REAL", correct_pct=100 * (g.pred == g.label).mean(),
        mean_p_fake=g.p_fake.mean()))).reset_index()
    sub.to_csv(P.metrics / "test_per_subset.csv", index=False)
    print(md_table(met))
    print(md_table(sub))
    # figures: ROC + histograms + confusion matrices
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    for d, c in zip(DATASETS, ["#2a78d6", "#eb6834"]):
        g = te[te.dataset == d]
        if g.label.nunique() > 1:
            f, t_, _ = roc_curve(g.label, g.p_fake)
            ax[0].plot(f, t_, color=c, label=f"{d} AUC {roc_auc_score(g.label, g.p_fake):.3f}")
    ax[0].plot([0, 1], [0, 1], "k:", lw=0.8)
    ax[0].set_xlabel("false positive rate")
    ax[0].set_ylabel("true positive rate")
    ax[0].legend()
    ax[0].set_title("ROC (test)")
    for k, d in enumerate(DATASETS, 1):
        g = te[te.dataset == d]
        ax[k].hist(g[g.label == 0].p_fake, bins=30, range=(0, 1), alpha=0.7, color="#1e8449", label="real")
        ax[k].hist(g[g.label == 1].p_fake, bins=30, range=(0, 1), alpha=0.7, color="#c0392b", label="fake")
        ax[k].axvline(thr, color="k", ls="--", lw=1)
        ax[k].set_title(f"{d}: P(fake) on test videos")
        ax[k].legend()
    plt.tight_layout()
    plt.savefig(P.figures / "roc_and_scores.png", dpi=130)
    plt.close(fig)
    fig, ax = plt.subplots(1, 2, figsize=(8, 3.6))
    for k, d in enumerate(DATASETS):
        g = te[te.dataset == d]
        cm = confusion_matrix(g.label, g.pred, labels=[0, 1])
        ax[k].imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):
            ax[k].text(j, i, str(v), ha="center", va="center", fontsize=11)
        ax[k].set_xticks([0, 1])
        ax[k].set_xticklabels(["pred REAL", "pred FAKE"])
        ax[k].set_yticks([0, 1])
        ax[k].set_yticklabels(["REAL", "FAKE"])
        ax[k].set_title(d)
    plt.tight_layout()
    plt.savefig(P.figures / "confusion_matrices.png", dpi=130)
    plt.close(fig)
    return pred, met, sub


def phase_xai(a, P: Paths, D: Data):
    """Faithfulness test of the explanations + example figures (correct and wrong decisions)."""
    print("\n" + "=" * 90 + "\n PHASE XAI: Grad-CAM++ faithfulness test + example explanations\n" + "=" * 90)
    pred = pd.read_csv(P.metrics / "predictions.csv")
    te = pred[pred.split == "test"].reset_index(drop=True)
    tr = Trainer(a, P, D, 0)
    model, obj = load_model(P.best, tr.dev)
    T, thr = obj["temperature"], obj["threshold"]
    rng = np.random.default_rng(SPLIT_SEED)
    pick = []
    for (d, lab), g in te.groupby(["dataset", "label"]):
        pick += list(rng.choice(g.index.values, size=min(len(g), max(1, a.faith_n // 4)), replace=False))
    rows = []
    for k in range(0, len(pick), 4):
        ch = te.loc[pick[k:k + 4]]
        clips = np.stack([np.asarray(D.C.arr[i]) for i in ch.idx])
        tgt = (ch.pred.values == 1).astype(int)
        cams = []
        for b in range(len(ch)):
            c, _, _ = gradcam_pp(model, normalise(clips[b:b + 1], tr.dev), None, int(tgt[b]))
            cams.append(c[0])
        cams = np.stack(cams)
        for b, (_, r) in enumerate(ch.iterrows()):
            cm = cams[b]
            q = np.quantile(cm.reshape(len(cm), -1), 1 - XAI_TOP, axis=1)[:, None, None]
            top = cm >= q
            rnd = rng.random(cm.shape) < XAI_TOP
            c_top, c_rnd = blur_mask(clips[b], top), blur_mask(clips[b], rnd)
            with torch.no_grad():
                p0 = sigmoid(model(normalise(clips[b:b + 1], tr.dev))["logit"].float().cpu().numpy() / T)[0]
                pt = sigmoid(model(normalise(c_top[None], tr.dev))["logit"].float().cpu().numpy() / T)[0]
                pr_ = sigmoid(model(normalise(c_rnd[None], tr.dev))["logit"].float().cpu().numpy() / T)[0]
            sgn = 1 if tgt[b] == 1 else -1                      # drop in confidence of the predicted class
            S = cm.shape[-1]
            yy, xx = np.mgrid[0:S, 0:S]
            centre = ((yy - S / 2) ** 2 + (xx - S / 2) ** 2) <= (0.35 * S) ** 2
            rows.append(dict(idx=int(r.idx), dataset=r.dataset, subset=r.subset, label=int(r.label), pred=int(r.pred),
                             correct=bool(r.pred == r.label), p_fake=p0, drop_top=sgn * (p0 - pt), drop_random=sgn * (p0 - pr_),
                             cam_in_face_centre=float(cm[:, centre].sum() / (cm.sum() + 1e-8))))
        sys.stdout.write(f"\r  {bar(min(k + 4, len(pick)), len(pick))} faithfulness test   ")
        sys.stdout.flush()
    sys.stdout.write("\n")
    ft = pd.DataFrame(rows)
    ft["cam_beats_random"] = ft.drop_top > ft.drop_random
    ft.to_csv(P.metrics / "xai_faithfulness.csv", index=False)
    summ = ft.groupby("dataset").agg(n=("idx", "size"), mean_drop_top20_cam=("drop_top", "mean"), mean_drop_random20=("drop_random", "mean"),
                                     cam_beats_random_pct=("cam_beats_random", lambda s: 100 * s.mean()),
                                     cam_energy_in_face_centre=("cam_in_face_centre", "mean")).reset_index()
    summ.to_csv(P.metrics / "xai_faithfulness_summary.csv", index=False)
    print(md_table(summ))
    # example explanations: 3 correct fakes, 3 correct reals, 3 most confident mistakes per dataset
    ex = []
    for d in DATASETS:
        g = te[te.dataset == d]
        ex += list(g[(g.label == 1) & (g.pred == 1)].nlargest(3, "p_fake").idx)
        ex += list(g[(g.label == 0) & (g.pred == 0)].nsmallest(3, "p_fake").idx)
        wrong = g[g.pred != g.label].assign(conf=lambda x: (x.p_fake - thr).abs())
        ex += list(wrong.nlargest(3, "conf").idx)
    for i in ex:
        r = te[te.idx == i].iloc[0]
        clip = np.asarray(D.C.arr[i])
        cam, lg, att = gradcam_pp(model, normalise(clip[None], tr.dev), None, int(r.pred))
        tag = "correct" if r.pred == r.label else "WRONG"
        save_xai_figure(P.xai / f"{tag}_{r.dataset}_{r.subset}_{Path(r.path).stem[:40]}.png", clip, cam[0], att[0], float(r.p_fake),
                        f"{'FAKE' if r.pred else 'REAL'} (true: {'FAKE' if r.label else 'REAL'}, {tag})", f"{r.dataset}/{r.subset}: {Path(r.path).name}")
    print(f"  {len(ex)} example explanations -> {P.xai}")
    return summ


def phase_report(a, P: Paths):
    met = pd.read_csv(P.metrics / "test_metrics.csv")
    sub = pd.read_csv(P.metrics / "test_per_subset.csv")
    fx = pd.read_csv(P.metrics / "xai_faithfulness_summary.csv") if (P.metrics / "xai_faithfulness_summary.csv").exists() else pd.DataFrame()
    obj = torch.load(P.best, map_location="cpu", weights_only=False)
    sel = read_json(P.reports / "model_selection.json")
    ss = read_json(P.splits / "split_summary.json")
    hist = pd.DataFrame(obj["history"])
    L = ["# 151026: Real / fake video detector with explainable AI\n",
         f"_Generated {now()} | BioDeepAV + FaceForensics++ C23 originals only (no degraded videos) | script {SCRIPT_PATH.name}_\n",
         "## Model\n",
         f"- EfficientNet-B0 ({'ImageNet-pretrained' if obj['config'].get('pretrained') else 'random init'}) per face frame -> temporal "
         f"transformer -> attention pooling; {N_FRAMES} frames, face crops {obj['config']['img']} px with a {obj['config']['face_px']} px "
         "effective-resolution bottleneck (removes the resolution shortcut).",
         f"- Chosen seed {sel['chosen_seed']} (validation macro-AUC {sel['val_macro_auc']:.4f}); best epoch {obj['best_epoch']}; "
         f"temperature {obj['temperature']:.2f}; decision threshold {obj['threshold']:.2f}; ECE on validation {obj['val_ece_raw']:.3f} -> {obj['val_ece']:.3f}.",
         f"- Split: {ss['groups']:,} identity/duplicate groups, groups crossing splits {ss['groups_crossing_splits']} (0 = no leakage); "
         f"exact duplicates merged {ss['exact_duplicates_merged']}, near-duplicate pairs merged {ss['near_duplicate_pairs']}.\n",
         "## Test results (held-out videos, never seen in training or model selection)\n", md_table(met), "",
         "![ROC](../figures/roc_and_scores.png)\n", "![Confusion](../figures/confusion_matrices.png)\n",
         "## Per subset (FF++ manipulation methods, real folders)\n", md_table(sub), "",
         "## Training history\n", md_table(hist), "",
         "## Explainability\n",
         "- Grad-CAM++ heat-maps (last EfficientNet block), temporal attention per frame and, in prediction mode, occlusion sensitivity.",
         "- Faithfulness test: blur the top-20 % Grad-CAM pixels vs a random 20 % of pixels; a faithful explanation lowers the confidence "
         "of the predicted class more (cam_beats_random_pct well above 50 %).\n", md_table(fx), "",
         f"Example explanations (correct and wrong decisions): `{P.xai}`\n",
         "## Using the detector on any video\n",
         "```\npython 151026.py --predict \"path\\to\\video.mp4\"\npython 151026.py --predict \"path\\to\\folder\"\npython 151026.py --phase gui\n```",
         f"Verdicts within ±{UNCERTAIN_BAND:.2f} of the threshold are reported as **uncertain**. The detector is trained on face-swap / "
         "re-enactment deepfakes of these two datasets; for other generators (e.g. fully synthetic diffusion videos) or videos without a "
         "visible face its output is less reliable. Heat-maps show where the evidence is, not proof of manipulation."]
    (P.reports / "REPORT_151026.md").write_text("\n".join(L), encoding="utf-8")
    print(f"  report -> {P.reports / 'REPORT_151026.md'}")


# =====================================================================================
# PREDICTION ON ANY VIDEO
# =====================================================================================
_DET = None


def predict_video(path: str, P: Paths, model=None, obj=None, device=None, occlusion: bool = True, save: bool = True) -> dict:
    global _DET
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model is None:
        if not P.best.exists():
            raise SystemExit(f"[STOP] no trained model at {P.best} - run the training first")
        model, obj = load_model(P.best, device)
    cfg = obj["config"]
    if _DET is None:
        _DET = FaceDetector(ensure_face_model())
    crops, info = extract_faces(path, _DET, cfg["n_frames"], cfg["img"], cfg["face_px"], cfg["margin"], keep_frames=True)
    if crops is None:
        return dict(video=str(path), verdict="ERROR", error=info.get("error", "cannot read video"))
    T, thr = obj["temperature"], obj["threshold"]
    with torch.no_grad():
        lg = model(normalise(crops[None], device))["logit"].float().cpu().numpy()[0]
    p = float(sigmoid(lg / T))
    pred = int(p >= thr)
    verdict = "UNCERTAIN" if abs(p - thr) < UNCERTAIN_BAND else ("FAKE" if pred else "REAL")
    conf = p if pred else 1 - p
    notes = []
    if info["fallback"]:
        notes.append("no face detected - a centre crop was analysed; result unreliable")
    elif info["detection_rate"] < 0.5:
        notes.append(f"face found in only {100 * info['detection_rate']:.0f}% of frames")
    res = dict(video=str(path), verdict=verdict, predicted=("FAKE" if pred else "REAL"), p_fake=round(p, 4), confidence=round(conf, 4),
               threshold=thr, face_detection_rate=info["detection_rate"], notes=notes, analysed_at=now())
    if save:
        out = P.predictions / f"{Path(path).stem[:60]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        out.mkdir(parents=True, exist_ok=True)
        cam, _, att = gradcam_pp(model, normalise(crops[None], device), None, pred)
        occ = None
        if occlusion:
            occ, _ = occlusion_map(model, crops, device, patch=max(16, cfg["img"] // 6), T_temp=T)
        save_xai_figure(out / "explanation.png", crops, cam[0], att[0], p, f"{verdict}  (confidence {conf:.1%})", Path(path).name, occ)
        # heat-map projected back onto the original frames
        frames, boxes = info["frames"], info["boxes"]
        h, w = frames[0].shape[:2]
        vw = cv2.VideoWriter(str(out / "explanation_overlay.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 4, (w, h))
        for t, f in enumerate(frames):
            x0, y0, s = [int(round(v)) for v in boxes[t]]
            s = max(8, s)
            hm = cv2.resize(cam[0][t], (s, s))
            sub = f[y0:y0 + s, x0:x0 + s]
            hm = hm[:sub.shape[0], :sub.shape[1]]
            col = cv2.applyColorMap(np.uint8(255 * hm), cv2.COLORMAP_JET)
            f2 = f.copy()
            f2[y0:y0 + sub.shape[0], x0:x0 + sub.shape[1]] = np.uint8(0.55 * sub + 0.45 * col)
            c = (0, 0, 255) if verdict == "FAKE" else ((0, 160, 0) if verdict == "REAL" else (0, 200, 255))
            cv2.rectangle(f2, (x0, y0), (x0 + s, y0 + s), c, 2)
            cv2.putText(f2, f"{verdict} P(fake)={p:.2f}  frame {t} att {att[0][t]:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, c, 2)
            vw.write(f2)
        vw.release()
        res.update(top_frames=[int(i) for i in np.argsort(-att[0])[:3]], output_folder=str(out))
        write_json(out / "verdict.json", res)
    return res


def phase_predict(a, P: Paths):
    tgt = Path(a.predict)
    vids = sorted(p for p in tgt.rglob("*") if p.suffix.lower() in VIDEO_EXT) if tgt.is_dir() else [tgt]
    if not vids:
        raise SystemExit(f"[STOP] no video found at {tgt}")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, obj = load_model(P.best, dev)
    print(f"  model: {P.best} | threshold {obj['threshold']:.2f} | {len(vids)} video(s)")
    rows = []
    for v in vids:
        r = predict_video(str(v), P, model, obj, dev, occlusion=not a.no_occlusion and len(vids) <= 20)
        rows.append(r)
        print(f"\n  {Path(v).name}\n    ==> {r['verdict']}" + (f"  (P(fake) = {r['p_fake']:.3f}, confidence {r['confidence']:.1%})" if 'p_fake' in r else
                                                             f"  {r.get('error', '')}"))
        for n in r.get("notes", []):
            print(f"    note: {n}")
        if r.get("output_folder"):
            print(f"    explanation: {r['output_folder']}")
    if len(rows) > 1:
        out = P.predictions / f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        pd.DataFrame(rows).drop(columns=["notes"], errors="ignore").to_csv(out, index=False)
        print(f"\n  summary CSV: {out}")


def phase_gui(a, P: Paths):
    import tkinter as tk
    from tkinter import filedialog, ttk
    from PIL import Image, ImageTk
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, obj = load_model(P.best, dev)
    root = tk.Tk()
    root.title("Deepfake detector (151026) - REAL or FAKE?")
    top = ttk.Frame(root, padding=8)
    top.pack(fill="x")
    lab = ttk.Label(top, text="Choose a video", font=("Segoe UI", 16, "bold"))
    lab.pack(side="left", padx=8)
    img_lab = ttk.Label(root)
    img_lab.pack(fill="both", expand=True)

    def run():
        f = filedialog.askopenfilename(filetypes=[("Videos", " ".join(f"*{e}" for e in sorted(VIDEO_EXT))), ("All", "*.*")])
        if not f:
            return
        lab.config(text="analysing ...")
        root.update()
        r = predict_video(f, P, model, obj, dev, occlusion=True)
        if r["verdict"] == "ERROR":
            lab.config(text=f"ERROR: {r.get('error')}", foreground="#b7950b")
            return
        col = {"FAKE": "#c0392b", "REAL": "#1e8449"}.get(r["verdict"], "#b7950b")
        lab.config(text=f"{r['verdict']}   P(fake) = {r['p_fake']:.3f}   ({Path(f).name})" + ("   " + r["notes"][0] if r["notes"] else ""),
                   foreground=col)
        im = Image.open(Path(r["output_folder"]) / "explanation.png")
        im.thumbnail((1400, 700))
        tkim = ImageTk.PhotoImage(im)
        img_lab.config(image=tkim)
        img_lab.image = tkim

    ttk.Button(top, text="Open video ...", command=run).pack(side="right")
    root.geometry("1450x820")
    root.mainloop()


# =====================================================================================
# SELFTEST
# =====================================================================================
def phase_selftest(a, P: Paths) -> bool:
    print("\n" + "=" * 90 + "\n PHASE SELFTEST\n" + "=" * 90)
    res = {}
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = DeepfakeDetector(pretrained=False).to(dev)
    x = np.random.default_rng(0).integers(0, 255, (2, CLIP_FRAMES, 96, 96, 3), dtype=np.uint8)
    o = m(normalise(x, dev), torch.arange(CLIP_FRAMES, device=dev).expand(2, -1))
    res["model output shapes"] = o["logit"].shape == (2,) and o["frame_logit"].shape == (2, CLIP_FRAMES) and o["attn"].shape == (2, CLIP_FRAMES)
    res["attention sums to 1"] = torch.allclose(o["attn"].sum(1), torch.ones(2, device=dev), atol=1e-4)
    cam, _, _ = gradcam_pp(m, normalise(x, dev), None, 1)
    res["Grad-CAM++ shape and range"] = cam.shape == (2, CLIP_FRAMES, 96, 96) and cam.min() >= 0 and cam.max() <= 1 + 1e-6
    rng = np.random.default_rng(1)
    au = augment_clip(x[0], rng)
    res["augmentation keeps shape/dtype"] = au.shape == x[0].shape and au.dtype == np.uint8
    img = np.full((200, 200, 3), 120, np.uint8)
    res["face detector runs (no face -> None)"] = FaceDetector(None, quiet=True).detect(img) is None
    real_det = FaceDetector(FACE_MODEL if FACE_MODEL.exists() else None, quiet=True)
    print(f"  face detector for the real data: {real_det.backend}" + (f" ({FACE_MODEL})" if real_det.backend == "yunet" else
          f"  <-- YuNet model not found at {FACE_MODEL}; it will be downloaded at the faces phase"))
    if real_det.backend == "yunet":
        res["YuNet face detector loads"] = real_det.detect(img) is None
    b = square_box(np.array([150, 150, 40, 40], np.float32), 200, 200, 0.3)
    res["square face box inside frame"] = b[0] >= 0 and b[1] >= 0 and b[0] + b[2] <= 200 + 1e-3 and b[1] + b[2] <= 200 + 1e-3
    inv = pd.DataFrame(dict(dataset=["A"] * 6 + ["B"] * 6, label=[0, 1] * 6))
    groups = {i: [i] for i in range(12)}
    sp = make_split(groups, inv, 0, (0.5, 0.25, 0.25))
    res["split assigns every group"] = len(sp) == 12 and set(sp.values()) <= {"train", "val", "test"}
    H = rng.integers(0, 2 ** 63, (3, 4), dtype=np.int64).astype(np.uint64)
    H[1] = H[0] ^ np.uint64(1)                                 # 1 bit away from row 0 -> near duplicate
    res["near-duplicate screen"] = near_duplicates(H, np.array([True, True, True])) == [(0, 1)]
    yv = rng.integers(0, 2, 3000)
    z = (2 * yv - 1) + rng.normal(0, 1, 3000)
    res["temperature recovers x5 over-confidence"] = fit_temperature(5 * z, yv) > 2
    res["threshold in range"] = 0.05 <= choose_threshold(yv, sigmoid(z), np.array(["A", "B"] * 1500)) <= 0.95
    res["identity tokens FF++ pair"] = identity_tokens("FF++", "Deepfakes", 1, "000_003") == ["ff:000", "ff:003"]
    for k, v in res.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    ok = all(res.values())
    print(f"  selftest {'PASSED' if ok else 'FAILED'} | torch {torch.__version__} | torchvision "
          f"{getattr(torchvision, '__version__', 'missing')} | CUDA {torch.cuda.is_available()}")
    return ok


# =====================================================================================
# MAIN
# =====================================================================================
PHASES = ["run", "selftest", "inventory", "faces", "split", "train", "evaluate", "xai", "report", "predict", "gui"]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="151026: real/fake video detector with XAI (BioDeepAV + FF++ originals)")
    p.add_argument("--phase", default="run", choices=PHASES)
    p.add_argument("--predict", default=None, help="video file or folder to classify (implies --phase predict)")
    p.add_argument("--root", default=None)
    p.add_argument("--seeds", default=SEEDS, help="train one model per seed, keep the best on validation")
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--batch", type=int, default=BATCH_CLIPS)
    p.add_argument("--accum", type=int, default=ACCUM)
    p.add_argument("--lr-backbone", type=float, default=LR_BACKBONE)
    p.add_argument("--lr-head", type=float, default=LR_HEAD)
    p.add_argument("--label-smooth", type=float, default=LABEL_SMOOTH)
    p.add_argument("--ema-decay", type=float, default=EMA_DECAY)
    p.add_argument("--img", type=int, default=IMG)
    p.add_argument("--face-px", type=int, default=FACE_PX)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--io-workers", type=int, default=4)
    p.add_argument("--loader-threads", type=int, default=4)
    p.add_argument("--max-per-folder", type=int, default=None)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--faith-n", type=int, default=FAITH_N)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--no-occlusion", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--source-root", default=None, help="(testing) replace E:\\Datasets by another root with the same sub-folders")
    a = p.parse_args(argv)
    if a.predict:
        a.phase = "predict"
    if a.smoke:
        a.max_per_folder = a.max_per_folder or 40
        a.epochs = min(a.epochs, 2)
        a.max_steps = a.max_steps or 6
        a.faith_n = min(a.faith_n, 16)
    a.root = Path(a.root) if a.root else (SMOKE_ROOT if a.smoke else DEFAULT_ROOT)
    a.workers = a.workers or max(1, min(8, (os.cpu_count() or 4) // 2))
    a.seed_list = [int(s) for s in str(a.seeds).split(",") if s.strip()]
    a.sources = [(d, s, l, f.replace("E:\\Datasets", a.source_root).replace("\\", os.sep) if a.source_root else f)
                 for d, s, l, f in SOURCES]
    return a


def phase_run(a, P: Paths):
    if not phase_selftest(a, P):
        raise SystemExit("[STOP] selftest failed")
    inv = pd.read_csv(P.inventory) if P.inventory.exists() else phase_inventory(a, P)
    C = phase_faces(a, P, inv)
    sp = phase_split(a, P, inv, C)
    D = Data(sp, C)
    phase_train(a, P, D)
    phase_evaluate(a, P, D)
    phase_xai(a, P, D)
    phase_report(a, P)
    print(f"\n  DONE. Report: {P.reports / 'REPORT_151026.md'}\n  Test any video:  python {SCRIPT_PATH.name} --predict \"path\\to\\video.mp4\"")


def main(argv=None):
    a = parse_args(argv)
    P = Paths(a.root)
    P.make()
    if a.phase not in ("predict", "gui"):
        lower_priority()
        sys.stdout = Tee(P.logs / f"{a.phase}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    torch.set_num_threads(max(1, min(4, (os.cpu_count() or 4) // 2)))
    print(f"151026 | phase {a.phase} | root {P.root} | {now()} | {platform.platform()} | torch {torch.__version__} | "
          f"GPU {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}")
    ph = a.phase
    if ph == "run":
        return phase_run(a, P)
    if ph == "selftest":
        sys.exit(0 if phase_selftest(a, P) else 1)
    if ph == "predict":
        return phase_predict(a, P)
    if ph == "gui":
        return phase_gui(a, P)
    inv = pd.read_csv(P.inventory) if (P.inventory.exists() and ph != "inventory") else phase_inventory(a, P)
    if ph == "inventory":
        return
    C = phase_faces(a, P, inv)
    if ph == "faces":
        return
    sp = phase_split(a, P, inv, C)
    D = Data(sp, C)
    if ph == "train":
        phase_train(a, P, D)
    elif ph == "evaluate":
        phase_evaluate(a, P, D)
    elif ph == "xai":
        phase_xai(a, P, D)
    elif ph == "report":
        phase_report(a, P)


if __name__ == "__main__":
    main()
