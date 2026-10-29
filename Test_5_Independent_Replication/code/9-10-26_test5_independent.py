"""
9-10-26 (TEST 5, INDEPENDENT): HARD-DEGRADATION-AWARE TRAINING AND GRAD-CAM EXPLANATION STABILITY
=================================================================================================

Thesis: Robust Deepfake Detection and Explainability Under Real-World Video Degradation

Research question
  Does hard-degradation-aware training (brightness, saturation, contrast, blur) improve detection robustness and
  Grad-CAM explanation stability of the AudioVisualDeepFakeCNN, and does explanation/feature-consistency training
  add to that improvement?

Independence (this file is a standalone experiment; it does NOT import 6-10-26.py or 8-10-26.py)
  * Fresh source inventory (full-file SHA-256, ffprobe metadata, manifest link checks) of the local BioDeepAV and
    FaceForensics++ C23 data, fresh identity grouping, fresh split (own seed), fresh face crops / audio caches in
    run_test5_independent/cache, fresh model initialisation, training, evaluation, Grad-CAM and statistics.
  * Test 4 (run_6-10-26, 6-10-26.py) is only READ for documentation (historical reported numbers, architecture
    signature check in --phase selftest). Every artifact load goes through guard(); loading anything from a Test 4 /
    old Test 5 run directory as an input raises ForbiddenArtifactError and is audited by the leakage gate.
  * The preprocessing and architecture code below is re-implemented from the documented Test 4 recipe (YuNet face
    crop, resolution normalisation 360 px / 80 px bottleneck, 16 cached frames, 8-frame clips, 128 px, log-STFT
    audio spectrogram). Its source hash is recorded in every cache/config/checkpoint.

Phases (run in order; every phase is resumable)
  python 9-10-26_test5_independent.py --phase selftest     # unit tests, no data, no training
  python 9-10-26_test5_independent.py --phase prepare      # audit -> clean preprocessing -> grouping/split ->
                                                            # degraded preprocessing -> LEAKAGE GATE, then STOP
  python 9-10-26_test5_independent.py --phase status       # what is done
  python 9-10-26_test5_independent.py --phase all          # train -> evaluate -> xai -> analyze -> report
                                                            # (refuses to start unless the gate PASSED)
  Individual phases: audit, preprocess, split, degraded, gate, train, evaluate, xai, analyze, report.
  Smoke test of the whole pipeline on a tiny subset (separate root): add --smoke

Variants (Stage 2; Stage 1 = clean-only training, shared by 5A-5D within a (training set, seed))
  5A  independent baseline: unpaired 50 % original / 50 % degraded clips, natural degradation mixture, CE only
  5B  5A + hard-degradation sampling (brightness .30 / saturation .30 / contrast .20 / blur .20, share 0.5)
  5C  5B sampling + paired original/degraded training + Grad-CAM consistency (lambda_cam 0.05)
  5D  5C + feature consistency (lambda_feat 0.10) + degradation curriculum (1-3 mild, 4-6 +medium, 7-10 all)
  Unavoidable difference: Grad-CAM consistency needs original/degraded pairs, so 5C also introduces pairing.
  lambda values are a starting hypothesis (not tuned; never selected on test data).
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import io
import json
import math
import os
import platform
import random
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
import warnings
from collections import Counter, defaultdict, deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import cv2
from scipy.io import wavfile
from scipy.signal import butter, sosfiltfilt, resample_poly
from scipy import sparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import roc_auc_score, confusion_matrix  # noqa: E402

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
DEFAULT_ROOT = SCRIPT_DIR / "run_test5_independent"
SMOKE_ROOT = SCRIPT_DIR / "run_test5_independent_smoke"
TEST4_DIR = SCRIPT_DIR / "run_6-10-26"
TEST4_SCRIPT = SCRIPT_DIR / "6-10-26.py"
# directories whose artifacts (caches, checkpoints, splits, predictions, CAMs) must never be Test 5 inputs
FORBIDDEN_ARTIFACT_ROOTS = [SCRIPT_DIR / d for d in ("run_6-10-26", "run_test5_hard_explanation", "run_1-10-26", "run_30-9-26",
                                                     "run_1-10-26_smoke", "run_6-10-26_smoke")]
FACE_MODEL = SCRIPT_DIR / "models" / "face_detection_yunet_2023mar.onnx"   # third-party YuNet weights (not a Test 4 artifact)
YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"

DATA_SEED = 20261009              # Test 5 split / degraded selection / evaluation-sample seed (Test 4 used 42)
TRAIN_SEEDS = "42,123,2026"
SPLIT_RATIOS = (0.70, 0.10, 0.20)  # train / val / test (originals; degraded versions inherit their source's split)
MIN_PER_CLASS_IN_SPLIT = 20        # split integrity: every dataset needs >= this many real and fake videos in val and test

IMG_SIZE, NUM_FRAMES, CLIP_LEN = 128, 16, 8
FACE_MARGIN, FACE_SCORE = 0.20, 0.6
NORM = dict(frame_side=360, face_px=80, jpeg=0)
AUDIO_SR, AUDIO_SAMPLES = 16000, 48000
AUDIO_PROBS = (0.50, 0.25, 0.25)  # mixed audio policy: clean / degraded / none (all variants, both stages)
AUDIO_DEGRADATIONS = ["noise", "volume", "lowpass", "resample", "dropout", "silence"]
AUDIO_DEG_LEVELS = {"noise": (20.0, 10.0, 0.0), "volume": (-10.0, -20.0, -30.0), "lowpass": (4000, 2000, 1000),
                    "resample": (11025, 8000, 4000), "dropout": (0.10, 0.25, 0.50), "silence": (1, 1, 1)}

BATCH_SIZE, LR1, LR2, EPOCHS1, EPOCHS2, WEIGHT_DECAY = 32, 1e-3, 3e-4, 8, 10, 1e-4
MAX_OVERSAMPLE = 4.0
LAMBDA_FEAT, LAMBDA_CAM = 0.10, 0.05
CURRICULUM = (3, 6)
HARD_SHARE = 0.50
HARD_WEIGHTS = "brightness:0.30,saturation:0.30,contrast:0.20,blur:0.20"
HARD_TYPES = ["brightness", "saturation", "contrast", "blur"]
DEG_PER_SOURCE = 3
THRESHOLD_GRID = (0.01, 0.99, 0.01)
EXPLANATION_CHANGE_IOU = 0.5
XAI_PER_TYPE = 40
N_BOOT = 1000
MIN_CLUSTERS = 10
MIN_PER_CLASS_AUC = 5             # per-degradation AUC needs >= 5 real and >= 5 fake items
# duplicate screens (dHash of face crops / spectrograms, 4 x 64 bit) + normalised cross-correlation confirmation
NEARDUP_STRONG_MEAN, NEARDUP_STRONG_MAX, NEARDUP_STRONG_NCC = 3.0, 6, 0.95
NEARDUP_CAND_MEAN, NEARDUP_CAND_NCC = 8.0, 0.90
NEARDUP_MAX_PAIRS = 2_000_000
DEG_LINK_DECOYS = 20

SEV_LEVEL = {"mild": 0, "medium": 1, "severe": 2}
BIO_FOLDERS = [("BioDeepAV", "real", 0, ("real", "videos")), ("BioDeepAV", "fake", 1, ("fake", "videos"))]
FFPP_FOLDERS = ["original", "Deepfakes", "Face2Face", "FaceShifter", "FaceSwap", "NeuralTextures", "DeepFakeDetection"]
NO_WINDOW = subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0
FFPROBE = shutil.which("ffprobe")
FFMPEG = shutil.which("ffmpeg")
METRIC_KEYS = ["auc", "accuracy", "balanced_accuracy", "precision", "recall", "sensitivity", "specificity", "f1"]

VARIANTS = {
    "5A": dict(hard=False, paired=False, lambda_feat=0.0, lambda_cam=0.0, curriculum=False,
               description="Independent baseline: Stage 2 on unpaired 50/50 original/degraded clips, natural degradation mixture, CE only"),
    "5B": dict(hard=True, paired=False, lambda_feat=0.0, lambda_cam=0.0, curriculum=False,
               description="5A + hard-degradation sampling (brightness, saturation, contrast, blur)"),
    "5C": dict(hard=True, paired=True, lambda_feat=0.0, lambda_cam=LAMBDA_CAM, curriculum=False,
               description="Hard-degradation sampling + paired original/degraded training + Grad-CAM consistency"),
    "5D": dict(hard=True, paired=True, lambda_feat=LAMBDA_FEAT, lambda_cam=LAMBDA_CAM, curriculum=True,
               description="5C + feature consistency + degradation curriculum"),
}
VARIANT_COLOR = {"S1": "#8a8984", "5A": "#2a78d6", "5B": "#eb6834", "5C": "#1baf7a", "5D": "#4a3aa7"}
COMPARISONS = [("5B", "5A"), ("5C", "5B"), ("5D", "5C"), ("5D", "5A")]


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


def progress_bar(cur, tot, n=20) -> str:
    f = min(1.0, max(0.0, cur / max(1, tot)))
    k = int(f * n)
    return f"[{('=' * k + ('>' if k < n else '')).ljust(n, '.')}] {100 * f:5.1f}%"


def imap_bounded(fn: Callable, items, workers: int, ahead: Optional[int] = None):
    """Yield (item, finished future) with at most `ahead` tasks in flight. Submitting everything at once (and keeping
    every future) holds every result in RAM until the loop ends - for video caches that is tens of GB."""
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
                nxt = next(it, _END)
                if nxt is not _END:
                    pending[ex.submit(fn, nxt)] = nxt
                yield x, fut


_END = object()


def lower_process_priority() -> None:
    """Keep the desktop responsive while all worker threads are busy."""
    try:
        if sys.platform == "win32":
            import ctypes
            ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x00004000)  # BELOW_NORMAL
        else:
            os.nice(5)
    except Exception:
        pass


def to_jsonable(o):
    if isinstance(o, dict):
        return {str(k): to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [to_jsonable(v) for v in o]
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if (np.isnan(o) or np.isinf(o)) else float(o)
    if isinstance(o, np.ndarray):
        return to_jsonable(o.tolist())
    return o


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(obj), f, indent=2)
    os.replace(tmp, path)


def atomic_torch_save(obj, path: Path) -> None:
    """torch.save that survives a power cut: write a temp file, fsync it, then atomically replace the target.
    The previous file stays intact until the new one is completely on disk."""
    path = Path(path)
    tmp = path.with_name(path.name + ".writing")
    with open(tmp, "wb") as f:
        torch.save(obj, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def sha256_file(path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def as_bool(s: pd.Series) -> pd.Series:
    return s.map(lambda v: str(v).strip().lower() in ("true", "1", "1.0", "yes"))


def md_table(df: pd.DataFrame, nd: int = 3, max_rows: int = 60) -> str:
    if df is None or not len(df):
        return "_(no rows)_"
    d = df.head(max_rows)

    def cell(v):
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            return "n/a"
        if isinstance(v, (float, np.floating)):
            return f"{v:.{nd}f}"
        return str(v).replace("|", "/")

    lines = ["| " + " | ".join(str(c) for c in d.columns) + " |", "|" + "---|" * len(d.columns)]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in d.itertuples(index=False)]
    if len(df) > max_rows:
        lines.append(f"\n_({len(df) - max_rows} more rows in the CSV)_")
    return "\n".join(lines)


def fmt(v, nd=3) -> str:
    return "n/a" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{nd}f}"


def code_sha256() -> str:
    return sha256_file(SCRIPT_PATH)


def environment_info() -> dict:
    return dict(python=sys.version.split()[0], platform=platform.platform(), pytorch=torch.__version__, cuda=torch.version.cuda,
                cudnn=torch.backends.cudnn.version() if torch.cuda.is_available() else None,
                gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, numpy=np.__version__, pandas=pd.__version__,
                opencv=cv2.__version__, ffprobe=FFPROBE, ffmpeg=FFMPEG, script=str(SCRIPT_PATH), script_sha256=code_sha256(),
                recorded_at=now())


class Tee(io.TextIOBase):
    """Mirror stdout into a phase log file."""

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


# =====================================================================================
# PATHS + ARTIFACT ACCESS GUARD
# =====================================================================================
class Paths:
    SUBDIRS = ("data_audit", "manifests", "splits", "cache", "configs", "checkpoints", "logs", "metrics", "figures", "reports")

    def __init__(self, root: Path):
        self.root = Path(root)
        for d in self.SUBDIRS:
            setattr(self, d, self.root / d)
        self.inventory = self.data_audit / "source_inventory.csv"
        self.clean_index = self.manifests / "clean_cache_index.csv"
        self.deg_index = self.manifests / "degraded_cache_index.csv"
        self.group_manifest = self.manifests / "source_group_manifest.csv"
        self.split_originals = self.splits / "split_originals.csv"
        self.split_manifest = self.splits / "split_manifest.csv"
        self.split_meta = self.splits / "split_meta.json"
        self.split_integrity = self.splits / "split_integrity_report.json"
        self.leakage = self.reports / "leakage_audit_report.json"
        self.gate = self.reports / "gate_status.json"
        self.provenance = self.cache / "provenance.json"
        self.access_log = self.logs / "file_access_log.csv"
        self.xai_pairs = self.manifests / "xai_pairs.csv"
        self.pred_dir = self.metrics / "predictions"
        self.cam_dir = self.metrics / "gradcam_runs"
        self.hist_dir = self.logs / "training"

    def make(self):
        for d in self.SUBDIRS:
            getattr(self, d).mkdir(parents=True, exist_ok=True)
        for d in (self.pred_dir, self.cam_dir, self.hist_dir):
            d.mkdir(parents=True, exist_ok=True)


class ForbiddenArtifactError(RuntimeError):
    pass


_ACCESS = dict(log=None, phase="")
_ACCESS_LOCK = threading.Lock()


def _is_under(p: Path, root: Path) -> bool:
    try:
        p.relative_to(root.resolve())
        return True
    except ValueError:
        return False


def guard(path, purpose: str) -> Path:
    """Every artifact read (cache, checkpoint, manifest, prediction, historical file) passes here. Artifacts from
    Test 4 / old Test 5 directories may only be read for 'historical_reference_readonly' documentation."""
    p = Path(path).resolve()
    forbidden = any(_is_under(p, r) for r in FORBIDDEN_ARTIFACT_ROOTS)
    historical = purpose.startswith("historical_reference_readonly")
    log = _ACCESS["log"]
    if log is not None:
        with _ACCESS_LOCK:
            new = not log.exists()
            with open(log, "a", encoding="utf-8") as f:
                if new:
                    f.write("time,phase,purpose,path,forbidden_root,historical_readonly,blocked\n")
                f.write(f"{now()},{_ACCESS['phase']},{purpose},\"{p}\",{forbidden},{historical},{forbidden and not historical}\n")
    if forbidden and not historical:
        raise ForbiddenArtifactError(f"Test 5 tried to load a non-Test-5 artifact as input: {p} ({purpose})")
    return p


def read_csv(path, purpose: str, **kw) -> pd.DataFrame:
    return pd.read_csv(guard(path, purpose), **kw)


def read_json(path, purpose: str):
    with open(guard(path, purpose), "r", encoding="utf-8") as f:
        return json.load(f)


# =====================================================================================
# CLI
# =====================================================================================
PHASES = ["run", "selftest", "prepare", "audit", "preprocess", "split", "degraded", "gate", "status", "train", "evaluate", "xai",
          "analyze", "report", "all"]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Test 5 (independent): hard-degradation-aware deepfake detection + Grad-CAM stability")
    p.add_argument("--phase", default="run", choices=PHASES)
    p.add_argument("--root", default=None, help=f"Experiment root (default {DEFAULT_ROOT})")
    p.add_argument("--bio-root", default=r"E:\Datasets\BioDeepAV")
    p.add_argument("--ffpp-root", default=r"E:\Datasets\FaceForensics++_C23")
    p.add_argument("--datasets", default="bio,ffpp")
    p.add_argument("--data-seed", type=int, default=DATA_SEED)
    p.add_argument("--seeds", default=TRAIN_SEEDS)
    p.add_argument("--variants", default="5A,5B,5C,5D")
    p.add_argument("--train-on", default="BioDeepAV,FF++", help="Each training set gets its own models (tested on both datasets)")
    p.add_argument("--dfd-grouping", choices=["actor", "pair"], default="actor",
                   help="actor (default, strict): DFD videos sharing ANY actor are one group; pair: Test 4 rule")
    p.add_argument("--hash-scope", choices=["all", "originals"], default="all",
                   help="all: full SHA-256 of every original + every manifest degraded file; originals: degraded files only when selected")
    p.add_argument("--deg-per-source", type=int, default=DEG_PER_SOURCE)
    p.add_argument("--max-per-folder", type=int, default=None, help="Random subset per folder (smoke tests only)")
    p.add_argument("--smoke", action="store_true", help="Tiny end-to-end run in a separate root (pipeline check only)")
    p.add_argument("--min-per-class", type=int, default=MIN_PER_CLASS_IN_SPLIT,
                   help="Gate: minimum real and fake videos per dataset in val and test")
    p.add_argument("--epochs1", type=int, default=EPOCHS1)
    p.add_argument("--epochs2", type=int, default=EPOCHS2)
    p.add_argument("--curriculum-epochs", default=f"{CURRICULUM[0]},{CURRICULUM[1]}")
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--lr1", type=float, default=LR1)
    p.add_argument("--lr2", type=float, default=LR2)
    p.add_argument("--lambda-feat", type=float, default=LAMBDA_FEAT)
    p.add_argument("--lambda-cam", type=float, default=LAMBDA_CAM)
    p.add_argument("--hard-share", type=float, default=HARD_SHARE)
    p.add_argument("--hard-weights", default=HARD_WEIGHTS)
    p.add_argument("--xai-per-type", type=int, default=XAI_PER_TYPE)
    p.add_argument("--n-boot", type=int, default=N_BOOT)
    p.add_argument("--workers", type=int, default=None, help="Video decoding threads (default: half the CPU cores, max 8)")
    p.add_argument("--io-workers", type=int, default=4, help="Hashing / ffprobe threads")
    p.add_argument("--loader-threads", type=int, default=3)
    p.add_argument("--gate-rehash", action="store_true", help="Gate: re-compute the full SHA-256 of every used file")
    p.add_argument("--no-share-stage1", action="store_true", help="Train a separate Stage 1 for every variant")
    p.add_argument("--save-every-steps", type=int, default=1,
                   help="Write the crash-safe training resume checkpoint every N steps (1 = every step; power-cut protection)")
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--gpu-max-temp", type=int, default=80)
    p.add_argument("--gpu-pause", type=int, default=120)
    a = p.parse_args(argv)
    if a.smoke:
        a.max_per_folder = a.max_per_folder or 30
        a.min_per_class = min(a.min_per_class, 2)
        a.epochs1, a.epochs2 = min(a.epochs1, 1), min(a.epochs2, 2)
        a.curriculum_epochs = "1,1"
        a.xai_per_type = min(a.xai_per_type, 3)
        a.n_boot = min(a.n_boot, 100)
        a.seeds = a.seeds if a.seeds != TRAIN_SEEDS else "42,123"
    root = Path(a.root) if a.root else (SMOKE_ROOT if a.max_per_folder else DEFAULT_ROOT)
    if a.max_per_folder and root.resolve() == DEFAULT_ROOT.resolve():
        raise SystemExit("--max-per-folder / --smoke must not write into the main Test 5 root (use another --root).")
    a.root = root
    a.workers = a.workers or max(1, min(8, (os.cpu_count() or 4) // 2))  # leave cores for the OS / desktop
    a.seed_list = [int(s) for s in str(a.seeds).split(",") if s.strip()]
    a.variant_list = [v.strip().upper() for v in a.variants.split(",") if v.strip()]
    bad = [v for v in a.variant_list if v not in VARIANTS]
    if bad:
        raise SystemExit(f"unknown variants {bad}")
    alias = {"bio": "BioDeepAV", "biodeepav": "BioDeepAV", "ffpp": "FF++", "ff++": "FF++"}
    a.train_list = [alias.get(t.strip().lower(), t.strip()) for t in a.train_on.split(",") if t.strip()]
    w = {}
    for tok in a.hard_weights.split(","):
        k, v = tok.split(":")
        w[k.strip()] = float(v)
    a.hard_w = {k: v / sum(w.values()) for k, v in w.items()}
    a.curriculum = tuple(int(x) for x in a.curriculum_epochs.split(","))[:2]
    return a


def variant_config(a, v: str) -> dict:
    c = dict(VARIANTS[v])
    if c["lambda_feat"] > 0:
        c["lambda_feat"] = a.lambda_feat
    if c["lambda_cam"] > 0:
        c["lambda_cam"] = a.lambda_cam
    return c


# =====================================================================================
# ARCHITECTURE (AudioVisualDeepFakeCNN family, same layer layout as the documented Test 4 model; fresh init)
# =====================================================================================
def block(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.ReLU(inplace=True), nn.BatchNorm2d(cout),
                         nn.Conv2d(cout, cout, 3, padding=1), nn.ReLU(inplace=True), nn.BatchNorm2d(cout), nn.MaxPool2d(2))


class VideoCNN(nn.Module):
    def __init__(self, width=(16, 32, 64, 128)):
        super().__init__()
        ch = (3,) + tuple(width)
        self.features = nn.Sequential(*[block(ch[i], ch[i + 1]) for i in range(len(width))])
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward_act(self, x):  # [B,T,3,H,W] -> (Grad-CAM activations [B*T,C,h,w], embedding [B,256])
        B, T, C, H, W = x.shape
        act = self.features(x.reshape(B * T, C, H, W))
        f = self.pool(act).flatten(1).view(B, T, -1)
        return act, torch.cat([f.mean(1), f.amax(1)], dim=1)

    def forward(self, x):
        return self.forward_act(x)[1]


class AudioCNN(nn.Module):
    def __init__(self, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(128, out_dim, 3, padding=1), nn.BatchNorm2d(out_dim), nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d((1, 1)))

    def forward(self, x):
        return self.net(x).flatten(1)


class AudioVisualDeepFakeCNN(nn.Module):
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

    def forward_train(self, x_v, x_a, a_mask):
        act, f_v = self.video_net.forward_act(x_v)
        f_a = self.audio_net(x_a) * a_mask.view(-1, 1).to(x_a.dtype)
        return self.fusion_head(torch.cat([f_v, f_a], dim=1)), f_v, act


def architecture_signature(model: nn.Module) -> str:
    sd = model.state_dict()
    return sha256_text(type(model).__name__ + "\n" + "\n".join(f"{k}:{tuple(v.shape)}" for k, v in sd.items()))


def architecture_source_sha() -> str:
    return sha256_text("".join(inspect.getsource(o) for o in (block, VideoCNN, AudioCNN, AudioVisualDeepFakeCNN)))


def weights_sha256(sd) -> str:
    h = hashlib.sha256()
    for k in sorted(sd):
        h.update(k.encode())
        h.update(sd[k].detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def to_video_tensor(arr: np.ndarray, device) -> torch.Tensor:
    """uint8 RGB [B,T,H,W,3] -> float [B,T,3,H,W] in [-1, 1]."""
    return torch.from_numpy(np.ascontiguousarray(arr)).permute(0, 1, 4, 2, 3).float().div_(127.5).sub_(1.0).to(device, non_blocking=True)


# =====================================================================================
# GRAD-CAM + CONSISTENCY LOSSES
# =====================================================================================
def grad_cam(model, video, audio, a_mask, target_class: int = 1, threshold: float = 0.5) -> Dict[str, np.ndarray]:
    """Grad-CAM on the last VideoCNN block. video [B,T,3,H,W]. Returns spatial [B,H,W] (min-max normalised temporal mean),
    temporal [B,T] (per-frame CAM mass, sums to 1), probability_fake [B], prediction [B]."""
    store = {}

    def hook(_m, _i, o):
        store["a"] = o
        o.register_hook(lambda g: store.__setitem__("g", g))

    B, T, _, H, W = video.shape
    h = model.video_net.features[-1].register_forward_hook(hook)
    try:
        model.eval()
        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            logits = model(video, audio, mode="av", a_mask=a_mask)
            logits[:, target_class].sum().backward()
    finally:
        h.remove()
    A, G = store["a"].detach(), store["g"].detach()
    cam = torch.relu((G.mean((2, 3), keepdim=True) * A).sum(1))
    cam = F.interpolate(cam[:, None], size=(H, W), mode="bilinear", align_corners=False)[:, 0].view(B, T, H, W)
    temporal = cam.flatten(2).sum(-1)
    temporal = temporal / (temporal.sum(1, keepdim=True) + 1e-8)
    spatial = cam.mean(1).flatten(1)
    mn, mx = spatial.min(1, keepdim=True)[0], spatial.max(1, keepdim=True)[0]
    spatial = ((spatial - mn) / (mx - mn + 1e-8)).view(B, H, W)
    prob = torch.softmax(logits.detach().float(), 1)[:, 1]
    model.zero_grad(set_to_none=True)
    return dict(spatial=spatial.cpu().numpy(), temporal=temporal.cpu().numpy(), p=prob.cpu().numpy(),
                pred=(prob >= threshold).int().cpu().numpy())


def soft_gradcam(logits, act, B: int, T: int, target_class: int = 1, create_graph: bool = True):
    """Differentiable Grad-CAM (same layer/recipe as grad_cam): [B,h,w] in [0,1]."""
    g = torch.autograd.grad(logits[:, target_class].sum(), act, create_graph=create_graph, retain_graph=True)[0]
    cam = torch.relu((g.mean((2, 3), keepdim=True) * act).sum(1))
    cam = cam.view(B, T, *cam.shape[-2:]).mean(1)
    return cam / (cam.flatten(1).amax(1).view(-1, 1, 1) + 1e-6)


def cam_consistency_loss(a, b):
    """1 - soft IoU of two [B,h,w] maps in [0,1]."""
    return (1.0 - torch.minimum(a, b).flatten(1).sum(1) / (torch.maximum(a, b).flatten(1).sum(1) + 1e-6)).mean()


def feature_consistency_loss(za, zb):
    return (F.normalize(za, dim=1) - F.normalize(zb, dim=1)).pow(2).sum(1).mean()


def cam_metrics(a: np.ndarray, b: np.ndarray, top: float = 0.2) -> Tuple[float, float, float]:
    """Pearson r, top-20 % IoU, centre-of-mass shift (fraction of the map side)."""
    r = float(np.corrcoef(a.ravel(), b.ravel())[0, 1]) if a.std() > 1e-8 and b.std() > 1e-8 else float("nan")
    ta, tb = a >= np.quantile(a, 1 - top), b >= np.quantile(b, 1 - top)
    iou = float((ta & tb).sum() / max(1, (ta | tb).sum()))
    yy, xx = np.mgrid[0:a.shape[0], 0:a.shape[1]]

    def com(c):
        w = c.sum() + 1e-8
        return np.array([(yy * c).sum() / w, (xx * c).sum() / w])

    return r, iou, float(np.linalg.norm(com(a) - com(b)) / a.shape[0])


# =====================================================================================
# METRICS
# =====================================================================================
def compute_metrics(y_true, y_prob, threshold: float) -> Dict[str, Any]:
    y = np.asarray(y_true).astype(int)
    p = np.asarray(y_prob, dtype=np.float64)
    keep = ~np.isnan(p)
    y, p = y[keep], p[keep]
    out = dict(n=int(len(y)), n_real=int((y == 0).sum()), n_fake=int((y == 1).sum()), threshold=float(threshold))
    if not len(y):
        out.update({k: float("nan") for k in METRIC_KEYS}, tn=0, fp=0, fn=0, tp=0)
        return out
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = (int(v) for v in confusion_matrix(y, pred, labels=[0, 1]).ravel())
    rec = tp / (tp + fn) if tp + fn else float("nan")
    spec = tn / (tn + fp) if tn + fp else float("nan")
    prec = tp / (tp + fp) if tp + fp else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if np.isfinite(prec) and np.isfinite(rec) and prec + rec > 0 else 0.0
    bal = (rec + spec) / 2 if np.isfinite(rec) and np.isfinite(spec) else float("nan")
    auc = float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")
    out.update(auc=auc, accuracy=float((pred == y).mean()), balanced_accuracy=float(bal), precision=float(prec), recall=float(rec),
               sensitivity=float(rec), specificity=float(spec), f1=float(f1), tn=tn, fp=fp, fn=fn, tp=tp)
    return out


def safe_auc(y, p, min_per_class: int = 1) -> float:
    y = np.asarray(y).astype(int)
    if (y == 0).sum() < min_per_class or (y == 1).sum() < min_per_class:
        return float("nan")
    return float(roc_auc_score(y, p))


def find_threshold(y, p) -> Dict[str, float]:
    """Validation-only threshold: max balanced accuracy, ties -> F1, then closeness to 0.5."""
    lo, hi, st = THRESHOLD_GRID
    best = None
    for t in np.round(np.arange(lo, hi + st / 2, st), 4):
        m = compute_metrics(y, p, float(t))
        key = (np.nan_to_num(m["balanced_accuracy"], nan=-1), np.nan_to_num(m["f1"], nan=-1), -abs(t - 0.5))
        if best is None or key > best[0]:
            best = (key, float(t), m)
    return dict(threshold=best[1], balanced_accuracy=best[2]["balanced_accuracy"], f1=best[2]["f1"], n=best[2]["n"])


class WeightedAUC:
    """AUC for many bootstrap weight vectors at once. Items are sorted into tie blocks once; for weights W [R, n]
    AUC_r = sum_b P_b (Ncum_before_b + 0.5 N_b) / (sum P * sum N)."""

    def __init__(self, y, p):
        self.y = np.asarray(y).astype(np.float64)
        u, inv = np.unique(np.asarray(p, dtype=np.float64), return_inverse=True)
        self.M = sparse.csr_matrix((np.ones(len(inv)), (np.arange(len(inv)), inv)), shape=(len(inv), len(u)))

    def __call__(self, W: np.ndarray) -> np.ndarray:
        W = np.atleast_2d(W).astype(np.float64)
        Pb = np.asarray(self.M.T @ (W * self.y).T).T          # [R, n_blocks] positive weight per tie block
        Nb = np.asarray(self.M.T @ (W * (1 - self.y)).T).T
        before = np.cumsum(Nb, axis=1) - Nb
        num = (Pb * (before + 0.5 * Nb)).sum(1)
        den = Pb.sum(1) * Nb.sum(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(den > 0, num / den, np.nan)


# =====================================================================================
# INVENTORY (source audit)
# =====================================================================================
def source_folders(a) -> List[dict]:
    ds = {d.strip().lower() for d in a.datasets.split(",") if d.strip()}
    out = []
    if "bio" in ds:
        for dname, sub, lab, rel in BIO_FOLDERS:
            out.append(dict(dataset=dname, subset=sub, label=lab, folder=Path(a.bio_root).joinpath(*rel)))
    if "ffpp" in ds:
        for sub in FFPP_FOLDERS:
            out.append(dict(dataset="FF++", subset=sub, label=0 if sub == "original" else 1, folder=Path(a.ffpp_root) / sub))
    return out


def yt_id(stem: str) -> Optional[str]:
    m = re.match(r"^(.*)_\d{4}_S\d+_E\d+", stem)
    return m.group(1) if m else None


def identity_tokens(dataset: str, subset: str, label: int, stem: str, dfd_grouping: str) -> Tuple[List[str], str, str]:
    """Returns (tokens, basis, status). Videos sharing any token are one identity group."""
    if dataset == "BioDeepAV":
        if label == 0:
            y = yt_id(stem)
            if y:
                return [f"bio_yt:{y}"], "YouTube id parsed from file name", "verified"
            return [f"bio_file:{stem}"], "file stem (YouTube-id pattern not matched)", "unverified"
        return [f"bio_fake:{stem}"], "file stem (UUID); no metadata links BioDeepAV fakes to a source identity", "unverifiable"
    if subset == "DeepFakeDetection":
        m = re.match(r"^(\d+)_(\d+)__", stem)
        if not m:
            return [f"dfd_file:{stem}"], "file stem (actor pattern not matched)", "unverified"
        if dfd_grouping == "pair":
            return [f"dfd_pair:{m.group(1)}_{m.group(2)}"], "DFD actor pair (Test 4 rule)", "verified (pair-level)"
        return [f"dfd_actor:{m.group(1)}", f"dfd_actor:{m.group(2)}"], "DFD target and source actor ids", "verified"
    if subset == "original":
        return [f"ff:{stem}"], "FF++ YouTube sequence id", "verified"
    parts = stem.split("_")
    if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
        return [f"ff:{parts[0]}", f"ff:{parts[1]}"], "FF++ target and source sequence ids", "verified"
    return [f"ff_file:{stem}"], "file stem (id pattern not matched)", "unverified"


def wav_candidates(folder: Path, stem: str) -> List[Path]:
    return [folder / f"{stem}.wav", folder.parent / "audios" / f"{stem}.wav"]


def ffprobe_meta(path: str) -> dict:
    out = dict(probe_ok=False, probe_tool="ffprobe" if FFPROBE else "opencv", duration_s=np.nan, fps=np.nan, width=np.nan,
               height=np.nan, n_frames=np.nan, video_codec="", has_audio_stream=None, audio_sample_rate=np.nan, probe_error="")
    if FFPROBE:
        try:
            raw = subprocess.check_output([FFPROBE, "-v", "error", "-show_entries",
                                           "format=duration:stream=codec_type,codec_name,width,height,avg_frame_rate,nb_frames,sample_rate",
                                           "-of", "json", str(path)], timeout=60, creationflags=NO_WINDOW, stderr=subprocess.DEVNULL)
            j = json.loads(raw.decode("utf-8", "ignore"))
            vs = [s for s in j.get("streams", []) if s.get("codec_type") == "video"]
            au = [s for s in j.get("streams", []) if s.get("codec_type") == "audio"]
            if vs:
                v = vs[0]
                try:
                    num, den = (str(v.get("avg_frame_rate") or "0/0") + "/1").split("/")[:2]
                    fps = float(num) / float(den) if float(den) else np.nan
                except ValueError:
                    fps = np.nan
                out.update(probe_ok=bool(v.get("width")), duration_s=float(j.get("format", {}).get("duration") or np.nan), fps=fps,
                           width=float(v.get("width") or np.nan), height=float(v.get("height") or np.nan),
                           n_frames=float(v.get("nb_frames")) if str(v.get("nb_frames", "")).isdigit() else np.nan,
                           video_codec=v.get("codec_name", ""), has_audio_stream=bool(au),
                           audio_sample_rate=float(au[0].get("sample_rate") or np.nan) if au else np.nan)
            return out
        except Exception as e:  # fall back to OpenCV
            out["probe_error"] = f"ffprobe: {type(e).__name__}"
    cap = open_video(path)
    if cap.isOpened():
        fps = float(cap.get(cv2.CAP_PROP_FPS) or np.nan)
        n = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or np.nan)
        out.update(probe_ok=True, probe_tool="opencv", fps=fps, n_frames=n, width=float(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                   height=float(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), duration_s=n / fps if fps and fps > 0 else np.nan)
    cap.release()
    return out


class KVCache:
    """JSON cache of per-file results keyed by path|size|mtime_ns (Test 5's own; resumable)."""

    def __init__(self, path: Path):
        self.path = path
        self.d = read_json(path, "test5_audit_cache") if path.exists() else {}
        self.lock = threading.Lock()
        self.last = time.time()

    def save(self, force=False):
        if force or time.time() - self.last > 60:
            with self.lock:
                write_json(self.path, self.d)
                self.last = time.time()


def probe_hash_files(paths: List[str], hash_set: set, probe_set: set, P: Paths, workers: int, label: str) -> Dict[str, dict]:
    cache = KVCache(P.data_audit / "probe_hash_cache.json")
    res, todo = {}, []
    for p in paths:
        try:
            st = os.stat(p)
        except OSError:
            res[p] = dict(exists=False)
            continue
        k = f"{p}|{st.st_size}|{st.st_mtime_ns}"
        c = cache.d.setdefault(k, {})
        np_, nh = (p in probe_set and "probe_ok" not in c), (p in hash_set and "sha256" not in c)
        if np_ or nh:
            todo.append((p, k, np_, nh))
        res[p] = k
    if todo:
        print(f"  {label}: probing/hashing {len(todo):,} files ({workers} threads) ...")
        t0, done, last = time.time(), 0, 0

        def work(t):
            p, k, np_, nh = t
            out = {}
            if np_:
                out.update(ffprobe_meta(p))
            if nh:
                out["sha256"] = sha256_file(p)
            return k, out

        for _, fut in imap_bounded(work, todo, workers):
            k, out = fut.result()
            with cache.lock:
                cache.d[k].update(to_jsonable(out))
            done += 1
            if time.time() - last > 0.5 or done == len(todo):
                rate = done / max(1e-3, time.time() - t0)
                sys.stdout.write(f"\r    {progress_bar(done, len(todo))} {done:,}/{len(todo):,} | {rate:5.1f} files/s | "
                                 f"ETA {fmt_time((len(todo) - done) / max(rate, 1e-3))}   ")
                sys.stdout.flush()
                last = time.time()
                cache.save()
        sys.stdout.write("\n")
    cache.save(force=True)
    out = {}
    for p, k in res.items():
        if isinstance(k, dict):
            out[p] = k
        else:
            size, mt = k.rsplit("|", 2)[1:]
            out[p] = dict(exists=True, file_size=int(size), mtime_ns=str(mt), **cache.d[k])  # str: ns precision survives CSV
    return out


def phase_audit(a, P: Paths) -> pd.DataFrame:
    print("\n" + "=" * 100 + "\n PHASE AUDIT: fresh source inventory (originals + manifest-linked degraded files)\n" + "=" * 100)
    folders = source_folders(a)
    cfg = dict(folders=[dict(f, folder=str(f["folder"])) for f in folders], bio_root=a.bio_root, ffpp_root=a.ffpp_root,
               datasets=a.datasets, max_per_folder=a.max_per_folder, discovery="folder glob *.mp4 + <folder>/degraded_output/manifest.json",
               dfd_grouping=a.dfd_grouping, data_seed=a.data_seed, recorded_at=now())
    src_cfg = P.configs / "data_sources.json"
    if src_cfg.exists():
        old = read_json(src_cfg, "test5_config")
        if [f["folder"] for f in old["folders"]] != [f["folder"] for f in cfg["folders"]] or old.get("max_per_folder") != a.max_per_folder:
            raise SystemExit(f"[STOP] data source configuration differs from {src_cfg}; use a new --root for a different data set.")
    write_json(src_cfg, cfg)
    rng = random.Random(a.data_seed)
    originals = []
    for f in folders:
        if not f["folder"].exists():
            print(f"  [WARN] missing folder (skipped): {f['folder']}")
            continue
        files = sorted(f["folder"].glob("*.mp4"))
        if a.max_per_folder and len(files) > a.max_per_folder:
            files = sorted(rng.sample(files, a.max_per_folder))
        for p in files:
            toks, basis, status = identity_tokens(f["dataset"], f["subset"], f["label"], p.stem, a.dfd_grouping)
            wav = next((w for w in wav_candidates(f["folder"], p.stem) if w.exists()), None)
            originals.append(dict(kind="original", item_id=f"{f['dataset']}/{f['subset']}/{p.name}", dataset=f["dataset"], subset=f["subset"],
                                  folder=str(f["folder"]), label=f["label"], label_name="real" if f["label"] == 0 else "fake",
                                  original_video_id=p.stem, identity_tokens="|".join(toks), identity_basis=basis, identity_status=status,
                                  path=str(p), external_wav=str(wav) if wav else ""))
    for r in originals:
        r["source_key"] = r["item_id"]
    by_folder = {(r["folder"].lower(), Path(r["path"]).name.lower()): r for r in originals}
    by_path = {r["path"].lower(): r for r in originals}
    degraded, manifests, orphans = [], [], []
    for f in folders:
        mpath = f["folder"] / "degraded_output" / "manifest.json"
        if not mpath.exists():
            manifests.append(dict(folder=str(f["folder"]), manifest="", present=False))
            continue
        m = read_json(mpath, "dataset_manifest_readonly")
        base = mpath.parent
        st = dict(folder=str(f["folder"]), manifest=str(mpath), present=True, entries=len(m.get("files", [])), tool=m.get("tool"),
                  version=m.get("version"), created=m.get("createdAt"), matched=0, unmatched=0, outputs=0, missing=0)
        listed = {str(base / o["output"]).lower() for it in m.get("files", []) for o in it.get("outputs", [])}
        listed |= {str(Path(o["outputPath"])).lower() for it in m.get("files", []) for o in it.get("outputs", []) if o.get("outputPath")}
        for item in m.get("files", []):
            sname = Path(item.get("source", "")).name
            r_f = by_folder.get((str(f["folder"]).lower(), sname.lower()))
            sp = item.get("sourcePath")
            r_p = by_path.get(str(Path(sp)).lower()) if sp else None
            src = r_f or r_p
            if src is None:
                st["unmatched"] += 1
                continue
            st["matched"] += 1
            for out in item.get("outputs", []):
                st["outputs"] += 1
                p = base / out["output"]
                if not p.exists() and out.get("outputPath"):
                    p = Path(out["outputPath"])
                exists = p.exists()
                st["missing"] += int(not exists)
                sev = str(out.get("severity", "")).lower()
                degraded.append(dict(kind="degraded", item_id=f"{src['item_id']}::{out.get('degradation', 'unknown')}:{sev}:{p.name}",
                                     source_key=src["item_id"], dataset=src["dataset"], subset=src["subset"], folder=src["folder"],
                                     label=src["label"], label_name=src["label_name"], original_video_id=src["original_video_id"],
                                     identity_tokens=src["identity_tokens"], identity_basis=src["identity_basis"],
                                     identity_status=src["identity_status"], path=str(p), external_wav="",
                                     degradation=out.get("degradation", "unknown"), severity=sev, severity_level=SEV_LEVEL.get(sev, np.nan),
                                     level=str(out.get("level", "")), random_set=out.get("randomSet"), manifest=str(mpath),
                                     link_method="folder+filename" if r_f is not None else "manifest sourcePath",
                                     link_source_path_match=(r_p is not None and (r_f is None or r_p is r_f)) if sp else None,
                                     link_stem_match=Path(out["output"]).stem.lower().startswith(Path(sname).stem.lower()),
                                     exists=exists))
        for q in base.rglob("*.mp4"):
            if str(q).lower() not in listed:
                orphans.append(dict(folder=str(f["folder"]), path=str(q), size=q.stat().st_size))
        manifests.append(st)
    print(f"  originals found: {len(originals):,} | manifest-linked degraded outputs: {len(degraded):,} | orphan degraded files "
          f"(on disk, not in a manifest; not used): {len(orphans):,}")
    orig_paths = [r["path"] for r in originals]
    wav_paths = [r["external_wav"] for r in originals if r["external_wav"]]
    deg_paths = [r["path"] for r in degraded if r["exists"]]
    hash_set = set(orig_paths) | set(wav_paths) | (set(deg_paths) if a.hash_scope == "all" else set())
    meta = probe_hash_files(orig_paths + wav_paths + deg_paths, hash_set, set(orig_paths) | set(deg_paths), P, a.io_workers, "inventory")
    for r in originals + degraded:
        m = meta.get(r["path"], dict(exists=False))
        r.update({k: m.get(k) for k in ("exists", "file_size", "mtime_ns", "sha256", "probe_ok", "probe_tool", "duration_s", "fps", "width",
                                        "height", "n_frames", "video_codec", "has_audio_stream", "audio_sample_rate", "probe_error")})
        if r["external_wav"]:
            w = meta.get(r["external_wav"], {})
            r.update(external_wav_sha256=w.get("sha256"), external_wav_size=w.get("file_size"), external_wav_mtime_ns=w.get("mtime_ns"))
    dur = {r["item_id"]: r.get("duration_s") for r in originals}
    avail = defaultdict(list)
    for r in degraded:
        sd = dur.get(r["source_key"])
        r["link_duration_ratio"] = (r["duration_s"] / sd) if (r.get("duration_s") and sd and np.isfinite(r["duration_s"]) and sd > 0) else np.nan
        dr = r["link_duration_ratio"]
        r["link_metadata_ok"] = bool(r["exists"] and r.get("probe_ok") and r["link_stem_match"] and r["link_source_path_match"] is not False
                                     and (not np.isfinite(dr) or 0.5 <= dr <= 2.0) and np.isfinite(r["severity_level"]))
        if r["exists"]:
            avail[r["source_key"]].append(f"{r['degradation']}:{r['severity']}")
    for r in originals:
        r["available_degradations"] = ";".join(sorted(avail[r["item_id"]]))
        r["n_available_degraded"] = len(avail[r["item_id"]])
    inv = pd.DataFrame(originals + degraded)
    inv.to_csv(P.inventory, index=False)
    pd.DataFrame(orphans).to_csv(P.data_audit / "orphan_degraded_files.csv", index=False)
    pd.DataFrame(manifests).to_csv(P.data_audit / "manifest_summary.csv", index=False)
    # exact duplicate screen over every hashed file (originals, degraded, external wavs)
    rows = [(r["item_id"], r["kind"], r["dataset"], r.get("sha256")) for r in originals + degraded if r.get("sha256")]
    rows += [(r["item_id"] + "::wav", "wav", r["dataset"], r.get("external_wav_sha256")) for r in originals if r.get("external_wav_sha256")]
    dd = pd.DataFrame(rows, columns=["item_id", "kind", "dataset", "sha256"])
    dup = dd[dd.duplicated("sha256", keep=False)].sort_values(["sha256", "item_id"])
    dup.to_csv(P.data_audit / "exact_duplicates.csv", index=False)
    masks = [str(q) for root in (Path(a.ffpp_root), Path(a.bio_root)) if root.exists() for q in root.iterdir() if q.is_dir() and "mask" in q.name.lower()]
    o = inv[inv.kind == "original"]
    d = inv[inv.kind == "degraded"]
    summary = dict(
        created=now(), n_originals=len(o), n_degraded_listed=len(d), n_degraded_exist=int(as_bool(d.exists).sum()) if len(d) else 0,
        n_degraded_link_ok=int(as_bool(d.link_metadata_ok).sum()) if len(d) else 0, n_orphans=len(orphans),
        originals_probe_failed=int((~as_bool(o.probe_ok)).sum()), exact_duplicate_files=int(len(dup)),
        exact_duplicate_sets=int(dup.sha256.nunique()) if len(dup) else 0, hash_scope=a.hash_scope,
        per_folder=o.groupby(["dataset", "subset"]).agg(videos=("item_id", "size"), probe_ok=("probe_ok", lambda s: int(as_bool(s).sum())),
                                                        audio_stream=("has_audio_stream", lambda s: int(as_bool(s).sum())),
                                                        external_wav=("external_wav", lambda s: int((s.fillna("") != "").sum())),
                                                        median_width=("width", "median"), median_height=("height", "median"),
                                                        median_fps=("fps", "median"), median_duration_s=("duration_s", "median"),
                                                        degraded_available=("n_available_degraded", "sum")).reset_index().to_dict(orient="records"),
        identity_status=o.groupby(["dataset", "subset", "identity_status"]).size().rename("n").reset_index().to_dict(orient="records"),
        degraded_types=(d[as_bool(d.exists)].groupby(["dataset", "label_name", "degradation", "severity"]).size().rename("n").reset_index()
                        .to_dict(orient="records") if len(d) else []),
        manifests=manifests, localization_annotation_dirs=masks,
        notes=["BioDeepAV fake videos carry no identity/source metadata (UUID names): identity overlap with real videos cannot be "
               "verified from metadata and is screened by face-crop and audio duplicate checks instead.",
               "FF++ manipulation folders: degradation type/severity counts are identical across Deepfakes/Face2Face/FaceShifter/"
               "FaceSwap/NeuralTextures (same random assignment per sequence id) - documented, not leakage."])
    write_json(P.data_audit / "inventory_summary.json", summary)
    print(pd.DataFrame(summary["per_folder"]).to_string(index=False))
    print(f"  degraded linked with valid metadata: {summary['n_degraded_link_ok']:,}/{summary['n_degraded_listed']:,} | exact duplicate files: "
          f"{summary['exact_duplicate_files']} in {summary['exact_duplicate_sets']} sets | localisation annotation dirs: {masks or 'none'}")
    return inv


# =====================================================================================
# PREPROCESSING (label-agnostic: no function below receives a label)
# =====================================================================================
class FaceDetector:
    """Thread-safe OpenCV YuNet wrapper (one detector per thread)."""

    def __init__(self, model_path: Optional[Path], score: float = FACE_SCORE, max_side: int = 640):
        self.model_path, self.score, self.max_side = model_path, score, max_side
        self.backend = "yunet" if (model_path and Path(model_path).exists() and hasattr(cv2, "FaceDetectorYN")) else "none"
        self._local = threading.local()

    def detect(self, frame_bgr):
        if self.backend != "yunet":
            return None
        H, W = frame_bgr.shape[:2]
        sc = min(1.0, self.max_side / max(H, W))
        img = cv2.resize(frame_bgr, (int(W * sc), int(H * sc)), interpolation=cv2.INTER_AREA) if sc < 1 else frame_bgr
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
        return dict(bbox=b[:4] / sc, score=float(b[14]))


def face_box_relative(bbox, W, H, margin):
    x, y, w, h = [float(v) for v in bbox]
    side = min(max(w, h) * (1.0 + 2.0 * margin), float(min(W, H)))
    return np.array([(x + w / 2) / W, (y + h / 2) / H, side / min(W, H)], dtype=np.float32)


def normalize_frame_resolution(frame, short_side: int):
    if not short_side:
        return frame
    h, w = frame.shape[:2]
    s = short_side / min(h, w)
    if abs(s - 1.0) < 1e-3:
        return frame
    return cv2.resize(frame, (max(1, int(round(w * s))), max(1, int(round(h * s)))), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)


def face_bottleneck(patch, size: int, face_px: int = 0, jpeg: int = 0):
    if face_px:
        if patch.shape[0] != face_px or patch.shape[1] != face_px:
            patch = cv2.resize(patch, (face_px, face_px), interpolation=cv2.INTER_AREA if patch.shape[0] > face_px else cv2.INTER_CUBIC)
        if jpeg and 0 < jpeg < 100:
            ok, enc = cv2.imencode(".jpg", patch, [cv2.IMWRITE_JPEG_QUALITY, int(jpeg)])
            if ok:
                patch = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    if patch.shape[0] != size or patch.shape[1] != size:
        patch = cv2.resize(patch, (size, size), interpolation=cv2.INTER_AREA if patch.shape[0] > size else cv2.INTER_CUBIC)
    return patch


def crop_face(frame, box_rel, size: int, norm: dict):
    H, W = frame.shape[:2]
    side = float(np.clip(box_rel[2] * min(W, H), 8, min(W, H)))
    cx = float(np.clip(box_rel[0] * W, side / 2, W - side / 2))
    cy = float(np.clip(box_rel[1] * H, side / 2, H - side / 2))
    s = int(round(side))
    x0, y0 = int(round(cx - side / 2)), int(round(cy - side / 2))
    x0, y0 = max(0, min(x0, W - s)), max(0, min(y0, H - s))
    return face_bottleneck(frame[y0:y0 + s, x0:x0 + s], size, norm.get("face_px", 0), norm.get("jpeg", 0))


def open_video(path) -> cv2.VideoCapture:
    """VideoCapture with a single FFmpeg decode thread: parallelism comes from the worker pool, and FFmpeg's default
    (one thread per core, per capture) multiplied by the pool size oversubscribes the CPU and freezes the PC."""
    prop = getattr(cv2, "CAP_PROP_N_THREADS", None)
    if prop is not None:
        try:
            cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG, [prop, 1])
            if cap.isOpened():
                return cap
            cap.release()
        except Exception:
            pass
    return cv2.VideoCapture(str(path))


def read_sampled_frames(path, n_frames: int, max_side: int = 960):
    """n_frames uniformly spaced BGR frames + metadata."""
    cap = open_video(path)
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
    if not frames:
        # frame count unknown: count decodable frames first, then decode only the wanted ones (never hold the whole video)
        cap.release()
        cap = open_video(path)
        count = 0
        while cap.grab():
            count += 1
        cap.release()
        if not count:
            return None, meta
        want = set(np.linspace(0, count - 1, n_frames).round().astype(int).tolist())
        cap = open_video(path)
        for j in range(max(want) + 1):
            ok, fr = cap.read()
            if not ok:
                break
            if j in want:
                frames[j] = shrink(fr)
        if not frames:
            cap.release()
            return None, meta
    cap.release()
    keys = sorted(frames)
    want_list = np.linspace(0, keys[-1], n_frames).round().astype(int)
    return [frames[min(keys, key=lambda k: abs(k - t))] for t in want_list], meta


def smooth_boxes(boxes):
    if len(boxes) < 3:
        return boxes
    pad = np.concatenate([boxes[:1], boxes, boxes[-1:]])
    return ((pad[:-2] + pad[1:-1] + pad[2:]) / 3.0).astype(np.float32)


def extract_face_clip(path, n_frames: int, size: int, detector: Optional[FaceDetector], margin: float, norm: dict,
                      ref: Optional[Tuple[np.ndarray, bool]] = None):
    """video -> frames -> resolution normalisation -> YuNet face boxes (or inherited ref boxes) -> crop -> face bottleneck
    -> RGB uint8 [T,S,S,3]. No label input: identical processing for real and fake. Returns (clip or None, info)."""
    info = dict(frames=0, detected=0, detection_rate=float("nan"), fallback=True, face_rel_size=float("nan"),
                boxes=np.zeros((n_frames, 3), np.float32), det=np.zeros(n_frames, np.int8), width=0, height=0, fps=0.0, total_frames=0, error="")
    try:
        frames, meta = read_sampled_frames(path, n_frames)
        info.update(meta)
        if frames is None:
            info["error"] = "no frames decoded"
            return None, info
        if norm.get("frame_side"):
            frames = [normalize_frame_resolution(f, norm["frame_side"]) for f in frames]
        h, w = frames[0].shape[:2]
        T = len(frames)
        info["frames"] = T
        if ref is not None:
            boxes, fallback = ref
            info["detected"] = -1
        else:
            dets = [detector.detect(f) if detector is not None else None for f in frames]
            detected = np.array([d is not None for d in dets])
            info["detected"], info["detection_rate"], info["det"] = int(detected.sum()), float(detected.mean()), detected.astype(np.int8)
            fallback = not detected.any()
            boxes = np.zeros((T, 3), np.float32)
            if not fallback:
                idx = np.where(detected)[0]
                for t in range(T):
                    d = dets[t] if detected[t] else dets[idx[np.argmin(np.abs(idx - t))]]
                    boxes[t] = face_box_relative(d["bbox"], w, h, margin)
                boxes = smooth_boxes(boxes)
                info["face_rel_size"] = float(boxes[:, 2].mean())
        info["boxes"], info["fallback"] = boxes, bool(fallback)
        crops = [face_bottleneck(f, size, norm.get("face_px", 0), norm.get("jpeg", 0)) if fallback else crop_face(f, boxes[t], size, norm)
                 for t, f in enumerate(frames)]
        return np.stack(crops)[..., ::-1].copy(), info
    except Exception as e:
        info["error"] = f"{type(e).__name__}: {e}"
        return None, info


def load_audio_waveform(wav: Optional[str], video_path: str, n_samples: int = AUDIO_SAMPLES) -> Tuple[Optional[np.ndarray], str]:
    """16 kHz mono float32 tiled/cropped to n_samples. Returns (waveform or None, source)."""
    audio, src = None, "none"
    if wav and Path(wav).exists():
        try:
            sr, raw = wavfile.read(wav)
            if raw.ndim > 1:
                raw = raw[:, 0]
            audio = raw.astype(np.float32) / 32768.0 if raw.dtype == np.int16 else raw.astype(np.float32)
            if sr != AUDIO_SR and len(audio):
                g = math.gcd(int(sr), AUDIO_SR)
                audio = resample_poly(audio, AUDIO_SR // g, int(sr) // g).astype(np.float32)
            src = "external_wav"
        except Exception:
            audio = None
    if (audio is None or not len(audio)) and FFMPEG:
        try:
            raw = subprocess.check_output([FFMPEG, "-v", "error", "-i", str(video_path), "-vn", "-ac", "1", "-ar", str(AUDIO_SR), "-f", "s16le", "-"],
                                          creationflags=NO_WINDOW, timeout=30, stderr=subprocess.DEVNULL)
            audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            src = "video_stream"
        except Exception:
            audio = None
    if audio is None or not len(audio) or float(np.abs(audio).max()) < 1e-6:
        return None, "none" if audio is None or not len(audio) else "silent"
    audio = np.tile(audio, int(np.ceil(n_samples / len(audio))))[:n_samples] if len(audio) < n_samples else audio[:n_samples]
    return np.ascontiguousarray(audio, dtype=np.float32), src


def waveform_to_spec(audio: np.ndarray) -> np.ndarray:
    """16 kHz waveform -> log-magnitude STFT (n_fft 512, hop 256, Hann) -> bilinear 128x128 -> per-clip z-score clipped to +-3, /3."""
    with torch.no_grad():
        t = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32))
        st = torch.stft(t, n_fft=512, hop_length=256, win_length=512, window=torch.hann_window(512), return_complex=True)
        mag = torch.log1p(st.abs())
        s = F.interpolate(mag[None, None], size=(128, 128), mode="bilinear", align_corners=False)
        s = ((s - s.mean()) / (s.std() + 1e-6)).clamp(-3.0, 3.0) / 3.0
        return s.squeeze(0).numpy().astype(np.float32)


def degrade_audio(x: np.ndarray, deg: str, sev_level: int, rng) -> np.ndarray:
    x = np.array(x, dtype=np.float64, copy=True)
    n = len(x)
    v = AUDIO_DEG_LEVELS[deg][sev_level]
    if deg == "noise":
        x = x + rng.normal(0.0, math.sqrt((float(np.mean(x ** 2)) + 1e-12) / 10 ** (v / 10)), n)
    elif deg == "volume":
        x = np.round(x * 10 ** (v / 20) * 32767.0) / 32767.0
    elif deg == "lowpass":
        x = sosfiltfilt(butter(8, v, btype="low", fs=AUDIO_SR, output="sos"), x)
    elif deg == "resample":
        g = math.gcd(int(v), AUDIO_SR)
        y = resample_poly(resample_poly(x, int(v) // g, AUDIO_SR // g), AUDIO_SR // g, int(v) // g)
        x = y[:n] if len(y) >= n else np.pad(y, (0, n - len(y)))
    elif deg == "dropout":
        seg = AUDIO_SR // 10
        ns = max(1, n // seg)
        for s in rng.choice(ns, size=min(ns, max(1, int(round(v * ns)))), replace=False):
            x[s * seg:(s + 1) * seg] = 0.0
    elif deg == "silence":
        x[:] = 0.0
    return np.clip(x, -1.0, 1.0).astype(np.float32)


PREPROC_FUNCS = (FaceDetector, face_box_relative, normalize_frame_resolution, face_bottleneck, crop_face, read_sampled_frames,
                 smooth_boxes, extract_face_clip, load_audio_waveform, waveform_to_spec)


def preprocessing_code_sha() -> str:
    return sha256_text("".join(inspect.getsource(f) for f in PREPROC_FUNCS))


def preprocessing_config(a) -> dict:
    det = FaceDetector(FACE_MODEL)
    return dict(frame_sampling=f"{NUM_FRAMES} frames uniformly spaced over the decoded video (cv2 grab/retrieve; sequential fallback)",
                frames_cached=NUM_FRAMES, clip_len=CLIP_LEN, clip_frames=f"model clips use every {NUM_FRAMES // CLIP_LEN}th cached frame "
                f"(evaluation averages the {NUM_FRAMES // CLIP_LEN} offsets; training draws {CLIP_LEN} random sorted frames)",
                decode_max_side=960, resolution_normalisation=NORM, face_detector=dict(backend=det.backend, model=str(FACE_MODEL),
                model_sha256=sha256_file(FACE_MODEL) if FACE_MODEL.exists() else None, opencv=cv2.__version__, score_threshold=FACE_SCORE,
                nms=0.3, top_k=50, detect_max_side=640, selection="largest area x score"),
                crop=dict(rule="square side = max(w,h)*(1+2*margin), clipped inside the frame, 3-frame box smoothing, nearest-detection "
                          "box for frames without a face, full frame when no face in the video", margin=FACE_MARGIN, alignment=False,
                          effective_face_resolution_px=NORM["face_px"], output_size=IMG_SIZE),
                degraded_crops="degraded videos are cropped with their SOURCE video's relative face boxes (pixel-aligned pairs)",
                normalisation="uint8 RGB -> float (x/127.5 - 1)", audio=dict(sample_rate=AUDIO_SR, samples=AUDIO_SAMPLES,
                source="external .wav if present, else ffmpeg audio stream; tiled/cropped to 3 s; silent/no audio -> zero spectrogram + mask 0",
                spectrogram="STFT n_fft 512, hop 256, win 512 Hann, log1p magnitude, bilinear 128x128, per-clip z-score clip +-3 /3"),
                failed_video_handling="originals whose frames cannot be decoded are excluded before grouping (status recorded); degraded "
                                      "videos that fail decoding or the content-link test are excluded",
                label_independence="no preprocessing function receives a label; real and fake videos share one code path",
                preprocessing_code_sha256=preprocessing_code_sha(), cache_namespace="t5_", cache_dir=str(Path(a.root) / "cache"))


# =====================================================================================
# CACHES (Test 5 namespace; never search other directories)
# =====================================================================================
_PROV_LOCK = threading.Lock()


def record_provenance(P: Paths, logical: str, fname: str, meta: dict):
    with _PROV_LOCK:
        prov = read_json(P.provenance, "test5_cache_provenance") if P.provenance.exists() else {}
        prov[logical] = dict(fname=fname, created_by=SCRIPT_PATH.name, script_sha256=code_sha256(),
                             preprocessing_code_sha256=preprocessing_code_sha(), recorded_at=now(), **meta)
        write_json(P.provenance, prov)


class ClipCache:
    """uint8 [N,T,S,S,3] memmap + status (0 pending, 1 ok, 2 failed) + face info. Resumable, Test 5 cache dir only."""

    def __init__(self, logical: str, key_src: str, n: int, P: Paths, T=NUM_FRAMES, S=IMG_SIZE):
        self.logical, self.n, self.T, self.S, self.P = logical, n, T, S, P
        self.key_sha = sha256_text(key_src)
        self.fname = f"t5_{logical}_T{T}_S{S}_n{n}_{self.key_sha[:16]}"
        self.a_path, self.s_path, self.i_path = (P.cache / f"{self.fname}{s}" for s in (".npy", ".status.npy", ".info.npz"))
        self.arr = self.status = self.info = None

    def complete(self) -> bool:
        return self.a_path.exists() and self.s_path.exists() and self.i_path.exists() and bool((np.load(guard(self.s_path, "test5_cache")) != 0).all())

    def load(self):
        self.status = np.load(guard(self.s_path, "test5_cache"))
        self.arr = np.load(guard(self.a_path, "test5_cache"), mmap_mode="r")
        z = np.load(guard(self.i_path, "test5_cache"))
        self.info = {k: z[k] for k in z.files}
        return self

    def build(self, work_fn: Callable[[int], Tuple[Optional[np.ndarray], Optional[dict]]], workers: int, label: str, meta: dict):
        if self.complete():
            print(f"  -> {label}: complete Test 5 cache found ({self.a_path.name})")
            return self.load()
        shape = (self.n, self.T, self.S, self.S, 3)
        need = int(np.prod(shape)) if not self.a_path.exists() else 0
        free = shutil.disk_usage(self.P.cache).free
        if need > free - 3e9:
            raise SystemExit(f"[STOP] cache {self.fname} needs {need / 1e9:.1f} GB, only {free / 1e9:.1f} GB free on {self.P.cache}")
        if self.a_path.exists() and self.s_path.exists():
            status = np.load(guard(self.s_path, "test5_cache"))
            arr = np.lib.format.open_memmap(guard(self.a_path, "test5_cache"), mode="r+")
            info = {k: v for k, v in np.load(guard(self.i_path, "test5_cache")).items()} if self.i_path.exists() else None
        else:
            status = np.zeros(self.n, np.int8)
            arr = np.lib.format.open_memmap(self.a_path, mode="w+", dtype=np.uint8, shape=shape)
            info = None
        if info is None:
            info = dict(boxes=np.zeros((self.n, self.T, 3), np.float32), det=np.full((self.n, self.T), -1, np.int8),
                        meta=np.full((self.n, 7), np.nan, np.float32), error=np.array([""] * self.n, dtype=object).astype("U80"))
        todo = np.where(status == 0)[0]
        print(f"\n[{label}] {len(todo):,} of {self.n:,} videos to extract | {workers} threads -> {self.a_path.name}")

        def save():
            arr.flush()
            np.save(self.s_path, status)
            np.savez(self.i_path, **info)

        t0, done, last_p, last_s = time.time(), 0, 0, time.time()
        for i, fut in imap_bounded(work_fn, [int(i) for i in todo], workers):
            try:
                clip, inf = fut.result()
            except Exception as e:
                clip, inf = None, dict(error=str(e))
            if clip is not None and clip.shape == shape[1:]:
                arr[i] = clip
                status[i] = 1
            else:
                arr[i] = 0
                status[i] = 2
            if inf:
                if "boxes" in inf:
                    info["boxes"][i] = inf["boxes"][:self.T]
                    info["det"][i] = inf["det"][:self.T] if inf.get("detected", 0) >= 0 else -1
                    info["meta"][i] = [inf["width"], inf["height"], inf["fps"], inf["total_frames"], float(inf["fallback"]),
                                       inf["face_rel_size"], inf["detected"]]
                info["error"][i] = str(inf.get("error", ""))[:80]
            del clip, inf, fut
            done += 1
            now_t = time.time()
            if now_t - last_s > 60:
                save()
                last_s = now_t
            if now_t - last_p > 0.3 or done == len(todo):
                rate = done / max(1e-3, now_t - t0)
                sys.stdout.write(f"\r  {progress_bar(done, len(todo))} {done:,}/{len(todo):,} | {rate:5.1f} vid/s | "
                                 f"ETA {fmt_time((len(todo) - done) / max(rate, 1e-3))}   ")
                sys.stdout.flush()
                last_p = now_t
        sys.stdout.write("\n")
        save()
        del arr
        record_provenance(self.P, self.logical, self.fname, dict(key_src_sha256=self.key_sha, n=self.n, T=self.T, S=self.S,
                                                                 ok=int((status == 1).sum()), failed=int((status == 2).sum()), **meta))
        print(f"  saved {self.a_path.name} ({self.a_path.stat().st_size / 1e9:.2f} GB, {int((status == 2).sum())} failed)")
        return self.load()


def clean_cache_key(index_df: pd.DataFrame) -> str:
    return ("\n".join(f"{r.item_id}|{r.sha256}" for r in index_df.itertuples()) + json.dumps(NORM, sort_keys=True)
            + f"|{NUM_FRAMES}|{IMG_SIZE}|{FACE_MARGIN}|{FACE_SCORE}|" + preprocessing_code_sha())


def deg_cache_key(index_df: pd.DataFrame, clean_fname: str) -> str:
    return ("\n".join(f"{r.item_id}|{r.sha256}|{r.source_key}" for r in index_df.itertuples()) + json.dumps(NORM, sort_keys=True)
            + f"|{NUM_FRAMES}|{IMG_SIZE}|boxes-from:{clean_fname}|" + preprocessing_code_sha())


def audio_cache_paths(P: Paths, key_sha: str, n: int):
    stem = f"t5_audio_n{n}_{key_sha[:16]}"
    return P.cache / f"{stem}.spec.npy", P.cache / f"{stem}.wave.npy", P.cache / f"{stem}.status.npy", stem


def build_audio_cache(P: Paths, index_df: pd.DataFrame, workers: int):
    """spectrograms float32 [N,1,128,128] + int16 waveforms [N,48000] + status (1 audio, 3 none/silent)."""
    key = sha256_text("\n".join(f"{r.item_id}|{r.sha256}|{r.external_wav_sha256}" for r in index_df.itertuples())
                      + f"|{AUDIO_SR}|{AUDIO_SAMPLES}|" + preprocessing_code_sha())
    n = len(index_df)
    sp, wp, stp, stem = audio_cache_paths(P, key, n)
    if sp.exists() and wp.exists() and stp.exists() and (np.load(guard(stp, "test5_cache")) != 0).all():
        print(f"  -> audio: complete Test 5 cache found ({stem})")
    else:
        if sp.exists() and stp.exists():
            status = np.load(guard(stp, "test5_cache"))
            spec = np.lib.format.open_memmap(guard(sp, "test5_cache"), mode="r+")
            wave = np.lib.format.open_memmap(guard(wp, "test5_cache"), mode="r+")
        else:
            status = np.zeros(n, np.int8)
            spec = np.lib.format.open_memmap(sp, mode="w+", dtype=np.float32, shape=(n, 1, 128, 128))
            wave = np.lib.format.open_memmap(wp, mode="w+", dtype=np.int16, shape=(n, AUDIO_SAMPLES))
        src_path = P.cache / f"{stem}.source.json"
        srcs = read_json(src_path, "test5_cache") if src_path.exists() else {}
        rows = list(index_df.itertuples())
        todo = np.where(status == 0)[0]
        print(f"\n[AUDIO] {len(todo):,} of {n:,} videos to extract")

        def work(i):
            r = rows[i]
            w, s = load_audio_waveform(r.external_wav if isinstance(r.external_wav, str) and r.external_wav else None, r.path)
            return i, w, s

        done, last = 0, 0
        for _, fut in imap_bounded(work, [int(i) for i in todo], workers):
            i, w, s = fut.result()
            if w is None:
                spec[i] = 0
                wave[i] = 0
                status[i] = 3
            else:
                spec[i] = waveform_to_spec(w)
                wave[i] = np.round(np.clip(w, -1, 1) * 32767).astype(np.int16)
                status[i] = 1
            srcs[str(i)] = s
            done += 1
            if time.time() - last > 0.3 or done == len(todo):
                sys.stdout.write(f"\r  {progress_bar(done, len(todo))} {done:,}/{len(todo):,}   ")
                sys.stdout.flush()
                last = time.time()
        sys.stdout.write("\n")
        spec.flush()
        wave.flush()
        np.save(stp, status)
        write_json(src_path, srcs)
        del spec, wave
        record_provenance(P, "audio", stem, dict(key_src_sha256=key, n=n, with_audio=int((status == 1).sum()),
                                                 spec_file=sp.name, wave_file=wp.name, status_file=stp.name))
    status = np.load(guard(stp, "test5_cache"))
    return (np.load(guard(sp, "test5_cache"), mmap_mode="r"), np.load(guard(wp, "test5_cache"), mmap_mode="r"), status == 1, key)


# =====================================================================================
# CONTENT FINGERPRINTS + NEAR-DUPLICATE SCREEN
# =====================================================================================
FP_SIDE = 24


def dhash64(img2d: np.ndarray) -> np.uint64:
    s = cv2.resize(img2d.astype(np.float32), (9, 8), interpolation=cv2.INTER_AREA)
    bits = (s[:, 1:] > s[:, :-1]).ravel()
    return np.uint64(int("".join("1" if b else "0" for b in bits), 2))


def standardize(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    m = x.reshape(x.shape[0], -1).mean(1).reshape(-1, *([1] * (x.ndim - 1)))
    s = x.reshape(x.shape[0], -1).std(1).reshape(-1, *([1] * (x.ndim - 1)))
    return (x - m) / (s + 1e-6)


def clip_fingerprint(clip: np.ndarray):
    """clip uint8 [T,S,S,3] -> (md5 of the decoded clip, degenerate flag, dHash [4] uint64, standardized gray [T,24,24] float16)."""
    g = clip.astype(np.float32).mean(-1)
    md5 = hashlib.md5(np.ascontiguousarray(clip).tobytes()).hexdigest()
    degenerate = bool(g.std() < 2.0)
    idx = np.linspace(0, len(g) - 1, 4).round().astype(int)
    dh = np.array([dhash64(g[t]) for t in idx], dtype=np.uint64)
    small = np.stack([cv2.resize(f, (FP_SIDE, FP_SIDE), interpolation=cv2.INTER_AREA) for f in g])
    return md5, degenerate, dh, standardize(small).astype(np.float16)


def spec_fingerprint(spec: np.ndarray, wave: np.ndarray):
    """spectrogram [1,128,128] (freq x time) -> (md5 of int16 waveform, dHash of 4 time segments, standardized 32x32)."""
    s = spec[0]
    dh = np.array([dhash64(s[:, k * 32:(k + 1) * 32]) for k in range(4)], dtype=np.uint64)
    small = cv2.resize(s, (32, 32), interpolation=cv2.INTER_AREA)[None]
    return hashlib.md5(np.ascontiguousarray(wave).tobytes()).hexdigest(), dh, standardize(small)[0].astype(np.float16)


def popcount(x: np.ndarray) -> np.ndarray:
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(x)
    lut = np.array([bin(i).count("1") for i in range(256)], np.uint8)
    return lut[x.view(np.uint8)].reshape(*x.shape, 8).sum(-1)


def near_dup_pairs(H: np.ndarray, valid: np.ndarray, mean_thr: float, chunk: int = 256) -> List[Tuple[int, int, float, int]]:
    """All pairs (i<j) whose mean Hamming distance over the 4 hashes is <= mean_thr."""
    iv = np.where(valid)[0]
    Hv = H[iv]
    out = []
    for a0 in range(0, len(Hv), chunk):
        pc = popcount(Hv[a0:a0 + chunk, None, :] ^ Hv[None, :, :]).astype(np.uint8)
        mean, mx = pc.mean(-1), pc.max(-1)
        ii, jj = np.where(mean <= mean_thr)
        k = jj > ii + a0
        for i, j in zip(ii[k], jj[k]):
            out.append((int(iv[a0 + i]), int(iv[j]), float(mean[i, j]), int(mx[i, j])))
        if len(out) > NEARDUP_MAX_PAIRS:
            print(f"  [WARN] near-duplicate candidates exceed {NEARDUP_MAX_PAIRS:,}; keeping the closest")
            out = sorted(out, key=lambda t: t[2])[:NEARDUP_MAX_PAIRS // 2]
    return out


def ncc(a: np.ndarray, b: np.ndarray) -> float:
    """Mean per-frame normalised cross-correlation of standardized arrays [T,h,w] (or [h,w])."""
    a, b = a.astype(np.float32), b.astype(np.float32)
    if a.ndim == 2:
        return float((a * b).mean())
    return float((a * b).reshape(len(a), -1).mean(1).mean())


# =====================================================================================
# PHASE PREPROCESS (clean originals + audio + fingerprints + preprocessing self-tests)
# =====================================================================================
def face_detector() -> FaceDetector:
    if not FACE_MODEL.exists():
        FACE_MODEL.parent.mkdir(parents=True, exist_ok=True)
        print(f"  downloading YuNet -> {FACE_MODEL}")
        urllib.request.urlretrieve(YUNET_URL, FACE_MODEL)
    det = FaceDetector(FACE_MODEL)
    if det.backend != "yunet":
        raise SystemExit("[STOP] YuNet face detector unavailable (OpenCV FaceDetectorYN missing)")
    return det


def load_inventory(P: Paths) -> pd.DataFrame:
    if not P.inventory.exists():
        raise SystemExit("[STOP] run --phase audit first")
    return read_csv(P.inventory, "test5_inventory", low_memory=False, dtype={"mtime_ns": str, "external_wav_mtime_ns": str})


def load_clean(P: Paths) -> Tuple[pd.DataFrame, ClipCache]:
    idx = read_csv(P.clean_index, "test5_manifest")
    c = ClipCache("faceclean", clean_cache_key(idx), len(idx), P)
    if not c.complete():
        raise SystemExit("[STOP] clean cache incomplete - run --phase preprocess")
    return idx, c.load()


def phase_preprocess(a, P: Paths):
    print("\n" + "=" * 100 + "\n PHASE PREPROCESS: fresh face crops + audio for every decodable original (label-agnostic)\n" + "=" * 100)
    inv = load_inventory(P)
    o = inv[(inv.kind == "original") & as_bool(inv.exists) & as_bool(inv.probe_ok) & inv.sha256.notna()].sort_values("item_id")
    idx = o[["item_id", "source_key", "dataset", "subset", "label", "path", "sha256", "external_wav", "external_wav_sha256"]].reset_index(drop=True)
    idx["external_wav"] = idx.external_wav.fillna("")
    idx.insert(1, "clean_idx", np.arange(len(idx)))
    idx.to_csv(P.clean_index, index=False)
    pcfg = preprocessing_config(a)
    write_json(P.configs / "preprocessing_config.json", pcfg)
    det = face_detector()
    cache = ClipCache("faceclean", clean_cache_key(idx), len(idx), P)
    paths = idx.path.tolist()
    cache.build(lambda i: extract_face_clip(paths[i], NUM_FRAMES, IMG_SIZE, det, FACE_MARGIN, NORM), a.workers,
                "ORIGINAL VIDEOS (face crop, resolution-normalised)", dict(kind="clean face clips", preprocessing=pcfg))
    spec, wave, has_audio, _ = build_audio_cache(P, idx, a.workers)
    # per-video face statistics
    meta = cache.info["meta"]
    face = idx[["item_id", "dataset", "subset", "label"]].copy()
    face["extraction_ok"] = cache.status == 1
    face["error"] = cache.info["error"]
    face["width"], face["height"], face["fps"], face["total_frames"] = meta[:, 0], meta[:, 1], meta[:, 2], meta[:, 3]
    face["fallback_full_frame"], face["face_rel_size"], face["detected_frames"] = meta[:, 4] > 0.5, meta[:, 5], meta[:, 6]
    face["detection_rate"] = face.detected_frames / NUM_FRAMES
    face["native_face_px"] = face.face_rel_size * np.minimum(face.width, face.height)
    face["crop_side_after_norm_px"] = face.face_rel_size * NORM["frame_side"]
    face["has_audio"] = has_audio
    print("  computing content fingerprints (decoded clip md5, dHash, standardized thumbnails) ...")
    N = len(idx)
    md5s, degen = np.array([""] * N, dtype="U32"), np.zeros(N, bool)
    dh = np.zeros((N, 4), np.uint64)
    gray = np.zeros((N, NUM_FRAMES, FP_SIDE, FP_SIDE), np.float16)
    bright, sharp = np.full(N, np.nan), np.full(N, np.nan)
    for i in range(N):
        if cache.status[i] != 1:
            continue
        c = np.asarray(cache.arr[i])
        md5s[i], degen[i], dh[i], gray[i] = clip_fingerprint(c)
        g = c.mean(-1)
        bright[i] = float(g.mean())
        sharp[i] = float(np.mean([cv2.Laplacian(f.astype(np.float32), cv2.CV_32F).var() for f in g[::4]]))
        if i % 500 == 0:
            sys.stdout.write(f"\r  {progress_bar(i + 1, N)}   ")
    sys.stdout.write("\n")
    a_md5, a_dh = np.array([""] * N, dtype="U32"), np.zeros((N, 4), np.uint64)
    a_small = np.zeros((N, 32, 32), np.float16)
    for i in np.where(has_audio)[0]:
        a_md5[i], a_dh[i], a_small[i] = spec_fingerprint(np.asarray(spec[i]), np.asarray(wave[i]))
    np.savez(P.data_audit / "fingerprints_clean.npz", md5=md5s, degenerate=degen, dhash=dh, gray=gray, audio_md5=a_md5,
             audio_dhash=a_dh, audio_small=a_small, has_audio=has_audio, ok=cache.status == 1)
    face["crop_brightness"], face["crop_sharpness"], face["degenerate_clip"] = bright, sharp, degen
    face.to_csv(P.data_audit / "face_detection_per_video.csv", index=False)
    summ = face.groupby(["dataset", "subset"]).agg(videos=("item_id", "size"), extraction_ok=("extraction_ok", "sum"),
                                                     fallback=("fallback_full_frame", "sum"), detection_rate=("detection_rate", "mean"),
                                                     native_face_px=("native_face_px", "median"), crop_side_px=("crop_side_after_norm_px", "median"),
                                                     with_audio=("has_audio", "sum")).reset_index()
    summ.to_csv(P.data_audit / "face_detection_summary.csv", index=False)
    print(summ.round(3).to_string(index=False))
    preprocessing_selftest_real(a, P, idx, cache, det)
    label_independence_report(P, face)


def preprocessing_selftest_real(a, P: Paths, idx: pd.DataFrame, cache: ClipCache, det: FaceDetector):
    """Re-extract a few cached videos per folder and compare bit-exactly (determinism), check shapes / ranges, and that the
    preprocessing entry points have no label argument."""
    rng = random.Random(a.data_seed + 1)
    rows = []
    for (ds, sub), g in idx.groupby(["dataset", "subset"]):
        ok = [int(i) for i in g.clean_idx if cache.status[int(i)] == 1]
        for i in rng.sample(ok, min(2, len(ok))):
            clip, info = extract_face_clip(idx.path[i], NUM_FRAMES, IMG_SIZE, det, FACE_MARGIN, NORM)
            cached = np.asarray(cache.arr[i])
            rows.append(dict(item_id=idx.item_id[i], dataset=ds, subset=sub, shape_ok=clip is not None and clip.shape == cached.shape,
                             dtype_ok=clip is not None and clip.dtype == np.uint8, deterministic=bool(clip is not None and np.array_equal(clip, cached)),
                             value_range=f"{int(cached.min())}-{int(cached.max())}", fallback=info.get("fallback")))
    df = pd.DataFrame(rows)
    sigs = {f.__name__: list(inspect.signature(f).parameters) for f in (extract_face_clip, crop_face, face_bottleneck, load_audio_waveform, waveform_to_spec)}
    label_free = all(not any("label" in p.lower() or p.lower() in ("y", "target", "cls") for p in ps) for ps in sigs.values())
    res = dict(created=now(), checks=df.to_dict(orient="records"), all_deterministic=bool(df.deterministic.all()) if len(df) else None,
               all_shapes_ok=bool(df.shape_ok.all()) if len(df) else None, preprocessing_signatures=sigs, label_free_signatures=label_free,
               cache_status=dict(ok=int((cache.status == 1).sum()), failed=int((cache.status == 2).sum())))
    write_json(P.data_audit / "preprocessing_selftest_real.json", res)
    print(f"  preprocessing self-test on real videos: deterministic {res['all_deterministic']} | shapes ok {res['all_shapes_ok']} | "
          f"label-free signatures {label_free}")


def label_independence_report(P: Paths, face: pd.DataFrame):
    """Diagnostic: does any nuisance property of the PREPROCESSED inputs separate the classes within a dataset?
    (The code path is label-blind; this checks whether the DATA still carries shortcuts after preprocessing.)"""
    feats = ["crop_brightness", "crop_sharpness", "detection_rate", "fallback_full_frame", "face_rel_size", "native_face_px",
             "crop_side_after_norm_px", "width", "height", "fps", "has_audio"]
    rows = []
    for ds, d in face[face.extraction_ok].groupby("dataset"):
        for f in feats:
            x = d[f].astype(float)
            ok = x.notna()
            if ok.sum() < 10 or x[ok].nunique() < 2 or d.label[ok].nunique() < 2:
                continue
            auc = roc_auc_score(d.label[ok], x[ok])
            sep = abs(auc - 0.5)
            rows.append(dict(dataset=ds, feature=f, auc_feature_vs_label=auc, separation=sep,
                             mean_real=float(x[ok & (d.label == 0)].mean()), mean_fake=float(x[ok & (d.label == 1)].mean()),
                             flag="potential shortcut (dataset property; not created by preprocessing)" if sep >= 0.3 else ""))
    df = pd.DataFrame(rows)
    df.to_csv(P.data_audit / "preprocessing_label_independence.csv", index=False)
    if len(df):
        print("  label-separating input properties after preprocessing (|AUC-0.5| >= 0.3 flagged):")
        print("    " + df[df.flag != ""][["dataset", "feature", "auc_feature_vs_label"]].round(3).to_string(index=False).replace("\n", "\n    "))


# =====================================================================================
# PHASE SPLIT (identity + content grouping, then a fresh grouped/stratified split)
# =====================================================================================
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


def make_splits(groups: Dict[str, List[int]], stratum_of: Dict[str, Tuple], seed: int, ratios=SPLIT_RATIOS) -> Dict[str, str]:
    """Grouped, stratified split: groups of each stratum are shuffled (seeded), then assigned largest-first to the split
    with the largest remaining fraction of its target size."""
    strata = defaultdict(list)
    for g in groups:
        strata[stratum_of[g]].append(g)
    rng = random.Random(seed)
    out = {}
    for key in sorted(strata, key=str):
        gs = sorted(strata[key])
        rng.shuffle(gs)
        n = sum(len(groups[g]) for g in gs)
        tv, tt = int(round(n * ratios[1])), int(round(n * ratios[2]))
        targets = {"test": tt, "val": tv, "train": n - tv - tt}
        filled = dict(test=0, val=0, train=0)
        for g in sorted(gs, key=lambda g: -len(groups[g])):
            name = max(targets, key=lambda k: (targets[k] - filled[k]) / max(1, targets[k]))
            filled[name] += len(groups[g])
            out[g] = name
    return out


def phase_split(a, P: Paths):
    print("\n" + "=" * 100 + "\n PHASE SPLIT: identity + duplicate grouping, then a NEW grouped split (seed "
          f"{a.data_seed})\n" + "=" * 100)
    inv = load_inventory(P)
    idx = read_csv(P.clean_index, "test5_manifest")
    fp = np.load(guard(P.data_audit / "fingerprints_clean.npz", "test5_fingerprints"))
    ok, degen, md5, dh, gray = fp["ok"], fp["degenerate"], fp["md5"], fp["dhash"], fp["gray"]
    has_audio, a_md5, a_dh, a_small = fp["has_audio"], fp["audio_md5"], fp["audio_dhash"], fp["audio_small"]
    orig = inv[inv.kind == "original"].set_index("item_id")
    items = idx.item_id.tolist()
    ds = idx.dataset.values
    uf = UnionFind()
    edges = []
    for i, it in enumerate(items):
        uf.find(f"v:{it}")
        if ok[i]:
            for t in str(orig.at[it, "identity_tokens"]).split("|"):
                uf.union(f"v:{it}", f"t:{t}")

    def link(i, j, reason, **kw):
        if not (ok[i] and ok[j]):
            return
        same = uf.find(f"v:{items[i]}") == uf.find(f"v:{items[j]}")
        edges.append(dict(item_a=items[i], item_b=items[j], dataset_a=ds[i], dataset_b=ds[j], label_a=int(idx.label[i]), label_b=int(idx.label[j]),
                          reason=reason, already_same_identity_group=same, **kw))
        uf.union(f"v:{items[i]}", f"v:{items[j]}")

    def dup_sets(keys: np.ndarray, valid: np.ndarray, reason: str):
        by = defaultdict(list)
        for i in np.where(valid)[0]:
            by[keys[i]].append(i)
        for k, lst in by.items():
            for j in lst[1:]:
                link(lst[0], j, reason, key=k)

    sha = idx.sha256.values
    wav_sha = idx.external_wav_sha256.fillna("").values
    dup_sets(sha, ok.copy(), "exact file SHA-256")
    dup_sets(wav_sha, ok & (wav_sha != ""), "exact external .wav SHA-256")
    dup_sets(md5, ok & ~degen, "identical decoded face clip")
    dup_sets(a_md5, ok & has_audio, "identical decoded audio waveform")
    # near-duplicate screen: video face crops and audio spectrograms
    cand_rows = []
    for kind, H, valid, thumbs in (("video", dh, ok & ~degen, gray), ("audio", a_dh, ok & has_audio, a_small)):
        print(f"  {kind} near-duplicate screen over {int(valid.sum()):,} items ...")
        pairs = near_dup_pairs(H, valid, NEARDUP_CAND_MEAN)
        print(f"    {len(pairs):,} hash candidates (mean Hamming <= {NEARDUP_CAND_MEAN}) -> confirming with NCC")
        for i, j, mh, mx in pairs:
            c = ncc(thumbs[i], thumbs[j])
            strong = mh <= NEARDUP_STRONG_MEAN and mx <= NEARDUP_STRONG_MAX and c >= NEARDUP_STRONG_NCC
            if strong:
                link(i, j, f"near-duplicate {kind} (strong)", mean_hamming=mh, max_hamming=mx, ncc=c)
            elif c >= NEARDUP_CAND_NCC:
                cand_rows.append(dict(modality=kind, item_a=items[i], item_b=items[j], dataset_a=ds[i], dataset_b=ds[j],
                                      label_a=int(idx.label[i]), label_b=int(idx.label[j]), mean_hamming=mh, max_hamming=mx, ncc=c,
                                      merged=False))
    pd.DataFrame(edges).to_csv(P.data_audit / "content_merges.csv", index=False)
    # groups
    root_of = {it: uf.find(f"v:{it}") for i, it in enumerate(items) if ok[i]}
    members = defaultdict(list)
    for i, it in enumerate(items):
        if ok[i]:
            members[root_of[it]].append(i)
    gid_of_root = {r: "G" + sha256_text("\n".join(sorted(items[i] for i in m)))[:12] for r, m in members.items()}
    groups = {gid_of_root[r]: m for r, m in members.items()}
    reasons = defaultdict(Counter)
    for e in edges:
        reasons[gid_of_root[root_of[e["item_a"]]]][e["reason"]] += 1
    stratum = {}
    grows = []
    for g, m in groups.items():
        dsets = sorted({ds[i] for i in m})
        labs = sorted({int(idx.label[i]) for i in m})
        stratum[g] = ("+".join(dsets), labs[0] if len(labs) == 1 else "mixed")
        toks = sorted({t for i in m for t in str(orig.at[items[i], "identity_tokens"]).split("|")})
        stat = sorted({orig.at[items[i], "identity_status"] for i in m})
        grows.append(dict(group_id=g, n_videos=len(m), datasets="+".join(dsets), labels="/".join(map(str, labs)),
                          subsets="+".join(sorted({idx.subset[i] for i in m})), stratum=f"{stratum[g][0]}|{stratum[g][1]}",
                          identity_tokens=";".join(toks[:20]) + (f";...(+{len(toks) - 20})" if len(toks) > 20 else ""), n_identity_tokens=len(toks),
                          identity_status="+".join(stat), merge_reasons=json.dumps(dict(reasons[g])) if reasons[g] else "identity tokens only",
                          members=";".join(items[i] for i in m[:50]) + (f";...(+{len(m) - 50})" if len(m) > 50 else "")))
    split_of_group = make_splits(groups, stratum, a.data_seed)
    gm = pd.DataFrame(grows)
    gm["split"] = gm.group_id.map(split_of_group)
    gm.sort_values(["stratum", "group_id"]).to_csv(P.group_manifest, index=False)
    cand = pd.DataFrame(cand_rows)
    if len(cand):
        gi = {it: gid_of_root[root_of[it]] for it in root_of}
        cand["group_a"], cand["group_b"] = cand.item_a.map(gi), cand.item_b.map(gi)
        cand["split_a"], cand["split_b"] = cand.group_a.map(split_of_group), cand.group_b.map(split_of_group)
        cand["same_group"] = cand.group_a == cand.group_b
        cand["cross_split"] = cand.split_a != cand.split_b
        cand["cross_dataset"] = cand.dataset_a != cand.dataset_b
    cand.to_csv(P.data_audit / "near_duplicate_candidates.csv", index=False)
    rows = []
    for i, it in enumerate(items):
        r = orig.loc[it]
        g = gid_of_root.get(root_of.get(it)) if ok[i] else ""
        rows.append(dict(item_id=it, kind="original", source_key=it, group_id=g, dataset=ds[i], subset=idx.subset[i], label=int(idx.label[i]),
                         split=split_of_group.get(g, "excluded") if ok[i] else "excluded",
                         status="used" if ok[i] else "excluded: face-clip extraction failed", degradation="", severity="", severity_level=np.nan,
                         path=idx.path[i], sha256=sha[i], clean_idx=i, deg_idx=np.nan, identity_status=r["identity_status"]))
    so = pd.DataFrame(rows)
    so.to_csv(P.split_originals, index=False)
    used = so[so.split != "excluded"]
    split_hash = sha256_text("\n".join(f"{k},{s}" for k, s in sorted(zip(used.item_id, used.split))))
    largest = gm.sort_values("n_videos", ascending=False).head(10)[["group_id", "n_videos", "stratum", "merge_reasons", "split"]]
    meta = dict(created=now(), data_seed=a.data_seed, ratios=dict(zip(("train", "val", "test"), SPLIT_RATIOS)),
                grouping_policy=dict(identity="union-find over identity tokens (BioDeepAV real: YouTube id; BioDeepAV fake: own file (no "
                                     "metadata); FF++: target+source sequence ids; DFD: " + ("individual actors" if a.dfd_grouping == "actor" else "actor pair") + ")",
                                     content="merged before splitting: identical file SHA-256, identical .wav SHA-256, identical decoded face clip, "
                                     "identical decoded audio, strong near-duplicate face crops / spectrograms (mean dHash <= "
                                     f"{NEARDUP_STRONG_MEAN}, max <= {NEARDUP_STRONG_MAX}, NCC >= {NEARDUP_STRONG_NCC})",
                                     stratification="(datasets, label or 'mixed') of each group; largest groups first",
                                     degraded="every degraded version inherits its source video's group and split (assigned after this step)"),
                n_groups=len(groups), n_originals_used=int(len(used)), n_excluded=int((so.split == "excluded").sum()),
                n_content_merges=len(edges), n_near_dup_candidates_unmerged=int(len(cand)), largest_groups=largest.to_dict(orient="records"),
                split_hash_originals=split_hash, split_originals_sha256=sha256_file(P.split_originals),
                group_manifest_sha256=sha256_file(P.group_manifest), counts=used.groupby(["dataset", "label", "split"]).size().rename("n")
                .reset_index().to_dict(orient="records"))
    write_json(P.split_meta, meta)
    print(f"  groups {len(groups):,} | content merges {len(edges):,} | unmerged near-duplicate candidates {len(cand):,} | "
          f"excluded originals {meta['n_excluded']}")
    print(used.pivot_table(index=["dataset", "subset"], columns="split", values="item_id", aggfunc="count", fill_value=0).to_string())
    print(f"  split hash (originals) {split_hash[:16]}...")


# =====================================================================================
# PHASE DEGRADED (selection of verified degraded files, aligned crops, content-link verification)
# =====================================================================================
def phase_degraded(a, P: Paths):
    print("\n" + "=" * 100 + "\n PHASE DEGRADED: verified degraded versions -> aligned face crops -> content-link test\n" + "=" * 100)
    inv = load_inventory(P)
    so = read_csv(P.split_originals, "test5_split")
    clean_idx_df, clean = load_clean(P)
    used_src = set(so[so.split != "excluded"].item_id)
    d = inv[inv.kind == "degraded"].copy()
    d["link_metadata_ok"] = as_bool(d.link_metadata_ok)
    excl = Counter()
    # exact-duplicate conflicts: degraded file byte-identical to an original or to a degraded file of another source
    o_sha = dict(zip(inv[inv.kind == "original"].sha256, inv[inv.kind == "original"].item_id))
    d_by_sha = d[d.sha256.notna()].groupby("sha256").source_key.agg(lambda s: set(s)).to_dict()
    reason = []
    for r in d.itertuples():
        why = ""
        if r.source_key not in used_src:
            why = "source excluded / not in split"
        elif not r.link_metadata_ok:
            why = "link metadata check failed"
        elif isinstance(r.sha256, str) and r.sha256 in o_sha:
            why = "byte-identical to original " + ("(its own source: no-op degradation)" if o_sha[r.sha256] == r.source_key else o_sha[r.sha256])
        elif isinstance(r.sha256, str) and len(d_by_sha.get(r.sha256, ())) > 1:
            why = "byte-identical to a degraded file of another source"
        reason.append(why)
    d["exclude_reason"] = reason
    cand = d[d.exclude_reason == ""].copy()
    # one-class guard + type matching (types present for both classes of a dataset)
    types = cand.groupby(["dataset", "label"]).degradation.agg(lambda s: set(s)).to_dict()
    shared = {ds: types.get((ds, 0), set()) & types.get((ds, 1), set()) for ds in cand.dataset.unique()}
    m = cand.apply(lambda r: r.degradation in shared.get(r.dataset, set()), axis=1) if len(cand) else pd.Series(dtype=bool)
    d.loc[cand.index[~m.values], "exclude_reason"] = "degradation type not present for both classes of this dataset"
    cand = cand[m.values]
    # byte-identical versions of the same source -> keep one
    dupkey = cand.sha256.notna() & cand.duplicated(["source_key", "sha256"], keep="first")
    d.loc[cand.index[dupkey.values], "exclude_reason"] = "duplicate version of the same source"
    cand = cand[~dupkey.values]
    # class-balanced per-source cap, distinct degradation types first (seeded)
    caps = {}
    n_cls = cand.groupby(["dataset", "label"]).source_key.nunique().to_dict()
    for (ds, lab), n in n_cls.items():
        other = n_cls.get((ds, 1 - lab), 0)
        caps[(ds, lab)] = a.deg_per_source if not other else max(1, min(a.deg_per_source, int(round(a.deg_per_source * min(n, other) / n))))
    rng = random.Random(a.data_seed + 99)
    chosen = []
    for src, g in cand.sort_values("item_id").groupby("source_key", sort=True):
        lst = list(g.index)
        cap = caps[(g.dataset.iloc[0], int(g.label.iloc[0]))]
        if len(lst) > cap:
            rng.shuffle(lst)
            pick, used_t = [], set()
            for k in lst:
                if cand.at[k, "degradation"] not in used_t:
                    pick.append(k)
                    used_t.add(cand.at[k, "degradation"])
                if len(pick) == cap:
                    break
            pick += [k for k in lst if k not in pick][:cap - len(pick)]
            for k in lst:
                if k not in pick:
                    d.at[k, "exclude_reason"] = "per-source cap (class-balanced)"
            lst = pick
        chosen += lst
    sel = d.loc[sorted(chosen)].sort_values("item_id").reset_index(drop=True)
    # hash anything not hashed yet (hash-scope originals)
    miss = sel[sel.sha256.isna()].path.tolist()
    if miss:
        meta = probe_hash_files(miss, set(miss), set(), P, a.io_workers, "selected degraded")
        sel.loc[sel.sha256.isna(), "sha256"] = [meta[p].get("sha256") for p in miss]
    sel.insert(1, "deg_idx", np.arange(len(sel)))
    ci = dict(zip(clean_idx_df.item_id, clean_idx_df.clean_idx))
    sel["source_clean_idx"] = sel.source_key.map(ci)
    sel[["item_id", "deg_idx", "source_key", "source_clean_idx", "dataset", "subset", "label", "degradation", "severity", "severity_level",
         "path", "sha256"]].to_csv(P.deg_index, index=False)
    d[d.exclude_reason != ""][["item_id", "source_key", "dataset", "label", "degradation", "severity", "path", "exclude_reason"]].to_csv(
        P.data_audit / "degraded_excluded.csv", index=False)
    print(f"  degraded selected {len(sel):,} | caps per (dataset,label): {caps} | shared types: "
          f"{ {k: sorted(v) for k, v in shared.items()} }")
    print("  exclusions: " + json.dumps(Counter(d.exclude_reason[d.exclude_reason != ""].str.split(" \\(").str[0]).most_common()))
    # aligned crops using the SOURCE video's boxes
    cache = ClipCache("facedeg", deg_cache_key(sel, clean.fname), len(sel), P)
    srcs = sel.source_clean_idx.astype(int).values
    paths = sel.path.tolist()

    def work(j):
        s = srcs[j]
        if clean.status[s] != 1:
            return None, dict(error="source clip failed")
        ref = (clean.info["boxes"][s], bool(clean.info["meta"][s, 4] > 0.5))
        return extract_face_clip(paths[j], NUM_FRAMES, IMG_SIZE, None, FACE_MARGIN, NORM, ref=ref)

    cache.build(work, a.workers, "DEGRADED VIDEOS (source face boxes)", dict(kind="degraded face clips (source boxes)", clean_cache=clean.fname))
    # content-link verification: the degraded clip must be more similar to its claimed source than to any of
    # DEG_LINK_DECOYS random other sources of the same folder from other identity groups
    fp = np.load(guard(P.data_audit / "fingerprints_clean.npz", "test5_fingerprints"))
    cgray = fp["gray"]
    grp = dict(zip(so.item_id, so.group_id))
    by_sub = defaultdict(list)
    for r in clean_idx_df.itertuples():
        if clean.status[r.clean_idx] == 1:
            by_sub[(r.dataset, r.subset)].append(r.clean_idx)
    by_sub = {k: np.array(v, np.int64) for k, v in by_sub.items()}
    cgrp = clean_idx_df.item_id.map(grp).fillna("").values
    rng2 = np.random.default_rng(a.data_seed + 7)
    rows = []
    for j, r in enumerate(sel.itertuples()):
        s = int(r.source_clean_idx)
        row = dict(item_id=r.item_id, source_key=r.source_key, dataset=r.dataset, label=r.label, degradation=r.degradation,
                   severity=r.severity, extraction_ok=bool(cache.status[j] == 1))
        if cache.status[j] == 1:
            md5, degen_d, _, g = clip_fingerprint(np.asarray(cache.arr[j]))
            sub = by_sub.get((r.dataset, r.subset), np.zeros(0, np.int64))
            pool = sub[(sub != s) & (cgrp[sub] != grp.get(r.source_key, "?"))]
            dec = rng2.choice(pool, size=min(DEG_LINK_DECOYS, len(pool)), replace=False) if len(pool) else []
            nt = ncc(g, cgray[s])
            nd = [ncc(g, cgray[k]) for k in dec]
            row.update(decoded_md5=md5, degenerate=degen_d, ncc_source=nt, max_ncc_decoy=max(nd) if nd else np.nan, n_decoys=len(nd),
                       content_link_verified=bool(len(nd) and nt > max(nd)), identical_to_source_clip=bool(md5 == fp["md5"][s]))
        else:
            row.update(decoded_md5="", degenerate=np.nan, ncc_source=np.nan, max_ncc_decoy=np.nan, n_decoys=0, content_link_verified=False,
                       identical_to_source_clip=False)
        rows.append(row)
        if j % 500 == 0:
            sys.stdout.write(f"\r  content-link test {progress_bar(j + 1, len(sel))}   ")
    sys.stdout.write("\n")
    ver = pd.DataFrame(rows)
    ver.to_csv(P.data_audit / "degraded_link_verification.csv", index=False)
    print(f"  content link verified: {int(ver.content_link_verified.sum()):,}/{len(ver):,} | extraction failed: "
          f"{int((~ver.extraction_ok).sum())} | decoded clip identical to source (no-op): {int(ver.identical_to_source_clip.sum())}")
    print(ver.groupby(["dataset", "degradation"]).content_link_verified.mean().unstack(0).round(3).to_string())
    # final split manifest
    v = ver.set_index("item_id")
    sp = dict(zip(so.item_id, so.split))
    rows = []
    for r in sel.itertuples():
        vv = v.loc[r.item_id]
        st = "used"
        if not vv.extraction_ok:
            st = "excluded: decode/crop failed"
        elif not vv.content_link_verified:
            st = "excluded: content link to source not verified"
        elif vv.identical_to_source_clip:
            st = "excluded: decoded clip identical to source (no-op degradation)"
        rows.append(dict(item_id=r.item_id, kind="degraded", source_key=r.source_key, group_id=grp.get(r.source_key, ""), dataset=r.dataset,
                         subset=r.subset, label=int(r.label), split=sp.get(r.source_key, "excluded") if st == "used" else "excluded", status=st,
                         degradation=r.degradation, severity=r.severity, severity_level=r.severity_level, path=r.path, sha256=r.sha256,
                         clean_idx=int(r.source_clean_idx), deg_idx=int(r.deg_idx), identity_status=""))
    man = pd.concat([so, pd.DataFrame(rows)], ignore_index=True)
    man.to_csv(P.split_manifest, index=False)
    u = man[man.split != "excluded"]
    print(u.pivot_table(index=["dataset", "kind"], columns="split", values="item_id", aggfunc="count", fill_value=0).to_string())


# =====================================================================================
# PHASE GATE (leakage + integrity + independence; training is impossible without PASS)
# =====================================================================================
def load_manifest(P: Paths) -> pd.DataFrame:
    if not P.split_manifest.exists():
        raise SystemExit("[STOP] split manifest missing - run --phase prepare")
    m = read_csv(P.split_manifest, "test5_split", low_memory=False)
    m["label"] = m.label.astype(int)
    return m


def phase_gate(a, P: Paths) -> bool:
    print("\n" + "=" * 100 + "\n PHASE GATE: leakage + integrity + independence checks (training needs PASS)\n" + "=" * 100)
    inv = load_inventory(P).set_index("item_id")
    man = load_manifest(P)
    used = man[man.split.isin(["train", "val", "test"])].copy()
    uo, ud = used[used.kind == "original"], used[used.kind == "degraded"]
    R = dict(created=now(), script_sha256=code_sha256())
    # 1. inventory integrity (size + mtime unchanged since the audit; optional full re-hash)
    bad_int, checked = [], 0
    for r in used.itertuples():
        p = r.path
        try:
            st = os.stat(p)
            iv = inv.loc[r.item_id]
            ok = st.st_size == int(iv.file_size) and str(st.st_mtime_ns) == str(iv.mtime_ns)
            if ok and a.gate_rehash:
                ok = sha256_file(p) == r.sha256
            if not ok:
                bad_int.append(r.item_id)
        except Exception:
            bad_int.append(r.item_id)
        checked += 1
    for r in inv[(inv.kind == "original") & inv.index.isin(uo.item_id) & inv.external_wav.fillna("").ne("")].itertuples():
        try:
            st = os.stat(r.external_wav)
            if st.st_size != int(r.external_wav_size) or str(st.st_mtime_ns) != str(r.external_wav_mtime_ns):
                bad_int.append(r.Index + "::wav")
        except Exception:
            bad_int.append(r.Index + "::wav")
    so_meta = read_json(P.split_meta, "test5_split")
    split_file_unchanged = sha256_file(P.split_originals) == so_meta["split_originals_sha256"]
    R["inventory_integrity"] = dict(files_checked=checked, changed_or_missing=len(bad_int), examples=bad_int[:20], full_rehash=a.gate_rehash,
                                    used_files_without_sha256=int(used.sha256.isna().sum()), split_originals_unchanged=split_file_unchanged,
                                    passed=not bad_int and not used.sha256.isna().any() and split_file_unchanged)
    # 2. group / identity isolation
    gsplit = uo.groupby("group_id").split.nunique()
    tok_split = defaultdict(set)
    for r in uo.itertuples():
        for t in str(inv.at[r.item_id, "identity_tokens"]).split("|"):
            tok_split[t].add(r.split)
    tok_cross = [t for t, s in tok_split.items() if len(s) > 1]
    R["group_isolation"] = dict(groups=int(len(gsplit)), groups_in_multiple_splits=int((gsplit > 1).sum()),
                                identity_tokens=len(tok_split), identity_tokens_in_multiple_splits=len(tok_cross), examples=tok_cross[:10])
    # 3. original / degraded grouping
    src_split = dict(zip(uo.item_id, uo.split))
    src_group = dict(zip(uo.item_id, uo.group_id))
    ver = read_csv(P.data_audit / "degraded_link_verification.csv", "test5_audit").set_index("item_id")
    pair_bad = int(sum(src_split.get(r.source_key) != r.split for r in ud.itertuples()))
    grp_bad = int(sum(src_group.get(r.source_key) != r.group_id for r in ud.itertuples()))
    unverified = int((~as_bool(ver.loc[ud.item_id, "content_link_verified"])).sum()) if len(ud) else 0
    meta_bad = int((~as_bool(inv.loc[ud.item_id, "link_metadata_ok"])).sum()) if len(ud) else 0
    R["original_degraded_grouping"] = dict(degraded_used=int(len(ud)), split_mismatch=pair_bad, group_mismatch=grp_bad,
                                           used_without_content_link=unverified, used_without_metadata_link=meta_bad,
                                           sources_with_degraded=int(ud.source_key.nunique()))
    # 4. exact file duplicates across splits (originals + degraded + wavs)
    files = [(r.item_id, r.split, r.sha256, r.dataset) for r in used.itertuples()]
    files += [(r.item_id + "::wav", r.split, inv.at[r.item_id, "external_wav_sha256"], r.dataset) for r in uo.itertuples()
              if isinstance(inv.at[r.item_id, "external_wav_sha256"], str)]
    fd = pd.DataFrame(files, columns=["item_id", "split", "sha256", "dataset"])
    dsets = fd[fd.duplicated("sha256", keep=False)].groupby("sha256")
    exact_cross, exact_within, exact_xds = [], 0, 0
    for h, g in dsets:
        if g.split.nunique() > 1:
            exact_cross.append(dict(sha256=h, items=g.item_id.tolist(), splits=g.split.tolist(),
                                    investigation="byte-identical files (full SHA-256) in different splits -> confirmed leakage"))
        else:
            exact_within += 1
        exact_xds += int(g.dataset.nunique() > 1)
    # 5. decoded content duplicates across splits (clean + degraded clips)
    fp = np.load(guard(P.data_audit / "fingerprints_clean.npz", "test5_fingerprints"))
    cm = [(r.item_id, r.split, fp["md5"][int(r.clean_idx)], bool(fp["degenerate"][int(r.clean_idx)])) for r in uo.itertuples()]
    cm += [(r.item_id, r.split, ver.at[r.item_id, "decoded_md5"], as_bool(pd.Series([ver.at[r.item_id, "degenerate"]]))[0]) for r in ud.itertuples()]
    cd = pd.DataFrame(cm, columns=["item_id", "split", "md5", "degenerate"])
    dec_cross, dec_degen, dec_within = [], [], 0
    for h, g in cd[cd.duplicated("md5", keep=False)].groupby("md5"):
        if g.split.nunique() > 1:
            (dec_degen if g.degenerate.any() else dec_cross).append(dict(md5=h, items=g.item_id.tolist(), splits=g.split.tolist()))
        else:
            dec_within += 1
    # 6. near-duplicates
    nd_path = P.data_audit / "near_duplicate_candidates.csv"
    nd = read_csv(nd_path, "test5_audit") if nd_path.exists() and nd_path.stat().st_size > 2 else pd.DataFrame()
    nd_cross = nd[as_bool(nd.cross_split)] if len(nd) else nd
    merges = read_csv(P.data_audit / "content_merges.csv", "test5_audit") if (P.data_audit / "content_merges.csv").stat().st_size > 2 else pd.DataFrame()
    strong_cross = 0
    if len(merges):
        gi = dict(zip(uo.item_id, uo.split))
        strong_cross = int(sum(gi.get(x) is not None and gi.get(y) is not None and gi.get(x) != gi.get(y) for x, y in zip(merges.item_a, merges.item_b)))
    R["duplicates"] = dict(
        exact_file_duplicate_sets_across_splits=len(exact_cross), exact_file_duplicate_sets_within_split=exact_within,
        exact_file_duplicate_sets_across_datasets=exact_xds, exact_cross_split_details=exact_cross[:50],
        decoded_duplicate_sets_across_splits=len(dec_cross), decoded_duplicate_sets_within_split=dec_within,
        decoded_degenerate_duplicates_across_splits_unresolved=len(dec_degen), decoded_cross_split_details=dec_cross[:50],
        content_merges_before_split=int(len(merges)), merged_pairs_ending_in_different_splits=strong_cross,
        near_duplicate_candidates_unmerged=int(len(nd)), near_duplicate_candidates_cross_split_unresolved=int(len(nd_cross)),
        near_duplicate_candidates_cross_dataset=int(as_bool(nd.cross_dataset).sum()) if len(nd) else 0,
        near_duplicate_note=("Unmerged candidates have high dHash similarity and NCC >= {:.2f} but miss the strong criterion; they are "
                             "documented for manual investigation (data_audit/near_duplicate_candidates.csv), not counted as confirmed "
                             "leakage. Fingerprint similarity alone is not proof of a shared source.").format(NEARDUP_CAND_NCC),
        method_limits="Exact screens find byte-identical files and identical decoded clips; the dHash/NCC screen finds re-encoded copies "
                      "with nearly identical framing; it does NOT detect the same person filmed in different videos.")
    # 7. independence: no forbidden artifact loads, cache provenance
    acc = read_csv(P.access_log, "test5_access_log") if P.access_log.exists() else pd.DataFrame()
    blocked = acc[as_bool(acc.blocked)] if len(acc) else acc
    forbidden_reads = acc[as_bool(acc.forbidden_root)] if len(acc) else acc
    prov = read_json(P.provenance, "test5_cache_provenance") if P.provenance.exists() else {}
    known = set()
    for v in prov.values():
        f = v["fname"]
        known |= {f + s for s in (".npy", ".status.npy", ".info.npz")}
        known |= {v.get(k) for k in ("spec_file", "wave_file", "status_file") if v.get(k)}
        known.add(f + ".source.json")
    on_disk = [q.name for q in P.cache.iterdir() if q.is_file() and q.name != "provenance.json"]
    unknown = [n for n in on_disk if n not in known]
    idx_clean = read_csv(P.clean_index, "test5_manifest")
    dsel = read_csv(P.deg_index, "test5_manifest")
    exp_keys = dict(faceclean=sha256_text(clean_cache_key(idx_clean)),
                    facedeg=sha256_text(deg_cache_key(dsel, ClipCache("faceclean", clean_cache_key(idx_clean), len(idx_clean), P).fname)))
    key_ok = {k: prov.get(k, {}).get("key_src_sha256") == v for k, v in exp_keys.items()}
    links = []
    for q in P.cache.iterdir():
        if q.is_symlink() or (q.is_file() and q.stat().st_nlink > 1):
            links.append(q.name)
    other_caches = [c for r in FORBIDDEN_ARTIFACT_ROOTS for c in ([r / "cache"] if (r / "cache").exists() else [])]
    same_inode = []
    for oc in other_caches:
        try:
            ino = {(q.stat().st_dev, q.stat().st_ino) for q in oc.iterdir() if q.is_file()}
        except Exception:
            ino = set()
        same_inode += [q.name for q in P.cache.iterdir() if q.is_file() and (q.stat().st_dev, q.stat().st_ino) in ino]
    R["independence"] = dict(
        artifact_reads_logged=int(len(acc)), blocked_forbidden_reads=int(len(blocked)),
        historical_readonly_reads=int(len(forbidden_reads) - len(blocked)),
        historical_readonly_files=sorted(set(forbidden_reads.path)) if len(forbidden_reads) else [],
        cache_files_on_disk=len(on_disk), cache_files_without_provenance=unknown, cache_keys_match_current_data=key_ok,
        cache_files_linked_or_hardlinked=links, cache_files_same_inode_as_other_runs=same_inode,
        test4_imported=False, note="Test 5 never searches other run directories for caches; all caches carry the t5_ namespace, "
                                   "a provenance record and a key derived from this run's inventory hashes.")
    R["provenance_passed"] = bool(not unknown and all(key_ok.values()) and not links and not same_inode)
    # 8. split integrity
    counts = uo.groupby(["dataset", "split", "label"]).size().unstack("label", fill_value=0)
    short = []
    for (dn, sp), row in counts.iterrows():
        for lab in (0, 1):
            if sp in ("val", "test") and row.get(lab, 0) < a.min_per_class:
                short.append(f"{dn}/{sp}/{'real' if lab == 0 else 'fake'}={row.get(lab, 0)}")
    ratios = uo.groupby("dataset").split.value_counts(normalize=True).rename("share").reset_index()
    SI = dict(created=now(), data_seed=a.data_seed, split_hash_originals=so_meta["split_hash_originals"],
              split_manifest_sha256=sha256_file(P.split_manifest), group_manifest_sha256=sha256_file(P.group_manifest),
              counts=uo.groupby(["dataset", "subset", "split"]).size().rename("originals").reset_index().to_dict(orient="records"),
              degraded_counts=ud.groupby(["dataset", "split", "degradation"]).size().rename("degraded").reset_index().to_dict(orient="records"),
              split_shares=ratios.to_dict(orient="records"), classes_below_minimum=short, min_per_class_required=a.min_per_class,
              excluded=man[man.split == "excluded"].status.value_counts().to_dict(),
              identity_status=uo.groupby(["dataset", "identity_status"]).size().rename("n").reset_index().to_dict(orient="records"),
              group_isolation=R["group_isolation"], original_degraded_grouping=R["original_degraded_grouping"],
              test4_split_overlap=test4_split_overlap(P, uo))
    write_json(P.split_integrity, SI)
    confirmed = (R["group_isolation"]["groups_in_multiple_splits"] + R["group_isolation"]["identity_tokens_in_multiple_splits"]
                 + pair_bad + grp_bad + len(exact_cross) + len(dec_cross) + strong_cross)
    R["confirmed_cross_split_leakage_count"] = int(confirmed)
    checks = {
        "source inventory integrity verified": R["inventory_integrity"]["passed"],
        "source-group split isolation verified": R["group_isolation"]["groups_in_multiple_splits"] == 0
        and R["group_isolation"]["identity_tokens_in_multiple_splits"] == 0,
        "original/degraded grouping verified": pair_bad == 0 and grp_bad == 0 and unverified == 0 and meta_bad == 0,
        "exact cross-split duplicates investigated (none confirmed)": len(exact_cross) == 0 and len(dec_cross) == 0,
        "confirmed cross-split source leakage == 0": confirmed == 0,
        "unresolved duplicates documented": (P.data_audit / "near_duplicate_candidates.csv").exists(),
        "no Test 4 / old Test 5 caches or checkpoints loaded": len(blocked) == 0,
        "fresh Test 5 cache provenance verified": R["provenance_passed"],
        "every dataset has >= {} real and fake videos in val and test".format(a.min_per_class): not short,
    }
    R["checks"] = checks
    R["gate"] = "PASS" if all(checks.values()) else "FAIL"
    write_json(P.leakage, R)
    if R["gate"] == "PASS":
        build_xai_pairs(a, P, man)
        write_experiment_configs(a, P, SI)
        write_test4_reference(P)
    status = dict(gate=R["gate"], checked_at=now(), checks=checks, split_manifest_sha256=SI["split_manifest_sha256"],
                  split_hash_originals=SI["split_hash_originals"], script_sha256=code_sha256(),
                  xai_pairs_sha256=sha256_file(P.xai_pairs) if P.xai_pairs.exists() else None)
    write_json(P.gate, status)
    write_prepare_report(a, P, R, SI)
    print("\n  LEAKAGE GATE CHECKS")
    for k, v in checks.items():
        print(f"    [{'PASS' if v else 'FAIL'}] {k}")
    print(f"  confirmed cross-split leakage = {confirmed} | unresolved near-duplicate candidates across splits = "
          f"{R['duplicates']['near_duplicate_candidates_cross_split_unresolved']} | GATE: {R['gate']}")
    if R["gate"] != "PASS":
        print(f"\n[STOP] Leakage gate FAILED - diagnostic report: {P.leakage}. Training will not start.")
    return R["gate"] == "PASS"


def test4_split_overlap(P: Paths, uo: pd.DataFrame) -> dict:
    """Documentation only (read-only): how much the new split differs from Test 4's. Never used for any decision."""
    out = {}
    for f in sorted(TEST4_DIR.glob("4D_train-*/*split_6-10-26.json")):
        try:
            t4 = read_json(f, "historical_reference_readonly:split_overlap_documentation")
        except Exception:
            continue
        t5 = dict(zip(uo.item_id, uo.split))
        res = {}
        for k in ("train", "val", "test"):
            keys = set(t4.get(k, []))
            res[f"test4_{k}"] = dict(n=len(keys), **{f"in_test5_{s}": int(sum(t5.get(x) == s for x in keys)) for s in ("train", "val", "test")})
        out[f.parent.name] = res
    return dict(note="Test 4 split read read-only for documentation; not used for any Test 5 decision.", files=out)


def build_xai_pairs(a, P: Paths, man: pd.DataFrame):
    """Pre-registered (before any training) type-stratified, class-balanced sample of test original/degraded pairs."""
    if P.xai_pairs.exists():
        return
    t = man[(man.kind == "degraded") & (man.split == "test")].sort_values("item_id")
    rng = np.random.default_rng(a.data_seed + 555)
    rows = []
    for (ds, dg), g in t.groupby(["dataset", "degradation"]):
        per = []
        for lab in (0, 1):
            gl = g[g.label == lab]
            per.append(list(gl.item_id.values[rng.permutation(len(gl))]))
        pick, k = [], 0
        while len(pick) < a.xai_per_type and (per[0] or per[1]):
            lst = per[k % 2] or per[(k + 1) % 2]
            pick.append(lst.pop(0))
            k += 1
        rows += [dict(pair_id=f"{ds}|{dg}|{n}", deg_item_id=x) for n, x in enumerate(pick)]
    xp = pd.DataFrame(rows).merge(t[["item_id", "source_key", "group_id", "dataset", "label", "degradation", "severity"]],
                                  left_on="deg_item_id", right_on="item_id").drop(columns="item_id")
    xp.to_csv(P.xai_pairs, index=False)
    print(f"  Grad-CAM evaluation pairs (pre-registered): {len(xp)} -> {P.xai_pairs.name}")


def write_experiment_configs(a, P: Paths, SI: dict):
    pc = read_json(P.configs / "preprocessing_config.json", "test5_config")
    m = AudioVisualDeepFakeCNN()
    for v in VARIANTS:
        c = variant_config(a, v)
        cfg = dict(
            test="Test 5 (independent)", variant=v, description=c["description"], created=now(),
            research_question="Does hard-degradation-aware training improve detection robustness and Grad-CAM explanation stability?",
            stage1=dict(data="clean training originals", epochs=a.epochs1, lr=a.lr1, sampling="class-balanced (oversampling cap "
                        f"{MAX_OVERSAMPLE}x)", shared_across_variants=not a.no_share_stage1, audio_policy=f"mixed {AUDIO_PROBS}"),
            stage2=dict(init="best Stage-1 checkpoint of the same (training set, seed)", epochs=a.epochs2, lr=a.lr2, paired=c["paired"],
                        hard_sampling=c["hard"], hard_share=a.hard_share if c["hard"] else 0.0, hard_weights=a.hard_w if c["hard"] else {},
                        lambda_feat=c["lambda_feat"], lambda_cam=c["lambda_cam"], consistency_anchor="clean (stop-gradient)",
                        curriculum=dict(enabled=c["curriculum"], phase_end_epochs=a.curriculum,
                                        rule="epochs 1-e1 mild, e1+1-e2 +medium, later all severities; if no allowed version exists the "
                                             "mildest version of the source is used"),
                        clip_mix="each step: 32 clips = 16 original + 16 degraded (unpaired 5A/5B) or 16 original/degraded pairs (5C/5D)",
                        degradation_mixture="natural: uniform over the source's verified versions; hard: with probability hard_share a "
                                            "hard-type version drawn from the same (dataset, class) cell",
                        audio_policy=f"mixed clean/degraded/none {AUDIO_PROBS}", class_balance="class-balanced source draws"),
            optimiser=dict(name="AdamW", weight_decay=WEIGHT_DECAY, scheduler="OneCycleLR (max lr = stage lr)", batch_clips=a.batch_size,
                           grad_clip=None),
            model=dict(name="AudioVisualDeepFakeCNN", parameters=int(sum(p.numel() for p in m.parameters())),
                       architecture_signature=architecture_signature(m), architecture_source_sha256=architecture_source_sha(),
                       init="PyTorch default initialisation, seeded by the training seed (no pretrained / Test 4 weights)"),
            seeds=a.seed_list, train_sets=a.train_list, model_selection="best epoch by validation score = mean(val original AUC, val "
            "degraded AUC) of the training dataset (Stage 1: val original AUC); threshold = validation balanced-accuracy optimum",
            evaluation=dict(test="untouched Test 5 test split of BOTH datasets (in-domain + cross-dataset)", gradcam_pairs=str(P.xai_pairs),
                            bootstrap=f"cluster (source group) bootstrap, {a.n_boot} replicates"),
            data=dict(split_manifest=str(P.split_manifest), split_manifest_sha256=SI["split_manifest_sha256"],
                      split_hash_originals=SI["split_hash_originals"], data_seed=a.data_seed),
            preprocessing=pc, unavoidable_differences=["5C introduces original/degraded pairing because Grad-CAM consistency needs "
                                                       "pairs; 5C vs 5B therefore measures pairing + Grad-CAM consistency together",
                                                       "5D adds feature consistency AND the curriculum; 5D vs 5C measures both together"],
            environment=environment_info())
        write_json(P.configs / f"experiment_config_{v}.json", cfg)


def write_test4_reference(P: Paths):
    """Read-only documentation of Test 4's REPORTED numbers (different split; never a Test 5 input)."""
    ref = dict(note="Historical Test 4 results as reported by Test 4 on ITS OWN split (seed 42). Test 5 uses a newly generated "
                    "split, so any comparison is across different evaluation splits, not a controlled paired comparison.",
               test4_script_sha256=sha256_file(TEST4_SCRIPT) if TEST4_SCRIPT.exists() else None, runs={})
    sc = TEST4_DIR / "suite_comparison.csv"
    if sc.exists():
        ref["suite_comparison"] = read_csv(sc, "historical_reference_readonly:test4_suite_comparison").to_dict(orient="records")
    for f in sorted(TEST4_DIR.glob("4D_train-*/results/*final_experiment_summary.json")):
        j = read_json(f, "historical_reference_readonly:test4_summary")
        g = j.get("gradcam_stability", {}).get("stage2", {})
        ref["runs"][f.parent.parent.name] = dict(
            stage2_metrics={k: {m: v.get(m) for m in ("auc", "balanced_accuracy", "f1", "n")} for k, v in j.get("stage2", {}).get("metrics", {}).items()},
            gradcam_stage2={k: g.get(k) for k in ("n", "mean_pearson_r", "mean_top20_iou", "mean_center_shift", "mean_temporal_cam_r",
                                                   "prediction_stable_rate", "unstable_explanation_given_stable_prediction", "per_degradation")},
            leakage_check=j.get("leakage_check"))
    write_json(P.data_audit / "test4_historical_reference.json", ref)


def write_prepare_report(a, P: Paths, R: dict, SI: dict):
    inv_s = read_json(P.data_audit / "inventory_summary.json", "test5_audit")
    st = read_json(P.data_audit / "preprocessing_selftest_real.json", "test5_audit") if (P.data_audit / "preprocessing_selftest_real.json").exists() else {}
    sm = read_json(P.split_meta, "test5_split")
    li = read_csv(P.data_audit / "preprocessing_label_independence.csv", "test5_audit") if (P.data_audit / "preprocessing_label_independence.csv").exists() else pd.DataFrame()
    L = [f"# Test 5 (independent) - pre-training checks\n", f"_Generated {now()} by {SCRIPT_PATH.name} (sha256 {code_sha256()[:16]}...)_\n",
         f"**Leakage gate: {R['gate']}** - confirmed cross-split leakage = {R['confirmed_cross_split_leakage_count']}\n", "## Gate checks\n",
         md_table(pd.DataFrame([dict(check=k, result="PASS" if v else "FAIL") for k, v in R["checks"].items()])), "\n## Source inventory\n",
         f"Originals {inv_s['n_originals']:,}; manifest degraded outputs {inv_s['n_degraded_listed']:,} ({inv_s['n_degraded_link_ok']:,} with valid "
         f"link metadata); orphan degraded files (not in a manifest, unused) {inv_s['n_orphans']:,}; exact duplicate files "
         f"{inv_s['exact_duplicate_files']} in {inv_s['exact_duplicate_sets']} sets; hash scope `{inv_s['hash_scope']}`.\n",
         md_table(pd.DataFrame(inv_s["per_folder"])), "\n### Identity metadata\n", md_table(pd.DataFrame(inv_s["identity_status"])),
         "\n" + "\n".join(f"- {n}" for n in inv_s["notes"]) + "\n", "## Preprocessing self-tests\n",
         f"Deterministic re-extraction: {st.get('all_deterministic')}; shapes ok: {st.get('all_shapes_ok')}; label-free signatures: "
         f"{st.get('label_free_signatures')}; clean clips ok/failed: {st.get('cache_status')}.\n",
         "Input properties that still separate the classes after preprocessing (dataset properties, potential shortcuts):\n",
         md_table(li[li.flag.fillna("") != ""][["dataset", "feature", "auc_feature_vs_label", "mean_real", "mean_fake"]] if len(li) else li),
         "\n## Grouping and split\n", f"Seed {sm['data_seed']}; {sm['n_groups']:,} groups; {sm['n_content_merges']} content merges before "
         f"splitting; {sm['n_excluded']} originals excluded; split hash `{sm['split_hash_originals'][:16]}...`.\n",
         md_table(pd.DataFrame(SI["counts"]).pivot_table(index=["dataset", "subset"], columns="split", values="originals", fill_value=0).reset_index()),
         "\nLargest groups:\n", md_table(pd.DataFrame(sm["largest_groups"])), "\n## Leakage audit\n",
         "```json\n" + json.dumps(to_jsonable({k: R[k] for k in ("group_isolation", "original_degraded_grouping")}), indent=1) + "\n```\n",
         "```json\n" + json.dumps(to_jsonable({k: v for k, v in R["duplicates"].items() if not k.endswith("details")}), indent=1) + "\n```\n",
         "```json\n" + json.dumps(to_jsonable({k: v for k, v in R["independence"].items() if k != "historical_readonly_files"}), indent=1) + "\n```\n",
         "## Next step\n", "Training has NOT been started. After reviewing this report run `--phase all`." if R["gate"] == "PASS"
         else "The gate FAILED: fix the reported problems; training is blocked."]
    (P.reports / "PRE_TRAINING_CHECKS.md").write_text("\n".join(L), encoding="utf-8")


def require_gate(P: Paths):
    if not P.gate.exists():
        raise SystemExit("[STOP] no gate result - run --phase prepare first")
    g = read_json(P.gate, "test5_gate")
    if g.get("gate") != "PASS":
        raise SystemExit(f"[STOP] leakage gate status is {g.get('gate')} - training/evaluation is blocked (see {P.leakage})")
    if sha256_file(P.split_manifest) != g["split_manifest_sha256"]:
        raise SystemExit("[STOP] split manifest changed after the gate - re-run --phase gate")
    if P.xai_pairs.exists() and g.get("xai_pairs_sha256") and sha256_file(P.xai_pairs) != g["xai_pairs_sha256"]:
        raise SystemExit("[STOP] pre-registered Grad-CAM pairs changed after the gate")
    return g


# =====================================================================================
# RUNTIME DATA
# =====================================================================================
class T5Data:
    def __init__(self, P: Paths):
        self.P = P
        man = load_manifest(P)
        o = man[(man.kind == "original") & man.split.isin(["train", "val", "test"])].reset_index(drop=True)
        d = man[(man.kind == "degraded") & man.split.isin(["train", "val", "test"])].reset_index(drop=True)
        self.o, self.d = o, d
        self.key = o.item_id.values
        self.labels = o.label.values.astype(np.int64)
        self.ds = o.dataset.values
        self.split = o.split.values
        self.group = o.group_id.values
        self.cidx = o.clean_idx.values.astype(np.int64)
        pos = {k: i for i, k in enumerate(self.key)}
        self.deg_src = d.source_key.map(pos).values.astype(np.int64)
        self.deg_type = d.degradation.values
        self.deg_sev = d.severity.values
        self.deg_level = d.severity_level.values.astype(np.int64)
        self.didx = d.deg_idx.values.astype(np.int64)
        self.versions = defaultdict(list)
        for j, s in enumerate(self.deg_src):
            self.versions[int(s)].append(j)
        self.versions = {k: np.array(v, np.int64) for k, v in self.versions.items()}
        idx_clean, self.clean = load_clean(P)
        dsel = read_csv(P.deg_index, "test5_manifest")
        dc = ClipCache("facedeg", deg_cache_key(dsel, self.clean.fname), len(dsel), P)
        if not dc.complete():
            raise SystemExit("[STOP] degraded cache incomplete")
        self.degc = dc.load()
        self.spec, self.wave, self.has_audio_all, _ = build_audio_cache(P, idx_clean, 1)
        self.has_audio = self.has_audio_all[self.cidx]

    def idx(self, split: str, dataset: Optional[str] = None) -> np.ndarray:
        m = self.split == split
        if dataset:
            m &= self.ds == dataset
        return np.where(m)[0]

    def deg_items(self, src_idx: np.ndarray) -> np.ndarray:
        return np.concatenate([self.versions.get(int(s), np.zeros(0, np.int64)) for s in src_idx]) if len(src_idx) else np.zeros(0, np.int64)

    def clip(self, kind: str, i: int, sel=None) -> np.ndarray:
        if kind == "deg":
            return np.asarray(self.degc.arr[self.didx[i]] if sel is None else self.degc.arr[self.didx[i], sel])
        return np.asarray(self.clean.arr[self.cidx[i]] if sel is None else self.clean.arr[self.cidx[i], sel])

    def src_of(self, kind: str, i: int) -> int:
        return int(self.deg_src[i]) if kind == "deg" else int(i)

    def audio(self, src: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return np.array(self.spec[self.cidx[src]], dtype=np.float32), self.has_audio[src].astype(np.float32)

    def wave_float(self, s: int) -> np.ndarray:
        return np.asarray(self.wave[self.cidx[s]], dtype=np.float32) / 32767.0


# =====================================================================================
# TRAINING
# =====================================================================================
def set_seed(seed: int, deterministic: bool):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.benchmark = True


def gpu_temp() -> Optional[int]:
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                                      creationflags=NO_WINDOW, timeout=2)
        return int(out.decode().split()[0])
    except Exception:
        return None


def train_tag(t: str) -> str:
    return "train-" + t.replace("+", "p")


def run_id(train_on: str, seed: int, variant: str) -> str:
    return f"{train_tag(train_on)}_seed{seed}_{variant}"


class Trainer:
    def __init__(self, a, P: Paths, D: T5Data, train_on: str, seed: int):
        self.a, self.P, self.D, self.train_on, self.seed = a, P, D, train_on, seed
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.T, self.L = NUM_FRAMES, CLIP_LEN
        self.stride = max(1, self.T // self.L)
        self.tr = D.idx("train", train_on)
        self.va = D.idx("val", train_on)
        self.cell = np.array([f"{d}|{'real' if l == 0 else 'fake'}" for d, l in zip(D.ds, D.labels)])
        self.tr_deg_src = np.array([s for s in self.tr if int(s) in D.versions], np.int64)
        self.n_tr_deg = int(sum(len(D.versions[int(s)]) for s in self.tr_deg_src))
        self.hard_pool = defaultdict(list)
        for s in self.tr_deg_src:
            for j in D.versions[int(s)]:
                self.hard_pool[(self.cell[s], D.deg_type[j])].append(int(j))
        self.hard_pool = {k: np.array(sorted(v), np.int64) for k, v in self.hard_pool.items()}
        self.ckdir = P.checkpoints / train_tag(train_on) / f"seed{seed}"
        self.ckdir.mkdir(parents=True, exist_ok=True)
        self.split_hash = read_json(P.gate, "test5_gate")["split_hash_originals"]

    # ---------------------------------------------------------------- sampling
    def weights(self, src: np.ndarray, epoch_size: int) -> np.ndarray:
        """Class-balanced (dataset x class cell) source probabilities with an oversampling cap."""
        cells = self.cell[src]
        names, counts = np.unique(cells, return_counts=True)
        share = {c: 1.0 / len(names) for c in names}
        cap = {c: MAX_OVERSAMPLE * n / epoch_size for c, n in zip(names, counts)}
        for _ in range(10):
            over = {c for c in names if share[c] > cap[c] + 1e-12}
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

    def choose_deg(self, s: int, rng, hard: bool, max_level: int) -> Tuple[int, int, str]:
        D = self.D
        if hard and rng.random() < self.a.hard_share:
            types = [t for t in self.a.hard_w if len(self.hard_pool.get((self.cell[s], t), ()))]
            if types:
                w = np.array([self.a.hard_w[t] for t in types])
                t = types[int(rng.choice(len(types), p=w / w.sum()))]
                pool = self.hard_pool[(self.cell[s], t)]
                lv = D.deg_level[pool]
                cand = pool[lv <= max_level]
                if not len(cand):
                    cand = pool[lv == lv.min()]
                j = int(cand[int(rng.integers(len(cand)))])
                return int(D.deg_src[j]), j, "hard"
        v = D.versions[int(s)]
        lv = D.deg_level[v]
        cand = v[lv <= max_level]
        if not len(cand):
            cand = v[lv == lv.min()]
        return int(s), int(cand[int(rng.integers(len(cand)))]), "natural"

    def sample_audio(self, src: np.ndarray, rng, max_level: int):
        spec, has = self.D.audio(src)
        u = rng.random(len(src))
        pc, pdg, _ = AUDIO_PROBS
        keep = (has > 0) & (u < pc + pdg)
        deg = keep & (u >= pc)
        for b in np.where(deg)[0]:
            d = AUDIO_DEGRADATIONS[int(rng.integers(len(AUDIO_DEGRADATIONS)))]
            lvl = 1 if d == "silence" else int(rng.integers(max_level + 1))
            spec[b] = waveform_to_spec(degrade_audio(self.D.wave_float(int(src[b])), d, lvl, rng))
        return spec, keep.astype(np.float32)

    def build(self, items: List[Tuple[str, int]], seed: int, max_level: int, paired: bool):
        """items: [(kind, idx)] (unpaired) or [(src, deg_j)] (paired: same frames / flip / jitter for both members)."""
        rng = np.random.default_rng(seed)
        S, D = IMG_SIZE, self.D
        pad = max(2, S // 24)
        B = len(items)
        a_ = np.empty((B, self.L, S, S, 3), np.uint8)
        b_ = np.empty((B, self.L, S, S, 3), np.uint8) if paired else None
        src = np.empty(B, np.int64)
        for k, it in enumerate(items):
            sel = np.sort(rng.choice(self.T, self.L, replace=False))
            flip = rng.random() < 0.5
            dy, dx = rng.integers(0, 2 * pad + 1, 2)
            if paired:
                s, j = it
                clips = [(a_, D.clip("clean", s, sel)), (b_, D.clip("deg", j, sel))]
                src[k] = s
            else:
                kind, i = it
                clips = [(a_, D.clip(kind, i, sel))]
                src[k] = D.src_of(kind, i)
            for out, c in clips:
                if flip:
                    c = c[:, :, ::-1, :]
                pp = np.pad(c, ((0, 0), (pad, pad), (pad, pad), (0, 0)), mode="edge")
                out[k] = pp[:, dy:dy + S, dx:dx + S]
        audio, mask = self.sample_audio(src, rng, max_level)
        return a_, b_, audio, mask, D.labels[src]

    def plan_epoch(self, rng, stage: int, vc: dict, max_level: int, steps: int, bs: int):
        """Deterministic per-epoch plan (training seed). Returns list of step item lists + realised sampling log."""
        D = self.D
        plan, log = [], Counter()
        if stage == 1:
            w = self.weights(self.tr, steps * bs)
            draws = rng.choice(self.tr, size=steps * bs, replace=True, p=w)
            for st in range(steps):
                its = [("clean", int(s)) for s in draws[st * bs:(st + 1) * bs]]
                plan.append(its)
                log.update((self.cell[s], "clean", "-") for _, s in its)
            return plan, log
        wd = self.weights(self.tr_deg_src, steps * bs)
        if vc["paired"]:
            n = bs // 2
            for st in range(steps):
                its = []
                for s in rng.choice(self.tr_deg_src, size=n, replace=True, p=wd):
                    s2, j, mode = self.choose_deg(int(s), rng, vc["hard"], max_level)
                    its.append((s2, j))
                    log[(self.cell[s2], mode, D.deg_type[j])] += 1
                plan.append(its)
        else:
            wc = self.weights(self.tr, steps * bs)
            nc = bs // 2
            for st in range(steps):
                its = [("clean", int(s)) for s in rng.choice(self.tr, size=nc, replace=True, p=wc)]
                for s in rng.choice(self.tr_deg_src, size=bs - nc, replace=True, p=wd):
                    s2, j, mode = self.choose_deg(int(s), rng, vc["hard"], max_level)
                    its.append(("deg", j))
                    log[(self.cell[s2], mode, D.deg_type[j])] += 1
                log.update((self.cell[s], "clean", "-") for _, s in its[:nc])
                plan.append(its)
        return plan, log

    # ---------------------------------------------------------------- prediction
    @torch.no_grad()
    def predict(self, model, items: List[Tuple[str, int]], cond: str = "normal", bs: int = 32) -> np.ndarray:
        model.eval()
        out = []
        for i in range(0, len(items), bs):
            ch = items[i:i + bs]
            clips = np.stack([self.D.clip(k, j) for k, j in ch])
            src = np.array([self.D.src_of(k, j) for k, j in ch])
            spec, has = self.D.audio(src)
            xa = torch.from_numpy(spec).to(self.device)
            am = torch.from_numpy(np.zeros_like(has) if cond == "audio_masked" else has).to(self.device)
            mode = "a_only" if cond == "audio_only" else "av"
            p = 0
            for v in range(self.stride):
                xv = to_video_tensor(clips[:, v::self.stride][:, :self.L], self.device)
                p = p + torch.softmax(model(xv, xa, mode=mode, a_mask=am).float(), 1)[:, 1].cpu().numpy()
            out.append(p / self.stride)
        return np.concatenate(out) if out else np.zeros(0)

    def val_sets(self, stage: int):
        vc = [("clean", int(s)) for s in self.va]
        vd = [("deg", int(j)) for j in self.D.deg_items(self.va)] if stage == 2 else []
        return vc, vd

    def validate(self, model, stage: int) -> dict:
        vc, vd = self.val_sets(stage)
        pc = self.predict(model, vc)
        yc = self.D.labels[[s for _, s in vc]]
        row = dict(val_clean_auc=safe_auc(yc, pc))
        if vd:
            pdg = self.predict(model, vd)
            row["val_degraded_auc"] = safe_auc(self.D.labels[self.D.deg_src[[j for _, j in vd]]], pdg)
            row["score"] = float(np.nanmean([row["val_clean_auc"], row["val_degraded_auc"]]))
        else:
            row["score"] = row["val_clean_auc"]
        return row

    def select_threshold(self, model, stage: int) -> dict:
        vc, vd = self.val_sets(stage)
        y = list(self.D.labels[[s for _, s in vc]])
        p = list(self.predict(model, vc))
        if vd:
            y += list(self.D.labels[self.D.deg_src[[j for _, j in vd]]])
            p += list(self.predict(model, vd))
        r = find_threshold(np.array(y), np.array(p))
        r["selected_on"] = "validation original" + (" + degraded" if vd else "") + f" ({self.train_on})"
        return r

    # ---------------------------------------------------------------- training loop
    def train(self, stage: int, variant: str, model, ckpt: Path) -> dict:
        a, D = self.a, self.D
        vc = dict(paired=False, hard=False, lambda_feat=0.0, lambda_cam=0.0, curriculum=False) if stage == 1 else variant_config(a, variant)
        name = run_id(self.train_on, self.seed, "S1" if stage == 1 else variant)
        bs = a.batch_size
        clips_per_epoch = len(self.tr) if stage == 1 else len(self.tr) + self.n_tr_deg
        steps = int(math.ceil(clips_per_epoch / bs))
        epochs = a.epochs1 if stage == 1 else a.epochs2
        lr = a.lr1 if stage == 1 else a.lr2
        print("\n" + "=" * 100 + f"\n TRAIN {name} | stage {stage} | {VARIANTS.get(variant, {}).get('description', 'clean originals')}\n"
              f"  train originals {len(self.tr):,} (with degraded {len(self.tr_deg_src):,}, degraded versions {self.n_tr_deg:,}) | val "
              f"{len(self.va):,} | {steps} steps x {epochs} epochs | {bs} clips/step | lr {lr} | paired {vc['paired']} | hard {vc['hard']} | "
              f"lambda_feat {vc['lambda_feat']} | lambda_cam {vc['lambda_cam']} | curriculum {vc['curriculum']}\n" + "=" * 100)
        if stage == 2 and not len(self.tr_deg_src):
            raise SystemExit(f"[STOP] no degraded training videos for {self.train_on}")
        opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
        sched = optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=max(1, steps * epochs))
        crit = nn.CrossEntropyLoss()
        rng = np.random.default_rng([self.seed, stage, sum(map(ord, variant))])
        loader = ThreadPoolExecutor(max_workers=max(1, a.loader_threads))
        history, best, samp_rows = [], dict(score=-1.0, epoch=-1), []
        t_start, overall = time.time(), 0
        # ---- crash-safe resume (power cut): state written every --save-every-steps steps and at every epoch end
        resume_path = ckpt.with_suffix(".resume.pt")
        run_sig = dict(run=name, stage=stage, variant=variant, split_hash=self.split_hash, steps=steps, epochs=epochs, bs=bs, lr=lr)
        start_ep, start_st, ep_state = 1, 0, None
        rs = None
        if resume_path.exists():
            try:
                rs = torch.load(guard(resume_path, "test5_checkpoint"), map_location=self.device, weights_only=False)
            except Exception as e:
                print(f"  [resume] {resume_path.name} unreadable ({type(e).__name__}) - starting this run from scratch")
            if rs is not None and rs.get("sig") != run_sig:
                print(f"  [resume] {resume_path.name} belongs to a different configuration - starting this run from scratch")
                rs = None
        if rs is not None:
            model.load_state_dict(rs["model"])
            opt.load_state_dict(rs["opt"])
            sched.load_state_dict(rs["sched"])
            rng.bit_generator.state = rs["rng_epoch_start"]
            torch.set_rng_state(rs["torch_rng"].cpu())
            if torch.cuda.is_available() and rs.get("cuda_rng") is not None:
                torch.cuda.set_rng_state_all([s.cpu() for s in rs["cuda_rng"]])
            history, best, samp_rows, overall = rs["history"], rs["best"], rs["samp_rows"], rs["overall"]
            start_ep, start_st = rs["epoch"], rs["step"]
            ep_state = rs.get("ep_state")
            t_start = time.time() - rs["elapsed"]
            print(f"  [resume] {name}: continuing at epoch {start_ep}/{epochs}, step {start_st}/{steps} (power-cut checkpoint)")
        else:
            ckpt.with_suffix(".best.tmp").unlink(missing_ok=True)  # stale best from an interrupted run without resume state

        def save_resume(ep_next, st_next, rng_state, ep_st):
            atomic_torch_save(dict(sig=run_sig, model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(),
                                   rng_epoch_start=rng_state, torch_rng=torch.get_rng_state(),
                                   cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                                   history=history, best=best, samp_rows=samp_rows, overall=overall, epoch=ep_next, step=st_next,
                                   ep_state=ep_st, elapsed=time.time() - t_start, saved_at=now()), resume_path)

        for ep in range(start_ep, epochs + 1):
            model.train()
            e1, e2 = a.curriculum
            max_level = (0 if ep <= e1 else 1 if ep <= e2 else 2) if vc["curriculum"] else 2
            rng_epoch_start = rng.bit_generator.state            # re-creates this epoch's plan + batch seeds on resume
            plan, slog = self.plan_epoch(rng, stage, vc, max_level, steps, bs)
            mid_epoch = ep == start_ep and start_st > 0
            if not mid_epoch:                                     # on a mid-epoch resume these rows were already saved
                for (cell, mode, typ), n in slog.items():
                    samp_rows.append(dict(run=name, epoch=ep, max_level=max_level, cell=cell, mode=mode, degradation=typ, n=n))
            seeds = rng.integers(0, 2 ** 31, size=steps)
            first = start_st if mid_epoch else 0
            q, nxt = deque(), first

            def submit():
                nonlocal nxt
                if nxt < steps:
                    q.append(loader.submit(self.build, plan[nxt], int(seeds[nxt]), max_level, vc["paired"]))
                    nxt += 1

            for _ in range(max(2, 2 * a.loader_threads)):
                submit()
            tot, comp, t0, last = 0.0, Counter(), time.time(), 0
            if mid_epoch and ep_state:
                tot, comp, t0 = ep_state["tot"], Counter(ep_state["comp"]), time.time() - ep_state["epoch_elapsed"]
            for st in range(first, steps):
                fa, fb, audio, mask, y = q.popleft().result()
                submit()
                overall += 1
                if vc["paired"]:
                    nb = len(y)
                    xa = torch.from_numpy(np.concatenate([audio, audio])).to(self.device)
                    am = torch.from_numpy(np.concatenate([mask, mask])).to(self.device)
                    logits, z, act = model.forward_train(to_video_tensor(np.concatenate([fa, fb]), self.device), xa, am)
                    l_ce = crit(logits, torch.from_numpy(np.concatenate([y, y])).to(self.device))
                    l_feat = feature_consistency_loss(z[:nb].detach(), z[nb:])
                    cam = soft_gradcam(logits, act, 2 * nb, self.L, create_graph=vc["lambda_cam"] > 0)
                    l_cam = cam_consistency_loss(cam[:nb].detach(), cam[nb:])
                    loss = l_ce + vc["lambda_feat"] * l_feat + vc["lambda_cam"] * l_cam
                    comp["ce"] += float(l_ce.item())
                    comp["feat"] += float(l_feat.item())
                    comp["cam"] += float(l_cam.item())
                else:
                    logits = model(to_video_tensor(fa, self.device), torch.from_numpy(audio).to(self.device), mode="av",
                                   a_mask=torch.from_numpy(mask).to(self.device))
                    loss = crit(logits, torch.from_numpy(y).to(self.device))
                    comp["ce"] += float(loss.item())
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                if sched.last_epoch < steps * epochs - 1:
                    sched.step()
                tot += float(loss.item())
                if (st + 1) % max(1, a.save_every_steps) == 0 and st + 1 < steps:
                    save_resume(ep, st + 1, rng_epoch_start,
                                dict(tot=tot, comp=dict(comp), epoch_elapsed=time.time() - t0))
                if time.time() - last > 0.5 or st + 1 == steps:
                    gt = gpu_temp() if st % 50 == 0 else None
                    if gt is not None and gt >= a.gpu_max_temp:
                        print(f"\n  [thermal] GPU {gt}C >= {a.gpu_max_temp}C - pausing {a.gpu_pause}s")
                        torch.cuda.empty_cache()
                        time.sleep(a.gpu_pause)
                    el = time.time() - t_start
                    eta = el / overall * (steps * epochs - overall)
                    sys.stdout.write(f"\r  [{name}] ep {ep}/{epochs} {progress_bar(st + 1, steps, 16)} loss {tot / (st + 1):.4f} "
                                     + (f"(ce {comp['ce'] / (st + 1):.3f} feat {comp['feat'] / (st + 1):.3f} cam {comp['cam'] / (st + 1):.3f}) "
                                        if vc["paired"] else "") + f"| ETA {fmt_time(eta)}   ")
                    sys.stdout.flush()
                    last = time.time()
            row = dict(run=name, epoch=ep, max_severity_level=max_level, train_loss=tot / steps, lr=float(sched.get_last_lr()[0]),
                       **{f"train_{k}": v / steps for k, v in comp.items()}, epoch_time_s=time.time() - t0)
            row.update(self.validate(model, stage))
            history.append(row)
            better = np.isfinite(row["score"]) and row["score"] > best["score"]
            if better:
                best = dict(score=row["score"], epoch=ep)
                atomic_torch_save(dict(model_state_dict=model.state_dict(), epoch=ep, val=row), ckpt.with_suffix(".best.tmp"))
            print(f"\r  [{name}] ep {ep}/{epochs} | loss {row['train_loss']:.4f} | val clean AUC {row['val_clean_auc']:.4f}"
                  + (f" | val degraded AUC {row['val_degraded_auc']:.4f}" if "val_degraded_auc" in row else "")
                  + f" | score {row['score']:.4f}{'  <- best' if better else ''} | {fmt_time(row['epoch_time_s'])}" + " " * 20)
            save_resume(ep + 1, 0, rng.bit_generator.state, None)   # epoch done (incl. validation): next start state
        loader.shutdown(wait=True)
        pd.DataFrame(history).to_csv(self.P.hist_dir / f"history_{name}.csv", index=False)
        pd.DataFrame(samp_rows).to_csv(self.P.hist_dir / f"sampling_{name}.csv", index=False)
        tmp = ckpt.with_suffix(".best.tmp")
        if not tmp.exists():
            raise RuntimeError(f"{name}: no valid validation score - nothing saved")
        model.load_state_dict(torch.load(guard(tmp, "test5_checkpoint"), map_location=self.device, weights_only=False)["model_state_dict"])
        thr = self.select_threshold(model, stage)
        cfg = dict(run=name, stage=stage, variant=variant if stage == 2 else "S1", variant_config=vc, train_on=self.train_on, seed=self.seed,
                   epochs=epochs, lr=lr, batch_clips=bs, steps_per_epoch=steps, data_seed=self.a.data_seed, split_hash=self.split_hash,
                   split_manifest_sha256=read_json(self.P.gate, "test5_gate")["split_manifest_sha256"],
                   architecture_signature=architecture_signature(model), script_sha256=code_sha256(),
                   hard_share=self.a.hard_share, hard_weights=self.a.hard_w, curriculum_epochs=self.a.curriculum)
        cfg_sha = sha256_text(json.dumps(to_jsonable(cfg), sort_keys=True))
        sd = model.state_dict()
        atomic_torch_save(dict(model_state_dict=sd, threshold=thr["threshold"], threshold_selection=thr, best_epoch=best["epoch"],
                               best_val_score=best["score"], history=history, config=cfg, config_sha256=cfg_sha,
                               weights_sha256=weights_sha256(sd), final=True, saved_at=now()), ckpt)
        tmp.unlink(missing_ok=True)
        resume_path.unlink(missing_ok=True)
        rec = dict(run_id=name, stage=stage, path=str(ckpt), file_sha256=sha256_file(ckpt), weights_sha256=weights_sha256(sd),
                   script_sha256=code_sha256(), config_sha256=cfg_sha, split_hash=self.split_hash, threshold=thr["threshold"],
                   best_epoch=best["epoch"], best_val_score=best["score"], train_time_s=time.time() - t_start, saved_at=now())
        hp = self.P.checkpoints / "checkpoint_hashes.csv"
        h = read_csv(hp, "test5_checkpoint_index") if hp.exists() else pd.DataFrame()
        h = pd.concat([h[h.run_id != name] if len(h) else h, pd.DataFrame([rec])], ignore_index=True)
        h.to_csv(hp.with_suffix(".csv.tmp"), index=False)
        os.replace(hp.with_suffix(".csv.tmp"), hp)
        print(f"  saved {ckpt.name} | best epoch {best['epoch']} | val score {best['score']:.4f} | threshold {thr['threshold']:.2f} "
              f"| sha256 {rec['file_sha256'][:16]}...")
        return rec

    def done(self, ckpt: Path) -> bool:
        if not ckpt.exists():
            return False
        try:
            obj = torch.load(guard(ckpt, "test5_checkpoint"), map_location="cpu", weights_only=False)
        except (RuntimeError, EOFError, OSError, ValueError) as e:   # e.g. half-written by a power cut
            print(f"  [resume] {ckpt.name} is unreadable ({type(e).__name__}) - this run will be redone")
            return False
        return bool(obj.get("final")) and obj.get("config", {}).get("split_hash") == self.split_hash

    def load(self, ckpt: Path):
        obj = torch.load(guard(ckpt, "test5_checkpoint"), map_location=self.device, weights_only=False)
        m = AudioVisualDeepFakeCNN().to(self.device)
        m.load_state_dict(obj["model_state_dict"])
        m.eval()
        return m, obj

    def stage1(self, variant_tag: str = "") -> Path:
        ck = self.ckdir / (f"stage1{variant_tag}.pt")
        if self.done(ck):
            print(f"  [resume] {ck} done")
            return ck
        set_seed(self.seed, self.a.deterministic)
        model = AudioVisualDeepFakeCNN().to(self.device)
        print(f"  fresh AudioVisualDeepFakeCNN | {sum(p.numel() for p in model.parameters()):,} parameters | init seed {self.seed}")
        self.train(1, "S1", model, ck)
        return ck

    def stage2(self, variant: str, s1: Path) -> Path:
        ck = self.ckdir / f"{variant}_stage2.pt"
        if self.done(ck):
            print(f"  [resume] {ck} done")
            return ck
        set_seed(self.seed * 1000 + sum(map(ord, variant)), self.a.deterministic)
        model, _ = self.load(s1)
        self.train(2, variant, model, ck)
        return ck


def phase_train(a, P: Paths, D: T5Data, seeds: Optional[List[int]] = None):
    require_gate(P)
    status = []
    for seed in (seeds or a.seed_list):          # seed-major: after one seed every variant x training set exists
        for t in a.train_list:
            tr = Trainer(a, P, D, t, seed)
            s1 = None if a.no_share_stage1 else tr.stage1()
            for v in a.variant_list:
                try:
                    s1v = s1 or tr.stage1(f"_{v}")
                    tr.stage2(v, s1v)
                    status.append(dict(run_id=run_id(t, seed, v), status="completed"))
                except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
                    status.append(dict(run_id=run_id(t, seed, v), status=f"failed: {type(e).__name__}: {str(e)[:200]}"))
                    print(f"  [FAILED] {run_id(t, seed, v)}: {e}")
                    torch.cuda.empty_cache()
            status.append(dict(run_id=run_id(t, seed, "S1"), status="completed" if s1 and s1.exists() else "not shared"))
    old = read_csv(P.logs / "run_status.csv", "test5_log") if (P.logs / "run_status.csv").exists() else pd.DataFrame()
    st = pd.concat([old, pd.DataFrame(status)]).drop_duplicates("run_id", keep="last") if len(old) else pd.DataFrame(status)
    st.to_csv(P.logs / "run_status.csv", index=False)


# =====================================================================================
# EVALUATION (predictions on the untouched test split) + GRAD-CAM
# =====================================================================================
def available_runs(a, P: Paths) -> List[dict]:
    out = []
    for t in a.train_list:
        for seed in a.seed_list:
            d = P.checkpoints / train_tag(t) / f"seed{seed}"
            for v in ["S1"] + a.variant_list:
                ck = d / ("stage1.pt" if v == "S1" else f"{v}_stage2.pt")
                if ck.exists():
                    out.append(dict(run_id=run_id(t, seed, v), train_on=t, seed=seed, variant=v, ckpt=ck))
    return out


def phase_evaluate(a, P: Paths, D: T5Data):
    require_gate(P)
    test = D.idx("test")
    clean_items = [("clean", int(s)) for s in test]
    deg_items = [("deg", int(j)) for j in D.deg_items(test)]
    audio_src = test[D.has_audio[test]]
    for r in available_runs(a, P):
        out = P.pred_dir / f"{r['run_id']}.csv.gz"
        if out.exists():
            continue
        tr = Trainer(a, P, D, r["train_on"], r["seed"])
        model, obj = tr.load(r["ckpt"])
        thr = float(obj["threshold"])
        print(f"  evaluating {r['run_id']} (threshold {thr:.2f}) on {len(clean_items):,} test originals + {len(deg_items):,} degraded ...")
        rows = []

        def add(items, p, cond):
            for (k, i), pp in zip(items, p):
                s = D.src_of(k, i)
                rows.append((r["run_id"], r["train_on"], r["variant"], r["seed"], D.d.item_id[i] if k == "deg" else D.key[i], k, D.key[s],
                             D.group[s], D.ds[s], int(D.labels[s]), D.deg_type[i] if k == "deg" else "original",
                             D.deg_sev[i] if k == "deg" else "", cond, float(pp), thr))

        add(clean_items, tr.predict(model, clean_items), "normal")
        add(deg_items, tr.predict(model, deg_items), "normal")
        ai = [("clean", int(s)) for s in audio_src]
        if ai:
            add(ai, tr.predict(model, ai, "audio_masked"), "audio_masked")
            add(ai, tr.predict(model, ai, "audio_only"), "audio_only")
        pd.DataFrame(rows, columns=["run_id", "train_on", "variant", "seed", "item_id", "kind", "source_key", "group_id", "dataset", "label",
                                    "degradation", "severity", "condition", "p_fake", "threshold"]).to_csv(out, index=False, compression="gzip")
        del model
        torch.cuda.empty_cache()


def phase_xai(a, P: Paths, D: T5Data):
    require_gate(P)
    xp = read_csv(P.xai_pairs, "test5_manifest")
    dpos = {k: i for i, k in enumerate(D.d.item_id)}
    spos = {k: i for i, k in enumerate(D.key)}
    pairs = [(spos[r.source_key], dpos[r.deg_item_id], r) for r in xp.itertuples() if r.deg_item_id in dpos]
    for r in available_runs(a, P):
        out = P.cam_dir / f"{r['run_id']}.csv"
        if out.exists():
            continue
        tr = Trainer(a, P, D, r["train_on"], r["seed"])
        model, obj = tr.load(r["ckpt"])
        thr = float(obj["threshold"])
        st, L = tr.stride, tr.L
        rows = []
        print(f"  Grad-CAM {r['run_id']} on {len(pairs)} pre-registered test pairs ...")
        for k in range(0, len(pairs), 16):
            ch = pairs[k:k + 16]
            src = np.array([s for s, _, _ in ch])
            cc = np.stack([D.clip("clean", s)[::st][:L] for s, _, _ in ch])
            cr = np.stack([D.clip("clean", s)[1::st][:L] if st > 1 else D.clip("clean", s)[::-1][:L] for s, _, _ in ch])
            dc = np.stack([D.clip("deg", j)[::st][:L] for _, j, _ in ch])
            spec, has = D.audio(src)
            xa, am = torch.from_numpy(spec).to(tr.device), torch.from_numpy(has).to(tr.device)
            gc = grad_cam(model, to_video_tensor(cc, tr.device), xa, am, 1, thr)
            gd = grad_cam(model, to_video_tensor(dc, tr.device), xa, am, 1, thr)
            gr = grad_cam(model, to_video_tensor(cr, tr.device), xa, am, 1, thr)
            for b, (s, j, x) in enumerate(ch):
                pr, iou, sh = cam_metrics(gc["spatial"][b], gd["spatial"][b])
                _, ref_iou, _ = cam_metrics(gc["spatial"][b], gr["spatial"][b])
                ta, tb = gc["temporal"][b], gd["temporal"][b]
                tcr = float(np.corrcoef(ta, tb)[0, 1]) if ta.std() > 1e-8 and tb.std() > 1e-8 else np.nan
                stable = bool(gc["pred"][b] == gd["pred"][b])
                rows.append(dict(run_id=r["run_id"], train_on=r["train_on"], variant=r["variant"], seed=r["seed"], pair_id=x.pair_id,
                                 source_key=x.source_key, group_id=x.group_id, dataset=x.dataset, label=int(x.label), degradation=x.degradation,
                                 severity=x.severity, threshold=thr, p_fake_original=float(gc["p"][b]), p_fake_degraded=float(gd["p"][b]),
                                 prediction_stable=stable, pearson_r=pr, top20_iou=iou, center_shift=sh, temporal_cam_r=tcr,
                                 explanation_changed=bool(iou < EXPLANATION_CHANGE_IOU),
                                 explanation_changed_while_prediction_stable=bool(stable and iou < EXPLANATION_CHANGE_IOU),
                                 clean_reference_iou=ref_iou))
        pd.DataFrame(rows).to_csv(out, index=False)
        if r["variant"] in ("5A", "5D") and r["seed"] == a.seed_list[0]:
            save_cam_examples(P, D, tr, model, thr, pairs[:: max(1, len(pairs) // 6)][:6], r["run_id"])
        del model
        torch.cuda.empty_cache()


def save_cam_examples(P, D, tr, model, thr, pairs, rid):
    st, L = tr.stride, tr.L
    fig, axes = plt.subplots(len(pairs), 4, figsize=(9, 2.4 * len(pairs)))
    axes = np.atleast_2d(axes)
    for row, (s, j, x) in enumerate(pairs):
        cc, dc = D.clip("clean", s)[::st][:L][None], D.clip("deg", j)[::st][:L][None]
        spec, has = D.audio(np.array([s]))
        xa, am = torch.from_numpy(spec).to(tr.device), torch.from_numpy(has).to(tr.device)
        gc = grad_cam(model, to_video_tensor(cc, tr.device), xa, am, 1, thr)
        gd = grad_cam(model, to_video_tensor(dc, tr.device), xa, am, 1, thr)
        r_, iou, _ = cam_metrics(gc["spatial"][0], gd["spatial"][0])
        for k, (img, cam, t) in enumerate([(cc[0, L // 2], None, f"original p={gc['p'][0]:.2f}"), (cc[0, L // 2], gc["spatial"][0], "CAM original"),
                                           (dc[0, L // 2], None, f"{x.degradation}/{x.severity} p={gd['p'][0]:.2f}"),
                                           (dc[0, L // 2], gd["spatial"][0], f"CAM degraded IoU={iou:.2f} r={r_:.2f}")]):
            axes[row, k].imshow(img)
            if cam is not None:
                axes[row, k].imshow(cam, cmap="magma", alpha=0.5)
            axes[row, k].set_title(t, fontsize=7)
            axes[row, k].axis("off")
    plt.suptitle(f"Grad-CAM examples ({rid}); stability is not correctness", fontsize=9)
    plt.tight_layout()
    plt.savefig(P.figures / f"gradcam_examples_{rid}.png", dpi=110)
    plt.close()


# =====================================================================================
# ANALYSIS
# =====================================================================================
def load_predictions(P: Paths) -> pd.DataFrame:
    fs = sorted(P.pred_dir.glob("*.csv.gz"))
    return pd.concat([read_csv(f, "test5_predictions") for f in fs], ignore_index=True) if fs else pd.DataFrame()


def load_cams(P: Paths) -> pd.DataFrame:
    fs = sorted(P.cam_dir.glob("*.csv"))
    return pd.concat([read_csv(f, "test5_gradcam") for f in fs], ignore_index=True) if fs else pd.DataFrame()


def per_run_tables(pred: pd.DataFrame):
    met, degm, rob, cross = [], [], [], []
    for rid, g in pred.groupby("run_id"):
        r0 = g.iloc[0]
        thr = float(r0.threshold)
        base = dict(run_id=rid, train_on=r0.train_on, variant=r0.variant, seed=int(r0.seed), threshold=thr)
        for ds, h in g.groupby("dataset"):
            setting = "in-domain" if ds == r0.train_on else "cross-dataset"
            n = h[h.condition == "normal"]
            c, d = n[n.kind == "clean"], n[n.kind == "deg"]
            conds = [("clean", c), ("degraded_all", d), ("degraded_hard", d[d.degradation.isin(HARD_TYPES)]),
                     ("degraded_other", d[~d.degradation.isin(HARD_TYPES)])]
            conds += [(f"clean_{cn}", h[(h.condition == cn)]) for cn in ("audio_masked", "audio_only") if (h.condition == cn).any()]
            if (h.condition == "audio_masked").any():
                conds.append(("clean_with_audio_normal", c[c.source_key.isin(h[h.condition == "audio_masked"].source_key)]))
            for cn, x in conds:
                m = compute_metrics(x.label, x.p_fake, thr)
                met.append(dict(base, test_dataset=ds, setting=setting, condition=cn, **m))
            cp = dict(zip(c.source_key, c.p_fake))
            d = d.assign(p_clean=d.source_key.map(cp))
            d = d.assign(flip=((d.p_fake >= thr) != (d.p_clean >= thr)), shift=(d.p_fake - d.p_clean).abs())
            c_same = c[c.source_key.isin(d.source_key)]
            auc_clean_same = safe_auc(c_same.label, c_same.p_fake)
            type_auc = {}
            for (dg, sv), x in list(d.groupby(["degradation", "severity"])) + [((dg, "all"), x) for dg, x in d.groupby("degradation")]:
                m = compute_metrics(x.label, x.p_fake, thr)
                cs = c[c.source_key.isin(x.source_key)]
                auc = safe_auc(x.label, x.p_fake, MIN_PER_CLASS_AUC)
                ca = safe_auc(cs.label, cs.p_fake, MIN_PER_CLASS_AUC)
                degm.append(dict(base, test_dataset=ds, setting=setting, degradation=dg, severity=sv, **{k: m[k] for k in
                                 ("n", "n_real", "n_fake", "accuracy", "balanced_accuracy", "precision", "recall", "specificity", "f1")},
                                 auc=auc, clean_auc_same_sources=ca, auc_change=auc - ca, prediction_flip_rate_pct=100 * x.flip.mean(),
                                 n_flips=int(x.flip.sum()), mean_abs_prob_shift=x["shift"].mean()))
                if sv == "all":
                    type_auc[dg] = auc
            sev_auc = {sv: safe_auc(x.label, x.p_fake, MIN_PER_CLASS_AUC) for sv, x in d.groupby("severity")}
            ts = [(k, v) for k, v in type_auc.items() if np.isfinite(v)]
            tsv = [(f"{x.degradation.iloc[0]}/{sv}", safe_auc(x.label, x.p_fake, MIN_PER_CLASS_AUC)) for (dg, sv), x in d.groupby(["degradation", "severity"])]
            tsv = [t for t in tsv if np.isfinite(t[1])]
            mc = compute_metrics(c.label, c.p_fake, thr)
            md_ = compute_metrics(d.label, d.p_fake, thr)
            rob.append(dict(base, test_dataset=ds, setting=setting, n_clean=len(c), n_degraded=len(d), clean_auc=mc["auc"],
                            clean_auc_sources_with_degraded=auc_clean_same, degraded_auc_pooled=md_["auc"],
                            clean_to_degraded_auc_change=md_["auc"] - auc_clean_same, clean_balanced_accuracy=mc["balanced_accuracy"],
                            degraded_balanced_accuracy=md_["balanced_accuracy"], prediction_flip_rate_pct=100 * d.flip.mean() if len(d) else np.nan,
                            n_flips=int(d.flip.sum()), mean_abs_prob_shift=d["shift"].mean(),
                            worst_type=min(ts, key=lambda t: t[1])[0] if ts else "", worst_type_auc=min(v for _, v in ts) if ts else np.nan,
                            worst_type_severity=min(tsv, key=lambda t: t[1])[0] if tsv else "",
                            worst_type_severity_auc=min(v for _, v in tsv) if tsv else np.nan,
                            macro_degradation_auc=float(np.mean([v for _, v in ts])) if ts else np.nan,
                            mean_degradation_auc=md_["auc"], n_types=len(ts),
                            **{f"auc_{s}": sev_auc.get(s, np.nan) for s in ("mild", "medium", "severe")},
                            **{f"auc_{t}": type_auc.get(t, np.nan) for t in HARD_TYPES},
                            auc_hard_pooled=safe_auc(d[d.degradation.isin(HARD_TYPES)].label, d[d.degradation.isin(HARD_TYPES)].p_fake, MIN_PER_CLASS_AUC)))
            for cn, x in (("clean", c), ("degraded", d)):
                m = compute_metrics(x.label, x.p_fake, thr)
                xr, xf = x[x.label == 0].p_fake, x[x.label == 1].p_fake
                cross.append(dict(base, test_dataset=ds, setting=setting, condition=cn, n=m["n"], n_real=m["n_real"], n_fake=m["n_fake"],
                                  auc=m["auc"], balanced_accuracy=m["balanced_accuracy"], error_rate_real_fpr=1 - m["specificity"],
                                  error_rate_fake_fnr=1 - m["recall"], predicted_fake_rate=float((x.p_fake >= thr).mean()) if len(x) else np.nan,
                                  mean_p_fake_real=xr.mean(), median_p_fake_real=xr.median(), mean_p_fake_fake=xf.mean(),
                                  median_p_fake_fake=xf.median(), q10_p_fake=x.p_fake.quantile(0.1), q90_p_fake=x.p_fake.quantile(0.9)))
            am = h[h.condition == "audio_masked"]
            if len(am):
                cn_ = dict(zip(c.source_key, c.p_fake))
                for cond in ("audio_masked", "audio_only"):
                    x = h[h.condition == cond]
                    pn = x.source_key.map(cn_)
                    m = compute_metrics(x.label, x.p_fake, thr)
                    mn = compute_metrics(x.label, pn, thr)
                    cross.append(dict(base, test_dataset=ds, setting=setting, condition=f"ablation_{cond}", n=m["n"], n_real=m["n_real"],
                                      n_fake=m["n_fake"], auc=m["auc"], balanced_accuracy=m["balanced_accuracy"],
                                      error_rate_real_fpr=1 - m["specificity"], error_rate_fake_fnr=1 - m["recall"],
                                      predicted_fake_rate=float((x.p_fake >= thr).mean()), mean_p_fake_real=x[x.label == 0].p_fake.mean(),
                                      mean_p_fake_fake=x[x.label == 1].p_fake.mean(), auc_normal_same_videos=mn["auc"],
                                      flip_rate_vs_normal_pct=100 * float(((x.p_fake >= thr) != (pn >= thr)).mean())))
    return pd.DataFrame(met), pd.DataFrame(degm), pd.DataFrame(rob), pd.DataFrame(cross)


def cam_summary(cams: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if not len(cams):
        return pd.DataFrame()
    cams = cams.copy()
    for c in ("prediction_stable", "explanation_changed", "explanation_changed_while_prediction_stable"):
        cams[c] = as_bool(cams[c])
    for (rid, ds), g in cams.groupby(["run_id", "dataset"]):
        r0 = g.iloc[0]
        for scope, x in [("all", g), ("hard", g[g.degradation.isin(HARD_TYPES)]), ("other", g[~g.degradation.isin(HARD_TYPES)])] + \
                        [(f"type:{t}", y) for t, y in g.groupby("degradation")]:
            if not len(x):
                continue
            st = x[x.prediction_stable]
            rows.append(dict(run_id=rid, train_on=r0.train_on, variant=r0.variant, seed=int(r0.seed), test_dataset=ds,
                             setting="in-domain" if ds == r0.train_on else "cross-dataset", scope=scope, n_pairs=len(x),
                             mean_pearson_r=x.pearson_r.mean(), mean_top20_iou=x.top20_iou.mean(), mean_center_shift=x.center_shift.mean(),
                             mean_temporal_cam_r=x.temporal_cam_r.mean(), prediction_stable_rate=x.prediction_stable.mean(),
                             n_prediction_stable=len(st), explanation_changed_given_stable_pct=100 * st.explanation_changed.mean() if len(st) else np.nan,
                             mean_clean_reference_iou=x.clean_reference_iou.mean()))
    return pd.DataFrame(rows)


SEED_METRICS = [("robustness", "clean_auc"), ("robustness", "degraded_auc_pooled"), ("robustness", "clean_to_degraded_auc_change"),
                ("robustness", "prediction_flip_rate_pct"), ("robustness", "worst_type_auc"), ("robustness", "macro_degradation_auc"),
                ("robustness", "auc_hard_pooled"), ("robustness", "auc_brightness"), ("robustness", "auc_saturation"),
                ("robustness", "auc_contrast"), ("robustness", "auc_blur"), ("robustness", "degraded_balanced_accuracy"),
                ("gradcam", "mean_top20_iou"), ("gradcam", "mean_pearson_r"), ("gradcam", "mean_center_shift"),
                ("gradcam", "explanation_changed_given_stable_pct")]


def seed_comparison(rob: pd.DataFrame, cs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    srcs = {"robustness": rob, "gradcam": cs[cs.scope.isin(["all", "hard"])] if len(cs) else cs}
    for src, metric in SEED_METRICS:
        df = srcs[src]
        if not len(df) or metric not in df:
            continue
        keys = ["train_on", "variant", "test_dataset", "setting"] + (["scope"] if src == "gradcam" else [])
        for k, g in df.groupby(keys):
            vals = dict(zip(g.seed, g[metric]))
            v = np.array([x for x in vals.values() if np.isfinite(x)])
            r = dict(zip(keys, k if isinstance(k, tuple) else (k,)))
            r.update(metric=metric + (f" [{r.pop('scope')}]" if src == "gradcam" else ""), n_seeds=len(v),
                     seeds=",".join(str(s) for s in sorted(vals)), mean=v.mean() if len(v) else np.nan,
                     sd_across_seeds=v.std(ddof=1) if len(v) > 1 else np.nan, min=v.min() if len(v) else np.nan, max=v.max() if len(v) else np.nan,
                     **{f"seed_{s}": vals[s] for s in sorted(vals)})
            rows.append(r)
    return pd.DataFrame(rows)


def bootstrap_stats(a, pred: pd.DataFrame, cams: pd.DataFrame) -> pd.DataFrame:
    """Paired comparisons on the same test videos. Seed uncertainty: per-seed differences (mean, sd, sign agreement).
    Sample uncertainty: cluster bootstrap (resampling source groups) of the seed-averaged difference."""
    if not len(pred):
        return pd.DataFrame()
    rows = []
    hard = set(HARD_TYPES)
    for (train_on, ds), g in pred[pred.condition == "normal"].groupby(["train_on", "dataset"]):
        groups = np.array(sorted(g.group_id.unique()))
        G = len(groups)
        gpos = {x: i for i, x in enumerate(groups)}
        rng = np.random.default_rng([a.data_seed, sum(map(ord, train_on + ds))])
        Wg = rng.multinomial(G, np.full(G, 1.0 / G), size=a.n_boot).astype(np.float64)
        Wg = np.vstack([np.ones(G), Wg])     # row 0 = point estimate
        camg = cams[(cams.train_on == train_on) & (cams.dataset == ds)] if len(cams) else cams
        per = {}
        for rid, h in g.groupby("run_id"):
            v, sd = h.variant.iloc[0], int(h.seed.iloc[0])
            thr = float(h.threshold.iloc[0])
            c, d = h[h.kind == "clean"], h[h.kind == "deg"]
            out = {}

            def wauc(x):
                if (x.label == 0).sum() < MIN_PER_CLASS_AUC or (x.label == 1).sum() < MIN_PER_CLASS_AUC:
                    return np.full(len(Wg), np.nan)
                return WeightedAUC(x.label.values, x.p_fake.values)(Wg[:, x.group_id.map(gpos).values])

            out["clean_auc"] = wauc(c)
            out["degraded_auc"] = wauc(d)
            out["hard_auc"] = wauc(d[d.degradation.isin(hard)])
            ta = []
            for t, x in d.groupby("degradation"):
                aa = wauc(x)
                if t in ("brightness", "saturation", "contrast", "blur"):
                    out[f"{t}_auc"] = aa
                if np.isfinite(aa[0]):
                    ta.append(aa)
            out["worst_type_auc"] = np.nanmin(np.vstack(ta), axis=0) if ta else np.full(len(Wg), np.nan)
            cp = dict(zip(c.source_key, c.p_fake))
            fl = ((d.p_fake >= thr) != (d.source_key.map(cp) >= thr)).astype(float).values
            W = Wg[:, d.group_id.map(gpos).values]
            out["flip_rate_pct"] = 100 * (W * fl).sum(1) / np.maximum(W.sum(1), 1e-9)
            if len(camg):
                x = camg[camg.run_id == rid]
                for nm, y in (("gradcam_iou_all", x), ("gradcam_iou_hard", x[x.degradation.isin(hard)])):
                    if len(y):
                        Wc = Wg[:, y.group_id.map(gpos).fillna(-1).astype(int).values.clip(0)] * (y.group_id.isin(gpos).values)
                        out[nm] = (Wc * y.top20_iou.values).sum(1) / np.maximum(Wc.sum(1), 1e-9)
                y = x[x.degradation.isin(hard)]
                if len(y):
                    Wc = Wg[:, y.group_id.map(gpos).fillna(-1).astype(int).values.clip(0)] * (y.group_id.isin(gpos).values)
                    out["gradcam_pearson_hard"] = (Wc * y.pearson_r.fillna(0).values).sum(1) / np.maximum(Wc.sum(1), 1e-9)
            per[(v, sd)] = out
        better_low = {"flip_rate_pct"}
        for hi, lo in COMPARISONS:
            seeds = sorted({s for (v, s) in per if v == hi} & {s for (v, s) in per if v == lo})
            if not seeds:
                continue
            for metric in per[(hi, seeds[0])]:
                if not all(metric in per[(hi, s)] and metric in per[(lo, s)] for s in seeds):
                    continue
                diffs = np.vstack([per[(hi, s)][metric] - per[(lo, s)][metric] for s in seeds])   # [n_seeds, 1+B]
                point = diffs[:, 0]
                if not np.isfinite(point).all():
                    continue
                boot = np.nanmean(diffs[:, 1:], axis=0)
                boot = boot[np.isfinite(boot)]
                lo_ci, hi_ci = (np.percentile(boot, [2.5, 97.5]) if len(boot) > 20 else (np.nan, np.nan))
                want = -1 if metric in better_low else 1
                same_sign = bool(np.all(np.sign(point) == np.sign(point.mean()))) if point.mean() != 0 else False
                if G < MIN_CLUSTERS or not np.isfinite(lo_ci):
                    verdict = "inconclusive (too few clusters)"
                elif len(seeds) < 2:
                    verdict = ("single seed: CI excludes 0 (not a multi-seed result)" if (lo_ci > 0 or hi_ci < 0) else "single seed: inconclusive")
                elif (lo_ci > 0 or hi_ci < 0) and same_sign:
                    verdict = "supported improvement" if np.sign(point.mean()) == want else "supported deterioration"
                else:
                    verdict = "inconclusive"
                p_two = float(2 * min((boot <= 0).mean(), (boot >= 0).mean())) if len(boot) else np.nan
                rows.append(dict(train_on=train_on, test_dataset=ds, setting="in-domain" if ds == train_on else "cross-dataset",
                                 comparison=f"{hi} - {lo}", metric=metric, better_direction="lower" if want < 0 else "higher",
                                 n_seeds=len(seeds), seeds=",".join(map(str, seeds)), mean_diff=point.mean(),
                                 sd_diff_across_seeds=point.std(ddof=1) if len(seeds) > 1 else np.nan,
                                 per_seed_diffs=";".join(f"{s}:{d:+.4f}" for s, d in zip(seeds, point)), sign_agreement=same_sign,
                                 ci95_low=lo_ci, ci95_high=hi_ci, bootstrap_p_two_sided=p_two, n_clusters=G, n_boot=len(boot), verdict=verdict))
    return pd.DataFrame(rows)


def make_figures(P: Paths, rob: pd.DataFrame, cs: pd.DataFrame, degm: pd.DataFrame, cross: pd.DataFrame):
    if not len(rob):
        return
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    order = [v for v in ["S1", "5A", "5B", "5C", "5D"] if v in set(rob.variant)]

    def panel(df, metric, title, fname, ylabel, ylim=None, setting="in-domain"):
        trains = sorted(df.train_on.unique())
        fig, axes = plt.subplots(1, len(trains), figsize=(4.8 * len(trains), 3.6), squeeze=False)
        for ax, t in zip(axes[0], trains):
            x = df[(df.train_on == t) & (df.setting == setting)]
            for k, v in enumerate(order):
                vals = x[x.variant == v][metric].dropna().values
                if not len(vals):
                    continue
                ax.bar(k, vals.mean(), color=VARIANT_COLOR[v], alpha=0.85, width=0.65)
                ax.scatter(np.full(len(vals), k) + np.linspace(-0.12, 0.12, len(vals)), vals, color="#0b0b0b", s=10, zorder=3)
                if len(vals) > 1:
                    ax.errorbar(k, vals.mean(), yerr=vals.std(ddof=1), color="#0b0b0b", capsize=3, lw=1)
            ax.set_xticks(range(len(order)))
            ax.set_xticklabels(order)
            ax.set_title(f"trained on {t} ({setting})")
            ax.set_ylabel(ylabel)
            if ylim:
                ax.set_ylim(*ylim)
            ax.grid(axis="y", alpha=0.3)
        fig.suptitle(title + " - bars: seed mean, dots: seeds, whiskers: SD across seeds", fontsize=9)
        plt.tight_layout()
        plt.savefig(P.figures / fname, dpi=130)
        plt.close()

    panel(rob, "clean_auc", "Clean test AUC", "fig_clean_auc.png", "AUC", (0.4, 1.0))
    panel(rob, "degraded_auc_pooled", "Degraded test AUC (all verified degraded versions)", "fig_degraded_auc.png", "AUC", (0.4, 1.0))
    panel(rob, "worst_type_auc", "Worst-case degradation-type AUC", "fig_worst_case_auc.png", "AUC", (0.3, 1.0))
    panel(rob, "prediction_flip_rate_pct", "Prediction flips original -> degraded", "fig_prediction_flips.png", "% flipped")
    panel(rob, "auc_brightness", "Brightness-degraded AUC", "fig_brightness_auc.png", "AUC", (0.3, 1.0))
    panel(rob, "auc_saturation", "Saturation-degraded AUC", "fig_saturation_auc.png", "AUC", (0.3, 1.0))
    panel(rob, "degraded_auc_pooled", "Cross-dataset degraded AUC", "fig_cross_dataset_degraded_auc.png", "AUC", (0.2, 1.0), "cross-dataset")
    panel(rob, "clean_auc", "Cross-dataset clean AUC", "fig_cross_dataset_clean_auc.png", "AUC", (0.2, 1.0), "cross-dataset")
    if len(cs):
        for scope in ("all", "hard"):
            x = cs[cs.scope == scope]
            if len(x):
                panel(x, "mean_top20_iou", f"Grad-CAM top-20% IoU original vs degraded ({scope} types)", f"fig_gradcam_iou_{scope}.png", "IoU", (0, 1))
    if len(degm):
        x = degm[(degm.setting == "in-domain") & degm.degradation.isin(["brightness", "saturation"]) & (degm.severity != "all")]
        if len(x):
            trains = sorted(x.train_on.unique())
            fig, axes = plt.subplots(len(trains), 2, figsize=(9, 3.2 * len(trains)), squeeze=False)
            for r, t in enumerate(trains):
                for c, dg in enumerate(["brightness", "saturation"]):
                    ax = axes[r, c]
                    for v in order:
                        y = x[(x.train_on == t) & (x.degradation == dg) & (x.variant == v)].groupby("severity").auc.mean()
                        y = y.reindex(["mild", "medium", "severe"])
                        ax.plot(range(3), y.values, marker="o", color=VARIANT_COLOR[v], label=v)
                    ax.set_xticks(range(3))
                    ax.set_xticklabels(["mild", "medium", "severe"])
                    ax.set_title(f"{dg} - trained on {t}")
                    ax.set_ylabel("AUC (seed mean)")
                    ax.grid(alpha=0.3)
            axes[0, 0].legend(fontsize=8, frameon=False)
            plt.tight_layout()
            plt.savefig(P.figures / "fig_brightness_saturation_by_severity.png", dpi=130)
            plt.close()


def phase_analyze(a, P: Paths):
    pred = load_predictions(P)
    if not len(pred):
        print("  no predictions yet - run --phase evaluate")
        return
    cams = load_cams(P)
    met, degm, rob, cross = per_run_tables(pred)
    cs = cam_summary(cams)
    met.to_csv(P.metrics / "metrics_test5.csv", index=False)
    degm.to_csv(P.metrics / "degradation_metrics_test5.csv", index=False)
    rob.to_csv(P.metrics / "robustness_test5.csv", index=False)
    cross.to_csv(P.metrics / "cross_dataset_test5.csv", index=False)
    if len(cams):
        cams.to_csv(P.metrics / "gradcam_stability_test5.csv", index=False)
        cs.to_csv(P.metrics / "gradcam_summary_test5.csv", index=False)
    sc = seed_comparison(rob, cs)
    sc.to_csv(P.metrics / "seed_comparison.csv", index=False)
    print(f"  bootstrap statistics ({a.n_boot} cluster replicates) ...")
    stt = bootstrap_stats(a, pred, cams)
    stt.to_csv(P.metrics / "statistics_test5.csv", index=False)
    make_figures(P, rob, cs, degm, cross)
    print(f"  metrics written to {P.metrics}; figures to {P.figures}")


# =====================================================================================
# REPORT
# =====================================================================================
def hypothesis_verdicts(stt: pd.DataFrame) -> List[dict]:
    H = [("H1", "Hard-degradation sampling improves detection on hard degradations", "5B - 5A", ["hard_auc", "brightness_auc", "saturation_auc"]),
         ("H2", "Grad-CAM consistency (with pairing) improves explanation stability", "5C - 5B", ["gradcam_iou_hard", "gradcam_iou_all"]),
         ("H3", "Feature consistency + curriculum improve robustness / stability beyond 5C", "5D - 5C",
          ["degraded_auc", "flip_rate_pct", "gradcam_iou_hard"]),
         ("H4", "The full method (5D) improves robustness over the independent baseline (5A)", "5D - 5A",
          ["degraded_auc", "hard_auc", "worst_type_auc", "flip_rate_pct", "gradcam_iou_hard"])]
    out = []
    for hid, text, comp, metrics in H:
        x = stt[(stt.comparison == comp) & (stt.setting == "in-domain") & stt.metric.isin(metrics)] if len(stt) else pd.DataFrame()
        if not len(x):
            out.append(dict(hypothesis=hid, statement=text, comparison=comp, verdict="not evaluated (runs missing)", evidence=""))
            continue
        prim = x[x.metric == metrics[0]]
        v = prim.verdict.tolist()
        if v and all(s == "supported improvement" for s in v):
            verdict = "supported (primary metric, all training sets)"
        elif any(s == "supported improvement" for s in v):
            verdict = "partially supported (some training sets)"
        elif any(s == "supported deterioration" for s in v):
            verdict = "not supported (statistically supported deterioration)"
        else:
            verdict = "inconclusive / not supported"
        ev = "; ".join(f"{r.train_on} {r.metric}: {r.mean_diff:+.4f} [{fmt(r.ci95_low, 4)}, {fmt(r.ci95_high, 4)}] ({r.verdict}, seeds {r.n_seeds})"
                       for r in x.itertuples())
        out.append(dict(hypothesis=hid, statement=text, comparison=comp, verdict=verdict, evidence=ev))
    return out


def phase_report(a, P: Paths):
    def rd(name):
        f = P.metrics / name
        return read_csv(f, "test5_metrics") if f.exists() and f.stat().st_size > 2 else pd.DataFrame()

    met, degm, rob, cross, cs, sc, stt = (rd(n) for n in ("metrics_test5.csv", "degradation_metrics_test5.csv", "robustness_test5.csv",
                                                           "cross_dataset_test5.csv", "gradcam_summary_test5.csv", "seed_comparison.csv",
                                                           "statistics_test5.csv"))
    gate = read_json(P.gate, "test5_gate") if P.gate.exists() else {}
    leak = read_json(P.leakage, "test5_gate") if P.leakage.exists() else {}
    si = read_json(P.split_integrity, "test5_split") if P.split_integrity.exists() else {}
    sm = read_json(P.split_meta, "test5_split") if P.split_meta.exists() else {}
    inv_s = read_json(P.data_audit / "inventory_summary.json", "test5_audit") if (P.data_audit / "inventory_summary.json").exists() else {}
    t4 = read_json(P.data_audit / "test4_historical_reference.json", "test5_audit") if (P.data_audit / "test4_historical_reference.json").exists() else {}
    hashes = read_csv(P.checkpoints / "checkpoint_hashes.csv", "test5_checkpoint_index") if (P.checkpoints / "checkpoint_hashes.csv").exists() else pd.DataFrame()
    runs = available_runs(a, P)
    expected = [run_id(t, s, v) for t in a.train_list for s in a.seed_list for v in a.variant_list]
    have = {r["run_id"] for r in runs}
    missing = [r for r in expected if r not in have]
    status = read_csv(P.logs / "run_status.csv", "test5_log") if (P.logs / "run_status.csv").exists() else pd.DataFrame()
    verdicts = hypothesis_verdicts(stt)
    seeds_done = {v: sorted({r["seed"] for r in runs if r["variant"] == v}) for v in a.variant_list}
    summary = dict(test="Test 5 (independent)", created=now(), script_sha256=code_sha256(), gate=gate.get("gate"),
                   confirmed_leakage=leak.get("confirmed_cross_split_leakage_count"), data_seed=a.data_seed,
                   split_hash=si.get("split_hash_originals"), split_manifest_sha256=si.get("split_manifest_sha256"),
                   runs_expected=len(expected), runs_completed=len(expected) - len(missing), runs_missing=missing,
                   seeds_completed_per_variant=seeds_done, hypotheses=verdicts,
                   headline=rob[rob.setting == "in-domain"].groupby(["train_on", "variant"])[[
                       "clean_auc", "degraded_auc_pooled", "worst_type_auc", "prediction_flip_rate_pct", "auc_brightness", "auc_saturation"]]
                   .agg(["mean", "std", "count"]).round(4).reset_index().to_dict(orient="records") if len(rob) else [],
                   checkpoints=hashes.to_dict(orient="records") if len(hashes) else [],
                   localization="No ground-truth manipulation-region annotations were found; explanation CORRECTNESS is not established - "
                                "only stability under degradation is measured." if not inv_s.get("localization_annotation_dirs")
                   else f"annotation folders found but not evaluated: {inv_s.get('localization_annotation_dirs')}",
                   comparison_with_test4="Across different evaluation splits (Test 4 seed 42 vs Test 5 seed {}); NOT a controlled paired "
                                         "comparison.".format(a.data_seed))
    write_json(P.reports / "final_experiment_summary.json", summary)

    def seedtab(metric, setting="in-domain"):
        x = sc[(sc.metric == metric) & (sc.setting == setting)] if len(sc) else sc
        if not len(x):
            return "_(not available)_"
        x = x.assign(value=x.apply(lambda r: f"{r['mean']:.3f}" + (f" ± {r.sd_across_seeds:.3f}" if np.isfinite(r.sd_across_seeds) else "")
                                   + f" (n={int(r.n_seeds)})", axis=1))
        return md_table(x.pivot_table(index=["train_on", "test_dataset"], columns="variant", values="value", aggfunc="first").reset_index())

    L = []
    w = L.append
    w("# Test 5 (independent): hard-degradation-aware training and Grad-CAM explanation stability\n")
    w(f"_Generated {now()} | script `{SCRIPT_PATH.name}` sha256 `{code_sha256()[:16]}…` | data seed {a.data_seed} | "
      f"split hash `{str(si.get('split_hash_originals', ''))[:16]}…`_\n")
    w(f"**Completion:** {len(expected) - len(missing)}/{len(expected)} Stage-2 runs finished. Seeds completed per variant: "
      + ", ".join(f"{v}: {s or 'none'}" for v, s in seeds_done.items()) + ".")
    if missing:
        w(f" Missing runs: {', '.join(missing)}. Results for a variant with fewer than 3 seeds are NOT multi-seed results.")
    w("\n## 1. Research question and hypotheses\n")
    w("Does targeted training on the degradation types with the least stable explanations in earlier work (brightness, saturation, "
      "contrast, blur) improve detection robustness and Grad-CAM explanation stability of the AudioVisualDeepFakeCNN?\n")
    for v in verdicts:
        w(f"- **{v['hypothesis']}** ({v['comparison']}): {v['statement']}")
    w("\n## 2. How independence was established\n")
    ind = dict(leak.get("independence", {}))
    if P.access_log.exists():  # counts over ALL phases so far (the gate snapshot predates later reads)
        acc = read_csv(P.access_log, "test5_access_log")
        ind.update(artifact_reads_logged=len(acc), blocked_forbidden_reads=int(as_bool(acc.blocked).sum()),
                   historical_readonly_reads=int((as_bool(acc.forbidden_root) & ~as_bool(acc.blocked)).sum()))
    w(f"- New root `{P.root}`; fresh inventory with full-file SHA-256; new grouping; new split (seed {a.data_seed}); own caches (`t5_` namespace, "
      f"provenance + inventory-derived keys); fresh model initialisation; no import of Test 4 code at run time.\n"
      f"- Artifact reads logged: {ind.get('artifact_reads_logged')}; blocked reads of Test 4 / old Test 5 artifacts: "
      f"{ind.get('blocked_forbidden_reads')}; read-only historical documentation reads: {ind.get('historical_readonly_reads')}.\n"
      f"- Cache files without provenance: {ind.get('cache_files_without_provenance')}; cache keys match current data: "
      f"{ind.get('cache_keys_match_current_data')}; hard/soft links to other runs: {ind.get('cache_files_linked_or_hardlinked')} / "
      f"{ind.get('cache_files_same_inode_as_other_runs')}.\n")
    w("## 3. Dataset inventory and exclusions\n")
    if inv_s:
        w(md_table(pd.DataFrame(inv_s.get("per_folder", []))))
        w(f"\nOriginals {inv_s.get('n_originals')}; manifest degraded outputs {inv_s.get('n_degraded_listed')} "
          f"(valid link metadata {inv_s.get('n_degraded_link_ok')}); orphan degraded files {inv_s.get('n_orphans')}; exact duplicate files "
          f"{inv_s.get('exact_duplicate_files')}.\n")
        w("\n".join(f"- {n}" for n in inv_s.get("notes", [])))
    w(f"\nExcluded items by reason: `{json.dumps(si.get('excluded', {}))}` (details: data_audit/degraded_excluded.csv, "
      "data_audit/degraded_link_verification.csv).\n")
    w("## 4. Split and leakage validation\n")
    w(f"Gate **{gate.get('gate')}**; confirmed cross-split leakage **{leak.get('confirmed_cross_split_leakage_count')}**.\n")
    if leak.get("checks"):
        w(md_table(pd.DataFrame([dict(check=k, result="PASS" if v else "FAIL") for k, v in leak["checks"].items()])))
    dup = leak.get("duplicates", {})
    w(f"\nGroups {sm.get('n_groups')}; content merges before splitting {sm.get('n_content_merges')}; unmerged near-duplicate candidates "
      f"{dup.get('near_duplicate_candidates_unmerged')} (across splits, unresolved and documented: "
      f"{dup.get('near_duplicate_candidates_cross_split_unresolved')}); within-split exact duplicate sets "
      f"{dup.get('exact_file_duplicate_sets_within_split')}. {dup.get('method_limits', '')}\n")
    if si.get("counts"):
        w(md_table(pd.DataFrame(si["counts"]).pivot_table(index=["dataset", "subset"], columns="split", values="originals", fill_value=0).reset_index()))
    w("\n## 5. Preprocessing and model configuration\n")
    cfgA = read_json(P.configs / "experiment_config_5A.json", "test5_config") if (P.configs / "experiment_config_5A.json").exists() else {}
    if cfgA:
        pc = cfgA["preprocessing"]
        w(f"- Frames: {pc['frame_sampling']}; {pc['clip_frames']}.\n- Face: YuNet (OpenCV {pc['face_detector']['opencv']}, model sha256 "
          f"`{str(pc['face_detector']['model_sha256'])[:16]}…`), margin {FACE_MARGIN}, frames normalised to {NORM['frame_side']} px shorter "
          f"side, effective face resolution {NORM['face_px']} px, output {IMG_SIZE} px; degraded crops reuse the source boxes.\n"
          f"- Audio: {pc['audio']['spectrogram']}; {pc['audio']['source']}.\n- Model: {cfgA['model']['name']}, "
          f"{cfgA['model']['parameters']:,} parameters, {cfgA['model']['init']}.\n- Optimiser: AdamW (wd {WEIGHT_DECAY}), OneCycle; Stage 1 "
          f"{a.epochs1} epochs lr {a.lr1}; Stage 2 {a.epochs2} epochs lr {a.lr2}; {a.batch_size} clips per step.\n"
          f"- Model selection: {cfgA['model_selection']}.\n- Unavoidable differences: " + "; ".join(cfgA["unavoidable_differences"]) + ".\n")
    w("## 6. Results for all four variants (in-domain, seed mean ± SD across seeds, n = seeds)\n")
    for m, t in (("clean_auc", "Clean AUC"), ("degraded_auc_pooled", "Degraded AUC (pooled)"), ("worst_type_auc", "Worst-case type AUC"),
                 ("prediction_flip_rate_pct", "Prediction flip rate %"), ("auc_hard_pooled", "Hard-degradation AUC")):
        w(f"\n**{t}**\n\n" + seedtab(m))
    w("\n## 7. Results across seeds\n")
    if len(sc):
        x = sc[(sc.setting == "in-domain") & sc.metric.isin(["clean_auc", "degraded_auc_pooled", "mean_top20_iou [hard]"])]
        w(md_table(x[[c for c in x.columns if c in ("train_on", "variant", "metric", "n_seeds") or c.startswith("seed_") and c != "seeds"]]))
    w("\n## 8. Per-degradation detection robustness\n")
    if len(degm):
        x = degm[(degm.setting == "in-domain") & (degm.severity == "all")].groupby(["train_on", "variant", "degradation"]).auc.mean().unstack("variant").round(3)
        w(md_table(x.reset_index()))
        w("\nBrightness and saturation by severity: figures/fig_brightness_saturation_by_severity.png; full table "
          "metrics/degradation_metrics_test5.csv (sample counts per cell included).\n")
    w("## 9. Explanation stability and limitations\n")
    w(seedtab("mean_top20_iou [hard]") + "\n\nAll types:\n\n" + seedtab("mean_top20_iou [all]") + "\n\nExplanation changed while prediction "
      "stable (%):\n\n" + seedtab("explanation_changed_given_stable_pct [all]"))
    w(f"\n\n{summary['localization']} The clean-to-clean reference IoU (same video, different frame subset) in "
      "metrics/gradcam_summary_test5.csv gives the stability expected without any degradation.\n")
    w("## 10. Cross-dataset generalization\n")
    w("Clean AUC:\n\n" + seedtab("clean_auc", "cross-dataset") + "\n\nDegraded AUC:\n\n" + seedtab("degraded_auc_pooled", "cross-dataset"))
    if len(cross):
        x = cross[cross.setting == "cross-dataset"].groupby(["train_on", "test_dataset", "variant", "condition"])[[
            "auc", "balanced_accuracy", "error_rate_real_fpr", "error_rate_fake_fnr", "predicted_fake_rate", "mean_p_fake_real", "mean_p_fake_fake"]].mean().round(3)
        w("\n\nClass-wise errors and prediction distributions (seed means):\n\n" + md_table(x.reset_index()))
        y = cross[cross.condition.str.startswith("ablation_")].groupby(["train_on", "test_dataset", "variant", "condition"])[[
            "auc", "auc_normal_same_videos", "flip_rate_vs_normal_pct"]].mean().round(3)
        w("\n\nAudio masking / modality ablation (BioDeepAV test videos with audio; FF++ has no audio):\n\n" + md_table(y.reset_index()))
    w("\n\nCross-dataset generalisation is claimed only where the cross-dataset AUC is clearly above 0.5 across seeds.\n")
    w("## 11. Statistical uncertainty\n")
    w("Sample uncertainty: cluster bootstrap over source groups (paired: same test videos and replicates for both models) of the "
      "seed-averaged difference. Seed uncertainty: per-seed differences, their SD and sign agreement. A difference is called "
      "'supported' only if the 95 % CI excludes 0 AND all seeds agree in sign (>= 2 seeds).\n")
    if len(stt):
        w(md_table(stt[stt.setting == "in-domain"][["train_on", "comparison", "metric", "n_seeds", "mean_diff", "sd_diff_across_seeds",
                                                     "ci95_low", "ci95_high", "verdict"]], nd=4, max_rows=120))
    w("\n## 12. Failure cases\n")
    if len(rob):
        x = rob[rob.setting == "in-domain"].groupby(["train_on", "variant"]).agg(worst_type=("worst_type", lambda s: Counter(s).most_common(1)[0][0]),
                                                                                worst_type_auc=("worst_type_auc", "mean"),
                                                                                worst_type_severity=("worst_type_severity", lambda s: Counter(s).most_common(1)[0][0]),
                                                                                flip_rate=("prediction_flip_rate_pct", "mean"))
        w(md_table(x.reset_index()))
        w("\nPer-video flips: metrics/predictions/*.csv.gz; Grad-CAM pairs with changed explanations: metrics/gradcam_stability_test5.csv.\n")
    w("## 13. Comparison with the historical Test 4 results\n")
    w("**Different evaluation splits** (Test 4: seed 42, its own grouping; Test 5: seed {} with stricter DFD actor grouping and "
      "content-duplicate merging). This is NOT a controlled paired comparison; the Test 4 model was never evaluated on the Test 5 "
      "test set (its training data overlaps that test set).\n".format(a.data_seed))
    if t4.get("suite_comparison"):
        x = pd.DataFrame(t4["suite_comparison"])
        cols = [c for c in x.columns if c in ("run", "train_on") or any(k in c for k in ("(in-domain)_original_auc", "(in-domain)_degraded_auc",
                                                                                          "(cross)_original_auc", "prediction_flip", "gradcam_top20_iou",
                                                                                          "explanation_changed"))]
        w(md_table(x[cols]))
    w("\n## 14. Hypothesis verdicts\n")
    w(md_table(pd.DataFrame(verdicts)[["hypothesis", "comparison", "verdict", "evidence"]], max_rows=10))
    w("\n## 15. Recommended thesis conclusions\n")
    w("- Report only effects marked 'supported' above; describe the rest as inconclusive at this sample size and seed count.\n"
      "- Explanation stability is not explanation correctness: no localisation ground truth exists in this data.\n"
      "- If 5B-5D do not beat 5A, state that hard-degradation weighting did not improve robustness under this protocol.\n"
      "- Cross-dataset numbers indicate whether the detector transfers; do not claim generalisation where AUC ≈ 0.5.\n"
      "- Test 4 vs Test 5 differences mix method and split changes and must be described as such.\n")
    w("\n---\nFiles: metrics_test5.csv, degradation_metrics_test5.csv, robustness_test5.csv, gradcam_stability_test5.csv, "
      "gradcam_summary_test5.csv, cross_dataset_test5.csv, seed_comparison.csv, statistics_test5.csv (metrics/); checkpoint hashes "
      "(checkpoints/checkpoint_hashes.csv); configs/experiment_config_5A-5D.json; reports/leakage_audit_report.json; "
      "splits/split_integrity_report.json.\n")
    (P.reports / "TEST5_INDEPENDENT_REPORT.md").write_text("\n".join(L), encoding="utf-8")
    print(f"  report -> {P.reports / 'TEST5_INDEPENDENT_REPORT.md'}")


# =====================================================================================
# SELF-TEST (no data)
# =====================================================================================
def phase_selftest(a, P: Paths) -> bool:
    print("\n" + "=" * 100 + "\n PHASE SELFTEST (synthetic data only)\n" + "=" * 100)
    res = {}
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def check(name, fn):
        try:
            out = fn()
            res[name] = dict(ok=bool(out[0]) if isinstance(out, tuple) else bool(out), detail=out[1] if isinstance(out, tuple) else "")
        except Exception as e:
            res[name] = dict(ok=False, detail=f"{type(e).__name__}: {e}")
        print(f"  [{'OK ' if res[name]['ok'] else 'FAIL'}] {name} {res[name]['detail']}")

    def arch():
        torch.manual_seed(0)
        m = AudioVisualDeepFakeCNN().to(dev)
        v = torch.randn(2, CLIP_LEN, 3, IMG_SIZE, IMG_SIZE, device=dev)
        x = torch.randn(2, 1, 128, 128, device=dev)
        o = m(v, x, a_mask=torch.ones(2, device=dev))
        lo, z, act = m.forward_train(v, x, torch.ones(2, device=dev))
        n = sum(p.numel() for p in m.parameters())
        return o.shape == (2, 2) and z.shape == (2, 256) and act.shape[0] == 2 * CLIP_LEN, f"params {n:,}, act {tuple(act.shape)}"

    def arch_vs_test4():
        if not TEST4_SCRIPT.exists():
            return True, "Test 4 script absent - skipped"
        import importlib.util
        spec = importlib.util.spec_from_file_location("t4_arch_check_only", TEST4_SCRIPT)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # code-level import for the signature comparison only; no weights, caches or splits are loaded
        s4 = architecture_signature(mod.AudioVisualDeepFakeCNN())
        s5 = architecture_signature(AudioVisualDeepFakeCNN())
        sys.modules.pop("t4_arch_check_only", None)
        return s4 == s5, f"signature {'identical' if s4 == s5 else 'DIFFERENT'} ({s5[:12]}), random init only"

    def init_independent():
        torch.manual_seed(42)
        a1 = weights_sha256(AudioVisualDeepFakeCNN().state_dict())
        torch.manual_seed(123)
        a2 = weights_sha256(AudioVisualDeepFakeCNN().state_dict())
        torch.manual_seed(42)
        a3 = weights_sha256(AudioVisualDeepFakeCNN().state_dict())
        return a1 != a2 and a1 == a3, "seeds give distinct, reproducible initialisations"

    def cams():
        m = AudioVisualDeepFakeCNN().to(dev)
        v = torch.randn(3, CLIP_LEN, 3, 64, 64, device=dev)
        g = grad_cam(m, v, torch.zeros(3, 1, 128, 128, device=dev), torch.zeros(3, device=dev))
        m.train()
        lo, z, act = m.forward_train(v, torch.zeros(3, 1, 128, 128, device=dev), torch.zeros(3, device=dev))
        c = soft_gradcam(lo, act, 3, CLIP_LEN)
        l = cam_consistency_loss(c[:2].detach(), c[1:]) + feature_consistency_loss(z[:2], z[1:])
        l.backward()
        gn = sum(float(p.grad.abs().sum()) for p in m.parameters() if p.grad is not None)
        r, iou, sh = cam_metrics(g["spatial"][0], g["spatial"][0])
        return g["spatial"].shape == (3, 64, 64) and gn > 0 and abs(iou - 1) < 1e-6 and sh < 1e-6, f"double-backprop grad {gn:.3g}, self IoU {iou:.2f}"

    def wauc():
        rng = np.random.default_rng(0)
        y = rng.integers(0, 2, 300)
        p = np.round(rng.random(300) + 0.3 * y, 2)
        w = WeightedAUC(y, p)(np.ones((1, 300)))[0]
        W = rng.integers(0, 3, 300).astype(float)
        ww = WeightedAUC(y, p)(W[None])[0]
        rep = np.repeat(np.arange(300), W.astype(int))
        return abs(w - roc_auc_score(y, p)) < 1e-9 and abs(ww - roc_auc_score(y[rep], p[rep])) < 1e-9, f"{w:.4f} == sklearn"

    def split_ok():
        groups = {f"g{i}": list(range(i * 3, i * 3 + 1 + i % 4)) for i in range(200)}
        strat = {g: ("D", i % 2) for i, g in enumerate(groups)}
        s = make_splits(groups, strat, 7)
        n = Counter()
        for g, sp in s.items():
            n[sp] += len(groups[g])
        tot = sum(n.values())
        return len(s) == len(groups) and abs(n["test"] / tot - 0.2) < 0.05, dict(n)

    def uf_tokens():
        t1 = identity_tokens("FF++", "Deepfakes", 1, "000_003", "actor")[0]
        t2 = identity_tokens("FF++", "original", 0, "003", "actor")[0]
        t3 = identity_tokens("FF++", "DeepFakeDetection", 1, "01_02__x__AB", "actor")[0]
        t4 = identity_tokens("BioDeepAV", "real", 0, "--Y9imYnfBw_0000_S1_E2_L1_T1_R1_B1", "actor")[0]
        return set(t1) & set(t2) and t3 == ["dfd_actor:01", "dfd_actor:02"] and t4 == ["bio_yt:--Y9imYnfBw"], f"{t1} {t3} {t4}"

    def neardup():
        rng = np.random.default_rng(1)
        base = (rng.random((16, 64, 64)) * 255).astype(np.uint8)
        a_ = np.repeat(base[..., None], 3, -1)
        b_ = np.clip(a_.astype(int) + 3, 0, 255).astype(np.uint8)
        c_ = np.repeat((rng.random((16, 64, 64)) * 255).astype(np.uint8)[..., None], 3, -1)
        fa, fb, fc = clip_fingerprint(a_), clip_fingerprint(b_), clip_fingerprint(c_)
        H = np.stack([fa[2], fb[2], fc[2]])
        pairs = near_dup_pairs(H, np.ones(3, bool), NEARDUP_STRONG_MEAN)
        return [(i, j) for i, j, _, _ in pairs] == [(0, 1)] and ncc(fa[3], fb[3]) > 0.95 > ncc(fa[3], fc[3]), f"pairs {pairs}"

    def face_pipeline():
        tmp = P.logs / "selftest_video.mp4"
        wr = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), 25, (320, 240))
        for t in range(40):
            fr = np.full((240, 320, 3), 90, np.uint8)
            cv2.circle(fr, (160 + t % 5, 120), 50, (180, 160, 150), -1)
            wr.write(fr)
        wr.release()
        clip, info = extract_face_clip(tmp, NUM_FRAMES, IMG_SIZE, FaceDetector(FACE_MODEL), FACE_MARGIN, NORM)
        clip2, _ = extract_face_clip(tmp, NUM_FRAMES, IMG_SIZE, FaceDetector(FACE_MODEL), FACE_MARGIN, NORM)
        ref, _ = extract_face_clip(tmp, NUM_FRAMES, IMG_SIZE, None, FACE_MARGIN, NORM, ref=(np.tile([[0.5, 0.5, 0.6]], (NUM_FRAMES, 1)).astype(np.float32), False))
        tmp.unlink(missing_ok=True)
        ok = clip is not None and clip.shape == (NUM_FRAMES, IMG_SIZE, IMG_SIZE, 3) and np.array_equal(clip, clip2) and ref.shape == clip.shape
        return ok, f"shape {None if clip is None else clip.shape}, fallback {info['fallback']}, deterministic"

    def label_free():
        bad = [f.__name__ for f in (extract_face_clip, crop_face, face_bottleneck, load_audio_waveform, waveform_to_spec)
               if any("label" in p for p in inspect.signature(f).parameters)]
        return not bad, "no preprocessing function takes a label"

    def audio():
        w = np.sin(np.linspace(0, 2000, AUDIO_SAMPLES)).astype(np.float32) * 0.3
        s = waveform_to_spec(w)
        dd = [degrade_audio(w, d, 2, np.random.default_rng(0)).shape == w.shape for d in AUDIO_DEGRADATIONS]
        return s.shape == (1, 128, 128) and all(dd) and np.abs(s).max() <= 1.0001, f"spec {s.shape}"

    def guard_test():
        saved, _ACCESS["log"] = _ACCESS["log"], None  # synthetic probe: not written to the run's access log
        try:
            try:
                guard(TEST4_DIR / "cache" / "x.npy", "test5_cache")
                return False, "not blocked"
            except ForbiddenArtifactError:
                pass
            guard(TEST4_DIR / "suite_comparison.csv", "historical_reference_readonly:selftest")
            return True, "Test 4 cache path blocked; historical read-only allowed"
        finally:
            _ACCESS["log"] = saved

    def hard_sampling():
        class Dm:
            pass
        D = Dm()
        types = ["blur", "brightness", "compression", "contrast", "frame_drop", "motion_blur", "noise", "saturation"]
        D.deg_type = np.array([types[j % 8] for j in range(800)])
        D.deg_level = np.array([j % 3 for j in range(800)])
        D.deg_src = np.array([j // 8 for j in range(800)])
        D.versions = {s: np.arange(s * 8, s * 8 + 8) for s in range(100)}
        t = Trainer.__new__(Trainer)
        t.a, t.D = argparse.Namespace(hard_share=0.5, hard_w=parse_args([]).hard_w), D
        t.cell = np.array(["X|real" if s < 50 else "X|fake" for s in range(100)])
        t.hard_pool = defaultdict(list)
        for j in range(800):
            t.hard_pool[(t.cell[D.deg_src[j]], D.deg_type[j])].append(j)
        t.hard_pool = {k: np.array(v) for k, v in t.hard_pool.items()}
        rng = np.random.default_rng(0)
        cnt, cls = Counter(), Counter()
        for _ in range(20000):
            s = int(rng.integers(100))
            s2, j, _ = t.choose_deg(s, rng, True, 2)
            cnt[D.deg_type[j]] += 1
            cls[(t.cell[s] == t.cell[s2])] += 1
        share = {k: v / 20000 for k, v in cnt.items()}
        exp = {"brightness": 0.5 * 0.3 + 0.5 / 8, "saturation": 0.5 * 0.3 + 0.5 / 8, "contrast": 0.1 + 0.0625, "blur": 0.1 + 0.0625, "noise": 0.0625}
        ok = all(abs(share[k] - v) < 0.015 for k, v in exp.items()) and cls[True] == 20000
        return ok, {k: round(v, 3) for k, v in sorted(share.items())}

    for name, fn in [("architecture forward/shapes", arch), ("architecture signature == documented Test 4 family", arch_vs_test4),
                     ("independent seeded initialisation", init_independent), ("Grad-CAM + consistency losses", cams),
                     ("weighted (bootstrap) AUC", wauc), ("grouped stratified split", split_ok), ("identity tokens", uf_tokens),
                     ("near-duplicate screen", neardup), ("face pipeline (synthetic video)", face_pipeline),
                     ("preprocessing is label-free", label_free), ("audio spectrogram + degradations", audio),
                     ("artifact guard", guard_test), ("hard-degradation sampling shares + class preservation", hard_sampling)]:
        check(name, fn)
    ok = all(v["ok"] for v in res.values())
    write_json(P.data_audit / "selftest_report.json", dict(created=now(), passed=ok, checks=res, environment=environment_info()))
    print(f"  SELFTEST {'PASSED' if ok else 'FAILED'}")
    return ok


# =====================================================================================
# STATUS + MAIN
# =====================================================================================
def phase_status(a, P: Paths):
    items = [("inventory", P.inventory), ("clean caches / fingerprints", P.data_audit / "fingerprints_clean.npz"),
             ("split (originals)", P.split_originals), ("split manifest (with degraded)", P.split_manifest), ("gate", P.gate),
             ("predictions", next(iter(P.pred_dir.glob("*.csv.gz")), None) if P.pred_dir.exists() else None),
             ("report", P.reports / "TEST5_INDEPENDENT_REPORT.md")]
    for n, p in items:
        print(f"  {n:34s} {'done' if p is not None and Path(p).exists() else '-'}")
    if P.gate.exists():
        print(f"  gate status: {read_json(P.gate, 'test5_gate').get('gate')}")
    runs = available_runs(a, P) if P.checkpoints.exists() else []
    print(f"  checkpoints: {len(runs)} ({', '.join(r['run_id'] for r in runs[:40])})")


def phase_run(a, P: Paths):
    """One command for the whole experiment: selftest -> prepare (skipped when a valid gate PASS exists) -> leakage gate ->
    5A, 5B, 5C, 5D trained, evaluated and explained seed by seed (both training sets), with the analysis and report
    refreshed after every seed. Stops before any training if the gate fails. Resumable: finished runs are skipped."""
    print("\n" + "=" * 100 + f"\n RUN ALL VARIANTS {a.variant_list} | seeds {a.seed_list} | training sets {a.train_list}\n" + "=" * 100)
    if not phase_selftest(a, P):
        raise SystemExit("[STOP] selftest failed")
    gate_ok = False
    if P.gate.exists() and P.split_manifest.exists():
        g = read_json(P.gate, "test5_gate")
        gate_ok = g.get("gate") == "PASS" and sha256_file(P.split_manifest) == g.get("split_manifest_sha256")
    if not gate_ok:
        phase_audit(a, P)
        phase_preprocess(a, P)
        phase_split(a, P)
        phase_degraded(a, P)
        if not phase_gate(a, P):
            raise SystemExit(f"[STOP] leakage gate FAILED - no training. Diagnostic report: {P.leakage}")
    D = T5Data(P)
    for k, seed in enumerate(a.seed_list, 1):
        print("\n" + "#" * 100 + f"\n SEED {seed} ({k}/{len(a.seed_list)}): training {', '.join(a.variant_list)} on "
              f"{', '.join(a.train_list)}\n" + "#" * 100)
        phase_train(a, P, D, seeds=[seed])
        phase_evaluate(a, P, D)
        phase_xai(a, P, D)
        phase_analyze(a, P)
        phase_report(a, P)
        print(f"\n  seed {seed} done - interim report (seeds so far: {a.seed_list[:k]}): {P.reports / 'TEST5_INDEPENDENT_REPORT.md'}")
    print("\n  ALL RUNS FINISHED.")


def main(argv=None):
    a = parse_args(argv)
    lower_process_priority()
    torch.set_num_threads(max(1, min(4, (os.cpu_count() or 4) // 2)))  # STFT/augmentation run inside worker threads
    P = Paths(a.root)
    P.make()
    _ACCESS["log"], _ACCESS["phase"] = P.access_log, a.phase
    sys.stdout = Tee(P.logs / f"{a.phase}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    print(f"TEST 5 (independent) | phase {a.phase} | root {P.root} | data seed {a.data_seed} | seeds {a.seed_list} | variants "
          f"{a.variant_list} | train sets {a.train_list} | {now()}")
    write_json(P.logs / f"environment_{a.phase}.json", environment_info())
    ph = a.phase
    if ph == "run":
        return phase_run(a, P)
    if ph == "selftest":
        sys.exit(0 if phase_selftest(a, P) else 1)
    if ph == "status":
        return phase_status(a, P)
    if ph in ("prepare", "audit"):
        phase_audit(a, P)
    if ph in ("prepare", "preprocess"):
        phase_preprocess(a, P)
    if ph in ("prepare", "split"):
        phase_split(a, P)
    if ph in ("prepare", "degraded"):
        phase_degraded(a, P)
    if ph in ("prepare", "gate"):
        ok = phase_gate(a, P)
        print(f"\n  Pre-training report: {P.reports / 'PRE_TRAINING_CHECKS.md'}")
        if ph == "prepare":
            print("  STOP: preparation finished. Review the checks, then run --phase all to train." if ok else "  STOP: gate failed.")
        sys.exit(0 if ok else 2)
    if ph in ("train", "evaluate", "xai", "all"):
        require_gate(P)
        D = T5Data(P)
        if ph in ("train", "all"):
            phase_train(a, P, D)
        if ph in ("evaluate", "all"):
            phase_evaluate(a, P, D)
        if ph in ("xai", "all"):
            phase_xai(a, P, D)
    if ph in ("analyze", "all"):
        phase_analyze(a, P)
    if ph in ("report", "all", "analyze"):
        phase_report(a, P)


if __name__ == "__main__":
    main()
