"""
16-10-26 (TEST 6): MULTI-STREAM SPATIO-TEMPORAL-FREQUENCY AUDIO-VISUAL DETECTOR WITH
           ROBUST, EXPLANATION-CONSISTENT AND SHORTCUT-ADVERSARIAL TRAINING
=================================================================================================

Thesis: Robust Deepfake Detection and Explainability Under Real-World Video Degradation

Why Test 6 (what Test 5 left open)
  * The AudioVisualDeepFakeCNN (585 k parameters, frame-wise CNN + mean/max pooling) reached a ceiling: no Test 5
    variant reliably beat the plain degraded-training baseline 5A.
  * Fixed hard-type sampling (5B-5D) starved other types: 5D lost severe-noise AUC (0.977 -> 0.665).
  * Grad-CAM consistency was the only replicated benefit (FF++ only).
  * Shortcuts remained (BioDeepAV face resolution, audio presence); cross-dataset AUC stayed at chance.
  * Probabilities were badly calibrated (BioDeepAV validation thresholds 0.01-0.12) and the selected epoch was noisy.

New architecture: MSTF-AV (Multi-Stream Spatio-Temporal-Frequency Audio-Visual network, ~2.3 M parameters)
  1. RGB spatial stream      residual CNN with squeeze-excitation on every 128x128 face frame
  2. Frequency/noise stream  fixed SRM high-pass residual filters + residual CNN (blending / up-sampling traces)
  3. Cross-stream gate       the noise stream gates the RGB stream at 16x16 (learned, per channel and position)
  4. Last spatial block      8x8 map = Grad-CAM layer (same CAM recipe and metrics as Tests 3-5)
  5. Temporal transformer    frame tokens + frame-difference tokens, learned positions, CLS token, 2 layers x 4 heads
  6. Audio stream            residual spectrogram CNN -> 16 time tokens
  7. Cross-modal attention   video CLS attends to audio tokens; a learned gate (x audio-present mask) adds audio
  8. Heads                   classifier; projection head (supervised contrastive); class-conditional adversaries
                             (gradient reversal) for face-resolution tercile and, in joint training, dataset

Training variants (Stage 2; Stage 1 = clean originals, CE only, shared by 6A-6D within a training set and seed)
  6A  MSTF-AV + Test 5 baseline recipe (unpaired 16 original + 16 degraded clips, natural mixture, CE)
  6B  6A + paired original/degraded batches + Grad-CAM consistency (0.05) + supervised contrastive consistency (0.10)
  6C  6B + adaptive group-DRO degradation sampling (loss-driven type weights with a 50 % uniform floor, so no type
      is starved) + EMA weights + label smoothing 0.05
  6D  6C + class-conditional shortcut adversaries (gradient reversal; resolution tercile, + dataset in joint
      training; lambda ramped to 0.10)
  All variants: AMP, temperature scaling on validation, threshold = validation balanced-accuracy optimum on the
  calibrated probability.

Training sets: BioDeepAV, FF++ (as Test 5) and Joint (both datasets; validation score = macro over datasets).

Data: Test 6 does NOT rebuild data. It re-uses the PREPARED Test 5 inputs read-only (split manifest, face-crop and
audio caches, XAI pair list), after verifying the Test 5 gate (PASS), the split-manifest SHA-256 and cache
completeness. Same split, same test videos and same seeds as Test 5 -> Test 6 vs Test 5 is a PAIRED comparison.
Test 5 checkpoints are never loaded (guard()); Test 5 predictions / Grad-CAM rows are read only in --phase analyze
as the baseline (T5-5A, T5-5C).

Hypotheses (paired cluster bootstrap over source groups + sign agreement across seeds)
  H1  6A - T5-5A   the new architecture improves robustness (degraded, worst-type AUC)
  H2  6B - 6A      consistency training improves explanation stability (Grad-CAM IoU, flips)
  H3  6C - 6B      DRO sampling + EMA improve worst-case robustness without starving types
  H4  6D - 6C      shortcut adversaries reduce shortcuts (cross-dataset AUC, resolution correlation, calibration)
  H5  6D - T5-5A   full Test 6 vs the Test 5 baseline

Phases
  python 161026.py --phase selftest      # unit tests, no data
  python 161026.py --phase link          # verify + link the Test 5 prepared data (no training)
  python 161026.py --phase run           # selftest -> link -> per seed: train, evaluate, xai, analyze, report
  python 161026.py --smoke               # tiny end-to-end pipeline check in a separate root
  Individual phases: train, evaluate, xai, analyze, report, status. Every phase is resumable (power-cut safe).
  Fewer runs: --train-on BioDeepAV,FF++  |  --variants 6A,6D  |  --seeds 42
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import inspect
import io
import json
import math
import os
import platform
import random
import shutil
import subprocess
import sys
import threading
import time
import warnings
from collections import Counter, defaultdict, deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.signal import butter, resample_poly, sosfiltfilt
from scipy.stats import spearmanr
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import confusion_matrix, roc_auc_score  # noqa: E402

warnings.filterwarnings("ignore")
# Grad-CAM consistency differentiates through a gradient (double backward); the fused flash / memory-efficient /
# cuDNN attention kernels have no double backward, so attention runs on the math kernel (short sequences, cheap).
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_cudnn_sdp(False)

# =====================================================================================
# CONFIGURATION
# =====================================================================================
SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent
DEFAULT_ROOT = SCRIPT_DIR / "run_16-10-26_test6"
SMOKE_ROOT = SCRIPT_DIR / "run_16-10-26_test6_smoke"
T5_ROOT = SCRIPT_DIR / "run_test5_independent"
# checkpoints of earlier tests must never be loaded (Test 6 trains from fresh initialisation)
FORBIDDEN_ARTIFACT_ROOTS = [SCRIPT_DIR / d for d in ("run_6-10-26", "run_1-10-26", "run_30-9-26", "run_test5_hard_explanation",
                                                     "run_6-10-26_smoke", "run_1-10-26_smoke")] + [T5_ROOT / "checkpoints"]

TRAIN_SEEDS = "42,123,2026"                 # same seeds as Test 5 (paired seed-wise comparison)
IMG_SIZE, NUM_FRAMES, CLIP_LEN = 128, 16, 8
AUDIO_SR, AUDIO_SAMPLES = 16000, 48000
AUDIO_PROBS = (0.50, 0.25, 0.25)            # clean / degraded / none (as Test 5)
AUDIO_DEGRADATIONS = ["noise", "volume", "lowpass", "resample", "dropout", "silence"]
AUDIO_DEG_LEVELS = {"noise": (20.0, 10.0, 0.0), "volume": (-10.0, -20.0, -30.0), "lowpass": (4000, 2000, 1000),
                    "resample": (11025, 8000, 4000), "dropout": (0.10, 0.25, 0.50), "silence": (1, 1, 1)}

BATCH_SIZE, LR1, LR2, EPOCHS1, EPOCHS2, WEIGHT_DECAY = 32, 1e-3, 3e-4, 8, 10, 5e-4
WARMUP_FRAC = 0.15
MAX_OVERSAMPLE = 4.0
EMB_DIM = 192
LAMBDA_CAM, LAMBDA_SUPCON, SUPCON_TAU = 0.05, 0.10, 0.10
LAMBDA_ADV, N_RES_BINS = 0.10, 3
LABEL_SMOOTH = 0.05
EMA_DECAY = 0.999
DRO_ETA, DRO_FLOOR, DRO_MOMENTUM = 2.0, 0.50, 0.5
GRAD_CLIP = 5.0
THRESHOLD_GRID = (0.01, 0.99, 0.01)
TEMP_GRID = np.exp(np.linspace(np.log(0.05), np.log(20.0), 241))
ECE_BINS = 15
EXPLANATION_CHANGE_IOU = 0.5
XAI_CHUNK = 8                       # Grad-CAM pairs per batch (peak memory)
N_BOOT = 1000
MIN_CLUSTERS = 10
MIN_PER_CLASS_AUC = 5
HARD_TYPES = ["brightness", "saturation", "contrast", "blur"]   # reported subset (Test 5 definition)
SEV_LEVEL = {"mild": 0, "medium": 1, "severe": 2}
DATASETS = ["BioDeepAV", "FF++"]
NO_WINDOW = subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0
METRIC_KEYS = ["auc", "accuracy", "balanced_accuracy", "precision", "recall", "sensitivity", "specificity", "f1"]

VARIANTS = {
    "6A": dict(paired=False, lambda_cam=0.0, lambda_supcon=0.0, dro=False, ema=False, label_smooth=0.0, lambda_adv=0.0,
               description="MSTF-AV + Test 5 baseline recipe (unpaired 50/50 original/degraded, natural mixture, CE)"),
    "6B": dict(paired=True, lambda_cam=LAMBDA_CAM, lambda_supcon=LAMBDA_SUPCON, dro=False, ema=False, label_smooth=0.0, lambda_adv=0.0,
               description="6A + paired batches + Grad-CAM consistency + supervised contrastive consistency"),
    "6C": dict(paired=True, lambda_cam=LAMBDA_CAM, lambda_supcon=LAMBDA_SUPCON, dro=True, ema=True, label_smooth=LABEL_SMOOTH, lambda_adv=0.0,
               description="6B + adaptive group-DRO degradation sampling (uniform floor) + EMA weights + label smoothing"),
    "6D": dict(paired=True, lambda_cam=LAMBDA_CAM, lambda_supcon=LAMBDA_SUPCON, dro=True, ema=True, label_smooth=LABEL_SMOOTH,
               lambda_adv=LAMBDA_ADV, description="6C + class-conditional shortcut adversaries (resolution tercile; dataset in joint training)"),
}
T5_BASELINES = {"5A": "T5-5A", "5C": "T5-5C"}
VARIANT_ORDER = ["S1", "T5-5A", "T5-5C", "6A", "6B", "6C", "6D"]
VARIANT_COLOR = {"S1": "#8a8984", "T5-5A": "#b9c4d6", "T5-5C": "#a8d5c2", "6A": "#2a78d6", "6B": "#1baf7a", "6C": "#eb6834", "6D": "#4a3aa7"}
COMPARISONS = [("6A", "T5-5A"), ("6B", "6A"), ("6C", "6B"), ("6D", "6C"), ("6D", "T5-5A"), ("6B", "T5-5C")]


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


def lower_process_priority() -> None:
    try:
        if sys.platform == "win32":
            import ctypes
            ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x00004000)
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
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if (np.isnan(o) or np.isinf(o)) else float(o)
    if isinstance(o, np.ndarray):
        return to_jsonable(o.tolist())
    return o


def replace_retry(src, dst, tries: int = 12) -> None:
    """os.replace that waits out short-lived locks on Windows (antivirus / indexer scanning a file just written)."""
    for k in range(tries):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if k == tries - 1:
                raise
            time.sleep(min(0.25 * 2 ** k, 5.0))


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(obj), f, indent=2)
    replace_retry(tmp, path)


def atomic_torch_save(obj, path: Path) -> None:
    path = Path(path)
    tmp = path.with_name(path.name + ".writing")
    with open(tmp, "wb") as f:
        torch.save(obj, f)
        f.flush()
        os.fsync(f.fileno())
    replace_retry(tmp, path)


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


def md_table(df: pd.DataFrame, nd: int = 3, max_rows: int = 80) -> str:
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


def code_sha256() -> str:
    return sha256_file(SCRIPT_PATH)


def environment_info() -> dict:
    return dict(python=sys.version.split()[0], platform=platform.platform(), pytorch=torch.__version__, cuda=torch.version.cuda,
                cudnn=torch.backends.cudnn.version() if torch.cuda.is_available() else None,
                gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, numpy=np.__version__, pandas=pd.__version__,
                script=str(SCRIPT_PATH), script_sha256=code_sha256(), recorded_at=now())


class Tee(io.TextIOBase):
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
    SUBDIRS = ("configs", "checkpoints", "logs", "metrics", "figures", "reports")

    def __init__(self, root: Path):
        self.root = Path(root)
        for d in self.SUBDIRS:
            setattr(self, d, self.root / d)
        self.link = self.reports / "test5_input_link.json"
        self.access_log = self.logs / "file_access_log.csv"
        self.pred_dir = self.metrics / "predictions"
        self.cam_dir = self.metrics / "gradcam_runs"
        self.hist_dir = self.logs / "training"

    def make(self):
        for d in self.SUBDIRS:
            getattr(self, d).mkdir(parents=True, exist_ok=True)
        for d in (self.pred_dir, self.cam_dir, self.hist_dir):
            d.mkdir(parents=True, exist_ok=True)


class T5Paths:
    """Read-only view of the prepared Test 5 run (inputs of Test 6)."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.split_manifest = self.root / "splits" / "split_manifest.csv"
        self.gate = self.root / "reports" / "gate_status.json"
        self.provenance = self.root / "cache" / "provenance.json"
        self.cache = self.root / "cache"
        self.xai_pairs = self.root / "manifests" / "xai_pairs.csv"
        self.pred_dir = self.root / "metrics" / "predictions"
        self.cam_dir = self.root / "metrics" / "gradcam_runs"


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
    """Every artifact read passes here and is logged. Checkpoints of earlier tests can never be loaded."""
    p = Path(path).resolve()
    forbidden = any(_is_under(p, r) for r in FORBIDDEN_ARTIFACT_ROOTS)
    log = _ACCESS["log"]
    if log is not None:
        with _ACCESS_LOCK:
            new = not log.exists()
            with open(log, "a", encoding="utf-8") as f:
                if new:
                    f.write("time,phase,purpose,path,blocked\n")
                f.write(f"{now()},{_ACCESS['phase']},{purpose},\"{p}\",{forbidden}\n")
    if forbidden:
        raise ForbiddenArtifactError(f"Test 6 must not load {p} ({purpose})")
    return p


def read_csv(path, purpose: str, **kw) -> pd.DataFrame:
    return pd.read_csv(guard(path, purpose), **kw)


def read_json(path, purpose: str):
    with open(guard(path, purpose), "r", encoding="utf-8") as f:
        return json.load(f)


# =====================================================================================
# CLI
# =====================================================================================
PHASES = ["run", "selftest", "link", "status", "train", "evaluate", "xai", "analyze", "report", "all"]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Test 6: MSTF-AV multi-stream detector with robust, consistent and shortcut-adversarial training")
    p.add_argument("--phase", default="run", choices=PHASES)
    p.add_argument("--root", default=None, help=f"Experiment root (default {DEFAULT_ROOT})")
    p.add_argument("--t5-root", default=str(T5_ROOT), help="Prepared Test 5 run (split + caches), read-only")
    p.add_argument("--seeds", default=TRAIN_SEEDS)
    p.add_argument("--variants", default="6A,6B,6C,6D")
    p.add_argument("--train-on", default="BioDeepAV,FF++,Joint")
    p.add_argument("--epochs1", type=int, default=EPOCHS1)
    p.add_argument("--epochs2", type=int, default=EPOCHS2)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="clips per step (paired: half originals, half degraded)")
    p.add_argument("--accum", type=int, default=1, help="gradient accumulation steps (use 2 with --batch-size 16 on 4 GB GPUs)")
    p.add_argument("--lr1", type=float, default=LR1)
    p.add_argument("--lr2", type=float, default=LR2)
    p.add_argument("--lambda-cam", type=float, default=LAMBDA_CAM)
    p.add_argument("--lambda-supcon", type=float, default=LAMBDA_SUPCON)
    p.add_argument("--lambda-adv", type=float, default=LAMBDA_ADV)
    p.add_argument("--dro-eta", type=float, default=DRO_ETA)
    p.add_argument("--dro-floor", type=float, default=DRO_FLOOR)
    p.add_argument("--ema-decay", type=float, default=EMA_DECAY)
    p.add_argument("--no-amp", action="store_true", help="disable mixed precision")
    p.add_argument("--n-boot", type=int, default=N_BOOT)
    p.add_argument("--max-steps", type=int, default=None, help="cap steps per epoch (debugging)")
    p.add_argument("--loader-threads", type=int, default=3)
    p.add_argument("--save-every-steps", type=int, default=25, help="crash-safe resume checkpoint every N steps")
    p.add_argument("--smoke", action="store_true", help="tiny end-to-end run in a separate root")
    p.add_argument("--smoke-per-cell", type=int, default=12)
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--gpu-max-temp", type=int, default=80)
    p.add_argument("--gpu-pause", type=int, default=120)
    a = p.parse_args(argv)
    if a.smoke:
        a.epochs1, a.epochs2 = min(a.epochs1, 1), min(a.epochs2, 2)
        a.n_boot = min(a.n_boot, 100)
        a.max_steps = a.max_steps or 3
        a.seeds = a.seeds if a.seeds != TRAIN_SEEDS else "42,123"
    root = Path(a.root) if a.root else (SMOKE_ROOT if a.smoke else DEFAULT_ROOT)
    if a.smoke and root.resolve() == DEFAULT_ROOT.resolve():
        raise SystemExit("--smoke must not write into the main Test 6 root")
    a.root = root
    a.t5 = T5Paths(Path(a.t5_root))
    a.seed_list = [int(s) for s in str(a.seeds).split(",") if s.strip()]
    a.variant_list = [v.strip().upper() for v in a.variants.split(",") if v.strip()]
    bad = [v for v in a.variant_list if v not in VARIANTS]
    if bad:
        raise SystemExit(f"unknown variants {bad}")
    alias = {"bio": "BioDeepAV", "biodeepav": "BioDeepAV", "ffpp": "FF++", "ff++": "FF++", "joint": "Joint", "both": "Joint"}
    a.train_list = [alias.get(t.strip().lower(), t.strip()) for t in a.train_on.split(",") if t.strip()]
    a.amp = (not a.no_amp) and torch.cuda.is_available()
    return a


def variant_config(a, v: str) -> dict:
    c = dict(VARIANTS[v])
    for k in ("lambda_cam", "lambda_supcon", "lambda_adv"):
        if c[k] > 0:
            c[k] = getattr(a, k)
    return c


# =====================================================================================
# ARCHITECTURE: MSTF-AV
# =====================================================================================
class ConvBNAct(nn.Sequential):
    def __init__(self, cin, cout, k=3, s=1):
        super().__init__(nn.Conv2d(cin, cout, k, s, k // 2, bias=False), nn.BatchNorm2d(cout), nn.SiLU(inplace=True))


class SqueezeExcite(nn.Module):
    def __init__(self, c, r=8):
        super().__init__()
        self.fc = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(c, max(8, c // r), 1), nn.SiLU(inplace=True),
                                nn.Conv2d(max(8, c // r), c, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.fc(x)


class ResBlock(nn.Module):
    """3x3 -> 3x3 residual block with squeeze-excitation; stride-2 shortcut by 1x1 conv."""

    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(cin, cout, 3, stride, 1, bias=False), nn.BatchNorm2d(cout), nn.SiLU(inplace=True),
                                  nn.Conv2d(cout, cout, 3, 1, 1, bias=False), nn.BatchNorm2d(cout), SqueezeExcite(cout))
        self.short = (nn.Identity() if cin == cout and stride == 1 else
                      nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout)))
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.body(x) + self.short(x))


def srm_kernels() -> torch.Tensor:
    """Three classic SRM high-pass residual filters (Fridrich & Kodovsky), 5x5, as [3,1,5,5]."""
    k1 = np.array([[0, 0, 0, 0, 0], [0, -1, 2, -1, 0], [0, 2, -4, 2, 0], [0, -1, 2, -1, 0], [0, 0, 0, 0, 0]], np.float32) / 4.0
    k2 = np.array([[-1, 2, -2, 2, -1], [2, -6, 8, -6, 2], [-2, 8, -12, 8, -2], [2, -6, 8, -6, 2], [-1, 2, -2, 2, -1]], np.float32) / 12.0
    k3 = np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 1, -2, 1, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]], np.float32) / 2.0
    return torch.from_numpy(np.stack([k1, k2, k3])[:, None])


class SRMFilter(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("k", srm_kernels())
        self.register_buffer("gray", torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1))

    def forward(self, x):                       # x in [-1, 1], [N,3,H,W] -> [N,3,H,W] residuals (truncated)
        g = (x * self.gray.to(x.dtype)).sum(1, keepdim=True)
        return torch.clamp(F.conv2d(g, self.k.to(x.dtype), padding=2) * 4.0, -1.0, 1.0)


class CrossStreamGate(nn.Module):
    """RGB features modulated by the noise-residual stream: rgb + sigmoid(W[rgb, frq]) * V frq."""

    def __init__(self, c):
        super().__init__()
        self.gate = nn.Sequential(nn.Conv2d(2 * c, c, 1), nn.Sigmoid())
        self.val = nn.Sequential(nn.Conv2d(c, c, 1, bias=False), nn.BatchNorm2d(c))

    def forward(self, rgb, frq):
        return rgb + self.gate(torch.cat([rgb, frq], 1)) * self.val(frq)


class SpatialFrequencyEncoder(nn.Module):
    def __init__(self, out_c=EMB_DIM):
        super().__init__()
        self.rgb_stem = ConvBNAct(3, 32, 3, 2)                                   # 64
        self.rgb1 = ResBlock(32, 48, 2)                                          # 32
        self.rgb2 = ResBlock(48, 96, 2)                                          # 16
        self.srm = SRMFilter()
        self.frq = nn.Sequential(ConvBNAct(3, 24, 3, 2), ResBlock(24, 48, 2), ResBlock(48, 96, 2))   # 16
        self.fuse = CrossStreamGate(96)
        self.rgb3 = ResBlock(96, out_c, 2)                                       # 8  <- Grad-CAM layer

    def forward(self, x):
        r = self.rgb2(self.rgb1(self.rgb_stem(x)))
        f = self.frq(self.srm(x))
        return self.rgb3(self.fuse(r, f))


class TemporalTransformer(nn.Module):
    def __init__(self, d=EMB_DIM, layers=2, heads=4, max_len=32):
        super().__init__()
        self.inp = nn.Linear(2 * d, d)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.pos = nn.Parameter(torch.zeros(1, max_len + 1, d))
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.trunc_normal_(self.cls, std=0.02)
        enc = nn.TransformerEncoderLayer(d, heads, 2 * d, dropout=0.1, activation="gelu", batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(enc, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)

    def forward(self, f):                       # f [B,T,d] frame tokens
        diff = torch.cat([torch.zeros_like(f[:, :1]), f[:, 1:] - f[:, :-1]], 1)
        x = self.inp(torch.cat([f, diff], -1))
        x = torch.cat([self.cls.expand(len(x), -1, -1), x], 1) + self.pos[:, :x.size(1) + 1]
        x = self.enc(x)
        return self.norm(x[:, 0] + x[:, 1:].mean(1))


class AudioEncoder(nn.Module):
    def __init__(self, d=EMB_DIM):
        super().__init__()
        self.net = nn.Sequential(ConvBNAct(1, 32, 3, 2), ResBlock(32, 64, 2), ResBlock(64, 128, 2))   # [B,128,16,16]
        self.proj = nn.Linear(128, d)

    def forward(self, x):                       # [B,1,128(freq),128(time)] -> [B,16,d] time tokens
        h = self.net(x).mean(2)                 # average over frequency
        return self.proj(h.transpose(1, 2))


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None


class MSTFAV(nn.Module):
    """Multi-Stream Spatio-Temporal-Frequency Audio-Visual detector."""

    def __init__(self, d=EMB_DIM, n_res_bins=N_RES_BINS, n_domains=2):
        super().__init__()
        self.d = d
        self.encoder = SpatialFrequencyEncoder(d)
        self.temporal = TemporalTransformer(d)
        self.audio = AudioEncoder(d)
        self.xattn = nn.MultiheadAttention(d, 4, dropout=0.1, batch_first=True)
        self.a_gate = nn.Sequential(nn.Linear(2 * d, d), nn.Sigmoid())
        self.a_proj = nn.Linear(d, d)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 128), nn.GELU(), nn.Dropout(0.3), nn.Linear(128, 2))
        self.proj = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 128))
        self.adv_res = nn.Sequential(nn.Linear(d + 2, 128), nn.ReLU(inplace=True), nn.Linear(128, n_res_bins))
        self.adv_dom = nn.Sequential(nn.Linear(d + 2, 128), nn.ReLU(inplace=True), nn.Linear(128, n_domains))

    @property
    def cam_module(self):
        return self.encoder.rgb3

    def encode_video(self, x):                  # [B,T,3,H,W] -> act [B*T,C,8,8], v [B,d]
        B, T = x.shape[:2]
        act = self.encoder(x.flatten(0, 1))
        return act, self.temporal(act.mean((2, 3)).view(B, T, -1))

    def fuse(self, v, x_a, a_mask):
        if not self.training and not bool((a_mask > 0).any()):
            # no audio in the batch: skip the branch. A model trained without audio (FF++) has audio-BN running_var ~0,
            # so its audio tokens overflow fp16 and 0 * inf = NaN would reach v. Training path unchanged (same RNG).
            return v
        tok = self.audio(x_a)
        ctx = self.xattn(v[:, None], tok, tok, need_weights=False)[0][:, 0]
        g = self.a_gate(torch.cat([v, ctx], 1)) * a_mask.view(-1, 1).to(v.dtype)
        return v + g * self.a_proj(ctx)

    def forward(self, x_v, x_a=None, mode="av", a_mask=None):
        ref = x_v if x_v is not None else x_a
        B = ref.size(0)
        if mode == "a_only":
            v = torch.zeros(B, self.d, device=ref.device, dtype=x_a.dtype)
        else:
            v = self.encode_video(x_v)[1]
        if mode == "v_only" or x_a is None:
            z = v
        else:
            am = a_mask if a_mask is not None else torch.ones(B, device=ref.device)
            z = self.fuse(v, x_a, am)
        return self.head(z)

    def forward_train(self, x_v, x_a, a_mask):
        act, v = self.encode_video(x_v)
        z = self.fuse(v, x_a, a_mask)
        return self.head(z), z, v, act

    def adversary(self, v, y, lam):
        h = torch.cat([GradReverse.apply(v, lam), F.one_hot(y, 2).to(v.dtype)], 1)
        return self.adv_res(h), self.adv_dom(h)


def architecture_signature(model: nn.Module) -> str:
    sd = model.state_dict()
    return sha256_text(type(model).__name__ + "\n" + "\n".join(f"{k}:{tuple(v.shape)}" for k, v in sd.items()))


def architecture_source_sha() -> str:
    objs = (ConvBNAct, SqueezeExcite, ResBlock, srm_kernels, SRMFilter, CrossStreamGate, SpatialFrequencyEncoder,
            TemporalTransformer, AudioEncoder, GradReverse, MSTFAV)
    return sha256_text("".join(inspect.getsource(o) for o in objs))


def weights_sha256(sd) -> str:
    h = hashlib.sha256()
    for k in sorted(sd):
        h.update(k.encode())
        h.update(sd[k].detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def to_video_tensor(arr: np.ndarray, device) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(arr)).permute(0, 1, 4, 2, 3).float().div_(127.5).sub_(1.0).to(device, non_blocking=True)


# =====================================================================================
# LOSSES, GRAD-CAM, CALIBRATION
# =====================================================================================
def soft_gradcam(logits, act, B: int, T: int, target_class: int = 1, create_graph: bool = True):
    g = torch.autograd.grad(logits[:, target_class].float().sum(), act, create_graph=create_graph, retain_graph=True)[0]
    cam = torch.relu((g.float().mean((2, 3), keepdim=True) * act.float()).sum(1))
    cam = cam.view(B, T, *cam.shape[-2:]).mean(1)
    return cam / (cam.flatten(1).amax(1).view(-1, 1, 1) + 1e-6)


def cam_consistency_loss(a, b):
    return (1.0 - torch.minimum(a, b).flatten(1).sum(1) / (torch.maximum(a, b).flatten(1).sum(1) + 1e-6)).mean()


def supcon_loss(z, y, tau: float = SUPCON_TAU):
    """Supervised contrastive loss (Khosla et al. 2020): positives = same label (incl. the paired version)."""
    z = F.normalize(z.float(), dim=1)
    with torch.autocast(z.device.type, enabled=False):         # fp32: under AMP the fp16 matmul cannot hold the -1e9 mask
        sim = z @ z.t() / tau
    n = len(z)
    eye = torch.eye(n, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(eye, -1e9)
    pos = (y[:, None] == y[None, :]) & ~eye
    logp = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    cnt = pos.sum(1)
    keep = cnt > 0
    if not keep.any():
        return sim.new_zeros(())
    return (-(logp * pos).sum(1)[keep] / cnt[keep]).mean()


def grad_cam(model, video, audio, a_mask, target_class: int = 1, threshold: float = 0.5, temperature: float = 1.0) -> Dict[str, np.ndarray]:
    store = {}

    def hook(_m, _i, o):
        store["a"] = o

    B, T, _, H, W = video.shape
    h = model.cam_module.register_forward_hook(hook)
    try:
        model.eval()
        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            logits = model(video, audio, mode="av", a_mask=a_mask)
            # autograd.grad, not a tensor hook: a hook closure referencing its own tensor forms a cycle through the C++
            # graph that Python cannot collect, leaking the activation + gradient on the GPU on every call
            G = torch.autograd.grad(logits[:, target_class].sum(), store["a"])[0]
    finally:
        h.remove()
    A, G = store.pop("a").detach().float(), G.detach().float()
    cam = torch.relu((G.mean((2, 3), keepdim=True) * A).sum(1))
    cam = F.interpolate(cam[:, None], size=(H, W), mode="bilinear", align_corners=False)[:, 0].view(B, T, H, W)
    temporal = cam.flatten(2).sum(-1)
    temporal = temporal / (temporal.sum(1, keepdim=True) + 1e-8)
    spatial = cam.mean(1).flatten(1)
    mn, mx = spatial.min(1, keepdim=True)[0], spatial.max(1, keepdim=True)[0]
    spatial = ((spatial - mn) / (mx - mn + 1e-8)).view(B, H, W)
    prob = torch.softmax(logits.detach().float() / temperature, 1)[:, 1]
    model.zero_grad(set_to_none=True)
    return dict(spatial=spatial.cpu().numpy(), temporal=temporal.cpu().numpy(), p=prob.cpu().numpy(),
                pred=(prob >= threshold).int().cpu().numpy())


def cam_metrics(a: np.ndarray, b: np.ndarray, top: float = 0.2) -> Tuple[float, float, float]:
    r = float(np.corrcoef(a.ravel(), b.ravel())[0, 1]) if a.std() > 1e-8 and b.std() > 1e-8 else float("nan")
    ta, tb = a >= np.quantile(a, 1 - top), b >= np.quantile(b, 1 - top)
    iou = float((ta & tb).sum() / max(1, (ta | tb).sum()))
    yy, xx = np.mgrid[0:a.shape[0], 0:a.shape[1]]

    def com(c):
        w = c.sum() + 1e-8
        return np.array([(yy * c).sum() / w, (xx * c).sum() / w])

    return r, iou, float(np.linalg.norm(com(a) - com(b)) / a.shape[0])


def fit_temperature(logit_diff: np.ndarray, y: np.ndarray) -> float:
    """Temperature T minimising the validation NLL of sigmoid(logit_diff / T) (grid search, robust)."""
    x = np.asarray(logit_diff, np.float64)
    y = np.asarray(y, np.float64)
    best, bt = np.inf, 1.0
    for t in TEMP_GRID:
        z = x / t
        nll = np.mean(np.logaddexp(0, z) - y * z)
        if nll < best:
            best, bt = nll, float(t)
    return bt


def ece_score(y, p, bins: int = ECE_BINS) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    if not len(y):
        return float("nan")
    b = np.minimum((p * bins).astype(int), bins - 1)
    tot = 0.0
    for k in range(bins):
        m = b == k
        if m.any():
            tot += m.sum() * abs(p[m].mean() - y[m].mean())
    return float(tot / len(y))


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
    lo, hi, st = THRESHOLD_GRID
    best = None
    for t in np.round(np.arange(lo, hi + st / 2, st), 4):
        m = compute_metrics(y, p, float(t))
        key = (np.nan_to_num(m["balanced_accuracy"], nan=-1), np.nan_to_num(m["f1"], nan=-1), -abs(t - 0.5))
        if best is None or key > best[0]:
            best = (key, float(t), m)
    return dict(threshold=best[1], balanced_accuracy=best[2]["balanced_accuracy"], f1=best[2]["f1"], n=best[2]["n"])


class WeightedAUC:
    """AUC for many bootstrap weight vectors at once (tie-block formulation)."""

    def __init__(self, y, p):
        self.y = np.asarray(y).astype(np.float64)
        u, inv = np.unique(np.asarray(p, dtype=np.float64), return_inverse=True)
        self.M = sparse.csr_matrix((np.ones(len(inv)), (np.arange(len(inv)), inv)), shape=(len(inv), len(u)))

    def __call__(self, W: np.ndarray) -> np.ndarray:
        W = np.atleast_2d(W).astype(np.float64)
        Pb = np.asarray(self.M.T @ (W * self.y).T).T
        Nb = np.asarray(self.M.T @ (W * (1 - self.y)).T).T
        before = np.cumsum(Nb, axis=1) - Nb
        num = (Pb * (before + 0.5 * Nb)).sum(1)
        den = Pb.sum(1) * Nb.sum(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(den > 0, num / den, np.nan)


def weighted_ece(y, p, W: np.ndarray, bins: int = ECE_BINS) -> np.ndarray:
    """ECE for many bootstrap weight vectors: sum_b |sum_i w_i (p_i - y_i)| / sum_i w_i."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    b = np.minimum((p * bins).astype(int), bins - 1)
    M = sparse.csr_matrix((np.ones(len(b)), (np.arange(len(b)), b)), shape=(len(b), bins))
    num = np.abs(np.asarray(M.T @ (W * (p - y)).T).T).sum(1)
    return num / np.maximum(W.sum(1), 1e-9)


# =====================================================================================
# AUDIO AUGMENTATION (same recipe as Test 5)
# =====================================================================================
def waveform_to_spec(audio: np.ndarray) -> np.ndarray:
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


# =====================================================================================
# TEST 5 INPUT LINK (read-only re-use of the prepared split + caches)
# =====================================================================================
def t5_cache_files(t5: T5Paths) -> dict:
    prov = read_json(t5.provenance, "t5_cache_provenance")
    need = {"faceclean", "facedeg", "audio"}
    if not need <= set(prov):
        raise SystemExit(f"[STOP] Test 5 provenance lacks {need - set(prov)}")
    c, d, a = prov["faceclean"], prov["facedeg"], prov["audio"]
    return dict(clean=t5.cache / f"{c['fname']}.npy", clean_status=t5.cache / f"{c['fname']}.status.npy",
                clean_info=t5.cache / f"{c['fname']}.info.npz", n_clean=int(c["n"]),
                deg=t5.cache / f"{d['fname']}.npy", deg_status=t5.cache / f"{d['fname']}.status.npy", n_deg=int(d["n"]),
                spec=t5.cache / a["spec_file"], wave=t5.cache / a["wave_file"], audio_status=t5.cache / a["status_file"], n_audio=int(a["n"]),
                provenance=prov)


def phase_link(a, P: Paths) -> dict:
    """Verify the Test 5 inputs before anything is trained. Writes reports/test5_input_link.json."""
    print("\n" + "=" * 100 + "\n PHASE LINK: verify the prepared Test 5 split + caches (read-only)\n" + "=" * 100)
    t5 = a.t5
    checks, info = {}, {}
    for nm, pth in (("split manifest", t5.split_manifest), ("gate", t5.gate), ("cache provenance", t5.provenance), ("xai pairs", t5.xai_pairs)):
        checks[f"{nm} exists"] = pth.exists()
    if not all(checks.values()):
        write_json(P.link, dict(status="FAIL", checks=checks, checked_at=now()))
        raise SystemExit(f"[STOP] Test 5 inputs missing: {[k for k, v in checks.items() if not v]} (--t5-root {t5.root})")
    gate = read_json(t5.gate, "t5_gate")
    checks["Test 5 leakage gate PASS"] = gate.get("gate") == "PASS"
    sm_sha = sha256_file(guard(t5.split_manifest, "t5_split"))
    checks["split manifest SHA-256 matches the Test 5 gate"] = sm_sha == gate.get("split_manifest_sha256")
    xp_sha = sha256_file(guard(t5.xai_pairs, "t5_xai_pairs"))
    checks["XAI pair list SHA-256 matches the Test 5 gate"] = (xp_sha == gate.get("xai_pairs_sha256")) if gate.get("xai_pairs_sha256") else True
    cf = t5_cache_files(t5)
    for k in ("clean", "clean_status", "clean_info", "deg", "deg_status", "spec", "wave", "audio_status"):
        checks[f"cache file {k} exists"] = cf[k].exists()
    if all(checks.values()):
        cs = np.load(guard(cf["clean_status"], "t5_cache"))
        ds = np.load(guard(cf["deg_status"], "t5_cache"))
        au = np.load(guard(cf["audio_status"], "t5_cache"))
        checks["clean cache complete (no pending rows)"] = bool((cs != 0).all()) and len(cs) == cf["n_clean"]
        checks["degraded cache complete (no pending rows)"] = bool((ds != 0).all()) and len(ds) == cf["n_deg"]
        checks["audio cache complete"] = bool((au != 0).all()) and len(au) == cf["n_audio"]
        man = read_csv(t5.split_manifest, "t5_split", low_memory=False)
        used = man[man.status == "used"]
        o, d = used[used.kind == "original"], used[used.kind == "degraded"]
        checks["every used original has an ok clean clip"] = bool((cs[o.clean_idx.astype(int).values] == 1).all())
        checks["every used degraded video has an ok clip"] = bool((ds[d.deg_idx.astype(int).values] == 1).all())
        checks["degraded videos are in their source's split"] = bool(
            (d.split.values == d.source_key.map(dict(zip(o.item_id, o.split))).values).all())
        info = dict(originals=len(o), degraded=len(d), split_counts=o.groupby(["dataset", "split"]).size().to_dict(),
                    split_manifest_sha256=sm_sha, xai_pairs_sha256=xp_sha, split_hash_originals=gate.get("split_hash_originals"),
                    test5_script_sha256=gate.get("script_sha256"), cache_files={k: str(v) for k, v in cf.items() if isinstance(v, Path)})
    ok = all(checks.values())
    rep = dict(status="PASS" if ok else "FAIL", checked_at=now(), test5_root=str(t5.root), checks=checks, **info,
               note="Test 6 re-uses the Test 5 split, caches and XAI pairs read-only; Test 5 checkpoints are blocked by guard().")
    write_json(P.link, rep)
    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print(f"  link status: {rep['status']} -> {P.link}")
    if not ok:
        raise SystemExit("[STOP] Test 5 inputs failed verification - nothing trained")
    write_experiment_configs(a, P, rep)
    return rep


def require_link(P: Paths) -> dict:
    if not P.link.exists():
        raise SystemExit("[STOP] run --phase link first")
    r = read_json(P.link, "t6_link")
    if r.get("status") != "PASS":
        raise SystemExit("[STOP] Test 5 input link did not PASS")
    return r


def write_experiment_configs(a, P: Paths, link: dict):
    m = MSTFAV()
    for v in a.variant_list:
        write_json(P.configs / f"experiment_config_{v}.json", dict(
            test="Test 6", variant=v, description=VARIANTS[v]["description"], created=now(), variant_config=variant_config(a, v),
            model=dict(name="MSTF-AV", parameters=n_params(m), embedding_dim=EMB_DIM, architecture_signature=architecture_signature(m),
                       architecture_source_sha256=architecture_source_sha(),
                       streams=["RGB residual CNN + SE", "SRM noise-residual CNN", "cross-stream gate (16x16)", "temporal transformer (2x4 heads)",
                                "audio residual CNN tokens", "gated cross-modal attention"]),
            stage1=dict(data="clean training originals", epochs=a.epochs1, lr=a.lr1, loss="cross-entropy", shared_across_variants=True),
            stage2=dict(epochs=a.epochs2, lr=a.lr2, batch_clips=a.batch_size, accum=a.accum, amp=a.amp, warmup_frac=WARMUP_FRAC,
                        dro=dict(eta=a.dro_eta, floor=a.dro_floor, momentum=DRO_MOMENTUM), ema_decay=a.ema_decay, grad_clip=GRAD_CLIP),
            optimiser=dict(name="AdamW", weight_decay=WEIGHT_DECAY, schedule="linear warm-up + cosine"),
            calibration="temperature scaling on validation (original + degraded), NLL grid search",
            threshold="validation balanced-accuracy optimum on the calibrated probability",
            model_selection="best epoch by validation mean(original AUC, degraded AUC); Joint: macro over datasets; 6C/6D on EMA weights",
            seeds=a.seed_list, train_sets=a.train_list, data=dict(source="Test 5 prepared run (read-only)", **{k: link.get(k) for k in (
                "split_manifest_sha256", "split_hash_originals", "xai_pairs_sha256", "test5_root")}),
            environment=environment_info()))


# =====================================================================================
# RUNTIME DATA
# =====================================================================================
class T6Data:
    def __init__(self, a, P: Paths):
        t5 = a.t5
        self.P = P
        man = read_csv(t5.split_manifest, "t5_split", low_memory=False)
        man["label"] = man.label.astype(int)
        man = man[man.status == "used"]
        o = man[(man.kind == "original") & man.split.isin(["train", "val", "test"])].reset_index(drop=True)
        d = man[(man.kind == "degraded") & man.split.isin(["train", "val", "test"])].reset_index(drop=True)
        if a.smoke:                                  # tiny balanced subset per (dataset, split, label)
            rng = np.random.default_rng(7)
            keep = []
            for _, g in o.groupby(["dataset", "split", "label"]):
                keep += list(rng.choice(g.index.values, size=min(len(g), a.smoke_per_cell), replace=False))
            o = o.loc[sorted(keep)].reset_index(drop=True)
            d = d[d.source_key.isin(set(o.item_id))].reset_index(drop=True)
        self.o, self.d = o, d
        self.key = o.item_id.values
        self.labels = o.label.values.astype(np.int64)
        self.ds = o.dataset.values
        self.dom = np.array([DATASETS.index(x) if x in DATASETS else 0 for x in self.ds], np.int64)
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
        cf = t5_cache_files(t5)
        self.clean = np.load(guard(cf["clean"], "t5_cache"), mmap_mode="r")
        self.degc = np.load(guard(cf["deg"], "t5_cache"), mmap_mode="r")
        info = np.load(guard(cf["clean_info"], "t5_cache"))
        meta = info["meta"][self.cidx]                        # width, height, fps, frames, fallback, face_rel_size, detected
        self.native_face_px = (meta[:, 5] * np.minimum(meta[:, 0], meta[:, 1])).astype(np.float64)
        self.spec = np.load(guard(cf["spec"], "t5_cache"), mmap_mode="r")
        self.wave = np.load(guard(cf["wave"], "t5_cache"), mmap_mode="r")
        self.has_audio = (np.load(guard(cf["audio_status"], "t5_cache")) == 1)[self.cidx]
        print(f"  data: {len(o):,} originals, {len(d):,} degraded (Test 5 split, read-only caches)"
              + (" [SMOKE SUBSET]" if a.smoke else ""))

    def idx(self, split: str, dataset: Optional[str] = None) -> np.ndarray:
        m = self.split == split
        if dataset and dataset != "Joint":
            m &= self.ds == dataset
        return np.where(m)[0]

    def deg_items(self, src_idx: np.ndarray) -> np.ndarray:
        return (np.concatenate([self.versions.get(int(s), np.zeros(0, np.int64)) for s in src_idx]) if len(src_idx)
                else np.zeros(0, np.int64))

    def clip(self, kind: str, i: int, sel=None) -> np.ndarray:
        if kind == "deg":
            return np.asarray(self.degc[self.didx[i]] if sel is None else self.degc[self.didx[i], sel])
        return np.asarray(self.clean[self.cidx[i]] if sel is None else self.clean[self.cidx[i], sel])

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


def lr_lambda(total: int):
    w = max(1, int(WARMUP_FRAC * total))

    def f(step):
        if step < w:
            return (step + 1) / w
        return 0.5 * (1 + math.cos(math.pi * min(1.0, (step - w) / max(1, total - w))))
    return f


class EMA:
    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        self.model = copy.deepcopy(model).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        for e, p in zip(self.model.parameters(), model.parameters()):
            e.mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)
        for e, b in zip(self.model.buffers(), model.buffers()):
            e.copy_(b)


class Trainer:
    def __init__(self, a, P: Paths, D: T6Data, train_on: str, seed: int):
        self.a, self.P, self.D, self.train_on, self.seed = a, P, D, train_on, seed
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.T, self.L = NUM_FRAMES, CLIP_LEN
        self.stride = max(1, self.T // self.L)
        self.tr = D.idx("train", train_on)
        self.va = D.idx("val", train_on)
        # FF++ has no audio: such a model never trained its audio branch, so it is evaluated as video-only
        self.uses_audio = bool(D.has_audio[self.tr].any())
        self.cell = np.array([f"{d}|{'real' if l == 0 else 'fake'}" for d, l in zip(D.ds, D.labels)])
        self.tr_deg_src = np.array([s for s in self.tr if int(s) in D.versions], np.int64)
        self.n_tr_deg = int(sum(len(D.versions[int(s)]) for s in self.tr_deg_src))
        self.type_pool = defaultdict(list)                 # (cell, type) -> degraded items (training split)
        for s in self.tr_deg_src:
            for j in D.versions[int(s)]:
                self.type_pool[(self.cell[s], D.deg_type[j])].append(int(j))
        self.type_pool = {k: np.array(sorted(v), np.int64) for k, v in self.type_pool.items()}
        self.types = sorted({t for _, t in self.type_pool})
        # resolution terciles (per dataset, cut points from TRAINING originals only)
        self.res_bin = np.full(len(D.key), -1, np.int64)
        lp = np.log(np.clip(D.native_face_px, 1, None))
        for ds in np.unique(D.ds[self.tr]) if len(self.tr) else []:
            m_tr = self.tr[(D.ds[self.tr] == ds) & np.isfinite(lp[self.tr])]
            if len(m_tr) < 3:
                continue
            cuts = np.quantile(lp[m_tr], [1 / 3, 2 / 3])
            m = (D.ds == ds) & np.isfinite(lp)
            self.res_bin[m] = np.digitize(lp[m], cuts)
        self.ckdir = P.checkpoints / train_tag(train_on) / f"seed{seed}"
        self.ckdir.mkdir(parents=True, exist_ok=True)
        self.split_hash = require_link(P).get("split_hash_originals")

    # ---------------------------------------------------------------- sampling
    def weights(self, src: np.ndarray, epoch_size: int) -> np.ndarray:
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

    def dro_probs(self, dro_loss: Dict[str, float]) -> Dict[str, float]:
        """Group-DRO type weights: q ~ exp(eta * L_t / mean L), mixed with a uniform floor (no type is starved)."""
        if not self.types:
            return {}
        L = np.array([dro_loss.get(t, 1.0) for t in self.types])
        q = np.exp(self.a.dro_eta * (L / max(L.mean(), 1e-6) - 1.0))
        q = q / q.sum()
        p = self.a.dro_floor / len(self.types) + (1 - self.a.dro_floor) * q
        return dict(zip(self.types, p / p.sum()))

    def choose_deg(self, s: int, rng, dro_p: Optional[Dict[str, float]]) -> Tuple[int, int, str]:
        D = self.D
        if dro_p:
            types = [t for t in self.types if len(self.type_pool.get((self.cell[s], t), ()))]
            if types:
                w = np.array([dro_p[t] for t in types])
                t = types[int(rng.choice(len(types), p=w / w.sum()))]
                pool = self.type_pool[(self.cell[s], t)]
                j = int(pool[int(rng.integers(len(pool)))])
                return int(D.deg_src[j]), j, "dro"
        v = D.versions[int(s)]
        return int(s), int(v[int(rng.integers(len(v)))]), "natural"

    def sample_audio(self, src: np.ndarray, rng):
        spec, has = self.D.audio(src)
        u = rng.random(len(src))
        pc, pdg, _ = AUDIO_PROBS
        keep = (has > 0) & (u < pc + pdg)
        deg = keep & (u >= pc)
        for b in np.where(deg)[0]:
            d = AUDIO_DEGRADATIONS[int(rng.integers(len(AUDIO_DEGRADATIONS)))]
            lvl = 1 if d == "silence" else int(rng.integers(3))
            spec[b] = waveform_to_spec(degrade_audio(self.D.wave_float(int(src[b])), d, lvl, rng))
        return spec, keep.astype(np.float32)

    def build(self, items: List[tuple], seed: int, paired: bool) -> dict:
        """items: [(kind, idx)] (unpaired) or [('pair', src, deg_j)] (paired: same frames / flip / shift for both)."""
        rng = np.random.default_rng(seed)
        S, D = IMG_SIZE, self.D
        pad = max(2, S // 24)
        B = len(items)
        a_ = np.empty((B, self.L, S, S, 3), np.uint8)
        b_ = np.empty((B, self.L, S, S, 3), np.uint8) if paired else None
        src = np.empty(B, np.int64)
        dtype = []
        for k, it in enumerate(items):
            sel = np.sort(rng.choice(self.T, self.L, replace=False))
            flip = rng.random() < 0.5
            dy, dx = rng.integers(0, 2 * pad + 1, 2)
            if paired:
                _, s, j = it
                clips = [(a_, D.clip("clean", s, sel)), (b_, D.clip("deg", j, sel))]
                src[k] = s
                dtype.append(D.deg_type[j])
            else:
                kind, i = it
                clips = [(a_, D.clip(kind, i, sel))]
                src[k] = D.src_of(kind, i)
                dtype.append(D.deg_type[i] if kind == "deg" else "original")
            for out, c in clips:
                if flip:
                    c = c[:, :, ::-1, :]
                pp = np.pad(c, ((0, 0), (pad, pad), (pad, pad), (0, 0)), mode="edge")
                out[k] = pp[:, dy:dy + S, dx:dx + S]
        audio, mask = self.sample_audio(src, rng)
        return dict(a=a_, b=b_, audio=audio, mask=mask, y=D.labels[src], res=self.res_bin[src], dom=D.dom[src], dtype=np.array(dtype))

    def plan_epoch(self, rng, stage: int, vc: dict, steps: int, bs: int, dro_p):
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
        dp = dro_p if vc["dro"] else None
        if vc["paired"]:
            for st in range(steps):
                its = []
                for s in rng.choice(self.tr_deg_src, size=bs // 2, replace=True, p=wd):
                    s2, j, mode = self.choose_deg(int(s), rng, dp)
                    its.append(("pair", s2, j))
                    log[(self.cell[s2], mode, self.D.deg_type[j])] += 1
                plan.append(its)
        else:
            wc = self.weights(self.tr, steps * bs)
            nc = bs // 2
            for st in range(steps):
                its = [("clean", int(s)) for s in rng.choice(self.tr, size=nc, replace=True, p=wc)]
                for s in rng.choice(self.tr_deg_src, size=bs - nc, replace=True, p=wd):
                    s2, j, mode = self.choose_deg(int(s), rng, dp)
                    its.append(("deg", j))
                    log[(self.cell[s2], mode, self.D.deg_type[j])] += 1
                log.update((self.cell[s], "clean", "-") for _, s in its[:nc])
                plan.append(its)
        return plan, log

    # ---------------------------------------------------------------- prediction
    def amask(self, has: np.ndarray) -> np.ndarray:
        return has if self.uses_audio else np.zeros_like(has)

    @torch.no_grad()
    def predict_logits(self, model, items: List[Tuple[str, int]], cond: str = "normal", bs: int = 32) -> np.ndarray:
        """logit(fake) - logit(real), averaged over the frame offsets."""
        model.eval()
        out = []
        for i in range(0, len(items), bs):
            ch = items[i:i + bs]
            clips = np.stack([self.D.clip(k, j) for k, j in ch])
            src = np.array([self.D.src_of(k, j) for k, j in ch])
            spec, has = self.D.audio(src)
            xa = torch.from_numpy(spec).to(self.device)
            am = torch.from_numpy(np.zeros_like(has) if cond == "audio_masked" else self.amask(has)).to(self.device)
            mode = "a_only" if cond == "audio_only" else "av"
            z = 0
            for v in range(self.stride):
                xv = to_video_tensor(clips[:, v::self.stride][:, :self.L], self.device)
                with torch.autocast("cuda", dtype=torch.float16, enabled=self.a.amp):
                    lg = model(xv, xa, mode=mode, a_mask=am).float()
                z = z + (lg[:, 1] - lg[:, 0]).cpu().numpy()
            out.append(z / self.stride)
        return np.concatenate(out) if out else np.zeros(0)

    def val_sets(self, stage: int):
        vc = [("clean", int(s)) for s in self.va]
        vd = [("deg", int(j)) for j in self.D.deg_items(self.va)] if stage == 2 else []
        return vc, vd

    def validate(self, model, stage: int) -> dict:
        vc, vd = self.val_sets(stage)
        zc = self.predict_logits(model, vc)
        sc = np.array([s for _, s in vc], np.int64)
        row = {}
        scores = []
        for ds in sorted(set(self.D.ds[sc])):
            m = self.D.ds[sc] == ds
            ca = safe_auc(self.D.labels[sc][m], zc[m])
            row[f"val_clean_auc_{ds}"] = ca
            s_ds = [ca]
            if vd:
                jd = np.array([j for _, j in vd], np.int64)
                md = self.D.ds[self.D.deg_src[jd]] == ds
                if md.any():
                    zd = self.predict_logits(model, [("deg", int(j)) for j in jd[md]])
                    da = safe_auc(self.D.labels[self.D.deg_src[jd[md]]], zd)
                    row[f"val_degraded_auc_{ds}"] = da
                    s_ds.append(da)
            scores.append(float(np.nanmean(s_ds)))
        row["val_clean_auc"] = float(np.nanmean([v for k, v in row.items() if k.startswith("val_clean_auc_")]))
        dv = [v for k, v in row.items() if k.startswith("val_degraded_auc_")]
        if dv:
            row["val_degraded_auc"] = float(np.nanmean(dv))
        row["score"] = float(np.nanmean(scores)) if scores else float("nan")
        return row

    def calibrate(self, model, stage: int) -> dict:
        vc, vd = self.val_sets(stage)
        items = vc + vd
        y = np.array([self.D.labels[self.D.src_of(k, i)] for k, i in items])
        z = self.predict_logits(model, items)
        T = fit_temperature(z, y)
        p_raw, p = 1 / (1 + np.exp(-z)), 1 / (1 + np.exp(-z / T))
        thr = find_threshold(y, p)
        thr.update(temperature=T, val_ece_raw=ece_score(y, p_raw), val_ece=ece_score(y, p),
                   selected_on="validation original" + (" + degraded" if vd else "") + f" ({self.train_on})")
        return thr

    # ---------------------------------------------------------------- training loop
    def train(self, stage: int, variant: str, model, ckpt: Path) -> dict:
        a, D = self.a, self.D
        vc = (dict(paired=False, lambda_cam=0.0, lambda_supcon=0.0, dro=False, ema=False, label_smooth=0.0, lambda_adv=0.0)
              if stage == 1 else variant_config(a, variant))
        name = run_id(self.train_on, self.seed, "S1" if stage == 1 else variant)
        bs = a.batch_size
        clips_per_epoch = len(self.tr) if stage == 1 else len(self.tr) + self.n_tr_deg
        steps = int(math.ceil(clips_per_epoch / (bs * a.accum))) * a.accum
        if a.max_steps:
            steps = min(steps, a.max_steps * a.accum)
        epochs = a.epochs1 if stage == 1 else a.epochs2
        lr = a.lr1 if stage == 1 else a.lr2
        print("\n" + "=" * 100 + f"\n TRAIN {name} | stage {stage} | {VARIANTS.get(variant, {}).get('description', 'clean originals, CE')}\n"
              f"  train originals {len(self.tr):,} (with degraded {len(self.tr_deg_src):,}; versions {self.n_tr_deg:,}) | val {len(self.va):,} | "
              f"{steps} steps x {epochs} epochs | {bs} clips/step x accum {a.accum} | lr {lr} | amp {a.amp} | {vc}\n" + "=" * 100)
        if stage == 2 and not len(self.tr_deg_src):
            raise SystemExit(f"[STOP] no degraded training videos for {self.train_on}")
        decay, no_decay = [], []
        for n_, p_ in model.named_parameters():
            (no_decay if p_.ndim <= 1 or n_.endswith(("pos", "cls")) else decay).append(p_)
        opt = optim.AdamW([dict(params=decay, weight_decay=WEIGHT_DECAY), dict(params=no_decay, weight_decay=0.0)], lr=lr)
        n_updates = max(1, steps // a.accum * epochs)
        sched = optim.lr_scheduler.LambdaLR(opt, lr_lambda(n_updates))
        scaler = torch.amp.GradScaler("cuda", enabled=a.amp)
        ema = EMA(model, a.ema_decay) if vc["ema"] else None
        rng = np.random.default_rng([self.seed, stage, sum(map(ord, variant))])
        loader = ThreadPoolExecutor(max_workers=max(1, a.loader_threads))
        history, best, samp_rows = [], dict(score=-1.0, epoch=-1), []
        dro_loss = {t: 1.0 for t in self.types}
        t_start, overall, updates = time.time(), 0, 0
        resume_path = ckpt.with_suffix(".resume.pt")
        run_sig = dict(run=name, stage=stage, variant=variant, split_hash=self.split_hash, steps=steps, epochs=epochs, bs=bs, lr=lr,
                       accum=a.accum, vc={k: v for k, v in vc.items() if k != "description"}, arch=architecture_signature(model))
        start_ep, start_st, ep_state, rs = 1, 0, None, None
        if resume_path.exists():
            try:
                rs = torch.load(guard(resume_path, "t6_checkpoint"), map_location=self.device, weights_only=False)
            except Exception as e:
                print(f"  [resume] {resume_path.name} unreadable ({type(e).__name__}) - starting from scratch")
            if rs is not None and rs.get("sig") != run_sig:
                print(f"  [resume] {resume_path.name} belongs to another configuration - starting from scratch")
                rs = None
        if rs is not None:
            model.load_state_dict(rs["model"])
            opt.load_state_dict(rs["opt"])
            sched.load_state_dict(rs["sched"])
            scaler.load_state_dict(rs["scaler"])
            if ema is not None and rs.get("ema") is not None:
                ema.model.load_state_dict(rs["ema"])
            rng.bit_generator.state = rs["rng_epoch_start"]
            torch.set_rng_state(rs["torch_rng"].cpu())
            if torch.cuda.is_available() and rs.get("cuda_rng") is not None:
                torch.cuda.set_rng_state_all([s.cpu() for s in rs["cuda_rng"]])
            history, best, samp_rows, overall, updates = rs["history"], rs["best"], rs["samp_rows"], rs["overall"], rs["updates"]
            dro_loss = rs["dro_loss_epoch_start"]
            start_ep, start_st, ep_state = rs["epoch"], rs["step"], rs.get("ep_state")
            t_start = time.time() - rs["elapsed"]
            print(f"  [resume] {name}: epoch {start_ep}/{epochs}, step {start_st}/{steps}")
        else:
            ckpt.with_suffix(".best.tmp").unlink(missing_ok=True)

        def save_resume(ep_next, st_next, rng_state, ep_st, dro_start):
            state = dict(sig=run_sig, model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(),
                         scaler=scaler.state_dict(), ema=ema.model.state_dict() if ema else None, rng_epoch_start=rng_state,
                         torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                         history=history, best=best, samp_rows=samp_rows, overall=overall, updates=updates,
                         dro_loss_epoch_start=dro_start, epoch=ep_next, step=st_next, ep_state=ep_st,
                         elapsed=time.time() - t_start, saved_at=now())
            try:
                atomic_torch_save(state, resume_path)
            except PermissionError as e:            # a skipped resume save only costs the steps since the previous one
                print(f"\n  [resume] could not save {resume_path.name} ({e}) - training continues")

        ce_main = nn.CrossEntropyLoss(label_smoothing=vc["label_smooth"])
        for ep in range(start_ep, epochs + 1):
            model.train()
            rng_epoch_start = rng.bit_generator.state
            dro_start = dict(dro_loss)
            dro_p = self.dro_probs(dro_loss) if vc["dro"] else None
            plan, slog = self.plan_epoch(rng, stage, vc, steps, bs, dro_p)
            mid = ep == start_ep and start_st > 0
            if not mid:
                for (cell, mode, typ), n in slog.items():
                    samp_rows.append(dict(run=name, epoch=ep, cell=cell, mode=mode, degradation=typ, n=n,
                                          dro_p=(dro_p or {}).get(typ, np.nan)))
            seeds = rng.integers(0, 2 ** 31, size=steps)
            first = start_st if mid else 0
            q, nxt = deque(), first

            def submit():
                nonlocal nxt
                if nxt < steps:
                    q.append(loader.submit(self.build, plan[nxt], int(seeds[nxt]), vc["paired"]))
                    nxt += 1

            for _ in range(max(2, 2 * a.loader_threads)):
                submit()
            tot, comp, t0, last = 0.0, Counter(), time.time(), 0
            tl_sum, tl_n = Counter(), Counter()
            if mid and ep_state:
                tot, comp, t0 = ep_state["tot"], Counter(ep_state["comp"]), time.time() - ep_state["epoch_elapsed"]
                tl_sum, tl_n = Counter(ep_state["tl_sum"]), Counter(ep_state["tl_n"])
            progress = lambda: (updates + 1) / n_updates  # noqa: E731
            opt.zero_grad(set_to_none=True)
            for st in range(first, steps):
                bt = q.popleft().result()
                submit()
                overall += 1
                y_np = bt["y"]
                with torch.autocast("cuda", dtype=torch.float16, enabled=a.amp):
                    if vc["paired"]:
                        nb = len(y_np)
                        xv = to_video_tensor(np.concatenate([bt["a"], bt["b"]]), self.device)
                        xa = torch.from_numpy(np.concatenate([bt["audio"], bt["audio"]])).to(self.device)
                        am = torch.from_numpy(np.concatenate([bt["mask"], bt["mask"]])).to(self.device)
                        y = torch.from_numpy(np.concatenate([y_np, y_np])).to(self.device)
                        logits, z, v, act = model.forward_train(xv, xa, am)
                        is_deg = np.r_[np.zeros(nb, bool), np.ones(nb, bool)]
                        dtypes = np.r_[np.array(["original"] * nb), bt["dtype"]]
                    else:
                        xv = to_video_tensor(bt["a"], self.device)
                        xa = torch.from_numpy(bt["audio"]).to(self.device)
                        am = torch.from_numpy(bt["mask"]).to(self.device)
                        y = torch.from_numpy(y_np).to(self.device)
                        logits, z, v, act = model.forward_train(xv, xa, am)
                        dtypes = bt["dtype"]
                        is_deg = dtypes != "original"
                    l_ce = ce_main(logits.float(), y)
                    loss = l_ce
                    comp["ce"] += float(l_ce.item())
                    if vc["paired"] and vc["lambda_supcon"] > 0:
                        l_sc = supcon_loss(model.proj(z), y)
                        loss = loss + vc["lambda_supcon"] * l_sc
                        comp["supcon"] += float(l_sc.item())
                    if vc["lambda_adv"] > 0:
                        lam = vc["lambda_adv"] * (2 / (1 + math.exp(-10 * progress())) - 1)
                        r_logit, d_logit = model.adversary(v.float(), y, lam)
                        res = torch.from_numpy(np.concatenate([bt["res"], bt["res"]]) if vc["paired"] else bt["res"]).to(self.device)
                        okr = res >= 0
                        l_adv = F.cross_entropy(r_logit[okr], res[okr]) if okr.any() else logits.new_zeros(())
                        if self.train_on == "Joint":
                            dom = torch.from_numpy(np.concatenate([bt["dom"], bt["dom"]]) if vc["paired"] else bt["dom"]).to(self.device)
                            l_adv = l_adv + F.cross_entropy(d_logit, dom)
                        loss = loss + l_adv            # gradient reversal scales the encoder side by lam
                        comp["adv"] += float(l_adv.item())
                if vc["paired"] and vc["lambda_cam"] > 0:
                    cam = soft_gradcam(logits, act, 2 * nb, self.L, create_graph=True)
                    l_cam = cam_consistency_loss(cam[:nb].detach(), cam[nb:])
                    loss = loss + vc["lambda_cam"] * l_cam
                    comp["cam"] += float(l_cam.item())
                if vc["dro"] and is_deg.any():
                    with torch.no_grad():
                        per = F.cross_entropy(logits.float(), y, reduction="none").cpu().numpy()
                    for t, l_ in zip(dtypes[is_deg], per[is_deg]):
                        tl_sum[t] += float(l_)
                        tl_n[t] += 1
                scaler.scale(loss / a.accum).backward()
                if (st + 1) % a.accum == 0:
                    scaler.unscale_(opt)
                    nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                    scaler.step(opt)
                    scaler.update()
                    opt.zero_grad(set_to_none=True)
                    if updates < n_updates - 1:
                        sched.step()
                    updates += 1
                    if ema is not None:
                        ema.update(model)
                tot += float(loss.item())
                if (st + 1) % max(1, a.save_every_steps) == 0 and st + 1 < steps and (st + 1) % a.accum == 0:
                    save_resume(ep, st + 1, rng_epoch_start, dict(tot=tot, comp=dict(comp), epoch_elapsed=time.time() - t0,
                                                                  tl_sum=dict(tl_sum), tl_n=dict(tl_n)), dro_start)
                if time.time() - last > 0.5 or st + 1 == steps:
                    gt = gpu_temp() if st % 50 == 0 else None
                    if gt is not None and gt >= a.gpu_max_temp:
                        print(f"\n  [thermal] GPU {gt}C - pausing {a.gpu_pause}s")
                        torch.cuda.empty_cache()
                        time.sleep(a.gpu_pause)
                    el = time.time() - t_start
                    eta = el / overall * (steps * epochs - overall)
                    sys.stdout.write(f"\r  [{name}] ep {ep}/{epochs} {progress_bar(st + 1, steps, 16)} loss {tot / (st + 1):.4f} ("
                                     + " ".join(f"{k} {v / (st + 1):.3f}" for k, v in comp.items()) + f") | ETA {fmt_time(eta)}   ")
                    sys.stdout.flush()
                    last = time.time()
            if vc["dro"]:                                            # update group-DRO type losses (momentum)
                for t in self.types:
                    if tl_n[t]:
                        dro_loss[t] = DRO_MOMENTUM * dro_loss[t] + (1 - DRO_MOMENTUM) * tl_sum[t] / tl_n[t]
            row = dict(run=name, epoch=ep, train_loss=tot / steps, lr=float(sched.get_last_lr()[0]),
                       **{f"train_{k}": v / steps for k, v in comp.items()}, epoch_time_s=time.time() - t0,
                       **({f"dro_p_{t}": p_ for t, p_ in self.dro_probs(dro_loss).items()} if vc["dro"] else {}))
            eval_model = ema.model if ema is not None else model
            row.update(self.validate(eval_model, stage))
            history.append(row)
            better = np.isfinite(row["score"]) and row["score"] > best["score"]
            if better:
                best = dict(score=row["score"], epoch=ep)
                atomic_torch_save(dict(model_state_dict=eval_model.state_dict(), epoch=ep, val=row), ckpt.with_suffix(".best.tmp"))
            print(f"\r  [{name}] ep {ep}/{epochs} | loss {row['train_loss']:.4f} | val clean AUC {row['val_clean_auc']:.4f}"
                  + (f" | val degraded AUC {row['val_degraded_auc']:.4f}" if "val_degraded_auc" in row else "")
                  + f" | score {row['score']:.4f}{'  <- best' if better else ''} | {fmt_time(row['epoch_time_s'])}" + " " * 20)
            save_resume(ep + 1, 0, rng.bit_generator.state, None, dict(dro_loss))
        loader.shutdown(wait=True)
        pd.DataFrame(history).to_csv(self.P.hist_dir / f"history_{name}.csv", index=False)
        pd.DataFrame(samp_rows).to_csv(self.P.hist_dir / f"sampling_{name}.csv", index=False)
        tmp = ckpt.with_suffix(".best.tmp")
        if not tmp.exists():
            raise RuntimeError(f"{name}: no valid validation score - nothing saved")
        model.load_state_dict(torch.load(guard(tmp, "t6_checkpoint"), map_location=self.device, weights_only=False)["model_state_dict"])
        cal = self.calibrate(model, stage)
        cfg = dict(run=name, stage=stage, variant=variant if stage == 2 else "S1", variant_config=vc, train_on=self.train_on, seed=self.seed,
                   epochs=epochs, lr=lr, batch_clips=bs, accum=a.accum, steps_per_epoch=steps, amp=a.amp, split_hash=self.split_hash,
                   architecture_signature=architecture_signature(model), architecture_source_sha256=architecture_source_sha(),
                   parameters=n_params(model), script_sha256=code_sha256())
        cfg_sha = sha256_text(json.dumps(to_jsonable(cfg), sort_keys=True))
        sd = model.state_dict()
        atomic_torch_save(dict(model_state_dict=sd, threshold=cal["threshold"], temperature=cal["temperature"], calibration=cal,
                               best_epoch=best["epoch"], best_val_score=best["score"], history=history, config=cfg, config_sha256=cfg_sha,
                               weights_sha256=weights_sha256(sd), final=True, saved_at=now()), ckpt)
        tmp.unlink(missing_ok=True)
        resume_path.unlink(missing_ok=True)
        rec = dict(run_id=name, stage=stage, path=str(ckpt), file_sha256=sha256_file(ckpt), weights_sha256=weights_sha256(sd),
                   script_sha256=code_sha256(), config_sha256=cfg_sha, split_hash=self.split_hash, threshold=cal["threshold"],
                   temperature=cal["temperature"], val_ece_raw=cal["val_ece_raw"], val_ece=cal["val_ece"], best_epoch=best["epoch"],
                   best_val_score=best["score"], train_time_s=time.time() - t_start, saved_at=now())
        hp = self.P.checkpoints / "checkpoint_hashes.csv"
        h = read_csv(hp, "t6_checkpoint_index") if hp.exists() else pd.DataFrame()
        h = pd.concat([h[h.run_id != name] if len(h) else h, pd.DataFrame([rec])], ignore_index=True)
        h.to_csv(hp.with_suffix(".csv.tmp"), index=False)
        replace_retry(hp.with_suffix(".csv.tmp"), hp)
        print(f"  saved {ckpt.name} | best epoch {best['epoch']} | val score {best['score']:.4f} | T {cal['temperature']:.2f} | "
              f"threshold {cal['threshold']:.2f} | val ECE {cal['val_ece_raw']:.3f} -> {cal['val_ece']:.3f}")
        return rec

    def done(self, ckpt: Path) -> bool:
        if not ckpt.exists():
            return False
        try:
            obj = torch.load(guard(ckpt, "t6_checkpoint"), map_location="cpu", weights_only=False)
        except (RuntimeError, EOFError, OSError, ValueError) as e:
            print(f"  [resume] {ckpt.name} unreadable ({type(e).__name__}) - redo")
            return False
        return bool(obj.get("final")) and obj.get("config", {}).get("split_hash") == self.split_hash

    def load(self, ckpt: Path):
        obj = torch.load(guard(ckpt, "t6_checkpoint"), map_location=self.device, weights_only=False)
        m = MSTFAV().to(self.device)
        m.load_state_dict(obj["model_state_dict"])
        m.eval()
        return m, obj

    def stage1(self) -> Path:
        ck = self.ckdir / "stage1.pt"
        if self.done(ck):
            print(f"  [resume] {ck} done")
            return ck
        set_seed(self.seed, self.a.deterministic)
        model = MSTFAV().to(self.device)
        print(f"  fresh MSTF-AV | {n_params(model):,} parameters | init seed {self.seed}")
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


def with_oom_retry(a, fn, *args):
    """On CUDA out-of-memory: halve the clips per step and double gradient accumulation (same effective batch), retry."""
    while True:
        try:
            return fn(*args)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if a.batch_size <= 8:
                raise
            a.batch_size //= 2
            a.accum *= 2
            print(f"\n  [OOM] retrying with --batch-size {a.batch_size} --accum {a.accum} (same effective batch); "
                  f"pass these flags next time to avoid the retry")


def phase_train(a, P: Paths, D: T6Data, seeds: Optional[List[int]] = None):
    require_link(P)
    status = []
    for seed in (seeds or a.seed_list):
        for t in a.train_list:
            tr = Trainer(a, P, D, t, seed)
            s1 = with_oom_retry(a, tr.stage1)
            for v in a.variant_list:
                try:
                    with_oom_retry(a, tr.stage2, v, s1)
                    status.append(dict(run_id=run_id(t, seed, v), status="completed"))
                except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
                    status.append(dict(run_id=run_id(t, seed, v), status=f"failed: {type(e).__name__}: {str(e)[:200]}"))
                    print(f"  [FAILED] {run_id(t, seed, v)}: {e}")
                    torch.cuda.empty_cache()
            status.append(dict(run_id=run_id(t, seed, "S1"), status="completed"))
    p = P.logs / "run_status.csv"
    old = read_csv(p, "t6_log") if p.exists() else pd.DataFrame()
    st = pd.concat([old, pd.DataFrame(status)]).drop_duplicates("run_id", keep="last") if len(old) else pd.DataFrame(status)
    st.to_csv(p, index=False)


# =====================================================================================
# EVALUATION + GRAD-CAM
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


PRED_COLS = ["run_id", "train_on", "variant", "seed", "item_id", "kind", "source_key", "group_id", "dataset", "label", "degradation",
             "severity", "condition", "p_fake", "p_fake_raw", "logit", "threshold", "temperature", "native_face_px"]


def phase_evaluate(a, P: Paths, D: T6Data):
    require_link(P)
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
        thr, T = float(obj["threshold"]), float(obj["temperature"])
        print(f"  evaluating {r['run_id']} (T {T:.2f}, threshold {thr:.2f}) on {len(clean_items):,} originals + {len(deg_items):,} degraded ...")
        rows = []

        def add(items, z, cond):
            for (k, i), zz in zip(items, z):
                s = D.src_of(k, i)
                rows.append((r["run_id"], r["train_on"], r["variant"], r["seed"], D.d.item_id[i] if k == "deg" else D.key[i], k, D.key[s],
                             D.group[s], D.ds[s], int(D.labels[s]), D.deg_type[i] if k == "deg" else "original",
                             D.deg_sev[i] if k == "deg" else "", cond, float(1 / (1 + np.exp(-zz / T))), float(1 / (1 + np.exp(-zz))),
                             float(zz), thr, T, float(D.native_face_px[s])))

        add(clean_items, tr.predict_logits(model, clean_items), "normal")
        add(deg_items, tr.predict_logits(model, deg_items), "normal")
        ai = [("clean", int(s)) for s in audio_src]
        if ai and tr.uses_audio:                           # audio ablations are undefined for a video-only (FF++) model
            add(ai, tr.predict_logits(model, ai, "audio_masked"), "audio_masked")
            add(ai, tr.predict_logits(model, ai, "audio_only"), "audio_only")
        pd.DataFrame(rows, columns=PRED_COLS).to_csv(out, index=False, compression="gzip")
        del model
        torch.cuda.empty_cache()


def load_xai_pairs(a, D: T6Data):
    xp = read_csv(a.t5.xai_pairs, "t5_xai_pairs")
    dpos = {k: i for i, k in enumerate(D.d.item_id)}
    spos = {k: i for i, k in enumerate(D.key)}
    return [(spos[r.source_key], dpos[r.deg_item_id], r) for r in xp.itertuples() if r.deg_item_id in dpos and r.source_key in spos]


def phase_xai(a, P: Paths, D: T6Data):
    require_link(P)
    pairs = load_xai_pairs(a, D)
    if a.smoke:
        pairs = pairs[::3]
    for r in available_runs(a, P):
        out = P.cam_dir / f"{r['run_id']}.csv"
        if out.exists():
            continue
        tr = Trainer(a, P, D, r["train_on"], r["seed"])
        model, obj = tr.load(r["ckpt"])
        thr, T = float(obj["threshold"]), float(obj["temperature"])
        st, L = tr.stride, tr.L
        rows = []
        print(f"  Grad-CAM {r['run_id']} on {len(pairs)} Test 5 XAI pairs ...")
        for k in range(0, len(pairs), XAI_CHUNK):
            ch = pairs[k:k + XAI_CHUNK]
            src = np.array([s for s, _, _ in ch])
            cc = np.stack([D.clip("clean", s)[::st][:L] for s, _, _ in ch])
            cr = np.stack([D.clip("clean", s)[1::st][:L] if st > 1 else D.clip("clean", s)[::-1][:L] for s, _, _ in ch])
            dc = np.stack([D.clip("deg", j)[::st][:L] for _, j, _ in ch])
            spec, has = D.audio(src)
            xa, am = torch.from_numpy(spec).to(tr.device), torch.from_numpy(tr.amask(has)).to(tr.device)
            gc = grad_cam(model, to_video_tensor(cc, tr.device), xa, am, 1, thr, T)
            gd = grad_cam(model, to_video_tensor(dc, tr.device), xa, am, 1, thr, T)
            gr = grad_cam(model, to_video_tensor(cr, tr.device), xa, am, 1, thr, T)
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
        if r["variant"] in ("6A", "6D") and r["seed"] == a.seed_list[0] and pairs:
            save_cam_examples(P, D, tr, model, thr, T, pairs[:: max(1, len(pairs) // 6)][:6], r["run_id"])
        del model
        torch.cuda.empty_cache()


def save_cam_examples(P, D, tr, model, thr, T, pairs, rid):
    st, L = tr.stride, tr.L
    fig, axes = plt.subplots(len(pairs), 4, figsize=(9, 2.4 * len(pairs)))
    axes = np.atleast_2d(axes)
    for row, (s, j, x) in enumerate(pairs):
        cc, dc = D.clip("clean", s)[::st][:L][None], D.clip("deg", j)[::st][:L][None]
        spec, has = D.audio(np.array([s]))
        xa, am = torch.from_numpy(spec).to(tr.device), torch.from_numpy(tr.amask(has)).to(tr.device)
        gc = grad_cam(model, to_video_tensor(cc, tr.device), xa, am, 1, thr, T)
        gd = grad_cam(model, to_video_tensor(dc, tr.device), xa, am, 1, thr, T)
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
# ANALYSIS (Test 6 runs + Test 5 baselines on the same test videos)
# =====================================================================================
def load_predictions(a, P: Paths) -> pd.DataFrame:
    fs = sorted(P.pred_dir.glob("*.csv.gz"))
    pred = pd.concat([read_csv(f, "t6_predictions") for f in fs], ignore_index=True) if fs else pd.DataFrame()
    if not len(pred):
        return pred
    pred["source_test"] = "Test 6"
    t5 = []
    for v5, name in T5_BASELINES.items():
        for t in [t for t in a.train_list if t in DATASETS]:
            for seed in a.seed_list:
                f = a.t5.pred_dir / f"{run_id(t, seed, v5)}.csv.gz"
                if f.exists():
                    x = read_csv(f, "t5_reference_predictions")
                    x = x[x.source_key.isin(set(pred.source_key))]          # same test videos (smoke subsets)
                    x["variant"], x["run_id"] = name, run_id(t, seed, name)
                    x["p_fake_raw"], x["source_test"] = x.p_fake, "Test 5"
                    t5.append(x)
    if t5:
        nfp = pred.drop_duplicates("source_key").set_index("source_key").native_face_px
        t5 = pd.concat(t5, ignore_index=True)
        t5["native_face_px"] = t5.source_key.map(nfp)
        pred = pd.concat([pred, t5], ignore_index=True)
    return pred


def load_cams(a, P: Paths) -> pd.DataFrame:
    fs = sorted(P.cam_dir.glob("*.csv"))
    cams = pd.concat([read_csv(f, "t6_gradcam") for f in fs], ignore_index=True) if fs else pd.DataFrame()
    if not len(cams):
        return cams
    t5 = []
    for v5, name in T5_BASELINES.items():
        for t in [t for t in a.train_list if t in DATASETS]:
            for seed in a.seed_list:
                f = a.t5.cam_dir / f"{run_id(t, seed, v5)}.csv"
                if f.exists():
                    x = read_csv(f, "t5_reference_gradcam")
                    x = x[x.pair_id.isin(set(cams.pair_id))]
                    x["variant"], x["run_id"] = name, run_id(t, seed, name)
                    t5.append(x)
    return pd.concat([cams] + t5, ignore_index=True) if t5 else cams


def setting_of(train_on: str, ds: str) -> str:
    return "joint" if train_on == "Joint" else ("in-domain" if ds == train_on else "cross-dataset")


def per_run_tables(pred: pd.DataFrame):
    met, degm, rob, cross = [], [], [], []
    for rid, g in pred.groupby("run_id"):
        r0 = g.iloc[0]
        thr = float(r0.threshold)
        base = dict(run_id=rid, train_on=r0.train_on, variant=r0.variant, seed=int(r0.seed), threshold=thr, source_test=r0.source_test)
        for ds, h in g.groupby("dataset"):
            setting = setting_of(r0.train_on, ds)
            n = h[h.condition == "normal"]
            c, d = n[n.kind == "clean"], n[n.kind == "deg"]
            for cn, x in [("clean", c), ("degraded_all", d), ("degraded_hard", d[d.degradation.isin(HARD_TYPES)])]:
                met.append(dict(base, test_dataset=ds, setting=setting, condition=cn, **compute_metrics(x.label, x.p_fake, thr)))
            cp = dict(zip(c.source_key, c.p_fake))
            d = d.assign(p_clean=d.source_key.map(cp))
            d = d.assign(flip=((d.p_fake >= thr) != (d.p_clean >= thr)), shift=(d.p_fake - d.p_clean).abs())
            type_auc = {}
            for (dg, sv), x in list(d.groupby(["degradation", "severity"])) + [((dg, "all"), x) for dg, x in d.groupby("degradation")]:
                m = compute_metrics(x.label, x.p_fake, thr)
                auc = safe_auc(x.label, x.p_fake, MIN_PER_CLASS_AUC)
                degm.append(dict(base, test_dataset=ds, setting=setting, degradation=dg, severity=sv, n=m["n"], n_real=m["n_real"],
                                 n_fake=m["n_fake"], balanced_accuracy=m["balanced_accuracy"], auc=auc,
                                 prediction_flip_rate_pct=100 * x.flip.mean(), mean_abs_prob_shift=x["shift"].mean()))
                if sv == "all":
                    type_auc[dg] = auc
            ts = [(k, v) for k, v in type_auc.items() if np.isfinite(v)]
            mc, md_ = compute_metrics(c.label, c.p_fake, thr), compute_metrics(d.label, d.p_fake, thr)
            rho = {}
            for lab, nm in ((0, "real"), (1, "fake")):
                x = c[(c.label == lab) & np.isfinite(c.native_face_px)]
                rho[nm] = float(spearmanr(x.native_face_px, x.p_fake)[0]) if len(x) > 5 and x.native_face_px.nunique() > 1 else np.nan
            rob.append(dict(base, test_dataset=ds, setting=setting, n_clean=len(c), n_degraded=len(d), clean_auc=mc["auc"],
                            degraded_auc_pooled=md_["auc"], clean_balanced_accuracy=mc["balanced_accuracy"],
                            degraded_balanced_accuracy=md_["balanced_accuracy"], real_recognised=mc["specificity"],
                            prediction_flip_rate_pct=100 * d.flip.mean() if len(d) else np.nan, mean_abs_prob_shift=d["shift"].mean(),
                            worst_type=min(ts, key=lambda t: t[1])[0] if ts else "", worst_type_auc=min(v for _, v in ts) if ts else np.nan,
                            macro_type_auc=float(np.mean([v for _, v in ts])) if ts else np.nan,
                            auc_hard_pooled=safe_auc(d[d.degradation.isin(HARD_TYPES)].label, d[d.degradation.isin(HARD_TYPES)].p_fake, MIN_PER_CLASS_AUC),
                            **{f"auc_{t}": type_auc.get(t, np.nan) for t in HARD_TYPES + ["noise", "compression"]},
                            ece_clean=ece_score(c.label, c.p_fake), ece_clean_raw=ece_score(c.label, c.p_fake_raw),
                            ece_degraded=ece_score(d.label, d.p_fake), brier_clean=float(np.mean((c.p_fake - c.label) ** 2)) if len(c) else np.nan,
                            rho_resolution_real=rho["real"], rho_resolution_fake=rho["fake"]))
            for cond in ("audio_masked", "audio_only"):
                x = h[h.condition == cond]
                if len(x):
                    pn = x.source_key.map(cp)
                    cross.append(dict(base, test_dataset=ds, setting=setting, condition=f"ablation_{cond}", n=len(x),
                                      auc=safe_auc(x.label, x.p_fake), predicted_fake_rate=float((x.p_fake >= thr).mean()),
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
        for scope, x in [("all", g), ("hard", g[g.degradation.isin(HARD_TYPES)])] + [(f"type:{t}", y) for t, y in g.groupby("degradation")]:
            if not len(x):
                continue
            st = x[x.prediction_stable]
            rows.append(dict(run_id=rid, train_on=r0.train_on, variant=r0.variant, seed=int(r0.seed), test_dataset=ds,
                             setting=setting_of(r0.train_on, ds), scope=scope, n_pairs=len(x), mean_pearson_r=x.pearson_r.mean(),
                             mean_top20_iou=x.top20_iou.mean(), mean_center_shift=x.center_shift.mean(),
                             mean_temporal_cam_r=x.temporal_cam_r.mean(), prediction_stable_rate=x.prediction_stable.mean(),
                             explanation_changed_given_stable_pct=100 * st.explanation_changed.mean() if len(st) else np.nan,
                             mean_clean_reference_iou=x.clean_reference_iou.mean()))
    return pd.DataFrame(rows)


SUMMARY_METRICS = ["clean_auc", "degraded_auc_pooled", "worst_type_auc", "macro_type_auc", "auc_hard_pooled", "auc_noise",
                   "prediction_flip_rate_pct", "real_recognised", "ece_clean", "ece_clean_raw", "rho_resolution_real"]


def seed_summary(rob: pd.DataFrame, cs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    srcs = [(m, rob) for m in SUMMARY_METRICS] + [(m, cs[cs.scope.isin(["all", "hard"])] if len(cs) else cs)
                                                  for m in ("mean_top20_iou", "mean_pearson_r", "explanation_changed_given_stable_pct")]
    for metric, df in srcs:
        if not len(df) or metric not in df:
            continue
        keys = ["train_on", "test_dataset", "setting", "variant"] + (["scope"] if "scope" in df else [])
        for k, g in df.groupby(keys):
            v = g[metric].dropna().values
            r = dict(zip(keys, k))
            r.update(metric=metric + (f" [{r.pop('scope')}]" if "scope" in r else ""), n_seeds=len(v), mean=v.mean() if len(v) else np.nan,
                     sd=v.std(ddof=1) if len(v) > 1 else np.nan, **{f"seed_{s}": x for s, x in zip(g.seed, g[metric])})
            rows.append(r)
    return pd.DataFrame(rows)


def bootstrap_stats(a, pred: pd.DataFrame, cams: pd.DataFrame) -> pd.DataFrame:
    """Paired comparisons on the same test videos (Test 6 variants and Test 5 baselines share split, items and seeds).
    Seed uncertainty: per-seed differences and sign agreement. Sample uncertainty: cluster bootstrap over source groups."""
    if not len(pred):
        return pd.DataFrame()
    rows = []
    for (train_on, ds), g in pred[pred.condition == "normal"].groupby(["train_on", "dataset"]):
        groups = np.array(sorted(g.group_id.unique()))
        G = len(groups)
        gpos = {x: i for i, x in enumerate(groups)}
        rng = np.random.default_rng([20261016, sum(map(ord, train_on + ds))])
        Wg = np.vstack([np.ones(G), rng.multinomial(G, np.full(G, 1.0 / G), size=a.n_boot).astype(np.float64)])
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

            out["clean_auc"], out["degraded_auc"] = wauc(c), wauc(d)
            out["hard_auc"] = wauc(d[d.degradation.isin(HARD_TYPES)])
            ta = [aa for _, x in d.groupby("degradation") for aa in [wauc(x)] if np.isfinite(aa[0])]
            out["worst_type_auc"] = np.nanmin(np.vstack(ta), axis=0) if ta else np.full(len(Wg), np.nan)
            out["noise_auc"] = wauc(d[d.degradation == "noise"])
            cp = dict(zip(c.source_key, c.p_fake))
            fl = ((d.p_fake >= thr) != (d.source_key.map(cp) >= thr)).astype(float).values
            W = Wg[:, d.group_id.map(gpos).values]
            out["flip_rate_pct"] = 100 * (W * fl).sum(1) / np.maximum(W.sum(1), 1e-9)
            if len(c):
                out["ece_clean"] = weighted_ece(c.label.values, c.p_fake.values, Wg[:, c.group_id.map(gpos).values])
            if len(camg):
                x = camg[camg.run_id == rid]
                for nm, y in (("gradcam_iou_all", x), ("gradcam_iou_hard", x[x.degradation.isin(HARD_TYPES)])):
                    if len(y):
                        Wc = Wg[:, y.group_id.map(gpos).fillna(-1).astype(int).values.clip(0)] * (y.group_id.isin(gpos).values)
                        out[nm] = (Wc * y.top20_iou.values).sum(1) / np.maximum(Wc.sum(1), 1e-9)
            per[(v, sd)] = out
        better_low = {"flip_rate_pct", "ece_clean"}
        for hi, lo in COMPARISONS:
            seeds = sorted({s for (v, s) in per if v == hi} & {s for (v, s) in per if v == lo})
            if not seeds:
                continue
            for metric in per[(hi, seeds[0])]:
                if not all(metric in per[(hi, s)] and metric in per[(lo, s)] for s in seeds):
                    continue
                diffs = np.vstack([per[(hi, s)][metric] - per[(lo, s)][metric] for s in seeds])
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
                    verdict = "single seed: CI excludes 0" if (lo_ci > 0 or hi_ci < 0) else "single seed: inconclusive"
                elif (lo_ci > 0 or hi_ci < 0) and same_sign:
                    verdict = "supported improvement" if np.sign(point.mean()) == want else "supported deterioration"
                else:
                    verdict = "inconclusive"
                rows.append(dict(train_on=train_on, test_dataset=ds, setting=setting_of(train_on, ds), comparison=f"{hi} - {lo}", metric=metric,
                                 better_direction="lower" if want < 0 else "higher", n_seeds=len(seeds), mean_diff=point.mean(),
                                 sd_diff_across_seeds=point.std(ddof=1) if len(seeds) > 1 else np.nan,
                                 per_seed_diffs=";".join(f"{s}:{d:+.4f}" for s, d in zip(seeds, point)), sign_agreement=same_sign,
                                 ci95_low=lo_ci, ci95_high=hi_ci, n_clusters=G, n_boot=len(boot), verdict=verdict))
    return pd.DataFrame(rows)


def make_figures(P: Paths, rob: pd.DataFrame, cs: pd.DataFrame, degm: pd.DataFrame):
    if not len(rob):
        return
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    order = [v for v in VARIANT_ORDER if v in set(rob.variant)]

    def panel(df, metric, title, fname, ylabel, ylim=None, settings=("in-domain", "joint")):
        keys = sorted({(t, ds) for t, ds, s in zip(df.train_on, df.test_dataset, df.setting) if s in settings})
        if not keys:
            return
        fig, axes = plt.subplots(1, len(keys), figsize=(4.2 * len(keys), 3.6), squeeze=False)
        for ax, (t, ds) in zip(axes[0], keys):
            x = df[(df.train_on == t) & (df.test_dataset == ds)]
            for k, v in enumerate(order):
                vals = x[x.variant == v][metric].dropna().values
                if not len(vals):
                    continue
                ax.bar(k, vals.mean(), color=VARIANT_COLOR[v], width=0.65)
                ax.scatter(np.full(len(vals), k) + np.linspace(-0.12, 0.12, len(vals)), vals, color="#0b0b0b", s=10, zorder=3)
                if len(vals) > 1:
                    ax.errorbar(k, vals.mean(), yerr=vals.std(ddof=1), color="#0b0b0b", capsize=3, lw=1)
            ax.set_xticks(range(len(order)))
            ax.set_xticklabels(order, rotation=30)
            ax.set_title(f"train {t} / test {ds}", fontsize=9)
            ax.set_ylabel(ylabel)
            if ylim:
                ax.set_ylim(*ylim)
            ax.grid(axis="y", alpha=0.3)
        fig.suptitle(title + " - bars: seed mean, dots: seeds, whiskers: SD", fontsize=9)
        plt.tight_layout()
        plt.savefig(P.figures / fname, dpi=130)
        plt.close()

    panel(rob, "clean_auc", "Original test AUC", "fig_clean_auc.png", "AUC", (0.4, 1.0))
    panel(rob, "degraded_auc_pooled", "Degraded test AUC", "fig_degraded_auc.png", "AUC", (0.4, 1.0))
    panel(rob, "worst_type_auc", "Worst degradation-type AUC", "fig_worst_case_auc.png", "AUC", (0.3, 1.0))
    panel(rob, "prediction_flip_rate_pct", "Prediction flips original -> degraded", "fig_prediction_flips.png", "% flipped")
    panel(rob, "ece_clean", "Calibration error (ECE, original test videos)", "fig_ece.png", "ECE")
    panel(rob, "clean_auc", "Cross-dataset AUC", "fig_cross_dataset_auc.png", "AUC", (0.2, 1.0), ("cross-dataset",))
    if len(cs):
        for scope in ("all", "hard"):
            x = cs[cs.scope == scope]
            if len(x):
                panel(x, "mean_top20_iou", f"Grad-CAM top-20% IoU original vs degraded ({scope} types)", f"fig_gradcam_iou_{scope}.png", "IoU", (0, 1))
    if len(degm):
        x = degm[degm.setting.isin(["in-domain", "joint"]) & (degm.severity == "all")]
        piv = x.groupby(["train_on", "test_dataset", "degradation", "variant"]).auc.mean().unstack("variant")
        piv = piv[[v for v in order if v in piv.columns]]
        if len(piv):
            fig, ax = plt.subplots(figsize=(1.2 + 0.7 * piv.shape[1], 0.28 * len(piv) + 1.5))
            im = ax.imshow(piv.values, cmap="RdYlGn", vmin=0.5, vmax=1.0, aspect="auto")
            ax.set_xticks(range(piv.shape[1]))
            ax.set_xticklabels(piv.columns, rotation=30)
            ax.set_yticks(range(len(piv)))
            ax.set_yticklabels([f"{t}->{d}: {g}" for t, d, g in piv.index], fontsize=7)
            for i in range(piv.shape[0]):
                for j in range(piv.shape[1]):
                    if np.isfinite(piv.values[i, j]):
                        ax.text(j, i, f"{piv.values[i, j]:.2f}", ha="center", va="center", fontsize=6)
            plt.colorbar(im, ax=ax, fraction=0.03)
            ax.set_title("AUC per degradation type (seed mean)", fontsize=9)
            plt.tight_layout()
            plt.savefig(P.figures / "fig_type_auc_heatmap.png", dpi=130)
            plt.close()


def phase_analyze(a, P: Paths):
    pred = load_predictions(a, P)
    if not len(pred):
        print("  no predictions yet - run --phase evaluate")
        return
    cams = load_cams(a, P)
    met, degm, rob, cross = per_run_tables(pred)
    cs = cam_summary(cams)
    met.to_csv(P.metrics / "metrics_test6.csv", index=False)
    degm.to_csv(P.metrics / "degradation_metrics_test6.csv", index=False)
    rob.to_csv(P.metrics / "robustness_test6.csv", index=False)
    cross.to_csv(P.metrics / "audio_ablation_test6.csv", index=False)
    if len(cams):
        cams.to_csv(P.metrics / "gradcam_stability_test6.csv", index=False)
        cs.to_csv(P.metrics / "gradcam_summary_test6.csv", index=False)
    seed_summary(rob, cs).to_csv(P.metrics / "seed_summary_test6.csv", index=False)
    print(f"  bootstrap statistics ({a.n_boot} cluster replicates) ...")
    bootstrap_stats(a, pred, cams).to_csv(P.metrics / "statistics_test6.csv", index=False)
    make_figures(P, rob, cs, degm)
    print(f"  metrics -> {P.metrics}; figures -> {P.figures}")


# =====================================================================================
# REPORT
# =====================================================================================
HYPOTHESES = [
    ("H1", "The MSTF-AV architecture improves robustness over the Test 5 baseline", "6A - T5-5A", ["degraded_auc", "worst_type_auc", "clean_auc"]),
    ("H2", "Paired Grad-CAM + supervised contrastive consistency improves explanation stability", "6B - 6A",
     ["gradcam_iou_hard", "gradcam_iou_all", "flip_rate_pct"]),
    ("H3", "Group-DRO sampling + EMA improve worst-case robustness without starving types", "6C - 6B", ["worst_type_auc", "noise_auc", "degraded_auc"]),
    ("H4", "Shortcut adversaries reduce shortcut use (better cross-dataset AUC, calibration)", "6D - 6C", ["clean_auc", "degraded_auc", "ece_clean"]),
    ("H5", "Full Test 6 (6D) beats the Test 5 baseline", "6D - T5-5A", ["degraded_auc", "worst_type_auc", "flip_rate_pct", "gradcam_iou_hard"]),
]


def hypothesis_verdicts(stt: pd.DataFrame) -> List[dict]:
    out = []
    for hid, text, comp, metrics in HYPOTHESES:
        x = stt[(stt.comparison == comp) & stt.metric.isin(metrics)] if len(stt) else stt
        if hid == "H4" and len(x):
            x = x[(x.setting == "cross-dataset") | (x.metric == "ece_clean")]
        elif len(x):
            x = x[x.setting.isin(["in-domain", "joint"])]
        if not len(x):
            out.append(dict(hypothesis=hid, comparison=comp, verdict="not tested (runs missing)", evidence=""))
            continue
        imp = (x.verdict == "supported improvement").sum()
        det = (x.verdict == "supported deterioration").sum()
        verdict = ("supported (all measured settings)" if imp and not det and imp == len(x) else "partially supported" if imp and not det
                   else "not supported (supported deterioration)" if det and not imp else "mixed" if imp and det else "inconclusive / not supported")
        ev = "; ".join(f"{r.train_on}->{r.test_dataset} {r.metric}: {r.mean_diff:+.4f} [{r.ci95_low:+.4f}, {r.ci95_high:+.4f}] ({r.verdict})"
                       for r in x.itertuples())
        out.append(dict(hypothesis=hid, statement=text, comparison=comp, verdict=verdict, evidence=ev))
    return out


def phase_report(a, P: Paths):
    f = P.metrics / "robustness_test6.csv"
    if not f.exists():
        print("  nothing to report yet")
        return
    rob = read_csv(f, "t6_metrics")
    cs = read_csv(P.metrics / "gradcam_summary_test6.csv", "t6_metrics") if (P.metrics / "gradcam_summary_test6.csv").exists() else pd.DataFrame()
    stt = read_csv(P.metrics / "statistics_test6.csv", "t6_metrics") if (P.metrics / "statistics_test6.csv").exists() else pd.DataFrame()
    aud = read_csv(P.metrics / "audio_ablation_test6.csv", "t6_metrics") if (P.metrics / "audio_ablation_test6.csv").exists() else pd.DataFrame()
    link = read_json(P.link, "t6_link")
    ck = read_csv(P.checkpoints / "checkpoint_hashes.csv", "t6_checkpoint_index") if (P.checkpoints / "checkpoint_hashes.csv").exists() else pd.DataFrame()
    order = [v for v in VARIANT_ORDER if v in set(rob.variant)]

    def ms_table(df, metric, settings, nd=3):
        x = df[df.setting.isin(settings)]
        if not len(x) or metric not in x:
            return "_(no rows)_"
        g = x.groupby(["train_on", "test_dataset", "variant"])[metric].agg(["mean", "std", "count"])
        rows = []
        for (t, ds), h in g.groupby(level=[0, 1]):
            r = {"train_on": t, "test_dataset": ds}
            for v in order:
                if (t, ds, v) in h.index:
                    m_, s_, n_ = h.loc[(t, ds, v)]
                    r[v] = f"{m_:.{nd}f} ± {s_:.{nd}f} (n={int(n_)})" if n_ > 1 else f"{m_:.{nd}f} (n=1)"
                else:
                    r[v] = "-"
            rows.append(r)
        return md_table(pd.DataFrame(rows))

    L = [f"# Test 6: MSTF-AV multi-stream detector with robust, consistent and shortcut-adversarial training\n",
         f"_Generated {now()} | script `{SCRIPT_PATH.name}` sha256 `{code_sha256()[:16]}…` | data = Test 5 split "
         f"`{str(link.get('split_hash_originals'))[:16]}…` (read-only, link {link.get('status')})_\n",
         "## 1. Design\n",
         "- **Architecture (MSTF-AV):** RGB residual CNN + SE and an SRM noise-residual stream joined by a cross-stream gate; "
         "temporal transformer over frame and frame-difference tokens; audio tokens fused by gated cross-modal attention; "
         f"{n_params(MSTFAV()):,} parameters (Test 5 model: 584,978).",
         "- **Variants:** " + "; ".join(f"{k}: {v['description']}" for k, v in VARIANTS.items()) + ".",
         "- **Baselines:** Test 5 variants 5A and 5C on the same split, test videos and seeds (T5-5A, T5-5C).",
         "- **Calibration:** temperature scaling on validation; thresholds chosen on calibrated probabilities.\n",
         "## 2. Runs\n", md_table(ck[["run_id", "best_epoch", "best_val_score", "temperature", "threshold", "val_ece_raw", "val_ece", "train_time_s"]])
         if len(ck) else "_(no checkpoints)_", "\n## 3. Results (seed mean ± SD)\n"]
    for metric, title in [("clean_auc", "AUC, original test videos"), ("degraded_auc_pooled", "AUC, degraded test videos"),
                          ("worst_type_auc", "Worst degradation-type AUC"), ("auc_noise", "Noise AUC (starvation check)"),
                          ("prediction_flip_rate_pct", "Prediction flip rate %"), ("real_recognised", "Real videos recognised"),
                          ("ece_clean", "ECE (calibrated)"), ("ece_clean_raw", "ECE (uncalibrated)")]:
        L += [f"**{title}** (in-domain / joint)\n", ms_table(rob, metric, ["in-domain", "joint"]), ""]
    L += ["**Cross-dataset AUC (original videos)**\n", ms_table(rob, "clean_auc", ["cross-dataset"]), "",
          "**Resolution shortcut: Spearman ρ(p_fake, native face px) within real videos**\n",
          ms_table(rob, "rho_resolution_real", ["in-domain", "joint", "cross-dataset"]), ""]
    if len(aud):
        L += ["**Audio ablation (flip rate vs normal, %)**\n", ms_table(aud[aud.condition == "ablation_audio_masked"], "flip_rate_vs_normal_pct",
                                                                         ["in-domain", "joint", "cross-dataset"], 1), ""]
    if len(cs):
        L += ["## 4. Grad-CAM stability\n", "**Top-20% IoU, all types**\n", ms_table(cs[cs.scope == "all"], "mean_top20_iou", ["in-domain", "joint"]),
              "", "**Top-20% IoU, hard types**\n", ms_table(cs[cs.scope == "hard"], "mean_top20_iou", ["in-domain", "joint"]), "",
              "**Explanation changed while prediction stable (%)**\n",
              ms_table(cs[cs.scope == "all"], "explanation_changed_given_stable_pct", ["in-domain", "joint"], 1), "",
              "Stability is not correctness: no manipulation-region ground truth exists.\n"]
    L += ["## 5. Statistics (paired cluster bootstrap + seed sign agreement)\n",
          md_table(stt[stt.verdict.str.startswith("supported")] if len(stt) else stt, 4),
          f"\n_{(stt.verdict == 'inconclusive').sum() if len(stt) else 0} further comparisons inconclusive (statistics_test6.csv)._\n",
          "## 6. Hypothesis verdicts\n", md_table(pd.DataFrame(hypothesis_verdicts(stt)), 4), "",
          "## 7. Notes for the thesis\n",
          "- Test 6 and Test 5 share split, test videos, XAI pairs and seeds: differences are paired and attributable to model/training.",
          "- 6A − T5-5A isolates the architecture (same recipe); 6B/6C/6D add one training block each.",
          "- Report only 'supported' effects; describe the rest as inconclusive at this sample size and seed count.",
          "- Joint models have no held-out dataset; cross-dataset claims come only from single-dataset models.",
          "- Grad-CAM consistency is trained in 6B-6D, so their stability is partly optimised (report alongside clean reference IoU)."]
    (P.reports / "TEST6_REPORT.md").write_text("\n".join(L), encoding="utf-8")
    write_json(P.reports / "final_summary_test6.json", dict(created=now(), script_sha256=code_sha256(), link=link,
                                                            hypotheses=hypothesis_verdicts(stt), checkpoints=ck.to_dict("records") if len(ck) else []))
    print(f"  report -> {P.reports / 'TEST6_REPORT.md'}")


# =====================================================================================
# SELFTEST (no data)
# =====================================================================================
def phase_selftest(a, P: Paths) -> bool:
    print("\n" + "=" * 100 + "\n PHASE SELFTEST (no data, no training)\n" + "=" * 100)
    res = {}
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    m = MSTFAV().to(dev)
    res["parameter count 1.5-4 M"] = 1.5e6 <= n_params(m) <= 4e6
    B = 2
    xv = torch.rand(B, CLIP_LEN, 3, IMG_SIZE, IMG_SIZE, device=dev) * 2 - 1
    xa = torch.randn(B, 1, 128, 128, device=dev)
    am = torch.tensor([1.0, 0.0], device=dev)
    m.eval()
    with torch.no_grad():
        outs = {md: m(xv, xa, mode=md, a_mask=am) for md in ("av", "v_only", "a_only")}
        res["forward shapes (av / v_only / a_only)"] = all(o.shape == (B, 2) and torch.isfinite(o).all() for o in outs.values())
        z0 = m(xv, xa, mode="av", a_mask=torch.zeros(B, device=dev))
        res["audio mask 0 == video only"] = torch.allclose(z0, m(xv, None, mode="v_only"), atol=1e-4)
    m.train()
    logits, z, v, act = m.forward_train(xv, xa, am)
    res["Grad-CAM layer is 8x8"] = tuple(act.shape[-2:]) == (8, 8) and act.shape[0] == B * CLIP_LEN
    cam = soft_gradcam(logits, act, B, CLIP_LEN)
    lc = cam_consistency_loss(cam[:1].detach(), cam[1:])
    lc.backward(retain_graph=True)
    res["soft Grad-CAM consistency is differentiable"] = any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.encoder.parameters())
    m.zero_grad()
    x = torch.randn(4, 5, requires_grad=True)
    GradReverse.apply(x, 0.5).sum().backward()
    res["gradient reversal (-lambda)"] = torch.allclose(x.grad, torch.full_like(x, -0.5))
    zz = torch.randn(8, 16)
    yy = torch.tensor([0, 0, 1, 1, 0, 1, 0, 1])
    with torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):     # as in training (AMP)
        res["supervised contrastive finite"] = bool(torch.isfinite(supcon_loss(nn.Linear(16, 16).to(dev)(zz.to(dev)), yy.to(dev))))
    za = F.normalize(torch.randn(8, 16), dim=1)
    aligned = torch.cat([za[:4] + 5 * torch.tensor([1.0] + [0] * 15), za[4:] - 5 * torch.tensor([1.0] + [0] * 15)])
    res["supcon lower for class-aligned embeddings"] = float(supcon_loss(aligned, torch.tensor([1, 1, 1, 1, 0, 0, 0, 0]))) < float(
        supcon_loss(za, torch.tensor([1, 1, 1, 1, 0, 0, 0, 0])))
    r = np.random.default_rng(0)
    yv = r.integers(0, 2, 4000)
    zt = (2 * yv - 1) * 1.0 + r.normal(0, 1, 4000)
    res["temperature recovers over-confidence (T>2 for x5 logits)"] = fit_temperature(5 * zt, yv) > 2.0
    res["ECE drops after temperature scaling"] = ece_score(yv, 1 / (1 + np.exp(-5 * zt / fit_temperature(5 * zt, yv)))) < ece_score(yv, 1 / (1 + np.exp(-5 * zt)))
    p = r.random(300)
    y = (r.random(300) < p).astype(int)
    res["WeightedAUC == roc_auc_score"] = abs(WeightedAUC(y, p)(np.ones((1, 300)))[0] - roc_auc_score(y, p)) < 1e-9
    res["weighted ECE == ECE"] = abs(weighted_ece(y, p, np.ones((1, 300)))[0] - ece_score(y, p)) < 1e-9
    tr = Trainer.__new__(Trainer)
    tr.a, tr.types = a, ["blur", "noise", "brightness"]
    dp = tr.dro_probs({"blur": 0.2, "noise": 2.0, "brightness": 0.4})
    res["DRO up-weights the hardest type"] = dp["noise"] == max(dp.values())
    res["DRO floor keeps every type >= floor/K"] = min(dp.values()) >= a.dro_floor / 3 - 1e-9
    f = lr_lambda(100)
    res["warm-up + cosine schedule"] = f(0) < f(14) <= 1.0 and f(99) < 0.01
    e = EMA(m, 0.5)
    with torch.no_grad():
        for p_ in m.parameters():
            p_.add_(1.0)
    e.update(m)
    res["EMA moves halfway"] = all(torch.allclose(pe, pm - 0.5, atol=1e-5) for pe, pm in zip(e.model.parameters(), m.parameters()))
    res["audio augmentation + spectrogram"] = waveform_to_spec(degrade_audio(r.normal(0, 0.1, AUDIO_SAMPLES).astype(np.float32), "lowpass", 1, r)).shape == (1, 128, 128)
    try:
        guard(T5_ROOT / "checkpoints" / "x.pt", "selftest")
        res["guard blocks Test 5 checkpoints"] = False
    except ForbiddenArtifactError:
        res["guard blocks Test 5 checkpoints"] = True
    for k, v in res.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    ok = all(res.values())
    write_json(P.reports / "selftest_report.json", dict(ok=ok, results=res, parameters=n_params(m), checked_at=now(),
                                                        architecture_signature=architecture_signature(MSTFAV())))
    print(f"  selftest {'PASSED' if ok else 'FAILED'} | MSTF-AV {n_params(m):,} parameters")
    return ok


# =====================================================================================
# STATUS + MAIN
# =====================================================================================
def phase_status(a, P: Paths):
    print(f"  link: {read_json(P.link, 't6_link').get('status') if P.link.exists() else '-'}")
    runs = available_runs(a, P) if P.checkpoints.exists() else []
    print(f"  checkpoints: {len(runs)} ({', '.join(r['run_id'] for r in runs[:60])})")
    print(f"  predictions: {len(list(P.pred_dir.glob('*.csv.gz')))} | Grad-CAM: {len(list(P.cam_dir.glob('*.csv')))}")
    print(f"  report: {'done' if (P.reports / 'TEST6_REPORT.md').exists() else '-'}")


def phase_run(a, P: Paths):
    print("\n" + "=" * 100 + f"\n TEST 6 | variants {a.variant_list} | seeds {a.seed_list} | training sets {a.train_list}\n" + "=" * 100)
    if not phase_selftest(a, P):
        raise SystemExit("[STOP] selftest failed")
    phase_link(a, P)
    D = T6Data(a, P)
    for k, seed in enumerate(a.seed_list, 1):
        print("\n" + "#" * 100 + f"\n SEED {seed} ({k}/{len(a.seed_list)})\n" + "#" * 100)
        phase_train(a, P, D, seeds=[seed])
        phase_evaluate(a, P, D)
        phase_xai(a, P, D)
        phase_analyze(a, P)
        phase_report(a, P)
        print(f"\n  seed {seed} done - interim report: {P.reports / 'TEST6_REPORT.md'}")
    print("\n  ALL RUNS FINISHED.")


def main(argv=None):
    a = parse_args(argv)
    lower_process_priority()
    torch.set_num_threads(max(1, min(4, (os.cpu_count() or 4) // 2)))
    P = Paths(a.root)
    P.make()
    _ACCESS["log"], _ACCESS["phase"] = P.access_log, a.phase
    sys.stdout = Tee(P.logs / f"{a.phase}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    print(f"TEST 6 | phase {a.phase} | root {P.root} | Test 5 inputs {a.t5.root} | seeds {a.seed_list} | variants {a.variant_list} | "
          f"train sets {a.train_list} | amp {a.amp} | {now()}")
    write_json(P.logs / f"environment_{a.phase}.json", environment_info())
    ph = a.phase
    if ph == "run":
        return phase_run(a, P)
    if ph == "selftest":
        sys.exit(0 if phase_selftest(a, P) else 1)
    if ph == "status":
        return phase_status(a, P)
    if ph == "link":
        phase_link(a, P)
        return
    if ph in ("train", "evaluate", "xai", "all"):
        require_link(P)
        D = T6Data(a, P)
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
