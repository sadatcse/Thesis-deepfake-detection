"""
FINALTEST.py - FINAL TEST OF THE THESIS: ROBUST, EXPLAINABLE AND CALIBRATED VIDEO DEEPFAKE DETECTOR
=====================================================================================================

Thesis: Robust Deepfake Detection and Explainability Under Real-World Video Degradation

Why this test (what Tests 1-7 established)
  * Degraded training restores robustness (Tests 1-4; replicated with three seeds in Test 5 and with a new model in Test 7).
  * Paired original/degraded training with a Grad-CAM consistency loss is the only component that replicated as an
    improvement of explanation stability (Test 5 variant 5C, Test 7 variant 6B).
  * A larger model trained from scratch did not help (Test 7). An ImageNet-pretrained EfficientNet-B0 with a temporal
    transformer gave the best detection on original videos (basic retest 151026.py; improvement suite 171026.py, best
    sampler = adaptive group-DRO over dataset x folder cells, "harddro12") - but it was never trained or tested on
    degraded videos.
  * Audio presence is a dataset shortcut (FF++ has no audio track) -> the final detector is video-only.
  * Isotonic calibration and a validation-tuned abstention band made the decisions reliable (171026, E5).
  * Grad-CAM++ was faithful for REAL decisions only (171026, E6); faithfulness under degradation was never measured.

Design (all settings fixed before any result is seen; written to configs/final_test_protocol.json before training)
  Data      The prepared Test 5 run is re-used READ-ONLY (split, face-crop caches, degraded versions, Grad-CAM pairs),
            after verifying its leakage gate and SHA-256 hashes. Every result is therefore PAIRED with Test 5 (5A, 5C)
            and Test 7 (6A, 6B): same test videos, same degraded versions, same Grad-CAM pairs, same seeds.
  Model     EfficientNet-B0 (ImageNet) per face frame -> temporal transformer over frame positions -> attention pooling
            -> P(fake). Video only. The 128 px Test 5 face crops (80 px face bottleneck = resolution shortcut removed) are
            up-sampled to 192 px on the GPU (the input scale of the basic retest).
  Variants  F0  Stage 1: fine-tuning on clean originals for 12 epochs (151026 recipe + the harddro12 group-DRO dataset x
                folder sampler of 171026, label smoothing, frame loss, EMA, real-world augmentation)
            F1  F0 + degradation-aware Stage 2: unpaired 50 % original / 50 % degraded clips (natural mixture)
            F2  F0 + explanation-consistent Stage 2: paired original/degraded clips + Grad-CAM consistency (0.05) +
                supervised contrastive consistency (0.10)                   <- pre-registered FINAL SYSTEM
  Training  Joint (both datasets: the final single detector), BioDeepAV and FF++ (dataset-specific models for the paired
  sets     comparison with Test 5 / Test 7 and for cross-dataset generalisation); seeds 42, 123, 2026.
  Decision  temperature scaling + validation threshold for every run; for the frozen final model additionally isotonic
            calibration and an abstention ("UNCERTAIN") band tuned on validation for 90 % coverage.

Pre-registered hypotheses (paired cluster bootstrap over source groups, 1,000 replicates + sign agreement of all seeds;
Holm correction over the eight primary comparisons; non-inferiority margins 0.01 AUC and 0.05 IoU)
  H1  pretrained backbone:   F1 vs Test 5 baseline 5A (in-domain) improves degraded and worst-type AUC
  H2  degraded training:     F1 vs F0 improves degraded AUC, worst-type AUC and prediction flips
  H3  explanation consistency: F2 vs F1 improves Grad-CAM stability while detection stays non-inferior
  H4  final vs earlier best: F2 (Joint) vs the dataset-specific 5A of Test 5 on the same test videos
  H5  generalisation:        cross-dataset AUC of the dataset-specific F2 models is above chance (95 % CI > 0.5)
  H6  reliability:           isotonic calibration lowers ECE on original AND degraded videos; abstention raises accuracy
  RQ  faithfulness under degradation (exploratory): deletion / insertion gains of Grad-CAM vs a random map on the
      original and on the degraded version of the same video

Phases (every phase is resumable; outputs only under run_FINALTEST)
  python FINALTEST.py --phase selftest     # unit tests, no data (about 1 min)
  python FINALTEST.py --smoke              # tiny end-to-end check in run_FINALTEST_smoke (about 15-25 min)
  python FINALTEST.py                      # everything: selftest -> link -> per seed: train, evaluate, Grad-CAM,
                                           # faithfulness, analyse, report -> freeze final model -> prediction check
                                           # (about 16 h on an RTX 3050; a full report exists after every seed,
                                           #  i.e. after about 5 h, 10 h and at the end)
  python FINALTEST.py --train-on Joint     # only the final single detector (about half the time; no paired 5A/5C runs)
  python FINALTEST.py --phase status
  python FINALTEST.py --predict "D:\\video.mp4"    # final detector on any video or folder: REAL / FAKE / UNCERTAIN
  python FINALTEST.py --phase gui          # small window: open a video, see the verdict and the explanation
  Individual phases: link, train, evaluate, xai, faith, analyze, report, freeze, predictcheck, all.
  Default micro-batch 8 clips x accumulation 2 (the 151026 setting). On a 4 GB GPU add  --batch-size 4 --accum 4  (an
  out-of-memory error also triggers this automatically; the effective batch, and therefore the protocol, stays the same).

Main outputs
  reports/FINAL_TEST_REPORT.md, reports/final_summary.json, metrics/*_final.csv, figures/*.png,
  final_model/FINAL_MODEL.pt + MODEL_CARD.md (+ examples), reports/predict_pipeline_check.json
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import gc
import hashlib
import importlib.util
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
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import cv2
from scipy import sparse
from scipy.stats import spearmanr
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.isotonic import IsotonicRegression  # noqa: E402
from sklearn.metrics import confusion_matrix, roc_auc_score  # noqa: E402

try:
    import torchvision
    from torchvision.models import efficientnet_b0
except ImportError:  # pragma: no cover
    torchvision = None
    efficientnet_b0 = None

warnings.filterwarnings("ignore")
cv2.setNumThreads(1)
try:
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
except Exception:
    pass
# Grad-CAM consistency differentiates through a gradient (double backward through the temporal transformer). The fused
# flash / memory-efficient / cuDNN attention kernels have no double backward, so attention runs on the math kernel.
for _sdp in ("enable_flash_sdp", "enable_mem_efficient_sdp", "enable_cudnn_sdp"):
    try:
        getattr(torch.backends.cuda, _sdp)(False)
    except Exception:
        pass

# =====================================================================================
# CONFIGURATION
# =====================================================================================
SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent
DEFAULT_ROOT = SCRIPT_DIR / "run_FINALTEST"
SMOKE_ROOT = SCRIPT_DIR / "run_FINALTEST_smoke"
T5_ROOT = SCRIPT_DIR / "run_test5_independent"          # prepared split + caches + Grad-CAM pairs (read-only input)
T5_SCRIPT = SCRIPT_DIR / "9-10-26_test5_independent.py"  # its preprocessing code is imported for --predict (identical crops)
T7_ROOT = SCRIPT_DIR / "run_16-10-26_test6"              # Test 7 of the thesis (161026.py; its own files say "Test 6")
# checkpoints / runs of earlier tests are never loaded: the final test starts from ImageNet weights only
FORBIDDEN_ARTIFACT_ROOTS = ([T5_ROOT / "checkpoints", T7_ROOT / "checkpoints"]
                            + [SCRIPT_DIR / d for d in ("run_151026", "run_171026", "run_201026", "run_201026_smoke", "run_6-10-26",
                                                        "run_1-10-26", "run_30-9-26", "run_test5_hard_explanation")])

TRAIN_SEEDS = "42,123,2026"                 # the Test 5 / Test 7 seeds (paired seed-wise comparison)
TRAIN_SETS = "Joint,BioDeepAV,FF++"         # Joint first: the final single detector exists before the dataset-specific runs
NUM_FRAMES, CLIP_LEN, CACHE_SIZE = 16, 8, 128
IMG = 192                                    # network input (cache crops are up-sampled on the GPU)
EMB_DIM = 256
BATCH_CLIPS, ACCUM = 8, 2                    # clips per micro-step x accumulation = 16 clips per update (151026 setting)
EPOCHS1, EPOCHS2, PATIENCE = 12, 8, 4          # Stage 1 = the 12-epoch harddro12 recipe of 171026
LR_BACKBONE1, LR_HEAD1 = 1e-4, 1e-3          # 151026 settings
LR_BACKBONE2, LR_HEAD2 = 5e-5, 5e-4          # Stage 2 continues from the Stage 1 checkpoint
WEIGHT_DECAY = 1e-4
WARMUP_FRAC = 0.08
LABEL_SMOOTH = 0.05
FRAME_LOSS_W = 0.3
EMA_DECAY = 0.998
GRAD_CLIP = 5.0
LAMBDA_CAM, LAMBDA_SUPCON, SUPCON_TAU = 0.05, 0.10, 0.10    # Test 5 (5C) / Test 7 (6B) values
DRO_ETA, DRO_FLOOR, DRO_MOM = 2.0, 0.5, 0.5                  # 171026 harddro12 values
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
TEMP_GRID = np.exp(np.linspace(np.log(0.05), np.log(20.0), 241))
THRESHOLD_GRID = (0.01, 0.99, 0.01)
ECE_BINS = 15
COVERAGES = (1.00, 0.95, 0.90, 0.80)
DEPLOY_COVERAGE = 0.90
EXPLANATION_CHANGE_IOU = 0.5
XAI_CHUNK = 8
FAITH_STRIDE = 4                             # every 4th pre-registered Grad-CAM pair (180 of 720)
FAITH_FRACS = [0.0, 0.05, 0.10, 0.20, 0.30, 0.50, 0.75]
FAITH_SEED = 20261023
N_BOOT = 1000
MIN_CLUSTERS = 10
MIN_PER_CLASS_AUC = 5
NONINF_AUC = 0.01
NONINF_XAI = 0.05
SELECT_TOL = 0.005                           # final model: F2 unless its mean validation score is > 0.005 below F1
HARD_TYPES = ["brightness", "saturation", "contrast", "blur"]
DATASETS = ["BioDeepAV", "FF++"]
FF_FAKES = ["Deepfakes", "DeepFakeDetection", "Face2Face", "FaceShifter", "FaceSwap", "NeuralTextures"]
VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".mpg", ".mpeg", ".wmv", ".flv"}
NO_WINDOW = subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0
METRIC_KEYS = ["auc", "accuracy", "balanced_accuracy", "precision", "recall", "sensitivity", "specificity", "f1"]

STAGE1 = "F0"
F0_DESCRIPTION = ("ImageNet EfficientNet-B0 + temporal transformer + attention pooling, fine-tuned on clean originals with the "
                  "group-DRO dataset x folder sampler (151026 / harddro12 recipe)")
VARIANTS = {
    "F1": dict(paired=False, lambda_cam=0.0, lambda_supcon=0.0,
               description="F0 + degradation-aware Stage 2: unpaired 50/50 original/degraded clips, natural degradation mixture"),
    "F2": dict(paired=True, lambda_cam=LAMBDA_CAM, lambda_supcon=LAMBDA_SUPCON,
               description="F0 + explanation-consistent Stage 2: paired original/degraded clips + Grad-CAM consistency + "
                           "supervised contrastive consistency (final system)"),
}
BASELINES = [("T5", "5A", "T5-5A"), ("T5", "5C", "T5-5C"), ("T7", "6A", "T7-6A"), ("T7", "6B", "T7-6B")]
BASELINE_TEXT = {"T5-5A": "Test 5 baseline 5A (from-scratch AV CNN, degraded training)",
                 "T5-5C": "Test 5 variant 5C (+ paired Grad-CAM consistency)",
                 "T7-6A": "Test 7 variant 6A (MSTF-AV, degraded training)",
                 "T7-6B": "Test 7 variant 6B (MSTF-AV + Grad-CAM / contrastive consistency)"}
VARIANT_ORDER = ["T5-5A", "T5-5C", "T7-6A", "T7-6B", "F0", "F1", "F2"]
VARIANT_COLOR = {"T5-5A": "#cfcdc4", "T5-5C": "#b9b7ad", "T7-6A": "#a9c1df", "T7-6B": "#a9d9c4", "F0": "#8a8984",
                 "F1": "#2a78d6", "F2": "#eb6834"}
VARIANT_HATCH = {"T5-5C": "////", "T7-6B": "////"}
# paired comparisons within a training set (hi - lo)
COMPARISONS = [("F1", "F0"), ("F2", "F1"), ("F2", "F0"), ("F1", "T5-5A"), ("F2", "T5-5A"), ("F2", "T5-5C"),
               ("F1", "T7-6A"), ("F2", "T7-6B")]
# paired comparisons across training sets on the same test videos: the joint final detector vs dataset-specific models
CROSS_COMPARISONS = [(("Joint", "F2"), ("BioDeepAV", "T5-5A"), "BioDeepAV"), (("Joint", "F2"), ("FF++", "T5-5A"), "FF++"),
                     (("Joint", "F2"), ("BioDeepAV", "F2"), "BioDeepAV"), (("Joint", "F2"), ("FF++", "F2"), "FF++")]
# the eight primary comparisons (Holm-corrected family): (comparison, train_on, test_dataset, metric)
PRIMARY = [("F1 - T5-5A", "BioDeepAV", "BioDeepAV", "degraded_auc"), ("F1 - T5-5A", "FF++", "FF++", "degraded_auc"),
           ("F1 - F0", "Joint", "BioDeepAV", "degraded_auc"), ("F1 - F0", "Joint", "FF++", "degraded_auc"),
           ("F2 - F1", "Joint", "BioDeepAV", "gradcam_iou_hard"), ("F2 - F1", "Joint", "FF++", "gradcam_iou_hard"),
           ("F2@Joint - T5-5A@BioDeepAV", "Joint", "BioDeepAV", "degraded_auc"),
           ("F2@Joint - T5-5A@FF++", "Joint", "FF++", "degraded_auc")]
LOWER_BETTER = {"flip_rate_pct", "ece_clean", "ece_degraded", "changed_while_stable_pct"}
AUC_METRICS = {"clean_auc", "degraded_auc", "hard_auc", "worst_type_auc", "noise_auc"}
IOU_METRICS = {"gradcam_iou_all", "gradcam_iou_hard"}
HYPOTHESES = [
    ("H1", "An ImageNet-pretrained backbone improves robustness over the Test 5 baseline (F1 vs 5A, in-domain)"),
    ("H2", "Degraded training also helps the pretrained detector (F1 vs F0)"),
    ("H3", "Explanation-consistency training improves Grad-CAM stability without losing detection (F2 vs F1)"),
    ("H4", "The final joint detector beats the best earlier detector on the same test videos (F2 Joint vs 5A)"),
    ("H5", "The dataset-specific final detectors generalise across datasets (cross-dataset AUC CI above 0.5)"),
    ("H6", "Isotonic calibration and abstention make decisions reliable on original and degraded videos"),
]


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
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(obj), f, indent=2)
    replace_retry(tmp, path)


def write_csv(df: pd.DataFrame, path: Path, **kw) -> None:
    """Atomic CSV write (a half-written file is never mistaken for a finished result after a power cut)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    df.to_csv(tmp, index=False, **kw)
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


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(x, dtype=np.float64), -60, 60)))


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
                torchvision=getattr(torchvision, "__version__", None), opencv=cv2.__version__,
                gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                gpu_memory_gb=round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2) if torch.cuda.is_available() else None,
                numpy=np.__version__, pandas=pd.__version__, script=str(SCRIPT_PATH), script_sha256=code_sha256(), recorded_at=now())


class Tee(io.TextIOBase):
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(path, "a", encoding="utf-8")
        self.out = sys.stdout

    def write(self, s):
        try:
            self.out.write(s)
        except UnicodeEncodeError:
            self.out.write(s.encode("ascii", "replace").decode("ascii"))
        self.f.write(s)
        if "\n" in s or "\r" in s:
            self.out.flush()
            self.f.flush()
        return len(s)

    def flush(self):
        self.out.flush()
        self.f.flush()


def prefetch(fn: Callable, chunks: List, workers: int = 2):
    """Yield (chunk, fn(chunk)) in order while the next chunks are loaded in background threads (memmap reads)."""
    ex = ThreadPoolExecutor(max_workers=max(1, workers))
    q, it = deque(), iter(chunks)
    try:
        for _ in range(workers + 1):
            try:
                c = next(it)
            except StopIteration:
                break
            q.append((c, ex.submit(fn, c)))
        while q:
            c, fut = q.popleft()
            res = fut.result()
            try:
                c2 = next(it)
                q.append((c2, ex.submit(fn, c2)))
            except StopIteration:
                pass
            yield c, res
    finally:
        ex.shutdown(wait=True)


# =====================================================================================
# PATHS + ARTIFACT ACCESS GUARD
# =====================================================================================
class Paths:
    SUBDIRS = ("configs", "checkpoints", "logs", "metrics", "figures", "reports", "final_model", "predict")

    def __init__(self, root: Path):
        self.root = Path(root)
        for d in self.SUBDIRS:
            setattr(self, d, self.root / d)
        self.link = self.reports / "data_link.json"
        self.protocol = self.configs / "final_test_protocol.json"
        self.access_log = self.logs / "file_access_log.csv"
        self.pred_dir = self.metrics / "predictions"
        self.cam_dir = self.metrics / "gradcam_runs"
        self.faith_dir = self.metrics / "faithfulness_runs"
        self.hist_dir = self.logs / "training"
        self.examples = self.figures / "gradcam_examples"
        self.final = self.final_model / "FINAL_MODEL.pt"
        self.ck_index = self.checkpoints / "checkpoint_hashes.csv"

    def make(self):
        for d in self.SUBDIRS:
            getattr(self, d).mkdir(parents=True, exist_ok=True)
        for d in (self.pred_dir, self.cam_dir, self.faith_dir, self.hist_dir, self.examples):
            d.mkdir(parents=True, exist_ok=True)


class T5Paths:
    """Read-only view of the prepared Test 5 run (split, caches, Grad-CAM pairs, baseline predictions)."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.split_manifest = self.root / "splits" / "split_manifest.csv"
        self.gate = self.root / "reports" / "gate_status.json"
        self.provenance = self.root / "cache" / "provenance.json"
        self.cache = self.root / "cache"
        self.xai_pairs = self.root / "manifests" / "xai_pairs.csv"
        self.pred_dir = self.root / "metrics" / "predictions"
        self.cam_dir = self.root / "metrics" / "gradcam_runs"


class T7Paths:
    """Read-only view of Test 7 (161026.py, run_16-10-26_test6): baseline predictions and Grad-CAM rows only."""

    def __init__(self, root: Path):
        self.root = Path(root)
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
    """Every artifact read passes here and is logged. Checkpoints / runs of earlier tests can never be loaded."""
    p = Path(path).resolve()
    forbidden = any(_is_under(p, r) for r in FORBIDDEN_ARTIFACT_ROOTS)
    log = _ACCESS["log"]
    if log is not None:
        with _ACCESS_LOCK:
            new = not Path(log).exists()
            with open(log, "a", encoding="utf-8") as f:
                if new:
                    f.write("time,phase,purpose,path,blocked\n")
                f.write(f"{now()},{_ACCESS['phase']},{purpose},\"{p}\",{forbidden}\n")
    if forbidden:
        raise ForbiddenArtifactError(f"the final test must not load {p} ({purpose})")
    return p


def read_csv(path, purpose: str, **kw) -> pd.DataFrame:
    return pd.read_csv(guard(path, purpose), **kw)


def read_json(path, purpose: str):
    with open(guard(path, purpose), "r", encoding="utf-8") as f:
        return json.load(f)


# =====================================================================================
# CLI
# =====================================================================================
PHASES = ["run", "selftest", "link", "status", "train", "evaluate", "xai", "faith", "analyze", "report", "freeze",
          "predictcheck", "predict", "gui", "all"]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="FINAL TEST: robust, explainable and calibrated video deepfake detector")
    p.add_argument("--phase", default="run", choices=PHASES)
    p.add_argument("--predict", default=None, help="video file or folder for the final detector (implies --phase predict)")
    p.add_argument("--root", default=None, help=f"experiment root (default {DEFAULT_ROOT})")
    p.add_argument("--t5-root", default=str(T5_ROOT), help="prepared Test 5 run (split + caches), read-only")
    p.add_argument("--t5-script", default=str(T5_SCRIPT), help="Test 5 script (its preprocessing is imported for --predict)")
    p.add_argument("--t7-root", default=str(T7_ROOT), help="Test 7 run (161026.py) for the paired baselines, read-only")
    p.add_argument("--seeds", default=TRAIN_SEEDS)
    p.add_argument("--variants", default="F1,F2", help="Stage 2 variants (F0 = Stage 1 is always trained)")
    p.add_argument("--train-on", default=TRAIN_SETS)
    p.add_argument("--epochs1", type=int, default=EPOCHS1)
    p.add_argument("--epochs2", type=int, default=EPOCHS2)
    p.add_argument("--patience", type=int, default=PATIENCE)
    p.add_argument("--batch-size", type=int, default=BATCH_CLIPS, help="clips per micro-step (Stage 2: half originals, half degraded)")
    p.add_argument("--accum", type=int, default=ACCUM, help="gradient accumulation (use --batch-size 4 --accum 4 on 4 GB GPUs)")
    p.add_argument("--img", type=int, default=IMG, help="network input size (cache crops are 128 px)")
    p.add_argument("--lr-backbone1", type=float, default=LR_BACKBONE1)
    p.add_argument("--lr-head1", type=float, default=LR_HEAD1)
    p.add_argument("--lr-backbone2", type=float, default=LR_BACKBONE2)
    p.add_argument("--lr-head2", type=float, default=LR_HEAD2)
    p.add_argument("--lambda-cam", type=float, default=LAMBDA_CAM)
    p.add_argument("--lambda-supcon", type=float, default=LAMBDA_SUPCON)
    p.add_argument("--ema-decay", type=float, default=EMA_DECAY)
    p.add_argument("--no-augment", action="store_true", help="disable the 151026 real-world augmentation")
    p.add_argument("--no-amp", action="store_true", help="disable mixed precision")
    p.add_argument("--allow-random-init", action="store_true", help="pipeline check without ImageNet weights (not for results)")
    p.add_argument("--n-boot", type=int, default=N_BOOT)
    p.add_argument("--faith-scope", default="joint", choices=["joint", "all", "none"],
                   help="runs for the faithfulness test (joint = the 9 jointly trained models)")
    p.add_argument("--max-steps", type=int, default=None, help="cap steps per epoch (debugging)")
    p.add_argument("--loader-threads", type=int, default=4)
    p.add_argument("--eval-batch", type=int, default=16, help="clips per evaluation batch")
    p.add_argument("--eval-models", type=int, default=9, help="models evaluated together in one pass over the cached clips")
    p.add_argument("--skip-selftest", action="store_true", help="start the run without the selftest (not recommended)")
    p.add_argument("--save-every-steps", type=int, default=200, help="crash-safe resume checkpoint every N steps")
    p.add_argument("--smoke", action="store_true", help="tiny end-to-end run in a separate root")
    p.add_argument("--smoke-per-cell", type=int, default=12)
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--gpu-max-temp", type=int, default=82)
    p.add_argument("--gpu-pause", type=int, default=120)
    p.add_argument("--predictcheck-n", type=int, default=24, help="videos re-extracted from disk in the prediction check")
    p.add_argument("--overlay-video", action="store_true", help="--predict: also write an MP4 with the heat-map on the frames")
    p.add_argument("--force", action="store_true", help="--predict: run even if the Test 5 preprocessing code changed")
    a = p.parse_args(argv)
    if a.predict:
        a.phase = "predict"
    if a.smoke:
        a.epochs1, a.epochs2 = min(a.epochs1, 1), min(a.epochs2, 1)
        a.n_boot = min(a.n_boot, 100)
        a.max_steps = a.max_steps or 3
        a.seeds = a.seeds if a.seeds != TRAIN_SEEDS else "42,123"
        a.predictcheck_n = min(a.predictcheck_n, 4)
    root = Path(a.root) if a.root else (SMOKE_ROOT if a.smoke else DEFAULT_ROOT)
    if a.smoke and root.resolve() == DEFAULT_ROOT.resolve():
        raise SystemExit("--smoke must not write into the main final-test root")
    a.root = root
    a.t5 = T5Paths(Path(a.t5_root))
    a.t7 = T7Paths(Path(a.t7_root))
    a.seed_list = [int(s) for s in str(a.seeds).split(",") if s.strip()]
    a.variant_list = [v.strip().upper() for v in a.variants.split(",") if v.strip()]
    bad = [v for v in a.variant_list if v not in VARIANTS]
    if bad:
        raise SystemExit(f"unknown variants {bad} (choose from {list(VARIANTS)})")
    alias = {"bio": "BioDeepAV", "biodeepav": "BioDeepAV", "ffpp": "FF++", "ff++": "FF++", "joint": "Joint", "both": "Joint"}
    a.train_list = [alias.get(t.strip().lower(), t.strip()) for t in a.train_on.split(",") if t.strip()]
    badt = [t for t in a.train_list if t not in ("Joint", "BioDeepAV", "FF++")]
    if badt:
        raise SystemExit(f"unknown training sets {badt}")
    a.augment = not a.no_augment
    a.amp = (not a.no_amp) and torch.cuda.is_available()
    if a.batch_size < 4 or a.batch_size % 2:
        raise SystemExit("--batch-size must be an even number >= 4")
    return a


def variant_config(a, v: str) -> dict:
    if v == STAGE1:
        return dict(paired=False, lambda_cam=0.0, lambda_supcon=0.0, description=F0_DESCRIPTION)
    c = dict(VARIANTS[v])
    if c["lambda_cam"] > 0:
        c["lambda_cam"] = a.lambda_cam
    if c["lambda_supcon"] > 0:
        c["lambda_supcon"] = a.lambda_supcon
    return c


# =====================================================================================
# MODEL: EfficientNet-B0 (ImageNet) -> temporal transformer -> attention pooling (151026 architecture)
# =====================================================================================
class FinalDetector(nn.Module):
    """EfficientNet-B0 per face frame -> temporal transformer over frame positions -> attention pooling -> P(fake).
    Identical to the basic-retest / improvement-suite detector (151026, 171026), plus a small projection head that is
    only used by the supervised contrastive term of F2 (it never influences the prediction)."""

    def __init__(self, pretrained: bool = True, d: int = EMB_DIM, max_t: int = NUM_FRAMES, allow_random: bool = False):
        super().__init__()
        if torchvision is None or efficientnet_b0 is None:
            raise SystemExit("[STOP] torchvision is required: pip install torchvision")
        self.pretrained = False
        net = None
        if pretrained:
            try:
                net = efficientnet_b0(weights=torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1)
                self.pretrained = True
            except Exception as e:
                if not allow_random:
                    raise SystemExit(f"[STOP] ImageNet weights of EfficientNet-B0 could not be loaded ({type(e).__name__}: {e}). "
                                     "The final test needs the pretrained backbone: connect to the internet once (torchvision "
                                     "caches the weights in its hub folder) or use --allow-random-init for a pipeline check only.")
                print(f"  [warn] ImageNet weights unavailable ({type(e).__name__}) - random initialisation (pipeline check only)")
        if net is None:
            net = efficientnet_b0(weights=None)
        self.backbone = net.features                                   # [N,1280,h,w]; last block = Grad-CAM layer
        self.proj = nn.Sequential(nn.Linear(1280, d), nn.LayerNorm(d), nn.GELU(), nn.Dropout(0.2))
        self.pos = nn.Embedding(max_t, d)
        self.temporal = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 4, 2 * d, 0.1, activation="gelu", batch_first=True,
                                                                         norm_first=True), 1, enable_nested_tensor=False)
        self.att = nn.Linear(d, 1)
        self.frame_head = nn.Linear(d, 1)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Dropout(0.3), nn.Linear(d, 1))
        self.con_head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 128))

    @property
    def cam_layer(self):
        return self.backbone[-1]

    def from_act(self, act: torch.Tensor, B: int, T: int, t_idx=None) -> Dict[str, torch.Tensor]:
        """Everything after the last convolutional block: act [B*T,C,h,w] -> logits, frame logits, attention, embedding."""
        h = self.proj(act.mean((2, 3))).view(B, T, -1)
        if t_idx is None:
            t_idx = torch.arange(T, device=act.device).expand(B, T)
        h = self.temporal(h + self.pos(t_idx))
        w = torch.softmax(self.att(h).squeeze(-1).float(), 1)
        z = (w.unsqueeze(-1) * h.float()).sum(1)
        return dict(logit=self.head(z.to(h.dtype)).squeeze(-1).float(), frame_logit=self.frame_head(h).squeeze(-1).float(),
                    attn=w, z=z)

    def forward(self, x: torch.Tensor, t_idx=None) -> Dict[str, torch.Tensor]:
        """x [B,T,3,H,W] ImageNet-normalised; t_idx [B,T] frame positions (0..15)."""
        B, T = x.shape[:2]
        act = self.backbone(x.flatten(0, 1))
        out = self.from_act(act, B, T, t_idx)
        out["act"] = act
        return out


def architecture_signature(model: nn.Module) -> str:
    sd = model.state_dict()
    return sha256_text(type(model).__name__ + "\n" + "\n".join(f"{k}:{tuple(v.shape)}" for k, v in sd.items()))


def architecture_source_sha() -> str:
    return sha256_text(inspect.getsource(FinalDetector))


def weights_sha256(sd) -> str:
    h = hashlib.sha256()
    for k in sorted(sd):
        h.update(k.encode())
        h.update(sd[k].detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


_NORM_CACHE: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}


def to_input(clips_u8: np.ndarray, device, img: int = IMG) -> torch.Tensor:
    """uint8 RGB [B,T,H,W,3] -> float [B,T,3,img,img], ImageNet-normalised; bilinear resize on the device."""
    x = torch.from_numpy(np.ascontiguousarray(clips_u8)).to(device, non_blocking=True)
    B, T, H, W, C = x.shape
    x = x.reshape(B * T, H, W, C).permute(0, 3, 1, 2).float().div_(255.0)
    if (H, W) != (img, img):
        x = F.interpolate(x, size=(img, img), mode="bilinear", align_corners=False)
    key = str(device)
    if key not in _NORM_CACHE:
        _NORM_CACHE[key] = (torch.tensor(MEAN, device=device).view(1, 3, 1, 1), torch.tensor(STD, device=device).view(1, 3, 1, 1))
    m, s = _NORM_CACHE[key]
    return ((x - m) / s).reshape(B, T, 3, img, img)


def frame_positions(B: int, idx, device) -> torch.Tensor:
    return torch.as_tensor(np.asarray(idx), dtype=torch.long, device=device).view(1, -1).expand(B, -1)


# =====================================================================================
# AUGMENTATION (151026 real-world operators; identical parameters for both clips of an original/degraded pair)
# =====================================================================================
def aug_params(rng, S: int) -> dict:
    sc = rng.uniform(0.80, 1.0)
    cs = int(S * sc)
    y0, x0 = (int(v) for v in rng.integers(0, S - cs + 1, 2))
    p = dict(cs=cs, y0=y0, x0=x0, flip=bool(rng.random() < 0.5), bright=float(rng.uniform(-0.15, 0.15)),
             contrast=float(rng.uniform(0.85, 1.15)), sat=float(rng.uniform(0.8, 1.2)))
    p["jpeg"] = int(rng.integers(40, 96)) if rng.random() < 0.5 else None
    p["blur"] = float(rng.uniform(0.4, 1.4)) if rng.random() < 0.2 else None
    p["down"] = float(rng.uniform(0.45, 0.9)) if rng.random() < 0.3 else None
    p["erase"] = None
    if rng.random() < 0.25:
        ew, eh = int(S * rng.uniform(0.15, 0.4)), int(S * rng.uniform(0.15, 0.4))
        p["erase"] = (int(rng.integers(0, S - eh)), int(rng.integers(0, S - ew)), eh, ew, rng.integers(0, 256, 3).astype(np.uint8))
    return p


def apply_aug(clip: np.ndarray, p: Optional[dict]) -> np.ndarray:
    """clip uint8 [T,S,S,3] RGB; the same operators and parameters for every frame of the clip."""
    if p is None:
        return np.ascontiguousarray(clip)
    T, S = clip.shape[:2]
    out = np.empty_like(clip)
    for t in range(T):
        f = clip[t, p["y0"]:p["y0"] + p["cs"], p["x0"]:p["x0"] + p["cs"]]
        f = cv2.resize(np.ascontiguousarray(f), (S, S), interpolation=cv2.INTER_LINEAR)
        if p["flip"]:
            f = np.ascontiguousarray(f[:, ::-1])
        if p["down"]:
            k = max(16, int(S * p["down"]))
            f = cv2.resize(cv2.resize(f, (k, k), interpolation=cv2.INTER_AREA), (S, S), interpolation=cv2.INTER_LINEAR)
        if p["blur"]:
            f = cv2.GaussianBlur(f, (0, 0), p["blur"])
        g = f.astype(np.float32)
        gray = g.mean(-1, keepdims=True)
        g = gray + (g - gray) * p["sat"]
        g = (g - 128.0) * p["contrast"] + 128.0 + 255.0 * p["bright"]
        f = np.clip(g, 0, 255).astype(np.uint8)
        if p["jpeg"]:
            ok, enc = cv2.imencode(".jpg", np.ascontiguousarray(f[..., ::-1]), [cv2.IMWRITE_JPEG_QUALITY, int(p["jpeg"])])
            if ok:
                f = cv2.imdecode(enc, cv2.IMREAD_COLOR)[..., ::-1]
        if p["erase"] is not None:
            ey, ex, eh, ew, col = p["erase"]
            f = np.array(f, copy=True)
            f[ey:ey + eh, ex:ex + ew] = col
        out[t] = f
    return out


# =====================================================================================
# LOSSES, GRAD-CAM, CALIBRATION
# =====================================================================================
def soft_gradcam(logit: torch.Tensor, act: torch.Tensor, B: int, T: int, create_graph: bool = True) -> torch.Tensor:
    """Differentiable Grad-CAM of the FAKE logit, averaged over the frames of each clip and max-normalised: [B,h,w]."""
    g = torch.autograd.grad(logit.float().sum(), act, create_graph=create_graph, retain_graph=True)[0]
    cam = torch.relu((g.float().mean((2, 3), keepdim=True) * act.float()).sum(1))
    cam = cam.view(B, T, *cam.shape[-2:]).mean(1)
    return cam / (cam.flatten(1).amax(1).view(-1, 1, 1) + 1e-6)


def cam_consistency_loss(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """1 - soft IoU between the (detached) original map a and the degraded map b (Test 5 / Test 7 definition)."""
    return (1.0 - torch.minimum(a, b).flatten(1).sum(1) / (torch.maximum(a, b).flatten(1).sum(1) + 1e-6)).mean()


def supcon_loss(z: torch.Tensor, y: torch.Tensor, tau: float = SUPCON_TAU) -> torch.Tensor:
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


def math_sdp():
    """Force the math attention kernel: the Grad-CAM consistency loss differentiates through a gradient, and only the math
    kernel has a double backward (also set globally at import; this context makes it independent of the PyTorch version)."""
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
        return sdpa_kernel([SDPBackend.MATH])
    except Exception:
        return contextlib.nullcontext()


def grad_cam_eval(model: FinalDetector, x: torch.Tensor, t_idx: torch.Tensor, sign: Any = 1.0, out_size: int = CACHE_SIZE,
                  amp: bool = False) -> Dict[str, np.ndarray]:
    """Grad-CAM and Grad-CAM++ on the last EfficientNet block, up-sampled to the 128 px crop.
    The backbone runs without an autograd graph; only the head is differentiated (cheap; no hook, so no GPU-memory leak).
    sign: +1 = evidence for FAKE, -1 = evidence for REAL, an array per clip, or callable(logits) -> signs."""
    model.eval()
    B, T = x.shape[:2]
    with torch.no_grad():
        with torch.autocast("cuda", dtype=torch.float16, enabled=bool(amp and x.is_cuda)):
            act = model.backbone(x.flatten(0, 1))
    act = act.detach().float().requires_grad_(True)
    with torch.enable_grad():
        o = model.from_act(act, B, T, t_idx)
        lg = o["logit"].detach().cpu().numpy().astype(np.float64)
        s = sign(lg) if callable(sign) else sign
        s = np.broadcast_to(np.asarray(s, dtype=np.float32), (B,)).copy()
        st = torch.from_numpy(s).to(act.device)
        G = torch.autograd.grad((st * o["logit"]).sum(), act)[0]
    A, G = act.detach(), G.detach().float()
    cam = torch.relu((G.mean((2, 3), keepdim=True) * A).sum(1))                     # Grad-CAM          [B*T,h,w]
    g2, g3 = G ** 2, G ** 3
    den = 2 * g2 + A.sum((2, 3), keepdim=True) * g3
    alpha = g2 / torch.where(den.abs() > 1e-12, den, torch.ones_like(den))
    campp = torch.relu(((alpha * torch.relu(G)).sum((2, 3), keepdim=True) * A).sum(1))  # Grad-CAM++    [B*T,h,w]

    def up(c):
        return F.interpolate(c[:, None], size=(out_size, out_size), mode="bilinear", align_corners=False)[:, 0].view(B, T, out_size, out_size)

    def spatial(c):
        s_ = c.mean(1).flatten(1)
        mn, mx = s_.min(1, keepdim=True)[0], s_.max(1, keepdim=True)[0]
        return ((s_ - mn) / (mx - mn + 1e-8)).view(B, out_size, out_size)

    def per_video(c):
        return c / (c.flatten(1).amax(1).view(B, 1, 1, 1) + 1e-8)

    cam, campp = up(cam), up(campp)
    temporal = cam.flatten(2).sum(-1).double()                       # scale-free: tiny (untrained) maps still sum to 1
    tsum = temporal.sum(1, keepdim=True)
    temporal = torch.where(tsum > 0, temporal / tsum.clamp_min(1e-300), torch.zeros_like(temporal)).float()
    out = dict(cam=per_video(cam).cpu().numpy(), campp=per_video(campp).cpu().numpy(), spatial=spatial(cam).cpu().numpy(),
               spatial_pp=spatial(campp).cpu().numpy(), temporal=temporal.cpu().numpy(), attn=o["attn"].detach().float().cpu().numpy(),
               logit=lg, sign=s)
    model.zero_grad(set_to_none=True)
    return out


def cam_metrics(a: np.ndarray, b: np.ndarray, top: float = 0.2) -> Tuple[float, float, float]:
    """Pearson r, top-20 % IoU and centre-of-mass shift (fraction of the side) of two [H,W] maps (Test 5 / 7 definition)."""
    r = float(np.corrcoef(a.ravel(), b.ravel())[0, 1]) if a.std() > 1e-8 and b.std() > 1e-8 else float("nan")
    ta, tb = a >= np.quantile(a, 1 - top), b >= np.quantile(b, 1 - top)
    iou = float((ta & tb).sum() / max(1, (ta | tb).sum()))
    yy, xx = np.mgrid[0:a.shape[0], 0:a.shape[1]]

    def com(c):
        w = c.sum() + 1e-8
        return np.array([(yy * c).sum() / w, (xx * c).sum() / w])

    return r, iou, float(np.linalg.norm(com(a) - com(b)) / a.shape[0])


def safe_corr(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.corrcoef(a, b)[0, 1]) if len(a) > 1 and a.std() > 1e-8 and b.std() > 1e-8 else float("nan")


def fit_temperature(logit, y) -> float:
    """Temperature T minimising the validation NLL of sigmoid(logit / T) (grid search, robust)."""
    x, y = np.asarray(logit, np.float64), np.asarray(y, np.float64)
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


def mean_bacc(y, pred, ds) -> float:
    """Balanced accuracy averaged over datasets (each dataset counts equally)."""
    y, pred, ds = np.asarray(y).astype(int), np.asarray(pred).astype(int), np.asarray(ds)
    v = []
    for d in np.unique(ds):
        m = ds == d
        r = (pred[m][y[m] == 1] == 1).mean() if (y[m] == 1).any() else np.nan
        s = (pred[m][y[m] == 0] == 0).mean() if (y[m] == 0).any() else np.nan
        if np.isfinite(r) or np.isfinite(s):
            v.append(np.nanmean([r, s]))
    return float(np.mean(v)) if v else float("nan")


def choose_threshold(y, p, ds) -> float:
    """Threshold maximising the MEAN balanced accuracy over the datasets present (151026 rule), ties -> closest to 0.5."""
    y, p, ds = np.asarray(y).astype(int), np.asarray(p, float), np.asarray(ds)
    lo, hi, st = THRESHOLD_GRID
    best, bt = -np.inf, 0.5
    for t in np.round(np.arange(lo, hi + st / 2, st), 4):
        v = mean_bacc(y, (p >= t).astype(int), ds)
        if not np.isfinite(v):
            continue
        v -= 1e-4 * abs(t - 0.5)
        if v > best:
            best, bt = v, float(t)
    return bt


class IsoCal:
    """Isotonic calibration of the logit (fitted on validation), stored as a piecewise-linear table."""

    def __init__(self, x=None, y=None):
        self.x = None if x is None else np.asarray(x, float)
        self.y = None if y is None else np.asarray(y, float)

    def fit(self, z, y):
        ir = IsotonicRegression(out_of_bounds="clip", y_min=1e-4, y_max=1 - 1e-4).fit(np.asarray(z, float), np.asarray(y, float))
        self.x, self.y = np.asarray(ir.X_thresholds_, float), np.asarray(ir.y_thresholds_, float)
        return self

    def __call__(self, z) -> np.ndarray:
        z = np.asarray(z, float)
        if self.x is None or len(self.x) == 0:
            return sigmoid(z)
        if len(self.x) == 1:
            return np.full(z.shape, float(self.y[0]))
        return np.interp(z, self.x, self.y)


def abstain_band(p_val, thr: float, coverage: float) -> float:
    """Half-width of the 'uncertain' band around the threshold that keeps `coverage` of the validation videos decided."""
    if coverage >= 1.0 or not len(p_val):
        return 0.0
    return float(np.quantile(np.abs(np.asarray(p_val, float) - thr), 1.0 - coverage))


# =====================================================================================
# METRICS + BOOTSTRAP HELPERS
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
    return float(roc_auc_score(y, np.asarray(p, float)))


class WeightedAUC:
    """AUC for many bootstrap weight vectors at once (tie-block formulation; == roc_auc_score for unit weights)."""

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


def p_boot(boot: np.ndarray) -> float:
    boot = np.asarray(boot, float)
    boot = boot[np.isfinite(boot)]
    if len(boot) <= 20:
        return float("nan")
    return float(min(1.0, 2 * min((boot <= 0).mean(), (boot >= 0).mean()) + 1 / len(boot)))


def holm(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    ok = np.isfinite(p)
    adj = np.full(len(p), np.nan)
    pv = p[ok]
    order = np.argsort(pv)
    m = len(pv)
    run_max, res = 0.0, np.empty(m)
    for k, i in enumerate(order):
        run_max = max(run_max, min(1.0, (m - k) * pv[i]))
        res[i] = run_max
    adj[ok] = res
    return adj


_trap = getattr(np, "trapezoid", None) or getattr(np, "trapz")


# =====================================================================================
# PRE-REGISTERED PROTOCOL + TEST 5 INPUT LINK (read-only re-use of the prepared split + caches)
# =====================================================================================
def protocol_dict(a) -> dict:
    """Every scientific setting of the final test. Hashed and written BEFORE any training; a later run with different
    settings in the same root is refused (seeds / training sets only decide which runs exist and are not hashed)."""
    return dict(
        test="Final test of the thesis (FINALTEST.py)",
        question="Does an ImageNet-pretrained, degradation-aware and explanation-consistent video detector with calibrated, "
                 "selective decisions give robust detection and stable, faithful explanations under real-world degradation?",
        data=dict(source="prepared Test 5 run (split, face-crop caches, verified degraded versions, Grad-CAM pairs), read-only"),
        model=dict(name="FinalDetector", backbone="EfficientNet-B0, torchvision IMAGENET1K_V1", temporal="1 transformer layer, 4 heads",
                   pooling="attention over frames", embedding=EMB_DIM, input_px=a.img, cache_px=CACHE_SIZE, frames_cached=NUM_FRAMES,
                   clip_len_training=CLIP_LEN, frames_evaluation=NUM_FRAMES, video_only=True,
                   gradcam_layer="last EfficientNet block (features[-1])"),
        stage1=dict(name=STAGE1, epochs=a.epochs1, patience=a.patience, lr_backbone=a.lr_backbone1, lr_head=a.lr_head1,
                    data="clean training originals"),
        stage2=dict(epochs=a.epochs2, patience=a.patience, lr_backbone=a.lr_backbone2, lr_head=a.lr_head2,
                    init="best F0 checkpoint (EMA weights) of the same training set and seed",
                    batch="each step: half originals + half degraded clips (F1) or original/degraded pairs (F2)"),
        common=dict(effective_batch_clips=a.batch_size * a.accum, weight_decay=WEIGHT_DECAY, warmup_frac=WARMUP_FRAC,
                    schedule="linear warm-up + cosine", label_smoothing=LABEL_SMOOTH, frame_loss_weight=FRAME_LOSS_W,
                    ema_decay=a.ema_decay, grad_clip=GRAD_CLIP, amp="fp16 autocast on CUDA",
                    augmentation=("151026 real-world operators (crop, flip, colour, JPEG, blur, down-scaling, erasing); identical "
                                  "parameters for both clips of a pair") if a.augment else "none",
                    sampler=dict(kind="adaptive group-DRO over dataset x folder cells (171026 harddro)", eta=DRO_ETA, floor=DRO_FLOOR,
                                 momentum=DRO_MOM)),
        variants={v: {k: x for k, x in variant_config(a, v).items()} for v in [STAGE1] + list(VARIANTS)},
        model_selection="best epoch by validation score (EMA weights): F0 = mean over datasets of validation original AUC; "
                        "F1/F2 = mean over datasets of mean(validation original AUC, validation degraded AUC)",
        calibration="temperature scaling (NLL grid) on validation originals + degraded versions of the training set's datasets; "
                    "threshold = maximum of the balanced accuracy averaged over datasets",
        final_model=dict(rule=f"Joint F2 unless its mean validation score over seeds is more than {SELECT_TOL} below Joint F1 (then F1); "
                              "within the variant the seed with the highest validation score (validation data only)",
                         calibration="isotonic regression of the logit on validation", decision="threshold = max mean balanced "
                         "accuracy on the isotonic probability", abstention=f"uncertain band tuned on validation for {DEPLOY_COVERAGE:.0%} coverage"),
        evaluation=dict(test="Test 5 test split of both datasets: originals + all verified degraded versions (in-domain, joint, cross-dataset)",
                        gradcam=dict(pairs="720 pre-registered Test 5 original/degraded pairs", frames="every 2nd cached frame (offset 0; "
                                     "clean reference = offset 1)", target="FAKE evidence", maps=["Grad-CAM", "Grad-CAM++"],
                                     change_rule=f"explanation changed = top-20 % IoU < {EXPLANATION_CHANGE_IOU}"),
                        faithfulness=dict(pairs=f"every {FAITH_STRIDE}th Grad-CAM pair, original and degraded version", maps=["gradcam", "gradcam++", "random"],
                                          perturbations=["grey", "blur"], fractions=FAITH_FRACS, target="predicted class")),
        statistics=dict(bootstrap="paired cluster bootstrap over source groups, seeds averaged", supported="95 % CI excludes 0 and all "
                        "seeds agree in sign", noninferiority=dict(auc=NONINF_AUC, iou=NONINF_XAI), multiplicity="Holm over PRIMARY",
                        primary=PRIMARY, comparisons=COMPARISONS, cross_comparisons=CROSS_COMPARISONS),
        hypotheses=HYPOTHESES, baselines=BASELINE_TEXT)


def ensure_protocol(a, P: Paths) -> str:
    proto = protocol_dict(a)
    sha = sha256_text(json.dumps(to_jsonable(proto), sort_keys=True))
    if P.protocol.exists():
        old = read_json(P.protocol, "final_protocol")
        if old.get("protocol_sha256") != sha:
            raise SystemExit(f"[STOP] these settings differ from the pre-registered protocol {P.protocol} (sha {str(old.get('protocol_sha256'))[:12]}"
                             f" vs {sha[:12]}). Run with the same settings, or use a new --root for a new experiment.")
        return sha
    write_json(P.protocol, dict(protocol=proto, protocol_sha256=sha, seeds_planned=a.seed_list, train_sets_planned=a.train_list,
                                variants_planned=[STAGE1] + a.variant_list, n_boot=a.n_boot, faith_scope=a.faith_scope,
                                registered_at=now(), script_sha256=code_sha256(),
                                note="written before any training; every checkpoint records this protocol hash"))
    print(f"  protocol registered: {P.protocol.name} (sha256 {sha[:16]}...)")
    return sha


def t5_cache_files(t5: T5Paths) -> dict:
    prov = read_json(t5.provenance, "t5_cache_provenance")
    need = {"faceclean", "facedeg"}
    if not need <= set(prov):
        raise SystemExit(f"[STOP] Test 5 cache provenance lacks {need - set(prov)}")
    c, d = prov["faceclean"], prov["facedeg"]
    return dict(clean=t5.cache / f"{c['fname']}.npy", clean_status=t5.cache / f"{c['fname']}.status.npy",
                clean_info=t5.cache / f"{c['fname']}.info.npz", n_clean=int(c["n"]),
                deg=t5.cache / f"{d['fname']}.npy", deg_status=t5.cache / f"{d['fname']}.status.npy", n_deg=int(d["n"]),
                provenance=prov, preprocessing_sha=c.get("preprocessing_code_sha256"))


def run_id(train_on: str, seed: int, variant: str) -> str:
    return f"{train_tag(train_on)}_seed{seed}_{variant}"


def train_tag(t: str) -> str:
    return "train-" + t.replace("+", "p")


def phase_link(a, P: Paths) -> dict:
    """Verify the prepared Test 5 inputs and register the protocol before anything is trained."""
    print("\n" + "=" * 100 + "\n PHASE LINK: verify the prepared Test 5 split + caches (read-only) and register the protocol\n" + "=" * 100)
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
    checks["Grad-CAM pair list SHA-256 matches the Test 5 gate"] = (xp_sha == gate.get("xai_pairs_sha256")) if gate.get("xai_pairs_sha256") else True
    cf = t5_cache_files(t5)
    for k in ("clean", "clean_status", "clean_info", "deg", "deg_status"):
        checks[f"cache file {k} exists"] = cf[k].exists()
    prov = cf["provenance"]
    checks["degraded cache was cropped with the boxes of this clean cache"] = prov["facedeg"].get("clean_cache") == prov["faceclean"].get("fname")
    checks["clean and degraded caches share one preprocessing code"] = (prov["facedeg"].get("preprocessing_code_sha256")
                                                                       == prov["faceclean"].get("preprocessing_code_sha256"))
    if all(checks.values()):
        cs = np.load(guard(cf["clean_status"], "t5_cache"))
        dss = np.load(guard(cf["deg_status"], "t5_cache"))
        arr = np.load(guard(cf["clean"], "t5_cache"), mmap_mode="r")
        checks["clean cache layout [N,16,128,128,3] uint8"] = (tuple(arr.shape[1:]) == (NUM_FRAMES, CACHE_SIZE, CACHE_SIZE, 3)
                                                                and arr.dtype == np.uint8)
        checks["clean cache complete (no pending rows)"] = bool((cs != 0).all()) and len(cs) == cf["n_clean"]
        checks["degraded cache complete (no pending rows)"] = bool((dss != 0).all()) and len(dss) == cf["n_deg"]
        man = read_csv(t5.split_manifest, "t5_split", low_memory=False)
        used = man[man.status == "used"]
        o, d = used[used.kind == "original"], used[used.kind == "degraded"]
        checks["every used original has an ok clean clip"] = bool((cs[o.clean_idx.astype(int).values] == 1).all())
        checks["every used degraded video has an ok clip"] = bool((dss[d.deg_idx.astype(int).values] == 1).all())
        checks["degraded videos are in their source's split"] = bool(
            (d.split.values == d.source_key.map(dict(zip(o.item_id, o.split))).values).all())
        checks["train / val / test contain real and fake videos of both datasets"] = all(
            ((o.dataset == ds) & (o.split == sp) & (o.label.astype(int) == lb)).any() for ds in DATASETS for sp in ("train", "val", "test")
            for lb in (0, 1))
        info = dict(originals=len(o), degraded=len(d), split_counts={f"{k[0]}|{k[1]}": int(v) for k, v in o.groupby(["dataset", "split"]).size().items()},
                    degraded_counts={f"{k[0]}|{k[1]}": int(v) for k, v in d.groupby(["dataset", "split"]).size().items()},
                    split_manifest_sha256=sm_sha, xai_pairs_sha256=xp_sha, split_hash_originals=gate.get("split_hash_originals"),
                    test5_script_sha256_at_gate=gate.get("script_sha256"), preprocessing_code_sha256=cf["preprocessing_sha"],
                    cache_files={k: str(v) for k, v in cf.items() if isinstance(v, Path)})
    base = {}
    for src, v, name in BASELINES:
        pdir, cdir = (a.t5.pred_dir, a.t5.cam_dir) if src == "T5" else (a.t7.pred_dir, a.t7.cam_dir)
        tsets = ["BioDeepAV", "FF++"] + ([] if src == "T5" else ["Joint"])
        base[name] = dict(predictions=int(sum((pdir / f"{run_id(t, s, v)}.csv.gz").exists() for t in tsets for s in a.seed_list)),
                          gradcam=int(sum((cdir / f"{run_id(t, s, v)}.csv").exists() for t in tsets for s in a.seed_list)),
                          description=BASELINE_TEXT[name])
    ok = all(checks.values())
    psha = ensure_protocol(a, P) if ok else None
    rep = dict(status="PASS" if ok else "FAIL", checked_at=now(), test5_root=str(t5.root), test7_root=str(a.t7.root), checks=checks,
               **info, baselines=base, protocol_sha256=psha,
               note="The final test re-uses the Test 5 split, caches and Grad-CAM pairs read-only; checkpoints of earlier tests are "
                    "blocked by guard(); Test 5 / Test 7 predictions and Grad-CAM rows are read as paired baselines only.")
    write_json(P.link, rep)
    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    for k, v in base.items():
        print(f"  baseline {k}: {v['predictions']} prediction files, {v['gradcam']} Grad-CAM files")
    print(f"  link status: {rep['status']} -> {P.link}")
    if not ok:
        raise SystemExit("[STOP] Test 5 inputs failed verification - nothing trained")
    write_experiment_configs(a, P, rep)
    return rep


def require_link(P: Paths) -> dict:
    if not P.link.exists():
        raise SystemExit("[STOP] run --phase link first")
    r = read_json(P.link, "final_link")
    if r.get("status") != "PASS":
        raise SystemExit("[STOP] the Test 5 input link did not PASS")
    return r


def write_experiment_configs(a, P: Paths, link: dict):
    m = FinalDetector(pretrained=False)
    for v in [STAGE1] + a.variant_list:
        write_json(P.configs / f"experiment_config_{v}.json", dict(
            test="Final test", variant=v, description=variant_config(a, v)["description"], created=now(),
            variant_config={k: x for k, x in variant_config(a, v).items() if k != "description"},
            model=dict(name="FinalDetector", parameters=n_params(m), architecture_signature=architecture_signature(m),
                       architecture_source_sha256=architecture_source_sha(), backbone="EfficientNet-B0 (ImageNet)", input_px=a.img),
            training=protocol_dict(a)["stage1" if v == STAGE1 else "stage2"], common=protocol_dict(a)["common"],
            protocol_sha256=link.get("protocol_sha256"), seeds=a.seed_list, train_sets=a.train_list,
            data={k: link.get(k) for k in ("split_manifest_sha256", "split_hash_originals", "xai_pairs_sha256", "preprocessing_code_sha256",
                                            "test5_root")},
            environment=environment_info()))


# =====================================================================================
# RUNTIME DATA (Test 5 split + caches, read-only)
# =====================================================================================
class FData:
    def __init__(self, a, P: Paths):
        t5 = a.t5
        man = read_csv(t5.split_manifest, "t5_split", low_memory=False)
        man = man[man.status == "used"].copy()
        man["label"] = man.label.astype(int)
        o = man[(man.kind == "original") & man.split.isin(["train", "val", "test"])].reset_index(drop=True)
        d = man[(man.kind == "degraded") & man.split.isin(["train", "val", "test"])].reset_index(drop=True)
        if a.smoke:                                  # tiny balanced subset per (dataset, split, label) + sources of some Grad-CAM pairs
            rng = np.random.default_rng(7)
            keep = set()
            for _, g in o.groupby(["dataset", "split", "label"]):
                keep |= set(int(i) for i in rng.choice(g.index.values, size=min(len(g), a.smoke_per_cell), replace=False))
            xp = read_csv(t5.xai_pairs, "t5_xai_pairs")
            xs = set(xp.iloc[::max(1, len(xp) // 24)].source_key)
            keep |= set(int(i) for i in o.index[o.item_id.isin(xs)])
            o = o.loc[sorted(keep)].reset_index(drop=True)
            d = d[d.source_key.isin(set(o.item_id))].reset_index(drop=True)
        self.o, self.d = o, d
        self.key = o.item_id.values.astype(str)
        self.labels = o.label.values.astype(np.int64)
        self.ds = o.dataset.values.astype(str)
        self.subset = o.subset.values.astype(str)
        self.fcell = np.array([f"{x}|{y}" for x, y in zip(self.ds, self.subset)])
        self.split = o.split.values.astype(str)
        self.group = o.group_id.values.astype(str)
        self.cidx = o.clean_idx.values.astype(np.int64)
        self.path = o.path.values.astype(str)
        pos = {k: i for i, k in enumerate(self.key)}
        self.deg_src = d.source_key.map(pos).values.astype(np.int64)
        self.deg_item = d.item_id.values.astype(str)
        self.deg_type = d.degradation.values.astype(str)
        self.deg_sev = d.severity.values.astype(str)
        self.didx = d.deg_idx.values.astype(np.int64)
        self.deg_path = d.path.values.astype(str)
        self.versions = defaultdict(list)
        for j, s in enumerate(self.deg_src):
            self.versions[int(s)].append(j)
        self.versions = {k: np.array(v, np.int64) for k, v in self.versions.items()}
        cf = t5_cache_files(t5)
        self.clean = np.load(guard(cf["clean"], "t5_cache"), mmap_mode="r")
        self.degc = np.load(guard(cf["deg"], "t5_cache"), mmap_mode="r")
        info = np.load(guard(cf["clean_info"], "t5_cache"))
        self.info_boxes = np.asarray(info["boxes"])                    # relative face boxes (degraded crops re-use them)
        self.info_meta = np.asarray(info["meta"])                      # width, height, fps, frames, fallback, face_rel_size, detected
        m = self.info_meta[self.cidx]
        self.native_face_px = (m[:, 5] * np.minimum(m[:, 0], m[:, 1])).astype(np.float64)
        self.preprocessing_sha = cf["preprocessing_sha"]
        print(f"  data: {len(o):,} originals, {len(d):,} degraded versions (Test 5 split, read-only caches)" + (" [SMOKE SUBSET]" if a.smoke else ""))

    def idx(self, split: str, dataset: Optional[str] = None) -> np.ndarray:
        m = self.split == split
        if dataset and dataset != "Joint":
            m &= self.ds == dataset
        return np.where(m)[0]

    def deg_items(self, src_idx: np.ndarray) -> np.ndarray:
        if not len(src_idx):
            return np.zeros(0, np.int64)
        parts = [self.versions[int(s)] for s in src_idx if int(s) in self.versions]
        return np.concatenate(parts) if parts else np.zeros(0, np.int64)

    def clip(self, kind: str, i: int, sel=None) -> np.ndarray:
        if kind == "deg":
            return np.asarray(self.degc[self.didx[i]] if sel is None else self.degc[self.didx[i], sel])
        return np.asarray(self.clean[self.cidx[i]] if sel is None else self.clean[self.cidx[i], sel])

    def src_of(self, kind: str, i: int) -> int:
        return int(self.deg_src[i]) if kind == "deg" else int(i)


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
    def __init__(self, a, P: Paths, D: FData, train_on: str, seed: int):
        self.a, self.P, self.D, self.train_on, self.seed = a, P, D, train_on, seed
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tr = D.idx("train", train_on)
        self.va = D.idx("val", train_on)
        self.tr_deg_src = np.array([s for s in self.tr if int(s) in D.versions], np.int64)
        self.n_tr_deg = int(sum(len(D.versions[int(s)]) for s in self.tr_deg_src))
        self.ckdir = P.checkpoints / train_tag(train_on) / f"seed{seed}"
        self.ckdir.mkdir(parents=True, exist_ok=True)
        link = require_link(P)
        self.split_hash = link.get("split_hash_originals")
        self.protocol_sha = link.get("protocol_sha256")
        self.init_info: dict = {}

    # ---------------------------------------------------------------- sampling (171026 harddro: group-DRO over dataset x folder cells)
    def cell_weights(self, src: np.ndarray, cell_loss: Dict[str, float]) -> Tuple[np.ndarray, Dict[str, float]]:
        cells = self.D.fcell[src]
        names = sorted(set(cells.tolist()))
        L = np.array([cell_loss.get(c, 1.0) for c in names], float)
        q = np.exp(DRO_ETA * (L / max(L.mean(), 1e-6) - 1.0))
        q = q / q.sum()
        p = DRO_FLOOR / len(names) + (1 - DRO_FLOOR) * q
        share = dict(zip(names, p / p.sum()))
        cnt = Counter(cells.tolist())
        w = np.array([share[c] / cnt[c] for c in cells], float)
        return w / w.sum(), share

    def pick_version(self, s: int, rng) -> int:
        v = self.D.versions[int(s)]
        return int(v[int(rng.integers(len(v)))])

    def plan_epoch(self, rng, stage: int, vc: dict, steps: int, bs: int, cell_loss: Dict[str, float]):
        D, plan, log = self.D, [], Counter()
        if stage == 1:
            w, share = self.cell_weights(self.tr, cell_loss)
            draws = rng.choice(self.tr, size=steps * bs, replace=True, p=w)
            for st in range(steps):
                its = [("clean", int(s)) for s in draws[st * bs:(st + 1) * bs]]
                plan.append(("single", its))
                log.update((D.fcell[s], "original") for _, s in its)
            return plan, log, share
        wd, share = self.cell_weights(self.tr_deg_src, cell_loss)
        half = bs // 2
        if vc["paired"]:
            draws = rng.choice(self.tr_deg_src, size=steps * half, replace=True, p=wd)
            for st in range(steps):
                its = []
                for s in draws[st * half:(st + 1) * half]:
                    j = self.pick_version(int(s), rng)
                    its.append((int(s), j))
                    log[(D.fcell[s], D.deg_type[j])] += 1
                plan.append(("pair", its))
            return plan, log, share
        wc, _ = self.cell_weights(self.tr, cell_loss)
        dc = rng.choice(self.tr, size=steps * half, replace=True, p=wc)
        dd = rng.choice(self.tr_deg_src, size=steps * (bs - half), replace=True, p=wd)
        for st in range(steps):
            its = [("clean", int(s)) for s in dc[st * half:(st + 1) * half]]
            log.update((D.fcell[s], "original") for _, s in its)
            for s in dd[st * (bs - half):(st + 1) * (bs - half)]:
                j = self.pick_version(int(s), rng)
                its.append(("deg", j))
                log[(D.fcell[s], D.deg_type[j])] += 1
            plan.append(("single", its))
        return plan, log, share

    def build(self, mode: str, items: list, seed: int) -> dict:
        """Load + augment one step. single: [(kind, idx)]; pair: [(src, deg_j)] with the same frames and augmentation."""
        rng = np.random.default_rng(seed)
        D, S, L = self.D, CACHE_SIZE, CLIP_LEN
        B = len(items)
        a_ = np.empty((B, L, S, S, 3), np.uint8)
        b_ = np.empty((B, L, S, S, 3), np.uint8) if mode == "pair" else None
        tix = np.empty((B, L), np.int64)
        src = np.empty(B, np.int64)
        dtype = []
        for k, it in enumerate(items):
            sel = np.sort(rng.choice(NUM_FRAMES, L, replace=False))
            prm = aug_params(rng, S) if self.a.augment else None
            if mode == "pair":
                s, j = it
                a_[k] = apply_aug(D.clip("clean", s, sel), prm)
                b_[k] = apply_aug(D.clip("deg", j, sel), prm)
                src[k] = s
                dtype.append(str(D.deg_type[j]))
            else:
                kind, i = it
                a_[k] = apply_aug(D.clip(kind, i, sel), prm)
                src[k] = D.src_of(kind, i)
                dtype.append(str(D.deg_type[i]) if kind == "deg" else "original")
            tix[k] = sel
        return dict(mode=mode, a=a_, b=b_, tix=tix, src=src, y=D.labels[src].astype(np.float32), dtype=np.array(dtype))

    # ---------------------------------------------------------------- prediction / validation / calibration
    @torch.no_grad()
    def predict_logits(self, model: FinalDetector, items: List[Tuple[str, int]], bs: Optional[int] = None) -> np.ndarray:
        """Video logit with all 16 cached frames (frame positions 0..15)."""
        if not len(items):
            return np.zeros(0)
        model.eval()
        bs = bs or self.a.eval_batch
        chunks = [items[i:i + bs] for i in range(0, len(items), bs)]
        out = []
        for ch, clips in prefetch(lambda c: np.stack([self.D.clip(k, j) for k, j in c]), chunks, 2):
            x = to_input(clips, self.device, self.a.img)
            with torch.autocast("cuda", dtype=torch.float16, enabled=self.a.amp):
                lg = model(x, frame_positions(len(ch), np.arange(NUM_FRAMES), self.device))["logit"]
            out.append(lg.float().cpu().numpy())
        return np.concatenate(out)

    def validate(self, model: FinalDetector, stage: int) -> dict:
        D = self.D
        zc = self.predict_logits(model, [("clean", int(s)) for s in self.va])
        jd = D.deg_items(self.va) if stage == 2 else np.zeros(0, np.int64)
        zd = self.predict_logits(model, [("deg", int(j)) for j in jd])
        dsrc = D.deg_src[jd]
        row, scores, cl, dg = {}, [], [], []
        for ds in sorted(set(D.ds[self.va])):
            m = D.ds[self.va] == ds
            ca = safe_auc(D.labels[self.va][m], zc[m])
            row[f"val_clean_auc_{ds}"] = ca
            cl.append(ca)
            parts = [ca]
            if stage == 2:
                md = D.ds[dsrc] == ds
                if md.any():
                    da = safe_auc(D.labels[dsrc][md], zd[md])
                    row[f"val_degraded_auc_{ds}"] = da
                    dg.append(da)
                    parts.append(da)
            scores.append(float(np.nanmean(parts)) if np.isfinite(parts).any() else float("nan"))
        row["val_clean_auc"] = float(np.nanmean(cl)) if np.isfinite(cl).any() else float("nan")
        if dg:
            row["val_degraded_auc"] = float(np.nanmean(dg)) if np.isfinite(dg).any() else float("nan")
        row["score"] = float(np.nanmean(scores)) if np.isfinite(scores).any() else float("nan")
        return row

    def calibrate(self, model: FinalDetector) -> dict:
        """Temperature + threshold on validation originals AND their degraded versions (same rule for F0, F1, F2)."""
        D = self.D
        jd = D.deg_items(self.va)
        items = [("clean", int(s)) for s in self.va] + [("deg", int(j)) for j in jd]
        src = np.array([D.src_of(k, i) for k, i in items], np.int64)
        y, ds = D.labels[src], D.ds[src]
        z = self.predict_logits(model, items)
        T = fit_temperature(z, y)
        p_raw, p = sigmoid(z), sigmoid(z / T)
        thr = choose_threshold(y, p, ds)
        iso = IsoCal().fit(z, y)
        return dict(temperature=T, threshold=thr, val_ece_raw=ece_score(y, p_raw), val_ece=ece_score(y, p), val_ece_isotonic=ece_score(y, iso(z)),
                    val_mean_bacc=mean_bacc(y, (p >= thr).astype(int), ds), n_val=int(len(y)),
                    selected_on=f"validation originals + degraded versions ({self.train_on})")

    # ---------------------------------------------------------------- training loop
    def train(self, stage: int, variant: str, model: FinalDetector, ckpt: Path) -> dict:
        a, D = self.a, self.D
        vc = variant_config(a, variant)
        name = run_id(self.train_on, self.seed, variant)
        bs = a.batch_size
        per_step = bs if stage == 1 else bs // 2                     # source videos per micro-step
        steps = int(math.ceil(len(self.tr) / per_step / a.accum)) * a.accum
        if a.max_steps:
            steps = min(steps, a.max_steps * a.accum)
        epochs = a.epochs1 if stage == 1 else a.epochs2
        lr_b, lr_h = (a.lr_backbone1, a.lr_head1) if stage == 1 else (a.lr_backbone2, a.lr_head2)
        print("\n" + "=" * 100 + f"\n TRAIN {name} | {vc['description']}\n"
              f"  train originals {len(self.tr):,} (with degraded versions {len(self.tr_deg_src):,}; versions {self.n_tr_deg:,}) | "
              f"val {len(self.va):,} | {steps} steps x {epochs} epochs (patience {a.patience}) | {bs} clips/step x accum {a.accum} | "
              f"lr {lr_b:g}/{lr_h:g} | img {a.img} | amp {a.amp} | augment {a.augment}\n" + "=" * 100)
        if stage == 2 and not len(self.tr_deg_src):
            raise SystemExit(f"[STOP] no degraded training videos for {self.train_on}")
        bb = list(model.backbone.parameters())
        other = [p for n, p in model.named_parameters() if not n.startswith("backbone.")]
        opt = torch.optim.AdamW([dict(params=bb, lr=lr_b), dict(params=other, lr=lr_h)], weight_decay=WEIGHT_DECAY)
        n_updates = max(1, steps // a.accum * epochs)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(n_updates))
        scaler = torch.amp.GradScaler("cuda", enabled=a.amp)
        ema = EMA(model, a.ema_decay)
        rng = np.random.default_rng([self.seed, stage, sum(map(ord, variant)), sum(map(ord, self.train_on))])
        history, best, samp_rows = [], dict(score=-1.0, epoch=-1), []
        cell_loss: Dict[str, float] = {}
        bad = 0
        t_start, overall, updates = time.time(), 0, 0
        resume_path = ckpt.with_suffix(".resume.pt")
        run_sig = dict(run=name, stage=stage, variant=variant, split_hash=self.split_hash, protocol=self.protocol_sha, steps=steps,
                       epochs=epochs, bs=bs, accum=a.accum, lr=[lr_b, lr_h], img=a.img, augment=a.augment,
                       vc={k: v for k, v in vc.items() if k != "description"}, arch=architecture_signature(model))
        start_ep, start_st, ep_state, rs = 1, 0, None, None
        if resume_path.exists():
            try:
                rs = torch.load(guard(resume_path, "final_checkpoint"), map_location=self.device, weights_only=False)
            except Exception as e:
                print(f"  [resume] {resume_path.name} unreadable ({type(e).__name__}) - starting this run from its beginning")
            if rs is not None and rs.get("sig") != run_sig:
                print(f"  [resume] {resume_path.name} belongs to another configuration - starting this run from its beginning")
                rs = None
        if rs is not None:
            model.load_state_dict(rs["model"])
            opt.load_state_dict(rs["opt"])
            sched.load_state_dict(rs["sched"])
            scaler.load_state_dict(rs["scaler"])
            ema.model.load_state_dict(rs["ema"])
            rng.bit_generator.state = rs["rng_epoch_start"]
            torch.set_rng_state(rs["torch_rng"].cpu())
            if torch.cuda.is_available() and rs.get("cuda_rng") is not None:
                torch.cuda.set_rng_state_all([s.cpu() for s in rs["cuda_rng"]])
            history, best, samp_rows, overall, updates, bad = (rs["history"], rs["best"], rs["samp_rows"], rs["overall"], rs["updates"],
                                                               rs["bad"])
            cell_loss = dict(rs["cell_loss_epoch_start"])
            start_ep, start_st, ep_state = rs["epoch"], rs["step"], rs.get("ep_state")
            self.init_info = rs.get("init_info", self.init_info)
            t_start = time.time() - rs["elapsed"]
            print(f"  [resume] {name}: epoch {start_ep}/{epochs}, step {start_st}/{steps}")
        else:
            ckpt.with_suffix(".best.tmp").unlink(missing_ok=True)

        def save_resume(ep_next, st_next, rng_state, ep_st, cell_start):
            state = dict(sig=run_sig, model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), scaler=scaler.state_dict(),
                         ema=ema.model.state_dict(), rng_epoch_start=rng_state, torch_rng=torch.get_rng_state(),
                         cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None, history=history, best=best,
                         samp_rows=samp_rows, overall=overall, updates=updates, bad=bad, cell_loss_epoch_start=cell_start, epoch=ep_next,
                         step=st_next, ep_state=ep_st, init_info=self.init_info, elapsed=time.time() - t_start, saved_at=now())
            try:
                atomic_torch_save(state, resume_path)
            except (PermissionError, OSError) as e:      # a skipped resume save only costs the steps since the previous one
                print(f"\n  [resume] could not save {resume_path.name} ({e}) - training continues")

        bce = nn.BCEWithLogitsLoss()
        loader = ThreadPoolExecutor(max_workers=max(1, a.loader_threads))
        try:
            for ep in range(start_ep, epochs + 1):
                if bad >= a.patience:
                    print(f"  early stop: no validation improvement for {a.patience} epochs")
                    break
                model.train()
                rng_epoch_start = rng.bit_generator.state
                cell_start = dict(cell_loss)
                plan, slog, share = self.plan_epoch(rng, stage, vc, steps, bs, cell_loss)
                mid = ep == start_ep and start_st > 0
                if not mid:
                    for (cell, typ), n in slog.items():
                        samp_rows.append(dict(run=name, epoch=ep, cell=cell, degradation=typ, n=int(n), cell_share=float(share.get(cell, np.nan))))
                seeds = rng.integers(0, 2 ** 31, size=steps)
                first = start_st if mid else 0
                q, nxt = deque(), first

                def submit():
                    nonlocal nxt
                    if nxt < steps:
                        q.append(loader.submit(self.build, plan[nxt][0], plan[nxt][1], int(seeds[nxt])))
                        nxt += 1

                for _ in range(max(2, 2 * a.loader_threads)):
                    submit()
                tot, comp, t0, last = 0.0, Counter(), time.time(), 0.0
                lsum, lcnt = Counter(), Counter()
                if mid and ep_state:
                    tot, comp, t0 = ep_state["tot"], Counter(ep_state["comp"]), time.time() - ep_state["epoch_elapsed"]
                    lsum, lcnt = Counter(ep_state["lsum"]), Counter(ep_state["lcnt"])
                opt.zero_grad(set_to_none=True)
                for st in range(first, steps):
                    bt = q.popleft().result()
                    submit()
                    overall += 1
                    paired = bt["mode"] == "pair"
                    y_np = bt["y"]
                    if paired:
                        nb = len(y_np)
                        x = to_input(np.concatenate([bt["a"], bt["b"]]), self.device, a.img)
                        tix = torch.from_numpy(np.concatenate([bt["tix"], bt["tix"]])).to(self.device)
                        y = torch.from_numpy(np.concatenate([y_np, y_np])).to(self.device)
                        src_all = np.concatenate([bt["src"], bt["src"]])
                    else:
                        nb = len(y_np)
                        x = to_input(bt["a"], self.device, a.img)
                        tix = torch.from_numpy(bt["tix"]).to(self.device)
                        y = torch.from_numpy(y_np).to(self.device)
                        src_all = bt["src"]
                    ys = y * (1 - LABEL_SMOOTH) + 0.5 * LABEL_SMOOTH
                    with math_sdp():
                        with torch.autocast("cuda", dtype=torch.float16, enabled=a.amp):
                            o = model(x, tix)
                            l_main = bce(o["logit"], ys)
                            l_frame = bce(o["frame_logit"], ys[:, None].expand_as(o["frame_logit"]))
                            loss = l_main + FRAME_LOSS_W * l_frame
                            comp["bce"] += float(l_main.item())
                            if paired and vc["lambda_supcon"] > 0:
                                l_sc = supcon_loss(model.con_head(o["z"]), y.long())
                                loss = loss + vc["lambda_supcon"] * l_sc
                                comp["supcon"] += float(l_sc.item())
                        if paired and vc["lambda_cam"] > 0:
                            cam = soft_gradcam(o["logit"], o["act"], 2 * nb, CLIP_LEN)
                            l_cam = cam_consistency_loss(cam[:nb].detach(), cam[nb:])
                            loss = loss + vc["lambda_cam"] * l_cam
                            comp["cam"] += float(l_cam.item())
                    with torch.no_grad():
                        per = F.binary_cross_entropy_with_logits(o["logit"].detach().float(), ys, reduction="none").cpu().numpy()
                    for c, l_ in zip(D.fcell[src_all], per):
                        lsum[c] += float(l_)
                        lcnt[c] += 1
                    if not torch.isfinite(loss):
                        comp["skipped_nonfinite"] += 1
                        opt.zero_grad(set_to_none=True)
                        print(f"\n  [warn] non-finite loss at epoch {ep} step {st} - step skipped")
                        del o, loss
                        continue
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
                        ema.update(model)
                    tot += float(loss.item())
                    del o, loss
                    if (st + 1) % max(1, a.save_every_steps) == 0 and st + 1 < steps and (st + 1) % a.accum == 0:
                        save_resume(ep, st + 1, rng_epoch_start, dict(tot=tot, comp=dict(comp), epoch_elapsed=time.time() - t0,
                                                                      lsum=dict(lsum), lcnt=dict(lcnt)), cell_start)
                    if time.time() - last > 0.5 or st + 1 == steps:
                        gt = gpu_temp() if st % 100 == 0 else None
                        if gt is not None and gt >= a.gpu_max_temp:
                            print(f"\n  [thermal] GPU {gt} C - pausing {a.gpu_pause} s")
                            torch.cuda.empty_cache()
                            time.sleep(a.gpu_pause)
                        el = time.time() - t_start
                        eta = el / max(1, overall) * (steps * epochs - overall)
                        sys.stdout.write(f"\r  [{name}] ep {ep}/{epochs} {progress_bar(st + 1, steps, 16)} loss {tot / (st + 1):.4f} ("
                                         + " ".join(f"{k} {v / (st + 1):.3f}" for k, v in comp.items()) + f") | ETA {fmt_time(eta)}   ")
                        sys.stdout.flush()
                        last = time.time()
                for c in lcnt:                                          # group-DRO cell losses (momentum)
                    cell_loss[c] = DRO_MOM * cell_loss.get(c, 1.0) + (1 - DRO_MOM) * lsum[c] / lcnt[c]
                row = dict(run=name, epoch=ep, train_loss=tot / steps, lr_backbone=float(sched.get_last_lr()[0]),
                           **{f"train_{k}": v / steps for k, v in comp.items()}, epoch_time_s=time.time() - t0,
                           **{f"share_{c}": float(v) for c, v in share.items()})
                row.update(self.validate(ema.model, stage))
                history.append(row)
                better = np.isfinite(row["score"]) and row["score"] > best["score"]
                if better:
                    best, bad = dict(score=row["score"], epoch=ep), 0
                    atomic_torch_save(dict(model_state_dict=ema.model.state_dict(), epoch=ep, val=row), ckpt.with_suffix(".best.tmp"))
                else:
                    bad += 1
                print(f"\r  [{name}] ep {ep}/{epochs} | loss {row['train_loss']:.4f} | val original AUC {row['val_clean_auc']:.4f}"
                      + (f" | val degraded AUC {row['val_degraded_auc']:.4f}" if "val_degraded_auc" in row else "")
                      + f" | score {row['score']:.4f}{'  <- best' if better else ''} | {fmt_time(row['epoch_time_s'])}" + " " * 20)
                save_resume(ep + 1, 0, rng.bit_generator.state, None, dict(cell_loss))
        finally:
            loader.shutdown(wait=True)
        write_csv(pd.DataFrame(history), self.P.hist_dir / f"history_{name}.csv")
        write_csv(pd.DataFrame(samp_rows), self.P.hist_dir / f"sampling_{name}.csv")
        tmp = ckpt.with_suffix(".best.tmp")
        if not tmp.exists():
            raise RuntimeError(f"{name}: no valid validation score - nothing saved")
        model.load_state_dict(torch.load(guard(tmp, "final_checkpoint"), map_location=self.device, weights_only=False)["model_state_dict"])
        cal = self.calibrate(model)
        cfg = dict(run=name, stage=stage, variant=variant, variant_config={k: v for k, v in vc.items() if k != "description"},
                   train_on=self.train_on, seed=self.seed, epochs=epochs, epochs_run=len(history), patience=a.patience, lr_backbone=lr_b,
                   lr_head=lr_h, batch_clips=bs, accum=a.accum, steps_per_epoch=steps, amp=a.amp, img=a.img, augment=a.augment,
                   split_hash=self.split_hash, protocol_sha256=self.protocol_sha, init=self.init_info,
                   architecture_signature=architecture_signature(model), architecture_source_sha256=architecture_source_sha(),
                   parameters=n_params(model), script_sha256=code_sha256())
        cfg_sha = sha256_text(json.dumps(to_jsonable(cfg), sort_keys=True))
        sd = model.state_dict()
        atomic_torch_save(dict(model_state_dict=sd, threshold=cal["threshold"], temperature=cal["temperature"], calibration=cal,
                               best_epoch=best["epoch"], best_val_score=best["score"], history=history, config=cfg, config_sha256=cfg_sha,
                               weights_sha256=weights_sha256(sd), final=True, saved_at=now()), ckpt)
        tmp.unlink(missing_ok=True)
        resume_path.unlink(missing_ok=True)
        rec = dict(run_id=name, train_on=self.train_on, variant=variant, seed=self.seed, stage=stage, path=str(ckpt), file_sha256=sha256_file(ckpt),
                   weights_sha256=weights_sha256(sd), script_sha256=code_sha256(), config_sha256=cfg_sha, protocol_sha256=self.protocol_sha,
                   split_hash=self.split_hash, threshold=cal["threshold"], temperature=cal["temperature"], val_ece_raw=cal["val_ece_raw"],
                   val_ece=cal["val_ece"], val_ece_isotonic=cal["val_ece_isotonic"], best_epoch=best["epoch"], best_val_score=best["score"],
                   epochs_run=len(history), batch_clips=bs, accum=a.accum, init=str(self.init_info.get("kind", "")),
                   train_time_s=time.time() - t_start, saved_at=now())
        h = read_csv(self.P.ck_index, "final_checkpoint_index") if self.P.ck_index.exists() else pd.DataFrame()
        h = pd.concat([h[h.run_id != name] if len(h) else h, pd.DataFrame([rec])], ignore_index=True)
        write_csv(h, self.P.ck_index)
        print(f"  saved {ckpt.name} | best epoch {best['epoch']} | val score {best['score']:.4f} | T {cal['temperature']:.2f} | "
              f"threshold {cal['threshold']:.2f} | val ECE {cal['val_ece_raw']:.3f} -> {cal['val_ece']:.3f} (isotonic {cal['val_ece_isotonic']:.3f})")
        return rec

    def done(self, ckpt: Path) -> bool:
        if not ckpt.exists():
            return False
        try:
            obj = torch.load(guard(ckpt, "final_checkpoint"), map_location="cpu", weights_only=False)
        except (RuntimeError, EOFError, OSError, ValueError) as e:
            print(f"  [resume] {ckpt.name} unreadable ({type(e).__name__}) - redo")
            return False
        cfg = obj.get("config", {})
        return bool(obj.get("final")) and cfg.get("split_hash") == self.split_hash and cfg.get("protocol_sha256") == self.protocol_sha

    def load(self, ckpt: Path):
        obj = torch.load(guard(ckpt, "final_checkpoint"), map_location=self.device, weights_only=False)
        m = FinalDetector(pretrained=False).to(self.device)
        m.load_state_dict(obj["model_state_dict"])
        m.eval()
        return m, obj

    def stage1(self) -> Path:
        ck = self.ckdir / f"{STAGE1}.pt"
        if self.done(ck):
            print(f"  [resume] {ck} done")
            return ck
        set_seed(self.seed, self.a.deterministic)
        model = FinalDetector(pretrained=True, allow_random=self.a.allow_random_init).to(self.device)
        self.init_info = dict(kind="ImageNet (torchvision EfficientNet_B0_Weights.IMAGENET1K_V1)" if model.pretrained else "random (pipeline check)",
                              pretrained=bool(model.pretrained), backbone_sha256=weights_sha256(model.backbone.state_dict()))
        print(f"  FinalDetector | {n_params(model):,} parameters | backbone {self.init_info['kind']} | head init seed {self.seed}")
        self.train(1, STAGE1, model, ck)
        return ck

    def stage2(self, variant: str, s1: Path) -> Path:
        ck = self.ckdir / f"{variant}.pt"
        if self.done(ck):
            print(f"  [resume] {ck} done")
            return ck
        set_seed(self.seed * 1000 + sum(map(ord, variant)), self.a.deterministic)
        model, _ = self.load(s1)
        self.init_info = dict(kind="F0 checkpoint (EMA weights at the best validation epoch)", path=str(s1), file_sha256=sha256_file(s1))
        self.train(2, variant, model, ck)
        return ck


def with_oom_retry(a, fn, *args):
    """On CUDA out-of-memory: halve the clips per step and double gradient accumulation (same effective batch, same
    protocol), halve the evaluation batch as well (validation runs inside training), and retry."""
    while True:
        try:
            return fn(*args)
        except torch.cuda.OutOfMemoryError:
            gc.collect()
            torch.cuda.empty_cache()
            if a.batch_size <= 4 and a.eval_batch <= 4:
                raise
            if a.batch_size > 4:
                a.batch_size //= 2
                a.accum *= 2
            a.eval_batch = max(4, a.eval_batch // 2)
            print(f"\n  [OOM] retrying with --batch-size {a.batch_size} --accum {a.accum} --eval-batch {a.eval_batch} (same effective "
                  f"batch); pass these flags next time to avoid the retry")


def phase_train(a, P: Paths, D: FData, seeds: Optional[List[int]] = None):
    require_link(P)
    status = []
    for seed in (seeds or a.seed_list):
        for t in a.train_list:
            tr = Trainer(a, P, D, t, seed)
            try:
                s1 = with_oom_retry(a, tr.stage1)
                status.append(dict(run_id=run_id(t, seed, STAGE1), status="completed"))
            except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
                status.append(dict(run_id=run_id(t, seed, STAGE1), status=f"failed: {type(e).__name__}: {str(e)[:200]}"))
                print(f"  [FAILED] {run_id(t, seed, STAGE1)}: {e}")
                gc.collect()
                torch.cuda.empty_cache()
                continue
            for v in a.variant_list:
                try:
                    with_oom_retry(a, tr.stage2, v, s1)
                    status.append(dict(run_id=run_id(t, seed, v), status="completed"))
                except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
                    status.append(dict(run_id=run_id(t, seed, v), status=f"failed: {type(e).__name__}: {str(e)[:200]}"))
                    print(f"  [FAILED] {run_id(t, seed, v)}: {e}")
                    gc.collect()
                    torch.cuda.empty_cache()
    p = P.logs / "run_status.csv"
    old = read_csv(p, "final_log") if p.exists() else pd.DataFrame()
    st = pd.concat([old, pd.DataFrame(status)]).drop_duplicates("run_id", keep="last") if len(old) else pd.DataFrame(status)
    write_csv(st, p)


# =====================================================================================
# EVALUATION, GRAD-CAM STABILITY, FAITHFULNESS UNDER DEGRADATION
# =====================================================================================
PRED_COLS = ["run_id", "train_on", "variant", "seed", "split", "item_id", "kind", "source_key", "group_id", "dataset", "subset", "label",
             "degradation", "severity", "condition", "p_fake", "p_fake_raw", "logit", "threshold", "temperature", "native_face_px"]


def available_runs(a, P: Paths, scope: str = "all") -> List[dict]:
    out = []
    for t in a.train_list:
        if scope == "joint" and t != "Joint":
            continue
        for seed in a.seed_list:
            d = P.checkpoints / train_tag(t) / f"seed{seed}"
            for v in [STAGE1] + a.variant_list:
                ck = d / f"{v}.pt"
                if ck.exists():
                    out.append(dict(run_id=run_id(t, seed, v), train_on=t, seed=seed, variant=v, ckpt=ck))
    return out


def phase_evaluate(a, P: Paths, D: FData):
    """Validation + test predictions (originals and every degraded version) of every finished run. Up to --eval-models models
    share one pass over the cached clips, so every batch is read from disk and up-sampled once."""
    require_link(P)
    todo = [r for r in available_runs(a, P) if not (P.pred_dir / f"{r['run_id']}.csv.gz").exists()]
    if not todo:
        return
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    items = []
    for split in ("val", "test"):
        o = D.idx(split)
        items += [(split, "clean", int(s)) for s in o] + [(split, "deg", int(j)) for j in D.deg_items(o)]
    step = max(1, a.eval_models)
    for g0 in range(0, len(todo), step):
        group = todo[g0:g0 + step]
        models = []
        for r in group:
            obj = torch.load(guard(r["ckpt"], "final_checkpoint"), map_location=dev, weights_only=False)
            m = FinalDetector(pretrained=False).to(dev)
            m.load_state_dict(obj["model_state_dict"])
            m.eval()
            models.append((r, m, float(obj["threshold"]), float(obj["temperature"])))
            del obj
        print(f"  evaluating {len(group)} run(s) on {len(items):,} validation + test clips: " + ", ".join(r["run_id"] for r in group))
        t0 = time.time()
        while True:
            chunks = [items[i:i + a.eval_batch] for i in range(0, len(items), a.eval_batch)]
            logits = {r["run_id"]: [] for r in group}
            n_done, last = 0, 0.0
            try:
                for ch, clips in prefetch(lambda c: np.stack([D.clip(k, i) for _, k, i in c]), chunks, 2):
                    x = to_input(clips, dev, a.img)
                    tp = frame_positions(len(ch), np.arange(NUM_FRAMES), dev)
                    with torch.no_grad():
                        for r, m, _, _ in models:
                            with torch.autocast("cuda", dtype=torch.float16, enabled=a.amp):
                                logits[r["run_id"]].append(m(x, tp)["logit"].float().cpu().numpy())
                    n_done += len(ch)
                    if time.time() - last > 0.5 or n_done == len(items):
                        el = time.time() - t0
                        sys.stdout.write(f"\r    {progress_bar(n_done, len(items), 16)} {n_done:,}/{len(items):,} | "
                                         f"ETA {fmt_time(el / n_done * (len(items) - n_done))}   ")
                        sys.stdout.flush()
                        last = time.time()
                break
            except torch.cuda.OutOfMemoryError:
                gc.collect()
                torch.cuda.empty_cache()
                if a.eval_batch <= 2:
                    raise
                a.eval_batch = max(2, a.eval_batch // 2)
                print(f"\n  [OOM] evaluation restarted with --eval-batch {a.eval_batch}")
        sys.stdout.write("\n")
        for r, m, thr, T in models:
            z = np.concatenate(logits[r["run_id"]])
            rows = []
            for (split, k, i), zz in zip(items, z):
                s = D.src_of(k, i)
                rows.append((r["run_id"], r["train_on"], r["variant"], r["seed"], split, D.deg_item[i] if k == "deg" else D.key[i], k, D.key[s],
                             D.group[s], D.ds[s], D.subset[s], int(D.labels[s]), D.deg_type[i] if k == "deg" else "original",
                             D.deg_sev[i] if k == "deg" else "", "normal", float(sigmoid(zz / T)), float(sigmoid(zz)), float(zz), thr, T,
                             float(D.native_face_px[s])))
            write_csv(pd.DataFrame(rows, columns=PRED_COLS), P.pred_dir / f"{r['run_id']}.csv.gz", compression="gzip")
        print(f"    done in {fmt_time(time.time() - t0)}")
        del models, logits
        gc.collect()
        torch.cuda.empty_cache()


def load_xai_pairs(a, D: FData) -> list:
    xp = read_csv(a.t5.xai_pairs, "t5_xai_pairs")
    dpos = {k: i for i, k in enumerate(D.deg_item)}
    spos = {k: i for i, k in enumerate(D.key)}
    return [(spos[r.source_key], dpos[r.deg_item_id], r) for r in xp.itertuples() if r.deg_item_id in dpos and r.source_key in spos]


EVEN_FRAMES = np.arange(0, NUM_FRAMES, 2)[:CLIP_LEN]       # frames of the Test 5 / Test 7 Grad-CAM protocol (offset 0)
ODD_FRAMES = np.arange(1, NUM_FRAMES, 2)[:CLIP_LEN]        # clean reference view (offset 1)


def phase_xai(a, P: Paths, D: FData):
    """Grad-CAM / Grad-CAM++ stability between each original and its degraded version on the 720 pre-registered Test 5 pairs."""
    require_link(P)
    pairs = load_xai_pairs(a, D)
    if not pairs:
        print("  no Grad-CAM pairs in this data subset")
        return
    chunks = [pairs[k:k + XAI_CHUNK] for k in range(0, len(pairs), XAI_CHUNK)]

    def load(ch):
        return (np.stack([D.clip("clean", s, EVEN_FRAMES) for s, _, _ in ch]), np.stack([D.clip("deg", j, EVEN_FRAMES) for _, j, _ in ch]),
                np.stack([D.clip("clean", s, ODD_FRAMES) for s, _, _ in ch]))

    for r in available_runs(a, P):
        out = P.cam_dir / f"{r['run_id']}.csv"
        if out.exists():
            continue
        tr = Trainer(a, P, D, r["train_on"], r["seed"])
        model, obj = tr.load(r["ckpt"])
        thr, T = float(obj["threshold"]), float(obj["temperature"])
        dev = tr.device
        rows, t0, n_done = [], time.time(), 0
        print(f"  Grad-CAM {r['run_id']} on {len(pairs)} Test 5 pairs ...")
        for ch, (cc, dc, cr) in prefetch(load, chunks, 2):
            n = len(ch)
            g_o = grad_cam_eval(model, to_input(cc, dev, a.img), frame_positions(n, EVEN_FRAMES, dev), 1.0, amp=a.amp)
            g_d = grad_cam_eval(model, to_input(dc, dev, a.img), frame_positions(n, EVEN_FRAMES, dev), 1.0, amp=a.amp)
            g_r = grad_cam_eval(model, to_input(cr, dev, a.img), frame_positions(n, ODD_FRAMES, dev), 1.0, amp=a.amp)
            for b, (s, j, x) in enumerate(ch):
                pr, iou, sh = cam_metrics(g_o["spatial"][b], g_d["spatial"][b])
                prp, ioup, _ = cam_metrics(g_o["spatial_pp"][b], g_d["spatial_pp"][b])
                _, ref_iou, _ = cam_metrics(g_o["spatial"][b], g_r["spatial"][b])
                _, ref_iou_pp, _ = cam_metrics(g_o["spatial_pp"][b], g_r["spatial_pp"][b])
                p_o, p_d = float(sigmoid(g_o["logit"][b] / T)), float(sigmoid(g_d["logit"][b] / T))
                stable = bool((p_o >= thr) == (p_d >= thr))
                rows.append(dict(run_id=r["run_id"], train_on=r["train_on"], variant=r["variant"], seed=r["seed"], pair_id=x.pair_id,
                                 source_key=x.source_key, group_id=x.group_id, dataset=x.dataset, label=int(x.label), degradation=x.degradation,
                                 severity=x.severity, threshold=thr, p_fake_original=p_o, p_fake_degraded=p_d, prediction_stable=stable,
                                 pearson_r=pr, top20_iou=iou, center_shift=sh, temporal_cam_r=safe_corr(g_o["temporal"][b], g_d["temporal"][b]),
                                 attention_r=safe_corr(g_o["attn"][b], g_d["attn"][b]), explanation_changed=bool(iou < EXPLANATION_CHANGE_IOU),
                                 explanation_changed_while_prediction_stable=bool(stable and iou < EXPLANATION_CHANGE_IOU),
                                 clean_reference_iou=ref_iou, pearson_r_pp=prp, top20_iou_pp=ioup, clean_reference_iou_pp=ref_iou_pp))
            n_done += n
            el = time.time() - t0
            sys.stdout.write(f"\r    {progress_bar(n_done, len(pairs), 16)} {n_done}/{len(pairs)} | ETA {fmt_time(el / n_done * (len(pairs) - n_done))}   ")
            sys.stdout.flush()
        sys.stdout.write("\n")
        write_csv(pd.DataFrame(rows), out)
        if r["train_on"] == "Joint" and r["seed"] == a.seed_list[0] and r["variant"] in a.variant_list:
            save_cam_examples(P, D, a, tr, model, pairs[:: max(1, len(pairs) // 6)][:6], r["run_id"], T)
        del model
        gc.collect()
        torch.cuda.empty_cache()


def save_cam_examples(P: Paths, D: FData, a, tr: Trainer, model, pairs, rid: str, T: float = 1.0):
    if not pairs:
        return
    dev = tr.device
    fig, axes = plt.subplots(len(pairs), 4, figsize=(9, 2.4 * len(pairs)))
    axes = np.atleast_2d(axes)
    for row, (s, j, x) in enumerate(pairs):
        cc, dc = D.clip("clean", s, EVEN_FRAMES)[None], D.clip("deg", j, EVEN_FRAMES)[None]
        g_o = grad_cam_eval(model, to_input(cc, dev, a.img), frame_positions(1, EVEN_FRAMES, dev), 1.0, amp=a.amp)
        g_d = grad_cam_eval(model, to_input(dc, dev, a.img), frame_positions(1, EVEN_FRAMES, dev), 1.0, amp=a.amp)
        r_, iou, _ = cam_metrics(g_o["spatial"][0], g_d["spatial"][0])
        mid = CLIP_LEN // 2
        panels = [(cc[0, mid], None, f"original  p={float(sigmoid(g_o['logit'][0] / T)):.2f}"), (cc[0, mid], g_o["spatial"][0], "Grad-CAM original"),
                  (dc[0, mid], None, f"{x.degradation}/{x.severity}  p={float(sigmoid(g_d['logit'][0] / T)):.2f}"),
                  (dc[0, mid], g_d["spatial"][0], f"Grad-CAM degraded  IoU={iou:.2f}  r={r_:.2f}")]
        for k, (img, cam, title) in enumerate(panels):
            axes[row, k].imshow(img)
            if cam is not None:
                axes[row, k].imshow(cam, cmap="magma", alpha=0.5)
            axes[row, k].set_title(title, fontsize=7)
            axes[row, k].axis("off")
    plt.suptitle(f"Grad-CAM examples ({rid}); stability is not correctness", fontsize=9)
    plt.tight_layout()
    plt.savefig(P.examples / f"gradcam_examples_{rid}.png", dpi=110)
    plt.close(fig)


def faith_pairs(a, D: FData) -> list:
    pairs = load_xai_pairs(a, D)
    if a.smoke:
        return pairs[:: max(1, len(pairs) // 6)][:6]
    return pairs[::FAITH_STRIDE]


def faith_curves(model: FinalDetector, dev, a, clip: np.ndarray, frames: np.ndarray, attr: np.ndarray, target: int, T: float,
                 tie_rng) -> dict:
    """Deletion and insertion curves (confidence of the predicted class) for grey and blur perturbations (171026 E6 protocol)."""
    blurred = np.stack([cv2.GaussianBlur(f, (0, 0), 8) for f in clip])
    grey = np.full_like(clip, 128)
    flat = attr.reshape(len(attr), -1).astype(np.float64)
    flat = flat + tie_rng.random(flat.shape) * 1e-9 * (np.abs(flat).max() + 1e-12)       # deterministic tie-break
    order = np.argsort(-flat, axis=1, kind="stable")
    npx = flat.shape[1]
    batch, keys = [], []
    for kind, base in (("grey", grey), ("blur", blurred)):
        for frac in FAITH_FRACS:
            k = int(round(frac * npx))
            m = np.zeros(flat.shape, dtype=bool)
            if k:
                np.put_along_axis(m, order[:, :k], True, axis=1)
            m = m.reshape(attr.shape)
            dele = clip.copy()
            dele[m] = base[m]
            ins = base.copy()
            ins[m] = clip[m]
            batch += [dele, ins]
            keys += [("del", kind, frac), ("ins", kind, frac)]
    probs = []
    with torch.no_grad():
        for i in range(0, len(batch), 12):
            xb = np.stack(batch[i:i + 12])
            with torch.autocast("cuda", dtype=torch.float16, enabled=a.amp):
                lg = model(to_input(xb, dev, a.img), frame_positions(len(xb), frames, dev))["logit"]
            probs += list(sigmoid(lg.float().cpu().numpy() / T))
    conf = {k: (p if target == 1 else 1.0 - p) for k, p in zip(keys, probs)}
    out = {}
    for kind in ("grey", "blur"):
        d = [conf[("del", kind, f)] for f in FAITH_FRACS]
        i_ = [conf[("ins", kind, f)] for f in FAITH_FRACS]
        out[f"deletion_auc_{kind}"] = float(_trap(d, FAITH_FRACS) / FAITH_FRACS[-1])
        out[f"insertion_auc_{kind}"] = float(_trap(i_, FAITH_FRACS) / FAITH_FRACS[-1])
        out[f"drop20_{kind}"] = float(d[0] - conf[("del", kind, 0.20)])
        for n_, (dv, iv) in enumerate(zip(d, i_)):
            out[f"del_{kind}_{n_}"] = float(dv)
            out[f"ins_{kind}_{n_}"] = float(iv)
    return out


def phase_faith(a, P: Paths, D: FData):
    """Explanation faithfulness on the ORIGINAL and the DEGRADED version of the same videos: deletion / insertion of the
    Grad-CAM, Grad-CAM++ and a random map (same random maps for every run: paired)."""
    require_link(P)
    if a.faith_scope == "none":
        return
    pairs = faith_pairs(a, D)
    if not pairs:
        print("  no faithfulness pairs in this data subset")
        return
    for r in available_runs(a, P, a.faith_scope):
        out = P.faith_dir / f"{r['run_id']}.csv"
        if out.exists():
            continue
        tr = Trainer(a, P, D, r["train_on"], r["seed"])
        model, obj = tr.load(r["ckpt"])
        thr, T = float(obj["threshold"]), float(obj["temperature"])
        dev = tr.device
        rows, t0 = [], time.time()
        print(f"  faithfulness {r['run_id']} on {len(pairs)} pairs x (original, degraded) ...")
        for k, (s, j, x) in enumerate(pairs):
            for vi, (version, kind, idx) in enumerate((("original", "clean", s), ("degraded", "deg", j))):
                clip = D.clip(kind, idx, EVEN_FRAMES)
                g = grad_cam_eval(model, to_input(clip[None], dev, a.img), frame_positions(1, EVEN_FRAMES, dev),
                                  sign=lambda z: np.where(sigmoid(z / T) >= thr, 1.0, -1.0), amp=a.amp)
                p = float(sigmoid(g["logit"][0] / T))
                pred = int(p >= thr)
                rmap = np.random.default_rng([FAITH_SEED, k, vi]).random(clip.shape[:3]).astype(np.float32)
                for meth, mp in (("gradcam", g["cam"][0]), ("gradcam++", g["campp"][0]), ("random", rmap)):
                    fc = faith_curves(model, dev, a, clip, EVEN_FRAMES, mp, pred, T, np.random.default_rng([FAITH_SEED, k, vi, 7]))
                    rows.append(dict(run_id=r["run_id"], train_on=r["train_on"], variant=r["variant"], seed=r["seed"], pair_id=x.pair_id,
                                     source_key=x.source_key, group_id=x.group_id, dataset=x.dataset, label=int(x.label),
                                     degradation=x.degradation, severity=x.severity, version=version, p_fake=p, pred=pred,
                                     correct=bool(pred == int(x.label)), method=meth, **fc))
            el = time.time() - t0
            sys.stdout.write(f"\r    {progress_bar(k + 1, len(pairs), 16)} {k + 1}/{len(pairs)} | ETA {fmt_time(el / (k + 1) * (len(pairs) - k - 1))}   ")
            sys.stdout.flush()
        sys.stdout.write("\n")
        write_csv(pd.DataFrame(rows), out)
        del model
        gc.collect()
        torch.cuda.empty_cache()


# =====================================================================================
# ANALYSIS (final runs + Test 5 / Test 7 baselines on the same test videos)
# =====================================================================================
def setting_of(train_on: str, ds: str) -> str:
    return "joint" if train_on == "Joint" else ("in-domain" if ds == train_on else "cross-dataset")


def subset_map(a) -> Dict[str, str]:
    man = read_csv(a.t5.split_manifest, "t5_split", low_memory=False, usecols=["item_id", "kind", "subset"])
    o = man[man.kind == "original"]
    return dict(zip(o.item_id, o.subset))


def load_predictions(a, P: Paths) -> pd.DataFrame:
    fs = sorted(P.pred_dir.glob("*.csv.gz"))
    pred = pd.concat([read_csv(f, "final_predictions") for f in fs], ignore_index=True) if fs else pd.DataFrame()
    if not len(pred):
        return pred
    pred["source_test"] = "Final"
    test_keys = set(pred[pred.split == "test"].source_key)
    sub_of = subset_map(a)
    nfp = pred.drop_duplicates("source_key").set_index("source_key").native_face_px
    base = []
    for src, v, name in BASELINES:
        pdir = a.t5.pred_dir if src == "T5" else a.t7.pred_dir
        tsets = [t for t in a.train_list if t in DATASETS] if src == "T5" else list(a.train_list)
        for t in tsets:
            for seed in a.seed_list:
                f = pdir / f"{run_id(t, seed, v)}.csv.gz"
                if not f.exists():
                    continue
                x = read_csv(f, "baseline_predictions")
                x = x[(x.condition == "normal") & x.source_key.isin(test_keys)].copy()
                if not len(x):
                    continue
                x["variant"], x["run_id"] = name, run_id(t, seed, name)
                x["split"], x["source_test"] = "test", ("Test 5" if src == "T5" else "Test 7")
                x["subset"] = x.source_key.map(sub_of)
                if "p_fake_raw" not in x:
                    x["p_fake_raw"] = x.p_fake                 # Test 5 stored uncalibrated probabilities only
                if "logit" not in x:
                    x["logit"] = np.nan
                if "temperature" not in x:
                    x["temperature"] = 1.0
                x["native_face_px"] = x.source_key.map(nfp)
                base.append(x[[c for c in PRED_COLS if c in x.columns] + ["source_test"]])
    return pd.concat([pred] + base, ignore_index=True) if base else pred


def load_cams(a, P: Paths) -> pd.DataFrame:
    fs = sorted(P.cam_dir.glob("*.csv"))
    cams = pd.concat([read_csv(f, "final_gradcam") for f in fs], ignore_index=True) if fs else pd.DataFrame()
    if not len(cams):
        return cams
    ids = set(cams.pair_id)
    base = []
    for src, v, name in BASELINES:
        cdir = a.t5.cam_dir if src == "T5" else a.t7.cam_dir
        tsets = [t for t in a.train_list if t in DATASETS] if src == "T5" else list(a.train_list)
        for t in tsets:
            for seed in a.seed_list:
                f = cdir / f"{run_id(t, seed, v)}.csv"
                if f.exists():
                    x = read_csv(f, "baseline_gradcam")
                    x = x[x.pair_id.isin(ids)].copy()
                    x["variant"], x["run_id"] = name, run_id(t, seed, name)
                    base.append(x)
    return pd.concat([cams] + base, ignore_index=True) if base else cams


def per_run_tables(pred: pd.DataFrame):
    met, degm, rob, manip = [], [], [], []
    te = pred[(pred.split == "test") & (pred.condition == "normal")]
    for rid, g in te.groupby("run_id"):
        r0 = g.iloc[0]
        thr = float(r0.threshold)
        base = dict(run_id=rid, train_on=r0.train_on, variant=r0.variant, seed=int(r0.seed), threshold=thr, source_test=r0.source_test)
        for ds, h in g.groupby("dataset"):
            setting = setting_of(r0.train_on, ds)
            c, d = h[h.kind == "clean"], h[h.kind == "deg"]
            for cn, x in (("clean", c), ("degraded_all", d), ("degraded_hard", d[d.degradation.isin(HARD_TYPES)])):
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
                x = c[(c.label == lab) & np.isfinite(c.native_face_px.astype(float))]
                rho[nm] = (float(spearmanr(x.native_face_px, x.p_fake)[0]) if len(x) > 5 and x.native_face_px.nunique() > 1
                           and x.p_fake.nunique() > 1 else np.nan)
            dh = d[d.degradation.isin(HARD_TYPES)]
            rob.append(dict(base, test_dataset=ds, setting=setting, n_clean=len(c), n_degraded=len(d), clean_auc=mc["auc"],
                            degraded_auc_pooled=md_["auc"], clean_balanced_accuracy=mc["balanced_accuracy"],
                            degraded_balanced_accuracy=md_["balanced_accuracy"], real_recognised=mc["specificity"], fake_detected=mc["recall"],
                            real_recognised_degraded=md_["specificity"], fake_detected_degraded=md_["recall"],
                            prediction_flip_rate_pct=100 * d.flip.mean() if len(d) else np.nan, mean_abs_prob_shift=d["shift"].mean(),
                            worst_type=min(ts, key=lambda t_: t_[1])[0] if ts else "", worst_type_auc=min(v for _, v in ts) if ts else np.nan,
                            macro_type_auc=float(np.mean([v for _, v in ts])) if ts else np.nan,
                            auc_hard_pooled=safe_auc(dh.label, dh.p_fake, MIN_PER_CLASS_AUC),
                            **{f"auc_{t}": type_auc.get(t, np.nan) for t in HARD_TYPES + ["noise", "compression", "motion_blur", "frame_drop"]},
                            ece_clean=ece_score(c.label, c.p_fake), ece_clean_raw=ece_score(c.label, c.p_fake_raw),
                            ece_degraded=ece_score(d.label, d.p_fake), brier_clean=float(np.mean((c.p_fake - c.label) ** 2)) if len(c) else np.nan,
                            rho_resolution_real=rho["real"], rho_resolution_fake=rho["fake"]))
            if ds == "FF++":
                for cond, x in (("original", c), ("degraded", d)):
                    reals = x[x.label == 0]
                    manip.append(dict(base, test_dataset=ds, setting=setting, condition=cond, subset="original (real)", n=len(reals),
                                      recall=np.nan, specificity=float((reals.p_fake < thr).mean()) if len(reals) else np.nan, auc_vs_reals=np.nan))
                    for sub in FF_FAKES:
                        f = x[x.subset == sub]
                        if not len(f):
                            continue
                        yy = np.r_[np.zeros(len(reals)), np.ones(len(f))]
                        pp = np.r_[reals.p_fake.values, f.p_fake.values]
                        manip.append(dict(base, test_dataset=ds, setting=setting, condition=cond, subset=sub, n=len(f),
                                          recall=float((f.p_fake >= thr).mean()), specificity=np.nan, auc_vs_reals=safe_auc(yy, pp, MIN_PER_CLASS_AUC)))
    return pd.DataFrame(met), pd.DataFrame(degm), pd.DataFrame(rob), pd.DataFrame(manip)


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
                             mean_clean_reference_iou=x.clean_reference_iou.mean(),
                             relative_iou=x.top20_iou.mean() / x.clean_reference_iou.mean() if x.clean_reference_iou.mean() > 0 else np.nan,
                             mean_attention_r=x["attention_r"].mean() if "attention_r" in x else np.nan,
                             mean_top20_iou_pp=x["top20_iou_pp"].mean() if "top20_iou_pp" in x else np.nan,
                             mean_clean_reference_iou_pp=x["clean_reference_iou_pp"].mean() if "clean_reference_iou_pp" in x else np.nan))
    return pd.DataFrame(rows)


SUMMARY_METRICS = ["clean_auc", "degraded_auc_pooled", "worst_type_auc", "macro_type_auc", "auc_hard_pooled", "auc_noise",
                   "prediction_flip_rate_pct", "real_recognised", "fake_detected", "clean_balanced_accuracy", "degraded_balanced_accuracy",
                   "ece_clean", "ece_clean_raw", "ece_degraded", "rho_resolution_real"]
CAM_SUMMARY_METRICS = ["mean_top20_iou", "mean_pearson_r", "explanation_changed_given_stable_pct", "mean_clean_reference_iou", "relative_iou",
                       "mean_attention_r", "mean_top20_iou_pp"]


def seed_summary(rob: pd.DataFrame, cs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    srcs = [(m, rob) for m in SUMMARY_METRICS] + [(m, cs[cs.scope.isin(["all", "hard"])] if len(cs) else cs) for m in CAM_SUMMARY_METRICS]
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


def run_vectors(h: pd.DataFrame, Wg: np.ndarray, gpos: Dict[str, int], camr: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Point estimate (row 0) + bootstrap replicates of every metric of one run on one test dataset."""
    nb = len(Wg)
    nan = np.full(nb, np.nan)
    thr = float(h.threshold.iloc[0])
    c, d = h[h.kind == "clean"], h[h.kind == "deg"]

    def W_of(x):
        return Wg[:, x.group_id.astype(str).map(gpos).values.astype(int)]

    def wauc(x):
        if (x.label == 0).sum() < MIN_PER_CLASS_AUC or (x.label == 1).sum() < MIN_PER_CLASS_AUC:
            return nan.copy()
        return WeightedAUC(x.label.values, x.p_fake.values)(W_of(x))

    def wbacc(x):
        if not len(x):
            return nan.copy()
        W = W_of(x)
        y = x.label.values.astype(int)
        pr = x.p_fake.values >= thr
        r = (W[:, y == 1] * pr[y == 1]).sum(1) / np.maximum(W[:, y == 1].sum(1), 1e-9) if (y == 1).any() else nan.copy()
        s = (W[:, y == 0] * ~pr[y == 0]).sum(1) / np.maximum(W[:, y == 0].sum(1), 1e-9) if (y == 0).any() else nan.copy()
        return (r + s) / 2

    out = dict(clean_auc=wauc(c), degraded_auc=wauc(d), hard_auc=wauc(d[d.degradation.isin(HARD_TYPES)]),
               noise_auc=wauc(d[d.degradation == "noise"]), bacc_clean=wbacc(c), bacc_degraded=wbacc(d))
    ta = [v for _, x in d.groupby("degradation") for v in [wauc(x)] if np.isfinite(v[0])]
    out["worst_type_auc"] = np.nanmin(np.vstack(ta), axis=0) if ta else nan.copy()
    if len(c) and len(d):
        pc = d.source_key.map(dict(zip(c.source_key, c.p_fake)))
        ok = pc.notna().values
        fl = ((d.p_fake.values >= thr) != (pc.fillna(-1).values >= thr)).astype(float)
        W = W_of(d) * ok
        out["flip_rate_pct"] = 100 * (W * fl).sum(1) / np.maximum(W.sum(1), 1e-9)
    if len(c):
        out["ece_clean"] = weighted_ece(c.label.values, c.p_fake.values, W_of(c))
    if len(d):
        out["ece_degraded"] = weighted_ece(d.label.values, d.p_fake.values, W_of(d))
    if camr is not None and len(camr):
        gi = camr.group_id.astype(str).map(gpos)
        okc = gi.notna().values
        Wc = Wg[:, gi.fillna(0).astype(int).values] * okc
        for nm, m in (("gradcam_iou_all", np.ones(len(camr), bool)), ("gradcam_iou_hard", camr.degradation.isin(HARD_TYPES).values)):
            if m.any():
                out[nm] = (Wc[:, m] * camr.top20_iou.values[m]).sum(1) / np.maximum(Wc[:, m].sum(1), 1e-9)
        stb = as_bool(camr.prediction_stable).values
        if stb.any():
            chg = as_bool(camr.explanation_changed).values.astype(float)
            out["changed_while_stable_pct"] = 100 * (Wc[:, stb] * chg[stb]).sum(1) / np.maximum(Wc[:, stb].sum(1), 1e-9)
    return out


def noninferior(metric: str, lo_ci: float) -> Optional[bool]:
    if not np.isfinite(lo_ci):
        return None
    if metric in AUC_METRICS:
        return bool(lo_ci > -NONINF_AUC)
    if metric in IOU_METRICS:
        return bool(lo_ci > -NONINF_XAI)
    return None


LEVEL_METRICS = ["clean_auc", "degraded_auc", "worst_type_auc", "flip_rate_pct", "ece_clean", "bacc_clean", "bacc_degraded",
                 "gradcam_iou_all", "gradcam_iou_hard"]


def bootstrap_stats(a, pred: pd.DataFrame, cams: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Paired comparisons on the same test videos (final variants and Test 5 / Test 7 baselines share split, items and seeds).
    Seed uncertainty: per-seed differences + sign agreement. Sample uncertainty: cluster bootstrap over source groups (the
    same replicate weights for every run of a test dataset, so comparisons across training sets are paired too)."""
    te = pred[(pred.split == "test") & (pred.condition == "normal")]
    rows, levels = [], []
    for ds, g in te.groupby("dataset"):
        groups = np.array(sorted(g.group_id.astype(str).unique()))
        G = len(groups)
        gpos = {x: i for i, x in enumerate(groups)}
        rng = np.random.default_rng([20261023, sum(map(ord, ds))])
        Wg = np.vstack([np.ones(G), rng.multinomial(G, np.full(G, 1.0 / G), size=a.n_boot).astype(np.float64)])
        camd = cams[cams.dataset == ds] if len(cams) else cams
        per = {}
        for rid, h in g.groupby("run_id"):
            camr = camd[camd.run_id == rid] if len(camd) else camd
            per[(str(h.train_on.iloc[0]), str(h.variant.iloc[0]), int(h.seed.iloc[0]))] = run_vectors(h, Wg, gpos, camr)
        for t, v in sorted({(k[0], k[1]) for k in per}):
            seeds = sorted(k[2] for k in per if k[0] == t and k[1] == v)
            for metric in LEVEL_METRICS:
                if not all(metric in per[(t, v, s)] for s in seeds):
                    continue
                arr = np.nanmean(np.vstack([per[(t, v, s)][metric] for s in seeds]), 0)
                b = arr[1:][np.isfinite(arr[1:])]
                levels.append(dict(train_on=t, test_dataset=ds, setting=setting_of(t, ds), variant=v, metric=metric, n_seeds=len(seeds),
                                   seed_mean=arr[0], ci95_low=np.percentile(b, 2.5) if len(b) > 20 else np.nan,
                                   ci95_high=np.percentile(b, 97.5) if len(b) > 20 else np.nan))
        specs = [((t, hi), (t, lo), f"{hi} - {lo}", setting_of(t, ds)) for t in sorted({k[0] for k in per}) for hi, lo in COMPARISONS]
        specs += [((th, vh), (tl, vl), f"{vh}@{th} - {vl}@{tl}", "joint vs dataset-specific")
                  for (th, vh), (tl, vl), dsx in CROSS_COMPARISONS if dsx == ds]
        for (th, vh), (tl, vl), label, setting in specs:
            seeds = sorted({k[2] for k in per if (k[0], k[1]) == (th, vh)} & {k[2] for k in per if (k[0], k[1]) == (tl, vl)})
            if not seeds:
                continue
            for metric in per[(th, vh, seeds[0])]:
                if not all(metric in per[(th, vh, s)] and metric in per[(tl, vl, s)] for s in seeds):
                    continue
                diffs = np.vstack([per[(th, vh, s)][metric] - per[(tl, vl, s)][metric] for s in seeds])
                point = diffs[:, 0]
                if not np.isfinite(point).all():
                    continue
                boot = np.nanmean(diffs[:, 1:], axis=0)
                boot = boot[np.isfinite(boot)]
                lo_ci, hi_ci = (np.percentile(boot, [2.5, 97.5]) if len(boot) > 20 else (np.nan, np.nan))
                want = -1 if metric in LOWER_BETTER else 1
                mean = float(point.mean())
                same_sign = bool(np.all(np.sign(point) == np.sign(mean))) if mean != 0 else False
                if G < MIN_CLUSTERS or not np.isfinite(lo_ci):
                    verdict = "inconclusive (too few clusters)"
                elif len(seeds) < 2:
                    verdict = "single seed: CI excludes 0" if (lo_ci > 0 or hi_ci < 0) else "single seed: inconclusive"
                elif (lo_ci > 0 or hi_ci < 0) and same_sign:
                    verdict = "supported improvement" if np.sign(mean) == want else "supported deterioration"
                else:
                    verdict = "inconclusive"
                rows.append(dict(train_on=th, baseline_train_on=tl, test_dataset=ds, setting=setting, comparison=label, metric=metric,
                                 better_direction="lower" if want < 0 else "higher", n_seeds=len(seeds), mean_diff=mean,
                                 sd_diff_across_seeds=point.std(ddof=1) if len(seeds) > 1 else np.nan,
                                 per_seed_diffs=";".join(f"{s}:{x:+.4f}" for s, x in zip(seeds, point)), sign_agreement=same_sign,
                                 ci95_low=lo_ci, ci95_high=hi_ci, p_boot=p_boot(boot), n_clusters=G, n_boot=len(boot), verdict=verdict,
                                 noninferior=noninferior(metric, lo_ci)))
    st = pd.DataFrame(rows)
    if len(st):
        prim = set(PRIMARY)
        st["primary"] = [k in prim for k in zip(st.comparison, st.train_on, st.test_dataset, st.metric)]
        st["p_holm"] = np.nan
        if st.primary.any():
            st.loc[st.primary, "p_holm"] = holm(st.loc[st.primary, "p_boot"].values)
        st["significant_holm_0.05"] = st.p_holm < 0.05
    return st, pd.DataFrame(levels)


# ------------------------------------------------------------------------------------- faithfulness
FAITH_VALS = ["deletion_auc_grey", "insertion_auc_grey", "deletion_auc_blur", "insertion_auc_blur"]


def load_faith(P: Paths) -> pd.DataFrame:
    fs = sorted(P.faith_dir.glob("*.csv"))
    return pd.concat([read_csv(f, "final_faithfulness") for f in fs], ignore_index=True) if fs else pd.DataFrame()


def faith_per_video(fx: pd.DataFrame) -> pd.DataFrame:
    """One row per run x pair x version; gains vs the random map (higher = more faithful)."""
    idx = ["run_id", "train_on", "variant", "seed", "pair_id", "group_id", "dataset", "label", "degradation", "version", "pred"]
    pv = fx.pivot_table(index=idx, columns="method", values=FAITH_VALS).reset_index()
    out = pv[idx].copy()
    out.columns = idx
    for m, tag in (("gradcam", "gc"), ("gradcam++", "gcpp")):
        if ("deletion_auc_grey", m) not in pv.columns or ("deletion_auc_grey", "random") not in pv.columns:
            continue
        for k in ("grey", "blur"):
            out[f"{tag}_del_gain_{k}"] = (pv[(f"deletion_auc_{k}", "random")] - pv[(f"deletion_auc_{k}", m)]).values
            out[f"{tag}_ins_gain_{k}"] = (pv[(f"insertion_auc_{k}", m)] - pv[(f"insertion_auc_{k}", "random")]).values
    out["pred_class"] = np.where(out.pred.astype(int) == 1, "FAKE", "REAL")
    return out


def faith_tables(a, fx: pd.DataFrame):
    if not len(fx):
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    pv = faith_per_video(fx)
    keys = [c for c in pv.columns if "_gain_" in c]
    groups = np.array(sorted(pv.group_id.astype(str).unique()))
    gpos = {x: i for i, x in enumerate(groups)}
    rng = np.random.default_rng(FAITH_SEED)
    Wg = np.vstack([np.ones(len(groups)), rng.multinomial(len(groups), np.full(len(groups), 1.0 / len(groups)), size=a.n_boot)]).astype(float)

    def est(vals: np.ndarray, grp: pd.Series) -> np.ndarray:
        vals = np.asarray(vals, float)
        ok = np.isfinite(vals)                         # a non-finite model output (fp16 on a heavily masked clip) drops that video
        if not ok.any():
            return np.full(len(Wg), np.nan)
        W = Wg[:, grp.astype(str).map(gpos).values.astype(int)][:, ok]
        return (W * vals[ok]).sum(1) / np.maximum(W.sum(1), 1e-9)

    summ, comps = [], []
    for pcls in ("ALL", "FAKE", "REAL"):
        x0 = pv if pcls == "ALL" else pv[pv.pred_class == pcls]
        for (t, v, ver), g in x0.groupby(["train_on", "variant", "version"]):
            m = g.groupby(["pair_id", "group_id"])[keys].mean().reset_index()
            for k in keys:
                e = est(m[k].values, m.group_id)
                lo, hi = np.nanpercentile(e[1:], [2.5, 97.5]) if np.isfinite(e[1:]).any() else (np.nan, np.nan)
                summ.append(dict(train_on=t, variant=v, version=ver, pred_class=pcls, metric=k, n_pairs=len(m), n_seeds=int(g.seed.nunique()),
                                 mean=e[0], ci95_low=lo, ci95_high=hi,
                                 verdict="faithful (better than random)" if lo > 0.005 else ("worse than random" if hi < -0.005 else
                                                                                           "not distinguishable from random")))
    for (t, v), g in pv.groupby(["train_on", "variant"]):           # degraded - original, paired by pair and seed
        o = g[g.version == "original"].set_index(["pair_id", "seed"])
        d = g[g.version == "degraded"].set_index(["pair_id", "seed"])
        common = o.index.intersection(d.index)
        if not len(common):
            continue
        diff = (d.loc[common, keys] - o.loc[common, keys]).groupby(level=0).mean()
        grp = o.loc[common, "group_id"].groupby(level=0).first().loc[diff.index]
        for k in keys:
            e = est(diff[k].values, grp)
            lo, hi = np.nanpercentile(e[1:], [2.5, 97.5]) if np.isfinite(e[1:]).any() else (np.nan, np.nan)
            comps.append(dict(comparison="degraded - original", train_on=t, variant=v, version="paired", metric=k, n_pairs=len(diff),
                              mean_diff=e[0], ci95_low=lo, ci95_high=hi, p_boot=p_boot(e[1:] - 0.0),
                              verdict="lower on degraded videos" if hi < 0 else ("higher on degraded videos" if lo > 0 else "no supported change"),
                              noninferior=bool(lo > -NONINF_XAI)))
    for t in sorted(pv.train_on.unique()):
        for ver in ("original", "degraded"):
            for hi_v, lo_v in (("F2", "F1"), ("F2", "F0"), ("F1", "F0")):
                A = pv[(pv.train_on == t) & (pv.variant == hi_v) & (pv.version == ver)].groupby("pair_id")[keys].mean()
                B = pv[(pv.train_on == t) & (pv.variant == lo_v) & (pv.version == ver)].groupby("pair_id")[keys].mean()
                common = A.index.intersection(B.index)
                if not len(common):
                    continue
                diff = A.loc[common] - B.loc[common]
                grp = pv.drop_duplicates("pair_id").set_index("pair_id").group_id.loc[common]
                for k in keys:
                    e = est(diff[k].values, grp)
                    lo, hi = np.nanpercentile(e[1:], [2.5, 97.5]) if np.isfinite(e[1:]).any() else (np.nan, np.nan)
                    comps.append(dict(comparison=f"{hi_v} - {lo_v}", train_on=t, variant=f"{hi_v} vs {lo_v}", version=ver, metric=k,
                                      n_pairs=len(diff), mean_diff=e[0], ci95_low=lo, ci95_high=hi, p_boot=p_boot(e[1:]),
                                      verdict="higher" if lo > 0 else ("lower" if hi < 0 else "no supported change"),
                                      noninferior=bool(lo > -NONINF_XAI)))
    return pv, pd.DataFrame(summ), pd.DataFrame(comps)


# ------------------------------------------------------------------------------------- calibration + abstention
def calibration_tables(pred: pd.DataFrame):
    """Post-hoc calibration (fitted on the run's own validation videos) evaluated on test originals and degraded versions."""
    rows, curves = [], []
    fin = pred[pred.source_test == "Final"] if "source_test" in pred else pred
    for rid, g in fin.groupby("run_id"):
        r0 = g.iloc[0]
        train_ds = DATASETS if r0.train_on == "Joint" else [r0.train_on]
        v = g[(g.split == "val") & g.dataset.isin(train_ds)]
        t = g[g.split == "test"]
        if v.label.nunique() < 2 or not len(t):
            continue
        zv, yv, dv = v.logit.values.astype(float), v.label.values.astype(int), v.dataset.values
        T = float(r0.temperature)
        iso = IsoCal().fit(zv, yv)
        fns = {"raw": sigmoid, "temperature": (lambda z, T_=T: sigmoid(np.asarray(z, float) / T_)), "isotonic": iso}
        for cm, fn in fns.items():
            pv_ = fn(zv)
            thr = choose_threshold(yv, pv_, dv)
            for cov in COVERAGES:
                band = abstain_band(pv_, thr, cov)
                for ts in [x for x in DATASETS if x in set(t.dataset)]:
                    xa = t[t.dataset == ts]
                    for cond, x in (("original", xa[xa.kind == "clean"]), ("degraded", xa[xa.kind == "deg"])):
                        if not len(x):
                            continue
                        pt, yt = fn(x.logit.values.astype(float)), x.label.values.astype(int)
                        keep = np.abs(pt - thr) >= band
                        pr = (pt >= thr).astype(int)
                        rows.append(dict(run_id=rid, train_on=r0.train_on, variant=r0.variant, seed=int(r0.seed), test_dataset=ts,
                                         setting=setting_of(r0.train_on, ts), calibration=cm, target_coverage=cov, threshold=thr, band=band,
                                         condition=cond, n=len(x), coverage=float(keep.mean()), ece=ece_score(yt, pt),
                                         brier=float(np.mean((pt - yt) ** 2)), bacc_all=mean_bacc(yt, pr, x.dataset.values),
                                         bacc_decided=mean_bacc(yt[keep], pr[keep], x.dataset.values[keep]) if keep.any() else np.nan,
                                         accuracy_decided=float((pr[keep] == yt[keep]).mean()) if keep.any() else np.nan))
                        if cm == "isotonic" and cov == 1.0:
                            conf = np.abs(pt - thr)
                            err = (pr != yt)[np.argsort(-conf, kind="stable")]
                            risk = np.cumsum(err) / np.arange(1, len(err) + 1)
                            for c_ in np.round(np.arange(0.50, 1.0001, 0.02), 2):
                                k = max(1, int(round(c_ * len(err))))
                                curves.append(dict(run_id=rid, train_on=r0.train_on, variant=r0.variant, seed=int(r0.seed), test_dataset=ts,
                                                   condition=cond, coverage=c_, risk=float(risk[k - 1])))
    cal = pd.DataFrame(rows)
    summ = pd.DataFrame()
    if len(cal):
        summ = cal.groupby(["train_on", "variant", "test_dataset", "setting", "calibration", "target_coverage", "condition"]).agg(
            n_runs=("run_id", "nunique"), coverage=("coverage", "mean"), ece=("ece", "mean"), ece_sd=("ece", "std"), brier=("brier", "mean"),
            bacc_all=("bacc_all", "mean"), bacc_decided=("bacc_decided", "mean"), bacc_decided_sd=("bacc_decided", "std")).reset_index()
    return cal, summ, pd.DataFrame(curves)


# ------------------------------------------------------------------------------------- hypotheses
def _verdict(x: pd.DataFrame) -> str:
    if not len(x):
        return "not tested (runs missing)"
    imp = int((x.verdict == "supported improvement").sum())
    det = int((x.verdict == "supported deterioration").sum())
    if imp == len(x):
        return "supported"
    if imp and not det:
        return "partially supported"
    if det and not imp:
        return "not supported (supported deterioration)"
    if imp and det:
        return "mixed"
    return "not supported (inconclusive)"


def _evidence(x: pd.DataFrame) -> str:
    return "; ".join(f"{r.train_on}->{r.test_dataset} {r.metric}: {r.mean_diff:+.4f} [{r.ci95_low:+.4f}, {r.ci95_high:+.4f}] ({r.verdict})"
                     for r in x.itertuples())


def hypothesis_verdicts(stt: pd.DataFrame, lev: pd.DataFrame, cal: pd.DataFrame, fcmp: pd.DataFrame) -> List[dict]:
    cols = ["comparison", "setting", "metric", "verdict", "train_on", "test_dataset", "mean_diff", "ci95_low", "ci95_high", "noninferior"]
    S = stt if len(stt) else pd.DataFrame(columns=cols)
    text = dict(HYPOTHESES)
    out = []
    x = S[(S.comparison == "F1 - T5-5A") & (S.setting == "in-domain") & S.metric.isin(["degraded_auc", "worst_type_auc"])]
    out.append(dict(hypothesis="H1", statement=text["H1"], comparison="F1 - T5-5A (in-domain)", verdict=_verdict(x), evidence=_evidence(x)))
    x = S[(S.comparison == "F1 - F0") & S.setting.isin(["in-domain", "joint"]) & S.metric.isin(["degraded_auc", "worst_type_auc", "flip_rate_pct"])]
    out.append(dict(hypothesis="H2", statement=text["H2"], comparison="F1 - F0", verdict=_verdict(x), evidence=_evidence(x)))
    xs = S[(S.comparison == "F2 - F1") & S.setting.isin(["in-domain", "joint"]) & S.metric.isin(["gradcam_iou_all", "gradcam_iou_hard"])]
    xd = S[(S.comparison == "F2 - F1") & S.setting.isin(["in-domain", "joint"]) & S.metric.isin(["clean_auc", "degraded_auc"])]
    stab = _verdict(xs)
    ni = bool(len(xd)) and all(bool(v) for v in xd.noninferior.tolist() if v is not None and v == v) and xd.noninferior.notna().all()
    if not len(xs):
        v3 = "not tested (runs missing)"
    elif stab in ("supported", "partially supported"):
        v3 = (stab if ni else f"{stab} for stability, but detection not non-inferior (margin {NONINF_AUC})")
    else:
        v3 = stab
    ev3 = _evidence(xs) + " | detection non-inferiority (CI lower bound > -0.01): " + "; ".join(
        f"{r.train_on}->{r.test_dataset} {r.metric}: {r.mean_diff:+.4f} [{r.ci95_low:+.4f}, {r.ci95_high:+.4f}] "
        f"({'non-inferior' if r.noninferior else 'not shown non-inferior'})" for r in xd.itertuples())
    out.append(dict(hypothesis="H3", statement=text["H3"], comparison="F2 - F1", verdict=v3, evidence=ev3))
    x = S[S.comparison.isin(["F2@Joint - T5-5A@BioDeepAV", "F2@Joint - T5-5A@FF++"]) & S.metric.isin(["degraded_auc", "worst_type_auc", "flip_rate_pct"])]
    out.append(dict(hypothesis="H4", statement=text["H4"], comparison="F2 (Joint) - 5A (dataset-specific)", verdict=_verdict(x), evidence=_evidence(x)))
    L = lev[(lev.variant == "F2") & (lev.setting == "cross-dataset") & lev.metric.isin(["clean_auc", "degraded_auc"])] if len(lev) else pd.DataFrame()
    if not len(L):
        v5, e5 = "not tested (dataset-specific F2 runs missing)", ""
    else:
        above = L.ci95_low > 0.5
        v5 = "supported" if above.all() else ("partially supported" if above.any() else "not supported")
        e5 = "; ".join(f"{r.train_on}->{r.test_dataset} {r.metric}: {r.seed_mean:.3f} [{r.ci95_low:.3f}, {r.ci95_high:.3f}]" for r in L.itertuples())
    out.append(dict(hypothesis="H5", statement=text["H5"], comparison="cross-dataset AUC level (F2)", verdict=v5, evidence=e5))
    c = cal[(cal.train_on == "Joint") & (cal.variant == "F2")] if len(cal) else pd.DataFrame()
    if not len(c):
        v6, e6 = "not tested (Joint F2 runs missing)", ""
    else:
        e = c[c.target_coverage == 1.0].pivot_table(index=["seed", "test_dataset", "condition"], columns="calibration", values="ece")
        b = c[c.calibration == "isotonic"].pivot_table(index=["seed", "test_dataset", "condition"], columns="target_coverage", values="bacc_decided")
        ece_ok = bool({"isotonic", "temperature"} <= set(e.columns) and (e["isotonic"] < e["temperature"]).all())
        ab_ok = bool({0.9, 1.0} <= set(b.columns) and (b[0.9] > b[1.0]).all())
        v6 = "supported" if ece_ok and ab_ok else ("partially supported" if ece_ok or ab_ok else "not supported")
        em = e.groupby(level="condition").mean() if len(e) else pd.DataFrame()
        bm = b.groupby(level="condition").mean() if len(b) else pd.DataFrame()
        e6 = "; ".join(f"{cond}: ECE temperature {em.loc[cond, 'temperature']:.3f} -> isotonic {em.loc[cond, 'isotonic']:.3f}"
                       for cond in em.index if {"temperature", "isotonic"} <= set(em.columns))
        e6 += " | " + "; ".join(f"{cond}: balanced accuracy 100 % coverage {bm.loc[cond, 1.0]:.3f} -> 90 % coverage {bm.loc[cond, 0.9]:.3f}"
                                for cond in bm.index if {0.9, 1.0} <= set(bm.columns))
        e6 += f" | isotonic ECE lower in every seed/test set/condition: {ece_ok}; abstention raises balanced accuracy everywhere: {ab_ok}"
    out.append(dict(hypothesis="H6", statement=text["H6"], comparison="Joint F2: isotonic vs temperature; 90 % vs 100 % coverage", verdict=v6, evidence=e6))
    f = fcmp[(fcmp.comparison == "degraded - original") & (fcmp.train_on == "Joint") & (fcmp.variant == "F2")
             & fcmp.metric.isin(["gc_del_gain_grey", "gc_ins_gain_grey"])] if len(fcmp) else pd.DataFrame()
    ef = "; ".join(f"{r.metric}: {r.mean_diff:+.4f} [{r.ci95_low:+.4f}, {r.ci95_high:+.4f}] ({r.verdict})" for r in f.itertuples()) if len(f) else ""
    out.append(dict(hypothesis="RQ", statement="Exploratory: does degradation change Grad-CAM faithfulness (Joint F2, degraded - original)?",
                    comparison="degraded - original", verdict="reported (exploratory)" if len(f) else "not tested", evidence=ef))
    return out


# ------------------------------------------------------------------------------------- figures
PANELS = [("Joint", "BioDeepAV"), ("Joint", "FF++"), ("BioDeepAV", "BioDeepAV"), ("FF++", "FF++")]
XLABEL = {"T5-5A": "5A\n(T5)", "T5-5C": "5C\n(T5)", "T7-6A": "6A\n(T7)", "T7-6B": "6B\n(T7)"}


def _style():
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif", "Times New Roman", "DejaVu Serif"], "font.size": 8.5,
                         "axes.titlesize": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#555",
                         "axes.linewidth": 0.6, "axes.grid": True, "grid.color": "#e3e3e3", "grid.linewidth": 0.6, "axes.axisbelow": True})


def _title(t: str, ds: str) -> str:
    return f"Joint \u2192 {ds}" if t == "Joint" else f"{t}-trained \u2192 {ds}"


def _bars(ax, df: pd.DataFrame, metric: str, t: str, ds: str, ylim, ylabel=None, pct=False, chance=None):
    """Seed-mean bar per model + one dot per seed; the mean is printed in a strip above the plotting range (never on a bar).
    ylim = (low, high); the high end grows when the data exceed it (e.g. a 78 % rate on a 0-70 % axis)."""
    x = df[(df.train_on == t) & (df.test_dataset == ds)]
    vs = [v for v in VARIANT_ORDER if v in set(x.variant)]
    lo, hi = ylim
    vals_all = pd.to_numeric(x[metric], errors="coerce").dropna() if metric in x else pd.Series(dtype=float)
    if len(vals_all) and vals_all.max() > hi:
        hi = float(vals_all.max()) * 1.08
        if pct:
            hi = float(np.ceil(hi / 10.0) * 10.0)
    if len(vals_all) and vals_all.min() < lo:
        lo = float(np.floor(vals_all.min() * 10.0) / 10.0)
    strip = 0.10 * (hi - lo)
    for i, v in enumerate(vs):
        vals = pd.to_numeric(x[x.variant == v][metric], errors="coerce").dropna().values.astype(float)
        if not len(vals):
            continue
        m = vals.mean()
        ax.bar(i, m, 0.7, color=VARIANT_COLOR[v], hatch=VARIANT_HATCH.get(v, ""), edgecolor="#8d8b83" if v in VARIANT_HATCH else "white",
               linewidth=0.4)
        ax.scatter(i + np.linspace(-0.15, 0.15, len(vals)), np.clip(vals, lo, hi), s=7, color="#222", zorder=3, linewidths=0)
        ax.text(i, hi + 0.5 * strip, f"{m:.0f}" if pct else (f"{m:.2f}".lstrip("0") if 0 <= m < 1 else f"{m:.2f}"),
                ha="center", va="center", fontsize=6.2, color="#333")
    if chance is not None:
        ax.axhline(chance, color="#555", lw=0.7, ls="--")
    ax.set_xticks(range(len(vs)))
    ax.set_xticklabels([XLABEL.get(v, v) for v in vs], fontsize=7)
    ax.set_ylim(lo, hi + strip)
    ax.set_yticks([y_ for y_ in ax.get_yticks() if lo - 1e-9 <= y_ <= hi + 1e-9])
    ax.spines["left"].set_bounds(lo, hi)
    ax.set_title(_title(t, ds), fontsize=8.5)
    ax.grid(axis="x", visible=False)
    if ylabel:
        ax.set_ylabel(ylabel)


def _bar_grid(P: Paths, df: pd.DataFrame, rows_spec, fname: str, panels=PANELS, chance=None):
    present = [(t, ds) for t, ds in panels if len(df) and ((df.train_on == t) & (df.test_dataset == ds)).any()]
    if not present:
        return
    fig, axs = plt.subplots(len(rows_spec), len(present), figsize=(1.85 * len(present) + 0.9, 2.4 * len(rows_spec)), squeeze=False)
    for i, (metric, ylim, ylabel, pct) in enumerate(rows_spec):
        for j, (t, ds) in enumerate(present):
            _bars(axs[i, j], df, metric, t, ds, ylim, ylabel if j == 0 else None, pct, chance)
    fig.tight_layout(h_pad=1.2)
    fig.savefig(P.figures / fname, dpi=200)
    plt.close(fig)


def make_figures(a, P: Paths, rob, cs, degm, manip, fsum, curves, pred):
    _style()
    jobs = []
    jobs.append(("fig_detection_auc.png", lambda: _bar_grid(P, rob, [("clean_auc", (0.4, 1.0), "AUC, original", False),
                                                                   ("degraded_auc_pooled", (0.4, 1.0), "AUC, degraded", False)], "fig_detection_auc.png")))
    jobs.append(("fig_worst_flip.png", lambda: _bar_grid(P, rob, [("worst_type_auc", (0.2, 1.0), "Worst-type AUC", False),
                                                                 ("prediction_flip_rate_pct", (0, 30), "Flip rate (%)", True)], "fig_worst_flip.png")))
    jobs.append(("fig_ece_real.png", lambda: _bar_grid(P, rob, [("ece_clean", (0, 0.45), "ECE (temperature)", False),
                                                               ("real_recognised", (0, 1.0), "Real videos recognised", False)], "fig_ece_real.png")))
    if len(cs):
        csx = cs.copy()
        csh = csx[csx.scope == "hard"]
        csa = csx[csx.scope == "all"]
        jobs.append(("fig_gradcam_stability.png", lambda: _gradcam_fig(P, csa, csh)))
    jobs.append(("fig_cross_dataset.png", lambda: _bar_grid(P, rob, [("clean_auc", (0.0, 1.0), "AUC, original", False),
                                                                    ("degraded_auc_pooled", (0.0, 1.0), "AUC, degraded", False)], "fig_cross_dataset.png",
                                                            panels=[("BioDeepAV", "FF++"), ("FF++", "BioDeepAV")], chance=0.5)))
    if len(degm):
        jobs.append(("fig_type_heatmap.png", lambda: _type_heatmap(P, degm)))
    if len(manip):
        jobs.append(("fig_per_manipulation.png", lambda: _manip_fig(P, manip)))
    if len(fsum):
        jobs.append(("fig_faithfulness.png", lambda: _faith_fig(P, fsum)))
    if len(curves):
        jobs.append(("fig_risk_coverage.png", lambda: _risk_fig(P, curves)))
    jobs.append(("fig_training_curves.png", lambda: _training_fig(a, P)))
    jobs.append(("fig_reliability_final.png", lambda: _reliability_fig(P, pred)))
    for name, fn in jobs:
        try:
            fn()
        except Exception as e:                                   # a figure must never stop the analysis
            print(f"  [figure] {name} skipped ({type(e).__name__}: {e})")
        finally:
            plt.close("all")


def _gradcam_fig(P, csa, csh):
    fig_df = pd.concat([csh, csa])
    present = [(t, ds) for t, ds in PANELS if ((fig_df.train_on == t) & (fig_df.test_dataset == ds)).any()]
    if not present:
        return
    fig, axs = plt.subplots(2, len(present), figsize=(1.85 * len(present) + 0.9, 4.8), squeeze=False)
    for j, (t, ds) in enumerate(present):
        _bars(axs[0, j], csh, "mean_top20_iou", t, ds, (0, 1.0), "Top-20 % IoU (hard)" if j == 0 else None)
        _bars(axs[1, j], csa, "explanation_changed_given_stable_pct", t, ds, (0, 60), "Expl. changed, pred.\nstable (%)" if j == 0 else None, True)
    fig.tight_layout(h_pad=1.2)
    fig.savefig(P.figures / "fig_gradcam_stability.png", dpi=200)
    plt.close(fig)


def _type_heatmap(P, degm):
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("b", ["#f4f8fd", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
    x = degm[(degm.severity == "severe") & degm.setting.isin(["in-domain", "joint"])]
    present = [(t, ds) for t, ds in PANELS if ((x.train_on == t) & (x.test_dataset == ds)).any()]
    if not present:
        return
    nc = 2 if len(present) > 2 else len(present)
    nr = int(math.ceil(len(present) / nc))
    fig, axs = plt.subplots(nr, nc, figsize=(4.3 * nc + 0.8, 3.9 * nr), squeeze=False, layout="constrained")
    for ax in axs.ravel()[len(present):]:
        ax.set_visible(False)
    im = None
    for ax, (t, ds) in zip(axs.ravel(), present):
        d = x[(x.train_on == t) & (x.test_dataset == ds)]
        p = d.pivot_table(index="degradation", columns="variant", values="auc", aggfunc="mean")
        cols = [v for v in VARIANT_ORDER if v in p.columns]
        p = p[cols]
        im = ax.imshow(p.values, cmap=cmap, vmin=0.3, vmax=1.0, aspect="auto")
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels([XLABEL.get(v, v) for v in cols], fontsize=7)
        ax.set_yticks(range(len(p)))
        ax.set_yticklabels([s.replace("_", " ") for s in p.index], fontsize=7.5)
        ax.grid(False)
        for i in range(p.shape[0]):
            for j in range(p.shape[1]):
                v = p.values[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5, color="white" if v > 0.75 else "#222")
        ax.set_title(_title(t, ds) + " (severe level)", fontsize=8.5)
        for sp in ax.spines.values():
            sp.set_visible(False)
    if im is not None:
        cb = fig.colorbar(im, ax=[ax for ax in axs.ravel()[:len(present)]], shrink=0.6)
        cb.set_label("AUC (seed mean, severe level)")
        cb.outline.set_visible(False)
    fig.savefig(P.figures / "fig_type_heatmap.png", dpi=200)
    plt.close(fig)


def _manip_fig(P, manip):
    x = manip[manip.train_on == "Joint"]
    if not len(x):
        x = manip[manip.setting == "in-domain"]
    if not len(x):
        return
    x = x.assign(value=np.where(x.subset == "original (real)", x.specificity, x.recall))
    groups = ["original (real)"] + [s for s in FF_FAKES if s in set(x.subset)]
    vs = [v for v in VARIANT_ORDER if v in set(x.variant)]
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.1), sharey=True)
    for ax, cond in zip(axs, ("original", "degraded")):
        y = x[x.condition == cond].groupby(["subset", "variant"]).value.mean().unstack("variant").reindex(groups)
        y = y[[v for v in vs if v in y.columns]]
        w = 0.82 / max(1, y.shape[1])
        for k, v in enumerate(y.columns):
            ax.bar(np.arange(len(y)) + k * w - 0.41 + w / 2, y[v].values, w, color=VARIANT_COLOR[v], label=XLABEL.get(v, v).replace("\n", " "),
                   hatch=VARIANT_HATCH.get(v, ""), edgecolor="white", linewidth=0.3)
        ax.axvline(0.5, color="#999", lw=0.6)
        ax.set_xticks(range(len(y)))
        ax.set_xticklabels([s.replace("original (real)", "REAL videos").replace("DeepFakeDetection", "DFD").replace("NeuralTextures", "NeuralTex.")
                            for s in y.index], fontsize=7, rotation=25, ha="right")
        ax.set_ylim(0, 1)
        ax.set_title(f"FF++ {cond} test videos", fontsize=8.5)
        ax.grid(axis="x", visible=False)
    axs[0].set_ylabel("share correct\n(reals recognised / fakes detected)")
    h, l_ = axs[0].get_legend_handles_labels()
    fig.legend(h, l_, loc="lower center", ncol=len(l_), frameon=False, fontsize=7)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(P.figures / "fig_per_manipulation.png", dpi=200)
    plt.close(fig)


def _faith_fig(P, fsum):
    x = fsum[(fsum.train_on == "Joint") & (fsum.pred_class.isin(["FAKE", "REAL"]))]
    if not len(x):
        x = fsum[fsum.pred_class.isin(["FAKE", "REAL"])]
    if not len(x):
        return
    vs = [v for v in ["F0", "F1", "F2"] if v in set(x.variant)]
    fig, axs = plt.subplots(2, 2, figsize=(7.2, 5.0), sharey="row")
    for i, metric in enumerate(["gc_ins_gain_grey", "gc_del_gain_grey"]):
        for j, pc in enumerate(["FAKE", "REAL"]):
            ax = axs[i, j]
            for k, ver in enumerate(["original", "degraded"]):
                y = x[(x.metric == metric) & (x.pred_class == pc) & (x.version == ver)].set_index("variant").reindex(vs)
                pos = np.arange(len(vs)) + (k - 0.5) * 0.36
                ax.bar(pos, y["mean"].values, 0.34, color=["#2a78d6", "#eb6834"][k], label=f"{ver} version", alpha=0.9)
                ax.errorbar(pos, y["mean"].values, yerr=[(y["mean"] - y["ci95_low"]).values, (y["ci95_high"] - y["mean"]).values],
                            fmt="none", ecolor="#222", lw=0.8, capsize=2)
            ax.axhline(0, color="#555", lw=0.7)
            ax.set_xticks(range(len(vs)))
            ax.set_xticklabels(vs)
            ax.set_title(f"{'insertion' if 'ins' in metric else 'deletion'} gain vs random map - predicted {pc}", fontsize=8.5)
            ax.grid(axis="x", visible=False)
        axs[i, 0].set_ylabel("AUC gain (grey perturbation)")
    h, l_ = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l_, loc="upper center", ncol=2, frameon=False, fontsize=7.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(P.figures / "fig_faithfulness.png", dpi=200)
    plt.close(fig)


def _risk_fig(P, curves):
    x = curves[(curves.train_on == "Joint") & (curves.variant == "F2")]
    if not len(x):
        return
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=True)
    for ax, ds in zip(axs, DATASETS):
        for cond, col in (("original", "#2a78d6"), ("degraded", "#eb6834")):
            y = x[(x.test_dataset == ds) & (x.condition == cond)].groupby("coverage").risk.mean()
            if len(y):
                ax.plot(y.index, y.values, color=col, lw=1.6, label=cond)
        ax.axvline(DEPLOY_COVERAGE, color="#555", lw=0.7, ls="--")
        ax.set_xlabel("coverage (share of videos decided)")
        ax.set_title(f"Joint F2 \u2192 {ds} (isotonic, seed mean)", fontsize=8.5)
    axs[0].set_ylabel("error rate on decided videos")
    axs[1].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(P.figures / "fig_risk_coverage.png", dpi=200)
    plt.close(fig)


def _training_fig(a, P):
    hs = sorted(P.hist_dir.glob(f"history_*_seed{a.seed_list[0]}_*.csv"))
    if not hs:
        return
    sets = [t for t in a.train_list if any(f"{train_tag(t)}_seed" in h.name for h in hs)]
    fig, axs = plt.subplots(1, len(sets), figsize=(2.6 * len(sets) + 0.6, 2.7), squeeze=False)
    for ax, t in zip(axs[0], sets):
        f0 = P.hist_dir / f"history_{run_id(t, a.seed_list[0], STAGE1)}.csv"
        off = len(pd.read_csv(f0)) if f0.exists() else a.epochs1
        for v in [STAGE1] + list(VARIANTS):
            f = P.hist_dir / f"history_{run_id(t, a.seed_list[0], v)}.csv"
            if f.exists():
                h = pd.read_csv(f)
                ax.plot(h.epoch + (0 if v == STAGE1 else off), h.score, "o-", ms=2.5, lw=1.4, color=VARIANT_COLOR[v], label=v)
        ax.set_title(f"{t} (seed {a.seed_list[0]})", fontsize=8.5)
        ax.set_xlabel("epoch (F1 / F2 continue after F0)")
    axs[0, 0].set_ylabel("validation score")
    axs[0, -1].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(P.figures / "fig_training_curves.png", dpi=200)
    plt.close(fig)


def _reliability_fig(P, pred):
    if not len(pred) or "split" not in pred:
        return
    fin = pred[(pred.source_test == "Final") & (pred.train_on == "Joint") & (pred.variant == "F2")]
    if P.final.exists():
        try:
            info = json.loads((P.final_model / "model_card.json").read_text(encoding="utf-8"))
            fin = pred[pred.run_id == info.get("source_run")]
        except Exception:
            pass
    if not len(fin):
        return
    rid = sorted(fin.run_id.unique())[0]
    g = fin[fin.run_id == rid]
    v, t = g[g.split == "val"], g[g.split == "test"]
    if v.label.nunique() < 2:
        return
    T = float(g.temperature.iloc[0])
    iso = IsoCal().fit(v.logit.values, v.label.values)
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.2), sharey=True)
    for ax, cond, kind in zip(axs, ("original", "degraded"), ("clean", "deg")):
        x = t[t.kind == kind]
        for nm, p, col in (("temperature", sigmoid(x.logit.values / T), "#8a8984"), ("isotonic", iso(x.logit.values), "#eb6834")):
            bins = np.linspace(0, 1, 11)
            b = np.clip(np.digitize(p, bins) - 1, 0, 9)
            xs = [p[b == k].mean() for k in range(10) if (b == k).sum() >= 5]
            ys = [x.label.values[b == k].mean() for k in range(10) if (b == k).sum() >= 5]
            ax.plot(xs, ys, "o-", color=col, ms=3, lw=1.4, label=f"{nm} (ECE {ece_score(x.label.values, p):.3f})")
        ax.plot([0, 1], [0, 1], ":", color="#555", lw=0.8)
        ax.set_xlabel("predicted P(fake)")
        ax.set_title(f"{rid}: test {cond} videos", fontsize=8.5)
        ax.legend(frameon=False, fontsize=7)
    axs[0].set_ylabel("observed fake rate")
    fig.tight_layout()
    fig.savefig(P.figures / "fig_reliability_final.png", dpi=200)
    plt.close(fig)


def phase_analyze(a, P: Paths):
    pred = load_predictions(a, P)
    if not len(pred):
        print("  no predictions yet - run --phase evaluate")
        return
    print(f"  analysing {pred.run_id.nunique()} runs ({(pred.source_test == 'Final').sum():,} final-test prediction rows) ...")
    cams = load_cams(a, P)
    met, degm, rob, manip = per_run_tables(pred)
    cs = cam_summary(cams)
    fx = load_faith(P)
    fpv, fsum, fcmp = faith_tables(a, fx)
    cal, cal_sum, curves = calibration_tables(pred)
    ss = seed_summary(rob, cs)
    print(f"  paired cluster bootstrap ({a.n_boot} replicates) ...")
    stt, lev = bootstrap_stats(a, pred, cams)
    H = hypothesis_verdicts(stt, lev, cal, fcmp)
    for df, name in ((met, "metrics"), (degm, "degradation_metrics"), (rob, "robustness"), (manip, "per_manipulation"), (cs, "gradcam_summary"),
                     (fpv, "faithfulness_per_video"), (fsum, "faithfulness_summary"), (fcmp, "faithfulness_comparisons"), (cal, "calibration"),
                     (cal_sum, "calibration_summary"), (curves, "risk_coverage"), (ss, "seed_summary"), (stt, "statistics"), (lev, "levels")):
        if len(df):
            write_csv(df, P.metrics / f"{name}_final.csv")
    if len(cams):
        write_csv(cams, P.metrics / "gradcam_stability_final.csv")
    write_json(P.metrics / "hypotheses_final.json", H)
    make_figures(a, P, rob, cs, degm, manip, fsum, curves, pred)
    print(f"  metrics -> {P.metrics}; figures -> {P.figures}")


# =====================================================================================
# FINAL MODEL: selection on validation only, isotonic calibration + abstention band, model card
# =====================================================================================
def overlay(img_rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    hm = cv2.applyColorMap(np.uint8(255 * np.clip(cam, 0, 1)), cv2.COLORMAP_JET)[..., ::-1]
    return np.uint8(np.clip((1 - alpha) * img_rgb.astype(np.float32) + alpha * hm, 0, 255))


def verdict_of(p: float, thr: float, band: float, fallback: bool = False) -> str:
    """Decision rule of the final model: UNCERTAIN inside the band around the threshold or when no face was found."""
    if fallback or abs(float(p) - thr) < band:
        return "UNCERTAIN"
    return "FAKE" if float(p) >= thr else "REAL"


def save_explanation(path: Path, crops: np.ndarray, cam: np.ndarray, attn: np.ndarray, verdict: str, p: float, title: str,
                     note: str = "") -> None:
    """crops [T,S,S,3] RGB, cam [T,S,S] in 0..1, attn [T]. Object-oriented matplotlib (safe in the GUI worker thread)."""
    from matplotlib.figure import Figure
    T = len(crops)
    show = np.linspace(0, T - 1, 8).round().astype(int)
    fig = Figure(figsize=(15, 6.2), layout="constrained")
    gs = fig.add_gridspec(3, 8, height_ratios=[1, 1, 0.55])
    for k, t in enumerate(show):
        ax = fig.add_subplot(gs[0, k])
        ax.imshow(crops[t])
        ax.set_title(f"frame {t}", fontsize=8)
        ax.axis("off")
        ax = fig.add_subplot(gs[1, k])
        ax.imshow(overlay(crops[t], cam[t]))
        ax.set_title(f"Grad-CAM (attention {attn[t]:.2f})", fontsize=7)
        ax.axis("off")
    ax = fig.add_subplot(gs[2, :])
    top = set(np.argsort(-np.asarray(attn))[:2].tolist())
    ax.bar(range(T), attn, color=["#c0392b" if t in top else "#2a78d6" for t in range(T)])
    ax.set_xticks(range(T))
    ax.set_xlabel("frame")
    ax.set_ylabel("temporal attention")
    col = {"FAKE": "#c0392b", "REAL": "#1e8449"}.get(verdict, "#b7950b")
    fig.suptitle(f"{title}\n{verdict}   P(fake) = {p:.3f}" + (f"   ({note})" if note else ""), fontsize=11, color=col)
    fig.savefig(path, dpi=110)


def select_final(a, P: Paths) -> Optional[dict]:
    """Pre-registered rule (validation data only): Joint F2 unless its mean validation score is more than SELECT_TOL below
    Joint F1 (means over the seeds both variants finished); within the chosen variant the seed with the best validation score."""
    cands = []
    for v in [x for x in ("F2", "F1") if x in a.variant_list]:
        for seed in a.seed_list:
            ck = P.checkpoints / train_tag("Joint") / f"seed{seed}" / f"{v}.pt"
            if not ck.exists():
                continue
            try:
                obj = torch.load(guard(ck, "final_checkpoint"), map_location="cpu", weights_only=False)
            except Exception as e:
                print(f"  [freeze] {ck} unreadable ({type(e).__name__}) - ignored")
                continue
            if not obj.get("final"):
                continue
            cands.append(dict(variant=v, seed=int(seed), run_id=run_id("Joint", seed, v), ckpt=str(ck), val_score=float(obj["best_val_score"]),
                              best_epoch=int(obj["best_epoch"]), file_sha256=sha256_file(ck)))
    if not cands:
        return None
    C = pd.DataFrame(cands)
    common = sorted(set(C[C.variant == "F2"].seed) & set(C[C.variant == "F1"].seed))
    base = C[C.seed.isin(common)] if common else C
    means = base.groupby("variant").val_score.mean().to_dict()
    if "F2" in means and ("F1" not in means or means["F2"] >= means["F1"] - SELECT_TOL):
        var = "F2"
    elif "F1" in means:
        var = "F1"
    else:
        var = "F2"
    best = C[C.variant == var].sort_values(["val_score", "seed"], ascending=[False, True]).iloc[0]
    key = sha256_text(";".join(f"{r.run_id}:{r.file_sha256}" for r in C.sort_values("run_id").itertuples()))
    return dict(rule=f"Joint F2 unless its mean validation score is more than {SELECT_TOL} below Joint F1; best-validation seed of the "
                     "chosen variant (validation data only)", candidates=cands, mean_val_score=means, means_over_seeds=common or "all available",
                chosen_variant=var, chosen_seed=int(best.seed), chosen_run=str(best.run_id), ckpt=str(best.ckpt), file_sha256=str(best.file_sha256),
                val_score=float(best.val_score), candidates_key=key,
                complete=bool(len(C) == 2 * len(a.seed_list) and {"F1", "F2"} <= set(a.variant_list)),
                deviation="" if var == "F2" else "F2 fell more than the tolerance below F1 on validation, so F1 was frozen (pre-registered fallback)")


def final_model_xai_summary(P: Paths, variant: str) -> dict:
    """Grad-CAM stability and faithfulness of the chosen variant (Joint, seed means) from the analysis tables, if present."""
    out = {}
    f = P.metrics / "gradcam_summary_final.csv"
    if f.exists():
        cs = read_csv(f, "final_metrics")
        x = cs[(cs.train_on == "Joint") & (cs.variant == variant) & cs.scope.isin(["all", "hard"])]
        for (ds, sc), g in x.groupby(["test_dataset", "scope"]):
            out[f"gradcam_top20_iou_{sc}_{ds}"] = float(g.mean_top20_iou.mean())
            if sc == "all":
                out[f"explanation_changed_while_stable_pct_{ds}"] = float(g.explanation_changed_given_stable_pct.mean())
                out[f"clean_reference_iou_{ds}"] = float(g.mean_clean_reference_iou.mean())
    f = P.metrics / "faithfulness_summary_final.csv"
    if f.exists():
        fs = read_csv(f, "final_metrics")
        x = fs[(fs.train_on == "Joint") & (fs.variant == variant) & fs.metric.isin(["gc_ins_gain_grey", "gc_del_gain_grey"])]
        for r in x.itertuples():
            out[f"faith_{r.metric}_{r.version}_{r.pred_class}"] = dict(mean=float(r.mean), ci95=[float(r.ci95_low), float(r.ci95_high)], verdict=r.verdict)
    return out


def phase_freeze(a, P: Paths, D: FData) -> Optional[dict]:
    print("\n" + "=" * 100 + "\n PHASE FREEZE: select the final model on validation, isotonic calibration + abstention band, model card\n" + "=" * 100)
    link = require_link(P)
    sel = select_final(a, P)
    if sel is None:
        print("  no finished Joint F1/F2 run yet - nothing to freeze")
        return None
    rid, ck = sel["chosen_run"], Path(sel["ckpt"])
    if P.final.exists():
        try:
            old = torch.load(guard(P.final, "final_model"), map_location="cpu", weights_only=False)
            if (old.get("source_run") == rid and old.get("source_checkpoint_sha256") == sel["file_sha256"]
                    and old.get("selection", {}).get("candidates_key") == sel["candidates_key"]):
                print(f"  [resume] final model already frozen from {rid}")
                return {k: v for k, v in old.items() if k != "model_state_dict"}
            print(f"  re-freezing: the finished runs changed since the last freeze (now {len(sel['candidates'])} candidates)")
        except Exception as e:
            print(f"  [freeze] existing {P.final.name} unreadable ({type(e).__name__}) - re-freezing")
    pf = P.pred_dir / f"{rid}.csv.gz"
    if not pf.exists():
        phase_evaluate(a, P, D)
    pr = read_csv(pf, "final_predictions")
    val, test = pr[pr.split == "val"], pr[pr.split == "test"]
    tr = Trainer(a, P, D, "Joint", sel["chosen_seed"])
    model, obj = tr.load(ck)
    T, thr_T = float(obj["temperature"]), float(obj["threshold"])
    s1 = torch.load(guard(ck.parent / f"{STAGE1}.pt", "final_checkpoint"), map_location="cpu", weights_only=False) if (ck.parent / f"{STAGE1}.pt").exists() else {}
    pretrained = bool(s1.get("config", {}).get("init", {}).get("pretrained", False))
    z, y, ds = val.logit.values.astype(float), val.label.values.astype(int), val.dataset.values
    iso = IsoCal().fit(z, y)
    pv = iso(z)
    thr = choose_threshold(y, pv, ds)
    band = abstain_band(pv, thr, DEPLOY_COVERAGE)
    fb_of = dict(zip(D.key, D.info_meta[D.cidx, 4] > 0.5))                      # no face found in the source video
    val_keep = (np.abs(pv - thr) >= band) & ~val.source_key.map(fb_of).fillna(False).values.astype(bool)
    perf = []
    for tds in DATASETS:                                     # the frozen rule on the test split: reported, never tuned
        for cond, kind in (("original", "clean"), ("degraded", "deg")):
            x = test[(test.dataset == tds) & (test.kind == kind)]
            if not len(x):
                continue
            zt, yt = x.logit.values.astype(float), x.label.values.astype(int)
            pt = iso(zt)
            fb = x.source_key.map(fb_of).fillna(False).values.astype(bool)
            keep = (np.abs(pt - thr) >= band) & ~fb
            prd = (pt >= thr).astype(int)
            mk = keep if keep.any() else None
            perf.append(dict(test_dataset=tds, condition=cond, n=len(x), n_no_face=int(fb.sum()), auc=safe_auc(yt, zt),
                             balanced_accuracy_all=mean_bacc(yt, prd, x.dataset.values),
                             real_recognised=float((prd[yt == 0] == 0).mean()) if (yt == 0).any() else np.nan,
                             fake_detected=float((prd[yt == 1] == 1).mean()) if (yt == 1).any() else np.nan,
                             coverage=float(keep.mean()), balanced_accuracy_decided=mean_bacc(yt[mk], prd[mk], x.dataset.values[mk]) if mk is not None else np.nan,
                             ece_temperature=ece_score(yt, sigmoid(zt / T)), ece_isotonic=ece_score(yt, pt)))
    perf_df = pd.DataFrame(perf)
    m_tmp = FinalDetector(pretrained=False)
    sd = obj["model_state_dict"]
    final = dict(
        format="FINALTEST final model v1", model_state_dict=sd,
        architecture=dict(name="FinalDetector", parameters=n_params(m_tmp), architecture_signature=architecture_signature(m_tmp),
                          architecture_source_sha256=architecture_source_sha(), embedding=EMB_DIM, max_frames=NUM_FRAMES,
                          backbone="EfficientNet-B0 (ImageNet IMAGENET1K_V1 initialisation)" if pretrained else "EfficientNet-B0 (RANDOM initialisation - pipeline check only)"),
        input=dict(frames=NUM_FRAMES, frame_positions=list(range(NUM_FRAMES)), crop_px=CACHE_SIZE, input_px=a.img, colour="RGB uint8",
                   mean=MEAN.tolist(), std=STD.tolist(), resize="bilinear 128 -> input_px"),
        preprocessing=dict(code=f"{Path(a.t5_script).name}: extract_face_clip (imported at prediction time, never copied)",
                           preprocessing_code_sha256=link.get("preprocessing_code_sha256"), face_detector="OpenCV YuNet face_detection_yunet_2023mar.onnx",
                           face_margin=0.20, frame_side=360, face_px=80, jpeg=0,
                           degraded_versions="evaluation crops of degraded videos re-use the source video's face boxes"),
        calibration=dict(temperature=T, threshold_temperature=thr_T, isotonic_x=iso.x.tolist(), isotonic_y=iso.y.tolist(), threshold=thr,
                         abstain_band=band, deploy_coverage=DEPLOY_COVERAGE, validation_coverage=float(val_keep.mean()), n_validation=int(len(val)),
                         fitted_on="validation originals + degraded versions of both datasets (Test 5 validation split)",
                         rule="P = isotonic(logit); UNCERTAIN if |P - threshold| < band or no face found; else FAKE if P >= threshold else REAL"),
        source_run=rid, source_checkpoint=str(ck), source_checkpoint_sha256=sel["file_sha256"], weights_sha256=weights_sha256(sd),
        pretrained_backbone=pretrained, selection=sel, test_performance=perf, protocol_sha256=link.get("protocol_sha256"),
        split_hash=link.get("split_hash_originals"), split_manifest_sha256=link.get("split_manifest_sha256"), script_sha256=code_sha256(),
        smoke=bool(a.smoke), frozen_at=now())
    atomic_torch_save(final, P.final)
    card = {k: v for k, v in final.items() if k != "model_state_dict"}
    card["file_sha256"] = sha256_file(P.final)
    card["explanations"] = final_model_xai_summary(P, sel["chosen_variant"])
    write_json(P.final_model / "model_card.json", card)
    write_model_card(a, P, card, perf_df)
    try:
        freeze_examples(a, P, D, model, final, iso, tr.device)
    except Exception as e:                                      # examples are illustrations; never block the freeze
        print(f"  [freeze] example figure skipped ({type(e).__name__}: {e})")
    print(f"  FINAL MODEL = {rid} (validation score {sel['val_score']:.4f}; mean validation score per variant "
          + ", ".join(f"{k} {v:.4f}" for k, v in sel["mean_val_score"].items()) + ")")
    print(f"  isotonic threshold {thr:.3f} | uncertain band +-{band:.3f} (validation coverage {val_keep.mean():.1%}) | -> {P.final}")
    if len(perf_df):
        print(md_table(perf_df[["test_dataset", "condition", "n", "auc", "balanced_accuracy_all", "coverage", "balanced_accuracy_decided", "ece_isotonic"]]))
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return card


def write_model_card(a, P: Paths, card: dict, perf: pd.DataFrame):
    c, sel = card["calibration"], card["selection"]
    warn = []
    if card.get("smoke"):
        warn.append("**SMOKE-TEST MODEL (tiny data, 1 epoch): for checking the pipeline only - not a result.**")
    if not card.get("pretrained_backbone"):
        warn.append("**The backbone was randomly initialised (--allow-random-init): pipeline check only - not the final model.**")
    ex = card.get("explanations", {})
    xai_lines = [f"- {k.replace('_', ' ')}: {v:.3f}" for k, v in ex.items() if isinstance(v, float)]
    faith_lines = [f"- {k.replace('faith_', '').replace('_', ' ')}: {v['mean']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}] - {v['verdict']}"
                   for k, v in ex.items() if isinstance(v, dict)]
    pt = perf.rename(columns={"test_dataset": "test set", "balanced_accuracy_all": "bal. acc. (all)", "balanced_accuracy_decided": "bal. acc. (decided)",
                              "ece_isotonic": "ECE (isotonic)", "ece_temperature": "ECE (temperature)", "real_recognised": "real recognised",
                              "fake_detected": "fake detected", "n_no_face": "no face"}) if len(perf) else perf
    md = ["# Model card - final deepfake detector of the thesis", ""] + ([*warn, ""] if warn else []) + [
          f"Frozen {card['frozen_at']} by FINALTEST.py from run **{card['source_run']}** (selected on validation data only).", "",
          "## Model",
          f"- {card['architecture']['backbone']} per face frame -> 1-layer temporal transformer over the 16 frame positions -> attention pooling -> P(fake); "
          f"{card['architecture']['parameters']:,} parameters; video only (no audio).",
          f"- Variant {sel['chosen_variant']}: " + (VARIANTS[sel['chosen_variant']]["description"] if sel["chosen_variant"] in VARIANTS else ""),
          f"- Selection: {sel['rule']}. Mean validation score: " + ", ".join(f"{k} {v:.4f}" for k, v in sel["mean_val_score"].items())
          + f"; chosen seed {sel['chosen_seed']} (validation score {sel['val_score']:.4f})." + (f" {sel['deviation']}." if sel.get("deviation") else ""),
          "", "## Input and preprocessing",
          "- 16 frames uniformly spaced over the video; frames normalised to a 360 px short side; OpenCV YuNet face box (margin 0.20, 3-frame "
          "smoothing); every face is passed through an 80 px bottleneck (removes the resolution shortcut) and cropped to 128 px RGB; the network "
          f"sees {card['input']['input_px']} px (bilinear) with ImageNet normalisation.",
          f"- The preprocessing is the Test 5 code itself ({card['preprocessing']['code']}); its source hash "
          f"`{str(card['preprocessing']['preprocessing_code_sha256'])[:16]}...` is checked before every prediction.",
          "", "## Decision rule",
          f"- P(fake) = isotonic calibration of the logit (fitted on {c['n_validation']:,} validation videos: originals + degraded versions).",
          f"- **FAKE** if P >= {c['threshold']:.3f}, **REAL** otherwise; **UNCERTAIN** if |P - {c['threshold']:.3f}| < {c['abstain_band']:.3f} "
          f"(band tuned for {c['deploy_coverage']:.0%} coverage on validation; realised validation coverage {c['validation_coverage']:.1%}) or if no face is found.",
          f"- Temperature scaling (T = {c['temperature']:.2f}, threshold {c['threshold_temperature']:.2f}) is kept for reference.",
          "", "## Test performance (Test 5 test split; never used for training, selection or calibration)",
          md_table(pt, 3) if len(pt) else "_(not available)_",
          "", "AUC is the ranking quality of the logit; 'decided' = videos outside the uncertain band (coverage = share decided).",
          "", "## Explanations",
          "- Grad-CAM of the last EfficientNet block for the predicted class (evidence for FAKE or for REAL), up-sampled to the face crop; "
          "temporal attention shows which frames carried the decision.",
          "- Grad-CAM stability between original and degraded versions (Joint, seed mean of the chosen variant):", *(xai_lines or ["- (analysis not available)"]),
          "- Faithfulness (insertion / deletion gain over a random map; > 0 = the map points at evidence the model uses):", *(faith_lines or ["- (not measured)"]),
          "- A stable or faithful explanation is not proof that the decision is correct.",
          "", "## Intended use and limits",
          "- Research and decision support: screening face videos similar to BioDeepAV and FaceForensics++ (C23), including the degradation types "
          "of the thesis (blur, brightness, contrast, saturation, noise, compression, motion blur, frame drop, fps reduction, low resolution; three severities).",
          "- Not a forensic or legal proof; not for identity decisions; not validated on other generators, other datasets, non-face content or audio deepfakes.",
          "- Two datasets only; dataset-specific cues may remain (see the cross-dataset results of the final test); small faces (< 80 px) and videos "
          "without a detectable face are less reliable (the latter are always UNCERTAIN).",
          "- Calibration and the uncertain band hold for the validation distribution; on new data, re-check calibration before trusting probabilities.",
          "", "## Reproducibility",
          f"- final model file SHA-256 `{card.get('file_sha256', '')}`; weights SHA-256 `{card['weights_sha256']}`",
          f"- source checkpoint `{card['source_checkpoint']}` (SHA-256 `{card['source_checkpoint_sha256']}`)",
          f"- protocol SHA-256 `{card['protocol_sha256']}`; split hash `{card['split_hash']}`; script SHA-256 `{card['script_sha256']}`",
          "", "## Use",
          "```", 'python FINALTEST.py --predict "D:\\video.mp4"        # or a folder; add --overlay-video for an MP4 with the heat-map',
          "python FINALTEST.py --phase gui", "```",
          "Outputs: run_FINALTEST/predict/<video>_<time>/verdict.json + explanation.png (+ explanation_overlay.mp4)."]
    (P.final_model / "MODEL_CARD.md").write_text("\n".join(md), encoding="utf-8")


def freeze_examples(a, P: Paths, D: FData, model, final: dict, iso: IsoCal, dev):
    """One original + degraded test pair per dataset and class with the final decision rule and the Grad-CAM of its decision."""
    c = final["calibration"]
    thr, band = c["threshold"], c["abstain_band"]
    pairs = load_xai_pairs(a, D)
    chosen = []
    for ds in DATASETS:
        for lab in (0, 1):
            m = [p for p in pairs if p[2].dataset == ds and int(p[2].label) == lab and p[2].degradation in HARD_TYPES and p[2].severity == "severe"]
            m = m or [p for p in pairs if p[2].dataset == ds and int(p[2].label) == lab]
            if m:
                chosen.append(m[0])
    if not chosen:
        return
    show = np.linspace(0, NUM_FRAMES - 1, 5).round().astype(int)
    fig, axes = plt.subplots(2 * len(chosen), len(show) + 1, figsize=(12.5, 2.25 * 2 * len(chosen)),
                             gridspec_kw=dict(width_ratios=[1] * len(show) + [1.5]))
    axes = np.atleast_2d(axes)
    tix = frame_positions(1, np.arange(NUM_FRAMES), dev)
    for k, (s, j, x) in enumerate(chosen):
        fb = bool(D.info_meta[D.cidx[s], 4] > 0.5)
        for vi, (name, clip) in enumerate((("original", D.clip("clean", s)), (f"{x.degradation}/{x.severity}", D.clip("deg", j)))):
            xin = to_input(clip[None], dev, a.img)
            with torch.no_grad():
                with torch.autocast("cuda", dtype=torch.float16, enabled=a.amp):
                    lg = float(model(xin, tix)["logit"].float().cpu().numpy()[0])
            p = float(iso(np.array([lg]))[0])
            vd = verdict_of(p, thr, band, fb)
            g = grad_cam_eval(model, xin, tix, sign=1.0 if p >= thr else -1.0, amp=a.amp)
            row = 2 * k + vi
            for q, t in enumerate(show):
                ax = axes[row, q]
                ax.imshow(overlay(clip[t], g["cam"][0][t]))
                ax.axis("off")
                if q == 0:
                    ax.set_title(f"{x.dataset} {'fake' if int(x.label) else 'real'} - {name}", fontsize=7, loc="left")
            ax = axes[row, -1]
            ax.bar(range(NUM_FRAMES), g["attn"][0], color="#2a78d6")
            ax.set_xticks([])
            ax.set_yticks([])
            col = {"FAKE": "#c0392b", "REAL": "#1e8449"}.get(vd, "#b7950b")
            ax.set_title(f"{vd}  P(fake) {p:.2f}", fontsize=8, color=col)
    fig.suptitle(f"Final model {final['source_run']}: Grad-CAM of the decision (evidence for the predicted class) and temporal attention\n"
                 "rows: original and degraded version of the same test video (true class in the row title)", fontsize=9)
    fig.tight_layout()
    fig.savefig(P.final_model / "final_model_examples.png", dpi=120)
    plt.close(fig)


def load_final(P: Paths, device) -> Tuple[FinalDetector, dict, IsoCal]:
    if not P.final.exists():
        raise SystemExit(f"[STOP] no final model at {P.final} - run the final test first (python FINALTEST.py) or --phase freeze")
    obj = torch.load(guard(P.final, "final_model"), map_location=device, weights_only=False)
    m = FinalDetector(pretrained=False).to(device)
    m.load_state_dict(obj["model_state_dict"])
    m.eval()
    c = obj["calibration"]
    return m, obj, IsoCal(c["isotonic_x"], c["isotonic_y"])


# =====================================================================================
# PREDICTION ON ANY VIDEO (Test 5 preprocessing imported -> identical face crops to the evaluation caches)
# =====================================================================================
_T5MOD: Dict[str, Any] = dict(mod=None, det=None)


def load_t5_module(a, expected_sha: Optional[str]) -> Tuple[Any, str, bool]:
    """Import the Test 5 script as a module (its main() does not run) and compare its preprocessing source hash with the
    hash recorded in the caches the final model was trained and evaluated on."""
    if _T5MOD["mod"] is None:
        path = Path(a.t5_script)
        if not path.exists():
            raise SystemExit(f"[STOP] {path} is needed for the face-crop preprocessing (use --t5-script)")
        spec = importlib.util.spec_from_file_location("t5_preprocessing", str(path))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod                       # inspect.getsource() of its classes needs the module registered
        try:
            spec.loader.exec_module(mod)
        except BaseException:
            sys.modules.pop(spec.name, None)
            raise
        _T5MOD["mod"] = mod
    mod = _T5MOD["mod"]
    sha = mod.preprocessing_code_sha()
    return mod, sha, (expected_sha is None or sha == expected_sha)


def t5_detector(mod):
    if _T5MOD["det"] is None:
        _T5MOD["det"] = mod.face_detector()
    return _T5MOD["det"]


def model_logit(model, clip: np.ndarray, dev, a, img: int) -> float:
    with torch.no_grad():
        with torch.autocast("cuda", dtype=torch.float16, enabled=bool(a.amp and dev.type == "cuda")):
            return float(model(to_input(clip[None], dev, img), frame_positions(1, np.arange(NUM_FRAMES), dev))["logit"].float().cpu().numpy()[0])


def write_overlay_video(mod, path: str, info: dict, cam: np.ndarray, verdict: str, p: float, out_path: Path) -> bool:
    """Heat-map projected back onto the (resolution-normalised) frames, using the same crop geometry as crop_face()."""
    frames, _ = mod.read_sampled_frames(str(path), NUM_FRAMES)
    if frames is None:
        return False
    side = mod.NORM.get("frame_side")
    if side:
        frames = [mod.normalize_frame_resolution(f, side) for f in frames]
    H, W = frames[0].shape[:2]
    vw = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), 4, (W, H))
    col = (0, 0, 255) if verdict == "FAKE" else ((0, 160, 0) if verdict == "REAL" else (0, 200, 255))
    for t, f in enumerate(frames[:len(cam)]):
        f2 = f.copy()
        if info.get("fallback"):
            x0, y0, w, h = 0, 0, W, H
        else:
            b = info["boxes"][t]
            s_ = float(np.clip(b[2] * min(W, H), 8, min(W, H)))
            cx = float(np.clip(b[0] * W, s_ / 2, W - s_ / 2))
            cy = float(np.clip(b[1] * H, s_ / 2, H - s_ / 2))
            s = int(round(s_))
            x0, y0 = int(round(cx - s_ / 2)), int(round(cy - s_ / 2))
            x0, y0 = max(0, min(x0, W - s)), max(0, min(y0, H - s))
            w = h = s
        hm = cv2.resize(cam[t].astype(np.float32), (w, h))
        sub = f2[y0:y0 + h, x0:x0 + w]
        cm = cv2.applyColorMap(np.uint8(255 * np.clip(hm[:sub.shape[0], :sub.shape[1]], 0, 1)), cv2.COLORMAP_JET)
        f2[y0:y0 + sub.shape[0], x0:x0 + sub.shape[1]] = np.uint8(0.55 * sub + 0.45 * cm)
        cv2.rectangle(f2, (x0, y0), (x0 + w - 1, y0 + h - 1), col, 2)
        cv2.putText(f2, f"{verdict}  P(fake)={p:.2f}  frame {t}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
        vw.write(f2)
    vw.release()
    return True


def predict_video(a, P: Paths, path: str, model, obj: dict, iso: IsoCal, dev, mod, det, save: bool = True) -> dict:
    c = obj["calibration"]
    thr, band, T = float(c["threshold"]), float(c["abstain_band"]), float(c["temperature"])
    img = int(obj["input"]["input_px"])
    clip, info = mod.extract_face_clip(str(path), NUM_FRAMES, CACHE_SIZE, det, mod.FACE_MARGIN, mod.NORM)
    if clip is None:
        return dict(video=str(path), verdict="ERROR", error=info.get("error") or "the video could not be decoded", analysed_at=now())
    lg = model_logit(model, clip, dev, a, img)
    p = float(iso(np.array([lg]))[0])
    fallback = bool(info.get("fallback"))
    verdict = verdict_of(p, thr, band, fallback)
    pred = int(p >= thr)
    notes = []
    if fallback:
        notes.append("no face was found - the whole frame was analysed; the verdict is withheld (UNCERTAIN)")
    elif np.isfinite(info.get("detection_rate", np.nan)) and info["detection_rate"] < 0.5:
        notes.append(f"a face was found in only {100 * info['detection_rate']:.0f} % of the sampled frames")
    if not fallback and abs(p - thr) < band:
        notes.append(f"P(fake) is within +-{band:.2f} of the threshold {thr:.2f}: too close to call (UNCERTAIN)")
    nfp = float(info.get("face_rel_size", np.nan)) * min(float(info.get("width") or 0), float(info.get("height") or 0))
    if np.isfinite(nfp) and 0 < nfp < 80:
        notes.append(f"small face (about {nfp:.0f} px in the video): less detail than the 80 px the detector was trained on")
    res = dict(video=str(path), verdict=verdict, predicted="FAKE" if pred else "REAL", p_fake=round(p, 4),
               confidence=round(p if pred else 1 - p, 4), p_fake_temperature=round(float(sigmoid(lg / T)), 4), logit=round(lg, 4),
               threshold=round(thr, 4), uncertain_band=round(band, 4), face_found=not fallback,
               face_detection_rate=round(float(info.get("detection_rate", np.nan)), 3) if np.isfinite(info.get("detection_rate", np.nan)) else None,
               native_face_px=round(nfp, 1) if np.isfinite(nfp) else None, notes=notes, model=obj.get("source_run"),
               model_sha256=obj.get("weights_sha256"), analysed_at=now())
    if save:
        stem = "".join(ch if (ch.isalnum() or ch in "-_.") else "_" for ch in Path(path).stem)[:60] or "video"
        out = P.predict / f"{stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        k = 1
        while out.exists():
            out = P.predict / f"{stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{k}"
            k += 1
        out.mkdir(parents=True, exist_ok=True)
        x = to_input(clip[None], dev, img)
        g = grad_cam_eval(model, x, frame_positions(1, np.arange(NUM_FRAMES), dev), sign=1.0 if pred else -1.0,
                          amp=bool(a.amp and dev.type == "cuda"))
        save_explanation(out / "explanation.png", clip, g["cam"][0], g["attn"][0], verdict, p, Path(path).name,
                         note=f"Grad-CAM = evidence for {'FAKE' if pred else 'REAL'}")
        if a.overlay_video:
            try:
                if write_overlay_video(mod, path, info, g["cam"][0], verdict, p, out / "explanation_overlay.mp4"):
                    res["overlay_video"] = str(out / "explanation_overlay.mp4")
            except Exception as e:
                notes.append(f"overlay video not written ({type(e).__name__})")
        res.update(top_frames=[int(i) for i in np.argsort(-g["attn"][0])[:3]], output_folder=str(out))
        write_json(out / "verdict.json", res)
    return res


def prepare_predictor(a, P: Paths):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, obj, iso = load_final(P, dev)
    exp = obj.get("preprocessing", {}).get("preprocessing_code_sha256")
    mod, sha, ok = load_t5_module(a, exp)
    if not ok:
        msg = (f"the Test 5 preprocessing code changed (hash {sha[:12]} vs {str(exp)[:12]} of the training caches): crops would differ "
               "from what the model was trained and calibrated on")
        if not a.force:
            raise SystemExit(f"[STOP] {msg}. Restore the original Test 5 script, or add --force to predict anyway.")
        print(f"  [WARN] {msg} (--force)")
    det = t5_detector(mod)
    if obj.get("smoke") or not obj.get("pretrained_backbone", True):
        print("  [WARN] this final model comes from a smoke test / random initialisation - for checking the pipeline only")
    return dev, model, obj, iso, mod, det


def phase_predict(a, P: Paths):
    tgt = Path(a.predict) if a.predict else None
    if tgt is None or not tgt.exists():
        raise SystemExit(f"[STOP] video or folder not found: {a.predict}")
    vids = sorted(p for p in tgt.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXT) if tgt.is_dir() else [tgt]
    if not vids:
        raise SystemExit(f"[STOP] no video found in {tgt}")
    dev, model, obj, iso, mod, det = prepare_predictor(a, P)
    c = obj["calibration"]
    print(f"  final model {obj.get('source_run')} | threshold {c['threshold']:.3f} | uncertain band +-{c['abstain_band']:.3f} | "
          f"{len(vids)} video(s) | {dev}")
    rows = []
    for v in vids:
        r = predict_video(a, P, str(v), model, obj, iso, dev, mod, det, save=True)
        rows.append(r)
        print(f"\n  {Path(v).name}\n    ==> {r['verdict']}" + (f"  (P(fake) = {r['p_fake']:.3f})" if "p_fake" in r else f"  {r.get('error', '')}"))
        for n in r.get("notes", []):
            print(f"    note: {n}")
        if r.get("output_folder"):
            print(f"    explanation: {r['output_folder']}")
    if len(rows) > 1:
        out = P.predict / f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df = pd.DataFrame(rows)
        if "notes" in df:
            df["notes"] = df["notes"].map(lambda x: " | ".join(x) if isinstance(x, list) else x)
        write_csv(df, out)
        print(f"\n  {pd.Series([r['verdict'] for r in rows]).value_counts().to_dict()} | summary CSV: {out}")


def phase_gui(a, P: Paths):
    try:
        import tkinter as tk
        from tkinter import filedialog, ttk
    except ImportError:
        raise SystemExit("[STOP] tkinter is not available in this Python - use --predict instead")
    dev, model, obj, iso, mod, det = prepare_predictor(a, P)
    c = obj["calibration"]
    root = tk.Tk()
    root.title("Final deepfake detector of the thesis - REAL / FAKE / UNCERTAIN")
    top = ttk.Frame(root, padding=8)
    top.pack(fill="x")
    lab = ttk.Label(top, text="Open a video to analyse", font=("Segoe UI", 15, "bold"))
    lab.pack(side="left", padx=8)
    sub = ttk.Label(root, text=f"model {obj.get('source_run')} | threshold {c['threshold']:.2f} | uncertain band +-{c['abstain_band']:.2f} "
                               f"({c['deploy_coverage']:.0%} coverage on validation) | Grad-CAM shows the evidence for the predicted class",
                    foreground="#555555")
    sub.pack(fill="x", padx=16)
    note_lab = ttk.Label(root, text="", foreground="#7a5c00")
    note_lab.pack(fill="x", padx=16)
    img_lab = ttk.Label(root)
    img_lab.pack(fill="both", expand=True)
    state: Dict[str, Any] = dict(thread=None, result=None, error=None)

    def show(png: Path):
        try:
            from PIL import Image, ImageTk
            im = Image.open(png)
            im.thumbnail((1400, 660))
            tkim = ImageTk.PhotoImage(im)
        except ImportError:
            tkim = tk.PhotoImage(file=str(png))
            if tkim.width() > 1400:
                tkim = tkim.subsample(2, 2)
        img_lab.config(image=tkim)
        img_lab.image = tkim

    def poll():
        th = state["thread"]
        if th is not None and th.is_alive():
            root.after(200, poll)
            return
        btn.state(["!disabled"])
        if state["error"]:
            lab.config(text=f"ERROR: {state['error']}", foreground="#b7950b")
            return
        r = state["result"]
        if r["verdict"] == "ERROR":
            lab.config(text=f"ERROR: {r.get('error')}", foreground="#b7950b")
            return
        col = {"FAKE": "#c0392b", "REAL": "#1e8449"}.get(r["verdict"], "#b7950b")
        lab.config(text=f"{r['verdict']}   P(fake) = {r['p_fake']:.3f}   ({Path(r['video']).name})", foreground=col)
        note_lab.config(text="   ".join(r.get("notes", [])))
        show(Path(r["output_folder"]) / "explanation.png")

    def work(f):
        try:
            state["result"] = predict_video(a, P, f, model, obj, iso, dev, mod, det, save=True)
        except Exception as e:
            state["error"] = f"{type(e).__name__}: {e}"

    def run():
        f = filedialog.askopenfilename(filetypes=[("Videos", " ".join(f"*{e}" for e in sorted(VIDEO_EXT))), ("All files", "*.*")])
        if not f:
            return
        lab.config(text=f"analysing {Path(f).name} ...", foreground="#333333")
        note_lab.config(text="")
        btn.state(["disabled"])
        state.update(result=None, error=None, thread=threading.Thread(target=work, args=(f,), daemon=True))
        state["thread"].start()
        root.after(200, poll)

    btn = ttk.Button(top, text="Open video ...", command=run)
    btn.pack(side="right")
    root.geometry("1450x840")
    root.mainloop()


# =====================================================================================
# PREDICTION-PIPELINE CHECK: raw video -> Test 5 preprocessing -> final model == cached evaluation crop
# =====================================================================================
def phase_predictcheck(a, P: Paths, D: FData) -> Optional[dict]:
    print("\n" + "=" * 100 + "\n PHASE PREDICTCHECK: --predict on raw test videos vs the cached evaluation crops\n" + "=" * 100)
    if not P.final.exists():
        print("  no final model yet - run --phase freeze first")
        return None
    out_json = P.reports / "predict_pipeline_check.json"
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, obj, iso = load_final(P, dev)
    c = obj["calibration"]
    thr, band, img = float(c["threshold"]), float(c["abstain_band"]), int(obj["input"]["input_px"])
    exp = obj.get("preprocessing", {}).get("preprocessing_code_sha256")
    try:
        mod, sha, sha_ok = load_t5_module(a, exp)
        det = t5_detector(mod)
    except BaseException as e:                                   # SystemExit included: the check reports instead of stopping the run
        rep = dict(status="FAIL", reason=f"Test 5 preprocessing could not be loaded: {type(e).__name__}: {e}", checked_at=now())
        write_json(out_json, rep)
        print(f"  [FAIL] {rep['reason']}")
        return rep
    rng = np.random.default_rng(FAITH_SEED)
    te = D.idx("test")
    per = max(1, int(math.ceil(a.predictcheck_n / 4)))
    picks, missing = [], 0
    for ds in DATASETS:
        for lab in (0, 1):
            cand = te[(D.ds[te] == ds) & (D.labels[te] == lab)]
            got = 0
            for i in rng.permutation(cand):
                if got >= per:
                    break
                if Path(D.path[i]).exists():
                    picks.append(int(i))
                    got += 1
                else:
                    missing += 1
                    if missing > 50 and not picks:
                        break
    rows = []
    t0 = time.time()
    for n_, i in enumerate(picks):
        ci = int(D.cidx[i])
        fb_src = bool(D.info_meta[ci, 4] > 0.5)
        jobs = [("original", D.path[i], D.clip("clean", i), None, None)]
        vs = D.versions.get(int(i))
        if vs is not None and len(vs):
            j = int(vs[0])
            jobs.append(("degraded", D.deg_path[j], D.clip("deg", j), (D.info_boxes[ci].astype(np.float32), fb_src), j))
        for version, path, cached, ref, j in jobs:
            base = dict(item_id=D.key[i] if version == "original" else D.deg_item[j], source_key=D.key[i], dataset=D.ds[i], label=int(D.labels[i]),
                        version=version, degradation="original" if version == "original" else D.deg_type[j], path=str(path))
            if not Path(path).exists():
                rows.append(dict(base, status="video file not found"))
                continue
            if ref is None:
                new, info = mod.extract_face_clip(str(path), NUM_FRAMES, CACHE_SIZE, det, mod.FACE_MARGIN, mod.NORM)
            else:
                new, info = mod.extract_face_clip(str(path), NUM_FRAMES, CACHE_SIZE, None, mod.FACE_MARGIN, mod.NORM, ref=ref)
            if new is None or new.shape != cached.shape:
                rows.append(dict(base, status=f"extraction failed: {info.get('error', '')}"[:120]))
                continue
            zc, zn = model_logit(model, cached, dev, a, img), model_logit(model, new, dev, a, img)
            pc, pn = float(iso(np.array([zc]))[0]), float(iso(np.array([zn]))[0])
            fb_new = bool(info.get("fallback")) if ref is None else fb_src
            rows.append(dict(base, status="ok", pixel_mae=float(np.abs(new.astype(np.int16) - cached.astype(np.int16)).mean()),
                             identical_fraction=float((new == cached).mean()), logit_cache=zc, logit_reextracted=zn, p_cache=pc,
                             p_reextracted=pn, abs_dp=abs(pn - pc), verdict_cache=verdict_of(pc, thr, band, fb_src),
                             verdict_reextracted=verdict_of(pn, thr, band, fb_new)))
        sys.stdout.write(f"\r  {progress_bar(n_ + 1, len(picks), 16)} {n_ + 1}/{len(picks)} source videos | {fmt_time(time.time() - t0)}   ")
        sys.stdout.flush()
    sys.stdout.write("\n")
    df = pd.DataFrame(rows)
    if len(df):
        write_csv(df, P.metrics / "predict_pipeline_check.csv")
    ok = df[df.status == "ok"] if len(df) else df
    crit = "Test 5 preprocessing hash unchanged, median pixel MAE <= 1.0 (0-255 scale) and max |delta P(fake)| <= 0.05"
    if not len(ok):
        n_fail = int(df.status.str.startswith("extraction failed").sum()) if len(df) else 0
        status = ("FAIL (preprocessing code changed)" if not sha_ok else
                  "FAIL (videos found but could not be re-extracted)" if n_fail else "SKIPPED (source videos not found on this computer)")
        summ = {}
    else:
        summ = dict(n_ok=int(len(ok)), pixel_mae_median=float(ok.pixel_mae.median()), pixel_mae_max=float(ok.pixel_mae.max()),
                    identical_fraction_mean=float(ok.identical_fraction.mean()), abs_dp_mean=float(ok.abs_dp.mean()), abs_dp_max=float(ok.abs_dp.max()),
                    verdict_agreement=float((ok.verdict_cache == ok.verdict_reextracted).mean()),
                    by_version={v: dict(n=int(len(g)), pixel_mae_median=float(g.pixel_mae.median()), abs_dp_max=float(g.abs_dp.max()))
                                for v, g in ok.groupby("version")})
        consistent = sha_ok and summ["pixel_mae_median"] <= 1.0 and summ["abs_dp_max"] <= 0.05
        status = "PASS" if consistent else ("FAIL" if not sha_ok else "WARN")
    rep = dict(status=status, checked_at=now(), criteria=crit, preprocessing_code_sha256_expected=exp, preprocessing_code_sha256_found=sha,
               preprocessing_hash_match=bool(sha_ok), n_source_videos=len(picks), n_rows=int(len(df)),
               n_missing=int((df.status == "video file not found").sum()) if len(df) else 0,
               n_failed=int(df.status.str.startswith("extraction failed").sum()) if len(df) else 0, **summ,
               final_model=obj.get("source_run"),
               note="Originals are re-extracted with fresh YuNet detection; degraded videos with their source's face boxes (as in the Test 5 cache). "
                    "PASS means --predict on a raw video reproduces the evaluation input and probability.")
    write_json(out_json, rep)
    print(f"  prediction-pipeline check: {status}" + (f" | pixel MAE median {summ['pixel_mae_median']:.3f} | max |dP| {summ['abs_dp_max']:.4f} | "
                                                     f"verdict agreement {summ['verdict_agreement']:.0%}" if summ else ""))
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return rep


# =====================================================================================
# REPORT
# =====================================================================================
TRAIN_ORDER = {"Joint": 0, "BioDeepAV": 1, "FF++": 2}
VARIANT_RANK = {v: i for i, v in enumerate(VARIANT_ORDER)}
VERDICT_TAG = {"supported improvement": "S+", "supported deterioration": "S-", "inconclusive": "ns",
               "inconclusive (too few clusters)": "n/a", "single seed: CI excludes 0": "1s*", "single seed: inconclusive": "1s"}


def _sort(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or not len(df):
        return df
    d = df.copy()
    keys = []
    if "train_on" in d:
        d["_t"] = d.train_on.map(TRAIN_ORDER).fillna(9)
        keys.append("_t")
    if "test_dataset" in d:
        keys.append("test_dataset")
    if "variant" in d:
        d["_v"] = d.variant.map(VARIANT_RANK).fillna(99)
        keys.append("_v")
    return d.sort_values(keys, kind="stable").drop(columns=[c for c in ("_t", "_v") if c in d]).reset_index(drop=True) if keys else d


def _ms(v: pd.Series, nd: int = 3, scale: float = 1.0) -> str:
    v = pd.to_numeric(v, errors="coerce").dropna() * scale
    if not len(v):
        return "n/a"
    return f"{v.mean():.{nd}f} +- {v.std(ddof=1):.{nd}f}" if len(v) > 1 else f"{v.mean():.{nd}f}"


def _load_metric(P: Paths, name: str) -> pd.DataFrame:
    f = P.metrics / f"{name}_final.csv"
    return read_csv(f, "final_metrics") if f.exists() else pd.DataFrame()


def _model_name(v: str) -> str:
    return {"F0": "F0 (Stage 1)", "F1": "F1 (degraded)", "F2": "F2 (final system)"}.get(v, v)


def detection_table(rob: pd.DataFrame) -> pd.DataFrame:
    spec = [("clean_auc", "AUC original", 3, 1), ("degraded_auc_pooled", "AUC degraded", 3, 1), ("worst_type_auc", "worst-type AUC", 3, 1),
            ("prediction_flip_rate_pct", "flips %", 1, 1), ("real_recognised", "real recognised", 3, 1), ("fake_detected", "fake detected", 3, 1),
            ("ece_clean", "ECE original", 3, 1), ("ece_degraded", "ECE degraded", 3, 1)]
    rows = []
    for (t, ds, st, v), g in rob.groupby(["train_on", "test_dataset", "setting", "variant"]):
        r = dict(train_on=t, test_dataset=ds, setting=st, variant=v, seeds=int(g.seed.nunique()))
        for m, lab, nd, sc in spec:
            r[lab] = _ms(g[m], nd, sc) if m in g else "n/a"
        wt = g.worst_type.dropna()
        r["worst type"] = wt.mode().iloc[0] if len(wt) else ""
        rows.append(r)
    d = _sort(pd.DataFrame(rows))
    if len(d):
        d["variant"] = d.variant.map(_model_name)
    return d.rename(columns={"train_on": "training", "test_dataset": "test", "variant": "model"})


def type_table(degm: pd.DataFrame) -> pd.DataFrame:
    x = degm[(degm.severity == "all") & degm.setting.isin(["in-domain", "joint"])]
    if not len(x):
        return pd.DataFrame()
    p = x.pivot_table(index=["train_on", "test_dataset", "variant"], columns="degradation", values="auc", aggfunc="mean").reset_index()
    p = _sort(p)
    p["variant"] = p.variant.map(_model_name)
    return p.rename(columns={"train_on": "training", "test_dataset": "test", "variant": "model"})


def manip_table(manip: pd.DataFrame) -> pd.DataFrame:
    x = manip[manip.setting.isin(["in-domain", "joint"])]
    if not len(x):
        return pd.DataFrame()
    x = x.assign(value=np.where(x.subset == "original (real)", x.specificity, x.recall))
    p = x.pivot_table(index=["train_on", "variant", "condition"], columns="subset", values="value", aggfunc="mean").reset_index()
    p = _sort(p)
    p["variant"] = p.variant.map(_model_name)
    return p.rename(columns={"train_on": "training", "variant": "model", "condition": "videos"})


def calibration_report_tables(cs: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not len(cs):
        return pd.DataFrame(), pd.DataFrame()
    x = cs[cs.setting.isin(["in-domain", "joint"])]
    e = x[x.target_coverage == 1.0].pivot_table(index=["train_on", "test_dataset", "variant", "condition"], columns="calibration",
                                                 values="ece", aggfunc="mean").reset_index()
    e = _sort(e)
    iso = x[x.calibration == "isotonic"]
    b = iso.pivot_table(index=["train_on", "test_dataset", "variant", "condition"], columns="target_coverage", values="bacc_decided",
                        aggfunc="mean").reset_index()
    cov = iso[iso.target_coverage == DEPLOY_COVERAGE].groupby(["train_on", "test_dataset", "variant", "condition"]).coverage.mean().rename(
        f"realised coverage @{DEPLOY_COVERAGE:.0%}").reset_index()
    b = _sort(b.merge(cov, on=["train_on", "test_dataset", "variant", "condition"], how="left"))
    b.columns = [f"bal. acc. @{c:.0%}" if isinstance(c, float) else c for c in b.columns]
    for d in (e, b):
        d["variant"] = d.variant.map(_model_name)
    ren = {"train_on": "training", "test_dataset": "test", "variant": "model", "condition": "videos", "raw": "ECE raw",
           "temperature": "ECE temperature", "isotonic": "ECE isotonic"}
    return e.rename(columns=ren), b.rename(columns=ren)


def gradcam_table(cs: pd.DataFrame) -> pd.DataFrame:
    if not len(cs):
        return pd.DataFrame()
    rows = []
    for (t, ds, st, v), g in cs[cs.scope.isin(["all", "hard"])].groupby(["train_on", "test_dataset", "setting", "variant"]):
        ga, gh = g[g.scope == "all"], g[g.scope == "hard"]
        rows.append(dict(train_on=t, test_dataset=ds, setting=st, variant=v, seeds=int(ga.seed.nunique()),
                         **{"IoU all": _ms(ga.mean_top20_iou), "IoU hard": _ms(gh.mean_top20_iou), "Pearson r": _ms(ga.mean_pearson_r),
                            "changed while stable %": _ms(ga.explanation_changed_given_stable_pct, 1),
                            "clean-reference IoU": _ms(ga.mean_clean_reference_iou), "relative IoU": _ms(ga.relative_iou),
                            "Grad-CAM++ IoU": _ms(ga.mean_top20_iou_pp) if "mean_top20_iou_pp" in ga else "n/a",
                            "attention r": _ms(ga.mean_attention_r) if "mean_attention_r" in ga else "n/a"}))
    d = _sort(pd.DataFrame(rows))
    d["variant"] = d.variant.map(_model_name)
    return d.rename(columns={"train_on": "training", "test_dataset": "test", "variant": "model"})


def faith_report_tables(fsum: pd.DataFrame, fcmp: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    s, c = pd.DataFrame(), pd.DataFrame()
    if len(fsum):
        x = fsum[fsum.metric.isin(["gc_ins_gain_grey", "gc_del_gain_grey", "gcpp_ins_gain_grey", "gcpp_del_gain_grey"])].copy()
        x["cell"] = [f"{m:+.3f} [{lo:+.3f}, {hi:+.3f}]" + (" *" if str(vd).startswith("faithful") else "")
                     for m, lo, hi, vd in zip(x["mean"], x.ci95_low, x.ci95_high, x.verdict)]
        s = x.pivot_table(index=["train_on", "variant", "version", "metric"], columns="pred_class", values="cell", aggfunc="first").reset_index()
        s = _sort(s)
        s["variant"] = s.variant.map(_model_name)
        s = s.rename(columns={"train_on": "training", "variant": "model", "version": "videos", "ALL": "all decisions",
                              "FAKE": "predicted FAKE", "REAL": "predicted REAL"})
    if len(fcmp):
        c = fcmp[fcmp.metric.isin(["gc_ins_gain_grey", "gc_del_gain_grey", "gcpp_ins_gain_grey"])][
            ["comparison", "train_on", "variant", "version", "metric", "n_pairs", "mean_diff", "ci95_low", "ci95_high", "verdict"]]
        c = c.rename(columns={"train_on": "training", "variant": "model", "version": "videos"})
    return s, c


def stats_tables(stt: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not len(stt):
        return pd.DataFrame(), pd.DataFrame()
    pr = stt[stt.primary] if "primary" in stt else stt.iloc[0:0]
    pr = pr[["comparison", "train_on", "test_dataset", "metric", "n_seeds", "mean_diff", "ci95_low", "ci95_high", "p_boot", "p_holm",
             "sign_agreement", "verdict", "per_seed_diffs"]].rename(columns={"train_on": "training", "test_dataset": "test"})
    keep = ["clean_auc", "degraded_auc", "worst_type_auc", "flip_rate_pct", "ece_clean", "gradcam_iou_hard", "changed_while_stable_pct"]
    x = stt[stt.metric.isin(keep) & stt.setting.isin(["in-domain", "joint", "joint vs dataset-specific"])].copy()
    if not len(x):
        return pr, pd.DataFrame()
    x["cell"] = [f"{d:+.3f} {VERDICT_TAG.get(v, '?')}" for d, v in zip(x.mean_diff, x.verdict)]
    w = x.pivot_table(index=["comparison", "train_on", "test_dataset"], columns="metric", values="cell", aggfunc="first").reset_index()
    order = {c: i for i, c in enumerate([f"{h} - {l}" for h, l in COMPARISONS])}
    w["_c"] = w.comparison.map(order).fillna(99)
    w["_t"] = w.train_on.map(TRAIN_ORDER).fillna(9)
    w = w.sort_values(["_c", "_t", "test_dataset"]).drop(columns=["_c", "_t"])
    return pr, w[["comparison", "train_on", "test_dataset"] + [m for m in keep if m in w.columns]].rename(
        columns={"train_on": "training", "test_dataset": "test"})


def figure_list(P: Paths) -> List[Tuple[str, str]]:
    cap = {"fig_detection_auc.png": "AUC on original (top) and degraded (bottom) test videos; dots = seeds; Test 5 / Test 7 baselines in grey / blue",
           "fig_worst_flip.png": "Worst degradation type AUC (top) and prediction flips between original and degraded version (bottom, %)",
           "fig_ece_real.png": "Calibration error (temperature) and share of real videos recognised",
           "fig_gradcam_stability.png": "Grad-CAM top-20 % IoU on the hard degradation types (top) and explanations changed while the prediction stayed (bottom, %)",
           "fig_cross_dataset.png": "Cross-dataset generalisation of the dataset-specific models (dashed = chance)",
           "fig_type_heatmap.png": "AUC per degradation type at the severe level (seed mean)",
           "fig_per_manipulation.png": "FF++ fakes detected per manipulation method (Joint models)",
           "fig_faithfulness.png": "Grad-CAM faithfulness: insertion / deletion gain over a random map, original vs degraded version, per predicted class (95 % CI)",
           "fig_risk_coverage.png": "Risk-coverage of the Joint F2 models with isotonic calibration (dashed = 90 % deployment coverage)",
           "fig_training_curves.png": "Validation score per epoch (first seed)",
           "fig_reliability_final.png": "Reliability of the final model: temperature vs isotonic calibration on test videos"}
    out = [(f.name, cap.get(f.name, "")) for f in sorted(P.figures.glob("*.png"))]
    if (P.final_model / "final_model_examples.png").exists():
        out.append(("../final_model/final_model_examples.png", "Final model: decisions and Grad-CAM on original / degraded test videos"))
    return out


def phase_report(a, P: Paths) -> dict:
    rob, degm, manip = _load_metric(P, "robustness"), _load_metric(P, "degradation_metrics"), _load_metric(P, "per_manipulation")
    cs, fsum, fcmp = _load_metric(P, "gradcam_summary"), _load_metric(P, "faithfulness_summary"), _load_metric(P, "faithfulness_comparisons")
    cal, stt, lev = _load_metric(P, "calibration_summary"), _load_metric(P, "statistics"), _load_metric(P, "levels")
    H = read_json(P.metrics / "hypotheses_final.json", "final_metrics") if (P.metrics / "hypotheses_final.json").exists() else []
    link = read_json(P.link, "final_link") if P.link.exists() else {}
    proto = read_json(P.protocol, "final_protocol") if P.protocol.exists() else {}
    card = read_json(P.final_model / "model_card.json", "final_model") if (P.final_model / "model_card.json").exists() else None
    pcheck = read_json(P.reports / "predict_pipeline_check.json", "final_report") if (P.reports / "predict_pipeline_check.json").exists() else None
    ck = read_csv(P.ck_index, "final_checkpoint_index") if P.ck_index.exists() else pd.DataFrame()
    planned = [(t, int(s), v) for t in proto.get("train_sets_planned", a.train_list) for s in proto.get("seeds_planned", a.seed_list)
               for v in proto.get("variants_planned", [STAGE1] + a.variant_list)]
    done = set(ck.run_id) if len(ck) else set()
    n_done = sum(run_id(t, s, v) in done for t, s, v in planned)
    complete = bool(planned) and n_done == len(planned) and card is not None and pcheck is not None
    rs_path = P.logs / "run_status.csv"
    failed = read_csv(rs_path, "final_log") if rs_path.exists() else pd.DataFrame()
    failed = failed[failed.status.astype(str).str.startswith("failed")] if len(failed) else failed
    sel = card["selection"] if card else None
    md = [f"# FINAL TEST - robust, explainable and calibrated video deepfake detector", "",
          f"_Generated {now()} | FINALTEST.py (SHA-256 {code_sha256()[:16]}...) | protocol SHA-256 {str(proto.get('protocol_sha256', 'n/a'))[:16]}..._", ""]
    if a.smoke:
        md += ["> **SMOKE TEST** - tiny data subset, 1 epoch, few bootstrap replicates: checks the pipeline only. Not a result.", ""]
    rand = ck[ck["init"].astype(str).str.startswith("random")] if len(ck) and "init" in ck else pd.DataFrame()
    if len(rand):
        md += [f"> **WARNING**: {len(rand)} Stage-1 run(s) started from RANDOM weights (--allow-random-init). Results are not valid.", ""]
    md += [f"**Status: {'COMPLETE' if complete else 'PARTIAL'}** - {n_done} of {len(planned)} planned runs finished"
           + ("; final model frozen" if card else "; final model not frozen yet")
           + (f"; prediction-pipeline check {pcheck['status']}" if pcheck else "") + ".", ""]
    md += ["## 1. Design (fixed before training; see configs/final_test_protocol.json)", "",
           "- **Data**: the prepared Test 5 run, read-only - same split, same face-crop caches (80 px face bottleneck), same verified degraded "
           "versions and the same 720 Grad-CAM pairs. Every result is therefore paired with Test 5 (5A, 5C) and Test 7 (6A, 6B) on identical "
           "test videos and seeds.",
           f"- **Model**: EfficientNet-B0 (ImageNet) per face frame -> temporal transformer -> attention pooling; video only; {a.img} px input.",
           f"- **F0** {F0_DESCRIPTION}.", f"- **F1** {VARIANTS['F1']['description']}.", f"- **F2** {VARIANTS['F2']['description']}.",
           "- **Training sets**: Joint (the final single detector), BioDeepAV, FF++; seeds " + ", ".join(str(s) for s in proto.get("seeds_planned", a.seed_list)) + ".",
           "- **Decisions**: temperature scaling + validation threshold per run; the frozen final model adds isotonic calibration and an "
           f"UNCERTAIN band tuned for {DEPLOY_COVERAGE:.0%} validation coverage.",
           "- **Statistics**: paired cluster bootstrap over source-identity groups (seeds averaged) + sign agreement of every seed; Holm over the "
           f"8 primary comparisons; non-inferiority margins {NONINF_AUC} AUC / {NONINF_XAI} IoU.", ""]
    if link:
        md += ["## 2. Input link (Test 5, read-only)", "",
               f"- status **{link.get('status')}**; {link.get('originals', 'n/a')} originals, {link.get('degraded', 'n/a')} degraded versions; "
               f"split manifest SHA-256 `{str(link.get('split_manifest_sha256', ''))[:16]}...`; preprocessing code `{str(link.get('preprocessing_code_sha256', ''))[:16]}...`",
               "- baselines found: " + "; ".join(f"{k}: {v.get('predictions', 0)} prediction / {v.get('gradcam', 0)} Grad-CAM files"
                                                for k, v in link.get("baselines", {}).items()), ""]
    md += ["## 3. Runs", ""]
    if len(ck):
        cols = [c for c in ["run_id", "best_epoch", "epochs_run", "best_val_score", "temperature", "threshold", "val_ece_raw", "val_ece",
                            "val_ece_isotonic", "batch_clips", "accum", "train_time_s"] if c in ck.columns]
        md += [md_table(ck.sort_values("run_id")[cols], 3, 120), ""]
    else:
        md += ["_(no finished run yet)_", ""]
    if len(failed):
        md += ["Failed runs (see logs):", "", md_table(failed, 3), ""]
    if len(rob):
        md += ["## 4. Detection on original and degraded test videos (seed mean +- SD)", "",
               "Test 5 stored uncalibrated probabilities (its ECE is not temperature-scaled); Test 7 and the final test use temperature scaling. "
               "AUC does not depend on calibration.", "", md_table(detection_table(rob), 3, 120), "",
               "### Per degradation type (AUC, seed mean, all severities)", "", md_table(type_table(degm), 3, 120), ""]
        mt = manip_table(manip) if len(manip) else pd.DataFrame()
        if len(mt):
            md += ["### FF++ per manipulation method (share detected; 'original (real)' = share of real videos recognised)", "", md_table(mt, 3, 80), ""]
    if len(cal):
        e, b = calibration_report_tables(cal)
        md += ["## 5. Calibration and selective decisions (test videos; calibrators fitted on each run's validation videos)", "",
               "### ECE by calibration method", "", md_table(e, 3, 80), "",
               "### Isotonic calibration + abstention band: balanced accuracy on the decided videos", "", md_table(b, 3, 80), ""]
    if len(cs):
        md += ["## 6. Grad-CAM stability between each original and its degraded version (720 pre-registered pairs)", "",
               "IoU = overlap of the top-20 % Grad-CAM regions; clean-reference IoU = the same video seen on the other 8 frames (natural "
               "variability); relative IoU = IoU / clean-reference IoU. Stability is not correctness.", "", md_table(gradcam_table(cs), 3, 80), ""]
    if len(fsum):
        s, c = faith_report_tables(fsum, fcmp)
        md += ["## 7. Faithfulness under degradation (exploratory RQ)", "",
               "Gain = AUC of the insertion (deletion) curve of the map minus that of a random map, grey perturbation; > 0 = the map points at "
               "evidence the model uses (* = 95 % CI above +0.005). Every 4th Grad-CAM pair, original and degraded version.", "",
               md_table(s, 3, 80), "", "### Paired comparisons", "", md_table(c, 3, 80), ""]
    if len(stt):
        pr, w = stats_tables(stt)
        md += ["## 8. Statistics", "", "### Primary comparisons (Holm-corrected family)", "", md_table(pr, 4, 20), "",
               "### Paired comparisons (difference hi - lo; S+ supported improvement, S- supported deterioration, ns inconclusive)", "",
               md_table(w, 3, 120), "", "Full table: metrics/statistics_final.csv (levels with CIs: metrics/levels_final.csv).", ""]
    if H:
        md += ["## 9. Hypotheses", "", md_table(pd.DataFrame(H)[["hypothesis", "statement", "verdict"]], 3), ""]
        for h in H:
            if h.get("evidence"):
                md.append(f"- **{h['hypothesis']}** ({h['comparison']}): {h['evidence']}")
        md.append("")
    if card:
        c_ = card["calibration"]
        md += ["## 10. Final model", "",
               f"- **{card['source_run']}** frozen {card['frozen_at']}; rule: {sel['rule']}; mean validation score "
               + ", ".join(f"{k} {v:.4f}" for k, v in sel["mean_val_score"].items()) + "." + (f" {sel['deviation']}." if sel.get("deviation") else ""),
               f"- Decision: P = isotonic(logit); FAKE if P >= {c_['threshold']:.3f}; UNCERTAIN if |P - threshold| < {c_['abstain_band']:.3f} "
               f"(validation coverage {c_['validation_coverage']:.1%}) or no face found.",
               "", md_table(pd.DataFrame(card.get("test_performance", [])), 3), "",
               "Model card: final_model/MODEL_CARD.md; examples: final_model/final_model_examples.png.", ""]
    if pcheck:
        md += ["## 11. Prediction-pipeline check (--predict on raw videos == evaluation input)", "",
               f"- status **{pcheck.get('status')}** ({pcheck.get('criteria', '')})",
               f"- {pcheck.get('n_ok', 0)} re-extracted clips; pixel MAE median {pcheck.get('pixel_mae_median', float('nan')):.3f}; "
               f"max |delta P(fake)| {pcheck.get('abs_dp_max', float('nan')):.4f}; verdict agreement {pcheck.get('verdict_agreement', float('nan')):.0%}", ""]
    figs = figure_list(P)
    if figs:
        md += ["## 12. Figures", ""] + [f"![{c or f}](../figures/{f})" if not f.startswith("..") else f"![{c}]({f})" for f, c in figs] + [""]
        md += [f"- `{f}`: {c}" for f, c in figs] + [""]
    notes = []
    if planned and n_done < len(planned):
        miss = [run_id(t, s, v) for t, s, v in planned if run_id(t, s, v) not in done]
        notes.append(f"{len(miss)} planned runs not finished yet: " + ", ".join(miss[:12]) + (" ..." if len(miss) > 12 else ""))
    if len(ck) and "batch_clips" in ck:
        diff = ck[(ck.batch_clips.astype(float) * ck.accum.astype(float)) != float(proto.get("protocol", {}).get("common", {}).get("effective_batch_clips", 0))]
        if len(diff):
            notes.append("runs with a different effective batch: " + ", ".join(diff.run_id))
        oom = ck[ck.batch_clips.astype(float) < a.batch_size] if "batch_clips" in ck else pd.DataFrame()
        if len(oom):
            notes.append("out-of-memory retries (same effective batch, smaller micro-batch): " + ", ".join(oom.run_id))
    notes.append("Test 5 / Test 7 numbers are read from their prediction and Grad-CAM files (normal audio condition) on the same test videos; "
                 "no earlier checkpoint is loaded (guard() + logs/file_access_log.csv).")
    md += ["## 13. Notes", ""] + [f"- {n}" for n in notes] + [""]
    (P.reports / "FINAL_TEST_REPORT.md").write_text("\n".join(md), encoding="utf-8")
    summ = dict(test="Final test of the thesis (FINALTEST.py)", generated_at=now(), status="complete" if complete else "partial", smoke=bool(a.smoke),
                script_sha256=code_sha256(), protocol_sha256=proto.get("protocol_sha256"), runs_planned=len(planned), runs_done=n_done)
    if len(rob):
        recs = []
        for (t, ds, st, v), g in rob.groupby(["train_on", "test_dataset", "setting", "variant"]):
            r = dict(train_on=t, test_dataset=ds, setting=st, variant=v, n_seeds=int(g.seed.nunique()))
            for m in ("clean_auc", "degraded_auc_pooled", "worst_type_auc", "prediction_flip_rate_pct", "real_recognised", "fake_detected", "ece_clean", "ece_degraded"):
                r[f"{m}_mean"], r[f"{m}_sd"] = float(g[m].mean()), (float(g[m].std(ddof=1)) if len(g) > 1 else None)
            recs.append(r)
        summ["detection"] = recs
    if len(cs):
        summ["gradcam"] = [dict(train_on=t, test_dataset=ds, variant=v, iou_all=float(g[g.scope == "all"].mean_top20_iou.mean()),
                                iou_hard=float(g[g.scope == "hard"].mean_top20_iou.mean()),
                                changed_while_stable_pct=float(g[g.scope == "all"].explanation_changed_given_stable_pct.mean()))
                           for (t, ds, v), g in cs.groupby(["train_on", "test_dataset", "variant"])]
    if len(stt) and "primary" in stt:
        summ["primary_comparisons"] = stt[stt.primary][["comparison", "train_on", "test_dataset", "metric", "mean_diff", "ci95_low", "ci95_high",
                                                        "p_boot", "p_holm", "verdict"]].to_dict("records")
    summ["hypotheses"] = H
    if card:
        summ["final_model"] = dict(source_run=card["source_run"], threshold=card["calibration"]["threshold"], band=card["calibration"]["abstain_band"],
                                   test_performance=card.get("test_performance"), selection=dict(rule=sel["rule"], mean_val_score=sel["mean_val_score"]))
    if pcheck:
        summ["predict_pipeline_check"] = pcheck.get("status")
    write_json(P.reports / "final_summary.json", summ)
    print(f"  report -> {P.reports / 'FINAL_TEST_REPORT.md'}")
    return summ


# =====================================================================================
# SELFTEST (no data needed)
# =====================================================================================
NONCRITICAL = {"Test 5 preprocessing importable and unchanged (needed for --predict only)", "YuNet face model present (needed for --predict only)"}


def phase_selftest(a, P: Paths) -> bool:
    print("\n" + "=" * 100 + "\n PHASE SELFTEST: unit tests of the model, losses, Grad-CAM, calibration and statistics (no data)\n" + "=" * 100)
    res, err = {}, {}
    t_all = time.time()

    def check(name, fn):
        t0 = time.time()
        try:
            ok = fn()
            ok = True if ok is None else bool(ok)
        except (Exception, SystemExit) as e:                      # SystemExit included: report, never abort the selftest
            ok = False
            err[name] = f"{type(e).__name__}: {str(e)[:300]}"
        res[name] = ok
        tag = "PASS" if ok else ("WARN" if name in NONCRITICAL else "FAIL")
        print(f"  [{tag}] {name}" + (f"  ({err[name]})" if name in err else "") + f"  [{time.time() - t0:.1f}s]")

    cpu = torch.device("cpu")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(0)
    torch.manual_seed(0)

    def t_weights():
        if torchvision is None:
            raise RuntimeError("torchvision missing")
        try:
            efficientnet_b0(weights=torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1)
            return True
        except Exception as e:
            if a.allow_random_init:
                print(f"    (ImageNet weights unavailable: {type(e).__name__}; --allow-random-init given - pipeline check only)")
                return True
            raise RuntimeError("ImageNet weights could not be loaded; connect to the internet once (torchvision caches them) - "
                               f"{type(e).__name__}: {e}")
    check("ImageNet EfficientNet-B0 weights available (torchvision cache)", t_weights)

    S = 96
    clips = rng.integers(0, 256, (2, CLIP_LEN, S, S, 3), dtype=np.uint8)
    built: Dict[str, Any] = {}

    def t_build():
        built["m"] = FinalDetector(pretrained=False).to(cpu).eval()
        built["x"] = to_input(clips, cpu, S)
        built["tix"] = frame_positions(2, EVEN_FRAMES, cpu)
        return True
    check("model builds (EfficientNet-B0 + temporal transformer + attention pooling)", t_build)
    m, x, tix = built.get("m"), built.get("x"), built.get("tix")

    def t_shapes():
        with torch.no_grad():
            o = m(x, tix)
        return (tuple(o["logit"].shape) == (2,) and tuple(o["frame_logit"].shape) == (2, CLIP_LEN) and tuple(o["z"].shape) == (2, EMB_DIM)
                and o["act"].shape[1] == 1280 and torch.allclose(o["attn"].sum(1), torch.ones(2), atol=1e-4)
                and bool(torch.isfinite(o["logit"]).all()))
    check("model: logit, frame logits, attention (sums to 1), embedding, 1280-channel Grad-CAM layer", t_shapes)

    def t_positions():
        with torch.no_grad():
            l1 = m(x, frame_positions(2, EVEN_FRAMES, cpu))["logit"]
            l2 = m(x, frame_positions(2, ODD_FRAMES, cpu))["logit"]
        return not torch.allclose(l1, l2)
    check("model: frame positions reach the temporal transformer", t_positions)

    def t_to_input():
        y = to_input(np.full((1, 2, 64, 64, 3), 128, np.uint8), cpu, 32)
        return tuple(y.shape) == (1, 2, 3, 32, 32) and np.allclose(y[0, 1, :, 5, 5].numpy(), (128 / 255 - MEAN) / STD, atol=1e-4)
    check("to_input: resize + ImageNet normalisation", t_to_input)

    def t_gradcam():
        g = grad_cam_eval(m, x, tix, 1.0)
        gn = grad_cam_eval(m, x, tix, -1.0)
        gc_ = grad_cam_eval(m, x, tix, sign=lambda z: np.where(z >= 0, 1.0, -1.0))
        tsum = g["temporal"].sum(1)
        sub = {"shapes": g["cam"].shape == (2, CLIP_LEN, CACHE_SIZE, CACHE_SIZE) and g["campp"].shape == g["cam"].shape
                         and g["spatial"].shape == (2, CACHE_SIZE, CACHE_SIZE) and g["spatial_pp"].shape == g["spatial"].shape,
               "finite": all(np.isfinite(g[k]).all() for k in ("cam", "campp", "spatial", "spatial_pp", "temporal", "attn")),
               "ranges": bool(g["cam"].min() >= 0 and g["cam"].max() <= 1 + 1e-4 and g["spatial"].min() >= -1e-6
                              and g["spatial"].max() <= 1 + 1e-4),
               "temporal sums to 1": bool(np.all((np.abs(tsum - 1) < 1e-3) | (tsum == 0))),
               "+1 and -1 give different maps": bool(not np.allclose(g["cam"], gn["cam"]) or (g["cam"].max() == 0 and gn["cam"].max() == 0)),
               "per-clip sign": bool(np.array_equal(gc_["sign"], np.where(gc_["logit"] >= 0, 1.0, -1.0).astype(np.float32)))}
        bad = [k for k, v in sub.items() if not v]
        if bad:
            raise AssertionError("failed: " + ", ".join(bad) + f" (temporal sums {np.round(tsum, 4).tolist()}, cam max "
                                 f"{float(g['cam'].max()):.3g}/{float(gn['cam'].max()):.3g})")
        return True
    check("Grad-CAM / Grad-CAM++: shapes, ranges, temporal profile, sign (+1 FAKE / -1 REAL / per clip)", t_gradcam)

    def t_double_backward():
        md_ = FinalDetector(pretrained=False).to(dev).train()
        xx = to_input(rng.integers(0, 256, (4, CLIP_LEN, 64, 64, 3), dtype=np.uint8), dev, 64)
        tt = frame_positions(4, EVEN_FRAMES, dev)
        with math_sdp():
            with torch.autocast("cuda", dtype=torch.float16, enabled=bool(a.amp and dev.type == "cuda")):
                o = md_(xx, tt)
            cam = soft_gradcam(o["logit"], o["act"], 4, CLIP_LEN)
            lc = cam_consistency_loss(cam[:2].detach(), cam[2:])
            (o["logit"].float().mean() + lc).backward()
        grads = [p.grad for p in md_.backbone.parameters() if p.grad is not None]
        hg = [p.grad for n, p in md_.named_parameters() if n.startswith("temporal.") and p.grad is not None]
        ok = bool(torch.isfinite(lc)) and len(grads) > 0 and len(hg) > 0
        ok = ok and any(bool(torch.isfinite(g_).all()) and float(g_.abs().sum()) > 0 for g_ in grads)
        del md_, o, cam
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        return ok
    check(f"Grad-CAM consistency loss: gradient of a gradient through the transformer on {dev.type}" + (" + AMP" if a.amp else ""),
          t_double_backward)

    def t_cam_loss():
        a1 = torch.rand(3, 4, 4)
        b1, b2 = torch.zeros(3, 4, 4), torch.zeros(3, 4, 4)
        b1[:, :2], b2[:, 2:] = 1, 1
        return abs(cam_consistency_loss(a1, a1).item()) < 1e-4 and abs(cam_consistency_loss(b1, b2).item() - 1) < 1e-4
    check("Grad-CAM consistency loss: 0 for identical maps, 1 for disjoint maps", t_cam_loss)

    def t_supcon():
        y = torch.tensor([0, 0, 1, 1, 0, 1])
        zg = torch.tensor([[1, 0], [1, 0.1], [0, 1], [0.1, 1], [1, -0.1], [-0.1, 1]], dtype=torch.float32)
        zb = torch.tensor([[1, 0], [0, 1], [1, 0.1], [0.1, 1], [0, 1.0], [1, 0.0]], dtype=torch.float32)
        lg_, lb_ = supcon_loss(zg, y), supcon_loss(zb, y)
        return bool(torch.isfinite(lg_)) and float(lg_) < float(lb_)
    check("supervised contrastive loss: lower when same-class embeddings cluster", t_supcon)

    def t_cam_metrics():
        mp = rng.random((32, 32))
        r_, iou, sh = cam_metrics(mp, mp)
        r2, iou2, _ = cam_metrics(mp, 1 - mp)
        return abs(r_ - 1) < 1e-9 and abs(iou - 1) < 1e-9 and sh < 1e-9 and r2 < -0.99 and iou2 < 0.05
    check("explanation metrics: Pearson r, top-20 % IoU, centre shift", t_cam_metrics)

    def t_temperature():
        z = rng.normal(0, 4, 5000)
        y = (rng.random(5000) < sigmoid(z / 2.5)).astype(int)
        T = fit_temperature(z, y)
        return 2.0 < T < 3.1 and ece_score(y, sigmoid(z / T)) < ece_score(y, sigmoid(z))
    check("temperature scaling recovers T and lowers ECE", t_temperature)

    def t_isotonic():
        z = rng.normal(0, 3, 3000)
        y = (rng.random(3000) < sigmoid(z / 2 + 0.5)).astype(int)
        iso = IsoCal().fit(z, y)
        p = iso(np.sort(z))
        iso2 = IsoCal(iso.x.tolist(), iso.y.tolist())
        return (bool(np.all(np.diff(p) >= -1e-12)) and p.min() >= 0 and p.max() <= 1 and ece_score(y, iso(z)) < ece_score(y, sigmoid(z))
                and np.allclose(iso2(z), iso(z)) and np.allclose(IsoCal()(z), sigmoid(z)))
    check("isotonic calibration: monotone, in [0,1], lowers ECE, survives save/load", t_isotonic)

    def t_threshold():
        ds = np.array(["A"] * 2000 + ["B"] * 2000)
        y = np.r_[rng.random(2000) < 0.5, rng.random(2000) < 0.3].astype(int)
        p = np.where(y == 1, rng.uniform(0.35, 1.0, 4000), rng.uniform(0.0, 0.33, 4000))
        return 0.33 <= choose_threshold(y, p, ds) <= 0.36
    check("threshold = maximum mean balanced accuracy over datasets", t_threshold)

    def t_band():
        p = rng.random(20000)
        b = abstain_band(p, 0.5, 0.9)
        return abs((np.abs(p - 0.5) >= b).mean() - 0.9) < 0.01 and abstain_band(p, 0.5, 1.0) == 0.0 and verdict_of(0.52, 0.5, b) == "UNCERTAIN"
    check("abstention band keeps the target coverage", t_band)

    def t_wauc():
        y = rng.integers(0, 2, 400)
        p = np.round(rng.random(400), 1)
        w = rng.integers(0, 3, 400)
        v = WeightedAUC(y, p)(np.vstack([np.ones(400), w]))
        return abs(v[0] - roc_auc_score(y, p)) < 1e-9 and abs(v[1] - roc_auc_score(np.repeat(y, w), np.repeat(p, w))) < 1e-9
    check("bootstrap AUC: weighted AUC == AUC of the resampled videos (with ties)", t_wauc)

    def t_wece():
        y, p = rng.integers(0, 2, 500), rng.random(500)
        w = rng.integers(0, 3, 500)
        v = weighted_ece(y, p, np.vstack([np.ones(500), w]))
        return abs(v[0] - ece_score(y, p)) < 1e-9 and abs(v[1] - ece_score(np.repeat(y, w), np.repeat(p, w))) < 1e-9
    check("bootstrap ECE: weighted ECE == ECE of the resampled videos", t_wece)

    def t_holm():
        adj = holm(np.array([0.01, 0.04, 0.03, np.nan]))
        return np.allclose(adj[:3], [0.03, 0.06, 0.06]) and np.isnan(adj[3]) and p_boot(rng.normal(1, 0.1, 1000)) < 0.01 \
            and p_boot(rng.normal(0, 1, 1000)) > 0.2
    check("Holm correction and bootstrap p-value", t_holm)

    def t_dro():
        dummy = type("TrainerStub", (), {})()
        dummy.D = type("DataStub", (), {})()
        dummy.D.fcell = np.array(["A|r"] * 20 + ["A|x"] * 10 + ["A|y"] * 10)
        w, share = Trainer.cell_weights(dummy, np.arange(40), {"A|r": 0.2, "A|x": 2.0, "A|y": 0.4})
        return max(share, key=share.get) == "A|x" and min(share.values()) >= DRO_FLOOR / 3 - 1e-12 and abs(w.sum() - 1) < 1e-9
    check("group-DRO sampler up-weights the hardest dataset x folder cell (with floor)", t_dro)

    def t_aug():
        c = rng.integers(0, 256, (4, 64, 64, 3), dtype=np.uint8)
        a1 = apply_aug(c, aug_params(np.random.default_rng(5), 64))
        a2 = apply_aug(c.copy(), aug_params(np.random.default_rng(5), 64))
        return np.array_equal(a1, a2) and a1.shape == c.shape and a1.dtype == np.uint8 and np.array_equal(apply_aug(c, None), c)
    check("paired augmentation: identical parameters -> identical clips", t_aug)

    def t_ema():
        net = nn.Linear(3, 2)
        e = EMA(net, 0.0)
        with torch.no_grad():
            net.weight.add_(1.0)
        e.update(net)
        e2 = EMA(net, 1.0)
        w0 = e2.model.weight.clone()
        with torch.no_grad():
            net.weight.add_(1.0)
        e2.update(net)
        return torch.allclose(e.model.weight, net.weight - 1.0) and torch.allclose(e2.model.weight, w0)
    check("EMA of the weights", t_ema)

    def t_schedule():
        f = lr_lambda(100)
        return abs(f(0) - 1 / 8) < 1e-9 and abs(f(7) - 1.0) < 1e-9 and f(99) < 0.01 and all(f(i) >= f(i + 1) - 1e-12 for i in range(8, 99))
    check("learning-rate schedule: warm-up + cosine", t_schedule)

    def t_guard():
        old = _ACCESS["log"]
        _ACCESS["log"] = None
        try:
            try:
                guard(T5_ROOT / "checkpoints" / "x.pt", "selftest")
                return False
            except ForbiddenArtifactError:
                pass
            return guard(SCRIPT_DIR / "selftest_dummy.csv", "selftest") is not None
        finally:
            _ACCESS["log"] = old
    check("artifact guard blocks checkpoints of earlier tests", t_guard)

    def t_faith():
        ns = argparse.Namespace(img=64, amp=False)
        clip = rng.integers(0, 256, (CLIP_LEN, 64, 64, 3), dtype=np.uint8)
        out = faith_curves(m, cpu, ns, clip, EVEN_FRAMES, rng.random(clip.shape[:3]), 1, 1.0, np.random.default_rng(1))
        with torch.no_grad():
            p0 = float(sigmoid(m(to_input(clip[None], cpu, 64), frame_positions(1, EVEN_FRAMES, cpu))["logit"].numpy()[0]))
        return (all(0 <= out[k] <= 1 for k in ("deletion_auc_grey", "insertion_auc_grey", "deletion_auc_blur", "insertion_auc_blur"))
                and abs(out["del_grey_0"] - p0) < 1e-4)
    check("faithfulness: deletion / insertion curves (grey, blur)", t_faith)

    def t_stats():
        ns = argparse.Namespace(n_boot=200)
        r2 = np.random.default_rng(3)
        rows = []
        for v, noise in (("F1", 0.15), ("F0", 0.45)):
            for seed in (42, 123):
                for gi in range(40):
                    for k in range(4):
                        lab = int((gi + k) % 2)
                        key = f"g{gi}_v{k}"
                        for kind, dg in (("clean", "original"), ("deg", "blur"), ("deg", "noise")):
                            p = float(np.clip(0.5 + (0.3 if lab else -0.3) + r2.normal(0, noise + (0.1 if kind == "deg" else 0)), 0, 1))
                            rows.append(dict(run_id=f"train-Joint_seed{seed}_{v}", train_on="Joint", variant=v, seed=seed, split="test",
                                             item_id=key + dg, kind=kind, source_key=key, group_id=f"G{gi}", dataset="BioDeepAV", label=lab,
                                             degradation=dg, severity="severe" if kind == "deg" else "", condition="normal", p_fake=p, threshold=0.5))
        st, lv = bootstrap_stats(ns, pd.DataFrame(rows), pd.DataFrame())
        r = st[(st.comparison == "F1 - F0") & (st.metric == "degraded_auc")]
        return len(r) == 1 and r.verdict.iloc[0] == "supported improvement" and len(lv) > 0
    check("paired cluster bootstrap detects a clear improvement on synthetic data", t_stats)

    def t_protocol():
        d1, d2 = protocol_dict(a), protocol_dict(a)
        return sha256_text(json.dumps(to_jsonable(d1), sort_keys=True)) == sha256_text(json.dumps(to_jsonable(d2), sort_keys=True))
    check("pre-registered protocol is serialisable and hash-stable", t_protocol)

    def t_t5():
        if not Path(a.t5_script).exists():
            raise FileNotFoundError(str(a.t5_script))
        mod, sha, _ = load_t5_module(a, None)
        if a.t5.provenance.exists():
            exp = read_json(a.t5.provenance, "t5_cache_provenance")["faceclean"].get("preprocessing_code_sha256")
            if sha != exp:
                raise RuntimeError(f"preprocessing hash {sha[:12]} != cache {str(exp)[:12]}")
        return callable(getattr(mod, "extract_face_clip", None))
    check("Test 5 preprocessing importable and unchanged (needed for --predict only)", t_t5)

    def t_yunet():
        return (SCRIPT_DIR / "models" / "face_detection_yunet_2023mar.onnx").exists() and hasattr(cv2, "FaceDetectorYN")
    check("YuNet face model present (needed for --predict only)", t_yunet)

    built.clear()
    del m, x, tix
    gc.collect()
    critical = {k: v for k, v in res.items() if k not in NONCRITICAL}
    ok = all(critical.values())
    write_json(P.reports / "selftest_report.json", dict(passed=ok, results=res, errors=err, noncritical=sorted(NONCRITICAL), checked_at=now(),
                                                        seconds=round(time.time() - t_all, 1), environment=environment_info()))
    print(f"  selftest {'PASSED' if ok else 'FAILED'} ({sum(critical.values())}/{len(critical)} critical checks"
          + (f"; warnings: {', '.join(k for k in NONCRITICAL if not res.get(k, True))}" if not all(res.get(k, True) for k in NONCRITICAL) else "")
          + f") in {time.time() - t_all:.0f} s -> {P.reports / 'selftest_report.json'}")
    return ok


# =====================================================================================
# STATUS, RUN, MAIN
# =====================================================================================
def phase_status(a, P: Paths):
    print("\n" + "=" * 100 + f"\n STATUS of {P.root}\n" + "=" * 100)
    link = read_json(P.link, "final_link") if P.link.exists() else {}
    proto = read_json(P.protocol, "final_protocol") if P.protocol.exists() else {}
    print(f"  input link: {link.get('status', 'not run')} | protocol: {str(proto.get('protocol_sha256', 'not registered'))[:16]}")
    rows = []
    for t in a.train_list:
        for seed in a.seed_list:
            for v in [STAGE1] + a.variant_list:
                rid = run_id(t, seed, v)
                ck = P.checkpoints / train_tag(t) / f"seed{seed}" / f"{v}.pt"
                state = "done" if ck.exists() else ("in progress" if ck.with_suffix(".resume.pt").exists() else "-")
                faith_needed = a.faith_scope == "all" or (a.faith_scope == "joint" and t == "Joint")
                rows.append(dict(run=rid, checkpoint=state, predictions="yes" if (P.pred_dir / f"{rid}.csv.gz").exists() else "-",
                                 gradcam="yes" if (P.cam_dir / f"{rid}.csv").exists() else "-",
                                 faithfulness=("yes" if (P.faith_dir / f"{rid}.csv").exists() else "-") if faith_needed else "n/a"))
    df = pd.DataFrame(rows)
    print(md_table(df, 3, 200))
    print(f"\n  checkpoints {int((df.checkpoint == 'done').sum())}/{len(df)} | predictions {int((df.predictions == 'yes').sum())} | "
          f"Grad-CAM {int((df.gradcam == 'yes').sum())} | faithfulness {int((df.faithfulness == 'yes').sum())}")
    print(f"  final model: {'yes' if P.final.exists() else 'no'} | prediction check: "
          + (read_json(P.reports / "predict_pipeline_check.json", "final_report").get("status") if (P.reports / "predict_pipeline_check.json").exists() else "no")
          + f" | report: {'yes' if (P.reports / 'FINAL_TEST_REPORT.md').exists() else 'no'}")


def phase_run(a, P: Paths, selftest: bool = True, per_seed: bool = True):
    """run: selftest -> link -> for each seed (train, evaluate, Grad-CAM, faithfulness, analyse, report) -> freeze -> predict check.
    all: the same without the selftest and with all seeds trained before the evaluation."""
    t0 = time.time()
    if selftest and not a.skip_selftest:
        if not phase_selftest(a, P):
            raise SystemExit("[STOP] selftest failed - see the FAIL lines above (reports/selftest_report.json). Nothing was trained.")
    _ACCESS["phase"] = "link"
    phase_link(a, P)
    D = FData(a, P)
    groups = [[s] for s in a.seed_list] if per_seed else [list(a.seed_list)]
    for k, sl in enumerate(groups):
        print("\n" + "#" * 100 + f"\n SEED {', '.join(map(str, sl))} ({k + 1}/{len(groups)}) | training sets {', '.join(a.train_list)} | "
              f"variants {', '.join([STAGE1] + a.variant_list)} | elapsed {fmt_time(time.time() - t0)}\n" + "#" * 100)
        for ph, fn in (("train", lambda: phase_train(a, P, D, sl)), ("evaluate", lambda: phase_evaluate(a, P, D)), ("xai", lambda: phase_xai(a, P, D)),
                       ("faith", lambda: phase_faith(a, P, D)), ("analyze", lambda: phase_analyze(a, P)), ("report", lambda: phase_report(a, P))):
            _ACCESS["phase"] = ph
            print("\n" + "-" * 100 + f"\n {ph.upper()}\n" + "-" * 100)
            fn()
    for ph, fn in (("freeze", lambda: phase_freeze(a, P, D)), ("predictcheck", lambda: phase_predictcheck(a, P, D)),
                   ("analyze", lambda: phase_analyze(a, P)), ("report", lambda: phase_report(a, P))):
        _ACCESS["phase"] = ph
        fn()
    print("\n" + "=" * 100 + f"\n FINAL TEST finished in {fmt_time(time.time() - t0)}\n  report:      {P.reports / 'FINAL_TEST_REPORT.md'}\n"
          f"  summary:     {P.reports / 'final_summary.json'}\n  final model: {P.final} (+ MODEL_CARD.md)\n" + "=" * 100)


def main(argv=None):
    a = parse_args(argv)
    P = Paths(a.root)
    P.make()
    if a.phase in ("predict", "gui"):
        _ACCESS["log"], _ACCESS["phase"] = None, a.phase
        return phase_predict(a, P) if a.phase == "predict" else phase_gui(a, P)
    lower_process_priority()
    log_path = P.logs / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    stdout0 = sys.stdout
    sys.stdout = Tee(log_path)
    _ACCESS["log"], _ACCESS["phase"] = P.access_log, a.phase
    try:
        print("=" * 100 + f"\n FINALTEST.py | phase {a.phase} | root {P.root}" + (" | SMOKE" if a.smoke else "") + f" | {now()}\n" + "=" * 100)
        env = environment_info()
        write_json(P.logs / f"environment_{a.phase}.json", env)
        print(f"  Python {env['python']} | PyTorch {env['pytorch']} (CUDA {env['cuda']}) | GPU {env['gpu']} ({env['gpu_memory_gb']} GB) | "
              f"OpenCV {env['opencv']} | batch {a.batch_size} x accum {a.accum} | AMP {a.amp}")
        ph = a.phase
        if ph == "selftest":
            ok = phase_selftest(a, P)
            sys.exit(0 if ok else 1)
        if ph == "status":
            return phase_status(a, P)
        if ph == "link":
            return phase_link(a, P)
        if ph == "run":
            return phase_run(a, P)
        if ph == "all":
            return phase_run(a, P, selftest=False, per_seed=False)
        if ph in ("analyze", "report"):
            if ph == "analyze":
                phase_analyze(a, P)
            return phase_report(a, P)
        require_link(P)
        D = FData(a, P)
        {"train": lambda: phase_train(a, P, D), "evaluate": lambda: phase_evaluate(a, P, D), "xai": lambda: phase_xai(a, P, D),
         "faith": lambda: phase_faith(a, P, D), "freeze": lambda: phase_freeze(a, P, D),
         "predictcheck": lambda: phase_predictcheck(a, P, D)}[ph]()
    except KeyboardInterrupt:
        print("\n  interrupted - run the same command again to resume (finished runs and resume points are kept)")
        sys.exit(130)
    finally:
        sys.stdout.flush()
        sys.stdout = stdout0


if __name__ == "__main__":
    main()
