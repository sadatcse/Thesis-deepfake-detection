"""
8-10-26 (Test 5): HARD-DEGRADATION AND EXPLANATION-AWARE TRAINING
=================================================================

Builds on 6-10-26.py (Test 4, kept unchanged and NOT modified). Baseline = Test 4 variant 4D.

Research question
  Does targeted Stage-2 training on the degradation types that produced the least stable Grad-CAM
  explanations in Test 4 (brightness, saturation, contrast, blur) improve BOTH detection robustness and
  Grad-CAM explanation stability, while keeping the model, data, split and evaluation protocol fixed?

Controlled design (nothing but Stage-2 degradation sampling and the consistency-loss weights changes)
  * The Test 4 module (6-10-26.py) is IMPORTED, not copied: AudioVisualDeepFakeCNN, face cropping,
    resolution normalisation, audio preprocessing, caches, the identity-grouped split, curriculum, mixed
    audio policy, optimiser / OneCycle schedule, batch size, epochs, class balancing, validation-only
    thresholds and model selection are therefore byte-for-byte the Test 4 code. The SHA-256 of 6-10-26.py
    is recorded so later edits to Test 4 would be visible.
  * The split is re-created with the Test 4 DATA seed (42) and compared key-by-key with the Test 4 split
    file (run_6-10-26/4D_<train>/...split_6-10-26.json). Any difference -> STOP.
  * Training seeds (--seeds, default 42,123,2026) only change initialisation and the order/augmentation of
    training batches. The split, the degraded test versions and every evaluation input use the data seed.
  * Stage 1 is identical for all variants. For seed 42 the Test 4D Stage-1 checkpoint is copied (read-only)
    when its configuration matches; other seeds train one Stage 1 per (training set, seed) shared by 5A-5D.
  * Test 4 outputs are only READ (checkpoints, split, caches). Everything is written to
    run_test5_hard_explanation/.

Variants (Stage 2; all: paired original/degraded training, curriculum, mixed audio policy, res. normalisation)
  5A  Test 4D control: lambda_feat 0.10, lambda_cam 0.05, Test 4 degradation sampling (no hard sampling)
  5B  hard-degradation sampling, CE on both pair members only (lambda_feat 0, lambda_cam 0)
  5C  hard-degradation sampling + Grad-CAM consistency (lambda_cam 0.05, lambda_feat 0)
  5D  hard-degradation sampling + feature (0.10) + Grad-CAM (0.05) consistency  (= 5A + hard sampling)
  Contrasts: 5D vs 5A isolates hard sampling; 5C vs 5B isolates the Grad-CAM loss; 5D vs 5C isolates the
  feature loss. 5B vs 5A changes two things (sampling and losses) and is only reported, not interpreted alone.

Hard-degradation sampling (5B-5D)
  Each Stage-2 pair is drawn exactly as in Test 4 (dataset x class balanced source draw). Then with
  probability --hard-share (0.5) the degraded member is replaced by a degraded video of a HARD type drawn
  with weights brightness 0.30 / saturation 0.30 / contrast 0.20 / blur 0.20 from the SAME dataset x class
  cell (its own original becomes the clean member); otherwise ("normal", 0.5) the Test 4 rule is used
  (random version of the drawn source over ALL types). Severity follows the curriculum. Only the provided
  (manifest) degraded videos of the TRAINING split are used. With 8 types the natural share of each hard
  type is ~12.5 %, so the effective shares become ~21 % (brightness, saturation), ~16 % (contrast, blur)
  and ~6 % for each other type - no type is removed. (--normal-pool nonhard restricts the normal half to
  non-hard types; then hard types get exactly 50 %, i.e. 15/15/10/10 %, which is a much weaker change.)
  The hard set comes from the completed Test 4 Grad-CAM results, never from Test 5 test data.

Evaluation (every final model + the frozen Test 4D model, same test videos, frozen validation thresholds)
  detection (AUC, acc, balanced acc, precision, recall/sensitivity, specificity, F1, confusion matrix),
  robustness (clean / degraded AUC, drop, prediction flips, degradation x type and x severity),
  Grad-CAM stability on a fixed, type-stratified sample of aligned original/degraded test pairs
  (Pearson r, top-20 % IoU, centre shift, temporal CAM r, explanation change while prediction stable,
  clean-to-clean reference IoU), worst-case analysis, cross-dataset (BioDeepAV <-> FF++ models),
  audio ablation (normal / zero / degraded audio, video only, audio only), paired bootstrap statistics.
  Model choice for "BEST MODEL" uses VALIDATION data only (pre-registered rule, see choose_best()).
  Explanation CORRECTNESS needs localisation annotations; none exist in this project, so only stability is
  measured unless --localization-csv is given (video,x0,y0,x1,y1 normalised to the face crop).

Usage
  python 8-10-26.py --mode selftest                      # quick checks, no data / training
  python 8-10-26.py --mode leakage                       # data preparation + leakage report only
  python 8-10-26.py                                      # full experiment (seeds 42,123,2026; both training sets)
  python 8-10-26.py --seeds 42                           # minimum: seed 42 only
  python 8-10-26.py --mode report                        # rebuild tables / figures / report from saved results
  Any other Test 4 option (--epochs2, --batch-size, --bio-root, --workers, ...) is passed through unchanged;
  changing protocol options breaks the controlled comparison with Test 4 and is recorded in the configs.
  Runs are resumable (finished runs / stages are skipped; --no-resume to redo).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import random
import shutil
import subprocess
import sys
import threading
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata, wilcoxon, binomtest

SCRIPT_DIR = Path(__file__).resolve().parent
TEST4_SCRIPT = SCRIPT_DIR / "6-10-26.py"


def load_test4(path: Path):
    """Imports 6-10-26.py as a module (the file name is not a valid identifier)."""
    if not path.exists():
        raise SystemExit(f"Test 4 script not found: {path}")
    spec = importlib.util.spec_from_file_location("test4_6_10_26", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


T4 = load_test4(TEST4_SCRIPT)
plt = T4.plt
from sklearn.metrics import roc_curve  # noqa: E402

# =====================================================================================
# CONFIGURATION
# =====================================================================================
DATA_SEED = 42                      # Test 4 seed: split, degraded-version selection, evaluation inputs
DEFAULT_SEEDS = "42,123,2026"       # training seeds
HARD_SHARE = 0.50
HARD_WEIGHTS = "brightness:0.30,saturation:0.30,contrast:0.20,blur:0.20"
ALL_DEG_TYPES = ["blur", "brightness", "compression", "contrast", "frame_drop", "fps_reduction", "low_resolution",
                 "motion_blur", "noise", "saturation"]
PRIMARY_HARD = ["brightness", "saturation"]
MIN_CLUSTERS_FOR_SIGNIFICANCE = 10  # fewer resampling units -> degenerate CIs, never called significant
EXPLANATION_CHANGE_IOU = 0.5        # same definition as Test 4 (top-20 % IoU < 0.5 = explanation changed)
MODEL_ORDER = ["T4D", "5A", "5B", "5C", "5D"]
MODEL_LABEL = {"T4D": "Test 4D", "5A": "Test 5A", "5B": "Test 5B", "5C": "Test 5C", "5D": "Test 5D"}
# validated categorical palette (dataviz reference slots blue / orange / aqua / violet) + neutral grey for Test 4
MODEL_COLOR = {"T4D": "#8a8984", "5A": "#2a78d6", "5B": "#eb6834", "5C": "#1baf7a", "5D": "#4a3aa7"}
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

VARIANTS5 = {
    "5A": dict(feat=True, cam=True, hard=False,
               description="Test 4D control: paired + feature + Grad-CAM consistency + curriculum + mixed audio, Test 4 degradation sampling"),
    "5B": dict(feat=False, cam=False, hard=True,
               description="hard-degradation sampling, paired CE only (no consistency losses), curriculum, mixed audio"),
    "5C": dict(feat=False, cam=True, hard=True,
               description="hard-degradation sampling + Grad-CAM consistency, curriculum, mixed audio"),
    "5D": dict(feat=True, cam=True, hard=True,
               description="hard-degradation sampling + paired feature + Grad-CAM consistency + curriculum + mixed audio (= 5A + hard sampling)"),
}


def register_variants(lambda_feat: float, lambda_cam: float) -> None:
    """Adds 5A-5D as presets of the Test 4 argument parser (all other settings = Test 4D)."""
    base = dict(T4.VARIANTS["4D"])
    for v, c in VARIANTS5.items():
        T4.VARIANTS[v] = dict(base, lambda_feat=lambda_feat if c["feat"] else 0.0, lambda_cam=lambda_cam if c["cam"] else 0.0,
                              paired=True, curriculum=True, audio_policy="mixed", res_norm=True,
                              description=f"Test {v}: {c['description']}")


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def weights_sha256(state_dict) -> str:
    """Hash of the parameter VALUES (independent of pickling metadata in the checkpoint file)."""
    h = hashlib.sha256()
    for k in sorted(state_dict):
        h.update(k.encode())
        h.update(state_dict[k].detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def architecture_signature(model) -> str:
    sd = model.state_dict()
    return sha256_bytes((type(model).__name__ + "\n" + "\n".join(f"{k}:{tuple(v.shape)}" for k, v in sd.items())).encode())


def partial_file_hash(path: Path, chunk: int = 1 << 16) -> str:
    """size + first 64 KiB + last 64 KiB (fast duplicate screen; collisions are confirmed with a full SHA-256)."""
    st = os.stat(path)
    h = hashlib.sha256(str(st.st_size).encode())
    with open(path, "rb") as f:
        h.update(f.read(chunk))
        if st.st_size > 2 * chunk:
            f.seek(-chunk, 2)
            h.update(f.read(chunk))
    return h.hexdigest()


def fast_auc(y, p) -> float:
    y = np.asarray(y).astype(bool)
    p = np.asarray(p, dtype=np.float64)
    ok = ~np.isnan(p)
    y, p = y[ok], p[ok]
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(p)
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def fmt(v, nd=3, pct=False) -> str:
    if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
        return "n/a"
    return f"{v:.{nd}f}%" if pct else f"{v:.{nd}f}"


def md(df: pd.DataFrame, index: bool = True) -> str:
    """DataFrame -> GitHub Markdown table (no tabulate dependency)."""
    d = df.reset_index() if index else df
    cell = lambda v: "" if v is None or (isinstance(v, float) and np.isnan(v)) else (f"{v:.4g}" if isinstance(v, float) else str(v))
    cols = [" / ".join(str(c) for c in k) if isinstance(k, tuple) else str(k) for k in d.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in d.itertuples(index=False)]
    return "\n".join(lines)


def git_hash(path: Path) -> Optional[str]:
    try:
        return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], stderr=subprocess.DEVNULL,
                                       creationflags=T4.NO_WINDOW, timeout=10).decode().strip()
    except Exception:
        return None


def environment_info() -> dict:
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    return dict(python=sys.version.split()[0], python_full=sys.version, platform=platform.platform(), pytorch=torch.__version__,
                cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version() if torch.cuda.is_available() else None, gpu=gpu,
                numpy=np.__version__, pandas=pd.__version__, git_commit=git_hash(SCRIPT_DIR),
                test5_script=str(Path(__file__).resolve()), test5_script_sha256=T4.file_sha256(Path(__file__).resolve()),
                test4_script=str(TEST4_SCRIPT), test4_script_sha256=T4.file_sha256(TEST4_SCRIPT),
                recorded_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


# =====================================================================================
# CLI
# =====================================================================================
def parse_args5(argv=None):
    p = argparse.ArgumentParser(description="8-10-26 (Test 5): hard-degradation and explanation-aware training. "
                                            "Unknown options are passed to the Test 4 parser (6-10-26.py).")
    p.add_argument("--mode", default="full", choices=["full", "leakage", "report", "selftest"])
    p.add_argument("--variants", default="5A,5B,5C,5D")
    p.add_argument("--seeds", default=DEFAULT_SEEDS, help="Training seeds (data seed stays fixed)")
    p.add_argument("--train-on", default="BioDeepAV,FF++", help="Training sets (each model is tested on both datasets)")
    p.add_argument("--primary-train", default="BioDeepAV", help="Training set of the headline results / final_model_5X.pt")
    p.add_argument("--primary-seed", type=int, default=42, help="Seed of final_model_5X.pt and of the paired bootstrap")
    p.add_argument("--data-seed", type=int, default=DATA_SEED, help="Must equal the Test 4 seed (split + evaluation inputs)")
    p.add_argument("--work-dir", default=None, help="Default ./run_test5_hard_explanation")
    p.add_argument("--cache-dir", default=None, help="New caches (only if needed); Test 4 caches are read from <test4-dir>/cache")
    p.add_argument("--test4-dir", default=None, help="Default ./run_6-10-26")
    p.add_argument("--lambda-feat", type=float, default=T4.LAMBDA_FEAT, help="Feature consistency weight of 5A/5D (Test 4 value)")
    p.add_argument("--lambda-cam", type=float, default=T4.LAMBDA_CAM, help="Grad-CAM consistency weight of 5A/5C/5D (Test 4 value)")
    p.add_argument("--hard-share", type=float, default=HARD_SHARE, help="Probability that a pair uses a hard degradation")
    p.add_argument("--hard-weights", default=HARD_WEIGHTS, help="type:weight list of the hard pool")
    p.add_argument("--normal-pool", choices=["natural", "nonhard"], default="natural",
                   help="natural: the other half uses Test 4 sampling over all types; nonhard: only non-hard types")
    p.add_argument("--xai-per-type", type=int, default=40, help="Grad-CAM test pairs per (test dataset, degradation type)")
    p.add_argument("--val-xai-per-type", type=int, default=15, help="Grad-CAM validation pairs per type (model choice only)")
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--no-test4-stage1", action="store_true", help="Train Stage 1 for seed 42 instead of reusing Test 4D's")
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--skip-test4-split-check", action="store_true",
                   help="ONLY for smoke tests on subsets (--max-per-folder): the split cannot equal Test 4's then")
    p.add_argument("--skip-hash-check", action="store_true", help="Skip the duplicate-hash leakage screen")
    p.add_argument("--full-file-hash", action="store_true", help="Full SHA-256 of every video (slow) instead of the partial screen")
    p.add_argument("--localization-csv", default=None, help="Optional annotations: video,x0,y0,x1,y1 (normalised face-crop coords)")
    a, rest = p.parse_known_args(argv)
    for bad in ("--variant", "--seed", "--suite-variants", "--lambda-search", "--mode", "--work-dir", "--train-on"):
        if any(r == bad or r.startswith(bad + "=") for r in rest):
            raise SystemExit(f"{bad} is controlled by Test 5 (use --variants / --seeds / --data-seed / --train-on)")
    a.passthrough = rest
    a.root = Path(a.work_dir) if a.work_dir else SCRIPT_DIR / "run_test5_hard_explanation"
    a.cache_dir = Path(a.cache_dir) if a.cache_dir else a.root / "cache"
    a.test4_dir = Path(a.test4_dir) if a.test4_dir else SCRIPT_DIR / "run_6-10-26"
    a.seed_list = [int(s) for s in str(a.seeds).split(",") if s.strip()]
    a.variant_list = [v.strip().upper() for v in a.variants.split(",") if v.strip()]
    unknown = [v for v in a.variant_list if v not in VARIANTS5]
    if unknown:
        raise SystemExit(f"Unknown variants {unknown}; choose from {list(VARIANTS5)}")
    w = {}
    for tok in a.hard_weights.split(","):
        k, v = tok.split(":")
        w[k.strip()] = float(v)
    s = sum(w.values())
    a.hard_weight_map = {k: v / s for k, v in w.items()}
    alias = {"bio": "BioDeepAV", "biodeepav": "BioDeepAV", "ffpp": "FF++", "ff++": "FF++", "combined": "combined"}
    a.train_list = [alias.get(t.strip().lower(), t.strip()) for t in a.train_on.split(",") if t.strip()]
    a.primary_train = alias.get(a.primary_train.strip().lower(), a.primary_train)
    if a.primary_train not in a.train_list:
        a.primary_train = a.train_list[0]
    if a.primary_seed not in a.seed_list:
        a.primary_seed = a.seed_list[0]
    return a


def t4_args(a5, variant: str, train_on: str):
    """Test 4 namespace for one run: Test 4D protocol + the variant's Stage-2 settings, data seed = Test 4 seed."""
    register_variants(a5.lambda_feat, a5.lambda_cam)
    argv = list(a5.passthrough) + ["--mode", "full", "--variant", variant, "--seed", str(a5.data_seed), "--train-on", train_on,
                                   "--work-dir", str(a5.root), "--cache-dir", str(a5.cache_dir)]
    if not a5.no_resume:
        argv.append("--resume")
    ns = T4.parse_args(argv)
    ns = T4.for_train_set(ns, ns.train_on_list[0])
    ns.compare_runs = []
    return ns


# =====================================================================================
# EXPERIMENT (subclass of the Test 4 experiment; only data paths, Stage-2 pair sampling and evaluation differ)
# =====================================================================================
class Experiment5(T4.Experiment):
    def __init__(self, args, a5, train_on: str):
        self.args, self.a5, self.train_on = args, a5, train_on
        self.script_dir = SCRIPT_DIR
        self.tag = T4.train_tag(train_on)
        self.root = a5.root
        self.data_dir = a5.root / "data" / self.tag
        self.cache_dir = a5.cache_dir
        self.test4_dir = a5.test4_dir
        self.baseline_dir = SCRIPT_DIR / "run_30-9-26"
        self.test3_dir = SCRIPT_DIR / "run_1-10-26"
        self.norm = args.norm
        self.wave = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_workers = args.workers or (os.cpu_count() or 8)
        self.T, self.L = args.frames, args.clip_len
        self.stride = max(1, self.T // self.L)
        self.work_dir = self.res = self.data_dir
        for d in (self.data_dir / "logs", self.cache_dir):
            d.mkdir(parents=True, exist_ok=True)
        self.status = defaultdict(lambda: "Not Completed")
        self.variant, self.train_seed, self.hard = None, args.seed, None
        self._lock = threading.Lock()
        self.sampling_log = Counter()
        self._deg_audio_spec = None
        self.localization = None

    # ------------------------------------------------------------------ helpers
    def test4_run_dir(self) -> Path:
        return self.test4_dir / f"4D_{self.tag}"

    def config(self) -> dict:
        c = super().config()
        c.update(test="Test 5", test5_variant=self.variant, test5_description=VARIANTS5.get(self.variant, {}).get("description"),
                 seed=self.train_seed, train_seed=self.train_seed, data_seed=self.a5.data_seed, hard_sampling=self.hard,
                 test4_baseline_dir=str(self.test4_run_dir()))
        return c

    # ------------------------------------------------------------------ data (Test 4 prepare_data + Test 4 caches + split check)
    def prepare_data(self):
        a = self.args
        print("=" * 96)
        print(f" 8-10-26 / TEST 5 data preparation | training set {self.train_on} | data seed {a.seed}")
        print(f"  * device {self.device}" + (f" ({torch.cuda.get_device_name(0)})" if self.device.type == "cuda" else ""))
        print(f"  * Test 4 reference: {self.test4_run_dir()} | caches searched in {self.test4_dir / 'cache'}")
        print("=" * 96)
        self.sources, self.folders = T4.discover_sources(a, random.Random(a.seed))
        if not self.sources:
            raise RuntimeError("No source videos found (check --bio-root / --ffpp-root / --datasets).")
        found, self.manifest_summary = T4.discover_degraded(self.sources, self.folders, a.extra_manifests)
        self.degraded, self.deg_report = T4.select_degraded(found, self.sources, a)
        split = T4.make_splits(self.sources, a.test_ratio, a.val_ratio, a.seed)
        T4.print_file_inventory(self.sources, self.folders, self.manifest_summary, self.degraded, self.deg_report, split, self.train_on)
        T4.write_json(self.data_dir / "file_inventory.json", dict(manifests=self.manifest_summary, selection=self.deg_report,
                                                                  n_originals=len(self.sources), n_degraded_used=len(self.degraded)))
        self.face_detector = T4.FaceDetector(T4.ensure_face_model(a.face_model, SCRIPT_DIR) if a.face_crop else None, a.face_score)
        self.labels = np.array([s["label"] for s in self.sources], dtype=np.int64)
        self.ds_of = np.array([s["dataset"] for s in self.sources])
        self.subset_of = np.array([f"{s['dataset']}/{s['subset']}" for s in self.sources])
        self.deg_src = np.array([d["src"] for d in self.degraded], dtype=np.int64)
        self.deg_type = np.array([d["degradation"] for d in self.degraded])

        print("\n--- Caches (Test 4 caches are reused read-only; identical keys = identical preprocessing) ---")
        src_paths = [str(s["path"]) for s in self.sources]
        deg_paths = [str(d["path"]) for d in self.degraded]
        search = [self.test4_dir / "cache", self.test3_dir / "cache", self.baseline_dir / "cache"]
        norm = self.norm
        if not a.face_crop:
            raise SystemExit("Test 5 keeps the Test 4 face-cropping pipeline (--no-face-crop is not allowed).")
        face_key = f"|face|m{a.face_margin}|a{int(a.face_align)}|s{a.face_score}|{self.face_detector.backend}"
        if norm:
            face_key += f"|norm-f{norm['frame_side']}-p{norm['face_px']}-j{norm['jpeg']}"
        clean = T4.ClipCache("faceclean", "\n".join(src_paths) + f"|{self.T}|{a.img_size}" + face_key, len(src_paths), self.T, a.img_size,
                             self.cache_dir, search, with_info=True)
        deg = T4.ClipCache("facedeg", "\n".join(deg_paths) + f"|{self.T}|{a.img_size}" + face_key + f"|boxes-{a.deg_face_boxes}",
                           len(deg_paths), self.T, a.img_size, self.cache_dir, search, with_info=True) if self.degraded else None
        self.check_disk_space([clean] + ([deg] if deg else []))
        det = self.face_detector
        clean.build(lambda i: T4.extract_face_clip(self.sources[i]["path"], self.T, a.img_size, det, a.face_margin, a.face_align, norm=norm),
                    self.n_workers, "ORIGINAL VIDEOS (face crop, resolution-normalised)")
        if deg is not None:
            def deg_work(j):
                s = int(self.deg_src[j])
                if not clean.ok[s]:
                    return None, None
                if a.deg_face_boxes == "source":
                    ref = (clean.info["boxes"][s], clean.info["angles"][s], bool(clean.info["meta"][s, 4] > 0.5))
                    return T4.extract_face_clip(self.degraded[j]["path"], self.T, a.img_size, None, a.face_margin, a.face_align, ref=ref, norm=norm)
                return T4.extract_face_clip(self.degraded[j]["path"], self.T, a.img_size, det, a.face_margin, a.face_align, norm=norm)

            deg.build(deg_work, self.n_workers, "DEGRADED VIDEOS (face crop)")
        self.view = T4.View(f"{a.img_size}_face", a.img_size, clean, deg, True)
        print("\n--- Audio ---")
        self.audio, self.has_audio = T4.build_audio_cache(self.sources, self.cache_dir, search, self.n_workers)
        if a.modality == "av" and self.has_audio.any():
            self.wave = T4.build_waveform_cache(self.sources, self.has_audio, self.cache_dir, search, self.n_workers)
        self.deg_level = np.array([T4.SEVERITY_LEVEL.get(str(d["severity"]).lower(), 1) for d in self.degraded], dtype=np.int64)

        self.split = np.where(clean.ok, split, "dropped")
        deg_ok = deg.ok if deg is not None else np.zeros(0, bool)
        self.src_to_deg = defaultdict(list)
        for j in range(len(self.degraded)):
            if deg_ok[j] and clean.ok[self.deg_src[j]]:
                self.src_to_deg[int(self.deg_src[j])].append(j)
        self.idx = {k: np.where(self.split == k)[0] for k in ("train", "val", "test")}
        if self.train_on != "combined":
            for k in ("train", "val"):
                self.idx[k] = self.idx[k][self.ds_of[self.idx[k]] == self.train_on]
        self.split_json = {k: [self.sources[i]["key"] for i in v] for k, v in self.idx.items()}
        T4.write_json(self.data_dir / "split_test5.json", self.split_json)
        self.verify_test4_split()
        self.verify_leakage()                    # Test 4 check (groups / pairs / FF++ identities)
        self.leakage_t5 = self.leakage_report()  # Test 5 extended check (stops on any leakage)
        self.report_inventory()
        self.report_face_stats()
        self.build_eval_sets()                   # Test 4 code, data seed -> same degraded test versions as Test 4
        self.write_manifest()
        self.setup_hard_pool()
        self.xai_test = self.build_xai_sample("test", self.a5.xai_per_type, 555)
        self.xai_val = self.build_xai_sample("val", self.a5.val_xai_per_type, 777)
        self.xai_examples = self.choose_examples()
        self.load_localization()
        self.check_5a_equivalence()
        return True

    def verify_test4_split(self):
        files = sorted(self.test4_run_dir().glob("*split_6-10-26.json"))
        res = dict(test4_split_file=str(files[0]) if files else None, identical=None, differences={})
        if files:
            t4 = json.load(open(files[0], encoding="utf-8"))
            same = True
            for k in ("train", "val", "test"):
                a_, b_ = set(t4.get(k, [])), set(self.split_json[k])
                res["differences"][k] = dict(test4=len(a_), test5=len(b_), only_test4=len(a_ - b_), only_test5=len(b_ - a_))
                same &= a_ == b_
            res["identical"] = bool(same)
        self.split_check = res
        msg = ("IDENTICAL to Test 4" if res["identical"] else "DIFFERENT from Test 4" if res["identical"] is False else
               "Test 4 split file not found")
        print(f"\n[Split check] {msg}: {res['differences'] or res['test4_split_file']}")
        T4.write_json(self.data_dir / "split_check_vs_test4.json", res)
        if res["identical"] is not True and not self.a5.skip_test4_split_check:
            raise SystemExit("[STOP] The Test 5 split is not verifiably identical to the Test 4 split. Test 5 must use exactly the "
                             "same split (do not pass --max-per-folder / other data options). For smoke tests only: --skip-test4-split-check")

    def _file_hashes(self, paths: List[Path], label: str) -> Dict[str, str]:
        """Cached partial (or full) hashes keyed by path|size|mtime."""
        cache_f = self.root / "data" / "hash_cache" / f"{'full' if self.a5.full_file_hash else 'partial'}_hashes.json"
        cache = json.load(open(cache_f, encoding="utf-8")) if cache_f.exists() else {}
        fn = T4.file_sha256 if self.a5.full_file_hash else partial_file_hash
        keys = {}
        todo = []
        for p in paths:
            try:
                st = os.stat(p)
            except OSError:
                continue
            k = f"{p}|{st.st_size}|{int(st.st_mtime)}"
            keys[str(p)] = k
            if k not in cache:
                todo.append((p, k))
        if todo:
            from concurrent.futures import ThreadPoolExecutor
            print(f"  hashing {len(todo):,} {label} files ({'full SHA-256' if self.a5.full_file_hash else 'size + first/last 64 KiB'}) ...")
            with ThreadPoolExecutor(max_workers=min(16, self.n_workers)) as ex:
                for (p, k), h in zip(todo, ex.map(lambda t: fn(t[0]), todo)):
                    cache[k] = h
            cache_f.parent.mkdir(parents=True, exist_ok=True)
            json.dump(cache, open(cache_f, "w", encoding="utf-8"))
        return {p: cache[k] for p, k in keys.items()}

    def _clip_hashes(self, idx_all: np.ndarray) -> Dict[int, str]:
        cache_f = self.root / "data" / "hash_cache" / f"clip_md5_{self.view.clean.fname}.json"
        cache = {int(k): v for k, v in json.load(open(cache_f, encoding="utf-8")).items()} if cache_f.exists() else {}
        todo = [int(i) for i in idx_all if int(i) not in cache and self.view.clean.ok[i]]
        if todo:
            print(f"  hashing {len(todo):,} cached face clips (decoded content) ...")
            for n, i in enumerate(todo):
                cache[i] = hashlib.md5(np.asarray(self.view.clean.arr[i]).tobytes()).hexdigest()
                if n % 500 == 0:
                    sys.stdout.write(f"\r  clip hashes {T4.progress_bar(n + 1, len(todo))}   ")
            sys.stdout.write("\n")
            cache_f.parent.mkdir(parents=True, exist_ok=True)
            json.dump({str(k): v for k, v in cache.items()}, open(cache_f, "w", encoding="utf-8"))
        return cache

    def leakage_report(self) -> dict:
        """Train/val/test identity overlap, duplicate file names, duplicate file / content hashes, pair leakage. Expected: 0."""
        idx = self.idx
        print("\n[Leakage report - Test 5]")
        grp = {k: {self.sources[i]["group"] for i in v} for k, v in idx.items()}
        tok = {k: {t for i in v if self.ds_of[i] == "FF++" for t in T4.ffpp_identity_tokens(self.sources[i]["path"].stem, self.sources[i]["subset"])}
               for k, v in idx.items()}
        r = dict(train_test_identity_overlap=len(grp["train"] & grp["test"]), val_test_identity_overlap=len(grp["val"] & grp["test"]),
                 train_val_identity_overlap=len(grp["train"] & grp["val"]),
                 ffpp_train_test_identity_token_overlap=len(tok["train"] & tok["test"]),
                 ffpp_val_test_identity_token_overlap=len(tok["val"] & tok["test"]))
        where = defaultdict(set)
        for k, v in idx.items():
            for i in v:
                where[(self.ds_of[i], self.sources[i]["path"].name.lower())].add(k)
        dup_names = [f"{d}/{n}" for (d, n), s in where.items() if len(s) > 1]
        r["duplicate_filenames_across_splits"] = len(dup_names)
        split_of = {}
        for k, v in idx.items():
            for i in v:
                split_of[int(i)] = k
        bad_pair, bad_folder = 0, 0
        for i, k in split_of.items():
            for j in self.src_to_deg[i]:
                bad_pair += int(self.split[self.deg_src[j]] != self.split[i])
                dp = Path(self.degraded[j]["path"])
                bad_folder += int(str(self.sources[i]["folder"]).lower() not in str(dp).lower() and dp.parent.name != "degraded_output"
                                  and "degraded" not in str(dp).lower())
        r["original_degraded_pair_split_mismatch"] = bad_pair
        r["degraded_file_outside_source_folder"] = bad_folder
        # degraded videos must not have been created before the split: every degraded item inherits its source's split here,
        # and a degraded file may not be byte-identical to a video of another split (checked below)
        r["duplicate_file_hashes_across_splits"] = r["duplicate_clip_hashes_across_splits"] = None
        r["duplicate_file_hashes_within_split"] = r["duplicate_clip_hashes_within_split"] = None
        if not self.a5.skip_hash_check:
            files = []
            for i, k in split_of.items():
                files.append((Path(self.sources[i]["path"]), k, f"orig:{self.sources[i]['key']}"))
                for j in self.src_to_deg[i]:
                    files.append((Path(self.degraded[j]["path"]), k, f"deg:{self.degraded[j]['path'].name}"))
            hashes = self._file_hashes([f[0] for f in files], "video")
            by_hash = defaultdict(list)
            for p, k, name in files:
                if str(p) in hashes:
                    by_hash[hashes[str(p)]].append((p, k, name))
            cross, within = [], 0
            for h, lst in by_hash.items():
                if len(lst) < 2:
                    continue
                if len({k for _, k, _ in lst}) > 1:
                    full = {T4.file_sha256(p) for p, _, _ in lst} if not self.a5.full_file_hash else {h}
                    if len(full) == 1:
                        cross.append([n for _, _, n in lst])
                else:
                    within += 1
            r["duplicate_file_hashes_across_splits"] = len(cross)
            r["duplicate_file_hashes_within_split"] = within
            r["duplicate_file_examples"] = cross[:10]
            ch = self._clip_hashes(np.array(sorted(split_of)))
            by_c = defaultdict(set)
            cnt = Counter()
            for i, k in split_of.items():
                if i in ch:
                    by_c[ch[i]].add(k)
                    cnt[ch[i]] += 1
            r["duplicate_clip_hashes_across_splits"] = sum(len(s) > 1 for s in by_c.values())
            r["duplicate_clip_hashes_within_split"] = sum(1 for h, s in by_c.items() if len(s) == 1 and cnt[h] > 1)
        leak_keys = ["train_test_identity_overlap", "val_test_identity_overlap", "train_val_identity_overlap",
                     "ffpp_train_test_identity_token_overlap", "ffpp_val_test_identity_token_overlap", "duplicate_filenames_across_splits",
                     "original_degraded_pair_split_mismatch", "duplicate_file_hashes_across_splits", "duplicate_clip_hashes_across_splits"]
        r["leakage_count"] = int(sum(int(r[k] or 0) for k in leak_keys))
        r["duplicate_filename_examples"] = dup_names[:10]
        r["note"] = ("Identity overlap uses the Test 4 identity groups (BioDeepAV real: YouTube id, BioDeepAV fake: file, FF++: union of "
                     "both ids of every pair). Hash screens detect byte-identical files and identical decoded face clips; re-encoded "
                     "near-duplicates are NOT detected. Within-split duplicates are reported but are not leakage.")
        for k in leak_keys + ["duplicate_file_hashes_within_split", "duplicate_clip_hashes_within_split", "degraded_file_outside_source_folder"]:
            print(f"  {k:48s} {r[k]}")
        print(f"  LEAKAGE COUNT = {r['leakage_count']} (expected 0)")
        T4.write_json(self.data_dir / "leakage_report.json", r)
        if r["leakage_count"] > 0:
            raise SystemExit(f"[STOP] Data leakage detected - see {self.data_dir / 'leakage_report.json'}")
        return r

    def write_manifest(self):
        rows = []
        for k, v in self.idx.items():
            for i in v:
                s = self.sources[i]
                rows.append(dict(split=k, kind="original", key=s["key"], dataset=s["dataset"], subset=s["subset"], label=s["label"],
                                 group=s["group"], degradation="", severity="", path=str(s["path"])))
                for j in self.src_to_deg[int(i)]:
                    d = self.degraded[j]
                    rows.append(dict(split=k, kind="degraded", key=s["key"], dataset=s["dataset"], subset=s["subset"], label=s["label"],
                                     group=s["group"], degradation=d["degradation"], severity=d["severity"], path=str(d["path"])))
        df = pd.DataFrame(rows)
        df.to_csv(self.data_dir / "dataset_manifest_test5.csv", index=False)
        self.manifest_sha256 = T4.file_sha256(self.data_dir / "dataset_manifest_test5.csv")

    # ------------------------------------------------------------------ hard-degradation sampling
    def setup_hard_pool(self):
        self.cell_of = np.array([f"{d}|{'real' if l == 0 else 'fake'}" for d, l in zip(self.ds_of, self.labels)])
        pool = defaultdict(list)
        for s in self.idx["train"]:
            for j in self.src_to_deg[int(s)]:
                pool[(self.cell_of[s], self.deg_type[j])].append(j)
        self.hard_pool = {k: np.array(sorted(v), dtype=np.int64) for k, v in pool.items()}

    def _draw_hard(self, cell: str, rng, max_level: int) -> Optional[int]:
        w = self.hard["weights"]
        types = [t for t in w if len(self.hard_pool.get((cell, t), ()))]
        if not types:
            return None
        pw = np.array([w[t] for t in types], dtype=np.float64)
        t = types[int(rng.choice(len(types), p=pw / pw.sum()))]
        pool = self.hard_pool[(cell, t)]
        lv = self.deg_level[pool]
        cand = pool[lv <= max_level]
        if not len(cand):  # curriculum: no version of this type at an allowed severity -> mildest available (Test 4 rule)
            cand = pool[lv == lv.min()]
        return int(cand[int(rng.integers(len(cand)))])

    def make_pair_batch(self, view, src_list, seed, modality, max_level=2):
        """Test 4 paired batch; hard variants replace the degraded member (and its original) of a pair with probability
        hard_share by a hard-type degraded video of the same dataset x class cell. Hard draws use a separate RNG
        stream, so with hard sampling off the batch is bit-identical to Test 4 (verified in check_5a_equivalence)."""
        h = self.hard or {}
        rng_h = np.random.default_rng([int(seed), 0x5EED])
        pairs = []
        for s in src_list:
            s, j, mode = int(s), None, "normal"
            if h.get("enabled") and rng_h.random() < h["share"]:
                jj = self._draw_hard(self.cell_of[s], rng_h, max_level)
                if jj is not None:
                    j, s, mode = jj, int(self.deg_src[jj]), "hard"
            pairs.append((s, j, mode))
        return self._pair_batch(view, pairs, seed, modality, max_level)

    def _pair_batch(self, view, pairs, seed, modality, max_level):
        """Copy of Test 4 Experiment.make_pair_batch with an optional fixed degraded version per pair."""
        rng = np.random.default_rng(seed)
        B, S = len(pairs), view.img_size
        fc = np.empty((B, self.L, S, S, 3), dtype=np.uint8)
        fd = np.empty((B, self.L, S, S, 3), dtype=np.uint8)
        pad_px = max(2, S // 24)
        log = Counter()
        nonhard = bool(self.hard and self.hard.get("enabled") and self.hard.get("normal_pool") == "nonhard")
        for b, (s, j, mode) in enumerate(pairs):
            sel = np.sort(rng.choice(self.T, self.L, replace=False))
            clean = np.asarray(view.clean.arr[s, sel])
            if j is None:
                versions = self.src_to_deg[s] if view.deg is not None else []
                if versions and nonhard:
                    versions = [v for v in versions if self.deg_type[v] not in self.hard["weights"]] or versions
                if versions:
                    cands = [v for v in versions if self.deg_level[v] <= max_level]
                    if not cands:
                        lo = min(self.deg_level[v] for v in versions)
                        cands = [v for v in versions if self.deg_level[v] == lo]
                    j = cands[int(rng.integers(len(cands)))]
                    dclip = np.asarray(view.deg.arr[j, sel])
                elif self.args.synth_per_video > 0:
                    dg, sv = T4.random_degradation_spec(rng, max_level=max_level)
                    dclip = T4.synth_degrade(clean, dg, sv, rng)
                else:
                    dclip = clean
            else:
                dclip = np.asarray(view.deg.arr[j, sel])
            log[(int(max_level), mode, str(self.deg_type[j]) if j is not None else "none")] += 1
            flip = rng.random() < 0.5
            dy, dx = rng.integers(0, 2 * pad_px + 1, 2)
            for out, clip in ((fc, clean), (fd, dclip)):
                if flip:
                    clip = clip[:, :, ::-1, :]
                pad = np.pad(clip, ((0, 0), (pad_px, pad_px), (pad_px, pad_px), (0, 0)), mode="edge")
                out[b] = pad[:, dy : dy + S, dx : dx + S]
        src = np.asarray([p[0] for p in pairs], dtype=np.int64)
        audio, a_mask = self.sample_audio(src, rng, modality, max_level)
        with self._lock:
            self.sampling_log.update(log)
        return fc, fd, audio, a_mask, self.labels[src]

    def check_5a_equivalence(self):
        """5A must train on exactly the Test 4D batches: compare against the unmodified Test 4 method."""
        tr = [int(s) for s in self.idx["train"] if self.src_to_deg[int(s)]][:6]
        if len(tr) < 2:
            self.equivalence_5a = None
            return
        old = self.hard
        self.hard = dict(enabled=False)
        ok = True
        for seed in (1, 2):
            for ml in (0, 2):
                a_ = T4.Experiment.make_pair_batch(self, self.view, tr, seed, "av", ml)
                b_ = self.make_pair_batch(self.view, tr, seed, "av", ml)
                ok &= all(np.array_equal(x, y) for x, y in zip(a_, b_))
        self.hard = old
        self.sampling_log.clear()
        self.equivalence_5a = bool(ok)
        print(f"\n[5A equivalence] Test 5 pair batches with hard sampling OFF == Test 4 make_pair_batch: {ok}")
        if not ok:
            raise SystemExit("[STOP] 5A batches differ from Test 4D batches - the control would not reproduce Test 4D.")

    def expected_type_distribution(self) -> pd.DataFrame:
        """Expected share of each degradation type among Stage-2 degraded members (all severities, cells equally balanced)."""
        items = [("clean", int(s), None) for s in self.idx["train"]]
        _, dist = self.sampling_weights(items, self.args.sampling)
        share = {r["cell"]: r["sampled_share_%"] / 100 for _, r in dist.iterrows()} if self.args.sampling == "dataset_class" else None
        cells = sorted({self.cell_of[s] for s in self.idx["train"]})
        tot = Counter()
        rows = []
        for c in cells:
            srcs = [int(s) for s in self.idx["train"] if self.cell_of[s] == c and self.src_to_deg[int(s)]]
            nat = Counter()
            for s in srcs:
                vs = self.src_to_deg[s]
                if self.hard and self.hard.get("enabled") and self.hard.get("normal_pool") == "nonhard":
                    vs = [v for v in vs if self.deg_type[v] not in self.hard["weights"]] or vs
                for v in vs:
                    nat[self.deg_type[v]] += 1 / len(vs) / max(1, len(srcs))
            exp = Counter(nat)
            if self.hard and self.hard.get("enabled"):
                avail = {t: w for t, w in self.hard["weights"].items() if len(self.hard_pool.get((c, t), ()))}
                z = sum(avail.values())
                exp = Counter({t: (1 - self.hard["share"]) * nat.get(t, 0) for t in set(nat) | set(avail)})
                for t, w in avail.items():
                    exp[t] += self.hard["share"] * w / z if z else 0
            cw = share.get(c, 1 / len(cells)) if share else 1 / len(cells)
            for t, v in exp.items():
                rows.append(dict(cell=c, degradation=t, natural_share_pct=100 * nat.get(t, 0), expected_share_pct=100 * v))
                tot[t] += cw * v
        df = pd.DataFrame(rows)
        for t, v in tot.items():
            df = pd.concat([df, pd.DataFrame([dict(cell="ALL (sampling-weighted)", degradation=t, expected_share_pct=100 * v)])], ignore_index=True)
        return df

    # ------------------------------------------------------------------ evaluation samples
    def build_xai_sample(self, split: str, per_type: int, off: int) -> pd.DataFrame:
        """Fixed Grad-CAM pairs, stratified by (dataset, degradation type), chosen with the DATA seed (same for every model)."""
        e = self.E[split]
        pos_of_src = {int(s): k for k, s in enumerate(e["idx"])}
        rng = np.random.default_rng(self.args.seed + off)
        groups = defaultdict(list)
        for q, it in enumerate(e["deg_all"]):
            s = self.item_src(it)
            groups[(self.ds_of[s], self.item_degradation(it)[0])].append(q)
        rows = []
        for key in sorted(groups):
            g = np.array(groups[key])
            for q in sorted(rng.choice(g, size=min(per_type, len(g)), replace=False).tolist()):
                it = e["deg_all"][q]
                s = self.item_src(it)
                dg, sv = self.item_degradation(it)
                rows.append(dict(pair_id=int(q), k=pos_of_src[s], video=self.sources[s]["key"], dataset=self.ds_of[s],
                                 true_label=int(self.labels[s]), degradation=dg, severity=T4.SEVERITY_NAME.get(str(sv), str(sv)),
                                 degraded_file=self.item_name(it)))
        df = pd.DataFrame(rows)
        df.to_csv(self.data_dir / f"xai_sample_{split}.csv", index=False)
        return df

    def choose_examples(self) -> List[int]:
        """Grad-CAM example pairs, fixed before any model is evaluated: first 2 in-domain pairs per hard type (data-seed order)."""
        d = self.xai_test
        dom = d[d.dataset == self.train_on] if self.train_on != "combined" else d
        out = []
        for t in ["brightness", "saturation", "contrast", "blur"]:
            out += dom[dom.degradation == t].pair_id.head(2).tolist()
        return out

    def load_localization(self):
        f = self.a5.localization_csv
        if not f:
            return
        df = pd.read_csv(f)
        need = {"video", "x0", "y0", "x1", "y1"}
        if not need <= set(df.columns):
            raise SystemExit(f"--localization-csv needs columns {sorted(need)}")
        self.localization = {r.video: (r.x0, r.y0, r.x1, r.y1) for r in df.itertuples()}
        print(f"  localisation annotations: {len(self.localization)} videos")

    # ------------------------------------------------------------------ runs
    def set_run(self, args, variant: str, seed: int):
        self.args, self.variant, self.train_seed = args, variant, int(seed)
        self.run_name = f"{variant}_{self.tag}_seed{seed}"
        self.work_dir = self.root / "runs" / self.run_name
        self.res = self.work_dir / "results"
        for d in [self.res / s / sub for s in ("stage1", "stage2") for sub in ("checkpoints", "metrics", "predictions", "confusion_matrices", "threshold")] + \
                 [self.res / d for d in ("final", "logs")] + [self.work_dir / "eval"]:
            d.mkdir(parents=True, exist_ok=True)
        self.ckpt = {1: self.res / "stage1" / "checkpoints" / "stage1_best.pt", 2: self.res / "stage2" / "checkpoints" / "stage2_best.pt"}
        self.status = defaultdict(lambda: "Not Completed")
        self.stage_info = {}
        self.hard = dict(enabled=VARIANTS5[variant]["hard"], share=self.a5.hard_share, weights=dict(self.a5.hard_weight_map),
                         normal_pool=self.a5.normal_pool)
        self.sampling_log = Counter()
        T4.set_seed(self.train_seed, args.deterministic)

    def train_stage(self, *a, **k):
        """Test 4 training loop, run with the TRAINING seed (the data seed stays in args.seed everywhere else)."""
        old = self.args.seed
        self.args.seed = self.train_seed
        try:
            return super().train_stage(*a, **k)
        finally:
            self.args.seed = old

    def stage1_compatible(self, ck: Path) -> bool:
        try:
            cfg = torch.load(ck, map_location="cpu", weights_only=False).get("config", {})
        except Exception:
            return False
        mine = self.config()
        keys = ["img_size", "frames", "clip_len", "modality", "face_crop", "face_align", "face_margin", "face_score", "deg_face_boxes", "sampling",
                "max_oversample", "epoch_size", "batch_size", "lr1", "epochs1", "patience", "datasets", "ffpp_folders", "max_per_folder", "test_ratio",
                "val_ratio", "train_on", "deg_per_source", "res_norm_params", "audio_policy", "audio_probs_clean_degraded_none"]
        diff = {k: (cfg.get(k), mine.get(k)) for k in keys if T4.to_jsonable(cfg.get(k)) != T4.to_jsonable(mine.get(k))}
        if int(cfg.get("seed", -1)) != self.train_seed:
            diff["seed"] = (cfg.get("seed"), self.train_seed)
        if diff:
            print(f"  [Stage 1] Test 4D Stage-1 config differs -> not reused: {diff}")
        return not diff

    def ensure_stage1(self):
        shared = self.root / "stage1_shared" / f"{self.tag}_seed{self.train_seed}" / "stage1_best.pt"
        if self.stage_done(1):
            return
        if shared.exists():
            shutil.copy(shared, self.ckpt[1])
            self.status["stage1"] = "Completed (shared across Test 5 variants)"
            self.stage1_source = str(shared)
            print(f"\n[Stage 1 shared] {shared}")
            return
        t4 = self.test4_run_dir() / "results" / "stage1" / "checkpoints" / "stage1_best.pt"
        if self.train_seed == self.a5.data_seed and not self.a5.no_test4_stage1 and t4.exists() and self.stage1_compatible(t4):
            obj = torch.load(t4, map_location="cpu", weights_only=False)
            if obj.get("threshold") is not None:
                shared.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(t4, shared)  # read from Test 4, written only into the Test 5 directory
                shutil.copy(shared, self.ckpt[1])
                T4.write_json(shared.parent / "SOURCE.json", dict(copied_from=str(t4), sha256=T4.file_sha256(t4),
                                                                   copied_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
                self.status["stage1"] = "Completed (Test 4D Stage 1 reused, identical configuration)"
                self.stage1_source = str(t4)
                print(f"\n[Stage 1] identical configuration -> Test 4D Stage-1 checkpoint reused (read-only copy): {t4}")
                return
        self.run_stage1()
        shared.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(self.ckpt[1], shared)
        self.stage1_source = "trained in " + self.run_name

    def run_variant(self) -> dict:
        a = self.args
        done = self.work_dir / "eval" / "summary.json"
        if done.exists() and not self.a5.no_resume and (self.res / "final" / "final_model.pt").exists():
            print(f"\n[resume] {self.run_name} already evaluated -> skipped")
            return json.load(open(done, encoding="utf-8"))
        self.stage1_source = None
        print("\n" + "#" * 96 + f"\n# TEST {self.variant} | train {self.train_on} | seed {self.train_seed}\n# {VARIANTS5[self.variant]['description']}\n"
              f"# lambda_feat {a.lambda_feat} | lambda_cam {a.lambda_cam} | hard sampling {self.hard['enabled']}"
              + (f" (share {self.hard['share']}, {self.hard['weights']}, normal pool {self.hard['normal_pool']})" if self.hard["enabled"] else "")
              + "\n" + "#" * 96)
        T4.write_json(self.work_dir / "config.json", self.config())
        exp_dist = self.expected_type_distribution()
        exp_dist.to_csv(self.res / "logs" / "hard_sampling_expected.csv", index=False)
        allrow = exp_dist[exp_dist.cell.str.startswith("ALL")].sort_values("expected_share_pct", ascending=False)
        print("  * Expected Stage-2 degradation-type shares: " + ", ".join(f"{r.degradation} {r.expected_share_pct:.1f}%" for r in allrow.itertuples()))
        t0 = time.time()
        self.ensure_stage1()
        trained2 = False
        if not self.stage_done(2):
            T4.set_seed(self.train_seed, a.deterministic)
            self.sampling_log.clear()
            self.run_stage2()
            trained2 = True
        else:
            self.status["stage2"] = "Completed (resumed)"
        if trained2:
            rl = pd.DataFrame([dict(curriculum_max_level=k[0], mode=k[1], degradation=k[2], pairs=v) for k, v in self.sampling_log.items()])
            if len(rl):
                rl["share_pct"] = 100 * rl.pairs / rl.pairs.sum()
                rl.to_csv(self.res / "logs" / "hard_sampling_realised.csv", index=False)
                tt = rl.groupby("degradation").pairs.sum()
                print("  * Realised Stage-2 degradation-type shares: " + ", ".join(f"{k} {100 * v / tt.sum():.1f}%" for k, v in tt.sort_values(ascending=False).items()))
        frozen = self.freeze_final()
        train_time = time.time() - t0
        model, meta = T4.load_checkpoint(self.res / "final" / "final_model.pt", self.device)
        info = dict(model=self.variant, variant=self.variant, train_on=self.train_on, seed=self.train_seed, run=self.run_name,
                    threshold=float(meta["threshold"]), final_checkpoint=str(self.res / "final" / "final_model.pt"),
                    checkpoint_sha256=frozen["sha256"] if frozen else None, weights_sha256=weights_sha256(model.state_dict()),
                    architecture_signature=architecture_signature(model), n_parameters=int(sum(p.numel() for p in model.parameters())),
                    stage1_source=self.stage1_source, stage1_status=self.status["stage1"], stage2_status=self.status["stage2"],
                    stage2_train_time_sec=(self.stage_info.get(2) or {}).get("train_time"),
                    stage1_train_time_sec=(self.stage_info.get(1) or {}).get("train_time"), wall_time_sec=train_time,
                    best_epoch=meta.get("epoch"), best_val_selection_score=meta.get("best_val_metric"),
                    lambda_feat=a.lambda_feat, lambda_cam=a.lambda_cam, hard_sampling=self.hard, config=self.config())
        hist = self.res / "logs" / "history_stage2.csv"
        if hist.exists():
            shutil.copy(hist, self.work_dir / "training_history.csv")
        summary = self.t5_evaluate(self.variant, model, info["threshold"], self.work_dir / "eval", info)
        return summary

    def evaluate_test4_reference(self) -> Optional[dict]:
        """Frozen Test 4D final model + its frozen threshold on the SAME Test 5 evaluation inputs (Test 4 files are only read)."""
        ck = self.test4_run_dir() / "results" / "final" / "final_model.pt"
        out = self.root / "test4_reference" / self.tag / "eval"
        if (out / "summary.json").exists() and not self.a5.no_resume:
            print(f"\n[resume] Test 4D reference ({self.tag}) already evaluated")
            return json.load(open(out / "summary.json", encoding="utf-8"))
        if not ck.exists():
            print(f"\n[Test 4D reference] {ck} not found -> Test 4 column will be missing")
            return None
        model, meta = T4.load_checkpoint(ck, self.device)
        if architecture_signature(model) != architecture_signature(T4.AudioVisualDeepFakeCNN()):
            raise SystemExit("[STOP] Test 4D checkpoint architecture differs from AudioVisualDeepFakeCNN")
        self.variant, self.train_seed = "T4D", int(meta.get("seed") or 42)
        print("\n" + "#" * 96 + f"\n# TEST 4D REFERENCE (frozen Test 4 model, threshold {meta['threshold']:.3f}) | {self.tag}\n" + "#" * 96)
        info = dict(model="T4D", variant="T4D", train_on=self.train_on, seed=self.train_seed, run=f"T4D_{self.tag}",
                    threshold=float(meta["threshold"]), final_checkpoint=str(ck), checkpoint_sha256=T4.file_sha256(ck),
                    weights_sha256=weights_sha256(model.state_dict()), architecture_signature=architecture_signature(model),
                    n_parameters=int(sum(p.numel() for p in model.parameters())), config=meta.get("config"),
                    best_epoch=meta.get("epoch"), best_val_selection_score=meta.get("best_val_metric"))
        return self.t5_evaluate("T4D", model, info["threshold"], out, info)

    # ------------------------------------------------------------------ prediction helpers
    @torch.no_grad()
    def predict_audio(self, model, items, spec: np.ndarray, mask: np.ndarray, net_mode="av", bs=32) -> np.ndarray:
        model.eval()
        out = []
        for i in range(0, len(items), bs):
            chunk = items[i : i + bs]
            clips = np.stack([self.get_clip(self.view, it) for it in chunk])
            xa = torch.from_numpy(np.ascontiguousarray(spec[i : i + bs], dtype=np.float32)).to(self.device)
            am = torch.from_numpy(np.ascontiguousarray(mask[i : i + bs], dtype=np.float32)).to(self.device)
            p = 0
            for v in range(self.stride):
                logits = model(T4.to_video_tensor(clips[:, v :: self.stride][:, : self.L], self.device), xa, mode=net_mode, a_mask=am)
                p = p + torch.softmax(logits.float(), 1)[:, 1].cpu().numpy()
            out.append(p / self.stride)
        return np.concatenate(out) if out else np.zeros(0)

    def degraded_audio_spec(self) -> np.ndarray:
        """One random audio degradation per test video (data seed, Test 4 recipe) - identical for every model."""
        if self._deg_audio_spec is None:
            idx = self.E["test"]["idx"]
            spec = np.array(self.audio[idx], dtype=np.float32)
            if self.wave is not None:
                for k, s in enumerate(idx):
                    if self.has_audio[s]:
                        rng = np.random.default_rng(self.args.seed * 7919 + int(s))
                        dg, sv = T4.random_audio_degradation_spec(rng)
                        spec[k] = T4.waveform_to_spec(T4.degrade_audio(self.wave_float(int(s)), dg, sv, rng))
            self._deg_audio_spec = spec
        return self._deg_audio_spec

    def xai_eval(self, model, thr: float, sample: pd.DataFrame, split: str, example_ids=()) -> Tuple[pd.DataFrame, dict]:
        e = self.E[split]
        rows, ex = [], {}
        loc = self.localization
        for b0 in range(0, len(sample), 16):
            ch = sample.iloc[b0 : b0 + 16]
            ci = [e["clean"][k] for k in ch.k]
            di = [e["deg_all"][q] for q in ch.pair_id]
            src = np.array([self.item_src(it) for it in ci])
            fcl = [self.get_clip(self.view, it) for it in ci]
            fdl = [self.get_clip(self.view, it) for it in di]
            cc = np.stack([c[:: self.stride][: self.L] for c in fcl])
            c1 = np.stack([c[1 :: self.stride][: self.L] for c in fcl]) if self.stride > 1 else cc
            dc = np.stack([c[:: self.stride][: self.L] for c in fdl])
            xa, am = self.audio_inputs(src, "av")
            gc = T4.grad_cam(model, T4.to_video_tensor(cc, self.device), xa, am, 1, thr)
            gd = T4.grad_cam(model, T4.to_video_tensor(dc, self.device), xa, am, 1, thr)
            g1 = T4.grad_cam(model, T4.to_video_tensor(c1, self.device), xa, am, 1, thr)
            for b, r in enumerate(ch.itertuples()):
                A, D, R = gc["spatial_cam"][b], gd["spatial_cam"][b], g1["spatial_cam"][b]
                pr, iou, sh = T4.cam_metrics(A, D)
                rr, iou_ref, _ = T4.cam_metrics(A, R)
                ta, td = gc["temporal_cam"][b], gd["temporal_cam"][b]
                tc = float(np.corrcoef(ta, td)[0, 1]) if ta.std() > 1e-8 and td.std() > 1e-8 else np.nan
                stable = bool(gc["prediction"][b] == gd["prediction"][b])
                row = dict(pair_id=int(r.pair_id), video=r.video, dataset=r.dataset, true_label=int(r.true_label), degradation=r.degradation,
                           severity=r.severity, threshold=thr, p_fake_original=float(gc["probability_fake"][b]),
                           p_fake_degraded=float(gd["probability_fake"][b]), prediction_original=int(gc["prediction"][b]),
                           prediction_degraded=int(gd["prediction"][b]), prediction_stable=stable, pearson_r=pr, top20_iou=iou,
                           center_shift=sh, temporal_cam_r=tc, explanation_changed=bool(iou < EXPLANATION_CHANGE_IOU),
                           clean_ref_iou=iou_ref, clean_ref_pearson_r=rr)
                if loc is not None and r.video in loc and int(r.true_label) == 1:
                    for nm, cam in (("original", A), ("degraded", D)):
                        row.update({f"loc_{nm}_{k}": v for k, v in localization_scores(cam, loc[r.video]).items()})
                rows.append(row)
                if int(r.pair_id) in example_ids:
                    ex[int(r.pair_id)] = dict(frame_original=cc[b, self.L // 2], frame_degraded=dc[b, self.L // 2], cam_original=A,
                                              cam_degraded=D, p_original=float(gc["probability_fake"][b]),
                                              p_degraded=float(gd["probability_fake"][b]), iou=iou, degradation=r.degradation, video=r.video)
        return pd.DataFrame(rows), ex

    # ------------------------------------------------------------------ evaluation
    def t5_evaluate(self, tag: str, model, thr: float, out: Path, info: dict) -> dict:
        out.mkdir(parents=True, exist_ok=True)
        meta = dict(model=tag, train_on=self.train_on, seed=info.get("seed"))
        print(f"\n  [eval {tag} | {self.tag} | seed {info.get('seed')}] predictions (val + test, original + every degraded version) ...")
        P = self.predict_sets(model, self.view, self.args.modality)
        e = self.E["test"]
        idx = e["idx"]
        y, ds, hv = self.labels[idx], self.ds_of[idx], e["pick"] >= 0
        pos_of_src = {int(s): k for k, s in enumerate(idx)}
        src_all = np.array([self.item_src(it) for it in e["deg_all"]], dtype=np.int64)
        k_all = np.array([pos_of_src[int(s)] for s in src_all], dtype=np.int64)
        typ_all = np.array([self.item_degradation(it)[0] for it in e["deg_all"]])
        sev_all = np.array([T4.SEVERITY_NAME.get(str(self.item_degradation(it)[1]), str(self.item_degradation(it)[1])) for it in e["deg_all"]])
        y_all, ds_all = self.labels[src_all], self.ds_of[src_all]
        pc, pdg, pall = P["test"]["clean"], P["test"]["deg"], P["test"]["deg_all"]
        flip_all = (pall >= thr) != (pc[k_all] >= thr)
        dsets = ["Combined"] + sorted(set(ds))
        dom = self.train_on if self.train_on != "combined" else "Combined"
        setting = lambda dn: "mixed" if dn == "Combined" else ("in-domain" if self.train_on in ("combined", dn) else "cross-dataset")

        def mrow(yy, pp):
            m = T4.compute_metrics(yy, pp, thr)
            pp = np.asarray(pp, dtype=np.float64)
            yy = np.asarray(yy)
            m.update(sensitivity=m["recall"], fpr=m["fp"] / (m["fp"] + m["tn"]) if (m["fp"] + m["tn"]) else np.nan,
                     fnr=m["fn"] / (m["fn"] + m["tp"]) if (m["fn"] + m["tp"]) else np.nan,
                     mean_p_fake=float(np.nanmean(pp)) if len(pp) else np.nan, pred_fake_rate=float(np.mean(pp >= thr)) if len(pp) else np.nan,
                     mean_p_fake_real=float(np.nanmean(pp[yy == 0])) if (yy == 0).any() else np.nan,
                     mean_p_fake_fake=float(np.nanmean(pp[yy == 1])) if (yy == 1).any() else np.nan)
            return m

        det = []
        for dn in dsets:
            sel = np.ones(len(idx), bool) if dn == "Combined" else ds == dn
            sa = np.ones(len(src_all), bool) if dn == "Combined" else ds_all == dn
            for cond, yy, pp in (("original", y[sel], pc[sel]), ("degraded", y[sel & hv], pdg[sel & hv]), ("degraded_all_versions", y_all[sa], pall[sa])):
                det.append(dict(meta, dataset=dn, setting=setting(dn), condition=cond, **mrow(yy, pp)))
        det = pd.DataFrame(det)
        rob = []
        for dn in dsets:
            sel = (np.ones(len(idx), bool) if dn == "Combined" else ds == dn) & hv
            sa = np.ones(len(src_all), bool) if dn == "Combined" else ds_all == dn
            mo, md = T4.compute_metrics(y[sel], pc[sel], thr), T4.compute_metrics(y[sel], pdg[sel], thr)
            ma = T4.compute_metrics(y_all[sa], pall[sa], thr)
            rob.append(dict(meta, dataset=dn, setting=setting(dn), n=int(sel.sum()), n_versions=int(sa.sum()), clean_auc=mo["auc"],
                            degraded_auc=md["auc"], auc_drop=mo["auc"] - md["auc"], degraded_all_versions_auc=ma["auc"],
                            auc_drop_all_versions=mo["auc"] - ma["auc"], clean_accuracy=mo["accuracy"], degraded_accuracy=md["accuracy"],
                            clean_balanced_accuracy=mo["balanced_accuracy"], degraded_balanced_accuracy=md["balanced_accuracy"],
                            prediction_flip_rate_pct=100 * float(np.mean((pc[sel] >= thr) != (pdg[sel] >= thr))) if sel.any() else np.nan,
                            prediction_flip_rate_all_versions_pct=100 * float(flip_all[sa].mean()) if sa.any() else np.nan,
                            mean_abs_prob_shift=float(np.abs(pc[sel] - pdg[sel]).mean()) if sel.any() else np.nan))
        rob = pd.DataFrame(rob)
        degr = []
        for dn in dsets:
            dmask = np.ones(len(src_all), bool) if dn == "Combined" else ds_all == dn
            for t in sorted(set(typ_all)):
                for sv in ("all", "mild", "medium", "severe"):
                    sel = dmask & (typ_all == t) & ((sev_all == sv) if sv != "all" else True)
                    if not sel.any():
                        continue
                    m = T4.compute_metrics(y_all[sel], pall[sel], thr)
                    degr.append(dict(meta, dataset=dn, setting=setting(dn), degradation=t, severity=sv, n=int(sel.sum()), auc=m["auc"],
                                     clean_auc_same_videos=fast_auc(y_all[sel], pc[k_all][sel]), accuracy=m["accuracy"],
                                     balanced_accuracy=m["balanced_accuracy"], precision=m["precision"], recall=m["recall"],
                                     specificity=m["specificity"], f1=m["f1"], tn=m["tn"], fp=m["fp"], fn=m["fn"], tp=m["tp"],
                                     prediction_flip_rate_pct=100 * float(flip_all[sel].mean()),
                                     mean_abs_prob_shift=float(np.abs(pall[sel] - pc[k_all][sel]).mean())))
        degr = pd.DataFrame(degr)

        print(f"  [eval {tag}] Grad-CAM on {len(self.xai_test)} fixed test pairs (+ {len(self.xai_val)} validation pairs) ...")
        xai, ex = self.xai_eval(model, thr, self.xai_test, "test", set(self.xai_examples))
        for k, v in meta.items():
            xai.insert(0, k, v)
        xv, _ = self.xai_eval(model, thr, self.xai_val, "val")

        print(f"  [eval {tag}] audio ablation ...")
        spec = np.array(self.audio[idx], dtype=np.float32)
        has = self.has_audio[idx]
        mask = has.astype(np.float32)
        conds = {"normal_audio": pc,
                 "zero_audio": self.predict_audio(model, e["clean"], np.zeros_like(spec), mask),
                 "degraded_audio": self.predict_audio(model, e["clean"], self.degraded_audio_spec(), mask),
                 "video_only": self.predict_audio(model, e["clean"], spec, np.zeros_like(mask)),
                 "audio_only": self.predict_audio(model, e["clean"], spec, mask, net_mode="a_only")}
        aud = []
        for c, p in conds.items():
            for scope in ("with_audio", "all"):
                for dn in dsets:
                    sel = (has if scope == "with_audio" else np.ones(len(idx), bool)) & ((ds == dn) if dn != "Combined" else True)
                    if not sel.any():
                        continue
                    m = T4.compute_metrics(y[sel], p[sel], thr)
                    aud.append(dict(meta, condition=c, scope=scope, dataset=dn, setting=setting(dn), n=int(sel.sum()), auc=m["auc"],
                                    accuracy=m["accuracy"], balanced_accuracy=m["balanced_accuracy"], f1=m["f1"], fpr=1 - m["specificity"],
                                    fnr=1 - m["recall"], mean_p_fake=float(p[sel].mean()), pred_fake_rate=float(np.mean(p[sel] >= thr)),
                                    flip_rate_vs_normal_pct=100 * float(np.mean((p[sel] >= thr) != (pc[sel] >= thr))),
                                    auc_difference_normal_minus_condition=fast_auc(y[sel], pc[sel]) - m["auc"]))
        aud = pd.DataFrame(aud)

        # validation (used ONLY for the pre-registered best-model choice)
        ev = self.E["val"]
        yv = self.labels[ev["idx"]]
        hvv = ev["pick"] >= 0
        val = dict(val_clean_auc=fast_auc(yv, P["val"]["clean"]), val_degraded_auc=fast_auc(yv[hvv], P["val"]["deg"][hvv]),
                   val_degraded_all_auc=fast_auc(self.labels[[self.item_src(it) for it in ev["deg_all"]]], P["val"]["deg_all"]),
                   val_cam_iou=float(xv.top20_iou.mean()) if len(xv) else np.nan,
                   val_cam_pearson=float(xv.pearson_r.mean()) if len(xv) else np.nan)
        xv.to_csv(out / "xai_val.csv", index=False)

        # summary for the comparison tables
        r_dom = rob[rob.dataset == dom].iloc[0] if (rob.dataset == dom).any() else rob.iloc[0]
        dclean = det[(det.dataset == dom) & (det.condition == "original")].iloc[0]
        ddeg = det[(det.dataset == dom) & (det.condition == "degraded")].iloc[0]
        tdom = degr[(degr.dataset == dom) & (degr.severity == "all")].set_index("degradation")
        x_stable = xai[xai.prediction_stable]
        iou_t = xai.groupby("degradation").top20_iou.mean()
        r_t = xai.groupby("degradation").pearson_r.mean()
        auc_t = tdom.auc.dropna()
        others = [d for d in sorted(set(ds)) if d != dom]
        cr = det[(det.dataset.isin(others)) & (det.condition == "original")]
        crd = det[(det.dataset.isin(others)) & (det.condition == "degraded")]
        aw = aud[(aud.scope == "with_audio") & (aud.dataset == "Combined")].set_index("condition")
        summary = dict(
            info, eval_dir=str(out), in_domain_dataset=dom, cross_datasets=others, threshold=thr,
            clean_auc=dclean.auc, clean_accuracy=dclean.accuracy, clean_balanced_accuracy=dclean.balanced_accuracy, clean_f1=dclean.f1,
            clean_precision=dclean.precision, clean_recall=dclean.recall, clean_specificity=dclean.specificity,
            degraded_auc=ddeg.auc, degraded_accuracy=ddeg.accuracy, degraded_balanced_accuracy=ddeg.balanced_accuracy,
            auc_drop=r_dom.auc_drop, degraded_all_versions_auc=r_dom.degraded_all_versions_auc, flip_rate_pct=r_dom.prediction_flip_rate_pct,
            flip_rate_all_versions_pct=r_dom.prediction_flip_rate_all_versions_pct,
            combined_clean_auc=float(det[(det.dataset == "Combined") & (det.condition == "original")].auc.iloc[0]),
            combined_degraded_auc=float(det[(det.dataset == "Combined") & (det.condition == "degraded")].auc.iloc[0]),
            combined_flip_rate_pct=float(rob[rob.dataset == "Combined"].prediction_flip_rate_pct.iloc[0]),
            cam_iou=float(xai.top20_iou.mean()), cam_pearson=float(xai.pearson_r.mean()), cam_center_shift=float(xai.center_shift.mean()),
            cam_temporal_r=float(xai.temporal_cam_r.mean()), cam_clean_ref_iou=float(xai.clean_ref_iou.mean()),
            cam_iou_in_domain=float(xai[xai.dataset == dom].top20_iou.mean()) if (xai.dataset == dom).any() else np.nan,
            cam_pearson_in_domain=float(xai[xai.dataset == dom].pearson_r.mean()) if (xai.dataset == dom).any() else np.nan,
            prediction_stable_rate=float(xai.prediction_stable.mean()),
            explanation_change_rate_pct=100 * float(x_stable.explanation_changed.mean()) if len(x_stable) else np.nan,
            auc_by_type=auc_t.to_dict(), accuracy_by_type=tdom.accuracy.to_dict(), flip_by_type=tdom.prediction_flip_rate_pct.to_dict(),
            iou_by_type=iou_t.to_dict(), pearson_by_type=r_t.to_dict(),
            worst_degradation_auc=float(auc_t.min()) if len(auc_t) else np.nan, worst_degradation_auc_type=auc_t.idxmin() if len(auc_t) else None,
            mean_degradation_auc=float(auc_t.mean()) if len(auc_t) else np.nan,
            worst_degradation_iou=float(iou_t.min()), worst_degradation_iou_type=iou_t.idxmin(), mean_degradation_iou=float(iou_t.mean()),
            cross_auc=float(cr.auc.mean()) if len(cr) else np.nan, cross_degraded_auc=float(crd.auc.mean()) if len(crd) else np.nan,
            cross_fpr=float(cr.fpr.mean()) if len(cr) else np.nan, cross_fnr=float(cr.fnr.mean()) if len(cr) else np.nan,
            cross_pred_fake_rate=float(cr.pred_fake_rate.mean()) if len(cr) else np.nan,
            audio_masking_flip_rate_pct=float(aw.loc["video_only", "flip_rate_vs_normal_pct"]) if "video_only" in aw.index else np.nan,
            audio_auc_normal_minus_masked=float(aw.loc["video_only", "auc_difference_normal_minus_condition"]) if "video_only" in aw.index else np.nan,
            audio_zero_flip_rate_pct=float(aw.loc["zero_audio", "flip_rate_vs_normal_pct"]) if "zero_audio" in aw.index else np.nan,
            audio_only_auc=float(aw.loc["audio_only", "auc"]) if "audio_only" in aw.index else np.nan, **val)
        summary["robustness_gap"] = summary["clean_auc"] - summary["worst_degradation_auc"]
        summary["explanation_gap"] = summary["cam_clean_ref_iou"] - summary["worst_degradation_iou"]
        if self.localization is not None and "loc_original_iou" in xai:
            summary["localization"] = {c: float(xai[c].mean()) for c in xai.columns if c.startswith("loc_")}

        det.to_csv(out / "detection.csv", index=False)
        rob.to_csv(out / "robustness.csv", index=False)
        degr.to_csv(out / "degradation.csv", index=False)
        xai.to_csv(out / "xai_test.csv", index=False)
        aud.to_csv(out / "audio_ablation.csv", index=False)
        np.savez_compressed(out / "preds_test.npz", idx=idx, y=y, ds=ds, hv=hv, p_clean=pc, p_deg=pdg, src_all=src_all, k_all=k_all,
                            typ_all=typ_all, sev_all=sev_all, p_all=pall, thr=thr, dom=dom, **{f"audio_{k}": v for k, v in conds.items()},
                            has_audio=has)
        if ex:
            np.savez_compressed(out / "xai_examples.npz", **{f"{k}__{f}": np.asarray(v) for k, d in ex.items() for f, v in d.items()})
        T4.write_json(out / "summary.json", summary)
        print(f"  [eval {tag}] {dom}: clean AUC {fmt(summary['clean_auc'], 4)} | degraded AUC {fmt(summary['degraded_auc'], 4)} | "
              f"flip {fmt(summary['flip_rate_pct'], 2)}% | Grad-CAM IoU {fmt(summary['cam_iou'])} r {fmt(summary['cam_pearson'])} | "
              f"explanation change {fmt(summary['explanation_change_rate_pct'], 1)}% | worst AUC {fmt(summary['worst_degradation_auc'])} "
              f"({summary['worst_degradation_auc_type']}) | worst IoU {fmt(summary['worst_degradation_iou'])} ({summary['worst_degradation_iou_type']}) | "
              f"cross AUC {fmt(summary['cross_auc'])}")
        return T4.to_jsonable(summary)


def localization_scores(cam: np.ndarray, box) -> dict:
    """IoU of the top-20 % CAM region with the box, pointing game (CAM maximum inside the box), hit (CAM mass inside box > box area)."""
    H, W = cam.shape
    x0, y0, x1, y1 = [float(v) for v in box]
    m = np.zeros_like(cam, dtype=bool)
    m[int(y0 * H) : max(int(y0 * H) + 1, int(np.ceil(y1 * H))), int(x0 * W) : max(int(x0 * W) + 1, int(np.ceil(x1 * W)))] = True
    top = cam >= np.quantile(cam, 0.8)
    yy, xx = np.unravel_index(int(np.argmax(cam)), cam.shape)
    mass = float(cam[m].sum() / (cam.sum() + 1e-8))
    return dict(iou=float((top & m).sum() / max(1, (top | m).sum())), pointing=float(m[yy, xx]), hit=float(mass > m.mean()), mass_in_box=mass)


# =====================================================================================
# STATISTICS
# =====================================================================================
class Boot:
    """Cluster bootstrap resamples (shared across models -> paired comparisons)."""

    def __init__(self, clusters, n_boot: int, seed: int):
        clusters = np.asarray(clusters)
        _, inv = np.unique(clusters, return_inverse=True)
        order = np.argsort(inv, kind="stable")
        counts = np.bincount(inv)
        groups = np.split(order, np.cumsum(counts)[:-1]) if len(counts) else []
        rng = np.random.default_rng(seed)
        K = len(groups)
        self.samples = [np.concatenate([groups[c] for c in rng.integers(0, K, K)]) for _ in range(n_boot)] if K else []

    def run(self, fn, arrays):
        est = fn(*arrays)
        bs = np.array([fn(*(a[s] for a in arrays)) for s in self.samples], dtype=np.float64)
        return float(est), bs


def ci(bs):
    b = bs[~np.isnan(bs)]
    return (float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))) if len(b) else (np.nan, np.nan)


def p_boot(diff_bs):
    d = diff_bs[~np.isnan(diff_bs)]
    return float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))) if len(d) else np.nan


def type_min(fn_type):
    def f(y, p, t):
        vals = [fn_type(y[t == u], p[t == u]) for u in np.unique(t)]
        vals = [v for v in vals if not np.isnan(v)]
        return min(vals) if vals else np.nan
    return f


def run_statistics(root: Path, train_on: str, seed: int, n_boot: int) -> pd.DataFrame:
    """Paired bootstrap (same test videos / same Grad-CAM pairs for all models) for one training set and seed."""
    tag = T4.train_tag(train_on)
    dirs = {"T4D": root / "test4_reference" / tag / "eval"}
    for v in VARIANTS5:
        dirs[v] = root / "runs" / f"{v}_{tag}_seed{seed}" / "eval"
    data = {}
    for m, d in dirs.items():
        if (d / "preds_test.npz").exists() and (d / "xai_test.csv").exists():
            z = np.load(d / "preds_test.npz", allow_pickle=True)
            data[m] = dict(z=z, x=pd.read_csv(d / "xai_test.csv").sort_values("pair_id").reset_index(drop=True))
    if len(data) < 1:
        return pd.DataFrame()
    ref = next(iter(data.values()))
    for m, D in data.items():
        if not np.array_equal(D["z"]["idx"], ref["z"]["idx"]) or not np.array_equal(D["x"].pair_id.values, ref["x"].pair_id.values):
            raise RuntimeError(f"Model {m} was evaluated on different test videos / Grad-CAM pairs -> paired statistics impossible")
    z0 = ref["z"]
    dom = str(z0["dom"])
    ds, hv = z0["ds"], z0["hv"]
    sel = ds == dom if dom != "Combined" else np.ones(len(ds), bool)
    seld = sel & hv
    ds_all = ds[z0["k_all"]]
    sa = ds_all == dom if dom != "Combined" else np.ones(len(ds_all), bool)
    others = sorted(set(ds) - {dom})
    selx = np.isin(ds, others)
    y, y_all, typ = z0["y"], z0["y"][z0["k_all"]], z0["typ_all"]
    hard = np.isin(typ, list(HARD_WEIGHTS_KEYS))
    rows = []
    boots = dict(video=Boot(np.arange(sel.sum()), n_boot, 11), video_deg=Boot(np.arange(seld.sum()), n_boot, 12),
                 cross=Boot(np.arange(selx.sum()), n_boot, 13) if selx.any() else None,
                 deg_all=Boot(z0["src_all"][sa], n_boot, 14))
    xr = ref["x"]
    boots["pairs"] = Boot(np.arange(len(xr)), n_boot, 15)
    mean_ = lambda a: float(np.nanmean(a)) if len(a) else np.nan

    def flip_of(z):
        return ((z["p_all"] >= z["thr"]) != (z["p_clean"][z["k_all"]] >= z["thr"]))

    specs = [  # (metric, boot, fn, arrays(z, x), higher_is_better)
        ("clean_auc", "video", fast_auc, lambda z, x: (y[sel], z["p_clean"][sel]), True),
        ("degraded_auc", "video_deg", fast_auc, lambda z, x: (y[seld], z["p_deg"][seld]), True),
        ("flip_rate_pct", "video_deg", lambda a, b: 100 * float(np.mean(a != b)),
         lambda z, x: (z["p_clean"][seld] >= z["thr"], z["p_deg"][seld] >= z["thr"]), False),
        ("degraded_all_versions_auc", "deg_all", fast_auc, lambda z, x: (y_all[sa], z["p_all"][sa]), True),
        ("flip_rate_all_versions_pct", "deg_all", lambda f: 100 * mean_(f), lambda z, x: (flip_of(z)[sa],), False),
        ("hard_types_auc", "deg_all", lambda yy, pp, h: fast_auc(yy[h], pp[h]), lambda z, x: (y_all[sa], z["p_all"][sa], hard[sa]), True),
        ("hard_types_flip_rate_pct", "deg_all", lambda f, h: 100 * mean_(f[h]), lambda z, x: (flip_of(z)[sa], hard[sa]), False),
        ("worst_degradation_auc", "deg_all", type_min(fast_auc), lambda z, x: (y_all[sa], z["p_all"][sa], typ[sa]), True),
        ("cam_iou", "pairs", mean_, lambda z, x: (x.top20_iou.values,), True),
        ("cam_pearson", "pairs", mean_, lambda z, x: (x.pearson_r.values,), True),
        ("explanation_change_rate_pct", "pairs", lambda iou, st: 100 * float(((iou < EXPLANATION_CHANGE_IOU) & st).sum() / max(1, st.sum())),
         lambda z, x: (x.top20_iou.values, x.prediction_stable.values.astype(bool)), False),
        ("worst_degradation_iou", "pairs", type_min(lambda a, b: mean_(a)), lambda z, x: (x.top20_iou.values, x.top20_iou.values, x.degradation.values), True),
        ("brightness_saturation_iou", "pairs", lambda iou, t: mean_(iou[np.isin(t, PRIMARY_HARD)]),
         lambda z, x: (x.top20_iou.values, x.degradation.values), True),
    ]
    if selx.any():
        specs.append(("cross_auc", "cross", fast_auc, lambda z, x: (y[selx], z["p_clean"][selx]), True))
    for t in sorted(set(typ[sa]) | set(xr.degradation)):
        specs += [(f"auc|{t}", "deg_all", lambda yy, pp, tt, t=t: fast_auc(yy[tt == t], pp[tt == t]), lambda z, x: (y_all[sa], z["p_all"][sa], typ[sa]), True),
                  (f"accuracy|{t}", "deg_all", lambda yy, pp, tt, th, t=t: float(np.mean((pp[tt == t] >= th[tt == t]) == yy[tt == t])) if (tt == t).any() else np.nan,
                   lambda z, x: (y_all[sa], z["p_all"][sa], typ[sa], np.full(sa.sum(), float(z["thr"]))), True),
                  (f"flip_rate_pct|{t}", "deg_all", lambda f, tt, t=t: 100 * mean_(f[tt == t]), lambda z, x: (flip_of(z)[sa], typ[sa]), False),
                  (f"iou|{t}", "pairs", lambda iou, tt, t=t: mean_(iou[tt == t]), lambda z, x: (x.top20_iou.values, x.degradation.values), True),
                  (f"pearson|{t}", "pairs", lambda r, tt, t=t: mean_(r[tt == t]), lambda z, x: (x.pearson_r.values, x.degradation.values), True)]
    print(f"\n[statistics] paired bootstrap ({n_boot} resamples) | train {train_on} | seed {seed} | models {list(data)} | {len(specs)} metrics")
    for metric, bname, fn, arrs, hib in specs:
        b = boots[bname]
        if b is None:
            continue
        res = {m: b.run(fn, arrs(D["z"], D["x"])) for m, D in data.items()}
        for m, (est, bs) in res.items():
            lo, hi = ci(bs)
            row = dict(train_on=train_on, seed=seed, metric=metric, model=m, estimate=est, ci_lo=lo, ci_hi=hi, higher_is_better=hib,
                       n_clusters=len(b.samples[0]) if b.samples else 0)
            rows.append(row)
            for vs in ("T4D", "5A"):
                if vs == m or vs not in res:
                    continue
                d = bs - res[vs][1]
                dl, dh = ci(d)
                extra = {}
                if metric in ("cam_iou", "cam_pearson") or metric.startswith(("iou|", "pearson|")) or metric == "brightness_saturation_iou":
                    col = "pearson_r" if "pearson" in metric else "top20_iou"
                    msk = np.ones(len(xr), bool)
                    if "|" in metric:
                        msk = xr.degradation.values == metric.split("|")[1]
                    elif metric == "brightness_saturation_iou":
                        msk = np.isin(xr.degradation.values, PRIMARY_HARD)
                    a_, b_ = data[m]["x"][col].values[msk], data[vs]["x"][col].values[msk]
                    ok = ~(np.isnan(a_) | np.isnan(b_))
                    if ok.sum() > 5 and np.any(a_[ok] != b_[ok]):
                        extra = dict(test="Wilcoxon signed-rank (pairs)", test_p=float(wilcoxon(a_[ok], b_[ok]).pvalue))
                if metric == "flip_rate_pct":
                    fa = (data[m]["z"]["p_clean"][seld] >= data[m]["z"]["thr"]) != (data[m]["z"]["p_deg"][seld] >= data[m]["z"]["thr"])
                    fb = (data[vs]["z"]["p_clean"][seld] >= data[vs]["z"]["thr"]) != (data[vs]["z"]["p_deg"][seld] >= data[vs]["z"]["thr"])
                    n01, n10 = int((~fa & fb).sum()), int((fa & ~fb).sum())
                    if n01 + n10 > 0:
                        extra = dict(test="exact McNemar (videos)", test_p=float(binomtest(n10, n01 + n10, 0.5).pvalue))
                rows.append(dict(row, comparison=f"{m} - {vs}", diff=est - res[vs][0], diff_ci_lo=dl, diff_ci_hi=dh, p_boot=p_boot(d),
                                 significant=bool(not np.isnan(dl) and (dl > 0 or dh < 0) and row["n_clusters"] >= MIN_CLUSTERS_FOR_SIGNIFICANCE),
                                 better=bool((est - res[vs][0] > 0) == hib) if not np.isnan(est - res[vs][0]) else None, **extra))
    return pd.DataFrame(rows)


HARD_WEIGHTS_KEYS = ["brightness", "saturation", "contrast", "blur"]


# =====================================================================================
# AGGREGATION, FIGURES, REPORT
# =====================================================================================
def collect_summaries(root: Path) -> List[dict]:
    out = []
    for f in sorted(list(root.glob("runs/*/eval/summary.json")) + list(root.glob("test4_reference/*/eval/summary.json"))):
        out.append(json.load(open(f, encoding="utf-8")))
    return out


def concat_csv(root: Path, name: str) -> pd.DataFrame:
    fs = sorted(list(root.glob(f"runs/*/eval/{name}")) + list(root.glob(f"test4_reference/*/eval/{name}")))
    return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True) if fs else pd.DataFrame()


COMP_COLS = [("clean_auc", "clean AUC"), ("degraded_auc", "degraded AUC"), ("auc_drop", "AUC drop"), ("flip_rate_pct", "flip rate %"),
             ("cam_iou", "Grad-CAM IoU"), ("cam_pearson", "Pearson r"), ("explanation_change_rate_pct", "explanation-change rate %"),
             ("worst_degradation_auc", "worst degradation AUC"), ("worst_degradation_iou", "worst degradation IoU"),
             ("cross_auc", "cross-dataset AUC"), ("mean_degradation_auc", "mean degradation AUC"), ("mean_degradation_iou", "mean degradation IoU"),
             ("robustness_gap", "robustness gap"), ("explanation_gap", "explanation gap"), ("degraded_all_versions_auc", "degraded AUC (all versions)"),
             ("flip_rate_all_versions_pct", "flip rate all versions %"), ("cam_clean_ref_iou", "clean reference IoU"),
             ("cam_center_shift", "centre shift"), ("cam_temporal_r", "temporal CAM r"), ("audio_masking_flip_rate_pct", "audio masking flip %"),
             ("audio_auc_normal_minus_masked", "AUC normal - masked audio"), ("val_degraded_auc", "VAL degraded AUC"), ("val_cam_iou", "VAL Grad-CAM IoU")]


def comparison_table(S: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (t, m), g in S.groupby(["train_on", "model"]):
        r = dict(train_on=t, model=m, label=MODEL_LABEL.get(m, m), n_seeds=len(g), seeds=",".join(str(int(s)) for s in sorted(g.seed)))
        for k, _ in COMP_COLS:
            v = pd.to_numeric(g[k], errors="coerce") if k in g else pd.Series(dtype=float)
            r[k] = float(v.mean()) if len(v.dropna()) else np.nan
            r[f"{k}_sd"] = float(v.std(ddof=1)) if len(v.dropna()) > 1 else np.nan
        for k in ("worst_degradation_auc_type", "worst_degradation_iou_type"):
            r[k] = "/".join(sorted(set(str(x) for x in g[k].dropna()))) if k in g else None
        rows.append(r)
    df = pd.DataFrame(rows)
    if len(df):
        df["_o"] = df.model.map({m: i for i, m in enumerate(MODEL_ORDER)})
        df = df.sort_values(["train_on", "_o"]).drop(columns="_o").reset_index(drop=True)
    return df


def choose_best(S: pd.DataFrame, train_on: str) -> dict:
    """Pre-registered, validation-only rule: among 5A-5D, keep the variants whose mean validation degraded AUC is within
    0.005 of the best one, then take the highest mean validation Grad-CAM IoU. The test set is never used."""
    g = S[(S.train_on == train_on) & S.model.isin(list(VARIANTS5))].groupby("model")[["val_degraded_auc", "val_cam_iou"]].mean()
    if not len(g):
        return dict(best=None)
    ok = g[g.val_degraded_auc >= g.val_degraded_auc.max() - 0.005]
    best = ok.val_cam_iou.idxmax()
    return dict(best=best, rule="val degraded AUC within 0.005 of the best variant, then highest val Grad-CAM IoU",
                validation=g.round(5).reset_index().to_dict(orient="records"))


def style_axes(ax, title=None, ylabel=None):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c3c2b7")
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, fontsize=10, color=INK, loc="left", fontweight="bold")
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9, color=INK2)


def bar_models(ax, comp, stats, metric, models, ylabel, title, pct=False, ylim=None, seed_points=None):
    vals, lo, hi = [], [], []
    for m in models:
        r = comp[comp.model == m]
        v = float(r[metric].iloc[0]) if len(r) else np.nan
        vals.append(v)
        s = stats[(stats.metric == metric) & (stats.model == m) & stats.comparison.isna()] if len(stats) and "comparison" in stats else pd.DataFrame()
        lo.append(v - float(s.ci_lo.iloc[0]) if len(s) and not np.isnan(v) else 0)
        hi.append(float(s.ci_hi.iloc[0]) - v if len(s) and not np.isnan(v) else 0)
    x = np.arange(len(models))
    ax.bar(x, vals, width=0.62, color=[MODEL_COLOR[m] for m in models], edgecolor=SURFACE, linewidth=2)
    ax.errorbar(x, vals, yerr=[np.maximum(lo, 0), np.maximum(hi, 0)], fmt="none", ecolor=INK2, elinewidth=1, capsize=3)
    if seed_points is not None:
        for i, m in enumerate(models):
            pts = seed_points[(seed_points.model == m)][metric].dropna().values
            ax.scatter(np.full(len(pts), i) + np.linspace(-0.12, 0.12, max(1, len(pts)))[: len(pts)], pts, s=12, color=INK, zorder=3, linewidths=0)
    for i, v in enumerate(vals):
        if not np.isnan(v):
            ax.text(i, v, f" {v:.1f}" + ("%" if pct else "") if pct else f" {v:.3f}", ha="center", va="bottom", fontsize=7, color=INK2, rotation=0)
    ax.set_xticks(x, [MODEL_LABEL[m] for m in models], fontsize=8)
    if ylim:
        ax.set_ylim(*ylim)
    style_axes(ax, title, ylabel)


def make_figures(root: Path, S: pd.DataFrame, comp: pd.DataFrame, stats: pd.DataFrame, primary: str, pseed: int, deg_table: pd.DataFrame):
    fig_dir = root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "text.color": INK, "axes.labelcolor": INK2, "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE})
    cp = comp[comp.train_on == primary]
    sp = stats[(stats.train_on == primary)] if len(stats) else pd.DataFrame(columns=["metric", "model", "comparison"])
    models = [m for m in MODEL_ORDER if m in set(cp.model)]
    pts = S[S.train_on == primary]
    note = f"bars: mean over seeds; whiskers: 95% bootstrap CI (seed {pseed}); dots: individual seeds"

    def single(name, metric, ylabel, title, pct=False, ylim=None):
        fig, ax = plt.subplots(figsize=(6.4, 3.8))
        bar_models(ax, cp, sp, metric, models, ylabel, title, pct, ylim, pts)
        fig.text(0.01, 0.01, note, fontsize=6.5, color=INK2)
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        fig.savefig(fig_dir / name, dpi=150)
        plt.close(fig)

    if not len(cp):
        return
    lo = np.nanmin(cp[["clean_auc", "degraded_auc"]].values) if len(cp) else 0.5
    yl = (max(0.5, lo - 0.03), 1.0)
    single("fig01_test4_vs_test5_clean_auc.png", "clean_auc", "ROC-AUC", f"Clean ROC-AUC ({primary} test, in-domain)", ylim=yl)
    single("fig02_test4_vs_test5_degraded_auc.png", "degraded_auc", "ROC-AUC", f"Degraded ROC-AUC ({primary} test, in-domain)", ylim=yl)
    single("fig07_prediction_flip_rate.png", "flip_rate_pct", "% of videos", "Prediction flip rate, original vs degraded", pct=True)
    single("fig08_explanation_change_rate.png", "explanation_change_rate_pct", "% of stable predictions",
           "Explanation changed (IoU < 0.5) while the prediction stayed the same", pct=True)

    # 3 / 4 degradation-wise AUC and IoU
    for name, key, ylabel, title in (("fig03_degradation_wise_auc.png", "auc_by_type", "ROC-AUC", f"ROC-AUC per degradation type ({primary} test, in-domain)"),
                                     ("fig04_degradation_wise_gradcam_iou.png", "iou_by_type", "top-20% IoU", "Grad-CAM top-20% IoU per degradation type (all test pairs)")):
        types = [t for t in ALL_DEG_TYPES if any(isinstance(d, dict) and t in d for d in pts[key])]
        fig, ax = plt.subplots(figsize=(11, 4))
        w = 0.8 / max(1, len(models))
        for i, m in enumerate(models):
            vals = [np.nanmean([d.get(t, np.nan) for d in pts[pts.model == m][key] if isinstance(d, dict)] or [np.nan]) for t in types]
            ax.bar(np.arange(len(types)) + (i - (len(models) - 1) / 2) * w, vals, width=w * 0.92, color=MODEL_COLOR[m], label=MODEL_LABEL[m],
                   edgecolor=SURFACE, linewidth=1)
        ax.set_xticks(np.arange(len(types)), types, fontsize=8, rotation=20)
        allv = [v for d in pts[key] if isinstance(d, dict) for v in d.values() if v is not None and not np.isnan(v)]
        ax.set_ylim(max(0, min(allv) - 0.05) if allv else 0, 1.0)
        ax.legend(fontsize=7, ncol=len(models), frameon=False, loc="upper left", bbox_to_anchor=(0, 1.12))
        style_axes(ax, None, ylabel)
        ax.set_title(title, fontsize=10, color=INK, loc="left", fontweight="bold", pad=22)
        fig.tight_layout()
        fig.savefig(fig_dir / name, dpi=150)
        plt.close(fig)

    # 5 / 6 brightness, saturation IoU
    for name, t in (("fig05_brightness_iou.png", "brightness"), ("fig06_saturation_iou.png", "saturation")):
        cpt = cp.copy()
        cpt[f"iou|{t}"] = [np.nanmean([d.get(t, np.nan) for d in pts[pts.model == m].iou_by_type if isinstance(d, dict)] or [np.nan]) for m in cpt.model]
        ptt = pts.copy()
        ptt[f"iou|{t}"] = [d.get(t, np.nan) if isinstance(d, dict) else np.nan for d in ptt.iou_by_type]
        fig, ax = plt.subplots(figsize=(6.4, 3.8))
        bar_models(ax, cpt, sp, f"iou|{t}", models, "top-20% IoU", f"Grad-CAM IoU under {t} (original vs degraded)", ylim=(0, 1), seed_points=ptt)
        fig.text(0.01, 0.01, note, fontsize=6.5, color=INK2)
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        fig.savefig(fig_dir / name, dpi=150)
        plt.close(fig)

    # 9 ROC clean vs degraded (primary seed)
    tag = T4.train_tag(primary)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for m in models:
        d = root / ("test4_reference/" + tag if m == "T4D" else f"runs/{m}_{tag}_seed{pseed}") / "eval" / "preds_test.npz"
        if not (Path(str(root / d)) if not d.is_absolute() else d).exists():
            continue
        z = np.load(d, allow_pickle=True)
        dom = str(z["dom"])
        sel = z["ds"] == dom if dom != "Combined" else np.ones(len(z["ds"]), bool)
        for ax, (cond, p, s2) in zip(axes, (("clean", z["p_clean"], sel), ("degraded", z["p_deg"], sel & z["hv"]))):
            if len(np.unique(z["y"][s2])) < 2:
                continue
            fpr, tpr, _ = roc_curve(z["y"][s2], p[s2])
            ax.plot(fpr, tpr, lw=2 if m in ("T4D", "5D") else 1.2, color=MODEL_COLOR[m], label=f"{MODEL_LABEL[m]} ({fast_auc(z['y'][s2], p[s2]):.3f})")
    for ax, cond in zip(axes, ("clean", "degraded")):
        ax.plot([0, 1], [0, 1], ":", color="#c3c2b7")
        ax.set_xlabel("false positive rate", fontsize=8, color=INK2)
        style_axes(ax, f"ROC - {cond} test videos ({primary}, seed {pseed})", "true positive rate")
        ax.legend(fontsize=7, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig09_clean_vs_degraded_roc.png", dpi=150)
    plt.close(fig)

    # 10 cross-dataset AUC
    fig, ax = plt.subplots(figsize=(8, 3.8))
    trains = [t for t in comp.train_on.unique()]
    w = 0.8 / max(1, len(models))
    for i, m in enumerate(models):
        vals = [float(comp[(comp.train_on == t) & (comp.model == m)].cross_auc.iloc[0]) if len(comp[(comp.train_on == t) & (comp.model == m)]) else np.nan for t in trains]
        ax.bar(np.arange(len(trains)) + (i - (len(models) - 1) / 2) * w, vals, width=w * 0.92, color=MODEL_COLOR[m], label=MODEL_LABEL[m], edgecolor=SURFACE)
    ax.axhline(0.5, color=INK2, lw=0.8, ls=":")
    ax.set_xticks(np.arange(len(trains)), [f"trained on {t} -> tested on the other dataset" for t in trains], fontsize=8)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7, ncol=len(models), frameon=False, loc="upper left", bbox_to_anchor=(0, 1.13))
    style_axes(ax, None, "ROC-AUC (original videos)")
    ax.set_title("Cross-dataset ROC-AUC (dotted line = chance)", fontsize=10, color=INK, loc="left", fontweight="bold", pad=22)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig10_cross_dataset_auc.png", dpi=150)
    plt.close(fig)

    # 11 worst case
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, keys, title, ylabel in ((axes[0], [("clean_auc", "clean"), ("mean_degradation_auc", "mean over types"), ("worst_degradation_auc", "worst type")],
                                     "Detection: clean vs mean vs worst degradation", "ROC-AUC"),
                                    (axes[1], [("cam_clean_ref_iou", "clean reference"), ("mean_degradation_iou", "mean over types"),
                                               ("worst_degradation_iou", "worst type")], "Grad-CAM IoU: clean reference vs mean vs worst", "top-20% IoU")):
        for i, (k, lab) in enumerate(keys):
            vals = [float(cp[cp.model == m][k].iloc[0]) for m in models]
            ax.plot(np.arange(len(models)), vals, marker="o", ms=6, lw=2, color=["#2a78d6", "#eb6834", "#4a3aa7"][i], label=lab)
            ax.text(len(models) - 1 + 0.08, vals[-1], lab, fontsize=7, color=INK2, va="center")
        ax.set_xticks(np.arange(len(models)), [MODEL_LABEL[m] for m in models], fontsize=8)
        ax.set_xlim(-0.3, len(models) + 0.6)
        style_axes(ax, title, ylabel)
        ax.legend(fontsize=7, frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig11_worst_case_robustness.png", dpi=150)
    plt.close(fig)

    # 12 training curves (primary train set + seed)
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.6))
    for v in VARIANTS5:
        f = root / "runs" / f"{v}_{tag}_seed{pseed}" / "training_history.csv"
        if not f.exists():
            continue
        h = pd.read_csv(f)
        for ax, col in zip(axes, ["train_ce", "train_feat", "train_cam_soft_iou", "score"]):
            if col in h:
                ax.plot(h.epoch, h[col], marker="o", ms=3, lw=1.8, color=MODEL_COLOR[v], label=MODEL_LABEL[v])
    for ax, t in zip(axes, ["classification loss (CE, pairs)", "feature consistency loss", "training Grad-CAM soft IoU", "validation selection score (macro AUC)"]):
        ax.set_xlabel("epoch", fontsize=8, color=INK2)
        style_axes(ax, t, None)
    axes[0].legend(fontsize=7, frameon=False)
    fig.suptitle(f"Stage-2 training curves ({primary}, seed {pseed}); losses are monitored even when their weight is 0", fontsize=9, color=INK2)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig12_training_loss_curves.png", dpi=150)
    plt.close(fig)

    # 13 Grad-CAM examples: same pairs, Test 4D vs best/5D
    e4 = root / "test4_reference" / tag / "eval" / "xai_examples.npz"
    e5 = root / "runs" / f"5D_{tag}_seed{pseed}" / "eval" / "xai_examples.npz"
    if e4.exists() and e5.exists():
        z4, z5 = np.load(e4, allow_pickle=True), np.load(e5, allow_pickle=True)
        ids = sorted({k.split("__")[0] for k in z4.files} & {k.split("__")[0] for k in z5.files}, key=int)
        if ids:
            fig, axes = plt.subplots(len(ids), 6, figsize=(13, 2.3 * len(ids)))
            axes = np.atleast_2d(axes)
            for r, i in enumerate(ids):
                g4 = lambda f: z4[f"{i}__{f}"]
                g5 = lambda f: z5[f"{i}__{f}"]
                fo, fd = g4("frame_original"), g4("frame_degraded")
                panels = [(fo, None, f"original\n{str(g4('video'))[-28:]}"), (fd, None, f"degraded: {g4('degradation')}"),
                          (fo, g4("cam_original"), f"Test 4D CAM orig p={float(g4('p_original')):.2f}"),
                          (fd, g4("cam_degraded"), f"Test 4D CAM deg IoU={float(g4('iou')):.2f}"),
                          (fo, g5("cam_original"), f"Test 5D CAM orig p={float(g5('p_original')):.2f}"),
                          (fd, g5("cam_degraded"), f"Test 5D CAM deg IoU={float(g5('iou')):.2f}")]
                for c, (img, cam, ttl) in enumerate(panels):
                    ax = axes[r, c]
                    ax.imshow(img)
                    if cam is not None:
                        ax.imshow(cam, cmap="magma", alpha=0.5)
                    ax.set_title(ttl, fontsize=7, color=INK)
                    ax.axis("off")
            fig.suptitle("Grad-CAM examples - identical test pairs for both models, pairs fixed before evaluation (first 2 per hard type)",
                         fontsize=9, color=INK2)
            fig.tight_layout(rect=(0, 0, 1, 0.975))
            fig.savefig(fig_dir / "fig13_gradcam_examples.png", dpi=130)
            plt.close(fig)


def build_report_tables(root: Path, a5) -> dict:
    S_list = collect_summaries(root)
    if not S_list:
        print("[report] no evaluated runs found")
        return {}
    S = pd.DataFrame(S_list)
    comp = comparison_table(S)
    comp.to_csv(root / "test5_comparison.csv", index=False)
    for name, out in (("detection.csv", "metrics_test5.csv"), ("degradation.csv", "degradation_metrics_test5.csv"),
                      ("xai_test.csv", "gradcam_stability_test5.csv"), ("audio_ablation.csv", "audio_ablation_test5.csv")):
        df = concat_csv(root, name)
        if len(df):
            df.to_csv(root / out, index=False)
    rob = concat_csv(root, "robustness.csv")
    deg = concat_csv(root, "degradation.csv")
    ps = []
    if len(rob):
        ps.append(rob.assign(level="dataset (one random degraded version per video, Test 4 definition)"))
    if len(deg):
        ps.append(deg[deg.severity == "all"].assign(level="degradation type (all versions)")[
            ["model", "train_on", "seed", "dataset", "setting", "degradation", "n", "prediction_flip_rate_pct", "mean_abs_prob_shift", "level"]])
    if ps:
        pd.concat(ps, ignore_index=True).to_csv(root / "prediction_stability_test5.csv", index=False)
    det = concat_csv(root, "detection.csv")
    aud = concat_csv(root, "audio_ablation.csv")
    if len(det):
        cr = det[(det.dataset != "Combined") & det.condition.isin(["original", "degraded"])].copy()
        cr["direction"] = cr.train_on + " -> " + cr.dataset
        if len(aud):
            am = aud[(aud.scope == "all") & (aud.condition == "video_only")][["model", "train_on", "seed", "dataset", "auc", "flip_rate_vs_normal_pct"]]
            am = am.rename(columns={"auc": "auc_audio_masked", "flip_rate_vs_normal_pct": "audio_masking_flip_rate_pct"})
            cr = cr.merge(am, on=["model", "train_on", "seed", "dataset"], how="left")
        cols = ["model", "train_on", "seed", "direction", "setting", "condition", "n", "auc", "accuracy", "balanced_accuracy", "mean_p_fake",
                "pred_fake_rate", "mean_p_fake_real", "mean_p_fake_fake", "fpr", "fnr", "auc_audio_masked", "audio_masking_flip_rate_pct"]
        cr[[c for c in cols if c in cr]].to_csv(root / "cross_dataset_test5.csv", index=False)

    stats = []
    for t in a5.train_list:
        try:
            st = run_statistics(root, t, a5.primary_seed, a5.n_boot)
        except RuntimeError as ex:
            print(f"[statistics] {ex}")
            st = pd.DataFrame()
        if len(st):
            stats.append(st)
    stats = pd.concat(stats, ignore_index=True) if stats else pd.DataFrame(columns=["train_on", "metric", "model", "comparison"])
    if "comparison" not in stats:
        stats["comparison"] = np.nan
    stats.to_csv(root / "statistics_test5.csv", index=False)

    # Section 7: degradation-wise explanation robustness (primary training set)
    P = S[S.train_on == a5.primary_train]
    dt = []
    for t in ALL_DEG_TYPES:
        r = dict(degradation=t, hard_type=t in a5.hard_weight_map)
        for m in MODEL_ORDER:
            vals = [d.get(t) for d in P[P.model == m].iou_by_type if isinstance(d, dict) and d.get(t) is not None]
            r[f"{MODEL_LABEL[m]} IoU"] = float(np.mean(vals)) if vals else np.nan
        r["delta_IoU_5D_minus_T4"] = r["Test 5D IoU"] - r["Test 4D IoU"]
        s = stats[(stats.train_on == a5.primary_train) & (stats.metric == f"iou|{t}") & (stats.comparison == "5D - T4D")]
        if len(s):
            r.update(delta_seed_primary=float(s["diff"].iloc[0]), delta_ci_lo=float(s.diff_ci_lo.iloc[0]), delta_ci_hi=float(s.diff_ci_hi.iloc[0]),
                     wilcoxon_p=float(s.test_p.iloc[0]) if "test_p" in s and not pd.isna(s.test_p.iloc[0]) else np.nan)
        dt.append(r)
    deg_table = pd.DataFrame(dt)
    deg_table.to_csv(root / "degradation_explanation_robustness_test5.csv", index=False)

    # Section 8: targeted improvement (all hard types, primary + other training sets)
    ha = []
    for t_on in a5.train_list:
        Pt = S[S.train_on == t_on]
        for t in HARD_WEIGHTS_KEYS:
            for metric, key in (("auc", "auc_by_type"), ("accuracy", "accuracy_by_type"), ("flip_rate_pct", "flip_by_type"),
                                ("iou", "iou_by_type"), ("pearson", "pearson_by_type")):
                base = [d.get(t) for d in Pt[Pt.model == "T4D"][key] if isinstance(d, dict) and d.get(t) is not None]
                b = float(np.mean(base)) if base else np.nan
                for m in VARIANTS5:
                    vals = [d.get(t) for d in Pt[Pt.model == m][key] if isinstance(d, dict) and d.get(t) is not None]
                    v = float(np.mean(vals)) if vals else np.nan
                    sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan
                    r = dict(train_on=t_on, degradation=t, primary_hypothesis=t in PRIMARY_HARD, metric=metric, model=m, test4D=b,
                             test5_mean_over_seeds=v, test5_sd=sd, n_seeds=len(vals), improvement=(v - b) if metric != "flip_rate_pct" else (b - v),
                             improvement_definition="T4 - T5 (flip reduction)" if metric == "flip_rate_pct" else "T5 - T4")
                    s = stats[(stats.train_on == t_on) & (stats.metric == f"{metric}|{t}") & (stats.comparison == f"{m} - T4D")]
                    if len(s):
                        r.update(seed_primary_diff=float(s["diff"].iloc[0]), diff_ci_lo=float(s.diff_ci_lo.iloc[0]), diff_ci_hi=float(s.diff_ci_hi.iloc[0]),
                                 p_boot=float(s.p_boot.iloc[0]), significant=bool(s.significant.iloc[0]))
                    ha.append(r)
    hard_df = pd.DataFrame(ha)
    hard_df.to_csv(root / "hard_degradation_analysis.csv", index=False)
    return dict(S=S, comp=comp, stats=stats, deg_table=deg_table, hard=hard_df, det=det, aud=aud)


def stat_row(stats, train_on, metric, comparison):
    s = stats[(stats.train_on == train_on) & (stats.metric == metric) & (stats.comparison == comparison)]
    return s.iloc[0] if len(s) else None


def diff_str(stats, t, metric, comparison, nd=3, pct=False):
    s = stat_row(stats, t, metric, comparison)
    if s is None:
        return "n/a"
    u = "%" if pct else ""
    return f"{s['diff']:+.{nd}f}{u} [95% CI {s.diff_ci_lo:+.{nd}f}, {s.diff_ci_hi:+.{nd}f}]" + (" *" if s.significant else "")


def hypothesis_status(stats, t, metric, comparison, want_up: bool):
    s = stat_row(stats, t, metric, comparison)
    if s is None:
        return "not evaluable", None
    if s.significant and ((s["diff"] > 0) == want_up):
        return "supported", s
    if s.significant:
        return "contradicted (significant change in the opposite direction)", s
    return "not supported (no significant change)", s


def seed_consistency(S, t, metric, m, ref="T4D", higher=True) -> str:
    g = S[(S.train_on == t) & (S.model == m)]
    b = S[(S.train_on == t) & (S.model == ref)]
    if not len(g) or not len(b):
        return "n/a"
    bv = float(pd.to_numeric(b[metric]).mean())
    d = pd.to_numeric(g[metric]).values - bv
    good = (d > 0) if higher else (d < 0)
    return f"{int(good.sum())}/{len(d)} seeds in the expected direction"


def write_report(root: Path, a5, T: dict, env: dict, data_info: dict):
    S, comp, stats, deg_table, hard = T["S"], T["comp"], T["stats"], T["deg_table"], T["hard"]
    pt, ps = a5.primary_train, a5.primary_seed
    cp = comp[comp.train_on == pt].set_index("model")
    best = choose_best(S, pt)
    bm = best.get("best")
    L = []
    w = L.append

    def val(m, k, nd=3, pct=False):
        return fmt(cp.loc[m, k], nd, pct) if m in cp.index and k in cp else "n/a"

    smoke = not (data_info.get("split_check", {}) or {}).get("identical", False)
    w("# Test 5 - Hard-Degradation and Explanation-Aware Training\n")
    w(f"*Generated {datetime.now():%Y-%m-%d %H:%M} by `8-10-26.py`. Baseline: Test 4 variant 4D (`6-10-26.py`, run `{a5.test4_dir.name}`). "
      f"Primary results: training set **{pt}**, seeds {', '.join(str(int(s)) for s in sorted(S[S.model != 'T4D'].seed.unique()))} "
      f"(paired bootstrap on seed {ps}).*\n")
    if smoke:
        w("> **WARNING - the split was NOT verified to be identical to the Test 4 split (smoke/subset run). These numbers must not be used in the thesis.**\n")
    w("## 1. Research question\n")
    w("Can targeted Stage-2 training on the degradation types that produced the least stable explanations in Test 4 "
      "(brightness, saturation, contrast, blur) improve **both** detection robustness and Grad-CAM explanation stability, "
      "without changing the model, the data split or the evaluation protocol?\n")
    w("## 2. Motivation from Test 4\n")
    w("Test 4D (BioDeepAV-trained, Stage 2) reached clean AUC ~0.992, degraded AUC ~0.988, ~2.2 % prediction flips, Grad-CAM top-20 % "
      "IoU ~0.674 and Pearson r ~0.855, but the explanation changed for ~19.6 % of the videos whose prediction stayed the same. "
      "Per degradation, brightness and saturation had the lowest IoU (Test 4 Grad-CAM file; only 6-15 pairs per type), contrast and blur followed. "
      "Cross-dataset generalisation was poor (BioDeepAV -> FF++ AUC ~0.53) and audio acted as a potential dataset shortcut.\n")
    w("Note: the Test 4 Grad-CAM numbers came from 150 random pairs. Test 5 re-evaluates the **frozen Test 4D model** on the larger, "
      "type-stratified Test 5 pair sample, so the \"Test 4D\" values below can differ slightly from the published Test 4 values; all "
      "comparisons use the re-evaluated values (identical inputs, paired).\n")
    w("## 3. Hypotheses\n")
    for h in ("H1: hard-degradation sampling improves robustness to difficult visual degradations.",
              "H2: hard-degradation sampling improves Grad-CAM explanation stability, especially under brightness and saturation.",
              "H3: hard sampling + feature + Grad-CAM consistency (5D) gives better worst-case robustness than Test 4D.",
              "H4: improving degradation robustness does NOT necessarily improve cross-dataset generalisation.",
              "H5: explanation stability and explanation correctness are different properties."):
        w(f"- {h}")
    w("")
    w("## 4. Experimental design\n")
    w("Controlled experiment: the Test 4 code is imported unchanged (SHA-256 of `6-10-26.py`: `" + str(env.get("test4_script_sha256")) + "`). "
      "Only two things change between variants: (i) how the degraded member of each Stage-2 training pair is sampled, and (ii) the weights of "
      "the feature and Grad-CAM consistency losses. Stage 1 is identical for every variant of a seed (seed 42: the Test 4D Stage-1 checkpoint "
      "itself, copied read-only when its configuration matches). Test data, degraded test versions, Grad-CAM pairs and audio degradations are "
      f"fixed by the data seed ({a5.data_seed}) and are identical for every model, including the frozen Test 4D model. "
      "Model choice uses validation data only; the test set is used once for the final evaluation.\n")
    w("## 5. Dataset\n")
    inv = data_info.get("inventory")
    if inv:
        w(pd.DataFrame(inv).pipe(md, index=False) + "\n")
    w("Only the provided degraded videos (`degraded_output/manifest.json`) are used; selection rules are those of Test 4 (types present for both "
      "classes of a dataset, max 3 versions per original, class-balanced). BioDeepAV therefore has 8 degradation types, FF++ 10 "
      "(fps_reduction and low_resolution exist only in FF++).\n")
    w("## 6. Data split\n")
    sc = data_info.get("split_check", {})
    w(f"Identity-grouped split from Test 4, recreated with data seed {a5.data_seed} and compared key-by-key with the Test 4 split file: "
      f"**{'identical' if sc.get('identical') else 'NOT verified identical'}** ({sc.get('differences')}).\n")
    lk = data_info.get("leakage", {})
    if lk:
        rows = [(k, lk.get(k)) for k in ("train_test_identity_overlap", "val_test_identity_overlap", "train_val_identity_overlap",
                                         "ffpp_train_test_identity_token_overlap", "duplicate_filenames_across_splits",
                                         "duplicate_file_hashes_across_splits", "duplicate_clip_hashes_across_splits",
                                         "original_degraded_pair_split_mismatch", "leakage_count")]
        w("| leakage check | count |\n|---|---|\n" + "\n".join(f"| {k} | {v} |" for k, v in rows) + "\n")
        w("Degraded versions are never split separately: every degraded video inherits the split of its original. Hash screens catch "
          "byte-identical files and identical decoded face clips, not re-encoded near-duplicates.\n")
    w("## 7. Model\n")
    w(f"`AudioVisualDeepFakeCNN` from Test 4, unchanged ({int(S.n_parameters.dropna().iloc[0]) if 'n_parameters' in S and S.n_parameters.notna().any() else 'n/a'} "
      "parameters): 4-block 2D CNN video branch (16-32-64-128) on 8 face crops of 128x128 (resolution-normalised: frames to 360 px shorter "
      "side, faces through an 80 px bottleneck), mean+max temporal pooling (256-d), 2D CNN audio branch on a 128x128 log-STFT (128-d), "
      "fusion MLP 384-128-2. All checkpoints have the same architecture signature (verified, see final_experiment_summary.json).\n")
    w("## 8. Training configuration\n")
    c0 = S[(S.model == "5A")].config.dropna()
    if len(c0):
        c = c0.iloc[0]
        w(f"Stage 1: {c.get('epochs1')} epochs, AdamW, OneCycle max lr {c.get('lr1')}, batch {c.get('batch_size')}; Stage 2: {c.get('epochs2')} "
          f"epochs, max lr {c.get('lr2')}, {c.get('pair_batch_size')} (original, degraded) pairs per step, curriculum {c.get('curriculum_epochs')} "
          f"(mild -> +medium -> all), mixed audio policy {c.get('audio_probs_clean_degraded_none')} (clean/degraded/none), sampling "
          f"{c.get('sampling')} (dataset x class balanced), consistency anchor {c.get('consistency_anchor')}, thresholds selected on validation "
          f"({c.get('threshold_metric')}), model selection by macro per-dataset validation AUC. Same as Test 4D.\n")
    w("## 9. Test 5 variants\n")
    w("| variant | hard sampling | lambda_feat | lambda_cam | paired | curriculum | audio |\n|---|---|---|---|---|---|---|")
    w("| Test 4D (frozen) | no | 0.10 | 0.05 | yes | yes | mixed |")
    for v, c in VARIANTS5.items():
        w(f"| {v} | {'yes' if c['hard'] else 'no'} | {a5.lambda_feat if c['feat'] else 0} | {a5.lambda_cam if c['cam'] else 0} | yes | yes | mixed |")
    w("\nContrasts: 5D vs 5A = effect of hard sampling (only difference); 5C vs 5B = Grad-CAM loss under hard sampling; 5D vs 5C = feature loss. "
      "5A vs Test 4D = reproducibility of Test 4D (same configuration, retrained).\n")
    w("## 10. Hard-degradation sampling\n")
    w(f"With probability {a5.hard_share} a Stage-2 pair is replaced by a pair whose degraded member is a hard type drawn with weights "
      + ", ".join(f"{k} {v:.2f}" for k, v in a5.hard_weight_map.items())
      + f" from the same dataset x class cell; otherwise the Test 4 rule is used ({a5.normal_pool} pool). Severity follows the curriculum. "
        "Realised shares per run: `runs/<run>/results/logs/hard_sampling_realised.csv`.\n")
    ed = root / "runs" / f"5D_{T4.train_tag(pt)}_seed{ps}" / "results" / "logs" / "hard_sampling_expected.csv"
    rl = root / "runs" / f"5D_{T4.train_tag(pt)}_seed{ps}" / "results" / "logs" / "hard_sampling_realised.csv"
    if ed.exists():
        e_ = pd.read_csv(ed)
        e_ = e_[e_.cell.str.startswith("ALL")][["degradation", "expected_share_pct"]]
        if rl.exists():
            r_ = pd.read_csv(rl).groupby("degradation").pairs.sum()
            e_["realised_share_pct"] = e_.degradation.map(100 * r_ / r_.sum())
        w(e_.sort_values("expected_share_pct", ascending=False).round(2).pipe(md, index=False) + "\n")
    w("## 11. Detection results\n")
    if len(T["det"]):
        d = T["det"]
        d = d[(d.train_on == pt) & (d.condition == "original") & (d.setting == "in-domain")]
        g = d.groupby("model")[["auc", "accuracy", "balanced_accuracy", "precision", "recall", "specificity", "f1", "tn", "fp", "fn", "tp"]].mean()
        g = g.reindex([m for m in MODEL_ORDER if m in g.index])
        w(f"In-domain {pt} test set, original videos, validation thresholds (mean over seeds; confusion counts are seed means):\n")
        w(g.round(4).pipe(md) + "\n")
    w("## 12. Robustness results\n")
    rcols = ["clean_auc", "degraded_auc", "auc_drop", "degraded_all_versions_auc", "flip_rate_pct", "flip_rate_all_versions_pct",
             "mean_degradation_auc", "worst_degradation_auc"]
    w(cp[[c for c in rcols if c in cp]].round(4).pipe(md) + "\n")
    w("`degraded_auc` / `flip_rate_pct` use one random degraded version per test video (Test 4 definition); `*_all_versions` use every version. "
      "Per type x severity: `degradation_metrics_test5.csv`.\n")
    w("## 13. Explanation stability\n")
    xcols = ["cam_iou", "cam_pearson", "cam_center_shift", "cam_temporal_r", "explanation_change_rate_pct", "cam_clean_ref_iou",
             "mean_degradation_iou", "worst_degradation_iou"]
    w(cp[[c for c in xcols if c in cp]].round(4).pipe(md) + "\n")
    w("Grad-CAM pairs: fixed, stratified by (test dataset, degradation type), aligned original/degraded crops, target class 'fake', same layer as "
      "Test 4. `cam_clean_ref_iou` = IoU between the clean clip and the neighbouring clean frames (no degradation) - a ceiling for "
      "stability. **Stability was partly optimised directly by the Grad-CAM consistency loss (5A, 5C, 5D, Test 4D), so higher stability of "
      "those models is expected by construction and is not evidence of better explanations.**\n")
    w("### Degradation-wise explanation robustness (top-20 % IoU)\n")
    w(deg_table.round(4).pipe(md, index=False) + "\n")
    dd = deg_table.dropna(subset=["delta_IoU_5D_minus_T4"])
    if len(dd):
        w(f"- Largest improvement (5D - Test 4): **{dd.loc[dd.delta_IoU_5D_minus_T4.idxmax(), 'degradation']}** "
          f"({dd.delta_IoU_5D_minus_T4.max():+.3f}); smallest change: **{dd.loc[dd.delta_IoU_5D_minus_T4.idxmin(), 'degradation']}** "
          f"({dd.delta_IoU_5D_minus_T4.min():+.3f}).")
        worse = dd[dd.delta_IoU_5D_minus_T4 < 0].degradation.tolist()
        w(f"- Degradations that became worse: {', '.join(worse) if worse else 'none'}"
          + (" (CI columns show whether the change is beyond sampling noise)." if worse else ".") + "\n")
    for sec, t in (("14. Brightness analysis", "brightness"), ("15. Saturation analysis", "saturation")):
        w(f"## {sec}\n")
        h = hard[(hard.train_on == pt) & (hard.degradation == t)]
        if len(h):
            tab = h.pivot_table(index="metric", columns="model", values="test5_mean_over_seeds")
            tab.insert(0, "Test 4D", h.groupby("metric").test4D.first())
            w(tab.round(4).pipe(md) + "\n")
            for m in ("5A", "5D"):
                w(f"- {m} vs Test 4D (seed {ps}, paired bootstrap): IoU {diff_str(stats, pt, f'iou|{t}', f'{m} - T4D')}, Pearson "
                  f"{diff_str(stats, pt, f'pearson|{t}', f'{m} - T4D')}, AUC {diff_str(stats, pt, f'auc|{t}', f'{m} - T4D', 4)}, accuracy "
                  f"{diff_str(stats, pt, f'accuracy|{t}', f'{m} - T4D', 4)}, flip rate {diff_str(stats, pt, f'flip_rate_pct|{t}', f'{m} - T4D', 2, True)}")
            w(f"- 5D vs 5A (effect of hard sampling alone): IoU {diff_str(stats, pt, f'iou|{t}', '5D - 5A')}, flip rate "
              f"{diff_str(stats, pt, f'flip_rate_pct|{t}', '5D - 5A', 2, True)}")
            w(f"- Test 4D IoU {fmt(h.test4D[h.metric == 'iou'].iloc[0])} -> Test 5D IoU "
              f"{fmt(h[(h.metric == 'iou') & (h.model == '5D')].test5_mean_over_seeds.iloc[0]) if len(h[(h.metric == 'iou') & (h.model == '5D')]) else 'n/a'}; "
              f"flip rate {fmt(h.test4D[h.metric == 'flip_rate_pct'].iloc[0], 2)}% -> "
              f"{fmt(h[(h.metric == 'flip_rate_pct') & (h.model == '5D')].test5_mean_over_seeds.iloc[0], 2) if len(h[(h.metric == 'flip_rate_pct') & (h.model == '5D')]) else 'n/a'}% "
              "(* = 95 % CI excludes 0).\n")
    w("## 16. Cross-dataset results\n")
    cdf = root / "cross_dataset_test5.csv"
    if cdf.exists():
        c = pd.read_csv(cdf)
        c = c[c.setting == "cross-dataset"]
        g = c.groupby(["direction", "condition", "model"])[["auc", "accuracy", "pred_fake_rate", "mean_p_fake", "fpr", "fnr", "auc_audio_masked"]].mean()
        w(g.round(4).pipe(md) + "\n")
    w("The cross-dataset models are the Stage-2 models of the other training set, tested with their own validation threshold (as in Test 4). "
      "A predicted-fake rate near 0 or 1 means the threshold does not transfer; AUC measures ranking independent of the threshold.\n")
    w("## 17. Audio ablation\n")
    if len(T["aud"]):
        a_ = T["aud"]
        a_ = a_[(a_.train_on == pt) & (a_.scope == "with_audio") & (a_.dataset == "Combined")]
        g = a_.groupby(["condition", "model"])[["auc", "flip_rate_vs_normal_pct", "auc_difference_normal_minus_condition", "mean_p_fake"]].mean().unstack("model")
        w(f"Test videos WITH audio, models trained on {pt} (mean over seeds):\n")
        w(g.round(4).pipe(md) + "\n")
    w("zero_audio = silent spectrogram, audio flag kept; video_only = audio features removed (audio-masking flip rate); degraded_audio = one "
      "random waveform degradation per video (data seed); audio_only = video features removed. Large flips under video_only would mean the "
      "decision depends on audio presence (dataset shortcut: only BioDeepAV has audio).\n")
    w("## 18. Worst-case robustness\n")
    w(cp[[c for c in ["clean_auc", "mean_degradation_auc", "worst_degradation_auc", "worst_degradation_auc_type", "robustness_gap",
                      "cam_clean_ref_iou", "mean_degradation_iou", "worst_degradation_iou", "worst_degradation_iou_type", "explanation_gap"] if c in cp]]
      .round(4).pipe(md) + "\n")
    w("Robustness gap = clean AUC - worst degradation AUC; explanation gap = clean reference IoU - worst degradation IoU. Goal: smaller gaps.\n")
    w("## 19. Comparison with Test 4\n")
    keep = [k for k, _ in COMP_COLS[:10]]
    tab = cp[[k for k in keep if k in cp]].copy()
    tab.columns = [dict(COMP_COLS)[k] for k in tab.columns]
    tab.index = [MODEL_LABEL.get(m, m) for m in tab.index]
    w(tab.round(4).pipe(md) + "\n")
    s4 = a5.test4_dir / "suite_comparison.csv"
    if s4.exists():
        w(f"Published Test 4 values (from `{s4.name}`, different Grad-CAM sample) are kept in final_experiment_summary.json for traceability.\n")
    w("## 20. Statistical analysis\n")
    w(f"Paired bootstrap ({a5.n_boot} resamples; clusters = test videos, or Grad-CAM pairs) on seed {ps}: 95 % percentile CIs of each metric "
      "and of differences between models evaluated on identical inputs; Wilcoxon signed-rank on per-pair IoU / r; exact McNemar on per-video "
      "prediction flips. Seed-to-seed variation (sd over seeds) is in test5_comparison.csv. No correction for multiple comparisons is applied - "
      "the primary, pre-specified tests are brightness/saturation IoU and flip rate (5D vs Test 4D and 5D vs 5A); everything else is exploratory.\n")
    key = ["clean_auc", "degraded_auc", "flip_rate_pct", "cam_iou", "cam_pearson", "explanation_change_rate_pct", "worst_degradation_auc",
           "worst_degradation_iou", "brightness_saturation_iou", "hard_types_auc", "hard_types_flip_rate_pct", "cross_auc"]
    if len(stats):
        st = stats[(stats.train_on == pt) & stats.metric.isin(key) & stats.comparison.isin(["5D - T4D", "5D - 5A", "5A - T4D"] + ([f"{bm} - T4D"] if bm else []))]
        if len(st):
            w(st[["metric", "comparison", "diff", "diff_ci_lo", "diff_ci_hi", "p_boot", "significant"] + (["test_p"] if "test_p" in st else [])]
              .drop_duplicates().round(4).pipe(md, index=False) + "\n")
    w("## 21. Failure cases\n")
    xg = root / "gradcam_stability_test5.csv"
    if xg.exists():
        x = pd.read_csv(xg)
        x = x[(x.train_on == pt) & ((x.model == "T4D") | ((x.model == "5D") & (x.seed == ps)))]
        if set(x.model) >= {"T4D", "5D"}:
            pv = x.pivot_table(index=["pair_id", "video", "dataset", "degradation", "severity"], columns="model", values="top20_iou").dropna()
            pv["delta"] = pv["5D"] - pv["T4D"]
            w("Ten pairs where 5D explanations became LEAST stable relative to Test 4D (same pairs):\n")
            w(pv.sort_values("delta").head(10).round(3).reset_index().pipe(md, index=False) + "\n")
            fx = x[(x.model == "5D") & x.prediction_stable & x.explanation_changed]
            w(f"5D pairs with a stable prediction but changed explanation (IoU < 0.5): {len(fx)} of {int((x.model == '5D').sum())}; by type: "
              + ", ".join(f"{k} {v}" for k, v in fx.degradation.value_counts().items()) + "\n")
            ff = x[(x.model == "5D") & ~x.prediction_stable]
            w(f"5D prediction flips in the Grad-CAM sample: {len(ff)}; by type: " + ", ".join(f"{k} {v}" for k, v in ff.degradation.value_counts().items()) + "\n")
    w("## 22. Limitations\n")
    lims = ["Grad-CAM stability is partly optimised by the training loss of 5A/5C/5D/Test 4D; stability is therefore not an independent "
            "measure of explanation quality, and it is not a measure of correctness.",
            "No localisation annotations (ExDDV-style masks or boxes) exist in this project, so explanation CORRECTNESS (IoU with the "
            "manipulated region, pointing game, localisation hit rate) could not be measured" + (" (except via the supplied --localization-csv)." if a5.localization_csv else "."),
            "The hard-degradation set was chosen from Test 4 Grad-CAM results based on only 6-15 pairs per type; the choice was fixed before "
            "Test 5 and not tuned on Test 5 test data.",
            "Only the provided degraded videos are used; for each original at most 3 degradation types exist, so hard sampling re-weights the "
            "existing training pairs instead of creating new degradations.",
            f"Seeds: {', '.join(str(int(s)) for s in sorted(S[S.model != 'T4D'].seed.unique()))}. Bootstrap CIs describe test-set sampling "
            "uncertainty for one seed; seed variation is reported separately (sd). GPU kernels are not fully deterministic.",
            "lambda_feat and lambda_cam were kept at the Test 4 values (no hyper-parameter search).",
            "The Grad-CAM sample includes cross-dataset pairs (pooled over test datasets, as in Test 4); in-domain-only values are in "
            "gradcam_stability_test5.csv.",
            "Hash-based duplicate screening detects identical files and identical decoded clips, not re-encoded near-duplicates."]
    for l_ in lims:
        w(f"- {l_}")
    w("")
    w("## 23. Whether hypotheses were supported\n")
    h1, s1 = hypothesis_status(stats, pt, "hard_types_auc", "5D - 5A", True)
    h1f, _ = hypothesis_status(stats, pt, "hard_types_flip_rate_pct", "5D - 5A", False)
    h2, s2 = hypothesis_status(stats, pt, "brightness_saturation_iou", "5D - 5A", True)
    h2b, _ = hypothesis_status(stats, pt, "brightness_saturation_iou", "5B - 5A", True)
    h3a, _ = hypothesis_status(stats, pt, "worst_degradation_auc", "5D - T4D", True)
    h3i, _ = hypothesis_status(stats, pt, "worst_degradation_iou", "5D - T4D", True)
    h4c, s4c = hypothesis_status(stats, pt, "cross_auc", "5D - T4D", True)
    h4 = ("supported (cross-dataset AUC did not improve significantly)" if not h4c.startswith("supported") else
          "not supported in this run (cross-dataset AUC improved significantly; see section 16 before drawing conclusions)")
    h5 = ("not empirically testable here (no localisation annotations); it remains a methodological statement" if not a5.localization_csv
          else "see localisation metrics in final_experiment_summary.json")
    hyp = [("H1", f"hard-type AUC 5D-5A: {h1}; hard-type flip rate 5D-5A: {h1f} "
                  f"({diff_str(stats, pt, 'hard_types_auc', '5D - 5A', 4)}; flips {diff_str(stats, pt, 'hard_types_flip_rate_pct', '5D - 5A', 2, True)}); "
                  f"seed consistency vs Test 4D: {seed_consistency(S, pt, 'degraded_auc', '5D')}"),
           ("H2", f"brightness+saturation IoU 5D-5A: {h2} ({diff_str(stats, pt, 'brightness_saturation_iou', '5D - 5A')}); 5B-5A: {h2b} "
                  f"({diff_str(stats, pt, 'brightness_saturation_iou', '5B - 5A')}; 5B also lacks the consistency losses)"),
           ("H3", f"worst-degradation AUC 5D-T4D: {h3a} ({diff_str(stats, pt, 'worst_degradation_auc', '5D - T4D', 4)}); worst-degradation IoU: "
                  f"{h3i} ({diff_str(stats, pt, 'worst_degradation_iou', '5D - T4D')})"),
           ("H4", f"{h4} ({diff_str(stats, pt, 'cross_auc', '5D - T4D', 4)})"), ("H5", h5)]
    w("| hypothesis | result |\n|---|---|\n" + "\n".join(f"| {h} | {r} |" for h, r in hyp) + "\n")
    w("'supported' requires a 95 % bootstrap CI that excludes 0 in the predicted direction (seed " + str(ps) + "); a same-direction but "
      "non-significant change is reported as 'not supported'.\n")
    w("## 24. Recommended thesis interpretation\n")
    verdict = improvement_verdict(stats, pt, bm or "5D")
    w(f"Validation-selected best Test 5 model: **{MODEL_LABEL.get(bm, bm)}** ({best.get('rule')}). Compared with Test 4D on identical test "
      f"inputs: {verdict['text']}\n")
    w("Interpretation guidance: report improvements only where the CI excludes 0 and the direction is consistent across seeds; treat "
      "Grad-CAM stability as *stability*, not correctness; and state explicitly that degradation robustness and cross-dataset generalisation are "
      "separate problems (section 16).\n")
    w("## 25. Suggested thesis paragraph\n")
    b = bm or "5D"
    w(f"> In Test 5 we kept the Test 4D architecture, data split and evaluation protocol fixed and changed only the Stage-2 training strategy: "
      f"half of the training pairs were drawn from the degradation types that produced the least stable explanations in Test 4 "
      f"(brightness, saturation, contrast, blur). On the {pt} test set, the validation-selected model ({MODEL_LABEL.get(b, b)}) reached a clean AUC of "
      f"{val(b, 'clean_auc')} and a degraded AUC of {val(b, 'degraded_auc')} (Test 4D: {val('T4D', 'clean_auc')} / {val('T4D', 'degraded_auc')}), "
      f"a prediction flip rate of {val(b, 'flip_rate_pct', 2)} % (Test 4D: {val('T4D', 'flip_rate_pct', 2)} %), and a mean Grad-CAM top-20 % IoU "
      f"of {val(b, 'cam_iou')} (Test 4D: {val('T4D', 'cam_iou')}). Under brightness the IoU changed by {diff_str(stats, pt, 'iou|brightness', f'{b} - T4D')} "
      f"and under saturation by {diff_str(stats, pt, 'iou|saturation', f'{b} - T4D')} (paired bootstrap, 95 % CI). {verdict['sentence']} "
      f"Cross-dataset AUC remained {val(b, 'cross_auc')} (Test 4D {val('T4D', 'cross_auc')}), confirming that robustness to quality degradations does "
      "not transfer to unseen manipulation datasets. Because the Grad-CAM consistency loss directly rewards stable maps and no localisation "
      "annotations were available, these results concern explanation stability only, not explanation correctness.\n")
    w("## 26. Recommended tables and figures\n")
    w("Tables: test5_comparison.csv (section 19), degradation_explanation_robustness_test5.csv (section 13), hard_degradation_analysis.csv "
      "(sections 14-15), cross_dataset_test5.csv (16), audio_ablation_test5.csv (17), statistics_test5.csv (20).\n")
    w("Figures (`figures/`): fig01/fig02 clean and degraded AUC, fig03/fig04 degradation-wise AUC and IoU, fig05/fig06 brightness and saturation "
      "IoU, fig07 flip rate, fig08 explanation-change rate, fig09 ROC clean vs degraded, fig10 cross-dataset AUC, fig11 worst case, fig12 "
      "training curves, fig13 Grad-CAM examples (same pairs for Test 4D and Test 5D). Thesis core: fig04, fig05, fig06, fig11, fig13.\n")
    (root / "TEST5_REPORT.md").write_text("\n".join(L), encoding="utf-8")
    return dict(best=best, verdict=verdict)


def improvement_verdict(stats, t, m) -> dict:
    keys = [("degraded_auc", "degraded AUC"), ("flip_rate_pct", "flip rate"), ("cam_iou", "Grad-CAM IoU"), ("cam_pearson", "Pearson r"),
            ("explanation_change_rate_pct", "explanation-change rate"), ("worst_degradation_auc", "worst-degradation AUC"),
            ("worst_degradation_iou", "worst-degradation IoU"), ("clean_auc", "clean AUC")]
    better, worse = [], []
    for k, lab in keys:
        s = stat_row(stats, t, k, f"{m} - T4D")
        if s is None or not s.significant:
            continue
        (better if s.better else worse).append(lab)
    if better and not worse:
        v = "YES - significant improvement"
    elif better and worse:
        v = "MIXED"
    elif worse:
        v = "NO - significantly worse"
    else:
        v = "NO significant difference"
    text = (f"**{v}**. Significantly better: {', '.join(better) or 'none'}; significantly worse: {', '.join(worse) or 'none'} "
            f"(95 % paired bootstrap CI excludes 0).")
    sentence = {"YES - significant improvement": f"Relative to Test 4D, the model improved significantly in {', '.join(better)} without significant losses.",
                "MIXED": f"The change was mixed: better in {', '.join(better)}, worse in {', '.join(worse)}.",
                "NO - significantly worse": f"The targeted training did not help; the model was significantly worse in {', '.join(worse)}.",
                "NO significant difference": "None of the headline differences to Test 4D exceeded test-set sampling noise."}[v]
    return dict(verdict=v, better=better, worse=worse, text=text, sentence=sentence)


def finalize_outputs(root: Path, a5, env: dict, data_info: dict):
    """Top-level artefacts: final_model_5X.pt, config_5X.json, training_history_5X.csv, tables, figures, report, summary."""
    T = build_report_tables(root, a5)
    if not T:
        return None
    tag = T4.train_tag(a5.primary_train)
    ckpts = {}
    for v in VARIANTS5:
        rd = root / "runs" / f"{v}_{tag}_seed{a5.primary_seed}"
        f = rd / "results" / "final" / "final_model.pt"
        if f.exists():
            shutil.copy(f, root / f"final_model_{v}.pt")
            ckpts[v] = dict(path=str(root / f"final_model_{v}.pt"), sha256=T4.file_sha256(root / f"final_model_{v}.pt"))
        if (rd / "config.json").exists():
            cfg = json.load(open(rd / "config.json", encoding="utf-8"))
            runs = sorted(str(p.name) for p in (root / "runs").glob(f"{v}_*"))
            T4.write_json(root / f"config_{v}.json", dict(primary_run=rd.name, config=cfg, all_runs_of_variant=runs))
        if (rd / "training_history.csv").exists():
            shutil.copy(rd / "training_history.csv", root / f"training_history_{v}.csv")
    sp = root / "data" / tag / "split_test5.json"
    if sp.exists():
        shutil.copy(sp, root / "split_test5.json")
    mf = root / "data" / tag / "dataset_manifest_test5.csv"
    if mf.exists():
        shutil.copy(mf, root / "dataset_manifest_test5.csv")
    make_figures(root, T["S"], T["comp"], T["stats"], a5.primary_train, a5.primary_seed, T["deg_table"])
    rep = write_report(root, a5, T, env, data_info)
    S = T["S"]
    published = None
    s4 = a5.test4_dir / "suite_comparison.csv"
    if s4.exists():
        published = pd.read_csv(s4).to_dict(orient="records")
    summary = dict(
        test="Test 5 - hard-degradation and explanation-aware training", script="8-10-26.py", baseline="Test 4D (6-10-26.py)",
        created=datetime.now().strftime("%Y-%m-%d %H:%M:%S"), primary_train=a5.primary_train, primary_seed=a5.primary_seed,
        seeds=sorted(int(s) for s in S[S.model != "T4D"].seed.unique()), data_seed=a5.data_seed, variants={k: v["description"] for k, v in VARIANTS5.items()},
        hard_sampling=dict(share=a5.hard_share, weights=a5.hard_weight_map, normal_pool=a5.normal_pool,
                           selection_basis="Test 4 Grad-CAM per-degradation IoU (completed experiment), not Test 5 test data"),
        lambda_feat=a5.lambda_feat, lambda_cam=a5.lambda_cam, best_model_validation=rep["best"], verdict_vs_test4=rep["verdict"],
        comparison=T["comp"].to_dict(orient="records"), runs=S.drop(columns=["config"], errors="ignore").to_dict(orient="records"),
        final_checkpoints=ckpts, data=data_info, reproducibility=env, published_test4_suite_comparison=published,
        explanation_correctness=("not measured: no localisation annotations in the project" if not a5.localization_csv else "see runs[*].localization"),
        architecture_signatures=sorted(set(S.architecture_signature.dropna())) if "architecture_signature" in S else [],
    )
    T4.write_json(root / "final_experiment_summary.json", summary)
    print_final(T, a5, rep)
    return summary


def print_final(T, a5, rep):
    cp = T["comp"][T["comp"].train_on == a5.primary_train].set_index("model")
    b = rep["best"].get("best")
    if b is None or b not in cp.index:
        return
    r = cp.loc[b]
    t4 = cp.loc["T4D"] if "T4D" in cp.index else None
    g = lambda row, k, nd=4: fmt(row[k], nd) if row is not None and k in row else "n/a"
    print("\n" + "=" * 72 + f"\n TEST 5 SUMMARY (train {a5.primary_train}; mean over {int(r.n_seeds)} seed(s); in-domain test)\n" + "=" * 72)
    print(f"BEST MODEL:              {MODEL_LABEL[b]}  (chosen on validation only)")
    for lab, k, nd in (("CLEAN AUC", "clean_auc", 4), ("DEGRADED AUC", "degraded_auc", 4), ("AUC DROP", "auc_drop", 4), ("FLIP RATE %", "flip_rate_pct", 2),
                       ("GRAD-CAM IoU", "cam_iou", 3), ("PEARSON", "cam_pearson", 3), ("EXPLANATION CHANGE RATE %", "explanation_change_rate_pct", 1)):
        print(f"{lab + ':':25s}{g(r, k, nd):>8s}   (Test 4D {g(t4, k, nd)})")
    print(f"{'WORST DEGRADATION:':25s}{r.worst_degradation_auc_type} (AUC), {r.worst_degradation_iou_type} (IoU)")
    print(f"{'WORST-CASE AUC:':25s}{g(r, 'worst_degradation_auc'):>8s}   (Test 4D {g(t4, 'worst_degradation_auc')})")
    print(f"{'CROSS-DATASET AUC:':25s}{g(r, 'cross_auc'):>8s}   (Test 4D {g(t4, 'cross_auc')})")
    print(f"\nIS TEST 5 AN IMPROVEMENT OVER TEST 4?  {rep['verdict']['text']}")
    print(f"Report: {a5.root / 'TEST5_REPORT.md'}")


# =====================================================================================
# SELF TEST
# =====================================================================================
def selftest(a5) -> bool:
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}{(' - ' + detail) if detail else ''}")

    print("=" * 80 + "\n TEST 5 SELF TEST\n" + "=" * 80)
    register_variants(a5.lambda_feat, a5.lambda_cam)
    ns = {v: t4_args(a5, v, "BioDeepAV") for v in VARIANTS5}
    check("variant presets", ns["5A"].lambda_feat == 0.1 and ns["5A"].lambda_cam == 0.05 and ns["5B"].lambda_feat == 0 and ns["5B"].lambda_cam == 0
          and ns["5B"].paired and ns["5C"].lambda_cam == 0.05 and ns["5C"].lambda_feat == 0 and ns["5D"].lambda_feat == 0.1
          and all(n.curriculum and n.audio_policy == "mixed" and n.res_norm and n.seed == a5.data_seed for n in ns.values()))
    p4 = T4.parse_args(["--variant", "4D"])
    same = all(getattr(ns["5A"], k) == getattr(p4, k) for k in ("img_size", "frames", "clip_len", "batch_size", "lr1", "lr2", "epochs1", "epochs2",
                                                                 "curriculum_epochs", "audio_probs", "pair_batch_size", "sampling", "norm"))
    check("5A protocol == Test 4D protocol", same)
    m = T4.AudioVisualDeepFakeCNN()
    check("architecture = Test 4 AudioVisualDeepFakeCNN", architecture_signature(m) == architecture_signature(T4.AudioVisualDeepFakeCNN()),
          f"{sum(p.numel() for p in m.parameters()):,} parameters")

    # synthetic experiment for the pair sampler
    rng = np.random.default_rng(0)
    N, T, S = 40, 16, 32
    types = ["blur", "brightness", "compression", "contrast", "frame_drop", "motion_blur", "noise", "saturation"]
    e = Experiment5.__new__(Experiment5)
    e.T, e.L = T, 8
    e.args = SimpleNamespace(audio_policy="mixed", audio_probs=(0.5, 0.25, 0.25), audio_dropout=0.2, synth_per_video=0, sampling="class",
                             epoch_size=None, max_oversample=4.0)
    e.labels = np.array([i % 2 for i in range(N)])
    e.ds_of = np.array(["BioDeepAV"] * N)
    deg_src, deg_type, deg_level = [], [], []
    for s in range(N):
        for t in rng.choice(types, 3, replace=False):
            deg_src.append(s)
            deg_type.append(t)
            deg_level.append(int(rng.integers(3)))
    e.deg_src, e.deg_type, e.deg_level = np.array(deg_src), np.array(deg_type), np.array(deg_level)
    e.src_to_deg = defaultdict(list)
    for j, s in enumerate(deg_src):
        e.src_to_deg[s].append(j)
    e.view = T4.View("x", S, SimpleNamespace(arr=rng.integers(0, 255, (N, T, S, S, 3), dtype=np.uint8)),
                     SimpleNamespace(arr=rng.integers(0, 255, (len(deg_src), T, S, S, 3), dtype=np.uint8)), True)
    e.audio = rng.normal(size=(N, 1, 128, 128)).astype(np.float32)
    e.has_audio = np.ones(N, bool)
    e.wave = None
    e._lock = threading.Lock()
    e.sampling_log = Counter()
    e.idx = {"train": np.arange(N)}
    e.setup_hard_pool()
    e.hard = dict(enabled=False)
    src = list(range(8))
    a_ = T4.Experiment.make_pair_batch(e, e.view, src, 7, "av", 2)
    b_ = e.make_pair_batch(e.view, src, 7, "av", 2)
    check("hard sampling OFF reproduces Test 4 make_pair_batch exactly", all(np.array_equal(x, y) for x, y in zip(a_, b_)))
    e.hard = dict(enabled=True, share=a5.hard_share, weights=dict(a5.hard_weight_map), normal_pool=a5.normal_pool)
    e.sampling_log.clear()
    for sd in range(400):
        e.make_pair_batch(e.view, list(rng.integers(0, N, 16)), sd, "av", 2)
    cnt = Counter()
    for (lv, mode, t), v in e.sampling_log.items():
        cnt[t] += v
    tot = sum(cnt.values())
    share = {t: cnt[t] / tot for t in types}
    exp = e.expected_type_distribution()
    exp_all = exp[exp.cell.str.startswith("ALL")].set_index("degradation").expected_share_pct / 100
    err = max(abs(share[t] - exp_all.get(t, 0)) for t in types)
    check("realised hard-sampling shares match the expected shares", err < 0.02,
          ", ".join(f"{t} {100 * share[t]:.1f}% (exp {100 * exp_all.get(t, 0):.1f}%)" for t in types))
    hard_share = sum(share[t] for t in a5.hard_weight_map)
    check("hard types over-represented, no type removed", hard_share > 0.6 and min(share.values()) > 0.02, f"hard total {100 * hard_share:.1f}%")
    y = rng.integers(0, 2, 500)
    p = np.clip(0.3 * y + rng.normal(0.4, 0.2, 500), 0, 1)
    from sklearn.metrics import roc_auc_score
    check("fast AUC == sklearn", abs(fast_auc(y, p) - roc_auc_score(y, p)) < 1e-12)
    bt = Boot(np.arange(500), 200, 0)
    est, bs = bt.run(fast_auc, (y, p))
    lo, hi = ci(bs)
    check("bootstrap CI contains estimate", lo <= est <= hi, f"{est:.3f} [{lo:.3f}, {hi:.3f}]")
    cam = np.zeros((16, 16))
    cam[2:6, 2:6] = 1
    ls = localization_scores(cam, (0.1, 0.1, 0.4, 0.4))
    check("localisation scores", ls["pointing"] == 1 and ls["hit"] == 1)
    print("=" * 80 + f"\n SELF TEST {'PASSED' if ok else 'FAILED'}\n" + "=" * 80)
    return ok


# =====================================================================================
# MAIN
# =====================================================================================
def main(argv=None):
    a5 = parse_args5(sys.argv[1:] if argv is None else argv)
    if a5.mode == "selftest":
        sys.exit(0 if selftest(a5) else 1)
    a5.root.mkdir(parents=True, exist_ok=True)
    env = environment_info()
    T4.write_json(a5.root / "reproducibility.json", dict(env, seeds=a5.seed_list, data_seed=a5.data_seed, argv=sys.argv))
    if a5.mode == "report":
        di = json.load(open(a5.root / "data_info.json", encoding="utf-8")) if (a5.root / "data_info.json").exists() else {}
        finalize_outputs(a5.root, a5, env, di)
        return
    if a5.data_seed != 42:
        print(f"[WARN] --data-seed {a5.data_seed} differs from the Test 4 seed (42): the split will not match Test 4.")
    print("=" * 96 + f"\n TEST 5 | variants {a5.variant_list} | seeds {a5.seed_list} | training sets {a5.train_list} | work dir {a5.root}\n"
          f" Test 4 baseline: {a5.test4_dir} (read only) | passthrough options: {a5.passthrough or 'none'}\n" + "=" * 96)
    exps: Dict[str, Experiment5] = {}
    data_info = {}
    t_start = time.time()
    for t in a5.train_list:  # data preparation + leakage check + Test 4D reference for every training set first
        ex = Experiment5(t4_args(a5, a5.variant_list[0], t), a5, t)
        ex.prepare_data()
        exps[t] = ex
        data_info[t] = dict(split_check=ex.split_check, leakage=ex.leakage_t5, equivalence_5a_vs_test4=ex.equivalence_5a,
                            manifest_sha256=ex.manifest_sha256, n_xai_test_pairs=len(ex.xai_test), n_xai_val_pairs=len(ex.xai_val),
                            inventory=ex.df_inv.reset_index().to_dict(orient="records"), face_detection=getattr(ex, "face_summary", None),
                            n_train=len(ex.idx["train"]), n_val=len(ex.idx["val"]), n_test=len(ex.idx["test"]))
    di = dict(data_info.get(a5.primary_train, {}), per_training_set=data_info)
    T4.write_json(a5.root / "data_info.json", di)
    if a5.mode == "leakage":
        print(f"\n[leakage mode] done - see {a5.root / 'data'}")
        return
    for t, ex in exps.items():
        ex.evaluate_test4_reference()
    order = [(s, t, v) for s in a5.seed_list for t in a5.train_list for v in a5.variant_list]
    for k, (s, t, v) in enumerate(order, 1):
        ex = exps[t]
        print(f"\n>>> RUN {k}/{len(order)}  ({T4.format_time(time.time() - t_start)} elapsed)")
        ex.set_run(t4_args(a5, v, t), v, s)
        ex.run_variant()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if k % len(a5.variant_list) == 0:  # refresh tables after every complete seed x training set
            try:
                finalize_outputs(a5.root, a5, env, di)
            except Exception as exc:  # reporting must never kill a long training run
                print(f"[report] intermediate report failed: {exc}")
    finalize_outputs(a5.root, a5, env, di)
    print(f"\nTest 5 finished in {T4.format_time(time.time() - t_start)} | results in {a5.root}")


if __name__ == "__main__":
    main()
