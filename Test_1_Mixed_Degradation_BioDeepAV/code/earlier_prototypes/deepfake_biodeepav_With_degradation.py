"""
DeepFake BioDeepAV Training Pipeline With Video Degradations
============================================================
Trains and evaluates DeepFakeVideoCNN on both authentic/original (clean)
videos and quality-degraded variants (blur, brightness, compression, contrast,
fps_reduction, frame_drop, motion_blur, noise, saturation).

Uses memory-mapped caching (np.memmap) to efficiently stream frames without
exceeding RAM limits, preserves exact baseline train/val/test splits to eliminate
data leakage, and produces comprehensive per-degradation robustness reports.
"""

import os
import re
import sys
import json
import time
import random
import warnings
import subprocess
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

warnings.filterwarnings("ignore")
cv2.setNumThreads(1)


def format_time(seconds):
    """Format seconds into HH:MM:SS or MM:SS."""
    if seconds is None or np.isnan(seconds) or seconds < 0:
        return "--:--"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h > 0 else f"{m:02d}:{s:02d}"


def progress_bar(current, total, bar_len=20):
    """Render an ASCII progress bar [====>.....] 45.0%."""
    frac = min(1.0, max(0.0, current / max(1, total)))
    filled = int(frac * bar_len)
    bar = "=" * filled + (">" if filled < bar_len else "")
    bar = bar.ljust(bar_len, ".")
    return f"[{bar}] {frac * 100:5.1f}%"


_last_gpu_temp_check = 0
_cached_gpu_temp = None


def get_gpu_temp(ttl=1.5):
    """Fetch GPU temperature in Celsius, cached for ttl seconds."""
    global _last_gpu_temp_check, _cached_gpu_temp
    now = time.time()
    if now - _last_gpu_temp_check < ttl and _cached_gpu_temp is not None:
        return _cached_gpu_temp
    if not torch.cuda.is_available():
        return None
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
            creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0,
            timeout=1.5,
        )
        _cached_gpu_temp = int(out.decode().strip().split()[0])
        _last_gpu_temp_check = now
        return _cached_gpu_temp
    except Exception:
        return _cached_gpu_temp


def check_gpu_thermal_guard(max_temp, pause_sec):
    """Check GPU temp. If >= max_temp, pause execution for pause_sec with live countdown."""
    curr_temp = get_gpu_temp(ttl=0.0)
    if curr_temp is None or curr_temp < max_temp:
        return 0

    print(f"\n\n{'!' * 78}")
    print(f" [THERMAL WARNING] GPU Temperature reached {curr_temp}C (Threshold: {max_temp}C)!")
    print(f" Pausing training for {pause_sec // 60} minutes to let the GPU cool down...")
    print(f"{'!' * 78}\n")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    t_pause_start = time.time()
    while True:
        elapsed_pause = time.time() - t_pause_start
        rem_pause = pause_sec - elapsed_pause
        if rem_pause <= 0:
            break
        temp_now = get_gpu_temp(ttl=0.0)
        temp_str = f"{temp_now}C" if temp_now is not None else "N/A"
        sys.stdout.write(f"\r  [Cooling Down] Remaining: {format_time(rem_pause)} | Current GPU Temp: {temp_str}   ")
        sys.stdout.flush()
        time.sleep(1.0)

    temp_after = get_gpu_temp(ttl=0.0)
    temp_after_str = f"{temp_after}C" if temp_after is not None else "N/A"
    sys.stdout.write(f"\r  [Cooling Down Complete] Cooled to {temp_after_str}. Resuming training!                  \n\n")
    sys.stdout.flush()
    return int(time.time() - t_pause_start)


# ------------------------- CONFIGURATION -------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_DEFAULT_ROOT = (
    _SCRIPT_DIR.parent / "BioDeepAV"
    if (_SCRIPT_DIR.parent / "BioDeepAV").exists()
    else Path(r"E:\Datasets\BioDeepAV")
)
_DEFAULT_WORK = _SCRIPT_DIR / "biodeep_run"
_DEFAULT_REPORT = _SCRIPT_DIR / "test"

DATA_ROOT = Path(os.environ.get("BIODEEP_ROOT", str(_DEFAULT_ROOT)))
WORK_DIR = Path(os.environ.get("BIODEEP_WORK", str(_DEFAULT_WORK)))
REPORT_DIR = Path(os.environ.get("BIODEEP_REPORT", str(_DEFAULT_REPORT)))
CACHE_DIR = WORK_DIR / "cache_degraded"

GPU_MAX_TEMP = int(os.environ.get("BIODEEP_GPU_MAX_TEMP", 72))
GPU_PAUSE_SECONDS = int(os.environ.get("BIODEEP_GPU_PAUSE", 300))

N_FRAMES = 16
IMG = 96
CLIP_LEN = 8
BATCH = 16
_epochs_env = os.environ.get("BIODEEP_EPOCHS", "").strip()
EPOCHS = int(_epochs_env) if _epochs_env.isdigit() else 12
LR = 1e-3
TEST_FRAC = 0.20
VAL_FRAC = 0.10
SEED = 42
_maxpc = os.environ.get("BIODEEP_MAXPC", "").strip()
MAX_PER_CLASS = int(_maxpc) if _maxpc.isdigit() and int(_maxpc) > 0 else None
GROUP_SPLIT_REAL = True
N_CPU = os.cpu_count() or 8

# Training mode: 'expanded' trains on all variants every epoch;
# 'stochastic' samples 1 variant per source video per epoch.
TRAIN_MODE = os.environ.get("BIODEEP_TRAIN_MODE", "expanded").strip().lower()

# Optional filter to run with only specific degradations (e.g. "blur,noise")
_deg_filter_env = os.environ.get("BIODEEP_DEGRADATIONS", "").strip()
ACTIVE_DEG_FILTER = [d.strip() for d in _deg_filter_env.split(",") if d.strip()] if _deg_filter_env else None
# -----------------------------------------------------------------

WORK_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)
(WORK_DIR / "report").mkdir(parents=True, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(N_CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 78)
print(" BIODEEP-AV MULTI-DEGRADATION DEEPFAKE TRAINING PIPELINE")
print(f"  * Device          : {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
print(f"  * Dataset Root    : {DATA_ROOT}")
print(f"  * Work Directory  : {WORK_DIR}")
print(f"  * Degradation Cache: {CACHE_DIR}")
print(f"  * Training Mode   : {TRAIN_MODE.upper()}")
print(f"  * Epochs          : {EPOCHS} | Batch: {BATCH} | LR: {LR}")
print("=" * 78)


# ------------------------- 1. MANIFEST DISCOVERY -------------------------
def load_manifest_files(data_root):
    """Parse real and fake manifests and build source -> degraded mappings."""
    manifest_info = {}
    deg_names = set()

    for label_str, label_int in [("real", 0), ("fake", 1)]:
        m_path = data_root / label_str / "videos" / "degraded_output" / "manifest.json"
        deg_base = data_root / label_str / "videos" / "degraded_output"
        orig_base = data_root / label_str / "videos"

        if not m_path.exists():
            raise FileNotFoundError(f"Manifest not found: {m_path}")

        with open(m_path, "r", encoding="utf-8") as f:
            m_data = json.load(f)

        for item in m_data["files"]:
            src = item["source"]
            clean_p = orig_base / src
            outputs_dict = {}
            for out in item["outputs"]:
                deg_type = out["degradation"]
                if ACTIVE_DEG_FILTER and deg_type not in ACTIVE_DEG_FILTER:
                    continue
                deg_names.add(deg_type)
                outputs_dict[deg_type] = deg_base / out["output"]

            manifest_info[src] = {
                "label": label_int,
                "clean_path": clean_p,
                "degradations": outputs_dict,
            }

    sorted_deg = sorted(list(deg_names))
    return manifest_info, sorted_deg


manifest_map, degradation_types = load_manifest_files(DATA_ROOT)
print(f"\nDiscovered {len(manifest_map)} total source videos.")
print(f"Available degradation types ({len(degradation_types)}): {', '.join(degradation_types)}")


# ------------------------- 2. VIDEO SAMPLES & ORDER -------------------------
# Determine video list and deterministic order
real_names = sorted([k for k, v in manifest_map.items() if v["label"] == 0])
fake_names = sorted([k for k, v in manifest_map.items() if v["label"] == 1])

if MAX_PER_CLASS:
    rng0 = random.Random(SEED)
    real_names = rng0.sample(real_names, min(MAX_PER_CLASS, len(real_names)))
    fake_names = rng0.sample(fake_names, min(MAX_PER_CLASS, len(fake_names)))
    print(f"Subsampled by MAX_PER_CLASS={MAX_PER_CLASS}: real={len(real_names)}, fake={len(fake_names)}")

all_source_names = np.array(real_names + fake_names)
all_labels = np.array([0] * len(real_names) + [1] * len(fake_names))
N_VIDEOS = len(all_source_names)
print(f"Active source videos: {N_VIDEOS} (Real: {len(real_names)} | Fake: {len(fake_names)})")
total_variants = 1 + len(degradation_types)
print(f"Total video variants per source: {total_variants} (1 Clean + {len(degradation_types)} Degraded)")
print(f"Total video dataset instances: {N_VIDEOS * total_variants:,}")


# ------------------------- 3. VIDEO FRAME EXTRACTION -------------------------
def extract_clip(path, n_frames=N_FRAMES, size=IMG):
    """Extract evenly spaced frames from video file and resize to (size, size)."""
    try:
        cap = cv2.VideoCapture(str(path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = {}
        if total > 0:
            want = set(np.linspace(0, total - 1, n_frames).round().astype(int).tolist())
            last = max(want)
            i = 0
            while i <= last:
                if not cap.grab():
                    break
                if i in want:
                    ok, fr = cap.retrieve()
                    if ok:
                        frames[i] = cv2.resize(fr, (size, size), interpolation=cv2.INTER_AREA)
                i += 1
        if len(frames) == 0:
            cap.release()
            cap = cv2.VideoCapture(str(path))
            tmp = []
            while True:
                ok, fr = cap.read()
                if not ok:
                    break
                tmp.append(cv2.resize(fr, (size, size), interpolation=cv2.INTER_AREA))
            if not tmp:
                return None
            total = len(tmp)
            frames = {j: tmp[j] for j in np.linspace(0, total - 1, n_frames).round().astype(int)}
        cap.release()
        keys = sorted(frames)
        want_list = np.linspace(0, keys[-1], n_frames).round().astype(int)
        seq = [frames[min(keys, key=lambda k: abs(k - t))] for t in want_list]
        arr = np.stack(seq)[..., ::-1].copy()  # BGR -> RGB
        return arr
    except Exception as e:
        print(f"skip {Path(path).name}: {e}")
        return None


def get_or_create_cache(deg_name, paths, tag, expected_shape):
    """Return memory-mapped numpy array of shape (N, N_FRAMES, IMG, IMG, 3)."""
    cache_path = CACHE_DIR / f"cache_{deg_name}_{tag}.npy"
    if cache_path.exists():
        print(f"  -> Found cached {deg_name.upper()}: {cache_path.name}")
        return np.load(cache_path, mmap_mode="r")

    total = len(paths)
    print(f"\n[Extracting & Caching {deg_name.upper()}] ({total} videos | {N_CPU} threads)")
    mmap_arr = np.lib.format.open_memmap(cache_path, mode="w+", dtype=np.uint8, shape=expected_shape)
    t0 = time.time()
    last_print = 0

    with ThreadPoolExecutor(max_workers=N_CPU) as ex:
        futures = {ex.submit(extract_clip, p): i for i, p in enumerate(paths)}
        completed = 0
        for fut in as_completed(futures):
            i = futures[fut]
            try:
                res = fut.result()
                if res is not None and res.shape == (N_FRAMES, IMG, IMG, 3):
                    mmap_arr[i] = res
                else:
                    mmap_arr[i] = 0
            except Exception:
                mmap_arr[i] = 0
            completed += 1
            now = time.time()
            if now - last_print >= 0.15 or completed == total:
                elapsed = now - t0
                rate = completed / max(0.001, elapsed)
                eta = (total - completed) / max(0.001, rate)
                bar = progress_bar(completed, total, bar_len=20)
                sys.stdout.write(
                    f"\r  {deg_name}: {bar} ({completed}/{total}) | {rate:5.1f} vid/s | "
                    f"Elapsed: {format_time(elapsed)} | ETA: {format_time(eta)}   "
                )
                sys.stdout.flush()
                last_print = now

    sys.stdout.write("\n")
    sys.stdout.flush()
    mmap_arr.flush()
    del mmap_arr
    print(f"  Saved memory-mapped cache: {cache_path.name} ({cache_path.stat().st_size / 1e9:.2f} GB)")
    return np.load(cache_path, mmap_mode="r")


# ------------------------- 4. BUILD / LOAD CACHES -------------------------
tag = f"T{N_FRAMES}_S{IMG}_n{N_VIDEOS}"
shape = (N_VIDEOS, N_FRAMES, IMG, IMG, 3)
variant_caches = {}

# 4A. Clean cache
clean_cache_npz = WORK_DIR / f"clean_{tag}.npz"
clean_memmap_npy = CACHE_DIR / f"cache_clean_{tag}.npy"

if clean_cache_npz.exists() and not clean_memmap_npy.exists():
    print(f"\nFound existing clean NPZ: {clean_cache_npz.name}")
    print("  Converting to memory-mapped format for zero-RAM overhead streaming...")
    z = np.load(clean_cache_npz)
    clean_mmap = np.lib.format.open_memmap(clean_memmap_npy, mode="w+", dtype=np.uint8, shape=shape)
    clean_mmap[:] = z["X"]
    clean_mmap.flush()
    del clean_mmap
    del z

if clean_memmap_npy.exists():
    print(f"  -> Loaded clean memory-map: {clean_memmap_npy.name}")
    variant_caches["clean"] = np.load(clean_memmap_npy, mmap_mode="r")
else:
    clean_paths = [manifest_map[n]["clean_path"] for n in all_source_names]
    variant_caches["clean"] = get_or_create_cache("clean", clean_paths, tag, shape)

# 4B. Degraded caches
for deg in degradation_types:
    deg_paths = [manifest_map[n]["degradations"].get(deg, manifest_map[n]["clean_path"]) for n in all_source_names]
    variant_caches[deg] = get_or_create_cache(deg, deg_paths, tag, shape)

print("\nAll video variant caches ready in memory-mapped mode!")
active_variants = ["clean"] + degradation_types
print(f"Active variants: {active_variants}")


# ------------------------- 5. SPLIT MANAGEMENT -------------------------
def yt_group(name):
    mm = re.match(r"^(.*)_\d{4}_S\d+_E\d+", name)
    return mm.group(1) if mm else name


def grouped_take(idx, groups, frac, rng):
    idx = list(idx)
    target = frac * len(idx)
    ug = sorted(set(groups[i] for i in idx))
    rng.shuffle(ug)
    held, count = set(), 0
    for g in ug:
        if count >= target:
            break
        members = [i for i in idx if groups[i] == g]
        if count + len(members) > target * 1.5 and count > 0:
            continue
        held.add(g)
        count += len(members)
    return [i for i in idx if groups[i] not in held], [i for i in idx if groups[i] in held]


split_file = WORK_DIR / "split.json"
train_idx, val_idx, test_idx = None, None, None

if split_file.exists() and not MAX_PER_CLASS:
    print(f"\nLoading baseline split from {split_file.name} to ensure 100% split consistency...")
    with open(split_file, "r") as f:
        split_data = json.load(f)
    name_to_idx = {name: i for i, name in enumerate(all_source_names)}
    train_idx = np.array(sorted([name_to_idx[n] for n in split_data["train"] if n in name_to_idx]))
    val_idx = np.array(sorted([name_to_idx[n] for n in split_data["val"] if n in name_to_idx]))
    test_idx = np.array(sorted([name_to_idx[n] for n in split_data["test"] if n in name_to_idx]))
    print(f"  Loaded from split.json: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
else:
    print("\nGenerating fresh grouped split (YouTube video group leakage prevention)...")
    rng = random.Random(SEED)
    groups = np.array([yt_group(n) if (lab == 0 and GROUP_SPLIT_REAL) else n for n, lab in zip(all_source_names, all_labels)])
    tr_list, va_list, te_list = [], [], []
    for cls in (0, 1):
        idx = [i for i in range(len(all_labels)) if all_labels[i] == cls]
        rest, te = grouped_take(idx, groups, TEST_FRAC, rng)
        tr, va = grouped_take(rest, groups, VAL_FRAC, rng)
        tr_list += tr
        va_list += va
        te_list += te
    train_idx = np.array(sorted(tr_list))
    val_idx = np.array(sorted(va_list))
    test_idx = np.array(sorted(te_list))

assert not (set(train_idx) & set(test_idx))
assert not (set(val_idx) & set(test_idx))
assert not (set(train_idx) & set(val_idx))

print("\nSource Video Splits:")
for nm, ix in (("Train", train_idx), ("Val", val_idx), ("Test", test_idx)):
    print(f"  * {nm:5s}: {len(ix):5d} source videos | Real: {int((all_labels[ix] == 0).sum())} | Fake: {int((all_labels[ix] == 1).sum())}")


# ------------------------- 6. MODEL ARCHITECTURE -------------------------
def block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(cout),
        nn.Conv2d(cout, cout, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(cout),
        nn.MaxPool2d(2),
    )


class DeepFakeVideoCNN(nn.Module):
    def __init__(self, width=(16, 32, 64, 128), num_classes=2):
        super().__init__()
        ch = (3,) + tuple(width)
        self.features = nn.Sequential(*[block(ch[i], ch[i + 1]) for i in range(len(width))])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Linear(2 * width[-1], 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):  # x: [B, T, 3, H, W]
        B, T, C, H, W = x.shape
        f = self.pool(self.features(x.reshape(B * T, C, H, W))).flatten(1).view(B, T, -1)
        return self.head(torch.cat([f.mean(1), f.amax(1)], dim=1))


def to_tensor(arr):
    t = torch.from_numpy(np.ascontiguousarray(arr)).permute(0, 1, 4, 2, 3).float().div_(127.5).sub_(1.0)
    return t.to(device)


model = DeepFakeVideoCNN().to(device)
print(f"\nModel: DeepFakeVideoCNN | Trainable Parameters: {sum(p.numel() for p in model.parameters()):,}")


# ------------------------- 7. DATASET BATCHING -------------------------
# Build training instance indexing
if TRAIN_MODE == "expanded":
    # Each video has |active_variants| instances
    # [(v_idx, var_name), ...]
    train_instances = [(v, var) for v in train_idx for var in active_variants]
else:
    # Stochastic mode: 1 instance per source video, variant chosen per epoch
    train_instances = [(v, None) for v in train_idx]

print(f"Training Mode: {TRAIN_MODE.upper()} | Training instances per epoch: {len(train_instances):,}")


def make_training_batch(batch_tuples, rng_np):
    """Collate a batch from memmap arrays with random temporal subsampling & augmentation."""
    B = len(batch_tuples)
    raw_frames = np.empty((B, CLIP_LEN, IMG, IMG, 3), dtype=np.uint8)
    labels = np.empty(B, dtype=np.int64)

    for b, (v_idx, var_name) in enumerate(batch_tuples):
        chosen_var = var_name if var_name is not None else rng_np.choice(active_variants)
        cache_arr = variant_caches[chosen_var]
        # Temporal subsampling: pick CLIP_LEN frames sorted
        sel = np.sort(rng_np.choice(N_FRAMES, CLIP_LEN, replace=False))
        clip = cache_arr[v_idx, sel]  # [CLIP_LEN, IMG, IMG, 3]

        # Spatial augmentation: random horizontal flip
        if rng_np.random() < 0.5:
            clip = clip[:, :, ::-1, :]

        # Random spatial translation pad/crop
        pad = np.pad(clip, ((0, 0), (4, 4), (4, 4), (0, 0)), mode="edge")
        dy, dx = rng_np.integers(0, 9, 2)
        raw_frames[b] = pad[:, dy : dy + IMG, dx : dx + IMG]
        labels[b] = all_labels[v_idx]

    return to_tensor(raw_frames), torch.from_numpy(labels).to(device)


@torch.no_grad()
def predict_proba_on_cache(model_eval, cache_arr, indices, bs=32):
    """Evaluate model on a specific cache array over given video indices."""
    model_eval.eval()
    stride = max(1, N_FRAMES // CLIP_LEN)
    out = []
    for i in range(0, len(indices), bs):
        chunk = indices[i : i + bs]
        p = 0
        for v in range(stride):
            xb = cache_arr[chunk][:, v::stride][:, :CLIP_LEN]
            p = p + torch.softmax(model_eval(to_tensor(xb)), 1)[:, 1].cpu().numpy()
        out.append(p / stride)
    return np.concatenate(out)


def metrics(y_true, p, thr=0.5):
    pred = (p >= thr).astype(int)
    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    return dict(
        accuracy=accuracy_score(y_true, pred),
        balanced_acc=balanced_accuracy_score(y_true, pred),
        precision=precision_score(y_true, pred, zero_division=0),
        recall=recall_score(y_true, pred, zero_division=0),
        f1=f1_score(y_true, pred, zero_division=0),
        auc=roc_auc_score(y_true, p) if len(set(y_true)) > 1 else float("nan"),
        tn=int(cm[0, 0]),
        fp=int(cm[0, 1]),
        fn=int(cm[1, 0]),
        tp=int(cm[1, 1]),
        threshold=float(thr),
    )


# ------------------------- 8. TRAINING SETUP -------------------------
counts = np.bincount(all_labels[train_idx], minlength=2)
class_w = torch.tensor(counts.sum() / (2.0 * counts), dtype=torch.float32).to(device)
criterion = nn.CrossEntropyLoss(weight=class_w)
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
steps_per_epoch = int(np.ceil(len(train_instances) / BATCH))
total_steps = EPOCHS * steps_per_epoch
scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LR, total_steps=total_steps)
rng_np = np.random.default_rng(SEED)

history, best = [], dict(score=-1, epoch=-1)
best_model_path = WORK_DIR / "best_model_degraded.pt"
gpu_temp_init = get_gpu_temp(ttl=0)
gpu_temp_str = f"{gpu_temp_init}C" if gpu_temp_init is not None else "N/A"

print("\n" + "=" * 78)
print(" MULTI-DEGRADATION TRAINING SCHEDULE")
print(f"  * Total training instances : {len(train_instances):,} ({steps_per_epoch} steps/epoch, batch={BATCH})")
print(f"  * Epochs                   : {EPOCHS} ({total_steps} optimization steps)")
print(f"  * Initial GPU Temp         : {gpu_temp_str} (Thermal guard: pause 5 min if >= {GPU_MAX_TEMP}C)")
print(f"  * Validation set           : {len(val_idx)} videos across {len(active_variants)} variants")
print(f"  * Checkpoint destination   : {best_model_path.name}")
print("=" * 78 + "\n")

t_start = time.time()
overall_step = 0

# ------------------------- 9. TRAINING LOOP -------------------------
for epoch in range(1, EPOCHS + 1):
    model.train()
    # Shuffle instances
    perm_order = rng_np.permutation(len(train_instances))
    total_loss = 0.0
    t0 = time.time()
    last_print = 0

    for s in range(steps_per_epoch):
        overall_step += 1
        batch_slice = perm_order[s * BATCH : (s + 1) * BATCH]
        if len(batch_slice) < 2:
            continue
        batch_tuples = [train_instances[k] for k in batch_slice]

        xb, yb = make_training_batch(batch_tuples, rng_np)
        loss = criterion(model(xb), yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()

        now = time.time()
        step_in_epoch = s + 1
        if now - last_print >= 0.15 or step_in_epoch == steps_per_epoch:
            gpu_temp = get_gpu_temp()
            if gpu_temp is not None and gpu_temp >= GPU_MAX_TEMP:
                paused_sec = check_gpu_thermal_guard(GPU_MAX_TEMP, GPU_PAUSE_SECONDS)
                t_start += paused_sec
                t0 += paused_sec
                now = time.time()
                gpu_temp = get_gpu_temp(ttl=0)

            gpu_str = f"{gpu_temp}C" if gpu_temp is not None else "N/A"
            epoch_elapsed = now - t0
            epoch_rate = step_in_epoch / max(0.001, epoch_elapsed)
            epoch_eta = (steps_per_epoch - step_in_epoch) / max(0.001, epoch_rate)

            total_elapsed = now - t_start
            total_rate = overall_step / max(0.001, total_elapsed)
            total_eta = (total_steps - overall_step) / max(0.001, total_rate)

            bar = progress_bar(step_in_epoch, steps_per_epoch, bar_len=16)
            avg_loss = total_loss / step_in_epoch
            sys.stdout.write(
                f"\rEpoch [{epoch:2d}/{EPOCHS}] {bar} step {step_in_epoch:3d}/{steps_per_epoch} "
                f"| loss {avg_loss:.4f} | GPU: {gpu_str} | ep ETA: {format_time(epoch_eta)} | total ETA: {format_time(total_eta)}   "
            )
            sys.stdout.flush()
            last_print = now

    # Validation: Evaluate on Clean validation set + a subset of degraded sets to pick best epoch
    pv_clean = predict_proba_on_cache(model, variant_caches["clean"], val_idx)
    vm_clean = metrics(all_labels[val_idx], pv_clean)

    # Fast validation score combining clean AUC and balanced accuracy
    score = vm_clean["auc"] + vm_clean["balanced_acc"]
    avg_train_loss = total_loss / max(1, steps_per_epoch)
    history.append(
        dict(
            epoch=epoch,
            train_loss=avg_train_loss,
            val_clean_auc=vm_clean["auc"],
            val_clean_bal_acc=vm_clean["balanced_acc"],
        )
    )

    gpu_temp = get_gpu_temp()
    if gpu_temp is not None and gpu_temp >= GPU_MAX_TEMP:
        paused_sec = check_gpu_thermal_guard(GPU_MAX_TEMP, GPU_PAUSE_SECONDS)
        t_start += paused_sec
        t0 += paused_sec
        gpu_temp = get_gpu_temp(ttl=0)

    gpu_str = f"{gpu_temp}C" if gpu_temp is not None else "N/A"
    epoch_elapsed = time.time() - t0
    total_elapsed = time.time() - t_start
    total_rate = overall_step / max(0.001, total_elapsed)
    total_eta = (total_steps - overall_step) / max(0.001, total_rate)

    flag = ""
    if score > best["score"]:
        best = dict(score=score, epoch=epoch, auc=vm_clean["auc"], bal_acc=vm_clean["balanced_acc"])
        torch.save(model.state_dict(), best_model_path)
        flag = "  <-- BEST"

    sys.stdout.write(
        f"\rEpoch [{epoch:2d}/{EPOCHS}] loss {avg_train_loss:.4f} | "
        f"val Clean AUC {vm_clean['auc']:.4f} bal-acc {vm_clean['balanced_acc']:.4f} | GPU: {gpu_str} | "
        f"time {format_time(epoch_elapsed)} | elapsed {format_time(total_elapsed)} | ETA {format_time(total_eta)}{flag}    \n"
    )
    sys.stdout.flush()

total_time = time.time() - t_start
print("-" * 78)
print(f"Training complete in {format_time(total_time)}! Best Epoch: {best['epoch']}")
print("-" * 78 + "\n")


# ------------------------- 10. COMPREHENSIVE EVALUATION -------------------------
print("[Evaluating Model Across All Degradation Types on Held-Out Test Set...]")
model.load_state_dict(torch.load(best_model_path, map_location=device))
y_test = all_labels[test_idx]

# Optimal threshold selected on Clean validation set
pv_clean_val = predict_proba_on_cache(model, variant_caches["clean"], val_idx)
grid = np.linspace(0.05, 0.95, 91)
thr_val = float(grid[np.argmax([balanced_accuracy_score(all_labels[val_idx], (pv_clean_val >= t).astype(int)) for t in grid])])

eval_results = {}
pred_table = {"name": all_source_names[test_idx], "label": y_test}

for var in active_variants:
    p_test = predict_proba_on_cache(model, variant_caches[var], test_idx)
    pred_table[f"p_fake_{var}"] = p_test
    eval_results[var] = {
        "thr_0.5": metrics(y_test, p_test, thr=0.5),
        "thr_val_tuned": metrics(y_test, p_test, thr=thr_val),
    }

# Summary comparison table
summary_rows = []
for var in active_variants:
    m05 = eval_results[var]["thr_0.5"]
    summary_rows.append(
        {
            "Variant": var.upper(),
            "Accuracy": m05["accuracy"],
            "Balanced Acc": m05["balanced_acc"],
            "Precision": m05["precision"],
            "Recall": m05["recall"],
            "F1-Score": m05["f1"],
            "ROC-AUC": m05["auc"],
        }
    )

summary_df = pd.DataFrame(summary_rows).set_index("Variant")
print("\n" + "=" * 78)
print(" TEST SET PERFORMANCE ACROSS ALL DEGRADATION TYPES (Threshold = 0.5)")
print("=" * 78)
print(summary_df.round(4).to_string())
print("=" * 78)

# Baseline comparison if results_clean.json exists
baseline_file = WORK_DIR / "results_clean.json"
if baseline_file.exists():
    try:
        with open(baseline_file, "r") as f:
            base_data = json.load(f)
        base_clean_auc = base_data.get("results", {}).get("threshold_0.5", {}).get("auc", None)
        base_clean_acc = base_data.get("results", {}).get("threshold_0.5", {}).get("accuracy", None)
        print("\n" + "-" * 78)
        print(" COMPARISON: CLEAN-ONLY TRAINED BASELINE vs DEGRADATION-TRAINED MODEL")
        print(f"  * Baseline Model Clean Test AUC : {base_clean_auc:.4f}" if base_clean_auc else "")
        print(f"  * Baseline Model Clean Accuracy : {base_clean_acc * 100:.2f}%" if base_clean_acc else "")
        print(f"  * Degraded-Trained Clean Test AUC: {summary_df.loc['CLEAN', 'ROC-AUC']:.4f}")
        print(f"  * Degraded-Trained Clean Accuracy: {summary_df.loc['CLEAN', 'Accuracy'] * 100:.2f}%")
        print(f"  * Macro Average Degraded AUC    : {summary_df.loc[summary_df.index != 'CLEAN', 'ROC-AUC'].mean():.4f}")
        print("-" * 78)
    except Exception as e:
        print(f"(Could not parse baseline results: {e})")

# Save predictions CSV
predictions_df = pd.DataFrame(pred_table)
predictions_df.to_csv(WORK_DIR / "test_predictions_degraded.csv", index=False)

# ------------------------- 11. REPORTS & VISUALIZATIONS -------------------------
now = datetime.now()
date_str = f"{now.year % 100}-{now.month}-{now.day}"
time_str = f"{now.hour:02d}.{now.minute:02d}"
report_filename = f"report_degraded_date_{date_str}_time_{time_str}.json"

report_data = {
    "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
    "date": date_str,
    "time": time_str,
    "results_by_variant": eval_results,
    "summary_table": summary_df.to_dict(orient="index"),
    "best_epoch": best["epoch"],
    "val_threshold": thr_val,
    "training_time": format_time(total_time),
    "gpu_final_temp": f"{get_gpu_temp(ttl=0)}C" if get_gpu_temp(ttl=0) is not None else "N/A",
    "history": history,
    "config": dict(
        N_FRAMES=N_FRAMES,
        IMG=IMG,
        CLIP_LEN=CLIP_LEN,
        BATCH=BATCH,
        EPOCHS=EPOCHS,
        LR=LR,
        SEED=SEED,
        TRAIN_MODE=TRAIN_MODE,
        n_source_train=len(train_idx),
        n_source_val=len(val_idx),
        n_source_test=len(test_idx),
        total_training_instances=len(train_instances),
        active_variants=active_variants,
        GPU_MAX_TEMP=GPU_MAX_TEMP,
        GPU_PAUSE_SECONDS=GPU_PAUSE_SECONDS,
    ),
}

# Save reports to test/ and biodeep_run/
for rdir in (REPORT_DIR, WORK_DIR / "test"):
    rdir.mkdir(parents=True, exist_ok=True)
    with open(rdir / report_filename, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    with open(rdir / "latest_report_degraded.json", "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

with open(WORK_DIR / "results_degraded.json", "w", encoding="utf-8") as f:
    json.dump(report_data, f, indent=2)

print(f"\n[Report saved] -> {REPORT_DIR / report_filename}")

# Generate Plots
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
h = pd.DataFrame(history)
ax[0].plot(h.epoch, h.train_loss, label="train loss", color="#e74c3c", linewidth=2)
ax[0].set_xlabel("Epoch")
ax[0].set_ylabel("Loss")
ax[0].set_title("Training Loss")
ax[0].grid(True, linestyle="--", alpha=0.5)
ax[0].legend()

ax[1].plot(h.epoch, h.val_clean_auc, label="val Clean AUC", color="#2980b9", linewidth=2)
ax[1].plot(h.epoch, h.val_clean_bal_acc, label="val Clean Bal-Acc", color="#27ae60", linewidth=2)
ax[1].set_xlabel("Epoch")
ax[1].set_ylabel("Score")
ax[1].set_title("Validation Metrics")
ax[1].grid(True, linestyle="--", alpha=0.5)
ax[1].legend()

plt.tight_layout()
plt.savefig(WORK_DIR / "training_curves_degraded.png", dpi=130)
plt.close()

# Robustness Bar Plot Across Degradations
fig, ax = plt.subplots(figsize=(10, 4.5))
variants_upper = [v.upper() for v in active_variants]
aucs = [eval_results[v]["thr_0.5"]["auc"] for v in active_variants]
accs = [eval_results[v]["thr_0.5"]["accuracy"] * 100 for v in active_variants]

x = np.arange(len(variants_upper))
width = 0.35
ax.bar(x - width / 2, aucs, width, label="ROC-AUC", color="#3498db")
ax.bar(x + width / 2, [a / 100.0 for a in accs], width, label="Accuracy", color="#2ecc71")
ax.set_ylabel("Score (0 - 1.0)")
ax.set_title("DeepFake Detection Robustness Across Degradation Types")
ax.set_xticks(x)
ax.set_xticklabels(variants_upper, rotation=35, ha="right")
ax.set_ylim([0.0, 1.05])
ax.grid(axis="y", linestyle="--", alpha=0.6)
ax.legend(loc="lower right")

plt.tight_layout()
plt.savefig(WORK_DIR / "robustness_by_degradation.png", dpi=130)
plt.close()

# Confusion Matrix for Clean and Combined
cm_clean = confusion_matrix(y_test, (pred_table["p_fake_clean"] >= 0.5).astype(int), labels=[0, 1])
fig, ax = plt.subplots(figsize=(3.8, 3.5))
ax.imshow(cm_clean, cmap="Blues")
for (i, j), v in np.ndenumerate(cm_clean):
    ax.text(j, i, str(v), ha="center", va="center", fontsize=14, color="black" if cm_clean[i, j] < cm_clean.max() / 2 else "white")
ax.set_xticks([0, 1])
ax.set_xticklabels(["Pred Real", "Pred Fake"])
ax.set_yticks([0, 1])
ax.set_yticklabels(["Real", "Fake"])
ax.set_title("Clean Test Set Confusion Matrix")
plt.tight_layout()
plt.savefig(WORK_DIR / "confusion_matrix_degraded.png", dpi=130)
plt.close("all")

print("\nSaved evaluation plots:")
print(f"  * {WORK_DIR / 'training_curves_degraded.png'}")
print(f"  * {WORK_DIR / 'robustness_by_degradation.png'}")
print(f"  * {WORK_DIR / 'confusion_matrix_degraded.png'}")
print("\nAll tasks completed successfully!")
