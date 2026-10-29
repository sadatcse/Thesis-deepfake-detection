"""
DeepFake BioDeepAV Audio-Visual 100-Sample Test Pipeline With Video Degradations
================================================================================
Multimodal Audio-Visual deepfake detection pipeline evaluating AudioVisualDeepFakeCNN on:
  * 100 Real source videos + 100 Fake source videos (200 source videos total)
  * Dual modalities: Video frames + Audio log-STFT spectrograms
  * 10 video variants per video (1 Clean + 9 Degraded: blur, brightness, compression,
    contrast, fps_reduction, frame_drop, motion_blur, noise, saturation)
  * Total = 2,000 video instances paired with biometric acoustic signals.

Features:
  1. Audio Extraction from companion WAV files or embedded AAC video streams via ffmpeg
  2. STFT Log-Spectrogram computation and memory-mapped audio caching
  3. Dual-branch AudioVisualDeepFakeCNN (VideoCNN + AudioCNN + Cross-Modal Fusion Head)
  4. Modality ablation evaluation: Audio-Visual vs Video-Only vs Audio-Only
  5. GPU Thermal Guard with auto-pause countdown
  6. Visual comparison plots and comprehensive JSON reporting
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
from scipy.io import wavfile
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
    if seconds is None or np.isnan(seconds) or seconds < 0:
        return "--:--"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h > 0 else f"{m:02d}:{s:02d}"


def progress_bar(current, total, bar_len=20):
    frac = min(1.0, max(0.0, current / max(1, total)))
    filled = int(frac * bar_len)
    bar = "=" * filled + (">" if filled < bar_len else "")
    bar = bar.ljust(bar_len, ".")
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

DATA_ROOT = Path(os.environ.get("BIODEEP_ROOT", str(_DEFAULT_ROOT)))
WORK_DIR = Path(os.environ.get("BIODEEP_WORK", str(_SCRIPT_DIR / "biodeep_run_av_test100")))
REPORT_DIR = Path(os.environ.get("BIODEEP_REPORT", str(_SCRIPT_DIR / "test")))
CACHE_DIR = WORK_DIR / "cache_degraded"
PREV_CACHE_DIR = _SCRIPT_DIR / "biodeep_run_test100" / "cache_degraded"

GPU_MAX_TEMP = int(os.environ.get("BIODEEP_GPU_MAX_TEMP", 72))
GPU_PAUSE_SECONDS = int(os.environ.get("BIODEEP_GPU_PAUSE", 300))

N_REAL = int(os.environ.get("BIODEEP_TEST_REAL", 100))
N_FAKE = int(os.environ.get("BIODEEP_TEST_FAKE", 100))

N_FRAMES = 16
IMG = 96
CLIP_LEN = 8
BATCH = 16

_epochs_env = os.environ.get("BIODEEP_EPOCHS", "").strip()
EPOCHS = int(_epochs_env) if _epochs_env.isdigit() else 5
LR = 1e-3
TEST_FRAC = 0.20
VAL_FRAC = 0.10
SEED = 42
GROUP_SPLIT_REAL = True
N_CPU = os.cpu_count() or 8
TRAIN_MODE = os.environ.get("BIODEEP_TRAIN_MODE", "expanded").strip().lower()
# -----------------------------------------------------------------

WORK_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)
(WORK_DIR / "test").mkdir(parents=True, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(N_CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 78)
print(" BIODEEP-AV AUDIO-VISUAL TEST PIPELINE (100 REAL + 100 FAKE)")
print(f"  * Device            : {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
print(f"  * Dataset Root      : {DATA_ROOT}")
print(f"  * Work Directory    : {WORK_DIR}")
print(f"  * Sample Selection  : {N_REAL} Real + {N_FAKE} Fake = {N_REAL + N_FAKE} Source Videos")
print(f"  * Modalities        : Visual Video Frames (96x96) + Audio Spectrograms (128x128)")
print(f"  * Training Mode     : {TRAIN_MODE.upper()} | Epochs: {EPOCHS} | Batch: {BATCH}")
print("=" * 78)


# ------------------------- 1. MANIFEST DISCOVERY -------------------------
def load_manifest_files(data_root):
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
print(f"\nDiscovered {len(manifest_map)} total videos in dataset manifests.")
print(f"Degradation types ({len(degradation_types)}): {', '.join(degradation_types)}")


# ------------------------- 2. SAMPLE 100 REAL & 100 FAKE -------------------------
all_real_pool = sorted([
    k for k, v in manifest_map.items()
    if v["label"] == 0 and all(d in v["degradations"] for d in degradation_types)
])
all_fake_pool = sorted([
    k for k, v in manifest_map.items()
    if v["label"] == 1 and all(d in v["degradations"] for d in degradation_types)
])

rng0 = random.Random(SEED)
selected_real = sorted(rng0.sample(all_real_pool, min(N_REAL, len(all_real_pool))))
selected_fake = sorted(rng0.sample(all_fake_pool, min(N_FAKE, len(all_fake_pool))))

all_source_names = np.array(selected_real + selected_fake)
all_labels = np.array([0] * len(selected_real) + [1] * len(selected_fake))
N_VIDEOS = len(all_source_names)
total_variants = 1 + len(degradation_types)

print(f"\nTarget test sample selected:")
print(f"  * Real videos      : {len(selected_real)}")
print(f"  * Fake videos      : {len(selected_fake)}")
print(f"  * Total source     : {N_VIDEOS} videos")
print(f"  * Variants/video   : {total_variants} (1 Clean + {len(degradation_types)} Degraded)")
print(f"  * Total instances  : {N_VIDEOS * total_variants:,} video instances")


# ------------------------- 3. AUDIO FEATURE EXTRACTION -------------------------
def extract_audio_spectrogram(source_name, label_int, clean_video_path, n_samples=48000):
    """Extract 16kHz audio from companion WAV or video stream and convert to (1, 128, 128) log-STFT."""
    audio_data = None
    # 1. Check companion WAV file
    audio_dir = DATA_ROOT / ("real" if label_int == 0 else "fake") / "audios"
    wav_path = audio_dir / (Path(source_name).stem + ".wav")
    if wav_path.exists():
        try:
            sr, raw = wavfile.read(wav_path)
            if raw.ndim > 1:
                raw = raw[:, 0]
            if raw.dtype == np.int16:
                audio_data = raw.astype(np.float32) / 32768.0
            else:
                audio_data = raw.astype(np.float32)
            if sr != 16000 and len(audio_data) > 0:
                from scipy.signal import resample
                new_len = int(len(audio_data) * 16000 / sr)
                audio_data = resample(audio_data, new_len).astype(np.float32)
        except Exception:
            audio_data = None

    # 2. Extract from video file via ffmpeg stream
    if audio_data is None or len(audio_data) == 0:
        try:
            cmd = [
                "ffmpeg", "-v", "error",
                "-i", str(clean_video_path),
                "-vn", "-ac", "1", "-ar", "16000",
                "-f", "s16le", "-"
            ]
            raw_bytes = subprocess.check_output(
                cmd,
                creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0,
                timeout=10,
            )
            audio_data = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        except Exception:
            audio_data = np.zeros(n_samples, dtype=np.float32)

    # Fixed length 3.0s (48,000 samples at 16kHz)
    if len(audio_data) < n_samples:
        if len(audio_data) > 0:
            reps = int(np.ceil(n_samples / len(audio_data)))
            audio_data = np.tile(audio_data, reps)[:n_samples]
        else:
            audio_data = np.zeros(n_samples, dtype=np.float32)
    else:
        audio_data = audio_data[:n_samples]

    # STFT Spectrogram: (1, 128, 128)
    with torch.no_grad():
        t = torch.from_numpy(audio_data)
        stft = torch.stft(t, n_fft=512, hop_length=256, win_length=512, window=torch.hann_window(512), return_complex=True)
        mag = torch.log1p(stft.abs())  # [Freq, Time]
        spec = torch.nn.functional.interpolate(mag.unsqueeze(0).unsqueeze(0), size=(128, 128), mode="bilinear", align_corners=False)
        spec = (spec - spec.mean()) / (spec.std() + 1e-6)
        spec = spec.clamp(-3.0, 3.0) / 3.0
        return spec.squeeze(0).numpy().astype(np.float32)


# ------------------------- 4. CACHING (VIDEO & AUDIO) -------------------------
def extract_video_clip(path, n_frames=N_FRAMES, size=IMG):
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
        return np.stack(seq)[..., ::-1].copy()
    except Exception as e:
        print(f"skip {Path(path).name}: {e}")
        return None


def get_or_create_video_cache(deg_name, paths, tag, expected_shape):
    cache_path = CACHE_DIR / f"cache_{deg_name}_{tag}.npy"
    prev_path = PREV_CACHE_DIR / f"cache_{deg_name}_{tag}.npy"

    # Reuse previous run cache if identical
    if cache_path.exists():
        print(f"  -> Found cached {deg_name.upper()}: {cache_path.name}")
        return np.load(cache_path, mmap_mode="r")
    elif prev_path.exists() and prev_path.stat().st_size == np.prod(expected_shape):
        print(f"  -> Reusing verified video cache from {prev_path.parent.name}/{prev_path.name}")
        return np.load(prev_path, mmap_mode="r")

    total = len(paths)
    print(f"\n[Extracting & Caching {deg_name.upper()}] ({total} videos | {N_CPU} threads)")
    mmap_arr = np.lib.format.open_memmap(cache_path, mode="w+", dtype=np.uint8, shape=expected_shape)
    t0 = time.time()
    last_print = 0

    with ThreadPoolExecutor(max_workers=N_CPU) as ex:
        futures = {ex.submit(extract_video_clip, p): i for i, p in enumerate(paths)}
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
                    f"\r  {deg_name:13s}: {bar} ({completed}/{total}) | {rate:5.1f} vid/s | "
                    f"Elapsed: {format_time(elapsed)} | ETA: {format_time(eta)}   "
                )
                sys.stdout.flush()
                last_print = now

    sys.stdout.write("\n")
    sys.stdout.flush()
    mmap_arr.flush()
    del mmap_arr
    print(f"  Saved memory-mapped cache: {cache_path.name} ({cache_path.stat().st_size / 1e6:.1f} MB)")
    return np.load(cache_path, mmap_mode="r")


tag = f"T{N_FRAMES}_S{IMG}_n{N_VIDEOS}"
video_shape = (N_VIDEOS, N_FRAMES, IMG, IMG, 3)
variant_caches = {}

print("\n--- 1. Preparing Video Caches (Memory-Mapped) ---")
clean_paths = [manifest_map[n]["clean_path"] for n in all_source_names]
variant_caches["clean"] = get_or_create_video_cache("clean", clean_paths, tag, video_shape)

for deg in degradation_types:
    deg_paths = [manifest_map[n]["degradations"].get(deg, manifest_map[n]["clean_path"]) for n in all_source_names]
    variant_caches[deg] = get_or_create_video_cache(deg, deg_paths, tag, video_shape)

active_variants = ["clean"] + degradation_types
print(f"All {len(active_variants)} video variants loaded!")

# Audio Caching: (N_VIDEOS, 1, 128, 128)
print("\n--- 2. Preparing Audio Spectrogram Cache ---")
audio_cache_path = CACHE_DIR / f"cache_audio_{tag}.npy"
if audio_cache_path.exists():
    print(f"  -> Found cached Audio Spectrograms: {audio_cache_path.name}")
    audio_cache = np.load(audio_cache_path, mmap_mode="r")
else:
    print(f"  -> Extracting and computing audio spectrograms for {N_VIDEOS} source videos...")
    audio_shape = (N_VIDEOS, 1, 128, 128)
    audio_mmap = np.lib.format.open_memmap(audio_cache_path, mode="w+", dtype=np.float32, shape=audio_shape)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=N_CPU) as ex:
        futures = {
            ex.submit(
                extract_audio_spectrogram,
                all_source_names[i],
                all_labels[i],
                manifest_map[all_source_names[i]]["clean_path"],
            ): i
            for i in range(N_VIDEOS)
        }
        for fut in as_completed(futures):
            idx = futures[fut]
            audio_mmap[idx] = fut.result()
    audio_mmap.flush()
    del audio_mmap
    print(f"  Saved Audio Spectrogram cache in {format_time(time.time() - t0)}: {audio_cache_path.name} ({audio_cache_path.stat().st_size / 1e6:.1f} MB)")
    audio_cache = np.load(audio_cache_path, mmap_mode="r")

print(f"Audio Cache ready! Shape: {audio_cache.shape} ({audio_cache.dtype})")


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
    print(f"  * {nm:5s}: {len(ix):3d} source videos | Real: {int((all_labels[ix] == 0).sum())} | Fake: {int((all_labels[ix] == 1).sum())}")

json.dump(
    {
        "train": all_source_names[train_idx].tolist(),
        "val": all_source_names[val_idx].tolist(),
        "test": all_source_names[test_idx].tolist(),
    },
    open(WORK_DIR / "split_av_test100.json", "w"),
    indent=1,
)


# ------------------------- 6. MULTIMODAL MODEL ARCHITECTURE -------------------------
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


class VideoCNN(nn.Module):
    def __init__(self, width=(16, 32, 64, 128)):
        super().__init__()
        ch = (3,) + tuple(width)
        self.features = nn.Sequential(*[block(ch[i], ch[i + 1]) for i in range(len(width))])
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):  # x: [B, T, 3, H, W]
        B, T, C, H, W = x.shape
        f = self.pool(self.features(x.reshape(B * T, C, H, W))).flatten(1).view(B, T, -1)
        return torch.cat([f.mean(1), f.amax(1)], dim=1)  # [B, 256]


class AudioCNN(nn.Module):
    def __init__(self, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 64x64
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 32x32
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 16x16
            nn.Conv2d(128, out_dim, 3, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

    def forward(self, x):  # x: [B, 1, 128, 128]
        return self.net(x).flatten(1)  # [B, 128]


class AudioVisualDeepFakeCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.video_net = VideoCNN(width=(16, 32, 64, 128))
        self.audio_net = AudioCNN(out_dim=128)
        self.fusion_head = nn.Sequential(
            nn.Linear(256 + 128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 2),
        )

    def forward(self, x_v, x_a=None, mode="av"):
        B = x_v.size(0)
        if mode == "av":
            f_v = self.video_net(x_v)
            f_a = self.audio_net(x_a)
            return self.fusion_head(torch.cat([f_v, f_a], dim=1))
        elif mode == "v_only":
            f_v = self.video_net(x_v)
            f_a = torch.zeros(B, 128, device=x_v.device)
            return self.fusion_head(torch.cat([f_v, f_a], dim=1))
        elif mode == "a_only":
            f_v = torch.zeros(B, 256, device=x_a.device)
            f_a = self.audio_net(x_a)
            return self.fusion_head(torch.cat([f_v, f_a], dim=1))


def to_video_tensor(arr):
    t = torch.from_numpy(np.ascontiguousarray(arr)).permute(0, 1, 4, 2, 3).float().div_(127.5).sub_(1.0)
    return t.to(device)


model = AudioVisualDeepFakeCNN().to(device)
v_params = sum(p.numel() for p in model.video_net.parameters())
a_params = sum(p.numel() for p in model.audio_net.parameters())
f_params = sum(p.numel() for p in model.fusion_head.parameters())
print(f"\nModel: AudioVisualDeepFakeCNN")
print(f"  * Video Backbone Parameters : {v_params:,}")
print(f"  * Audio Backbone Parameters : {a_params:,}")
print(f"  * Fusion Head Parameters    : {f_params:,}")
print(f"  * Total Trainable Parameters: {sum(p.numel() for p in model.parameters()):,}")


# ------------------------- 7. DATASET BATCHING & TRAINING -------------------------
if TRAIN_MODE == "expanded":
    train_instances = [(v, var) for v in train_idx for var in active_variants]
else:
    train_instances = [(v, None) for v in train_idx]

print(f"\nTraining Mode: {TRAIN_MODE.upper()} | Training instances per epoch: {len(train_instances):,}")


def make_training_batch(batch_tuples, rng_np):
    B = len(batch_tuples)
    raw_frames = np.empty((B, CLIP_LEN, IMG, IMG, 3), dtype=np.uint8)
    raw_audios = np.empty((B, 1, 128, 128), dtype=np.float32)
    labels = np.empty(B, dtype=np.int64)

    for b, (v_idx, var_name) in enumerate(batch_tuples):
        chosen_var = var_name if var_name is not None else rng_np.choice(active_variants)
        cache_arr = variant_caches[chosen_var]
        sel = np.sort(rng_np.choice(N_FRAMES, CLIP_LEN, replace=False))
        clip = cache_arr[v_idx, sel]

        if rng_np.random() < 0.5:
            clip = clip[:, :, ::-1, :]

        pad = np.pad(clip, ((0, 0), (4, 4), (4, 4), (0, 0)), mode="edge")
        dy, dx = rng_np.integers(0, 9, 2)
        raw_frames[b] = pad[:, dy : dy + IMG, dx : dx + IMG]
        raw_audios[b] = audio_cache[v_idx]
        labels[b] = all_labels[v_idx]

    return to_video_tensor(raw_frames), torch.from_numpy(raw_audios).to(device), torch.from_numpy(labels).to(device)


@torch.no_grad()
def predict_proba_multimodal(model_eval, cache_v, indices, mode="av", bs=32):
    model_eval.eval()
    stride = max(1, N_FRAMES // CLIP_LEN)
    out = []
    for i in range(0, len(indices), bs):
        chunk = indices[i : i + bs]
        p = 0
        audio_chunk = torch.from_numpy(audio_cache[chunk]).to(device)
        for v in range(stride):
            xb = cache_v[chunk][:, v::stride][:, :CLIP_LEN]
            logits = model_eval(to_video_tensor(xb), audio_chunk, mode=mode)
            p = p + torch.softmax(logits, 1)[:, 1].cpu().numpy()
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


counts = np.bincount(all_labels[train_idx], minlength=2)
class_w = torch.tensor(counts.sum() / (2.0 * counts), dtype=torch.float32).to(device)
criterion = nn.CrossEntropyLoss(weight=class_w)
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
steps_per_epoch = int(np.ceil(len(train_instances) / BATCH))
total_steps = EPOCHS * steps_per_epoch
scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LR, total_steps=total_steps)
rng_np = np.random.default_rng(SEED)

history, best = [], dict(score=-1, epoch=-1)
best_model_path = WORK_DIR / "best_model_av_test100.pt"
gpu_temp_init = get_gpu_temp(ttl=0)
gpu_temp_str = f"{gpu_temp_init}C" if gpu_temp_init is not None else "N/A"

print("\n" + "=" * 78)
print(" AUDIO-VISUAL 100-SAMPLE TRAINING SCHEDULE")
print(f"  * Total training instances : {len(train_instances):,} ({steps_per_epoch} steps/epoch, batch={BATCH})")
print(f"  * Epochs                   : {EPOCHS} ({total_steps} optimization steps)")
print(f"  * Initial GPU Temp         : {gpu_temp_str} (Thermal guard: pause 5 min if >= {GPU_MAX_TEMP}C)")
print(f"  * Validation set           : {len(val_idx)} videos")
print(f"  * Checkpoint destination   : {best_model_path.name}")
print("=" * 78 + "\n")

t_start = time.time()
overall_step = 0

for epoch in range(1, EPOCHS + 1):
    model.train()
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

        xv, xa, yb = make_training_batch(batch_tuples, rng_np)
        logits = model(xv, xa, mode="av")
        loss = criterion(logits, yb)

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

    # Validation in Audio-Visual mode
    pv_clean = predict_proba_multimodal(model, variant_caches["clean"], val_idx, mode="av")
    vm_clean = metrics(all_labels[val_idx], pv_clean)

    score = vm_clean["auc"] + vm_clean["balanced_acc"]
    avg_train_loss = total_loss / max(1, steps_per_epoch)
    history.append(
        dict(
            epoch=epoch,
            train_loss=avg_train_loss,
            val_av_clean_auc=vm_clean["auc"],
            val_av_clean_bal_acc=vm_clean["balanced_acc"],
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
        f"val AV-Clean AUC {vm_clean['auc']:.4f} bal-acc {vm_clean['balanced_acc']:.4f} | GPU: {gpu_str} | "
        f"time {format_time(epoch_elapsed)} | elapsed {format_time(total_elapsed)} | ETA {format_time(total_eta)}{flag}    \n"
    )
    sys.stdout.flush()

total_time = time.time() - t_start
print("-" * 78)
print(f"Training complete in {format_time(total_time)}! Best Epoch: {best['epoch']}")
print("-" * 78 + "\n")


# ------------------------- 8. COMPREHENSIVE MULTIMODAL EVALUATION -------------------------
print("[Evaluating Audio-Visual Model & Ablations Across All Variants on Held-Out Test Set...]")
model.load_state_dict(torch.load(best_model_path, map_location=device))
y_test = all_labels[test_idx]

pv_clean_val = predict_proba_multimodal(model, variant_caches["clean"], val_idx, mode="av")
grid = np.linspace(0.05, 0.95, 91)
thr_val = float(grid[np.argmax([balanced_accuracy_score(all_labels[val_idx], (pv_clean_val >= t).astype(int)) for t in grid])])

eval_results_av = {}
eval_results_vonly = {}
eval_results_aonly = {}
pred_table = {"name": all_source_names[test_idx], "label": y_test}

# 8A. Audio-Visual Evaluation across all 10 video variants
for var in active_variants:
    p_test_av = predict_proba_multimodal(model, variant_caches[var], test_idx, mode="av")
    pred_table[f"p_fake_av_{var}"] = p_test_av
    eval_results_av[var] = metrics(y_test, p_test_av, thr=0.5)

    p_test_vo = predict_proba_multimodal(model, variant_caches[var], test_idx, mode="v_only")
    pred_table[f"p_fake_vonly_{var}"] = p_test_vo
    eval_results_vonly[var] = metrics(y_test, p_test_vo, thr=0.5)

# 8B. Audio-Only Baseline (acoustic signal alone)
p_test_ao = predict_proba_multimodal(model, variant_caches["clean"], test_idx, mode="a_only")
pred_table["p_fake_aonly"] = p_test_ao
audio_only_metrics = metrics(y_test, p_test_ao, thr=0.5)

# Print Summary Table
summary_rows = []
for var in active_variants:
    m_av = eval_results_av[var]
    m_vo = eval_results_vonly[var]
    summary_rows.append(
        {
            "Variant": var.upper(),
            "AV AUC": m_av["auc"],
            "AV Accuracy": m_av["accuracy"],
            "AV F1": m_av["f1"],
            "Video-Only AUC": m_vo["auc"],
            "Video-Only Acc": m_vo["accuracy"],
            "Audio-Only AUC": audio_only_metrics["auc"],
            "Audio-Only Acc": audio_only_metrics["accuracy"],
        }
    )

summary_df = pd.DataFrame(summary_rows).set_index("Variant")
print("\n" + "=" * 78)
print(" MULTIMODAL ABLATION & ROBUSTNESS PERFORMANCE ON TEST SET (Threshold = 0.5)")
print("=" * 78)
print(summary_df.round(4).to_string())
print("=" * 78)

predictions_df = pd.DataFrame(pred_table)
predictions_df.to_csv(WORK_DIR / "test_predictions_av_test100.csv", index=False)


# ------------------------- 9. REPORTS & PLOTS -------------------------
now = datetime.now()
date_str = f"{now.year % 100}-{now.month}-{now.day}"
time_str = f"{now.hour:02d}.{now.minute:02d}"
report_filename = f"report_av_test100_date_{date_str}_time_{time_str}.json"

report_data = {
    "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
    "date": date_str,
    "time": time_str,
    "sample_config": {"real_videos": len(selected_real), "fake_videos": len(selected_fake)},
    "multimodal_results": {
        "audio_visual": eval_results_av,
        "video_only": eval_results_vonly,
        "audio_only": audio_only_metrics,
    },
    "summary_table": summary_df.to_dict(orient="index"),
    "best_epoch": best["epoch"],
    "val_threshold": thr_val,
    "training_time": format_time(total_time),
    "gpu_final_temp": f"{get_gpu_temp(ttl=0)}C" if get_gpu_temp(ttl=0) is not None else "N/A",
    "history": history,
}

for rdir in (REPORT_DIR, WORK_DIR / "test"):
    rdir.mkdir(parents=True, exist_ok=True)
    with open(rdir / report_filename, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    with open(rdir / "latest_report_av_test100.json", "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

with open(WORK_DIR / "results_av_test100.json", "w", encoding="utf-8") as f:
    json.dump(report_data, f, indent=2)

print(f"\n[Report saved] -> {REPORT_DIR / report_filename}")

# Plot 1: Training Curves
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
h = pd.DataFrame(history)
ax[0].plot(h.epoch, h.train_loss, label="train loss", color="#e74c3c", linewidth=2)
ax[0].set_xlabel("Epoch")
ax[0].set_ylabel("Loss")
ax[0].set_title("Audio-Visual Training Loss")
ax[0].grid(True, linestyle="--", alpha=0.5)
ax[0].legend()

ax[1].plot(h.epoch, h.val_av_clean_auc, label="val AV Clean AUC", color="#2980b9", linewidth=2)
ax[1].plot(h.epoch, h.val_av_clean_bal_acc, label="val AV Clean Bal-Acc", color="#27ae60", linewidth=2)
ax[1].set_xlabel("Epoch")
ax[1].set_ylabel("Score")
ax[1].set_title("Validation Metrics")
ax[1].grid(True, linestyle="--", alpha=0.5)
ax[1].legend()

plt.tight_layout()
plt.savefig(WORK_DIR / "training_curves_av_test100.png", dpi=130)
plt.close()

# Plot 2: Multimodal Ablation Comparison (Audio-Visual vs Video-Only vs Audio-Only)
fig, ax = plt.subplots(figsize=(11, 5))
variants_upper = [v.upper() for v in active_variants]
av_aucs = [eval_results_av[v]["auc"] for v in active_variants]
vo_aucs = [eval_results_vonly[v]["auc"] for v in active_variants]
ao_auc = audio_only_metrics["auc"]

x = np.arange(len(variants_upper))
width = 0.35
ax.bar(x - width / 2, av_aucs, width, label="Audio-Visual (Multimodal)", color="#2ecc71")
ax.bar(x + width / 2, vo_aucs, width, label="Video-Only", color="#3498db")
ax.axhline(ao_auc, color="#e67e22", linestyle="--", linewidth=2, label=f"Audio-Only Baseline ({ao_auc:.3f})")

ax.set_ylabel("ROC-AUC Score")
ax.set_title("Multimodal Robustness: Audio-Visual vs Video-Only Across Degradation Types")
ax.set_xticks(x)
ax.set_xticklabels(variants_upper, rotation=35, ha="right")
ax.set_ylim([0.0, 1.05])
ax.grid(axis="y", linestyle="--", alpha=0.6)
ax.legend(loc="lower right")

plt.tight_layout()
plt.savefig(WORK_DIR / "multimodal_ablation_comparison.png", dpi=130)
plt.close()

# Plot 3: Confusion Matrix
cm_clean = confusion_matrix(y_test, (pred_table["p_fake_av_clean"] >= 0.5).astype(int), labels=[0, 1])
fig, ax = plt.subplots(figsize=(3.8, 3.5))
ax.imshow(cm_clean, cmap="Greens")
for (i, j), v in np.ndenumerate(cm_clean):
    ax.text(j, i, str(v), ha="center", va="center", fontsize=14, color="black" if cm_clean[i, j] < cm_clean.max() / 2 else "white")
ax.set_xticks([0, 1])
ax.set_xticklabels(["Pred Real", "Pred Fake"])
ax.set_yticks([0, 1])
ax.set_yticklabels(["Real", "Fake"])
ax.set_title("Audio-Visual Clean Test CM")
plt.tight_layout()
plt.savefig(WORK_DIR / "confusion_matrix_av_test100.png", dpi=130)
plt.close("all")

print("\nSaved evaluation plots:")
print(f"  * {WORK_DIR / 'training_curves_av_test100.png'}")
print(f"  * {WORK_DIR / 'multimodal_ablation_comparison.png'}")
print(f"  * {WORK_DIR / 'confusion_matrix_av_test100.png'}")
print("\nAudio-Visual pipeline run completed successfully!")
