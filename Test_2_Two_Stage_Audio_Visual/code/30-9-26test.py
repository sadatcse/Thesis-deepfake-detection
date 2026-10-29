"""
30-9-26 test: Two-stage audio-visual deepfake training + clean/degraded evaluation
==================================================================================

Datasets (clean / original):
  BioDeepAV            E:\\Datasets\\BioDeepAV\\{real,fake}\\videos            (has audio)
  FaceForensics++ C23  E:\\Datasets\\FaceForensics++_C23\\<folder>            (no audio track)
                       original = real, all other folders = fake

Degraded versions (auto-discovered at <folder>\\degraded_output\\manifest.json):
  BioDeepAV real / fake, FF++ DeepFakeDetection
  The manifests are "random mode": every source video has 3 random degradations
  (from noise, blur, brightness, compression, saturation, contrast, motion_blur,
  frame_drop), so a video does NOT have every degradation type.
  Sources without a manifest (FF++ original + the other 5 fake folders) get
  on-the-fly synthetic degradations of the same 8 types, so that "degraded" is
  never a shortcut for "fake" (FF++ originals would otherwise never be degraded).

Pipeline:
  Stage 1  train on ALL original (clean) videos of the train split
  Stage 2  continue training on degraded + original videos
  Test     held-out random test videos (leakage-free groups), each model is tested on
           the original video AND a randomly selected degraded version of it.

Following the literature review (Section 4, "The Gap"), the test also measures whether
EXPLANATIONS stay reliable under degradation: Grad-CAM maps of the clean vs degraded
version of the same video are compared (Pearson r, top-20% IoU, centre shift), and
the rate of "prediction stable but explanation changed" is reported.

Usage:
  python 30-9-26test.py                          # stage1 -> stage2 -> test
  python 30-9-26test.py --max-per-folder 40 --epochs1 1 --epochs2 1   # quick smoke run
  python 30-9-26test.py --mode test              # evaluate saved checkpoints
  python 30-9-26test.py --datasets bio           # BioDeepAV only
"""

import os
import re
import sys
import json
import time
import random
import hashlib
import warnings
import argparse
import subprocess
from collections import deque, defaultdict
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
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
)

warnings.filterwarnings("ignore")
cv2.setNumThreads(1)

DEGRADATION_TYPES = ["noise", "blur", "brightness", "compression", "saturation", "contrast", "motion_blur", "frame_drop"]
SEVERITIES = ["mild", "medium", "severe"]
FRAME_DROP_LEVELS = ["5pct", "10pct", "30pct"]
SEVERITY_LEVEL = {"mild": 0, "medium": 1, "severe": 2, "5pct": 0, "10pct": 1, "30pct": 2}
FFPP_FOLDERS = ["original", "Deepfakes", "Face2Face", "FaceShifter", "FaceSwap", "NeuralTextures", "DeepFakeDetection"]
NO_WINDOW = subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0


# ------------------------- UTILITIES & THERMAL GUARD -------------------------
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
            creationflags=NO_WINDOW,
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
    print(f" Pausing execution for {pause_sec // 60} minutes to let the GPU cool down...")
    print(f"{'!' * 78}\n")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    t_pause_start = time.time()
    while True:
        rem_pause = pause_sec - (time.time() - t_pause_start)
        if rem_pause <= 0:
            break
        temp_now = get_gpu_temp(ttl=0.0)
        temp_str = f"{temp_now}C" if temp_now is not None else "N/A"
        sys.stdout.write(f"\r  [Cooling Down] Remaining: {format_time(rem_pause)} | Current GPU Temp: {temp_str}   ")
        sys.stdout.flush()
        time.sleep(1.0)

    temp_after = get_gpu_temp(ttl=0.0)
    temp_after_str = f"{temp_after}C" if temp_after is not None else "N/A"
    sys.stdout.write(f"\r  [Cooling Down Complete] Cooled to {temp_after_str}. Resuming!                  \n\n")
    sys.stdout.flush()
    return int(time.time() - t_pause_start)


# ------------------------- CLI ARGUMENT PARSER -------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Two-stage (original -> degraded+original) audio-visual deepfake training on BioDeepAV + FaceForensics++"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["full", "stage1", "stage2", "test"],
        default="full",
        help="'full' (stage1 -> stage2 -> test), 'stage1' (original only + test), "
        "'stage2' (fine-tune saved stage1 on degraded+original + test), 'test' (evaluate saved checkpoints)",
    )
    parser.add_argument("--datasets", type=str, default="bio,ffpp", help="Comma list: bio, ffpp")
    parser.add_argument("--bio-root", type=str, default=r"E:\Datasets\BioDeepAV")
    parser.add_argument("--ffpp-root", type=str, default=r"E:\Datasets\FaceForensics++_C23")
    parser.add_argument("--ffpp-folders", type=str, default=",".join(FFPP_FOLDERS), help="FF++ folders to use ('original' = real)")
    parser.add_argument(
        "--extra-manifests",
        type=str,
        nargs="*",
        default=[],
        help="Additional degraded_output/manifest.json files (folders' own degraded_output/manifest.json are found automatically)",
    )
    parser.add_argument("--max-per-folder", type=int, default=None, help="Randomly limit videos per folder (quick smoke tests)")
    parser.add_argument("--test-ratio", type=float, default=0.20)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--epochs1", type=int, default=8, help="Stage 1 epochs (original videos)")
    parser.add_argument("--epochs2", type=int, default=8, help="Stage 2 epochs (degraded + original videos)")
    parser.add_argument("--lr1", type=float, default=1e-3, help="Stage 1 max learning rate")
    parser.add_argument("--lr2", type=float, default=3e-4, help="Stage 2 max learning rate (fine-tuning)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--frames", type=int, default=16, help="Frames cached per video")
    parser.add_argument("--img-size", type=int, default=96)
    parser.add_argument("--clip-len", type=int, default=8, help="Frames per training clip")
    parser.add_argument(
        "--synth-per-video",
        type=int,
        default=3,
        help="Synthetic degraded copies per video for sources WITHOUT a degraded manifest (0 disables)",
    )
    parser.add_argument("--audio-dropout", type=float, default=0.2, help="Prob. of hiding audio during training (FF++ has no audio)")
    parser.add_argument("--xai-samples", type=int, default=150, help="Test videos used for Grad-CAM explanation stability (0 disables)")
    parser.add_argument("--show-random", type=int, default=20, help="Random test videos to print with clean/degraded predictions")
    parser.add_argument("--weights1", type=str, default=None, help="Stage 1 checkpoint (default: <work-dir>/best_stage1_original.pt)")
    parser.add_argument("--weights2", type=str, default=None, help="Stage 2 checkpoint (default: <work-dir>/best_stage2_degraded.pt)")
    parser.add_argument("--work-dir", type=str, default=None, help="Output dir for caches/models (default: ./run_30-9-26)")
    parser.add_argument("--report-dir", type=str, default=None, help="Output dir for reports (default: <work-dir>/test)")
    parser.add_argument("--workers", type=int, default=None, help="Threads for video extraction")
    parser.add_argument("--loader-threads", type=int, default=3, help="Threads preparing training batches")
    parser.add_argument("--gpu-max-temp", type=int, default=72, help="GPU thermal guard trigger temperature")
    parser.add_argument("--gpu-pause", type=int, default=300, help="Pause seconds when GPU exceeds thermal limit")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


# ------------------------- ARCHITECTURE DEFINITION -------------------------
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
    """Same architecture as deepfake_biodeepav_mixed_degradation_av.py, plus an audio mask:
    videos without an audio track (FF++) get zeroed audio features, exactly like 'v_only'."""

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

    def forward(self, x_v, x_a=None, mode="av", a_mask=None):
        ref = x_v if x_v is not None else x_a
        B, dev = ref.size(0), ref.device
        f_v = self.video_net(x_v) if mode in ("av", "v_only") else torch.zeros(B, 256, device=dev)
        f_a = self.audio_net(x_a) if mode in ("av", "a_only") else torch.zeros(B, 128, device=dev)
        if mode == "av" and a_mask is not None:
            f_a = f_a * a_mask.view(-1, 1).to(f_a.dtype)
        return self.fusion_head(torch.cat([f_v, f_a], dim=1))


# ------------------------- AUDIO & VIDEO EXTRACTION -------------------------
def extract_audio_spectrogram(wav_candidates, video_path, n_samples=48000):
    """Returns (spectrogram [1,128,128] float32, has_audio bool)."""
    audio_data = None
    for wav_path in wav_candidates:
        if not Path(wav_path).exists():
            continue
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

                audio_data = resample(audio_data, int(len(audio_data) * 16000 / sr)).astype(np.float32)
            break
        except Exception:
            audio_data = None

    if audio_data is None or len(audio_data) == 0:
        try:
            cmd = ["ffmpeg", "-v", "error", "-i", str(video_path), "-vn", "-ac", "1", "-ar", "16000", "-f", "s16le", "-"]
            raw_bytes = subprocess.check_output(cmd, creationflags=NO_WINDOW, timeout=20, stderr=subprocess.DEVNULL)
            audio_data = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        except Exception:
            audio_data = None

    if audio_data is None or len(audio_data) == 0 or float(np.abs(audio_data).max()) < 1e-6:
        return np.zeros((1, 128, 128), dtype=np.float32), False

    if len(audio_data) < n_samples:
        reps = int(np.ceil(n_samples / len(audio_data)))
        audio_data = np.tile(audio_data, reps)[:n_samples]
    else:
        audio_data = audio_data[:n_samples]

    with torch.no_grad():
        t = torch.from_numpy(np.ascontiguousarray(audio_data))
        stft = torch.stft(t, n_fft=512, hop_length=256, win_length=512, window=torch.hann_window(512), return_complex=True)
        mag = torch.log1p(stft.abs())
        spec = torch.nn.functional.interpolate(mag.unsqueeze(0).unsqueeze(0), size=(128, 128), mode="bilinear", align_corners=False)
        spec = (spec - spec.mean()) / (spec.std() + 1e-6)
        spec = spec.clamp(-3.0, 3.0) / 3.0
        return spec.squeeze(0).numpy().astype(np.float32), True


def extract_video_clip(path, n_frames=16, size=96):
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
        return np.stack(seq)[..., ::-1].copy()  # BGR -> RGB
    except Exception as e:
        print(f"skip {Path(path).name}: {e}")
        return None


# ------------------------- SYNTHETIC DEGRADATION -------------------------
def random_degradation_spec(rng, exclude=()):
    choices = [d for d in DEGRADATION_TYPES if d not in exclude] or DEGRADATION_TYPES
    deg = choices[int(rng.integers(len(choices)))]
    levels = FRAME_DROP_LEVELS if deg == "frame_drop" else SEVERITIES
    return deg, levels[int(rng.integers(len(levels)))]


def synth_degrade(clip, deg, sev, rng):
    """Apply one of the 8 manifest degradation types to an RGB uint8 clip [T,H,W,3].
    Used for sources that have no pre-rendered degraded_output videos."""
    x = np.array(clip, dtype=np.uint8, copy=True)
    lvl = SEVERITY_LEVEL.get(sev, 1)
    if deg == "noise":
        sigma = (5.0, 10.0, 18.0)[lvl]
        x = np.clip(x.astype(np.float32) + rng.normal(0, sigma, x.shape), 0, 255).astype(np.uint8)
    elif deg == "blur":
        s = (0.8, 1.4, 2.2)[lvl]
        x = np.stack([cv2.GaussianBlur(f, (0, 0), s) for f in x])
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
        q = (45, 25, 10)[lvl]
        out = []
        for f in x:
            ok, enc = cv2.imencode(".jpg", cv2.cvtColor(f, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, q])
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
        p = (0.05, 0.10, 0.30)[lvl]
        T = len(x)
        drop = rng.random(T) < p
        drop[0] = False
        if T > 1 and not drop.any():
            drop[int(rng.integers(1, T))] = True
        for t in range(1, T):
            if drop[t]:
                x[t] = x[t - 1]
    return x


# ------------------------- DATASET INDEXING -------------------------
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
    """Identities in an FF++ file name, used to keep every identity in a single split."""
    if subset == "DeepFakeDetection":
        mm = re.match(r"^(\d+)_(\d+)__", stem)
        return [f"dfd:{mm.group(1)}_{mm.group(2)}"] if mm else [f"dfd:{stem}"]
    if subset == "original":
        return [f"ff:{stem}"]
    parts = stem.split("_")
    return [f"ff:{p}" for p in parts[:2]] if len(parts) >= 2 else [f"ff:{stem}"]


def discover_sources(args, rng):
    datasets = {d.strip().lower() for d in args.datasets.split(",") if d.strip()}
    folders = []  # (dataset, subset, label, folder)
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
        files = sorted(folder.glob("*.mp4"))  # non-recursive: degraded_output/ is not included
        if args.max_per_folder and len(files) > args.max_per_folder:
            files = sorted(rng.sample(files, args.max_per_folder))
        for p in files:
            sources.append(
                dict(
                    key=f"{ds}/{sub}/{p.name}",
                    dataset=ds,
                    subset=sub,
                    label=label,
                    path=p,
                    folder=folder,
                    wavs=[folder / (p.stem + ".wav"), folder.parent / "audios" / (p.stem + ".wav")],
                )
            )

    # Leakage-free group ids
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
    """Read every degraded_output/manifest.json and link outputs to their source videos."""
    lookup = {(str(s["folder"]).lower(), s["path"].name): i for i, s in enumerate(sources)}
    manifest_paths = [f / "degraded_output" / "manifest.json" for _, _, _, f in folders]
    manifest_paths += [Path(m) for m in extra_manifests]

    degraded, seen, manifest_summary = [], set(), []
    for m_path in manifest_paths:
        if not m_path.exists() or str(m_path).lower() in seen:
            continue
        seen.add(str(m_path).lower())
        deg_base = m_path.parent
        src_folder = str(deg_base.parent).lower()
        with open(m_path, "r", encoding="utf-8") as f:
            m_data = json.load(f)
        n_linked, n_missing = 0, 0
        for item in m_data.get("files", []):
            si = lookup.get((src_folder, Path(item["source"]).name))
            if si is None:
                continue
            for out in item.get("outputs", []):
                p = deg_base / out["output"]
                if not p.exists():
                    n_missing += 1
                    continue
                degraded.append(
                    dict(
                        src=si,
                        path=p,
                        degradation=out.get("degradation", "unknown"),
                        severity=out.get("severity", out.get("level", "")),
                        set=out.get("randomSet", ""),
                    )
                )
                n_linked += 1
        manifest_summary.append(dict(manifest=str(m_path), linked_outputs=n_linked, missing_files=n_missing))
    return degraded, manifest_summary


def make_splits(sources, test_ratio, val_ratio, seed):
    """Grouped split: all videos of an identity group go to the same split.
    Stratified per (dataset, label) when a group is single-label, else per dataset."""
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
        test_target, val_target = int(round(n * test_ratio)), int(round(n * val_ratio))
        targets = {"test": test_target, "val": val_target, "train": n - test_target - val_target}
        filled = {"test": 0, "val": 0, "train": 0}
        # biggest groups first, each into the split with the largest remaining (relative) deficit
        for g in sorted(gs, key=lambda g: -len(groups[g])):
            name = max(targets, key=lambda k: (targets[k] - filled[k]) / max(1, targets[k]))
            filled[name] += len(groups[g])
            for i in groups[g]:
                split[i] = name
    return split


# ------------------------- CACHING (MEMORY-MAPPED, RESUMABLE) -------------------------
def build_video_cache(name, paths, n_frames, size, cache_dir, n_workers):
    """uint8 [N, T, S, S, 3] memmap. status: 0 = pending, 1 = ok, 2 = failed. Resumes if interrupted."""
    key = hashlib.md5(("\n".join(str(p) for p in paths) + f"|{n_frames}|{size}").encode("utf-8")).hexdigest()[:10]
    cache_path = cache_dir / f"cache_{name}_T{n_frames}_S{size}_n{len(paths)}_{key}.npy"
    status_path = cache_dir / f"cache_{name}_T{n_frames}_S{size}_n{len(paths)}_{key}.status.npy"
    shape = (len(paths), n_frames, size, size, 3)

    if cache_path.exists() and status_path.exists():
        status = np.load(status_path)
        arr = np.lib.format.open_memmap(cache_path, mode="r+")
    else:
        status = np.zeros(len(paths), dtype=np.int8)
        arr = np.lib.format.open_memmap(cache_path, mode="w+", dtype=np.uint8, shape=shape)

    todo = np.where(status == 0)[0]
    if len(todo) == 0:
        print(f"  -> Found cached {name.upper()}: {cache_path.name} ({int((status == 1).sum())} ok, {int((status == 2).sum())} failed)")
        del arr
        return np.load(cache_path, mmap_mode="r"), status == 1

    print(f"\n[Extracting & Caching {name.upper()}] ({len(todo)} of {len(paths)} videos remaining | {n_workers} threads)")
    t0 = time.time()
    last_print = last_save = 0
    completed = 0
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futures = {ex.submit(extract_video_clip, paths[i], n_frames, size): i for i in todo}
        for fut in as_completed(futures):
            i = futures[fut]
            try:
                res = fut.result()
            except Exception:
                res = None
            if res is not None and res.shape == shape[1:]:
                arr[i] = res
                status[i] = 1
            else:
                arr[i] = 0
                status[i] = 2
            completed += 1
            now = time.time()
            if now - last_save >= 30:
                arr.flush()
                np.save(status_path, status)
                last_save = now
            if now - last_print >= 0.15 or completed == len(todo):
                elapsed = now - t0
                rate = completed / max(0.001, elapsed)
                eta = (len(todo) - completed) / max(0.001, rate)
                sys.stdout.write(
                    f"\r  {name:13s}: {progress_bar(completed, len(todo))} ({completed}/{len(todo)}) | {rate:5.1f} vid/s | "
                    f"Elapsed: {format_time(elapsed)} | ETA: {format_time(eta)}   "
                )
                sys.stdout.flush()
                last_print = now
    sys.stdout.write("\n")
    arr.flush()
    np.save(status_path, status)
    del arr
    n_fail = int((status == 2).sum())
    print(f"  Saved cache: {cache_path.name} ({cache_path.stat().st_size / 1e9:.2f} GB, {n_fail} failed videos)")
    return np.load(cache_path, mmap_mode="r"), status == 1


def build_audio_cache(sources, cache_dir, n_workers):
    """float32 [N,1,128,128] spectrograms + has_audio flags. status: 0 pending, 1 audio, 3 no audio."""
    key = hashlib.md5("\n".join(str(s["path"]) for s in sources).encode("utf-8")).hexdigest()[:10]
    cache_path = cache_dir / f"cache_audio_n{len(sources)}_{key}.npy"
    status_path = cache_dir / f"cache_audio_n{len(sources)}_{key}.status.npy"
    shape = (len(sources), 1, 128, 128)

    if cache_path.exists() and status_path.exists():
        status = np.load(status_path)
        arr = np.lib.format.open_memmap(cache_path, mode="r+")
    else:
        status = np.zeros(len(sources), dtype=np.int8)
        arr = np.lib.format.open_memmap(cache_path, mode="w+", dtype=np.float32, shape=shape)

    todo = np.where(status == 0)[0]
    if len(todo):
        print(f"  -> Extracting audio spectrograms for {len(todo)} videos ({n_workers} threads)...")
        t0 = time.time()
        last_print = last_save = 0
        completed = 0
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            futures = {ex.submit(extract_audio_spectrogram, sources[i]["wavs"], sources[i]["path"]): i for i in todo}
            for fut in as_completed(futures):
                i = futures[fut]
                try:
                    spec, has = fut.result()
                except Exception:
                    spec, has = np.zeros((1, 128, 128), dtype=np.float32), False
                arr[i] = spec
                status[i] = 1 if has else 3
                completed += 1
                now = time.time()
                if now - last_save >= 30:
                    arr.flush()
                    np.save(status_path, status)
                    last_save = now
                if now - last_print >= 0.15 or completed == len(todo):
                    elapsed = now - t0
                    rate = completed / max(0.001, elapsed)
                    sys.stdout.write(
                        f"\r  audio spectrograms: {progress_bar(completed, len(todo))} ({completed}/{len(todo)}) | "
                        f"{rate:5.1f} vid/s | ETA: {format_time((len(todo) - completed) / max(0.001, rate))}   "
                    )
                    sys.stdout.flush()
                    last_print = now
        sys.stdout.write("\n")
        arr.flush()
        np.save(status_path, status)
    else:
        print(f"  -> Found cached Audio Spectrograms: {cache_path.name}")
    del arr
    return np.load(cache_path, mmap_mode="r"), status == 1


# ------------------------- METRICS COMPUTATION -------------------------
def compute_metrics(y_true, p, thr=0.5):
    y_true = np.asarray(y_true)
    p = np.asarray(p)
    if len(y_true) == 0:
        return dict(n=0, accuracy=float("nan"), balanced_acc=float("nan"), precision=float("nan"), recall=float("nan"),
                    specificity=float("nan"), f1=float("nan"), auc=float("nan"), tn=0, fp=0, fn=0, tp=0, threshold=float(thr))
    pred = (p >= thr).astype(int)
    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    auc = roc_auc_score(y_true, p) if len(set(y_true.tolist())) > 1 else float("nan")
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
    return dict(
        n=int(len(y_true)),
        accuracy=float(accuracy_score(y_true, pred)) if len(y_true) else float("nan"),
        balanced_acc=float(balanced_accuracy_score(y_true, pred)) if len(set(y_true.tolist())) > 1 else float("nan"),
        precision=float(precision_score(y_true, pred, zero_division=0)),
        recall=float(recall_score(y_true, pred, zero_division=0)),
        specificity=float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan"),
        f1=float(f1_score(y_true, pred, zero_division=0)),
        auc=float(auc),
        tn=tn,
        fp=fp,
        fn=fn,
        tp=tp,
        threshold=float(thr),
    )


def _nan0(v, default=0.5):
    return default if (v is None or np.isnan(v) or np.isinf(v)) else v


# ------------------------- MAIN WORKFLOW -------------------------
def main():
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    work_dir = Path(args.work_dir) if args.work_dir else (script_dir / "run_30-9-26")
    report_dir = Path(args.report_dir) if args.report_dir else (work_dir / "test")
    cache_dir = work_dir / "cache"
    for d in (work_dir, report_dir, cache_dir):
        d.mkdir(parents=True, exist_ok=True)

    n_frames, img_size, clip_len = args.frames, args.img_size, args.clip_len
    stride = max(1, n_frames // clip_len)
    batch_size = args.batch_size
    n_cpu = os.cpu_count() or 8
    n_workers = args.workers or n_cpu

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(n_cpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    stage1_path = Path(args.weights1) if args.weights1 else work_dir / "best_stage1_original.pt"
    stage2_path = Path(args.weights2) if args.weights2 else work_dir / "best_stage2_degraded.pt"

    print("=" * 90)
    print(" 30-9-26 TEST: TWO-STAGE AUDIO-VISUAL DEEPFAKE PIPELINE (ORIGINAL -> DEGRADED + ORIGINAL)")
    print(f"  * Mode              : {args.mode.upper()}")
    print(f"  * Device            : {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    print(f"  * Datasets          : {args.datasets}")
    print(f"  * Work Directory    : {work_dir}")
    print("=" * 90)

    # ---------------- 1. Index sources, degraded versions & splits ----------------
    sources, folders = discover_sources(args, random.Random(args.seed))
    if not sources:
        raise RuntimeError("No source videos found. Check --bio-root / --ffpp-root / --datasets.")
    degraded, manifest_summary = discover_degraded(sources, folders, args.extra_manifests)
    split = make_splits(sources, args.test_ratio, args.val_ratio, args.seed)

    # ---------------- 2. Caches ----------------
    print("\n--- Preparing Video Caches (Memory-Mapped) ---")
    clean_cache, clean_ok = build_video_cache("clean", [s["path"] for s in sources], n_frames, img_size, cache_dir, n_workers)
    if degraded:
        deg_cache, deg_ok = build_video_cache("degraded", [d["path"] for d in degraded], n_frames, img_size, cache_dir, n_workers)
    else:
        deg_cache, deg_ok = None, np.zeros(0, dtype=bool)
    print("\n--- Preparing Audio Spectrogram Cache ---")
    audio_cache, has_audio = build_audio_cache(sources, cache_dir, n_workers)

    labels = np.array([s["label"] for s in sources], dtype=np.int64)
    split = np.where(clean_ok, split, "dropped")
    deg_src = np.array([d["src"] for d in degraded], dtype=np.int64)
    deg_valid = np.array([bool(deg_ok[j]) and bool(clean_ok[d["src"]]) for j, d in enumerate(degraded)], dtype=bool)
    src_to_deg = defaultdict(list)
    for j, d in enumerate(degraded):
        if deg_valid[j]:
            src_to_deg[d["src"]].append(j)

    train_idx = np.where(split == "train")[0]
    val_idx = np.where(split == "val")[0]
    test_idx = np.where(split == "test")[0]
    json.dump(
        {k: [sources[i]["key"] for i in idx] for k, idx in (("train", train_idx), ("val", val_idx), ("test", test_idx))},
        open(work_dir / "split_30-9-26.json", "w", encoding="utf-8"),
        indent=1,
    )

    # Inventory banner
    inv_rows = []
    for ds, sub, label, folder in folders:
        idx = [i for i, s in enumerate(sources) if s["folder"] == folder]
        if not idx:
            continue
        n_deg = sum(len(src_to_deg[i]) for i in idx)
        inv_rows.append(
            {
                "Folder": f"{ds}/{sub}",
                "Label": "REAL" if label == 0 else "FAKE",
                "Original": len(idx),
                "Degraded (manifest)": n_deg,
                "Audio": int(has_audio[idx].sum()),
                "Train": int(sum(split[i] == "train" for i in idx)),
                "Val": int(sum(split[i] == "val" for i in idx)),
                "Test": int(sum(split[i] == "test" for i in idx)),
                "Failed": int(sum(split[i] == "dropped" for i in idx)),
            }
        )
    df_inv = pd.DataFrame(inv_rows).set_index("Folder")
    print("\n" + "=" * 90)
    print(" [DATASET INVENTORY: ORIGINAL + DEGRADED]")
    print("=" * 90)
    print(df_inv.to_string())
    print("-" * 90)
    for m in manifest_summary:
        print(f"  * Manifest: {m['manifest']} -> {m['linked_outputs']:,} degraded videos linked ({m['missing_files']} missing)")
    deg_types_found = sorted({degraded[j]["degradation"] for j in range(len(degraded)) if deg_valid[j]})
    print(f"  * Degradation types in manifests : {', '.join(deg_types_found) if deg_types_found else 'none'}")
    print(f"  * Total original videos          : {int(clean_ok.sum()):,} (Real {int((labels[clean_ok] == 0).sum()):,} | Fake {int((labels[clean_ok] == 1).sum()):,})")
    print(f"  * Total degraded videos          : {int(deg_valid.sum()):,}")
    print(f"  * Synthetic degraded per video   : {args.synth_per_video} (only for videos without a degraded manifest)")
    print(f"  * Split (leakage-free groups)    : train {len(train_idx):,} | val {len(val_idx):,} | test {len(test_idx):,}")
    print("=" * 90 + "\n")

    # ---------------- 3. Item helpers ----------------
    # item = (kind, idx, spec): ("clean", src, None) | ("deg", deg_j, None) | ("synth", src, (type, severity, seed) or None)
    def item_src(it):
        return int(deg_src[it[1]]) if it[0] == "deg" else int(it[1])

    def item_degradation(it):
        if it[0] == "clean":
            return "clean", ""
        if it[0] == "deg":
            d = degraded[it[1]]
            return d["degradation"], d["severity"]
        return it[2][0], it[2][1]

    def item_name(it):
        if it[0] == "deg":
            return str(degraded[it[1]]["path"].name)
        base = sources[it[1]]["path"].name
        return base if it[0] == "clean" else f"{base} [synthetic {it[2][0]}_{it[2][1]}]"

    def get_full_clip(it):
        if it[0] == "clean":
            return np.asarray(clean_cache[it[1]])
        if it[0] == "deg":
            return np.asarray(deg_cache[it[1]])
        deg, sev, seed = it[2]
        return synth_degrade(clean_cache[it[1]], deg, sev, np.random.default_rng(seed))

    def all_degraded_items(idx_list, seed_offset):
        """All degraded versions of the given sources: manifest outputs, or synthetic ones if none exist."""
        items = []
        for s in idx_list:
            s = int(s)
            if src_to_deg[s]:
                items += [("deg", j, None) for j in src_to_deg[s]]
            elif args.synth_per_video > 0:
                rng_s = np.random.default_rng(args.seed * 100003 + seed_offset + s)
                used = []
                for k in range(args.synth_per_video):
                    deg, sev = random_degradation_spec(rng_s, exclude=used)
                    used.append(deg)
                    items.append(("synth", s, (deg, sev, int(rng_s.integers(2**31)))))
        return items

    def random_degraded_choice(items, idx_list, seed):
        """One randomly selected degraded version per source (aligned with idx_list). Returns positions into items."""
        by_src = defaultdict(list)
        for pos, it in enumerate(items):
            by_src[item_src(it)].append(pos)
        rng_c = random.Random(seed)
        return np.array([rng_c.choice(by_src[int(s)]) if by_src[int(s)] else -1 for s in idx_list], dtype=np.int64)

    def to_video_tensor(arr):
        t = torch.from_numpy(np.ascontiguousarray(arr)).permute(0, 1, 4, 2, 3).float().div_(127.5).sub_(1.0)
        return t.to(device, non_blocking=True)

    @torch.no_grad()
    def predict(model_eval, items, mode="av", bs=32):
        model_eval.eval()
        out = []
        for i in range(0, len(items), bs):
            chunk = items[i : i + bs]
            clips = np.stack([get_full_clip(it) for it in chunk])
            src = np.array([item_src(it) for it in chunk])
            xa = torch.from_numpy(np.asarray(audio_cache[src])).to(device)
            am = torch.from_numpy(has_audio[src].astype(np.float32)).to(device)
            p = 0
            for v in range(stride):
                xb = clips[:, v::stride][:, :clip_len]
                logits = model_eval(to_video_tensor(xb), xa, mode=mode, a_mask=am)
                p = p + torch.softmax(logits.float(), 1)[:, 1].cpu().numpy()
            out.append(p / stride)
        return np.concatenate(out) if out else np.zeros(0)

    # Validation sets (clean + one random degraded version per val video)
    val_clean_items = [("clean", int(s), None) for s in val_idx]
    val_deg_all = all_degraded_items(val_idx, seed_offset=7)
    val_pick = random_degraded_choice(val_deg_all, val_idx, args.seed + 7)
    val_deg_items = [val_deg_all[p] for p in val_pick if p >= 0]
    y_val_clean = labels[val_idx]
    y_val_deg = np.array([labels[item_src(it)] for it in val_deg_items], dtype=np.int64)

    # ---------------- 4. Training ----------------
    def make_training_batch(batch_items, seed, audio_dropout):
        rng_b = np.random.default_rng(seed)
        B = len(batch_items)
        frames = np.empty((B, clip_len, img_size, img_size, 3), dtype=np.uint8)
        src = np.empty(B, dtype=np.int64)
        for b, it in enumerate(batch_items):
            sel = np.sort(rng_b.choice(n_frames, clip_len, replace=False))
            if it[0] == "deg":
                clip = np.asarray(deg_cache[it[1], sel])
            else:
                clip = np.asarray(clean_cache[it[1], sel])
                if it[0] == "synth":
                    deg, sev = random_degradation_spec(rng_b) if it[2] is None else it[2][:2]
                    clip = synth_degrade(clip, deg, sev, rng_b)
            if rng_b.random() < 0.5:
                clip = clip[:, :, ::-1, :]
            pad = np.pad(clip, ((0, 0), (4, 4), (4, 4), (0, 0)), mode="edge")
            dy, dx = rng_b.integers(0, 9, 2)
            frames[b] = pad[:, dy : dy + img_size, dx : dx + img_size]
            src[b] = item_src(it)
        audio = np.asarray(audio_cache[src]).astype(np.float32)
        a_mask = (has_audio[src] & (rng_b.random(B) >= audio_dropout)).astype(np.float32)
        return frames, audio, a_mask, labels[src]

    def train_stage(stage_name, model, train_items, epochs, lr, ckpt_path, val_sets):
        print("\n" + "=" * 90)
        print(f" {stage_name}")
        print(f"  * Training instances/epoch : {len(train_items):,}")
        kinds = pd.Series([it[0] for it in train_items]).value_counts().to_dict()
        print(f"  * Instance mix             : " + ", ".join(f"{k}={v:,}" for k, v in kinds.items()))
        steps_per_epoch = int(np.ceil(len(train_items) / batch_size))
        total_steps = max(1, epochs * steps_per_epoch)
        print(f"  * Epochs                   : {epochs} ({steps_per_epoch} steps/epoch, batch={batch_size}, max lr={lr})")
        print(f"  * Validation sets          : " + ", ".join(f"{n} ({len(its)})" for n, its, _ in val_sets))
        print(f"  * Checkpoint               : {ckpt_path.name}")
        print("=" * 90)

        y_items = np.array([labels[item_src(it)] for it in train_items])
        counts = np.bincount(y_items, minlength=2).astype(np.float64)
        class_w = torch.tensor(counts.sum() / (2.0 * np.maximum(counts, 1)), dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_w)
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, total_steps=total_steps)
        rng_np = np.random.default_rng(args.seed + len(stage_name))

        history, best = [], dict(score=-1.0, epoch=-1)
        t_start = time.time()
        overall_step = 0
        loader = ThreadPoolExecutor(max_workers=max(1, args.loader_threads))
        prefetch = max(2, args.loader_threads * 2)

        for epoch in range(1, epochs + 1):
            model.train()
            perm = rng_np.permutation(len(train_items))
            seeds = rng_np.integers(0, 2**31, size=steps_per_epoch)
            batches = [perm[s * batch_size : (s + 1) * batch_size] for s in range(steps_per_epoch)]
            queue = deque()
            next_submit = 0

            def submit_next():
                nonlocal next_submit
                if next_submit < steps_per_epoch:
                    its = [train_items[k] for k in batches[next_submit]]
                    queue.append(loader.submit(make_training_batch, its, int(seeds[next_submit]), args.audio_dropout))
                    next_submit += 1

            for _ in range(prefetch):
                submit_next()

            total_loss, n_loss = 0.0, 0
            t0 = time.time()
            last_print = 0
            for s in range(steps_per_epoch):
                frames, audio, a_mask, yb = queue.popleft().result()
                submit_next()
                overall_step += 1
                if len(yb) < 2:
                    continue
                xv = to_video_tensor(frames)
                xa = torch.from_numpy(audio).to(device)
                am = torch.from_numpy(a_mask).to(device)
                yt = torch.from_numpy(yb).to(device)
                logits = model(xv, xa, mode="av", a_mask=am)
                loss = criterion(logits, yt)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if scheduler.last_epoch < total_steps - 1:
                    scheduler.step()
                total_loss += loss.item()
                n_loss += 1

                now = time.time()
                step_in_epoch = s + 1
                if now - last_print >= 0.15 or step_in_epoch == steps_per_epoch:
                    gpu_temp = get_gpu_temp()
                    if gpu_temp is not None and gpu_temp >= args.gpu_max_temp:
                        paused = check_gpu_thermal_guard(args.gpu_max_temp, args.gpu_pause)
                        t_start += paused
                        t0 += paused
                        now = time.time()
                        gpu_temp = get_gpu_temp(ttl=0)
                    gpu_str = f"{gpu_temp}C" if gpu_temp is not None else "N/A"
                    ep_eta = (steps_per_epoch - step_in_epoch) / max(0.001, step_in_epoch / max(0.001, now - t0))
                    tot_eta = (total_steps - overall_step) / max(0.001, overall_step / max(0.001, now - t_start))
                    sys.stdout.write(
                        f"\rEpoch [{epoch:2d}/{epochs}] {progress_bar(step_in_epoch, steps_per_epoch, 16)} step {step_in_epoch:4d}/{steps_per_epoch} "
                        f"| loss {total_loss / max(1, n_loss):.4f} | GPU: {gpu_str} | ep ETA: {format_time(ep_eta)} | total ETA: {format_time(tot_eta)}   "
                    )
                    sys.stdout.flush()
                    last_print = now

            # Validation
            row = dict(epoch=epoch, train_loss=total_loss / max(1, n_loss))
            scores = []
            for vname, vitems, vy in val_sets:
                if len(vitems) == 0:
                    continue
                vm = compute_metrics(vy, predict(model, vitems, mode="av"))
                row[f"val_{vname}_auc"] = vm["auc"]
                row[f"val_{vname}_bal_acc"] = vm["balanced_acc"]
                scores.append(_nan0(vm["auc"]) + _nan0(vm["balanced_acc"]))
            score = float(np.mean(scores)) if scores else -row["train_loss"]
            row["score"] = score
            history.append(row)

            flag = ""
            if score > best["score"]:
                best = dict(score=score, epoch=epoch)
                torch.save(model.state_dict(), ckpt_path)
                flag = "  <-- BEST"
            val_str = " | ".join(
                f"val {vn} AUC {row.get(f'val_{vn}_auc', float('nan')):.4f} bal-acc {row.get(f'val_{vn}_bal_acc', float('nan')):.4f}"
                for vn, _, _ in val_sets
            )
            sys.stdout.write(
                f"\rEpoch [{epoch:2d}/{epochs}] loss {row['train_loss']:.4f} | {val_str} | "
                f"time {format_time(time.time() - t0)} | elapsed {format_time(time.time() - t_start)}{flag}          \n"
            )
            sys.stdout.flush()

        loader.shutdown(wait=True)
        train_time = time.time() - t_start
        print("-" * 90)
        print(f"{stage_name} complete in {format_time(train_time)} | Best epoch: {best['epoch']} (score {best['score']:.4f})")
        print("-" * 90)
        pd.DataFrame(history).to_csv(work_dir / f"history_{ckpt_path.stem}.csv", index=False)
        return history, train_time

    histories, train_times = {}, {}
    val_clean_set = ("clean", val_clean_items, y_val_clean)
    val_deg_set = ("degraded", val_deg_items, y_val_deg)

    if args.mode in ("full", "stage1"):
        model = AudioVisualDeepFakeCNN().to(device)
        print(f"\nModel: AudioVisualDeepFakeCNN | Parameters: {sum(p.numel() for p in model.parameters()):,}")
        train_items_1 = [("clean", int(s), None) for s in train_idx]
        histories["stage1"], train_times["stage1"] = train_stage(
            "STAGE 1: TRAIN WITH ALL ORIGINAL VIDEOS", model, train_items_1, args.epochs1, args.lr1, stage1_path, [val_clean_set]
        )

    if args.mode in ("full", "stage2"):
        if not stage1_path.exists():
            raise FileNotFoundError(f"Stage 1 checkpoint not found: {stage1_path} (run --mode stage1 first or pass --weights1)")
        model = AudioVisualDeepFakeCNN().to(device)
        model.load_state_dict(torch.load(stage1_path, map_location=device))
        print(f"\n[Stage 2 initialised from] {stage1_path}")
        train_items_2 = [("clean", int(s), None) for s in train_idx]
        for s in train_idx:
            s = int(s)
            if src_to_deg[s]:
                train_items_2 += [("deg", j, None) for j in src_to_deg[s]]
            else:
                train_items_2 += [("synth", s, None)] * args.synth_per_video  # new random degradation every epoch
        histories["stage2"], train_times["stage2"] = train_stage(
            "STAGE 2: TRAIN WITH DEGRADED + ORIGINAL VIDEOS", model, train_items_2, args.epochs2, args.lr2, stage2_path,
            [val_clean_set, val_deg_set],
        )

    # ---------------- 5. Evaluation on random held-out test videos ----------------
    models_to_test = []
    for tag, pth, title in (("stage1", stage1_path, "Stage 1 (original only)"), ("stage2", stage2_path, "Stage 2 (degraded + original)")):
        if args.mode == "stage1" and tag == "stage2":
            continue
        if pth.exists():
            m = AudioVisualDeepFakeCNN().to(device)
            m.load_state_dict(torch.load(pth, map_location=device))
            m.eval()
            models_to_test.append((tag, title, m))
    if not models_to_test:
        raise FileNotFoundError(f"No checkpoints to test. Expected {stage1_path} and/or {stage2_path}")

    test_clean_items = [("clean", int(s), None) for s in test_idx]
    test_deg_all = all_degraded_items(test_idx, seed_offset=2026)
    test_pick = random_degraded_choice(test_deg_all, test_idx, args.seed + 2026)
    has_deg_version = test_pick >= 0
    y_test = labels[test_idx]
    y_deg_all = np.array([labels[item_src(it)] for it in test_deg_all], dtype=np.int64)
    deg_all_types = [item_degradation(it)[0] for it in test_deg_all]
    deg_all_origin = ["manifest" if it[0] == "deg" else "synthetic" for it in test_deg_all]
    subsets = np.array([f"{sources[s]['dataset']}/{sources[s]['subset']}" for s in test_idx])
    ds_names = np.array([sources[s]["dataset"] for s in test_idx])
    audio_test_mask = has_audio[test_idx]

    print("\n" + "=" * 90)
    print(" TEST: RANDOM HELD-OUT VIDEOS, ORIGINAL vs DEGRADED VERSION")
    print(f"  * Test videos            : {len(test_idx):,} (Real {int((y_test == 0).sum()):,} | Fake {int((y_test == 1).sum()):,})")
    print(f"  * Degraded test versions : {len(test_deg_all):,} ({deg_all_origin.count('manifest'):,} manifest + {deg_all_origin.count('synthetic'):,} synthetic)")
    print(f"  * Random-degraded test   : 1 randomly selected degraded version per test video")
    print("=" * 90)

    results = {}
    pred_table = {
        "video": [sources[s]["key"] for s in test_idx],
        "subset": subsets,
        "true_label": y_test,
        "degraded_version": [item_name(test_deg_all[p]) if p >= 0 else "" for p in test_pick],
        "degradation": [item_degradation(test_deg_all[p])[0] if p >= 0 else "" for p in test_pick],
        "severity": [item_degradation(test_deg_all[p])[1] if p >= 0 else "" for p in test_pick],
    }
    deg_all_rows = {
        "video": [sources[item_src(it)]["key"] for it in test_deg_all],
        "degraded_version": [item_name(it) for it in test_deg_all],
        "origin": deg_all_origin,
        "degradation": deg_all_types,
        "severity": [item_degradation(it)[1] for it in test_deg_all],
        "true_label": y_deg_all,
    }

    for tag, title, m in models_to_test:
        print(f"\n[{title}] predicting original, degraded and video-only ablations...")
        p_clean = predict(m, test_clean_items, "av")
        p_clean_vo = predict(m, test_clean_items, "v_only")
        p_all = predict(m, test_deg_all, "av")
        p_all_vo = predict(m, test_deg_all, "v_only")
        p_deg = np.where(has_deg_version, p_all[np.maximum(test_pick, 0)] if len(p_all) else np.nan, np.nan)
        p_deg_vo = np.where(has_deg_version, p_all_vo[np.maximum(test_pick, 0)] if len(p_all_vo) else np.nan, np.nan)
        hv = has_deg_version

        r = dict(title=title)
        r["clean"] = compute_metrics(y_test, p_clean)
        r["degraded_random"] = compute_metrics(y_test[hv], p_deg[hv])
        r["clean_same_videos"] = compute_metrics(y_test[hv], p_clean[hv])
        r["clean_vonly"] = compute_metrics(y_test, p_clean_vo)
        r["degraded_random_vonly"] = compute_metrics(y_test[hv], p_deg_vo[hv])
        r["pooled_clean_plus_all_degraded"] = compute_metrics(np.concatenate([y_test, y_deg_all]), np.concatenate([p_clean, p_all]))
        r["all_degraded"] = compute_metrics(y_deg_all, p_all) if len(p_all) else {}
        if audio_test_mask.any():
            p_ao = predict(m, [test_clean_items[i] for i in np.where(audio_test_mask)[0]], "a_only")
            r["audio_only_clean"] = compute_metrics(y_test[audio_test_mask], p_ao)
        # Prediction stability: same decision for original and its degraded version
        same = ((p_clean[hv] >= 0.5) == (p_deg[hv] >= 0.5))
        r["prediction_flip_rate"] = float(1.0 - same.mean()) if hv.any() else float("nan")
        r["mean_abs_prob_shift"] = float(np.abs(p_clean[hv] - p_deg[hv]).mean()) if hv.any() else float("nan")

        per_type = {}
        for deg in sorted(set(deg_all_types)):
            sel = np.array([t == deg for t in deg_all_types])
            per_type[deg] = compute_metrics(y_deg_all[sel], p_all[sel])
            per_type[deg]["origin_manifest_share"] = float(np.mean([deg_all_origin[i] == "manifest" for i in np.where(sel)[0]]))
        r["per_degradation"] = per_type

        per_subset = {}
        for sub in sorted(set(subsets)):
            sel = subsets == sub
            sel_d = sel & hv
            per_subset[sub] = dict(
                n=int(sel.sum()),
                clean_accuracy=float(((p_clean[sel] >= 0.5) == y_test[sel]).mean()),
                degraded_accuracy=float(((p_deg[sel_d] >= 0.5) == y_test[sel_d]).mean()) if sel_d.any() else float("nan"),
            )
        r["per_subset"] = per_subset
        per_dataset = {}
        for ds in sorted(set(ds_names)):
            sel = ds_names == ds
            sel_d = sel & hv
            per_dataset[ds] = dict(clean=compute_metrics(y_test[sel], p_clean[sel]), degraded=compute_metrics(y_test[sel_d], p_deg[sel_d]))
        r["per_dataset"] = per_dataset
        r["_p_clean"], r["_p_deg"], r["_p_all"] = p_clean, p_deg, p_all
        results[tag] = r

        pred_table[f"p_fake_original_{tag}"] = p_clean
        pred_table[f"pred_original_{tag}"] = (p_clean >= 0.5).astype(int)
        pred_table[f"p_fake_degraded_{tag}"] = p_deg
        pred_table[f"pred_degraded_{tag}"] = np.where(hv, (p_deg >= 0.5).astype(int), -1)
        deg_all_rows[f"p_fake_{tag}"] = p_all

    # ---------------- 6. Tables ----------------
    comp_rows = []
    for tag, r in results.items():
        c, d = r["clean"], r["degraded_random"]
        comp_rows.append(
            {
                "Model": r["title"],
                "Original AUC": c["auc"],
                "Original Acc": c["accuracy"],
                "Original F1": c["f1"],
                "Degraded AUC": d["auc"],
                "Degraded Acc": d["accuracy"],
                "Degraded F1": d["f1"],
                "Delta AUC": d["auc"] - r["clean_same_videos"]["auc"],
                "Delta Acc": d["accuracy"] - r["clean_same_videos"]["accuracy"],
                "Pooled AUC": r["pooled_clean_plus_all_degraded"]["auc"],
                "Pred Flip %": r["prediction_flip_rate"] * 100,
            }
        )
    df_comp = pd.DataFrame(comp_rows).set_index("Model")

    type_rows = []
    for tag, r in results.items():
        for deg, mt in r["per_degradation"].items():
            type_rows.append(
                {"Model": tag, "Degradation": deg, "N": mt["n"], "AUC": mt["auc"], "Accuracy": mt["accuracy"], "Bal-Acc": mt["balanced_acc"],
                 "Delta AUC vs Original": mt["auc"] - r["clean"]["auc"], "Manifest share": mt["origin_manifest_share"]}
            )
    df_types = pd.DataFrame(type_rows)

    sub_rows = []
    for tag, r in results.items():
        for sub, v in r["per_subset"].items():
            sub_rows.append({"Model": tag, "Subset": sub, "N": v["n"], "Original Acc": v["clean_accuracy"], "Degraded Acc": v["degraded_accuracy"],
                             "Delta": v["degraded_accuracy"] - v["clean_accuracy"]})
    df_subsets = pd.DataFrame(sub_rows)

    print("\n" + "=" * 90)
    print(" TABLE 1: ORIGINAL vs RANDOM DEGRADED VERSION (held-out test videos)")
    print("=" * 90)
    print(df_comp.round(4).to_string())
    for tag, r in results.items():
        c, d = r["clean"], r["degraded_random"]
        print(f"\n  [{r['title']}] Confusion  original [TN FP / FN TP] = [{c['tn']} {c['fp']} / {c['fn']} {c['tp']}]"
              f"   degraded = [{d['tn']} {d['fp']} / {d['fn']} {d['tp']}]")
        print(f"  Video-only AUC: original {r['clean_vonly']['auc']:.4f} | degraded {r['degraded_random_vonly']['auc']:.4f}"
              + (f" | Audio-only AUC (BioDeepAV): {r['audio_only_clean']['auc']:.4f}" if "audio_only_clean" in r else ""))
        for ds, v in r["per_dataset"].items():
            print(f"  {ds:10s} original AUC {v['clean']['auc']:.4f} acc {v['clean']['accuracy']:.4f} | "
                  f"degraded AUC {v['degraded']['auc']:.4f} acc {v['degraded']['accuracy']:.4f}")

    print("\n" + "=" * 90)
    print(" TABLE 2: PER-DEGRADATION TYPE (all degraded versions of the test videos)")
    print("=" * 90)
    if len(df_types):
        print(df_types.pivot(index="Degradation", columns="Model", values="AUC").round(4).add_prefix("AUC ").join(
            df_types.pivot(index="Degradation", columns="Model", values="Accuracy").round(4).add_prefix("Acc ")).to_string())

    print("\n" + "=" * 90)
    print(" TABLE 3: PER-SUBSET ACCURACY (original vs degraded)")
    print("=" * 90)
    print(df_subsets.round(4).to_string(index=False))

    # Random test videos
    if args.show_random > 0 and len(test_idx):
        rng_show = random.Random(args.seed + 99)
        show = sorted(rng_show.sample(range(len(test_idx)), min(args.show_random, len(test_idx))))
        last_tag = list(results)[-1]
        print("\n" + "=" * 90)
        print(f" RANDOM TEST VIDEOS ({len(show)}) - {results[last_tag]['title']}")
        print("=" * 90)
        for i in show:
            y = "FAKE" if y_test[i] == 1 else "REAL"
            pc = results[last_tag]["_p_clean"][i]
            pdg = results[last_tag]["_p_deg"][i]
            ok_c = "OK " if (pc >= 0.5) == y_test[i] else "ERR"
            line = f"  {pred_table['video'][i][:58]:58s} true {y} | original p_fake {pc:.3f} {ok_c}"
            if has_deg_version[i]:
                ok_d = "OK " if (pdg >= 0.5) == y_test[i] else "ERR"
                line += f" | {pred_table['degradation'][i]:>11s}/{pred_table['severity'][i]:<6s} p_fake {pdg:.3f} {ok_d}"
            print(line)

    # ---------------- 7. Explanation stability under degradation (Grad-CAM) ----------------
    xai_summary, xai_rows = {}, []
    if args.xai_samples > 0 and has_deg_version.any():
        cand = np.where(has_deg_version)[0]
        rng_x = np.random.default_rng(args.seed + 5)
        xai_pos = np.sort(rng_x.choice(cand, size=min(args.xai_samples, len(cand)), replace=False))
        print("\n" + "=" * 90)
        print(f" EXPLANATION STABILITY UNDER DEGRADATION (Grad-CAM, {len(xai_pos)} test videos)")
        print("=" * 90)

        def gradcam(model_x, clips, src):
            store = {}

            def fwd_hook(_m, _i, o):
                store["a"] = o
                o.register_hook(lambda g: store.__setitem__("g", g))

            h = model_x.video_net.features[-1].register_forward_hook(fwd_hook)
            try:
                model_x.eval()
                model_x.zero_grad(set_to_none=True)
                xa = torch.from_numpy(np.asarray(audio_cache[src])).to(device)
                am = torch.from_numpy(has_audio[src].astype(np.float32)).to(device)
                with torch.enable_grad():
                    logits = model_x(to_video_tensor(clips), xa, mode="av", a_mask=am)
                    logits[:, 1].sum().backward()
            finally:
                h.remove()
            A, G = store["a"].detach(), store["g"].detach()
            B, T = clips.shape[0], clips.shape[1]
            cam = torch.relu((G.mean((2, 3), keepdim=True) * A).sum(1))  # [B*T, h, w]
            cam = cam.view(B, T, *cam.shape[-2:]).mean(1, keepdim=True)
            cam = torch.nn.functional.interpolate(cam, size=(img_size, img_size), mode="bilinear", align_corners=False)[:, 0]
            flat = cam.view(B, -1)
            mn, mx = flat.min(1, keepdim=True)[0], flat.max(1, keepdim=True)[0]
            cam = ((flat - mn) / (mx - mn + 1e-8)).view(B, img_size, img_size)
            model_x.zero_grad(set_to_none=True)
            return cam.cpu().numpy(), torch.softmax(logits.detach().float(), 1)[:, 1].cpu().numpy()

        def cam_metrics(a, b, top=0.2):
            r = float(np.corrcoef(a.ravel(), b.ravel())[0, 1]) if a.std() > 1e-8 and b.std() > 1e-8 else float("nan")
            ta, tb = a >= np.quantile(a, 1 - top), b >= np.quantile(b, 1 - top)
            iou = float((ta & tb).sum() / max(1, (ta | tb).sum()))
            yy, xx = np.mgrid[0 : a.shape[0], 0 : a.shape[1]]

            def com(c):
                w = c.sum() + 1e-8
                return np.array([(yy * c).sum() / w, (xx * c).sum() / w])

            shift = float(np.linalg.norm(com(a) - com(b)) / a.shape[0])
            return r, iou, shift

        examples = {}
        for tag, title, m in models_to_test:
            rs, ious, shifts = [], [], []
            for k in range(0, len(xai_pos), 16):
                pos = xai_pos[k : k + 16]
                c_items = [test_clean_items[i] for i in pos]
                d_items = [test_deg_all[test_pick[i]] for i in pos]
                src = np.array([item_src(it) for it in c_items])
                c_clips = np.stack([get_full_clip(it)[::stride][:clip_len] for it in c_items])
                d_clips = np.stack([get_full_clip(it)[::stride][:clip_len] for it in d_items])
                cam_c, pc = gradcam(m, c_clips, src)
                cam_d, pd_ = gradcam(m, d_clips, src)
                for b, i in enumerate(pos):
                    r_, iou_, sh_ = cam_metrics(cam_c[b], cam_d[b])
                    same_pred = bool((pc[b] >= 0.5) == (pd_[b] >= 0.5))
                    xai_rows.append(
                        dict(model=tag, video=pred_table["video"][i], true_label=int(y_test[i]), degradation=pred_table["degradation"][i],
                             severity=pred_table["severity"][i], p_fake_original=float(pc[b]), p_fake_degraded=float(pd_[b]),
                             prediction_stable=same_pred, pearson_r=r_, top20_iou=iou_, center_shift=sh_,
                             explanation_unstable=bool(iou_ < 0.5))
                    )
                    if tag == models_to_test[-1][0] and len(examples) < 6:
                        examples[i] = (c_clips[b, clip_len // 2], cam_c[b], d_clips[b, clip_len // 2], cam_d[b], pc[b], pd_[b], r_, iou_)
            df_t = pd.DataFrame([x for x in xai_rows if x["model"] == tag])
            stable = df_t[df_t.prediction_stable]
            xai_summary[tag] = dict(
                n=int(len(df_t)),
                mean_pearson_r=float(df_t.pearson_r.mean()),
                mean_top20_iou=float(df_t.top20_iou.mean()),
                mean_center_shift=float(df_t.center_shift.mean()),
                prediction_stable_rate=float(df_t.prediction_stable.mean()),
                explanation_unstable_rate=float(df_t.explanation_unstable.mean()),
                unstable_explanation_given_stable_prediction=float(stable.explanation_unstable.mean()) if len(stable) else float("nan"),
                per_degradation=df_t.groupby("degradation")[["pearson_r", "top20_iou", "center_shift"]].mean().round(4).to_dict(orient="index"),
            )
            s = xai_summary[tag]
            print(f"  [{title}] Pearson r {s['mean_pearson_r']:.3f} | top-20% IoU {s['mean_top20_iou']:.3f} | centre shift {s['mean_center_shift']:.3f}"
                  f" | prediction stable {s['prediction_stable_rate'] * 100:.1f}% | explanation changed (IoU<0.5) while prediction stable: "
                  f"{s['unstable_explanation_given_stable_prediction'] * 100:.1f}%")

        df_xai = pd.DataFrame(xai_rows)
        df_xai.to_csv(report_dir / "xai_stability_30-9-26.csv", index=False)
        print("\n  Mean top-20% IoU per degradation type:")
        print(df_xai.pivot_table(index="degradation", columns="model", values="top20_iou", aggfunc="mean").round(4).to_string())

        # Plot: IoU per degradation per model
        piv = df_xai.pivot_table(index="degradation", columns="model", values="top20_iou", aggfunc="mean")
        fig, ax = plt.subplots(figsize=(10, 4.8))
        piv.plot(kind="bar", ax=ax, color=["#e67e22", "#2ecc71"][: piv.shape[1]])
        ax.set_ylabel("Top-20% IoU (original vs degraded Grad-CAM)")
        ax.set_title("Explanation stability under degradation (higher = more stable)", fontweight="bold")
        ax.set_ylim(0, 1)
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(report_dir / "xai_stability_by_degradation.png", dpi=130)
        plt.close()

        # Plot: example overlays
        if examples:
            fig, axes = plt.subplots(len(examples), 4, figsize=(10, 2.6 * len(examples)))
            axes = np.atleast_2d(axes)
            for row, (i, (fc, cc, fd, cd, pc, pdg, r_, iou_)) in enumerate(examples.items()):
                axes[row, 0].imshow(fc)
                axes[row, 0].set_title(f"original p={pc:.2f}", fontsize=8)
                axes[row, 1].imshow(fc)
                axes[row, 1].imshow(cc, cmap="jet", alpha=0.45)
                axes[row, 1].set_title("Grad-CAM original", fontsize=8)
                axes[row, 2].imshow(fd)
                axes[row, 2].set_title(f"{pred_table['degradation'][i]} p={pdg:.2f}", fontsize=8)
                axes[row, 3].imshow(fd)
                axes[row, 3].imshow(cd, cmap="jet", alpha=0.45)
                axes[row, 3].set_title(f"Grad-CAM degraded r={r_:.2f} IoU={iou_:.2f}", fontsize=8)
                for a in axes[row]:
                    a.axis("off")
            plt.tight_layout()
            plt.savefig(report_dir / "xai_examples_original_vs_degraded.png", dpi=120)
            plt.close()

    # ---------------- 8. Save CSVs, plots, JSON ----------------
    pd.DataFrame(pred_table).to_csv(report_dir / "test_predictions_30-9-26.csv", index=False)
    pd.DataFrame(deg_all_rows).to_csv(report_dir / "test_predictions_all_degraded_30-9-26.csv", index=False)
    df_comp.to_csv(report_dir / "summary_original_vs_degraded.csv")
    df_types.to_csv(report_dir / "summary_per_degradation.csv", index=False)
    df_subsets.to_csv(report_dir / "summary_per_subset.csv", index=False)

    # ROC curves
    fig, ax = plt.subplots(figsize=(8, 6.5))
    colors = {"stage1": ("#e67e22", "#c0392b"), "stage2": ("#27ae60", "#2980b9")}
    for tag, r in results.items():
        c1, c2 = colors.get(tag, ("#7f8c8d", "#34495e"))
        fpr, tpr, _ = roc_curve(y_test, r["_p_clean"])
        ax.plot(fpr, tpr, color=c1, lw=2.2, label=f"{r['title']} - original (AUC {r['clean']['auc']:.3f})")
        hv = has_deg_version
        if hv.any() and len(set(y_test[hv].tolist())) > 1:
            fpr, tpr, _ = roc_curve(y_test[hv], r["_p_deg"][hv])
            ax.plot(fpr, tpr, color=c2, lw=2.2, linestyle="--", label=f"{r['title']} - degraded (AUC {r['degraded_random']['auc']:.3f})")
    ax.plot([0, 1], [0, 1], color="#7f8c8d", lw=1.2, linestyle=":")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC: original vs degraded test videos", fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", fontsize=8.5)
    plt.tight_layout()
    plt.savefig(report_dir / "roc_original_vs_degraded.png", dpi=130)
    plt.close()

    # Per-degradation AUC bars
    if len(df_types):
        piv = df_types.pivot(index="Degradation", columns="Model", values="AUC")
        base = pd.DataFrame({t: [results[t]["clean"]["auc"]] for t in piv.columns}, index=["ORIGINAL"])
        piv = pd.concat([base, piv])
        fig, ax = plt.subplots(figsize=(11, 5))
        piv.plot(kind="bar", ax=ax, color=["#e67e22", "#2ecc71"][: piv.shape[1]])
        ax.set_ylabel("ROC-AUC")
        ax.set_ylim(0, 1.05)
        ax.set_title("Detection AUC per degradation type (Stage 1 vs Stage 2)", fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(report_dir / "auc_per_degradation.png", dpi=130)
        plt.close()

    # Confusion matrices
    fig, axes = plt.subplots(len(results), 2, figsize=(9.5, 4.0 * len(results)))
    axes = np.atleast_2d(axes)
    for row, (tag, r) in enumerate(results.items()):
        for col, (key, name) in enumerate((("clean", "original"), ("degraded_random", "degraded"))):
            mt = r[key]
            cm = np.array([[mt["tn"], mt["fp"]], [mt["fn"], mt["tp"]]])
            ax_cm = axes[row, col]
            ax_cm.imshow(cm, cmap="Blues")
            for (i, j), val in np.ndenumerate(cm):
                ax_cm.text(j, i, f"{val:,}", ha="center", va="center", fontsize=12, fontweight="bold",
                           color="white" if cm[i, j] > cm.max() / 2 else "black")
            ax_cm.set_xticks([0, 1])
            ax_cm.set_xticklabels(["Pred Real", "Pred Fake"])
            ax_cm.set_yticks([0, 1])
            ax_cm.set_yticklabels(["Real", "Fake"])
            ax_cm.set_title(f"{r['title']} - {name}", fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.savefig(report_dir / "confusion_original_vs_degraded.png", dpi=130)
    plt.close()

    # Training curves
    if histories:
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        offset = 0
        for stg, h in histories.items():
            h = pd.DataFrame(h)
            ep = h.epoch + offset
            ax[0].plot(ep, h.train_loss, marker="o", label=f"{stg} train loss")
            for col in [c for c in h.columns if c.startswith("val_") and c.endswith("_auc")]:
                ax[1].plot(ep, h[col], marker="o", label=f"{stg} {col}")
            offset += len(h)
        ax[0].set_xlabel("Epoch (stage 1 then stage 2)")
        ax[0].set_title("Training loss")
        ax[1].set_xlabel("Epoch (stage 1 then stage 2)")
        ax[1].set_title("Validation AUC")
        for a in ax:
            a.grid(True, linestyle="--", alpha=0.5)
            a.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(report_dir / "training_curves_30-9-26.png", dpi=130)
        plt.close()

    now = datetime.now()
    report = {
        "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
        "args": {k: (v if not isinstance(v, Path) else str(v)) for k, v in vars(args).items()},
        "inventory": df_inv.reset_index().to_dict(orient="records"),
        "manifests": manifest_summary,
        "split_sizes": dict(train=int(len(train_idx)), val=int(len(val_idx)), test=int(len(test_idx))),
        "training_time": {k: format_time(v) for k, v in train_times.items()},
        "results": {t: {k: v for k, v in r.items() if not k.startswith("_")} for t, r in results.items()},
        "explanation_stability": xai_summary,
        "summary_table": df_comp.reset_index().to_dict(orient="records"),
    }
    report_name = f"report_30-9-26_{now.strftime('%Y%m%d_%H%M')}.json"
    for target in (report_dir / report_name, report_dir / "latest_report_30-9-26.json"):
        with open(target, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))

    print("\nSaved reports & plots to:", report_dir)
    print(f"  * {report_name}, latest_report_30-9-26.json")
    print("  * test_predictions_30-9-26.csv, test_predictions_all_degraded_30-9-26.csv")
    print("  * summary_original_vs_degraded.csv, summary_per_degradation.csv, summary_per_subset.csv")
    print("  * roc_original_vs_degraded.png, auc_per_degradation.png, confusion_original_vs_degraded.png")
    if xai_rows:
        print("  * xai_stability_30-9-26.csv, xai_stability_by_degradation.png, xai_examples_original_vs_degraded.png")
    print("\n30-9-26 pipeline completed successfully!")


if __name__ == "__main__":
    main()
