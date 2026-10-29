import os
import re
import sys
import json
import time
import random
import warnings
import argparse
import subprocess
import shutil
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
    roc_curve,
    confusion_matrix,
)

warnings.filterwarnings("ignore")
cv2.setNumThreads(1)


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
    print(f" Pausing execution for {pause_sec // 60} minutes to let the GPU cool down...")
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
    sys.stdout.write(f"\r  [Cooling Down Complete] Cooled to {temp_after_str}. Resuming!                  \n\n")
    sys.stdout.flush()
    return int(time.time() - t_pause_start)


# ------------------------- CLI ARGUMENT PARSER -------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Unified Deepfake Video Detection: BioDeepAV & FaceForensics++ C23 under Mixed Degradation"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train_test", "test", "train"],
        default="train_test",
        help="Operational mode: 'train_test', 'test', or 'train'",
    )
    parser.add_argument(
        "--modality",
        type=str,
        choices=["v_only", "av", "a_only"],
        default="v_only",
        help="Model modality: 'v_only' (recommended for FaceForensics++), 'av', or 'a_only'",
    )
    # BioDeepAV Paths
    parser.add_argument("--biodeep-root", type=str, default=r"E:\Datasets\BioDeepAV")
    parser.add_argument(
        "--bio-real-manifest",
        type=str,
        default=r"E:\Datasets\BioDeepAV\real\videos\degraded_output\manifest.json",
    )
    parser.add_argument(
        "--bio-fake-manifest",
        type=str,
        default=r"E:\Datasets\BioDeepAV\fake\videos\degraded_output\manifest.json",
    )
    # FaceForensics++ C23 Paths
    parser.add_argument("--ff-root", type=str, default=r"E:\Datasets\FaceForensics++_C23")
    parser.add_argument(
        "--ff-dfd-manifest",
        type=str,
        default=r"E:\Datasets\FaceForensics++_C23\DeepFakeDetection\degraded_output\manifest.json",
    )
    # Dataset and Training Hyperparameters
    parser.add_argument(
        "--dataset-scale",
        type=str,
        choices=["full", "370", "test100", "custom"],
        default="full",
        help="Dataset scale for loading",
    )
    parser.add_argument("--n-real", type=int, default=None, help="Max real videos to load")
    parser.add_argument("--n-fake", type=int, default=None, help="Max fake videos to load")
    parser.add_argument(
        "--train-mode",
        type=str,
        choices=["clean_first_then_deg", "expanded", "random_mix"],
        default="clean_first_then_deg",
        help="Training strategy: 'clean_first_then_deg' (clean initial train followed by degraded fine-tuning), 'expanded', or 'random_mix'",
    )
    parser.add_argument("--clean-epochs", type=int, default=8, help="Epochs for clean video training stage")
    parser.add_argument("--deg-epochs", type=int, default=6, help="Epochs for degraded video training stage")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for AdamW")
    parser.add_argument("--weights", type=str, default=None, help="Path to checkpoint (.pt)")
    parser.add_argument("--work-dir", type=str, default="unified_deepfake_run", help="Output directory")
    parser.add_argument("--report-dir", type=str, default="reports_evaluation", help="Reports directory")
    parser.add_argument("--gpu-max-temp", type=int, default=72, help="Thermal threshold in Celsius")
    parser.add_argument("--gpu-pause", type=int, default=300, help="Pause duration in seconds")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


# ------------------------- DYNAMIC MANIFEST & DATASET PARSER -------------------------
class DatasetManager:
    @staticmethod
    def inspect_and_parse_manifest(manifest_path, dataset_name, label_int, orig_base, deg_base):
        m_path = Path(manifest_path)
        if not m_path.exists():
            return {}
        with open(m_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        parsed = {}
        files_list = data.get("files", data) if isinstance(data, dict) else data
        if not isinstance(files_list, list):
            return parsed

        for item in files_list:
            src = item.get("source") or item.get("original") or item.get("video")
            if not src:
                continue
            clean_p = orig_base / src
            outputs_dict = {}
            for out in item.get("outputs", []):
                deg_type = str(out.get("degradation", out.get("type", "unknown"))).lower()
                sev = str(out.get("severity", out.get("level", ""))).lower()
                key = f"{deg_type}_{sev}" if sev else deg_type
                out_name = out.get("output", out.get("path"))
                if out_name:
                    outputs_dict[key] = {
                        "path": deg_base / out_name,
                        "type": deg_type,
                        "severity": sev or "standard",
                    }
            parsed[f"{dataset_name}::{src}"] = {
                "dataset": dataset_name,
                "label": label_int,
                "clean_path": clean_p,
                "degradations": outputs_dict,
                "raw_id": Path(src).stem,
            }
        return parsed

    @staticmethod
    def scan_directory_clean_videos(video_dir, dataset_name, label_int, max_videos=None):
        vdir = Path(video_dir)
        if not vdir.exists():
            return {}
        valid_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
        files = sorted([f for f in vdir.glob("*.*") if f.suffix.lower() in valid_exts])
        if max_videos:
            files = files[:max_videos]

        parsed = {}
        for f in files:
            parsed[f"{dataset_name}::{f.name}"] = {
                "dataset": dataset_name,
                "label": label_int,
                "clean_path": f,
                "degradations": {},
                "raw_id": f.stem,
            }
        return parsed


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
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
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

    def forward(self, x_v=None, x_a=None, mode="v_only"):
        if mode == "v_only":
            B = x_v.size(0)
            f_v = self.video_net(x_v)
            f_a = torch.zeros(B, 128, device=x_v.device)
            return self.fusion_head(torch.cat([f_v, f_a], dim=1))
        elif mode == "a_only":
            B = x_a.size(0)
            f_v = torch.zeros(B, 256, device=x_a.device)
            f_a = self.audio_net(x_a)
            return self.fusion_head(torch.cat([f_v, f_a], dim=1))
        else:  # 'av'
            B = x_v.size(0)
            f_v = self.video_net(x_v)
            f_a = self.audio_net(x_a) if x_a is not None else torch.zeros(B, 128, device=x_v.device)
            return self.fusion_head(torch.cat([f_v, f_a], dim=1))


# ------------------------- DATA PREPROCESSING -------------------------
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
        return np.stack(seq)[..., ::-1].copy()
    except Exception:
        return None


def extract_audio_spectrogram(clean_video_path, n_samples=48000):
    audio_data = None
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
            timeout=8,
        )
        audio_data = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    except Exception:
        audio_data = np.zeros(n_samples, dtype=np.float32)

    if len(audio_data) < n_samples:
        if len(audio_data) > 0:
            reps = int(np.ceil(n_samples / len(audio_data)))
            audio_data = np.tile(audio_data, reps)[:n_samples]
        else:
            audio_data = np.zeros(n_samples, dtype=np.float32)
    else:
        audio_data = audio_data[:n_samples]

    with torch.no_grad():
        t = torch.from_numpy(audio_data)
        stft = torch.stft(
            t,
            n_fft=512,
            hop_length=256,
            win_length=512,
            window=torch.hann_window(512),
            return_complex=True,
        )
        mag = torch.log1p(stft.abs())
        spec = torch.nn.functional.interpolate(
            mag.unsqueeze(0).unsqueeze(0),
            size=(128, 128),
            mode="bilinear",
            align_corners=False,
        )
        spec = (spec - spec.mean()) / (spec.std() + 1e-6)
        spec = spec.clamp(-3.0, 3.0) / 3.0
        return spec.squeeze(0).numpy().astype(np.float32)


# ------------------------- CACHING HELPER -------------------------
def find_or_create_cache(tag_name, paths, tag, expected_shape, cache_dir, n_cpu):
    cache_path = cache_dir / f"cache_{tag_name}_{tag}.npy"
    if cache_path.exists():
        print(f"  -> Found cached video tensor: {cache_path.name}")
        return np.load(cache_path, mmap_mode="r")

    total = len(paths)
    print(f"\n[Extracting & Caching: {tag_name.upper()}] ({total} videos | {n_cpu} threads)")
    mmap_arr = np.lib.format.open_memmap(cache_path, mode="w+", dtype=np.uint8, shape=expected_shape)
    t0 = time.time()
    last_print = 0

    with ThreadPoolExecutor(max_workers=n_cpu) as ex:
        futures = {ex.submit(extract_video_clip, p, expected_shape[1], expected_shape[2]): i for i, p in enumerate(paths)}
        completed = 0
        for fut in as_completed(futures):
            i = futures[fut]
            try:
                res = fut.result()
                mmap_arr[i] = res if (res is not None and res.shape == expected_shape[1:]) else 0
            except Exception:
                mmap_arr[i] = 0
            completed += 1
            now = time.time()
            if now - last_print >= 0.20 or completed == total:
                elapsed = now - t0
                rate = completed / max(0.001, elapsed)
                eta = (total - completed) / max(0.001, rate)
                bar = progress_bar(completed, total, bar_len=20)
                sys.stdout.write(
                    f"\r  {tag_name:12s}: {bar} ({completed}/{total}) | {rate:5.1f} vids/s | ETA: {format_time(eta)}   "
                )
                sys.stdout.flush()
                last_print = now

    sys.stdout.write("\n")
    sys.stdout.flush()
    mmap_arr.flush()
    del mmap_arr
    return np.load(cache_path, mmap_mode="r")


# ------------------------- METRICS -------------------------
def compute_metrics(y_true, p, thr=0.5):
    pred = (p >= thr).astype(int)
    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    auc = roc_auc_score(y_true, p) if len(set(y_true)) > 1 else float("nan")
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return dict(
        accuracy=float(accuracy_score(y_true, pred)),
        balanced_acc=float(balanced_accuracy_score(y_true, pred)),
        precision=float(precision_score(y_true, pred, zero_division=0)),
        recall=float(recall_score(y_true, pred, zero_division=0)),
        specificity=float(specificity),
        f1=float(f1_score(y_true, pred, zero_division=0)),
        auc=float(auc),
        tn=tn, fp=fp, fn=fn, tp=tp, threshold=float(thr),
    )


# ------------------------- MAIN WORKFLOW -------------------------
def main():
    args = parse_args()
    script_path = Path(__file__).resolve()

    # Step 1: Automatic Script Backup
    timestamp_backup = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_file = script_path.parent / f"{script_path.stem}_backup_{timestamp_backup}.py"
    shutil.copy2(script_path, backup_file)
    print(f"[Backup Created] Initialized backup: {backup_file.name}")

    work_dir = Path(args.work_dir)
    report_dir = Path(args.report_dir)
    cache_dir = work_dir / "cache_tensors"
    work_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    n_frames, img_size, clip_len, batch_size = 16, 96, 8, args.batch_size
    n_cpu = os.cpu_count() or 8

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "=" * 82)
    print(" UNIFIED DEEPFAKE PIPELINE: BIODEEPAV & FACEFORENSICS++ C23")
    print(f"  * Mode              : {args.mode.upper()}")
    print(f"  * Modality          : {args.modality.upper()}")
    print(f"  * Device            : {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    print(f"  * Training Regime   : {args.train_mode.upper()}")
    print("=" * 82)

    # Step 2: Multi-Dataset Ingestion
    all_dataset_items = {}

    # 1. BioDeepAV (Real & Fake with Manifests)
    bio_root = Path(args.biodeep_root)
    if bio_root.exists():
        real_m = Path(args.bio_real_manifest)
        fake_m = Path(args.bio_fake_manifest)
        all_dataset_items.update(
            DatasetManager.inspect_and_parse_manifest(
                real_m, "BioDeepAV_Real", 0, real_m.parent.parent, real_m.parent
            )
        )
        all_dataset_items.update(
            DatasetManager.inspect_and_parse_manifest(
                fake_m, "BioDeepAV_Fake", 1, fake_m.parent.parent, fake_m.parent
            )
        )

    # 2. FaceForensics++ C23 Categories
    ff_root = Path(args.ff_root)
    if ff_root.exists():
        # Real Videos
        ff_orig_dir = ff_root / "original"
        all_dataset_items.update(
            DatasetManager.scan_directory_clean_videos(ff_orig_dir, "FF++_Original", 0)
        )

        # Fake Manipulation Categories
        ff_fake_categories = [
            "NeuralTextures",
            "FaceSwap",
            "FaceShifter",
            "Face2Face",
            "Deepfakes",
        ]
        for cat in ff_fake_categories:
            cat_dir = ff_root / cat
            all_dataset_items.update(
                DatasetManager.scan_directory_clean_videos(cat_dir, f"FF++_{cat}", 1)
            )

        # DeepFakeDetection (Manifest-governed)
        ff_dfd_m = Path(args.ff_dfd_manifest)
        if ff_dfd_m.exists():
            all_dataset_items.update(
                DatasetManager.inspect_and_parse_manifest(
                    ff_dfd_m, "FF++_DeepFakeDetection", 1, ff_dfd_m.parent.parent, ff_dfd_m.parent
                )
            )
        else:
            # Fallback scan if manifest is not yet generated
            all_dataset_items.update(
                DatasetManager.scan_directory_clean_videos(
                    ff_root / "DeepFakeDetection", "FF++_DeepFakeDetection", 1
                )
            )

    # Filter to existing clean video files
    existing_items = {k: v for k, v in all_dataset_items.items() if v["clean_path"].exists()}
    if not existing_items:
        raise RuntimeError("No valid video files found at specified dataset paths.")

    # Ingestion Summary
    source_keys = sorted(list(existing_items.keys()))
    labels = np.array([existing_items[k]["label"] for k in source_keys])
    real_keys = [k for k in source_keys if existing_items[k]["label"] == 0]
    fake_keys = [k for k in source_keys if existing_items[k]["label"] == 1]

    print(f"\n[Dataset Inventory Summary]")
    print(f"  * Total Original Real Videos: {len(real_keys):,}")
    print(f"  * Total Original Fake Videos: {len(fake_keys):,}")
    print(f"  * Total Unified Videos Loaded: {len(source_keys):,}")

    # Subsample if requested
    if args.dataset_scale == "370":
        rng_sub = random.Random(args.seed)
        real_keys = rng_sub.sample(real_keys, min(185, len(real_keys)))
        fake_keys = rng_sub.sample(fake_keys, min(185, len(fake_keys)))
        source_keys = sorted(real_keys + fake_keys)
    elif args.dataset_scale == "test100":
        rng_sub = random.Random(args.seed)
        real_keys = rng_sub.sample(real_keys, min(args.n_real or 50, len(real_keys)))
        fake_keys = rng_sub.sample(fake_keys, min(args.n_fake or 50, len(fake_keys)))
        source_keys = sorted(real_keys + fake_keys)

    labels = np.array([existing_items[k]["label"] for k in source_keys])
    name_to_idx = {name: i for i, name in enumerate(source_keys)}

    # Step 3: Leakage-Free Splitting by Video Identity
    def get_identity_group(key):
        item = existing_items[key]
        raw_id = item["raw_id"]
        # FaceForensics pattern: "000_003" or "000"
        m_ff = re.match(r"^(\d{3})(?:_\d{3})?", raw_id)
        if m_ff:
            return f"FF_{m_ff.group(1)}"
        # BioDeepAV pattern
        m_bio = re.match(r"^(.*)_\d{4}_S\d+_E\d+", raw_id)
        if m_bio:
            return f"BIO_{m_bio.group(1)}"
        return f"GEN_{raw_id.split('_')[0]}"

    groups = {}
    for k in source_keys:
        gid = get_identity_group(k)
        groups.setdefault(gid, []).append(k)

    rng_grp = random.Random(args.seed)
    grp_list = list(groups.keys())
    rng_grp.shuffle(grp_list)

    train_names, val_names, test_names = [], [], []
    target_test = int(len(source_keys) * 0.20)
    target_val = int(len(source_keys) * 0.10)

    for g in grp_list:
        members = groups[g]
        if len(test_names) + len(members) <= target_test:
            test_names.extend(members)
        elif len(val_names) + len(members) <= target_val:
            val_names.extend(members)
        else:
            train_names.extend(members)

    train_idx = np.array([name_to_idx[n] for n in train_names], dtype=int)
    val_idx = np.array([name_to_idx[n] for n in val_names], dtype=int)
    test_idx = np.array([name_to_idx[n] for n in test_names], dtype=int)

    # Verify Leakage Prevention
    assert not (set(train_idx) & set(test_idx)), "Data Leakage: Overlap between Train and Test!"
    assert not (set(val_idx) & set(test_idx)), "Data Leakage: Overlap between Val and Test!"

    # In accordance with requirements: 80% Original/Clean videos, 20% Degraded videos in test set
    n_test_total = len(test_idx)
    n_test_deg = max(1, int(round(0.20 * n_test_total)))
    n_test_clean = n_test_total - n_test_deg

    rng_test = random.Random(args.seed + 99)
    test_shuffled = list(test_idx)
    rng_test.shuffle(test_shuffled)
    test_clean_sub_idx = np.array(test_shuffled[:n_test_clean], dtype=int)
    test_deg_sub_idx = np.array(test_shuffled[n_test_clean:], dtype=int)

    print(f"\n[Identity-Partitioned Allocation]")
    print(f"  * Training Set (70%)     : {len(train_idx):,} original videos")
    print(f"  * Validation Set (10%)   : {len(val_idx):,} original videos")
    print(f"  * Final Held-Out Test Set: {len(test_idx):,} original videos")
    print(f"    - Clean Test Evaluation (80%): {len(test_clean_sub_idx):,} videos")
    print(f"    - Degraded Test Evaluation (20%): {len(test_deg_sub_idx):,} videos")

    # Step 4: Caching
    tag = f"T{n_frames}_S{img_size}_n{len(source_keys)}"
    clean_paths = [existing_items[k]["clean_path"] for k in source_keys]
    video_shape = (len(source_keys), n_frames, img_size, img_size, 3)
    clean_cache = find_or_create_cache("clean_baseline", clean_paths, tag, video_shape, cache_dir, n_cpu)

    # Audio Spectrogram Cache (if modality uses audio)
    audio_cache = None
    if args.modality in ["av", "a_only"]:
        audio_cache_path = cache_dir / f"cache_audio_{tag}.npy"
        audio_shape = (len(source_keys), 1, 128, 128)
        if audio_cache_path.exists():
            audio_cache = np.load(audio_cache_path, mmap_mode="r")
        else:
            print(f"\n[Extracting Audio Features] ({len(source_keys)} items | {n_cpu} threads)")
            mmap_aud = np.lib.format.open_memmap(audio_cache_path, mode="w+", dtype=np.float32, shape=audio_shape)
            with ThreadPoolExecutor(max_workers=n_cpu) as ex:
                futures = {ex.submit(extract_audio_spectrogram, clean_paths[i]): i for i in range(len(source_keys))}
                for fut in as_completed(futures):
                    mmap_aud[futures[fut]] = fut.result()
            mmap_aud.flush()
            del mmap_aud
            audio_cache = np.load(audio_cache_path, mmap_mode="r")

    # Step 5: Model Initialization
    def to_video_tensor(arr):
        t = torch.from_numpy(np.ascontiguousarray(arr)).permute(0, 1, 4, 2, 3).float().div_(127.5).sub_(1.0)
        return t.to(device)

    model = AudioVisualDeepFakeCNN().to(device)
    best_model_path = work_dir / "best_model_unified.pt"

    if args.weights and Path(args.weights).exists():
        model.load_state_dict(torch.load(args.weights, map_location=device))
        print(f"\n[Loaded Checkpoint]: {args.weights}")

    # Step 6: Training (Clean Stage then Degraded Stage)
    if args.mode in ["train", "train_test"]:
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

        def run_epoch(indices, is_degraded_stage, epoch_num, total_eps):
            model.train()
            rng_np = np.random.default_rng(args.seed + epoch_num)
            order = rng_np.permutation(len(indices))
            steps = int(np.ceil(len(indices) / batch_size))
            total_loss = 0.0
            t0 = time.time()
            last_p = 0

            for s in range(steps):
                b_idx = indices[order[s * batch_size : (s + 1) * batch_size]]
                if len(b_idx) < 2:
                    continue

                # Batch extraction
                raw_v = np.empty((len(b_idx), clip_len, img_size, img_size, 3), dtype=np.uint8)
                for b_i, v_i in enumerate(b_idx):
                    sel = np.sort(rng_np.choice(n_frames, clip_len, replace=False))
                    clip = clean_cache[v_i, sel].copy()

                    # On-the-fly degradation augmentation if degraded stage
                    if is_degraded_stage and rng_np.random() < 0.65:
                        d_choice = rng_np.choice(["blur", "noise", "contrast", "brightness"])
                        if d_choice == "blur":
                            clip = cv2.GaussianBlur(clip.reshape(-1, img_size, 3), (5, 5), 1.6).reshape(clip_len, img_size, img_size, 3)
                        elif d_choice == "noise":
                            noise = rng_np.normal(0, 15, clip.shape).astype(np.int16)
                            clip = np.clip(clip.astype(np.int16) + noise, 0, 255).astype(np.uint8)
                        elif d_choice == "contrast":
                            clip = np.clip(clip.astype(np.float32) * 0.75, 0, 255).astype(np.uint8)
                        elif d_choice == "brightness":
                            clip = np.clip(clip.astype(np.float32) - 30, 0, 255).astype(np.uint8)

                    raw_v[b_i] = clip

                xv = to_video_tensor(raw_v)
                xa = torch.from_numpy(audio_cache[b_idx]).to(device) if audio_cache is not None else None
                yb = torch.from_numpy(labels[b_idx]).to(device)

                optimizer.zero_grad()
                logits = model(xv, xa, mode=args.modality)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

                now = time.time()
                if now - last_p >= 0.20 or (s + 1) == steps:
                    bar = progress_bar(s + 1, steps, bar_len=16)
                    sys.stdout.write(
                        f"\r  [{'DEGRADED' if is_degraded_stage else 'CLEAN'} Stage Ep {epoch_num}/{total_eps}] {bar} "
                        f"loss: {total_loss / (s + 1):.4f}   "
                    )
                    sys.stdout.flush()
                    last_p = now

            sys.stdout.write("\n")

        print("\n" + "=" * 82)
        print(" STAGE 1: TRAINING ON ORIGINAL / CLEAN VIDEOS")
        print("=" * 82)
        for ep in range(1, args.clean_epochs + 1):
            run_epoch(train_idx, is_degraded_stage=False, epoch_num=ep, total_eps=args.clean_epochs)

        print("\n" + "=" * 82)
        print(" STAGE 2: ROBUSTNESS TRAINING ON DEGRADED VIDEOS")
        print("=" * 82)
        for ep in range(1, args.deg_epochs + 1):
            run_epoch(train_idx, is_degraded_stage=True, epoch_num=ep, total_eps=args.deg_epochs)

        torch.save(model.state_dict(), best_model_path)
        print(f"\n[Training Completed] Checkpoint saved: {best_model_path.name}")

    # Step 7: Final Standard Evaluation (80% Clean vs 20% Degraded)
    print("\n" + "=" * 82)
    print(" HELD-OUT TEST EVALUATION (80% CLEAN / 20% DEGRADED)")
    print("=" * 82)

    if best_model_path.exists():
        model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    @torch.no_grad()
    def evaluate_indices(indices, apply_degradation=False):
        probs = []
        stride = max(1, n_frames // clip_len)
        rng_eval = np.random.default_rng(args.seed)

        for i in range(0, len(indices), batch_size):
            chunk = indices[i : i + batch_size]
            xb_raw = clean_cache[chunk][:, :clip_len].copy()

            if apply_degradation:
                # Apply deterministic test degradation to evaluate robustness
                for idx in range(len(chunk)):
                    xb_raw[idx] = cv2.GaussianBlur(xb_raw[idx].reshape(-1, img_size, 3), (7, 7), 2.5).reshape(clip_len, img_size, img_size, 3)

            xv = to_video_tensor(xb_raw)
            xa = torch.from_numpy(audio_cache[chunk]).to(device) if audio_cache is not None else None
            logits = model(xv, xa, mode=args.modality)
            probs.append(torch.softmax(logits, 1)[:, 1].cpu().numpy())

        return np.concatenate(probs)

    # 1. Evaluate 80% Clean portion
    p_clean = evaluate_indices(test_clean_sub_idx, apply_degradation=False)
    metrics_clean = compute_metrics(labels[test_clean_sub_idx], p_clean)

    # 2. Evaluate 20% Degraded portion
    p_deg = evaluate_indices(test_deg_sub_idx, apply_degradation=True)
    metrics_deg = compute_metrics(labels[test_deg_sub_idx], p_deg)

    # 3. Overall Test Set
    p_all = np.concatenate([p_clean, p_deg])
    y_all = np.concatenate([labels[test_clean_sub_idx], labels[test_deg_sub_idx]])
    metrics_overall = compute_metrics(y_all, p_all)

    # Print Summary Table
    df_eval = pd.DataFrame([
        {
            "Condition": "Original/Clean (80%)",
            "Samples": len(test_clean_sub_idx),
            "Accuracy": f"{metrics_clean['accuracy']*100:.2f}%",
            "ROC-AUC": f"{metrics_clean['auc']:.4f}",
            "F1-Score": f"{metrics_clean['f1']:.4f}",
            "Precision": f"{metrics_clean['precision']*100:.2f}%",
            "Recall": f"{metrics_clean['recall']*100:.2f}%",
        },
        {
            "Condition": "Degraded (20%)",
            "Samples": len(test_deg_sub_idx),
            "Accuracy": f"{metrics_deg['accuracy']*100:.2f}%",
            "ROC-AUC": f"{metrics_deg['auc']:.4f}",
            "F1-Score": f"{metrics_deg['f1']:.4f}",
            "Precision": f"{metrics_deg['precision']*100:.2f}%",
            "Recall": f"{metrics_deg['recall']*100:.2f}%",
        },
        {
            "Condition": "Overall Final Test (100%)",
            "Samples": len(y_all),
            "Accuracy": f"{metrics_overall['accuracy']*100:.2f}%",
            "ROC-AUC": f"{metrics_overall['auc']:.4f}",
            "F1-Score": f"{metrics_overall['f1']:.4f}",
            "Precision": f"{metrics_overall['precision']*100:.2f}%",
            "Recall": f"{metrics_overall['recall']*100:.2f}%",
        },
    ]).set_index("Condition")

    print(df_eval.to_string())
    print("-" * 82)
    print(f"Robustness Drop (Clean -> Degraded):")
    print(f"  * Delta AUC : {metrics_deg['auc'] - metrics_clean['auc']:+.4f}")
    print(f"  * Delta Acc : {(metrics_deg['accuracy'] - metrics_clean['accuracy'])*100:+.2f}%")
    print("=" * 82)

    # Step 8: Save Structured JSON Report & Visualizations
    report_file = report_dir / f"evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump({
            "metrics_clean_80pct": metrics_clean,
            "metrics_degraded_20pct": metrics_deg,
            "metrics_overall_100pct": metrics_overall,
            "delta_auc": metrics_deg['auc'] - metrics_clean['auc'],
            "delta_acc": metrics_deg['accuracy'] - metrics_clean['accuracy'],
        }, f, indent=2)

    # ROC Plot
    plt.figure(figsize=(7, 6))
    fpr_c, tpr_c, _ = roc_curve(labels[test_clean_sub_idx], p_clean)
    fpr_d, tpr_d, _ = roc_curve(labels[test_deg_sub_idx], p_deg)
    plt.plot(fpr_c, tpr_c, color="#27ae60", lw=2.5, label=f"Original/Clean 80% (AUC = {metrics_clean['auc']:.3f})")
    plt.plot(fpr_d, tpr_d, color="#e74c3c", lw=2.5, label=f"Degraded 20% (AUC = {metrics_deg['auc']:.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Deepfake Detection: Original vs Degraded Conditions")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(report_dir / "clean_vs_degraded_roc.png", dpi=130)
    plt.close()

    print(f"\n[Reports & Figures Saved] -> {report_dir}")


if __name__ == "__main__":
    main()