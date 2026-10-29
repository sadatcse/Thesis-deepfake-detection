import os
import re
import sys
import json
import time
import random
import warnings
import argparse
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
        description="BioDeepAV Multimodal Audio-Visual Pipeline for Mixed Degradation Training & Testing"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train_test", "test", "train"],
        default="train_test",
        help="Operational mode: 'train_test' (train + evaluate), 'test' (evaluate existing weights), or 'train'",
    )
    parser.add_argument(
        "--real-manifest",
        type=str,
        default=r"E:\Datasets\BioDeepAV\real\videos\degraded_output\manifest.json",
        help="Path to real videos degraded output manifest.json",
    )
    parser.add_argument(
        "--fake-manifest",
        type=str,
        default=r"E:\Datasets\BioDeepAV\fake\videos\degraded_output\manifest.json",
        help="Path to fake videos degraded output manifest.json",
    )
    parser.add_argument(
        "--dataset-scale",
        type=str,
        choices=["full", "370", "test100", "custom"],
        default="full",
        help="Dataset scale: 'full' (all 3,691 videos / 36,910 clean+deg), '370' (185 real + 185 fake), 'test100' (100 real + 100 fake), or 'custom'",
    )
    parser.add_argument("--n-real", type=int, default=None, help="Custom number of real videos to load")
    parser.add_argument("--n-fake", type=int, default=None, help="Custom number of fake videos to load")
    parser.add_argument(
        "--train-mode",
        type=str,
        choices=["expanded", "random_mix", "compound_mix"],
        default="expanded",
        help="Mixed degradation training regime: 'expanded' (all clean+degraded variants) or 'random_mix'",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to pre-trained model checkpoint (.pt) to load for testing or resuming",
    )
    parser.add_argument("--epochs", type=int, default=12, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for AdamW")
    parser.add_argument("--data-root", type=str, default=None, help="Path to BioDeepAV dataset root")
    parser.add_argument("--work-dir", type=str, default=None, help="Output directory for caches and model")
    parser.add_argument("--report-dir", type=str, default=None, help="Output directory for test reports")
    parser.add_argument("--gpu-max-temp", type=int, default=72, help="GPU thermal guard trigger temperature")
    parser.add_argument("--gpu-pause", type=int, default=300, help="Pause seconds when GPU exceeds thermal limit")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
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


# ------------------------- AUDIO & VIDEO EXTRACTION -------------------------
def extract_audio_spectrogram(source_name, label_int, clean_video_path, data_root, n_samples=48000):
    audio_data = None
    audio_dir = data_root / ("real" if label_int == 0 else "fake") / "audios"
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
    except Exception as e:
        print(f"skip {Path(path).name}: {e}")
        return None


# ------------------------- CACHING HELPER -------------------------
def find_or_create_cache(
    deg_name, paths, tag, expected_shape, cache_dir, search_dirs, n_cpu
):
    target_cache_path = cache_dir / f"cache_{deg_name}_{tag}.npy"

    # 1. Check target dir
    if target_cache_path.exists():
        print(f"  -> Found cached {deg_name.upper()}: {target_cache_path.name}")
        return np.load(target_cache_path, mmap_mode="r")

    # 2. Check alternative cache dirs
    for sdir in search_dirs:
        candidate = sdir / f"cache_{deg_name}_{tag}.npy"
        if candidate.exists():
            try:
                cand_arr = np.load(candidate, mmap_mode="r")
                if cand_arr.shape == expected_shape and cand_arr.dtype == np.uint8:
                    print(f"  -> Reusing existing {deg_name.upper()} cache from {sdir.name}: {candidate.name}")
                    return cand_arr
            except Exception:
                pass

    # 3. Extract fresh
    total = len(paths)
    print(f"\n[Extracting & Caching {deg_name.upper()}] ({total} videos | {n_cpu} threads)")
    mmap_arr = np.lib.format.open_memmap(target_cache_path, mode="w+", dtype=np.uint8, shape=expected_shape)
    t0 = time.time()
    last_print = 0

    with ThreadPoolExecutor(max_workers=n_cpu) as ex:
        futures = {ex.submit(extract_video_clip, p, expected_shape[1], expected_shape[2]): i for i, p in enumerate(paths)}
        completed = 0
        for fut in as_completed(futures):
            i = futures[fut]
            try:
                res = fut.result()
                if res is not None and res.shape == expected_shape[1:]:
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
    print(f"  Saved memory-mapped cache: {target_cache_path.name} ({target_cache_path.stat().st_size / 1e9:.2f} GB)")
    return np.load(target_cache_path, mmap_mode="r")


# ------------------------- METRICS COMPUTATION -------------------------
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
        tn=tn,
        fp=fp,
        fn=fn,
        tp=tp,
        threshold=float(thr),
    )


# ------------------------- MAIN WORKFLOW -------------------------
def main():
    args = parse_args()
    script_dir = Path(__file__).resolve().parent

    # Configuration setup
    default_root = (
        script_dir.parent / "BioDeepAV"
        if (script_dir.parent / "BioDeepAV").exists()
        else Path(r"E:\Datasets\BioDeepAV")
    )
    data_root = Path(args.data_root) if args.data_root else default_root
    work_dir = Path(args.work_dir) if args.work_dir else (script_dir / "biodeep_run_mixed_av")
    report_dir = Path(args.report_dir) if args.report_dir else (script_dir / "test")
    cache_dir = work_dir / "cache_degraded"

    work_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / "test").mkdir(parents=True, exist_ok=True)

    n_frames = 16
    img_size = 96
    clip_len = 8
    batch_size = args.batch_size
    n_cpu = os.cpu_count() or 8

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(n_cpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 82)
    print(" BIODEEP-AV MULTIMODAL AUDIO-VISUAL PIPELINE: MIXED DEGRADATION")
    print(f"  * Mode              : {args.mode.upper()}")
    print(f"  * Device            : {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    print(f"  * Dataset Root      : {data_root}")
    print(f"  * Work Directory    : {work_dir}")
    print(f"  * Dataset Scale     : {args.dataset_scale.upper()}")
    print(f"  * Training Regime   : {args.train_mode.upper()}")
    print("=" * 82)

    # 1. Manifest Discovery & Ingestion
    real_m_path = Path(args.real_manifest) if args.real_manifest else (data_root / "real" / "videos" / "degraded_output" / "manifest.json")
    fake_m_path = Path(args.fake_manifest) if args.fake_manifest else (data_root / "fake" / "videos" / "degraded_output" / "manifest.json")

    manifest_info = {}
    deg_names = set()
    for label_str, label_int, m_path in [("real", 0, real_m_path), ("fake", 1, fake_m_path)]:
        if not m_path.exists():
            raise FileNotFoundError(f"Manifest not found: {m_path}")
        deg_base = m_path.parent
        orig_base = deg_base.parent

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
    active_variants = ["clean"] + sorted_deg

    # 2. Sample Pool Discovery (Valid Videos With All Degradations)
    all_real_pool = sorted([
        k for k, v in manifest_info.items()
        if v["label"] == 0 and all(d in v["degradations"] for d in sorted_deg)
    ])
    all_fake_pool = sorted([
        k for k, v in manifest_info.items()
        if v["label"] == 1 and all(d in v["degradations"] for d in sorted_deg)
    ])

    # 3. Splits Selection (80% Train / 20% Held-Out Test, Leakage-Free Grouped Allocation)
    def yt_group(name):
        mm = re.match(r"^(.*)_\d{4}_S\d+_E\d+", name)
        return mm.group(1) if mm else name

    split_file = None
    search_split_files = [
        work_dir / "split_mixed_av.json",
        Path(r"E:\Datasets\Code 2\biodeep_run_cnngru_mixed\split_mixed_cnngru.json") if args.dataset_scale == "full" else None,
        script_dir / "biodeep_run_av_test100" / "split_av_test100.json" if args.dataset_scale == "test100" else None,
    ]
    for sp in search_split_files:
        if sp and sp.exists():
            split_file = sp
            break

    if args.dataset_scale == "full":
        # Full dataset: 3,691 paired source videos
        target_total_real = len(all_real_pool)
        target_total_fake = len(all_fake_pool)
    elif args.dataset_scale == "370":
        target_total_real = 185
        target_total_fake = 185
    elif args.dataset_scale == "test100":
        target_total_real = args.n_real or 100
        target_total_fake = args.n_fake or 100
    else:  # custom
        target_total_real = args.n_real or 100
        target_total_fake = args.n_fake or 100

    target_source_count = target_total_real + target_total_fake
    test_ratio = 0.20
    val_ratio = 0.10
    total_test_target = int(round(target_source_count * test_ratio))
    per_deg = max(2, (total_test_target // len(sorted_deg)) // 2 * 2)
    mixed_test_spec = {d: per_deg for d in sorted_deg}
    n_test_total = sum(mixed_test_spec.values())
    n_test_real = sum(count // 2 for count in mixed_test_spec.values())
    n_test_fake = sum(count // 2 for count in mixed_test_spec.values())

    real_groups = {}
    for rf in all_real_pool:
        g = yt_group(rf)
        real_groups.setdefault(g, []).append(rf)

    rng_grp = random.Random(args.seed)
    grp_keys = list(real_groups.keys())
    rng_grp.shuffle(grp_keys)

    test_real_names, val_real_names, train_real_names = [], [], []
    target_val_real = int((target_total_real - n_test_real) * val_ratio)
    target_train_real = target_total_real - n_test_real - target_val_real

    for g in grp_keys:
        members = real_groups[g]
        if len(test_real_names) + len(members) <= n_test_real:
            test_real_names.extend(members)
        elif len(val_real_names) + len(members) <= target_val_real:
            val_real_names.extend(members)
        elif args.dataset_scale == "full" or (len(train_real_names) + len(members) <= target_train_real):
            train_real_names.extend(members)

    rng_fake = random.Random(args.seed + 1)
    shuffled_fake = list(all_fake_pool)
    rng_fake.shuffle(shuffled_fake)

    target_val_fake = len(val_real_names)
    target_train_fake = (target_total_fake - n_test_fake - target_val_fake) if args.dataset_scale == "full" else len(train_real_names)

    test_fake_names = shuffled_fake[:n_test_fake]
    val_fake_names = shuffled_fake[n_test_fake : n_test_fake + target_val_fake]
    train_fake_names = shuffled_fake[n_test_fake + target_val_fake : n_test_fake + target_val_fake + target_train_fake]

    selected_real = train_real_names + val_real_names + test_real_names
    selected_fake = train_fake_names + val_fake_names + test_fake_names
    all_source_names = np.array(selected_real + selected_fake)
    all_labels = np.array([0] * len(selected_real) + [1] * len(selected_fake))
    name_to_idx = {name: i for i, name in enumerate(all_source_names)}

    train_idx = np.array(sorted([name_to_idx[n] for n in train_real_names + train_fake_names]), dtype=int)
    val_idx = np.array(sorted([name_to_idx[n] for n in val_real_names + val_fake_names]), dtype=int)
    test_idx = np.array(sorted([name_to_idx[n] for n in test_real_names + test_fake_names]), dtype=int)

    assert not (set(train_idx) & set(test_idx)), "Data leakage: train & test overlap!"
    assert not (set(val_idx) & set(test_idx)), "Data leakage: val & test overlap!"
    assert not (set(train_idx) & set(val_idx)), "Data leakage: train & val overlap!"

    json.dump(
        {
            "train": all_source_names[train_idx].tolist(),
            "val": all_source_names[val_idx].tolist(),
            "test": all_source_names[test_idx].tolist(),
        },
        open(work_dir / "split_mixed_av.json", "w"),
        indent=1,
    )

    n_videos = len(all_source_names)
    total_variants = len(active_variants)
    n_deg_types = len(sorted_deg)

    # 4. Prominent Startup Banner: Total Clean + Degraded Video Statistics & Partitions
    n_clean_real = int((all_labels == 0).sum())
    n_clean_fake = int((all_labels == 1).sum())
    n_clean_total = n_videos
    n_deg_total = n_clean_total * n_deg_types
    n_grand_total = n_clean_total + n_deg_total

    n_train_pool = len(train_idx) + len(val_idx)
    n_train_pool_deg = n_train_pool * n_deg_types
    n_test_pool = len(test_idx)
    n_test_pool_deg = n_test_pool * n_deg_types

    print("\n" + "=" * 82)
    print(" BIODEEPAV MULTIMODAL AUDIO-VISUAL PIPELINE: MIXED DEGRADATION")
    print("=" * 82)
    print(" [DATASET INVENTORY & DISCOVERY]")
    print(f"  * Real Video Manifest        : {real_m_path}")
    print(f"  * Fake Video Manifest        : {fake_m_path}")
    print(f"  * Active Degradation Types ({n_deg_types}): {', '.join(sorted_deg)}")
    print("\n [TOTAL VIDEO COUNTS: CLEAN + DEGRADATION]")
    print(f"  * Clean Real Videos          : {n_clean_real:,}")
    print(f"  * Clean Fake Videos          : {n_clean_fake:,}")
    print(f"  * Total Clean Videos         : {n_clean_total:,}")
    print(f"  * Total Degraded Videos      : {n_deg_total:,} ({n_deg_types} degraded versions per video)")
    print(f"  * TOTAL ALL VIDEOS (Clean+Deg): {n_grand_total:,} video files")
    print("\n [PARTITION ALLOCATION (80% Train / 20% Test)]")
    print(f"  * Training Set Pool (80%):")
    print(f"      - Clean Videos           : {n_train_pool:,} (Real: {sum(all_labels[train_idx]==0)+sum(all_labels[val_idx]==0):,} | Fake: {sum(all_labels[train_idx]==1)+sum(all_labels[val_idx]==1):,})")
    print(f"      - Degraded Videos        : {n_train_pool_deg:,} ({n_deg_types} variants per video)")
    print(f"      - Total Train (Clean+Deg): {n_train_pool + n_train_pool_deg:,} videos")
    print(f"        [Active Train: {len(train_idx):,} clean + {len(train_idx)*n_deg_types:,} deg = {len(train_idx)*(n_deg_types+1):,} | Val: {len(val_idx):,} clean + {len(val_idx)*n_deg_types:,} deg = {len(val_idx)*(n_deg_types+1):,}]")
    print(f"  * Held-Out Test Set (Remaining 20%):")
    print(f"      - Clean Test Videos      : {n_test_pool:,} (Real: {int((all_labels[test_idx]==0).sum()):,} | Fake: {int((all_labels[test_idx]==1).sum()):,})")
    print(f"      - Degraded Test Variants : {n_test_pool_deg:,} ({n_deg_types} variants per video)")
    print(f"      - Total Test (Clean+Deg) : {n_test_pool + n_test_pool_deg:,} videos")
    print(f"      - Test Evaluation Mode   : 100% Degraded Version (Auto-selected random degradation per video)")
    print("=" * 82 + "\n")

    # 5. Memory-Mapped Caches (Video & Audio)
    tag = f"T{n_frames}_S{img_size}_n{n_videos}"
    video_shape = (n_videos, n_frames, img_size, img_size, 3)
    variant_caches = {}

    search_dirs = [
        work_dir / "cache_degraded",
        Path(r"E:\Datasets\Code 2\biodeep_run_cnngru_mixed\cache_degraded"),
        script_dir / "biodeep_run_mixed_av" / "cache_degraded",
        script_dir / "biodeep_run_av_test100" / "cache_degraded",
        script_dir / "biodeep_run_test100" / "cache_degraded",
        script_dir / "biodeep_run_av" / "cache_degraded",
        script_dir / "biodeep_run",
    ]

    print("\n--- 1. Preparing Video Caches (Memory-Mapped) ---")
    clean_paths = [manifest_info[n]["clean_path"] for n in all_source_names]
    variant_caches["clean"] = find_or_create_cache(
        "clean", clean_paths, tag, video_shape, cache_dir, search_dirs, n_cpu
    )

    for deg in sorted_deg:
        deg_paths = [manifest_info[n]["degradations"].get(deg, manifest_info[n]["clean_path"]) for n in all_source_names]
        variant_caches[deg] = find_or_create_cache(
            deg, deg_paths, tag, video_shape, cache_dir, search_dirs, n_cpu
        )

    print(f"\nAll {len(active_variants)} video variant caches verified and ready!")

    # Audio Spectrogram Cache
    print("\n--- 2. Preparing Audio Spectrogram Cache ---")
    audio_cache = None
    target_audio_cache = cache_dir / f"cache_audio_{tag}.npy"
    audio_shape = (n_videos, 1, 128, 128)

    if target_audio_cache.exists():
        try:
            cand_arr = np.load(target_audio_cache, mmap_mode="r")
            if cand_arr.shape == audio_shape:
                print(f"  -> Found cached Audio Spectrograms: {target_audio_cache.name}")
                audio_cache = cand_arr
        except Exception:
            pass

    if audio_cache is None:
        for sdir in search_dirs:
            cand = sdir / f"cache_audio_{tag}.npy"
            if cand.exists():
                try:
                    cand_arr = np.load(cand, mmap_mode="r")
                    if cand_arr.shape == audio_shape:
                        print(f"  -> Reusing cached Audio Spectrograms from {sdir.name}: {cand.name}")
                        audio_cache = cand_arr
                        break
                except Exception:
                    pass

    if audio_cache is None:
        print(f"  -> Extracting audio spectrograms for {n_videos} source videos ({n_cpu} threads)...")
        audio_mmap = np.lib.format.open_memmap(target_audio_cache, mode="w+", dtype=np.float32, shape=audio_shape)
        t0 = time.time()
        last_print = 0

        with ThreadPoolExecutor(max_workers=n_cpu) as ex:
            futures = {
                ex.submit(
                    extract_audio_spectrogram,
                    all_source_names[i],
                    all_labels[i],
                    manifest_info[all_source_names[i]]["clean_path"],
                    data_root,
                ): i
                for i in range(n_videos)
            }
            completed = 0
            for fut in as_completed(futures):
                idx = futures[fut]
                audio_mmap[idx] = fut.result()
                completed += 1
                now = time.time()
                if now - last_print >= 0.15 or completed == n_videos:
                    elapsed = now - t0
                    rate = completed / max(0.001, elapsed)
                    eta = (n_videos - completed) / max(0.001, rate)
                    bar = progress_bar(completed, n_videos, bar_len=20)
                    sys.stdout.write(
                        f"\r  audio spectrograms: {bar} ({completed}/{n_videos}) | {rate:5.1f} vid/s | "
                        f"Elapsed: {format_time(elapsed)} | ETA: {format_time(eta)}   "
                    )
                    sys.stdout.flush()
                    last_print = now

        sys.stdout.write("\n")
        sys.stdout.flush()
        audio_mmap.flush()
        del audio_mmap
        print(f"  Saved Audio Spectrogram cache: {target_audio_cache.name} ({target_audio_cache.stat().st_size / 1e6:.1f} MB)")
        audio_cache = np.load(target_audio_cache, mmap_mode="r")

    print(f"Audio Cache ready! Shape: {audio_cache.shape} ({audio_cache.dtype})")

    # 5. Initialize Model
    def to_video_tensor(arr):
        t = torch.from_numpy(np.ascontiguousarray(arr)).permute(0, 1, 4, 2, 3).float().div_(127.5).sub_(1.0)
        return t.to(device)

    model = AudioVisualDeepFakeCNN().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: AudioVisualDeepFakeCNN | Total Parameters: {total_params:,}")

    best_model_path = work_dir / "best_model_mixed_av.pt"
    loaded_checkpoint = False

    # Checkpoint loading logic
    target_weights = None
    if args.weights:
        target_weights = Path(args.weights)
    elif args.mode == "test":
        # Search for available checkpoints
        cands = [
            best_model_path,
            script_dir / "biodeep_run_av_test100" / "best_model_av_test100.pt",
            script_dir / "biodeep_run_av" / "best_model_degraded_av.pt",
        ]
        for c in cands:
            if c.exists():
                target_weights = c
                break

    if target_weights and target_weights.exists():
        print(f"\n[Loading Pretrained Weights] -> {target_weights}")
        state_dict = torch.load(target_weights, map_location=device)
        model.load_state_dict(state_dict)
        loaded_checkpoint = True
        print("Model checkpoint loaded successfully!")
    elif args.mode == "test":
        raise FileNotFoundError(
            f"No checkpoint found to evaluate in test mode! Please specify --weights <path>"
        )

    history = []
    best = dict(score=-1, epoch=-1)
    total_training_time = 0

    # 6. Training Pipeline (if mode is train_test or train)
    if args.mode in ["train_test", "train"]:
        epochs = args.epochs if args.epochs is not None else (5 if args.dataset_scale == "test100" else 12)
        lr = args.lr
        train_mode = args.train_mode

        if train_mode == "expanded":
            train_instances = [(v, var) for v in train_idx for var in active_variants]
        else:
            # random_mix or compound_mix: one instance per video per epoch, dynamically degraded
            train_instances = [(v, None) for v in train_idx]

        print(f"\nTraining Mode: {train_mode.upper()} | Training instances per epoch: {len(train_instances):,}")

        def make_training_batch(batch_tuples, rng_np):
            B = len(batch_tuples)
            raw_frames = np.empty((B, clip_len, img_size, img_size, 3), dtype=np.uint8)
            raw_audios = np.empty((B, 1, 128, 128), dtype=np.float32)
            labels = np.empty(B, dtype=np.int64)

            for b, (v_idx, var_name) in enumerate(batch_tuples):
                chosen_var = var_name if var_name is not None else rng_np.choice(active_variants)
                cache_arr = variant_caches[chosen_var]
                sel = np.sort(rng_np.choice(n_frames, clip_len, replace=False))
                clip = cache_arr[v_idx, sel]

                # Random spatial augmentation
                if rng_np.random() < 0.5:
                    clip = clip[:, :, ::-1, :]
                pad = np.pad(clip, ((0, 0), (4, 4), (4, 4), (0, 0)), mode="edge")
                dy, dx = rng_np.integers(0, 9, 2)
                augmented_clip = pad[:, dy : dy + img_size, dx : dx + img_size].copy()

                # Compound mixed augmentation: add secondary stochastic degradation on-the-fly
                if train_mode == "compound_mix" and rng_np.random() < 0.35:
                    sec_type = rng_np.choice(["noise", "blur", "gamma"])
                    if sec_type == "noise":
                        noise = rng_np.normal(0, 8, augmented_clip.shape).astype(np.int16)
                        augmented_clip = np.clip(augmented_clip.astype(np.int16) + noise, 0, 255).astype(np.uint8)
                    elif sec_type == "blur":
                        augmented_clip = cv2.GaussianBlur(augmented_clip.reshape(-1, img_size, 3), (3, 3), 0.8).reshape(clip_len, img_size, img_size, 3)
                    elif sec_type == "gamma":
                        gamma = rng_np.uniform(0.7, 1.3)
                        inv_gamma = 1.0 / gamma
                        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
                        augmented_clip = cv2.LUT(augmented_clip, table)

                raw_frames[b] = augmented_clip
                raw_audios[b] = audio_cache[v_idx]
                labels[b] = all_labels[v_idx]

            return to_video_tensor(raw_frames), torch.from_numpy(raw_audios).to(device), torch.from_numpy(labels).to(device)

        counts = np.bincount(all_labels[train_idx], minlength=2)
        class_w = torch.tensor(counts.sum() / (2.0 * counts), dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_w)
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        steps_per_epoch = int(np.ceil(len(train_instances) / batch_size))
        total_steps = epochs * steps_per_epoch
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, total_steps=total_steps)
        rng_np = np.random.default_rng(args.seed)

        gpu_temp_init = get_gpu_temp(ttl=0)
        gpu_temp_str = f"{gpu_temp_init}C" if gpu_temp_init is not None else "N/A"

        print("\n" + "=" * 82)
        print(" MULTIMODAL AUDIO-VISUAL TRAINING SCHEDULE (MIXED DEGRADATION)")
        print(f"  * Total instances/epoch    : {len(train_instances):,} ({steps_per_epoch} steps/epoch, batch={batch_size})")
        print(f"  * Epochs                   : {epochs} ({total_steps} total optimization steps)")
        print(f"  * Initial GPU Temp         : {gpu_temp_str} (Thermal guard: pause if >= {args.gpu_max_temp}C)")
        print(f"  * Validation set           : {len(val_idx)} videos")
        print(f"  * Checkpoint destination   : {best_model_path.name}")
        print("=" * 82 + "\n")

        t_start = time.time()
        overall_step = 0

        @torch.no_grad()
        def predict_proba_eval(model_eval, cache_v, indices, mode="av", bs=32):
            model_eval.eval()
            stride = max(1, n_frames // clip_len)
            out = []
            for i in range(0, len(indices), bs):
                chunk = indices[i : i + bs]
                p = 0
                audio_chunk = torch.from_numpy(audio_cache[chunk]).to(device)
                for v in range(stride):
                    xb = cache_v[chunk][:, v::stride][:, :clip_len]
                    logits = model_eval(to_video_tensor(xb), audio_chunk, mode=mode)
                    p = p + torch.softmax(logits, 1)[:, 1].cpu().numpy()
                out.append(p / stride)
            return np.concatenate(out)

        for epoch in range(1, epochs + 1):
            model.train()
            perm_order = rng_np.permutation(len(train_instances))
            total_loss = 0.0
            t0 = time.time()
            last_print = 0

            for s in range(steps_per_epoch):
                overall_step += 1
                batch_slice = perm_order[s * batch_size : (s + 1) * batch_size]
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
                    if gpu_temp is not None and gpu_temp >= args.gpu_max_temp:
                        paused_sec = check_gpu_thermal_guard(args.gpu_max_temp, args.gpu_pause)
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
                        f"\rEpoch [{epoch:2d}/{epochs}] {bar} step {step_in_epoch:3d}/{steps_per_epoch} "
                        f"| loss {avg_loss:.4f} | GPU: {gpu_str} | ep ETA: {format_time(epoch_eta)} | total ETA: {format_time(total_eta)}   "
                    )
                    sys.stdout.flush()
                    last_print = now

            # Validation evaluation on Clean + Mixed subset
            pv_clean = predict_proba_eval(model, variant_caches["clean"], val_idx, mode="av")
            vm_clean = compute_metrics(all_labels[val_idx], pv_clean)

            auc_val = 0.5 if (np.isnan(vm_clean["auc"]) or np.isinf(vm_clean["auc"])) else vm_clean["auc"]
            bal_acc_val = 0.5 if (np.isnan(vm_clean["balanced_acc"]) or np.isinf(vm_clean["balanced_acc"])) else vm_clean["balanced_acc"]
            score = auc_val + bal_acc_val
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
            if gpu_temp is not None and gpu_temp >= args.gpu_max_temp:
                paused_sec = check_gpu_thermal_guard(args.gpu_max_temp, args.gpu_pause)
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
                f"\rEpoch [{epoch:2d}/{epochs}] loss {avg_train_loss:.4f} | "
                f"val Clean AUC {vm_clean['auc']:.4f} bal-acc {vm_clean['balanced_acc']:.4f} | GPU: {gpu_str} | "
                f"time {format_time(epoch_elapsed)} | elapsed {format_time(total_elapsed)} | ETA {format_time(total_eta)}{flag}    \n"
            )
            sys.stdout.flush()

        total_training_time = time.time() - t_start
        print("-" * 82)
        print(f"Training complete in {format_time(total_training_time)}! Best Epoch: {best['epoch']}")
        print("-" * 82 + "\n")

    # 7. Comprehensive Evaluation: Mixed Degradation Benchmark Suite
    print("\n" + "=" * 82)
    print(" COMPREHENSIVE MIXED DEGRADATION TEST BENCHMARK")
    print("=" * 82)

    if best_model_path.exists():
        print(f"Loading best checkpoint for testing: {best_model_path.name}")
        model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    @torch.no_grad()
    def predict_proba_multimodal(model_eval, cache_v, indices, mode="av", bs=32):
        model_eval.eval()
        stride = max(1, n_frames // clip_len)
        out = []
        for i in range(0, len(indices), bs):
            chunk = indices[i : i + bs]
            p = 0
            audio_chunk = torch.from_numpy(audio_cache[chunk]).to(device)
            for v in range(stride):
                xb = cache_v[chunk][:, v::stride][:, :clip_len]
                logits = model_eval(to_video_tensor(xb), audio_chunk, mode=mode)
                p = p + torch.softmax(logits, 1)[:, 1].cpu().numpy()
            out.append(p / stride)
        return np.concatenate(out)

    y_test = all_labels[test_idx]
    n_test = len(test_idx)
    n_real_test = int((y_test == 0).sum())
    n_fake_test = int((y_test == 1).sum())

    # 7.1 Individual Variant Evaluation
    eval_results_av = {}
    eval_results_vonly = {}
    eval_preds_av = {}
    eval_preds_vo = {}
    pred_table = {
        "video_name": all_source_names[test_idx],
        "true_label": y_test,
    }

    print(f"\n[1/4] Evaluating {len(active_variants)} Variants on {n_test} Held-Out 20% Test Videos ({n_real_test} Real + {n_fake_test} Fake)...")
    for var in active_variants:
        p_av = predict_proba_multimodal(model, variant_caches[var], test_idx, mode="av")
        eval_preds_av[var] = p_av
        pred_table[f"p_fake_av_{var}"] = p_av
        eval_results_av[var] = compute_metrics(y_test, p_av, thr=0.5)

        p_vo = predict_proba_multimodal(model, variant_caches[var], test_idx, mode="v_only")
        eval_preds_vo[var] = p_vo
        pred_table[f"p_fake_vonly_{var}"] = p_vo
        eval_results_vonly[var] = compute_metrics(y_test, p_vo, thr=0.5)

    # Audio-Only Evaluation (evaluated on clean baseline)
    p_test_ao = predict_proba_multimodal(model, variant_caches["clean"], test_idx, mode="a_only")
    pred_table["p_fake_aonly"] = p_test_ao
    audio_only_metrics = compute_metrics(y_test, p_test_ao, thr=0.5)

    # 7.2 PRIMARY BENCHMARK: AUTO-SELECTED RANDOM DEGRADATION TEST (HELD-OUT 20%)
    # Each test video is tested on its DEGRADED version with auto-selected random degradation option
    print(f"\n[2/4] Computing Auto-Selected Random Degradation Test ({n_test} videos)...")
    rng_deg = random.Random(args.seed + 2026)
    assigned_degradations = [rng_deg.choice(sorted_deg) for _ in range(n_test)]
    pred_table["assigned_degradation"] = assigned_degradations

    p_deg_av = np.array([eval_preds_av[deg][i] for i, deg in enumerate(assigned_degradations)])
    p_deg_vo = np.array([eval_preds_vo[deg][i] for i, deg in enumerate(assigned_degradations)])

    pred_table["p_fake_clean_av"] = eval_preds_av["clean"]
    pred_table["pred_clean_av"] = (eval_preds_av["clean"] >= 0.5).astype(int)
    pred_table["p_fake_degraded_av"] = p_deg_av
    pred_table["pred_degraded_av"] = (p_deg_av >= 0.5).astype(int)
    pred_table["p_fake_clean_vonly"] = eval_preds_vo["clean"]
    pred_table["pred_clean_vonly"] = (eval_preds_vo["clean"] >= 0.5).astype(int)
    pred_table["p_fake_degraded_vonly"] = p_deg_vo
    pred_table["pred_degraded_vonly"] = (p_deg_vo >= 0.5).astype(int)

    degraded_metrics_av = compute_metrics(y_test, p_deg_av, thr=0.5)
    degraded_metrics_vo = compute_metrics(y_test, p_deg_vo, thr=0.5)
    clean_metrics_av = eval_results_av["clean"]
    clean_metrics_vo = eval_results_vonly["clean"]

    # 7.3 POOLED MIXED DEGRADATION BENCHMARK (All 10 Variants Pooled)
    print(f"[3/4] Computing Pooled Mixed Degradation Benchmark ({n_test * total_variants:,} instances)...")
    y_test_pooled = np.tile(y_test, total_variants)
    p_pooled_av = np.concatenate([eval_preds_av[var] for var in active_variants])
    p_pooled_vo = np.concatenate([eval_preds_vo[var] for var in active_variants])
    p_pooled_ao = np.tile(p_test_ao, total_variants)

    pooled_mixed_metrics_av = compute_metrics(y_test_pooled, p_pooled_av, thr=0.5)
    pooled_mixed_metrics_vo = compute_metrics(y_test_pooled, p_pooled_vo, thr=0.5)
    pooled_mixed_metrics_ao = compute_metrics(y_test_pooled, p_pooled_ao, thr=0.5)

    # 7.4 PERFORMANCE DROP & SENSITIVITY ANALYSIS
    delta_auc_deg = degraded_metrics_av["auc"] - clean_metrics_av["auc"]
    delta_acc_deg = degraded_metrics_av["accuracy"] - clean_metrics_av["accuracy"]
    multimodal_gain_clean = clean_metrics_av["auc"] - clean_metrics_vo["auc"]
    multimodal_gain_deg = degraded_metrics_av["auc"] - degraded_metrics_vo["auc"]

    # 7.5 CONSTRUCT SUMMARY TABLES
    # Primary Table: Head-to-Head Clean vs Degraded Version Comparison
    head_to_head_rows = [
        {
            "Metric": "Accuracy",
            "Clean Baseline": f"{clean_metrics_av['accuracy']*100:.2f}%",
            "Degraded Version (Random Deg)": f"{degraded_metrics_av['accuracy']*100:.2f}%",
            "Delta / Robustness Gap": f"{(degraded_metrics_av['accuracy'] - clean_metrics_av['accuracy'])*100:+.2f}%",
        },
        {
            "Metric": "Balanced Accuracy",
            "Clean Baseline": f"{clean_metrics_av['balanced_acc']*100:.2f}%",
            "Degraded Version (Random Deg)": f"{degraded_metrics_av['balanced_acc']*100:.2f}%",
            "Delta / Robustness Gap": f"{(degraded_metrics_av['balanced_acc'] - clean_metrics_av['balanced_acc'])*100:+.2f}%",
        },
        {
            "Metric": "ROC-AUC",
            "Clean Baseline": f"{clean_metrics_av['auc']:.4f}",
            "Degraded Version (Random Deg)": f"{degraded_metrics_av['auc']:.4f}",
            "Delta / Robustness Gap": f"{delta_auc_deg:+.4f}",
        },
        {
            "Metric": "Precision (Fake)",
            "Clean Baseline": f"{clean_metrics_av['precision']*100:.2f}%",
            "Degraded Version (Random Deg)": f"{degraded_metrics_av['precision']*100:.2f}%",
            "Delta / Robustness Gap": f"{(degraded_metrics_av['precision'] - clean_metrics_av['precision'])*100:+.2f}%",
        },
        {
            "Metric": "Recall / Sensitivity",
            "Clean Baseline": f"{clean_metrics_av['recall']*100:.2f}%",
            "Degraded Version (Random Deg)": f"{degraded_metrics_av['recall']*100:.2f}%",
            "Delta / Robustness Gap": f"{(degraded_metrics_av['recall'] - clean_metrics_av['recall'])*100:+.2f}%",
        },
        {
            "Metric": "Specificity (Real)",
            "Clean Baseline": f"{clean_metrics_av['specificity']*100:.2f}%",
            "Degraded Version (Random Deg)": f"{degraded_metrics_av['specificity']*100:.2f}%",
            "Delta / Robustness Gap": f"{(degraded_metrics_av['specificity'] - clean_metrics_av['specificity'])*100:+.2f}%",
        },
        {
            "Metric": "F1-Score",
            "Clean Baseline": f"{clean_metrics_av['f1']:.4f}",
            "Degraded Version (Random Deg)": f"{degraded_metrics_av['f1']:.4f}",
            "Delta / Robustness Gap": f"{degraded_metrics_av['f1'] - clean_metrics_av['f1']:+.4f}",
        },
        {
            "Metric": "Confusion Matrix [TN, FP]",
            "Clean Baseline": f"[{clean_metrics_av['tn']}, {clean_metrics_av['fp']}]",
            "Degraded Version (Random Deg)": f"[{degraded_metrics_av['tn']}, {degraded_metrics_av['fp']}]",
            "Delta / Robustness Gap": f"FP change: {degraded_metrics_av['fp'] - clean_metrics_av['fp']:+d}",
        },
        {
            "Metric": "Confusion Matrix [FN, TP]",
            "Clean Baseline": f"[{clean_metrics_av['fn']}, {clean_metrics_av['tp']}]",
            "Degraded Version (Random Deg)": f"[{degraded_metrics_av['fn']}, {degraded_metrics_av['tp']}]",
            "Delta / Robustness Gap": f"FN change: {degraded_metrics_av['fn'] - clean_metrics_av['fn']:+d}",
        },
    ]
    df_head_to_head = pd.DataFrame(head_to_head_rows).set_index("Metric")

    # Table 2: Benchmark Comparison across Modalities
    benchmark_rows = [
        {
            "Evaluation Benchmark": "Clean Test Baseline (Authentic Quality)",
            "Samples": n_test,
            "AV AUC": clean_metrics_av["auc"],
            "AV Accuracy": clean_metrics_av["accuracy"],
            "AV F1": clean_metrics_av["f1"],
            "Video-Only AUC": clean_metrics_vo["auc"],
            "Video-Only Acc": clean_metrics_vo["accuracy"],
            "Audio-Only AUC": audio_only_metrics["auc"],
            "AV Gain (AV - VO)": multimodal_gain_clean,
        },
        {
            "Evaluation Benchmark": "Degraded Version Test (Auto-Selected Random Deg)",
            "Samples": n_test,
            "AV AUC": degraded_metrics_av["auc"],
            "AV Accuracy": degraded_metrics_av["accuracy"],
            "AV F1": degraded_metrics_av["f1"],
            "Video-Only AUC": degraded_metrics_vo["auc"],
            "Video-Only Acc": degraded_metrics_vo["accuracy"],
            "Audio-Only AUC": audio_only_metrics["auc"],
            "AV Gain (AV - VO)": multimodal_gain_deg,
        },
        {
            "Evaluation Benchmark": "Pooled Mixed Benchmark (All 10 Variants)",
            "Samples": n_test * total_variants,
            "AV AUC": pooled_mixed_metrics_av["auc"],
            "AV Accuracy": pooled_mixed_metrics_av["accuracy"],
            "AV F1": pooled_mixed_metrics_av["f1"],
            "Video-Only AUC": pooled_mixed_metrics_vo["auc"],
            "Video-Only Acc": pooled_mixed_metrics_vo["accuracy"],
            "Audio-Only AUC": pooled_mixed_metrics_ao["auc"],
            "AV Gain (AV - VO)": pooled_mixed_metrics_av["auc"] - pooled_mixed_metrics_vo["auc"],
        },
    ]
    df_benchmarks = pd.DataFrame(benchmark_rows).set_index("Evaluation Benchmark")

    # Table 3: Detailed Per-Degradation Robustness Breakdown
    breakdown_rows = []
    for var in active_variants:
        m_av = eval_results_av[var]
        m_vo = eval_results_vonly[var]
        breakdown_rows.append(
            {
                "Variant": var.upper(),
                "AV AUC": m_av["auc"],
                "AV Accuracy": m_av["accuracy"],
                "AV Bal-Acc": m_av["balanced_acc"],
                "AV F1": m_av["f1"],
                "Delta AUC (vs Clean)": m_av["auc"] - clean_metrics_av["auc"],
                "Delta Acc (vs Clean)": m_av["accuracy"] - clean_metrics_av["accuracy"],
                "Video-Only AUC": m_vo["auc"],
                "Video-Only Acc": m_vo["accuracy"],
                "Audio Gain (AV - VO)": m_av["auc"] - m_vo["auc"],
            }
        )

    # Append Degraded Version Row and Pooled Mixed Row
    breakdown_rows.append(
        {
            "Variant": "--> DEGRADED VERSION (RANDOM) <--",
            "AV AUC": degraded_metrics_av["auc"],
            "AV Accuracy": degraded_metrics_av["accuracy"],
            "AV Bal-Acc": degraded_metrics_av["balanced_acc"],
            "AV F1": degraded_metrics_av["f1"],
            "Delta AUC (vs Clean)": delta_auc_deg,
            "Delta Acc (vs Clean)": delta_acc_deg,
            "Video-Only AUC": degraded_metrics_vo["auc"],
            "Video-Only Acc": degraded_metrics_vo["accuracy"],
            "Audio Gain (AV - VO)": multimodal_gain_deg,
        }
    )
    breakdown_rows.append(
        {
            "Variant": "--> POOLED MIXED (ALL 10) <--",
            "AV AUC": pooled_mixed_metrics_av["auc"],
            "AV Accuracy": pooled_mixed_metrics_av["accuracy"],
            "AV Bal-Acc": pooled_mixed_metrics_av["balanced_acc"],
            "AV F1": pooled_mixed_metrics_av["f1"],
            "Delta AUC (vs Clean)": pooled_mixed_metrics_av["auc"] - clean_metrics_av["auc"],
            "Delta Acc (vs Clean)": pooled_mixed_metrics_av["accuracy"] - clean_metrics_av["accuracy"],
            "Video-Only AUC": pooled_mixed_metrics_vo["auc"],
            "Video-Only Acc": pooled_mixed_metrics_vo["accuracy"],
            "Audio Gain (AV - VO)": pooled_mixed_metrics_av["auc"] - pooled_mixed_metrics_vo["auc"],
        }
    )
    df_breakdown = pd.DataFrame(breakdown_rows).set_index("Variant")

    # DISPLAY RESULTS
    print("\n" + "=" * 82)
    print(" TABLE 1: HELD-OUT 20% TEST BENCHMARK RESULTS (CLEAN vs DEGRADED VERSION)")
    print(f" [Test Cohort: {n_test} Held-Out Videos | Real: {n_real_test}, Fake: {n_fake_test} | Identity Leakage-Free]")
    print("=" * 82)
    print(df_head_to_head.to_string())
    print("-" * 82)
    print(" MULTIMODAL RESILIENCE UNDER DEGRADATION:")
    print(f"  * Audio-Visual Multimodal AUC : {degraded_metrics_av['auc']:.4f} (Accuracy: {degraded_metrics_av['accuracy']*100:.2f}%)")
    print(f"  * Video-Only Ablation AUC     : {degraded_metrics_vo['auc']:.4f} (Accuracy: {degraded_metrics_vo['accuracy']*100:.2f}%)")
    print(f"  * Audio-Only Baseline AUC     : {audio_only_metrics['auc']:.4f} (Accuracy: {audio_only_metrics['accuracy']*100:.2f}%)")
    print(f"  * Multimodal Gain (AV - VO)   : {multimodal_gain_deg:+.4f} AUC ({(degraded_metrics_av['accuracy'] - degraded_metrics_vo['accuracy'])*100:+.2f}% Accuracy)")
    print("=" * 82)

    print("\n" + "=" * 82)
    print(" TABLE 2: BENCHMARK COMPARISON ACROSS MODALITIES")
    print("=" * 82)
    print(df_benchmarks.round(4).to_string())

    print("\n" + "=" * 82)
    print(" TABLE 3: PER-DEGRADATION ROBUSTNESS & PERFORMANCE DROP ANALYSIS")
    print("=" * 82)
    print(df_breakdown.round(4).to_string())
    print("=" * 82)

    # Save detailed predictions CSV
    predictions_df = pd.DataFrame(pred_table)
    pred_csv_path = work_dir / "test_predictions_degraded_av.csv"
    predictions_df.to_csv(pred_csv_path, index=False)
    if report_dir != work_dir:
        predictions_df.to_csv(report_dir / "test_predictions_degraded_av.csv", index=False)
    print(f"\n[Saved Predictions CSV] -> {pred_csv_path.name} ({len(predictions_df):,} test videos)")

    # 8. VISUALIZATION ARTIFACTS
    print(f"[4/4] Generating Publication-Quality Plots...")

    # Plot 1: Clean vs Degraded Version ROC Curves
    fig, ax = plt.subplots(figsize=(8, 6.5))
    fpr_c, tpr_c, _ = roc_curve(y_test, eval_preds_av["clean"])
    ax.plot(fpr_c, tpr_c, color="#27ae60", lw=2.5, label=f"Clean AV Baseline (AUC = {clean_metrics_av['auc']:.3f})")

    fpr_deg, tpr_deg, _ = roc_curve(y_test, p_deg_av)
    ax.plot(fpr_deg, tpr_deg, color="#e74c3c", lw=3.0, linestyle="-", label=f"Degraded Version AV (Random Deg, AUC = {degraded_metrics_av['auc']:.3f})")

    fpr_vo, tpr_vo, _ = roc_curve(y_test, p_deg_vo)
    ax.plot(fpr_vo, tpr_vo, color="#3498db", lw=2.0, linestyle="--", label=f"Degraded Video-Only (AUC = {degraded_metrics_vo['auc']:.3f})")

    fpr_ao, tpr_ao, _ = roc_curve(y_test, p_test_ao)
    ax.plot(fpr_ao, tpr_ao, color="#e67e22", lw=1.8, linestyle=":", label=f"Audio-Only Baseline (AUC = {audio_only_metrics['auc']:.3f})")

    ax.plot([0, 1], [0, 1], color="#7f8c8d", lw=1.5, linestyle="--", alpha=0.7)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    ax.set_ylabel("True Positive Rate (Recall / Sensitivity)", fontsize=11)
    ax.set_title("ROC Curves: Audio-Visual DeepFake Detection (Clean vs Degraded Version)", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", fontsize=10)
    plt.tight_layout()
    roc_plot_path = work_dir / "degraded_av_roc_curves.png"
    plt.savefig(roc_plot_path, dpi=130)
    if report_dir != work_dir:
        plt.savefig(report_dir / "degraded_av_roc_curves.png", dpi=130)
    plt.close()

    # Plot 2: Multimodal Resilience Bar Chart
    fig, ax = plt.subplots(figsize=(12, 5.5))
    plot_variants = [v.upper() for v in active_variants] + ["DEGRADED (RANDOM)", "POOLED MIXED"]
    av_plot_aucs = [eval_results_av[v]["auc"] for v in active_variants] + [degraded_metrics_av["auc"], pooled_mixed_metrics_av["auc"]]
    vo_plot_aucs = [eval_results_vonly[v]["auc"] for v in active_variants] + [degraded_metrics_vo["auc"], pooled_mixed_metrics_vo["auc"]]

    x = np.arange(len(plot_variants))
    w = 0.38
    rects1 = ax.bar(x - w / 2, av_plot_aucs, w, label="Audio-Visual (Multimodal)", color="#2ecc71")
    rects2 = ax.bar(x + w / 2, vo_plot_aucs, w, label="Video-Only Ablation", color="#3498db")
    ax.axhline(audio_only_metrics["auc"], color="#e67e22", linestyle="--", lw=2, label=f"Audio-Only Baseline ({audio_only_metrics['auc']:.3f})")

    ax.set_ylabel("ROC-AUC Score", fontsize=11)
    ax.set_title("Multimodal Resilience: Audio-Visual vs Video-Only Across Degradations", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(plot_variants, rotation=30, ha="right", fontsize=9.5)
    ax.set_ylim([0.0, 1.05])
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    ax.legend(loc="lower right", fontsize=10)
    plt.tight_layout()
    ablation_plot_path = work_dir / "multimodal_degraded_ablation.png"
    plt.savefig(ablation_plot_path, dpi=130)
    if report_dir != work_dir:
        plt.savefig(report_dir / "multimodal_degraded_ablation.png", dpi=130)
    plt.close()

    # Plot 3: Dual Confusion Matrices (Clean Baseline vs Degraded Version)
    cm_clean = confusion_matrix(y_test, (eval_preds_av["clean"] >= 0.5).astype(int), labels=[0, 1])
    cm_deg = confusion_matrix(y_test, (p_deg_av >= 0.5).astype(int), labels=[0, 1])

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))
    for ax_cm, cm, title in zip(
        axes,
        [cm_clean, cm_deg],
        [f"Clean AV Test CM (N={n_test})", f"Degraded AV Test CM (Random Deg, N={n_test})"],
    ):
        ax_cm.imshow(cm, cmap="Blues")
        for (i, j), val in np.ndenumerate(cm):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax_cm.text(j, i, f"{val:,}", ha="center", va="center", fontsize=13, color=color, fontweight="bold")
        ax_cm.set_xticks([0, 1])
        ax_cm.set_xticklabels(["Pred Real", "Pred Fake"])
        ax_cm.set_yticks([0, 1])
        ax_cm.set_yticklabels(["Real", "Fake"])
        ax_cm.set_title(title, fontsize=11, fontweight="bold")

    plt.tight_layout()
    cm_plot_path = work_dir / "confusion_matrix_clean_vs_degraded.png"
    plt.savefig(cm_plot_path, dpi=130)
    if report_dir != work_dir:
        plt.savefig(report_dir / "confusion_matrix_clean_vs_degraded.png", dpi=130)
    plt.close()

    # Plot 4: Training Curves (if training occurred)
    if history:
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        h = pd.DataFrame(history)
        ax[0].plot(h.epoch, h.train_loss, label="Train Loss", color="#e74c3c", linewidth=2)
        ax[0].set_xlabel("Epoch")
        ax[0].set_ylabel("Loss")
        ax[0].set_title("Mixed Degradation Training Loss")
        ax[0].grid(True, linestyle="--", alpha=0.5)
        ax[0].legend()

        ax[1].plot(h.epoch, h.val_clean_auc, label="Val Clean AUC", color="#2980b9", linewidth=2)
        ax[1].plot(h.epoch, h.val_clean_bal_acc, label="Val Clean Bal-Acc", color="#27ae60", linewidth=2)
        ax[1].set_xlabel("Epoch")
        ax[1].set_ylabel("Score")
        ax[1].set_title("Validation Metrics")
        ax[1].grid(True, linestyle="--", alpha=0.5)
        ax[1].legend()

        plt.tight_layout()
        plt.savefig(work_dir / "training_curves_mixed_av.png", dpi=130)
        plt.close()

    # 9. STRUCTURED JSON REPORT
    now = datetime.now()
    date_str = f"{now.year % 100}-{now.month}-{now.day}"
    time_str = f"{now.hour:02d}.{now.minute:02d}"
    report_filename = f"report_degraded_av_date_{date_str}_time_{time_str}.json"

    report_data = {
        "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
        "date": date_str,
        "time": time_str,
        "mode": args.mode,
        "dataset_scale": args.dataset_scale,
        "sample_counts": {
            "source_videos": n_videos,
            "real_videos": n_clean_real,
            "fake_videos": n_clean_fake,
            "train_videos": len(train_idx),
            "val_videos": len(val_idx),
            "test_videos": len(test_idx),
            "total_variants": total_variants,
            "total_clean_videos": n_clean_total,
            "total_degraded_videos": n_deg_total,
            "grand_total_videos": n_grand_total,
        },
        "benchmarks": {
            "clean_baseline": {
                "audio_visual": clean_metrics_av,
                "video_only": clean_metrics_vo,
                "audio_only": audio_only_metrics,
            },
            "degraded_version_test": {
                "audio_visual": degraded_metrics_av,
                "video_only": degraded_metrics_vo,
                "audio_only": audio_only_metrics,
            },
            "pooled_mixed_benchmark": {
                "audio_visual": pooled_mixed_metrics_av,
                "video_only": pooled_mixed_metrics_vo,
                "audio_only": pooled_mixed_metrics_ao,
            },
        },
        "robustness_drop_metrics": {
            "auc_drop_clean_to_degraded": float(clean_metrics_av["auc"] - degraded_metrics_av["auc"]),
            "acc_drop_clean_to_degraded": float(clean_metrics_av["accuracy"] - degraded_metrics_av["accuracy"]),
            "multimodal_gain_on_degraded": float(multimodal_gain_deg),
        },
        "per_degradation_results": {
            "audio_visual": eval_results_av,
            "video_only": eval_results_vonly,
        },
        "summary_table_head_to_head": df_head_to_head.to_dict(orient="index"),
        "summary_table_benchmarks": df_benchmarks.to_dict(orient="index"),
        "summary_table_breakdown": df_breakdown.to_dict(orient="index"),
        "training_time": format_time(total_training_time) if total_training_time > 0 else "N/A (Eval only)",
        "gpu_temp": f"{get_gpu_temp(ttl=0)}C" if get_gpu_temp(ttl=0) is not None else "N/A",
    }

    # Save reports in both work_dir and report_dir
    for target_dir in [report_dir, work_dir / "test", work_dir]:
        target_dir.mkdir(parents=True, exist_ok=True)
        with open(target_dir / report_filename, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2)
        with open(target_dir / "latest_report_degraded_av.json", "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2)

    print(f"\nSaved Reports and Visualizations:")
    print(f"  * JSON Report   : {report_dir / report_filename}")
    print(f"  * Latest Report : {report_dir / 'latest_report_degraded_av.json'}")
    print(f"  * ROC Curves    : {roc_plot_path}")
    print(f"  * Ablation Plot : {ablation_plot_path}")
    print(f"  * Confusion Mat : {cm_plot_path}")
    print(f"  * Predictions   : {pred_csv_path}")
    print("\nBioDeepAV Degradation Pipeline execution completed successfully!")


if __name__ == "__main__":
    main()
