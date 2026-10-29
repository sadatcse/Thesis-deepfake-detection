"""
BioDeepAV 200-Video Multimodal Deepfake Testing Pipeline with Mixed Degradation
=============================================================================
Dedicated evaluation pipeline that tests exactly 200 videos on:
  1. Random In-The-Wild Mixed Degradation: Each video is assigned a randomly
     sampled degradation (clean, blur, brightness, compression, contrast,
     fps_reduction, frame_drop, motion_blur, noise, saturation).
  2. Pooled Mixed Degradation: All 200 videos evaluated across all 10 variants
     (2,000 video instances evaluated).
  3. Per-Degradation Robustness Breakdown: Clean vs all 9 degradation types.
  4. Multimodal Ablation: Audio-Visual vs Video-Only vs Audio-Only.
  5. Detailed Per-Video Predictions CSV and Publication-Quality Visualizations.
"""

import os
import re
import sys
import json
import time
import random
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

cv2.setNumThreads(1)


# ------------------------- CLI ARGUMENTS -------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Test 200 videos under Random & Pooled Mixed Degradation with AudioVisualDeepFakeCNN"
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to pre-trained model checkpoint (.pt). If None, automatically searches for best model.",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Path to BioDeepAV dataset root (default: E:\\Datasets\\BioDeepAV)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save test outputs and plots (default: test_200_mixed_results)",
    )
    parser.add_argument(
        "--report-dir",
        type=str,
        default=None,
        help="Directory to save JSON reports (default: test)",
    )
    parser.add_argument(
        "--n-videos",
        type=int,
        default=200,
        help="Total number of videos to test (default: 200, 100 real + 100 fake)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for video sampling and degradation assignment (default: 42)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Classification decision threshold (default: 0.5)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for model inference (default: 32)",
    )
    parser.add_argument(
        "--force-fresh-sampling",
        action="store_true",
        help="Force fresh sampling and feature extraction of 200 videos instead of reusing pre-extracted cache.",
    )
    return parser.parse_args()


# ------------------------- MODEL ARCHITECTURE -------------------------
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


# ------------------------- FEATURE EXTRACTION HELPERS -------------------------
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


# ------------------------- METRICS CALCULATION -------------------------
def calculate_metrics(y_true, p, thr=0.5):
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


# ------------------------- MAIN EVALUATION ROUTINE -------------------------
def main():
    args = parse_args()
    script_dir = Path(__file__).resolve().parent

    default_root = (
        script_dir.parent / "BioDeepAV"
        if (script_dir.parent / "BioDeepAV").exists()
        else Path(r"E:\Datasets\BioDeepAV")
    )
    data_root = Path(args.data_root) if args.data_root else default_root
    output_dir = Path(args.output_dir) if args.output_dir else (script_dir / "test_200_mixed_results")
    report_dir = Path(args.report_dir) if args.report_dir else (script_dir / "test")

    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 84)
    print(" BIODEEP-AV 200-VIDEO DEDICATED TEST PIPELINE: MIXED & RANDOM DEGRADATION")
    print("=" * 84)
    print(f"  * Device            : {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    print(f"  * Dataset Root      : {data_root}")
    print(f"  * Output Directory  : {output_dir}")
    print(f"  * Test Sample Size  : {args.n_videos} Videos (Balanced: {args.n_videos // 2} Real + {args.n_videos // 2} Fake)")
    print(f"  * Random Seed       : {args.seed}")
    print(f"  * Decision Threshold: {args.threshold}")
    print("=" * 84)

    # 1. Discover Manifest and Degradations
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
    active_variants = ["clean"] + sorted_deg
    print(f"\nDiscovered {len(manifest_info):,} total videos in BioDeepAV dataset.")
    print(f"Active degradation types ({len(sorted_deg)}): {', '.join(sorted_deg)}")

    # 2. Select 200 Random Test Videos (100 Real + 100 Fake)
    all_real_pool = sorted([
        k for k, v in manifest_info.items()
        if v["label"] == 0 and all(d in v["degradations"] for d in sorted_deg)
    ])
    all_fake_pool = sorted([
        k for k, v in manifest_info.items()
        if v["label"] == 1 and all(d in v["degradations"] for d in sorted_deg)
    ])

    n_half = args.n_videos // 2
    rng_sample = random.Random(args.seed)
    selected_real = sorted(rng_sample.sample(all_real_pool, min(n_half, len(all_real_pool))))
    selected_fake = sorted(rng_sample.sample(all_fake_pool, min(n_half, len(all_fake_pool))))

    selected_source_names = np.array(selected_real + selected_fake)
    selected_labels = np.array([0] * len(selected_real) + [1] * len(selected_fake))
    n_test = len(selected_source_names)
    total_variants = len(active_variants)

    print(f"\nTest Cohort Selection:")
    print(f"  * Real Videos       : {len(selected_real)}")
    print(f"  * Fake Videos       : {len(selected_fake)}")
    print(f"  * Total Test Videos : {n_test}")
    print(f"  * Variants/Video    : {total_variants} (1 Clean + {len(sorted_deg)} Degraded)")
    print(f"  * Total Evaluations : {n_test * total_variants:,} instances in Pooled Mixed Benchmark")

    # 3. Load or Extract Features (Video & Audio Caches)
    n_frames = 16
    img_size = 96
    clip_len = 8
    video_shape = (n_test, n_frames, img_size, img_size, 3)
    audio_shape = (n_test, 1, 128, 128)
    n_cpu = os.cpu_count() or 8

    # Check if pre-extracted caches exist for these 200 videos (seed 42 matches test100 cache!)
    existing_cache_dir = script_dir / "biodeep_run_av_test100" / "cache_degraded"
    local_cache_dir = output_dir / "cache_degraded"
    local_cache_dir.mkdir(parents=True, exist_ok=True)

    variant_caches = {}
    print("\n--- 1. Loading / Caching Video Variants (Memory-Mapped) ---")

    can_reuse_existing = (
        not args.force_fresh_sampling
        and args.seed == 42
        and existing_cache_dir.exists()
        and (existing_cache_dir / "cache_clean_T16_S96_n200.npy").exists()
    )

    if can_reuse_existing:
        print(f"  -> Found pre-extracted 200-sample caches in {existing_cache_dir.name}!")
        print("     Loading memory-mapped arrays instantly without re-extraction...")
        for var in active_variants:
            c_path = existing_cache_dir / f"cache_{var}_T16_S96_n200.npy"
            variant_caches[var] = np.load(c_path, mmap_mode="r")
            print(f"     * Loaded {var:14s}: {c_path.name}")
    else:
        print("  -> Performing extraction for 200 test videos...")
        for var in active_variants:
            c_path = local_cache_dir / f"cache_{var}_n{n_test}.npy"
            if c_path.exists() and c_path.stat().st_size >= np.prod(video_shape):
                variant_caches[var] = np.load(c_path, mmap_mode="r")
                print(f"     * Found existing cache for {var:14s}: {c_path.name}")
                continue

            paths = (
                [manifest_info[n]["clean_path"] for n in selected_source_names]
                if var == "clean"
                else [manifest_info[n]["degradations"][var] for n in selected_source_names]
            )
            print(f"     * Extracting {var.upper()} ({len(paths)} videos)...")
            mmap_arr = np.lib.format.open_memmap(c_path, mode="w+", dtype=np.uint8, shape=video_shape)
            with ThreadPoolExecutor(max_workers=n_cpu) as ex:
                futures = {ex.submit(extract_video_clip, p, n_frames, img_size): i for i, p in enumerate(paths)}
                for fut in as_completed(futures):
                    i = futures[fut]
                    res = fut.result()
                    mmap_arr[i] = res if (res is not None and res.shape == video_shape[1:]) else 0
            mmap_arr.flush()
            del mmap_arr
            variant_caches[var] = np.load(c_path, mmap_mode="r")

    print("\n--- 2. Loading / Caching Audio Spectrograms ---")
    audio_cache = None
    if can_reuse_existing and (existing_cache_dir / "cache_audio_T16_S96_n200.npy").exists():
        a_path = existing_cache_dir / "cache_audio_T16_S96_n200.npy"
        audio_cache = np.load(a_path, mmap_mode="r")
        print(f"  -> Loaded audio spectrogram cache: {a_path.name} ({audio_cache.shape})")
    else:
        a_path = local_cache_dir / f"cache_audio_n{n_test}.npy"
        if a_path.exists() and a_path.stat().st_size >= np.prod(audio_shape) * 4:
            audio_cache = np.load(a_path, mmap_mode="r")
            print(f"  -> Found existing audio spectrogram cache: {a_path.name}")
        else:
            print(f"  -> Extracting audio spectrograms for {n_test} videos...")
            mmap_audio = np.lib.format.open_memmap(a_path, mode="w+", dtype=np.float32, shape=audio_shape)
            with ThreadPoolExecutor(max_workers=n_cpu) as ex:
                futures = {
                    ex.submit(
                        extract_audio_spectrogram,
                        selected_source_names[i],
                        selected_labels[i],
                        manifest_info[selected_source_names[i]]["clean_path"],
                        data_root,
                    ): i
                    for i in range(n_test)
                }
                for fut in as_completed(futures):
                    mmap_audio[futures[fut]] = fut.result()
            mmap_audio.flush()
            del mmap_audio
            audio_cache = np.load(a_path, mmap_mode="r")
            print(f"  -> Saved audio spectrogram cache: {a_path.name}")

    # 4. Load Model Checkpoint
    checkpoint_candidates = [
        Path(args.weights) if args.weights else None,
        script_dir / "biodeep_run_mixed_av" / "best_model_mixed_av.pt",
        script_dir / "biodeep_run_av_test100" / "best_model_av_test100.pt",
        script_dir / "biodeep_run_av" / "best_model_degraded_av.pt",
    ]
    model_path = None
    for cand in checkpoint_candidates:
        if cand and cand.exists():
            model_path = cand
            break

    if not model_path:
        raise FileNotFoundError(
            "Could not find a valid model checkpoint (.pt)! Please specify with --weights <path>"
        )

    print(f"\n--- 3. Loading Model Checkpoint ---")
    print(f"  -> Checkpoint Path: {model_path}")
    model = AudioVisualDeepFakeCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("  -> AudioVisualDeepFakeCNN initialized and weights loaded successfully!")

    def to_video_tensor(arr):
        t = torch.from_numpy(np.ascontiguousarray(arr)).permute(0, 1, 4, 2, 3).float().div_(127.5).sub_(1.0)
        return t.to(device)

    @torch.no_grad()
    def predict_proba_multimodal(cache_v, indices, mode="av", bs=args.batch_size):
        model.eval()
        stride = max(1, n_frames // clip_len)
        out = []
        for i in range(0, len(indices), bs):
            chunk = indices[i : i + bs]
            p = 0
            audio_chunk = torch.from_numpy(audio_cache[chunk]).to(device)
            for v in range(stride):
                xb = cache_v[chunk][:, v::stride][:, :clip_len]
                logits = model(to_video_tensor(xb), audio_chunk, mode=mode)
                p = p + torch.softmax(logits, 1)[:, 1].cpu().numpy()
            out.append(p / stride)
        return np.concatenate(out)

    all_indices = np.arange(n_test)

    # 5. Run Evaluations
    print("\n" + "=" * 84)
    print(f" EXECUTING EVALUATIONS ON {n_test} VIDEOS")
    print("=" * 84)

    # 5.1 Individual Variant Evaluation
    print(f"\n[1/3] Evaluating each of the {total_variants} degradation types across all {n_test} videos...")
    eval_results_av = {}
    eval_results_vo = {}
    pred_table = {
        "index": np.arange(n_test),
        "source_name": selected_source_names,
        "true_label": selected_labels,
        "class_name": ["Fake" if l == 1 else "Real" for l in selected_labels],
    }

    t0_eval = time.time()
    for var in active_variants:
        p_av = predict_proba_multimodal(variant_caches[var], all_indices, mode="av")
        p_vo = predict_proba_multimodal(variant_caches[var], all_indices, mode="v_only")
        pred_table[f"p_fake_av_{var}"] = p_av
        pred_table[f"p_fake_vo_{var}"] = p_vo
        eval_results_av[var] = calculate_metrics(selected_labels, p_av, thr=args.threshold)
        eval_results_vo[var] = calculate_metrics(selected_labels, p_vo, thr=args.threshold)

    # Audio-Only evaluation on clean audio
    p_ao = predict_proba_multimodal(variant_caches["clean"], all_indices, mode="a_only")
    pred_table["p_fake_ao"] = p_ao
    audio_only_metrics = calculate_metrics(selected_labels, p_ao, thr=args.threshold)

    # 5.2 Random In-The-Wild Mixed Degradation Assignment
    # For each video, assign a randomly chosen degradation type from the active variants
    print(f"\n[2/3] Generating Random In-The-Wild Mixed Test Set ({n_test} videos)...")
    rng_degrade = random.Random(args.seed + 999)
    # Ensure balanced or random distribution
    assigned_degradations = [rng_degrade.choice(active_variants) for _ in range(n_test)]
    pred_table["assigned_random_degradation"] = assigned_degradations

    p_random_mixed_av = np.array([pred_table[f"p_fake_av_{deg}"][i] for i, deg in enumerate(assigned_degradations)])
    p_random_mixed_vo = np.array([pred_table[f"p_fake_vo_{deg}"][i] for i, deg in enumerate(assigned_degradations)])

    pred_table["p_fake_random_mixed_av"] = p_random_mixed_av
    pred_table["pred_class_random_mixed_av"] = (p_random_mixed_av >= args.threshold).astype(int)
    pred_table["pred_label_random_mixed_av"] = ["Fake" if p == 1 else "Real" for p in pred_table["pred_class_random_mixed_av"]]
    pred_table["is_correct_random_mixed_av"] = [
        "CORRECT" if pred == true else "WRONG"
        for pred, true in zip(pred_table["pred_class_random_mixed_av"], selected_labels)
    ]

    random_mixed_metrics_av = calculate_metrics(selected_labels, p_random_mixed_av, thr=args.threshold)
    random_mixed_metrics_vo = calculate_metrics(selected_labels, p_random_mixed_vo, thr=args.threshold)

    # 5.3 Pooled Mixed Degradation Benchmark
    # Evaluates all 200 videos * 10 variants = 2,000 instances
    print(f"\n[3/3] Generating Pooled Mixed Degradation Benchmark ({n_test * total_variants:,} instances)...")
    y_test_pooled = np.tile(selected_labels, total_variants)
    p_pooled_av = np.concatenate([pred_table[f"p_fake_av_{var}"] for var in active_variants])
    p_pooled_vo = np.concatenate([pred_table[f"p_fake_vo_{var}"] for var in active_variants])
    p_pooled_ao = np.tile(p_ao, total_variants)

    pooled_mixed_metrics_av = calculate_metrics(y_test_pooled, p_pooled_av, thr=args.threshold)
    pooled_mixed_metrics_vo = calculate_metrics(y_test_pooled, p_pooled_vo, thr=args.threshold)
    pooled_mixed_metrics_ao = calculate_metrics(y_test_pooled, p_pooled_ao, thr=args.threshold)

    eval_duration = time.time() - t0_eval
    print(f"Completed all evaluations in {eval_duration:.2f} seconds!")

    # 6. Construct Summary Tables
    clean_auc = eval_results_av["clean"]["auc"]
    clean_acc = eval_results_av["clean"]["accuracy"]

    benchmark_rows = [
        {
            "Evaluation Benchmark": "Clean Videos (Authentic Quality)",
            "Samples": n_test,
            "AV AUC": eval_results_av["clean"]["auc"],
            "AV Accuracy": eval_results_av["clean"]["accuracy"],
            "AV F1": eval_results_av["clean"]["f1"],
            "Video-Only AUC": eval_results_vo["clean"]["auc"],
            "Video-Only Acc": eval_results_vo["clean"]["accuracy"],
            "Audio-Only AUC": audio_only_metrics["auc"],
            "Audio-Visual Gain": eval_results_av["clean"]["auc"] - eval_results_vo["clean"]["auc"],
        },
        {
            "Evaluation Benchmark": "Random In-The-Wild Mixed (1 Degradation/Video)",
            "Samples": n_test,
            "AV AUC": random_mixed_metrics_av["auc"],
            "AV Accuracy": random_mixed_metrics_av["accuracy"],
            "AV F1": random_mixed_metrics_av["f1"],
            "Video-Only AUC": random_mixed_metrics_vo["auc"],
            "Video-Only Acc": random_mixed_metrics_vo["accuracy"],
            "Audio-Only AUC": audio_only_metrics["auc"],
            "Audio-Visual Gain": random_mixed_metrics_av["auc"] - random_mixed_metrics_vo["auc"],
        },
        {
            "Evaluation Benchmark": "Pooled Mixed Degradation (All 10 Variants)",
            "Samples": n_test * total_variants,
            "AV AUC": pooled_mixed_metrics_av["auc"],
            "AV Accuracy": pooled_mixed_metrics_av["accuracy"],
            "AV F1": pooled_mixed_metrics_av["f1"],
            "Video-Only AUC": pooled_mixed_metrics_vo["auc"],
            "Video-Only Acc": pooled_mixed_metrics_vo["accuracy"],
            "Audio-Only AUC": pooled_mixed_metrics_ao["auc"],
            "Audio-Visual Gain": pooled_mixed_metrics_av["auc"] - pooled_mixed_metrics_vo["auc"],
        },
    ]
    df_benchmarks = pd.DataFrame(benchmark_rows).set_index("Evaluation Benchmark")

    breakdown_rows = []
    for var in active_variants:
        m_av = eval_results_av[var]
        m_vo = eval_results_vo[var]
        breakdown_rows.append(
            {
                "Degradation Variant": var.upper(),
                "AV AUC": m_av["auc"],
                "AV Accuracy": m_av["accuracy"],
                "AV Bal-Acc": m_av["balanced_acc"],
                "AV F1": m_av["f1"],
                "Delta AUC (vs Clean)": m_av["auc"] - clean_auc,
                "Delta Acc (vs Clean)": m_av["accuracy"] - clean_acc,
                "Video-Only AUC": m_vo["auc"],
                "Video-Only Acc": m_vo["accuracy"],
                "Audio Gain (AV - VO)": m_av["auc"] - m_vo["auc"],
            }
        )

    # Add Mixed Benchmarks into breakdown for complete view
    breakdown_rows.append(
        {
            "Degradation Variant": "-> RANDOM IN-THE-WILD <-",
            "AV AUC": random_mixed_metrics_av["auc"],
            "AV Accuracy": random_mixed_metrics_av["accuracy"],
            "AV Bal-Acc": random_mixed_metrics_av["balanced_acc"],
            "AV F1": random_mixed_metrics_av["f1"],
            "Delta AUC (vs Clean)": random_mixed_metrics_av["auc"] - clean_auc,
            "Delta Acc (vs Clean)": random_mixed_metrics_av["accuracy"] - clean_acc,
            "Video-Only AUC": random_mixed_metrics_vo["auc"],
            "Video-Only Acc": random_mixed_metrics_vo["accuracy"],
            "Audio Gain (AV - VO)": random_mixed_metrics_av["auc"] - random_mixed_metrics_vo["auc"],
        }
    )
    breakdown_rows.append(
        {
            "Degradation Variant": "-> POOLED MIXED (2,000) <-",
            "AV AUC": pooled_mixed_metrics_av["auc"],
            "AV Accuracy": pooled_mixed_metrics_av["accuracy"],
            "AV Bal-Acc": pooled_mixed_metrics_av["balanced_acc"],
            "AV F1": pooled_mixed_metrics_av["f1"],
            "Delta AUC (vs Clean)": pooled_mixed_metrics_av["auc"] - clean_auc,
            "Delta Acc (vs Clean)": pooled_mixed_metrics_av["accuracy"] - clean_acc,
            "Video-Only AUC": pooled_mixed_metrics_vo["auc"],
            "Video-Only Acc": pooled_mixed_metrics_vo["accuracy"],
            "Audio Gain (AV - VO)": pooled_mixed_metrics_av["auc"] - pooled_mixed_metrics_vo["auc"],
        }
    )
    df_breakdown = pd.DataFrame(breakdown_rows).set_index("Degradation Variant")

    print("\n" + "=" * 84)
    print(" BENCHMARK 1: OVERALL BENCHMARK COMPARISON ON 200 VIDEOS")
    print("=" * 84)
    print(df_benchmarks.round(4).to_string())

    print("\n" + "=" * 84)
    print(" BENCHMARK 2: DETAILED PER-DEGRADATION ROBUSTNESS & GAP ANALYSIS")
    print("=" * 84)
    print(df_breakdown.round(4).to_string())
    print("=" * 84)

    # 7. Print Sample of Individual Video Predictions
    predictions_df = pd.DataFrame(pred_table)
    csv_path = output_dir / f"test_200_videos_predictions.csv"
    predictions_df.to_csv(csv_path, index=False)
    print(f"\n[Saved Full Predictions CSV] -> {csv_path}")

    print("\nSample of Individual 200-Video Predictions (First 15 Videos):")
    sample_cols = [
        "index", "source_name", "class_name", "assigned_random_degradation",
        "p_fake_random_mixed_av", "pred_label_random_mixed_av", "is_correct_random_mixed_av"
    ]
    sample_display = predictions_df[sample_cols].head(15).copy()
    sample_display["source_name"] = sample_display["source_name"].apply(lambda s: s[:28] + "..." if len(s) > 31 else s)
    sample_display["p_fake_random_mixed_av"] = sample_display["p_fake_random_mixed_av"].round(4)
    print(sample_display.to_string(index=False))

    # 8. Publication-Quality Plots
    print("\nGenerating publication-quality visualization charts...")

    # Plot 1: ROC Curves
    fig, ax = plt.subplots(figsize=(8, 6.5))
    fpr_c, tpr_c, _ = roc_curve(selected_labels, pred_table["p_fake_av_clean"])
    ax.plot(fpr_c, tpr_c, color="#27ae60", lw=2.5, label=f"Clean AV (AUC = {clean_auc:.3f})")

    fpr_w, tpr_w, _ = roc_curve(selected_labels, p_random_mixed_av)
    ax.plot(fpr_w, tpr_w, color="#9b59b6", lw=3.0, linestyle="-", label=f"Random In-The-Wild AV (AUC = {random_mixed_metrics_av['auc']:.3f})")

    fpr_m, tpr_m, _ = roc_curve(y_test_pooled, p_pooled_av)
    ax.plot(fpr_m, tpr_m, color="#e74c3c", lw=2.5, linestyle="--", label=f"Pooled Mixed AV (AUC = {pooled_mixed_metrics_av['auc']:.3f})")

    fpr_vo, tpr_vo, _ = roc_curve(y_test_pooled, p_pooled_vo)
    ax.plot(fpr_vo, tpr_vo, color="#3498db", lw=2.0, linestyle=":", label=f"Pooled Mixed Video-Only (AUC = {pooled_mixed_metrics_vo['auc']:.3f})")

    ax.plot([0, 1], [0, 1], color="#7f8c8d", lw=1.5, linestyle="--", alpha=0.7)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    ax.set_ylabel("True Positive Rate (Recall / Sensitivity)", fontsize=11)
    ax.set_title(f"ROC Curves: 200-Video Mixed Degradation Test", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", fontsize=10)
    plt.tight_layout()
    roc_plot_path = output_dir / "test_200_roc_curves.png"
    plt.savefig(roc_plot_path, dpi=130)
    plt.close()

    # Plot 2: Multimodal Ablation Bar Chart
    fig, ax = plt.subplots(figsize=(12, 5.5))
    plot_labels = [v.upper() for v in active_variants] + ["RANDOM WILD", "POOLED MIXED"]
    av_aucs = [eval_results_av[v]["auc"] for v in active_variants] + [random_mixed_metrics_av["auc"], pooled_mixed_metrics_av["auc"]]
    vo_aucs = [eval_results_vo[v]["auc"] for v in active_variants] + [random_mixed_metrics_vo["auc"], pooled_mixed_metrics_vo["auc"]]

    x = np.arange(len(plot_labels))
    w = 0.38
    ax.bar(x - w / 2, av_aucs, w, label="Audio-Visual (Multimodal)", color="#2ecc71")
    ax.bar(x + w / 2, vo_aucs, w, label="Video-Only Ablation", color="#3498db")
    ax.axhline(audio_only_metrics["auc"], color="#e67e22", linestyle="--", lw=2, label=f"Audio-Only Baseline ({audio_only_metrics['auc']:.3f})")

    ax.set_ylabel("ROC-AUC Score", fontsize=11)
    ax.set_title("200-Video Degradation Robustness: Audio-Visual vs Video-Only Across All Conditions", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(plot_labels, rotation=30, ha="right", fontsize=9.5)
    ax.set_ylim([0.0, 1.05])
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    ax.legend(loc="lower right", fontsize=10)
    plt.tight_layout()
    ablation_plot_path = output_dir / "test_200_degradation_barchart.png"
    plt.savefig(ablation_plot_path, dpi=130)
    plt.close()

    # Plot 3: Triple Confusion Matrices (Clean vs Random In-the-Wild vs Pooled Mixed)
    cm_clean = confusion_matrix(selected_labels, (pred_table["p_fake_av_clean"] >= args.threshold).astype(int), labels=[0, 1])
    cm_wild = confusion_matrix(selected_labels, (p_random_mixed_av >= args.threshold).astype(int), labels=[0, 1])
    cm_pooled = confusion_matrix(y_test_pooled, (p_pooled_av >= args.threshold).astype(int), labels=[0, 1])

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    configs = [
        (cm_clean, f"Clean AV (N={n_test})\nAcc: {eval_results_av['clean']['accuracy']*100:.1f}%"),
        (cm_wild, f"Random Mixed AV (N={n_test})\nAcc: {random_mixed_metrics_av['accuracy']*100:.1f}%"),
        (cm_pooled, f"Pooled Mixed AV (N={len(y_test_pooled):,})\nAcc: {pooled_mixed_metrics_av['accuracy']*100:.1f}%"),
    ]

    for ax_cm, (cm, title) in zip(axes, configs):
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
    cm_plot_path = output_dir / "test_200_confusion_matrix.png"
    plt.savefig(cm_plot_path, dpi=130)
    plt.close()

    # 9. Save JSON Report
    now = datetime.now()
    date_str = f"{now.year % 100}-{now.month}-{now.day}"
    time_str = f"{now.hour:02d}.{now.minute:02d}"
    report_filename = f"report_test_200_mixed_date_{date_str}_time_{time_str}.json"

    report_data = {
        "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
        "date": date_str,
        "time": time_str,
        "model_checkpoint": str(model_path),
        "test_config": {
            "total_test_videos": n_test,
            "real_videos": len(selected_real),
            "fake_videos": len(selected_fake),
            "seed": args.seed,
            "decision_threshold": args.threshold,
            "total_variants": total_variants,
            "pooled_evaluations": len(y_test_pooled),
        },
        "benchmarks": {
            "clean_baseline": {
                "audio_visual": eval_results_av["clean"],
                "video_only": eval_results_vo["clean"],
                "audio_only": audio_only_metrics,
            },
            "random_wild_mixed": {
                "audio_visual": random_mixed_metrics_av,
                "video_only": random_mixed_metrics_vo,
            },
            "pooled_mixed": {
                "audio_visual": pooled_mixed_metrics_av,
                "video_only": pooled_mixed_metrics_vo,
                "audio_only": pooled_mixed_metrics_ao,
            },
        },
        "per_degradation_breakdown": {
            "audio_visual": eval_results_av,
            "video_only": eval_results_vo,
        },
        "summary_table_benchmarks": df_benchmarks.to_dict(orient="index"),
        "summary_table_breakdown": df_breakdown.to_dict(orient="index"),
    }

    for tdir in [report_dir, output_dir]:
        with open(tdir / report_filename, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2)
        with open(tdir / "latest_report_test_200_mixed.json", "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2)

    print(f"\nSaved Test Results & Reports:")
    print(f"  * Predictions CSV : {csv_path}")
    print(f"  * JSON Report     : {report_dir / report_filename}")
    print(f"  * Latest JSON     : {report_dir / 'latest_report_test_200_mixed.json'}")
    print(f"  * ROC Curves      : {roc_plot_path}")
    print(f"  * Robustness Plot : {ablation_plot_path}")
    print(f"  * Confusion Matrix: {cm_plot_path}")
    print("\n200-Video Mixed Degradation Testing completed successfully!")


if __name__ == "__main__":
    main()
