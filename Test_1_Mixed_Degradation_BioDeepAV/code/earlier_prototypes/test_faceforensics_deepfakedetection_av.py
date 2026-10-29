"""
Dedicated Standalone Testing & Evaluation Pipeline for FaceForensics++ DeepFakeDetection
Using Multimodal Audio-Visual Model (biodeep_run_mixed_av / best_model_mixed_av.pt)
========================================================================================
Features:
  1. Automated target detection: F:\\Datasets\\archive\\FaceForensics++_C23\\DeepFakeDetection
  2. Automatic checkpoint loading: biodeep_run_mixed_av\\best_model_mixed_av.pt
  3. Integrated YuNet Neural Face Cropper with fallback center crop
  4. Supports Video-Only (v_only) and Multimodal Audio-Visual (av) inference modes
  5. Multi-threaded video frame decoding with fast cap.grab() sequential reads
  6. Dual-stride 8-frame sub-clip inference with div(127.5).sub(1.0) normalization
  7. Spatial Grad-CAM explainability heatmaps on the final CNN convolutional block
  8. Optional --include-real flag to evaluate alongside FaceForensics++ original (real) videos
     computing full ROC-AUC, Balanced Accuracy, Precision, Recall, Specificity, and F1
  9. Exports audited CSV summary, JSON metrics report, and Grad-CAM visualization images
"""

import os
import sys
import time
import json
import urllib.request
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
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
    script_dir = Path(__file__).resolve().parent
    default_video_dir = Path(r"F:\Datasets\archive\FaceForensics++_C23\DeepFakeDetection")
    default_real_dir = Path(r"F:\Datasets\archive\FaceForensics++_C23\original")
    default_weights = script_dir / "biodeep_run_mixed_av" / "best_model_mixed_av.pt"
    if not default_weights.exists():
        cand = script_dir / "biodeep_run_mixed_av" / "best_model_mixed_av.pt"
        if cand.exists():
            default_weights = cand

    parser = argparse.ArgumentParser(
        description="Standalone Multimodal Deepfake Testing Pipeline for FaceForensics++ DeepFakeDetection"
    )
    parser.add_argument(
        "--video-dir",
        type=str,
        default=str(default_video_dir),
        help=f"Path to video directory to test (default: {default_video_dir})",
    )
    parser.add_argument(
        "--single-video",
        type=str,
        default=None,
        help="Path to a single video file to test individually",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=str(default_weights),
        help=f"Path to pre-trained model weights .pt (default: {default_weights})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="test_output_deepfakedetection_av",
        help="Directory to save CSV predictions, JSON report, and Grad-CAM heatmaps",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["v_only", "av"],
        default="v_only",
        help="Inference mode: 'v_only' (video-only) or 'av' (multimodal audio-visual). Default: v_only",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of videos to evaluate (e.g. 50, 100). Default is None (evaluates all)",
    )
    parser.add_argument(
        "--face-crop",
        dest="face_crop",
        action="store_true",
        default=True,
        help="Use YuNet neural face detection & cropping (default: enabled)",
    )
    parser.add_argument(
        "--no-face-crop",
        dest="face_crop",
        action="store_false",
        help="Disable face cropping and resize entire video frame",
    )
    parser.add_argument(
        "--include-real",
        action="store_true",
        help="Also test original (real) videos from FaceForensics++ to calculate full ROC-AUC, Precision, Recall, and Confusion Matrix",
    )
    parser.add_argument(
        "--real-dir",
        type=str,
        default=str(default_real_dir),
        help=f"Path to original (real) video directory if --include-real is enabled (default: {default_real_dir})",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decision threshold for classifying as FAKE (default: 0.5)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for GPU forward pass (default: 16)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of worker threads for parallel video decoding (default: CPU count * 2)",
    )
    parser.add_argument(
        "--cam-samples",
        type=int,
        default=10,
        help="Number of Grad-CAM visualization heatmaps to save (default: 10)",
    )
    parser.add_argument(
        "--save-all-cam",
        action="store_true",
        help="Save Grad-CAM visualization heatmaps for every video tested",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Inference device: 'cuda' or 'cpu' (default: cuda if available)",
    )
    return parser.parse_args()


# ------------------------- UTILITY FUNCTIONS -------------------------
def format_time(seconds):
    """Format seconds into HH:MM:SS or MM:SS."""
    if seconds is None or np.isnan(seconds) or seconds < 0:
        return "--:--"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h > 0 else f"{m:02d}:{s:02d}"


def progress_bar(current, total, bar_len=22):
    """Render an ASCII progress bar."""
    frac = min(1.0, max(0.0, current / max(1, total)))
    filled = int(frac * bar_len)
    bar = "=" * filled + (">" if filled < bar_len else "")
    bar = bar.ljust(bar_len, ".")
    return f"[{bar}] {frac * 100:5.1f}%"


def get_gpu_temp():
    """Query current GPU temperature via nvidia-smi."""
    if not torch.cuda.is_available():
        return None
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
            creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0,
            timeout=1.5,
        )
        return int(out.decode().strip().split()[0])
    except Exception:
        return None


# ------------------------- FACE DETECTOR SETUP -------------------------
def ensure_yunet_model(script_dir):
    """Ensures YuNet ONNX face detection model exists locally."""
    yunet_path = script_dir / "face_detection_yunet_2023mar.onnx"
    if yunet_path.exists() and yunet_path.stat().st_size > 100000:
        return str(yunet_path)

    print("  -> Downloading YuNet neural face detection model...")
    url = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = resp.read()
        with open(yunet_path, "wb") as f:
            f.write(data)
        print(f"  -> YuNet downloaded successfully ({len(data):,} bytes)")
        return str(yunet_path)
    except Exception as e:
        print(f"  [!] Warning: Failed to download YuNet ({e}). Face detection will fall back to center crop.")
        return None


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

    def forward(self, x_v, x_a=None, mode="v_only"):
        B = x_v.size(0) if x_v is not None else x_a.size(0)
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


# ------------------------- GRAD-CAM ENGINE -------------------------
def generate_gradcam_overlay(model_eval, clip_sub, mode="v_only", class_idx=1, target_frame=0, img_size=96):
    """
    Computes class-discriminative spatial Grad-CAM activation heatmap
    on the final convolutional block of VideoCNN.
    clip_sub: [1, 8, 3, 96, 96] torch float tensor normalized [-1, 1]
    """
    model_eval.eval()
    model_eval.zero_grad()

    activations = []
    gradients = []

    target_layer = model_eval.video_net.features[3]

    def forward_hook(module, inp, out):
        activations.append(out)

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    h1 = target_layer.register_forward_hook(forward_hook)
    h2 = target_layer.register_full_backward_hook(backward_hook)

    try:
        clip_tensor = clip_sub.clone().detach().requires_grad_(True)
        logits = model_eval(clip_tensor, mode=mode)
        score = logits[0, class_idx]
        score.backward()

        if not activations or not gradients:
            return np.zeros((img_size, img_size), dtype=np.float32)

        # Activations: [B*T, 128, 6, 6] -> target_frame corresponds to target_frame in T=8
        acts = activations[0][target_frame].detach()  # [128, 6, 6]
        grads = gradients[0][target_frame].detach()   # [128, 6, 6]

        weights = grads.mean(dim=(1, 2), keepdim=True)  # [128, 1, 1]
        cam = torch.relu((weights * acts).sum(dim=0)).cpu().numpy()  # [6, 6]
        cam_min, cam_max = cam.min(), cam.max()
        cam_norm = (cam - cam_min) / (cam_max - cam_min + 1e-8) if cam_max > cam_min else np.zeros_like(cam)
        cam_resized = cv2.resize(cam_norm, (img_size, img_size))
        return cam_resized
    finally:
        h1.remove()
        h2.remove()


# ------------------------- VIDEO & AUDIO EXTRACTION -------------------------
_THREAD_LOCAL_DETECTOR = None

def get_thread_detector(yunet_path):
    global _THREAD_LOCAL_DETECTOR
    if yunet_path is None:
        return None
    try:
        return cv2.FaceDetectorYN.create(yunet_path, "", (300, 300), score_threshold=0.55)
    except Exception:
        return None


def extract_video_clip(path, n_frames=16, size=96, crop_face=True, yunet_path=None):
    """
    Extracts equidistant video frames with smart face detection & cropping.
    Returns RGB uint8 numpy array [n_frames, size, size, 3].
    """
    try:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            return None
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        want = set(np.linspace(0, max(0, total - 1), n_frames).round().astype(int).tolist())
        last = max(want) if want else 0
        raw_frames = {}
        i = 0
        while i <= last:
            if not cap.grab():
                break
            if i in want:
                ok, fr = cap.retrieve()
                if ok and fr is not None:
                    raw_frames[i] = fr
            i += 1
        cap.release()

        if not raw_frames:
            # Fallback sequential read
            cap = cv2.VideoCapture(str(path))
            tmp = []
            while True:
                ok, fr = cap.read()
                if not ok or fr is None:
                    break
                tmp.append(fr)
            cap.release()
            if not tmp:
                return None
            total = len(tmp)
            indices = np.linspace(0, max(0, total - 1), n_frames).round().astype(int)
            raw_frames = {j: tmp[j] for j in indices}

        keys = sorted(raw_frames.keys())
        if not keys:
            return None

        # Detect face on initial frames
        bbox = None
        if crop_face and yunet_path is not None:
            detector = get_thread_detector(yunet_path)
            if detector is not None:
                for k in keys[:3]:
                    fr = raw_frames[k]
                    h, w = fr.shape[:2]
                    detector.setInputSize((w, h))
                    _, faces = detector.detect(fr)
                    if faces is not None and len(faces) > 0:
                        # Select largest face
                        best = max(faces, key=lambda f: f[2] * f[3])
                        x, y, fw, fh = best[0:4].astype(int)
                        # Add 25-30% contextual margin around face
                        pad_x = int(fw * 0.28)
                        pad_y = int(fh * 0.28)
                        x1 = max(0, x - pad_x)
                        y1 = max(0, y - pad_y)
                        x2 = min(w, x + fw + pad_x)
                        y2 = min(h, y + fh + pad_y)
                        if (x2 - x1) > 20 and (y2 - y1) > 20:
                            bbox = (x1, y1, x2, y2)
                            break

        processed = []
        want_list = np.linspace(0, keys[-1], n_frames).round().astype(int)
        for t in want_list:
            fr = raw_frames[min(keys, key=lambda k: abs(k - t))]
            if bbox is not None:
                x1, y1, x2, y2 = bbox
                fr = fr[y1:y2, x1:x2]
            elif crop_face:
                # Fallback center square crop
                h, w = fr.shape[:2]
                sz = min(h, w)
                cy, cx = h // 2, w // 2
                fr = fr[cy - sz // 2 : cy + sz // 2, cx - sz // 2 : cx + sz // 2]

            resized = cv2.resize(fr, (size, size), interpolation=cv2.INTER_AREA)
            processed.append(resized[..., ::-1])  # Convert BGR to RGB

        return np.stack(processed)
    except Exception as e:
        return None


def extract_audio_spectrogram(video_path, n_samples=48000):
    """
    Extracts 16kHz mono audio spectrogram [1, 128, 128] float32 tensor.
    If video has no audio, returns zero spectrogram.
    """
    audio_data = None
    try:
        cmd = [
            "ffmpeg", "-v", "error",
            "-i", str(video_path),
            "-vn", "-ac", "1", "-ar", "16000",
            "-f", "s16le", "-"
        ]
        raw_bytes = subprocess.check_output(
            cmd,
            creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0,
            timeout=8,
        )
        if len(raw_bytes) > 0:
            audio_data = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    except Exception:
        audio_data = None

    if audio_data is None or len(audio_data) == 0:
        return np.zeros((1, 128, 128), dtype=np.float32)

    if len(audio_data) < n_samples:
        reps = int(np.ceil(n_samples / len(audio_data)))
        audio_data = np.tile(audio_data, reps)[:n_samples]
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
        ).squeeze(0)
        spec = (spec - spec.mean()) / (spec.std() + 1e-6)
        spec = spec.clamp(-3.0, 3.0) / 3.0
        return spec.numpy().astype(np.float32)


# ------------------------- MAIN EVALUATION PIPELINE -------------------------
def main():
    args = parse_args()
    script_dir = Path(__file__).resolve().parent

    print("\n" + "=" * 82)
    print(" STANDALONE MULTIMODAL DEEPFAKE DETECTION TESTING PIPELINE")
    print(" FaceForensics++ C23 DeepFakeDetection Benchmark")
    print("=" * 82)
    print(f"  * Device             : {args.device}" + (f" ({torch.cuda.get_device_name(0)})" if args.device == "cuda" else ""))
    print(f"  * Model Checkpoint   : {args.weights}")
    print(f"  * Architecture       : AudioVisualDeepFakeCNN (biodeep_run_mixed_av)")
    print(f"  * Modality Mode      : {args.mode.upper()} ({'Video-Only' if args.mode == 'v_only' else 'Audio-Visual'})")
    print(f"  * Face Cropper       : {'YuNet Neural Face Detection (Enabled)' if args.face_crop else 'Full Frame / Disabled'}")
    print(f"  * Output Directory   : {args.output_dir}")
    print(f"  * Decision Threshold : {args.threshold}")
    print(f"  * Batch Size         : {args.batch_size}")
    print("=" * 82)

    device = torch.device(args.device)

    # 1. Load Model Checkpoint
    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found at: {weights_path}")

    print(f"\n[1/4] Loading model checkpoint from: {weights_path.name}")
    model = AudioVisualDeepFakeCNN().to(device)
    state = torch.load(str(weights_path), map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.eval()
    print("  -> AudioVisualDeepFakeCNN weights loaded successfully!")

    # 2. Setup YuNet face detector model
    yunet_path = ensure_yunet_model(script_dir) if args.face_crop else None

    # 3. Discover and prepare test queue
    print(f"\n[2/4] Scanning target directory: {args.video_dir}")
    test_queue = []

    if args.single_video:
        s_path = Path(args.single_video)
        if not s_path.exists():
            raise FileNotFoundError(f"Single video not found: {s_path}")
        test_queue.append((s_path, 1, "Single Video"))
    else:
        v_dir = Path(args.video_dir)
        if not v_dir.exists():
            raise FileNotFoundError(f"Target video directory does not exist: {v_dir}")

        exts = {".mp4", ".avi", ".mov", ".mkv"}
        fake_videos = sorted([p for p in v_dir.iterdir() if p.suffix.lower() in exts])
        if args.limit:
            fake_videos = fake_videos[: args.limit]

        print(f"  -> Found {len(fake_videos):,} video files in DeepFakeDetection")
        for p in fake_videos:
            test_queue.append((p, 1, "DeepFakeDetection (FAKE)"))

        if args.include_real:
            real_dir = Path(args.real_dir)
            if real_dir.exists():
                real_videos = sorted([p for p in real_dir.iterdir() if p.suffix.lower() in exts])
                limit_real = len(fake_videos) if args.limit is None else min(args.limit, len(real_videos))
                real_videos = real_videos[:limit_real]
                print(f"  -> Also included {len(real_videos):,} original (REAL) videos for complete ROC-AUC evaluation")
                for p in real_videos:
                    test_queue.append((p, 0, "original (REAL)"))
            else:
                print(f"  [!] Warning: Real video directory not found ({real_dir}). Testing fake videos only.")

    total_videos = len(test_queue)
    if total_videos == 0:
        print("No video files found to test! Exiting.")
        return

    print(f"\nTotal videos prepared for testing: {total_videos:,}")

    # Output directory setup
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cam_dir = out_dir / "gradcam_explanations"
    cam_dir.mkdir(parents=True, exist_ok=True)

    n_workers = args.workers or min(16, (os.cpu_count() or 4) * 2)
    print(f"[3/4] Starting inference with {n_workers} CPU worker threads on {device.type.upper()}...")

    n_frames = 16
    img_size = 96
    clip_len = 8
    stride = max(1, n_frames // clip_len)

    results = []
    cam_saved_count = 0
    t_start = time.time()
    last_print = 0

    batch_size = max(1, args.batch_size)

    for b_start in range(0, total_videos, batch_size):
        b_items = test_queue[b_start : b_start + batch_size]

        # Parallel extraction of video frames
        batch_clips = {}
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            fut_map = {
                executor.submit(
                    extract_video_clip,
                    item[0],
                    n_frames,
                    img_size,
                    args.face_crop,
                    yunet_path,
                ): (idx, item)
                for idx, item in enumerate(b_items)
            }
            for fut in as_completed(fut_map):
                idx, item = fut_map[fut]
                try:
                    batch_clips[idx] = fut.result()
                except Exception:
                    batch_clips[idx] = None

        # Parallel extraction of audio spectrograms if mode is 'av'
        batch_audios = {}
        if args.mode == "av":
            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                fut_map_a = {
                    executor.submit(extract_audio_spectrogram, item[0]): idx
                    for idx, item in enumerate(b_items)
                }
                for fut in as_completed(fut_map_a):
                    idx = fut_map_a[fut]
                    try:
                        batch_audios[idx] = fut.result()
                    except Exception:
                        batch_audios[idx] = np.zeros((1, 128, 128), dtype=np.float32)

        # Inference for each video in batch
        for idx, (v_path, ground_truth, cat) in enumerate(b_items):
            clip = batch_clips.get(idx)
            if clip is None:
                results.append({
                    "video_name": v_path.name,
                    "video_path": str(v_path),
                    "category": cat,
                    "ground_truth": ground_truth,
                    "predicted_label": "ERROR",
                    "predicted_int": -1,
                    "p_fake": float("nan"),
                    "confidence": float("nan"),
                    "gradcam_image": None,
                    "status": "decoding_failed",
                })
                continue

            # Multi-stride sub-clips [2, 8, 96, 96, 3]
            sub1 = clip[0::stride][:clip_len]
            sub2 = clip[1::stride][:clip_len]
            sub_batch = np.stack([sub1, sub2])

            # Convert to tensor and normalize to [-1, 1] matching training
            xv = torch.from_numpy(np.ascontiguousarray(sub_batch)).permute(0, 1, 4, 2, 3).float().div_(127.5).sub_(1.0).to(device)

            xa = None
            if args.mode == "av":
                audio_spec = batch_audios.get(idx, np.zeros((1, 128, 128), dtype=np.float32))
                # Tile for 2 sub-clips: [2, 1, 128, 128]
                xa = torch.from_numpy(np.repeat(audio_spec[np.newaxis, ...], 2, axis=0)).to(device)

            with torch.no_grad():
                logits = model(xv, xa, mode=args.mode)
                probs = torch.softmax(logits, dim=1)[:, 1].mean().item()

            p_fake = float(probs)
            pred_label = "FAKE" if p_fake >= args.threshold else "REAL"
            pred_int = 1 if pred_label == "FAKE" else 0
            conf = p_fake if pred_label == "FAKE" else (1.0 - p_fake)

            # Grad-CAM heatmap generation
            save_cam_file = None
            should_save_cam = args.save_all_cam or (cam_saved_count < args.cam_samples)
            if should_save_cam:
                try:
                    sub_cam = xv[:1]  # [1, 8, 3, 96, 96]
                    target_cls = 1 if p_fake >= 0.5 else 0
                    cam_map = generate_gradcam_overlay(
                        model, sub_cam, mode=args.mode, class_idx=target_cls, target_frame=0, img_size=img_size
                    )

                    orig_fr = sub1[0]
                    heat = cv2.applyColorMap(np.uint8(255 * cam_map), cv2.COLORMAP_JET)[..., ::-1]
                    overlay = np.uint8(orig_fr * 0.55 + heat * 0.45)

                    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.2))
                    axes[0].imshow(orig_fr)
                    axes[0].set_title(f"Extracted Frame 1 ({'Face' if args.face_crop else 'Full'})", fontsize=9.5)
                    axes[0].axis("off")

                    axes[1].imshow(overlay)
                    axes[1].set_title(f"Grad-CAM ({pred_label} Conf: {conf*100:.1f}%)", fontsize=9.5)
                    axes[1].axis("off")

                    fig.suptitle(f"{v_path.name[:32]} -> {pred_label} (P_fake={p_fake:.3f})", fontsize=11, fontweight="bold")
                    plt.tight_layout()

                    save_cam_file = cam_dir / f"{v_path.stem}_gradcam.png"
                    fig.savefig(save_cam_file, dpi=120, bbox_inches="tight")
                    plt.close(fig)
                    cam_saved_count += 1
                except Exception as cam_err:
                    save_cam_file = None

            results.append({
                "video_name": v_path.name,
                "video_path": str(v_path),
                "category": cat,
                "ground_truth": ground_truth,
                "predicted_label": pred_label,
                "predicted_int": pred_int,
                "p_fake": round(p_fake, 5),
                "confidence": round(conf, 5),
                "gradcam_image": save_cam_file.name if save_cam_file else None,
                "status": "success",
            })

        # Live Progress Rendering
        done = len(results)
        now = time.time()
        if now - last_print >= 0.25 or done == total_videos:
            elapsed = now - t_start
            rate = done / max(0.001, elapsed)
            eta = (total_videos - done) / max(0.001, rate)
            bar = progress_bar(done, total_videos, bar_len=22)
            n_fake = sum(1 for r in results if r["predicted_label"] == "FAKE")
            n_real = sum(1 for r in results if r["predicted_label"] == "REAL")
            temp = get_gpu_temp()
            temp_str = f" | GPU: {temp}C" if temp is not None else ""
            sys.stdout.write(
                f"\r  {bar} ({done}/{total_videos}) | {rate:4.1f} vid/s | Elapsed: {format_time(elapsed)} | ETA: {format_time(eta)} | Fake: {n_fake} Real: {n_real}{temp_str}   "
            )
            sys.stdout.flush()
            last_print = now

    total_time = time.time() - t_start
    print("\n")

    # 4. Compile Metrics and Export Reports
    df = pd.DataFrame(results)
    csv_path = out_dir / "test_predictions_deepfakedetection_av.csv"
    df.to_csv(csv_path, index=False)

    valid_df = df[df["status"] == "success"]
    n_valid = len(valid_df)
    n_errors = len(df) - n_valid

    if n_valid == 0:
        print("[!] No videos were successfully processed.")
        return

    n_fake_preds = int((valid_df["predicted_label"] == "FAKE").sum())
    n_real_preds = int((valid_df["predicted_label"] == "REAL").sum())
    fake_pct = (n_fake_preds / n_valid) * 100.0
    mean_p_fake = float(valid_df["p_fake"].mean())
    mean_conf = float(valid_df["confidence"].mean())

    has_both_classes = len(valid_df["ground_truth"].unique()) > 1
    y_true = valid_df["ground_truth"].values
    y_pred = valid_df["predicted_int"].values
    p_fake_arr = valid_df["p_fake"].values

    metrics = {}
    if has_both_classes:
        acc = float(accuracy_score(y_true, y_pred))
        b_acc = float(balanced_accuracy_score(y_true, y_pred))
        prec = float(precision_score(y_true, y_pred, zero_division=0))
        rec = float(recall_score(y_true, y_pred, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        try:
            auc = float(roc_auc_score(y_true, p_fake_arr))
        except Exception:
            auc = float("nan")

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

        metrics = {
            "accuracy": round(acc, 4),
            "balanced_accuracy": round(b_acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "specificity": round(spec, 4),
            "f1_score": round(f1, 4),
            "roc_auc": round(auc, 4),
            "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        }

        # Save ROC & Confusion Matrix plot
        fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))
        fpr, tpr, _ = roc_curve(y_true, p_fake_arr)
        axes[0].plot(fpr, tpr, color="#2563eb", lw=2, label=f"BioDeep-AV (AUC = {auc:.4f})")
        axes[0].plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6)
        axes[0].set_xlim([0.0, 1.0])
        axes[0].set_ylim([0.0, 1.05])
        axes[0].set_xlabel("False Positive Rate (1 - Specificity)")
        axes[0].set_ylabel("True Positive Rate (Recall)")
        axes[0].set_title("FaceForensics++ ROC Curve", fontweight="bold")
        axes[0].legend(loc="lower right")
        axes[0].grid(True, alpha=0.3)

        im = axes[1].imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        axes[1].figure.colorbar(im, ax=axes[1])
        classes = ["REAL", "FAKE"]
        axes[1].set(xticks=np.arange(2), yticks=np.arange(2), xticklabels=classes, yticklabels=classes)
        axes[1].set_ylabel("True Label", fontweight="bold")
        axes[1].set_xlabel("Predicted Label", fontweight="bold")
        axes[1].set_title(f"Confusion Matrix (Acc: {acc*100:.1f}%)", fontweight="bold")
        for r in range(2):
            for c in range(2):
                axes[1].text(c, r, format(cm[r, c], "d"), ha="center", va="center", color="white" if cm[r, c] > cm.max()/2 else "black")

        plt.tight_layout()
        roc_plot_path = out_dir / "evaluation_roc_confusion.png"
        fig.savefig(roc_plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        # Ground truth is all 1 (FAKE)
        recall_fake = float(n_fake_preds / n_valid)
        metrics = {
            "fake_detection_rate": round(recall_fake, 4),
            "accuracy": round(recall_fake, 4),
        }

    # Plot Distribution of Prediction Probabilities
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.hist(valid_df["p_fake"], bins=25, color="#3b82f6", edgecolor="black", alpha=0.75)
    ax.axvline(args.threshold, color="#dc2626", linestyle="--", lw=1.8, label=f"Decision Threshold ({args.threshold})")
    ax.set_title("FaceForensics++ DeepFakeDetection Probability Distribution", fontweight="bold")
    ax.set_xlabel("Predicted Probability of FAKE")
    ax.set_ylabel("Video Count")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    dist_plot_path = out_dir / "distribution_predictions.png"
    fig.savefig(dist_plot_path, dpi=130, bbox_inches="tight")
    plt.close(fig)

    report_data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_architecture": "AudioVisualDeepFakeCNN",
        "checkpoint_path": str(weights_path),
        "target_path": str(args.video_dir),
        "modality_mode": args.mode,
        "face_cropping_enabled": args.face_crop,
        "total_videos_tested": len(df),
        "successfully_decoded": n_valid,
        "decoding_errors": n_errors,
        "classification_summary": {
            "predicted_fake": n_fake_preds,
            "predicted_real": n_real_preds,
            "fake_percentage": round(fake_pct, 2),
            "mean_fake_probability": round(mean_p_fake, 4),
            "mean_confidence": round(mean_conf, 4),
            "threshold": args.threshold,
        },
        "metrics": metrics,
        "execution_summary": {
            "total_elapsed_time_sec": round(total_time, 2),
            "throughput_videos_per_sec": round(n_valid / max(0.001, total_time), 2),
            "gradcam_images_generated": cam_saved_count,
        },
        "predictions_csv": str(csv_path),
    }

    report_path = out_dir / "test_report_deepfakedetection_av.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    # 5. Print Results Summary
    print("=" * 82)
    print(" EVALUATION RESULTS SUMMARY")
    print("=" * 82)
    print(f"  * Total Videos Tested         : {len(df):,}")
    print(f"  * Successfully Processed      : {n_valid:,} (Errors: {n_errors})")
    print(f"  * Total Inference Time        : {format_time(total_time)} ({n_valid / max(0.001, total_time):.1f} vid/s)")
    print(f"  * Predicted as FAKE           : {n_fake_preds:,} ({fake_pct:.1f}%)")
    print(f"  * Predicted as REAL           : {n_real_preds:,} ({100.0 - fake_pct:.1f}%)")
    print(f"  * Average Fake Probability    : {mean_p_fake:.4f}")
    print(f"  * Average Prediction Conf     : {mean_conf * 100:.1f}%")

    if has_both_classes:
        print(f"  * Overall Accuracy            : {metrics['accuracy']*100:.2f}%")
        print(f"  * Balanced Accuracy           : {metrics['balanced_accuracy']*100:.2f}%")
        print(f"  * Precision (Fake)            : {metrics['precision']*100:.2f}%")
        print(f"  * Recall / FAKE Detection Rate: {metrics['recall']*100:.2f}%")
        print(f"  * Specificity (Real Accuracy) : {metrics['specificity']*100:.2f}%")
        print(f"  * F1-Score                    : {metrics['f1_score']:.4f}")
        print(f"  * ROC-AUC                     : {metrics['roc_auc']:.4f}")
        print(f"  * Confusion Matrix            : TN={metrics['confusion_matrix']['tn']}, FP={metrics['confusion_matrix']['fp']}, FN={metrics['confusion_matrix']['fn']}, TP={metrics['confusion_matrix']['tp']}")
    else:
        print(f"  * FAKE Detection Rate (Recall): {metrics['fake_detection_rate']*100:.2f}%")

    print("\n  Outputs Generated:")
    print(f"  -> Predictions CSV : {csv_path}")
    print(f"  -> Detailed JSON   : {report_path}")
    print(f"  -> P(Fake) Plot    : {dist_plot_path}")
    if has_both_classes:
        print(f"  -> ROC / CM Plot   : {out_dir / 'evaluation_roc_confusion.png'}")
    print(f"  -> Grad-CAM Images : {cam_dir} ({cam_saved_count} saved)")
    print("=" * 82 + "\n")


if __name__ == "__main__":
    main()
