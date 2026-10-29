
import os, re, json, time, random, warnings, sys, subprocess
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
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)

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
    """Fetch GPU temperature in Celsius, cached for ttl seconds to keep overhead negligible."""
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
            timeout=1.5
        )
        _cached_gpu_temp = int(out.decode().strip().split()[0])
        _last_gpu_temp_check = now
        return _cached_gpu_temp
    except Exception:
        return _cached_gpu_temp


def check_gpu_thermal_guard(max_temp, pause_sec):
    """Check GPU temp. If >= max_temp, pause execution for pause_sec with live countdown. Returns paused duration."""
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


# ------------------------- CONFIG (edit here) -------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_DEFAULT_ROOT = _SCRIPT_DIR.parent / "BioDeepAV" if (_SCRIPT_DIR.parent / "BioDeepAV").exists() else Path(r"E:\Datasets\BioDeepAV")
_DEFAULT_WORK = _SCRIPT_DIR / "biodeep_run"
_DEFAULT_REPORT = _SCRIPT_DIR / "test"

DATA_ROOT  = Path(os.environ.get("BIODEEP_ROOT", str(_DEFAULT_ROOT)))
WORK_DIR   = Path(os.environ.get("BIODEEP_WORK", str(_DEFAULT_WORK)))   # cache, model, results
REPORT_DIR = Path(os.environ.get("BIODEEP_REPORT", str(_DEFAULT_REPORT)))    # folder for timestamped JSON reports

GPU_MAX_TEMP      = int(os.environ.get("BIODEEP_GPU_MAX_TEMP", 72))   # Pause if GPU reaches 70°C+
GPU_PAUSE_SECONDS = int(os.environ.get("BIODEEP_GPU_PAUSE", 300))    # 5 minutes pause = 300s



N_FRAMES   = 16        # frames sampled evenly from each video (cached)
IMG        = 96        # every frame is resized to IMG x IMG  (removes the resolution shortcut)
CLIP_LEN   = 8         # frames per training clip (random subset of the 16, order kept)
BATCH      = 16        # clips per step  (= 128 frames)
_epochs_env = os.environ.get("BIODEEP_EPOCHS", "").strip()
EPOCHS     = int(_epochs_env) if _epochs_env.isdigit() else 12
LR         = 1e-3
TEST_FRAC  = 0.20      # as requested: 20% of videos for testing
VAL_FRAC   = 0.10      # 10% of the remaining train videos, used only to pick the best epoch
SEED       = 42
_maxpc     = os.environ.get("BIODEEP_MAXPC", "").strip()
MAX_PER_CLASS = int(_maxpc) if _maxpc.isdigit() and int(_maxpc) > 0 else None
GROUP_SPLIT_REAL = True   # real clips cut from the same YouTube video never appear in both train and test
N_CPU = os.cpu_count() or 8
# ----------------------------------------------------------------------

WORK_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
(WORK_DIR / "report").mkdir(parents=True, exist_ok=True)
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.set_num_threads(N_CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device} | CPU threads: {N_CPU} | torch {torch.__version__}")


real_paths = sorted((DATA_ROOT / "real" / "videos").glob("*.mp4"))
fake_paths = sorted((DATA_ROOT / "fake" / "videos").glob("*.mp4"))
if MAX_PER_CLASS:
    rng0 = random.Random(SEED)
    real_paths = rng0.sample(real_paths, min(MAX_PER_CLASS, len(real_paths)))
    fake_paths = rng0.sample(fake_paths, min(MAX_PER_CLASS, len(fake_paths)))
print(f"real videos: {len(real_paths)} | fake videos: {len(fake_paths)}")
all_paths = real_paths + fake_paths
all_labels = np.array([0] * len(real_paths) + [1] * len(fake_paths))


def extract_clip(path, n_frames=N_FRAMES, size=IMG):
    """Return (frames uint8 [T,size,size,3] RGB, meta dict) or None if the video cannot be read."""
    try:
        cap = cv2.VideoCapture(str(path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
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
            cap.release(); cap = cv2.VideoCapture(str(path)); tmp = []
            while True:
                ok, fr = cap.read()
                if not ok: break
                tmp.append(cv2.resize(fr, (size, size), interpolation=cv2.INTER_AREA))
            if not tmp: return None
            total = len(tmp)
            frames = {j: tmp[j] for j in np.linspace(0, total - 1, n_frames).round().astype(int)}
        cap.release()
        keys = sorted(frames)
        want_list = np.linspace(0, keys[-1], n_frames).round().astype(int)
        seq = [frames[min(keys, key=lambda k: abs(k - t))] for t in want_list]     # nearest available frame
        arr = np.stack(seq)[..., ::-1].copy()                                       # BGR -> RGB
        meta = dict(width=w, height=h, fps=fps, n_frames=total,
                    duration=(total / fps if fps else 0.0), size_bytes=os.path.getsize(path))
        return arr, meta
    except Exception as e:
        print("skip", Path(path).name, e)
        return None


def build_cache(paths, desc="videos"):
    t0 = time.time()
    total = len(paths)
    print(f"\n[Extracting & Caching {desc.upper()}] ({total} videos | {N_CPU} worker threads)")
    results = [None] * total
    with ThreadPoolExecutor(max_workers=N_CPU) as ex:
        futures = {ex.submit(extract_clip, p): i for i, p in enumerate(paths)}
        completed = 0
        last_print = 0
        for fut in as_completed(futures):
            i = futures[fut]
            try:
                results[i] = fut.result()
            except Exception as e:
                print(f"\n  [Skip] {Path(paths[i]).name}: {e}")
                results[i] = None
            completed += 1
            now = time.time()
            if now - last_print >= 0.15 or completed == total:
                elapsed = now - t0
                rate = completed / max(0.001, elapsed)
                eta = (total - completed) / max(0.001, rate)
                bar = progress_bar(completed, total, bar_len=20)
                sys.stdout.write(
                    f"\r  {desc}: {bar} ({completed}/{total}) | {rate:5.1f} vid/s | "
                    f"Elapsed: {format_time(elapsed)} | ETA: {format_time(eta)}   "
                )
                sys.stdout.flush()
                last_print = now
    sys.stdout.write("\n")
    sys.stdout.flush()
    ok = [i for i, r in enumerate(results) if r is not None]
    X = np.stack([results[i][0] for i in ok])
    meta = pd.DataFrame([results[i][1] for i in ok])
    print(f"  Completed in {format_time(time.time() - t0)} ({len(ok)}/{total} videos successfully decoded).")
    return X, meta, ok


tag = f"T{N_FRAMES}_S{IMG}_n{len(all_paths)}"
cache_file = WORK_DIR / f"clean_{tag}.npz"
if cache_file.exists():
    print(f"\nFound existing cached frames: {cache_file.name}")
    print("  Loading cache from disk (extraction not needed)...")
    z = np.load(cache_file, allow_pickle=True)
    X, y, names = z["X"], z["y"], z["names"]
    meta = pd.read_csv(WORK_DIR / f"clean_{tag}_meta.csv")
    print(f"  Loaded cache: {cache_file.name} | shape {X.shape} ({X.nbytes / 1e9:.2f} GB)")
else:
    X, meta, ok = build_cache(all_paths, "clean")
    y = all_labels[ok]
    names = np.array([all_paths[i].name for i in ok])
    print(f"  Saving cache to {cache_file.name}...")
    np.savez(cache_file, X=X, y=y, names=names)
    meta.insert(0, "name", names); meta.insert(1, "label", y)
    meta.to_csv(WORK_DIR / f"clean_{tag}_meta.csv", index=False)
    print(f"  Saved cache {cache_file.name} ({X.nbytes / 1e9:.2f} GB).")
print(f"X: {X.shape}  ({X.nbytes / 1e9:.2f} GB)   real={int((y == 0).sum())} fake={int((y == 1).sum())}")


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold

m = meta.copy() if "name" in meta else meta.assign(name=names, label=y)
m["bytes_per_pixel_frame"] = m.size_bytes / (m.width * m.height * m.n_frames.clip(lower=1))
gray = X.mean(-1).astype(np.uint8)
m["lap_var"] = [np.mean([cv2.Laplacian(f, cv2.CV_64F).var() for f in v]) for v in gray]   # sharpness after resizing
m["mean_luma"] = X.mean((1, 2, 3, 4))
m["frame_diff"] = np.abs(np.diff(X.astype(np.int16), axis=1)).mean((1, 2, 3, 4))
feat_sets = {
    "resolution (width)": ["width"], "fps": ["fps"], "duration": ["duration"], "bytes/pixel": ["bytes_per_pixel_frame"],
    "sharpness after resize": ["lap_var"], "motion (frame diff)": ["frame_diff"], "mean brightness": ["mean_luma"],
}
rows = []
min_class = min((m.label == 0).sum(), (m.label == 1).sum())
n_splits = min(5, min_class)
if n_splits >= 2:
    skf = StratifiedKFold(n_splits, shuffle=True, random_state=SEED)
    for k, cols in feat_sets.items():
        F = m[cols].replace([np.inf, -np.inf], np.nan).fillna(0).values
        F = (F - F.mean(0)) / (F.std(0) + 1e-9)
        p = cross_val_predict(LogisticRegression(max_iter=1000), F, m.label.values, cv=skf, method="predict_proba")[:, 1]
        rows.append((k, roc_auc_score(m.label.values, p)))
    audit = pd.DataFrame(rows, columns=["single cheap feature", f"AUC ({n_splits}-fold, LogReg)"]).round(3)
    print(audit.to_string(index=False))
    print("\nRead this as: AUC near 1.0 = that feature alone almost separates real/fake, so a high CNN score can be partly a"
          " shortcut. Report it in the thesis as a dataset limitation.")
    audit.to_csv(WORK_DIR / "shortcut_audit.csv", index=False)
else:
    print(f"\n[Skipping shortcut audit: minimum class count {min_class} is too small for cross-validation]")


def yt_group(name):
    mm = re.match(r"^(.*)_\d{4}_S\d+_E\d+", name)
    return mm.group(1) if mm else name


def grouped_take(idx, groups, frac, rng):
    """Move whole groups into a held-out set until it holds ~frac of idx. Returns (kept, held)."""
    idx = list(idx); target = frac * len(idx)
    ug = sorted(set(groups[i] for i in idx)); rng.shuffle(ug)
    held, count = set(), 0
    for g in ug:
        if count >= target: break
        members = [i for i in idx if groups[i] == g]
        if count + len(members) > target * 1.5 and count > 0: continue      # avoid overshooting badly
        held.add(g); count += len(members)
    return [i for i in idx if groups[i] not in held], [i for i in idx if groups[i] in held]


rng = random.Random(SEED)
groups = np.array([yt_group(n) if (lab == 0 and GROUP_SPLIT_REAL) else n for n, lab in zip(names, y)])
train_idx, val_idx, test_idx = [], [], []
for cls in (0, 1):
    idx = [i for i in range(len(y)) if y[i] == cls]
    rest, te = grouped_take(idx, groups, TEST_FRAC, rng)
    tr, va = grouped_take(rest, groups, VAL_FRAC, rng)
    train_idx += tr; val_idx += va; test_idx += te
train_idx, val_idx, test_idx = map(lambda a: np.array(sorted(a)), (train_idx, val_idx, test_idx))
assert not (set(train_idx) & set(test_idx)) and not (set(val_idx) & set(test_idx)) and not (set(train_idx) & set(val_idx))
assert not (set(groups[train_idx]) & set(groups[test_idx])), "group leakage"
for nm, ix in (("train", train_idx), ("val", val_idx), ("test", test_idx)):
    print(f"{nm:5s}: {len(ix):5d} videos | real={int((y[ix] == 0).sum())} fake={int((y[ix] == 1).sum())}")
json.dump({"train": names[train_idx].tolist(), "val": names[val_idx].tolist(), "test": names[test_idx].tolist()},
          open(WORK_DIR / "split.json", "w"), indent=1)   # reuse the SAME test videos for the degraded datasets later


def block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1), nn.ReLU(inplace=True), nn.BatchNorm2d(cout),
        nn.Conv2d(cout, cout, 3, padding=1), nn.ReLU(inplace=True), nn.BatchNorm2d(cout),
        nn.MaxPool2d(2))


class DeepFakeVideoCNN(nn.Module):
    def __init__(self, width=(16, 32, 64, 128), num_classes=2):
        super().__init__()
        ch = (3,) + tuple(width)
        self.features = nn.Sequential(*[block(ch[i], ch[i + 1]) for i in range(len(width))])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(nn.Linear(2 * width[-1], 128), nn.ReLU(inplace=True), nn.Dropout(0.5),
                                  nn.Linear(128, num_classes))

    def forward(self, x):                                   # x: [B, T, 3, H, W]
        B, T, C, H, W = x.shape
        f = self.pool(self.features(x.reshape(B * T, C, H, W))).flatten(1).view(B, T, -1)
        return self.head(torch.cat([f.mean(1), f.amax(1)], dim=1))


def to_tensor(arr):                    
    t = torch.from_numpy(np.ascontiguousarray(arr)).permute(0, 1, 4, 2, 3).float().div_(127.5).sub_(1.0)
    return t.to(device)


model = DeepFakeVideoCNN().to(device)
print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")


def make_batch(X, idx, train, rng_np):
    if train:
        sel = np.stack([np.sort(rng_np.choice(X.shape[1], CLIP_LEN, replace=False)) for _ in idx])
        xb = X[idx[:, None], sel]                                        
        flip = rng_np.random(len(idx)) < 0.5
        xb[flip] = xb[flip][:, :, :, ::-1]
        pad = np.pad(xb, ((0, 0), (0, 0), (4, 4), (4, 4), (0, 0)), mode="edge")
        out = np.empty_like(xb)
        for b in range(len(idx)):
            dy, dx = rng_np.integers(0, 9, 2)
            out[b] = pad[b, :, dy:dy + IMG, dx:dx + IMG]
        return to_tensor(out)
    return to_tensor(X[idx])


@torch.no_grad()
def predict_proba(model, Xarr, idx, bs=32):
    """P(fake) per video. Averages several CLIP_LEN-frame views (frames 0,2,4.. and 1,3,5..) of the 16 cached frames."""
    model.eval()
    T = Xarr.shape[1]; stride = max(1, T // CLIP_LEN)
    out = []
    for i in range(0, len(idx), bs):
        chunk = np.asarray(idx[i:i + bs])
        p = 0
        for v in range(stride):
            xb = Xarr[chunk][:, v::stride][:, :CLIP_LEN]
            p = p + torch.softmax(model(to_tensor(xb)), 1)[:, 1].cpu().numpy()
        out.append(p / stride)
    return np.concatenate(out)


def metrics(y_true, p, thr=0.5):
    pred = (p >= thr).astype(int)
    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    return dict(accuracy=accuracy_score(y_true, pred), balanced_acc=balanced_accuracy_score(y_true, pred),
                precision=precision_score(y_true, pred, zero_division=0), recall=recall_score(y_true, pred, zero_division=0),
                f1=f1_score(y_true, pred, zero_division=0),
                auc=roc_auc_score(y_true, p) if len(set(y_true)) > 1 else float("nan"),
                tn=int(cm[0, 0]), fp=int(cm[0, 1]), fn=int(cm[1, 0]), tp=int(cm[1, 1]), threshold=float(thr))


counts = np.bincount(y[train_idx], minlength=2)
class_w = torch.tensor(counts.sum() / (2.0 * counts), dtype=torch.float32).to(device)
criterion = nn.CrossEntropyLoss(weight=class_w)
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
steps_per_epoch = int(np.ceil(len(train_idx) / BATCH))
scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LR, total_steps=EPOCHS * steps_per_epoch)
rng_np = np.random.default_rng(SEED)

history, best = [], dict(score=-1, epoch=-1)
best_path = WORK_DIR / "best_model.pt"
total_training_steps = EPOCHS * steps_per_epoch
gpu_temp_init = get_gpu_temp(ttl=0)
gpu_temp_str = f"{gpu_temp_init}C" if gpu_temp_init is not None else "N/A"
print("\n" + "=" * 78)
print(" TRAINING PIPELINE & PROGRESS")
print(f"  * Device          : {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
print(f"  * GPU Temp        : {gpu_temp_str} (Thermal guard: auto-pause 5 min if >= {GPU_MAX_TEMP}C)")
print(f"  * Train dataset   : {len(train_idx)} videos ({steps_per_epoch} steps/epoch, batch={BATCH})")
print(f"  * Val dataset     : {len(val_idx)} videos")
print(f"  * Test dataset    : {len(test_idx)} videos")
print(f"  * Schedule        : {EPOCHS} epochs | {total_training_steps} total optimization steps")
print("=" * 78 + "\n")

t_start = time.time()
overall_step = 0

for epoch in range(1, EPOCHS + 1):
    model.train()
    order = rng_np.permutation(train_idx)
    total_loss = 0.0
    t0 = time.time()
    last_print = 0

    for s in range(steps_per_epoch):
        overall_step += 1
        bi = order[s * BATCH:(s + 1) * BATCH]
        if len(bi) < 2:
            continue
        xb = make_batch(X, bi, True, rng_np)
        yb = torch.from_numpy(y[bi]).long().to(device)
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
            # Thermal guard check: pause 5 min if GPU temp >= GPU_MAX_TEMP (70°C+)
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
            total_eta = (total_training_steps - overall_step) / max(0.001, total_rate)

            bar = progress_bar(step_in_epoch, steps_per_epoch, bar_len=16)
            avg_loss = total_loss / step_in_epoch
            sys.stdout.write(
                f"\rEpoch [{epoch:2d}/{EPOCHS}] {bar} step {step_in_epoch:3d}/{steps_per_epoch} "
                f"| loss {avg_loss:.4f} | GPU: {gpu_str} | ep ETA: {format_time(epoch_eta)} | total ETA: {format_time(total_eta)}   "
            )
            sys.stdout.flush()
            last_print = now

    pv = predict_proba(model, X, val_idx)
    vm = metrics(y[val_idx], pv)
    score = vm["auc"] + vm["balanced_acc"]
    avg_train_loss = total_loss / max(1, steps_per_epoch)
    history.append(dict(epoch=epoch, train_loss=avg_train_loss, val_auc=vm["auc"], val_bal_acc=vm["balanced_acc"]))

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
    total_eta = (total_training_steps - overall_step) / max(0.001, total_rate)

    flag = ""
    if score > best["score"]:
        best = dict(score=score, epoch=epoch, auc=vm["auc"], bal_acc=vm["balanced_acc"])
        torch.save(model.state_dict(), best_path)
        flag = "  <-- BEST"

    sys.stdout.write(
        f"\rEpoch [{epoch:2d}/{EPOCHS}] loss {avg_train_loss:.4f} | "
        f"val AUC {vm['auc']:.4f} bal-acc {vm['balanced_acc']:.4f} | GPU: {gpu_str} | "
        f"time {format_time(epoch_elapsed)} | elapsed {format_time(total_elapsed)} | ETA {format_time(total_eta)}{flag}    \n"
    )
    sys.stdout.flush()

total_time = time.time() - t_start
print("-" * 78)
print(f"Training complete in {format_time(total_time)}! Best Epoch: {best['epoch']}")
print("-" * 78 + "\n")


# ## 7. Final test on the untouched 20 %


print("[Evaluating Final Clean Test Set...]")
model.load_state_dict(torch.load(best_path, map_location=device))
pv = predict_proba(model, X, val_idx)
# threshold chosen on the validation set only (helps when real/fake counts are unequal)
grid = np.linspace(0.05, 0.95, 91)
thr_val = float(grid[np.argmax([balanced_accuracy_score(y[val_idx], (pv >= t).astype(int)) for t in grid])])

pt = predict_proba(model, X, test_idx)
res = {"threshold_0.5": metrics(y[test_idx], pt, 0.5), "threshold_val_tuned": metrics(y[test_idx], pt, thr_val)}
print(pd.DataFrame(res).T.round(4).to_string())
clean_test_auc = res["threshold_0.5"]["auc"]

pd.DataFrame({"name": names[test_idx], "label": y[test_idx], "p_fake": pt}).to_csv(WORK_DIR / "test_predictions_clean.csv", index=False)

now = datetime.now()
date_str = f"{now.year % 100}-{now.month}-{now.day}"
time_str = f"{now.hour:02d}.{now.minute:02d}"
report_filename = f"report_date_{date_str}_time_{time_str}.json"

report_data = {
    "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
    "date": date_str,
    "time": time_str,
    "results": res,
    "best_epoch": best["epoch"],
    "val_threshold": thr_val,
    "training_time": format_time(total_time),
    "gpu_final_temp": f"{get_gpu_temp(ttl=0)}C" if get_gpu_temp(ttl=0) is not None else "N/A",
    "history": history,
    "config": dict(
        N_FRAMES=N_FRAMES, IMG=IMG, CLIP_LEN=CLIP_LEN, BATCH=BATCH, EPOCHS=EPOCHS, LR=LR, SEED=SEED,
        n_train=len(train_idx), n_val=len(val_idx), n_test=len(test_idx),
        GPU_MAX_TEMP=GPU_MAX_TEMP, GPU_PAUSE_SECONDS=GPU_PAUSE_SECONDS
    )
}

# Save to test/ folder in workspace and WORK_DIR
for rdir in (REPORT_DIR, WORK_DIR / "test"):
    rdir.mkdir(parents=True, exist_ok=True)
    with open(rdir / report_filename, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    with open(rdir / "latest_report.json", "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

json.dump(report_data, open(WORK_DIR / "results_clean.json", "w"), indent=1)
print(f"\n[Report saved] -> {REPORT_DIR / report_filename}")

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
h = pd.DataFrame(history)
ax[0].plot(h.epoch, h.train_loss, label="train loss"); ax[0].set_xlabel("epoch"); ax[0].legend()
ax[1].plot(h.epoch, h.val_auc, label="val AUC"); ax[1].plot(h.epoch, h.val_bal_acc, label="val balanced acc")
ax[1].set_xlabel("epoch"); ax[1].legend(); plt.tight_layout(); plt.savefig(WORK_DIR / "training_curves.png", dpi=120)
cm = confusion_matrix(y[test_idx], (pt >= 0.5).astype(int), labels=[0, 1])
fig, ax = plt.subplots(figsize=(3.6, 3.4)); ax.imshow(cm, cmap="Blues")
for (i, j), v in np.ndenumerate(cm): ax.text(j, i, v, ha="center", va="center", fontsize=14)
ax.set_xticks([0, 1]); ax.set_xticklabels(["pred real", "pred fake"]); ax.set_yticks([0, 1]); ax.set_yticklabels(["real", "fake"])
ax.set_title("Clean test set"); plt.tight_layout(); plt.savefig(WORK_DIR / "confusion_matrix_clean.png", dpi=120)
plt.close("all")
print(f"\nTest accuracy (video level): {res['threshold_0.5']['accuracy'] * 100:.2f}%   AUC: {clean_test_auc:.4f}")
