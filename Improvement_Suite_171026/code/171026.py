"""
171026.py - TEST 5.7 IMPROVEMENT SUITE FOR THE REAL / FAKE VIDEO DETECTOR (follow-up of the basic retest, 151026.py)
==================================================================================================================

Six experiments, one script. All of them use the SAME data as the basic retest (original real / fake videos of
BioDeepAV and FaceForensics++ C23, no degraded videos): the face cache, inventory and split of run_151026 are re-used
read-only (no 19 GB re-extraction), and the model / preprocessing / augmentation code is imported from 151026.py so
the baseline is the identical pipeline.

  E1  Baseline reproduction   re-run seed 42 with the 151026 settings; verify (a) the split is re-derived identically,
                              (b) the same checkpoint epoch is selected, (c) validation / test metrics agree with run_151026
                              (exact equality is impossible on GPUs with cuDNN autotuning; tolerances are reported).
  E2  Three-seed evaluation   seeds 42, 123, 2026 of the baseline -> mean +- SD of every metric.
  E3  Hard-type sampling      ONLY the training sampler changes:
                                hardfix - fixed weights per FF++ method / BioDeepAV class from the basic-retest VALIDATION
                                          miss rates (NeuralTextures, Face2Face, FaceShifter, FaceSwap get more samples)
                                harddro - online group-DRO: per-cell weights follow the running TRAINING loss
                                          (no validation data used), with a 50 % uniform floor
                              3 seeds each; paired comparison with the baseline (same seeds, same test videos).
  E4  Longer training         12 (= E2) vs 20 vs 30 epochs (cosine schedule stretched), early stopping patience 5.
  E5  Threshold + calibration post-hoc on every run: raw / temperature / Platt / isotonic calibration; thresholds 0.5,
                              validation mean-balanced-accuracy, validation Youden; selective prediction ("uncertain"
                              band) tuned on validation for 100 / 95 / 90 / 80 % coverage. Reports coverage, ECE, Brier,
                              balanced accuracy and risk-coverage curves on the test set.
  E6  XAI faithfulness        Grad-CAM++ vs Integrated Gradients vs occlusion vs a random map on balanced test videos:
                              deletion and insertion curves with two perturbations (blur and grey - so blur-sensitivity is
                              not mistaken for faithfulness), area under the curves, top-20 % drop, face-centre energy,
                              and a model-randomisation sanity check (maps must change when the weights are randomised).

Statistics: every configuration is compared with the baseline on the SAME test videos and seeds (paired cluster
bootstrap over identity/duplicate groups, 1,000 replicates); an effect counts as "supported" only if the 95 % CI
excludes 0 AND all seeds agree in sign. The best configuration (validation only) is exported as best_model.pt and
can be used exactly like 151026 (--predict / --phase gui), now with an extra Integrated-Gradients explanation.

Usage (from E:\\Code, after 151026.py has run)
  python 171026.py --phase selftest
  python 171026.py                         # all experiments (resumable; ~14 h on an RTX 3050)
  python 171026.py --only E1               # one experiment (E1..E6); --only E1,E2 also works
  python 171026.py --predict "video.mp4"   # best model of this suite
  python 171026.py --phase gui
  python 171026.py --smoke                 # tiny run on run_151026_smoke (pipeline check)
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy import sparse  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.isotonic import IsotonicRegression  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score, roc_curve  # noqa: E402

SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent
DEFAULT_ROOT = SCRIPT_DIR / "run_171026"
SMOKE_ROOT = SCRIPT_DIR / "run_171026_smoke"
BASE_SCRIPT = SCRIPT_DIR / "151026.py"
BASE_ROOT = SCRIPT_DIR / "run_151026"
BASE_SMOKE_ROOT = SCRIPT_DIR / "run_151026_smoke"

SEEDS = [42, 123, 2026]
CONFIGS = {   # name: sampler, epochs, patience, experiment
    "base12": dict(sampler="baseline", epochs=12, patience=4, exp="E1/E2", label="Baseline (151026 settings), 12 epochs"),
    "hardfix12": dict(sampler="hardfix", epochs=12, patience=4, exp="E3", label="Fixed hard-type sampler, 12 epochs"),
    "harddro12": dict(sampler="harddro", epochs=12, patience=4, exp="E3", label="Group-DRO hard-type sampler, 12 epochs"),
    "base20": dict(sampler="baseline", epochs=20, patience=5, exp="E4", label="Baseline sampler, 20 epochs, patience 5"),
    "base30": dict(sampler="baseline", epochs=30, patience=5, exp="E4", label="Baseline sampler, 30 epochs, patience 5"),
}
EXPERIMENTS = ["E1", "E2", "E3", "E4", "E5", "E6"]
EXP_CONFIGS = {"E1": ["base12"], "E2": ["base12"], "E3": ["hardfix12", "harddro12"], "E4": ["base20", "base30"]}
COMPARISONS = [("hardfix12", "base12"), ("harddro12", "base12"), ("base20", "base12"), ("base30", "base12")]
CONFIG_COLOR = {"base12": "#8a8984", "hardfix12": "#eb6834", "harddro12": "#4a3aa7", "base20": "#2a78d6", "base30": "#1baf7a"}
HARD_FLOOR, DRO_ETA, DRO_FLOOR, DRO_MOM = 0.05, 2.0, 0.5, 0.5
N_BOOT = 1000
REPRO_TOL = dict(auc=0.010, accuracy=0.015)
CAL_METHODS = ["raw", "temperature", "platt", "isotonic"]
THR_METHODS = ["fixed_0.5", "val_mean_bacc", "val_youden"]
COVERAGES = [1.00, 0.95, 0.90, 0.80]
XAI_METHODS = ["gradcam++", "integrated_gradients", "occlusion", "random"]
XAI_FRACS = [0.0, 0.05, 0.10, 0.20, 0.30, 0.50, 0.75]
IG_STEPS = 24
XAI_N = 160


# =====================================================================================
# BASE PIPELINE (151026.py) - imported so the baseline is byte-identical code
# =====================================================================================
def load_base(path: Path):
    if not path.exists():
        raise SystemExit(f"[STOP] {path} not found - 171026.py needs 151026.py in the same folder (or --base-script)")
    spec = importlib.util.spec_from_file_location("base151026", str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["base151026"] = mod
    spec.loader.exec_module(mod)
    return mod


B = None  # base module (set in main)


def sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


class Paths:
    def __init__(self, root: Path):
        self.root = Path(root)
        for d in ("runs", "checkpoints", "metrics", "figures", "reports", "logs", "xai", "predictions", "repro"):
            setattr(self, d, self.root / d)
        self.best = self.checkpoints / "best_model.pt"

    def make(self):
        for d in ("runs", "checkpoints", "metrics", "figures", "reports", "logs", "xai", "predictions", "repro"):
            getattr(self, d).mkdir(parents=True, exist_ok=True)


# =====================================================================================
# DATA (read-only re-use of run_151026)
# =====================================================================================
def load_base_data(a):
    BP = B.Paths(a.base_root)
    for f in (BP.inventory, BP.split_csv):
        if not f.exists():
            raise SystemExit(f"[STOP] {f} missing - run 151026.py first (its face cache and split are re-used)")
    inv = pd.read_csv(BP.inventory)
    sp = pd.read_csv(BP.split_csv)
    C = B.FaceCache(BP, inv, a.img, a.face_px)
    if not C.complete():
        raise SystemExit(f"[STOP] face cache {C.arr_p.name} incomplete in {BP.cache} (check --img / --face-px)")
    C.load()
    D = B.Data(sp, C)
    D.subset = sp.subset.values
    D.group = sp.group.values
    D.fcell = np.array([f"{d}|{s}" for d, s in zip(sp.dataset, sp.subset)])   # dataset x folder cell
    print(f"  data from {a.base_root}: {len(sp):,} videos | train {len(D.idx('train')):,} | val {len(D.idx('val')):,} | "
          f"test {len(D.idx('test')):,} | cache {C.arr_p.name}")
    return BP, inv, C, D


# =====================================================================================
# SAMPLERS (E3) - the ONLY thing that differs between base12 and hardfix12 / harddro12
# =====================================================================================
def hardfix_weights(a, D, tr: np.ndarray, epoch_size: int) -> Tuple[np.ndarray, dict]:
    """Cell (dataset x folder) shares: each dataset keeps 50 % real / 50 % fake; within the fake half, folders are
    weighted by their basic-retest VALIDATION miss rate (floor HARD_FLOOR). Source: run_151026/metrics/predictions.csv."""
    pv = pd.read_csv(a.base_root / "metrics" / "predictions.csv")
    pv = pv[pv.split == "val"]
    miss = {f"{d}|{s}": float((g.pred != g.label).mean()) for (d, s), g in pv.groupby(["dataset", "subset"])}
    cells = D.fcell[tr]
    share = {}
    for ds in np.unique(D.ds[tr]):
        real = sorted({c for c, l in zip(cells, D.y[tr]) if c.startswith(ds + "|") and l == 0})
        fake = sorted({c for c, l in zip(cells, D.y[tr]) if c.startswith(ds + "|") and l == 1})
        wf = {c: max(HARD_FLOOR, miss.get(c, HARD_FLOOR)) for c in fake}
        for c in real:
            share[c] = 0.5 / len(DATASETS_PRESENT(D, tr)) / len(real)
        for c in fake:
            share[c] = 0.5 / len(DATASETS_PRESENT(D, tr)) * wf[c] / sum(wf.values())
    cnt = pd.Series(cells).value_counts().to_dict()
    w = np.array([share[c] / cnt[c] for c in cells])
    return w / w.sum(), dict(cell_share=share, val_miss_rate=miss)


def DATASETS_PRESENT(D, tr):
    return sorted(np.unique(D.ds[tr]))


def dro_weights(D, tr: np.ndarray, cell_loss: Dict[str, float]) -> Tuple[np.ndarray, dict]:
    """Online group-DRO over dataset x folder cells (training loss only): q ~ exp(eta * L/mean L), 50 % uniform floor."""
    cells = D.fcell[tr]
    names = sorted(set(cells))
    L = np.array([cell_loss.get(c, 1.0) for c in names])
    q = np.exp(DRO_ETA * (L / max(L.mean(), 1e-6) - 1))
    q /= q.sum()
    p = DRO_FLOOR / len(names) + (1 - DRO_FLOOR) * q
    share = dict(zip(names, p / p.sum()))
    cnt = pd.Series(cells).value_counts().to_dict()
    w = np.array([share[c] / cnt[c] for c in cells])
    return w / w.sum(), dict(cell_share=share, cell_loss=dict(zip(names, L)))


# =====================================================================================
# TRAINER (mirrors 151026 Trainer.train; sampler / epochs / patience are parameters)
# =====================================================================================
class Run:
    def __init__(self, a, P: Paths, D, config: str, seed: int):
        self.a, self.P, self.D, self.config, self.seed = a, P, D, config, seed
        self.cfg = CONFIGS[config]
        self.id = f"{config}_s{seed}"
        self.dir = P.runs / self.id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.amp = self.dev.type == "cuda" and not a.no_amp
        self.ck_best, self.ck_last = self.dir / "best.pt", self.dir / "last.pt"
        self.pred_csv = self.dir / "predictions.csv"

    def batch(self, items, seed, train=True):
        rng = np.random.default_rng(seed)
        clips, tix = [], []
        for i in items:
            if train:
                sel = np.sort(rng.choice(B.N_FRAMES, B.CLIP_FRAMES, replace=False))
                clips.append(B.augment_clip(np.asarray(self.D.C.arr[i, sel]), rng))
            else:
                sel = np.arange(B.N_FRAMES)
                clips.append(np.asarray(self.D.C.arr[i]))
            tix.append(sel)
        return np.stack(clips), np.stack(tix), self.D.y[items], np.asarray(items)

    @torch.no_grad()
    def predict(self, model, idx, bs=8):
        model.eval()
        out = []
        for s in range(0, len(idx), bs):
            x, t, _, _ = self.batch(idx[s:s + bs], 0, train=False)
            with torch.autocast("cuda", dtype=torch.float16, enabled=self.amp):
                o = model(B.normalise(x, self.dev), torch.from_numpy(t).to(self.dev))
            out.append(o["logit"].float().cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0)

    def validate(self, model):
        va = self.D.idx("val")
        lg = self.predict(model, va)
        row = {}
        for d in B.DATASETS:
            m = self.D.ds[va] == d
            if m.any() and len(np.unique(self.D.y[va][m])) > 1:
                row[f"val_auc_{d}"] = roc_auc_score(self.D.y[va][m], lg[m])
        row["val_score"] = float(np.mean([v for k, v in row.items() if k.startswith("val_auc_")])) if row else np.nan
        return row

    def done(self) -> bool:
        return self.ck_best.exists() and self.pred_csv.exists() and torch.load(self.ck_best, map_location="cpu", weights_only=False).get("final", False)

    def train(self):
        a, D, cfg = self.a, self.D, self.cfg
        if self.done():
            print(f"  [resume] {self.id} already finished")
            return
        print("\n" + "-" * 90 + f"\n RUN {self.id}: {cfg['label']} (sampler {cfg['sampler']})\n" + "-" * 90)
        if a.deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        B.set_seed(self.seed)
        if a.deterministic:
            torch.backends.cudnn.benchmark = False
        model = B.DeepfakeDetector(pretrained=not a.no_pretrained).to(self.dev)
        bb = list(model.backbone.parameters())
        other = [p for n, p in model.named_parameters() if not n.startswith("backbone.")]
        opt = torch.optim.AdamW([dict(params=bb, lr=B.LR_BACKBONE), dict(params=other, lr=B.LR_HEAD)], weight_decay=B.WEIGHT_DECAY)
        tr = D.idx("train")
        steps = int(math.ceil(len(tr) / (a.batch * a.accum))) * a.accum
        if a.max_steps:
            steps = min(steps, a.max_steps * a.accum)
        epochs = min(cfg["epochs"], a.max_epochs) if a.max_epochs else cfg["epochs"]
        total_upd = max(1, steps // a.accum * epochs)
        warm = max(1, int(B.WARMUP_FRAC * total_upd))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda u: (u + 1) / warm if u < warm else
                                                  0.5 * (1 + math.cos(math.pi * min(1.0, (u - warm) / max(1, total_upd - warm)))))
        scaler = torch.amp.GradScaler("cuda", enabled=self.amp)
        ema = B.EMA(model, B.EMA_DECAY)
        hist, best, start_ep, updates, bad = [], dict(score=-1.0, epoch=0), 1, 0, 0
        cell_loss = {c: 1.0 for c in set(D.fcell[tr])}
        samp_log = []
        if self.ck_last.exists():
            st = torch.load(self.ck_last, map_location=self.dev, weights_only=False)
            model.load_state_dict(st["model"])
            ema.model.load_state_dict(st["ema"])
            opt.load_state_dict(st["opt"])
            sched.load_state_dict(st["sched"])
            scaler.load_state_dict(st["scaler"])
            hist, best, start_ep, updates, bad = st["hist"], st["best"], st["epoch"] + 1, st["updates"], st["bad"]
            cell_loss, samp_log = st.get("cell_loss", cell_loss), st.get("samp_log", [])
            print(f"  [resume] {self.id}: continuing at epoch {start_ep}")
        if cfg["sampler"] == "baseline":
            w, winfo = B.balanced_weights(D.cell[tr], steps * a.batch), dict(cells="dataset x class (151026)")
        elif cfg["sampler"] == "hardfix":
            w, winfo = hardfix_weights(a, D, tr, steps * a.batch)
        bce = nn.BCEWithLogitsLoss()
        bce_n = nn.BCEWithLogitsLoss(reduction="none")
        loader = ThreadPoolExecutor(max_workers=max(1, a.loader_threads))
        print(f"  train {len(tr):,} | val {len(D.idx('val')):,} | {steps} steps x {epochs} epochs (patience {cfg['patience']}) | "
              f"{a.batch} clips x {B.CLIP_FRAMES} frames x accum {a.accum} | AMP {self.amp} | deterministic {a.deterministic}")
        t_run = time.time()
        for ep in range(start_ep, epochs + 1):
            if bad >= cfg["patience"]:
                print(f"  early stop: no validation improvement for {cfg['patience']} epochs")
                break
            if cfg["sampler"] == "harddro":
                w, winfo = dro_weights(D, tr, cell_loss)
            samp_log.append(dict(epoch=ep, **{f"share_{k}": v for k, v in winfo.get("cell_share", {}).items()}))
            model.train()
            rng = np.random.default_rng([self.seed, ep])
            draws = rng.choice(tr, size=steps * a.batch, replace=True, p=w)
            seeds = rng.integers(0, 2 ** 31, steps)
            q, nxt = deque(), 0

            def submit():
                nonlocal nxt
                if nxt < steps:
                    q.append(loader.submit(self.batch, draws[nxt * a.batch:(nxt + 1) * a.batch], int(seeds[nxt])))
                    nxt += 1

            for _ in range(2 * a.loader_threads + 1):
                submit()
            tot, t0, last = 0.0, time.time(), 0
            lsum, lcnt = {}, {}
            opt.zero_grad(set_to_none=True)
            for st in range(steps):
                x, tix, y, items = q.popleft().result()
                submit()
                ys = torch.from_numpy(y).float().to(self.dev) * (1 - B.LABEL_SMOOTH) + 0.5 * B.LABEL_SMOOTH
                with torch.autocast("cuda", dtype=torch.float16, enabled=self.amp):
                    o = model(B.normalise(x, self.dev), torch.from_numpy(tix).to(self.dev))
                    loss = bce(o["logit"], ys) + B.FRAME_LOSS_W * bce(o["frame_logit"], ys[:, None].expand_as(o["frame_logit"]))
                if cfg["sampler"] == "harddro":
                    with torch.no_grad():
                        per = bce_n(o["logit"].float(), ys).cpu().numpy()
                    for c, l_ in zip(D.fcell[items], per):
                        lsum[c] = lsum.get(c, 0.0) + float(l_)
                        lcnt[c] = lcnt.get(c, 0) + 1
                scaler.scale(loss / a.accum).backward()
                if (st + 1) % a.accum == 0:
                    scaler.unscale_(opt)
                    nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    scaler.step(opt)
                    scaler.update()
                    opt.zero_grad(set_to_none=True)
                    if updates < total_upd - 1:
                        sched.step()
                    updates += 1
                    ema.update(model)
                tot += float(loss.item())
                if time.time() - last > 0.5 or st + 1 == steps:
                    el = time.time() - t0
                    sys.stdout.write(f"\r  [{self.id}] ep {ep}/{epochs} {B.bar(st + 1, steps, 16)} loss {tot / (st + 1):.4f} | "
                                     f"epoch ETA {B.fmt_time(el / (st + 1) * (steps - st - 1))}   ")
                    sys.stdout.flush()
                    last = time.time()
            for c in lsum:
                cell_loss[c] = DRO_MOM * cell_loss.get(c, 1.0) + (1 - DRO_MOM) * lsum[c] / lcnt[c]
            row = dict(epoch=ep, train_loss=tot / steps, epoch_time_s=time.time() - t0, **self.validate(ema.model))
            hist.append(row)
            better = np.isfinite(row["val_score"]) and row["val_score"] > best["score"]
            if better:
                best, bad = dict(score=row["val_score"], epoch=ep), 0
                B.atomic_save(dict(model=ema.model.state_dict(), epoch=ep, val=row), self.ck_best.with_suffix(".tmp"))
            else:
                bad += 1
            print(f"\r  [{self.id}] ep {ep}/{epochs} | loss {row['train_loss']:.4f} | "
                  + " | ".join(f"{k.replace('val_auc_', 'val AUC ')} {v:.4f}" for k, v in row.items() if k.startswith("val_auc_"))
                  + f" | score {row['val_score']:.4f}{'  <- best' if better else ''} | {B.fmt_time(row['epoch_time_s'])}" + " " * 10)
            B.atomic_save(dict(model=model.state_dict(), ema=ema.model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(),
                               scaler=scaler.state_dict(), hist=hist, best=best, epoch=ep, updates=updates, bad=bad, cell_loss=cell_loss,
                               samp_log=samp_log), self.ck_last)
        loader.shutdown(wait=True)
        pd.DataFrame(hist).to_csv(self.dir / "history.csv", index=False)
        pd.DataFrame(samp_log).to_csv(self.dir / "sampler_shares.csv", index=False)
        tmp = self.ck_best.with_suffix(".tmp")
        if not tmp.exists():
            raise RuntimeError(f"{self.id}: no valid validation score")
        model.load_state_dict(torch.load(tmp, map_location=self.dev, weights_only=False)["model"])
        # predictions (val + test) with raw logits -> E5 calibrates post hoc; 151026-style T / threshold kept for E1
        rows = []
        for split in ("val", "test"):
            idx = D.idx(split)
            lg = self.predict(model, idx)
            for i, l in zip(idx, lg):
                rows.append(dict(split=split, idx=int(i), dataset=D.ds[i], subset=D.subset[i], label=int(D.y[i]), group=D.group[i], logit=float(l)))
        pr = pd.DataFrame(rows)
        va = pr[pr.split == "val"]
        T = B.fit_temperature(va.logit.values, va.label.values)
        thr = B.choose_threshold(va.label.values, B.sigmoid(va.logit.values / T), va.dataset.values)
        pr["p_fake"] = B.sigmoid(pr.logit / T)
        pr["pred"] = (pr.p_fake >= thr).astype(int)
        pr.to_csv(self.pred_csv, index=False)
        cfgd = dict(img=a.img, face_px=a.face_px, n_frames=B.N_FRAMES, margin=B.FACE_MARGIN, mean=B.MEAN.tolist(), std=B.STD.tolist(),
                    backbone="efficientnet_b0", pretrained=bool(model.pretrained))
        B.atomic_save(dict(model=model.state_dict(), temperature=T, threshold=thr, best_epoch=best["epoch"], val_score=best["score"],
                           history=hist, config=cfgd, run=self.id, sampler=cfg["sampler"], epochs=epochs, patience=cfg["patience"],
                           seed=self.seed, script=SCRIPT_PATH.name, saved_at=B.now(), final=True), self.ck_best)
        tmp.unlink(missing_ok=True)
        print(f"  {self.id}: best epoch {best['epoch']} | val macro-AUC {best['score']:.4f} | T {T:.2f} | threshold {thr:.2f} | "
              f"{B.fmt_time(time.time() - t_run)}")


def run_configs(a, P, D, configs: List[str]):
    for seed in a.seed_list:                     # seed-major: base12 of a seed exists before its variants
        for c in configs:
            Run(a, P, D, c, seed).train()


def load_runs(P: Paths) -> pd.DataFrame:
    rows = []
    for d in sorted(P.runs.glob("*_s*")):
        f = d / "predictions.csv"
        if f.exists():
            x = pd.read_csv(f)
            cfg, seed = d.name.rsplit("_s", 1)
            x["config"], x["seed"], x["run"] = cfg, int(seed), d.name
            rows.append(x)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


# =====================================================================================
# METRICS + BOOTSTRAP
# =====================================================================================
class WeightedAUC:
    def __init__(self, y, p):
        self.y = np.asarray(y, float)
        u, inv = np.unique(np.asarray(p, float), return_inverse=True)
        self.M = sparse.csr_matrix((np.ones(len(inv)), (np.arange(len(inv)), inv)), shape=(len(inv), len(u)))

    def __call__(self, W):
        W = np.atleast_2d(W).astype(float)
        Pb = np.asarray(self.M.T @ (W * self.y).T).T
        Nb = np.asarray(self.M.T @ (W * (1 - self.y)).T).T
        before = np.cumsum(Nb, 1) - Nb
        num = (Pb * (before + 0.5 * Nb)).sum(1)
        den = Pb.sum(1) * Nb.sum(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(den > 0, num / den, np.nan)


def run_metrics(pr: pd.DataFrame) -> pd.DataFrame:
    """Per run x test set: AUC, accuracy, balanced accuracy, real recognised, fake detected, ECE (151026-style T + threshold)."""
    out = []
    for run, g in pr[pr.split == "test"].groupby("run"):
        for name, x in [("ALL", g)] + [(d, g[g.dataset == d]) for d in sorted(g.dataset.unique())]:
            thr_pred = x.pred.values
            y = x.label.values
            rec = (thr_pred[y == 1] == 1).mean() if (y == 1).any() else np.nan
            spec = (thr_pred[y == 0] == 0).mean() if (y == 0).any() else np.nan
            out.append(dict(run=run, config=g.config.iloc[0], seed=int(g.seed.iloc[0]), test_set=name, n=len(x),
                            auc=roc_auc_score(y, x.p_fake) if len(np.unique(y)) > 1 else np.nan, accuracy=float((thr_pred == y).mean()),
                            balanced_accuracy=np.nanmean([rec, spec]), fake_detected=rec, real_recognised=spec, ece=B.ece(y, x.p_fake)))
        for (d, s), x in g.groupby(["dataset", "subset"]):
            out.append(dict(run=run, config=g.config.iloc[0], seed=int(g.seed.iloc[0]), test_set=f"{d}/{s}", n=len(x),
                            accuracy=float((x.pred == x.label).mean())))
    return pd.DataFrame(out)


def bootstrap_compare(a, pr: pd.DataFrame) -> pd.DataFrame:
    rows = []
    te = pr[pr.split == "test"]
    for ds, g in [("ALL", te)] + [(d, te[te.dataset == d]) for d in sorted(te.dataset.unique())]:
        groups = np.array(sorted(g.group.unique()))
        gpos = {x: i for i, x in enumerate(groups)}
        rng = np.random.default_rng([171026, len(ds)])
        Wg = np.vstack([np.ones(len(groups)), rng.multinomial(len(groups), np.full(len(groups), 1 / len(groups)), size=a.n_boot)])
        per = {}
        for run, h in g.groupby("run"):
            W = Wg[:, h.group.map(gpos).values]
            y, pd_, pf = h.label.values, h.pred.values, h.p_fake.values
            o = dict(auc=WeightedAUC(y, pf)(W))
            for lab, nm in ((1, "fake_detected"), (0, "real_recognised")):
                m = y == lab
                o[nm] = (W[:, m] * (pd_[m] == lab)).sum(1) / np.maximum(W[:, m].sum(1), 1e-9) if m.any() else np.full(len(Wg), np.nan)
            o["balanced_accuracy"] = (o["fake_detected"] + o["real_recognised"]) / 2
            for sub in ("NeuralTextures", "Face2Face", "FaceShifter", "FaceSwap"):
                m = h.subset.values == sub
                if m.any():
                    o[f"recall_{sub}"] = (W[:, m] * (pd_[m] == 1)).sum(1) / np.maximum(W[:, m].sum(1), 1e-9)
            per[(h.config.iloc[0], int(h.seed.iloc[0]))] = o
        for hi, lo in COMPARISONS:
            seeds = sorted({s for c, s in per if c == hi} & {s for c, s in per if c == lo})
            if not seeds:
                continue
            for met in per[(hi, seeds[0])]:
                if not all(met in per[(hi, s)] and met in per[(lo, s)] for s in seeds):
                    continue
                d = np.vstack([per[(hi, s)][met] - per[(lo, s)][met] for s in seeds])
                pt = d[:, 0]
                if not np.isfinite(pt).all():
                    continue
                boot = np.nanmean(d[:, 1:], 0)
                boot = boot[np.isfinite(boot)]
                lo_ci, hi_ci = np.percentile(boot, [2.5, 97.5]) if len(boot) > 20 else (np.nan, np.nan)
                same = bool(np.all(np.sign(pt) == np.sign(pt.mean()))) if pt.mean() != 0 else False
                if len(seeds) < 2:
                    v = "single seed: CI excludes 0" if (lo_ci > 0 or hi_ci < 0) else "single seed: inconclusive"
                elif (lo_ci > 0 or hi_ci < 0) and same:
                    v = "supported improvement" if pt.mean() > 0 else "supported deterioration"
                else:
                    v = "inconclusive"
                rows.append(dict(test_set=ds, comparison=f"{hi} - {lo}", metric=met, n_seeds=len(seeds), mean_diff=pt.mean(),
                                 per_seed=";".join(f"{s}:{x:+.4f}" for s, x in zip(seeds, pt)), ci95_low=lo_ci, ci95_high=hi_ci, verdict=v))
    return pd.DataFrame(rows)


def seed_table(m: pd.DataFrame, metrics=("auc", "accuracy", "balanced_accuracy", "real_recognised", "fake_detected", "ece")) -> pd.DataFrame:
    rows = []
    for (c, t), g in m.groupby(["config", "test_set"]):
        r = dict(config=c, test_set=t, n_seeds=g.seed.nunique())
        for k in metrics:
            if k in g and g[k].notna().any():
                r[k] = f"{g[k].mean():.3f} ± {g[k].std(ddof=1):.3f}" if g[k].notna().sum() > 1 else f"{g[k].mean():.3f}"
        rows.append(r)
    return pd.DataFrame(rows)


# =====================================================================================
# E1 BASELINE REPRODUCTION
# =====================================================================================
def exp_e1(a, P, BP, inv, C, D):
    print("\n" + "=" * 90 + "\n E1  BASELINE REPRODUCTION (seed 42, 151026 settings)\n" + "=" * 90)
    res = {}
    # (a) split re-derivation with the 151026 code into a scratch folder
    RP = B.Paths(P.repro / "split_rederived")
    RP.make()
    a_split = argparse.Namespace(smoke=a.smoke)
    sp2 = B.phase_split(a_split, RP, inv, C)
    sp1 = pd.read_csv(BP.split_csv)
    same_split = bool((sp1.split.values == sp2.split.values).all())
    # group ids are union-find roots of the same graph -> identical partition if the labels match
    same_groups = bool(pd.crosstab(sp1.group, sp2.group).astype(bool).sum(1).max() == 1)
    res["split_identical"] = same_split
    res["groups_identical"] = same_groups
    res["split_sha256_151026"] = sha_file(BP.split_csv)
    res["split_sha256_rederived"] = sha_file(RP.split_csv)
    print(f"  split re-derived: identical assignment {same_split} | identical groups {same_groups}")
    # (b)+(c) re-run seed 42
    Run(a, P, D, "base12", 42).train()
    new = torch.load(P.runs / "base12_s42" / "best.pt", map_location="cpu", weights_only=False)
    ref_ck = BP.checkpoints / "seed42_best.pt"
    ref = torch.load(ref_ck if ref_ck.exists() else BP.best, map_location="cpu", weights_only=False)
    res["best_epoch_151026"], res["best_epoch_rerun"] = int(ref["best_epoch"]), int(new["best_epoch"])
    res["same_best_epoch"] = res["best_epoch_151026"] == res["best_epoch_rerun"]
    hr, hn = pd.DataFrame(ref["history"]), pd.DataFrame(new["history"])
    k = min(len(hr), len(hn))
    res["max_abs_val_auc_diff_per_epoch"] = float(np.nanmax(np.abs(hr.val_score.values[:k] - hn.val_score.values[:k]))) if k else np.nan
    res["val_score_151026"], res["val_score_rerun"] = float(ref["val_score"]), float(new["val_score"])
    res["threshold_151026"], res["threshold_rerun"] = float(ref["threshold"]), float(new["threshold"])
    res["temperature_151026"], res["temperature_rerun"] = float(ref["temperature"]), float(new["temperature"])
    # test metrics: 151026 predictions.csv vs rerun
    p1 = pd.read_csv(BP.metrics / "predictions.csv")
    p1 = p1[p1.split == "test"]
    p2 = pd.read_csv(P.runs / "base12_s42" / "predictions.csv")
    p2 = p2[p2.split == "test"]
    rows = []
    for name in ["ALL"] + sorted(p1.dataset.unique()):
        x1 = p1 if name == "ALL" else p1[p1.dataset == name]
        x2 = p2 if name == "ALL" else p2[p2.dataset == name]
        r = dict(test_set=name, n_151026=len(x1), n_rerun=len(x2))
        for nm, x in (("151026", x1), ("rerun", x2)):
            r[f"auc_{nm}"] = roc_auc_score(x.label, x.p_fake)
            r[f"accuracy_{nm}"] = float((x.pred == x.label).mean())
        r["auc_diff"] = r["auc_rerun"] - r["auc_151026"]
        r["accuracy_diff"] = r["accuracy_rerun"] - r["accuracy_151026"]
        r["within_tolerance"] = abs(r["auc_diff"]) <= REPRO_TOL["auc"] and abs(r["accuracy_diff"]) <= REPRO_TOL["accuracy"]
        rows.append(r)
    cmp_ = pd.DataFrame(rows)
    mm = p1.merge(p2, on="idx", suffixes=("_151026", "_rerun"))
    res["same_test_videos"] = len(mm) == len(p1) == len(p2)
    res["prediction_agreement"] = float((mm.pred_151026 == mm.pred_rerun).mean()) if len(mm) else np.nan
    res["p_fake_spearman"] = float(spearmanr(mm.p_fake_151026, mm.p_fake_rerun)[0]) if len(mm) > 2 else np.nan
    res["bitwise_identical_predictions"] = bool(len(mm) and np.allclose(mm.p_fake_151026, mm.p_fake_rerun, atol=1e-6))
    res["metrics_within_tolerance"] = bool(cmp_.within_tolerance.all())
    res["reproduced"] = bool(same_split and res["metrics_within_tolerance"])
    cmp_.to_csv(P.metrics / "E1_reproduction_metrics.csv", index=False)
    B.write_json(P.reports / "E1_reproduction.json", dict(res, tolerance=REPRO_TOL, checked_at=B.now(),
                                                          note="cuDNN autotuning / atomics make GPU training non-bitwise reproducible; "
                                                               "use --deterministic for a slower, more repeatable run"))
    print(B.md_table(cmp_, 4))
    print(f"  best epoch {res['best_epoch_151026']} -> {res['best_epoch_rerun']} | prediction agreement {res['prediction_agreement']:.3f} | "
          f"REPRODUCED: {res['reproduced']}")
    return res


# =====================================================================================
# E5 CALIBRATION + THRESHOLD + SELECTIVE PREDICTION
# =====================================================================================
def calibrate(method, zv, yv, zt):
    if method == "raw":
        return B.sigmoid(zv), B.sigmoid(zt), {}
    if method == "temperature":
        T = B.fit_temperature(zv, yv)
        return B.sigmoid(zv / T), B.sigmoid(zt / T), dict(T=T)
    if method == "platt":
        lr = LogisticRegression(C=1e6, max_iter=1000).fit(zv[:, None], yv)
        return lr.predict_proba(zv[:, None])[:, 1], lr.predict_proba(zt[:, None])[:, 1], dict(a=float(lr.coef_[0, 0]), b=float(lr.intercept_[0]))
    if method == "isotonic":
        iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-4, y_max=1 - 1e-4).fit(B.sigmoid(zv), yv)
        return iso.predict(B.sigmoid(zv)), iso.predict(B.sigmoid(zt)), {}
    raise ValueError(method)


def threshold(method, pv, yv, dsv):
    if method == "fixed_0.5":
        return 0.5
    if method == "val_mean_bacc":
        return B.choose_threshold(yv, pv, dsv)
    f, t, th = roc_curve(yv, pv)
    return float(np.clip(th[np.argmax(t - f)], 0.01, 0.99))


def mean_bacc(y, pred, ds):
    v = []
    for d in np.unique(ds):
        m = ds == d
        r = (pred[m][y[m] == 1] == 1).mean() if (y[m] == 1).any() else np.nan
        s = (pred[m][y[m] == 0] == 0).mean() if (y[m] == 0).any() else np.nan
        v.append(np.nanmean([r, s]))
    return float(np.nanmean(v)) if v else np.nan


def brier(y, p):
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def nll(y, p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def exp_e5(a, P, pr: pd.DataFrame):
    print("\n" + "=" * 90 + "\n E5  THRESHOLD + CALIBRATION + SELECTIVE PREDICTION (post hoc, validation-tuned)\n" + "=" * 90)
    rows, curves = [], []
    for run, g in pr.groupby("run"):
        v, t = g[g.split == "val"], g[g.split == "test"]
        zv, yv, dv = v.logit.values, v.label.values, v.dataset.values
        zt, yt, dt = t.logit.values, t.label.values, t.dataset.values
        for cm in CAL_METHODS:
            pv, pt, par = calibrate(cm, zv, yv, zt)
            for tm in THR_METHODS:
                thr = threshold(tm, pv, yv, dv)
                for cov in COVERAGES:
                    band = 0.0 if cov >= 1 else float(np.quantile(np.abs(pv - thr), 1 - cov))   # tuned on validation
                    keep = np.abs(pt - thr) >= band
                    pred = (pt >= thr).astype(int)
                    r = dict(run=run, config=g.config.iloc[0], seed=int(g.seed.iloc[0]), calibration=cm, threshold_method=tm,
                             target_coverage=cov, threshold=thr, band=band, test_coverage=float(keep.mean()),
                             ece=B.ece(yt, pt), brier=brier(yt, pt), nll=nll(yt, pt),
                             mean_bacc=mean_bacc(yt[keep], pred[keep], dt[keep]), accuracy=float((pred[keep] == yt[keep]).mean()))
                    for d in np.unique(dt):
                        m = (dt == d)
                        r[f"coverage_{d}"] = float(keep[m].mean())
                        r[f"bacc_{d}"] = mean_bacc(yt[m & keep], pred[m & keep], dt[m & keep])
                        r[f"ece_{d}"] = B.ece(yt[m], pt[m])
                    rows.append(r)
            if cm == "temperature":            # risk-coverage curve for the default decision rule
                thr = threshold("val_mean_bacc", pv, yv, dv)
                conf = np.abs(pt - thr)
                order = np.argsort(-conf)
                err = ((pt >= thr).astype(int) != yt)[order]
                cov_ = np.arange(1, len(err) + 1) / len(err)
                risk = np.cumsum(err) / np.arange(1, len(err) + 1)
                curves.append(pd.DataFrame(dict(run=run, config=g.config.iloc[0], coverage=cov_, risk=risk)))
    res = pd.DataFrame(rows)
    res.to_csv(P.metrics / "E5_calibration_threshold.csv", index=False)
    summ = res.groupby(["config", "calibration", "threshold_method", "target_coverage"]).agg(
        n_runs=("run", "nunique"), test_coverage=("test_coverage", "mean"), ece=("ece", "mean"), brier=("brier", "mean"),
        mean_bacc=("mean_bacc", "mean"), mean_bacc_sd=("mean_bacc", "std"), accuracy=("accuracy", "mean")).reset_index()
    summ.to_csv(P.metrics / "E5_summary.csv", index=False)
    cv = pd.concat(curves) if curves else pd.DataFrame()
    if len(cv):
        aurc = cv.groupby(["run", "config"]).risk.mean().reset_index().rename(columns=dict(risk="AURC"))
        aurc.to_csv(P.metrics / "E5_aurc.csv", index=False)
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        for c, x in cv.groupby("config"):
            for r_, y in x.groupby("run"):
                ax[0].plot(y.coverage, y.risk, color=CONFIG_COLOR.get(c, "k"), alpha=0.6, label=c if r_.endswith("_s42") else None)
        ax[0].set_xlabel("coverage (share of videos decided)")
        ax[0].set_ylabel("error rate on decided videos")
        ax[0].set_title("Risk-coverage (temperature, val mean-BAcc threshold)")
        ax[0].legend(fontsize=8)
        ax[0].grid(alpha=.3)
        b = summ[(summ.config == "base12") & (summ.threshold_method == "val_mean_bacc") & (summ.target_coverage == 1.0)]
        ax[1].bar(b.calibration, b.ece, color="#2a78d6")
        ax[1].set_title("Test ECE by calibration method (base12, seed mean)")
        ax[1].grid(axis="y", alpha=.3)
        plt.tight_layout()
        plt.savefig(P.figures / "E5_risk_coverage_ece.png", dpi=130)
        plt.close(fig)
        # reliability diagram for base12 seed 42
        g = pr[(pr.run == "base12_s42")]
        if len(g):
            v, t = g[g.split == "val"], g[g.split == "test"]
            fig, ax = plt.subplots(figsize=(4.5, 4.2))
            for cm, col in zip(CAL_METHODS, ["#8a8984", "#2a78d6", "#eb6834", "#1baf7a"]):
                _, pt, _ = calibrate(cm, v.logit.values, v.label.values, t.logit.values)
                bins = np.linspace(0, 1, 11)
                idx = np.clip(np.digitize(pt, bins) - 1, 0, 9)
                xs = [pt[idx == k].mean() for k in range(10) if (idx == k).any()]
                ys = [t.label.values[idx == k].mean() for k in range(10) if (idx == k).any()]
                ax.plot(xs, ys, "o-", color=col, label=f"{cm} (ECE {B.ece(t.label.values, pt):.3f})", ms=3)
            ax.plot([0, 1], [0, 1], "k:", lw=0.8)
            ax.set_xlabel("predicted P(fake)")
            ax.set_ylabel("observed fake rate")
            ax.legend(fontsize=7)
            ax.set_title("Reliability (base12, seed 42, test)")
            plt.tight_layout()
            plt.savefig(P.figures / "E5_reliability.png", dpi=130)
            plt.close(fig)
    best = summ[summ.target_coverage == 1.0].sort_values(["ece"]).head(1)
    print(B.md_table(summ[(summ.config == "base12") & (summ.target_coverage.isin([1.0, 0.9]))], 3))
    return res, summ


# =====================================================================================
# E6 XAI FAITHFULNESS
# =====================================================================================
def integrated_gradients(model, x: torch.Tensor, x0: torch.Tensor, target: int, steps: int = IG_STEPS, chunk: int = 4) -> np.ndarray:
    """IG for the video logit (sign by target). x, x0: [1,T,3,H,W] normalised. Returns |attribution| summed over
    channels, [T,H,W]. Riemann (midpoint) rule."""
    model.eval()
    sign = 1.0 if target == 1 else -1.0
    total = torch.zeros_like(x)
    alphas = (torch.arange(steps, device=x.device, dtype=torch.float32) + 0.5) / steps
    for s in range(0, steps, chunk):
        al = alphas[s:s + chunk].view(-1, 1, 1, 1, 1)
        xi = (x0 + al * (x - x0)).detach().requires_grad_(True)
        out = model(xi)                                    # xi: [k,T,3,H,W]
        g = torch.autograd.grad((sign * out["logit"]).sum(), xi)[0]
        total += g.sum(0, keepdim=True)
    attr = (x - x0) * total / steps
    return attr[0].sum(1).clamp(min=0).detach().cpu().numpy(), float(sign * (attr.sum()).item())


_trap = getattr(np, "trapezoid", None) or getattr(np, "trapz")


def perturb(clip_u8: np.ndarray, mask: np.ndarray, kind: str, blurred: np.ndarray) -> np.ndarray:
    out = clip_u8.copy()
    if kind == "blur":
        out[mask] = blurred[mask]
    else:
        out[mask] = 128
    return out


def faith_curves(model, dev, clip: np.ndarray, attr: np.ndarray, target: int, T_: float, rng) -> dict:
    """Deletion and insertion curves (confidence of the predicted class) for blur and grey perturbations."""
    blurred = np.stack([cv2.GaussianBlur(f, (0, 0), 8) for f in clip])
    flat = attr.reshape(len(attr), -1) + rng.random(attr.reshape(len(attr), -1).shape) * 1e-9   # tie-break
    order = np.argsort(-flat, axis=1)
    npx = flat.shape[1]
    sign = 1 if target == 1 else -1
    batch, keys = [], []
    for kind in ("blur", "grey"):
        base_img = blurred if kind == "blur" else np.full_like(clip, 128)
        for frac in XAI_FRACS:
            k = int(round(frac * npx))
            m = np.zeros_like(flat, dtype=bool)
            if k:
                np.put_along_axis(m, order[:, :k], True, axis=1)
            m = m.reshape(attr.shape)
            batch.append(perturb(clip, m, kind, blurred))                                       # deletion
            ins = base_img.copy()
            ins[m] = clip[m]
            batch.append(ins)                                                                   # insertion
            keys += [("del", kind, frac), ("ins", kind, frac)]
    ps = []
    with torch.no_grad():
        for s in range(0, len(batch), 4):
            lg = model(B.normalise(np.stack(batch[s:s + 4]), dev))["logit"].float().cpu().numpy()
            ps += list(B.sigmoid(lg / T_))
    conf = {k: (p if sign == 1 else 1 - p) for k, p in zip(keys, ps)}
    out = {}
    for kind in ("blur", "grey"):
        d = [conf[("del", kind, f)] for f in XAI_FRACS]
        i = [conf[("ins", kind, f)] for f in XAI_FRACS]
        out[f"deletion_auc_{kind}"] = float(_trap(d, XAI_FRACS) / XAI_FRACS[-1])
        out[f"insertion_auc_{kind}"] = float(_trap(i, XAI_FRACS) / XAI_FRACS[-1])
        out[f"drop20_{kind}"] = d[0] - conf[("del", kind, 0.20)]
        out[f"curve_del_{kind}"] = d
        out[f"curve_ins_{kind}"] = i
    return out


def explain_all(model, dev, clip: np.ndarray, target: int, T_: float, rng, methods=XAI_METHODS) -> Dict[str, np.ndarray]:
    x = B.normalise(clip[None], dev)
    maps = {}
    t0 = time.time()
    if "gradcam++" in methods:
        cam, _, _ = B.gradcam_pp(model, x, None, target)
        maps["gradcam++"] = (cam[0], time.time() - t0)
    if "integrated_gradients" in methods:
        t0 = time.time()
        blurred = np.stack([cv2.GaussianBlur(f, (0, 0), 8) for f in clip])
        ig, _ = integrated_gradients(model, x, B.normalise(blurred[None], dev), target)
        ig = np.stack([cv2.GaussianBlur(f, (0, 0), 2) for f in ig])          # light smoothing for region metrics
        maps["integrated_gradients"] = (ig, time.time() - t0)
    if "occlusion" in methods:
        t0 = time.time()
        occ, _ = B.occlusion_map(model, clip, dev, patch=max(16, clip.shape[1] // 6), T_temp=T_)
        occ = occ if target == 1 else -occ
        maps["occlusion"] = (np.repeat(np.clip(occ, 0, None)[None], len(clip), 0), time.time() - t0)
    if "random" in methods:
        maps["random"] = (rng.random(clip.shape[:3]).astype(np.float32), 0.0)
    return maps


def randomise_top(model):
    m = copy.deepcopy(model)
    with torch.no_grad():
        for mod in list(m.backbone[-2:]) + [m.proj, m.temporal, m.att, m.head, m.frame_head]:
            for p in mod.parameters():
                p.normal_(0, 0.02)
    return m.eval()


def exp_e6(a, P, D, ck: Path):
    print("\n" + "=" * 90 + f"\n E6  XAI FAITHFULNESS: Grad-CAM++ vs Integrated Gradients vs occlusion vs random ({ck.parent.name})\n" + "=" * 90)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    obj = torch.load(ck, map_location=dev, weights_only=False)
    model = B.DeepfakeDetector(pretrained=False).to(dev)
    model.load_state_dict(obj["model"])
    model.eval()
    T_, thr = obj["temperature"], obj["threshold"]
    pr = pd.read_csv(ck.parent / "predictions.csv")
    te = pr[pr.split == "test"]
    rng = np.random.default_rng(171026)
    pick = []
    for _, g in te.groupby(["dataset", "label"]):
        pick += list(rng.choice(g.idx.values, size=min(len(g), max(1, a.xai_n // 4)), replace=False))
    rand_model = randomise_top(model)
    rows, curves = [], []
    t_all = time.time()
    for n, i in enumerate(pick):
        r = te[te.idx == i].iloc[0]
        clip = np.asarray(D.C.arr[i])
        target = int(r.pred)
        maps = explain_all(model, dev, clip, target, T_, rng)
        rmaps = explain_all(rand_model, dev, clip, target, T_, rng, methods=["gradcam++", "integrated_gradients", "occlusion"])
        S = clip.shape[1]
        yy, xx = np.mgrid[0:S, 0:S]
        centre = ((yy - S / 2) ** 2 + (xx - S / 2) ** 2) <= (0.35 * S) ** 2
        for meth, (mp, sec) in maps.items():
            fc = faith_curves(model, dev, clip, mp, target, T_, rng)
            row = dict(idx=int(i), dataset=r.dataset, subset=r.subset, label=int(r.label), pred=target, correct=bool(target == r.label),
                       method=meth, seconds=sec, face_centre_energy=float(mp[:, centre].sum() / (mp.sum() + 1e-8)),
                       **{k: v for k, v in fc.items() if not k.startswith("curve")})
            if meth in rmaps:
                rm = rmaps[meth][0]
                row["randomisation_spearman"] = float(spearmanr(mp.reshape(-1)[::7], rm.reshape(-1)[::7])[0])
            rows.append(row)
            for kind in ("blur", "grey"):
                for f, dd, ii in zip(XAI_FRACS, fc[f"curve_del_{kind}"], fc[f"curve_ins_{kind}"]):
                    curves.append(dict(method=meth, kind=kind, frac=f, deletion=dd, insertion=ii, pred=target, dataset=r.dataset))
        if n < 4:
            fig, ax = plt.subplots(2, 5, figsize=(13, 5.4))
            t = int(np.argmax(maps["gradcam++"][0].reshape(len(clip), -1).sum(1)))
            ax[0, 0].imshow(clip[t])
            ax[0, 0].set_title(f"{r.dataset}/{r.subset}\ntrue {'FAKE' if r.label else 'REAL'}, pred {'FAKE' if target else 'REAL'}", fontsize=8)
            ax[1, 0].axis("off")
            for k, meth in enumerate(XAI_METHODS, 1):
                mp = maps[meth][0][t]
                ax[0, k].imshow(B.overlay(clip[t], mp / (mp.max() + 1e-8)))
                ax[0, k].set_title(meth, fontsize=9)
                cv = pd.DataFrame([c for c in curves[-len(XAI_METHODS) * 2 * len(XAI_FRACS):] if c["method"] == meth and c["kind"] == "grey"])
                ax[1, k].plot(cv.frac, cv.deletion, "o-", label="deletion (grey)")
                ax[1, k].plot(cv.frac, cv.insertion, "s-", label="insertion (grey)")
                ax[1, k].set_ylim(0, 1)
                ax[1, k].set_xlabel("fraction of pixels")
                if k == 1:
                    ax[1, k].legend(fontsize=7)
            for a_ in ax[0]:
                a_.axis("off")
            plt.tight_layout()
            plt.savefig(P.xai / f"E6_example_{n}_{r.dataset}_{r.subset}.png", dpi=110)
            plt.close(fig)
        el = time.time() - t_all
        sys.stdout.write(f"\r  {B.bar(n + 1, len(pick))} {n + 1}/{len(pick)} videos | ETA {B.fmt_time(el / (n + 1) * (len(pick) - n - 1))}   ")
        sys.stdout.flush()
    sys.stdout.write("\n")
    fx = pd.DataFrame(rows)
    fx.to_csv(P.metrics / "E6_xai_faithfulness.csv", index=False)
    pd.DataFrame(curves).to_csv(P.metrics / "E6_xai_curves.csv", index=False)
    fx["pred_class"] = np.where(fx.pred == 1, "FAKE", "REAL")
    summ = fx.groupby(["method", "pred_class"]).agg(n=("idx", "size"), deletion_auc_blur=("deletion_auc_blur", "mean"),
                                                    deletion_auc_grey=("deletion_auc_grey", "mean"), insertion_auc_blur=("insertion_auc_blur", "mean"),
                                                    insertion_auc_grey=("insertion_auc_grey", "mean"), drop20_grey=("drop20_grey", "mean"),
                                                    face_centre_energy=("face_centre_energy", "mean"),
                                                    randomisation_spearman=("randomisation_spearman", "mean"), seconds=("seconds", "mean")).reset_index()
    # paired comparison vs random map (per video) for deletion / insertion AUC (grey)
    pv = fx.pivot_table(index=["idx", "pred_class"], columns="method", values=["deletion_auc_grey", "insertion_auc_grey"]).reset_index()
    comp = []
    for meth in [m for m in XAI_METHODS if m != "random"]:
        for cls in ("REAL", "FAKE"):
            x = pv[pv.pred_class == cls]
            if not len(x):
                continue
            dd = (x[("deletion_auc_grey", "random")] - x[("deletion_auc_grey", meth)]).values      # >0: deletes evidence faster than random
            ii = (x[("insertion_auc_grey", meth)] - x[("insertion_auc_grey", "random")]).values    # >0: restores evidence faster
            bs = np.random.default_rng(1).integers(0, len(dd), (a.n_boot, len(dd)))
            for nm, v in (("deletion_gain_vs_random", dd), ("insertion_gain_vs_random", ii)):
                b = v[bs].mean(1)
                comp.append(dict(method=meth, pred_class=cls, metric=nm, n=len(v), mean=v.mean(), ci95_low=np.percentile(b, 2.5),
                                 ci95_high=np.percentile(b, 97.5), better_than_random_pct=100 * (v > 0).mean(),
                                 verdict="faithful (better than random)" if np.percentile(b, 2.5) > 0.005 else
                                 ("worse than random" if np.percentile(b, 97.5) < -0.005 else "not distinguishable from random")))
    comp = pd.DataFrame(comp)
    summ.to_csv(P.metrics / "E6_xai_summary.csv", index=False)
    comp.to_csv(P.metrics / "E6_xai_vs_random.csv", index=False)
    cv = pd.DataFrame(curves)
    fig, ax = plt.subplots(2, 4, figsize=(15, 6.5), sharey=True)
    for c_, cls in enumerate([1, 0]):
        for k, (kind, mode) in enumerate([("grey", "deletion"), ("grey", "insertion"), ("blur", "deletion"), ("blur", "insertion")]):
            axx = ax[c_, k]
            for meth, col in zip(XAI_METHODS, ["#c0392b", "#2a78d6", "#1baf7a", "#8a8984"]):
                y = cv[(cv.method == meth) & (cv.kind == kind) & (cv.pred == cls)].groupby("frac")[mode].mean()
                axx.plot(y.index, y.values, "o-", color=col, label=meth, ms=3)
            axx.set_title(f"{mode} ({kind}) - predicted {'FAKE' if cls else 'REAL'}", fontsize=9)
            axx.set_xlabel("fraction of pixels")
            axx.grid(alpha=.3)
    ax[0, 0].set_ylabel("confidence of predicted class")
    ax[1, 0].set_ylabel("confidence of predicted class")
    ax[0, 0].legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(P.figures / "E6_deletion_insertion_curves.png", dpi=130)
    plt.close(fig)
    print(B.md_table(summ, 3))
    print(B.md_table(comp, 3))
    return summ, comp


# =====================================================================================
# FIGURES, SELECTION, REPORT
# =====================================================================================
def figures_main(P, m: pd.DataFrame):
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    order = [c for c in CONFIGS if c in set(m.config)]
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
    for k, (ts, met, title) in enumerate([("BioDeepAV", "auc", "BioDeepAV test AUC"), ("FF++", "auc", "FF++ test AUC"),
                                          ("FF++/NeuralTextures", "accuracy", "NeuralTextures detected")]):
        x = m[m.test_set == ts]
        for j, c in enumerate(order):
            v = x[x.config == c][met].dropna().values
            if not len(v):
                continue
            ax[k].bar(j, v.mean(), color=CONFIG_COLOR[c], width=0.65)
            ax[k].scatter(np.full(len(v), j), v, color="k", s=10, zorder=3)
            if len(v) > 1:
                ax[k].errorbar(j, v.mean(), yerr=v.std(ddof=1), color="k", capsize=3)
        ax[k].set_xticks(range(len(order)))
        ax[k].set_xticklabels(order, rotation=20)
        ax[k].set_title(title)
        ax[k].grid(axis="y", alpha=.3)
    fig.suptitle("Bars: seed mean, dots: seeds, whiskers: SD", fontsize=9)
    plt.tight_layout()
    plt.savefig(P.figures / "E2_E4_configs.png", dpi=130)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for d in sorted(P.runs.glob("*_s*")):
        h = d / "history.csv"
        if h.exists():
            x = pd.read_csv(h)
            c = d.name.rsplit("_s", 1)[0]
            ax.plot(x.epoch, x.val_score, color=CONFIG_COLOR.get(c, "k"), alpha=0.7, label=c if d.name.endswith("_s42") else None)
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation macro-AUC")
    ax.legend(fontsize=8)
    ax.grid(alpha=.3)
    plt.tight_layout()
    plt.savefig(P.figures / "E4_validation_curves.png", dpi=130)
    plt.close(fig)


def select_best(P, pr: pd.DataFrame) -> Tuple[str, Path]:
    """Validation only: config with the best seed-mean validation macro-AUC; within it the best seed."""
    rows = []
    for d in sorted(P.runs.glob("*_s*")):
        f = d / "best.pt"
        if f.exists():
            o = torch.load(f, map_location="cpu", weights_only=False)
            rows.append(dict(run=d.name, config=d.name.rsplit("_s", 1)[0], val_score=o["val_score"], best_epoch=o["best_epoch"]))
    sel = pd.DataFrame(rows)
    cm = sel.groupby("config").val_score.mean().sort_values(ascending=False)
    best_cfg = cm.index[0]
    best_run = sel[sel.config == best_cfg].sort_values("val_score", ascending=False).run.iloc[0]
    sel.to_csv(P.metrics / "model_selection.csv", index=False)
    import shutil
    shutil.copyfile(P.runs / best_run / "best.pt", P.best)
    B.write_json(P.reports / "model_selection.json", dict(best_config=best_cfg, best_run=best_run, config_val_means=cm.to_dict(),
                                                          rule="highest seed-mean validation macro-AUC; best seed within it", selected_at=B.now()))
    print(f"  best configuration {best_cfg} (val macro-AUC mean {cm.iloc[0]:.4f}); exported {best_run} -> {P.best}")
    return best_run, P.runs / best_run / "best.pt"


def report(a, P, m, cmp_, e1, e5s, e6s, e6c, sel):
    st = seed_table(m[m.test_set.isin(["ALL", "BioDeepAV", "FF++"])])
    sub = m[m.test_set.str.contains("/")].groupby(["config", "test_set"]).accuracy.agg(["mean", "std", "count"]).reset_index()
    status = {
        "E1 Baseline reproduction": "done" if e1 else "not run",
        "E2 Three-seed evaluation": f"{m[m.config == 'base12'].seed.nunique()} / 3 seeds",
        "E3 Hard-type sampling": f"{m[m.config.str.startswith('hard')].run.nunique()} / 6 runs",
        "E4 Longer training": f"{m[m.config.isin(['base20', 'base30'])].run.nunique()} / 6 runs",
        "E5 Threshold and calibration": "done" if e5s is not None else "not run",
        "E6 XAI faithfulness": "done" if e6s is not None else "not run"}
    L = ["# 171026: Test 5.7 improvement suite (basic-retest follow-up)\n", f"_Generated {B.now()} | data: {a.base_root} (read-only) | "
         f"base script 151026.py sha256 {sha_file(a.base_script)[:16]}…_\n", "## Checklist\n",
         B.md_table(pd.DataFrame([dict(experiment=k, status=v) for k, v in status.items()])), "",
         "## E1 Baseline reproduction\n", B.md_table(pd.DataFrame([e1])) if e1 else "_not run_", "",
         (B.md_table(pd.read_csv(P.metrics / "E1_reproduction_metrics.csv"), 4) if (P.metrics / "E1_reproduction_metrics.csv").exists() else ""), "",
         "## E2-E4 Test results per configuration (seed mean ± SD)\n", B.md_table(st), "", "![configs](../figures/E2_E4_configs.png)\n",
         "### Per folder (accuracy, seed mean)\n", B.md_table(sub, 3), "",
         "### Paired comparisons with base12 (cluster bootstrap + seed sign agreement)\n",
         B.md_table(cmp_[cmp_.verdict.str.startswith("supported")] if len(cmp_) else cmp_, 4),
         f"\n_{int((cmp_.verdict == 'inconclusive').sum()) if len(cmp_) else 0} further comparisons inconclusive (E3_E4_statistics.csv)._\n",
         "![curves](../figures/E4_validation_curves.png)\n",
         "## E5 Threshold, calibration, coverage\n",
         B.md_table(e5s[e5s.target_coverage.isin([1.0, 0.9])] if e5s is not None else pd.DataFrame(), 3), "",
         "![rc](../figures/E5_risk_coverage_ece.png)\n![rel](../figures/E5_reliability.png)\n",
         "## E6 XAI faithfulness\n", B.md_table(e6s, 3) if e6s is not None else "_not run_", "",
         B.md_table(e6c, 3) if e6c is not None else "", "", "![xai](../figures/E6_deletion_insertion_curves.png)\n",
         "Deletion AUC: lower = removing the highlighted pixels destroys the evidence faster (more faithful). Insertion AUC: higher = "
         "restoring the highlighted pixels recovers the decision faster. Grey and blur perturbations are both reported because blur "
         "alone removes high-frequency texture everywhere. randomisation_spearman near 0 = the map depends on the learned weights "
         "(passes the sanity check).\n",
         "## Selected model\n", f"{sel}\n",
         "Use: `python 171026.py --predict \"video.mp4\"` (adds an Integrated-Gradients panel to the 151026 explanation)."]
    (P.reports / "REPORT_171026.md").write_text("\n".join(L), encoding="utf-8")
    print(f"  report -> {P.reports / 'REPORT_171026.md'}")


# =====================================================================================
# PREDICT / GUI (best model of the suite)
# =====================================================================================
def phase_predict(a, P):
    BP = B.Paths(P.root)          # 151026 helpers expect .best and .predictions
    BP.best, BP.predictions = P.best, P.predictions
    if not P.best.exists():
        raise SystemExit(f"[STOP] no best_model.pt in {P.checkpoints} - run the experiments first")
    tgt = Path(a.predict)
    vids = sorted(p for p in tgt.rglob("*") if p.suffix.lower() in B.VIDEO_EXT) if tgt.is_dir() else [tgt]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, obj = B.load_model(P.best, dev)
    print(f"  model {obj.get('run', '?')} | threshold {obj['threshold']:.2f} | {len(vids)} video(s)")
    rows = []
    for v in vids:
        r = B.predict_video(str(v), BP, model, obj, dev, occlusion=len(vids) <= 20)
        if r.get("output_folder"):                                         # extra: Integrated Gradients panel
            crops, _ = B.extract_faces(str(v), B._DET, obj["config"]["n_frames"], obj["config"]["img"], obj["config"]["face_px"], obj["config"]["margin"])
            if crops is not None:
                tgt_c = 1 if r["predicted"] == "FAKE" else 0
                blurred = np.stack([cv2.GaussianBlur(f, (0, 0), 8) for f in crops])
                ig, _ = integrated_gradients(model, B.normalise(crops[None], dev), B.normalise(blurred[None], dev), tgt_c)
                ig = np.stack([cv2.GaussianBlur(f, (0, 0), 2) for f in ig])
                show = np.linspace(0, len(crops) - 1, 8).round().astype(int)
                fig, ax = plt.subplots(1, 8, figsize=(16, 2.6))
                for k, t in enumerate(show):
                    ax[k].imshow(B.overlay(crops[t], ig[t] / (ig.max() + 1e-8)))
                    ax[k].set_title(f"IG frame {t}", fontsize=8)
                    ax[k].axis("off")
                fig.suptitle(f"Integrated Gradients (evidence for {r['predicted']})", fontsize=10)
                plt.tight_layout()
                plt.savefig(Path(r["output_folder"]) / "integrated_gradients.png", dpi=110)
                plt.close(fig)
        rows.append(r)
        print(f"\n  {Path(v).name}\n    ==> {r['verdict']}" + (f"  (P(fake) = {r['p_fake']:.3f})" if "p_fake" in r else f"  {r.get('error', '')}"))
        for n in r.get("notes", []):
            print(f"    note: {n}")
        if r.get("output_folder"):
            print(f"    explanation: {r['output_folder']}")
    if len(rows) > 1:
        out = P.predictions / f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        pd.DataFrame(rows).drop(columns=["notes"], errors="ignore").to_csv(out, index=False)
        print(f"\n  summary CSV: {out}")


def phase_gui(a, P):
    BP = B.Paths(P.root)
    BP.best, BP.predictions = P.best, P.predictions
    B.phase_gui(a, BP)


# =====================================================================================
# SELFTEST
# =====================================================================================
def phase_selftest(a, P) -> bool:
    print("\n" + "=" * 90 + "\n SELFTEST\n" + "=" * 90)
    res = {}
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    m = B.DeepfakeDetector(pretrained=False).to(dev).eval()
    rng = np.random.default_rng(0)
    clip = rng.integers(0, 255, (B.CLIP_FRAMES, 64, 64, 3), dtype=np.uint8)
    x = B.normalise(clip[None], dev)
    x0 = B.normalise(np.stack([cv2.GaussianBlur(f, (0, 0), 8) for f in clip])[None], dev)
    ig, total = integrated_gradients(m, x, x0, 1, steps=64, chunk=8)
    with torch.no_grad():
        fx, f0 = m(x)["logit"].item(), m(x0)["logit"].item()
    res["IG completeness (sum attr ~ f(x) - f(x0))"] = abs(total - (fx - f0)) <= 0.15 * max(1e-3, abs(fx - f0)) + 0.05
    res["IG map shape"] = ig.shape == clip.shape[:3]
    fc = faith_curves(m, dev, clip, rng.random(clip.shape[:3]), 1, 1.0, rng)
    res["deletion / insertion curves"] = len(fc["curve_del_grey"]) == len(XAI_FRACS) and 0 <= fc["deletion_auc_grey"] <= 1
    zv = rng.normal(0, 3, 2000)
    yv = (rng.random(2000) < B.sigmoid(zv / 3)).astype(int)
    pv, pt, par = calibrate("temperature", zv, yv, zv)
    res["temperature calibration lowers ECE"] = B.ece(yv, pt) < B.ece(yv, B.sigmoid(zv))
    for cm in CAL_METHODS:
        a1, b1, _ = calibrate(cm, zv, yv, zv)
        res[f"calibration {cm} in [0,1]"] = bool(np.all((b1 >= 0) & (b1 <= 1)))
    ds = np.array(["A", "B"] * 1000)
    res["Youden threshold in range"] = 0.01 <= threshold("val_youden", pt, yv, ds) <= 0.99
    tr = np.arange(40)

    class Dm:
        pass
    Dm.fcell = np.array(["A|r"] * 20 + ["A|x"] * 10 + ["A|y"] * 10)
    w, info = dro_weights(Dm, tr, {"A|r": 0.2, "A|x": 2.0, "A|y": 0.4})
    sh = info["cell_share"]
    res["DRO up-weights the hardest cell"] = sh["A|x"] == max(sh.values()) and abs(w.sum() - 1) < 1e-9
    res["DRO floor"] = min(sh.values()) >= DRO_FLOOR / 3 - 1e-9
    p = rng.random(300)
    y = (rng.random(300) < p).astype(int)
    res["weighted AUC == roc_auc"] = abs(WeightedAUC(y, p)(np.ones((1, 300)))[0] - roc_auc_score(y, p)) < 1e-9
    mr = randomise_top(m)
    res["model randomisation changes outputs"] = not torch.allclose(mr(x)["logit"], m(x)["logit"])
    for k, v in res.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    ok = all(res.values())
    print(f"  selftest {'PASSED' if ok else 'FAILED'}")
    return ok


# =====================================================================================
# MAIN
# =====================================================================================
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="171026: improvement suite for the 151026 real/fake detector")
    p.add_argument("--phase", default="run", choices=["run", "selftest", "predict", "gui", "report"])
    p.add_argument("--predict", default=None)
    p.add_argument("--only", default=None, help="comma list of experiments, e.g. E1,E5")
    p.add_argument("--root", default=None)
    p.add_argument("--base-root", default=None)
    p.add_argument("--base-script", default=str(BASE_SCRIPT))
    p.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    p.add_argument("--img", type=int, default=192)
    p.add_argument("--face-px", type=int, default=112)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--accum", type=int, default=2)
    p.add_argument("--loader-threads", type=int, default=4)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--max-epochs", type=int, default=None)
    p.add_argument("--xai-n", type=int, default=XAI_N)
    p.add_argument("--xai-run", default=None, help="run used for E6 (default: base12_s42)")
    p.add_argument("--n-boot", type=int, default=N_BOOT)
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args(argv)
    if a.predict:
        a.phase = "predict"
    if a.smoke:
        a.max_steps, a.max_epochs, a.xai_n, a.n_boot = a.max_steps or 3, a.max_epochs or 2, min(a.xai_n, 8), min(a.n_boot, 100)
    a.root = Path(a.root) if a.root else (SMOKE_ROOT if a.smoke else DEFAULT_ROOT)
    a.base_root = Path(a.base_root) if a.base_root else (BASE_SMOKE_ROOT if a.smoke else BASE_ROOT)
    a.base_script = Path(a.base_script)
    a.seed_list = [int(s) for s in a.seeds.split(",") if s.strip()]
    a.exps = [e.strip().upper() for e in a.only.split(",")] if a.only else EXPERIMENTS
    return a


def main(argv=None):
    global B
    a = parse_args(argv)
    B = load_base(a.base_script)
    P = Paths(a.root)
    P.make()
    if a.phase not in ("predict", "gui"):
        B.lower_priority()
        sys.stdout = B.Tee(P.logs / f"{a.phase}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    torch.set_num_threads(max(1, min(4, (os.cpu_count() or 4) // 2)))
    print(f"171026 | phase {a.phase} | experiments {a.exps} | seeds {a.seed_list} | root {P.root} | base {a.base_root} | {B.now()} | "
          f"GPU {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}")
    if a.phase == "selftest":
        sys.exit(0 if phase_selftest(a, P) else 1)
    if a.phase == "predict":
        return phase_predict(a, P)
    if a.phase == "gui":
        return phase_gui(a, P)
    if not phase_selftest(a, P):
        raise SystemExit("[STOP] selftest failed")
    BP, inv, C, D = load_base_data(a)
    e1 = None
    if a.phase == "run":
        if "E1" in a.exps:
            e1 = exp_e1(a, P, BP, inv, C, D)
        cfgs = []
        for e in ("E2", "E3", "E4"):
            if e in a.exps:
                cfgs += [c for c in EXP_CONFIGS[e] if c not in cfgs]
        if "E3" in a.exps or "E4" in a.exps:
            cfgs = ["base12"] + [c for c in cfgs if c != "base12"]          # baseline of each seed first
        if cfgs:
            print("\n" + "=" * 90 + f"\n E2-E4  TRAINING {cfgs} x seeds {a.seed_list}\n" + "=" * 90)
            run_configs(a, P, D, cfgs)
    if e1 is None and (P.reports / "E1_reproduction.json").exists():
        e1 = B.read_json(P.reports / "E1_reproduction.json")
    pr = load_runs(P)
    if not len(pr):
        print("  no finished runs yet")
        return
    m = run_metrics(pr)
    m.to_csv(P.metrics / "E2_E4_run_metrics.csv", index=False)
    seed_table(m).to_csv(P.metrics / "E2_E4_seed_summary.csv", index=False)
    cmp_ = bootstrap_compare(a, pr)
    cmp_.to_csv(P.metrics / "E3_E4_statistics.csv", index=False)
    figures_main(P, m)
    e5s = None
    if "E5" in a.exps or a.phase == "report":
        _, e5s = exp_e5(a, P, pr)
    best_run, best_ck = select_best(P, pr)
    e6s = e6c = None
    if "E6" in a.exps and a.phase == "run":
        ck = P.runs / (a.xai_run or "base12_s42") / "best.pt"
        e6s, e6c = exp_e6(a, P, D, ck if ck.exists() else best_ck)
    elif (P.metrics / "E6_xai_summary.csv").exists():
        e6s, e6c = pd.read_csv(P.metrics / "E6_xai_summary.csv"), pd.read_csv(P.metrics / "E6_xai_vs_random.csv")
    report(a, P, m, cmp_, e1, e5s, e6s, e6c, f"best run {best_run} (see model_selection.json)")
    print(f"\n  DONE. Report: {P.reports / 'REPORT_171026.md'}")


if __name__ == "__main__":
    main()
