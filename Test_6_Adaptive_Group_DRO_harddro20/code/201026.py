"""
201026.py - TEST 6: ADAPTIVE GROUP-DRO + 20-EPOCH TRAINING (harddro20), follow-up of the improvement suite 171026.py
===================================================================================================================

Research question: does the online group-DRO sampler of harddro12 combined with the 20-epoch schedule of base20 improve
real / fake detection on BioDeepAV and FaceForensics++ C23 without sacrificing either dataset or explanation reliability?

Nothing of 151026.py / 171026.py is modified and no existing run is retrained. Both scripts are IMPORTED read-only so the
model, preprocessing, augmentation, loss, optimiser, EMA, group-DRO sampler, trainer, checkpoint selection, cluster
bootstrap and E5 calibration code are byte-identical. The only addition is one in-memory configuration:

  harddro20 = sampler "harddro" (as harddro12) + 20 epochs, patience 5, cosine schedule stretched (as base20)

Compared configurations (same split, same three seeds 42 / 123 / 2026, same test videos):
  base12, harddro12, base20  existing runs of run_171026 (checkpoints + predictions re-used read-only, re-scored here)
  harddro20                  NEW, trained by this script into run_201026/runs

Phases (all resumable; outputs only under run_201026):
  selftest   unit checks of the additions (bootstrap extension == 171026 bootstrap, probes, Holm, CAM metrics)
  preflight  data + leakage audit: split / inventory / cache hashes, split re-derived from scratch with the 151026 code,
             identity / duplicate groups do not cross partitions, labels, existing runs use the identical val / test
             videos -> reports/preflight_audit.json. ANY failure stops the script (no replacement split is ever made).
  train      harddro20 x seeds (epoch log, per-cell DRO loss + sampling shares, best + final checkpoint, run_meta.json)
  xai        Grad-CAM++ faithfulness (deletion / insertion, grey + blur, vs a random map) and stability (photometric
             probes taken from the 151026 training-augmentation operators) on the 160 E6 test videos, every run
  evaluate   all metrics, per-manipulation analysis, calibration (171026 E5), paired cluster bootstrap with Holm
             correction, figures, REPORT_201026.md + summary.json
  all        selftest -> preflight -> train -> xai -> evaluate (default)

Robustness (original vs degraded) is NOT evaluated: the 171026 protocol uses original videos only and the preflight
verifies that the manifest contains no degraded videos; new test degradations are deliberately not created.

Usage (from E:\\Code, after 151026.py and 171026.py have run)
  python 201026.py --phase selftest
  python 201026.py --phase preflight
  python 201026.py                                   # everything (~3 h training + ~1.5 h XAI on an RTX 3050)
  python 201026.py --phase train --seeds 42          # one seed
  python 201026.py --phase evaluate                  # re-score + report from whatever has finished
  python 201026.py --smoke                           # 3 steps x 2 epochs, seed 42, 8 XAI videos -> run_201026_smoke
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import cv2
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import roc_auc_score, roc_curve  # noqa: E402

SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent
DEFAULT_ROOT = SCRIPT_DIR / "run_201026"
SMOKE_ROOT = SCRIPT_DIR / "run_201026_smoke"
BASE_SCRIPT = SCRIPT_DIR / "151026.py"
BASE_ROOT = SCRIPT_DIR / "run_151026"
SUITE_SCRIPT = SCRIPT_DIR / "171026.py"
SUITE_ROOT = SCRIPT_DIR / "run_171026"

SEEDS = [42, 123, 2026]
NEW = "harddro20"
NEW_CONFIG = dict(sampler="harddro", epochs=20, patience=5, exp="T6",
                  label="Group-DRO hard-type sampler, 20 epochs, patience 5 (Test 6)")
CONFIGS = ["base12", "harddro12", "base20", NEW]
PRIMARY = [(NEW, "harddro12"), (NEW, "base20")]             # the two comparisons the hypotheses are tested on
SECONDARY = [(NEW, "base12"), ("harddro12", "base12"), ("base20", "base12")]
COLOR = {"base12": "#8a8984", "harddro12": "#4a3aa7", "base20": "#2a78d6", NEW: "#eb6834"}
HARD_TYPES = ["NeuralTextures", "FaceSwap", "Face2Face", "FaceShifter"]
FF_FAKES = ["Deepfakes", "DeepFakeDetection", "Face2Face", "FaceShifter", "FaceSwap", "NeuralTextures"]
N_BOOT = 1000
XAI_N = 160
XAI_SEED = 201026
BIO_MARGIN = 0.01          # H3: BioDeepAV AUC non-inferiority margin (CI lower bound of the difference > -0.01)
XAI_MARGIN = 0.05          # H4: explanation non-inferiority margin (faithfulness gain / stability, CI lower bound > -0.05)
# Explanation-stability probes: the photometric operators of 151026.augment_clip at mid-range strength, one at a time.
# Geometric operators (crop, flip, erase) are excluded so the maps stay pixel-aligned. These probe the explanation; they
# are NOT a detection-robustness test set.
PROBES = {"jpeg_q60": ("jpeg", 60), "blur_s1.0": ("blur", 1.0), "downscale_0.6": ("down", 0.6),
          "brightness_+0.10": ("bright", 0.10), "contrast_1.15": ("contrast", 1.15), "saturation_0.80": ("sat", 0.80)}

B = None   # 151026 module
S = None   # 171026 module


# =====================================================================================
# MODULES + PROVENANCE
# =====================================================================================
def load_module(path: Path, name: str):
    if not path.exists():
        raise SystemExit(f"[STOP] {path} not found - 201026.py needs it in the same folder")
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def sha_json(o) -> str:
    return hashlib.sha256(json.dumps(o, sort_keys=True, default=str).encode()).hexdigest()


_DRO_LOG: List[dict] = []


def gradcam_pp_noleak(model, x: torch.Tensor, t_idx, target: int = 1):
    """151026.gradcam_pp with the activation gradient taken by autograd.grad instead of a tensor hook. Same maths, same
    outputs (checked in selftest); the hook version leaks ~5.6 MB of GPU memory per call (tensor <-> hook cycle held by
    the C++ autograd node, invisible to Python's gc), which exhausts the GPU after a few thousand calls."""
    store = {}
    h = model.cam_layer.register_forward_hook(lambda _m, _i, o: store.__setitem__("a", o))
    try:
        model.eval()
        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            o = model(x, t_idx)
            sign = 1.0 if target == 1 else -1.0
            G = torch.autograd.grad((sign * o["logit"]).sum(), store["a"])[0]
    finally:
        h.remove()
    A, G = store["a"].detach().float(), G.detach().float()
    g2, g3 = G ** 2, G ** 3
    alpha = g2 / (2 * g2 + (A.sum((2, 3), keepdim=True) * g3) + 1e-8)
    wts = (alpha * torch.nn.functional.relu(G)).sum((2, 3), keepdim=True)
    cam = torch.nn.functional.relu((wts * A).sum(1))
    B_, T = x.shape[:2]
    cam = torch.nn.functional.interpolate(cam[:, None], size=x.shape[-2:], mode="bilinear", align_corners=False)[:, 0].view(B_, T, *x.shape[-2:])
    cam = cam / (cam.flatten(1).amax(1).view(B_, 1, 1, 1) + 1e-8)
    return cam.cpu().numpy(), o["logit"].detach().float().cpu().numpy(), o["attn"].detach().float().cpu().numpy()


def install(a):
    """Register harddro20 in the IMPORTED 171026 module (in memory; 171026.py on disk is untouched)."""
    S.B = B
    B.gradcam_pp_hook = B.gradcam_pp                # original kept for the equivalence check
    B.gradcam_pp = gradcam_pp_noleak                # in memory only; 151026.py on disk is untouched
    S.CONFIGS[NEW] = dict(NEW_CONFIG)
    S.CONFIG_COLOR.update(COLOR)
    S.SCRIPT_PATH = SCRIPT_PATH                     # checkpoints record script=201026.py
    orig = S.dro_weights

    def dro_logged(D, tr, cell_loss):               # unchanged algorithm; only records the per-cell losses it used
        w, info = orig(D, tr, cell_loss)
        _DRO_LOG.append(dict(info))
        return w, info
    dro_logged.__wrapped__ = orig
    S.dro_weights = dro_logged


def training_constants(a) -> dict:
    return dict(lr_backbone=B.LR_BACKBONE, lr_head=B.LR_HEAD, weight_decay=B.WEIGHT_DECAY, warmup_frac=B.WARMUP_FRAC,
                label_smooth=B.LABEL_SMOOTH, frame_loss_w=B.FRAME_LOSS_W, ema_decay=B.EMA_DECAY, n_frames=B.N_FRAMES,
                clip_frames=B.CLIP_FRAMES, max_oversample=B.MAX_OVERSAMPLE, img=a.img, face_px=a.face_px, batch=a.batch,
                accum=a.accum, amp=not a.no_amp, deterministic=a.deterministic, pretrained=not a.no_pretrained,
                dro=dict(eta=S.DRO_ETA, floor=S.DRO_FLOOR, momentum=S.DRO_MOM, cells="dataset x folder"),
                grad_clip=5.0, optimiser="AdamW", schedule="linear warm-up + cosine over max_epochs x steps",
                checkpoint_rule="EMA weights at the epoch with the best validation macro-AUC (mean of BioDeepAV and FF++ AUC)")


def provenance(a) -> dict:
    return dict(scripts={p.name: sha_file(p) for p in (SCRIPT_PATH, a.base_script, a.suite_script)},
                config_sha256={c: sha_json(S.CONFIGS[c]) for c in CONFIGS if c in S.CONFIGS},
                training_constants_sha256=sha_json(training_constants(a)), git="not a git repository - sha256 of the scripts is the source revision")


def environment(P) -> dict:
    env = dict(python=sys.version, platform=platform.platform(), torch=torch.__version__, cuda=torch.version.cuda,
               cudnn=torch.backends.cudnn.version() if torch.cuda.is_available() else None,
               gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, numpy=np.__version__,
               pandas=pd.__version__, opencv=cv2.__version__, matplotlib=matplotlib.__version__)
    try:
        import sklearn
        import scipy
        import torchvision
        env.update(sklearn=sklearn.__version__, scipy=scipy.__version__, torchvision=torchvision.__version__)
    except Exception:
        pass
    try:
        fr = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, timeout=120).stdout
        (P.reports / "pip_freeze.txt").write_text(fr, encoding="utf-8")
    except Exception as e:
        env["pip_freeze_error"] = str(e)
    B.write_json(P.reports / "environment.json", env)
    return env


# =====================================================================================
# PREFLIGHT: DATA + LEAKAGE AUDIT (fails hard)
# =====================================================================================
def run_dir(a, P, run: str) -> Path:
    return (P.runs if run.startswith(NEW + "_") else a.suite_root / "runs") / run


def preflight(a, P, BP, inv, C, D) -> dict:
    print("\n" + "=" * 90 + "\n PREFLIGHT: data + leakage audit\n" + "=" * 90)
    sp = D.sp
    chk, info = {}, {}
    e1 = a.suite_root / "reports" / "E1_reproduction.json"
    ref_sha = B.read_json(e1)["split_sha256_151026"] if e1.exists() else None
    info["split_sha256"] = sha_file(BP.split_csv)
    info["split_sha256_reference_171026"] = ref_sha
    info["inventory_sha256"] = sha_file(BP.inventory)
    info["face_cache"] = C.arr_p.name
    info["face_cache_status_sha256"] = sha_file(C.st_p)
    chk["split.csv identical to the split used by 171026 (sha256)"] = ref_sha is not None and info["split_sha256"] == ref_sha
    chk["split rows == inventory rows, same paths"] = len(sp) == len(inv) and bool((sp.path.values == inv.path.values).all())
    chk["face cache complete"] = bool(C.complete())
    use = sp[sp.split != "excluded"]
    chk["no group in more than one partition"] = int(use.groupby("group").split.nunique().max()) == 1
    tok = use.assign(tok=use.tokens.str.split(";")).explode("tok")
    chk["no identity token in more than one partition"] = int(tok.groupby("tok").split.nunique().max()) == 1
    chk["no exact duplicate (file hash) in more than one partition"] = int(use.groupby("quick_hash").split.nunique().max()) == 1
    real = ((sp.dataset == "BioDeepAV") & (sp.subset == "real")) | ((sp.dataset == "FF++") & (sp.subset == "original"))
    chk["labels: real folders = 0, fake folders = 1"] = bool(((sp.label == 0) == real).all())
    chk["datasets are exactly BioDeepAV + FF++ C23"] = set(sp.dataset) == {"BioDeepAV", "FF++"}
    deg = sp.path.str.lower().str.contains("degrad") | sp.subset.str.lower().str.contains("degrad")
    info["degraded_videos_in_manifest"] = int(deg.sum())
    tr, va, te = D.idx("train"), D.idx("val"), D.idx("test")
    chk["train / val / test disjoint"] = not (set(tr) & set(va) or set(tr) & set(te) or set(va) & set(te))
    w, winfo = S.dro_weights.__wrapped__(D, tr, {c: 1.0 for c in set(D.fcell[tr])})
    sh = np.array(list(winfo["cell_share"].values()))
    chk["DRO epoch-1 weights: uniform over cells, sum 1, only training videos"] = (
        abs(w.sum() - 1) < 1e-9 and np.allclose(sh, 1 / len(sh)) and len(w) == len(tr))
    info["dro_cells"] = sorted(winfo["cell_share"])
    if not a.skip_rederive:
        RP = B.Paths(P.repro / "split_rederived")
        RP.make()
        sp2 = B.phase_split(argparse.Namespace(smoke=False), RP, inv, C)
        chk["split re-derived from scratch with the 151026 code is identical"] = bool((sp.split.values == sp2.split.values).all())
        chk["re-derived groups identical"] = bool(pd.crosstab(sp.group, sp2.group).astype(bool).sum(1).max() == 1)
        info["split_sha256_rederived"] = sha_file(RP.split_csv)
    # existing runs: same videos, labels, metadata
    ref = {s: set(D.idx(s)) for s in ("val", "test")}
    runs = {}
    for c in CONFIGS:
        for s in a.seed_list:
            run = f"{c}_s{s}"
            d = run_dir(a, P, run)
            if c == NEW and not (d / "predictions.csv").exists():
                continue
            ok = (d / "predictions.csv").exists() and (d / "best.pt").exists()
            r = dict(found=ok)
            if ok:
                pr = pd.read_csv(d / "predictions.csv")
                ck = torch.load(d / "best.pt", map_location="cpu", weights_only=False)
                same = all(set(pr[pr.split == x].idx) == ref[x] for x in ("val", "test"))
                meta = (pr.label.values == D.y[pr.idx.values]).all() and (pr.dataset.values == D.ds[pr.idx.values]).all() and \
                       (pr.subset.values == D.subset[pr.idx.values]).all() and (pr.group.values == D.group[pr.idx.values]).all()
                cal = np.allclose(pr.p_fake.values, B.sigmoid(pr.logit.values / ck["temperature"]), atol=1e-6)
                pred = int(((pr.p_fake >= ck["threshold"]).astype(int) != pr.pred).sum())
                r.update(final=bool(ck.get("final")), sampler=ck["sampler"], epochs=ck["epochs"], patience=ck["patience"],
                         same_val_test_videos=same, metadata_match=bool(meta), p_fake_is_sigmoid_logit_over_T=bool(cal),
                         pred_mismatch=pred, predictions_sha256=sha_file(d / "predictions.csv"), best_sha256=sha_file(d / "best.pt"))
                exp = S.CONFIGS[c]
                ep_exp = min(exp["epochs"], a.max_epochs) if (c == NEW and a.max_epochs) else exp["epochs"]
                r["config_matches"] = (ck["sampler"], ck["epochs"], ck["patience"]) == (exp["sampler"], ep_exp, exp["patience"])
                ok = r["final"] and same and meta and cal and pred == 0 and r["config_matches"]
            runs[run] = r
            chk[f"run {run}: same val/test videos, labels, config, calibration"] = bool(ok)
    res = dict(checked_at=B.now(), passed=all(chk.values()), checks=chk, info=info, runs=runs,
               counts={f"{k[0]}|{k[1]}|{k[2]}": int(v) for k, v in sp.groupby(["dataset", "subset", "split"]).size().items()},
               robustness_evaluable=info["degraded_videos_in_manifest"] > 0, provenance=provenance(a))
    B.write_json(P.reports / "preflight_audit.json", res)
    B.write_json(P.reports / "data_manifest.json", dict(split_csv=str(BP.split_csv), split_sha256=info["split_sha256"],
                                                        inventory_csv=str(BP.inventory), inventory_sha256=info["inventory_sha256"],
                                                        face_cache=str(C.arr_p), n_videos=len(sp), counts=res["counts"]))
    for k, v in chk.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print(f"  degraded videos in manifest: {info['degraded_videos_in_manifest']} -> robustness evaluable: {res['robustness_evaluable']}")
    if not res["passed"]:
        raise SystemExit("[STOP] preflight failed - see reports/preflight_audit.json (nothing was trained or overwritten)")
    print("  preflight PASSED")
    return res


# =====================================================================================
# TRAIN harddro20
# =====================================================================================
def train(a, P, D):
    print("\n" + "=" * 90 + f"\n TRAIN {NEW} x seeds {a.seed_list}: {NEW_CONFIG['label']}\n" + "=" * 90)
    for seed in a.seed_list:
        run = S.Run(a, P, D, NEW, seed)
        start_ep = torch.load(run.ck_last, map_location="cpu", weights_only=False)["epoch"] + 1 if run.ck_last.exists() else 1
        was_done = run.done()
        _DRO_LOG.clear()
        t0, started = time.time(), B.now()
        run.train()
        if not was_done:                                       # per-epoch DRO cell losses (the loss each epoch's weights used)
            rows = [dict(epoch=start_ep + k, **{f"loss_{c}": float(v) for c, v in d["cell_loss"].items()},
                         **{f"share_{c}": float(v) for c, v in d["cell_share"].items()}) for k, d in enumerate(_DRO_LOG)]
            f = run.dir / "dro_cell_losses.csv"
            old = pd.read_csv(f) if f.exists() else pd.DataFrame()
            new = pd.concat([old, pd.DataFrame(rows)], ignore_index=True)
            if len(new):
                new.drop_duplicates("epoch", keep="last").sort_values("epoch").to_csv(f, index=False)
        ck = torch.load(run.ck_best, map_location="cpu", weights_only=False)
        last = torch.load(run.ck_last, map_location="cpu", weights_only=False)
        h = pd.DataFrame(ck["history"])
        meta = dict(run=run.id, config=NEW, seed=seed, sampler=ck["sampler"], epochs_max=ck["epochs"], patience=ck["patience"],
                    epochs_completed=int(len(h)), early_stopped=bool(len(h) < ck["epochs"]), best_epoch=int(ck["best_epoch"]),
                    val_macro_auc=float(ck["val_score"]), val_at_best={k: float(v) for k, v in h[h.epoch == ck["best_epoch"]].iloc[0].items()},
                    temperature=float(ck["temperature"]), threshold=float(ck["threshold"]), last_epoch=int(last["epoch"]),
                    optimizer_updates=int(last["updates"]), lr_final=[g["lr"] for g in last["opt"]["param_groups"]],
                    checkpoints=dict(best=str(run.ck_best), final=str(run.ck_last)),
                    best_contains="EMA model weights, T, threshold, history, config",
                    final_contains="model, EMA, optimizer, LR scheduler, AMP scaler, history, DRO cell losses, sampler log, epoch",
                    training_constants=training_constants(a), provenance=provenance(a),
                    resumed_from_epoch=start_ep if start_ep > 1 else None, this_call_started=started,
                    this_call_seconds=time.time() - t0, already_finished_before_this_call=was_done)
        B.write_json(run.dir / "run_meta.json", meta)
        print(f"  {run.id}: {meta['epochs_completed']}/{meta['epochs_max']} epochs completed"
              f"{' (early stop)' if meta['early_stopped'] else ''} | best epoch {meta['best_epoch']} | val macro-AUC {meta['val_macro_auc']:.4f}")


# =====================================================================================
# LOAD ALL FOUR CONFIGURATIONS
# =====================================================================================
def load_all(a, P) -> Tuple[pd.DataFrame, pd.DataFrame]:
    prs, info = [], []
    for c in CONFIGS:
        for s in SEEDS:
            run = f"{c}_s{s}"
            d = run_dir(a, P, run)
            if not ((d / "predictions.csv").exists() and (d / "best.pt").exists()):
                continue
            ck = torch.load(d / "best.pt", map_location="cpu", weights_only=False)
            if not ck.get("final"):
                continue
            x = pd.read_csv(d / "predictions.csv")
            x["config"], x["seed"], x["run"] = c, s, run
            prs.append(x)
            h = pd.DataFrame(ck["history"])
            info.append(dict(run=run, config=c, seed=s, source="new (201026)" if c == NEW else "existing (run_171026)",
                             sampler=ck["sampler"], epochs_max=ck["epochs"], patience=ck["patience"], epochs_completed=len(h),
                             early_stopped=len(h) < ck["epochs"], best_epoch=ck["best_epoch"], val_macro_auc=ck["val_score"],
                             val_auc_BioDeepAV=float(h.loc[h.epoch == ck["best_epoch"], "val_auc_BioDeepAV"].iloc[0]),
                             val_auc_FFpp=float(h.loc[h.epoch == ck["best_epoch"], "val_auc_FF++"].iloc[0]),
                             temperature=ck["temperature"], threshold=ck["threshold"], dir=str(d)))
    pr = pd.concat(prs, ignore_index=True) if prs else pd.DataFrame()
    ri = pd.DataFrame(info)
    if len(pr):   # every run must hold exactly the same val / test videos
        sets = pr.groupby(["run", "split"]).idx.apply(lambda v: tuple(sorted(v)))
        for sp_ in ("val", "test"):
            if sets.xs(sp_, level="split").nunique() != 1:
                raise SystemExit(f"[STOP] runs do not share the same {sp_} videos - comparison would not be paired")
    return pr, ri


# =====================================================================================
# METRICS
# =====================================================================================
def brier(y, p):
    return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))


def full_metrics(pr: pd.DataFrame, ri: pd.DataFrame) -> pd.DataFrame:
    """Per run x test set: AUC, accuracy, balanced accuracy, precision, recall, specificity, F1, confusion matrix, ECE, Brier.
    Decision rule = the run's validation-selected threshold on temperature-scaled P(fake) (151026 rule, stored in best.pt)."""
    thr = ri.set_index("run").threshold
    out = []
    for run, g in pr[pr.split == "test"].groupby("run"):
        for name, x in [("ALL", g)] + [(d, g[g.dataset == d]) for d in sorted(g.dataset.unique())]:
            m = B.compute_metrics(x.label.values, x.p_fake.values, thr[run])
            out.append(dict(run=run, config=g.config.iloc[0], seed=int(g.seed.iloc[0]), test_set=name, **m,
                            ece=B.ece(x.label.values, x.p_fake.values), brier=brier(x.label.values, x.p_fake.values)))
    return pd.DataFrame(out)


def manip_metrics(pr: pd.DataFrame) -> pd.DataFrame:
    """Per run x folder: fake folders -> recall + AUC against the real videos of the same dataset; real folders -> specificity."""
    out = []
    for run, g in pr[pr.split == "test"].groupby("run"):
        for (d, s), x in g.groupby(["dataset", "subset"]):
            r = dict(run=run, config=g.config.iloc[0], seed=int(g.seed.iloc[0]), dataset=d, subset=s, n=len(x))
            if x.label.iloc[0] == 1:
                reals = g[(g.dataset == d) & (g.label == 0)]
                both = pd.concat([x, reals])
                r.update(kind="fake", n_real_reference=len(reals), recall=float((x.pred == 1).mean()),
                         auc_vs_real=roc_auc_score(both.label, both.p_fake), mean_p_fake=float(x.p_fake.mean()))
            else:
                r.update(kind="real", specificity=float((x.pred == 0).mean()), mean_p_fake=float(x.p_fake.mean()))
            out.append(r)
    return pd.DataFrame(out)


def mean_sd(df: pd.DataFrame, by: List[str], cols: List[str]) -> pd.DataFrame:
    rows = []
    for k, g in df.groupby(by):
        r = dict(zip(by, k if isinstance(k, tuple) else (k,)), n_seeds=g.seed.nunique())
        for c in cols:
            if c in g and g[c].notna().any():
                r[f"{c}_mean"] = g[c].mean()
                r[f"{c}_sd"] = g[c].std(ddof=1) if g[c].notna().sum() > 1 else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


def fmt(m, s, nd=3):
    if not np.isfinite(m):
        return "n/a"
    return f"{m:.{nd}f} ± {s:.{nd}f}" if np.isfinite(s) else f"{m:.{nd}f}"


# =====================================================================================
# PAIRED CLUSTER BOOTSTRAP (171026 resampling, extended metrics, bootstrap p + Holm)
# =====================================================================================
def boot_replicates(a, pr: pd.DataFrame) -> Dict[str, dict]:
    """Per test set: {(config, seed): {metric: [1 + n_boot]}} (index 0 = point estimate). Group weights, RNG seeds and core
    metric formulas are those of 171026.bootstrap_compare (verified equal in selftest / evaluate); added: per-manipulation
    AUC vs the dataset's reals, recall of every FF++ method, specificity per dataset."""
    te = pr[pr.split == "test"]
    out = {}
    for ds, g in [("ALL", te)] + [(d, te[te.dataset == d]) for d in sorted(te.dataset.unique())]:
        groups = np.array(sorted(g.group.unique()))
        gpos = {x: i for i, x in enumerate(groups)}
        rng = np.random.default_rng([171026, len(ds)])
        Wg = np.vstack([np.ones(len(groups)), rng.multinomial(len(groups), np.full(len(groups), 1 / len(groups)), size=a.n_boot)])
        per = {}
        for run, h in g.groupby("run"):
            W = Wg[:, h.group.map(gpos).values]
            y, pd_, pf, sub, dset = h.label.values, h.pred.values, h.p_fake.values, h.subset.values, h.dataset.values
            o = dict(auc=S.WeightedAUC(y, pf)(W))
            for lab, nm in ((1, "fake_detected"), (0, "real_recognised")):
                m = y == lab
                o[nm] = (W[:, m] * (pd_[m] == lab)).sum(1) / np.maximum(W[:, m].sum(1), 1e-9) if m.any() else np.full(len(Wg), np.nan)
            o["balanced_accuracy"] = (o["fake_detected"] + o["real_recognised"]) / 2
            for s_ in (HARD_TYPES if ds == "ALL" else FF_FAKES):  # ALL keeps 171026's four recall metrics only
                m = sub == s_
                if m.any():
                    o[f"recall_{s_}"] = (W[:, m] * (pd_[m] == 1)).sum(1) / np.maximum(W[:, m].sum(1), 1e-9)
                    if ds != "ALL":
                        mm = m | ((y == 0) & (dset == "FF++"))
                        o[f"auc_{s_}"] = S.WeightedAUC(y[mm], pf[mm])(W[:, mm])
            per[(h.config.iloc[0], int(h.seed.iloc[0]))] = o
        out[ds] = per
    return out


def p_boot(boot: np.ndarray) -> float:
    boot = boot[np.isfinite(boot)]
    if len(boot) <= 20:
        return np.nan
    return float(min(1.0, 2 * min((boot <= 0).mean(), (boot >= 0).mean()) + 1 / len(boot)))


def holm(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    ok = np.isfinite(p)
    adj = np.full(len(p), np.nan)
    pv = p[ok]
    order = np.argsort(pv)
    m = len(pv)
    run_max, res = 0.0, np.empty(m)
    for k, i in enumerate(order):
        run_max = max(run_max, min(1.0, (m - k) * pv[i]))
        res[i] = run_max
    adj[ok] = res
    return adj


def compare(reps: Dict[str, dict], comparisons) -> pd.DataFrame:
    rows = []
    for ds, per in reps.items():
        for hi, lo in comparisons:
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
                                 per_seed=";".join(f"{s}:{x:+.4f}" for s, x in zip(seeds, pt)), ci95_low=lo_ci, ci95_high=hi_ci,
                                 verdict=v, p_boot=p_boot(boot), primary=(hi, lo) in PRIMARY))
    df = pd.DataFrame(rows)
    if len(df):
        df["p_holm"] = np.nan
        prim = df.primary.values
        df.loc[prim, "p_holm"] = holm(df.loc[prim, "p_boot"].values)
        df["significant_holm_0.05"] = df.p_holm < 0.05
    return df


def config_ci(reps: Dict[str, dict]) -> pd.DataFrame:
    """Seed-mean of every metric per configuration with a 95 % cluster-bootstrap CI (groups resampled, seeds averaged)."""
    rows = []
    for ds, per in reps.items():
        for c in CONFIGS:
            seeds = sorted(s for cc, s in per if cc == c)
            if not seeds:
                continue
            for met in per[(c, seeds[0])]:
                arr = np.nanmean(np.vstack([per[(c, s)][met] for s in seeds]), 0)
                b = arr[1:][np.isfinite(arr[1:])]
                rows.append(dict(test_set=ds, config=c, metric=met, n_seeds=len(seeds), seed_mean=arr[0],
                                 ci95_low=np.percentile(b, 2.5) if len(b) > 20 else np.nan,
                                 ci95_high=np.percentile(b, 97.5) if len(b) > 20 else np.nan))
    return pd.DataFrame(rows)


# =====================================================================================
# XAI: Grad-CAM++ faithfulness + stability
# =====================================================================================
def apply_probe(clip: np.ndarray, kind: str, v: float) -> np.ndarray:
    """Same cv2 / arithmetic operators as 151026.augment_clip, applied alone and identically to every frame."""
    out = np.empty_like(clip)
    S_ = clip.shape[1]
    for t, f in enumerate(clip):
        if kind == "jpeg":
            ok, enc = cv2.imencode(".jpg", f[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, int(v)])
            f = cv2.imdecode(enc, cv2.IMREAD_COLOR)[..., ::-1] if ok else f
        elif kind == "blur":
            f = cv2.GaussianBlur(f, (0, 0), v)
        elif kind == "down":
            k = max(16, int(S_ * v))
            f = cv2.resize(cv2.resize(f, (k, k), interpolation=cv2.INTER_AREA), (S_, S_), interpolation=cv2.INTER_LINEAR)
        else:
            g = f.astype(np.float32)
            if kind == "sat":
                gray = g.mean(-1, keepdims=True)
                g = gray + (g - gray) * v
            elif kind == "contrast":
                g = (g - 128.0) * v + 128.0
            elif kind == "bright":
                g = g + 255.0 * v
            f = np.clip(g, 0, 255).astype(np.uint8)
        out[t] = f
    return out


def cam_metrics(a_: np.ndarray, b_: np.ndarray, top: float = 0.2) -> Tuple[float, float, float]:
    """Pearson r, top-20 % IoU, centre-of-mass shift (fraction of side) of two [H,W] maps (definition of 161026.cam_metrics)."""
    r = float(np.corrcoef(a_.ravel(), b_.ravel())[0, 1]) if a_.std() > 1e-8 and b_.std() > 1e-8 else float("nan")
    ta, tb = a_ >= np.quantile(a_, 1 - top), b_ >= np.quantile(b_, 1 - top)
    iou = float((ta & tb).sum() / max(1, (ta | tb).sum()))
    yy, xx = np.mgrid[0:a_.shape[0], 0:a_.shape[1]]

    def com(c):
        w = c.sum() + 1e-8
        return np.array([(yy * c).sum() / w, (xx * c).sum() / w])
    return r, iou, float(np.linalg.norm(com(a_) - com(b_)) / a_.shape[0])


def xai_pick(D, n: int) -> np.ndarray:
    """Exactly the 171026 E6 video sample (same RNG seed, same (dataset, label) strata, same order)."""
    te = D.idx("test")
    df = pd.DataFrame(dict(idx=te, dataset=D.ds[te], label=D.y[te]))
    rng = np.random.default_rng(171026)
    pick = []
    for _, g in df.groupby(["dataset", "label"]):
        pick += list(rng.choice(g.idx.values, size=min(len(g), max(1, n // 4)), replace=False))
    return np.array(pick)


def xai_run(a, P, D, run: str, pick: np.ndarray):
    out = P.xai / run
    f_faith, f_stab = out / "faithfulness.csv", out / "stability.csv"
    if f_faith.exists() and f_stab.exists():
        print(f"  [resume] XAI {run} done")
        return
    ck_p = run_dir(a, P, run) / "best.pt"
    if not ck_p.exists():
        print(f"  [skip] XAI {run}: no checkpoint")
        return
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    obj = torch.load(ck_p, map_location=dev, weights_only=False)
    model = B.DeepfakeDetector(pretrained=False).to(dev)
    model.load_state_dict(obj["model"])
    model.eval()
    T_, thr = obj["temperature"], obj["threshold"]
    rng = np.random.default_rng(XAI_SEED)                  # same random maps / tie-breaks for every run (paired)
    fr, sr = [], []
    t0 = time.time()
    for n, i in enumerate(pick):
        clip = np.asarray(D.C.arr[i])
        with torch.no_grad():
            p = float(B.sigmoid(model(B.normalise(clip[None], dev))["logit"].float().cpu().numpy() / T_)[0])
        target = int(p >= thr)
        maps = S.explain_all(model, dev, clip, target, T_, rng, methods=["gradcam++", "random"])
        base = dict(run=run, config=run.rsplit("_s", 1)[0], seed=int(run.rsplit("_s", 1)[1]), idx=int(i), dataset=D.ds[i],
                    subset=D.subset[i], group=D.group[i], label=int(D.y[i]), p_fake=p, pred=target, correct=bool(target == D.y[i]))
        for meth, (mp, sec) in maps.items():
            fc = S.faith_curves(model, dev, clip, mp, target, T_, rng)
            fr.append(dict(base, method=meth, seconds=sec, **{k: v for k, v in fc.items() if not k.startswith("curve")},
                           **{f"{k}_{j}": x for k, v in fc.items() if k.startswith("curve") for j, x in enumerate(v)}))
        cam = maps["gradcam++"][0]
        for name, (kind, v) in PROBES.items():
            pc = apply_probe(clip, kind, v)
            x = B.normalise(pc[None], dev)
            cp, lg, _ = B.gradcam_pp(model, x, None, target)
            pp = float(B.sigmoid(lg / T_)[0])
            ms = np.array([cam_metrics(cam[t], cp[0][t]) for t in range(len(clip))], float)
            sr.append(dict(base, probe=name, pearson=float(np.nanmean(ms[:, 0])), top20_iou=float(ms[:, 1].mean()),
                           com_shift=float(ms[:, 2].mean()), p_fake_probe=pp, prediction_flip=bool(int(pp >= thr) != target)))
        gc.collect()                                   # gradcam_pp leaves a tensor <-> hook reference cycle that holds the graph
        el = time.time() - t0
        sys.stdout.write(f"\r  [XAI {run}] {B.bar(n + 1, len(pick), 16)} {n + 1}/{len(pick)} | ETA {B.fmt_time(el / (n + 1) * (len(pick) - n - 1))}   ")
        sys.stdout.flush()
    sys.stdout.write("\n")
    pd.DataFrame(sr).to_csv(f_stab, index=False)
    pd.DataFrame(fr).to_csv(f_faith, index=False)       # written last: its presence marks the run as complete
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def xai_all(a, P, D):
    print("\n" + "=" * 90 + f"\n XAI: Grad-CAM++ faithfulness + stability, {a.xai_n} test videos x every run\n" + "=" * 90)
    pick = xai_pick(D, a.xai_n)
    e6 = a.suite_root / "metrics" / "E6_xai_faithfulness.csv"
    if e6.exists() and a.xai_n == S.XAI_N:
        same = set(pd.read_csv(e6).idx) == set(pick)
        print(f"  sample identical to the 171026 E6 sample: {same}")
        if not same:
            raise SystemExit("[STOP] XAI sample differs from the 171026 E6 sample")
    for c in CONFIGS:
        for s in a.seed_list:
            xai_run(a, P, D, f"{c}_s{s}", pick)


def xai_load(P) -> Tuple[pd.DataFrame, pd.DataFrame]:
    fa = [pd.read_csv(f) for f in sorted(P.xai.glob("*/faithfulness.csv"))]
    st = [pd.read_csv(P.xai / f.parent.name / "stability.csv") for f in sorted(P.xai.glob("*/faithfulness.csv"))]
    return (pd.concat(fa, ignore_index=True) if fa else pd.DataFrame()), (pd.concat(st, ignore_index=True) if st else pd.DataFrame())


def xai_per_video(fa: pd.DataFrame, st: pd.DataFrame) -> pd.DataFrame:
    """One row per run x video, every metric oriented so that HIGHER = better."""
    pv = fa.pivot_table(index=["run", "config", "seed", "idx", "group", "dataset", "pred"], columns="method",
                        values=["deletion_auc_grey", "insertion_auc_grey", "deletion_auc_blur", "insertion_auc_blur"]).reset_index()
    out = pv[["run", "config", "seed", "idx", "group", "dataset", "pred"]].copy()
    out.columns = ["run", "config", "seed", "idx", "group", "dataset", "pred"]
    for k in ("grey", "blur"):
        out[f"deletion_gain_{k}"] = (pv[(f"deletion_auc_{k}", "random")] - pv[(f"deletion_auc_{k}", "gradcam++")]).values
        out[f"insertion_gain_{k}"] = (pv[(f"insertion_auc_{k}", "gradcam++")] - pv[(f"insertion_auc_{k}", "random")]).values
        out[f"gradcam_deletion_auc_{k}"] = pv[(f"deletion_auc_{k}", "gradcam++")].values
        out[f"gradcam_insertion_auc_{k}"] = pv[(f"insertion_auc_{k}", "gradcam++")].values
    s = st.groupby(["run", "idx"]).agg(stability_iou=("top20_iou", "mean"), stability_pearson=("pearson", "mean"),
                                       stability_com_shift=("com_shift", "mean"), probe_flip_rate=("prediction_flip", "mean")).reset_index()
    out = out.merge(s, on=["run", "idx"], how="left")
    out["pred_class"] = np.where(out.pred == 1, "FAKE", "REAL")
    return out


XAI_KEYS = ["deletion_gain_grey", "insertion_gain_grey", "deletion_gain_blur", "insertion_gain_blur", "stability_iou", "stability_pearson"]


def xai_compare(a, xv: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Config means (seed-averaged per video) + paired differences; cluster bootstrap over the videos' identity groups."""
    m = xv.groupby(["config", "idx", "group"])[XAI_KEYS + ["probe_flip_rate"]].mean().reset_index()
    groups = np.array(sorted(m.group.unique()))
    gpos = {g: i for i, g in enumerate(groups)}
    rng = np.random.default_rng(201026)
    Wg = np.vstack([np.ones(len(groups)), rng.multinomial(len(groups), np.full(len(groups), 1 / len(groups)), size=a.n_boot)])
    est = {}
    for c, g in m.groupby("config"):
        g = g.sort_values("idx")
        W = Wg[:, g.group.map(gpos).values]
        est[c] = {k: (W * g[k].values).sum(1) / W.sum(1) for k in XAI_KEYS + ["probe_flip_rate"]}
        est[c]["_idx"] = tuple(g.idx)
    rows, crow = [], []
    for c in est:
        n_seeds = xv[xv.config == c].seed.nunique()
        for k in XAI_KEYS + ["probe_flip_rate"]:
            v = est[c][k]
            crow.append(dict(config=c, metric=k, n_videos=len(est[c]["_idx"]), n_seeds=n_seeds, mean=v[0],
                             ci95_low=np.percentile(v[1:], 2.5), ci95_high=np.percentile(v[1:], 97.5)))
    for hi, lo in PRIMARY + SECONDARY:
        if hi not in est or lo not in est or est[hi]["_idx"] != est[lo]["_idx"]:
            continue
        for k in XAI_KEYS:
            d = est[hi][k] - est[lo][k]
            l_, h_ = np.percentile(d[1:], [2.5, 97.5])
            v = ("non-inferior (CI lower bound > -margin)" if l_ > -XAI_MARGIN else
                 ("material reduction (CI upper bound < -margin)" if h_ < -XAI_MARGIN else "inconclusive"))
            rows.append(dict(comparison=f"{hi} - {lo}", metric=k, mean_diff=d[0], ci95_low=l_, ci95_high=h_, p_boot=p_boot(d[1:]),
                             margin=XAI_MARGIN, verdict=v, primary=(hi, lo) in PRIMARY))
    return pd.DataFrame(crow), pd.DataFrame(rows)


# =====================================================================================
# FIGURES
# =====================================================================================
def _style():
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                         "grid.alpha": 0.25, "grid.linewidth": 0.6, "legend.frameon": False})


def fig_training(a, P, ri):
    fig, ax = plt.subplots(1, 4, figsize=(16, 3.6))
    for _, r in ri.iterrows():
        h = pd.read_csv(Path(r.dir) / "history.csv")
        kw = dict(color=COLOR[r.config], alpha=0.8, lw=1.6 if r.config == NEW else 1.0, label=r.config if r.seed == 42 else None)
        for k, col in enumerate(["train_loss", "val_score", "val_auc_BioDeepAV", "val_auc_FF++"]):
            ax[k].plot(h.epoch, h[col], **kw)
            if col == "val_score":
                ax[k].scatter([r.best_epoch], [r.val_macro_auc], color=COLOR[r.config], s=18, zorder=3)
    for k, t in enumerate(["training loss", "validation macro-AUC (selection score)", "validation AUC BioDeepAV", "validation AUC FF++"]):
        ax[k].set_title(t)
        ax[k].set_xlabel("epoch")
    ax[0].legend(fontsize=8)
    fig.suptitle("Lines: seeds 42 / 123 / 2026; dots: selected (best validation) epoch", fontsize=9)
    plt.tight_layout()
    plt.savefig(P.figures / "T6_training_validation_curves.png", dpi=130)
    plt.close(fig)
    rows = []
    for c in ("harddro12", NEW):
        for s in SEEDS:
            f = run_dir(a, P, f"{c}_s{s}") / "sampler_shares.csv"
            if f.exists():
                x = pd.read_csv(f)
                x["config"] = c
                rows.append(x)
    if rows:
        x = pd.concat(rows)
        cols = [c for c in x.columns if c.startswith("share_")]
        fig, ax = plt.subplots(1, 2, figsize=(13, 3.8), sharey=True)
        cmap = plt.get_cmap("tab10")
        for k, c in enumerate(("harddro12", NEW)):
            g = x[x.config == c].groupby("epoch")[cols].mean()
            for j, col in enumerate(cols):
                ax[k].plot(g.index, g[col], color=cmap(j), lw=1.4, label=col.replace("share_", ""))
                if len(g):
                    ax[k].annotate(col.replace("share_", "").replace("FF++|", ""), (g.index[-1], g[col].iloc[-1]), fontsize=6,
                                   xytext=(3, 0), textcoords="offset points", va="center")
            ax[k].set_title(f"{c}: group-DRO sampling share per cell (seed mean)")
            ax[k].set_xlabel("epoch")
        ax[0].set_ylabel("share of training draws")
        ax[1].legend(fontsize=6, loc="upper left", bbox_to_anchor=(1.08, 1))
        plt.tight_layout()
        plt.savefig(P.figures / "T6_dro_sampling_shares.png", dpi=130)
        plt.close(fig)


def fig_roc_cm(P, pr, ri):
    te = pr[pr.split == "test"]
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
    for k, ts in enumerate(["ALL", "BioDeepAV", "FF++"]):
        x = te if ts == "ALL" else te[te.dataset == ts]
        for (c, s), g in x.groupby(["config", "seed"]):
            fpr, tpr, _ = roc_curve(g.label, g.p_fake)
            ax[k].plot(fpr, tpr, color=COLOR[c], lw=1.0, alpha=0.75,
                       label=f"{c} (AUC {np.mean([roc_auc_score(h.label, h.p_fake) for _, h in x[x.config == c].groupby('seed')]):.3f})" if s == 42 else None)
        ax[k].plot([0, 1], [0, 1], ":", color="#999", lw=0.8)
        ax[k].set_title(f"ROC - test {ts} (one line per seed)")
        ax[k].set_xlabel("false positive rate (real called fake)")
        ax[k].set_ylabel("true positive rate (fake detected)")
        ax[k].legend(fontsize=7, loc="lower right")
    plt.tight_layout()
    plt.savefig(P.figures / "T6_roc_curves.png", dpi=130)
    plt.close(fig)
    fig, ax = plt.subplots(2, 4, figsize=(14, 6.4))
    for j, c in enumerate(CONFIGS):
        for i, d in enumerate(["BioDeepAV", "FF++"]):
            g = te[(te.config == c) & (te.dataset == d)]
            axx = ax[i, j]
            axx.grid(False)
            if not len(g):
                axx.axis("off")
                continue
            cm = np.array([[((g.label == t) & (g.pred == p)).sum() for p in (0, 1)] for t in (0, 1)])
            rn = cm / cm.sum(1, keepdims=True)
            axx.imshow(rn, cmap="Blues", vmin=0, vmax=1)
            for t in (0, 1):
                for p in (0, 1):
                    axx.text(p, t, f"{cm[t, p]}\n{rn[t, p]:.1%}", ha="center", va="center", fontsize=8,
                             color="white" if rn[t, p] > 0.6 else "#222")
            axx.set_xticks([0, 1], ["pred REAL", "pred FAKE"])
            axx.set_yticks([0, 1], ["true REAL", "true FAKE"])
            axx.set_title(f"{c} - {d} ({g.seed.nunique()} seeds summed)", fontsize=8)
    plt.tight_layout()
    plt.savefig(P.figures / "T6_confusion_matrices.png", dpi=130)
    plt.close(fig)


def fig_manip(P, mm):
    f = mm[(mm.kind == "fake")]
    subs = [s for s in ["NeuralTextures", "FaceSwap", "Face2Face", "FaceShifter", "Deepfakes", "DeepFakeDetection"] if s in set(f.subset)]
    subs += [("BioDeepAV", "fake")] if ((f.dataset == "BioDeepAV")).any() else []
    fig, ax = plt.subplots(2, 1, figsize=(13, 7))
    w = 0.8 / len(CONFIGS)
    for k, (met, title) in enumerate([("recall", "fake detected (recall) at the validation threshold"),
                                      ("auc_vs_real", "AUC vs the real test videos of the same dataset")]):
        for j, c in enumerate(CONFIGS):
            for i, s in enumerate(subs):
                x = f[(f.config == c) & ((f.subset == s) if isinstance(s, str) else ((f.dataset == s[0]) & (f.subset == s[1])))][met]
                if not len(x):
                    continue
                xpos = i + (j - (len(CONFIGS) - 1) / 2) * w
                ax[k].bar(xpos, x.mean(), width=w * 0.9, color=COLOR[c], label=c if i == 0 else None)
                if len(x) > 1:
                    ax[k].errorbar(xpos, x.mean(), yerr=x.std(ddof=1), color="#333", capsize=2, lw=0.8)
                ax[k].scatter(np.full(len(x), xpos), x, s=5, color="#222", zorder=3)
        ax[k].set_xticks(range(len(subs)), [s if isinstance(s, str) else "BioDeepAV fake" for s in subs])
        ax[k].set_title(title)
        ax[k].set_ylim(0.3 if met == "recall" else 0.6, 1.01)
    h_, l_ = ax[0].get_legend_handles_labels()
    fig.legend(h_, l_, ncol=4, fontsize=8, loc="lower center")
    fig.suptitle("Per manipulation, test set. Bars: seed mean; whiskers: SD over seeds; dots: seeds. NT/FS/F2F/FSh = hard types (H2)", fontsize=9)
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(P.figures / "T6_per_manipulation.png", dpi=130)
    plt.close(fig)


def fig_forest(P, cmp_):
    keys = [("ALL", "auc"), ("BioDeepAV", "auc"), ("FF++", "auc"), ("ALL", "balanced_accuracy"), ("BioDeepAV", "balanced_accuracy"),
            ("FF++", "balanced_accuracy")] + [("FF++", f"recall_{t}") for t in HARD_TYPES] + [("FF++", f"auc_{t}") for t in HARD_TYPES]
    fig, ax = plt.subplots(1, len(PRIMARY), figsize=(13, 6), sharey=True)
    for k, (hi, lo) in enumerate(PRIMARY):
        x = cmp_[cmp_.comparison == f"{hi} - {lo}"].set_index(["test_set", "metric"])
        for i, key in enumerate(keys):
            if key not in x.index:
                continue
            r = x.loc[key]
            col = "#1baf7a" if r.verdict == "supported improvement" else ("#c0392b" if r.verdict == "supported deterioration" else "#666")
            ax[k].plot([r.ci95_low, r.ci95_high], [i, i], color=col, lw=2)
            ax[k].plot(r.mean_diff, i, "o", color=col, ms=5)
        ax[k].axvline(0, color="#333", lw=0.8)
        ax[k].set_yticks(range(len(keys)), [f"{t}: {m}" for t, m in keys])
        ax[k].invert_yaxis()
        ax[k].set_title(f"{hi} - {lo} (seed-mean difference, 95 % cluster-bootstrap CI)", fontsize=9)
    fig.suptitle("green = supported improvement, red = supported deterioration, grey = inconclusive (CI excl. 0 AND all seeds agree)", fontsize=8)
    plt.tight_layout()
    plt.savefig(P.figures / "T6_bootstrap_forest.png", dpi=130)
    plt.close(fig)


def fig_reliability(P, pr):
    te = pr[pr.split == "test"]
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 4.2))
    for k, d in enumerate(["BioDeepAV", "FF++"]):
        for c in CONFIGS:
            g = te[(te.config == c) & (te.dataset == d)]
            if not len(g):
                continue
            bins = np.linspace(0, 1, 11)
            b = np.clip(np.digitize(g.p_fake, bins) - 1, 0, 9)
            xs = [g.p_fake.values[b == j].mean() for j in range(10) if (b == j).sum() >= 10]
            ys = [g.label.values[b == j].mean() for j in range(10) if (b == j).sum() >= 10]
            ax[k].plot(xs, ys, "o-", color=COLOR[c], ms=3, lw=1.2, label=f"{c} (ECE {B.ece(g.label, g.p_fake):.3f})")
        ax[k].plot([0, 1], [0, 1], ":", color="#999")
        ax[k].set_title(f"Reliability, test {d} (seeds pooled, temperature-scaled)")
        ax[k].set_xlabel("predicted P(fake)")
        ax[k].set_ylabel("observed fake rate")
        ax[k].legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(P.figures / "T6_reliability.png", dpi=130)
    plt.close(fig)


def fig_xai(P, xc):
    if not len(xc):
        return
    ks = [k for k in XAI_KEYS if k in set(xc.metric)]
    fig, ax = plt.subplots(1, len(ks), figsize=(3 * len(ks), 3.6))
    for k, met in enumerate(ks):
        x = xc[xc.metric == met].set_index("config")
        for j, c in enumerate([c for c in CONFIGS if c in x.index]):
            r = x.loc[c]
            ax[k].bar(j, r["mean"], color=COLOR[c], width=0.65)
            ax[k].errorbar(j, r["mean"], yerr=[[r["mean"] - r.ci95_low], [r.ci95_high - r["mean"]]], color="#333", capsize=3, lw=0.8)
        ax[k].set_xticks(range(len(x)), [c for c in CONFIGS if c in x.index], rotation=30)
        ax[k].set_title(met.replace("_", " "), fontsize=8)
        ax[k].axhline(0, color="#333", lw=0.6)
    fig.suptitle("Grad-CAM++ (higher = better). Gains = vs a random map on the same video; stability = vs clean map under "
                 "photometric probes. Whiskers: 95 % cluster-bootstrap CI", fontsize=8)
    plt.tight_layout()
    plt.savefig(P.figures / "T6_xai_comparison.png", dpi=130)
    plt.close(fig)


# =====================================================================================
# EVALUATE + REPORT
# =====================================================================================
def verdict_word(rows: pd.DataFrame) -> str:
    v = rows.verdict.tolist()
    return ", ".join(f"{r.comparison.split(' - ')[1]}: {r.verdict} ({r.mean_diff:+.4f}, CI {r.ci95_low:+.4f}..{r.ci95_high:+.4f}, "
                     f"Holm p {r.p_holm:.3f})" for r in rows.itertuples()) if v else "n/a"


def hypotheses(cmp_: pd.DataFrame, xcmp: pd.DataFrame) -> dict:
    P_ = cmp_[cmp_.primary] if len(cmp_) else cmp_

    def get(ts, met):
        return P_[(P_.test_set == ts) & (P_.metric == met)]
    H = {}
    # H1: overall AUC improves vs BOTH comparators
    r = get("ALL", "auc")
    sup = (r.verdict == "supported improvement").sum()
    det = (r.verdict == "supported deterioration").sum()
    H["H1"] = dict(statement="harddro20 improves overall test ROC-AUC vs harddro12 and base20",
                   verdict="supported" if sup == len(PRIMARY) and len(r) == len(PRIMARY) else
                   ("contradicted" if det else ("partially supported" if sup else "not supported (inconclusive)")),
                   evidence=verdict_word(r), holm_significant=int(r["significant_holm_0.05"].sum()) if len(r) else 0)
    # H2: hard-type recall and AUC
    rr = P_[(P_.test_set == "FF++") & P_.metric.isin([f"recall_{t}" for t in HARD_TYPES] + [f"auc_{t}" for t in HARD_TYPES])]
    n_imp, n_det, n = int((rr.verdict == "supported improvement").sum()), int((rr.verdict == "supported deterioration").sum()), len(rr)
    H["H2"] = dict(statement="harddro20 improves recall and AUC on NeuralTextures, FaceSwap, Face2Face, FaceShifter",
                   verdict="supported" if n and n_imp == n else ("contradicted" if n_det and not n_imp else
                                                                 ("mixed" if n_det else ("partially supported" if n_imp else "not supported (inconclusive)"))),
                   evidence=f"{n_imp}/{n} (type x metric x comparator) supported improvements, {n_det} supported deteriorations; "
                            f"{int(rr['significant_holm_0.05'].sum()) if n else 0} survive Holm",
                   detail=rr[["comparison", "metric", "mean_diff", "ci95_low", "ci95_high", "verdict", "p_holm"]].to_dict("records"))
    # H3: BioDeepAV maintained (non-inferiority) AND FF++ improved
    bio, ff = get("BioDeepAV", "auc"), get("FF++", "auc")
    bio_ok = len(bio) == len(PRIMARY) and bool((bio.ci95_low > -BIO_MARGIN).all()) and not (bio.verdict == "supported deterioration").any()
    ff_ok = len(ff) == len(PRIMARY) and bool((ff.verdict == "supported improvement").all())
    bio_bad = bool((bio.verdict == "supported deterioration").any())
    H["H3"] = dict(statement=f"harddro20 maintains BioDeepAV AUC (non-inferiority margin {BIO_MARGIN}) while improving FF++ AUC, vs both",
                   verdict="supported" if bio_ok and ff_ok else ("contradicted" if bio_bad else
                                                                 ("partially supported" if bio_ok or (ff.verdict == "supported improvement").any() else "not supported (inconclusive)")),
                   evidence=f"BioDeepAV: {verdict_word(bio)} | non-inferior: {bio_ok}. FF++: {verdict_word(ff)}")
    # H4: explanation reliability not materially reduced
    if len(xcmp):
        xp = xcmp[xcmp.primary & xcmp.metric.isin(["deletion_gain_grey", "insertion_gain_grey", "stability_iou", "stability_pearson"])]
        ni = (xp.verdict.str.startswith("non-inferior")).all() and len(xp) > 0
        bad = xp.verdict.str.startswith("material").any()
        H["H4"] = dict(statement=f"harddro20 does not materially reduce Grad-CAM++ faithfulness or stability (margin {XAI_MARGIN})",
                       verdict="supported" if ni else ("contradicted" if bad else "inconclusive"),
                       evidence="; ".join(f"{r.comparison} {r.metric}: {r.mean_diff:+.3f} (CI {r.ci95_low:+.3f}..{r.ci95_high:+.3f}) {r.verdict}"
                                          for r in xp.itertuples()))
    else:
        H["H4"] = dict(statement="explanation reliability", verdict="not evaluated (XAI phase not run)", evidence="")
    return H


def evaluate(a, P, D, pre: dict):
    print("\n" + "=" * 90 + "\n EVALUATE: four configurations, paired statistics, report\n" + "=" * 90)
    _style()
    pr, ri = load_all(a, P)
    if not len(pr):
        print("  no finished runs")
        return
    ri.to_csv(P.metrics / "T6_runs.csv", index=False)
    pr.to_csv(P.predictions / "T6_all_predictions.csv", index=False)
    for run, g in pr[pr.config == NEW].groupby("run"):
        g.to_csv(P.predictions / f"{run}_predictions.csv", index=False)
    fm = full_metrics(pr, ri)
    fm.to_csv(P.metrics / "T6_run_metrics.csv", index=False)
    mcols = ["auc", "accuracy", "balanced_accuracy", "precision", "recall", "specificity", "f1", "ece", "brier"]
    fs = mean_sd(fm, ["config", "test_set"], mcols + ["tn", "fp", "fn", "tp"])
    fs.to_csv(P.metrics / "T6_seed_summary.csv", index=False)
    mm = manip_metrics(pr)
    mm.to_csv(P.metrics / "T6_per_manipulation_runs.csv", index=False)
    ms = mean_sd(mm, ["config", "dataset", "subset", "kind"], ["n", "recall", "auc_vs_real", "specificity", "mean_p_fake"])
    ms.to_csv(P.metrics / "T6_per_manipulation_summary.csv", index=False)
    # statistics: existing 171026 bootstrap must equal the extension on the shared metrics (same methodology)
    reps = boot_replicates(a, pr)
    cmp_ = compare(reps, PRIMARY + SECONDARY)
    S.COMPARISONS = PRIMARY + SECONDARY
    ref = S.bootstrap_compare(a, pr)
    j = ref.merge(cmp_, on=["test_set", "comparison", "metric"], suffixes=("_171026", ""))
    boot_check = dict(rows_compared=len(j), max_abs_diff_mean=float((j.mean_diff_171026 - j.mean_diff).abs().max()) if len(j) else np.nan,
                      max_abs_diff_ci=float(np.nanmax(np.abs(np.r_[j.ci95_low_171026 - j.ci95_low, j.ci95_high_171026 - j.ci95_high]))) if len(j) else np.nan,
                      verdicts_identical=bool((j.verdict_171026 == j.verdict).all()) if len(j) else None)
    if len(j) and (boot_check["max_abs_diff_mean"] > 1e-9 or boot_check["max_abs_diff_ci"] > 1e-9):
        raise SystemExit(f"[STOP] bootstrap extension disagrees with 171026.bootstrap_compare: {boot_check}")
    print(f"  bootstrap extension == 171026.bootstrap_compare on {len(j)} shared rows")
    cmp_.to_csv(P.metrics / "T6_bootstrap_comparisons.csv", index=False)
    cci = config_ci(reps)
    cci.to_csv(P.metrics / "T6_config_bootstrap_ci.csv", index=False)
    # calibration: 171026 E5 on the four configurations (validation-fitted, applied to test)
    S.CONFIG_COLOR.update(COLOR)
    e5, e5s = S.exp_e5(a, P, pr)
    cal = e5[(e5.target_coverage == 1.0) & (e5.threshold_method == "val_mean_bacc")]
    cal_s = cal.groupby(["config", "calibration"]).agg(n_runs=("run", "nunique"), threshold=("threshold", "mean"), threshold_sd=("threshold", "std"),
                                                       ece=("ece", "mean"), ece_sd=("ece", "std"), brier=("brier", "mean"), brier_sd=("brier", "std"),
                                                       mean_bacc=("mean_bacc", "mean"), mean_bacc_sd=("mean_bacc", "std"),
                                                       ece_BioDeepAV=("ece_BioDeepAV", "mean"), **{"ece_FF++": ("ece_FF++", "mean")},
                                                       bacc_BioDeepAV=("bacc_BioDeepAV", "mean"), **{"bacc_FF++": ("bacc_FF++", "mean")}).reset_index()
    cal_s.to_csv(P.metrics / "T6_calibration_summary.csv", index=False)
    # XAI
    fa, st = xai_load(P)
    xv = xai_per_video(fa, st) if len(fa) and len(st) else pd.DataFrame()
    xc, xcmp, xcls = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    if len(xv):
        xv.to_csv(P.metrics / "T6_xai_per_video.csv", index=False)
        xc, xcmp = xai_compare(a, xv)
        xc.to_csv(P.metrics / "T6_xai_config_summary.csv", index=False)
        xcmp.to_csv(P.metrics / "T6_xai_comparisons.csv", index=False)
        xcls = xv.groupby(["config", "dataset", "pred_class"]).agg(n=("idx", "size"), **{k: (k, "mean") for k in XAI_KEYS + ["probe_flip_rate"]}).reset_index()
        xcls.to_csv(P.metrics / "T6_xai_by_dataset_class.csv", index=False)
        st.groupby(["config", "probe"]).agg(top20_iou=("top20_iou", "mean"), pearson=("pearson", "mean"),
                                            flip_rate=("prediction_flip", "mean")).reset_index().to_csv(P.metrics / "T6_xai_stability_by_probe.csv", index=False)
    # figures
    fig_training(a, P, ri)
    fig_roc_cm(P, pr, ri)
    fig_manip(P, mm)
    if len(cmp_):
        fig_forest(P, cmp_)
    fig_reliability(P, pr)
    fig_xai(P, xc)
    H = hypotheses(cmp_, xcmp)
    summary = dict(experiment="Test 6 - harddro20", generated=B.now(), configs={c: S.CONFIGS[c] for c in CONFIGS},
                   runs=ri.drop(columns=["dir"]).to_dict("records"), hypotheses=H, bootstrap_check=boot_check,
                   preflight_passed=pre.get("passed"), split_sha256=pre.get("info", {}).get("split_sha256"),
                   inventory_sha256=pre.get("info", {}).get("inventory_sha256"), robustness_evaluable=pre.get("robustness_evaluable"),
                   seed_summary=fs.to_dict("records"), provenance=provenance(a), n_boot=a.n_boot, xai_videos=int(xv.idx.nunique()) if len(xv) else 0)
    B.write_json(P.reports / "summary.json", summary)
    report(a, P, pre, ri, fs, ms, cmp_, cci, cal_s, xc, xcmp, xcls, st, H, boot_check)


def report(a, P, pre, ri, fs, ms, cmp_, cci, cal_s, xc, xcmp, xcls, st, H, boot_check):
    def t(df, nd=3):
        return B.md_table(df, nd, max_rows=200) if len(df) else "_none_"

    def ms_(c, ts, k):
        r = fs[(fs.config == c) & (fs.test_set == ts)]
        return fmt(r[f"{k}_mean"].iloc[0], r[f"{k}_sd"].iloc[0]) if len(r) and f"{k}_mean" in r else "n/a"
    new_runs = ri[ri.config == NEW]
    seeds_new = sorted(new_runs.seed)
    L = [f"# Test 6 (201026): adaptive group-DRO + 20-epoch training (`{NEW}`)\n",
         f"_Generated {B.now()} by 201026.py | existing runs: {a.suite_root} (read-only) | data: {a.base_root} (read-only) | "
         f"new runs: {P.runs}_\n",
         "**Provenance labels.** *Existing* = base12 / harddro12 / base20 checkpoints and predictions produced by 171026.py on 2026-10-17, "
         "re-scored here without retraining. *New* = harddro20, trained by this script. All XAI numbers are newly measured "
         "(171026 E6 had evaluated base12 seed 42 only).\n",
         "## 1. Research objective\n",
         "Does combining adaptive group-DRO sampling with a 20-epoch training schedule improve deepfake detection across BioDeepAV "
         "and FaceForensics++ C23 without sacrificing either dataset or Grad-CAM++ explanation reliability?\n",
         "## 2. Hypotheses (evaluated independently; pre-specified decision rules)\n",
         "- **H1** harddro20 improves overall test ROC-AUC vs harddro12 and vs base20. Rule: 'supported' = supported improvement vs both.\n"
         "- **H2** harddro20 improves recall and AUC on NeuralTextures, FaceSwap, Face2Face, FaceShifter. Rule: count of supported "
         "improvements / deteriorations over 4 types x 2 metrics x 2 comparators.\n"
         f"- **H3** BioDeepAV AUC maintained (CI lower bound of the difference > -{BIO_MARGIN} and no supported deterioration) while FF++ AUC "
         "improves (supported improvement), vs both comparators.\n"
         f"- **H4** Grad-CAM++ faithfulness (deletion / insertion gain over a random map, grey perturbation) and stability (top-20 % IoU, "
         f"Pearson r under photometric probes) are non-inferior: CI lower bound of the difference > -{XAI_MARGIN}, vs both comparators.\n",
         "'Supported improvement / deterioration' is the 171026 rule: 95 % paired cluster-bootstrap CI excludes 0 AND all seeds agree in sign. "
         "Holm-adjusted bootstrap p-values over the whole primary family are reported alongside.\n",
         "## 3. Configurations\n",
         t(pd.DataFrame([dict(config=c, source="new" if c == NEW else "existing (171026)", sampler=S.CONFIGS[c]["sampler"],
                              max_epochs=S.CONFIGS[c]["epochs"], patience=S.CONFIGS[c]["patience"], label=S.CONFIGS[c]["label"]) for c in CONFIGS])), "",
         "Differences: harddro20 vs harddro12 = 20 instead of 12 epochs, early-stopping patience 5 instead of 4, cosine LR schedule stretched "
         "over 20 epochs (all three inherited from the verified base20 definition). harddro20 vs base20 = sampler only (group-DRO over the 9 "
         "dataset x folder cells instead of dataset x class balancing). Model, preprocessing, augmentation, losses, optimiser, EMA, batch, "
         "learning rates and checkpoint rule are identical (same imported code).\n",
         "## 4. Experimental setup\n",
         f"- Seeds {SEEDS}; held-out test set of the 151026 split ({int(pre.get('counts') and sum(v for k, v in pre['counts'].items() if k.endswith('|test')) or 0)} videos).\n"
         f"- Checkpoint = EMA weights at the epoch with the best validation macro-AUC (mean of BioDeepAV and FF++ AUC, equal weight).\n"
         "- Decision threshold and temperature are fitted on validation only (151026 rule: maximise mean balanced accuracy over datasets).\n"
         f"- Paired cluster bootstrap: {a.n_boot} replicates, identity/duplicate groups resampled (multinomial weights), the same "
         "replicate weights applied to every configuration and seed; seed-mean difference per replicate -> percentile 95 % CI. "
         "Bootstrap p = two-sided percentile p (+1/B); Holm correction over all primary rows (harddro20 - harddro12, harddro20 - base20; "
         "every test set and metric). Secondary comparisons (vs base12, and the existing harddro12 / base20 vs base12) are uncorrected context.\n"
         f"- Extension check: the bootstrap extension reproduces 171026.bootstrap_compare exactly on {boot_check['rows_compared']} shared rows "
         f"(max |diff| {boot_check['max_abs_diff_mean']:.1e} mean, {boot_check['max_abs_diff_ci']:.1e} CI).\n",
         "## 5. Dataset and leakage audit\n",
         t(pd.DataFrame([dict(check=k, result="PASS" if v else "FAIL") for k, v in pre.get("checks", {}).items()])), "",
         f"split.csv sha256 `{pre.get('info', {}).get('split_sha256')}`; inventory sha256 `{pre.get('info', {}).get('inventory_sha256')}`; "
         f"face cache `{pre.get('info', {}).get('face_cache')}`. Full record: reports/preflight_audit.json, reports/data_manifest.json.\n",
         "## 6. Training configuration (harddro20)\n",
         "```json\n" + json.dumps(training_constants(a), indent=1) + "\n```\n",
         t(new_runs[["run", "epochs_max", "epochs_completed", "early_stopped", "best_epoch", "val_macro_auc", "val_auc_BioDeepAV",
                     "val_auc_FFpp", "temperature", "threshold"]], 4) if len(new_runs) else "_harddro20 not trained yet_", "",
         "![curves](../figures/T6_training_validation_curves.png)\n![dro](../figures/T6_dro_sampling_shares.png)\n",
         "## 7. Results across seeds (all runs)\n",
         t(ri[["run", "source", "epochs_max", "epochs_completed", "early_stopped", "best_epoch", "val_macro_auc", "val_auc_BioDeepAV", "val_auc_FFpp",
               "threshold"]], 4), "",
         "## 8. Overall detection comparison (test, seed mean ± SD)\n"]
    rows = []
    for c in CONFIGS:
        r = ri[ri.config == c]
        rows.append(dict(config=c, source="new" if c == NEW else "existing", seeds=len(r),
                         epochs_completed=",".join(map(str, r.epochs_completed)), val_macro_auc=fmt(r.val_macro_auc.mean(), r.val_macro_auc.std(ddof=1) if len(r) > 1 else np.nan),
                         AUC_all=ms_(c, "ALL", "auc"), AUC_BioDeepAV=ms_(c, "BioDeepAV", "auc"), AUC_FFpp=ms_(c, "FF++", "auc"),
                         BAcc_all=ms_(c, "ALL", "balanced_accuracy"), F1_all=ms_(c, "ALL", "f1"), ECE_all=ms_(c, "ALL", "ece")))
    L += [t(pd.DataFrame(rows)), ""]
    full = fs[fs.test_set == "ALL"].copy()
    keep = ["config", "n_seeds"] + [c for c in full.columns if c.endswith("_mean")]
    L += ["**All metrics, pooled test set (seed means)**\n", t(full[keep], 3), "", "![roc](../figures/T6_roc_curves.png)\n",
          "![cm](../figures/T6_confusion_matrices.png)\n"]
    for k, ds in ((9, "BioDeepAV"), (10, "FF++")):
        x = fs[fs.test_set == ds]
        L += [f"## {k}. {'BioDeepAV' if ds == 'BioDeepAV' else 'FaceForensics++ C23'} results (test)\n",
              t(pd.DataFrame([dict(config=r.config, **{m: fmt(getattr(r, f'{m}_mean'), getattr(r, f'{m}_sd')) for m in
                                                       ["auc", "accuracy", "balanced_accuracy", "precision", "recall", "specificity", "f1", "ece", "brier"]},
                                   TN_TP=f"{r.tn_mean:.0f} / {r.tp_mean:.0f}", FP_FN=f"{r.fp_mean:.0f} / {r.fn_mean:.0f}") for r in x.itertuples()])), "",
              "**Paired comparisons (primary)**\n",
              t(cmp_[(cmp_.test_set == ds) & cmp_.primary & cmp_.metric.isin(["auc", "balanced_accuracy", "fake_detected", "real_recognised"])]
                [["comparison", "metric", "mean_diff", "per_seed", "ci95_low", "ci95_high", "verdict", "p_holm"]], 4), ""]
    L += ["## 11. Per-manipulation analysis (test)\n",
          "Fake folders: recall at the validation threshold and AUC against the real test videos of the same dataset (FF++ originals or "
          "BioDeepAV reals). Real folders: specificity. n = test videos per seed. CI = 95 % cluster-bootstrap CI of the seed mean.\n"]
    mrows = []
    for r in ms.itertuples():
        ci = cci[(cci.test_set == r.dataset) & (cci.config == r.config) & (cci.metric == f"recall_{r.subset}")]
        cia = cci[(cci.test_set == r.dataset) & (cci.config == r.config) & (cci.metric == f"auc_{r.subset}")]
        mrows.append(dict(config=r.config, folder=f"{r.dataset}/{r.subset}", n=int(r.n_mean),
                          recall_or_specificity=fmt(getattr(r, "recall_mean", np.nan) if r.kind == "fake" else getattr(r, "specificity_mean", np.nan),
                                                    getattr(r, "recall_sd", np.nan) if r.kind == "fake" else getattr(r, "specificity_sd", np.nan)),
                          recall_ci95=f"{ci.ci95_low.iloc[0]:.3f}..{ci.ci95_high.iloc[0]:.3f}" if len(ci) else "",
                          auc_vs_real=fmt(getattr(r, "auc_vs_real_mean", np.nan), getattr(r, "auc_vs_real_sd", np.nan)) if r.kind == "fake" else "",
                          auc_ci95=f"{cia.ci95_low.iloc[0]:.3f}..{cia.ci95_high.iloc[0]:.3f}" if len(cia) else ""))
    md = pd.DataFrame(mrows).sort_values(["folder", "config"], key=lambda s: s.map({c: i for i, c in enumerate(CONFIGS)}) if s.name == "config" else s)
    L += [t(md), "", "All FF++ manipulation folders of the split are present in the test partition (FaceShifter, DeepFakeDetection "
          "included); none had to be skipped.\n", "**Paired per-manipulation comparisons (FF++, primary)**\n",
          t(cmp_[(cmp_.test_set == "FF++") & cmp_.primary & (cmp_.metric.str.startswith("recall_") | cmp_.metric.str.startswith("auc_"))]
            [["comparison", "metric", "mean_diff", "per_seed", "ci95_low", "ci95_high", "verdict", "p_holm"]], 4), "",
          "![manip](../figures/T6_per_manipulation.png)\n![forest](../figures/T6_bootstrap_forest.png)\n",
          "## 12. Original versus degraded performance\n",
          f"**Not evaluated.** The 171026 improvement-suite protocol uses original BioDeepAV and FF++ C23 videos only; the preflight found "
          f"{pre.get('info', {}).get('degraded_videos_in_manifest', 'n/a')} degraded videos in the test manifest, so there are no paired "
          "degraded samples. As instructed, no new test degradations were created. Original-video AUC is reported in sections 8-10; "
          "degraded AUC, AUC change, prediction flip rate and per-degradation results cannot be reported for this experiment. "
          "(Section 14 uses mild photometric probes only to measure explanation stability; they are not a robustness test set.)\n",
          "## 13. Calibration analysis\n",
          "171026 E5 procedure: calibrators and thresholds fitted on validation, applied unchanged to test; threshold = validation "
          "mean-balanced-accuracy rule; 100 % coverage. 'raw' = sigmoid(logit); 'temperature' = the rule used by the stored predictions.\n",
          t(cal_s, 3), "", "![rel](../figures/T6_reliability.png)\n"]
    L += ["## 14. Explainability analysis (Grad-CAM++)\n"]
    if len(xc):
        L += [f"Videos: the 171026 E6 sample ({int(xc.n_videos.max())} test videos, balanced over dataset x label), every run "
              f"({int(xc.n_seeds.max())} seeds per configuration); per video the seed values are averaged, then identity groups are bootstrapped.\n",
              "- **Faithfulness** (does the map point at pixels the model uses?): deletion / insertion curves of 171026 E6 with grey and blur "
              "fill; gain = Grad-CAM++ minus a random map on the same video (deletion: random AUC - Grad-CAM++ AUC; insertion: Grad-CAM++ - random). "
              "> 0 = better than random.\n"
              "- **Stability** (does the map stay put under small input changes?): Grad-CAM++ for the same target class after each photometric "
              f"probe ({', '.join(PROBES)}; the 151026 augmentation operators, applied alone) vs the clean map: top-20 % IoU and Pearson r, "
              "averaged over frames and probes. Stability is not faithfulness, and neither is correctness: no manipulation-region ground "
              "truth exists, so no claim is made that the maps localise the manipulated region.\n",
              t(xc, 3), "", "**Paired differences (95 % cluster-bootstrap CI; margin = H4 non-inferiority margin)**\n", t(xcmp, 4), "",
              "**By dataset and predicted class (seed-pooled means; classes are model-dependent, so these sets differ between configurations)**\n",
              t(xcls, 3), "", "![xai](../figures/T6_xai_comparison.png)\n"]
    else:
        L += ["_XAI phase not run yet._\n"]
    L += ["## 15. Statistical significance and confidence intervals\n",
          f"Primary family: {int(cmp_.primary.sum()) if len(cmp_) else 0} rows; Holm-significant at 0.05: "
          f"{int(cmp_['significant_holm_0.05'].sum()) if len(cmp_) else 0}. Rows with a 'supported' verdict (all comparisons):\n",
          t(cmp_[cmp_.verdict.str.startswith("supported")][["test_set", "comparison", "metric", "n_seeds", "mean_diff", "per_seed", "ci95_low",
                                                             "ci95_high", "verdict", "p_boot", "p_holm"]], 4),
          f"\n_{int((cmp_.verdict == 'inconclusive').sum()) if len(cmp_) else 0} further comparisons inconclusive - full table: "
          "metrics/T6_bootstrap_comparisons.csv._\n",
          "Seed-mean ± SD describes training-run variability; the cluster-bootstrap CI describes test-set sampling variability for these "
          "trained models. With three seeds, the sign-agreement requirement is a coarse check of training variability, not a test.\n",
          "## 16. Limitations and threats to validity\n",
          "- Three seeds; GPU training with cuDNN autotuning is not bitwise reproducible (171026 E1: a seed-42 re-run differed by up to "
          "0.013 test AUC), so part of any small difference is run-to-run noise.\n"
          "- harddro20 differs from harddro12 in epochs AND patience AND LR-schedule length (one 'longer training' factor, as in base20 vs base12); "
          "the effect of each cannot be separated.\n"
          "- Group-DRO and the longer schedule both change how much each cell is trained; the 2x2 design (base12, harddro12, base20, harddro20) "
          "allows an interaction reading, but with three seeds the interaction itself is not tested formally.\n"
          "- The existing runs were trained on 2026-10-17 and harddro20 later on the same machine and code; no environment change was detected "
          "(script hashes in summary.json), but driver / library state is not fully controlled.\n"
          "- Test-set bootstrap treats the 1,602 test videos' identity groups as the sampling unit; it does not cover other datasets, "
          "compressions or real-world degradations (section 12).\n"
          "- Per-manipulation results use 164 fakes per FF++ method (63 for DeepFakeDetection): differences of a few points are within noise.\n"
          "- Grad-CAM++ evaluation uses 160 videos; per-class sets depend on each model's predictions. Faithfulness metrics depend on the "
          "perturbation (grey vs blur) - both are reported. No localisation ground truth.\n"
          "- Thresholds are selected on validation; calibrated metrics on test are honest, but the validation set is also used for checkpoint "
          "selection (as in all 171026 runs).\n",
          "## 17. Conclusions per hypothesis\n"]
    for k, h in H.items():
        L += [f"- **{k} - {h['verdict'].upper()}**: {h['statement']}. Evidence: {h['evidence']}\n"]
    L += ["\n## 18. Recommended next experiment\n", recommend(H, cmp_), "",
          "## 19. Suggested thesis-ready results paragraph\n", thesis_paragraph(fs, cmp_, H, ri), "",
          "## 20. Reproduction commands and output locations\n",
          "```\npython 201026.py --phase selftest\npython 201026.py --phase preflight\npython 201026.py --phase train      # harddro20 x 3 seeds\n"
          "python 201026.py --phase xai        # Grad-CAM++ faithfulness + stability, 12 runs\npython 201026.py --phase evaluate   # metrics, statistics, figures, this report\n```\n",
          f"- New runs (best.pt = selected EMA checkpoint, last.pt = final full training state, history.csv, sampler_shares.csv, "
          f"dro_cell_losses.csv, predictions.csv, run_meta.json): `{P.runs}`\n"
          f"- Metrics CSV: `{P.metrics}` (T6_*.csv, E5_*.csv) | figures: `{P.figures}` | XAI per run: `{P.xai}`\n"
          f"- Machine-readable summary: `{P.reports / 'summary.json'}`; audit: `preflight_audit.json`, `data_manifest.json`; environment: "
          "`environment.json`, `pip_freeze.txt`; logs: `logs/`\n"
          f"- Script sha256: " + ", ".join(f"{k} `{v[:16]}…`" for k, v in provenance(a)["scripts"].items()) + "\n"]
    (P.reports / "REPORT_201026.md").write_text("\n".join(L), encoding="utf-8")
    print(f"  report -> {P.reports / 'REPORT_201026.md'}")


def _row(cmp_, ts, met, comp):
    r = cmp_[(cmp_.test_set == ts) & (cmp_.metric == met) & (cmp_.comparison == comp)]
    return r.iloc[0] if len(r) else None


def recommend(H, cmp_) -> str:
    h1, h2, h3 = H["H1"]["verdict"], H["H2"]["verdict"], H["H3"]["verdict"]
    nt = [_row(cmp_, "FF++", "recall_NeuralTextures", f"{NEW} - {c}") for c in ("harddro12", "base20")]
    out = []
    if h1 == "supported" and h3 == "supported":
        out.append("harddro20 is the new reference configuration. Next: test whether the gain holds out of distribution (train on the "
                   "suite split, evaluate on a held-out manipulation family or dataset) before adding any new component.")
    elif h3 in ("contradicted",) or "deterioration" in H["H3"]["evidence"]:
        out.append("The 20-epoch DRO schedule costs BioDeepAV performance. Next: keep 20 epochs but select the checkpoint per dataset "
                   "or cap the DRO share of FF++ cells (one change only), and re-test H3.")
    else:
        out.append("The combination does not give a statistically supported gain over its best single component. Next: a 30-epoch "
                   "harddro run (harddro30) to complete the 2x3 sampler x duration grid, or more seeds (5) for the existing "
                   "harddro12 / base20 / harddro20 cells to reduce training variance before concluding - choose one.")
    if any(r is not None and r.verdict != "supported improvement" for r in nt):
        out.append("NeuralTextures remains the weakest type; a targeted, single-factor follow-up (e.g. a texture-sensitive input "
                   "branch or higher face resolution) should be evaluated with the same split, seeds and statistics.")
    return " ".join(out)


def thesis_paragraph(fs, cmp_, H, ri) -> str:
    def v(c, ts, k="auc"):
        r = fs[(fs.config == c) & (fs.test_set == ts)]
        return f"{r[f'{k}_mean'].iloc[0]:.3f} ± {r[f'{k}_sd'].iloc[0]:.3f}" if len(r) and np.isfinite(r[f"{k}_sd"].iloc[0]) else "n/a"

    def d(ts, met, lo):
        r = _row(cmp_, ts, met, f"{NEW} - {lo}")
        return f"{r.mean_diff:+.3f} (95 % CI {r.ci95_low:+.3f} to {r.ci95_high:+.3f}; {r.verdict})" if r is not None else "n/a"
    n = ri[ri.config == NEW]
    ep = ", ".join(f"{int(x)}" for x in n.epochs_completed) if len(n) else "n/a"
    return (f"Combining online group-DRO sampling with a 20-epoch schedule (harddro20; three seeds, {ep} epochs completed, "
            f"best epochs {', '.join(map(str, n.best_epoch)) if len(n) else 'n/a'}) yielded a pooled test ROC-AUC of {v(NEW, 'ALL')} "
            f"compared with {v('harddro12', 'ALL')} for harddro12 and {v('base20', 'ALL')} for base20 (mean ± SD over seeds). "
            f"Paired cluster-bootstrap differences in pooled AUC were {d('ALL', 'auc', 'harddro12')} against harddro12 and "
            f"{d('ALL', 'auc', 'base20')} against base20. On BioDeepAV the AUC was {v(NEW, 'BioDeepAV')} (difference "
            f"{d('BioDeepAV', 'auc', 'harddro12')} vs harddro12, {d('BioDeepAV', 'auc', 'base20')} vs base20), and on FaceForensics++ "
            f"C23 it was {v(NEW, 'FF++')} ({d('FF++', 'auc', 'harddro12')} vs harddro12, {d('FF++', 'auc', 'base20')} vs base20). "
            f"NeuralTextures recall changed by {d('FF++', 'recall_NeuralTextures', 'harddro12')} relative to harddro12 and "
            f"{d('FF++', 'recall_NeuralTextures', 'base20')} relative to base20. "
            f"Under the pre-specified decision rules, H1 was {H['H1']['verdict']}, H2 {H['H2']['verdict']}, H3 {H['H3']['verdict']} "
            f"and H4 {H['H4']['verdict']}. An effect is called supported only when the 95 % CI excludes zero and all three seeds agree "
            "in sign; Holm-adjusted p-values over the primary comparison family are reported in the accompanying tables.")


# =====================================================================================
# SELFTEST
# =====================================================================================
def phase_selftest(a, P) -> bool:
    print("\n" + "=" * 90 + "\n SELFTEST (201026 additions)\n" + "=" * 90)
    res = {"171026 selftest": S.phase_selftest(a, P)}
    c = S.CONFIGS[NEW]
    res["harddro20 = harddro12 sampler + base20 epochs/patience"] = (
        c["sampler"] == S.CONFIGS["harddro12"]["sampler"] and c["epochs"] == S.CONFIGS["base20"]["epochs"] == 20
        and c["patience"] == S.CONFIGS["base20"]["patience"])
    res["existing configs unchanged"] = (S.CONFIGS["harddro12"] == dict(sampler="harddro", epochs=12, patience=4, exp="E3",
                                                                       label="Group-DRO hard-type sampler, 12 epochs")
                                         and S.CONFIGS["base20"]["epochs"] == 20 and S.CONFIGS["base12"]["epochs"] == 12)

    class Dm:
        pass
    Dm.fcell = np.array(["A|r"] * 20 + ["A|x"] * 10 + ["A|y"] * 10)
    w1, i1 = S.dro_weights(Dm, np.arange(40), {"A|r": 0.2, "A|x": 2.0, "A|y": 0.4})
    w2, i2 = S.dro_weights.__wrapped__(Dm, np.arange(40), {"A|r": 0.2, "A|x": 2.0, "A|y": 0.4})
    res["DRO logging wrapper does not change weights"] = np.array_equal(w1, w2) and _DRO_LOG[-1]["cell_loss"]["A|x"] == 2.0
    _DRO_LOG.clear()
    torch.manual_seed(1)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mdl = B.DeepfakeDetector(pretrained=False).to(dev).eval()
    xx = B.normalise(np.random.default_rng(3).integers(0, 255, (1, 6, 64, 64, 3), dtype=np.uint8), dev)
    eq = []
    for tg in (0, 1):
        c1, l1, a1 = B.gradcam_pp_hook(mdl, xx, None, tg)
        c2, l2, a2 = gradcam_pp_noleak(mdl, xx, None, tg)
        eq.append(np.allclose(c1, c2, atol=1e-5) and np.allclose(l1, l2) and np.allclose(a1, a2))
    res["leak-free Grad-CAM++ == 151026.gradcam_pp (both targets)"] = all(eq)
    if torch.cuda.is_available():
        m0 = torch.cuda.memory_allocated()
        for _ in range(5):
            gradcam_pp_noleak(mdl, xx, None, 1)
        res["leak-free Grad-CAM++ does not accumulate GPU memory"] = torch.cuda.memory_allocated() <= m0 + (1 << 20)
    del mdl
    rng = np.random.default_rng(0)
    m = rng.random((8, 8))
    r, iou, sh = cam_metrics(m, m)
    res["CAM metrics: identical maps -> r 1, IoU 1, shift 0"] = abs(r - 1) < 1e-9 and iou == 1 and sh < 1e-9
    clip = rng.integers(0, 255, (4, 32, 32, 3), dtype=np.uint8)
    res["probes keep shape / dtype and change pixels"] = all(
        (lambda o: o.shape == clip.shape and o.dtype == np.uint8 and not np.array_equal(o, clip))(apply_probe(clip, k, v)) for k, v in PROBES.values())
    res["Holm"] = np.allclose(holm(np.array([0.01, 0.04, 0.03, np.nan])), [0.03, 0.06, 0.06, np.nan], equal_nan=True)
    # bootstrap extension == 171026 bootstrap on synthetic paired predictions
    n = 240
    grp = np.repeat([f"G{i}" for i in range(60)], 4)
    lab = np.tile([0, 1, 1, 1], 60)
    sub = np.where(lab == 0, "original", rng.choice(["NeuralTextures", "Face2Face", "FaceShifter", "FaceSwap"], n))
    rows = []
    for cfg in ("base12", "harddro12", "base20", NEW):
        for s in SEEDS:
            p = np.clip(lab * 0.3 + rng.random(n) * 0.7, 0, 1)
            rows.append(pd.DataFrame(dict(split="test", idx=np.arange(n), dataset="FF++", subset=sub, label=lab, group=grp, p_fake=p,
                                          pred=(p >= 0.5).astype(int), config=cfg, seed=s, run=f"{cfg}_s{s}")))
    pr = pd.concat(rows, ignore_index=True)
    an = argparse.Namespace(n_boot=60)
    S.COMPARISONS = PRIMARY + SECONDARY
    ref = S.bootstrap_compare(an, pr)
    mine = compare(boot_replicates(an, pr), PRIMARY + SECONDARY)
    j = ref.merge(mine, on=["test_set", "comparison", "metric"], suffixes=("_r", ""))
    res["bootstrap extension == 171026.bootstrap_compare"] = len(j) == len(ref) > 0 and np.allclose(j.mean_diff_r, j.mean_diff) and \
        np.allclose(j.ci95_low_r, j.ci95_low) and (j.verdict_r == j.verdict).all()
    for k, v in res.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    ok = all(res.values())
    print(f"  selftest {'PASSED' if ok else 'FAILED'}")
    return ok


# =====================================================================================
# MAIN
# =====================================================================================
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="201026: Test 6 - harddro20 (group-DRO + 20 epochs) vs base12 / harddro12 / base20")
    p.add_argument("--phase", default="all", choices=["all", "selftest", "preflight", "train", "xai", "evaluate"])
    p.add_argument("--root", default=None)
    p.add_argument("--base-root", default=None)
    p.add_argument("--suite-root", default=None)
    p.add_argument("--base-script", default=str(BASE_SCRIPT))
    p.add_argument("--suite-script", default=str(SUITE_SCRIPT))
    p.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    p.add_argument("--img", type=int, default=192)
    p.add_argument("--face-px", type=int, default=112)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--accum", type=int, default=2)
    p.add_argument("--loader-threads", type=int, default=4)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--max-epochs", type=int, default=None)
    p.add_argument("--xai-n", type=int, default=XAI_N)
    p.add_argument("--n-boot", type=int, default=N_BOOT)
    p.add_argument("--skip-rederive", action="store_true", help="preflight: skip re-deriving the split from scratch (~3 min)")
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--smoke", action="store_true", help="pipeline check: 3 steps x 2 epochs, seed 42, 8 XAI videos, separate root")
    a = p.parse_args(argv)
    if a.smoke:
        a.max_steps, a.max_epochs, a.xai_n, a.n_boot = a.max_steps or 3, a.max_epochs or 2, min(a.xai_n, 8), min(a.n_boot, 100)
        a.seeds = "42" if a.seeds == ",".join(map(str, SEEDS)) else a.seeds
    a.root = Path(a.root) if a.root else (SMOKE_ROOT if a.smoke else DEFAULT_ROOT)
    a.base_root = Path(a.base_root) if a.base_root else BASE_ROOT
    a.suite_root = Path(a.suite_root) if a.suite_root else SUITE_ROOT
    a.base_script, a.suite_script = Path(a.base_script), Path(a.suite_script)
    a.seed_list = [int(s) for s in a.seeds.split(",") if s.strip()]
    if a.root.resolve() in (a.suite_root.resolve(), a.base_root.resolve()):
        raise SystemExit("[STOP] --root must not be an existing experiment folder")
    return a


def main(argv=None):
    global B, S
    a = parse_args(argv)
    B = load_module(a.base_script, "base151026")
    S = load_module(a.suite_script, "suite171026")
    install(a)
    P = S.Paths(a.root)
    P.make()
    B.lower_priority()
    sys.stdout = B.Tee(P.logs / f"{a.phase}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    torch.set_num_threads(max(1, min(4, (os.cpu_count() or 4) // 2)))
    print(f"201026 | phase {a.phase} | seeds {a.seed_list} | root {P.root} | suite {a.suite_root} | base {a.base_root} | {B.now()} | "
          f"GPU {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}")
    if not phase_selftest(a, P):
        raise SystemExit("[STOP] selftest failed")
    if a.phase == "selftest":
        return
    environment(P)
    B.write_json(P.reports / f"{NEW}_config.json", dict(config=NEW, **NEW_CONFIG, training_constants=training_constants(a),
                                                       comparisons=dict(primary=PRIMARY, secondary=SECONDARY), seeds=a.seed_list,
                                                       margins=dict(bio_auc=BIO_MARGIN, xai=XAI_MARGIN), probes=PROBES, provenance=provenance(a)))
    BP, inv, C, D = S.load_base_data(a)
    pre = preflight(a, P, BP, inv, C, D) if a.phase in ("all", "preflight", "train") or not (P.reports / "preflight_audit.json").exists() \
        else B.read_json(P.reports / "preflight_audit.json")
    if not pre.get("passed"):
        raise SystemExit("[STOP] preflight did not pass")
    if a.phase in ("all", "train"):
        train(a, P, D)
    if a.phase in ("all", "xai"):
        xai_all(a, P, D)
    if a.phase in ("all", "evaluate"):
        if (P.reports / "preflight_audit.json").exists():
            pre = B.read_json(P.reports / "preflight_audit.json")
        evaluate(a, P, D, pre)
    print(f"\n  DONE ({a.phase}). Report: {P.reports / 'REPORT_201026.md'}")


if __name__ == "__main__":
    main()
