"""
============================================================================
IID experiment core — shared pipeline for single-class and multiclass runs.
============================================================================

What this does (in one call to run_iid(cfg, mode)):

  NO PCA / NO AE: every detector consumes the RAW full bands. Our score nets
  (DSM, LRao) carry a FROZEN ZCA whitening first layer (fit on the training
  background, relative eigen-floor) so they take raw input, whiten internally,
  and return DATA-SPACE scores (detection uses the raw signature).

  Data (raw 103-D)
  ----------------
    1. Load raw (mode='none' — original sensor values, no scaling).
    2. Extract background / target pools (single = one bkg class multi =
       union of all non-target classes minus exclude_classes).
    3. Target signature  s_raw = mean(tgt_pix)   (NOT unit-normalized).
    4. Shuffle bkg with one seed -> train pool of size max(n_train_list)
       + held-out test pool of size test_size.
    5. Plant targets in raw 103-D (additive + replacement) ONCE; the test
       set is shared across every n_train.

  Sweep (for n in n_train_list)
  -----------------------------
    6. Train 2 score models on RAW bands: DSM, LRao (each with a frozen ZCA
       whitening first layer). Per-epoch loss recorded; weights saved. Score
       the test set with each (DSM has separate additive / replacement
       statistics; the LRao Mode-2 statistic is the same for both).

  Classical baselines (raw 103-D, depend only on n)
  -------------------------------------------------
    7. For each n: run AMF, Reg-AMF, CEM, GMM-GLRT, (DLTD, SMGLRT in multi),
       AMF-rep, GMM-GLRT-rep, Exact-GLRT.

  Save
  ----
    9. config.yaml, metrics.json (hierarchical AUCs), loss_curves.json
       (per-epoch losses, flat keys), scores.npz (per-pixel scores + labels,
       enough to re-render any figure offline), models/*, figures/*.

Use this module from run_iid_single.py and run_iid_multi.py.
============================================================================
"""

import os
import sys
import json
import time
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import torch
import yaml
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from tqdm import tqdm

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from final_paper_experiments.data_utils import (
    load_and_normalize, compute_sigma_from_data, plant_targets,
)
from final_paper_experiments.baselines.detectors import (
    amf, dsm_additive, gmm_glrt,            # ADDITIVE-only experiment
    _fit_gmm_shared_cov, _dltd_score, _smglrt_score,  # multi-class GLRT (fit once, score twice)
)
from final_paper_experiments.baselines.gmm_glrt_levin import gmm_glrt_levin_additive
from dsm_model import (
    ScoreNet, TwoBranchScore, Whitening, dsm_loss,
    lfi_loss_mode2, compute_lfi_detector_scores_mode2,
)


def _make_whitening(train_raw, cfg):
    """Frozen ZCA whitener fit on the RAW training pool.

    Default eig_floor=0 → spectral-gap adaptive floor (recommended).
    Set whiten_eig_floor > 0 in config to override with a fixed relative floor.
    """
    return Whitening.from_data(np.asarray(train_raw, dtype=np.float32),
                               mode=cfg.get('whiten_mode', 'zca'),
                               eig_floor=float(cfg.get('whiten_eig_floor', 0.0)))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _auc(labels: np.ndarray, scores: np.ndarray) -> float:
    try:
        return float(roc_auc_score(labels, scores))
    except Exception:
        return float('nan')


def _roc(labels: np.ndarray, scores: np.ndarray):
    try:
        fpr, tpr, _ = roc_curve(labels, scores)
        return fpr, tpr, _auc(labels, scores)
    except Exception:
        n = len(labels)
        return np.linspace(0, 1, n), np.linspace(0, 1, n), float('nan')


def _pauc(labels: np.ndarray, scores: np.ndarray, fpr_max: float = 0.1) -> float:
    """Partial AUC over FPR ∈ [0, fpr_max], normalized to [0,1] (so 0.5≈chance)."""
    try:
        fpr, tpr, _ = roc_curve(labels, scores)
        tpr_at = float(np.interp(fpr_max, fpr, tpr))
        keep = fpr < fpr_max
        fpr_c = np.concatenate([fpr[keep], [fpr_max]])
        tpr_c = np.concatenate([tpr[keep], [tpr_at]])
        return float(np.trapz(tpr_c, fpr_c) / fpr_max)
    except Exception:
        return float('nan')


def _pd_at_fa(labels: np.ndarray, scores: np.ndarray, pfa: float = 0.1) -> float:
    """Detection probability (TPR) at a fixed false-alarm rate Pfa."""
    try:
        fpr, tpr, _ = roc_curve(labels, scores)
        return float(np.interp(pfa, fpr, tpr))
    except Exception:
        return float('nan')


def _safe(name: str, fn, n_out: int):
    try:
        return fn()
    except Exception as exc:
        print(f"      [warn] {name}: {exc}", flush=True)
        return np.full(n_out, np.nan)


def _ensure_list(v):
    if isinstance(v, (list, tuple)):
        return list(v)
    return [v]


# ---------------------------------------------------------------------------
# Local training loops (record per-epoch loss tqdm with ratio diagnostic)
# ---------------------------------------------------------------------------

def _make_loader(X: np.ndarray, batch_size: int):
    Xt = torch.tensor(X, dtype=torch.float32)
    return Xt


def train_dsm_local(train_raw: np.ndarray, cfg: dict,
                    seed: int, label: str) -> Tuple[ScoreNet, List[float]]:
    """DSM on RAW bands with a frozen ZCA whitening first layer (no PCA/AE).

    The net whitens internally, DSM noise is isotropic in whitened space
    (σ=√ρ), and forward returns the DATA-SPACE score (detection uses raw sig).
    Trains on GPU if cfg['device'] is set (or CUDA is available and not disabled).
    """
    torch.manual_seed(seed)
    device = torch.device(cfg.get('device', 'cpu'))
    D = train_raw.shape[1]
    W = _make_whitening(train_raw, cfg)
    sigma = float(np.sqrt(cfg['dsm_sigma_rho']))     # whitened cov≈I ⇒ σ²=ρ·1
    model = ScoreNet(D, list(cfg['hidden_dims']), cfg['activation'], whitening=W).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg['lr'],
                           weight_decay=cfg['weight_decay'])
    X = torch.tensor(np.asarray(train_raw, dtype=np.float32)).to(device)
    N, bs = len(X), min(cfg['batch_size'], len(X))
    baseline = D / (sigma ** 2)
    hist = []
    pbar = tqdm(range(1, cfg['dsm_epochs'] + 1), desc=f'DSM {label}',
                dynamic_ncols=True, leave=False)
    for _ in pbar:
        perm = torch.randperm(N)
        tot = 0.0
        nb = 0
        for i in range(0, N, bs):
            b = X[perm[i:i + bs]]
            loss = dsm_loss(model, b, sigma)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
            nb += 1
        hist.append(tot / max(nb, 1))
        pbar.set_postfix(loss=f"{hist[-1]:.2f}",
                         ratio=f"{hist[-1] / baseline:.3f}")
    model.cpu().eval()   # move back to CPU so scoring helpers get numpy-compatible model
    return model, hist


def train_two_branch_local(train_raw: np.ndarray, cfg: dict,
                           seed: int, label: str) -> Tuple[TwoBranchScore, List[float]]:
    """Two-branch DSM on RAW bands with a frozen ZCA whitening front-end.

    Architecture (in whitened space): psi(y) = u(||Ay_w||^2) * (-A^T A y_w),
    with output mapped back to data space for detection.  Warmup: freeze the
    u-MLP for the first `two_branch_warmup_frac` fraction of epochs so the
    W-branch (linear covariance estimate) stabilises before scalar adaptation.
    Sigma schedule: same as DSM — sigma = sqrt(rho) in whitened space.
    """
    torch.manual_seed(seed)
    device  = torch.device(cfg.get('device', 'cpu'))
    D       = train_raw.shape[1]
    W       = _make_whitening(train_raw, cfg)
    sigma   = float(np.sqrt(cfg['dsm_sigma_rho']))
    u_dims  = list(cfg.get('two_branch_u_hidden', [32, 32]))
    model   = TwoBranchScore(D, u_hidden_dims=u_dims, whitening=W).to(device)
    opt     = torch.optim.Adam(model.parameters(), lr=cfg['lr'],
                                weight_decay=cfg['weight_decay'])
    X       = torch.tensor(np.asarray(train_raw, dtype=np.float32)).to(device)
    N, bs   = len(X), min(cfg['batch_size'], len(X))
    epochs  = int(cfg['dsm_epochs'])
    warmup  = int(cfg.get('two_branch_warmup_frac', 0.1) * epochs)
    baseline = D / (sigma ** 2)
    hist    = []

    pbar = tqdm(range(1, epochs + 1), desc=f'TwoBranch {label}',
                dynamic_ncols=True, leave=False)
    for ep in pbar:
        # freeze u_mlp during warmup so W-branch learns covariance first
        for p in model.u_mlp.parameters():
            p.requires_grad = (ep > warmup)

        perm = torch.randperm(N); tot = 0.0; nb = 0
        for i in range(0, N, bs):
            b    = X[perm[i:i + bs]]
            loss = dsm_loss(model, b, sigma)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        hist.append(tot / max(nb, 1))
        pbar.set_postfix(loss=f"{hist[-1]:.2f}",
                         ratio=f"{hist[-1] / baseline:.3f}")
    model.cpu().eval()
    return model, hist


def train_lrao_local(train_raw: np.ndarray, cfg: dict,
                     seed: int, label: str) -> Tuple[ScoreNet, List[float]]:
    """LRao Mode-2 on RAW bands with a frozen ZCA whitening first layer.

    The tr(J*) objective is invariant to the whitening reparametrization, so the
    net learns the data-space score with whitened-space conditioning.
    NaN-guarded: if the in-graph SVD blows up, abort and return what we have.
    """
    torch.manual_seed(seed)
    device = torch.device(cfg.get('device', 'cpu'))
    D = train_raw.shape[1]
    # LRao can use its own eigenvalue floor (lrao_whiten_eig_floor) — a larger
    # floor cuts off more near-zero directions, keeping C_Psi well-conditioned.
    # Falls back to the shared whiten_eig_floor if not set.
    lrao_cfg = cfg if 'lrao_whiten_eig_floor' not in cfg else {
        **cfg, 'whiten_eig_floor': cfg['lrao_whiten_eig_floor']}
    W = _make_whitening(train_raw, lrao_cfg)
    model = ScoreNet(D, list(cfg['hidden_dims']), cfg['activation'], whitening=W).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=cfg['lr'],
                              weight_decay=cfg['weight_decay'])
    X     = torch.tensor(np.asarray(train_raw, dtype=np.float32)).to(device)
    N, bs = len(X), min(cfg['batch_size'], len(train_raw))
    hist  = []

    pbar = tqdm(range(1, cfg['lrao_epochs'] + 1), desc=f'LRao {label}',
                dynamic_ncols=True, leave=False)
    for ep in pbar:
        model.train()
        perm = torch.randperm(N); tot = 0.0; nb = 0
        try:
            for i in range(0, N, bs):
                b    = X[perm[i:i + bs]]
                loss = lfi_loss_mode2(model, b, cfg['lfi_delta_theta'],
                                      cfg['lfi_sigma_cutoff'],
                                      detach_sigma=cfg['lfi_detach_sigma'])
                if not torch.isfinite(loss):
                    raise FloatingPointError("non-finite LRao loss")
                opt.zero_grad(); loss.backward(); opt.step()
                tot += loss.item(); nb += 1
        except Exception as exc:
            print(f"      [warn] LRao {label} aborted at epoch {ep}: {exc}",
                  flush=True)
            break
        hist.append(-tot / max(nb, 1))   # tr(J*)
        pbar.set_postfix(trJ=f"{hist[-1]:.2f}")

    model.cpu().eval()   # back to CPU so scoring helpers receive numpy-compatible model
    return model, hist


# ---------------------------------------------------------------------------
# Data prep
# ---------------------------------------------------------------------------

def build_pools(data: np.ndarray, gt_flat: np.ndarray, cfg: dict, mode: str):
    """Return (bkg_pixels, tgt_pixels) in raw 103-D."""
    H_W_D = data.shape
    flat = data.reshape(-1, H_W_D[-1])
    if mode == 'single':
        bkg = flat[gt_flat == cfg['bkg_cls']]
    else:
        excl = list(cfg.get('exclude_classes', []))
        bkg_classes = sorted(int(c) for c in np.unique(gt_flat)
                             if c != cfg['target_cls'] and c not in excl)
        bkg = np.vstack([flat[gt_flat == c] for c in bkg_classes])
    tgt = flat[gt_flat == cfg['target_cls']]
    return bkg, tgt


# ---------------------------------------------------------------------------
# DSM / LRao scoring helpers in latent
# ---------------------------------------------------------------------------

def score_dsm_add(model, train_lat, test_lat, s_lat):
    return dsm_additive(test_lat, train_lat, model, s_lat)


def score_lrao(model, train_lat, test_lat, s_lat, cfg):
    return compute_lfi_detector_scores_mode2(
        model, train_lat, test_lat, s_lat,
        delta_theta=cfg['lfi_delta_theta'],
        sigma_cutoff=cfg['lfi_sigma_cutoff'])


# ---------------------------------------------------------------------------
# Classical baselines (ADDITIVE only)
# ---------------------------------------------------------------------------

# single-class: AMF only  (single Gaussian bkg; Reg-AMF/GMM redundant)
# multi-class:  AMF + GMM-Levin + DLTD + SMGLRT
#   GMM-Levin: Levin 2019 product-of-GMMs GLRT — handles mixed background
#   DLTD/SMGLRT: shared-cov K-component GMM GLRT variants (Ma 2025/2026)
#                K clamped to ≥ 3 (K=1 is degenerate constant score)
CLASSICAL_DETS_SINGLE = ['AMF']
CLASSICAL_DETS_MULTI  = ['AMF', 'GMM-Levin']


def run_classical_additive(train_raw, test_planted, s_raw, reg_sigma, cfg, mode):
    """Returns {detector -> scores} for the ADDITIVE target model on raw bands."""
    dets = CLASSICAL_DETS_MULTI if mode == 'multi' else CLASSICAL_DETS_SINGLE
    n    = len(test_planted)
    K    = max(int(cfg.get('gmm_K', 6)), 3)   # DLTD/SMGLRT require K ≥ 3

    # Pre-fit the shared-cov GMM ONCE and share between DLTD + SMGLRT.
    # Both detectors need the same model; fitting twice wastes ~2× time.
    _gmm = None
    if 'DLTD' in dets or 'SMGLRT' in dets:
        _gmm = _fit_gmm_shared_cov(train_raw, K)

    all_jobs = {
        'AMF':       lambda: amf(test_planted, train_raw, s_raw),
        'GMM-Levin': lambda: gmm_glrt_levin_additive(
                         test_planted, train_raw, s_raw,
                         p_steps=50, p_max=1.0),
        'DLTD':      lambda: _dltd_score(test_planted, *_gmm, s_raw),
        'SMGLRT':    lambda: _smglrt_score(test_planted, *_gmm, s_raw),
    }
    return {nm: _safe(nm, all_jobs[nm], n) for nm in dets}


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

DETECTOR_COLORS = {
    'AMF':        '#1f77b4',
    'GMM-Levin':  '#9467bd',
    'DLTD':       '#e6550d',   # orange
    'SMGLRT':     '#8c564b',   # brown
    'DSM':        '#d62728',   # red   — nonlinear DSM
    'LDSM':       '#9b2226',   # dark red — linear DSM
    'DSM-lin':    '#9b2226',   # dark red  — legacy alias
    'DSM-MLP':    '#e07070',   # light red — legacy alias
    'LRao':       '#2ca02c',
    'TwoBranch':  '#8B0000',   # dark red — two-branch DSM
}


def _det_color(det: str):
    return DETECTOR_COLORS.get(det, '#444444')


def _savefig(fig, pdf_path: str, dpi: int = 150):
    """Save as PDF (paper) + PNG (inline display) side by side."""
    fig.savefig(pdf_path, bbox_inches='tight')
    fig.savefig(pdf_path.replace('.pdf', '.png'), dpi=dpi, bbox_inches='tight')


def _apply_log_xticks(ax, x):
    """Force log-axis ticks at exactly the data points with clean labels."""
    ax.set_xscale('log')
    ax.set_xticks(x)
    ax.xaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _: f'{v:g}'))
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    # ha='right' + rotation_mode='anchor' prevents label overlap on dense log grids
    for lbl in ax.get_xticklabels():
        lbl.set_rotation(45)
        lbl.set_ha('right')
        lbl.set_rotation_mode('anchor')
        lbl.set_fontsize(8)


def _plot_vs(xvals, series: dict, xlabel: str, ylabel: str, title: str,
             out_pdf: str, logx: bool = False, series_std: dict = None):
    """Generic 'metric vs x' line plot.

    series      : {det -> list of mean values}
    series_std  : optional {det -> list of std values} — draws ±1σ shaded band
    """
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    x = np.asarray(xvals, dtype=float)
    for det, ys in series.items():
        if ys is None or all(v != v for v in ys):     # all-NaN
            continue
        ys = np.asarray(ys, dtype=float)
        style = dict(marker='D', lw=2.2) if det in ('DSM', 'LDSM', 'DSM-lin', 'DSM-MLP', 'LRao') \
            else dict(marker='o', lw=1.4)
        c = _det_color(det)
        ax.plot(x, ys, color=c, label=det, **style)
        if series_std and det in series_std:
            sd = np.asarray(series_std[det], dtype=float)
            ax.fill_between(x, ys - sd, ys + sd, alpha=0.15, color=c)
    if logx:
        _apply_log_xticks(ax, x)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3, which='both')
    ax.legend(fontsize=8, loc='upper left', bbox_to_anchor=(1.02, 1.0),
              borderaxespad=0.)
    fig.tight_layout()
    _savefig(fig, out_pdf)
    plt.close(fig)


def _plot_roc(det_scores: dict, labels: np.ndarray, title: str, out_pdf: str):
    """Multi-detector ROC curve plot — all detectors on one axes."""
    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    ax.plot([0, 1], [0, 1], 'k--', lw=0.7, label='_no_legend_')
    for det, sc in det_scores.items():
        fpr, tpr, auc_v = _roc(labels, sc)
        lw  = 2.2 if det in ('DSM', 'LDSM', 'DSM-lin', 'DSM-MLP', 'LRao') else 1.4
        ax.plot(fpr, tpr, color=_det_color(det), lw=lw,
                label=f'{det}  (AUC={auc_v:.3f})')
    ax.set_xlabel('False Alarm Rate')
    ax.set_ylabel('Detection Rate')
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7, loc='lower right')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    _savefig(fig, out_pdf)
    plt.close(fig)
    print(f"  [fig] {os.path.basename(out_pdf)}", flush=True)


def _roc_to_dict(labels: np.ndarray, sc: np.ndarray) -> dict:
    fpr, tpr, auc_v = _roc(labels, sc)
    return {'fpr': fpr.tolist(), 'tpr': tpr.tolist(), 'auc': float(auc_v)}


def plot_loss_panel(loss_curves: dict, tag: str, out_png: str):
    keys = [
        (f'DSM_{tag}',  f'DSM  ({tag})', 'loss'),
        (f'LRao_{tag}', f'LRao ({tag})', 'tr(J*)'),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    for ax, (key, ttl, ylab) in zip(axes, keys):
        hist = loss_curves.get(key, [])
        ax.plot(hist) if hist else ax.text(0.5, 0.5, 'no data', ha='center')
        ax.set_title(ttl, fontsize=9)
        ax.set_xlabel('epoch'); ax.set_ylabel(ylab); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def run_iid(cfg: dict, mode: str):
    assert mode in ('single', 'multi')
    # Auto-dispatch: if seed is a list with >1 entry, run multi-seed aggregation
    _s = cfg.get('seed', 42)
    if isinstance(_s, (list, tuple)) and len(_s) > 1:
        return run_iid_multi_seed(cfg, mode)
    t_start = time.time()
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(cfg['results_dir'], f'iid_{mode}_{ts}')
    mdl_dir = os.path.join(run_dir, 'models')
    fig_dir = os.path.join(run_dir, 'figures')
    os.makedirs(mdl_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)
    yaml.dump(cfg, open(os.path.join(run_dir, 'config.yaml'), 'w'))
    print(f"Run dir: {run_dir}", flush=True)

    # ===================================================================
    # ADDITIVE-ONLY experiment. Two 1-D sweeps (same code, single vs multi
    # differ only in the data pools):
    #   vs n   (at fixed ρ): AUC, partial-AUC(Pfa<pauc_fpr), Pd@Pfa
    #   vs ρ   (at fixed n): AUC, Pd@Pfa
    # Detectors: classical {AMF, Reg-AMF, GMM-GLRT} + score {DSM, LRao}.
    # ===================================================================
    n_list   = sorted(set(_ensure_list(cfg['n_train_list'])))
    rho_list = sorted(set(_ensure_list(cfg.get(
        'rho_list', [0.001, 0.003, 0.01, 0.03, 0.1, 0.3]))))
    rho_fixed = float(cfg['dsm_sigma_rho'])          # ρ used for the vs-n sweep
    n_fixed   = int(cfg.get('n_fixed_for_rho', max(n_list)))   # n for the vs-ρ sweep
    pfa       = float(cfg.get('pfa', 0.1))           # operating Pfa for Pd@Pfa
    pauc_fpr  = float(cfg.get('pauc_fpr', 0.1))      # partial-AUC upper FPR

    # GPU / device selection — set cfg['device'] = 'cuda' in the notebook for GPU
    _dev_str = cfg.get('device', None)
    if _dev_str is None:
        _dev_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg = {**cfg, 'device': _dev_str}   # propagate to train helpers
    print(f"device = {_dev_str}", flush=True)
    print(f"n_train_list = {n_list}", flush=True)
    print(f"rho_list     = {rho_list}  (vs-ρ at n={n_fixed})", flush=True)
    print(f"Pfa={pfa}  pAUC FPR<{pauc_fpr}  (ADDITIVE only)", flush=True)

    # ----- load data: RAW bands everywhere (no PCA/AE). Score nets whiten
    #       internally; classical baselines consume the same raw bands. -----
    _s = cfg['seed']
    seed = int(_s[0] if isinstance(_s, (list, tuple)) else _s)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    data, gt = load_and_normalize(cfg['dataset'], mode=cfg.get('norm_mode', 'none'))
    H, W, D_RAW = data.shape
    gt_flat = gt.flatten()
    print(f"Image {H}x{W}x{D_RAW}  (RAW band space, no PCA)", flush=True)
    for cls_id in sorted(np.unique(gt_flat).tolist()):
        print(f"  class {int(cls_id):>2}: {int((gt_flat == cls_id).sum()):>6} px", flush=True)

    bkg_raw, tgt_raw = build_pools(data, gt_flat, cfg, mode)
    s_raw = tgt_raw.mean(axis=0).astype(np.float32)
    if cfg.get('normalize_signature', False):
        s_raw = (s_raw / (np.linalg.norm(s_raw) + 1e-12)).astype(np.float32)
    print(f"bkg pool: {len(bkg_raw):>6} | tgt pool: {len(tgt_raw):>6} | "
          f"||s_raw|| = {np.linalg.norm(s_raw):.4f}", flush=True)

    idx = np.arange(len(bkg_raw)); rng.shuffle(idx)
    bkg_shuf  = bkg_raw[idx]
    max_n     = max(max(n_list), n_fixed)
    test_size = int(cfg['test_size'])
    assert len(bkg_shuf) >= max_n + test_size, \
        f"need {max_n + test_size} bkg pixels, have {len(bkg_shuf)}"
    train_pool_raw = bkg_shuf[:max_n].astype(np.float32)
    test_bkg_raw   = bkg_shuf[-test_size:].astype(np.float32)

    # ----- plant ADDITIVE targets -----
    test_planted, labels, _ = plant_targets(
        test_bkg_raw, s_raw, cfg['amplitude'], cfg['target_fraction'],
        model='additive', seed=seed)
    test_planted = test_planted.astype(np.float32)
    print(f"planted {int(labels.sum())} targets in test (amp={cfg['amplitude']})\n",
          flush=True)

    _cls = CLASSICAL_DETS_MULTI if mode == 'multi' else CLASSICAL_DETS_SINGLE

    # DSM variant names: primary label from 'dsm_label' (default 'DSM').
    # If hidden_dims_2 is set in config, a second DSM is also run.
    _dsm1_label = cfg.get('dsm_label', 'DSM')
    _dsm_names = [_dsm1_label]
    if cfg.get('hidden_dims_2') is not None:
        _h2 = list(cfg['hidden_dims_2'])
        _dsm2_label = cfg.get('dsm2_label', 'DSM-MLP' if _h2 else 'DSM-lin')
        _dsm_names.append(_dsm2_label)

    DETS = _cls + _dsm_names + ['LRao', 'TwoBranch']
    loss_curves: Dict[str, list] = {}
    metrics = {
        'n_list': n_list, 'rho_list': rho_list, 'n_fixed': n_fixed,
        'rho_fixed': rho_fixed, 'pfa': pfa, 'pauc_fpr': pauc_fpr, 'mode': mode,
        'vs_n':   {d: {'auc': [], 'pauc': [], 'pd': []} for d in DETS},
        'vs_rho': {d: {'auc': [], 'pd': []} for d in DETS},
    }
    store = {'labels': labels.astype(np.int8)}

    def _mtr(sc):
        return (_auc(labels, sc), _pauc(labels, sc, pauc_fpr),
                _pd_at_fa(labels, sc, pfa))

    def _train_dsm_variant(train_raw_n, cfg_rho, name, tag):
        """Train one DSM variant and return its additive scores."""
        net, h = train_dsm_local(train_raw_n, cfg_rho, seed, f'{name}_{tag}')
        loss_curves[f'{name}_{tag}'] = h
        torch.save({'state_dict': net.state_dict(), 'tag': tag},
                   os.path.join(mdl_dir, f'{name}_{tag}.pt'))
        return score_dsm_add(net, train_raw_n, test_planted, s_raw)

    def _train_score_models(train_raw_n, cfg_rho, tag):
        """Train all DSM variants + LRao on train_raw_n; return score dict."""
        scores = {}
        # --- primary DSM ---
        scores[_dsm1_label] = _train_dsm_variant(train_raw_n, cfg_rho, _dsm1_label, tag)
        # --- secondary DSM (if configured) ---
        if len(_dsm_names) > 1:
            label2 = _dsm_names[1]
            cfg2 = {**cfg_rho,
                    'hidden_dims': list(cfg['hidden_dims_2']),
                    'activation':  cfg.get('activation_2', cfg_rho['activation'])}
            scores[label2] = _train_dsm_variant(train_raw_n, cfg2, label2, tag)
        # --- LRao ---
        lrao_net, h_lrao = train_lrao_local(train_raw_n, cfg, seed, tag)
        loss_curves[f'LRao_{tag}'] = h_lrao
        torch.save({'state_dict': lrao_net.state_dict(), 'tag': tag},
                   os.path.join(mdl_dir, f'lrao_{tag}.pt'))
        scores['LRao'] = _safe(f'LRao {tag}',
                               lambda: score_lrao(lrao_net, train_raw_n, test_planted,
                                                  s_raw, cfg),
                               len(labels))
        # --- TwoBranch ---
        tb_net, h_tb = train_two_branch_local(train_raw_n, cfg_rho, seed, f'TwoBranch_{tag}')
        loss_curves[f'TwoBranch_{tag}'] = h_tb
        torch.save({'state_dict': tb_net.state_dict(), 'tag': tag},
                   os.path.join(mdl_dir, f'TwoBranch_{tag}.pt'))
        scores['TwoBranch'] = _safe(f'TwoBranch {tag}',
                                    lambda: score_dsm_add(tb_net, train_raw_n,
                                                          test_planted, s_raw),
                                    len(labels))
        return scores

    # ------------------------------------------------------------------ vs n
    print(f"\n=== SWEEP vs n  (ρ={rho_fixed}) ===", flush=True)
    for n in n_list:
        t0 = time.time()
        tr = train_pool_raw[:n]
        reg_sigma = compute_sigma_from_data(tr, rho_fixed)
        cl = run_classical_additive(tr, test_planted, s_raw, reg_sigma, cfg, mode)
        sc_dict = _train_score_models(tr, cfg, f'n{n}')
        det_scores = {**cl, **sc_dict}
        for det, sc in det_scores.items():
            au, pa, pd = _mtr(sc)
            metrics['vs_n'][det]['auc'].append(au)
            metrics['vs_n'][det]['pauc'].append(pa)
            metrics['vs_n'][det]['pd'].append(pd)
            store[f'vsN/{det}_n{n}'] = np.asarray(sc, np.float32)
        line = "  ".join(f"{d}  AUC={metrics['vs_n'][d]['auc'][-1]:.3f}  Pd={metrics['vs_n'][d]['pd'][-1]:.3f}" for d in DETS)
        print(f"  n={n:>4}  ({time.time()-t0:.0f}s)  AUC: {line}", flush=True)
        json.dump(metrics, open(os.path.join(run_dir, 'metrics.json'), 'w'),
                  indent=2, default=str)
        json.dump(loss_curves, open(os.path.join(run_dir, 'loss_curves.json'), 'w'),
                  default=str)

    # ROC at n_max (largest training set from vs-n sweep)
    n_roc = n_list[-1]
    roc_n = {d: store[f'vsN/{d}_n{n_roc}'] for d in DETS}
    _plot_roc(roc_n, labels,
              f'ROC  n={n_roc}, ρ={rho_fixed}  (additive)',
              os.path.join(fig_dir, 'roc_at_nmax.pdf'))
    metrics['roc_at_nmax'] = {d: _roc_to_dict(labels, sc) for d, sc in roc_n.items()}
    json.dump(metrics, open(os.path.join(run_dir, 'metrics.json'), 'w'),
              indent=2, default=str)

    # ----------------------------------------------------------------- vs ρ
    # classical + LRao are ρ-independent → computed once at n_fixed, drawn flat.
    print(f"\n=== SWEEP vs ρ  (n={n_fixed}) ===", flush=True)
    tr_f = train_pool_raw[:n_fixed]
    reg_sigma_f = compute_sigma_from_data(tr_f, rho_fixed)
    cl_f = run_classical_additive(tr_f, test_planted, s_raw, reg_sigma_f, cfg, mode)
    lrao_f, h_lrao_f = train_lrao_local(tr_f, cfg, seed, f'rhofix_n{n_fixed}')
    loss_curves[f'LRao_rhofix_n{n_fixed}'] = h_lrao_f
    sc_lrao_f = _safe('LRao vsρ ref',
                      lambda: score_lrao(lrao_f, tr_f, test_planted, s_raw, cfg),
                      len(labels))
    flat = {**cl_f, 'LRao': sc_lrao_f}            # ρ-independent reference scores

    for rho in rho_list:
        t0 = time.time()
        cfg_rho = {**cfg, 'dsm_sigma_rho': float(rho)}
        tag_rho = f'rho{rho}_n{n_fixed}'
        dsm_scores_rho = {_dsm1_label: _train_dsm_variant(tr_f, cfg_rho, _dsm1_label, tag_rho)}
        if len(_dsm_names) > 1:
            label2 = _dsm_names[1]
            cfg2 = {**cfg_rho,
                    'hidden_dims': list(cfg['hidden_dims_2']),
                    'activation':  cfg.get('activation_2', cfg['activation'])}
            dsm_scores_rho[label2] = _train_dsm_variant(tr_f, cfg2, label2, tag_rho)
        tb_net_rho, h_tb_rho = train_two_branch_local(tr_f, cfg_rho, seed,
                                                       f'TwoBranch_{tag_rho}')
        loss_curves[f'TwoBranch_{tag_rho}'] = h_tb_rho
        torch.save({'state_dict': tb_net_rho.state_dict(), 'tag': tag_rho},
                   os.path.join(mdl_dir, f'TwoBranch_{tag_rho}.pt'))
        dsm_scores_rho['TwoBranch'] = _safe(
            f'TwoBranch vsρ rho={rho}',
            lambda m=tb_net_rho: score_dsm_add(m, tr_f, test_planted, s_raw),
            len(labels))
        det_scores = {**flat, **dsm_scores_rho}
        for det, sc in det_scores.items():
            au, _, pd = _mtr(sc)
            metrics['vs_rho'][det]['auc'].append(au)
            metrics['vs_rho'][det]['pd'].append(pd)
            store[f'vsRho/{det}_rho{rho}'] = np.asarray(sc, np.float32)
        dsm_aucs = "  ".join(f"{n}  AUC={metrics['vs_rho'][n]['auc'][-1]:.3f}  Pd={metrics['vs_rho'][n]['pd'][-1]:.3f}"
                             for n in _dsm_names + ['TwoBranch'])
        print(f"  ρ={rho:<6}  ({time.time()-t0:.0f}s)  {dsm_aucs}", flush=True)
        json.dump(metrics, open(os.path.join(run_dir, 'metrics.json'), 'w'),
                  indent=2, default=str)

    # ROC at rho_fixed (use the ρ in rho_list closest to dsm_sigma_rho)
    rho_roc = min(rho_list, key=lambda r: abs(r - rho_fixed))
    roc_rho = {d: store[f'vsRho/{d}_rho{rho_roc}'] for d in DETS}
    _plot_roc(roc_rho, labels,
              f'ROC  n={n_fixed}, ρ={rho_roc}  (additive)',
              os.path.join(fig_dir, 'roc_at_nfixed.pdf'))
    metrics['roc_at_nfixed'] = {d: _roc_to_dict(labels, sc) for d, sc in roc_rho.items()}
    json.dump(metrics, open(os.path.join(run_dir, 'metrics.json'), 'w'),
              indent=2, default=str)

    # ---- final saves ----
    np.savez_compressed(os.path.join(run_dir, 'scores.npz'), **store)

    # ---- figures (additive only) ----
    auc_n  = {d: metrics['vs_n'][d]['auc']  for d in DETS}
    pauc_n = {d: metrics['vs_n'][d]['pauc'] for d in DETS}
    pd_n   = {d: metrics['vs_n'][d]['pd']   for d in DETS}
    auc_r  = {d: metrics['vs_rho'][d]['auc'] for d in DETS}
    pd_r   = {d: metrics['vs_rho'][d]['pd']  for d in DETS}

    _plot_vs(n_list, auc_n, 'training samples  n', 'AUC',
             'AUC vs n (additive)', os.path.join(fig_dir, 'auc_vs_n.pdf'), logx=True)
    _plot_vs(n_list, pauc_n, 'training samples  n', f'partial AUC (Pfa<{pauc_fpr})',
             f'Partial AUC (Pfa<{pauc_fpr}) vs n',
             os.path.join(fig_dir, 'pauc_vs_n.pdf'), logx=True)
    _plot_vs(n_list, pd_n, 'training samples  n', f'Pd @ Pfa={pfa}',
             f'Pd @ Pfa={pfa} vs n', os.path.join(fig_dir, 'pd_at_fa_vs_n.pdf'),
             logx=True)
    _plot_vs(rho_list, auc_r, r'DSM noise level  $\rho$', 'AUC',
             f'AUC vs ρ (n={n_fixed}, additive)',
             os.path.join(fig_dir, 'auc_vs_rho.pdf'), logx=True)
    _plot_vs(rho_list, pd_r, r'DSM noise level  $\rho$', f'Pd @ Pfa={pfa}',
             f'Pd @ Pfa={pfa} vs ρ (n={n_fixed})',
             os.path.join(fig_dir, 'pdet_at_pfa_vs_rho.pdf'), logx=True)
    plot_loss_panel(loss_curves, f'n{n_fixed}',
                    os.path.join(fig_dir, f'loss_curves_n{n_fixed}.png'))

    elapsed_min = (time.time() - t_start) / 60.0
    print(f"\nDONE in {elapsed_min:.1f} min. Results -> {run_dir}", flush=True)
    return run_dir, metrics


# ---------------------------------------------------------------------------
# Multi-seed runner — aggregates mean ± std across seeds
# ---------------------------------------------------------------------------

def run_iid_multi_seed(cfg: dict, mode: str):
    """Run run_iid once per seed; emit aggregated (mean ± std) figures.

    cfg['seed'] must be a list, e.g. [42, 43, 44, 45, 46].
    Each per-seed run goes into  <results_dir>/seed_<s>/
    Aggregated figures + metrics land in  <results_dir>/aggregate/
    Returns (aggregate_dir, aggregate_metrics).
    """
    seeds = list(cfg['seed'])
    assert len(seeds) > 1, "use run_iid directly for a single seed"
    t0 = time.time()
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')

    base_dir   = cfg['results_dir']
    agg_dir    = os.path.join(base_dir, f'iid_{mode}_agg_{ts}')
    fig_dir    = os.path.join(agg_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)

    print(f"\n{'='*60}", flush=True)
    print(f"MULTI-SEED run  mode={mode}  seeds={seeds}", flush=True)
    print(f"{'='*60}\n", flush=True)

    all_metrics = []
    seed_dirs   = []
    for s in seeds:
        cfg_s = {**cfg, 'seed': int(s),
                 'results_dir': os.path.join(agg_dir, f'seed_{s}')}
        os.makedirs(cfg_s['results_dir'], exist_ok=True)
        print(f"\n--- seed {s} ---", flush=True)
        run_dir_s, m_s = run_iid(cfg_s, mode)
        all_metrics.append(m_s)
        seed_dirs.append(run_dir_s)

    # ---- aggregate ----
    DETS     = list(all_metrics[0]['vs_n'].keys())
    n_list   = all_metrics[0]['n_list']
    rho_list = all_metrics[0]['rho_list']
    n_fixed  = all_metrics[0]['n_fixed']
    pfa      = all_metrics[0]['pfa']
    pauc_fpr = all_metrics[0]['pauc_fpr']

    def _agg(vals_list):
        """[array_seed0, array_seed1, ...] → (mean_arr, std_arr)"""
        arr = np.array([[float(v) if v == v else np.nan for v in row]
                        for row in vals_list])   # (n_seeds, n_points)
        return arr.mean(axis=0).tolist(), arr.std(axis=0).tolist()

    agg = {
        'mode': mode, 'seeds': seeds, 'n_list': n_list, 'rho_list': rho_list,
        'n_fixed': n_fixed, 'pfa': pfa, 'pauc_fpr': pauc_fpr,
        'vs_n':   {d: {} for d in DETS},
        'vs_rho': {d: {} for d in DETS},
    }
    for d in DETS:
        for metric in ('auc', 'pauc', 'pd'):
            if metric in all_metrics[0]['vs_n'][d]:
                mu, sd = _agg([m['vs_n'][d][metric] for m in all_metrics])
                agg['vs_n'][d][metric]     = mu
                agg['vs_n'][d][f'{metric}_std'] = sd
        for metric in ('auc', 'pd'):
            mu, sd = _agg([m['vs_rho'][d][metric] for m in all_metrics])
            agg['vs_rho'][d][metric]     = mu
            agg['vs_rho'][d][f'{metric}_std'] = sd

    # ROC: aggregate at nmax and nfixed — store per-seed + mean
    for roc_key in ('roc_at_nmax', 'roc_at_nfixed'):
        if roc_key in all_metrics[0]:
            agg[roc_key] = {}
            for d in DETS:
                aucs = [m[roc_key][d]['auc'] for m in all_metrics if roc_key in m]
                agg[roc_key][d] = {
                    'auc_mean': float(np.mean(aucs)),
                    'auc_std':  float(np.std(aucs)),
                    'auc_per_seed': aucs,
                }

    yaml.dump({**cfg, 'seed': seeds},
              open(os.path.join(agg_dir, 'config.yaml'), 'w'))
    json.dump(agg, open(os.path.join(agg_dir, 'metrics_aggregate.json'), 'w'),
              indent=2, default=str)

    # ---- aggregated figures (mean ± std shading) ----
    def _mean(sweep, metric):
        return {d: agg[sweep][d][metric] for d in DETS}

    def _std(sweep, metric):
        return {d: agg[sweep][d].get(f'{metric}_std') for d in DETS}

    n_seeds = len(seeds)
    suf = f'  (n={n_seeds} seeds)'

    _plot_vs(n_list, _mean('vs_n','auc'), 'training samples  n', 'AUC',
             f'AUC vs n{suf}', os.path.join(fig_dir,'auc_vs_n.pdf'),
             logx=True, series_std=_std('vs_n','auc'))
    _plot_vs(n_list, _mean('vs_n','pauc'), 'training samples  n',
             f'partial AUC (Pfa<{pauc_fpr})',
             f'Partial AUC (Pfa<{pauc_fpr}) vs n{suf}',
             os.path.join(fig_dir,'pauc_vs_n.pdf'),
             logx=True, series_std=_std('vs_n','pauc'))
    _plot_vs(n_list, _mean('vs_n','pd'), 'training samples  n',
             f'Pd @ Pfa={pfa}', f'Pd @ Pfa={pfa} vs n{suf}',
             os.path.join(fig_dir,'pd_at_fa_vs_n.pdf'),
             logx=True, series_std=_std('vs_n','pd'))
    _plot_vs(rho_list, _mean('vs_rho','auc'), r'DSM noise level  $\rho$', 'AUC',
             f'AUC vs ρ (n={n_fixed}){suf}',
             os.path.join(fig_dir,'auc_vs_rho.pdf'),
             logx=True, series_std=_std('vs_rho','auc'))
    _plot_vs(rho_list, _mean('vs_rho','pd'), r'DSM noise level  $\rho$',
             f'Pd @ Pfa={pfa}', f'Pd @ Pfa={pfa} vs ρ (n={n_fixed}){suf}',
             os.path.join(fig_dir,'pdet_at_pfa_vs_rho.pdf'),
             logx=True, series_std=_std('vs_rho','pd'))

    # ROC summary bar: mean AUC ± std at n_max
    if 'roc_at_nmax' in agg:
        fig, ax = plt.subplots(figsize=(5.5, 3.5))
        dets_r = list(agg['roc_at_nmax'].keys())
        means  = [agg['roc_at_nmax'][d]['auc_mean'] for d in dets_r]
        stds   = [agg['roc_at_nmax'][d]['auc_std']  for d in dets_r]
        colors = [_det_color(d) for d in dets_r]
        bars = ax.bar(dets_r, means, yerr=stds, color=colors, capsize=4, alpha=0.85)
        ax.set_ylabel('AUC'); ax.set_ylim(0, 1)
        ax.set_title(f'AUC at n={n_list[-1]}  (mean±std, {n_seeds} seeds)', fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        fig.tight_layout()
        _savefig(fig, os.path.join(fig_dir, 'roc_auc_bar_nmax.pdf'))
        plt.close(fig)

    elapsed = (time.time() - t0) / 60.0
    print(f"\nMULTI-SEED DONE  {elapsed:.1f} min  →  {agg_dir}", flush=True)
    return agg_dir, agg
