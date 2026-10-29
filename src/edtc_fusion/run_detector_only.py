"""
run_detector_only.py — Detector-only baselines
===============================================
Runs YOLO-only inference (no SiamCAR tracking, no DETECT/TRACK switching) on
the Anti-UAV RGB-T validation split and reports EDTC Acc.  Modes:

    --mode ir    : single-modality thermal YOLO
    --mode rgb   : single-modality visible YOLO
    --mode dbf   : dual YOLO + DBF/ABF belief fusion + suppression gate + DMS
    --mode naive : dual YOLO + averaged beliefs + suppression gate + DMS
    --mode gated : dual YOLO + learned MLP gate fusion + suppression gate + DMS
    --mode dms   : dual YOLO + DMS box selection only (no fusion/suppression)

Suppression gate (dbf/abf/naive/gated): emit the DMS box iff
    b_fused[UAV] > b_fused[bg]  AND  u_fused < theta_cls.
The DMS box, dms_winner, b_uav, b_bg and u_fused are logged for EVERY frame
(pre-gate), so the theta_cls threshold can be swept offline (exactly) over the
logged values via scripts/calibrate_detonly_threshold.py.

Per-frame pipeline (per mode)
-----------------------------
    ir / rgb : box = top YOLO detection in that modality (or [0,0,0,0])
    dual     : run both YOLO heads; fuse beliefs; gate; emit DMS winner's box

Per-sequence outputs (under --results_dir):
    <seq>.txt       : predicted [x, y, w, h] per frame
    <seq>_log.csv   : per-frame u (or u_fused), b_uav, sigma2, dms_winner
    summary.json    : aggregate Acc and config

Usage (Snellius)::

    python -m src.edtc_fusion.run_detector_only --mode ir \\
        --dataset_root /home/ssuttorp/Anti-UAV-RGBT/val \\
        --results_dir  /home/ssuttorp/pretrained_models/edtc_fusion/val_results_detector_ir \\
        --yolo_ir      /home/ssuttorp/pretrained_models/evidential/evidential_ir_stage2_best.pt \\
        --yolo_ir_base /home/ssuttorp/pretrained_models/best_clean.pt
"""

from __future__ import absolute_import, annotations

import argparse
import csv
import json
import logging
import math
import os
import sys
from copy import deepcopy
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.edtc_fusion.tracker import (
    FusionConfig,
    _yolo_run,
    _dms_with_winner,
    _ZERO_UNC,
)
from src.edtc_fusion.run_edtc_fusion import (
    compute_edtc_acc,
    compute_edtc_acc_dms,
    load_paired_sequence,
    discover_sequences,
)

# Path bootstrap mirrors tracker.py so EvidentialYOLO / belief_fusion import cleanly.
_EDTC_ROOT = _REPO_ROOT / "resources" / "EDTC"
_YOLO_ROOT = _EDTC_ROOT / "yolov5"
for _p in (str(_EDTC_ROOT), str(_YOLO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from lib.utils.belief_fusion import abf, dbf  # noqa: E402
from lib.models.yolo_evidential.model import EvidentialYOLO  # noqa: E402
from yolov5.utils.general import check_img_size  # noqa: E402
from src.edtc_fusion.tracker_gated_fusion import GateFusionModule  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("detector_only")


# ===========================================================================
# Detectors
# ===========================================================================

class _SingleYOLO:
    """Thin wrapper around _yolo_run for one modality."""

    def __init__(self, weights, base, device, cfg, K=2, miss_belief="dogmatic_bg"):
        self.model = EvidentialYOLO.load(weights, base, device, K=K).eval()
        self.device = device
        self.cfg = cfg
        self.stride = int(self.model.stride.max())
        self.imgsz = check_img_size(cfg.yolo_imgsz, s=self.stride)
        self.miss_belief = miss_belief

    def __call__(self, image_bgr):
        box, unc = _yolo_run(image_bgr, self.model, self.cfg, self.device,
                             self.imgsz, self.stride)
        # On a miss, _yolo_run returns the _ZERO_UNC placeholder b=(0,1), u=1
        # — a dogmatic background opinion that also claims full vacuity
        # (violates b.sum() + u = 1).  'vacuous' replaces it with the
        # partition-consistent zero-evidence opinion b=(0,0), u=1, matching
        # what compute_beliefs yields for alpha=(1,1).
        if box is None and self.miss_belief == "vacuous":
            unc = dict(unc)
            unc["b"] = torch.zeros_like(unc["b"])
        return box, unc


# ===========================================================================
# Per-sequence runners
# ===========================================================================

def run_sequence_single(yolo: _SingleYOLO, frames, modality: str, timer=None):
    """Single-modality YOLO over a sequence.  Returns (preds, logs)."""
    key = "ir" if modality == "ir" else "rgb"
    cuda = (yolo.device.type == "cuda")
    preds, logs = [], []
    for fr in frames:
        img = cv2.imread(fr[key])
        if img is None:
            preds.append([0.0, 0.0, 0.0, 0.0])
            logs.append({"u": None, "b_uav": None, "sigma2": None, "bbox": None})
            continue
        if timer is not None and cuda:
            torch.cuda.synchronize(yolo.device)
        t0 = perf_counter()
        box, unc = yolo(img)
        if timer is not None:
            if cuda:
                torch.cuda.synchronize(yolo.device)
            timer["t"] += perf_counter() - t0
            timer["n"] += 1
        if box is None:
            preds.append([0.0, 0.0, 0.0, 0.0])
            logs.append({
                "u":      float(unc["u"].item()),
                "b_uav":  float(unc["b"][0].item()),
                "sigma2": float(unc["sigma2_norm"].mean().item()),
                "bbox":   None,
            })
            continue
        preds.append(box)
        logs.append({
            "u":      float(unc["u"].item()),
            "b_uav":  float(unc["b"][0].item()),
            "sigma2": float(unc["sigma2_norm"].mean().item()),
            "bbox":   box,
        })
    return preds, logs


# ---------------------------------------------------------------------------
# Belief-vector fusion operators.  Each returns (b_fused, u_fused).
# Argument order to abf/dbf is (rgb, ir); both operators are symmetric.
# ---------------------------------------------------------------------------

def _fuse_naive(ir_unc, rgb_unc, device, **kw):
    b = 0.5 * (ir_unc["b"] + rgb_unc["b"])
    return b, (1.0 - b[0])


def _fuse_abf(ir_unc, rgb_unc, device, **kw):
    return abf(rgb_unc["b"], rgb_unc["u"], ir_unc["b"], ir_unc["u"])


def _fuse_dbf(ir_unc, rgb_unc, device, lam=1.0, **kw):
    return dbf(rgb_unc["b"], rgb_unc["u"], ir_unc["b"], ir_unc["u"], lam=lam)


def _fuse_gated(ir_unc, rgb_unc, device, gate=None, **kw):
    b_rgb_d = rgb_unc["b"].to(device)
    b_ir_d  = ir_unc["b"].to(device)
    with torch.no_grad():
        g = gate.detect_gate(torch.cat([b_rgb_d, b_ir_d])).squeeze()
    b = (g * b_rgb_d + (1.0 - g) * b_ir_d).cpu()
    return b, (1.0 - b[0])


def run_sequence_dual(yolo_ir: _SingleYOLO, yolo_rgb: _SingleYOLO, frames, device,
                      fuse_fn=None, theta_cls=1.0, apply_gate=True,
                      fuse_kwargs=None, timer=None):
    """Unified dual-modality runner.

    The DMS box and per-frame fused beliefs are ALWAYS computed and logged,
    independent of the suppression gate, so that the uncertainty threshold can
    later be swept offline (exactly) over the logged values.

    Gate (when apply_gate): emit DMS box iff  b_uav > b_bg  AND  u_fused < theta_cls.
    DMS-only mode sets apply_gate=False — the DMS box is always emitted.
    """
    fuse_kwargs = fuse_kwargs or {}
    cuda = (device.type == "cuda")
    preds, logs = [], []

    def _empty_log(**over):
        d = {"u_fused": None, "b_uav": None, "b_bg": None, "sigma2_best": None,
             "dms_box": None, "ir_bbox": None, "rgb_bbox": None, "dms_winner": None}
        d.update(over)
        return d

    for fr in frames:
        ir_img  = cv2.imread(fr["ir"])
        rgb_img = cv2.imread(fr["rgb"])
        if ir_img is None or rgb_img is None:
            preds.append([0.0, 0.0, 0.0, 0.0])
            logs.append(_empty_log())
            continue

        if timer is not None and cuda:
            torch.cuda.synchronize(device)
        t0 = perf_counter()

        ir_box,  ir_unc  = yolo_ir(ir_img)
        rgb_box, rgb_unc = yolo_rgb(rgb_img)

        if ir_box is None and rgb_box is None:
            if timer is not None:
                if cuda:
                    torch.cuda.synchronize(device)
                timer["t"] += perf_counter() - t0
                timer["n"] += 1
            preds.append([0.0, 0.0, 0.0, 0.0])
            logs.append(_empty_log(u_fused=1.0, b_uav=0.0, b_bg=1.0))
            continue

        # DMS box + winner — always, regardless of the gate.
        if ir_box is not None and rgb_box is not None:
            best_box_t, sigma2_best_t, dms_winner = _dms_with_winner(
                torch.tensor(rgb_box, dtype=torch.float32), rgb_unc["sigma2_norm"],
                torch.tensor(ir_box,  dtype=torch.float32), ir_unc["sigma2_norm"],
            )
            dms_box     = best_box_t.tolist()
            sigma2_best = float(sigma2_best_t.item()
                                if torch.is_tensor(sigma2_best_t) else sigma2_best_t)
        elif ir_box is not None:
            dms_box, dms_winner = ir_box, "ir"
            sigma2_best = float(ir_unc["sigma2_norm"].mean())
        else:
            dms_box, dms_winner = rgb_box, "rgb"
            sigma2_best = float(rgb_unc["sigma2_norm"].mean())

        # Fusion + suppression gate.
        if apply_gate:
            b_fused, u_fused = fuse_fn(ir_unc, rgb_unc, device, **fuse_kwargs)
            b_uav = float(b_fused[0]); b_bg = float(b_fused[1])
            u_f   = float(u_fused)
            passes = (b_uav > b_bg) and (u_f < theta_cls)
        else:  # DMS-only — no fusion, no suppression
            b_uav = b_bg = u_f = None
            passes = True

        if timer is not None:
            if cuda:
                torch.cuda.synchronize(device)
            timer["t"] += perf_counter() - t0
            timer["n"] += 1

        out_box = dms_box if passes else [0.0, 0.0, 0.0, 0.0]
        preds.append(out_box)
        logs.append({
            "u_fused": u_f, "b_uav": b_uav, "b_bg": b_bg,
            "sigma2_best": sigma2_best, "dms_box": dms_box,
            "ir_bbox": ir_box, "rgb_bbox": rgb_box, "dms_winner": dms_winner,
        })
    return preds, logs


# ===========================================================================
# Computational-efficiency measurement (params / FLOPs)
# ===========================================================================

def _count_params(model):
    try:
        return int(sum(p.numel() for p in model.parameters()))
    except Exception:
        try:
            return int(sum(p.numel() for p in model.model.parameters()))
        except Exception:
            return None


def _measure_gflops(model, imgsz, device):
    """Best-effort GFLOPs via thop (same path yolov5 uses).  None on failure."""
    try:
        import thop
        im = torch.zeros((1, 3, imgsz, imgsz), device=device)
        flops = thop.profile(deepcopy(model), inputs=(im,), verbose=False)[0]
        return float(flops / 1e9 * 2)   # *2: MACs -> FLOPs, matching yolov5
    except Exception as e:
        logger.warning("GFLOPs measurement failed: %s", e)
        return None


# ===========================================================================
# CLI
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Anti-UAV — detector-only baselines")

    p.add_argument("--mode", choices=["ir", "rgb", "dbf", "naive", "gated", "dms"], required=True,
                   help="ir=thermal YOLO only; rgb=visible YOLO only; "
                        "dbf=both YOLOs + DBF/ABF belief fusion + gate + DMS; "
                        "naive=both YOLOs + average beliefs + gate + DMS; "
                        "gated=both YOLOs + learned MLP gate fusion + gate + DMS; "
                        "dms=both YOLOs + DMS box selection only (no fusion gate)")

    # YOLO checkpoints + base weights.  For mode=ir only --yolo_ir is needed;
    # for mode=rgb only --yolo_rgb; for mode=dbf both pairs are required.
    p.add_argument("--yolo_ir",       default=None)
    p.add_argument("--yolo_ir_base",  default=None)
    p.add_argument("--yolo_rgb",      default=None)
    p.add_argument("--yolo_rgb_base", default=None)

    p.add_argument("--dataset_root", required=True)
    p.add_argument("--results_dir",  required=True)
    p.add_argument("--sequence", default=None,
                   help="Run a single sequence by name instead of the whole split.")

    # Fusion knobs (dbf mode)
    p.add_argument("--fusion", choices=["dbf", "abf"], default="dbf")
    p.add_argument("--lam", type=float, default=1.0)

    # Gate knobs (gated mode)
    p.add_argument("--gate_weights", default=None,
                   help="Path to trained GateFusionModule checkpoint (required for --mode gated).")
    p.add_argument("--gate_hidden", type=int, default=16)

    # Missing-detection encoding for dual modes.
    p.add_argument("--miss_belief", choices=["dogmatic_bg", "vacuous"],
                   default="dogmatic_bg",
                   help="Belief assigned to a stream whose NMS returns nothing. "
                        "'dogmatic_bg' = legacy _ZERO_UNC placeholder b=(0,1), "
                        "u=1 (used by all original runs); 'vacuous' = "
                        "partition-consistent zero-evidence opinion b=(0,0), "
                        "u=1.")

    # Uncertainty suppression threshold for the fusion gate (dbf/abf/naive/gated).
    # Emit a box iff  b_uav > b_bg AND u_fused < theta_cls.  Default 1.0 means
    # argmax-only (no uncertainty gating), reproducing the pre-calibration runs.
    p.add_argument("--theta_cls", type=float, default=1.0,
                   help="Upper bound on u_fused for the detection gate. "
                        "1.0 = argmax-only (no uncertainty suppression).")

    # YOLO knobs
    p.add_argument("--yolo_imgsz",   type=int,   default=640)
    p.add_argument("--yolo_conf",    type=float, default=0.25)
    p.add_argument("--yolo_iou",     type=float, default=0.45)
    p.add_argument("--yolo_classes", type=int,   nargs="*", default=None,
                   help="NMS class filter (default: [0] = UAV only). "
                        "Pass '--yolo_classes' with no values for no filtering.")

    # YOLO uncertainty aggregation (see FusionConfig.unc_agg).
    p.add_argument("--unc_agg", choices=["mean", "peak"], default="mean",
                   help="'peak' = belief/uncertainty of the selected detection only; "
                        "'mean' = average over all pre-NMS anchors (legacy default).")

    # EDTC Acc
    p.add_argument("--alpha", type=float, default=0.2)
    p.add_argument("--beta",  type=float, default=0.3)
    p.add_argument("--failure_iou_thresh", type=float, default=1e-5)

    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


def _require(args, *names):
    missing = [n for n in names if getattr(args, n) is None]
    if missing:
        raise SystemExit(f"mode={args.mode} requires --{', --'.join(missing)}")


def main():
    args = parse_args()

    # ABF is selected via --mode dbf --fusion abf (not a distinct --mode).
    DUAL_MODES = ("dbf", "naive", "gated", "dms")

    if args.mode == "ir":
        _require(args, "yolo_ir", "yolo_ir_base")
    elif args.mode == "rgb":
        _require(args, "yolo_rgb", "yolo_rgb_base")
    elif args.mode == "gated":
        _require(args, "yolo_ir", "yolo_ir_base", "yolo_rgb", "yolo_rgb_base")
        if not args.gate_weights:
            raise SystemExit("--mode gated requires --gate_weights")
    else:  # dbf / naive / dms
        _require(args, "yolo_ir", "yolo_ir_base", "yolo_rgb", "yolo_rgb_base")

    # Reuse FusionConfig purely as a settings bag for _yolo_run; only the
    # yolo_* fields are read.
    cfg = FusionConfig(
        yolo_ir_weights  = args.yolo_ir       or "",
        yolo_rgb_weights = args.yolo_rgb      or "",
        yolo_ir_base     = args.yolo_ir_base  or "",
        yolo_rgb_base    = args.yolo_rgb_base or "",
        fusion          = args.fusion,
        lam             = args.lam,
        yolo_imgsz      = (args.yolo_imgsz, args.yolo_imgsz),
        yolo_conf_thres = args.yolo_conf,
        yolo_iou_thres  = args.yolo_iou,
        yolo_classes    = args.yolo_classes,
        unc_agg         = args.unc_agg,
        device          = args.device,
    )
    device = torch.device(args.device)

    logger.info("Loading detectors for mode=%s...", args.mode)
    yolo_ir  = (_SingleYOLO(args.yolo_ir,  args.yolo_ir_base,  device, cfg,
                            miss_belief=args.miss_belief)
                if args.mode in ("ir",) + DUAL_MODES else None)
    yolo_rgb = (_SingleYOLO(args.yolo_rgb, args.yolo_rgb_base, device, cfg,
                            miss_belief=args.miss_belief)
                if args.mode in ("rgb",) + DUAL_MODES else None)
    gate     = (GateFusionModule.load(args.gate_weights, device,
                                      K=cfg.K, hidden=args.gate_hidden)
                if args.mode == "gated" else None)

    # Resolve the fusion operator for dual modes (DMS-only has no fusion gate).
    if args.mode == "dbf":
        fuse_fn     = _fuse_abf if args.fusion == "abf" else _fuse_dbf
        fuse_kwargs = {} if args.fusion == "abf" else {"lam": args.lam}
    elif args.mode == "naive":
        fuse_fn, fuse_kwargs = _fuse_naive, {}
    elif args.mode == "gated":
        fuse_fn, fuse_kwargs = _fuse_gated, {"gate": gate}
    else:  # dms-only
        fuse_fn, fuse_kwargs = None, {}
    apply_gate = (args.mode != "dms")
    timer = {"t": 0.0, "n": 0}

    sequences = discover_sequences(args.dataset_root)
    if args.sequence is not None:
        sequences = [s for s in sequences if s == args.sequence]
        if not sequences:
            raise SystemExit(f'Sequence "{args.sequence}" not found.')

    os.makedirs(args.results_dir, exist_ok=True)

    # In single-modality mode we eval against that modality's GT (Option-A).
    # In dbf mode we eval against the DMS winner's GT (Option-B), matching the
    # convention used by run_edtc_fusion.py.
    accs = []

    for v_idx, seq_name in enumerate(sequences):
        seq_dir = os.path.join(args.dataset_root, seq_name)
        frames  = load_paired_sequence(seq_dir)
        if not frames:
            logger.warning("No paired frames for %s — skipping.", seq_name)
            continue

        if args.mode in ("ir", "rgb"):
            yolo = yolo_ir if args.mode == "ir" else yolo_rgb
            preds, fr_logs = run_sequence_single(yolo, frames, args.mode, timer=timer)
            gt_key = "gt" if args.mode == "ir" else "rgb_gt"
            eval_frames = [{"gt": fr[gt_key]} for fr in frames]
            acc = compute_edtc_acc(preds, eval_frames,
                                   alpha=args.alpha, beta=args.beta,
                                   failure_iou_thresh=args.failure_iou_thresh)
        else:  # dual modes: dbf / abf / naive / gated / dms
            preds, fr_logs = run_sequence_dual(
                yolo_ir, yolo_rgb, frames, device,
                fuse_fn=fuse_fn, theta_cls=args.theta_cls,
                apply_gate=apply_gate, fuse_kwargs=fuse_kwargs, timer=timer)
            dms_winners = [r.get("dms_winner") or "ir" for r in fr_logs]
            acc = compute_edtc_acc_dms(preds, frames, dms_winners,
                                       alpha=args.alpha, beta=args.beta,
                                       failure_iou_thresh=args.failure_iou_thresh)

        if not math.isnan(acc):
            accs.append(acc)

        logger.info("[%3d/%3d] %-30s  Acc=%.4f  (%d frames)",
                    v_idx + 1, len(sequences), seq_name, acc, len(frames))

        # Per-sequence bbox file
        bbox_path = os.path.join(args.results_dir, seq_name + ".txt")
        with open(bbox_path, "w") as f:
            for bbox in preds:
                f.write(",".join("{:.4f}".format(v) for v in bbox) + "\n")

        # Per-frame log
        log_path = os.path.join(args.results_dir, seq_name + "_log.csv")
        with open(log_path, "w", newline="") as f:
            w = csv.writer(f)
            if args.mode in ("ir", "rgb"):
                w.writerow(["frame", "u", "b_uav", "sigma2", "bbox"])
                for i, r in enumerate(fr_logs):
                    w.writerow([
                        i,
                        r["u"]      if r["u"]      is not None else "",
                        r["b_uav"]  if r["b_uav"]  is not None else "",
                        r["sigma2"] if r["sigma2"] is not None else "",
                        json.dumps(r["bbox"]) if r["bbox"] else "",
                    ])
            else:  # dual modes (dbf/abf/naive/gated/dms) — shared schema.
                # dms_box and dms_winner are always recorded (pre-gate) so the
                # uncertainty threshold can be swept offline over these logs.
                w.writerow(["frame", "u_fused", "b_uav", "b_bg", "sigma2_best",
                            "dms_box", "ir_bbox", "rgb_bbox", "dms_winner"])
                for i, r in enumerate(fr_logs):
                    ir_b   = json.dumps(r["ir_bbox"])  if r.get("ir_bbox")  else ""
                    rgb_b  = json.dumps(r["rgb_bbox"]) if r.get("rgb_bbox") else ""
                    dms_b  = json.dumps(r["dms_box"])  if r.get("dms_box")  else ""
                    w.writerow([
                        i,
                        r["u_fused"]     if r["u_fused"]     is not None else "",
                        r["b_uav"]       if r["b_uav"]       is not None else "",
                        r["b_bg"]        if r["b_bg"]        is not None else "",
                        r["sigma2_best"] if r["sigma2_best"] is not None else "",
                        dms_b, ir_b, rgb_b, r.get("dms_winner") or "",
                    ])

    mean_acc = float(np.mean(accs)) if accs else float("nan")
    logger.info("=" * 70)
    logger.info("mode=%s  Acc=%.4f  [%d sequences]  theta_cls=%.3f",
                args.mode, mean_acc, len(accs), args.theta_cls)
    logger.info("=" * 70)

    # ── Computational efficiency (params / FLOPs / FPS) ──────────────────
    imgsz_sq = args.yolo_imgsz
    eff = {"imgsz": imgsz_sq, "device": str(device)}
    p_ir  = _count_params(yolo_ir.model)  if yolo_ir  is not None else None
    p_rgb = _count_params(yolo_rgb.model) if yolo_rgb is not None else None
    p_gate = _count_params(gate)          if gate     is not None else None
    g_ir  = _measure_gflops(yolo_ir.model,  imgsz_sq, device) if yolo_ir  is not None else None
    g_rgb = _measure_gflops(yolo_rgb.model, imgsz_sq, device) if yolo_rgb is not None else None
    eff["params_yolo_ir"]  = p_ir
    eff["params_yolo_rgb"] = p_rgb
    eff["params_gate"]     = p_gate
    eff["params_total"]    = int(sum(x for x in (p_ir, p_rgb, p_gate) if x))
    eff["gflops_yolo_ir"]  = g_ir
    eff["gflops_yolo_rgb"] = g_rgb
    eff["gflops_total"]    = (None if (g_ir is None and g_rgb is None)
                              else float(sum(x for x in (g_ir, g_rgb) if x)))
    eff["frames_timed"]    = timer["n"]
    eff["mean_fps"]        = (timer["n"] / timer["t"]) if timer["t"] > 0 else None
    eff["mean_latency_ms"] = (1000.0 * timer["t"] / timer["n"]) if timer["n"] else None
    if eff["mean_fps"] is not None:
        logger.info("Efficiency: params_total=%s  gflops_total=%s  mean_fps=%.2f  latency=%.2fms",
                    eff["params_total"], eff["gflops_total"],
                    eff["mean_fps"], eff["mean_latency_ms"])

    summary = {
        "mode":               args.mode,
        "acc":                mean_acc,
        "n_sequences":        len(accs),
        "theta_cls":          args.theta_cls,
        "unc_agg":            args.unc_agg,
        "miss_belief":        args.miss_belief,
        "fusion":             (args.fusion    if args.mode == "dbf"
                              else "naive_avg" if args.mode == "naive"
                              else "gated_mlp" if args.mode == "gated"
                              else "dms_only"  if args.mode == "dms"
                              else None),
        "lam":                args.lam         if args.mode == "dbf"   else None,
        "gate_weights":       args.gate_weights if args.mode == "gated" else None,
        "alpha":              args.alpha,
        "beta":               args.beta,
        "failure_iou_thresh": args.failure_iou_thresh,
        "yolo_conf_thres":    args.yolo_conf,
        "yolo_iou_thres":     args.yolo_iou,
        "efficiency":         eff,
    }
    with open(os.path.join(args.results_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Summary -> %s", os.path.join(args.results_dir, "summary.json"))


if __name__ == "__main__":
    main()
