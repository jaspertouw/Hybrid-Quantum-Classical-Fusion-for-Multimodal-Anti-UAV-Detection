"""
rescore_belief_metrics.py — Belief-quality metrics for a finished run
=====================================================================
Post-hoc evaluation of ECE / AUROC / AUSE on a run's already-saved
per-frame logs.  No model is reloaded and no GPU is needed — everything
needed is in ``<seq>.txt`` (bboxes) and ``<seq>_log.csv`` (b_uav, u).

These metrics score the *belief distribution* — not the bbox — so they
distinguish fusion operators (DBF vs ABF) that produce different fused
beliefs but identical bbox streams.

Confidence signals (matches EDTC's tracking/compute_metrics.py)
---------------------------------------------------------------
For K=2 the Dirichlet evidential head gives belief (b_uav, b_bg) and
vacuity u, with b_uav + b_bg + u = 1.  We expose two derived signals:

    conf_prob = E[P(UAV)] = b_uav + u / 2     (Dirichlet expected prob.)
    conf_1mu  = 1 - u                          (total non-vacuity)

Per-frame correctness label (also matches compute_metrics.py)
-------------------------------------------------------------
    present frame : correct iff IoU(pred, gt) >= iou_threshold  (default 0.5)
    absent  frame : correct iff conf_prob <= cls_threshold       (predict bg;
                    strict <= so that ZERO_UNC frames with conf_prob == 0.5 are
                    treated as correct background predictions)

Metrics reported (computed by lib.utils.eval_metrics from EDTC)
----------------------------------------------------------------
    Accuracy         : binary correctness over all frames
    Mean IoU         : present frames only
    ECE              : expected calibration error
    AUROC            : ranking of correct vs incorrect frames by confidence
    AUSE             : area under sparsification error (vs. oracle)

Usage
-----
    python -m src.edtc_fusion.rescore_belief_metrics \\
        --results_dir  /home/ssuttorp/pretrained_models/edtc_fusion/val_results_detector_only_dbf \\
        --dataset_root /home/ssuttorp/Anti-UAV-RGBT/val

Writes ``summary_belief_metrics.json`` next to the existing ``summary.json``.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
_EDTC_ROOT = _REPO_ROOT / "resources" / "EDTC"
for _p in (str(_REPO_ROOT), str(_EDTC_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Re-use EDTC's metric implementations verbatim — no reimplementation.
from lib.utils.eval_metrics import (  # noqa: E402
    compute_ece,
    compute_auroc,
    compute_sparsification,
)

from src.edtc_fusion.run_edtc_fusion import (  # noqa: E402
    iou_xywh,
    load_paired_sequence,
    discover_sequences,
)


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def _load_preds(txt_path: str):
    """Read <seq>.txt — one CSV row per frame: x,y,w,h."""
    preds = []
    with open(txt_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            preds.append([float(p) for p in parts[:4]])
    return preds


def _load_log(csv_path: str):
    """Normalise per-frame log into ``[{b_uav, u, state, dms_winner}, ...]``.

    Schemas covered:

        Det+Track fusion : state, u_fused, b_uav, sigma2_best, switched,
                           ir_bbox, rgb_bbox, dms_winner
        Det+Track single : state, u, b_uav, sigma2, switched, bbox
        Det-only fusion  : u_fused, b_uav, sigma2_best, ir_bbox, rgb_bbox,
                           dms_winner
        Det-only single  : u, b_uav, sigma2, bbox
    """
    def _f(v):
        if v in (None, "", "None"):
            return None
        try:
            return float(v)
        except ValueError:
            return None

    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            u_val = _f(r.get("u_fused")) if r.get("u_fused") else _f(r.get("u"))
            sig_val = _f(r.get("sigma2_best"))
            if sig_val is None:
                sig_val = _f(r.get("sigma2"))
            rows.append({
                "b_uav":      _f(r.get("b_uav")),
                "u":          u_val,
                "sigma2":     sig_val,
                "state":      (r.get("state") or None),
                "dms_winner": (r.get("dms_winner") or None),
            })
    return rows


def _detect_run_type(summary: dict) -> str:
    """Classify run from summary.json keys.  See rescore_belief_gated for the
    same scheme — same labels."""
    if "mode" in summary:
        m = summary["mode"]
        if m == "ir":  return "detonly-ir"
        if m == "rgb": return "detonly-rgb"
        return "detonly-fusion"
    if "modality" in summary:
        return "single-" + summary["modality"]
    return "fusion"


# ---------------------------------------------------------------------------
# Detection AP — AP@0.5 and COCO-style mAP@[0.5:0.95]
# ---------------------------------------------------------------------------
# Anti-UAV is single-object: each frame holds at most one GT (present/absent)
# and the runner emits at most one box, already thresholded at yolo_conf=0.25
# plus the suppression gate.  We therefore compute AP over the *emitted* boxes,
# pooled across every frame of the split, ranked by a per-box confidence score:
#     belief runs : conf_prob = b_uav + u/2   (Dirichlet expected P(UAV), K=2)
#     DMS run     : -sigma2_best              (lower variance = more confident)
# n_gt = number of present frames (recall denominator).  Because sub-0.25
# detections were never emitted, the high-recall tail is truncated, so AP is a
# (conservative) LOWER BOUND — but every variant shares the same operating
# point, so cross-variant comparison stays fair.

AP_IOU_THRESHOLDS = [round(0.5 + 0.05 * i, 2) for i in range(10)]  # 0.50..0.95


def _average_precision(scores, is_tp, n_gt):
    """COCO 101-point interpolated AP at a single IoU threshold."""
    if n_gt == 0:
        return float("nan")
    scores = np.asarray(scores, dtype=float)
    if scores.size == 0:
        return 0.0
    is_tp = np.asarray(is_tp, dtype=float)
    order = np.argsort(-scores, kind="mergesort")          # high conf first
    tp_cum = np.cumsum(is_tp[order])
    fp_cum = np.cumsum(1.0 - is_tp[order])
    recall    = tp_cum / float(n_gt)
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)
    # Monotone-decreasing precision envelope (max precision at recall >= r).
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    ap = 0.0
    for t in np.linspace(0.0, 1.0, 101):
        mask = recall >= t
        ap += precision[mask].max() if mask.any() else 0.0
    return ap / 101.0


def _compute_map(scores, ious, present, n_gt, thresholds=AP_IOU_THRESHOLDS):
    """Return ({iou_thr: AP}, AP@0.5, mAP@[0.5:0.95]) for emitted detections."""
    scores  = np.asarray(scores,  dtype=float)
    ious    = np.asarray(ious,    dtype=float)
    present = np.asarray(present, dtype=bool)
    per_iou = {}
    for r in thresholds:
        is_tp = present & (ious >= r)        # absent frames -> never a TP
        per_iou[r] = _average_precision(scores, is_tp, n_gt)
    ap50  = per_iou.get(0.5, float("nan"))
    valid = [v for v in per_iou.values() if not np.isnan(v)]
    mAP   = float(np.mean(valid)) if valid else float("nan")
    return per_iou, ap50, mAP


# ---------------------------------------------------------------------------
# Per-sequence array assembly
# ---------------------------------------------------------------------------

def _per_sequence_arrays(preds, log_rows, frames, run_type, iou_threshold,
                         cls_threshold):
    """Build aligned per-frame arrays for the metric functions.

    Frames where b_uav or u are missing are SKIPPED entirely (no confidence →
    no scoring possible), matching the convention in compute_metrics.py
    where init/NaN frames are excluded from confidence-based metrics.
    """
    is_rgb_single = run_type in ("single-rgb", "detonly-rgb")
    is_fusion     = run_type in ("fusion", "detonly-fusion")

    n = min(len(preds), len(log_rows), len(frames))

    correctness   = []
    conf_prob_all = []
    conf_1mu_all  = []
    ious_pres     = []   # all present frames (bbox metric + sigma2 AUSE)
    ious_belief   = []   # present frames that also have beliefs (belief AUSE)
    cp_pres       = []
    cm_pres       = []
    cs_pres       = []   # sigma2-based confidence (-sigma2_best) for AUSE
    # Per-frame flags for the belief-based classification confusion matrix.
    # "Predicted UAV" := conf_prob > cls_threshold  (Dirichlet expected P(UAV)
    # strictly exceeds the decision threshold). This is a pure classification
    # signal — independent of the bbox — so ABF and DBF produce genuinely
    # different confusion matrices even when their bbox streams are identical.
    present_flags  = []
    pred_uav_flags = []
    # Official Anti-UAV State Accuracy building blocks (belief-INDEPENDENT).
    # Replicates tracking/evaluate_antiuav_performance.py::eval verbatim:
    #   measure = IoU on present frames (0 if no box) / 1 if empty pred on absent
    #   penalty = 1 on present frames with IoU < 1e-5 (miss / way-off box)
    # The sequence SA is mean(measure) - 0.2 * mean(penalty)**0.3 (computed below;
    # the nonlinear penalty forces a PER-SEQUENCE reduction, never pooled).
    sa_measure = []
    sa_penalty = []

    # ── AP / mAP collection (one row per EMITTED box) ─────────────────────
    ap_score   = []   # ranking confidence per emitted detection
    ap_iou     = []   # IoU with GT (0.0 on absent frames / misses)
    ap_present = []   # was this a present (GT-bearing) frame?
    ap_n_gt    = 0    # number of present frames (recall denominator)

    for i in range(n):
        pred = preds[i]
        r    = log_rows[i]
        fr   = frames[i]

        b_uav, u = r["b_uav"], r["u"]
        has_belief = (b_uav is not None and u is not None)

        # Presence anchor: IR GT is the canonical "exist" signal in this dataset,
        # EXCEPT for the single-modality RGB run, which is evaluated entirely
        # against RGB GT (matching run_detector_only.py's gt_key="rgb_gt").  Using
        # IR presence there would mis-score frames where the UAV is visible in IR
        # but not RGB (or vice versa) and is why RGB diverged from the runner's Acc.
        if is_rgb_single:
            present = (fr.get("rgb_gt") is not None)
        else:
            present = (fr["gt"] is not None)
        pred_empty_now = (pred[2] <= 0 or pred[3] <= 0)

        # ── Present-frame localisation arrays (belief-INDEPENDENT) ────────
        # IoU and the sigma2-confidence are available whenever a GT box exists,
        # so they are recorded even for the belief-free DMS run (which has no
        # b_uav/u column).  This drives mean IoU and the sigma2-based AUSE.
        iou = None
        if present:
            if is_fusion:
                winner = r["dms_winner"] or "ir"
                gt = fr.get("rgb_gt") if winner == "rgb" else fr["gt"]
            elif is_rgb_single:
                gt = fr.get("rgb_gt")
            else:
                gt = fr["gt"]
            iou = iou_xywh(pred, gt) if (gt is not None and not pred_empty_now) else 0.0
            # Localisation confidence = -sigma2_best.  Missing sigma (no detection
            # on a present frame) → worst confidence, matching the _ZERO_UNC
            # sentinel (sigma2 = 1e6) so such frames sparsify out first.
            sigma2 = r.get("sigma2")
            conf_sigma = -(sigma2 if sigma2 is not None else 1e6)
            ious_pres.append(iou)
            cs_pres.append(conf_sigma)
            # State Accuracy (present frame): credit the IoU, penalise a miss.
            sa_measure.append(iou)
            sa_penalty.append(1.0 if iou < 1e-5 else 0.0)
        else:
            # State Accuracy (absent frame): reward predicting "no UAV" (empty
            # box); never penalised (penalty term only applies to present frames).
            sa_measure.append(1.0 if pred_empty_now else 0.0)
            sa_penalty.append(0.0)

        # ── AP / mAP bookkeeping (belief-INDEPENDENT; one row per emitted box)
        # n_gt counts every present frame; a detection row is added only when a
        # non-empty box was emitted.  Score: conf_prob if beliefs exist, else
        # -sigma2 (belief-free DMS run).
        if present:
            ap_n_gt += 1
        if not pred_empty_now:
            if has_belief:
                det_score = b_uav + u / 2.0
            else:
                sg = r.get("sigma2")
                det_score = -(sg if sg is not None else 1e6)
            ap_score.append(det_score)
            ap_present.append(present)
            ap_iou.append(iou if (present and iou is not None) else 0.0)

        # ── Belief-dependent metrics — only when beliefs are logged ───────
        if not has_belief:
            continue                  # DMS / belief-free run: bbox+sigma done

        conf_prob = b_uav + u / 2.0   # E[P(UAV)] for K=2
        conf_1mu  = 1.0 - u           # 1 - vacuity
        # Belief-based class decision (independent of the bbox).
        # Strict > so that ZERO_UNC frames (conf_prob == 0.5 when u=1, b_uav=0,
        # i.e. no NMS detection) fall into "no UAV predicted".
        pred_uav_now = (conf_prob > cls_threshold)
        present_flags.append(present)
        pred_uav_flags.append(pred_uav_now)

        if present:
            correct = int(iou >= iou_threshold)
            ious_belief.append(iou)
            cp_pres.append(conf_prob)
            cm_pres.append(conf_1mu)
        else:
            # Absent → correct iff the model votes background (complement of the
            # class-decision flag above).
            correct = int(not pred_uav_now)

        correctness.append(correct)
        conf_prob_all.append(conf_prob)
        conf_1mu_all.append(conf_1mu)

    # Per-sequence State Accuracy (official Anti-UAV reduction).
    if sa_measure:
        sa_arr = np.array(sa_measure, dtype=float)
        sp_arr = np.array(sa_penalty, dtype=float)
        state_accuracy = float(sa_arr.mean()) - 0.2 * float(sp_arr.mean()) ** 0.3
    else:
        state_accuracy = float("nan")

    return {
        "correctness":      np.array(correctness,      dtype=int),
        "conf_prob":        np.array(conf_prob_all,    dtype=float),
        "conf_1mu":         np.array(conf_1mu_all,     dtype=float),
        "ious_present":     np.array(ious_pres,        dtype=float),
        "ious_belief":      np.array(ious_belief,      dtype=float),
        "cp_present":       np.array(cp_pres,          dtype=float),
        "cm_present":       np.array(cm_pres,          dtype=float),
        "cs_present":       np.array(cs_pres,          dtype=float),
        "present_flags":    np.array(present_flags,    dtype=bool),
        "pred_uav_flags":   np.array(pred_uav_flags,   dtype=bool),
        "state_accuracy":   state_accuracy,
        "ap_score":         np.array(ap_score,   dtype=float),
        "ap_iou":           np.array(ap_iou,     dtype=float),
        "ap_present":       np.array(ap_present, dtype=bool),
        "ap_n_gt":          ap_n_gt,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Belief-quality metrics (ECE / AUROC / AUSE) for a finished run.")
    p.add_argument("--results_dir",  required=True)
    p.add_argument("--dataset_root", required=True,
                   help="Anti-UAV-RGBT split root (e.g. .../val or .../test)")
    p.add_argument("--iou_threshold", type=float, default=0.5,
                   help="IoU threshold for present-frame correctness (default 0.5; "
                        "matches EDTC compute_metrics.py).")
    p.add_argument("--cls_threshold", type=float, default=0.5,
                   help="Threshold on conf_prob for the belief-based class "
                        "decision (default 0.5; matches EDTC's bg-prediction rule).")
    return p.parse_args()


def main():
    args = parse_args()

    summary_path = os.path.join(args.results_dir, "summary.json")
    if not os.path.isfile(summary_path):
        raise SystemExit(f"missing {summary_path}")
    with open(summary_path) as f:
        original_summary = json.load(f)
    run_type = _detect_run_type(original_summary)

    print(f"Run type      : {run_type}")
    print(f"Results dir   : {args.results_dir}")
    print(f"Dataset root  : {args.dataset_root}")
    print(f"IoU threshold : {args.iou_threshold}")
    print()

    sequences = discover_sequences(args.dataset_root)

    all_correct      = []
    all_cp_all       = []
    all_cm_all       = []
    all_ious_pres    = []
    all_ious_belief  = []
    all_cp_pres      = []
    all_cm_pres      = []
    all_cs_pres      = []
    all_present      = []
    all_pred_uav     = []
    all_state_acc    = []   # one State Accuracy per sequence (unweighted mean)
    all_ap_score     = []
    all_ap_iou       = []
    all_ap_present   = []
    total_ap_n_gt    = 0
    n_seq_used       = 0

    for v_idx, seq in enumerate(sequences):
        bbox_path = os.path.join(args.results_dir, seq + ".txt")
        log_path  = os.path.join(args.results_dir, seq + "_log.csv")
        seq_dir   = os.path.join(args.dataset_root, seq)
        if not (os.path.isfile(bbox_path) and os.path.isfile(log_path)):
            continue
        preds    = _load_preds(bbox_path)
        log_rows = _load_log(log_path)
        frames   = load_paired_sequence(seq_dir)
        if not frames:
            continue

        a = _per_sequence_arrays(preds, log_rows, frames, run_type,
                                 args.iou_threshold, args.cls_threshold)
        # Keep the sequence if it has EITHER belief data OR present-frame IoUs
        # (the latter alone suffices for the belief-free DMS run → sigma2 AUSE).
        if len(a["correctness"]) == 0 and len(a["ious_present"]) == 0:
            continue

        all_correct.append(a["correctness"])
        all_cp_all.append(a["conf_prob"])
        all_cm_all.append(a["conf_1mu"])
        all_ious_pres.append(a["ious_present"])
        all_ious_belief.append(a["ious_belief"])
        all_cp_pres.append(a["cp_present"])
        all_cm_pres.append(a["cm_present"])
        all_cs_pres.append(a["cs_present"])
        all_present.append(a["present_flags"])
        all_pred_uav.append(a["pred_uav_flags"])
        if not np.isnan(a["state_accuracy"]):
            all_state_acc.append(a["state_accuracy"])
        all_ap_score.append(a["ap_score"])
        all_ap_iou.append(a["ap_iou"])
        all_ap_present.append(a["ap_present"])
        total_ap_n_gt += a["ap_n_gt"]
        n_seq_used += 1

        acc_seq = float(a["correctness"].mean()) if len(a["correctness"]) else float("nan")
        miou    = float(a["ious_present"].mean())  if len(a["ious_present"])  else float("nan")
        print(f"[{v_idx + 1:3d}/{len(sequences)}] {seq:<40s}  "
              f"acc={acc_seq:.4f}  meanIoU={miou:.4f}  SA={a['state_accuracy']:.4f}  "
              f"frames_scored={len(a['correctness'])}  present={len(a['ious_present'])}")

    if not all_correct and not all_ious_pres:
        raise SystemExit("No sequences with usable data were found.")

    correctness  = np.concatenate(all_correct)     if all_correct    else np.array([], dtype=int)
    cp_all       = np.concatenate(all_cp_all)       if all_cp_all     else np.array([])
    cm_all       = np.concatenate(all_cm_all)       if all_cm_all     else np.array([])
    ious_pres    = np.concatenate(all_ious_pres)    if all_ious_pres  else np.array([])
    ious_belief  = np.concatenate(all_ious_belief)  if all_ious_belief else np.array([])
    cp_pres      = np.concatenate(all_cp_pres)      if all_cp_pres    else np.array([])
    cm_pres      = np.concatenate(all_cm_pres)      if all_cm_pres    else np.array([])
    cs_pres      = np.concatenate(all_cs_pres)      if all_cs_pres    else np.array([])
    present_flags  = np.concatenate(all_present)    if all_present    else np.array([], dtype=bool)
    pred_uav_flags = np.concatenate(all_pred_uav)   if all_pred_uav   else np.array([], dtype=bool)

    # ── Belief-based classification confusion matrix ───────────────────────
    # Class decision: predicted UAV iff conf_prob >= cls_threshold.  This is
    # independent of the bbox — it scores the Dirichlet *belief* itself, which
    # is exactly what differs between ABF and DBF.
    tp = int(np.sum( present_flags &  pred_uav_flags))
    fn = int(np.sum( present_flags & ~pred_uav_flags))
    fp = int(np.sum(~present_flags &  pred_uav_flags))
    tn = int(np.sum(~present_flags & ~pred_uav_flags))

    def _safe_div(num, den):
        return float(num) / float(den) if den > 0 else float("nan")

    cls_acc     = _safe_div(tp + tn, tp + fp + tn + fn)        # (TP+TN) / total
    precision   = _safe_div(tp, tp + fp)                       # PPV
    recall      = _safe_div(tp, tp + fn)                       # TPR / sensitivity
    specificity = _safe_div(tn, tn + fp)                       # TNR
    far         = _safe_div(fp, fp + tn)                       # 1 - specificity
    f1 = (2.0 * precision * recall / (precision + recall)
          if (precision + recall) and not np.isnan(precision) and not np.isnan(recall)
          else float("nan"))
    bal_acc = ((recall + specificity) / 2.0
               if not np.isnan(recall) and not np.isnan(specificity)
               else float("nan"))

    # ── EDTC's metric functions ────────────────────────────────────────────
    # Belief metrics require belief frames (absent for the belief-free DMS run).
    # sigma2-AUSE uses ALL present frames, so it is available even without beliefs.
    has_belief = len(correctness) > 0
    ece_prob   = compute_ece(cp_all, correctness)   if has_belief else float("nan")
    ece_1mu    = compute_ece(cm_all, correctness)   if has_belief else float("nan")
    auroc_prob = compute_auroc(cp_all, correctness) if has_belief else float("nan")
    auroc_1mu  = compute_auroc(cm_all, correctness) if has_belief else float("nan")
    sp_prob    = compute_sparsification(ious_belief, cp_pres)   # belief AUSE
    sp_1mu     = compute_sparsification(ious_belief, cm_pres)
    sp_sigma   = compute_sparsification(ious_pres,   cs_pres)   # sigma2 AUSE (all present)

    accuracy   = float(correctness.mean()) if has_belief else float("nan")
    mean_iou   = float(ious_pres.mean()) if len(ious_pres) else float("nan")
    # Official Anti-UAV State Accuracy: unweighted mean over per-sequence SA
    # (matches evaluate_antiuav_performance.py's np.mean(overall_performance)).
    state_acc  = float(np.mean(all_state_acc)) if all_state_acc else float("nan")

    # ── Detection AP@0.5 and COCO mAP@[0.5:0.95] ───────────────────────────
    ap_score   = np.concatenate(all_ap_score)   if all_ap_score   else np.array([])
    ap_iou     = np.concatenate(all_ap_iou)     if all_ap_iou     else np.array([])
    ap_present = np.concatenate(all_ap_present) if all_ap_present else np.array([], dtype=bool)
    ap_per_iou, ap50, map5095 = _compute_map(ap_score, ap_iou, ap_present, total_ap_n_gt)
    ap_score_signal = ("conf_prob = b_uav + u/2  (Dirichlet expected P(UAV), K=2)"
                       if has_belief else "-sigma2_best  (belief-free DMS run)")

    out = {
        "run_type":          run_type,
        "results_dir":       args.results_dir,
        "dataset_root":      args.dataset_root,
        "iou_threshold":     args.iou_threshold,
        "n_sequences":       n_seq_used,
        "n_frames_scored":   int(len(correctness)),
        "n_frames_present":  int(len(ious_pres)),
        # Sanity-check metrics (depend on bbox)
        "accuracy_threshold": (accuracy if not np.isnan(accuracy) else None),
        "mean_iou_present":   (mean_iou if not np.isnan(mean_iou) else None),
        # Official Anti-UAV State Accuracy (per-seq, then averaged; benchmark metric)
        "state_accuracy":     (state_acc if not np.isnan(state_acc) else None),
        # Detection AP (single-object; emitted boxes pooled over frames, COCO 101-pt)
        "ap_50":              (float(ap50)    if not np.isnan(ap50)    else None),
        "map_50_95":          (float(map5095) if not np.isnan(map5095) else None),
        "ap_per_iou":         {("%.2f" % k): (float(v) if not np.isnan(v) else None)
                               for k, v in ap_per_iou.items()},
        "ap_n_detections":    int(ap_score.size),
        "ap_n_gt":            int(total_ap_n_gt),
        "ap_score_signal":    ap_score_signal,
        "ap_note": ("AP over emitted boxes (conf>=0.25) pooled across all frames; "
                    "COCO 101-point interpolation; n_gt = #present frames. The "
                    "high-recall tail is truncated by the 0.25 operating point, so "
                    "values are a conservative lower bound — fair across variants."),
        # Belief-based classification (handles class imbalance; ABF vs DBF
        # produce different counts because their fused belief differs)
        "belief_classification": {
            "cls_threshold":     args.cls_threshold,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "n_present_gt":      int(np.sum(present_flags)),
            "n_absent_gt":       int(np.sum(~present_flags)),
            "accuracy":          cls_acc,          # (TP + TN) / total
            "precision":         precision,        # TP / (TP + FP)
            "recall":            recall,           # TP / (TP + FN)   sensitivity
            "specificity":       specificity,      # TN / (TN + FP)
            "false_alarm_rate":  far,              # FP / (FP + TN)
            "f1":                f1,
            "balanced_accuracy": bal_acc,
            "definition": (
                "predicted_uav := conf_prob > cls_threshold; "
                "conf_prob = b_uav + u_fused / 2  (Dirichlet expected P(UAV), K=2)"
            ),
        },
        # Belief-quality metrics — the actual fusion comparison signals
        "ece_conf_prob":   (float(ece_prob) if not np.isnan(ece_prob) else None),
        "ece_conf_1mu":    (float(ece_1mu)  if not np.isnan(ece_1mu)  else None),
        "auroc_conf_prob": float(auroc_prob) if not np.isnan(auroc_prob) else None,
        "auroc_conf_1mu":  float(auroc_1mu)  if not np.isnan(auroc_1mu)  else None,
        "ause_conf_prob":  float(sp_prob["ause"]) if not np.isnan(sp_prob["ause"]) else None,
        "ause_conf_1mu":   float(sp_1mu["ause"])  if not np.isnan(sp_1mu["ause"])  else None,
        "ause_conf_sigma": float(sp_sigma["ause"]) if not np.isnan(sp_sigma["ause"]) else None,
        "signals": {
            "conf_prob":  "b_uav + u_fused / 2  (Dirichlet expected P(UAV), K=2)",
            "conf_1mu":   "1 - u_fused          (total non-vacuity)",
            "conf_sigma": "-sigma2_best         (localisation confidence; lower variance = higher conf)",
        },
    }

    out_path = os.path.join(args.results_dir, "summary_belief_metrics.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    # ── Single consolidated table ──────────────────────────────────────────
    n_present = int(np.sum(present_flags))
    n_absent  = int(np.sum(~present_flags))
    width = 64
    print()
    print("=" * width)
    print(f"Run type     : {run_type}")
    print(f"Frames       : {len(correctness)} scored  "
          f"({n_present} present, {n_absent} absent)")
    print(f"Thresholds   : IoU>={args.iou_threshold}  "
          f"Cls(conf_prob)>{args.cls_threshold}")
    print("=" * width)
    print(f"{'Metric':<24} {'conf_prob':>12} {'conf_1mu':>12}     {'Notes':<10}")
    print("-" * width)
    # Bbox/IoU sanity-check (single column)
    print(f"{'Accuracy (thresh.)':<24} {accuracy:>12.4f} {'-':>12}     "
          f"per-frame correct")
    print(f"{'Mean IoU (present)':<24} {mean_iou:>12.4f} {'-':>12}     "
          f"localization")
    print(f"{'State Accuracy':<24} {state_acc:>12.4f} {'-':>12}     "
          f"Anti-UAV benchmark")
    # Detection AP (single column — uses the ranking score noted below)
    print(f"{'AP@0.5':<24} {ap50:>12.4f} {'-':>12}     higher better")
    print(f"{'mAP@[0.5:0.95]':<24} {map5095:>12.4f} {'-':>12}     "
          f"{ap_score.size} dets / {total_ap_n_gt} GT")
    # Calibration / belief-quality (two signals)
    print(f"{'ECE':<24} {ece_prob:>12.4f} {ece_1mu:>12.4f}     lower better")
    print(f"{'AUROC':<24} {auroc_prob:>12.4f} {auroc_1mu:>12.4f}     higher better")
    print(f"{'AUSE':<24} {sp_prob['ause']:>12.4f} {sp_1mu['ause']:>12.4f}     "
          f"lower better")
    print(f"{'AUSE (sigma2)':<24} {sp_sigma['ause']:>12.4f} {'-':>12}     "
          f"lower better (localisation)")
    print("-" * width)
    # Belief-based classification (single column — uses conf_prob only)
    print(f"{'Confusion (TP/FP/TN/FN)':<24} {tp:>5d}/{fp:<3d} {tn:>4d}/{fn:<3d}  "
          f"     belief argmax")
    print(f"{'Accuracy':<24} {cls_acc:>12.4f} {'-':>12}     (TP+TN)/total")
    print(f"{'Precision':<24} {precision:>12.4f} {'-':>12}     TP/(TP+FP)")
    print(f"{'Recall (sensitivity)':<24} {recall:>12.4f} {'-':>12}     TP/(TP+FN)")
    print(f"{'Specificity':<24} {specificity:>12.4f} {'-':>12}     TN/(TN+FP)")
    print(f"{'False alarm rate':<24} {far:>12.4f} {'-':>12}     FP/(FP+TN)")
    print(f"{'F1':<24} {f1:>12.4f} {'-':>12}     harmonic P,R")
    print(f"{'Balanced accuracy':<24} {bal_acc:>12.4f} {'-':>12}     (R+Spec)/2")
    print("=" * width)
    print(f"Summary -> {out_path}")


if __name__ == "__main__":
    main()
