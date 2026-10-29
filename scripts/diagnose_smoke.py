#!/usr/bin/env python
"""
diagnose_smoke.py
-----------------
After running smoke_edtc_fusion.job, inspect what the tracker actually
predicted frame-by-frame against the IR ground-truth.

Answers the key questions:
  1. "Why is Acc negative?"
  2. "Was IR SiamCAR actually initialised and running?"
  3. "Which modality does DMS select most often, and does its choice make sense?"

It prints:
  - Frame-0 detail (YOLO init frame — crucial for init diagnosis)
  - Modality analysis: dms_winner distribution, ir_bbox/rgb_bbox presence
  - Per-frame IoU for the first N visible frames
  - Summary statistics (mean IoU, fraction > 0.5, etc.)
  - Top-5 best predictions (check if *any* frame has overlap)
  - Centroid displacement: how far is the predicted centre from the GT centre?

Usage (Snellius):
    # DMS-selected predictions vs IR GT (default, uses <seq>.txt):
    python scripts/diagnose_smoke.py \\
        --pred_dir  /home/ssuttorp/pretrained_models/edtc_fusion/smoke \\
        --data_root /home/ssuttorp/Anti-UAV-RGBT/val \\
        --sequence  20190925_101846_1_4 \\
        --log_csv   /home/ssuttorp/pretrained_models/edtc_fusion/smoke/20190925_101846_1_4_log.csv

    # IR-only predictions vs IR GT (uses <seq>_ir.txt):
    python scripts/diagnose_smoke.py \\
        --pred_dir  /home/ssuttorp/pretrained_models/edtc_fusion/smoke \\
        --data_root /home/ssuttorp/Anti-UAV-RGBT/val \\
        --sequence  20190925_101846_1_4 \\
        --pred_suffix _ir \\
        --log_csv   /home/ssuttorp/pretrained_models/edtc_fusion/smoke/20190925_101846_1_4_log.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def iou_xywh(pred, gt):
    px, py, pw, ph = pred
    gx, gy, gw, gh = gt
    if pw <= 0 or ph <= 0 or gw <= 0 or gh <= 0:
        return 0.0
    ix1 = max(px, gx)
    iy1 = max(py, gy)
    ix2 = min(px + pw, gx + gw)
    iy2 = min(py + ph, gy + gh)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = pw * ph + gw * gh - inter
    return inter / (union + 1e-8)


def centroid_dist(pred, gt):
    cx_p = pred[0] + pred[2] / 2.0
    cy_p = pred[1] + pred[3] / 2.0
    cx_g = gt[0]   + gt[2]   / 2.0
    cy_g = gt[1]   + gt[3]   / 2.0
    return ((cx_p - cx_g) ** 2 + (cy_p - cy_g) ** 2) ** 0.5


def bbox_str(b):
    if b is None:
        return "       None       "
    return "[{:.0f}, {:.0f}, {:.0f}, {:.0f}]".format(*b)


def parse_bbox_field(s):
    """Parse a JSON-encoded bbox string from the CSV (e.g. '[12.0, 34.0, 5.0, 6.0]')."""
    if not s or s.strip() == "":
        return None
    try:
        vals = json.loads(s)
        return [float(v) for v in vals]
    except (ValueError, TypeError, json.JSONDecodeError):
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Diagnose smoke-test predictions vs GT")
    p.add_argument("--pred_dir",    required=True, help="Directory with <seq>[suffix].txt files")
    p.add_argument("--data_root",   required=True, help="Val split root (e.g. Anti-UAV-RGBT/val)")
    p.add_argument("--sequence",    required=True, help="Sequence name")
    p.add_argument("--pred_suffix", default="",
                   help="Suffix to append to sequence name when loading predictions "
                        "(e.g. '_ir' → loads <seq>_ir.txt).  Default: '' → <seq>.txt")
    p.add_argument("--log_csv",     default=None,  help="Per-frame log CSV from the run")
    p.add_argument("--show_n",      type=int, default=20,
                   help="How many present frames to print in detail (default 20)")
    args = p.parse_args()

    # ── Load predictions ────────────────────────────────────────────────────
    pred_path = os.path.join(args.pred_dir, args.sequence + args.pred_suffix + ".txt")
    if not os.path.exists(pred_path):
        sys.exit(f"Prediction file not found: {pred_path}")
    preds = []
    with open(pred_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            vals = [float(v) for v in line.split(",")]
            preds.append(vals)

    # ── Load IR ground-truth ────────────────────────────────────────────────
    gt_path = os.path.join(args.data_root, args.sequence, "IR_label.json")
    if not os.path.exists(gt_path):
        sys.exit(f"GT file not found: {gt_path}")
    with open(gt_path) as f:
        ann = json.load(f)

    # ── Load RGB ground-truth (if available, for DMS winner context) ────────
    rgb_gt_path = os.path.join(args.data_root, args.sequence, "visible_label.json")
    rgb_ann = None
    if os.path.exists(rgb_gt_path):
        with open(rgb_gt_path) as f:
            rgb_ann = json.load(f)

    n = min(len(preds), len(ann["exist"]))

    # ── Load per-frame log ──────────────────────────────────────────────────
    frame_log = {}
    has_modality_cols = False
    if args.log_csv and os.path.exists(args.log_csv):
        with open(args.log_csv, newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
            has_modality_cols = "dms_winner" in fieldnames
            for row in reader:
                frame_log[int(row["frame"])] = row

    # ── Build per-frame table ────────────────────────────────────────────────
    present_rows = []   # (frame_idx, ir_gt, pred, iou, dist)
    absent_rows  = []

    for i in range(n):
        exist = int(ann["exist"][i])
        gt    = ann["gt_rect"][i] if exist else None
        pred  = preds[i]

        if gt is not None:
            iou  = iou_xywh(pred, gt)
            dist = centroid_dist(pred, gt)
            present_rows.append((i, gt, pred, iou, dist))
        else:
            absent_rows.append(i)

    # ── Print results ────────────────────────────────────────────────────────
    sep  = "=" * 110
    sep2 = "-" * 110

    print(sep)
    pred_label = args.sequence + args.pred_suffix + ".txt"
    print(f"  Prediction file : {pred_label}")
    print(f"  Sequence        : {args.sequence}")
    print(f"  Frames          : {n}   (IR GT present: {len(present_rows)}, absent: {len(absent_rows)})")
    print(sep)

    # ── Modality analysis (new — requires log CSV with dms_winner column) ────
    if frame_log and has_modality_cols:
        print("\n[MODALITY ANALYSIS — DMS winner & per-stream init status]\n")

        n_ir_win  = sum(1 for r in frame_log.values() if r.get("dms_winner") == "ir")
        n_rgb_win = sum(1 for r in frame_log.values() if r.get("dms_winner") == "rgb")
        n_none    = sum(1 for r in frame_log.values() if not r.get("dms_winner"))

        print(f"  DMS winner distribution over {len(frame_log)} frames:")
        print(f"    IR  wins : {n_ir_win:5d}  ({100*n_ir_win/max(len(frame_log),1):.1f}%)")
        print(f"    RGB wins : {n_rgb_win:5d}  ({100*n_rgb_win/max(len(frame_log),1):.1f}%)")
        print(f"    None     : {n_none:5d}  (DETECT_NONE frames)")

        # Check how often ir_bbox is non-None (IR tracker was actually running)
        n_ir_bbox  = sum(1 for r in frame_log.values()
                        if parse_bbox_field(r.get("ir_bbox","")) is not None)
        n_rgb_bbox = sum(1 for r in frame_log.values()
                        if parse_bbox_field(r.get("rgb_bbox","")) is not None)

        print()
        print(f"  Frames with non-null ir_bbox  (IR tracker running)  : {n_ir_bbox:5d} / {len(frame_log)}")
        print(f"  Frames with non-null rgb_bbox (RGB tracker running)  : {n_rgb_bbox:5d} / {len(frame_log)}")

        if n_ir_bbox == 0:
            print()
            print("  [!!] ir_bbox is NULL on every frame → IR SiamCAR was NEVER initialised.")
            print("       This means IR YOLO found no class-0 UAV detection on frame 0.")
            print("       Likely cause: evidential_ir model predicts UAV as class 1 (background),")
            print("       not class 0, so classes=[0] NMS filter eliminates all IR detections.")
            print("       Fix: test IR YOLO standalone with classes=None to see what class it outputs.")
        elif n_ir_bbox < len(frame_log) // 2:
            print()
            print(f"  [!] IR tracker running on only {n_ir_bbox}/{len(frame_log)} frames — partial init or drift.")

        if n_rgb_bbox == 0:
            print()
            print("  [!!] rgb_bbox is NULL on every frame → RGB SiamCAR was NEVER initialised.")

        # Frame-0 modality detail
        r0 = frame_log.get(0, {})
        ir0  = parse_bbox_field(r0.get("ir_bbox",  ""))
        rgb0 = parse_bbox_field(r0.get("rgb_bbox", ""))
        w0   = r0.get("dms_winner", "?")
        gt0  = ann["gt_rect"][0] if ann["exist"][0] else None
        rgb_gt0 = (rgb_ann["gt_rect"][0] if (rgb_ann and rgb_ann["exist"][0]) else None)

        print()
        print(f"  Frame-0 detail:")
        print(f"    IR  YOLO bbox : {bbox_str(ir0)}"
              + (f"  IR  GT: {bbox_str(gt0)}"
                 + (f"  IoU={iou_xywh(ir0, gt0):.3f}" if ir0 and gt0 else "") if gt0 else ""))
        print(f"    RGB YOLO bbox : {bbox_str(rgb0)}"
              + (f"  RGB GT: {bbox_str(rgb_gt0)}"
                 + (f"  IoU={iou_xywh(rgb0, rgb_gt0):.3f}" if rgb0 and rgb_gt0 else "") if rgb_gt0 else ""))
        print(f"    DMS winner    : {w0}")

        print()
    elif frame_log and not has_modality_cols:
        print("\n[NOTE] Log CSV found but has no dms_winner/ir_bbox/rgb_bbox columns.")
        print("       Re-run smoke test with the updated runner to get modality analysis.\n")

    # ── Frame-0 IoU against IR GT ────────────────────────────────────────────
    print("\n[FRAME 0 — prediction vs IR GT]\n")
    gt0   = ann["gt_rect"][0] if ann["exist"][0] else None
    pred0 = preds[0] if preds else None
    if gt0 and pred0:
        iou0  = iou_xywh(pred0, gt0)
        dist0 = centroid_dist(pred0, gt0)
        print(f"  IR GT  : {bbox_str(gt0)}")
        print(f"  Pred   : {bbox_str(pred0)}")
        print(f"  IoU    : {iou0:.4f}    centroid dist vs IR GT: {dist0:.1f} px")
        if iou0 == 0:
            print("  *** IoU = 0 on frame 0 → init box does NOT overlap with IR GT UAV ***")
    elif pred0:
        print(f"  Pred : {bbox_str(pred0)}   (IR GT: UAV absent on frame 0)")
    print()

    # ── Per-frame detail table ────────────────────────────────────────────────
    hdr = (f"{'Frame':>6}  {'State':>11}  {'Winner':>6}  "
           f"{'IR GT (x,y,w,h)':>22}  {'Pred (x,y,w,h)':>22}  "
           f"{'IoU':>6}  {'Dist':>7}")
    if frame_log:
        hdr += f"  {'u_fused':>8}  {'σ²_best':>8}"
    print(hdr)
    print(sep2)

    shown = 0
    for (i, gt, pred, iou, dist) in present_rows:
        if shown >= args.show_n:
            break
        r     = frame_log.get(i, {})
        state  = r.get("state", "?")
        winner = r.get("dms_winner", "?") if has_modality_cols else ""
        row = (f"{i:>6}  {state:>11}  {winner:>6}  "
               f"{bbox_str(gt):>22}  {bbox_str(pred):>22}  "
               f"{iou:>6.3f}  {dist:>7.1f}")
        if frame_log:
            uf = r.get("u_fused", "?")
            s2 = r.get("sigma2_best", "?")
            try:
                row += f"  {float(uf):>8.4f}  {float(s2):>8.4f}"
            except (ValueError, TypeError):
                row += f"  {'?':>8}  {'?':>8}"
        print(row)
        shown += 1

    if len(present_rows) > args.show_n:
        print(f"  ... ({len(present_rows) - args.show_n} more present frames not shown)")

    # ── Summary stats ─────────────────────────────────────────────────────────
    ious  = [r[3] for r in present_rows]
    dists = [r[4] for r in present_rows]
    print()
    print(sep)
    print("  IoU statistics (present frames, prediction vs IR GT)")
    print(sep)
    if ious:
        import statistics
        print(f"  Mean IoU    : {sum(ious)/len(ious):.4f}")
        print(f"  Median IoU  : {statistics.median(ious):.4f}")
        print(f"  Max IoU     : {max(ious):.4f}")
        print(f"  IoU > 0.0   : {sum(1 for x in ious if x > 0.0):4d} / {len(ious)}")
        print(f"  IoU > 0.2   : {sum(1 for x in ious if x > 0.2):4d} / {len(ious)}")
        print(f"  IoU > 0.5   : {sum(1 for x in ious if x > 0.5):4d} / {len(ious)}")
        print(f"  Mean centroid dist (vs IR GT): {sum(dists)/len(dists):.1f} px")
        print()
        best = sorted(present_rows, key=lambda r: -r[3])[:5]
        print("  Top-5 best-IoU present frames:")
        for (i, gt, pred, iou, dist) in best:
            print(f"    frame {i:4d}  GT={bbox_str(gt)}  pred={bbox_str(pred)}  "
                  f"IoU={iou:.4f}  dist={dist:.1f}px")
    else:
        print("  (no present frames with valid GT found)")

    # ── Diagnosis checklist ───────────────────────────────────────────────────
    print()
    print(sep)
    print("  Diagnosis checklist")
    print(sep)
    if ious:
        if max(ious) == 0.0:
            print("  [!] Max IoU = 0.0  →  predictions never touch IR GT region")
            print("      → Run with --pred_suffix _ir to check IR-only tracker output")
            print("      → Check Modality Analysis above: if ir_bbox=NULL throughout,")
            print("        IR YOLO never fired class-0 → UAV labelled as class 1 in model")
            if dists:
                md = sum(dists) / len(dists)
                print(f"      Average centroid distance = {md:.1f} px  "
                      f"(if ~400–600 px, coords are in wrong space; if <100 px, slight drift)")
        elif max(ious) > 0.5:
            print("  [OK] Some frames have IoU > 0.5 — predictions partially correct")
        elif sum(1 for x in ious if x > 0.2) < len(ious) * 0.1:
            print("  [!] < 10% of frames have IoU > 0.2  →  tracker drifts quickly after init")
        print()
        if len(present_rows) > 1:
            first_iou = present_rows[0][3]
            print(f"  Frame-0 IoU = {first_iou:.4f}  (YOLO init box quality vs IR GT)")
            if first_iou == 0:
                print("  → YOLO detected wrong object or wrong class; SiamCAR init has NO overlap with UAV")
                print("  → If dms_winner='rgb' on frame 0, IR YOLO returned None and RGB-space")
                print("    coords are being evaluated against IR GT (expected IoU=0)")
            elif first_iou > 0.3:
                print("  → YOLO init looks reasonable; drift is in SiamCAR tracking")
    print()


if __name__ == "__main__":
    main()
