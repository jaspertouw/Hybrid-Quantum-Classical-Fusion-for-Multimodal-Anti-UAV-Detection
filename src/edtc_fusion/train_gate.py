"""
train_gate.py — Training script for the Gated Fusion gate MLP
=============================================================
Trains GateFusionModule (~100 params) with all YOLO and SiamCAR backbone
models frozen.  Only the gate parameters are updated.

Strategy: pre-extract + batched training
-----------------------------------------
Running YOLO frame-by-frame for 20 epochs is prohibitively slow (~7 hrs/epoch).
Instead:

  Phase 1 — Feature extraction  (runs ONCE per sequence, ~6–8 hrs total):
      Run both frozen YOLOs on every frame  → cache (b_ir, b_rgb, label) tensors
      Run both frozen SiamCAR on each sequence → cache tracking belief tensors
      Each sequence saved to --cache_dir as a .npz file immediately after
      extraction.  On restart, already-cached sequences are skipped so the
      job can resume from where it was cancelled.

  Phase 2 — Gate training  (batched, 20 epochs, < 1 min total):
      Build a simple TensorDataset from the cached tensors.
      Train with DataLoader(batch_size=256) — no backbone inference here.

No switching thresholds are required for training.

Usage (Snellius)::

    python -m src.edtc_fusion.train_gate \\
        --train_root /home/ssuttorp/Anti-UAV-RGBT/train \\
        --val_root   /home/ssuttorp/Anti-UAV-RGBT/val   \\
        --yolo_ir   /home/ssuttorp/pretrained_models/evidential/evidential_ir_stage2_best.pt  \\
        --yolo_rgb  /home/ssuttorp/pretrained_models/evidential/evidential_rgb_stage2_best.pt \\
        --yolo_ir_base  /home/ssuttorp/pretrained_models/best_clean.pt     \\
        --yolo_rgb_base /home/ssuttorp/pretrained_models/best_clean_rgb.pt \\
        --siam_ir_stage2  /home/ssuttorp/pretrained_models/siamcar_gaussian/siamcar_ir_stage2_best.pth  \\
        --siam_rgb_stage2 /home/ssuttorp/pretrained_models/siamcar_gaussian/siamcar_rgb_stage2_best.pth \\
        --siam_ir_stage1  /home/ssuttorp/pretrained_models/siamcar_gaussian/siamcar_ir_stage1g_best.pth \\
        --siam_rgb_stage1 /home/ssuttorp/pretrained_models/siamcar_gaussian/siamcar_rgb_stage1g_best.pth \\
        --siam_ir_pretrained  /home/ssuttorp/pretrained_models/siamcar_thermal_final.pth \\
        --siam_rgb_pretrained /home/ssuttorp/pretrained_models/siamcar_rgb_final.pth     \\
        --siam_ir_cfg  resources/SiamCAR/experiments/siamcar_r50/config_antiuav_thermal.yaml \\
        --siam_rgb_cfg resources/SiamCAR/experiments/siamcar_r50/config_antiuav_rgb.yaml     \\
        --gate_out /home/ssuttorp/pretrained_models/gated_fusion/gate_best.pth \\
        --epochs 20 --lr 1e-3 --batch_size 256
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.edtc_fusion.tracker_gated_fusion import GateFusionModule
from src.edtc_fusion.tracker import (
    FusionConfig,
    _yolo_run,
    _EvidentialSiamCARTracker,
    _read_hp,
)
from src.edtc_fusion.run_edtc_fusion import load_paired_sequence, discover_sequences

_EDTC_ROOT    = _REPO_ROOT / "resources" / "EDTC"
_SIAMCAR_ROOT = _REPO_ROOT / "resources" / "SiamCAR"
for _p in (str(_EDTC_ROOT), str(_EDTC_ROOT / "yolov5"), str(_SIAMCAR_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from lib.models.yolo_evidential.model import EvidentialYOLO
from yolov5.utils.general import check_img_size
from pysot.core.config import cfg as siam_cfg
from pysot.models.siamcar_gaussian import SiamCARStage2

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_gate")


# ===========================================================================
# Phase 1: feature extraction (runs once per split)
# ===========================================================================

def _seq_cache_path(cache_dir: str, seq_name: str) -> str:
    return os.path.join(cache_dir, f"{seq_name}.npz")


@torch.no_grad()
def extract_beliefs(
    sequences, dataset_root,
    yolo_ir, yolo_rgb,
    siam_ir_model, siam_rgb_model,
    hp_ir, hp_rgb,
    cfg_obj, device,
    ir_imgsz, rgb_imgsz, ir_stride, rgb_stride,
    cache_dir: str | None = None,
    log_every: int = 10,
    split_name: str = "",
):
    """Extract (b_rgb, b_ir, label) tuples for every usable frame.

    Each sequence is saved to cache_dir/<seq_name>.npz immediately after
    extraction.  On restart, already-cached sequences are loaded from disk
    and skipped — extraction resumes from the first missing sequence.

    Returns two TensorDatasets — one for detection-phase examples (YOLO
    beliefs), one for tracking-phase examples (SiamCAR beliefs).
    Both share the schema: columns [b_rgb_0, b_rgb_1, b_ir_0, b_ir_1, label].
    """
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)

    det_rows = []
    trk_rows = []
    n_seq    = len(sequences)
    n_cached = 0

    hp_ir_d  = {"lr": hp_ir[0],  "penalty_k": hp_ir[1],  "window_lr": hp_ir[2]}
    hp_rgb_d = {"lr": hp_rgb[0], "penalty_k": hp_rgb[1], "window_lr": hp_rgb[2]}

    for s_idx, seq_name in enumerate(sequences):
        cache_path = _seq_cache_path(cache_dir, seq_name) if cache_dir else None

        # ── Load from cache if available ──────────────────────────────────
        if cache_path and os.path.exists(cache_path):
            data = np.load(cache_path)
            det_rows.extend(data["det"].tolist())
            trk_rows.extend(data["trk"].tolist())
            n_cached += 1
            if (s_idx + 1) % log_every == 0 or (s_idx + 1) == n_seq:
                logger.info(
                    "  [extract %s %3d/%3d]  seq=%-32s  CACHED  "
                    "det_total=%6d  trk_total=%6d",
                    split_name, s_idx + 1, n_seq, seq_name,
                    len(det_rows), len(trk_rows),
                )
            continue

        # ── Extract from frames ───────────────────────────────────────────
        frames   = load_paired_sequence(os.path.join(dataset_root, seq_name))
        seq_det  = []
        seq_trk  = []

        # Detection phase: YOLO on every frame
        for fr in frames:
            ir_bgr  = cv2.imread(fr["ir"])
            rgb_bgr = cv2.imread(fr["rgb"])
            if ir_bgr is None or rgb_bgr is None:
                continue
            _, ir_unc  = _yolo_run(ir_bgr,  yolo_ir,  cfg_obj, device, ir_imgsz,  ir_stride)
            _, rgb_unc = _yolo_run(rgb_bgr, yolo_rgb, cfg_obj, device, rgb_imgsz, rgb_stride)
            label = 1.0 if fr["gt"] is not None else 0.0
            seq_det.append([
                rgb_unc["b"][0].item(), rgb_unc["b"][1].item(),
                ir_unc["b"][0].item(),  ir_unc["b"][1].item(),
                label,
            ])

        # Tracking phase: SiamCAR after first joint YOLO detection
        siam_ir  = _EvidentialSiamCARTracker(siam_ir_model,  siam_cfg.TRACK)
        siam_rgb = _EvidentialSiamCARTracker(siam_rgb_model, siam_cfg.TRACK)
        initialized = False

        for fr in frames:
            ir_bgr  = cv2.imread(fr["ir"])
            rgb_bgr = cv2.imread(fr["rgb"])
            if ir_bgr is None or rgb_bgr is None:
                continue
            if not initialized:
                ir_box,  _ = _yolo_run(ir_bgr,  yolo_ir,  cfg_obj, device, ir_imgsz,  ir_stride)
                rgb_box, _ = _yolo_run(rgb_bgr, yolo_rgb, cfg_obj, device, rgb_imgsz, rgb_stride)
                if ir_box is not None and rgb_box is not None:
                    siam_ir.init(ir_bgr,   ir_box)
                    siam_rgb.init(rgb_bgr, rgb_box)
                    initialized = True
                continue
            ir_out  = siam_ir.track(ir_bgr,   hp_ir_d)
            rgb_out = siam_rgb.track(rgb_bgr, hp_rgb_d)
            label = 1.0 if fr["gt"] is not None else 0.0
            seq_trk.append([
                rgb_out["b"][0].item(), rgb_out["b"][1].item(),
                ir_out["b"][0].item(),  ir_out["b"][1].item(),
                label,
            ])

        # Save this sequence to disk immediately
        if cache_path:
            np.savez_compressed(
                cache_path,
                det=np.array(seq_det, dtype=np.float32),
                trk=np.array(seq_trk, dtype=np.float32),
            )

        det_rows.extend(seq_det)
        trk_rows.extend(seq_trk)

        if (s_idx + 1) % log_every == 0 or (s_idx + 1) == n_seq:
            logger.info(
                "  [extract %s %3d/%3d]  seq=%-32s  det_seq=%4d  trk_seq=%4d  "
                "det_total=%6d  trk_total=%6d  trk_init=%s",
                split_name, s_idx + 1, n_seq, seq_name,
                len(seq_det), len(seq_trk),
                len(det_rows), len(trk_rows),
                "yes" if initialized else "no",
            )

    if n_cached:
        logger.info("  [extract %s]  %d/%d sequences loaded from cache.", split_name, n_cached, n_seq)

    def to_dataset(rows):
        if not rows:
            return None
        t = torch.tensor(rows, dtype=torch.float32)
        return TensorDataset(t[:, :4], t[:, 4])

    return to_dataset(det_rows), to_dataset(trk_rows)


# ===========================================================================
# Phase 2: batched gate training on cached features
# ===========================================================================

def train_on_cache(
    gate, optimizer,
    det_train, trk_train,
    det_val, trk_val,
    epochs, batch_size,
    gate_out,
):
    """Train the gate MLP on pre-extracted belief tensors.

    Both detection and tracking datasets are concatenated into a single
    training set so the gate sees both sources of examples per epoch.
    """
    # Combine detection + tracking examples into one dataset
    parts = [d for d in [det_train, trk_train] if d is not None]
    if not parts:
        raise RuntimeError("No training examples extracted — check dataset paths.")
    from torch.utils.data import ConcatDataset
    train_ds = ConcatDataset(parts)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)

    val_parts = [d for d in [det_val, trk_val] if d is not None]
    val_loader = None
    if val_parts:
        from torch.utils.data import ConcatDataset as CD
        val_ds = CD(val_parts)
        val_loader = DataLoader(val_ds, batch_size=batch_size * 4, shuffle=False)

    # Report informative fraction (frames where at least one YOLO detects).
    # Both-miss frames (b_rgb[0]=b_ir[0]=0) inflate BCE but have zero gradient,
    # so we exclude them from loss/gradient — they don't affect gate weights.
    all_train_feats = torch.cat([ds[:][0] for ds in parts])
    n_informative   = ((all_train_feats[:, 0] + all_train_feats[:, 2]) > 1e-6).sum().item()
    logger.info(
        "Gate training: %d train examples  %d val examples  batch=%d  epochs=%d  "
        "informative_train=%d (%.1f%%)",
        len(train_ds),
        sum(len(d) for d in val_parts) if val_parts else 0,
        batch_size, epochs,
        n_informative, 100.0 * n_informative / max(len(train_ds), 1),
    )

    best_val_loss = float("inf")
    best_epoch    = -1

    for epoch in range(1, epochs + 1):
        gate.train()
        total_loss = 0.0
        n = 0
        for feats, labels in train_loader:
            feats  = feats.to(next(gate.parameters()).device)
            labels = labels.to(feats.device)
            b_rgb = feats[:, :2]
            b_ir  = feats[:, 2:]
            # Exclude both-miss frames: when YOLO returns _ZERO_UNC (b[0]=0)
            # for *both* modalities, p_fused=0 exactly → BCE→∞, gradient=0.
            # These frames inflate the loss metric without affecting gate weights.
            informative = (b_rgb[:, 0] + b_ir[:, 0]) > 1e-6
            if informative.sum() == 0:
                continue
            feats_i  = feats[informative]
            labels_i = labels[informative]
            b_rgb_i  = b_rgb[informative]
            b_ir_i   = b_ir[informative]
            g       = gate.detect_gate(feats_i).squeeze(-1)      # (B',)
            p_fused = g * b_rgb_i[:, 0] + (1.0 - g) * b_ir_i[:, 0]
            p_fused = p_fused.clamp(1e-6, 1 - 1e-6)
            loss = F.binary_cross_entropy(p_fused, labels_i)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * informative.sum().item()
            n += informative.sum().item()
        train_loss = total_loss / max(n, 1)

        val_loss = float("nan")
        if val_loader is not None:
            gate.eval()
            vl, vn = 0.0, 0
            with torch.no_grad():
                for feats, labels in val_loader:
                    feats  = feats.to(next(gate.parameters()).device)
                    labels = labels.to(feats.device)
                    b_rgb = feats[:, :2]
                    b_ir  = feats[:, 2:]
                    informative = (b_rgb[:, 0] + b_ir[:, 0]) > 1e-6
                    if informative.sum() == 0:
                        continue
                    feats_i  = feats[informative]
                    labels_i = labels[informative]
                    b_rgb_i  = b_rgb[informative]
                    b_ir_i   = b_ir[informative]
                    g       = gate.detect_gate(feats_i).squeeze(-1)
                    p_fused = (g * b_rgb_i[:, 0] + (1.0 - g) * b_ir_i[:, 0]).clamp(1e-6, 1 - 1e-6)
                    vl += F.binary_cross_entropy(p_fused, labels_i).item() * informative.sum().item()
                    vn += informative.sum().item()
            val_loss = vl / max(vn, 1)

        logger.info(
            "Epoch %2d/%2d  train_loss=%.4f  val_loss=%.4f",
            epoch, epochs,
            train_loss,
            val_loss if not math.isnan(val_loss) else -1,
        )

        monitor = val_loss if (val_loader and not math.isnan(val_loss)) else train_loss
        if monitor < best_val_loss:
            best_val_loss = monitor
            best_epoch    = epoch
            gate.save(gate_out)
            logger.info("  → saved best checkpoint (epoch %d, loss=%.4f)", epoch, monitor)

    logger.info("Done.  Best epoch=%d  loss=%.4f  checkpoint=%s",
                best_epoch, best_val_loss, gate_out)


# ===========================================================================
# Main
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Train gate MLP for Gated Fusion baseline")

    p.add_argument("--train_root", required=True)
    p.add_argument("--val_root",   default=None)
    p.add_argument("--gate_out",   required=True)
    p.add_argument("--cache_dir",  default=None,
                   help="Directory to save/load per-sequence .npz belief caches. "
                        "Strongly recommended — allows resuming after job cancellation.")
    p.add_argument("--epochs",     type=int,   default=20)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--batch_size", type=int,   default=256)
    p.add_argument("--K",          type=int,   default=2)
    p.add_argument("--hidden",     type=int,   default=16)
    p.add_argument("--log_every",  type=int,   default=10,
                   help="Log progress every N sequences during extraction.")

    p.add_argument("--yolo_ir",   required=True)
    p.add_argument("--yolo_rgb",  required=True)
    p.add_argument("--yolo_ir_base",  required=True)
    p.add_argument("--yolo_rgb_base", required=True)

    p.add_argument("--siam_ir_stage2",  required=True)
    p.add_argument("--siam_rgb_stage2", required=True)
    p.add_argument("--siam_ir_stage1",  required=True)
    p.add_argument("--siam_rgb_stage1", required=True)
    p.add_argument("--siam_ir_pretrained",  required=True)
    p.add_argument("--siam_rgb_pretrained", required=True)
    p.add_argument("--siam_ir_cfg",  required=True)
    p.add_argument("--siam_rgb_cfg", required=True)

    p.add_argument("--yolo_imgsz", type=int,   default=640)
    p.add_argument("--yolo_conf",  type=float, default=0.25)
    p.add_argument("--yolo_iou",   type=float, default=0.45)

    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    cfg_obj = FusionConfig(
        yolo_imgsz=(args.yolo_imgsz, args.yolo_imgsz),
        yolo_conf_thres=args.yolo_conf,
        yolo_iou_thres=args.yolo_iou,
        K=args.K,
    )

    logger.info("Loading YOLO models (frozen)...")
    yolo_ir  = EvidentialYOLO.load(args.yolo_ir,  args.yolo_ir_base,  device, K=args.K).eval()
    yolo_rgb = EvidentialYOLO.load(args.yolo_rgb, args.yolo_rgb_base, device, K=args.K).eval()
    ir_stride  = int(yolo_ir.stride.max())
    rgb_stride = int(yolo_rgb.stride.max())
    ir_imgsz   = check_img_size((args.yolo_imgsz, args.yolo_imgsz), s=ir_stride)
    rgb_imgsz  = check_img_size((args.yolo_imgsz, args.yolo_imgsz), s=rgb_stride)
    for param in list(yolo_ir.parameters()) + list(yolo_rgb.parameters()):
        param.requires_grad_(False)

    logger.info("Loading SiamCAR models (frozen)...")
    siam_cfg.merge_from_file(args.siam_ir_cfg)
    siam_ir_model  = SiamCARStage2.load(
        args.siam_ir_stage2,
        stage1_path=args.siam_ir_stage1,
        pretrained_path=args.siam_ir_pretrained,
        K=args.K,
    ).to(device).eval()
    siam_rgb_model = SiamCARStage2.load(
        args.siam_rgb_stage2,
        stage1_path=args.siam_rgb_stage1,
        pretrained_path=args.siam_rgb_pretrained,
        K=args.K,
    ).to(device).eval()
    for param in list(siam_ir_model.parameters()) + list(siam_rgb_model.parameters()):
        param.requires_grad_(False)

    hp_ir  = _read_hp(args.siam_ir_cfg,  "ANTIUAVTHERMAL")
    hp_rgb = _read_hp(args.siam_rgb_cfg, "ANTIUAVRGB")

    cache_train = os.path.join(args.cache_dir, "train") if args.cache_dir else None
    cache_val   = os.path.join(args.cache_dir, "val")   if args.cache_dir else None

    extract_kwargs = dict(
        yolo_ir=yolo_ir, yolo_rgb=yolo_rgb,
        siam_ir_model=siam_ir_model, siam_rgb_model=siam_rgb_model,
        hp_ir=hp_ir, hp_rgb=hp_rgb,
        cfg_obj=cfg_obj, device=device,
        ir_imgsz=ir_imgsz, rgb_imgsz=rgb_imgsz,
        ir_stride=ir_stride, rgb_stride=rgb_stride,
        log_every=args.log_every,
    )

    # ── Phase 1: extract ─────────────────────────────────────────────────
    train_seqs = discover_sequences(args.train_root)
    logger.info("=== Phase 1: extracting train beliefs (%d sequences) ===", len(train_seqs))
    if cache_train:
        logger.info("    cache_dir: %s  (already-cached sequences will be skipped)", cache_train)
    det_train, trk_train = extract_beliefs(
        train_seqs, args.train_root, split_name="train",
        cache_dir=cache_train, **extract_kwargs,
    )

    det_val, trk_val = None, None
    if args.val_root:
        val_seqs = discover_sequences(args.val_root)
        logger.info("=== Phase 1: extracting val beliefs (%d sequences) ===", len(val_seqs))
        if cache_val:
            logger.info("    cache_dir: %s", cache_val)
        det_val, trk_val = extract_beliefs(
            val_seqs, args.val_root, split_name="val",
            cache_dir=cache_val, **extract_kwargs,
        )

    # ── Phase 2: train gate on cached features ───────────────────────────
    gate      = GateFusionModule(K=args.K, hidden=args.hidden).to(device)
    optimizer = torch.optim.Adam(gate.parameters(), lr=args.lr)
    os.makedirs(Path(args.gate_out).parent, exist_ok=True)

    logger.info("=== Phase 2: training gate MLP ===")
    train_on_cache(
        gate, optimizer,
        det_train, trk_train,
        det_val, trk_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        gate_out=args.gate_out,
    )


if __name__ == "__main__":
    main()
