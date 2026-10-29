"""
Stage 1 independent training for one EvidentialYOLO stream (Phase 3.3).

Two-phase training strategy:
  Phase A (warmup_epochs): backbone frozen, train only m_var + m_evi.
                           Gives the new branches a stable starting point.
  Phase B (remaining):     unfreeze all, full fine-tuning with val mAP tracking.

Run once for IR, once for RGB:

  # IR
  python lib/models/yolo_evidential/train_single_stream.py \
      --weights /home/ssuttorp/pretrained_models/best_clean.pt \
      --data    .../antiuav_clean.yaml \
      --train_txt .../train_ir_images.txt \
      --val_txt   .../val_ir_images.txt \
      --name evidential_ir_stage1 --device 0

  # RGB
  python lib/models/yolo_evidential/train_single_stream.py \
      --weights /home/ssuttorp/pretrained_models/best_clean_rgb.pt \
      --data    .../antiuav_clean_rgb.yaml \
      --train_txt .../train_rgb_images.txt \
      --val_txt   .../val_rgb_images.txt \
      --name evidential_rgb_stage1 --device 0
"""

import os
import sys
import csv
import time
import argparse
from pathlib import Path

import torch
import torch.optim as optim
import numpy as np

_edtc_root = Path(__file__).resolve().parents[3]
_yolo_root = _edtc_root / 'yolov5'
for _p in [str(_edtc_root), str(_yolo_root)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from yolov5.utils.torch_utils import select_device
from yolov5.utils.dataloaders import create_dataloader
from yolov5.utils.general import check_img_size, colorstr, non_max_suppression, xywh2xyxy
from yolov5.utils.metrics import ap_per_class, box_iou

from lib.models.yolo_evidential.model  import EvidentialYOLO
from lib.models.yolo_evidential.losses import ComputeEvidentialLoss

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_HYP = {
    # Loss weights
    'box': 0.05, 'obj': 1.0, 'cls': 0.5,
    'anchor_t': 4.0, 'fl_gamma': 0.0,
    'cls_pw': 1.0, 'obj_pw': 1.0,
    'label_smoothing': 0.0,
    # Augmentation — required by create_dataloader when augment=True
    'hsv_h': 0.015, 'hsv_s': 0.7, 'hsv_v': 0.4,
    'degrees': 0.0, 'translate': 0.1, 'scale': 0.5,
    'shear': 0.0, 'perspective': 0.0,
    'flipud': 0.0, 'fliplr': 0.5,
    'mosaic': 1.0, 'mixup': 0.0, 'copy_paste': 0.0,
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def build_loaders(train_txt, val_txt, data_yaml, imgsz, batch_size, stride,
                  workers=8, hyp=None):
    """Create train and val dataloaders.

    Args:
        hyp: optional hyperparameter dict override.  Merged on top of
             DEFAULT_HYP so you only need to supply keys you want to change
             (e.g. ``{'mosaic': 0.0}`` to disable mosaic augmentation).
    """
    gs   = max(int(stride), 32)
    imgsz = check_img_size(imgsz, s=gs)

    effective_hyp = dict(DEFAULT_HYP)
    if hyp:
        effective_hyp.update(hyp)

    train_loader, train_dataset = create_dataloader(
        train_txt, imgsz=imgsz, batch_size=batch_size, stride=gs,
        single_cls=True, hyp=effective_hyp, augment=True,
        cache=False, rect=False, rank=-1, workers=workers,
        image_weights=False, quad=False,
        prefix=colorstr('train: '), shuffle=True,
    )
    val_loader, _ = create_dataloader(
        val_txt, imgsz=imgsz, batch_size=batch_size * 2, stride=gs,
        single_cls=True, hyp=effective_hyp, augment=False,
        cache=False, rect=True, rank=-1, workers=workers,
        image_weights=False, quad=False,
        prefix=colorstr('val: '), shuffle=False,
    )
    return train_loader, val_loader, imgsz


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate(model, val_loader, device, conf_thres=0.25, iou_thres=0.45, K=2):
    """Compute mAP50 + evidential health metrics on the validation set.

    Returns dict with keys: mAP50, mAP5095, u_mean, b_uav_mean.
    """
    model.eval()
    iouv   = torch.linspace(0.5, 0.95, 10, device=device)
    niou   = iouv.numel()
    stats  = []
    u_vals = []
    b_vals = []

    with torch.no_grad():
        for imgs, targets, paths, shapes in val_loader:
            imgs    = imgs.to(device).float() / 255.0
            targets = targets.to(device)
            nb, _, h, w = imgs.shape

            # Forward — inference mode returns tuple
            out = model(imgs)
            if isinstance(out, tuple) and len(out) >= 5:
                nms_preds, _, _, _, unc_scales = out
                # Collect uncertainty stats
                for s in unc_scales:
                    u_vals.append(s['u'].mean().item())
                    b_vals.append(s['b'][..., 0].mean().item())  # b_UAV
            else:
                nms_preds = out[0] if isinstance(out, tuple) else out

            # NMS
            preds = non_max_suppression(
                nms_preds, conf_thres, iou_thres,
                classes=None, agnostic=False, max_det=1000,
            )

            # Accumulate per-image stats
            for si, pred in enumerate(preds):
                # Ground-truth for this image
                gt_mask = targets[:, 0] == si
                gt      = targets[gt_mask, 1:]          # (M, 5) cls cx cy w h (norm)
                nl      = gt.shape[0]
                cls_gt  = gt[:, 0].long() if nl else torch.tensor([], device=device)

                # Convert norm cx,cy,w,h → absolute xyxy in padded-image space
                if nl:
                    gt_box = xywh2xyxy(gt[:, 1:5] * torch.tensor(
                        [w, h, w, h], device=device
                    ))

                correct = torch.zeros(
                    pred.shape[0], niou, dtype=torch.bool, device=device
                )
                if pred.shape[0]:
                    if nl:
                        iou_mat = box_iou(gt_box, pred[:, :4])  # (M, N)
                        # Greedy matching: for each GT, find best pred above iou threshold
                        for j in range(niou):
                            thresh = iouv[j]
                            matched_gt  = set()
                            matched_pred = set()
                            for gt_idx in range(nl):
                                best_iou, best_pred = -1, -1
                                for pred_idx in range(pred.shape[0]):
                                    if pred_idx in matched_pred:
                                        continue
                                    if iou_mat[gt_idx, pred_idx] > best_iou:
                                        best_iou  = iou_mat[gt_idx, pred_idx].item()
                                        best_pred = pred_idx
                                if best_iou >= thresh and best_pred >= 0:
                                    matched_gt.add(gt_idx)
                                    matched_pred.add(best_pred)
                                    correct[best_pred, j] = True

                stats.append((
                    correct.cpu(),
                    pred[:, 4].cpu(),
                    pred[:, 5].cpu(),
                    cls_gt.cpu(),
                ))

    # mAP
    if not stats:
        return {'mAP50': 0.0, 'mAP5095': 0.0, 'u_mean': float('nan'), 'b_uav_mean': float('nan')}

    # Concatenate
    tp_all   = torch.cat([s[0] for s in stats], 0).numpy()
    conf_all = torch.cat([s[1] for s in stats], 0).numpy()
    pcls_all = torch.cat([s[2] for s in stats], 0).numpy()
    tcls_all = torch.cat([s[3] for s in stats], 0).numpy()

    try:
        p, r, ap, f1, ap_class = ap_per_class(
            tp_all, conf_all, pcls_all, tcls_all,
            plot=False, save_dir='.', names={0: 'UAV'},
        )
        mAP50   = ap[:, 0].mean()       # AP at IoU=0.50
        mAP5095 = ap.mean()             # AP at IoU=0.50:0.95
    except Exception:
        mAP50 = mAP5095 = 0.0

    return {
        'mAP50':    float(mAP50),
        'mAP5095':  float(mAP5095),
        'u_mean':   float(np.mean(u_vals)) if u_vals else float('nan'),
        'b_uav_mean': float(np.mean(b_vals)) if b_vals else float('nan'),
    }


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, loss_fn, optimizer, device):
    """Train for one epoch. Returns dict of average loss components."""
    model.train()
    totals = {'loss': 0.0, 'box': 0.0, 'obj': 0.0, 'evi': 0.0, 'var': 0.0}
    nb = len(loader)

    for imgs, targets, *_ in loader:
        imgs    = imgs.to(device).float() / 255.0
        targets = targets.to(device)

        preds = model(imgs)

        # preds must be (base_raw, var_raw, evi_raw) in training mode
        if not (isinstance(preds, tuple) and len(preds) == 3):
            raise RuntimeError(
                f"Expected (base_raw, var_raw, evi_raw) from model in train mode. "
                f"Got: {type(preds)}, len={len(preds) if isinstance(preds, tuple) else 'N/A'}"
            )

        loss, components = loss_fn(preds, targets)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()

        bs = imgs.shape[0]
        totals['loss'] += loss.item() / bs
        totals['box']  += components[0].item()
        totals['obj']  += components[1].item()
        totals['evi']  += components[2].item()
        totals['var']  += components[3].item()

    return {k: v / nb for k, v in totals.items()}


# ---------------------------------------------------------------------------
# Main training orchestrator
# ---------------------------------------------------------------------------

def train(args):
    device = select_device(args.device)
    os.makedirs(args.save_dir, exist_ok=True)

    # ── Build model ───────────────────────────────────────────────────
    print(f'\n[Stage 1] Loading: {args.weights}')
    model = EvidentialYOLO.from_pretrained(args.weights, device, K=args.K)
    model.hyp = DEFAULT_HYP

    # ── Data loaders ─────────────────────────────────────────────────
    stride = int(model.stride.max())
    train_loader, val_loader, imgsz = build_loaders(
        args.train_txt, args.val_txt, args.data,
        args.imgsz, args.batch_size, stride,
        workers=args.workers,
    )
    print(f'[Stage 1] Train: {len(train_loader.dataset)} images | '
          f'Val: {len(val_loader.dataset)} images')

    # ── Loss function ─────────────────────────────────────────────────
    total_steps = args.epochs * len(train_loader)
    loss_fn = ComputeEvidentialLoss(
        model, DEFAULT_HYP,
        lambda_var=args.lambda_var,
        lambda_evi=args.lambda_evi,
        K=args.K,
    )
    loss_fn._anneal_steps = total_steps

    history = []
    best_map50 = -1.0

    # ── Phase A: warm up new branches (backbone frozen) ───────────────
    if args.warmup_epochs > 0:
        print(f'\n[Phase A] Warming up new branches for {args.warmup_epochs} epoch(s).')
        model.freeze_backbone()
        opt_a = optim.AdamW(model.new_params(), lr=args.lr_phase_a, weight_decay=1e-4)

        for epoch in range(args.warmup_epochs):
            t0   = time.time()
            comp = train_one_epoch(model, train_loader, loss_fn, opt_a, device)
            elapsed = time.time() - t0
            print(
                f'  A {epoch+1:3d}/{args.warmup_epochs}  '
                f'loss={comp["loss"]:.4f}  '
                f'[box={comp["box"]:.4f} obj={comp["obj"]:.4f} '
                f'evi={comp["evi"]:.4f} var={comp["var"]:.4f}]  '
                f'{elapsed:.1f}s'
            )
            history.append({
                'epoch': epoch + 1, 'phase': 'A',
                **comp,
                'mAP50': '', 'mAP5095': '', 'u_mean': '', 'b_uav_mean': '',
            })

    # ── Phase B: full fine-tuning with validation ─────────────────────
    phase_b_epochs = args.epochs - args.warmup_epochs
    if phase_b_epochs > 0:
        print(f'\n[Phase B] Full fine-tuning for {phase_b_epochs} epoch(s).')
        model.unfreeze_all()
        opt_b = optim.AdamW(
            model.parameters(), lr=args.lr_phase_b, weight_decay=1e-4
        )
        sched_b = optim.lr_scheduler.CosineAnnealingLR(
            opt_b, T_max=phase_b_epochs, eta_min=args.lr_phase_b * 0.01
        )

        for epoch in range(phase_b_epochs):
            t0   = time.time()
            comp = train_one_epoch(model, train_loader, loss_fn, opt_b, device)
            val_metrics = validate(
                model, val_loader, device,
                conf_thres=args.conf_thres,
                iou_thres=args.iou_thres,
                K=args.K,
            )
            sched_b.step()
            elapsed = time.time() - t0

            total_epoch = args.warmup_epochs + epoch + 1
            print(
                f'  B {total_epoch:3d}/{args.epochs}  '
                f'loss={comp["loss"]:.4f}  '
                f'mAP50={val_metrics["mAP50"]:.4f}  '
                f'u_mean={val_metrics["u_mean"]:.4f}  '
                f'{elapsed:.1f}s'
            )
            history.append({
                'epoch': total_epoch, 'phase': 'B',
                **comp, **val_metrics,
            })

            # Save best by mAP50
            if val_metrics['mAP50'] > best_map50:
                best_map50 = val_metrics['mAP50']
                best_path  = os.path.join(args.save_dir, f'{args.name}_best.pt')
                model.save(best_path)
                print(f'  → New best mAP50={best_map50:.4f} saved to {best_path}')

    # ── Save last + history ───────────────────────────────────────────
    last_path = os.path.join(args.save_dir, f'{args.name}_last.pt')
    model.save(last_path)
    _save_csv(history, os.path.join(args.save_dir, f'{args.name}_history.csv'))
    print(f'\n[Stage 1] Done. Best mAP50={best_map50:.4f}')


def _save_csv(history, path):
    if not history:
        return
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    print(f'[Stage 1] History → {path}')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Phase 3.3 Stage 1: independent EvidentialYOLO training.'
    )
    parser.add_argument('--weights',       required=True)
    parser.add_argument('--data',          required=True,
                        help='Dataset YAML (antiuav_clean.yaml or antiuav_clean_rgb.yaml).')
    parser.add_argument('--train_txt',     required=True)
    parser.add_argument('--val_txt',       required=True)
    parser.add_argument('--save_dir',      default='/home/ssuttorp/pretrained_models/evidential')
    parser.add_argument('--name',          default='evidential_stage1')
    parser.add_argument('--device',        default='0')
    parser.add_argument('--epochs',        type=int,   default=30)
    parser.add_argument('--warmup_epochs', type=int,   default=5,
                        help='Phase A duration (backbone frozen). Set 0 to skip.')
    parser.add_argument('--batch_size',    type=int,   default=64)
    parser.add_argument('--workers',       type=int,   default=8)
    parser.add_argument('--imgsz',         type=int,   default=640)
    parser.add_argument('--lr_phase_a',    type=float, default=1e-3)
    parser.add_argument('--lr_phase_b',    type=float, default=1e-4)
    parser.add_argument('--lambda_var',    type=float, default=0.1)
    parser.add_argument('--lambda_evi',    type=float, default=1.0)
    parser.add_argument('--K',             type=int,   default=2)
    parser.add_argument('--conf_thres',    type=float, default=0.25)
    parser.add_argument('--iou_thres',     type=float, default=0.45)
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
