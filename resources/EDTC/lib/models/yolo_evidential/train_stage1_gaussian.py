"""
Phase 3.3 Stage 1 (Gaussian) — train the Gaussian bbox head per stream.

Protocol
--------
Phase A (warmup_epochs, default 5):
  Backbone + neck frozen.  Only m_var trains (log-variance head).
  Optimizer: AdamW on model.gaussian_params() at lr_phase_a.
  No validation — just let the variance head find sensible initialisation.

Phase B (remaining epochs, default 25):
  m_evi frozen.  Backbone, neck, m (bbox+obj) and m_var all train together.
  Optimizer: AdamW on all requires_grad params at lr_phase_b.
  CosineAnnealingLR scheduler.
  Each epoch: validate mAP50 and check spatial variance health.
  Best checkpoint saved by mAP50.

The evidential head (m_evi) is NEVER touched during Stage 1.
Run Stage 2 (train_stage2_evidential.py) once Stage 1 is complete.

Usage
-----
IR stream:
  python lib/models/yolo_evidential/train_stage1_gaussian.py \\
      --weights  /home/ssuttorp/pretrained_models/best_clean.pt \\
      --data     .../antiuav_clean.yaml \\
      --train_txt .../train_ir_images.txt \\
      --val_txt   .../val_ir_images.txt \\
      --name evidential_ir_stage1g --device 0

RGB stream:
  python lib/models/yolo_evidential/train_stage1_gaussian.py \\
      --weights  /home/ssuttorp/pretrained_models/best_clean_rgb.pt \\
      --data     .../antiuav_clean_rgb.yaml \\
      --train_txt .../train_rgb_images.txt \\
      --val_txt   .../val_rgb_images.txt \\
      --name evidential_rgb_stage1g --device 0
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
from tqdm import tqdm

_edtc_root = Path(__file__).resolve().parents[3]
_yolo_root = _edtc_root / 'yolov5'
for _p in [str(_edtc_root), str(_yolo_root)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from yolov5.utils.torch_utils import select_device
from yolov5.utils.general     import init_seeds

# Reuse data/validation utilities from train_single_stream without modification
from lib.models.yolo_evidential.train_single_stream import (
    build_loaders, validate, DEFAULT_HYP,
)
from lib.models.yolo_evidential.model         import EvidentialYOLO
from lib.models.yolo_evidential.losses_staged import ComputeGaussianBboxLoss


# ---------------------------------------------------------------------------
# Training loop (Gaussian loss only)
# ---------------------------------------------------------------------------

def train_one_epoch_gaussian(model, loader, loss_fn, optimizer, device,
                             desc: str = 'train'):
    """One epoch with ComputeGaussianBboxLoss.

    Returns dict of average loss components: loss, box, obj, var.
    """
    model.train()
    totals = {'loss': 0.0, 'box': 0.0, 'obj': 0.0, 'var': 0.0}
    nb = len(loader)

    pbar = tqdm(loader, desc=desc, ncols=100, leave=False)
    for imgs, targets, *_ in pbar:
        imgs    = imgs.to(device).float() / 255.0
        targets = targets.to(device)

        preds = model(imgs)
        if not (isinstance(preds, tuple) and len(preds) == 3):
            raise RuntimeError(
                f'Expected (base_raw, var_raw, evi_raw) in train mode. '
                f'Got: {type(preds)}'
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
        totals['var']  += components[2].item()

        pbar.set_postfix(
            loss=f'{loss.item()/bs:.4f}',
            box=f'{components[0].item():.4f}',
            obj=f'{components[1].item():.4f}',
            var=f'{components[2].item():.4f}',
        )

    pbar.close()
    return {k: v / nb for k, v in totals.items()}


# ---------------------------------------------------------------------------
# Spatial variance health check
# ---------------------------------------------------------------------------

def check_variance_health(model, val_loader, device, max_images: int = 200):
    """Run a quick eval pass and report statistics on the normalised variances.

    Checks that sigma2_norm shows meaningful spread across frames.
    Warns if the distribution is degenerate (all uniform) or exploded.

    Args:
        model:       EvidentialYOLO in eval mode after the call.
        val_loader:  validation DataLoader.
        device:      torch.device.
        max_images:  maximum images to inspect (keep it fast).

    Returns:
        dict with keys: sigma2_mean, sigma2_std, sigma2_min, sigma2_max, healthy
    """
    model.eval()
    sigma2_vals = []
    n_seen = 0

    with torch.no_grad():
        for imgs, targets, *_ in val_loader:
            imgs = imgs.to(device).float() / 255.0
            out  = model(imgs)

            if isinstance(out, tuple) and len(out) >= 5:
                _, _, _, _, unc_scales = out
                for s in unc_scales:
                    sigma2_vals.append(s['sigma2_norm'].cpu().float())
            n_seen += imgs.shape[0]
            if n_seen >= max_images:
                break

    if not sigma2_vals:
        return {'sigma2_mean': float('nan'), 'sigma2_std': float('nan'),
                'sigma2_min': float('nan'), 'sigma2_max': float('nan'),
                'healthy': False}

    all_vals = torch.cat([v.reshape(-1) for v in sigma2_vals])
    s_mean = all_vals.mean().item()
    s_std  = all_vals.std().item()
    s_min  = all_vals.min().item()
    s_max  = all_vals.max().item()

    healthy = True
    msgs = []
    if s_std < 0.01:
        healthy = False
        msgs.append(
            f'[WARN] Variance is nearly constant (std={s_std:.4f} < 0.01). '
            'Gaussian head may be degenerate — check gradient flow to m_var.'
        )
    if s_mean > 20.0:
        healthy = False
        msgs.append(
            f'[WARN] Mean normalised variance is very large (mean={s_mean:.2f}). '
            'Consider reducing lambda_var or checking log-var clipping.'
        )
    if not msgs:
        msgs.append(
            f'[OK] Variance health: mean={s_mean:.4f}, std={s_std:.4f}, '
            f'min={s_min:.4f}, max={s_max:.4f}'
        )
    for m in msgs:
        print(f'  {m}')

    return {
        'sigma2_mean': s_mean,
        'sigma2_std':  s_std,
        'sigma2_min':  s_min,
        'sigma2_max':  s_max,
        'healthy':     healthy,
    }


# ---------------------------------------------------------------------------
# Main training orchestrator
# ---------------------------------------------------------------------------

def train(args):
    device = select_device(args.device)
    os.makedirs(args.save_dir, exist_ok=True)

    # deterministic=False: seed all RNGs (init of m_var/m_evi heads, data shuffling,
    # augmentation) for reproducible per-seed runs, but do not force
    # torch.use_deterministic_algorithms — some custom ops here lack a
    # deterministic kernel and would raise.
    init_seeds(args.seed, deterministic=False)
    print(f'[Stage 1g] Seed = {args.seed}')

    # ── Build model ────────────────────────────────────────────────────
    print(f'\n[Stage 1g] Loading pretrained YOLOv5: {args.weights}')
    model = EvidentialYOLO.from_pretrained(args.weights, device, K=args.K)
    model.hyp = DEFAULT_HYP

    # ── Data loaders ──────────────────────────────────────────────────
    stride = int(model.stride.max())
    # Disable mosaic augmentation for Stage 1: the variance head only needs
    # clean bbox supervision; mosaic loads 4× as many images per step and
    # would make each epoch ~4× slower on GPFS with no benefit here.
    train_loader, val_loader, imgsz = build_loaders(
        args.train_txt, args.val_txt, args.data,
        args.imgsz, args.batch_size, stride,
        workers=args.workers,
        hyp={'mosaic': 0.0},
    )
    print(f'[Stage 1g] Train: {len(train_loader.dataset)} images | '
          f'Val: {len(val_loader.dataset)} images | imgsz={imgsz}')

    # ── Loss function ─────────────────────────────────────────────────
    loss_fn = ComputeGaussianBboxLoss(model, DEFAULT_HYP, lambda_var=args.lambda_var)

    resume_path = os.path.join(args.save_dir, f'{args.name}_resume.pt')
    history     = []
    best_map50  = 0.0
    completed   = 0   # total epochs already done this run

    if args.resume and os.path.isfile(resume_path):
        rckpt      = torch.load(resume_path, map_location=device)
        model.load_state_dict(rckpt['model_state'])
        completed  = rckpt['completed']
        history    = rckpt['history']
        best_map50 = rckpt['best_map50']
        print(f'[Stage 1g] Resuming from epoch {completed}/{args.epochs}.')
    elif args.resume:
        print(f'[Stage 1g] --resume set but {resume_path} not found. Starting fresh.')

    # ── Phase A: warm m_var only (backbone+neck+m frozen) ────────────
    if args.warmup_epochs > 0 and completed < args.warmup_epochs:
        phase_a_start = completed
        print(f'\n[Phase A] Warming Gaussian head: epochs {phase_a_start+1}-{args.warmup_epochs} '
              f'(m_var only, all else frozen).')
        model.freeze_backbone_train_gaussian()
        opt_a = optim.AdamW(model.gaussian_params(), lr=args.lr_phase_a,
                            weight_decay=1e-4)
        if args.resume and os.path.isfile(resume_path):
            rckpt = torch.load(resume_path, map_location=device)
            if rckpt.get('phase') == 'A':
                opt_a.load_state_dict(rckpt['opt_state'])

        for epoch in range(phase_a_start, args.warmup_epochs):
            t0   = time.time()
            comp = train_one_epoch_gaussian(
                model, train_loader, loss_fn, opt_a, device,
                desc=f'A {epoch+1:3d}/{args.warmup_epochs}',
            )
            elapsed = time.time() - t0
            completed += 1
            print(
                f'  A {epoch + 1:3d}/{args.warmup_epochs}  '
                f'loss={comp["loss"]:.4f}  '
                f'[box={comp["box"]:.4f}  obj={comp["obj"]:.4f}  var={comp["var"]:.4f}]  '
                f'{elapsed:.1f}s'
            )
            history.append({
                'epoch': epoch + 1, 'phase': 'A',
                **comp,
                'mAP50': '', 'mAP5095': '', 'u_mean': '', 'b_uav_mean': '',
                'sigma2_mean': '', 'sigma2_std': '', 'sigma2_healthy': '',
            })
            torch.save({
                'completed':   completed,
                'phase':       'A',
                'model_state': model.state_dict(),
                'opt_state':   opt_a.state_dict(),
                'sched_state': None,
                'history':     history,
                'best_map50':  best_map50,
            }, resume_path)
    elif args.warmup_epochs > 0:
        print(f'\n[Phase A] Already complete ({args.warmup_epochs} epochs). Skipping.')

    # ── Phase B: unfreeze all except m_evi, full fine-tune ────────────
    phase_b_epochs = args.epochs - args.warmup_epochs
    phase_b_done   = max(0, completed - args.warmup_epochs)

    if phase_b_epochs > 0 and phase_b_done < phase_b_epochs:
        phase_b_start = phase_b_done
        print(f'\n[Phase B] Full fine-tuning: epochs {phase_b_start+1}-{phase_b_epochs} '
              f'(backbone+neck+m+m_var; m_evi stays frozen).')
        model.unfreeze_all()
        model.freeze_evidential_head()

        trainable = [p for p in model.parameters() if p.requires_grad]
        opt_b = optim.AdamW(trainable, lr=args.lr_phase_b, weight_decay=1e-4)
        sched_b = optim.lr_scheduler.CosineAnnealingLR(
            opt_b, T_max=phase_b_epochs, eta_min=args.lr_phase_b * 0.01
        )
        if args.resume and os.path.isfile(resume_path):
            rckpt = torch.load(resume_path, map_location=device)
            if rckpt.get('phase') == 'B':
                opt_b.load_state_dict(rckpt['opt_state'])
                if rckpt.get('sched_state') is not None:
                    sched_b.load_state_dict(rckpt['sched_state'])

        for epoch in range(phase_b_start, phase_b_epochs):
            t0   = time.time()
            comp = train_one_epoch_gaussian(
                model, train_loader, loss_fn, opt_b, device,
                desc=f'B {epoch+1:3d}/{phase_b_epochs}',
            )

            model.model[-1].use_cls_for_conf = True
            val_metrics  = validate(
                model, val_loader, device,
                conf_thres=args.conf_thres,
                iou_thres=args.iou_thres,
                K=args.K,
            )
            model.model[-1].use_cls_for_conf = False
            var_health   = check_variance_health(model, val_loader, device)
            sched_b.step()
            elapsed = time.time() - t0
            completed += 1

            total_epoch = args.warmup_epochs + epoch + 1
            print(
                f'  B {total_epoch:3d}/{args.epochs}  '
                f'loss={comp["loss"]:.4f}  '
                f'mAP50={val_metrics["mAP50"]:.4f}  '
                f'σ̂²_mean={var_health["sigma2_mean"]:.4f} '
                f'σ̂²_std={var_health["sigma2_std"]:.4f}  '
                f'{elapsed:.1f}s'
                + ('' if var_health['healthy'] else '  [VARIANCE WARN]')
            )

            history.append({
                'epoch': total_epoch, 'phase': 'B',
                **comp,
                **val_metrics,
                'sigma2_mean':    var_health['sigma2_mean'],
                'sigma2_std':     var_health['sigma2_std'],
                'sigma2_healthy': int(var_health['healthy']),
            })

            if val_metrics['mAP50'] > best_map50:
                best_map50 = val_metrics['mAP50']
                best_path  = os.path.join(args.save_dir, f'{args.name}_best.pt')
                model.save(best_path)
                print(f'  → New best mAP50={best_map50:.4f} → {best_path}')

            # Save last + resume checkpoint after every epoch
            model.save(os.path.join(args.save_dir, f'{args.name}_last.pt'))
            torch.save({
                'completed':   completed,
                'phase':       'B',
                'model_state': model.state_dict(),
                'opt_state':   opt_b.state_dict(),
                'sched_state': sched_b.state_dict(),
                'history':     history,
                'best_map50':  best_map50,
            }, resume_path)

    # ── Done: clean up resume checkpoint, save history ────────────────
    if os.path.isfile(resume_path):
        os.remove(resume_path)
    last_path = os.path.join(args.save_dir, f'{args.name}_last.pt')
    model.save(last_path)
    _save_csv(history, os.path.join(args.save_dir, f'{args.name}_history.csv'))
    print(f'\n[Stage 1g] Done.  Best mAP50={best_map50:.4f}')
    stage2_ckpt = (
        os.path.join(args.save_dir, args.name + '_best.pt')
        if best_map50 > 0.0
        else last_path
    )
    print(f'  Checkpoint for Stage 2: {stage2_ckpt}')
    print(f'  (best_map50={best_map50:.4f}; using last checkpoint if no improvement)')


def _save_csv(history, path):
    if not history:
        return
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    print(f'[Stage 1g] History → {path}')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Phase 3.3 Stage 1g: Gaussian bbox head training.'
    )
    parser.add_argument('--weights',        required=True,
                        help='Pretrained YOLOv5 .pt checkpoint (best_clean.pt or best_clean_rgb.pt).')
    parser.add_argument('--data',           required=True,
                        help='Dataset YAML (antiuav_clean.yaml or antiuav_clean_rgb.yaml).')
    parser.add_argument('--train_txt',      required=True,
                        help='Image list .txt for training split.')
    parser.add_argument('--val_txt',        required=True,
                        help='Image list .txt for validation split.')
    parser.add_argument('--save_dir',       default='/home/ssuttorp/pretrained_models/evidential')
    parser.add_argument('--name',           default='evidential_stage1g',
                        help='Output name prefix (e.g. evidential_ir_stage1g).')
    parser.add_argument('--device',         default='0')
    parser.add_argument('--epochs',         type=int,   default=30,
                        help='Total epochs (Phase A + Phase B).')
    parser.add_argument('--warmup_epochs',  type=int,   default=5,
                        help='Phase A duration (m_var only). Set 0 to skip.')
    parser.add_argument('--batch_size',     type=int,   default=64)
    parser.add_argument('--workers',        type=int,   default=8)
    parser.add_argument('--imgsz',          type=int,   default=640)
    parser.add_argument('--lr_phase_a',     type=float, default=1e-3,
                        help='AdamW lr for Phase A (m_var warm-up).')
    parser.add_argument('--lr_phase_b',     type=float, default=1e-4,
                        help='AdamW lr for Phase B (full fine-tune).')
    parser.add_argument('--lambda_var',     type=float, default=0.1,
                        help='Weight on Gaussian NLL term.')
    parser.add_argument('--K',              type=int,   default=2,
                        help='Evidential classes (architecture param, not used in loss here).')
    parser.add_argument('--conf_thres',     type=float, default=0.25)
    parser.add_argument('--iou_thres',      type=float, default=0.45)
    parser.add_argument('--resume',         action='store_true',
                        help='Resume training from _resume.pt checkpoint if available.')
    parser.add_argument('--seed',           type=int,   default=0,
                        help='Global RNG seed for reproducible / multi-seed runs.')
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
