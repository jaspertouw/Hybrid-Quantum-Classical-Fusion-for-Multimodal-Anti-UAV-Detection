"""
Phase 3.3 Stage 2 (Evidential) — train the evidential classification head per stream.

Protocol
--------
Starting point: Stage 1g checkpoint (backbone + m + m_var all trained).
Freeze: the entire Stage 1 network (backbone, neck, m, m_var) is frozen.
Train:  only m_evi (Dirichlet evidence head) trains with EDL loss.

KL annealing: KL regularisation weight ramps linearly 0 → 1 over
``--epochs`` training steps so the model first fits the expected CE signal
before the uniform-prior regulariser activates.

Monitoring
----------
Every epoch  : u_mean + b_UAV_mean on val set (from unc_scales inference output).
Every N epochs: full numerical verification via verify.py's run_verification().
  - [WARN] if u_mean < 0.05 → overconfident collapse
  - [WARN] if u_mean > 0.90 after epoch 5 → head has learned nothing

Saving best checkpoint: by lowest running-average training EDL loss.
(Val mAP50 is unchanged since backbone is frozen; we do not re-evaluate it.)

Usage
-----
IR stream:
  python lib/models/yolo_evidential/train_stage2_evidential.py \\
      --stage1_checkpoint .../evidential_ir_stage1g_best.pt \\
      --base_weights      .../best_clean.pt \\
      --data              .../antiuav_clean.yaml \\
      --train_txt         .../train_ir_images.txt \\
      --val_txt           .../val_ir_images.txt \\
      --name evidential_ir_stage2 --device 0

RGB stream:
  python lib/models/yolo_evidential/train_stage2_evidential.py \\
      --stage1_checkpoint .../evidential_rgb_stage1g_best.pt \\
      --base_weights      .../best_clean_rgb.pt \\
      --data              .../antiuav_clean_rgb.yaml \\
      --train_txt         .../train_rgb_images.txt \\
      --val_txt           .../val_rgb_images.txt \\
      --name evidential_rgb_stage2 --device 0
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

from lib.models.yolo_evidential.train_single_stream import build_loaders, DEFAULT_HYP
from lib.models.yolo_evidential.model              import EvidentialYOLO
from lib.models.yolo_evidential.losses_staged      import ComputeEvidentialOnlyLoss
from lib.models.yolo_evidential.verify             import (
    verify_evidential_outputs, run_verification,
)


# ---------------------------------------------------------------------------
# Training loop (evidential-only loss)
# ---------------------------------------------------------------------------

def train_one_epoch_evidential(model, loader, loss_fn, optimizer, device,
                               desc: str = 'train'):
    """One epoch with ComputeEvidentialOnlyLoss.

    Returns dict: loss (average per image).
    """
    model.train()
    total_loss = 0.0
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

        loss, _ = loss_fn(preds, targets)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()

        total_loss += loss.item() / imgs.shape[0]
        pbar.set_postfix(loss=f'{loss.item() / imgs.shape[0]:.4f}')

    pbar.close()
    return {'loss': total_loss / nb}


# ---------------------------------------------------------------------------
# Evidential uncertainty validation
# ---------------------------------------------------------------------------

def validate_evidential(model, val_loader, device, K: int = 2,
                        epoch: int = 0, max_images: int = 500):
    """Evaluate uncertainty statistics on the validation set.

    Args:
        model:       EvidentialYOLO (eval mode after call).
        val_loader:  validation DataLoader.
        device:      torch.device.
        K:           number of evidential classes.
        epoch:       current epoch (used for collapse warning threshold).
        max_images:  cap on images evaluated (for speed).

    Returns:
        dict: u_mean, u_std, u_min, u_max, b_uav_mean
    """
    model.eval()
    u_vals, b_vals = [], []
    n_seen = 0

    with torch.no_grad():
        for imgs, targets, *_ in val_loader:
            imgs = imgs.to(device).float() / 255.0
            out  = model(imgs)

            if isinstance(out, tuple) and len(out) >= 5:
                _, _, _, _, unc_scales = out
                for s in unc_scales:
                    u_vals.append(s['u'].cpu().float())
                    b_vals.append(s['b'][..., 0].cpu().float())   # b_UAV

            n_seen += imgs.shape[0]
            if n_seen >= max_images:
                break

    if not u_vals:
        return {k: float('nan') for k in
                ['u_mean', 'u_std', 'u_min', 'u_max', 'b_uav_mean']}

    all_u = torch.cat([v.reshape(-1) for v in u_vals])
    all_b = torch.cat([v.reshape(-1) for v in b_vals])

    u_mean = all_u.mean().item()
    u_std  = all_u.std().item()
    u_min  = all_u.min().item()
    u_max  = all_u.max().item()

    # Collapse detection
    OVERCONF_THRESHOLD   = 0.05
    UNINFORM_THRESHOLD   = 0.90
    if u_mean < OVERCONF_THRESHOLD:
        print(
            f'  [WARN] Epoch {epoch}: u_mean={u_mean:.4f} < {OVERCONF_THRESHOLD}. '
            'Overconfident collapse detected. '
            'Increase lambda_evi or reduce KL annealing speed.'
        )
    elif epoch >= 5 and u_mean > UNINFORM_THRESHOLD:
        print(
            f'  [WARN] Epoch {epoch}: u_mean={u_mean:.4f} > {UNINFORM_THRESHOLD} '
            'after warmup. Evidential head may not be learning — '
            'check that targets are assigned and loss is non-zero.'
        )

    return {
        'u_mean':     u_mean,
        'u_std':      u_std,
        'u_min':      u_min,
        'u_max':      u_max,
        'b_uav_mean': all_b.mean().item(),
    }


# ---------------------------------------------------------------------------
# Main training orchestrator
# ---------------------------------------------------------------------------

def train(args):
    device = select_device(args.device)
    os.makedirs(args.save_dir, exist_ok=True)

    # deterministic=False: seed m_evi init, data shuffling and augmentation for
    # reproducible per-seed runs without forcing deterministic algorithms
    # (some custom ops lack a deterministic kernel and would raise).
    init_seeds(args.seed, deterministic=False)
    print(f'[Stage 2] Seed = {args.seed}')

    resume_path = os.path.join(args.save_dir, f'{args.name}_resume.pt')

    # ── Load Stage 1g checkpoint ──────────────────────────────────────
    print(f'\n[Stage 2] Loading Stage 1g checkpoint: {args.stage1_checkpoint}')
    print(f'[Stage 2] Base architecture from:       {args.base_weights}')
    model = EvidentialYOLO.load(
        path=args.stage1_checkpoint,
        pretrained_weights=args.base_weights,
        device=device,
        K=args.K,
    )
    model.hyp = DEFAULT_HYP

    # ── Freeze everything except m_evi ────────────────────────────────
    model.freeze_all_train_evidential()

    # ── Data loaders ─────────────────────────────────────────────────
    stride = int(model.stride.max())
    train_loader, val_loader, imgsz = build_loaders(
        args.train_txt, args.val_txt, args.data,
        args.imgsz, args.batch_size, stride,
        workers=args.workers,
        hyp={'mosaic': 0.0},
    )
    print(f'[Stage 2] Train: {len(train_loader.dataset)} images | '
          f'Val: {len(val_loader.dataset)} images | imgsz={imgsz}')

    # ── Loss function ─────────────────────────────────────────────────
    anneal_epochs = min(10, args.epochs)
    anneal_steps  = anneal_epochs * len(train_loader)
    loss_fn = ComputeEvidentialOnlyLoss(
        model, DEFAULT_HYP,
        lambda_evi=args.lambda_evi,
        K=args.K,
    )
    loss_fn._anneal_steps = anneal_steps
    print(f'[Stage 2] KL annealing over {anneal_steps} steps '
          f'(first {anneal_epochs} of {args.epochs} epochs × {len(train_loader)} batches).')

    # ── Optimizer + scheduler ─────────────────────────────────────────
    optimizer = optim.AdamW(
        model.evidential_params(), lr=args.lr, weight_decay=1e-4
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    history     = []
    best_loss   = float('inf')
    loss_window = []
    start_epoch = 0

    # ── Resume ───────────────────────────────────────────────────────
    if args.resume:
        if os.path.isfile(resume_path):
            rckpt = torch.load(resume_path, map_location=device)
            model.load_state_dict(rckpt['model_state'])
            optimizer.load_state_dict(rckpt['opt_state'])
            scheduler.load_state_dict(rckpt['sched_state'])
            loss_fn._step  = rckpt['loss_step']
            start_epoch    = rckpt['completed']
            history        = rckpt['history']
            best_loss      = rckpt['best_loss']
            loss_window    = rckpt['loss_window']
            print(f'[Stage 2] Resumed from epoch {start_epoch}/{args.epochs}.')
        else:
            print(f'[Stage 2] --resume set but {resume_path} not found. Starting fresh.')

    # ── Training loop ─────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs):
        t0   = time.time()
        comp = train_one_epoch_evidential(
            model, train_loader, loss_fn, optimizer, device,
            desc=f'E {epoch+1:3d}/{args.epochs}',
        )
        val_unc = validate_evidential(
            model, val_loader, device, K=args.K, epoch=epoch + 1
        )
        scheduler.step()
        elapsed = time.time() - t0

        loss_window.append(comp['loss'])
        if len(loss_window) > 5:
            loss_window.pop(0)
        rolling_loss = float(np.mean(loss_window))

        kl_w = loss_fn._kl_weight
        print(
            f'  E {epoch + 1:3d}/{args.epochs}  '
            f'loss={comp["loss"]:.4f}  '
            f'u_mean={val_unc["u_mean"]:.4f}  '
            f'b_UAV={val_unc["b_uav_mean"]:.4f}  '
            f'KL_w={kl_w:.3f}  '
            f'{elapsed:.1f}s'
        )

        if (epoch + 1) % args.check_every == 0:
            print(f'\n  [Verify] Epoch {epoch + 1} numerical check:')
            for imgs, *_ in val_loader:
                sample_imgs = imgs[:4].to(device).float() / 255.0
                break
            run_verification(model, sample_imgs, device, K=args.K)

        history.append({
            'epoch':      epoch + 1,
            'loss':       comp['loss'],
            'u_mean':     val_unc['u_mean'],
            'u_std':      val_unc['u_std'],
            'u_min':      val_unc['u_min'],
            'u_max':      val_unc['u_max'],
            'b_uav_mean': val_unc['b_uav_mean'],
            'kl_weight':  kl_w,
        })

        if rolling_loss < best_loss:
            best_loss = rolling_loss
            best_path = os.path.join(args.save_dir, f'{args.name}_best.pt')
            model.save(best_path)
            print(f'  → New best loss={best_loss:.4f} → {best_path}')

        # Save last + resume checkpoint after every epoch
        model.save(os.path.join(args.save_dir, f'{args.name}_last.pt'))
        torch.save({
            'completed':   epoch + 1,
            'model_state': model.state_dict(),
            'opt_state':   optimizer.state_dict(),
            'sched_state': scheduler.state_dict(),
            'loss_step':   loss_fn._step,
            'history':     history,
            'best_loss':   best_loss,
            'loss_window': loss_window,
        }, resume_path)

    # ── Done: clean up resume checkpoint, save history ────────────────
    if os.path.isfile(resume_path):
        os.remove(resume_path)
    _save_csv(history, os.path.join(args.save_dir, f'{args.name}_history.csv'))

    print(f'\n[Stage 2] Done.  Best train loss={best_loss:.4f}')
    print(f'  Checkpoint: {os.path.join(args.save_dir, args.name + "_best.pt")}')

    print('\n[Stage 2] Final numerical verification:')
    for imgs, *_ in val_loader:
        sample_imgs = imgs[:8].to(device).float() / 255.0
        break
    run_verification(model, sample_imgs, device, K=args.K)


def _save_csv(history, path):
    if not history:
        return
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    print(f'[Stage 2] History → {path}')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Phase 3.3 Stage 2: evidential classification head training.'
    )
    parser.add_argument('--stage1_checkpoint', required=True,
                        help='Stage 1g best checkpoint (evidential_*_stage1g_best.pt).')
    parser.add_argument('--base_weights',  required=True,
                        help='Original YOLOv5 .pt (for architecture rebuild). '
                             'E.g. best_clean.pt or best_clean_rgb.pt.')
    parser.add_argument('--data',          required=True,
                        help='Dataset YAML.')
    parser.add_argument('--train_txt',     required=True)
    parser.add_argument('--val_txt',       required=True)
    parser.add_argument('--save_dir',      default='/home/ssuttorp/pretrained_models/evidential')
    parser.add_argument('--name',          default='evidential_stage2',
                        help='Output name prefix (e.g. evidential_ir_stage2).')
    parser.add_argument('--device',        default='0')
    parser.add_argument('--epochs',        type=int,   default=20)
    parser.add_argument('--batch_size',    type=int,   default=64)
    parser.add_argument('--workers',       type=int,   default=8)
    parser.add_argument('--imgsz',         type=int,   default=640)
    parser.add_argument('--lr',            type=float, default=1e-3,
                        help='AdamW learning rate for m_evi.')
    parser.add_argument('--lambda_evi',    type=float, default=1.0,
                        help='Overall weight on the EDL loss term.')
    parser.add_argument('--K',             type=int,   default=2,
                        help='Number of evidential classes.')
    parser.add_argument('--check_every',   type=int,   default=5,
                        help='Run full numerical verification every N epochs.')
    parser.add_argument('--resume',        action='store_true',
                        help='Resume training from _resume.pt checkpoint if available.')
    parser.add_argument('--seed',          type=int,   default=0,
                        help='Global RNG seed for reproducible / multi-seed runs.')
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
