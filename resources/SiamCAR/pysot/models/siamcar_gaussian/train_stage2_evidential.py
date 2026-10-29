"""
SiamCAR Stage 2 — Evidential classification head training.

Protocol
--------
Single phase (no warm-up): entire Stage 1 network frozen, only cls_evi trains.
KL annealing from 0 → 1 over the first min(10, epochs) epochs provides an
implicit curriculum: pure ECE first, then regularisation pulls toward
uniform Dirichlet prior.

40 epochs, SGD lr=1e-4, CosineAnnealingLR, batch=64, λ_evi=1.0.

Uncertainty monitoring:
    u_mean < 0.05          → [WARN] overconfident collapse (any epoch)
    u_mean > 0.90, ep ≥ 10 → [WARN] head not learning

Resume: --resume flag reloads {name}_resume.pt including the KL step counter
so the annealing schedule continues from where it left off.

Usage
-----
    cd resources/SiamCAR
    python pysot/models/siamcar_gaussian/train_stage2_evidential.py \\
        --cfg         experiments/siamcar_r50/config_antiuav_thermal.yaml \\
        --stage2_cfg  pysot/models/siamcar_gaussian/config_stage2.yaml \\
        --stage1_path /path/to/siamcar_ir_stage1g_best.pth \\
        --pretrained_path /path/to/siamcar_thermal_final.pth \\
        --save_dir    ./snapshot_stage2 \\
        --name        siamcar_stage2_ir
"""

import os
import csv
import time
import logging
import argparse
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pysot.core.config import cfg
from pysot.utils.log_helper import init_log, add_file_handler
from pysot.datasets.dataset import TrkDataset

from pysot.models.siamcar_gaussian.model_stage2 import SiamCARStage2
from pysot.models.siamcar_gaussian.losses_stage2 import SiamCARStage2Loss

logger = logging.getLogger('global')


# ---------------------------------------------------------------------------
# Data loader
# ---------------------------------------------------------------------------

def build_data_loader(batch_size: int, num_workers: int = 4):
    dataset = TrkDataset()
    loader  = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    return loader


# ---------------------------------------------------------------------------
# Single-epoch training loop
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, loss_fn, optimizer,
                    grad_clip: float, desc: str = 'train',
                    max_steps: int = 0) -> dict:
    """Run one epoch.  If max_steps > 0, stop after that many batches."""
    model.train()
    total_evi = 0.0
    steps     = min(max_steps, len(loader)) if max_steps > 0 else len(loader)

    for idx, data in enumerate(loader):
        if max_steps > 0 and idx >= max_steps:
            break

        outputs = model(data)
        outputs['label_cls'] = data['label_cls'].cuda()
        outputs['label_loc'] = data['bbox'].cuda()

        losses = loss_fn(outputs)
        loss   = losses['evi_loss']
        if isinstance(loss, torch.Tensor):
            loss = loss.mean()

        val = loss.item()
        if not (val != val or val == float('inf') or val > 1e4):
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                grad_clip,
            )
            optimizer.step()

        total_evi += (loss.item() if isinstance(loss, torch.Tensor) else float(loss))

        if (idx + 1) % 50 == 0:
            logger.info(
                f'{desc} [{idx+1}/{steps}]  '
                f'evi_loss={total_evi/(idx+1):.4f}  '
                f'kl_w={losses["kl_weight"]:.3f}'
            )

    return {'evi_loss': total_evi / steps}


# ---------------------------------------------------------------------------
# Uncertainty validation
# ---------------------------------------------------------------------------

def validate_evidential(model, loader, K: int = 2,
                        epoch: int = 0, max_batches: int = 50) -> dict:
    """Collect uncertainty statistics over a subset of the training data."""
    model.eval()
    all_u     = []
    all_b_uav = []

    with torch.no_grad():
        for i, data in enumerate(loader):
            if i >= max_batches:
                break
            out      = model(data)
            evidence = F.relu(out['evi_raw'])           # (B, K, H, W)
            alpha    = evidence + 1.0
            S        = alpha.sum(dim=1, keepdim=True)   # (B, 1, H, W)
            u        = (K / S).squeeze(1).cpu()         # (B, H, W) vacuity
            b_uav    = (evidence[:, 1, ...] / S.squeeze(1)).cpu()  # (B, H, W)
            all_u.append(u.reshape(-1))
            all_b_uav.append(b_uav.reshape(-1))

    u_vals     = torch.cat(all_u)
    b_uav_vals = torch.cat(all_b_uav)

    u_mean    = u_vals.mean().item()
    u_std     = u_vals.std().item()
    u_min     = u_vals.min().item()
    u_max     = u_vals.max().item()
    b_uav_mean = b_uav_vals.mean().item()

    # Collapse warnings
    if u_mean < 0.05:
        logger.warning(
            f'[UNC WARN] epoch {epoch}: u_mean={u_mean:.4f} < 0.05 — '
            'cls_evi may be collapsing to overconfident predictions.'
        )
    if u_mean > 0.90 and epoch >= 10:
        logger.warning(
            f'[UNC WARN] epoch {epoch}: u_mean={u_mean:.4f} > 0.90 after '
            'epoch 10 — cls_evi does not appear to be learning.'
        )

    model.train()
    return {
        'u_mean':     round(u_mean,     4),
        'u_std':      round(u_std,      4),
        'u_min':      round(u_min,      4),
        'u_max':      round(u_max,      4),
        'b_uav_mean': round(b_uav_mean, 4),
    }


# ---------------------------------------------------------------------------
# Numerical sanity check
# ---------------------------------------------------------------------------

def run_numerical_check(model, loader, K: int = 2, n_batches: int = 2) -> None:
    """Assert Dirichlet parameters are valid and no NaN/Inf values present."""
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(loader):
            if i >= n_batches:
                break
            out   = model(data)
            e     = F.relu(out['evi_raw'])
            alpha = e + 1.0
            S     = alpha.sum(dim=1)
            u     = K / S
            nan_count = torch.isnan(out['evi_raw']).sum().item()
            logger.info(
                f'  [check] alpha.min={alpha.min():.3f} (expect >=1.0)  '
                f'u.range=[{u.min():.3f},{u.max():.3f}] (expect (0,1])  '
                f'NaN count={nan_count} (expect 0)'
            )
            assert alpha.min().item() >= 1.0 - 1e-5, \
                f'alpha < 1 detected: {alpha.min().item()}'
            assert nan_count == 0, 'NaN in evi_raw'
    model.train()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def train(args) -> None:
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_file(args.stage2_cfg)

    os.makedirs(args.save_dir, exist_ok=True)
    init_log('global', logging.INFO)
    add_file_handler(
        'global',
        os.path.join(args.save_dir, f'{args.name}_train.log'),
        logging.INFO,
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    logger.info(f'Loading Stage 1 checkpoint from {args.stage1_path}')
    model = SiamCARStage2(
        stage1_path=args.stage1_path,
        pretrained_path=args.pretrained_path,
        K=args.K,
    ).cuda()

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    raw_model = model.module if isinstance(model, nn.DataParallel) else model

    raw_model.freeze_all_train_evidential()

    train_loader = build_data_loader(cfg.TRAIN.BATCH_SIZE, args.num_workers)

    anneal_epochs  = min(10, args.epochs)
    steps_per_epoch = (min(args.max_steps_per_epoch, len(train_loader))
                       if args.max_steps_per_epoch > 0 else len(train_loader))
    anneal_steps   = anneal_epochs * steps_per_epoch
    logger.info(f'Steps per epoch: {steps_per_epoch}/{len(train_loader)} '
                f'({"capped" if args.max_steps_per_epoch > 0 else "full dataset"})')
    logger.info(f'KL annealing: 0→1 over {anneal_epochs} epochs '
                f'({anneal_steps} gradient steps)')

    loss_fn = SiamCARStage2Loss(K=args.K, lambda_evi=args.lambda_evi)
    loss_fn._anneal_steps = anneal_steps

    optimizer = torch.optim.SGD(
        raw_model.evidential_params(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 0.01,
    )

    # ------------------------------------------------------------------
    # Resume
    # ------------------------------------------------------------------
    history     = []
    start_epoch = 0
    best_loss   = float('inf')
    loss_window = deque(maxlen=5)

    resume_path = os.path.join(args.save_dir, f'{args.name}_resume.pt')
    if args.resume and os.path.isfile(resume_path):
        rckpt = torch.load(resume_path, map_location='cpu')
        raw_model.load_state_dict(rckpt['model_state'])
        optimizer.load_state_dict(rckpt['opt_state'])
        scheduler.load_state_dict(rckpt['sched_state'])
        loss_fn._step  = rckpt['loss_step']   # restore KL counter
        start_epoch    = rckpt['completed']
        history        = rckpt.get('history', [])
        best_loss      = rckpt.get('best_loss', float('inf'))
        loss_window    = deque(rckpt.get('loss_window', []), maxlen=5)
        logger.info(f'Resumed from epoch {start_epoch}/{args.epochs} '
                    f'(KL step={loss_fn._step})')

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    for epoch in range(start_epoch, args.epochs):
        t0   = time.time()
        comp = train_one_epoch(
            model, train_loader, loss_fn, optimizer,
            args.grad_clip,
            desc=f'E{epoch+1:3d}/{args.epochs}',
            max_steps=args.max_steps_per_epoch,
        )

        val_unc = validate_evidential(
            raw_model, train_loader, K=args.K, epoch=epoch + 1
        )

        scheduler.step()
        elapsed = time.time() - t0

        logger.info(
            f'[epoch {epoch+1:3d}/{args.epochs}]  '
            f'evi_loss={comp["evi_loss"]:.4f}  '
            f'u_mean={val_unc["u_mean"]}  '
            f'b_uav={val_unc["b_uav_mean"]}  '
            f'kl_w={loss_fn._kl_weight:.3f}  '
            f'lr={optimizer.param_groups[0]["lr"]:.2e}  '
            f'({elapsed:.0f}s)'
        )

        # Numerical check every check_every epochs
        if (epoch + 1) % args.check_every == 0:
            logger.info(f'Running numerical check at epoch {epoch+1}...')
            run_numerical_check(raw_model, train_loader, K=args.K)

        # Rolling-window best checkpoint
        loss_window.append(comp['evi_loss'])
        rolling = sum(loss_window) / len(loss_window)
        if rolling < best_loss:
            best_loss = rolling
            best_path = os.path.join(args.save_dir, f'{args.name}_best.pth')
            torch.save(
                {'epoch': epoch + 1, 'state_dict': raw_model.state_dict()},
                best_path,
            )
            logger.info(f'  New best rolling_loss={best_loss:.4f} → {best_path}')

        # Per-epoch checkpoint
        epoch_path = os.path.join(args.save_dir, f'{args.name}_e{epoch+1}.pth')
        torch.save(
            {'epoch': epoch + 1, 'state_dict': raw_model.state_dict()},
            epoch_path,
        )

        # History record
        history.append({
            'epoch':      epoch + 1,
            'evi_loss':   comp['evi_loss'],
            'kl_weight':  round(loss_fn._kl_weight, 4),
            **val_unc,
        })

        # Resume checkpoint (includes KL step counter)
        torch.save({
            'completed':   epoch + 1,
            'model_state': raw_model.state_dict(),
            'opt_state':   optimizer.state_dict(),
            'sched_state': scheduler.state_dict(),
            'loss_step':   loss_fn._step,
            'history':     history,
            'best_loss':   best_loss,
            'loss_window': list(loss_window),
        }, resume_path)

    # ------------------------------------------------------------------
    # Finalise
    # ------------------------------------------------------------------
    if os.path.isfile(resume_path):
        os.remove(resume_path)

    last_path = os.path.join(args.save_dir, f'{args.name}_last.pth')
    torch.save({'state_dict': raw_model.state_dict()}, last_path)
    logger.info(f'Stage 2 complete. Best rolling_loss={best_loss:.4f}')

    csv_path = os.path.join(args.save_dir, f'{args.name}_history.csv')
    if history:
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
            writer.writeheader()
            writer.writerows(history)
        logger.info(f'History → {csv_path}')

    logger.info(f'Best checkpoint : {best_path}')
    logger.info(f'Last checkpoint : {last_path}')
    logger.info(f'Next step: Stage 3 — fine-tune all heads jointly.')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser('SiamCAR Stage 2 evidential head training')
    p.add_argument('--cfg',            required=True,
                   help='Stream config (e.g. config_antiuav_thermal.yaml)')
    p.add_argument('--stage2_cfg',     required=True,
                   help='Stage 2 overrides (config_stage2.yaml)')
    p.add_argument('--stage1_path',    required=True,
                   help='Stage 1 checkpoint (_best.pth or _last.pth)')
    p.add_argument('--pretrained_path', required=True,
                   help='Original SiamCAR baseline .pth (architecture only)')
    p.add_argument('--save_dir',       default='./snapshot_stage2')
    p.add_argument('--name',           default='siamcar_stage2')
    p.add_argument('--epochs',         type=int,   default=40)
    p.add_argument('--K',              type=int,   default=2,
                   help='Number of evidential classes')
    p.add_argument('--lr',             type=float, default=1e-4)
    p.add_argument('--lambda_evi',     type=float, default=1.0)
    p.add_argument('--weight_decay',   type=float, default=1e-4)
    p.add_argument('--momentum',       type=float, default=0.9)
    p.add_argument('--grad_clip',      type=float, default=10.0)
    p.add_argument('--num_workers',    type=int,   default=4)
    p.add_argument('--max_steps_per_epoch', type=int, default=0,
                   help='Cap gradient steps per epoch (0 = full dataset). '
                        'Use ~1500 to keep each epoch ~15 min on A100.')
    p.add_argument('--check_every',    type=int,   default=5,
                   help='Run numerical checks every N epochs')
    p.add_argument('--resume',         action='store_true')
    p.add_argument('--seed',           type=int,   default=123456)
    return p.parse_args()


if __name__ == '__main__':
    train(parse_args())
