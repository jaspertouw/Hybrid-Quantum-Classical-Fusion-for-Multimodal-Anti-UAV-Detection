"""
SiamCAR Stage 1 — Gaussian variance head training.

Protocol
--------
Phase A (warmup_epochs=5 by default):
    Freeze entire base; train only bbox_var.
    SGD on gaussian_params(), lr=1e-3.
    No validation — stabilize bbox_var initialization.

Phase B (remaining epochs, default 25):
    Freeze backbone; train neck+car_head+down+bbox_var.
    SGD, lr=1e-4, CosineAnnealingLR → eta_min=1e-6.
    Saves per-epoch checkpoint + best checkpoint by var_loss.
    Runs variance health check each epoch.

Signals of healthy training:
    - var_loss decreases and stabilises (does not diverge or collapse)
    - loc_loss stays flat or slightly improves
    - log_var values remain in [-6, 6] after clamping

Usage
-----
    cd resources/SiamCAR
    python pysot/models/siamcar_gaussian/train_stage1_gaussian.py \\
        --cfg  experiments/siamcar_r50/config.yaml \\
        --stage1_cfg  pysot/models/siamcar_gaussian/config_stage1.yaml \\
        --pretrained  /path/to/siamcar_baseline.pth \\
        --save_dir    ./snapshot_stage1 \\
        --name        siamcar_stage1g
"""

import os
import csv
import time
import logging
import argparse
import random

import numpy as np
import torch
import torch.nn as nn

from pysot.core.config import cfg
from pysot.utils.log_helper import init_log, add_file_handler
from pysot.datasets.dataset import TrkDataset

from pysot.models.siamcar_gaussian.model import SiamCARGaussian
from pysot.models.siamcar_gaussian.losses import SiamCARGaussianLoss

logger = logging.getLogger('global')


# ---------------------------------------------------------------------------
# Data loader
# ---------------------------------------------------------------------------

def build_data_loader(batch_size, num_workers=4):
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
# Optimizer builders
# ---------------------------------------------------------------------------

def build_phase_a_optimizer(model, lr_a, weight_decay, momentum):
    return torch.optim.SGD(
        model.gaussian_params(),
        lr=lr_a,
        momentum=momentum,
        weight_decay=weight_decay,
    )


def build_phase_b_optimizer(model, lr_b, weight_decay, momentum, phase_b_epochs):
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        trainable,
        lr=lr_b,
        momentum=momentum,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=phase_b_epochs,
        eta_min=1e-6,
    )
    return optimizer, scheduler


# ---------------------------------------------------------------------------
# Single-epoch training loop
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, loss_fn, optimizer, grad_clip, desc='train'):
    model.train()
    totals = {k: 0.0 for k in ['total_loss', 'cls_loss', 'loc_loss', 'cen_loss', 'var_loss']}
    nb = len(loader)

    for idx, data in enumerate(loader):
        outputs = model(data)
        outputs['label_cls'] = data['label_cls'].cuda()
        outputs['label_loc'] = data['bbox'].cuda()

        losses = loss_fn(outputs)
        total  = losses['total_loss']
        if isinstance(total, torch.Tensor):
            total = total.mean()

        val = total.item()
        if not (val != val or val == float('inf') or val > 1e4):
            optimizer.zero_grad()
            total.backward()
            nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                grad_clip,
            )
            optimizer.step()

        for k in totals:
            v = losses[k]
            totals[k] += (v.mean().item() if isinstance(v, torch.Tensor) else float(v))

        if (idx + 1) % 50 == 0:
            avgs = {k: totals[k] / (idx + 1) for k in totals}
            logger.info(
                f'{desc} [{idx+1}/{nb}]  '
                + '  '.join(f'{k}={avgs[k]:.4f}' for k in avgs)
            )

    return {k: v / nb for k, v in totals.items()}


# ---------------------------------------------------------------------------
# Variance health check
# ---------------------------------------------------------------------------

def check_variance_health(model, loader, max_batches=50):
    model.eval()
    all_log_var = []
    with torch.no_grad():
        for i, data in enumerate(loader):
            if i >= max_batches:
                break
            out = model(data)
            lv  = out['log_var'].clamp(-6, 6).cpu().reshape(-1)
            all_log_var.append(lv)

    vals    = torch.cat(all_log_var)
    mean_   = vals.mean().item()
    std_    = vals.std().item()
    healthy = True

    if std_ < 0.01:
        logger.warning(
            f'[VARIANCE WARN] log_var nearly constant (std={std_:.4f}). '
            'bbox_var may not be learning spatial variation.'
        )
        healthy = False
    if abs(mean_) > 5.5:
        logger.warning(
            f'[VARIANCE WARN] log_var mean={mean_:.4f} near clamp boundary. '
            'Consider reducing --lambda_var.'
        )
        healthy = False

    model.train()
    return {'log_var_mean': round(mean_, 4), 'log_var_std': round(std_, 4), 'healthy': healthy}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def train(args):
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_file(args.stage1_cfg)

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

    logger.info(f'Loading SiamCAR baseline from {args.pretrained}')
    model = SiamCARGaussian(pretrained_path=args.pretrained)
    model = model.cuda()

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    raw_model = model.module if isinstance(model, nn.DataParallel) else model

    train_loader = build_data_loader(cfg.TRAIN.BATCH_SIZE, args.num_workers)
    loss_fn      = SiamCARGaussianLoss(lambda_var=args.lambda_var)

    history      = []
    completed    = 0
    best_var_loss = float('inf')

    resume_path = os.path.join(args.save_dir, f'{args.name}_resume.pt')
    if args.resume and os.path.isfile(resume_path):
        rckpt = torch.load(resume_path, map_location='cpu')
        raw_model.load_state_dict(rckpt['model_state'])
        completed     = rckpt['completed']
        history       = rckpt.get('history', [])
        best_var_loss = rckpt.get('best_var_loss', float('inf'))
        logger.info(f'Resumed from epoch {completed}/{args.epochs}.')

    # ── Phase A ──────────────────────────────────────────────────────────
    if args.warmup_epochs > 0 and completed < args.warmup_epochs:
        phase_a_start = completed
        logger.info(
            f'Phase A: warming up bbox_var for epochs '
            f'{phase_a_start + 1}–{args.warmup_epochs}'
        )
        raw_model.freeze_base()
        opt_a = build_phase_a_optimizer(
            raw_model, args.lr_phase_a, args.weight_decay, args.momentum
        )

        if args.resume and os.path.isfile(resume_path):
            rckpt = torch.load(resume_path, map_location='cpu')
            if rckpt.get('phase') == 'A' and 'opt_state' in rckpt:
                opt_a.load_state_dict(rckpt['opt_state'])

        for epoch in range(phase_a_start, args.warmup_epochs):
            t0   = time.time()
            comp = train_one_epoch(
                model, train_loader, loss_fn, opt_a,
                args.grad_clip, desc=f'A {epoch+1:3d}/{args.warmup_epochs}'
            )
            elapsed    = time.time() - t0
            completed += 1
            logger.info(
                f'[Phase A] epoch {epoch+1}/{args.warmup_epochs}  '
                + '  '.join(f'{k}={v:.4f}' for k, v in comp.items())
                + f'  ({elapsed:.0f}s)'
            )
            history.append({'epoch': epoch + 1, 'phase': 'A', **comp,
                            'log_var_mean': '', 'log_var_std': '', 'healthy': ''})
            torch.save({
                'completed':   completed,
                'phase':       'A',
                'model_state': raw_model.state_dict(),
                'opt_state':   opt_a.state_dict(),
                'history':     history,
                'best_var_loss': best_var_loss,
            }, resume_path)
    elif args.warmup_epochs > 0:
        logger.info('Phase A already complete. Skipping.')

    # ── Phase B ──────────────────────────────────────────────────────────
    phase_b_epochs = args.epochs - args.warmup_epochs
    phase_b_done   = max(0, completed - args.warmup_epochs)

    if phase_b_epochs > 0 and phase_b_done < phase_b_epochs:
        phase_b_start = phase_b_done
        logger.info(
            f'Phase B: joint training for epochs '
            f'{phase_b_start + 1}–{phase_b_epochs} '
            f'(backbone frozen)'
        )
        raw_model.freeze_backbone_only()
        opt_b, sched_b = build_phase_b_optimizer(
            raw_model, args.lr_phase_b, args.weight_decay, args.momentum, phase_b_epochs
        )

        if args.resume and os.path.isfile(resume_path):
            rckpt = torch.load(resume_path, map_location='cpu')
            if rckpt.get('phase') == 'B':
                if 'opt_state' in rckpt:
                    opt_b.load_state_dict(rckpt['opt_state'])
                if 'sched_state' in rckpt:
                    sched_b.load_state_dict(rckpt['sched_state'])

        for epoch in range(phase_b_start, phase_b_epochs):
            t0   = time.time()
            comp = train_one_epoch(
                model, train_loader, loss_fn, opt_b,
                args.grad_clip,
                desc=f'B {epoch+1:3d}/{phase_b_epochs}'
            )
            vhealth   = check_variance_health(raw_model, train_loader)
            sched_b.step()
            elapsed    = time.time() - t0
            completed += 1
            total_epoch = args.warmup_epochs + epoch + 1

            logger.info(
                f'[Phase B] epoch {total_epoch}/{args.epochs}  '
                + '  '.join(f'{k}={v:.4f}' for k, v in comp.items())
                + f'  log_var_mean={vhealth["log_var_mean"]}'
                + f'  log_var_std={vhealth["log_var_std"]}'
                + ('' if vhealth['healthy'] else '  [VARIANCE WARN]')
                + f'  ({elapsed:.0f}s)'
            )

            history.append({'epoch': total_epoch, 'phase': 'B', **comp, **vhealth})

            # Per-epoch checkpoint (mirrors original train.py naming)
            epoch_ckpt = os.path.join(args.save_dir, f'{args.name}_e{total_epoch}.pth')
            torch.save(
                {'epoch': total_epoch, 'state_dict': raw_model.state_dict()},
                epoch_ckpt,
            )

            if comp['var_loss'] < best_var_loss:
                best_var_loss = comp['var_loss']
                best_path = os.path.join(args.save_dir, f'{args.name}_best.pth')
                torch.save(
                    {'epoch': total_epoch, 'state_dict': raw_model.state_dict()},
                    best_path,
                )
                logger.info(f'  New best var_loss={best_var_loss:.4f} → {best_path}')

            torch.save({
                'completed':     completed,
                'phase':         'B',
                'model_state':   raw_model.state_dict(),
                'opt_state':     opt_b.state_dict(),
                'sched_state':   sched_b.state_dict(),
                'history':       history,
                'best_var_loss': best_var_loss,
            }, resume_path)

    # ── Finalise ─────────────────────────────────────────────────────────
    if os.path.isfile(resume_path):
        os.remove(resume_path)

    last_path = os.path.join(args.save_dir, f'{args.name}_last.pth')
    torch.save({'state_dict': raw_model.state_dict()}, last_path)
    logger.info(f'Stage 1 complete. Best var_loss={best_var_loss:.4f}')

    csv_path = os.path.join(args.save_dir, f'{args.name}_history.csv')
    if history:
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
            writer.writeheader()
            writer.writerows(history)
        logger.info(f'History → {csv_path}')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser('SiamCAR Stage 1 Gaussian head training')
    p.add_argument('--cfg',          required=True,
                   help='Base config (experiments/siamcar_r50/config.yaml)')
    p.add_argument('--stage1_cfg',   required=True,
                   help='Stage 1 overrides (siamcar_gaussian/config_stage1.yaml)')
    p.add_argument('--pretrained',   required=True,
                   help='SiamCAR baseline checkpoint (.pth)')
    p.add_argument('--save_dir',     default='./snapshot_stage1')
    p.add_argument('--name',         default='siamcar_stage1g')
    p.add_argument('--epochs',       type=int,   default=30)
    p.add_argument('--warmup_epochs',type=int,   default=5)
    p.add_argument('--lr_phase_a',   type=float, default=1e-3,
                   help='SGD lr for Phase A (bbox_var warm-up)')
    p.add_argument('--lr_phase_b',   type=float, default=1e-4,
                   help='SGD lr for Phase B (joint fine-tune)')
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--momentum',     type=float, default=0.9)
    p.add_argument('--lambda_var',   type=float, default=0.1)
    p.add_argument('--grad_clip',    type=float, default=10.0)
    p.add_argument('--num_workers',  type=int,   default=4)
    p.add_argument('--resume',       action='store_true')
    p.add_argument('--seed',         type=int,   default=123456)
    return p.parse_args()


if __name__ == '__main__':
    train(parse_args())
