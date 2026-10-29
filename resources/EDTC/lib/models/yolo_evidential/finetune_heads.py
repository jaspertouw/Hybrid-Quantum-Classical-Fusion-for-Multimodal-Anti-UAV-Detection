"""
Fine-tune the Gaussian variance + evidential classification heads (Phase 3.2).

WHAT THIS SCRIPT DOES
─────────────────────
1. Loads a pretrained YOLOv5 checkpoint (best_clean.pt or best_clean_rgb.pt).
2. Wraps it with EvidentialYOLO, replacing the Detect head with EvidentialDetect.
3. Freezes the backbone; only m_var and m_evi are trainable.
4. Optionally runs numerical verification on a small batch BEFORE training.
5. Trains for a configurable number of epochs using ComputeEvidentialLoss.
6. Saves the fine-tuned model.

USAGE (on Snellius)
───────────────────
# IR model
python lib/models/yolo_evidential/finetune_heads.py \\
    --weights  /home/ssuttorp/pretrained_models/best_clean.pt \\
    --data     /home/ssuttorp/mm-uav-threat-detection/resources/EDTC/yolov5/data/antiuav_clean.yaml \\
    --img_list /home/ssuttorp/Anti-UAV-RGBT/train_ir_images.txt \\
    --save_dir /home/ssuttorp/pretrained_models/evidential \\
    --name     evidential_ir \\
    --device   0 \\
    --epochs   10 \\
    --verify

# RGB model (same but swap weights, data, img_list, name)
"""

import os
import sys
import argparse
import time
from pathlib import Path

import torch
import torch.optim as optim
import numpy as np

_edtc_root = Path(__file__).resolve().parents[3]
_yolo_root = _edtc_root / 'yolov5'
for p in [str(_edtc_root), str(_yolo_root)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from yolov5.utils.torch_utils import select_device
from yolov5.utils.dataloaders import create_dataloader
from yolov5.utils.general import check_img_size, colorstr

from lib.models.yolo_evidential.model  import EvidentialYOLO
from lib.models.yolo_evidential.losses import ComputeEvidentialLoss
from lib.models.yolo_evidential.verify import run_verification


# ---------------------------------------------------------------------------
# Default hyperparameters (subset of YOLOv5 hyp.scratch-low.yaml)
# ---------------------------------------------------------------------------

DEFAULT_HYP = {
    'box':        0.05,   # bbox loss weight
    'obj':        1.0,    # objectness loss weight
    'cls':        0.5,    # evidential classification loss weight
    'anchor_t':   4.0,    # anchor matching threshold
    'fl_gamma':   0.0,    # focal loss gamma (0 = off)
    'cls_pw':     1.0,
    'obj_pw':     1.0,
    'label_smoothing': 0.0,
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def build_dataloader(img_list: str, data_yaml: str, imgsz: int,
                     batch_size: int, stride: int, device: torch.device):
    """Build a YOLOv5 dataloader from a pre-built image list file.

    Args:
        img_list:   path to a .txt file listing absolute image paths.
        data_yaml:  dataset YAML (needed for augmentation settings).
        imgsz:      inference image size (square).
        batch_size: mini-batch size.
        stride:     model stride (for image size checking).
        device:     torch.device.

    Returns:
        (dataloader, dataset)
    """
    import yaml
    with open(data_yaml) as f:
        data_dict = yaml.safe_load(f)

    gs = max(stride, 32)
    imgsz = check_img_size(imgsz, s=gs)

    loader, dataset = create_dataloader(
        img_list,
        imgsz=imgsz,
        batch_size=batch_size,
        stride=gs,
        single_cls=True,    # Anti-UAV is single-class
        hyp=DEFAULT_HYP,
        augment=True,
        cache=False,
        rect=False,
        rank=-1,
        workers=4,
        image_weights=False,
        quad=False,
        prefix=colorstr('train: '),
        shuffle=True,
    )
    return loader, dataset


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    device = select_device(args.device)

    # ── Build model ───────────────────────────────────────────────────
    print(f'\n[finetune] Loading pretrained YOLO: {args.weights}')
    model = EvidentialYOLO.from_pretrained(
        args.weights, device, K=args.K, fuse=True
    )
    model.hyp = DEFAULT_HYP

    # ── Freeze backbone, train only new heads ─────────────────────────
    if not args.full_finetune:
        model.freeze_backbone()
    else:
        model.unfreeze_all()
        print('[finetune] Full fine-tune: all parameters trainable.')

    # ── Optional pre-training verification ────────────────────────────
    if args.verify:
        print('\n[finetune] Running pre-training verification …')
        _run_quick_verify(model, args, device)

    # ── Data loader ───────────────────────────────────────────────────
    stride = int(model.stride.max())
    loader, dataset = build_dataloader(
        args.img_list, args.data,
        imgsz=args.imgsz, batch_size=args.batch_size,
        stride=stride, device=device,
    )
    nb = len(loader)
    print(f'[finetune] {len(dataset)} images, {nb} batches/epoch')

    # ── Loss and optimiser ────────────────────────────────────────────
    loss_fn = ComputeEvidentialLoss(
        model, DEFAULT_HYP,
        lambda_var=args.lambda_var,
        lambda_evi=args.lambda_evi,
        K=args.K,
    )
    # Anneal KL over all training steps
    loss_fn._anneal_steps = args.epochs * nb

    params = model.new_params() if not args.full_finetune else model.parameters()
    optimizer = optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * nb, eta_min=args.lr * 0.01
    )

    # ── Training ──────────────────────────────────────────────────────
    os.makedirs(args.save_dir, exist_ok=True)
    best_loss = float('inf')
    history   = []

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_components = torch.zeros(4)

        t0 = time.time()
        for batch_i, (imgs, targets, paths, _) in enumerate(loader):
            imgs    = imgs.to(device).float() / 255.0
            targets = targets.to(device)

            # Forward
            preds = model(imgs)
            if not isinstance(preds, tuple) or len(preds) != 3:
                raise RuntimeError(
                    "Expected (base_raw, var_raw, evi_raw) from model in train mode. "
                    f"Got: {type(preds)}"
                )

            loss, components = loss_fn(preds, targets)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item() / imgs.shape[0]  # per-image
            epoch_components += components.cpu()

        elapsed = time.time() - t0
        avg_comp = epoch_components / nb
        print(
            f'Epoch {epoch+1:3d}/{args.epochs}  '
            f'loss={epoch_loss/nb:.4f}  '
            f'[box={avg_comp[0]:.4f} obj={avg_comp[1]:.4f} '
            f'evi={avg_comp[2]:.4f} var={avg_comp[3]:.4f}]  '
            f'{elapsed:.1f}s'
        )
        history.append({
            'epoch': epoch + 1,
            'loss':  epoch_loss / nb,
            'box':   avg_comp[0].item(),
            'obj':   avg_comp[1].item(),
            'evi':   avg_comp[2].item(),
            'var':   avg_comp[3].item(),
        })

        # Save best
        if epoch_loss / nb < best_loss:
            best_loss = epoch_loss / nb
            save_path = os.path.join(args.save_dir, f'{args.name}_best.pt')
            model.save(save_path)

    # ── Save final ────────────────────────────────────────────────────
    save_path = os.path.join(args.save_dir, f'{args.name}_last.pt')
    model.save(save_path)

    # ── Post-training verification ─────────────────────────────────────
    if args.verify:
        print('\n[finetune] Running post-training verification …')
        _run_quick_verify(model, args, device)

    print(f'\n[finetune] Done. Best loss: {best_loss:.4f}')
    _save_history(history, args)


def _run_quick_verify(model, args, device):
    """Grab a handful of images from the training list and run verify."""
    from lib.models.yolo_evidential.verify import _load_sample_images
    # Use the directory of the first listed image
    with open(args.img_list) as f:
        first_img = f.readline().strip()
    img_dir = os.path.dirname(first_img)
    images = _load_sample_images(img_dir, n=4, imgsz=args.imgsz, device=device)
    run_verification(model, images, device, K=args.K)


def _save_history(history: list, args):
    import csv
    csv_path = os.path.join(args.save_dir, f'{args.name}_history.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    print(f'[finetune] Training history → {csv_path}')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Fine-tune EvidentialYOLO variance + evidence heads.'
    )
    parser.add_argument('--weights',      required=True,
                        help='Path to pretrained YOLOv5 .pt checkpoint.')
    parser.add_argument('--data',         required=True,
                        help='Dataset YAML path.')
    parser.add_argument('--img_list',     required=True,
                        help='Path to train images .txt file.')
    parser.add_argument('--save_dir',     default='/home/ssuttorp/pretrained_models/evidential',
                        help='Directory to save fine-tuned checkpoints.')
    parser.add_argument('--name',         default='evidential_ir',
                        help='Checkpoint filename stem.')
    parser.add_argument('--device',       default='0')
    parser.add_argument('--epochs',       type=int,   default=10)
    parser.add_argument('--batch_size',   type=int,   default=16)
    parser.add_argument('--imgsz',        type=int,   default=640)
    parser.add_argument('--lr',           type=float, default=1e-3)
    parser.add_argument('--lambda_var',   type=float, default=0.1,
                        help='Weight on Gaussian NLL variance loss.')
    parser.add_argument('--lambda_evi',   type=float, default=1.0,
                        help='Weight on Dirichlet evidential loss.')
    parser.add_argument('--K',            type=int,   default=2,
                        help='Number of evidential classes (default 2: UAV/bg).')
    parser.add_argument('--full_finetune', action='store_true',
                        help='Fine-tune the full model (default: heads only).')
    parser.add_argument('--verify',       action='store_true',
                        help='Run numerical verification before and after training.')
    args = parser.parse_args()

    train(args)


if __name__ == '__main__':
    main()
