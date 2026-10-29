"""
Numerical verification for the evidential + Gaussian uncertainty heads (Phase 3.2).

Run this script BEFORE full training to confirm that the model's outputs
satisfy the theoretical guarantees:

  1. Evidence values e_k ≥ 0  (ReLU guarantee — should be exact)
  2. Uncertainty mass u ∈ (0, 1]
  3. Belief + uncertainty masses sum to 1:  Σ b_k + u = 1
  4. Distribution is not collapsed:
       - u ≈ 0 everywhere → model is overconfident (undertrained / diverged)
       - u ≈ 1 everywhere → model is uninformative (underfitted / init only)
  5. Spatial variances σ² > 0  (exp guarantee — should be exact)
  6. Normalised variances σ̂² are finite and positive

Usage::
    # From the EDTC root directory:
    python lib/models/yolo_evidential/verify.py \
        --weights /path/to/best_clean.pt \
        --data    /path/to/antiuav_clean.yaml \
        --img_dir /path/to/Anti-UAV-RGBT/test/<seq>/ir_frames \
        --device  0

Or call programmatically::
    from lib.models.yolo_evidential.verify import run_verification
    report = run_verification(model, images_tensor, device)
"""

import os
import sys
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np

_edtc_root = Path(__file__).resolve().parents[3]
_yolo_root = _edtc_root / 'yolov5'
for p in [str(_edtc_root), str(_yolo_root)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from yolov5.utils.torch_utils import select_device
from lib.models.yolo_evidential.model import EvidentialYOLO


# ---------------------------------------------------------------------------
# Core verification logic
# ---------------------------------------------------------------------------

def verify_evidential_outputs(unc_scales: list, K: int = 2, tol: float = 1e-4):
    """Check numerical guarantees on a single batch's uncertainty outputs.

    Args:
        unc_scales: list of per-scale uncertainty dicts as returned by
                    EvidentialDetect.forward() in inference mode.
        K:          number of evidential classes.
        tol:        numerical tolerance for sum-to-one check.

    Returns:
        dict with keys:
          'e_min'        : min evidence value across all scales/anchors
          'u_min'        : min uncertainty mass
          'u_max'        : max uncertainty mass
          'u_mean'       : mean uncertainty mass
          'sum_error_max': max |Σb_k + u - 1|  (should be < tol)
          'sigma2_min'   : min absolute variance
          'sigma2_norm_min': min normalised variance
          'warnings'     : list of human-readable warning strings
    """
    warnings = []
    e_mins, u_mins, u_maxes, u_means = [], [], [], []
    sum_errs = []
    sigma2_mins, sigma2_norm_mins = [], []

    for scale_dict in unc_scales:
        u          = scale_dict['u']           # (bs, na, H, W, 1)
        b          = scale_dict['b']           # (bs, na, H, W, K)
        sigma2     = scale_dict['sigma2']      # (bs, na, H, W, 4)
        sigma2_norm = scale_dict['sigma2_norm']

        # Recover evidence from beliefs: e_k = b_k * S, S = K / u
        # → e_k = b_k * K / u
        S = K / (u + 1e-12)
        e = b * S - 1.0    # α_k - 1 = e_k  (since α_k = e_k + 1, b_k = e_k/S → e_k = b_k*S - 1... wait)
        # Actually: b_k = e_k / S, so e_k = b_k * S.  And S = K/u.
        e = b * S           # approximate evidence values

        e_mins.append(e.min().item())
        u_mins.append(u.min().item())
        u_maxes.append(u.max().item())
        u_means.append(u.mean().item())

        # Sum-to-one: Σ b_k + u should = 1
        sum_val    = b.sum(dim=-1, keepdim=True) + u   # (bs, na, H, W, 1)
        sum_err    = (sum_val - 1.0).abs()
        sum_errs.append(sum_err.max().item())

        sigma2_mins.append(sigma2.min().item())
        sigma2_norm_mins.append(sigma2_norm.min().item())

    result = {
        'e_min':           min(e_mins),
        'u_min':           min(u_mins),
        'u_max':           max(u_maxes),
        'u_mean':          float(np.mean(u_means)),
        'sum_error_max':   max(sum_errs),
        'sigma2_min':      min(sigma2_mins),
        'sigma2_norm_min': min(sigma2_norm_mins),
        'warnings':        warnings,
    }

    # ── Checks ─────────────────────────────────────────────────────────
    if result['e_min'] < -tol:
        warnings.append(
            f"[FAIL] Negative evidence detected: e_min = {result['e_min']:.6f}. "
            "ReLU activation may not be applied correctly."
        )
    else:
        warnings.append(f"[OK] Evidence ≥ 0: e_min = {result['e_min']:.6f}")

    if not (0.0 - tol < result['u_min'] and result['u_max'] < 1.0 + tol):
        warnings.append(
            f"[FAIL] Uncertainty out of (0,1]: u ∈ [{result['u_min']:.4f}, {result['u_max']:.4f}]"
        )
    else:
        warnings.append(
            f"[OK] Uncertainty in (0,1]: u ∈ [{result['u_min']:.4f}, {result['u_max']:.4f}]"
        )

    if result['sum_error_max'] > tol:
        warnings.append(
            f"[FAIL] Σb_k + u ≠ 1 (max error = {result['sum_error_max']:.2e})"
        )
    else:
        warnings.append(
            f"[OK] Σb_k + u = 1 (max error = {result['sum_error_max']:.2e})"
        )

    # Collapsed distributions
    OVERCONF_THRESHOLD   = 0.05   # u_mean below this → overconfident
    UNINFORMATIVE_THRESHOLD = 0.9  # u_mean above this → uninformative
    if result['u_mean'] < OVERCONF_THRESHOLD:
        warnings.append(
            f"[WARN] Distribution may be overconfident: u_mean = {result['u_mean']:.4f} "
            f"(threshold = {OVERCONF_THRESHOLD}). Check for mode collapse."
        )
    elif result['u_mean'] > UNINFORMATIVE_THRESHOLD:
        warnings.append(
            f"[WARN] Distribution is uninformative: u_mean = {result['u_mean']:.4f} "
            f"(threshold = {UNINFORMATIVE_THRESHOLD}). Expected at initialisation; "
            "should decrease after training."
        )
    else:
        warnings.append(
            f"[OK] Uncertainty not collapsed: u_mean = {result['u_mean']:.4f}"
        )

    if result['sigma2_min'] <= 0.0:
        warnings.append(
            f"[FAIL] Non-positive variance: σ²_min = {result['sigma2_min']:.6f}"
        )
    else:
        warnings.append(f"[OK] σ² > 0: σ²_min = {result['sigma2_min']:.6f}")

    if result['sigma2_norm_min'] <= 0.0:
        warnings.append(
            f"[FAIL] Non-positive normalised variance: σ̂²_min = {result['sigma2_norm_min']:.6f}"
        )
    else:
        warnings.append(
            f"[OK] σ̂² > 0: σ̂²_min = {result['sigma2_norm_min']:.6f}"
        )

    return result


def run_verification(model: EvidentialYOLO, images: torch.Tensor,
                     device: torch.device, K: int = 2) -> dict:
    """Run verification on a batch of images.

    Args:
        model:   EvidentialYOLO in eval mode.
        images:  (bs, 3, H, W) float tensor, values in [0, 1].
        device:  torch.device.
        K:       number of evidential classes.

    Returns:
        Verification report dict (same as verify_evidential_outputs).
    """
    model.eval()
    with torch.no_grad():
        images = images.to(device)
        out = model(images)

    if len(out) < 5:
        raise RuntimeError(
            "Model did not return uncertainty outputs. "
            "Ensure model.eval() is called before verification."
        )

    _nms_preds, _base_raw, _var_raw, _evi_raw, unc_scales = out
    report = verify_evidential_outputs(unc_scales, K=K)

    print("\n── Evidential Uncertainty Verification ──────────────────────")
    for msg in report['warnings']:
        print(" ", msg)
    print(f"  u  : mean={report['u_mean']:.4f}, "
          f"min={report['u_min']:.4f}, max={report['u_max']:.4f}")
    print(f"  σ² : min={report['sigma2_min']:.6f}")
    print(f"  σ̂² : min={report['sigma2_norm_min']:.6f}")
    print("─────────────────────────────────────────────────────────────\n")
    return report


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _load_sample_images(img_dir: str, n: int = 4, imgsz: int = 640,
                         device: torch.device = torch.device('cpu')):
    """Load a few images from a directory for verification."""
    import cv2
    from yolov5.utils.augmentations import letterbox

    paths = sorted(Path(img_dir).glob('*.jpg'))[:n]
    if not paths:
        raise FileNotFoundError(f"No .jpg images found in {img_dir}")

    imgs = []
    for p in paths:
        im = cv2.imread(str(p))
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = letterbox(im, imgsz, stride=32, auto=True)[0]
        im = im.transpose(2, 0, 1)[::-1].copy()
        imgs.append(torch.from_numpy(im).float() / 255.0)

    return torch.stack(imgs).to(device)


def main():
    parser = argparse.ArgumentParser(
        description='Verify evidential + Gaussian outputs of EvidentialYOLO.'
    )
    parser.add_argument('--weights',  required=True,
                        help='Pretrained YOLOv5 .pt path (e.g. best_clean.pt).')
    parser.add_argument('--data',     required=True,
                        help='Dataset YAML path (for class names).')
    parser.add_argument('--img_dir',  required=True,
                        help='Directory containing .jpg images for verification.')
    parser.add_argument('--n',        type=int, default=4,
                        help='Number of images to use (default 4).')
    parser.add_argument('--imgsz',    type=int, default=640)
    parser.add_argument('--device',   default='0')
    parser.add_argument('--K',        type=int, default=2,
                        help='Number of evidential classes (default 2).')
    args = parser.parse_args()

    device = select_device(args.device)
    print(f'[verify] Building EvidentialYOLO from {args.weights}')
    model = EvidentialYOLO.from_pretrained(args.weights, device, K=args.K)

    images = _load_sample_images(args.img_dir, n=args.n,
                                  imgsz=args.imgsz, device=device)
    print(f'[verify] Running on {images.shape[0]} images '
          f'(shape {tuple(images.shape)})')

    run_verification(model, images, device, K=args.K)


if __name__ == '__main__':
    main()
