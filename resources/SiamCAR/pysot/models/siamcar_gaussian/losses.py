import torch
import torch.nn.functional as F

from pysot.core.config import cfg
from pysot.models.loss_car import make_siamcar_loss_evaluator


def gaussian_nll_ltrb(loc_pred, log_var, loc_target, reduction='mean'):
    """
    Gaussian NLL over 4 ltrb pixel-distance coordinates.

    Args:
        loc_pred:   (N, 4) predicted ltrb means, post-exp() activation
        log_var:    (N, 4) predicted log-variances (unclamped)
        loc_target: (N, 4) ground-truth ltrb distances
        reduction:  'mean' | 'sum' | 'none'
    """
    log_var = log_var.clamp(-6, 6)
    sq_err  = (loc_pred - loc_target) ** 2
    nll     = 0.5 * log_var + 0.5 * sq_err * torch.exp(-log_var)
    if reduction == 'mean':
        return nll.mean()
    elif reduction == 'sum':
        return nll.sum()
    return nll


class SiamCARGaussianLoss:
    """
    SiamCAR Stage 1 loss = baseline losses (cls + loc + cen) + λ_var * Gaussian NLL.

    The Gaussian NLL is computed only over positive anchor positions (same mask
    as the IoU regression loss), keeping the variance head focused on foreground.

    Args:
        lambda_var: weight on the Gaussian NLL term (default 0.1)
    """

    def __init__(self, lambda_var: float = 0.1):
        self.base_loss  = make_siamcar_loss_evaluator(cfg)
        self.lambda_var = lambda_var

    def __call__(self, outputs):
        """
        Args:
            outputs: dict with keys cls, loc, cen, log_var, locations,
                     label_cls, label_loc  (last two injected by training loop)

        Returns:
            dict: total_loss, cls_loss, loc_loss, cen_loss, var_loss
        """
        locations = outputs['locations']
        cls       = outputs['cls']       # (B, 2, H, W) raw logits
        loc       = outputs['loc']       # (B, 4, H, W) post-exp ltrb
        cen       = outputs['cen']       # (B, 1, H, W)
        log_var   = outputs['log_var']   # (B, 4, H, W)
        label_cls = outputs['label_cls'] # (B, 1, H, W) or (B, H*W)
        label_loc = outputs['label_loc'] # (B, 4) GT bboxes

        # Replicate ModelBuilder.log_softmax()
        b, a2, h, w = cls.size()
        cls_ls = cls.view(b, 2, a2 // 2, h, w)
        cls_ls = cls_ls.permute(0, 2, 3, 4, 1).contiguous()
        cls_ls = F.log_softmax(cls_ls, dim=4)

        # Baseline losses (unchanged from original SiamCAR)
        cls_loss, loc_loss, cen_loss = self.base_loss(
            locations, cls_ls, loc, cen, label_cls, label_loc
        )

        # Gaussian NLL over positive positions.
        # prepare_targets is called a second time (same args, CPU-side, no side effects
        # beyond label modification already done in the first call above).
        label_cls_proc, reg_targets = self.base_loss.prepare_targets(
            locations, label_cls, label_loc
        )

        labels_flatten      = label_cls_proc.view(-1)
        loc_flatten         = loc.permute(0, 2, 3, 1).contiguous().view(-1, 4)
        log_var_flatten     = log_var.permute(0, 2, 3, 1).contiguous().view(-1, 4)
        reg_targets_flatten = reg_targets.view(-1, 4)

        pos_inds = torch.nonzero(labels_flatten > 0).squeeze(1)

        if pos_inds.numel() > 0:
            var_loss = gaussian_nll_ltrb(
                loc_flatten[pos_inds],
                log_var_flatten[pos_inds],
                reg_targets_flatten[pos_inds],
            )
        else:
            # No positive positions — zero gradient, avoids NaN
            var_loss = log_var_flatten.sum() * 0.0

        total_loss = (
            cfg.TRAIN.CLS_WEIGHT * cls_loss
            + cfg.TRAIN.LOC_WEIGHT * loc_loss
            + cfg.TRAIN.CEN_WEIGHT * cen_loss
            + self.lambda_var * var_loss
        )

        return {
            'total_loss': total_loss,
            'cls_loss':   cls_loss,
            'loc_loss':   loc_loss,
            'cen_loss':   cen_loss,
            'var_loss':   var_loss,
        }
