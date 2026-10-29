import torch
import torch.nn.functional as F

from pysot.core.config import cfg
from pysot.models.loss_car import make_siamcar_loss_evaluator


def dirichlet_ece_loss(evidence, y_onehot, K: int, lam: float = 1.0):
    """Evidential classification loss (Sensoy et al. 2018).

    Self-contained port of the YOLO evidential loss — no EDTC imports.

    Args:
        evidence: (N, K) ReLU-activated evidence values, e_k >= 0
        y_onehot: (N, K) one-hot ground-truth labels
        K:        number of classes
        lam:      KL regularisation weight, annealed 0 → 1

    Loss:
        α = evidence + 1                 Dirichlet parameters
        S = Σ α                          precision
        ECE = Σ_k y_k [ψ(S) − ψ(α_k)]  expected cross-entropy under Dir(α)

        ã_k = y_k + (1 − y_k)·α_k       true-class evidence removed (no penalty
                                          for correct confidence)
        KL  = lgamma(S̃) − lgamma(K)
              − Σ lgamma(ã_k)
              + Σ (ã_k−1)·[ψ(ã_k) − ψ(S̃)]

        return mean(ECE + lam · KL)
    """
    alpha   = evidence + 1.0                              # (N, K)
    S       = alpha.sum(dim=-1, keepdim=True)             # (N, 1)

    # Expected cross-entropy
    ece = (y_onehot * (torch.digamma(S) - torch.digamma(alpha))).sum(dim=-1)  # (N,)

    # KL divergence toward Dir(1,...,1), with true-class evidence removed
    alpha_tilde = y_onehot + (1.0 - y_onehot) * alpha    # (N, K)
    S_tilde     = alpha_tilde.sum(dim=-1, keepdim=True)   # (N, 1)

    K_tensor = torch.tensor(K, dtype=alpha.dtype, device=alpha.device)
    kl = (
        torch.lgamma(S_tilde)
        - torch.lgamma(K_tensor)
        - torch.lgamma(alpha_tilde).sum(dim=-1, keepdim=True)
        + ((alpha_tilde - 1.0) * (
            torch.digamma(alpha_tilde) - torch.digamma(S_tilde)
        )).sum(dim=-1, keepdim=True)
    ).squeeze(-1)                                          # (N,)

    return (ece + lam * kl).mean()


class SiamCARStage2Loss:
    """EDL-only loss for SiamCAR Stage 2.

    Applies the Dirichlet ECE loss to both positive (UAV) and negative
    (background) grid positions with equal 0.5 / 0.5 weighting — matching
    the existing select_cross_entropy_loss behaviour exactly.

    Class convention (K=2):
        class 0 = background  → y_onehot = [1, 0]
        class 1 = UAV         → y_onehot = [0, 1]

    Unlike YOLO Stage 2 (positives only), both pos and neg are required
    here because SiamCAR is binary: the head must learn both UAV evidence
    AND background evidence to produce calibrated Dirichlet beliefs.

    KL annealing:
        _kl_weight = min(1.0, _step / _anneal_steps)
        _step incremented once per __call__
        Set _anneal_steps = min(10, epochs) * batches_per_epoch before training.

    Args:
        K:          number of evidential classes (default 2)
        lambda_evi: overall weight on the EDL loss term (default 1.0)
    """

    def __init__(self, K: int = 2, lambda_evi: float = 1.0):
        self.K          = K
        self.lambda_evi = lambda_evi
        self._loss_eval = make_siamcar_loss_evaluator(cfg)
        self._step         = 0
        self._anneal_steps = 1000   # overridden by trainer before first call

    @property
    def _kl_weight(self) -> float:
        return min(1.0, self._step / max(self._anneal_steps, 1))

    def __call__(self, outputs: dict) -> dict:
        """
        Args:
            outputs: dict with keys
                evi_raw   (B, K, H, W)  raw evidence (pre-ReLU) from cls_evi
                locations (H*W, 2)
                label_cls (B, H*W)      injected by training loop
                label_loc (B, 4)        injected by training loop (GT bboxes)
                (cls, loc, cen, log_var also present but not used here)

        Returns:
            dict: evi_loss (scalar), kl_weight (float)
        """
        kl_w      = self._kl_weight
        K         = self.K
        device    = outputs['evi_raw'].device

        # Get positive/negative position assignment via SiamCAR target matching.
        label_cls_proc, _ = self._loss_eval.prepare_targets(
            outputs['locations'],
            outputs['label_cls'],
            outputs['label_loc'],
        )
        # label_cls_proc: (B, H*W) — 0=background, 1=UAV (inside GT box)

        labels_flat = label_cls_proc.view(-1)                          # (B*H*W,)
        evi_flat    = (outputs['evi_raw']
                       .permute(0, 2, 3, 1).contiguous()
                       .view(-1, K))                                   # (B*H*W, K)

        pos_inds = torch.nonzero(labels_flat > 0).squeeze(1)          # UAV positions
        neg_inds = torch.nonzero(labels_flat == 0).squeeze(1)         # background

        # --- Positive loss (class 1 = UAV) ---
        if pos_inds.numel() > 0:
            y_pos = torch.zeros(pos_inds.numel(), K, device=device)
            y_pos[:, 1] = 1.0
            e_pos    = F.relu(evi_flat[pos_inds])
            loss_pos = dirichlet_ece_loss(e_pos, y_pos, K, lam=kl_w)
        else:
            loss_pos = evi_flat.sum() * 0.0   # zero gradient, avoids NaN

        # --- Negative loss (class 0 = background) ---
        if neg_inds.numel() > 0:
            y_neg = torch.zeros(neg_inds.numel(), K, device=device)
            y_neg[:, 0] = 1.0
            e_neg    = F.relu(evi_flat[neg_inds])
            loss_neg = dirichlet_ece_loss(e_neg, y_neg, K, lam=kl_w)
        else:
            loss_neg = evi_flat.sum() * 0.0

        # Equal weighting mirrors select_cross_entropy_loss (0.5 pos + 0.5 neg)
        evi_loss = self.lambda_evi * (0.5 * loss_pos + 0.5 * loss_neg)

        self._step += 1

        return {'evi_loss': evi_loss, 'kl_weight': kl_w}
