"""Losses used by CARF-Net.

* `AdjustedCE`   - logit-adjusted cross entropy (Menon et al., 2021) with soft
                   targets so it also works under mixup.
* `WeightedCE`   - inverse-frequency weighting, kept for ablations.
* `WeightedSupCoL` - AU-similarity weighted supervised contrastive loss.
* `mixup`        - applied to both face halves with the same lambda.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdjustedCE(nn.Module):
    """Cross entropy with a class-prior correction on the logits.

    During training the logits are shifted by tau * log(prior); at inference the
    raw logits are used. This optimises the balanced error rate, which is what
    UAR and macro F1 measure.
    """

    def __init__(self, class_counts, tau=1.0, label_smoothing=0.1, gamma=0.0):
        super().__init__()
        counts = torch.as_tensor(class_counts, dtype=torch.float32).clamp(min=1.0)
        prior = counts / counts.sum()
        self.register_buffer("adjustment", tau * torch.log(prior))
        self.label_smoothing = label_smoothing
        self.gamma = gamma
        self.num_classes = len(counts)

    def _soft_target(self, target):
        if target.dim() == 2:
            return target
        onehot = F.one_hot(target, self.num_classes).float()
        if self.label_smoothing <= 0:
            return onehot
        smooth = self.label_smoothing / self.num_classes
        return onehot * (1.0 - self.label_smoothing) + smooth

    def forward(self, logits, target):
        logits = logits + self.adjustment.to(logits.device)
        log_prob = F.log_softmax(logits, dim=-1)
        soft = self._soft_target(target)
        if self.gamma > 0:
            prob = log_prob.exp()
            log_prob = log_prob * (1.0 - prob).clamp(min=0.0) ** self.gamma
        return -(soft * log_prob).sum(dim=-1).mean()


class WeightedCE(nn.Module):
    """Inverse max(n)/n_c weighting, as in the original repository."""

    def __init__(self, class_counts, label_smoothing=0.1):
        super().__init__()
        counts = torch.as_tensor(class_counts, dtype=torch.float32).clamp(min=1.0)
        self.register_buffer("weight", counts.max() / counts)
        self.label_smoothing = label_smoothing

    def forward(self, logits, target):
        if target.dim() == 2:
            log_prob = F.log_softmax(logits, dim=-1)
            weight = self.weight.to(logits.device).unsqueeze(0)
            return -(target * weight * log_prob).sum(dim=-1).mean()
        return F.cross_entropy(logits, target, weight=self.weight.to(logits.device),
                               label_smoothing=self.label_smoothing)


class WeightedSupCoL(nn.Module):
    """Supervised contrastive loss whose positives are weighted by AU overlap.

    Adapted from https://github.com/GuillaumeErhard/Supervised_contrastive_loss_pytorch
    (credit: Guillaume Erhard), as used by Ruan et al. (2022).
    """

    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, projections, targets):
        if len(targets) == 1:
            return None
        device = targets.device
        mask_anchor_out = (1 - torch.eye(projections.shape[0])).to(device)
        cardinality = targets.sum(dim=-1)
        logits = torch.div(torch.matmul(projections, projections.T), self.temperature)
        exp_logits = torch.exp(logits) + 1e-5
        denominator = (exp_logits * mask_anchor_out).sum(dim=1, keepdim=True)
        log_prob = -torch.log(exp_logits / denominator)
        per_sample = (log_prob * targets).sum(dim=1) / cardinality.clamp(min=1)
        return per_sample.mean()


def au_similarity(au):
    """Pairwise Jaccard similarity between AU multi-hot vectors."""
    bool_au = torch.clone(au.bool()).unsqueeze(1).repeat(1, au.shape[0], 1)
    union = torch.logical_or(bool_au, bool_au.permute(1, 0, 2)).sum(dim=-1)
    inter = torch.logical_and(bool_au, bool_au.permute(1, 0, 2)).sum(dim=-1)
    return inter / union.clamp(min=1)


def build_criterion(kind, class_counts, device, tau=1.0, label_smoothing=0.1, gamma=0.0):
    if kind == "logit_adjusted":
        return AdjustedCE(class_counts, tau, label_smoothing, gamma).to(device)
    if kind == "weighted_ce":
        return WeightedCE(class_counts, label_smoothing).to(device)
    if kind == "plain_ce":
        return AdjustedCE(class_counts, 0.0, label_smoothing, gamma).to(device)
    raise ValueError(f"Unknown loss: {kind}")


def mixup(eyes, mouth, labels, num_classes, alpha, smoothing=0.1):
    """Mixup on both halves with a shared lambda and permutation."""
    smooth = smoothing / num_classes
    onehot = F.one_hot(labels, num_classes).float()
    if smoothing > 0:
        onehot = onehot * (1.0 - smoothing) + smooth
    if alpha <= 0:
        return eyes, mouth, onehot

    lam = float(torch.distributions.Beta(alpha, alpha).sample())
    lam = max(lam, 1.0 - lam)   # keep the primary sample dominant
    perm = torch.randperm(eyes.size(0), device=eyes.device)
    eyes = lam * eyes + (1.0 - lam) * eyes[perm]
    mouth = lam * mouth + (1.0 - lam) * mouth[perm]
    target = lam * onehot + (1.0 - lam) * onehot[perm]
    return eyes, mouth, target
