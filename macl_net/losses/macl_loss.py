import torch
import torch.nn as nn
import torch.nn.functional as F


class MACLLoss(nn.Module):
    """
    Margin Adaptive Contrastive Loss (STABILIZED)

    - Patch-wise contrastive learning
    - Bounded adaptive margin
    - Numerically stable
    """

    def __init__(
        self,
        temperature: float = 0.07,
        margin_scale: float = 0.5,
        eps: float = 1e-8
    ):
        super().__init__()
        self.temperature = temperature
        self.margin_scale = margin_scale
        self.eps = eps

    def forward(
        self,
        feat_fake: torch.Tensor,
        feat_real: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            feat_fake: [N, P, C] or [N, C]
            feat_real: [N, P, C] or [N, C]
        """

        # ---------------------------------------------
        # Case 1: Global features [N, C]
        # ---------------------------------------------
        if feat_fake.dim() == 2:
            feat_fake = F.normalize(feat_fake, dim=1)
            feat_real = F.normalize(feat_real, dim=1)

            sim = torch.sum(feat_fake * feat_real, dim=1)
            loss = (1.0 - sim) / self.temperature
            return loss.mean()

        # ---------------------------------------------
        # Case 2: Patch features [N, P, C]
        # ---------------------------------------------
        N, P, C = feat_fake.shape

        feat_fake = F.normalize(feat_fake, dim=2)
        feat_real = F.normalize(feat_real, dim=2)

        total_loss = 0.0

        for i in range(N):
            f_fake = feat_fake[i]  # [P, C]
            f_real = feat_real[i]  # [P, C]

            # Similarity matrix (P x P)
            sim = torch.matmul(f_fake, f_real.t()) / self.temperature

            # Positive similarities
            pos_sim = torch.diag(sim)

            # Adaptive margin (bounded)
            margin = self.margin_scale * (1.0 - pos_sim)
            margin = margin.clamp(min=0.0, max=self.margin_scale)

            logits = sim - margin.unsqueeze(1)

            # Numerical stabilization
            logits = logits - logits.max(dim=1, keepdim=True)[0]

            labels = torch.arange(P, device=logits.device)

            loss_i = F.cross_entropy(logits, labels)
            total_loss += loss_i

        return total_loss / N
