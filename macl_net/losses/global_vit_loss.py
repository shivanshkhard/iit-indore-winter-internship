import torch
import torch.nn as nn
import torch.nn.functional as F


class GlobalViTLoss(nn.Module):
    """
    Global Semantic Consistency Loss using ViT features (STABILIZED).

    Encourages global structural similarity between
    generated and real images.
    """

    def __init__(
        self,
        reduction: str = "mean",
        temperature: float = 0.1   # 🔑 smoothing factor
    ):
        """
        Args:
            reduction (str): 'mean' or 'sum'
            temperature (float): temperature scaling for stability
        """
        super().__init__()
        self.reduction = reduction
        self.temperature = temperature

    def forward(
        self,
        feat_fake: torch.Tensor,
        feat_real: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            feat_fake: Global ViT features from fake image (N, C)
            feat_real: Global ViT features from real image (N, C)

        Returns:
            Scalar loss
        """

        # ---------------------------------------------
        # Normalize features (cosine space)
        # ---------------------------------------------
        feat_fake = F.normalize(feat_fake, dim=1)
        feat_real = F.normalize(feat_real, dim=1)

        # ---------------------------------------------
        # Cosine similarity
        # ---------------------------------------------
        cos_sim = torch.sum(feat_fake * feat_real, dim=1)
        cos_sim = cos_sim.clamp(-1.0, 1.0)  # numerical safety

        loss = (1.0 - cos_sim) / self.temperature

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            raise ValueError(
                f"Unknown reduction type: {self.reduction}"
            )
