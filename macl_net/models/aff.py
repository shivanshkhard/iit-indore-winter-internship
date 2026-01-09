import torch
import torch.nn as nn
import torch.nn.functional as F


class AFF(nn.Module):
    """
    Adaptive Frequency Fusion (AFF)

    Stable version:
    - Bounded fusion weights
    - Normalized high-frequency input
    - Residual-safe fusion
    """

    def __init__(self, channels: int, hidden_channels: int = None):
        super().__init__()

        if hidden_channels is None:
            hidden_channels = max(channels // 2, 16)

        # ---------------------------------------------
        # Fusion network to predict alpha
        # ---------------------------------------------
        self.fusion_net = nn.Sequential(
            nn.Conv2d(channels * 2, hidden_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(hidden_channels),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(hidden_channels, channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, ll: torch.Tensor, hf: torch.Tensor) -> torch.Tensor:
        """
        Args:
            ll: Low-frequency features (N, C, H, W)
            hf: High-frequency features (N, C, H, W)
        """

        if ll.shape != hf.shape:
            raise ValueError(
                f"LL and HF must have same shape, got {ll.shape} vs {hf.shape}"
            )

        # ---------------------------------------------
        # Normalize HF (CRITICAL FIX)
        # ---------------------------------------------
        hf_mean = hf.mean(dim=(2, 3), keepdim=True)
        hf_std = hf.std(dim=(2, 3), keepdim=True) + 1e-5
        hf_norm = (hf - hf_mean) / hf_std

        # ---------------------------------------------
        # Predict bounded fusion weight alpha
        # ---------------------------------------------
        fusion_input = torch.cat([ll, hf_norm], dim=1)
        alpha = self.fusion_net(fusion_input)

        # Bound alpha away from extremes
        alpha = 0.1 + 0.8 * alpha

        # ---------------------------------------------
        # Residual-safe adaptive fusion
        # ---------------------------------------------
        fused = ll + alpha * (hf_norm - ll)

        return fused
