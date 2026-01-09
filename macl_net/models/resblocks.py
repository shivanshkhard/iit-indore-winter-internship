import torch
import torch.nn as nn
import torch.nn.functional as F

from models.fftn import FFTN


# --------------------------------------------------
# 1. Low-Frequency FFTN Residual Block
# --------------------------------------------------
class LowFFTNResBlock(nn.Module):
    """
    Low-frequency FFTN Residual Block

    Focus:
        - Global structure
        - Semantic consistency
    """

    def __init__(self, channels: int, residual_scale: float = 0.1):
        super().__init__()

        self.residual_scale = residual_scale

        self.fftn = FFTN(channels)

        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(channels)
        )

    def forward(self, x: torch.Tensor, freq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:    spatial feature (N, C, H, W)
            freq: low-frequency guidance (N, C, Hf, Wf)
        """

        # 🔒 SAFETY: enforce spatial alignment
        if freq.shape[2:] != x.shape[2:]:
            freq = F.interpolate(freq, size=x.shape[2:], mode="nearest")

        out = self.fftn(x, freq)
        out = self.conv(out)

        # 🔒 STABILITY: scaled residual
        return x + self.residual_scale * out


# --------------------------------------------------
# 2. High-Frequency FFTN Residual Block
# --------------------------------------------------
class HighFFTNResBlock(nn.Module):
    """
    High-frequency FFTN Residual Block

    Focus:
        - Texture
        - Local detail refinement
    """

    def __init__(self, channels: int, residual_scale: float = 0.1):
        super().__init__()

        self.residual_scale = residual_scale

        self.fftn = FFTN(channels)

        self.conv = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(channels),

            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(channels)
        )

    def forward(self, x: torch.Tensor, freq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:    spatial feature (N, C, H, W)
            freq: high-frequency guidance (N, C, Hf, Wf)
        """

        # 🔒 SAFETY: enforce spatial alignment
        if freq.shape[2:] != x.shape[2:]:
            freq = F.interpolate(freq, size=x.shape[2:], mode="nearest")

        out = self.fftn(x, freq)
        out = self.conv(out)

        # 🔒 STABILITY: scaled residual
        return x + self.residual_scale * out
