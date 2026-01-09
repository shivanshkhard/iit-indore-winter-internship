import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------
# Haar Discrete Wavelet Transform (DWT)
# --------------------------------------------------
class HaarDWT(nn.Module):
    """
    Haar Discrete Wavelet Transform.

    Decomposes input feature map into:
        - Low-frequency (LL)
        - High-frequency (HF = LH + HL + HH)
    """

    def __init__(self):
        super().__init__()

        # Haar filters
        ll = torch.tensor([[0.5, 0.5],
                           [0.5, 0.5]])

        lh = torch.tensor([[-0.5, -0.5],
                           [ 0.5,  0.5]])

        hl = torch.tensor([[-0.5,  0.5],
                           [-0.5,  0.5]])

        hh = torch.tensor([[ 0.5, -0.5],
                           [-0.5,  0.5]])

        # Register as buffers (move with device, no gradients)
        self.register_buffer("ll", ll[None, None])
        self.register_buffer("lh", lh[None, None])
        self.register_buffer("hl", hl[None, None])
        self.register_buffer("hh", hh[None, None])

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: Input tensor (N, C, H, W)

        Returns:
            ll: Low-frequency component (N, C, H/2, W/2)
            hf: High-frequency component (N, C, H/2, W/2)
        """
        N, C, H, W = x.shape

        # Apply filters per channel
        ll = F.conv2d(x, self.ll.repeat(C, 1, 1, 1),
                      stride=2, groups=C)
        lh = F.conv2d(x, self.lh.repeat(C, 1, 1, 1),
                      stride=2, groups=C)
        hl = F.conv2d(x, self.hl.repeat(C, 1, 1, 1),
                      stride=2, groups=C)
        hh = F.conv2d(x, self.hh.repeat(C, 1, 1, 1),
                      stride=2, groups=C)

        # Combine high-frequency components
        hf = lh + hl + hh

        return ll, hf


# --------------------------------------------------
# Haar Inverse Discrete Wavelet Transform (IDWT)
# --------------------------------------------------
class HaarIDWT(nn.Module):
    """
    Haar Inverse Discrete Wavelet Transform.

    Reconstructs spatial feature map from:
        - Low-frequency (LL)
        - High-frequency (HF)
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        ll: torch.Tensor,
        hf: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            ll: Low-frequency component (N, C, H, W)
            hf: High-frequency component (N, C, H, W)

        Returns:
            Reconstructed feature map (N, C, 2H, 2W)
        """
        # Simple and stable reconstruction
        x = ll + hf
        x = F.interpolate(
            x,
            scale_factor=2,
            mode="nearest"
        )
        return x
