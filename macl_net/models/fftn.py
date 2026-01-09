import torch
import torch.nn as nn
import torch.nn.functional as F


class FFTN(nn.Module):
    """
    Frequency Feature Transform Normalization (FFTN)

    Stable version:
    - Bounded frequency modulation
    - Normalized frequency encoder
    - Residual-safe behavior
    """

    def __init__(self, num_features, hidden_dim=None, eps=1e-5):
        super().__init__()

        self.num_features = num_features
        self.eps = eps

        if hidden_dim is None:
            hidden_dim = num_features // 2
            hidden_dim = max(hidden_dim, 16)

        # --------------------------------
        # Base affine (spatial)
        # --------------------------------
        self.gamma_s = nn.Parameter(torch.ones(num_features))
        self.beta_s = nn.Parameter(torch.zeros(num_features))

        # --------------------------------
        # Frequency encoder (STABILIZED)
        # --------------------------------
        self.freq_encoder = nn.Sequential(
            nn.Conv2d(num_features, hidden_dim, kernel_size=3, padding=1),
            nn.InstanceNorm2d(hidden_dim),
            nn.ReLU(inplace=True),

            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.InstanceNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )

        self.to_gamma_beta = nn.Conv2d(
            hidden_dim,
            num_features * 2,
            kernel_size=3,
            padding=1
        )

    def forward(self, x: torch.Tensor, freq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:    (N, C, H, W)
            freq: (N, C, h, w)
        """

        # --------------------------------
        # 1. Normalize spatial features
        # --------------------------------
        mean = x.mean(dim=(2, 3), keepdim=True)
        std = x.std(dim=(2, 3), keepdim=True) + self.eps
        x_norm = (x - mean) / std

        gamma_s = self.gamma_s.view(1, -1, 1, 1)
        beta_s = self.beta_s.view(1, -1, 1, 1)

        out = gamma_s * x_norm + beta_s

        # --------------------------------
        # 2. Encode frequency features
        # --------------------------------
        if freq.shape[2:] != x.shape[2:]:
            freq = F.interpolate(
                freq,
                size=x.shape[2:],
                mode="nearest"
            )

        freq_feat = self.freq_encoder(freq)
        gamma_f, beta_f = self.to_gamma_beta(freq_feat).chunk(2, dim=1)

        # --------------------------------
        # 3. Bounded modulation (KEY FIX)
        # --------------------------------
        gamma_f = 1.0 + 0.1 * torch.tanh(gamma_f)
        beta_f = 0.1 * torch.tanh(beta_f)

        out = out * gamma_f + beta_f
        return out
