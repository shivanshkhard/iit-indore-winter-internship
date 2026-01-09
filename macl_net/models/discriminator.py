import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


class PatchDiscriminator(nn.Module):
    """
    Multi-layer PatchGAN Discriminator (STABILIZED)

    Output: N x 1 x H' x W' realism map
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        n_layers: int = 4
    ):
        super().__init__()

        layers = []

        # --------------------------------------------------
        # Initial layer (no normalization)
        # --------------------------------------------------
        layers += [
            spectral_norm(
                nn.Conv2d(
                    in_channels,
                    base_channels,
                    kernel_size=4,
                    stride=2,
                    padding=1
                )
            ),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        # --------------------------------------------------
        # Intermediate layers
        # --------------------------------------------------
        curr_channels = base_channels
        for i in range(1, n_layers):
            next_channels = min(curr_channels * 2, 512)
            stride = 2 if i < n_layers - 1 else 1

            layers += [
                spectral_norm(
                    nn.Conv2d(
                        curr_channels,
                        next_channels,
                        kernel_size=4,
                        stride=stride,
                        padding=1
                    )
                ),
                nn.GroupNorm(
                    num_groups=min(32, next_channels // 2),
                    num_channels=next_channels
                ),
                nn.LeakyReLU(0.2, inplace=True)
            ]

            curr_channels = next_channels

        # --------------------------------------------------
        # Output layer (NO spectral norm)
        # --------------------------------------------------
        layers += [
            nn.Conv2d(
                curr_channels,
                1,
                kernel_size=4,
                stride=1,
                padding=1
            )
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# --------------------------------------------------
# Alias for compatibility
# --------------------------------------------------
Discriminator = PatchDiscriminator
