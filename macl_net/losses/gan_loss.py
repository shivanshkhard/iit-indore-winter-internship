import torch
import torch.nn as nn


class GANLoss(nn.Module):
    """
    GAN Loss wrapper (STABILIZED).

    Default: Least Squares GAN (LSGAN)
    - Uses one-sided label smoothing
    - Compatible with PatchGAN discriminators
    """

    def __init__(
        self,
        gan_mode: str = "lsgan",
        real_label: float = 0.9,   # 🔑 smoothed real label
        fake_label: float = 0.0
    ):
        super().__init__()

        self.gan_mode = gan_mode
        self.real_label = real_label
        self.fake_label = fake_label

        if gan_mode == "lsgan":
            self.loss = nn.MSELoss()
        elif gan_mode == "vanilla":
            self.loss = nn.BCEWithLogitsLoss()
        else:
            raise NotImplementedError(
                f"GAN mode '{gan_mode}' not implemented"
            )

    def get_target_tensor(
        self,
        prediction: torch.Tensor,
        target_is_real: bool
    ) -> torch.Tensor:
        """
        Create (or reuse) target tensor with same size as prediction.
        """
        target_value = self.real_label if target_is_real else self.fake_label
        return torch.full_like(prediction, target_value)

    def forward(
        self,
        prediction: torch.Tensor,
        target_is_real: bool
    ) -> torch.Tensor:
        """
        Compute GAN loss.

        Args:
            prediction: discriminator output (N, 1, H, W)
            target_is_real: True for real, False for fake

        Returns:
            Scalar loss
        """
        target_tensor = self.get_target_tensor(
            prediction,
            target_is_real
        )

        loss = self.loss(prediction, target_tensor)
        return loss
