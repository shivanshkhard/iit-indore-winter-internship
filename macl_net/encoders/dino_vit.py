import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class DINOEncoder(nn.Module):
    """
    DINO ViT encoder for global semantic consistency.

    - Frozen during training
    - Outputs normalized global image representation
    """

    def __init__(
        self,
        model_name: str = "vit_base_patch16_224",
        device: torch.device = None
    ):
        super().__init__()

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.device = device

        # ---------------------------------------------
        # Load DINO ViT (FP32, no classifier head)
        # ---------------------------------------------
        self.vit = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=0,
            global_pool="avg"
        ).to(device)

        # Force FP32 (IMPORTANT)
        self.vit = self.vit.float()

        # Freeze encoder
        for p in self.vit.parameters():
            p.requires_grad = False

        self.vit.eval()

        # ---------------------------------------------
        # ImageNet normalization (DINO standard)
        # ---------------------------------------------
        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    @torch.no_grad()
    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        # [-1, 1] → [0, 1]
        x = (x + 1.0) / 2.0

        x = F.interpolate(
            x,
            size=224,
            mode="bilinear",
            align_corners=False
        )

        x = (x - self.mean) / self.std
        return x

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: image tensor (N, 3, H, W), range [-1,1]

        Returns:
            Normalized global feature vector (N, C)
        """
        x = x.float()
        x = self.preprocess(x)

        feat = self.vit(x)

        # 🔑 CRITICAL FIX: normalize features
        feat = F.normalize(feat, dim=-1)

        return feat
