import torch
import torch.nn as nn
import torch.nn.functional as F
import clip


class CLIPEncoder(nn.Module):
    """
    CLIP image encoder for MACL-Net.

    - Extracts patch-level features (for MACL)
    - Optionally returns global features
    - Frozen during training
    """

    def __init__(
        self,
        model_name: str = "ViT-B/32",
        device: torch.device = None,
        return_patches: bool = True
    ):
        super().__init__()

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.device = device
        self.return_patches = return_patches

        # ---------------------------------------------
        # Load CLIP (FP32, frozen)
        # ---------------------------------------------
        self.clip_model, _ = clip.load(
            model_name,
            device=device,
            jit=False
        )

        self.clip_model = self.clip_model.float()

        for p in self.clip_model.parameters():
            p.requires_grad = False

        self.clip_model.eval()

        # ---------------------------------------------
        # CLIP normalization
        # ---------------------------------------------
        self.register_buffer(
            "mean",
            torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std",
            torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)
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
    def forward(self, x: torch.Tensor):
        x = x.float()
        x = self.preprocess(x)

        visual = self.clip_model.visual

        # ---------------------------------------------
        # Patch embedding
        # ---------------------------------------------
        x = visual.conv1(x)                        # (N, C, H', W')
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)                     # (N, P, C)

        # ---------------------------------------------
        # Class token + position
        # ---------------------------------------------
        cls_token = visual.class_embedding.to(x.dtype)
        cls_token = cls_token.unsqueeze(0).expand(x.shape[0], 1, -1)

        x = torch.cat([cls_token, x], dim=1)

        pos_embed = visual.positional_embedding.to(x.dtype)
        x = x + pos_embed

        x = visual.ln_pre(x)

        # ---------------------------------------------
        # Transformer
        # ---------------------------------------------
        x = x.permute(1, 0, 2)
        x = visual.transformer(x)
        x = x.permute(1, 0, 2)

        # ---------------------------------------------
        # Global feature
        # ---------------------------------------------
        global_feat = visual.ln_post(x[:, 0, :])
        global_feat = global_feat @ visual.proj
        global_feat = F.normalize(global_feat, dim=-1)

        if self.return_patches:
            patch_feats = x[:, 1:, :]
            patch_feats = visual.ln_post(patch_feats)
            patch_feats = patch_feats @ visual.proj
            patch_feats = F.normalize(patch_feats, dim=-1)
            return patch_feats

        return global_feat
