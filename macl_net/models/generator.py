import torch
import torch.nn as nn
import torch.nn.functional as F

from models.aff import AFF
from models.resblocks import LowFFTNResBlock, HighFFTNResBlock
from utils.wavelet import HaarDWT, HaarIDWT


# =====================================================
# Encoder
# =====================================================
class Encoder(nn.Module):
    def __init__(self, in_channels=3, base=64):
        super().__init__()

        self.blocks = nn.ModuleList()
        self.dwts = nn.ModuleList()
        self.affs = nn.ModuleList()

        ch = in_channels
        for i in range(4):
            out_ch = base * (2 ** i)

            self.blocks.append(
                nn.Sequential(
                    nn.Conv2d(ch, out_ch, kernel_size=4, stride=2, padding=1),
                    nn.InstanceNorm2d(out_ch),
                    nn.ReLU(inplace=True)
                )
            )

            self.dwts.append(HaarDWT())
            self.affs.append(AFF(out_ch))

            ch = out_ch

    def forward(self, x):
        low_feats, high_feats = [], []

        for block, dwt, aff in zip(self.blocks, self.dwts, self.affs):
            x = block(x)                  # Downsample
            ll, hf = dwt(x)               # Wavelet split

            fused = aff(ll, hf)
            fused = F.interpolate(
                fused,
                size=x.shape[2:],
                mode="nearest"
            )

            x = x + fused                 # Frequency-enhanced residual

            low_feats.append(ll)
            high_feats.append(hf)

        return x, low_feats, high_feats


# =====================================================
# Decoder
# =====================================================
class Decoder(nn.Module):
    def __init__(self, base=64):
        super().__init__()

        self.idwt = HaarIDWT()

        # Project reconstructed frequency maps
        self.freq_proj = nn.ModuleList([
            nn.Conv2d(base * 8, base * 4, kernel_size=1),  # 512 → 256
            nn.Conv2d(base * 4, base * 2, kernel_size=1),  # 256 → 128
            nn.Conv2d(base * 2, base, kernel_size=1),      # 128 → 64
        ])

        self.low_blocks = nn.ModuleList([
            LowFFTNResBlock(base * 4),
            LowFFTNResBlock(base * 2),
        ])

        self.high_blocks = nn.ModuleList([
            HighFFTNResBlock(base),
        ])

        self.up = nn.ModuleList([
            nn.ConvTranspose2d(base * 8, base * 4, 4, 2, 1),
            nn.ConvTranspose2d(base * 4, base * 2, 4, 2, 1),
            nn.ConvTranspose2d(base * 2, base, 4, 2, 1),
            nn.ConvTranspose2d(base, 3, 4, 2, 1),
        ])

    def forward(self, x, low_feats, high_feats):
        # -------------------------
        # Stage 1 (256 → 128)
        # -------------------------
        x = self.up[0](x)
        freq = self.idwt(low_feats[-1], high_feats[-1])
        freq = self.freq_proj[0](freq)
        x = self.low_blocks[0](x, freq)

        # -------------------------
        # Stage 2 (128 → 64)
        # -------------------------
        x = self.up[1](x)
        freq = self.idwt(low_feats[-2], high_feats[-2])
        freq = self.freq_proj[1](freq)
        x = self.low_blocks[1](x, freq)

        # -------------------------
        # Stage 3 (64 → 32)
        # -------------------------
        x = self.up[2](x)
        freq = self.idwt(low_feats[-3], high_feats[-3])
        freq = self.freq_proj[2](freq)
        x = self.high_blocks[0](x, freq)

        # -------------------------
        # Output (32 → 256)
        # -------------------------
        x = self.up[3](x)
        return torch.tanh(x)


# =====================================================
# Full Generator
# =====================================================
class MACLGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        latent, low_feats, high_feats = self.encoder(x)
        return self.decoder(latent, low_feats, high_feats)
