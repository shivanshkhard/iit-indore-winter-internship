import os
import torch

from torch.utils.data import DataLoader
from torchvision.utils import save_image

# ================================
# Imports
# ================================
from data.unpaired_dataset import UnpairedDataset

from models.generator import MACLGenerator
from models.discriminator import Discriminator

from losses.gan_loss import GANLoss
from losses.macl_loss import MACLLoss
from losses.global_vit_loss import GlobalViTLoss

from encoders.clip_encoder import CLIPEncoder
from encoders.dino_vit import DINOEncoder




# ================================
# Configuration
# ================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_ROOT = "datasets/horse2zebra"
BATCH_SIZE = 1
IMG_SIZE = 256
TOTAL_EPOCHS = 30
SAVE_INTERVAL = 100

# Optimizer settings
LR_G = 2e-4
LR_D = 1e-4

# Loss weights
LAMBDA_GAN = 0.5
LAMBDA_MACL = 0.5
LAMBDA_GLOBAL = 0.2

os.makedirs("outputs", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

print("Using device:", DEVICE)


# ================================
# Dataset
# ================================
dataset = UnpairedDataset(
    root=DATA_ROOT,
    phase="train",
    img_size=IMG_SIZE
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=True
)


# ================================
# Models
# ================================
G = MACLGenerator().to(DEVICE)
D = Discriminator().to(DEVICE)

clip_encoder = CLIPEncoder(return_patches=True).to(DEVICE)
dino_encoder = DINOEncoder().to(DEVICE)

clip_encoder.eval()
dino_encoder.eval()

for p in clip_encoder.parameters():
    p.requires_grad = False

for p in dino_encoder.parameters():
    p.requires_grad = False


# ================================
# Losses
# ================================
gan_loss = GANLoss(gan_mode="lsgan")
macl_loss_fn = MACLLoss()
global_vit_loss_fn = GlobalViTLoss()


# ================================
# Optimizers
# ================================
optimizer_G = torch.optim.Adam(
    G.parameters(), lr=LR_G, betas=(0.5, 0.999)
)

optimizer_D = torch.optim.Adam(
    D.parameters(), lr=LR_D, betas=(0.5, 0.999)
)

# ================================
# LR Schedulers
# ================================
scheduler_G = torch.optim.lr_scheduler.StepLR(
    optimizer_G, step_size=15, gamma=0.5
)

scheduler_D = torch.optim.lr_scheduler.StepLR(
    optimizer_D, step_size=15, gamma=0.5
)


# ================================
# 🔁 AUTO RESUME LOGIC (SAFE)
# ================================
start_epoch = 0
step = 0
best_G_loss = float("inf")

resume_path = "checkpoints/last.pth"

if os.path.exists(resume_path):
    print("🔁 Resuming training from last checkpoint")

    ckpt = torch.load(resume_path, map_location=DEVICE)

    G.load_state_dict(ckpt["G"])
    D.load_state_dict(ckpt["D"])

    optimizer_G.load_state_dict(ckpt["opt_G"])
    optimizer_D.load_state_dict(ckpt["opt_D"])

    scheduler_G.load_state_dict(ckpt["sch_G"])
    scheduler_D.load_state_dict(ckpt["sch_D"])

    start_epoch = ckpt["epoch"] + 1
    step = ckpt["step"]
    best_G_loss = ckpt.get("best_G_loss", best_G_loss)

    print(f"➡ Resumed from epoch {start_epoch}")
else:
    print("🆕 No checkpoint found — starting from epoch 0")


# ================================
# Training Loop
# ================================
for epoch in range(start_epoch, TOTAL_EPOCHS):
    print(f"\n========== Epoch {epoch} ==========")

    for batch in loader:
        step += 1

        real_A = batch["A"].to(DEVICE)
        real_B = batch["B"].to(DEVICE)

        # -------------------------
        # Train Discriminator
        # -------------------------
        optimizer_D.zero_grad()

        with torch.no_grad():
            fake_B = G(real_A)

        loss_D = 0.5 * (
            gan_loss(D(real_B), True) +
            gan_loss(D(fake_B.detach()), False)
        )

        loss_D.backward()

        # Train D less frequently (stability)
        if step % 2 == 0:
            optimizer_D.step()

        # -------------------------
        # Train Generator
        # -------------------------
        optimizer_G.zero_grad()

        fake_B = G(real_A)
        pred_fake = D(fake_B)

        loss_G_gan = gan_loss(pred_fake, True)

        with torch.no_grad():
            feat_real_clip = clip_encoder(real_B)

        feat_fake_clip = clip_encoder(fake_B)
        loss_G_macl = macl_loss_fn(feat_fake_clip, feat_real_clip)

        with torch.no_grad():
            feat_real_global = dino_encoder(real_B)

        feat_fake_global = dino_encoder(fake_B)
        loss_G_global = global_vit_loss_fn(
            feat_fake_global, feat_real_global
        )

        loss_G = (
            LAMBDA_GAN * loss_G_gan +
            LAMBDA_MACL * loss_G_macl +
            LAMBDA_GLOBAL * loss_G_global
        )

        loss_G.backward()
        optimizer_G.step()

        # -------------------------
        # Save BEST generator
        # -------------------------
        if loss_G.item() < best_G_loss:
            best_G_loss = loss_G.item()
            torch.save(
                {
                    "G": G.state_dict(),
                    "epoch": epoch,
                    "loss_G": best_G_loss
                },
                "checkpoints/best_model.pth"
            )

        # -------------------------
        # Logging
        # -------------------------
        if step % 50 == 0:
            print(
                f"[Epoch {epoch}] [Step {step}] "
                f"D: {loss_D.item():.4f} | "
                f"G_GAN: {loss_G_gan.item():.4f} | "
                f"G_MACL: {loss_G_macl.item():.4f} | "
                f"G_Global: {loss_G_global.item():.4f}"
            )

        # -------------------------
        # Save images
        # -------------------------
        if step % SAVE_INTERVAL == 0:
            save_image(fake_B * 0.5 + 0.5, f"outputs/fake_{step}.png")
            save_image(real_B * 0.5 + 0.5, f"outputs/real_{step}.png")

    # -------------------------
    # Save epoch checkpoint
    # -------------------------
    torch.save(
        {
            "epoch": epoch,
            "step": step,
            "G": G.state_dict(),
            "D": D.state_dict(),
            "opt_G": optimizer_G.state_dict(),
            "opt_D": optimizer_D.state_dict(),
            "sch_G": scheduler_G.state_dict(),
            "sch_D": scheduler_D.state_dict(),
            "best_G_loss": best_G_loss
        },
        "checkpoints/last.pth"
    )

    torch.save(
        {
            "G": G.state_dict(),
            "D": D.state_dict(),
            "epoch": epoch
        },
        f"checkpoints/macl_epoch_{epoch}.pth"
    )

    scheduler_G.step()
    scheduler_D.step()

    print(
        f"LR after epoch {epoch}: "
        f"G = {scheduler_G.get_last_lr()[0]:.6f}, "
        f"D = {scheduler_D.get_last_lr()[0]:.6f}"
    )

print("\nTraining completed successfully.")
