import json
import matplotlib.pyplot as plt

# ----------------------------
# Load loss log
# ----------------------------
with open("outputs/loss_log.json", "r") as f:
    log = json.load(f)

steps = log["step"]
D_loss = log["D"]
G_GAN = log["G_GAN"]
G_MACL = log["G_MACL"]
G_Global = log["G_Global"]

# ----------------------------
# Plot Discriminator Loss
# ----------------------------
plt.figure(figsize=(8, 5))
plt.plot(steps, D_loss, label="Discriminator Loss", color="red")
plt.xlabel("Training Steps")
plt.ylabel("Loss")
plt.title("Discriminator Loss vs Training Steps")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("outputs/discriminator_loss.png")
plt.show()

# ----------------------------
# Plot Generator Losses
# ----------------------------
plt.figure(figsize=(8, 5))
plt.plot(steps, G_GAN, label="GAN Loss")
plt.plot(steps, G_MACL, label="MACL Loss")
plt.plot(steps, G_Global, label="Global ViT Loss")
plt.xlabel("Training Steps")
plt.ylabel("Loss")
plt.title("Generator Loss Components")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("outputs/generator_losses.png")
plt.show()

# ----------------------------
# Plot Total Generator Loss
# ----------------------------
total_G = [
    g1 + g2 + g3
    for g1, g2, g3 in zip(G_GAN, G_MACL, G_Global)
]

plt.figure(figsize=(8, 5))
plt.plot(steps, total_G, label="Total Generator Loss", color="green")
plt.xlabel("Training Steps")
plt.ylabel("Loss")
plt.title("Total Generator Loss vs Training Steps")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("outputs/total_generator_loss.png")
plt.show()

print("✅ Loss graphs saved in outputs/")
