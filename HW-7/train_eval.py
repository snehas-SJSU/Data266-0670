import os
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.manifold import TSNE

# Step 0: Parameters from Assignment 1
SID4 = 670
SEED = SID4
SLICE = SID4 % 1000
HP_ID = SID4 % 6
CLS_A = SID4 % 10
CLS_B = (CLS_A + 1 + ((SID4 // 10) % 9)) % 10

# Set seeds
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
print(f"Using device: {device}")
print(f"Step 0: SID4={SID4}, SEED={SEED}, SLICE={SLICE}, HP_ID={HP_ID}, CLS_A={CLS_A}, CLS_B={CLS_B}")

# Paths: robust whether run from Gen-Ai/ or HW-7/
CURRENT_DIR = Path.cwd()
if CURRENT_DIR.name == "HW-7":
    ROOT = CURRENT_DIR
else:
    ROOT = CURRENT_DIR / "HW-7"
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "figures"
CKPT_DIR = ROOT / "checkpoints"
FIG_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR.mkdir(parents=True, exist_ok=True)

# Dataset & DataLoaders
transform = transforms.ToTensor()
train_dataset = datasets.FashionMNIST(DATA_DIR, train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST(DATA_DIR, train=False, download=True, transform=transform)

batch_size = 128
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

LATENT_DIM = 20

# Model 1: Fully Connected AutoEncoder (FC-AE)
class FCAE(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()
        # Encoder: 784 -> 512 -> 128 -> latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(True),
            nn.Linear(512, 128),
            nn.ReLU(True),
            nn.Linear(128, latent_dim)
        )
        # Decoder: latent_dim -> 128 -> 512 -> 784
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 512),
            nn.ReLU(True),
            nn.Linear(512, 784),
            nn.Sigmoid()
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x.view(-1, 784))
        recon = self.decode(z)
        return recon, z

# Model 2: Variational AutoEncoder (VAE)
class VAE(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()
        # Encoder backbone
        self.enc_backbone = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(True),
            nn.Linear(512, 128),
            nn.ReLU(True)
        )
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 512),
            nn.ReLU(True),
            nn.Linear(512, 784),
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.enc_backbone(x.view(-1, 784))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar, z

# Loss Functions
def fc_ae_loss(recon_x, x):
    # BCE loss summed over pixels, averaged over batch
    return F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction="sum") / x.size(0)

def vae_loss_fn(recon_x, x, mu, logvar):
    # Reconstruction loss (BCE) summed over pixels
    recon_loss = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction="sum") / x.size(0)
    # KL Divergence: D_KL(q(z|x) || p(z)) = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    total_loss = recon_loss + kl_div
    return total_loss, recon_loss, kl_div

# Count parameters
def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

fc_ae = FCAE().to(device)
vae = VAE().to(device)

print(f"FC-AE parameters: {count_params(fc_ae):,}")
print(f"VAE parameters:   {count_params(vae):,}")

# Training Setup
epochs = 20
lr = 1e-3
opt_ae = torch.optim.Adam(fc_ae.parameters(), lr=lr)
opt_vae = torch.optim.Adam(vae.parameters(), lr=lr)

ae_train_losses = []
vae_train_losses = []
vae_train_recons = []
vae_train_kls = []

start_time = time.time()
print("\n--- Training Fully Connected AutoEncoder (FC-AE) ---")
for epoch in range(1, epochs + 1):
    fc_ae.train()
    running_loss = 0.0
    for x, _ in train_loader:
        x = x.to(device)
        opt_ae.zero_grad()
        recon, _ = fc_ae(x)
        loss = fc_ae_loss(recon, x)
        loss.backward()
        opt_ae.step()
        running_loss += loss.item() * x.size(0)
    epoch_loss = running_loss / len(train_dataset)
    ae_train_losses.append(epoch_loss)
    if True:
        print(f"FC-AE Epoch [{epoch:2d}/{epochs}] - Train Loss (BCE): {epoch_loss:.4f}")

print("\n--- Training Variational AutoEncoder (VAE) ---")
for epoch in range(1, epochs + 1):
    vae.train()
    running_total = 0.0
    running_recon = 0.0
    running_kl = 0.0
    for x, _ in train_loader:
        x = x.to(device)
        opt_vae.zero_grad()
        recon, mu, logvar, _ = vae(x)
        total_loss, recon_loss, kl_div = vae_loss_fn(recon, x, mu, logvar)
        total_loss.backward()
        opt_vae.step()
        running_total += total_loss.item() * x.size(0)
        running_recon += recon_loss.item() * x.size(0)
        running_kl += kl_div.item() * x.size(0)
    epoch_total = running_total / len(train_dataset)
    epoch_recon = running_recon / len(train_dataset)
    epoch_kl = running_kl / len(train_dataset)
    vae_train_losses.append(epoch_total)
    vae_train_recons.append(epoch_recon)
    vae_train_kls.append(epoch_kl)
    if True:
        print(f"VAE Epoch [{epoch:2d}/{epochs}] - Total: {epoch_total:.4f} | Recon: {epoch_recon:.4f} | KL: {epoch_kl:.4f}")

train_time = time.time() - start_time
print(f"\nTraining completed in {train_time:.2f} seconds.")

# Save checkpoints
torch.save(fc_ae.state_dict(), CKPT_DIR / "fc_ae.pt")
torch.save(vae.state_dict(), CKPT_DIR / "vae.pt")
print("Checkpoints saved successfully.")

# Quantitative Evaluation on Test Set
fc_ae.eval()
vae.eval()

ae_test_bce = 0.0
ae_test_mse = 0.0
vae_test_bce = 0.0
vae_test_mse = 0.0
vae_test_kl = 0.0

with torch.no_grad():
    for x, _ in test_loader:
        x = x.to(device)
        flat_x = x.view(-1, 784)
        
        # FC-AE
        recon_ae, _ = fc_ae(x)
        ae_test_bce += F.binary_cross_entropy(recon_ae, flat_x, reduction="sum").item()
        ae_test_mse += F.mse_loss(recon_ae, flat_x, reduction="sum").item()
        
        # VAE
        recon_vae, mu, logvar, _ = vae(x)
        vae_test_bce += F.binary_cross_entropy(recon_vae, flat_x, reduction="sum").item()
        vae_test_mse += F.mse_loss(recon_vae, flat_x, reduction="sum").item()
        vae_test_kl += (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())).item()

n_test = len(test_dataset)
ae_test_bce /= n_test
ae_test_mse /= (n_test * 784)
vae_test_bce /= n_test
vae_test_mse /= (n_test * 784)
vae_test_kl /= n_test

print("\n--- Quantitative Test Set Evaluation ---")
print(f"FC-AE Test BCE Loss (per image): {ae_test_bce:.4f}")
print(f"FC-AE Test MSE Loss (per pixel): {ae_test_mse:.6f}")
print(f"VAE Test BCE Loss (per image):   {vae_test_bce:.4f}")
print(f"VAE Test MSE Loss (per pixel):   {vae_test_mse:.6f}")
print(f"VAE Test KL Divergence:          {vae_test_kl:.4f}")
print(f"VAE Test Total ELBO Loss:        {(vae_test_bce + vae_test_kl):.4f}")

# 1. Plot Training Loss Curves
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(range(1, epochs + 1), ae_train_losses, label="FC-AE Reconstruction (BCE)", color="#1f77b4", lw=2)
plt.plot(range(1, epochs + 1), vae_train_recons, label="VAE Reconstruction (BCE)", color="#2ca02c", lw=2, linestyle="--")
plt.plot(range(1, epochs + 1), vae_train_losses, label="VAE Total Loss (BCE + KL)", color="#d62728", lw=2)
plt.xlabel("Epoch")
plt.ylabel("Loss (per image)")
plt.title("Reconstruction & Total Loss vs Epoch")
plt.grid(True, linestyle=":", alpha=0.6)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(1, epochs + 1), vae_train_kls, label="VAE KL Divergence", color="#9467bd", lw=2)
plt.xlabel("Epoch")
plt.ylabel("KL Divergence")
plt.title("VAE Latent Regularization (KL Divergence)")
plt.grid(True, linestyle=":", alpha=0.6)
plt.legend()

plt.tight_layout()
plt.savefig(FIG_DIR / "loss_curves.png", dpi=300)
plt.close()
print("Saved figures/loss_curves.png")

# 2. Visual Reconstruction Comparison across 10 Classes
# Pick 1 sample for each of the 10 classes from test set
samples_per_class = {}
for img, label in test_dataset:
    if label not in samples_per_class:
        samples_per_class[label] = img
    if len(samples_per_class) == 10:
        break

ordered_imgs = [samples_per_class[c] for c in range(10)]
batch_imgs = torch.stack(ordered_imgs).to(device)

with torch.no_grad():
    recon_ae_imgs, _ = fc_ae(batch_imgs)
    recon_vae_imgs, _, _, _ = vae(batch_imgs)

recon_ae_imgs = recon_ae_imgs.view(-1, 28, 28).cpu().numpy()
recon_vae_imgs = recon_vae_imgs.view(-1, 28, 28).cpu().numpy()
orig_imgs = batch_imgs.view(-1, 28, 28).cpu().numpy()

fig, axes = plt.subplots(3, 10, figsize=(15, 5))
for col in range(10):
    # Original
    axes[0, col].imshow(orig_imgs[col], cmap="gray")
    axes[0, col].axis("off")
    axes[0, col].set_title(class_names[col], fontsize=8)
    # FC-AE
    axes[1, col].imshow(recon_ae_imgs[col], cmap="gray")
    axes[1, col].axis("off")
    # VAE
    axes[2, col].imshow(recon_vae_imgs[col], cmap="gray")
    axes[2, col].axis("off")

for row, name in enumerate(["Original", "FC-AE", "VAE"]):
    axes[row, 0].text(-0.08, 0.5, name, transform=axes[row, 0].transAxes,
                      rotation=90, va="center", ha="right", fontsize=10)
fig.suptitle("Test Sample Reconstruction: Original vs FC-AE vs VAE", fontsize=13, y=0.98)
plt.tight_layout()
plt.savefig(FIG_DIR / "reconstructions_comparison.png", dpi=300)
plt.close()
print("Saved figures/reconstructions_comparison.png")

# 3. Random Generation from Latent Prior z ~ N(0, I)
n_samples = 25
torch.manual_seed(SEED)
z_prior = torch.randn(n_samples, LATENT_DIM).to(device)

with torch.no_grad():
    gen_ae = fc_ae.decode(z_prior).view(-1, 28, 28).cpu().numpy()
    gen_vae = vae.decode(z_prior).view(-1, 28, 28).cpu().numpy()

fig, axes = plt.subplots(5, 10, figsize=(14, 7))
for i in range(25):
    r, c = divmod(i, 5)
    # FC-AE on left 5 cols
    axes[r, c].imshow(gen_ae[i], cmap="gray")
    axes[r, c].axis("off")
    # VAE on right 5 cols
    axes[r, c + 5].imshow(gen_vae[i], cmap="gray")
    axes[r, c + 5].axis("off")

axes[0, 2].set_title("FC-AE (Random z ~ N(0,I))", fontsize=11, fontweight="bold")
axes[0, 7].set_title("VAE (Sampled from Prior z ~ N(0,I))", fontsize=11, fontweight="bold")
fig.suptitle("Generative Capability: Decoding Random Latent Noise z ~ N(0, I)", fontsize=13)
plt.tight_layout()
plt.savefig(FIG_DIR / "random_generation.png", dpi=300)
plt.close()
print("Saved figures/random_generation.png")

# 4. Latent Space Linear Interpolation between CLS_A (0: T-shirt) and CLS_B (5: Sandal)
# Find one test sample for CLS_A and one for CLS_B
idx_a = None
idx_b = None
for idx, (img, label) in enumerate(test_dataset):
    if label == CLS_A and idx_a is None:
        idx_a = idx
    elif label == CLS_B and idx_b is None:
        idx_b = idx
    if idx_a is not None and idx_b is not None:
        break

img_a, _ = test_dataset[idx_a]
img_b, _ = test_dataset[idx_b]
pair_imgs = torch.stack([img_a, img_b]).to(device)

with torch.no_grad():
    z_ae_a = fc_ae.encode(pair_imgs[0:1].view(1, -1))
    z_ae_b = fc_ae.encode(pair_imgs[1:2].view(1, -1))
    mu_a, _ = vae.encode(pair_imgs[0:1].view(1, -1))
    mu_b, _ = vae.encode(pair_imgs[1:2].view(1, -1))

steps = 10
alphas = np.linspace(0, 1, steps)
interp_ae = []
interp_vae = []

with torch.no_grad():
    for alpha in alphas:
        z_ae_interp = (1 - alpha) * z_ae_a + alpha * z_ae_b
        z_vae_interp = (1 - alpha) * mu_a + alpha * mu_b
        out_ae = fc_ae.decode(z_ae_interp).view(28, 28).cpu().numpy()
        out_vae = vae.decode(z_vae_interp).view(28, 28).cpu().numpy()
        interp_ae.append(out_ae)
        interp_vae.append(out_vae)

fig, axes = plt.subplots(2, steps, figsize=(14, 3.5))
for col, alpha in enumerate(alphas):
    axes[0, col].imshow(interp_ae[col], cmap="gray")
    axes[0, col].axis("off")
    axes[0, col].set_title(f"α={alpha:.2f}", fontsize=8)
    axes[1, col].imshow(interp_vae[col], cmap="gray")
    axes[1, col].axis("off")

for row, name in enumerate(["FC-AE", "VAE"]):
    axes[row, 0].text(-0.08, 0.5, name, transform=axes[row, 0].transAxes,
                      rotation=90, va="center", ha="right", fontsize=11)
fig.suptitle(f"Latent Interpolation: {class_names[CLS_A]} (CLS_A=0) to {class_names[CLS_B]} (CLS_B=5)", fontsize=12)
plt.tight_layout()
plt.savefig(FIG_DIR / "latent_interpolation.png", dpi=300)
plt.close()
print("Saved figures/latent_interpolation.png")

# 5. Latent Space t-SNE Projection (2,000 test points)
n_eval = 2000
subset_loader = DataLoader(test_dataset, batch_size=n_eval, shuffle=False)
x_sub, y_sub = next(iter(subset_loader))
x_sub = x_sub.to(device)

with torch.no_grad():
    _, z_ae_eval = fc_ae(x_sub)
    _, mu_vae_eval, _, _ = vae(x_sub)

z_ae_np = z_ae_eval.cpu().numpy()
z_vae_np = mu_vae_eval.cpu().numpy()
y_np = y_sub.numpy()

tsne = TSNE(n_components=2, random_state=SEED, perplexity=30)
tsne_ae = tsne.fit_transform(z_ae_np)
tsne_vae = tsne.fit_transform(z_vae_np)

plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
scatter_ae = plt.scatter(tsne_ae[:, 0], tsne_ae[:, 1], c=y_np, cmap="tab10", alpha=0.7, s=12)
plt.title("FC-AE Latent Space (t-SNE)", fontsize=12)
plt.xlabel("t-SNE Dim 1")
plt.ylabel("t-SNE Dim 2")
plt.grid(True, linestyle=":", alpha=0.5)

plt.subplot(1, 2, 2)
scatter_vae = plt.scatter(tsne_vae[:, 0], tsne_vae[:, 1], c=y_np, cmap="tab10", alpha=0.7, s=12)
plt.title("VAE Latent Space (t-SNE)", fontsize=12)
plt.xlabel("t-SNE Dim 1")
plt.ylabel("t-SNE Dim 2")
plt.grid(True, linestyle=":", alpha=0.5)

handles = [plt.Line2D([0], [0], marker="o", ls="", color=plt.cm.tab10(c / 9), label=class_names[c], markersize=6)
           for c in range(10)]
plt.figlegend(handles=handles, loc="lower center", ncol=5, frameon=False)
plt.subplots_adjust(bottom=0.2)

plt.savefig(FIG_DIR / "latent_tsne.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved figures/latent_tsne.png")
print("\nAll experiments, evaluations, and figures completed successfully!")
