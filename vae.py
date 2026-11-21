import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.fid import FrechetInceptionDistance

from data_setup import train_loader, val_loader, test_loader, test_data, classes, seed

def set_seed(s=42):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)

set_seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs("outputs/models", exist_ok=True)
os.makedirs("outputs/figs", exist_ok=True)


vae_enc = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU()
).to(device)

vae_mu = nn.Linear(256, 64).to(device)
vae_logvar = nn.Linear(256, 64).to(device)

vae_dec = nn.Sequential(
    nn.Linear(64, 256),
    nn.ReLU(),
    nn.Linear(256, 784),
    nn.Sigmoid()
).to(device)

vae_params = (
    list(vae_enc.parameters()) +
    list(vae_mu.parameters()) +
    list(vae_logvar.parameters()) +
    list(vae_dec.parameters())
)
vae_opt = optim.Adam(vae_params, lr=1e-3)


vae_epochs = 20
vae_losses = []
vae_times = []

for epoch in range(vae_epochs):
    t_start = time.time()
    vae_enc.train()
    vae_mu.train()
    vae_logvar.train()
    vae_dec.train()
    train_loss = 0.0

    for imgs, _ in train_loader:
        x = imgs.view(imgs.size(0), -1).to(device)

        vae_opt.zero_grad()
        h = vae_enc(x)
        mu = vae_mu(h)
        logvar = vae_logvar(h)

        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)
        recon = vae_dec(z)

        bce = F.binary_cross_entropy(recon, x, reduction="sum") / x.size(0)
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
        loss = bce + kl

        loss.backward()
        vae_opt.step()
        train_loss += loss.item()

    ep_time = time.time() - t_start
    
    vae_enc.eval()
    vae_mu.eval()
    vae_logvar.eval()
    vae_dec.eval()
    
    val_loss = 0.0
    with torch.no_grad():
        for imgs, _ in val_loader:
            x = imgs.view(imgs.size(0), -1).to(device)
            h = vae_enc(x)
            mu = vae_mu(h)
            logvar = vae_logvar(h)
            z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
            recon = vae_dec(z)

            bce = F.binary_cross_entropy(recon, x, reduction="sum") / x.size(0)
            kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
            val_loss += (bce + kl).item()

    avg_train = train_loss / len(train_loader)
    avg_val = val_loss / len(val_loader)

    vae_losses.append(avg_train)
    vae_times.append(ep_time)
    print(f"Epoch {epoch+1:02d}/{vae_epochs} | Train Loss: {avg_train:.2f} | Val Loss: {avg_val:.2f} | Time: {ep_time:.2f}s")

torch.save(vae_enc.state_dict(), "outputs/models/vae_enc.pt")
torch.save(vae_mu.state_dict(), "outputs/models/vae_mu.pt")
torch.save(vae_logvar.state_dict(), "outputs/models/vae_logvar.pt")
torch.save(vae_dec.state_dict(), "outputs/models/vae_dec.pt")
np.save("outputs/models/vae_losses.npy", np.array(vae_losses))


psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
fid_metric = FrechetInceptionDistance(feature=2048, reset_real_features=False).to(device)

vae_enc.eval()
vae_mu.eval()
vae_logvar.eval()
vae_dec.eval()
mse_sum = 0.0
total_pixels = 0

with torch.no_grad():
    for imgs, _ in test_loader:
        x_img = imgs.to(device)
        x_flat = x_img.view(x_img.size(0), -1)

        mu = vae_mu(vae_enc(x_flat))
        recon_img = vae_dec(mu).view(-1, 1, 28, 28)

        mse_sum += F.mse_loss(recon_img, x_img, reduction="sum").item()
        total_pixels += x_img.numel()
        psnr_metric.update(recon_img, x_img)
        ssim_metric.update(recon_img, x_img)

        real_rgb = (x_img.clamp(0, 1) * 255).to(torch.uint8).repeat(1, 3, 1, 1)
        fid_metric.update(real_rgb, real=True)

vae_mse = mse_sum / total_pixels
vae_psnr = psnr_metric.compute().item()
vae_ssim = ssim_metric.compute().item()
vae_avg_time = np.mean(vae_times)

# Generate gambar baru untuk hitung FID Score
set_seed(seed)
n_total = len(test_data)
n_done = 0

with torch.no_grad():
    while n_done < n_total:
        bs = min(128, n_total - n_done)
        z_rand = torch.randn(bs, 64, device=device)
        fake_img = vae_dec(z_rand).view(-1, 1, 28, 28)
        fake_rgb = (fake_img.clamp(0, 1) * 255).to(torch.uint8).repeat(1, 3, 1, 1)
        fid_metric.update(fake_rgb, real=False)
        n_done += bs

vae_fid = fid_metric.compute().item()

# Inference Latency single-pass decoder
z_64 = torch.randn(64, 64, device=device)
with torch.no_grad():
    for _ in range(10):
        _ = vae_dec(z_64)
    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(100):
        _ = vae_dec(z_64)
    if device.type == "cuda":
        torch.cuda.synchronize()
    vae_latency = ((time.time() - t0) / 100) * 1000

print("=== Evaluasi VAE ===")
print("Dimensi Laten: 64")
print(f"Training Time/Epoch: {vae_avg_time:.2f} detik")
print(f"Latency Generation (prior->citra, batch 64): {vae_latency:.3f} ms")
print(f"Test MSE: {vae_mse:.5f}")
print(f"Test PSNR: {vae_psnr:.2f} dB")
print(f"Test SSIM: {vae_ssim:.4f}")
print(f"FID Score: {vae_fid:.2f}")


picked_imgs = {}
for img, label in test_data:
    if label not in picked_imgs:
        picked_imgs[label] = img
    if len(picked_imgs) == 10:
        break

sample_imgs = torch.stack([picked_imgs[i] for i in range(10)])
x_real = sample_imgs.view(10, -1).to(device)

with torch.no_grad():
    x_vae = vae_dec(vae_mu(vae_enc(x_real))).view(10, 28, 28).cpu()

fig, axes = plt.subplots(2, 10, figsize=(14, 3.2))
for i in range(10):
    axes[0, i].imshow(sample_imgs[i].squeeze(), cmap="gray")
    axes[0, i].set_title(classes[i], fontsize=8)
    axes[0, i].set_xticks([])
    axes[0, i].set_yticks([])

    axes[1, i].imshow(x_vae[i], cmap="gray")
    axes[1, i].set_xticks([])
    axes[1, i].set_yticks([])

axes[0, 0].set_ylabel("Asli", fontsize=10)
axes[1, 0].set_ylabel("VAE", fontsize=10)
plt.suptitle("Citra Asli vs VAE")
plt.tight_layout()
plt.savefig("outputs/figs/vae_recon.png", dpi=150)
plt.close()


# Random Sampling
set_seed(seed)
with torch.no_grad():
    z_16 = torch.randn(16, 64, device=device)
    vae_samples = vae_dec(z_16).view(16, 28, 28).cpu()

fig, axes = plt.subplots(4, 4, figsize=(5.5, 5.5))
for i in range(16):
    ax = axes[i // 4, i % 4]
    ax.imshow(vae_samples[i], cmap="gray", vmin=0, vmax=1)
    ax.axis("off")
plt.suptitle("VAE Random Sampling")
plt.tight_layout()
plt.savefig("outputs/figs/vae_samples.png", dpi=150)
plt.close()
