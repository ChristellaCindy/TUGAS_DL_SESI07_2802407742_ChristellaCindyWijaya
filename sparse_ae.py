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

sae_enc = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 64),
).to(device)

sae_dec = nn.Sequential(
    nn.Linear(64, 256),
    nn.ReLU(),
    nn.Linear(256, 784),
    nn.Sigmoid()
).to(device)

mse_fn = nn.MSELoss()
sae_opt = optim.Adam(list(sae_enc.parameters()) + list(sae_dec.parameters()), lr=1e-3)


lam = 1e-4
sae_epochs = 20
sae_losses = []
sae_times = []

for epoch in range(sae_epochs):
    t_start = time.time()
    sae_enc.train()
    sae_dec.train()
    train_loss = 0.0

    for imgs, _ in train_loader:
        x = imgs.view(imgs.size(0), -1).to(device)

        sae_opt.zero_grad()
        h = sae_enc(x)
        recon = sae_dec(h)

        recon_loss = mse_fn(recon, x)
        l1_penalty = h.abs().sum(dim=1).mean()
        loss = recon_loss + lam * l1_penalty

        loss.backward()
        sae_opt.step()
        train_loss += loss.item()

    ep_time = time.time() - t_start
    
    sae_enc.eval()
    sae_dec.eval()
    val_mse = 0.0
    with torch.no_grad():
        for imgs, _ in val_loader:
            x = imgs.view(imgs.size(0), -1).to(device)
            val_mse += mse_fn(sae_dec(sae_enc(x)), x).item()

    avg_train = train_loss / len(train_loader)
    avg_val = val_mse / len(val_loader)

    sae_losses.append(avg_train)
    sae_times.append(ep_time)
    print(f"Epoch {epoch+1:02d}/{sae_epochs} | Train Loss: {avg_train:.5f} | Val MSE: {avg_val:.5f} | Time: {ep_time:.2f}s")

torch.save(sae_enc.state_dict(), "outputs/models/sae_enc.pt")
torch.save(sae_dec.state_dict(), "outputs/models/sae_dec.pt")
np.save("outputs/models/sae_losses.npy", np.array(sae_losses))


psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

sae_enc.eval()
sae_dec.eval()
mse_sum = 0.0
total_pixels = 0
sae_h_list = []

with torch.no_grad():
    for imgs, _ in test_loader:
        x_img = imgs.to(device)
        x_flat = x_img.view(x_img.size(0), -1)

        h = sae_enc(x_flat)
        recon_img = sae_dec(h).view(-1, 1, 28, 28)

        mse_sum += F.mse_loss(recon_img, x_img, reduction="sum").item()
        total_pixels += x_img.numel()
        psnr_metric.update(recon_img, x_img)
        ssim_metric.update(recon_img, x_img)
        sae_h_list.append(h)

sae_mse = mse_sum / total_pixels
sae_psnr = psnr_metric.compute().item()
sae_ssim = ssim_metric.compute().item()
sae_avg_time = np.mean(sae_times)

all_h = torch.cat(sae_h_list, dim=0)
sae_zero = (all_h == 0).float().mean().item() * 100
sae_near_zero = (all_h.abs() < 0.05).float().mean().item() * 100
sae_l1 = all_h.abs().sum(dim=1).mean().item()

# Inference latency per batch 64
x_64 = next(iter(test_loader))[0][:64].view(64, -1).to(device)
with torch.no_grad():
    for _ in range(10):
        _ = sae_dec(sae_enc(x_64))
    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(100):
        _ = sae_dec(sae_enc(x_64))
    if device.type == "cuda":
        torch.cuda.synchronize()
    sae_latency = ((time.time() - t0) / 100) * 1000

print("=== Evaluasi Sparse AE (Linear + L1) ===")
print("Dimensi Laten: 64")
print(f"Lambda L1: {lam}")
print(f"Training Time/Epoch: {sae_avg_time:.2f} detik")
print(f"Latency Rekonstruksi (enc+dec, batch 64): {sae_latency:.3f} ms")
print(f"Test MSE: {sae_mse:.5f}")
print(f"Test PSNR: {sae_psnr:.2f} dB")
print(f"Test SSIM: {sae_ssim:.4f}")
print("FID Score: N/A (Latent Voids)")
print(f"Aktivasi tepat 0: {sae_zero:.1f}%")
print(f"Aktivasi hampir 0 (<0.05): {sae_near_zero:.1f}%")
print(f"Rata-rata ||h||_1: {sae_l1:.2f}")



picked_imgs = {}
for img, label in test_data:
    if label not in picked_imgs:
        picked_imgs[label] = img
    if len(picked_imgs) == 10:
        break

sample_imgs = torch.stack([picked_imgs[i] for i in range(10)])
x_real = sample_imgs.view(10, -1).to(device)

with torch.no_grad():
    x_sae = sae_dec(sae_enc(x_real)).view(10, 28, 28).cpu()

fig, axes = plt.subplots(2, 10, figsize=(14, 3.2))
for i in range(10):
    axes[0, i].imshow(sample_imgs[i].squeeze(), cmap="gray")
    axes[0, i].set_title(classes[i], fontsize=8)
    axes[0, i].set_xticks([])
    axes[0, i].set_yticks([])

    axes[1, i].imshow(x_sae[i], cmap="gray")
    axes[1, i].set_xticks([])
    axes[1, i].set_yticks([])

axes[0, 0].set_ylabel("Asli", fontsize=10)
axes[1, 0].set_ylabel("Sparse AE", fontsize=10)
plt.suptitle("Citra Asli vs Sparse AE")
plt.tight_layout()
plt.savefig("outputs/figs/sparse_ae.png", dpi=150)
plt.close()

# Sparse AE dengan aktivasi ReLU di bottleneck (ReLU + L1)
set_seed(seed)
sae_relu_enc = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 64),
    nn.ReLU()
).to(device)

sae_relu_dec = nn.Sequential(
    nn.Linear(64, 256),
    nn.ReLU(),
    nn.Linear(256, 784),
    nn.Sigmoid()
).to(device)

sae_relu_opt = optim.Adam(list(sae_relu_enc.parameters()) + list(sae_relu_dec.parameters()), lr=1e-3)

for epoch in range(sae_epochs):
    sae_relu_enc.train()
    sae_relu_dec.train()
    train_loss = 0.0

    for imgs, _ in train_loader:
        x = imgs.view(imgs.size(0), -1).to(device)

        sae_relu_opt.zero_grad()
        h = sae_relu_enc(x)
        recon = sae_relu_dec(h)

        loss = mse_fn(recon, x) + lam * h.abs().sum(dim=1).mean()
        loss.backward()
        sae_relu_opt.step()
        train_loss += loss.item()

    print(f"[ReLU+L1] Epoch {epoch+1:02d}/{sae_epochs} | Train Loss: {train_loss/len(train_loader):.5f}")

psnr_metric.reset()
ssim_metric.reset()
sae_relu_enc.eval()
sae_relu_dec.eval()

mse_sum = 0.0
total_pixels = 0
h_relu_list = []

with torch.no_grad():
    for imgs, _ in test_loader:
        x_img = imgs.to(device)
        x_flat = x_img.view(x_img.size(0), -1)

        h = sae_relu_enc(x_flat)
        recon_img = sae_relu_dec(h).view(-1, 1, 28, 28)

        mse_sum += F.mse_loss(recon_img, x_img, reduction="sum").item()
        total_pixels += x_img.numel()
        psnr_metric.update(recon_img, x_img)
        ssim_metric.update(recon_img, x_img)
        h_relu_list.append(h)

sae_relu_mse = mse_sum / total_pixels
sae_relu_psnr = psnr_metric.compute().item()
sae_relu_ssim = ssim_metric.compute().item()

all_h_relu = torch.cat(h_relu_list, dim=0)
sae_relu_zero = (all_h_relu == 0).float().mean().item() * 100
sae_relu_near_zero = (all_h_relu.abs() < 0.05).float().mean().item() * 100
sae_relu_l1 = all_h_relu.abs().sum(dim=1).mean().item()

print("=== Evaluasi Sparse AE (ReLU + L1) ===")
print(f"Test MSE: {sae_relu_mse:.5f}")
print(f"Test PSNR: {sae_relu_psnr:.2f} dB")
print(f"Test SSIM: {sae_relu_ssim:.4f}")
print(f"Aktivasi tepat 0: {sae_relu_zero:.1f}%")
print(f"Aktivasi hampir 0 (<0.05): {sae_relu_near_zero:.1f}%")
print(f"Rata-rata ||h||_1: {sae_relu_l1:.2f}")
