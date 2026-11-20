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


ae_enc = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 64)
).to(device)

ae_dec = nn.Sequential(
    nn.Linear(64, 256),
    nn.ReLU(),
    nn.Linear(256, 784),
    nn.Sigmoid()
).to(device)

mse_fn = nn.MSELoss()
ae_opt = optim.Adam(list(ae_enc.parameters()) + list(ae_dec.parameters()), lr=1e-3)


ae_epochs = 20
ae_losses = []
ae_times = []

for epoch in range(ae_epochs):
    t_start = time.time()
    ae_enc.train()
    ae_dec.train()
    train_loss = 0.0

    for imgs, _ in train_loader:
        x = imgs.view(imgs.size(0), -1).to(device)

        ae_opt.zero_grad()
        recon = ae_dec(ae_enc(x))
        loss = mse_fn(recon, x)
        loss.backward()
        ae_opt.step()
        train_loss += loss.item()

    ep_time = time.time() - t_start
    
    # Cek validation loss
    ae_enc.eval()
    ae_dec.eval()
    val_loss = 0.0
    with torch.no_grad():
        for imgs, _ in val_loader:
            x = imgs.view(imgs.size(0), -1).to(device)
            val_loss += mse_fn(ae_dec(ae_enc(x)), x).item()

    avg_train = train_loss / len(train_loader)
    avg_val = val_loss / len(val_loader)

    ae_losses.append(avg_train)
    ae_times.append(ep_time)
    print(f"Epoch {epoch+1:02d}/{ae_epochs} | Train MSE: {avg_train:.5f} | Val MSE: {avg_val:.5f} | Time: {ep_time:.2f}s")

torch.save(ae_enc.state_dict(), "outputs/models/ae_enc.pt")
torch.save(ae_dec.state_dict(), "outputs/models/ae_dec.pt")
np.save("outputs/models/ae_losses.npy", np.array(ae_losses))


psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

ae_enc.eval()
ae_dec.eval()
mse_sum = 0.0
total_pixels = 0
ae_z_list = []

with torch.no_grad():
    for imgs, _ in test_loader:
        x_img = imgs.to(device)
        x_flat = x_img.view(x_img.size(0), -1)
        z = ae_enc(x_flat)
        recon_img = ae_dec(ae_enc(x_flat)).view(-1, 1, 28, 28)

        mse_sum += F.mse_loss(recon_img, x_img, reduction="sum").item()
        total_pixels += x_img.numel()
        psnr_metric.update(recon_img, x_img)
        ssim_metric.update(recon_img, x_img)
        ae_z_list.append(z)

ae_mse = mse_sum / total_pixels
ae_psnr = psnr_metric.compute().item()
ae_ssim = ssim_metric.compute().item()
ae_avg_time = np.mean(ae_times)

all_z = torch.cat(ae_z_list, dim=0)
ae_near_zero = (all_z.abs() < 0.05).float().mean().item() * 100
ae_l1 = all_z.abs().sum(dim=1).mean().item()

# Inference Latency per batch 64 sampel
x_64 = next(iter(test_loader))[0][:64].view(64, -1).to(device)
with torch.no_grad():
    for _ in range(10):
        _ = ae_dec(ae_enc(x_64))
    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(100):
        _ = ae_dec(ae_enc(x_64))
    if device.type == "cuda":
        torch.cuda.synchronize()
    ae_latency = ((time.time() - t0) / 100) * 1000

print("=== Evaluasi Vanilla AE ===")
print(f"Dimensi Laten: 64")
print(f"Training Time/Epoch: {ae_avg_time:.2f} detik")
print(f"Latency Rekonstruksi (enc+dec, batch 64): {ae_latency:.3f} ms")
print(f"Test MSE: {ae_mse:.5f}")
print(f"Test PSNR: {ae_psnr:.2f} dB")
print(f"Test SSIM: {ae_ssim:.4f}")
print("FID Score: N/A (Latent Voids)")
print(f"Aktivasi hampir 0 (<0.05): {ae_near_zero:.1f}%")
print(f"Rata-rata ||z||_1: {ae_l1:.2f}")



picked_imgs = {}
for img, label in test_data:
    if label not in picked_imgs:
        picked_imgs[label] = img
    if len(picked_imgs) == 10:
        break

sample_imgs = torch.stack([picked_imgs[i] for i in range(10)])
x_real = sample_imgs.view(10, -1).to(device)

with torch.no_grad():
    x_ae = ae_dec(ae_enc(x_real)).view(10, 28, 28).cpu()

fig, axes = plt.subplots(2, 10, figsize=(14, 3.2))
for i in range(10):
    axes[0, i].imshow(sample_imgs[i].squeeze(), cmap="gray")
    axes[0, i].set_title(classes[i], fontsize=8)
    axes[0, i].set_xticks([])
    axes[0, i].set_yticks([])

    axes[1, i].imshow(x_ae[i], cmap="gray")
    axes[1, i].set_xticks([])
    axes[1, i].set_yticks([])

axes[0, 0].set_ylabel("Asli", fontsize=10)
axes[1, 0].set_ylabel("Vanilla AE", fontsize=10)
plt.suptitle("Citra Asli vs Vanilla AE")
plt.tight_layout()
plt.savefig("outputs/figs/vanilla_ae_recon.png", dpi=150)
plt.close()
