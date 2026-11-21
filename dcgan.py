import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchmetrics.image.fid import FrechetInceptionDistance

from data_setup import train_loader, test_loader, test_data, seed

def set_seed(s=42):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)

set_seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs("outputs/models", exist_ok=True)
os.makedirs("outputs/figs", exist_ok=True)

z_dim = 100

def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight, 0.0, 0.02)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)

# Generator (ConvTranspose)
dcgan_G = nn.Sequential(
    nn.Unflatten(1, (z_dim, 1, 1)),
    nn.ConvTranspose2d(z_dim, 256, kernel_size=7, stride=1, padding=0, bias=False),
    nn.BatchNorm2d(256),
    nn.ReLU(True),
    nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(128),
    nn.ReLU(True),
    nn.ConvTranspose2d(128, 1, kernel_size=4, stride=2, padding=1, bias=False),
    nn.Tanh()
).to(device)

# Discriminator (Conv)
dcgan_D = nn.Sequential(
    nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1, bias=False),
    nn.LeakyReLU(0.2, inplace=True),
    nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(128),
    nn.LeakyReLU(0.2, inplace=True),
    nn.Conv2d(128, 1, kernel_size=7, stride=1, padding=0, bias=False),
    nn.Flatten(),
    nn.Sigmoid()
).to(device)

dcgan_G.apply(init_weights)
dcgan_D.apply(init_weights)

bce_fn = nn.BCELoss()
opt_dcgan_G = optim.Adam(dcgan_G.parameters(), lr=2e-4, betas=(0.5, 0.999))
opt_dcgan_D = optim.Adam(dcgan_D.parameters(), lr=2e-4, betas=(0.5, 0.999))

dcgan_epochs = 30
dcgan_g_losses = []
dcgan_d_losses = []
dcgan_time = []

for epoch in range(dcgan_epochs):
    t_start = time.time()
    dcgan_G.train()
    dcgan_D.train()
    d_total = 0.0
    g_total = 0.0

    for imgs, _ in train_loader:
        bs = imgs.size(0)
        real_x = imgs.to(device) * 2 - 1
        real_y = torch.ones(bs, 1, device=device)
        fake_y = torch.zeros(bs, 1, device=device)

        # Train Discriminator
        opt_dcgan_D.zero_grad()
        z = torch.randn(bs, z_dim, device=device)
        fake_x = dcgan_G(z).detach()

        d_loss_real = bce_fn(dcgan_D(real_x), real_y)
        d_loss_fake = bce_fn(dcgan_D(fake_x), fake_y)
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        opt_dcgan_D.step()

        # Train Generator
        opt_dcgan_G.zero_grad()
        z = torch.randn(bs, z_dim, device=device)
        gen_x = dcgan_G(z)
        g_loss = bce_fn(dcgan_D(gen_x), real_y)
        g_loss.backward()
        opt_dcgan_G.step()

        d_total += d_loss.item()
        g_total += g_loss.item()

    ep_time = time.time() - t_start
    avg_d = d_total / len(train_loader)
    avg_g = g_total / len(train_loader)

    dcgan_d_losses.append(avg_d)
    dcgan_g_losses.append(avg_g)
    dcgan_time.append(ep_time)
    print(f"Epoch {epoch+1:02d}/{dcgan_epochs} | D Loss: {avg_d:.4f} | G Loss: {avg_g:.4f} | Time: {ep_time:.2f}s")

torch.save(dcgan_G.state_dict(), "outputs/models/dcgan_G.pt")
torch.save(dcgan_D.state_dict(), "outputs/models/dcgan_D.pt")

np.save("outputs/models/dcgan_g_losses.npy", np.array(dcgan_g_losses))
np.save("outputs/models/dcgan_d_losses.npy", np.array(dcgan_d_losses))

fid_metric = FrechetInceptionDistance(feature=2048, reset_real_features=False).to(device)
dcgan_G.eval()

with torch.no_grad():
    for imgs, _ in test_loader:
        real_rgb = (imgs.to(device).clamp(0, 1) * 255).to(torch.uint8).repeat(1, 3, 1, 1)
        fid_metric.update(real_rgb, real=True)


# Generate sampel baru untuk hitung FID Score
set_seed(seed)
n_total = len(test_data)
n_done = 0

with torch.no_grad():
    while n_done < n_total:
        bs = min(128, n_total - n_done)
        z_rand = torch.randn(bs, z_dim, device=device)
        fake_img = (dcgan_G(z_rand) + 1) / 2
        fake_rgb = (fake_img.clamp(0, 1) * 255).to(torch.uint8).repeat(1, 3, 1, 1)
        fid_metric.update(fake_rgb, real=False)
        n_done += bs

dcgan_fid = fid_metric.compute().item()
dcgan_avg_time = np.mean(dcgan_time)


# Inference latency single-pass
z_dcgan_64 = torch.randn(64, z_dim, device=device)
with torch.no_grad():
    for _ in range(10):
        _ = dcgan_G(z_dcgan_64)
    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(100):
        _ = dcgan_G(z_dcgan_64)
    if device.type == "cuda":
        torch.cuda.synchronize()
    dcgan_latency = ((time.time() - t0) / 100) * 1000

print("=== Evaluasi DCGAN ===")
print("Dimensi Laten: 100 (Prior Noise)")
print(f"Training Time/Epoch: {dcgan_avg_time:.2f} detik")
print(f"Latency Generation (prior->citra, batch 64): {dcgan_latency:.3f} ms")
print("Test MSE: N/A (Tanpa Encoder)")
print("Test PSNR: N/A")
print("Test SSIM: N/A")
print(f"FID Score: {dcgan_fid:.2f}")


set_seed(seed)
with torch.no_grad():
    z_16 = torch.randn(16, z_dim, device=device)
    dcgan_samples = ((dcgan_G(z_16) + 1) / 2).clamp(0, 1).squeeze(1).cpu()

fig, axes = plt.subplots(4, 4, figsize=(5.5, 5.5))
for i in range(16):
    ax = axes[i // 4, i % 4]
    ax.imshow(dcgan_samples[i], cmap="gray")
    ax.axis("off")

plt.suptitle("DCGAN Random Sampling")
plt.tight_layout()
plt.savefig("outputs/figs/dcgan_samples.png", dpi=150)
plt.close()

plt.figure(figsize=(7, 4))
plt.plot(range(1, dcgan_epochs + 1), dcgan_d_losses, label="Discriminator Loss")
plt.plot(range(1, dcgan_epochs + 1), dcgan_g_losses, label="Generator Loss")
plt.xlabel("Epoch")
plt.ylabel("BCE Loss")
plt.title("Training Loss DCGAN")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/figs/dcgan_loss.png", dpi=150)
plt.close()
