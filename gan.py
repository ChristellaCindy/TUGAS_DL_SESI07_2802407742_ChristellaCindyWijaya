import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchmetrics.image.fid import FrechetInceptionDistance

from data_setup import train_loader, test_loader, test_data, seed

torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs("outputs/models", exist_ok=True)
os.makedirs("outputs/figs", exist_ok=True)

z_dim = 100

#Generator
gan_G = nn.Sequential(
    nn.Linear(z_dim, 256),
    nn.LeakyReLU(0.2),
    nn.Linear(256, 512),
    nn.LeakyReLU(0.2),
    nn.Linear(512, 784),
    nn.Tanh()
).to(device)

# Discriminator
gan_D = nn.Sequential(
    nn.Linear(784, 512),
    nn.LeakyReLU(0.2),
    nn.Dropout(0.3),
    nn.Linear(512, 256),
    nn.LeakyReLU(0.2),
    nn.Dropout(0.3),
    nn.Linear(256, 1),
    nn.Sigmoid()
).to(device)

bce_fn = nn.BCELoss()
opt_G = optim.Adam(gan_G.parameters(), lr=2e-4, betas=(0.5, 0.999))
opt_D = optim.Adam(gan_D.parameters(), lr=2e-4, betas=(0.5, 0.999))


gan_epochs = 30
gan_g_losses = []
gan_d_losses = []
gan_times = []

for epoch in range(gan_epochs):
    t_start = time.time()
    gan_G.train()
    gan_D.train()
    d_total = 0.0
    g_total = 0.0

    for imgs, _ in train_loader:
        bs = imgs.size(0)
        real_x = imgs.view(bs, -1).to(device) * 2 - 1
        real_y = torch.ones(bs, 1, device=device)
        fake_y = torch.zeros(bs, 1, device=device)

        # Train Discriminator
        opt_D.zero_grad()
        z = torch.randn(bs, z_dim, device=device)
        fake_x = gan_G(z).detach()

        d_loss_real = bce_fn(gan_D(real_x), real_y)
        d_loss_fake = bce_fn(gan_D(fake_x), fake_y)
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        opt_D.step()

        # Train Generator
        opt_G.zero_grad()
        z = torch.randn(bs, z_dim, device=device)
        gen_x = gan_G(z)
        g_loss = bce_fn(gan_D(gen_x), real_y)
        g_loss.backward()
        opt_G.step()

        d_total += d_loss.item()
        g_total += g_loss.item()

    ep_time = time.time() - t_start
    avg_d = d_total / len(train_loader)
    avg_g = g_total / len(train_loader)

    gan_d_losses.append(avg_d)
    gan_g_losses.append(avg_g)
    gan_times.append(ep_time)
    print(f"Epoch {epoch+1:02d}/{gan_epochs} | D Loss: {avg_d:.4f} | G Loss: {avg_g:.4f} | Time: {ep_time:.2f}s")


torch.save(gan_G.state_dict(), "outputs/models/gan_G.pt")
torch.save(gan_D.state_dict(), "outputs/models/gan_D.pt")

np.save("outputs/models/gan_g_losses.npy", np.array(gan_g_losses))
np.save("outputs/models/gan_d_losses.npy", np.array(gan_d_losses))



fid_metric = FrechetInceptionDistance(feature=2048).to(device)
gan_G.eval()

with torch.no_grad():
    for imgs, _ in test_loader:
        real_rgb = (imgs.to(device).clamp(0, 1) * 255).to(torch.uint8).repeat(1, 3, 1, 1)
        fid_metric.update(real_rgb, real=True)

# Generate sampel baru untuk hitung FID Score
torch.manual_seed(seed)
n_total = len(test_data)
n_done = 0
with torch.no_grad():
    while n_done < n_total:
        bs = min(128, n_total - n_done)
        z_rand = torch.randn(bs, z_dim, device=device)
        fake_img = (gan_G(z_rand).view(-1, 1, 28, 28) + 1) / 2
        fake_rgb = (fake_img.clamp(0, 1) * 255).to(torch.uint8).repeat(1, 3, 1, 1)
        fid_metric.update(fake_rgb, real=False)
        n_done += bs

gan_fid = fid_metric.compute().item()
gan_avg_time = np.mean(gan_times)

# inference latency single-pass
z_64 = torch.randn(64, z_dim, device=device)
with torch.no_grad():
    for _ in range(10):
        _ = gan_G(z_64)
    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(100):
        _ = gan_G(z_64)
    if device.type == "cuda":
        torch.cuda.synchronize()
    gan_latency = ((time.time() - t0) / 100) * 1000

print(" === Evaluasi Vanilla GAN ===")
print("Dimensi Laten: 100 (Prior Noise)")
print(f"Training Time/Epoch: {gan_avg_time:.2f} detik")
print(f"Latency: {gan_latency:.3f} ms")
print("Test MSE: N/A (Tanpa Encoder)")
print("Test PSNR: N/A")
print("Test SSIM: N/A")
print(f"FID Score: {gan_fid:.2f}")



torch.manual_seed(seed)
with torch.no_grad():
    z_16 = torch.randn(16, z_dim, device=device)
    gan_samples = ((gan_G(z_16).view(16, 28, 28) + 1) / 2).clamp(0, 1).cpu()

fig, axes = plt.subplots(4, 4, figsize=(5.5, 5.5))
for i in range(16):
    ax = axes[i // 4, i % 4]
    ax.imshow(gan_samples[i], cmap="gray")
    ax.axis("off")

plt.suptitle("GAN Random Sampling")
plt.tight_layout()
plt.savefig("outputs/figs/gan_samples.png", dpi=150)
plt.close()

plt.figure(figsize=(7, 4))
plt.plot(range(1, gan_epochs + 1), gan_d_losses, label="Discriminator Loss")
plt.plot(range(1, gan_epochs + 1), gan_g_losses, label="Generator Loss")
plt.xlabel("Epoch")
plt.ylabel("BCE Loss")
plt.title("Training Loss GAN")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/figs/gan_loss.png", dpi=150)
plt.close()
