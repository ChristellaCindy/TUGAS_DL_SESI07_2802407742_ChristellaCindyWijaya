import os
import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchmetrics.image.fid import FrechetInceptionDistance

from data_setup import train_loader, val_loader, test_loader, test_data, seed

torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs("outputs/models", exist_ok=True)
os.makedirs("outputs/figs", exist_ok=True)


T = 300
beta = torch.linspace(1e-4, 0.035, T, device=device)
alpha = 1.0 - beta
alpha_bar = torch.cumprod(alpha, dim=0)

def get_time_embeded(t, dim=64):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    args = t[:, None].float() * freqs[None, :]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

def conv_block(in_c, out_c):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 3, padding=1),
        nn.GroupNorm(8, out_c),
        nn.SiLU(),
        nn.Conv2d(out_c, out_c, 3, padding=1),
        nn.GroupNorm(8, out_c),
        nn.SiLU()
    )

class MiniUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(64, 128),
            nn.SiLU(),
            nn.Linear(128, 128)
        )

        self.down1 = conv_block(1, 64)
        self.pool1 = nn.Conv2d(64, 64, 4, stride=2, padding=1)
        self.down2 = conv_block(64, 128)
        self.pool2 = nn.Conv2d(128, 128, 4, stride=2, padding=1)
        self.mid = conv_block(128, 128)

        self.up2 = nn.ConvTranspose2d(128, 128, 4, stride=2, padding=1)
        self.dec2 = conv_block(256, 64)
        self.up1 = nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1)
        self.dec1 = conv_block(128, 64)
        self.out = nn.Conv2d(64, 1, 1)

        self.t_d1 = nn.Linear(128, 64)
        self.t_d2 = nn.Linear(128, 128)
        self.t_mid = nn.Linear(128, 128)
        self.t_u2 = nn.Linear(128, 64)
        self.t_u1 = nn.Linear(128, 64)

    def forward(self, x, t):
        te = self.time_mlp(get_time_embeded(t, 64))

        h1 = self.down1(x) + self.t_d1(te)[:, :, None, None]
        h2 = self.down2(self.pool1(h1)) + self.t_d2(te)[:, :, None, None]
        m = self.mid(self.pool2(h2)) + self.t_mid(te)[:, :, None, None]

        u2 = torch.cat([self.up2(m), h2], dim=1)
        u2 = self.dec2(u2) + self.t_u2(te)[:, :, None, None]
        u1 = torch.cat([self.up1(u2), h1], dim=1)
        u1 = self.dec1(u1) + self.t_u1(te)[:, :, None, None]
        return self.out(u1)

ddpm_net = MiniUNet().to(device)
mse_fn = nn.MSELoss()
ddpm_opt = optim.Adam(ddpm_net.parameters(), lr=2e-4)


ddpm_epochs = 30
ddpm_loss = []
ddpm_times = []

for epoch in range(ddpm_epochs):
    t_start = time.time()
    ddpm_net.train()
    train_loss = 0.0

    for imgs, _ in train_loader:
        x0 = imgs.to(device) * 2 - 1
        bs = x0.size(0)
        t = torch.randint(0, T, (bs,), device=device)
        noise = torch.randn_like(x0)

        # Forward
        ab = alpha_bar[t][:, None, None, None]
        x_t = torch.sqrt(ab) * x0 + torch.sqrt(1 - ab) * noise

        ddpm_opt.zero_grad()
        pred_noise = ddpm_net(x_t, t)
        loss = mse_fn(pred_noise, noise)
        loss.backward()
        ddpm_opt.step()
        train_loss += loss.item()

    ddpm_net.eval()
    val_loss = 0.0

    with torch.no_grad():

        for imgs, _ in val_loader:
            x0 = imgs.to(device) * 2 - 1
            bs = x0.size(0)
            t = torch.randint(0, T, (bs,), device=device)
            noise = torch.randn_like(x0)
            ab = alpha_bar[t][:, None, None, None]
            x_t = torch.sqrt(ab) * x0 + torch.sqrt(1 - ab) * noise
            val_loss += mse_fn(ddpm_net(x_t, t), noise).item()

    ep_time = time.time() - t_start
    avg_train = train_loss / len(train_loader)
    avg_val = val_loss / len(val_loader)

    ddpm_loss.append(avg_train)
    ddpm_times.append(ep_time)
    print(f"Epoch {epoch+1:02d}/{ddpm_epochs} | Train MSE: {avg_train:.5f} | Val MSE: {avg_val:.5f} | Time: {ep_time:.2f}s")

torch.save(ddpm_net.state_dict(), "outputs/models/ddpm.pt")
np.save("outputs/models/ddpm_loss.npy", np.array(ddpm_loss))

# Reverse
def sample_ddpm(n):
    ddpm_net.eval()

    with torch.no_grad():
        x = torch.randn(n, 1, 28, 28, device=device)

        for i in reversed(range(T)):
            t = torch.full((n,), i, device=device, dtype=torch.long)
            pred = ddpm_net(x, t)
            a = alpha[i]
            ab = alpha_bar[i]
            b = beta[i]

            x = (1 / torch.sqrt(a)) * (x - ((1 - a) / torch.sqrt(1 - ab)) * pred)
            if i > 0:
                var = b * (1 - alpha_bar[i - 1]) / (1 - ab)
                x = x + torch.sqrt(var) * torch.randn_like(x)
        return (x.clamp(-1, 1) + 1) / 2

fid_metric = FrechetInceptionDistance(feature=2048).to(device)

with torch.no_grad():
    for imgs, _ in test_loader:
        real_rgb = (imgs.to(device).clamp(0, 1) * 255).to(torch.uint8).repeat(1, 3, 1, 1)
        fid_metric.update(real_rgb, real=True)

# Generate sample baru untuk hitung FID Score
torch.manual_seed(seed)
n_total = len(test_data)
n_done = 0
while n_done < n_total:
    bs = min(256, n_total - n_done)
    fake_img = sample_ddpm(bs)
    fake_rgb = (fake_img.clamp(0, 1) * 255).to(torch.uint8).repeat(1, 3, 1, 1)
    fid_metric.update(fake_rgb, real=False)
    n_done += bs

ddpm_fid = fid_metric.compute().item()
ddpm_avg_time = np.mean(ddpm_times)

# inference latency
_ = sample_ddpm(64)

if device.type == "cuda":
    torch.cuda.synchronize()

t0 = time.time()

for _ in range(5):
    _ = sample_ddpm(64)

if device.type == "cuda":
    torch.cuda.synchronize()

ddpm_latency = ((time.time() - t0) / 5) * 1000

print("=== Evaluasi DDPM ===")
print("Dimensi Laten: 28x28 (Dimensi Citra)")
print(f"Timesteps (T): {T}")
print(f"Training Time/Epoch: {ddpm_avg_time:.2f} detik")
print(f"Latency: {ddpm_latency:.2f} ms")
print("Test MSE: N/A (Denoising Model)")
print("Test PSNR: N/A")
print("Test SSIM: N/A")
print(f"FID Score: {ddpm_fid:.2f}")


torch.manual_seed(seed)
ddpm_samples = sample_ddpm(16).squeeze(1).cpu()

fig, axes = plt.subplots(4, 4, figsize=(5.5, 5.5))
for i in range(16):
    ax = axes[i // 4, i % 4]
    ax.imshow(ddpm_samples[i], cmap="gray")
    ax.axis("off")

plt.suptitle(f"DDPM Random Sampling")
plt.tight_layout()
plt.savefig("outputs/figs/ddpm_samples.png", dpi=150)
plt.close()

plt.figure(figsize=(7, 4))
plt.plot(range(1, ddpm_epochs + 1), ddpm_loss, label="Noise Prediction MSE Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Training Loss DDPM")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/figs/ddpm_loss.png", dpi=150)
plt.close()
