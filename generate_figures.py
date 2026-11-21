import os
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from torchvision.utils import make_grid

from data_setup import test_loader, test_data, classes, seed

def set_seed(s=42):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("outputs/figs", exist_ok=True)


z_dim = 100

# MLP & Frozen Decoder
mlp_enc = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 64),
    nn.ReLU()
).to(device)
mlp_head = nn.Linear(64, 10).to(device)
frozen_dec = nn.Sequential(
    nn.Linear(64, 256),
    nn.ReLU(),
    nn.Linear(256, 784),
    nn.Sigmoid()
).to(device)

# Vanilla AE
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

# Sparse AE
sae_enc = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 64)
).to(device)
sae_dec = nn.Sequential(
    nn.Linear(64, 256),
    nn.ReLU(),
    nn.Linear(256, 784),
    nn.Sigmoid()
).to(device)

# VAE
vae_enc = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU()
).to(device)
vae_mu = nn.Linear(256, 64).to(device)
vae_dec = nn.Sequential(
    nn.Linear(64, 256),
    nn.ReLU(),
    nn.Linear(256, 784),
    nn.Sigmoid()
).to(device)

# Vanilla GAN Generator
gan_G = nn.Sequential(
    nn.Linear(z_dim, 256),
    nn.LeakyReLU(0.2),
    nn.Linear(256, 512),
    nn.LeakyReLU(0.2),
    nn.Linear(512, 784),
    nn.Tanh()
).to(device)

# DCGAN Generator
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

# DDPM
T = 300
betas = torch.linspace(1e-4, 0.035, T, device=device)
alpha = 1.0 - betas
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

model_files = [
    (mlp_enc, "mlp_enc.pt"),
    (mlp_head, "mlp_head.pt"),
    (frozen_dec, "frozen_dec.pt"),
    (ae_enc, "ae_enc.pt"),
    (ae_dec, "ae_dec.pt"),
    (sae_enc, "sae_enc.pt"),
    (sae_dec, "sae_dec.pt"),
    (vae_enc, "vae_enc.pt"),
    (vae_mu, "vae_mu.pt"),
    (vae_dec, "vae_dec.pt"),
    (gan_G, "gan_G.pt"),
    (dcgan_G, "dcgan_G.pt"),
    (ddpm_net, "ddpm_net.pt"),
]

for model, fname in model_files:
    state = torch.load(f"outputs/models/{fname}", map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

# 10 Sampel Kelas & Hasil Rekonstruksi
picked_imgs = {}
for img, label in test_data:
    if label not in picked_imgs:
        picked_imgs[label] = img
    if len(picked_imgs) == 10:
        break

sample_labels = list(range(10))
sample_imgs = torch.stack([picked_imgs[i] for i in sample_labels])
x_real = sample_imgs.view(10, -1).to(device)

with torch.no_grad():
    x_ae = ae_dec(ae_enc(x_real)).view(10, 28, 28).cpu()
    x_sae = sae_dec(sae_enc(x_real)).view(10, 28, 28).cpu()
    x_vae = vae_dec(vae_mu(vae_enc(x_real))).view(10, 28, 28).cpu()


# Figur 1: Perbandingan Rekonstruksi Autoencoder
fig, axes = plt.subplots(4, 10, figsize=(14, 5.8))
rows = [sample_imgs.squeeze(1), x_ae, x_sae, x_vae]
row_names = ["Asli", "Vanilla AE", "Sparse AE", "VAE"]

for r in range(4):
    for c in range(10):
        axes[r, c].imshow(rows[r][c], cmap="gray", vmin=0, vmax=1)
        if r == 0:
            axes[r, c].set_title(classes[sample_labels[c]], fontsize=8)
        axes[r, c].set_xticks([])
        axes[r, c].set_yticks([])
    axes[r, 0].set_ylabel(row_names[r], fontsize=10, fontweight="bold")

plt.suptitle("Perbandingan Rekonstruksi Citra", fontsize=12)
plt.tight_layout()
plt.savefig("outputs/figs/rekonstruksi_ae.png", dpi=300, bbox_inches="tight")
plt.close()


# Figur 2: Perbandingan Random Generative Sampling
def sample_ddpm(n):
    ddpm_net.eval()
    with torch.no_grad():
        x = torch.randn(n, 1, 28, 28, device=device)
        for i in reversed(range(T)):
            t = torch.full((n,), i, device=device, dtype=torch.long)
            pred = ddpm_net(x, t)
            a = alpha[i]
            ab = alpha_bar[i]
            b = betas[i]

            x = (1 / torch.sqrt(a)) * (x - ((1 - a) / torch.sqrt(1 - ab)) * pred)
            if i > 0:
                var = b * (1 - alpha_bar[i - 1]) / (1 - ab)
                x = x + torch.sqrt(var) * torch.randn_like(x)
        return (x.clamp(-1, 1) + 1) / 2

with torch.no_grad():
    set_seed(seed)
    vae_samples = vae_dec(torch.randn(16, 64, device=device)).view(16, 28, 28).cpu()

    set_seed(seed)
    gan_samples = ((gan_G(torch.randn(16, z_dim, device=device)).view(16, 28, 28) + 1) / 2).clamp(0, 1).cpu()

    set_seed(seed)
    dcgan_samples = ((dcgan_G(torch.randn(16, z_dim, device=device)) + 1) / 2).clamp(0, 1).squeeze(1).cpu()

set_seed(seed)
ddpm_samples = sample_ddpm(16).squeeze(1).cpu()

panels = [
    ("VAE", vae_samples),
    ("Vanilla GAN", gan_samples),
    ("DCGAN", dcgan_samples),
    ("DDPM", ddpm_samples),
]

fig, axes = plt.subplots(2, 2, figsize=(9, 9.2))
for idx, (title, imgs) in enumerate(panels):
    ax = axes[idx // 2, idx % 2]
    grid = make_grid(imgs.detach().cpu().view(16, 1, 28, 28), nrow=4, padding=2, pad_value=1.0)
    ax.imshow(grid[0], cmap="gray", vmin=0, vmax=1)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.axis("off")

plt.suptitle("Perbandingan Random Generative Sampling", fontsize=13)
plt.tight_layout()
plt.savefig("outputs/figs/random_sampling.png", dpi=300, bbox_inches="tight")
plt.close()



# Figur 3: Batasan Bobot dan Latent Features MLP Classifier
w_enc = [m.weight for m in mlp_enc.modules() if isinstance(m, nn.Linear)]
w_head = [m.weight for m in mlp_head.modules() if isinstance(m, nn.Linear)]
w1, w2, w3 = w_enc[0], w_enc[1], w_head[0]

z_all = []
with torch.no_grad():
    for imgs, _ in test_loader:
        z_all.append(mlp_enc(imgs.view(imgs.size(0), -1).to(device)))

z_all = torch.cat(z_all, dim=0)
mu_z = z_all.mean(dim=0)
std_z = z_all.std(dim=0)

with torch.no_grad():
    z_mlp = mlp_enc(x_real)

    # Proyeksi balik W^T dari vektor kelas y_c
    y_idx = torch.as_tensor(sample_labels, device=device, dtype=torch.long)
    y_c = F.one_hot(y_idx, num_classes=len(classes)).float()
    x_wt_yc = torch.sigmoid(F.relu(F.relu(y_c @ w3) @ w2) @ w1).view(10, 28, 28).cpu()

    # Proyeksi balik W^T dari fitur laten z_MLP
    x_wt_z = torch.sigmoid(F.relu(z_mlp @ w2) @ w1).view(10, 28, 28).cpu()
    
    # Rekonstruksi Frozen Latent Decoder dari z_MLP
    x_frozen = frozen_dec(z_mlp).view(10, 28, 28).cpu()
    
    # Random Prior Sampling z_rand
    set_seed(seed)
    z_rand_mlp = mu_z + std_z * torch.randn(10, mu_z.numel(), device=device)
    x_mlp_rand = frozen_dec(z_rand_mlp).view(10, 28, 28).cpu()

fig, axes = plt.subplots(6, 10, figsize=(14, 8.6))
mlp_rows = [
    sample_imgs.squeeze(1),
    x_wt_yc,
    x_wt_z,
    x_frozen,
    x_ae,
    x_mlp_rand,
]
mlp_names = [
    "(1) Asli",
    "(2a) W^T (y_c)",
    "(2b) W^T (z_MLP)",
    "(3a) Frozen Dec",
    "(3b) Vanilla AE",
    "(4) Random Prior",
]

for r in range(6):
    for c in range(10):
        axes[r, c].imshow(mlp_rows[r][c], cmap="gray", vmin=0, vmax=1)
        if r == 0:
            axes[r, c].set_title(classes[sample_labels[c]], fontsize=8)
        axes[r, c].set_xticks([])
        axes[r, c].set_yticks([])
    axes[r, 0].set_ylabel(mlp_names[r], fontsize=9, fontweight="bold")

plt.suptitle("Analisis Batasan Bobot dan Latent Features MLP Classifier", fontsize=13)
plt.tight_layout()
plt.savefig("outputs/figs/mlp_analisis.png", dpi=300, bbox_inches="tight")
plt.close()


# Figur 4: Linear Latent Space Interpolation
alphas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

cls_a = classes.index("Trouser")
cls_b = classes.index("Ankle boot")

idx_a = sample_labels.index(cls_a)
idx_b = sample_labels.index(cls_b)

x_a = x_real[idx_a:idx_a + 1]
x_b = x_real[idx_b:idx_b + 1]

def pick_gan_pair(gen_model):
    gen_model.eval()
    set_seed(seed)
    with torch.no_grad():
        z_pool = torch.randn(512, z_dim, device=device)
        imgs = (gen_model(z_pool) + 1) / 2
        imgs_flat = imgs.view(512, -1).clamp(0, 1)
        probs = F.softmax(mlp_head(mlp_enc(imgs_flat)), dim=1)
        best_a = probs[:, cls_a].argmax().item()
        best_b = probs[:, cls_b].argmax().item()
    return z_pool[best_a:best_a + 1], z_pool[best_b:best_b + 1], probs[best_a, cls_a].item(), probs[best_b, cls_b].item()

def interp_decode(z_a, z_b, dec_fn):
    out = []
    with torch.no_grad():
        for a in alphas:
            z = (1.0 - a) * z_a + a * z_b
            out.append(dec_fn(z).view(28, 28).cpu())
    return out

with torch.no_grad():
    ae_row = interp_decode(ae_enc(x_a), ae_enc(x_b), ae_dec)
    sae_row = interp_decode(sae_enc(x_a), sae_enc(x_b), sae_dec)
    vae_row = interp_decode(vae_mu(vae_enc(x_a)), vae_mu(vae_enc(x_b)), vae_dec)

z_gan_a, z_gan_b, p_gan_a, p_gan_b = pick_gan_pair(gan_G)
gan_row = interp_decode(z_gan_a, z_gan_b, lambda z: ((gan_G(z) + 1) / 2).clamp(0, 1))

z_dc_a, z_dc_b, p_dc_a, p_dc_b = pick_gan_pair(dcgan_G)
dcgan_row = interp_decode(z_dc_a, z_dc_b, lambda z: ((dcgan_G(z) + 1) / 2).clamp(0, 1))

print(f"Kelas A: {classes[cls_a]}")
print(f"Kelas B: {classes[cls_b]}")
print(f"Confidence Vanilla GAN >> {classes[cls_a]}: {p_gan_a*100:.1f}%")
print(f"Confidence Vanilla GAN >> {classes[cls_b]}: {p_gan_b*100:.1f}%")
print(f"Confidence DCGAN >> {classes[cls_a]}: {p_dc_a*100:.1f}%")
print(f"Confidence DCGAN >> {classes[cls_b]}: {p_dc_b*100:.1f}%")

interp_rows = [ae_row, sae_row, vae_row, gan_row, dcgan_row]
interp_names = ["Vanilla AE", "Sparse AE", "VAE", "Vanilla GAN", "DCGAN"]

fig, axes = plt.subplots(5, len(alphas), figsize=(10.5, 8.8))
for r in range(5):
    for c, a in enumerate(alphas):
        axes[r, c].imshow(interp_rows[r][c], cmap="gray", vmin=0, vmax=1)
        if r == 0:
            axes[r, c].set_title(f"alpha = {a:.1f}", fontsize=10, fontweight="bold")
        axes[r, c].set_xticks([])
        axes[r, c].set_yticks([])
    axes[r, 0].set_ylabel(interp_names[r], fontsize=10, fontweight="bold")

plt.suptitle(f"Linear Latent Space Interpolation ({classes[cls_a]} -> {classes[cls_b]})", fontsize=13)
plt.tight_layout()
plt.savefig("outputs/figs/latent_interp.png", dpi=300, bbox_inches="tight")
plt.close()

# Figur 5a: Training Loss (6 Model)
def load_loss(name):
    path = f"outputs/models/{name}_losses.npy"
    if not os.path.exists(path):
        raise FileNotFoundError(f"'{path}' tidak ditemukan")
    return np.load(path)

hist_ae = load_loss("ae")
hist_sae = load_loss("sae")
hist_vae = load_loss("vae")
hist_gan_g = load_loss("gan_g")
hist_gan_d = load_loss("gan_d")
hist_dcgan_g = load_loss("dcgan_g")
hist_dcgan_d = load_loss("dcgan_d")
hist_ddpm = load_loss("ddpm")

fig, axes = plt.subplots(2, 3, figsize=(15, 8))

single_plots = [
    (axes[0, 0], hist_ae, "Vanilla AE", "MSE Loss"),
    (axes[0, 1], hist_sae, "Sparse AE", "MSE + L1 Loss"),
    (axes[0, 2], hist_vae, "VAE", "ELBO Loss (BCE + KL)"),
]
for ax, data, title, ylabel in single_plots:
    ep = range(1, len(data) + 1)
    ax.plot(ep, data, linewidth=2)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, alpha=0.3)

gan_plots = [
    (axes[1, 0], hist_gan_d, hist_gan_g, "Vanilla GAN"),
    (axes[1, 1], hist_dcgan_d, hist_dcgan_g, "DCGAN"),
]
for ax, d_data, g_data, title in gan_plots:
    ep = range(1, len(d_data) + 1)
    ax.plot(ep, d_data, label="Discriminator", linewidth=2)
    ax.plot(ep, g_data, label="Generator", linewidth=2)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("BCE Loss")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend()
    ax.grid(True, alpha=0.3)

ep_ddpm = range(1, len(hist_ddpm) + 1)
axes[1, 2].plot(ep_ddpm, hist_ddpm, linewidth=2)
axes[1, 2].set_title("DDPM", fontsize=11, fontweight="bold")
axes[1, 2].set_xlabel("Epoch")
axes[1, 2].set_ylabel("Noise MSE Loss")
axes[1, 2].xaxis.set_major_locator(MaxNLocator(integer=True))
axes[1, 2].grid(True, alpha=0.3)

plt.suptitle("Training Loss Model Generatif", fontsize=13)
plt.tight_layout()
plt.savefig("outputs/figs/kurva_loss.png", dpi=300, bbox_inches="tight")
plt.close()


# Figur 5b: DDPM Denoising Trajectory
set_seed(seed)
n_traj = 4
x = torch.randn(n_traj, 1, 28, 28, device=device)

ratio = [0.60, 0.35, 0.20, 0.12, 0.06, 0.02, 0.0]
snap_steps = sorted({int(r * (T - 1)) for r in ratio}, reverse=True)

traj_imgs = [((x.clamp(-1, 1) + 1) / 2).squeeze(1).cpu()]
traj_labels = [f"t = {T}\n(Noise)"]

ddpm_net.eval()
with torch.no_grad():
    for i in reversed(range(T)):
        t = torch.full((n_traj,), i, device=device, dtype=torch.long)
        pred = ddpm_net(x, t)
        a = alpha[i]
        ab = alpha_bar[i]
        b = betas[i]

        x = (1 / torch.sqrt(a)) * (x - ((1 - a) / torch.sqrt(1 - ab)) * pred)
        if i > 0:
            var = b * (1 - alpha_bar[i - 1]) / (1 - ab)
            x = x + torch.sqrt(var) * torch.randn_like(x)

        if i in snap_steps:
            traj_imgs.append(((x.clamp(-1, 1) + 1) / 2).squeeze(1).cpu())
            label = "t = 0\n(Final)" if i == 0 else f"t = {i}"
            traj_labels.append(label)

n_cols = len(traj_imgs)
fig, axes = plt.subplots(n_traj, n_cols, figsize=(13, 6.2))
for r in range(n_traj):
    for c in range(n_cols):
        axes[r, c].imshow(traj_imgs[c][r], cmap="gray", vmin=0, vmax=1)
        if r == 0:
            axes[r, c].set_title(traj_labels[c], fontsize=9, fontweight="bold")
        axes[r, c].set_xticks([])
        axes[r, c].set_yticks([])
    axes[r, 0].set_ylabel(f"Sampel {r+1}", fontsize=10, fontweight="bold")

plt.suptitle("DDPM Denoising Trajectory", fontsize=13)
plt.tight_layout()
plt.savefig("outputs/figs/ddpm_trajectory.png", dpi=300, bbox_inches="tight")
plt.close()
