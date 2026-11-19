import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

from data_setup import train_loader, val_loader, test_loader, test_data, classes, seed

torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs("outputs/models", exist_ok=True)
os.makedirs("outputs/figs", exist_ok=True)

# MLP Classifier
mlp_enc = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 64),
    nn.ReLU()
).to(device)

mlp_head = nn.Linear(64, 10).to(device)

loss_fn = nn.CrossEntropyLoss()
opt = optim.Adam(list(mlp_enc.parameters()) + list(mlp_head.parameters()), lr=1e-3)

epochs = 14


for epoch in range(epochs):
    mlp_enc.train()
    mlp_head.train()
    total_loss = 0

    for imgs, labels in train_loader:
        x = imgs.view(imgs.size(0), -1).to(device)
        y = labels.to(device)

        opt.zero_grad()
        out = mlp_head(mlp_enc(x))
        loss = loss_fn(out, y)
        loss.backward()
        opt.step()
        total_loss += loss.item()

    mlp_enc.eval()
    mlp_head.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for imgs, labels in val_loader:
            x = imgs.view(imgs.size(0), -1).to(device)
            y = labels.to(device)
            preds = mlp_head(mlp_enc(x)).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    val_acc = 100 * correct / total
    print(f"MLP Epoch {epoch+1:02d}/{epochs} | Loss: {total_loss/len(train_loader):.4f} | Val Acc: {val_acc:.2f}%")

correct, total = 0, 0
with torch.no_grad():
    for imgs, labels in test_loader:
        x = imgs.view(imgs.size(0), -1).to(device)
        y = labels.to(device)
        preds = mlp_head(mlp_enc(x)).argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

test_acc = 100 * correct / total
print(f"Test Accuracy MLP: {test_acc:.2f}%")

torch.save(mlp_enc.state_dict(), "outputs/models/mlp_enc.pt")
torch.save(mlp_head.state_dict(), "outputs/models/mlp_head.pt")

W1 = mlp_enc[0].weight.detach()
W2 = mlp_enc[2].weight.detach()
W3 = mlp_head.weight.detach()

eye64 = torch.eye(64, device=device)
ortho_err = (W3.T @ W3 - eye64).abs().max().item()
print(f"{ortho_err:.4f}")


# Top-Down Weight Transposition
y_onehot = torch.eye(10, device=device)
z_hat = F.relu(y_onehot @ W3)
h1_hat = F.relu(z_hat @ W2)
x_from_y = torch.sigmoid(h1_hat @ W1).view(10, 28, 28).cpu()

fig, axes = plt.subplots(2, 5, figsize=(10, 4))
for i in range(10):
    ax = axes[i // 5, i % 5]
    ax.imshow(x_from_y[i], cmap="gray")
    ax.set_title(classes[i], fontsize=9)
    ax.axis("off")
plt.suptitle("Top-Down Weight Transposition")
plt.tight_layout()
plt.savefig("outputs/figs/mlp_topdown.png", dpi=150)
plt.close()

# Latent Transpose Projection
picked_imgs = {}
for img, label in test_data:
    if label not in picked_imgs:
        picked_imgs[label] = img
    if len(picked_imgs) == 10:
        break

sample_imgs = torch.stack([picked_imgs[i] for i in range(10)])
sample_labels = list(range(10))
x_real = sample_imgs.view(10, -1).to(device)


with torch.no_grad():
    z_sample = mlp_enc(x_real)
    x_from_z = torch.sigmoid(F.relu(z_sample @ W2) @ W1).view(10, 28, 28).cpu()

fig, axes = plt.subplots(2, 10, figsize=(14, 3.2))
for i in range(10):
    axes[0, i].imshow(sample_imgs[i].squeeze(), cmap="gray")
    axes[0, i].set_title(classes[sample_labels[i]], fontsize=8)
    axes[0, i].set_xticks([])
    axes[0, i].set_yticks([])

    axes[1, i].imshow(x_from_z[i], cmap="gray")
    axes[1, i].set_xticks([])
    axes[1, i].set_yticks([])

axes[0, 0].set_ylabel("Asli", fontsize=10)
axes[1, 0].set_ylabel("W^T (z_MLP)", fontsize=10)
plt.suptitle("Citra Asli vs Proyeksi Balik dari z_MLP")
plt.tight_layout()
plt.savefig("outputs/figs/mlp_latent_transpose.png", dpi=150)
plt.close()

# Frozen Latent Decoding

# Freeze seluruh bobot MLP Classifier
for p in mlp_enc.parameters():
    p.requires_grad = False
for p in mlp_head.parameters():
    p.requires_grad = False
mlp_enc.eval()

# Decoder terpisah (64 -> 256 -> 784)
frozen_dec = nn.Sequential(
    nn.Linear(64, 256),
    nn.ReLU(),
    nn.Linear(256, 784),
    nn.Sigmoid()
).to(device)

mse_fn = nn.MSELoss()
dec_opt = optim.Adam(frozen_dec.parameters(), lr=1e-3)

dec_epochs = 15
for epoch in range(dec_epochs):
    frozen_dec.train()
    total_loss = 0

    for imgs, _ in train_loader:
        x = imgs.view(imgs.size(0), -1).to(device)
        with torch.no_grad():
            z_mlp = mlp_enc(x)

        dec_opt.zero_grad()
        recon = frozen_dec(z_mlp)
        loss = mse_fn(recon, x)
        loss.backward()
        dec_opt.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1:02d}/{dec_epochs} | Train MSE: {total_loss/len(train_loader):.4f}")


psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

frozen_dec.eval()
mse_sum = 0.0
total_pixels = 0

with torch.no_grad():
    for imgs, _ in test_loader:
        x_img = imgs.to(device)
        x_flat = x_img.view(x_img.size(0), -1)
        recon_img = frozen_dec(mlp_enc(x_flat)).view(-1, 1, 28, 28)

        mse_sum += F.mse_loss(recon_img, x_img, reduction="sum").item()
        total_pixels += x_img.numel()
        psnr_metric.update(recon_img, x_img)
        ssim_metric.update(recon_img, x_img)

mlp_dec_mse = mse_sum / total_pixels
mlp_dec_psnr = psnr_metric.compute().item()
mlp_dec_ssim = ssim_metric.compute().item()

print(f"Test MSE: {mlp_dec_mse:.5f}")
print(f"PSNR: {mlp_dec_psnr:.2f} dB")
print(f"SSIM: {mlp_dec_ssim:.4f}")

torch.save(frozen_dec.state_dict(), "outputs/models/frozen_dec.pt")

with torch.no_grad():
    x_frozen = frozen_dec(mlp_enc(x_real)).view(10, 28, 28).cpu()

fig, axes = plt.subplots(2, 10, figsize=(14, 3.2))
for i in range(10):
    axes[0, i].imshow(sample_imgs[i].squeeze(), cmap="gray")
    axes[0, i].set_title(classes[sample_labels[i]], fontsize=8)
    axes[0, i].set_xticks([])
    axes[0, i].set_yticks([])

    axes[1, i].imshow(x_frozen[i], cmap="gray")
    axes[1, i].set_xticks([])
    axes[1, i].set_yticks([])

axes[0, 0].set_ylabel("Asli", fontsize=10)
axes[1, 0].set_ylabel("Frozen Dec", fontsize=10)
plt.suptitle("Citra Asli vs Rekonstruksi Frozen Latent Decoder")
plt.tight_layout()
plt.savefig("outputs/figs/mlp_frozen_dec.png", dpi=150)
plt.close()


# Uji Sampling Generatif pada Ruang Laten MLP Classifier

z_list = []

with torch.no_grad():
    for imgs, _ in test_loader:
        x = imgs.view(imgs.size(0), -1).to(device)
        z_list.append(mlp_enc(x))

all_z_mlp = torch.cat(z_list, dim=0)
mu_z = all_z_mlp.mean(dim=0)
std_z = all_z_mlp.std(dim=0)
zero_pct = (all_z_mlp == 0).float().mean().item() * 100
print(f"Mean: [{mu_z.min():.2f}, {mu_z.max():.2f}]")
print(f"Std: [{std_z.min():.2f}, {std_z.max():.2f}]")
print(f"Elemen bernilai 0: {zero_pct:.1f}%")


z_rand = mu_z + std_z * torch.randn(16, 64, device=device)
with torch.no_grad():
    x_rand = frozen_dec(z_rand).view(16, 28, 28).cpu()

fig, axes = plt.subplots(4, 4, figsize=(5.5, 5.5))

for i in range(16):
    ax = axes[i // 4, i % 4]
    ax.imshow(x_rand[i], cmap="gray")
    ax.axis("off")
plt.suptitle("Random Sampling z_rand dari Latent Space MLP Classifier")
plt.tight_layout()
plt.savefig("outputs/figs/mlp_prior_sampling.png", dpi=150)
plt.close()
