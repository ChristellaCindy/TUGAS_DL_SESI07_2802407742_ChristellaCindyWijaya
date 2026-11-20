# TUGAS_DL_SESI07_2802407742_ChristellaCindyWijaya
Studi Komparatif Empiris Model Generatif: Autoencoder (Vanilla &amp; Sparse), GAN (Vanilla &amp; DCGAN), VAE, dan  DDPM disertai Pembuktian Batasan Teoretis Model Klasifikasi untuk Image Generation 

Repositori ini berisi implementasi berbasis PyTorch untuk eksperimen MLP Classifier serta enam arsitektur model generatif (**Vanilla AE, Sparse AE, VAE, Vanilla GAN, DCGAN, dan DDPM**) menggunakan dataset **Fashion-MNIST**.

**Identitas**

* **Nama:** Christella Cindy Wijaya

* **NIM:** 2802407742



---



## Struktur Repositori



```text

.

├── TUGAS_DL_SESI07_2802407742_ChristellaCindyWijaya.ipynb  # Notebook utama eksperimen (Google Colab)

├── data_setup.py            # Preprocessing, split dataset, dan DataLoader Fashion-MNIST

├── mlp_experiment.py        # Eksperimen MLP Classifier, uji inversi bobot, dan Frozen Decoder

├── vanilla_ae.py            # Training dan evaluasi Vanilla Autoencoder

├── sparse_ae.py             # Training dan evaluasi Sparse Autoencoder (L1)

├── vae.py                   # Training dan evaluasi Variational Autoencoder (VAE)

├── gan.py                   # Training dan evaluasi Vanilla GAN

├── dcgan.py                 # Training dan evaluasi Deep Convolutional GAN (DCGAN)

├── ddpm.py                  # Training dan evaluasi DDPM (Mini U-Net)

├── generate_figures.py      # Skrip untuk menghasilkan seluruh visualisasi laporan dari bobot tersimpan

├── requirements.txt         # Daftar pustaka Python yang diperlukan

├── .gitignore               # Pengecualian folder dataset dan output lokal

└── README.md                # Dokumentasi repositori

```



---



## Instalasi



1. Clone repositori ini:

```bash

git clone https://github.com/ChristellaCindy/TUGAS_DL_SESI07_2802407742_ChristellaCindyWijaya.git

cd TUGAS_DL_SESI07_2802407742_ChristellaCindyWijaya

```



2. Instal seluruh dependensi:

```bash

pip install -r requirements.txt

```



---



## Cara Menjalankan



### 1. Menggunakan Skrip Modular (`.py`)

Jalankan masing-masing skrip eksperimen untuk melatih model dan menyimpan bobot (`.pt`) serta riwayat loss (`.npy`) ke folder `outputs/models/`:



```bash

python mlp_experiment.py

python vanilla_ae.py

python sparse_ae.py

python vae.py

python gan.py

python dcgan.py

python ddpm.py

```



Setelah seluruh bobot tersimpan, jalankan skrip berikut untuk menghasilkan seluruh gambar evaluasi di folder `outputs/figs/`:



```bash

python generate_figures.py

```



### 2. Menggunakan Notebook (`.ipynb`)

Buka berkas `TUGAS_DL_SESI07_2802407742_ChristellaCindyWijaya.ipynb` di Google Colab dengan runtime GPU (disarankan T4 GPU), lalu jalankan sel secara berurutan.
