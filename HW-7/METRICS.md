# METRICS — DATA 266 HW7

Sneha Singh  
SID4 = 0670 | SEED = 670 | SLICE = 670 | HP_ID = 4 | CLS_A = 0 | CLS_B = 5  
HP_ID is reported only. Two trained models: fully connected autoencoder (FC-AE) and VAE.

From `autoencoders_fashionmnist.ipynb` on Apple Silicon MPS (PyTorch 2.14.0). Same numbers as `train_eval.py`, run separately.

## Setup

| Field | Value |
|-------|-------|
| Dataset | FashionMNIST, 28×28 grayscale, 10 classes |
| Train / test | 60,000 / 10,000 |
| Latent size | 20 |
| Architecture | 784-512-128-20, mirrored decoder, sigmoid output |
| Optimizer | Adam, lr 0.001 |
| Batch / epochs | 128 / 20 |
| Loss | BCE summed over 784 pixels, averaged per image (VAE adds KL, weight 1) |
| Device | mps |
| Interpolation pair | first T-shirt/top (class 0) and first Sandal (class 5) in the test set |
| t-SNE | 2,000 test images, perplexity 30 |

## Parameters

| Model | Trainable |
|-------|----------:|
| FC-AE | 941,092 |
| VAE | 943,672 |

The VAE has 2,580 more: a second 128 to 20 head (μ and log σ²).

## Training loss

| Epoch | FC-AE BCE | VAE BCE | VAE KL | VAE total |
|------:|----------:|--------:|-------:|----------:|
| 1 | 267.6440 | 279.8671 | 11.4876 | 291.3547 |
| 5 | 220.9310 | 233.3050 | 11.6220 | 244.9271 |
| 10 | 215.0799 | 227.6862 | 12.5984 | 240.2846 |
| 15 | 212.4944 | 225.8245 | 12.8035 | 238.6279 |
| 20 | 211.3226 | 224.8707 | 12.9265 | 237.7973 |

All 20 epochs are in `RUN_LOG.txt`.

## Test set (10,000 images)

| Metric | FC-AE | VAE |
|--------|------:|----:|
| Recon BCE per image | 213.6070 | 226.8435 |
| Recon MSE per pixel | 0.009739 | 0.015663 |
| KL per image | none | 13.0439 |
| BCE + KL | 213.6070 | 239.8874 |

BCE + KL is not a fair comparison, since the FC-AE has no KL term.

Plots: `figures/loss_curves.png`, `figures/reconstructions_comparison.png`, `figures/random_generation.png`, `figures/latent_interpolation.png`, `figures/latent_tsne.png`
