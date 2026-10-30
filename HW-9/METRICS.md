# METRICS — DATA 266 HW9

Sneha Singh  
SID4 = 0670 | SEED = 670 | SLICE = 670 | HP_ID = 4 | CLS_A = 0 | CLS_B = 5

HP_ID is reported only. Two models: a frozen pretrained VAE, and one conditional U-Net.

The numbers below are from `latent_diffusion.ipynb`, executed on Apple Silicon MPS. PyTorch 2.14.0.

## Setup

| Item | Value |
|------|-------|
| Dataset | FashionMNIST, resized to 128×128, 3 channels |
| VAE | `stabilityai/sd-vae-ft-mse`, frozen |
| Latent | 4×16×16, posterior mean, times 0.18215 |
| Device | MPS |
| Diffusion steps | 1000 |
| β start | 0.0001 |
| β end | 0.02 |
| Batch size | 128 |
| Epochs | 50 |
| Learning rate | 2e-5 |
| Optimizer | Adam |
| Label dropout | 0.1, null label = 10 |
| Guidance scale w | 3 |
| U-Net | base 64, multipliers (1, 2, 4), GroupNorm, SiLU, self-attention at 16×16 and the bottleneck |
| U-Net parameters | 12,289,540 |
| Model output shape | `(2, 4, 16, 16)` |

## VAE reconstruction

| Split | MSE per element, inputs in [-1, 1] |
|-------|-----------------------------------:|
| Test, 10,000 images | 0.000781 |

## Forward schedule

| time | index | ᾱ |
|------|------:|---:|
| 200 | 200 | 6.563466e-01 |
| 500 | 500 | 7.779661e-02 |
| 800 | 800 | 1.507520e-03 |
| 1000 | 999 | 4.035827e-05 |

## Training loss

| Epoch | Noise-prediction MSE |
|------:|---------------------:|
| 1 | 0.384546 |
| 10 | 0.112722 |
| 20 | 0.103292 |
| 30 | 0.099690 |
| 40 | 0.099033 |
| 50 | 0.096441 |

The loss dropped quickly at the beginning and then flattened near 0.096. Training took 111.44 minutes. All 50 epochs are in `RUN_LOG.txt`.

## Sampling

| Item | Value |
|------|-------|
| Reverse steps | 1000 |
| Grid | 64 images, classes 0–9 repeated row-major, w = 3 |
| 64-image sampling time | 1.08 minutes |

The notebook saved the grid under `outputs/` and the figures under `figures/`.
