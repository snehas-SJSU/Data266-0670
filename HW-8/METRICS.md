# METRICS — DATA 266 HW8

Sneha Singh  
SID4 = 0670 | SEED = 670 | SLICE = 670 | HP_ID = 4 | CLS_A = 0 | CLS_B = 5

HP_ID is reported only. One timestep-conditioned CNN is used for all three samplers.

The numbers below are from `mini_diffusion.ipynb`, executed on Apple Silicon MPS.

## Setup

| Item | Value |
|------|-------|
| Dataset | MNIST, 28×28 grayscale |
| Device | MPS |
| Diffusion steps | 100 |
| β start | 0.0001 |
| β end | 0.02 |
| Batch size | 128 |
| Epochs | 50 |
| Learning rate | 0.001 |
| Optimizer | Adam |
| Architecture | Five convolutional layers, ReLU, timestep conditioning, one output channel |
| CNN parameters | 97,537 |
| Model output shape | `(2, 1, 28, 28)` |

## Training loss

| Epoch | Noise-prediction MSE |
|------:|---------------------:|
| 1 | 0.164547 |
| 50 | 0.061000 |

The loss dropped quickly at the beginning and then flattened near 0.061.

## Sampling

| Sampler | Reverse steps |
|---------|--------------:|
| DDPM sampling steps | 100 |
| DDIM sampling steps | 50, 20 |

The notebook saved the three image grids under `outputs/` and the comparison figures under `figures/`.
