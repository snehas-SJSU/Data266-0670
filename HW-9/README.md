# DATA 266 — Homework 9

Sneha Singh  
SID4 = 0670 | SEED = 670 | SLICE = 670 | HP_ID = 4 | CLS_A = 0 | CLS_B = 5

Findings: [`HW9_Report.pdf`](HW9_Report.pdf)

## Latent diffusion on FashionMNIST

Notebook: `latent_diffusion.ipynb`. Dataset: FashionMNIST.

I load `stabilityai/sd-vae-ft-mse` and freeze it. Images are resized to 128×128 and repeated to 3 channels, so the latent is 4×16×16. The latent is scaled by `s = 0.18215` before diffusion, and divided by that factor again before decoding.

The U-Net has base channels 64 and multipliers (1, 2, 4), with self-attention at 16×16 and at the bottleneck. Training is 50 epochs, Adam, learning rate 2e-5, batch size 128, noise-prediction MSE. Labels are dropped to a null class with probability 0.1. Sampling uses all 1000 reverse steps and guidance scale w = 3.

HP_ID is reported only. I did not train a second hyperparameter network. The two models are the pretrained VAE and this U-Net.

The run was on my Mac with MPS, PyTorch 2.14.0. Test reconstruction MSE was 0.000781. Training loss went from 0.384546 at epoch 1 to 0.096441 at epoch 50.

The notebook is split into the same kind of sections as my earlier homework:

1. VAE encode, scale, and reconstruction;
2. forward diffusion on the latent, decoded back to an image;
3. the conditional U-Net;
4. noise-prediction training;
5. classifier-free guidance sampling, including the 64-image grid.

The code cells include comments explaining what each block is doing. The report uses the numbers and figures from the executed notebook.

## Files

| File | Role |
|------|------|
| `latent_diffusion.ipynb` | Executed notebook |
| `HW9_Report.pdf` | Findings and discussion |
| `figures/reconstructions.png` | Input versus VAE decode, one test image per class |
| `figures/forward_diffusion.png` | Decoded latent at t = 0, 200, 500, 800, 1000 |
| `figures/train_loss.png` | Training loss curve |
| `figures/cfg_samples.png` | 64-image grid, w = 3, classes 0–9 repeated |
| `figures/guidance_compare.png` | Unconditional, conditional, and w = 3 on one noise draw |
| `outputs/cfg_grid_w3.png` | Saved sample grid |
| `checkpoints/latent_unet_weights.pt` | Saved U-Net weights (no Adam state; under GitHub 100 MB limit) |
| `METRICS.md` / `RUN_LOG.txt` / `AI_USE.md` | Standing notes |
