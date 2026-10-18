# DATA 266 — Homework 7

Sneha Singh  
SID4 = 0670 | SEED = 670 | SLICE = 670 | HP_ID = 4 | CLS_A = 0 | CLS_B = 5

Findings: [`HW7_Report.pdf`](HW7_Report.pdf)

HP_ID is reported only. No second hyperparameter network. The two models are a fully connected autoencoder and a VAE.

Test reconstruction BCE: FC-AE 213.6070, VAE 226.8435. VAE test KL 13.0439. Run on my Mac (MPS), PyTorch 2.14.0. Interpolation goes from class 0 (T-shirt/top) to class 5 (Sandal).

## Files

| File | Role |
|------|------|
| `autoencoders_fashionmnist.ipynb` | GAN questions, then FC-AE and VAE, executed |
| `HW7_Report.pdf` | Findings |
| `train_eval.py` | Same training and figures as the notebook, as a script. Run it from `HW-7/`. |
| `checkpoints/` | Saved weights for both models |
| `figures/loss_curves.png` | Training loss vs epoch |
| `figures/reconstructions_comparison.png` | One test image per class, original / FC-AE / VAE |
| `figures/random_generation.png` | Decoded z ~ N(0, I) |
| `figures/latent_interpolation.png` | T-shirt to sandal in latent space |
| `figures/latent_tsne.png` | t-SNE of the 20-d latent codes |
| `METRICS.md` / `RUN_LOG.txt` / `AI_USE.md` | Standing notes |
