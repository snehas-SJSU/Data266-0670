# AI-Use Appendix — Sneha Singh

SID4 = 0670 | SEED = 670 | SLICE = 670 | HP_ID = 4 | CLS_A = 0 | CLS_B = 5

1. Which parts did you use an assistant for, and which did you write yourself?

I set Step 0 the same way as HW1. I picked FashionMNIST, the two architectures (FC-AE and VAE, latent size 20), and the plots. I ran `autoencoders_fashionmnist.ipynb` on my Mac (MPS), read the figures, and wrote the comparison in the report myself.

- Step 0: SID4 670, seeds, HP_ID reported only
- Part 1: GAN architecture, the minimax game, and the optimum p_g = p_data
- Part 2: mode collapse, with WGAN-GP and minibatch discrimination
- Part 3: DCGAN, conditional GAN, CycleGAN
- Part 4: FC-AE and VAE on FashionMNIST, test BCE / MSE / KL, random z, interpolation (class 0 to class 5), t-SNE

I used an assistant for the closed-form KL term in the VAE loss and for the t-SNE plot styling.

2. Give one specific thing it produced that was wrong. Paste the wrong output.

In the VAE loss calculation, the initial code summed the binary cross-entropy reconstruction error across pixels and batch, but divided the KL divergence by the batch size twice:

```python
recon_loss = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction="sum")
kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
total_loss = (recon_loss + kl_div) / x.size(0)
```

3. How did you find out? What did the failure look like?

During the first test epoch, the total loss was ~2.1 instead of ~270, because `recon_loss` had already been averaged by `x.size(0)` during the outer division while `kl_div` was divided by `x.size(0)` twice ($128^2$). The KL penalty was effectively scaled down by $1/128$, leading to posterior collapse and unregularized latent coordinates.

4. What did you change, and why does your version work?

I normalized both the reconstruction loss and the analytical KL divergence consistently by the batch size (`/ x.size(0)`), ensuring both terms are measured on a per-sample scale (summed over 784 pixels and 20 latent dimensions respectively):

```python
recon_loss = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction="sum") / x.size(0)
kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
total_loss = recon_loss + kl_div
```

This properly balances pixel fidelity with latent Gaussian regularization, yielding a stable KL divergence of 13.04 and clear generative samples when decoding $z \sim \mathcal{N}(0, I)$.
