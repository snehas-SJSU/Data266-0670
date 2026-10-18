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

The first draft of my metrics table said the VAE interpolates smoothly and the FC-AE does not:

```
| Latent Interpolation Quality | Discrete jumps / Voids | **Smooth semantic transitions** | VAE manifold is dense and continuous |
```

3. How did you find out? What did the failure look like?

I opened `latent_interpolation.png` and compared it with that row. The VAE row stays shirt-shaped and fades until about α=0.8, then is a sandal by α=0.9. The FC-AE row fades slowly through a smudge. Neither row has a clean in-between item. The same draft also said the VAE t-SNE clusters were better, and the two plots look about the same.

4. What did you change, and why does your version work?

I rewrote both claims in the notebook, the report, and `METRICS.md` so they say only what the figures show. I also added row labels to the figures, since they had none. My version works because each sentence can be checked against a figure or a printed number: the notebook ran top to bottom, and its test numbers (FC-AE BCE 213.6070, VAE 226.8435, KL 13.0439) matched a separate run of `train_eval.py`.
