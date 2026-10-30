# AI-Use Appendix — Sneha Singh

SID4 = 0670 | SEED = 670 | SLICE = 670 | HP_ID = 4 | CLS_A = 0 | CLS_B = 5

1. Which parts did you use an assistant for, and which did you write yourself?

I set Step 0 the same way as HW1 and HW8. I kept FashionMNIST, the section order, and the rule that HP_ID is reported only. The two models are the pretrained VAE and one U-Net. I ran `latent_diffusion.ipynb` on my Mac (MPS) and used the printed numbers and the saved figures in the report.

- Step 0: SID4 670, seeds, HP_ID reported only
- VAE: `stabilityai/sd-vae-ft-mse`, resize to 128×128, scale 0.18215, test MSE
- Forward process: T = 1000, linear β from 1e-4 to 0.02, decoded noisy latents
- U-Net: base 64, multipliers (1, 2, 4), GroupNorm, SiLU, self-attention at 16×16 and the bottleneck, 10 classes plus a null label
- Training: 50 epochs, Adam, lr 2e-5, label drop 0.1, noise-prediction MSE
- Sampling: 1000 reverse steps, w = 3, 64-image grid with classes 0–9 repeated

I used an assistant to draft the U-Net blocks, the latent scaling, and the classifier-free guidance sampling, in the same notebook layout as HW8.

2. Give one specific thing it produced that was wrong. Paste the wrong output.

The first reconstruction check decoded the scaled latent directly. On random tensors in [-1, 1] both versions looked the same, about 0.35, so the mistake was easy to miss. On a real FashionMNIST batch the check printed:

```
mse unscale 0.0007698756526224315
mse scaled-direct 0.3539625406265259
```

3. How did you find out? What did the failure look like?

I ran the same encode on a real batch of resized FashionMNIST images, then decoded twice: once after dividing by 0.18215, and once on the scaled latent itself. The direct decode was not a garment. Its MSE stayed near 0.35, which is about the error of guessing the pixel range, while the unscaled decode was about 0.00077.

4. What did you change, and why does your version work?

The notebook divides by 0.18215 before `vae.decode`. The test-set number in the report is from that version: MSE 0.000781 on all 10,000 test images. The reconstruction figure matches that path. The loss curve and the sample grids are from the same executed notebook.
