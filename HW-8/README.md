# DATA 266 — Homework 8

Sneha Singh  
SID4 = 0670 | SEED = 670 | SLICE = 670 | HP_ID = 4 | CLS_A = 0 | CLS_B = 5

Findings: [`HW8_Report.md`](HW8_Report.md)

## Mini Diffusion Model (DDPM → DDIM)

Notebook: `mini_diffusion.ipynb`. Dataset: MNIST.

I use `T = 100`, `BETA_START = 0.0001`, and `BETA_END = 0.02` with a linear β schedule. Batch size is `128` and the run is 50 epochs. The CNN has five convolutional layers, ReLU activations, timestep conditioning, and the same input/output image shape.

The run was done on my Mac with MPS. The report uses the numbers and figures from the executed notebook. The generated samples are still noisy on this run, so the comparison does not claim that DDIM is better just because it uses fewer reverse steps.

The notebook is split into the same kind of sections as my earlier homework:

1. forward diffusion and the noisy-image figure;
2. model setup and noise-prediction training;
3. DDPM sampling with all 100 reverse steps;
4. DDIM sampling with 50 and 20 steps;
5. comparison and discussion.

The code cells include comments explaining what each block is doing. The report will use only results from the executed notebook.

## Files

| File | Role |
|------|------|
| `mini_diffusion.ipynb` | Executed notebook with forward diffusion, training, DDPM, and DDIM |
| `HW8_Report.md` | Findings and discussion |
| `figures/forward_diffusion.png` | Forward-noise visualization |
| `figures/train_loss.png` | Training loss curve |
| `figures/ddpm_vs_ddim.png` | DDPM versus DDIM comparison |
| `outputs/` | Saved DDPM and DDIM image grids |
| `checkpoints/mini_diffusion.pt` | Saved trained CNN checkpoint |
| `METRICS.md` / `RUN_LOG.txt` / `AI_USE.md` | Standing notes |
