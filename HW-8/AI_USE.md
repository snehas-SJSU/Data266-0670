# AI-Use Appendix — Sneha Singh

SID4 = 0670 | SEED = 670 | SLICE = 670 | HP_ID = 4 | CLS_A = 0 | CLS_B = 5

1. Which parts did you use an assistant for, and which did you write yourself?

I set `T = 100`, `BETA_START = 0.0001`, `BETA_END = 0.02`, batch size 128, and 50 epochs. I picked MNIST, organized the notebook sections, ran `mini_diffusion.ipynb` on my Mac (MPS), checked the loss curve and generated-image grids, and wrote the comparison in the report myself.

- Step 0: SID4 670, seeds, HP_ID reported only
- Forward process: beta schedule, alpha-bar calculation, `q_sample`, and the noisy-image figure
- Model: five convolutional layers, ReLU activations, timestep conditioning, same input/output shape
- Training: random timestep per batch, noise-prediction MSE, 50 epochs
- Sampling: DDPM with 100 steps and DDIM with 50 and 20 steps
- Compare: the loss numbers and the three generated-image grids

I used an assistant for a limited part of the work, about 20%: checking the posterior variance formula, reviewing the DDPM/DDIM sampling idea, and suggesting a simple timestep-conditioning pattern. I wrote the forward process, CNN details, training loop, sampling code, figures, metrics, and final discussion. I also ran the notebook, checked that the model output was `torch.Size([2, 1, 28, 28])`, and used the printed loss values and figures in the report.

2. Give one specific thing it produced that was wrong. Paste the wrong output.

The first draft of the notebook/report planned to compare sampling time, but the code did not measure it. The report placeholder said:

```
The final discussion will be based on the executed images and timing results.
```

There were no `time.perf_counter()` calls or printed DDPM/DDIM timings in the notebook.

3. How did you find out? What did the failure look like?

I read the final sampling cell and the metrics table after the notebook ran. The cell saved the DDPM and DDIM images, but it never started or stopped a timer. The metrics rows for sampling time therefore had no numbers. The generated images were also not clean digits, so I could not honestly claim that the shorter DDIM runs were better.

4. What did you change, and why does your version work?

I removed the unsupported sampling-time claim and updated the report, `METRICS.md`, and `RUN_LOG.txt` with values that actually came from the run: 97,537 model parameters, output shape `(2, 1, 28, 28)`, loss `0.164547` at epoch 1, and loss `0.061000` at epoch 50. The comparison now says that DDPM, DDIM-50, and DDIM-20 were all still noisy, and does not call DDIM better just because it used fewer steps. The notebook ran through all six code cells and saved all three output grids.
