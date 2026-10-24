# DATA 266 — Homework 8

Sneha Singh  
SID4 = 0670 | SEED = 670 | SLICE = 670 | HP_ID = 4 | CLS_A = 0 | CLS_B = 5

## Mini Diffusion Model (DDPM → DDIM)

Notebook: `mini_diffusion.ipynb`. Dataset: MNIST.

I use `T = 100`, a linear β schedule from `0.0001` to `0.02`, batch size `128`, and 50 planned training epochs. The CNN has five convolutional layers, ReLU activations, timestep conditioning, and the same input/output image shape.

The notebook is split into the same kind of sections as my earlier homework:

1. forward diffusion and the noisy-image figure;
2. model setup and noise-prediction training;
3. DDPM sampling with all 100 reverse steps;
4. DDIM sampling with 50 and 20 steps;
5. comparison and discussion.

The code cells include comments explaining what each block is doing. The report will use only results from the executed notebook.
