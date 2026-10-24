# DATA 266 — Homework 8

Sneha Singh  
SID4 = 0670 | SEED = 670 | SLICE = 670 | HP_ID = 4 | CLS_A = 0 | CLS_B = 5

## Mini Diffusion Model (DDPM → DDIM)

Notebook: `mini_diffusion.ipynb`. Dataset: MNIST.

Configuration:

- Diffusion steps: `T = 100`
- Noise schedule: linear β from `0.0001` to `0.02`
- Batch size: `128`
- Epochs: `50`
- Model: small timestep-conditioned CNN with five convolutional layers
- Training target: the Gaussian noise added to the image

The notebook contains the forward diffusion visualization, training loss, DDPM sampling, DDIM sampling with 50 and 20 steps, and a comparison of the generated images.
