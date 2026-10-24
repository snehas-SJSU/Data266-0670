# DATA 266 — Homework 8

Sneha Singh  
SID4 = 0670 | SEED = 670 | SLICE = 670 | HP_ID = 4 | CLS_A = 0 | CLS_B = 5

Notebook: `mini_diffusion.ipynb`. Dataset: MNIST.

## 1. Forward diffusion

I used `T = 100` diffusion steps and a linear β schedule from `0.0001` to `0.02`. The closed-form forward process was:

\[
x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon,
\qquad \epsilon \sim \mathcal{N}(0,I).
\]

The visualization shows one MNIST image at several timesteps. The image becomes less recognizable as the cumulative noise increases.

## 2. Mini DDPM model and training

The noise-prediction network is a small CNN with at least four convolutional layers, ReLU activations, timestep conditioning, and an output with the same shape as the input image. For each training example I sampled a random timestep, created `x_t`, and trained the network to predict the noise `ε` using mean squared error.

Training settings and measured loss will be filled in after execution.

## 3. DDPM sampling

Starting from Gaussian noise at timestep `T - 1`, I applied the learned reverse process for all 100 steps. The DDPM sampler adds the posterior variance noise at intermediate steps, so repeated samples can differ even with the same trained model.

## 4. DDIM sampling

I used the same trained noise-prediction model but changed the sampling schedule. DDIM jumps across a subset of timesteps and uses `eta = 0`, making the sampling deterministic for a fixed starting noise tensor. I generated samples with 50 steps and 20 steps.

## 5. Comparison

The generated-image comparison and exact observations will be added after the notebook is run. I will compare image quality, sampling time, digit recognizability, and the effect of reducing the number of denoising steps.

## Conclusion

Pending notebook execution.
