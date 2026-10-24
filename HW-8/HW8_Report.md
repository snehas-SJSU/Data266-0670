# DATA 266 — Homework 8

Sneha Singh  
SID4 = 0670 | SEED = 670 | SLICE = 670 | HP_ID = 4 | CLS_A = 0 | CLS_B = 5

Notebook: `mini_diffusion.ipynb`. Dataset: MNIST.

Same SID/SEED as the earlier homework. I did not train a second HP_ID network. The trained model is one timestep-conditioned CNN; DDPM and DDIM are two sampling procedures applied to that same model.

## 1. Forward diffusion

I used `T = 100` diffusion steps and a linear β schedule from `0.0001` to `0.02`. The closed-form forward process was:

\[
x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon,
\qquad \epsilon \sim \mathcal{N}(0,I).
\]

I showed one MNIST image at several timesteps. At the first few steps the digit is still clear. Later it is mostly noise. That is the fixed forward process; there is no learned model in this part.

## 2. CNN and training

I used a small CNN with five convolutional layers, ReLU activations, timestep conditioning, and one output channel. The output has the same shape as the input image. The timestep embedding is added to the convolution features so the network knows how much noise is present.

For every batch I sampled a random timestep, made `x_t`, and trained the network to predict the exact Gaussian noise used to make it. The loss is mean squared error between the true noise and the predicted noise.

Batch size was `128`, the learning rate was `1e-3`, and the planned run was 50 epochs. I will add the first and final losses from the executed notebook here.

## 3. DDPM sampling

I started from Gaussian noise and used the learned reverse process for all 100 steps. At each step the model predicts the noise, I compute the reverse mean, and I add posterior noise except at the final step.

DDPM is stochastic. Two runs can produce different images because fresh random noise is added during the reverse chain.

## 4. DDIM sampling

I kept the same trained CNN and changed only the sampling procedure. DDIM uses a shorter timestep schedule and jumps over some of the 100 training steps. I used 50 steps and 20 steps with `eta = 0`, so a fixed starting noise tensor gives a deterministic result.

## 5. Comparison

I will compare DDPM with 100 steps, DDIM with 50 steps, and DDIM with 20 steps. I will look at digit recognizability, blur or artifacts, and sampling time. I am not claiming that fewer steps are better unless the generated images and timings show it.

## Conclusion

Pending the full 50-epoch run.
