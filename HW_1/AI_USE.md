# AI-Use Appendix — Sneha Singh

SID4 = 0670 | SEED = 670 | SLICE = 670 | HP_ID = 4 | CLS_A = 0 | CLS_B = 5

1. Which parts did you use an assistant for, and which did you write yourself?

I did this homework. Parameters, AR from lecture, diabetes csv, plots, 70/15/15 split, both neural nets (PyTorch and TensorFlow), the 3-seed table, and the conclusion are mine. I ran `neural_networks.ipynb` locally and put those numbers in METRICS.md.

CUDA is C, which I have not used much, so I used an assistant for the `matmul.cu` syntax (pointers, `cudaMalloc`, the kernel). I still set the design: 16x16 threads, one thread writes one `C[row, col]` from `blockIdx` / `threadIdx`. I ran it on Colab GPU, copied the times into the table, and wrote the crossover answer myself.

I only got stuck on two other things:
- TensorFlow on my Mac (kernel died on Anaconda 3.13).
- Colab profiler (`nsys` missing, so `nvprof`).

2. Give one specific thing it produced that was wrong. Paste the wrong output.

It told me to install TensorFlow in my Anaconda 3.13 setup. After that, `import tensorflow` crashed the kernel. No training happened. This is what I saw:

```
The kernel for Desktop/Gen-Ai/HW_1/neural_networks.ipynb appears to have died. It will restart automatically.
```

3. How did you find out? What did the failure look like?

I ran the first cell (imports and seeds). PyTorch was fine. When TensorFlow was in that cell the kernel died in under a minute and I lost all the variables, so I had to restart.

4. What did you change, and why does your version work?

I changed the kernel from Anaconda 3.13 to Python (HW1) which is Python 3.12. TensorFlow imports there, so I could train the same [64, 32] network in TF and PyTorch. On Colab I compiled `matmul.cu` with nvcc and ran `./matmul` myself. `nsys` said command not found, so I ran `nvprof ./matmul`. That output shows `gpu_matmul` separate from the HtoD and DtoH copies.
