# METRICS — DATA 266 HW1

SID4 = 0670 | SEED = 670 | SLICE = 670 | HP_ID = 4 | CLS_A = 0 | CLS_B = 5  
HP_ID 4: hidden `[64, 32]`, lr `0.001`, 15 epochs.

Split is fixed with `random_state = 670` (70 / 15 / 15).  
Training seeds: 670, 671, 672. Metric: test accuracy. Std is `np.std` from `neural_networks.ipynb` (3-seed cell).

## Neural networks — test accuracy (mean ± std over 3 training seeds)

| Framework | Model | Config | Mean test acc | Std |
|-----------|--------|--------|---------------|-----|
| PyTorch | Baseline | `[64, 32]`, lr 0.001, 30 epochs | 0.6696 | 0.0252 |
| PyTorch | Modified (HP_ID 4) | `[64, 32]`, lr 0.001, 15 epochs | 0.5994 | 0.0180 |
| TensorFlow | Baseline | `[64, 32]`, lr 0.001, 30 epochs | 0.6784 | 0.0149 |
| TensorFlow | Modified (HP_ID 4) | `[64, 32]`, lr 0.001, 15 epochs | 0.6374 | 0.0083 |

Per-seed test accuracies (from the 3-seed cell):

| Framework | Model | seed 670 | seed 671 | seed 672 |
|-----------|--------|----------|----------|----------|
| PyTorch | Baseline | 0.7018 | 0.6404 | 0.6667 |
| PyTorch | Modified | 0.6228 | 0.5965 | 0.5789 |
| TensorFlow | Baseline | 0.6579 | 0.6930 | 0.6842 |
| TensorFlow | Modified | 0.6316 | 0.6491 | 0.6316 |

Seed-670 loss-curve run (separate cells, used for the plots): PyTorch 0.7018 / 0.6228, TensorFlow 0.7193 / 0.6491. Modified is underfitting / stopped early; val loss still falling at epoch 15.

## CUDA matrix multiplication

GPU: Tesla T4 (Colab). Kernel uses 16×16 thread blocks; each thread writes one `C[row,col]`.  
Timing: warmup first, then one timed run per N (256 / 1024 / 4096). Not repeated or averaged. N=4096 CPU is ~13 min.  
Profiler used: **nvprof** (`nsys` was not installed). nvprof GPU activities: `gpu_matmul` 88.8%, `[CUDA memcpy HtoD]` 6.6%, `[CUDA memcpy DtoH]` 4.6%.

Table from `./matmul` (cudaEvent, not under nvprof):

| Matrix size | CPU (ms) | GPU kernel (ms) | H2D+D2H (ms) | Speedup |
|-------------|----------|-----------------|--------------|---------|
| 256 | 25.308 | 0.146 | 0.437 | 43.38 |
| 1024 | 3291.187 | 9.187 | 4.756 | 236.05 |
| 4096 | 763322.870 | 362.831 | 72.603 | 1753.02 |

Speedup = CPU time / (GPU kernel + H2D+D2H). N=256 max |CPU-GPU| = 1.79e-07.

Crossover: GPU end-to-end already wins at **N=256**. Not at size 0 because H2D+D2H still happens; at 256 copies (0.44 ms) are larger than the kernel (0.15 ms).
