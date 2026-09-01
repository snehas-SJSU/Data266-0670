# DATA 266 — Homework 1

Sneha Singh  
SID4 = 0670 | SEED = 670 | SLICE = 670 | HP_ID = 4 | CLS_A = 0 | CLS_B = 5

HP_ID 4: same network `[64, 32]` and lr `0.001`, **15** epochs instead of 30.

Written report: [`HW1_Report.pdf`](HW1_Report.pdf)

## Files

| File | Role |
|------|------|
| `HW1_Report.pdf` | Written report |
| `neural_networks.ipynb` | Autoregressive models + diabetes NN (PyTorch and TensorFlow) |
| `cuda.ipynb` | CUDA build/run, timing table, nvprof output |
| `matmul.cu` | CUDA matrix multiplication |
| `diabetes.csv` | Dataset (no header; 8 features + Outcome) |
| `METRICS.md` | Mean/std and CUDA tables |
| `RUN_LOG.txt` | Console numbers from the runs |
| `AI_USE.md` | AI-use appendix |

## Models

- Baseline: hidden `[64, 32]`, lr `0.001`, **30** epochs
- Modified (HP_ID 4): hidden `[64, 32]`, lr `0.001`, **15** epochs
- Split: 70/15/15, `random_state = 670`, same split for every model
- Training seeds: 670, 671, 672
