# METRICS — DATA 266 HW2.5

Sneha Singh  
SID4 = 0670 | SEED = 670 | SLICE = 670 | HP_ID = 4 | CLS_A = 0 | CLS_B = 5  
HP_ID is reported only. No second hyperparameter model.

Every cell is traceable to a UUID-labelled run in `RUN_LOG.txt` and `results/<gpu>/` JSON.


## Table HW2.5.1 — summary

| Measurement | NVIDIA GeForce RTX 4090 | Notes |
|-------------|------|-------|
| Peak achieved TFLOPS (BF16) | 162.00 (N=8192, UUID GPU-5b052ad1-4272-40db-4b25-c930bf32b547) | max BF16 GEMM over N=1024..16384; dense peak from vendor whitepaper |
| % of theoretical peak (BF16) | 98.1% (peak 165.2 TFLOPS, UUID GPU-5b052ad1-4272-40db-4b25-c930bf32b547) | BF16 tensor, FP32 accumulate, dense (not sparse) |
| Effective bandwidth (GB/s) | 915.1 GB/s (90.8% of 1008.0, UUID GPU-5b052ad1-4272-40db-4b25-c930bf32b547) | elementwise add, 3-operand traffic |
| Naive attention OOM length | ok≤26752 fail≥26816 (gap 64, UUID GPU-5b052ad1-4272-40db-4b25-c930bf32b547) | largest tested success and smallest tested fail; not 1-token unless gap=1 |
| Fused attention OOM length | ok≤131072 fail≥None (gap None, UUID GPU-5b052ad1-4272-40db-4b25-c930bf32b547) | same search on F.scaled_dot_product_attention |
| Steady-state / peak throughput | 98.0% (UUID GPU-5b052ad1-4272-40db-4b25-c930bf32b547) | last 5 min mean TFLOPS / first 30 s mean TFLOPS |
| Throttle onset (s, or none) | 10s  temp=62.0 C  power=449.97 W  UUID GPU-5b052ad1-4272-40db-4b25-c930bf32b547 | first non-idle throttle reason after 10 s |

Table HW2.5.1 — The summary table. Every cell is UUID-labelled.

## Part A — provenance

### NVIDIA GeForce RTX 4090

- UUID: `GPU-5b052ad1-4272-40db-4b25-c930bf32b547`
- driver: 595.95
- CUDA (nvidia-smi): 13.2
- VRAM: 24564 MiB
- power limit: 450.00 W
- architecture: Ada Lovelace (AD102)
- memory: 24 GB GDDR6X, 384-bit, 21 Gbps
- bandwidth: 1008.0 GB/s
- tensor cores: 4th generation
- reduced precisions: TF32, FP16, BF16, FP8, INT8, INT4
- sources: ['NVIDIA Ada GPU Architecture whitepaper, Table of GeForce RTX 4090 specs', 'https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf', 'https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4090/']
- nvidia-smi -q file: `results\NVIDIAGeForceRTX4090_5b052ad1\nvidia_smi_q.txt`

## Part B — GEMM

### NVIDIA GeForce RTX 4090  UUID `GPU-5b052ad1-4272-40db-4b25-c930bf32b547`

| N | precision | TFLOPS | % of dense peak | reps | mean ms | std ms |
|---|-----------|--------|-----------------|------|---------|--------|
| 1024 | fp32 | 34.47 | 41.7 | 200 | 0.062 | 0.007 |
| 4096 | fp32 | 53.74 | 65.1 | 200 | 2.558 | 0.067 |
| 8192 | fp32 | 54.57 | 66.1 | 75 | 20.150 | 0.151 |
| 16384 | fp32 | 50.63 | 61.3 | 20 | 173.731 | 1.339 |
| 1024 | tf32 | 26.67 | 32.3 | 200 | 0.081 | 0.014 |
| 4096 | tf32 | 86.13 | 104.3 | 200 | 1.596 | 0.046 |
| 8192 | tf32 | 87.62 | 106.1 | 120 | 12.548 | 0.128 |
| 16384 | tf32 | 87.02 | 105.3 | 20 | 101.086 | 0.615 |
| 1024 | fp16 | 62.31 | 37.7 | 200 | 0.034 | 0.008 |
| 4096 | fp16 | 164.80 | 99.8 | 200 | 0.834 | 0.028 |
| 8192 | fp16 | 158.40 | 95.9 | 200 | 6.941 | 0.178 |
| 16384 | fp16 | 157.80 | 95.5 | 27 | 55.741 | 0.508 |
| 1024 | bf16 | 72.71 | 44.0 | 200 | 0.030 | 0.005 |
| 4096 | bf16 | 152.14 | 92.1 | 200 | 0.903 | 0.030 |
| 8192 | bf16 | 162.00 | 98.1 | 200 | 6.787 | 0.154 |
| 16384 | bf16 | 160.21 | 97.0 | 28 | 54.905 | 0.490 |

BF16 reaches 95% of its own max (162.0 TFLOPS) at N=8192. FP16 reaches 95% of its own max (164.8 TFLOPS) at N=4096. FP32 reaches 95% of its own max (54.6 TFLOPS) at N=4096. TF32 reaches 95% of its own max (87.6 TFLOPS) at N=4096. Small N never plateaus: kernel launch and cuBLAS setup dominate, the tile count is too small to fill the SMs, and the working set sits in cache so neither DRAM bandwidth nor peak tensor-core throughput is visible.

Lower precision attempts:
- fp8 OK via `torch._scaled_mm(a8, b8.t().contiguous().t(), ...)` → 263.65 TFLOPS (79.82055149102824% of peak)
- fp4 FAILED. tried: ['float4_e2m1fn_x2']. failure: Found symbols ['float4_e2m1fn_x2'] but no public dense GEMM path is wired in this script. Treated as tooling-maturity finding unless a later cell succeeds.

## Part C — roofline

### NVIDIA GeForce RTX 4090  UUID `GPU-5b052ad1-4272-40db-4b25-c930bf32b547`

- add: 915.1 GB/s (90.8% of 1008.0 GB/s), AI=0.0833 FLOP/B, memory-bound (left of ridge)
- copy check: 908.5 GB/s (90.1%)
- BF16 GEMM N=16384: 160.76 TFLOPS (97.3% of peak), AI=5461.3 FLOP/B, compute-bound (right of ridge)
- ridge: FP32 81.9 FLOP/B, BF16 163.9 FLOP/B

## Part D — attention

### NVIDIA GeForce RTX 4090  UUID `GPU-5b052ad1-4272-40db-4b25-c930bf32b547`

batch=1, n_heads=32, head_dim=64, dtype=bfloat16

| seq | naive ms | naive peak GB | fused ms | fused peak GB | speedup |
|-----|----------|---------------|----------|---------------|---------|
| 512 | 0.10 | 0.050 | 0.13 | 0.025 | 0.82x |
| 1024 | 0.45 | 0.160 | 0.31 | 0.034 | 1.48x |
| 2048 | 1.89 | 0.579 | 1.08 | 0.050 | 1.75x |
| 4096 | 7.23 | 2.223 | 2.61 | 0.084 | 2.77x |
| 8192 | 28.67 | 8.733 | 4.40 | 0.151 | 6.51x |
| 16384 | 918.11 | 34.637 | 17.46 | 0.285 | 52.59x |

Naive OOM bounds: largest success 26752, smallest fail 26816, gap 64 (single-token resolution: False).
Fused OOM bounds: largest success 131072, smallest fail None, gap None.
Quadratic fit on naive peak memory: a=1.280000e+02 bytes/token², b=1.638400e+04, c=8.519680e+06. RMSE quadratic 7.256e-08 vs linear 3.012e+09. Quadratic better: True. a>0: True.

The fused / FlashAttention-style kernel never writes the S×S score matrix to HBM. It loads Q/K/V tiles into SRAM, computes local products, keeps a running softmax (m, l statistics) in on-chip memory, and writes only the output tile. That removes the O(S²) allocation that OOMs the naive path and the O(S²) memory traffic that makes naive attention bandwidth-bound.

## Part E — thermal

### NVIDIA GeForce RTX 4090  UUID `GPU-5b052ad1-4272-40db-4b25-c930bf32b547`

- duration: 1202.3s  N=8192 BF16
- first 30s TFLOPS: 154.96752464639664
- last 5 min TFLOPS: 151.93678907960563
- steady/peak: 98.04427697112249
- throttle onset (s): 10.0
- max temp C: 76.0
- log: `results\NVIDIAGeForceRTX4090_5b052ad1\thermal_smi.csv`

