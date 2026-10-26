# METRICS — DATA 266 HW4.5

Sneha Singh  
SID4 = 0670 | SEED = 670 | SLICE = 670 | HP_ID = 4 | CLS_A = 0 | CLS_B = 5

Part C skipped (optional). One GPU: NVIDIA L4 (Colab Pro).

This block is the HW4.5 summary. It is not an HW5 baseline.

Chosen configuration: Best combination (AMP + 8-bit AdamW). Peak allocated 1.64 GB, 222.0 samples/s, wall-clock 47.3 s, test accuracy 0.7160.  
One baseline fine-tune: 113.1 s / 3600 = 0.0314 GPU-hours.

Rows 670–5669 inclusive. Split 3500/750/750.

## Setup

| Item | Value |
|------|-------|
| Dataset | `cardiffnlp/tweet_eval` sentiment, train split |
| Rows | 670–5669 (5,000) |
| Split | 3500 / 750 / 750, SEED 670 |
| Model | bert-base-uncased |
| Tokenizer | BertTokenizer |
| Max length | 128 |
| Baseline batch | 16 |
| Epochs | 3 |
| Optimizer | AdamW |
| Learning rate | 2e-5 |
| Weight decay | 0.01 |
| Device | NVIDIA L4 |
| Serving model | facebook/opt-125m |
| Generated tokens | 32 |
| Accumulation | physical batch 8, 16 steps, effective 128 |

## HW4.a baseline

| Epoch | Train loss | Val loss | Val accuracy | Seconds |
|------:|-----------:|---------:|-------------:|--------:|
| 1 | 0.8892 | 0.7410 | 0.6640 | 37.0 |
| 2 | 0.5950 | 0.7395 | 0.6773 | 37.8 |
| 3 | 0.3922 | 0.7976 | 0.6813 | 38.3 |

Best validation-loss epoch: 2.  
Test accuracy at that epoch: 0.7267.  
Peak allocated memory: 2.48 GB.  
Train wall-clock: 113.1 s.

| Class | Precision | Recall | Support |
|-------|----------:|-------:|--------:|
| negative | 0.6667 | 0.6723 | 119 |
| neutral | 0.7082 | 0.7169 | 325 |
| positive | 0.7708 | 0.7582 | 306 |

Worst class: negative.

## Table HW4.1 — Serving comparison (L4)

| Configuration | Tokens/sec | p50 latency (ms) | p95 latency (ms) |
|---------------|----------:|-----------------:|-----------------:|
| HF pipeline, batch 1 | 115.7 | 269.3 | 280.8 |
| vLLM, batch 1 | 97.8 | 319.1 | 328.4 |
| vLLM, batch 8 | 627.8 | 349.2 | 354.9 |
| vLLM, batch 32 | 2273.5 | 383.7 | 386.5 |

Part D re-run (same L4): vLLM batch 1 / 8 / 32 → 97.9 / 622.9 / 2354.5 tok/s; p50 320.4 / 348.8 / 368.4 ms. Nearly identical to HW4.a (same hardware and precision).

Latency vs throughput: at batch 8, p50 is worse than HF batch 1 while tokens/s is still rising. Interactive apps care about p50.

## Part A

| Batch | Samples/s | Peak GB | Seconds / epoch |
|------:|----------:|--------:|----------------:|
| 8 | 70.5 | 2.09 | 49.7 |
| 16 | 92.5 | 2.49 | 37.8 |
| 32 | 106.5 | 3.76 | 32.9 |
| 64 | 110.6 | 6.15 | 31.7 |
| 128 | 119.6 | 11.00 | 29.3 |
| 256 | 121.0 | 20.65 | 28.9 |
| 512 | OOM | — | — |

Largest batch: 256. Fastest batch: 256 (121.0 samp/s). Throughput within 5% of peak from batch 128.

## Table HW4.5.1

| Technique | Max batch | Throughput (samp/s) | Peak memory (GB) | Eval accuracy |
|-----------|----------:|--------------------:|-----------------:|--------------:|
| Baseline (HW4 settings) | 256 | 92.8 | 2.48 | 0.7267 |
| Mixed precision | 256 | 189.3 | 2.26 | 0.7040 |
| Gradient checkpointing | 1024 | 70.7 | 2.09 | 0.7267 |
| Gradient accumulation | 128 (physical 8) | 99.5 | 2.32 | 0.6520 |
| 8-bit optimiser | 256 | 99.3 | 1.90 | 0.7187 |
| Best combination | 256 | 222.0 | 1.64 | 0.7160 |

Checkpointing throughput drop at batch 16: 92.8 → 70.7 samp/s (−23.8%).  
Accumulation: effective batch 128; same optimiser-step wall-clock 816.9 s (baseline train 113.1 s).

Ranking by memory saved per unit throughput lost (batch-16 peaks):

| Rank | Technique | Saved GB | Δ samp/s | Saved GB / samp/s lost |
|-----:|-----------|---------:|---------:|-----------------------:|
| 1 | 8-bit optimiser | 0.58 | +6.5 | free — apply first |
| 2 | Mixed precision | 0.22 | +96.5 | free |
| 3 | Gradient accumulation | 0.16 | +6.7 | free |
| 4 | Gradient checkpointing | 0.39 | −22.1 | 0.018 |

Not worth on a 32 GB card for this BERT run: gradient checkpointing.  
Combination flags: AMP + 8-bit AdamW.

## Part C

Part C skipped (optional).

## Part D — KV tensors at batch 8

| Context | Peak GB |
|--------:|--------:|
| 128 | 0.05 |
| 256 | 0.09 |
| 512 | 0.16 |
| 1024 | 0.30 |
| 2048 | 0.58 |
| 4096 | 1.14 |
| 8192 | 2.27 |
| 16384 | 4.52 |
| 32768 | 9.02 |
| 65536 | 18.02 |

Longest context at batch 8 that fit: 65536 (no allocation OOM on this L4 ladder).  
GPU captures: `results/GPU_CAPTURE.md` (training + serving on the same NVIDIA L4).
