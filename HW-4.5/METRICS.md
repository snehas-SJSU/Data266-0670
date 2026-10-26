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

## Serving

| Setup | Tokens/s | p50 (ms) | p95 (ms) |
|-------|---------:|---------:|---------:|
| HF pipeline, batch 1 | 115.7 | 269.3 | 280.8 |
| vLLM batch 1 (HW4.a) | 97.8 | 319.1 | 328.4 |
| vLLM batch 8 (HW4.a) | 627.8 | 349.2 | 354.9 |
| vLLM batch 32 (HW4.a) | 2273.5 | 383.7 | 386.5 |
| vLLM batch 1 (HW4.5) | 97.9 | 320.4 | 325.3 |
| vLLM batch 8 (HW4.5) | 622.9 | 348.8 | 355.4 |
| vLLM batch 32 (HW4.5) | 2354.5 | 368.4 | 378.6 |

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

Apply first: 8-bit optimiser.  
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

No OOM in this ladder on the L4.
