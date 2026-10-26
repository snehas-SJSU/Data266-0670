# DATA 266 — Homework 4.5

Sneha Singh  
SID4 = 0670 | SEED = 670 | SLICE = 670 | HP_ID = 4 | CLS_A = 0 | CLS_B = 5

Written report: [`HW4.5_Report.pdf`](HW4.5_Report.pdf)

Part C skipped (optional). One GPU: NVIDIA L4 (Colab Pro).

## Submit

| File | Role |
|------|------|
| `hw4.5.ipynb` | Executed notebook (BERT, serving, Parts A–E) |
| `HW4.5_Report.pdf` | Findings |
| `METRICS.md` | HW4.5 summary (not an HW5 baseline) |
| `RUN_LOG.txt` / `AI_USE.md` | Standing notes |
| `reservations/GPU_HOURS.md` | Session and GPU-hour record |
| `results/GPU_CAPTURE.md` | Training + serving GPU captures |
| `figures/` | Loss, confusion, Part A throughput, KV cache |
| tag `hw4-5` | Repository tag |

## Supporting

| File | Role |
|------|------|
| `data/twitter_slice.jsonl` | Frozen rows 670–5669 (5,000) |
| `checkpoints/bert_baseline/` | Best validation-loss tokenizer/config (weights local/Drive) |

Notebook data: `cardiffnlp/tweet_eval` sentiment, split 3500/750/750 with SEED 670. Baseline test accuracy 0.7267. Part A max batch 256 (512 OOM). Chosen config: AMP + 8-bit AdamW (222.0 samp/s, 1.64 GB).
