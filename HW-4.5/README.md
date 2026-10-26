# DATA 266 — Homework 4.5

Sneha Singh  
SID4 = 0670 | SEED = 670 | SLICE = 670 | HP_ID = 4 | CLS_A = 0 | CLS_B = 5

Findings: [`HW4.5_Report.pdf`](HW4.5_Report.pdf)

Part C skipped (optional). One GPU: NVIDIA L4 (Colab Pro).

## BERT sentiment, batch scaling, and serving

Notebook: `hw4.5.ipynb`. Data: `cardiffnlp/tweet_eval` sentiment, rows 670–5669, split 3500/750/750 with SEED 670.

I ran the notebook on Colab Pro with an L4. The report uses only numbers from that run. Baseline test accuracy was 0.7267. Part A OOMed at batch 512. The configuration I would keep is AMP + 8-bit AdamW (222.0 samp/s, 1.64 GB). This is the HW4.5 summary, not an HW5 baseline.

## Files

| File | Role |
|------|------|
| `hw4.5.ipynb` | Executed Colab notebook (L4) |
| `HW4.5_Report.pdf` | Findings and discussion |
| `data/twitter_slice.jsonl` | Frozen 5,000-row slice |
| `figures/baseline_loss.png` | Train / val loss |
| `figures/confusion.png` | Test confusion matrix |
| `figures/part_a_throughput.png` | Batch vs samples/s |
| `figures/kv_cache.png` | KV tensor size vs context |
| `checkpoints/bert_baseline/` | Best validation-loss BERT only |
| `METRICS.md` / `RUN_LOG.txt` / `AI_USE.md` | Standing notes |
| `reservations/GPU_HOURS.md` | Colab L4 session record |
