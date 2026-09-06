# DATA 266 — Homework 2

Sneha Singh  
SID4 = 0670 | SEED = 670 | SLICE = 670 | HP_ID = 4 | CLS_A = 0 | CLS_B = 5

HP_ID is reported only. HW2 has no HP_ID mapping.

## Files

| File | Role |
|------|------|
| `embeddings.ipynb` | Part 1 — Word2Vec + IMDB |
| `rag.ipynb` | Part 2 — Wikipedia RAG |
| `optimizations.ipynb` | Part 3 — training tricks (MPS; AMP pointer) |
| `amp_cuda_colab.ipynb` | Part 3 — CUDA fp16 mixed precision (Colab T4) |
| `METRICS.md` | Required tables |
| `RUN_LOG.txt` | Console numbers from the runs |
| `AI_USE.md` | AI-use appendix |

## What I ran

- Kernel: **Python (HW1)**
- Part 1: `word2vec-google-news-300`, then Word2Vec on 5,000 IMDB reviews, seed 670
- Part 2: 10 Wikipedia movies, chunk 500/50, then 200/20 on Q1 and Q2
- Part 3: same 12×512 MLP, batch 64, 40 steps, on **mps**; mixed precision on Colab **Tesla T4** (`amp_cuda_colab.ipynb`)
