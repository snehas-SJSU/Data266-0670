# DATA 266 — Homework 6

Sneha Singh  
SID4 = 0670 | SEED = 670 | SLICE = 670 | HP_ID = 4 | CLS_A = 0 | CLS_B = 5

Findings: [`report.pdf`](report.pdf)

HP_ID is reported only. No second hyperparameter network. The three models are supervised, rotation SSL, and SimCLR.

Test accuracy: supervised 0.3149, rotation 0.3957, SimCLR 0.4467. Run on Colab GPU.

## Files

| File | Role |
|------|------|
| `stl10_ssl.ipynb` | Parts A–D, executed |
| `report.pdf` | Findings |
| `checkpoints/` | Saved weights and the shared 500-index list |
| `figures/nearest_neighbors.png` | Same 3 queries, top-5 cosine neighbors |
| `figures/test_accuracy.png` | Test accuracy vs epoch |
| `METRICS.md` / `RUN_LOG.txt` / `AI_USE.md` | Standing notes |
