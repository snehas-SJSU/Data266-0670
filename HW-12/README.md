# DATA 266 — Homework 12

Sneha Singh  
SID4 = 0670 | SEED = 670 | SLICE = 670 | HP_ID = 4 | CLS_A = 0 | CLS_B = 5

Findings: [`HW12_Report.pdf`](HW12_Report.pdf)

## LLM Optimisation with GPTQ and Knowledge Distillation

Notebook: `llm_compress.ipynb`.

**Part 1 — GPTQ** on `facebook/opt-125m` with [Salesforce/wikitext](https://huggingface.co/datasets/Salesforce/wikitext) (`wikitext-2-raw-v1`). Disk falls from 242 MB → 123 MB; perplexity 43.96 → 47.47.

**Part 2 — Distillation** from local `teacher_model` (BERT-base AG News) into `distilbert/distilbert-base-uncased`. Best student: **T=2, α=0.5**, test accuracy **0.9389** (teacher 0.9443).

## Files

| File | Role |
|------|------|
| `llm_compress.ipynb` | Executed notebook |
| `HW12_Report.pdf` | Findings and discussion |
| `figures/` | Sweep plots, per-class bars, confusion matrix |
| `outputs/metrics.json` | Printed metric tables |
| `METRICS.md` / `RUN_LOG.txt` / `AI_USE.md` | Standing notes |

Local-only (not in git): `teacher_model/`, GPTQ/base checkpoints, `student_runs/`, `outputs/teacher_logits.pt`.
