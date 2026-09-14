# DATA 266 — Homework 3

Sneha Singh  
SID4 = 0670 | SEED = 670 | SLICE = 670 | HP_ID = 4 | CLS_A = 0 | CLS_B = 5

HP_ID is reported only. No second HP_ID model. Unmasked and masked attention are the two trained models.

Written report: [`HW3_Report.pdf`](HW3_Report.pdf)

## Files

| File | Role |
|------|------|
| `attention.ipynb` | prompts (LangChain) + self-attention from scratch |
| `HW3_Report.pdf` | findings |
| `METRICS.md` | prompt compare + attention notes |
| `RUN_LOG.txt` | console from the run |
| `AI_USE.md` | AI-use appendix |

## What I ran

- Kernel: **Python (HW1)**
- Prompts: LangChain `PromptTemplate` + local Flan-T5-small (same as HW2)
- Attention: word tokens, `nn.Embedding` + my own QKV, seed 670
