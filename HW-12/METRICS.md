# METRICS — DATA 266 HW12

Sneha Singh  
SID4 = 0670 | SEED = 670 | SLICE = 670 | HP_ID = 4 | CLS_A = 0 | CLS_B = 5

From `llm_compress.ipynb` on Apple Silicon (MPS for DistilBERT; GPTQ quantize/infer on CPU).

## Table HW12.1 — GPTQ comparison (OPT-125m / Wikitext-2)

| Configuration | Quality metric | p50 / p95 latency (ms) | Peak memory (GB) | Size on disk (MB) |
|---------------|----------------|------------------------:|-----------------:|------------------:|
| Base LLM | ppl 43.963 | 18.2 / 21.7 | 0.69 | 242.3 |
| GPTQ-quantized LLM | ppl 47.465 | 68.3 / 73.2 | 0.55 | 122.7 |

## Table HW12.2 — Distillation comparison (AG News)

| Configuration | Accuracy | p50 / p95 latency (ms) | Peak memory (GB) | Size on disk (MB) |
|---------------|---------:|------------------------:|-----------------:|------------------:|
| Teacher (BERT-base) | 0.9443 | 461.6 / 499.5 | 0.41 | 418.6 |
| Best distilled student | 0.9389 | 144.0 / 147.7 | 1.43 | 256.1 |

Best student: `T2_a50` (T=2, α=0.5).

## Temperature sweep (α=0.5)

| T | Accuracy |
|--:|---------:|
| 1 | 0.9384 |
| 2 | **0.9389** |
| 5 | 0.9376 |
| 10 | 0.9384 |

## Loss-weight sweep (T=2)

| α | Accuracy |
|--:|---------:|
| 0.25 | 0.9384 |
| 0.50 | **0.9389** |
| 0.75 | 0.9382 |

## Per-class drop (teacher − student)

| Class | Teacher | Student | Drop |
|-------|--------:|--------:|-----:|
| WORLD | 0.9489 | 0.9484 | +0.0005 |
| SPORTS | 0.9884 | 0.9874 | +0.0011 |
| BUSINESS | 0.9137 | 0.8942 | **+0.0195** |
| SCI/TECH | 0.9263 | 0.9258 | +0.0005 |
