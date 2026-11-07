# METRICS — DATA 266 HW10

Sneha Singh  
SID4 = 0670 | SEED = 670 | SLICE = 670 | HP_ID = 4 | CLS_A = 0 | CLS_B = 5

From `rag_rerank_memory.ipynb` on Apple Silicon (local ChromaDB).

## Setup

| Item | Value |
|------|-------|
| Dataset | `ccdv/arxiv-summarization`, `document`, train[:100] |
| Vector DB | ChromaDB (local `chroma_store/`) |
| Embedding | `sentence-transformers/all-MiniLM-L6-v2` |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Chunk A | fixed-size 500 / 50 → 7,220 chunks |
| Chunk B | recursive 500 / 50 → 7,224 chunks |
| Retrieve | top-5 baseline; top-20 then rerank to top-5 |
| K | 5 |

## Queries and gold docs

| Query | Text | Gold doc_id |
|-------|------|------------:|
| Q1 | learning rates for regularized kernel based methods for additive models | 0 |
| Q2 | positive linear magnetoresistance in topological insulator surface states | 20 |
| Q3 follow-up | What learning rates does the paper report? | 0 |

A chunk is relevant if its `doc_id` is in the gold set.

## Full metric table

| Strategy | Query | Mode | Precision@5 | Recall@5 | MRR |
|----------|-------|------|------------:|---------:|----:|
| fixed | Q1 | no rerank | 1.000 | 1.000 | 1.000 |
| fixed | Q1 | rerank | 1.000 | 1.000 | 1.000 |
| fixed | Q2 | no rerank | 0.800 | 1.000 | 1.000 |
| fixed | Q2 | rerank | 1.000 | 1.000 | 1.000 |
| recursive | Q1 | no rerank | 1.000 | 1.000 | 1.000 |
| recursive | Q1 | rerank | 1.000 | 1.000 | 1.000 |
| recursive | Q2 | no rerank | 0.800 | 1.000 | 1.000 |
| recursive | Q2 | rerank | 1.000 | 1.000 | 1.000 |
| recursive | Q3 follow-up | no memory | 0.800 | 1.000 | 1.000 |
| recursive | Q3 follow-up | with memory | 1.000 | 1.000 | 1.000 |
| recursive | Q3 follow-up | memory + rerank | 1.000 | 1.000 | 1.000 |

## Averages used in the report

| Comparison | Precision@5 | MRR |
|------------|------------:|----:|
| fixed, no rerank (avg Q1/Q2) | 0.900 | 1.000 |
| fixed, rerank (avg Q1/Q2) | 1.000 | 1.000 |
| recursive, no rerank (avg Q1/Q2) | 0.900 | 1.000 |
| recursive, rerank (avg Q1/Q2) | 1.000 | 1.000 |
