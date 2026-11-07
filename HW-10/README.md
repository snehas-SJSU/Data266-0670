# DATA 266 — Homework 10

Sneha Singh  
SID4 = 0670 | SEED = 670 | SLICE = 670 | HP_ID = 4 | CLS_A = 0 | CLS_B = 5

Findings: [`HW10_Report.pdf`](HW10_Report.pdf)

## Improving RAG with Reranking, Chunking, and Conversational Memory

Notebook: `rag_rerank_memory.ipynb`. Dataset: `ccdv/arxiv-summarization` (`document`, first 100 train articles).

I build two Chroma indexes (fixed-size and recursive chunking), retrieve with and without a cross-encoder reranker, and add a short conversational memory for a dependent follow-up. Metrics are Precision@5, Recall@5, and MRR. Relevance is manual by source document id.

Average Precision@5 on Q1/Q2: 0.900 without rerank, 1.000 with rerank. Follow-up Precision@5: 0.800 without memory, 1.000 with memory.

## Files

| File | Role |
|------|------|
| `rag_rerank_memory.ipynb` | Executed notebook |
| `HW10_Report.pdf` | Findings and discussion |
| `figures/chunk_rerank_compare.png` | Chunking / rerank Precision@5 and MRR |
| `figures/memory_compare.png` | Follow-up MRR with and without memory |
| `outputs/metrics.json` | Printed metric table |
| `METRICS.md` / `RUN_LOG.txt` / `AI_USE.md` | Standing notes |
