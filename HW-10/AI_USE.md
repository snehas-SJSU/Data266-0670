# AI-Use Appendix — Sneha Singh

SID4 = 0670 | SEED = 670 | SLICE = 670 | HP_ID = 4 | CLS_A = 0 | CLS_B = 5

1. Which parts did you use an assistant for, and which did you write yourself?

I set Step 0 with SID4 670 and seed 670. I picked the first 100 `document` train articles, the two chunkers (fixed and recursive), local ChromaDB, the two content queries, and the follow-up. I ran `rag_rerank_memory.ipynb` on my Mac and used the printed metrics and figures in the report.

- Step 0: SID4 670, seeds, HP_ID reported only
- Data: `ccdv/arxiv-summarization`, document, train[:100]
- Chunking: fixed-size 500/50 and recursive 500/50
- Index: Chroma + `all-MiniLM-L6-v2`
- Rerank: cross-encoder `ms-marco-MiniLM-L-6-v2`
- Memory: rewrite the follow-up from the previous user turn
- Metrics: Precision@5, Recall@5, MRR with manual gold doc ids

I used an assistant to draft the notebook cells and the Chroma / rerank wiring.

2. Give one specific thing it produced that was wrong. Paste the wrong output.

The first draft treated every top-5 hit as equally “successful” and planned to say recursive chunking was clearly better. An early placeholder said:

```
Recursive chunking beats fixed-size on both queries after reranking.
```

3. How did you find out? What did the failure look like?

I read `outputs/metrics.json` after the run. Fixed and recursive had the same average Precision@5 (0.900 without rerank, 1.000 with rerank) and the same MRR (1.000). The only clear gains were reranking on Q2 and memory on the follow-up Precision@5.

4. What did you change, and why does your version work?

I rewrote the discussion and the report so they only claim what the table shows: rerank raised average Precision@5 from 0.900 to 1.000, memory raised follow-up Precision@5 from 0.800 to 1.000, and fixed vs recursive looked almost the same on this query set. The notebook ran top to bottom and the printed numbers match `METRICS.md`.
