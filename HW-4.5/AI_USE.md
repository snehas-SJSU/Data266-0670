# AI-Use Appendix — Sneha Singh

SID4 = 0670 | SEED = 670 | SLICE = 670 | HP_ID = 4 | CLS_A = 0 | CLS_B = 5

1. Which parts did you use an assistant for, and which did you write yourself?

I set the standing parameters the same way as HW1: SID4 670, SEED 670, SLICE 670, HP_ID 4 reported only, CLS_A 0, CLS_B 5. I kept Part C skipped because the October 23 TA note says it is optional, and I used one GPU. I ran `hw4.5.ipynb` on Colab Pro (NVIDIA L4), checked the printed tables against `RUN_LOG.txt`, and wrote the report myself from those numbers. I also rewrote the five misclassified-tweet diagnoses so they match the wording in each tweet.

- Data: `cardiffnlp/tweet_eval` sentiment, train split, rows 670–5669, then a 70/15/15 split with SEED 670
- HW4.a baseline: `bert-base-uncased`, BertTokenizer, max length 128, batch 16, AdamW, lr 2e-5, 3 epochs
- Part A: power-of-two batch until out of memory
- Part B: mixed precision, gradient checkpointing, gradient accumulation, 8-bit AdamW, then the combination the ranking rule selects
- Serving: `facebook/opt-125m`, my prompt plus 31 fixed prompts, HF pipeline batch 1 and vLLM batches 1, 8, and 32
- Part C: skipped

An assistant helped draft the notebook cells and the Colab Drive save path. The numbers in the report are not from the assistant. They are whatever the L4 kernel printed.

2. Give one specific thing it produced that was wrong. Paste the wrong output.

The first plan for Part A said:

```
Throughput at each batch size is measured over a fixed 50 optimizer steps.
```

3. How did you find out? What did the failure look like?

The training split is 3,500 rows. At batch 256 that is about 14 steps, so 50 steps would run more than one epoch and repeat rows. The assignment asks for time per epoch and for the largest batch that runs, not for a fixed step budget. A 50-step timer would not be either of those measurements.

4. What did you change, and why does your version work?

In the notebook, each batch size does two measured things. One forward and backward on exactly that many length-128 rows decides whether the batch fits. One pass over the training rows records samples per second, peak allocated memory, and time per epoch. The ladder stops at the first out-of-memory batch and keeps the last size that finished. On the L4 that was batch 256 (512 OOM). The last cell wrote `METRICS.md` from that run only. The report uses those printed numbers: baseline test accuracy 0.7267, best combination 222.0 samples/s at 1.64 GB, and 0.0314 GPU-hours.
