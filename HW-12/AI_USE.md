# AI-Use Appendix — Sneha Singh

SID4 = 0670 | SEED = 670 | SLICE = 670 | HP_ID = 4 | CLS_A = 0 | CLS_B = 5

1. Which parts did you use an assistant for, and which did you write yourself?

I set Step 0 with SID4 670 and seed 670. I chose `facebook/opt-125m` for the GPTQ part on Wikitext-2, used the provided `teacher_model` folder, and set the student to `distilbert/distilbert-base-uncased`. I ran `llm_compress.ipynb` on my Mac and used the printed metrics and figures in the report.

- Part 1: GPTQ 4-bit on OPT-125m, Wikitext-2 only
- Part 2: AG News distillation, T ∈ {1,2,5,10}, α ∈ {0.25,0.5,0.75}
- Separate HW12.1 / HW12.2 tables (no mixed four-model accuracy table)

2. Give one specific thing it produced that was wrong. Paste the wrong output.

The first AG News load used the old dataset id and failed with:

```
HfUriError: Repository id must be 'namespace/name', got 'ag_news'.
```

3. How did you find out? What did the failure look like?

The notebook finished GPTQ, then stopped at the distillation data cell before any teacher logits were written.

4. What did you change, and why does your version work?

I switched to `fancyzhx/ag_news`. After that the run completed: GPTQ disk ~half at a small ppl cost, best student `T2_a50` at 0.9389 accuracy, and BUSINESS as the largest per-class drop. Numbers match `METRICS.md`.
