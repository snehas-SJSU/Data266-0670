# AI-Use Appendix — Sneha Singh

SID4 = 0670 | SEED = 670 | SLICE = 670 | HP_ID = 4 | CLS_A = 0 | CLS_B = 5

1. Which parts did you use an assistant for, and which did you write yourself?

I set Step 0 with SID4 670 and seed 670. I chose `facebook/opt-125m` for GPTQ on Wikitext-2, used the provided `teacher_model` folder, set the student to `distilbert/distilbert-base-uncased`, and kept the HW12.1 / HW12.2 tables separate. I ran `llm_compress.ipynb` on my Mac and used the printed metrics and figures in the report.

- Step 0: SID4 670, seeds, HP_ID reported only
- Part 1: GPTQ 4-bit on `facebook/opt-125m`, `Salesforce/wikitext` (`wikitext-2-raw-v1`)
- Part 2: AG News distillation, teacher logits for all train samples
- Sweeps: T ∈ {1,2,5,10} with α=0.5; then α ∈ {0.25,0.5,0.75} at best T
- Metrics: perplexity / latency / memory / disk for GPTQ; accuracy / latency / memory / disk for distill
- Error analysis: per-class drop + five teacher-right / student-wrong examples

I used an assistant to draft the notebook cells, the GPTQ measurement helpers, and the distillation train loop.

2. Give one specific thing it produced that was wrong. Paste the wrong output.

The first AG News load used the old dataset id and failed with:

```
HfUriError: Repository id must be 'namespace/name', got 'ag_news'.
```

3. How did you find out? What did the failure look like?

The notebook finished the GPTQ section, then stopped at the distillation data cell. No teacher logits file was written, and the T / alpha sweeps never started.

4. What did you change, and why does your version work?

I switched the dataset id to `fancyzhx/ag_news`. After that the notebook ran top to bottom: GPTQ cut disk from about 242 MB to 123 MB with a small ppl rise, best student was `T2_a50` at 0.9389 accuracy, and BUSINESS had the largest per-class drop. The printed numbers match `METRICS.md`.
