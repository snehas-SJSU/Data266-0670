# AI-Use Appendix — Sneha Singh

SID4 = 0670 | SEED = 670 | SLICE = 670 | HP_ID = 4 | CLS_A = 0 | CLS_B = 5

1. Which parts did you use an assistant for, and which did you write yourself?

I set Step 0 with SID4 670 and seed 670. I chose caption `"a photo of a dog"`, 100 images per CIFAR-10 class for Part A, and the full CIFAR-10 test set for zero-shot Part B. I ran `clip_cifar.ipynb` on my Mac (MPS) and used the printed metrics and figures in the report.

- Step 0: SID4 670, seeds, HP_ID reported only
- Model: `openai/clip-vit-base-patch32`
- Part A: caption–image cosine similarity, 100 images/class
- Part B: zero-shot prompts `a photo of a {class}`, per-class accuracy + confusion matrix

I used an assistant to draft the notebook cells and the CLIP embedding helpers.

2. Give one specific thing it produced that was wrong. Paste the wrong output.

The first draft treated `model.get_text_features(...)` as a tensor and called `F.normalize` on it directly. On my transformers version that call returns a `BaseModelOutputWithPooling`, so the run failed with:

```
AttributeError: 'BaseModelOutputWithPooling' object has no attribute 'norm'
```

3. How did you find out? What did the failure look like?

The notebook stopped right after CIFAR-10 finished downloading, before any similarity scores printed. The traceback pointed at `embed_texts` → `F.normalize`.

4. What did you change, and why does your version work?

I read `.pooler_output` from the feature helpers (that field already has the CLIP projection applied), then L2-normalize. After that fix the notebook ran top to bottom: dog ranked highest for the dog caption (mean 0.260), and zero-shot test accuracy was 0.888. The printed numbers match `METRICS.md`.
