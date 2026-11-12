# DATA 266 — Homework 11

Sneha Singh  
SID4 = 0670 | SEED = 670 | SLICE = 670 | HP_ID = 4 | CLS_A = 0 | CLS_B = 5

Findings: [`HW11_Report.pdf`](HW11_Report.pdf)

## CLIP Caption–Image Similarity and Zero-Shot CIFAR-10

Notebook: `clip_cifar.ipynb`. Model: `openai/clip-vit-base-patch32`. Dataset: CIFAR-10.

Part A scores caption `"a photo of a dog"` against 100 images per class (1,000 total). Dog has the highest mean similarity (0.260). Part B runs zero-shot classification on the full CIFAR-10 test set with prompts `a photo of a {class}`. Overall accuracy: **0.888**.

## Files

| File | Role |
|------|------|
| `clip_cifar.ipynb` | Executed notebook |
| `HW11_Report.pdf` | Findings and discussion |
| `figures/caption_class_similarity.png` | Mean caption similarity by class |
| `figures/caption_top_matches.png` | Top-12 images for the caption |
| `figures/zeroshot_per_class_accuracy.png` | Per-class zero-shot accuracy |
| `figures/zeroshot_confusion_matrix.png` | Confusion matrix |
| `outputs/metrics.json` | Printed metric table |
| `METRICS.md` / `RUN_LOG.txt` / `AI_USE.md` | Standing notes |
