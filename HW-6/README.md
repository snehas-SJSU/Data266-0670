# DATA 266 — Homework 6

Sneha Singh  
SID4 = 0670 | SEED = 670 | SLICE = 670 | HP_ID = 4 | CLS_A = 0 | CLS_B = 5

Notebook: `stl10_ssl.ipynb`. Dataset: STL-10. Backbone: ResNet-18, no pretrained weights.

HP_ID is reported only. No second hyperparameter network. The three models are supervised, rotation SSL, and SimCLR.

## Run

Colab GPU. Runtime → GPU, then Run all.

Part A is 500 labeled images, 12 epochs. Part B is the full unlabeled split (100,000) for 15 epochs, then a frozen linear probe for 15 epochs. Part C is 20,000 unlabeled images, two views, 15 epochs, then the same frozen linear probe. The 10% label list is saved and reused.

## Files

| File | Role |
|------|------|
| `stl10_ssl.ipynb` | Parts A–D |
| `checkpoints/` | Saved weights and the shared 500-index list |
| `figures/nearest_neighbors.png` | Same 3 queries, top-5 cosine neighbors |
| `AI_USE.md` | Standing note |
