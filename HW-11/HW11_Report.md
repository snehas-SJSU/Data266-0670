# DATA 266 — Homework 11

Sneha Singh  
SID4 = 0670 | SEED = 670 | SLICE = 670 | HP_ID = 4 | CLS_A = 0 | CLS_B = 5

Notebook: `clip_cifar.ipynb`. Model: `openai/clip-vit-base-patch32`. Dataset: CIFAR-10.

HP_ID is reported only.

## 1. Setup

| Item | Value |
|------|-------|
| Model | `openai/clip-vit-base-patch32` (~151M params) |
| Dataset | CIFAR-10 |
| Caption | `a photo of a dog` |
| Part A sample | 100 images per class (SEED=670) |
| Zero-shot prompts | `a photo of a {class}` |
| Device | MPS |
| Batch | 64 |

I load CLIP with the matching processor. Text and image embeddings are L2-normalized, then compared with cosine similarity (dot product of unit vectors).

## 2. Caption–image similarity

I scored the caption against 1,000 CIFAR-10 train images (100 per class).

| Class | Mean similarity |
|-------|----------------:|
| airplane | 0.204 |
| automobile | 0.200 |
| bird | 0.227 |
| cat | 0.240 |
| deer | 0.234 |
| dog | **0.260** |
| frog | 0.229 |
| horse | 0.221 |
| ship | 0.195 |
| truck | 0.196 |

![Mean caption similarity by class](figures/caption_class_similarity.png)

Dog has the highest mean score (0.260). Cat is next (0.240), then other animals. Vehicles sit at the bottom. That is the expected ranking for a dog caption.

![Top matches for the caption](figures/caption_top_matches.png)

The top-12 gallery is mostly dog images, with a few other animals that still look somewhat dog-like at 32×32.

## 3. Zero-shot CIFAR-10 classification

I embedded all 10 class prompts and classified every CIFAR-10 test image by the highest cosine similarity. Overall accuracy: **0.888**.

| Class | Accuracy |
|-------|---------:|
| airplane | 0.901 |
| automobile | 0.976 |
| bird | 0.888 |
| cat | 0.860 |
| deer | 0.765 |
| dog | 0.884 |
| frog | 0.774 |
| horse | 0.976 |
| ship | 0.955 |
| truck | 0.901 |

![Per-class zero-shot accuracy](figures/zeroshot_per_class_accuracy.png)

![Confusion matrix](figures/zeroshot_confusion_matrix.png)

Automobile and horse are strongest (97.6%). Deer is weakest (76.5%), with many deer predicted as horse. Cat↔dog and frog↔bird/cat swaps show up in the matrix; those classes share texture and shape at CIFAR resolution.

## Conclusion

CLIP ranked the dog caption highest on dog images without any fine-tuning. Zero-shot CIFAR-10 reached 88.8% accuracy. Most errors are between visually similar classes (deer/horse, cat/dog), which matches what the confusion matrix shows.
