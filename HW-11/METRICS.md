# METRICS — DATA 266 HW11

Sneha Singh  
SID4 = 0670 | SEED = 670 | SLICE = 670 | HP_ID = 4 | CLS_A = 0 | CLS_B = 5

From `clip_cifar.ipynb` on Apple Silicon (MPS).

## Setup

| Item | Value |
|------|-------|
| Model | `openai/clip-vit-base-patch32` (~151M params) |
| Dataset | CIFAR-10 (train for Part A, test for Part B) |
| Caption | `a photo of a dog` |
| Part A sample | 100 images per class (1,000 total), SEED=670 |
| Zero-shot prompts | `a photo of a {class}` for all 10 classes |
| Device | MPS |
| Batch size | 64 |

## Part A — Caption–image mean cosine similarity

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

Highest mean class: **dog** (0.260).

## Part B — Zero-shot CIFAR-10 (test set, n=10,000)

Overall accuracy: **0.888**

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
