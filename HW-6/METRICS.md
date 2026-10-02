# METRICS — DATA 266 HW6

Sneha Singh  
SID4 = 0670 | SEED = 670 | SLICE = 670 | HP_ID = 4 | CLS_A = 0 | CLS_B = 5  
HP_ID is reported only. Three trained models: supervised, rotation SSL, SimCLR.

From `stl10_ssl.ipynb` on Colab GPU. Second run, after matching Demo 6.

## Setup

| Field | Value |
|-------|-------|
| Dataset | STL-10 |
| Backbone | ResNet-18, no pretrained weights |
| Labeled subset | 500 / 5,000 train, seed 670, shared by A, B, C |
| Test | 8,000 |
| Unlabeled | 100,000 (rotation uses all; SimCLR uses 20,000) |
| Device | cuda |
| Optimizer | Adam, lr 0.001 |
| SimCLR τ | 0.2 |
| SimCLR views | random resized crop (scale 0.6 to 1.0), horizontal flip, color jitter, random grayscale (p=0.2) |
| Queries | test 3806 car, 1720 ship, 2051 dog |

## Parameters

| Stage | Trainable |
|-------|----------:|
| A end-to-end | 11,181,642 |
| B rotation pretrain | 11,178,564 |
| B linear probe (encoder frozen) | 5,130 |
| C SimCLR pretrain | 11,258,688 |
| C linear probe (encoder frozen) | 5,130 |

## Test accuracy

| Model | Epochs | Final train loss | Test accuracy |
|-------|-------:|-----------------:|--------------:|
| A supervised | 12 | 0.1047 | 0.3426 |
| B rotation pretrain | 15 | 0.2880 | |
| B linear probe | 15 | 1.5839 | 0.4260 |
| C SimCLR pretrain | 15 | 1.6038 | |
| C linear probe | 15 | 0.9113 | 0.4980 |

C probe epoch 14 was 0.5012. The reported test accuracy is the last epoch.

Plot: `figures/test_accuracy.png`  
Neighbors: `figures/nearest_neighbors.png`
