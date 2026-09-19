# METRICS — DATA 266 HW4

Sneha Singh  
SID4 = 0670 | SEED = 670 | SLICE = 670 | HP_ID = 4 | CLS_A = 0 | CLS_B = 5  
HP_ID is reported only. No second hyperparameter model.

## Model

| Field | Value |
|-------|-------|
| Dataset | course `data/shakespeare.txt` (balcony scene) |
| Chars in file | 1738 |
| Vocab size (char) | 49 |
| Seq length | 128 |
| Hidden dim | 128 |
| Heads | 4 |
| Layers | 2 |
| Parameters | 425,728 |
| Optimizer | Adam, lr=3e-4 |
| Batch | 64 |
| Epochs | 8 (300 random minibatches / epoch; assignment 5–8) |
| Device | cpu |

## Training loss by epoch

| Epoch | Train loss |
|-------|------------|
| 1 | 1.849 |
| 2 | 0.118 |
| 3 | 0.020 |
| 4 | 0.015 |
| 5 | 0.013 |
| 6 | 0.012 |
| 7 | 0.012 |
| 8 | 0.012 |

Plot: `figures/train_loss.png`

## Sampling prompt

`ROMEO:` — 200 new characters each.

| Method | Setting | Behavior (short) |
|--------|---------|------------------|
| Greedy | argmax | Repeats training opening, then drifts |
| Temperature | 0.2 | Almost greedy |
| Temperature | 0.7 | More spelling/spacing noise in the tail |
| Temperature | 1.2 | Noisiest / most diverse tail |
| Top-k | 5 | Narrower after memorized start |
| Top-k | 50 | Wider / odder fragments than k=5 |

Full strings are printed in `mini_gpt.ipynb` Part 4.
