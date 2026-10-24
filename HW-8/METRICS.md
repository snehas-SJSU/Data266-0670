# Metrics

These are from the executed notebook run.

| Item | Value |
|------|-------|
| Device | MPS |
| Diffusion steps | 100 |
| β start | 0.0001 |
| β end | 0.02 |
| Batch size | 128 |
| Epochs | 50 |
| Learning rate | 0.001 |
| CNN parameters | 97,537 |
| Model output shape | `(2, 1, 28, 28)` |
| Training loss, epoch 1 | 0.164547 |
| Training loss, epoch 50 | 0.061000 |
| DDPM sampling steps | 100 |
| DDIM sampling steps | 50, 20 |

The notebook saved the three image grids under `outputs/` and the comparison figures under `figures/`.
