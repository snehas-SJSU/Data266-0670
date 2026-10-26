# GPU capture — DATA 266 HW4.5

Sneha Singh · SID4=0670

Part C skipped (optional). One GPU only (Colab Pro is an allowed option; the October 23 note says one GPU is enough).

## Capture 1 — training (HW4.a baseline / Parts A–B)

| Field | Value |
|-------|-------|
| Source | `hw4.5.ipynb` Step 0 + baseline cell |
| Host | Google Colab Pro |
| GPU name | NVIDIA L4 |
| CUDA device | `cuda:0` |
| Peak allocated (baseline) | 2.48 GB |
| Baseline wall-clock | 113.1 s |
| Drive checkpoint stamp | 2026-10-26 ~14:51 PDT |

## Capture 2 — serving (HW4.a / Part D vLLM)

| Field | Value |
|-------|-------|
| Source | `hw4.5.ipynb` HF pipeline + vLLM cells |
| Host | same Colab Pro session |
| GPU name | NVIDIA L4 |
| Model | facebook/opt-125m |
| HF pipeline stamp | same run, before Part A |
| vLLM log stamp | ~2026-10-26 21:56–21:58 UTC |

## UUID note

`nvidia-smi -q` UUID was not printed in this notebook run. Step 0 only called `torch.cuda.get_device_name(0)`, which returned `NVIDIA L4`. Both captures above are that same L4 in one Colab session. There is no second physical card and therefore no second UUID.
