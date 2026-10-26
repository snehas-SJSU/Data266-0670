# GPU reservation and hours — DATA 266 HW4.5

Sneha Singh  SID4=0670

Part C skipped (optional). One GPU.

Times below are from the Colab Pro L4 run of `hw4.5.ipynb` on 26 October 2026. Local time is PDT (UTC−7). The METRICS append in `RUN_LOG.txt` is stamped `2026-10-26 22:35:30 UTC` (15:35 PDT). Checkpoint `model.safetensors` on Drive is dated 14:51 PDT the same day.

## Card

| Field | Value |
|-------|-------|
| Lab / hostname | Google Colab Pro |
| GPU name | NVIDIA L4 |
| GPU UUID (from `nvidia-smi -q`) | not captured in the notebook (Step 0 printed the device name only) |
| Reservation start (local) | Monday 26 October 2026 |
| Reservation end (local) | Monday 26 October 2026 |
| Hours reserved | Colab Pro session, same day |
| Login time | ~2026-10-26 14:45 PDT (Drive mount / first training cells) |
| Logout time | 2026-10-26 15:35 PDT (`RUN_LOG` METRICS stamp 22:35:30 UTC) |
| Time at the machine | ~50 min (notebook run in the browser) |
| GPU hours used | ~0.8 h wall-clock for the full notebook; **0.0314 GPU-hours** for one baseline fine-tune (113.1 s / 3600) |
| Job command | Run all cells in `hw4.5.ipynb` on Colab (GPU runtime) |
| Notes | Saved to `MyDrive/HW-4.5`. One checkpoint only: `checkpoints/bert_baseline/`. Part C skipped. |

## Totals

| Card | Reserved | Session wall time | Baseline GPU-hours |
|------|----------|-------------------|--------------------|
| L4 (Colab Pro) | Mon 26 Oct 2026 | ~50 min | **0.0314** (113.1 s / 3600) |

GPU-hours for one baseline fine-tune are training seconds / 3600. That is the measured training time, not the hours the Colab seat was open.
