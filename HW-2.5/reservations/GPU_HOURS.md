# GPU reservation and hours — DATA 266 HW2.5

Sneha Singh  SID4=0670

Times below are from `results/NVIDIAGeForceRTX4090_5b052ad1/RUN_LOG.txt` and `nvidia_smi_q.txt` (machine local = PDT, UTC−7). I ran this on the RTX 4090.

## RTX 4090 workstation

| Field | Value |
|-------|-------|
| Lab / hostname | ADS-R15-840-01 |
| Machine label (door / inventory) | RTX 4090 workstation |
| Reservation start (local) | Monday 14 September 2026 |
| Reservation end (local) | Monday 14 September 2026 |
| Hours reserved | Monday 14 September 2026 (same day) |
| Login time | 2026-09-14 16:06 PDT (first RUN_LOG block, 23:06 UTC) |
| Logout time | 2026-09-14 18:07 PDT (last thermal sample / Part E finished, 01:07 UTC 15 Sep) |
| Hours actually used | **2 h 1 min** from machine logs (16:06–18:07 PDT). First stamp: smoke-test Part A. Last stamp: Part E thermal end in `thermal_smi.csv` / RUN_LOG. |
| GPU UUID (from `nvidia-smi -q` / Part A) | `GPU-5b052ad1-4272-40db-4b25-c930bf32b547` |
| Job command | `run_quick.bat` then `run_full.bat` (`python scripts\run_all.py`) |
| Notes (queue wait, reboot, someone else on the card) | Ran from Desktop\HW-2.5 on the lab Windows login. Smoke test first, then A–D. The first Part E window looked frozen, so I ran Part E again by itself. The committed log is that second run: 1202 s (20 min) BF16 GEMM N=8192. Same UUID. Driver in `nvidia-smi -q`: 595.95, CUDA 13.2, VRAM 24564 MiB, power limit 450.00 W. |

Breakdown from RUN_LOG (full run, UUID `GPU-5b052ad1-…`):

| Part | Local PDT | UTC |
|------|-----------|-----|
| A (`nvidia-smi -q`) | 16:22 | 23:22:58 |
| B GEMM (N=1024…16384) | 16:23 | 23:23:22 |
| C roofline | 16:23 | 23:23:28 |
| D attention / OOM | 16:29 | 23:29:22 |
| E thermal finished | 18:07 | 01:07:07 (15 Sep UTC) |

## Totals

| Card | Reserved (h) | Used (h) |
|------|--------------|----------|
| 4090 | Mon 14 Sep 2026 | **2 h 1 min** (16:06–18:07 PDT) |

Reserved: Monday 14 September 2026. Used: **2 h 1 min** (from `RUN_LOG.txt` and `thermal_smi.csv`).
