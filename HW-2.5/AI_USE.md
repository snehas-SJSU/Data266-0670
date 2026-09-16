# AI-Use Appendix — Sneha Singh

SID4 = 0670 | SEED = 670 | SLICE = 670 | HP_ID = 4 | CLS_A = 0 | CLS_B = 5

1. Which parts did you use an assistant for, and which did you write yourself?

**What I did**

SID4 / SEED. RTX 4090 on `ADS-R15-840-01`, UUID `GPU-5b052ad1-4272-40db-4b25-c930bf32b547`. I ran the jobs on that machine (`run_quick.bat` then `run_full.bat`), copied `results/` off, and wrote the tables and answers.

- Part A: `nvidia-smi -q`, UUID, driver, CUDA, VRAM, power limit, Ada whitepaper specs
- Part B: GEMM at N = 1024, 4096, 8192, 16384 in FP32 / TF32 / FP16 / BF16, reps, % of peak, the plot, where it plateaus, FP8 and FP4 from the log
- Part C: elementwise add vs large GEMM, GB/s, which side of the roofline
- Part D: I picked batch 1, 32 heads, head size 64. Naive vs fused sweep, OOM 26752 / 26816, quadratic fit, speedup table
- Part E: 20 minute load, clock/temp plot, steady vs first 30 s
- Part F Table HW2.5.1, GPU hours (2 h 1 min on the card; 3 h 20 min at the lab including scripts), conclusion

**Where I took help (topics)**

CUDA event timing (so TFLOPS is GPU time, not Python `time`). The fused attention call (`scaled_dot_product_attention`). How to start a `.py` job on the lab Windows PC from Command Prompt.

A few things still went wrong:

- Run All on my laptop does nothing CUDA. `run_quick.bat` / `run_full.bat` on the lab PC is what actually saw the 4090.
- naive attention at seq=16384 did not OOM on the 24 GB card.
- FP4: PyTorch had a dtype name but no multiply.

2. Give one specific thing it produced that was wrong. Paste the wrong output.

It said naive attention would OOM at seq=16384 on 24 GB if I used 32 heads. On this 4090 that length still ran. Peak memory was 34.6 GB (Windows shared memory). This is what I saw in `RUN_LOG.txt`:

```
config: B=1 H=32 D=64 dtype=bfloat16
  seq=16384  naive=918.11ms  fused=17.46ms  speedup=52.59x  mem naive=34.64GB  fused=0.29GB
  naive seq=16384  918.11ms  peak=34.637 GB  reps=10
naive OOM: largest ok=26752  smallest fail=26816  gap=64
```

3. How did you find out? What did the failure look like?

I ran Part D on the lab PC and opened the log. I expected a CUDA OOM at 16384. The line was `ok` with `peak=34.637 GB` and a 918 ms forward. nvidia-smi said 24564 MiB. The search kept going and died later: 26752 worked, 26816 failed.

FP4 was a printed `ok=False` after FP8 had already worked. No kernel ran.

4. What did you change, and why does your version work?

I ran it on the lab 4090 with `run_quick.bat` then `run_full.bat`. I did not rerun Part D to force an OOM at 16384. I reported 26752 and 26816, gap 64.

FP8 ran (263.65 TFLOPS). I left the FP4 fail in the log. I did not invent FP4 TFLOPS.
