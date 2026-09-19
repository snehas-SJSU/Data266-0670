# AI-Use Appendix — Sneha Singh

SID4 = 0670 | SEED = 670 | SLICE = 670 | HP_ID = 4 | CLS_A = 0 | CLS_B = 5

1. Which parts did you use an assistant for, and which did you write yourself?

**What I did**

SID4 / SEED. I ran `mini_gpt.ipynb` on kernel Python (HW1), watched the loss go down, generated the `ROMEO:` samples, and wrote METRICS.md / the PDF myself.

- Part 1: course `shakespeare.txt`, char vocab, `char_to_idx` / `idx_to_char`, seq=128 windows, two decoded input-target pairs
- Part 2: I picked 4 heads, 2 layers, hidden 128. Causal Mini-GPT (no `nn.Transformer` / `nn.MultiheadAttention`)
- Part 3: Adam, lr 3e-4, batch 64, 8 epochs (assignment 5–8), loss plot
- Part 4: greedy, temperature 0.2 / 0.7 / 1.2, top-k 5 and 50
- Part 5: which looked coherent vs diverse from my prints

**Where I took help (topics)**

How to split Q/K/V into heads in the attention reshape. How to apply the causal `-inf` mask before softmax. Small syntax help on the sampling loop.

2. Give one specific thing it produced that was wrong. Paste the wrong output.

It first pointed me at the huge Karpathy shakespeare dump (~1M chars) instead of the short week-4 course file.

3. How did you find out? What did the failure look like?

I compared file sizes: Downloads `shakespeare.txt` was ~1.8KB (balcony scene). The big dump was a different file.

4. What did you change, and why does your version work?

I swapped in the course `shakespeare.txt`, kept seq_len=128 / 4 heads / 2 layers, and trained 8 epochs. Loss dropped to ~0.012 (short file = easy to memorize). Sampling still shows greedy vs temperature vs top-k differences in the tail.
