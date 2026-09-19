# AI-Use Appendix — Sneha Singh

SID4 = 0670 | SEED = 670 | SLICE = 670 | HP_ID = 4 | CLS_A = 0 | CLS_B = 5

1. Which parts did you use an assistant for, and which did you write yourself?

**What I did**

I used the course `shakespeare.txt` from week-4 files. I ran `mini_gpt.ipynb`, checked the loss curve, generated the `ROMEO:` samples, and wrote the findings PDF.

- Part 1: char vocab, `char_to_idx` / `idx_to_char`, seq=128 windows, two decoded input-target pairs
- Part 2: 4 heads, 2 layers, hidden 128. Causal Mini-GPT (no `nn.Transformer` / `nn.MultiheadAttention`)
- Part 3: Adam, lr 3e-4, batch 64, 8 epochs, loss plot
- Part 4: greedy, temperature 0.2 / 0.7 / 1.2, top-k 5 and 50
- Part 5: most coherent = greedy; most diverse = temperature 1.2

**Where I took help (topics)**

How to split Q/K/V into heads in the attention reshape. How to apply the causal `-inf` mask before softmax. Small syntax help on the sampling loop.

2. Give one specific thing it produced that was wrong. Paste the wrong output.

In the attention `__init__`, `self.qkv` got an extra indent so the cell would not run:

```
IndentationError: unexpected indent
        self.head_dim = n_embd // n_head
                self.qkv = nn.Linear(n_embd, 3 * n_embd)
        self.proj = nn.Linear(n_embd, n_embd)
```

3. How did you find out? What did the failure look like?

I tried Run All / compile on that cell. Only that cell failed; older printed outputs were still sitting under it from a previous successful run.

4. What did you change, and why does your version work?

I lined `self.qkv` up with `self.head_dim` / `self.proj` (same indent). After that, Run All finished: loss ~1.85 → 0.012 over 8 epochs, and all three sampling methods printed.
