# METRICS — DATA 266 HW2

SID4 = 0670 | SEED = 670 | SLICE = 670 | HP_ID = 4 | CLS_A = 0 | CLS_B = 5  
HP_ID is reported only. No HW2 mapping.

From the executed notebooks. Timing is not exact equality (standing 0.2).

## Part 1 — Word2Vec neighbors and shift

See `embeddings.ipynb`. IMDB: 5000 reviews, seed 670. Fine-tune: Word2Vec, 2 epochs.

**Table 1 — top-3 neighbors**

| Word | Before (Google News) | After (IMDB) |
|------|----------------------|--------------|
| cast | casts=0.722; casting=0.719; Cast=0.664 | casting=0.726; actors=0.706; supporting=0.700 |
| score | scoring=0.720; scores=0.660; scored=0.638 | scores=0.678; scoring=0.675; scored=0.618 |
| plot | plots=0.762; Plot=0.652; plotting=0.633 | plots=0.729; story=0.719; storyline=0.699 |
| screen | screens=0.773; onscreen=0.612; LCD_screen=0.560 | screens=0.691; stage=0.619; theatre=0.593 |
| review | reviewed=0.663; reviewing=0.661; reviews=0.638 | reviews=0.784; uwe=0.694; seagal=0.687 |

**Table 2 — cosine(original, fine-tuned)**

| Word | cosine |
|------|--------|
| cast | 0.8019 |
| score | 0.8618 |
| plot | 0.7806 |
| screen | 0.7885 |
| review | 0.7306 |

Most shifted: **review** (0.7306)  
Least shifted: **score** (0.8618)  
Figure: t-SNE in `embeddings.ipynb` §1.5

## Part 2 — RAG retrieval

See `rag.ipynb`. Chunk 500 / overlap 50, k=3. LLM: Flan-T5-small.

**Table 3 — retrieval success (top-3)**

| Q | Gold | In top-3? | Rank | Notes |
|---|------|-----------|------|-------|
| 1 | Matt Reeves / The Batman (film) | Yes | 1 | right page |
| 2 | Dune year 2021 | Yes | 1 | rank 1 is Part Two page talking about 2021 |
| 3 | Palme d'Or at Cannes | Yes | 1 | LLM still said Oscar |
| 4 | Hans Zimmer / Inception | No | — | composer not in top-3 |
| 5 | Leonardo DiCaprio as Jack Dawson | Yes | 2 | rank 1 is the J. Dawson grave |

Retrieval Success Rate = **4 / 5 = 0.80**

Chunk ablation Q1/Q2, 200/20 vs 500/50: Q1 answer stayed Matt Reeves; ranks 2–3 became Batman Begins. Q2 still 2021; Part Two still ranked high.

## Part 3 — Optimization experiment

See `optimizations.ipynb`. Device: **mps** for init / checkpoint / accum / tensor-create. Mixed precision: Colab **Tesla T4** in `amp_cuda_colab.ipynb`. Same 12×512 MLP, batch 64, 40 steps, Adam 0.001. MPS `mem_mb` is current allocated, not CUDA peak.

**Tensor create** 2048×2048, mean of 3 after warmup, with `torch.mps.synchronize()`: CPU 61.087 ms; MPS 2.167 ms.

**Weight init**

| init | seconds | mem_mb | final_loss |
|------|---------|--------|------------|
| default | 4.559 | 28.4 | 1.4873 |
| xavier | 0.231 | 26.3 | 1.6886 |
| zeros | 0.204 | 26.3 | 2.2914 |

Default time includes a cold MPS graph (first train). Zeros did not train (loss 2.29).

**Checkpointing**

| checkpoint | seconds | mem_mb | final_loss |
|------------|---------|--------|------------|
| False | 0.273 | 26.3 | 1.4873 |
| True | 0.333 | 26.3 | 1.4873 |

**Gradient accumulation** (effective batch 64, 40 opt steps)

| accum | micro | seconds | mem_mb | final_loss |
|-------|-------|---------|--------|------------|
| 1 | 64 | 0.395 | 26.3 | 1.4873 |
| 4 | 16 | 0.990 | 26.2 | 1.5353 |

**Mixed precision** — Colab Tesla T4 (`amp_cuda_colab.ipynb`). Same 12×512 MLP, batch 64, 40 steps, seed 670. Warmup, then mean of 3 with `torch.cuda.synchronize()`. Memory = CUDA peak allocated. Autocast: `cuda` + `float16` + GradScaler. Probe: `used_fp16: True`.

| amp | seconds | peak_mem_mb | final_loss | forward dtype |
|-----|---------|-------------|------------|---------------|
| False | 0.2343 | 82.6 | 1.4924 | torch.float32 |
| True | 0.2688 | 82.6 | 1.4793 | torch.float16 |

AMP used real fp16. It was slightly slower on this small net. Peak memory did not drop.
