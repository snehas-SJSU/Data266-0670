# AI-Use Appendix — Sneha Singh

SID4 = 0670 | SEED = 670 | SLICE = 670 | HP_ID = 4 | CLS_A = 0 | CLS_B = 5

1. Which parts did you use an assistant for, and which did you write yourself?

I set SID4 / SEED, the five words, the 10 movie titles (I wanted Batman and Dune to collide), the five RAG questions, and the writeups next to the tables. I ran `embeddings.ipynb`, `rag.ipynb`, and `optimizations.ipynb` on kernel Python (HW1). I copied the printed tables into METRICS.md myself.

I used an assistant for gensim / LangChain boilerplate and for the Part 3 train loop (checkpoint, accum). I still had to fix what it wrote.

This Mac has no NVIDIA GPU, so mixed precision had to run on Colab CUDA. I used an assistant for the Colab CUDA notebook (`amp_cuda_colab.ipynb`): `torch.autocast` with fp16, `GradScaler`, `torch.cuda.synchronize()` for timing, and `max_memory_allocated` for peak memory. I still ran it myself on a Tesla T4, checked `used_fp16: True`, and copied the FP32 vs AMP time / memory / loss into METRICS.md. The neighbor tables, cosine numbers, RAG Yes/No ranks, and MPS timing are from my runs, not from it guessing.

2. Give one specific thing it produced that was wrong. Paste the wrong output.

It fine-tuned with `w2v.build_vocab` / `w2v.train` on the object from `api.load("word2vec-google-news-300")`. That object is KeyedVectors. It has `most_similar`. It does not train. This is what I got:

```
AttributeError: 'KeyedVectors' object has no attribute 'build_vocab'
```

from

```
w2v.build_vocab(sentences, update=True)
w2v.train(corpus_iterable=sentences, total_examples=len(sentences), epochs=FT_EPOCHS)
```

3. How did you find out? What did the failure look like?

I ran the fine-tune cell after Google News and IMDB had already loaded. Neighbors-before had printed. The train cell died on line 2. No AFTER table, no cosine, no t-SNE.

(Later it also pointed Keras IMDB at `.../aclImdb`. Keras 3 put the files in `aclImdb_v1_extracted/aclImdb`. That cell printed 0 review files, then `IMDB vocab: 0 reviews: 0`, then the same “build vocabulary first” error for a different reason. And `pipeline("text2text-generation")` crashed on transformers 5: `Unknown task text2text-generation`.)

4. What did you change, and why does your version work?

I left `w2v` as the frozen Google News table for neighbors-before and for the original vectors. I made a real `Word2Vec` model (`ft`), `build_vocab` on the IMDB sentences, copied overlapping rows into `ft.wv.vectors[idx]` (not `ft.wv[word] = ...`, which wiped the vocab in gensim 4), then `ft.train`. AFTER neighbors and cosine use `ft.wv`.

For IMDB I check both extract folders. For the LLM I call Flan-T5 `generate()` instead of the removed pipeline task. Those three cells then ran and produced the tables in METRICS.md.
