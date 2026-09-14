# AI-Use Appendix — Sneha Singh

SID4 = 0670 | SEED = 670 | SLICE = 670 | HP_ID = 4 | CLS_A = 0 | CLS_B = 5

1. Which parts did you use an assistant for, and which did you write yourself?

I set SID4 / SEED, picked the two prompt tasks (15% of 80 and the river-bank / age / telescope ones), ran `attention.ipynb` on kernel Python (HW1), and copied the printed outs into METRICS.md and the compare cell myself.

I used an assistant for LangChain `PromptTemplate` boilerplate and for the TinyAttn class (Q, K, V, the `-inf` mask). I still ran it and checked the heatmaps. I reused the Flan-T5 `generate()` wrapper from HW2 because transformers 5 still has no `text2text-generation` pipeline.

2. Give one specific thing it produced that was wrong. Paste the wrong output.

It wrote meta-prompting as: ask the model to write a better prompt, then `llm.invoke` that string. Flan-T5-small does not write a prompt. Cell output:

```
generated prompt: 80%
OUT: 80% of the population is white.
```

and for the bank one:

```
generated prompt: bank
OUT: bank
```

3. How did you find out? What did the failure look like?

I ran those two cells after zero-shot / few-shot / CoT. I expected a sentence like "Compute 15/100 * 80". I got `80%` instead. Feeding that back in produced an unrelated sentence about population. The bank cell just echoed `bank`. So the second call was not "the improved prompt." It was garbage in.

4. What did you change, and why does your version work?

I left the two-step meta cells in, because that is the technique, and I wrote down that Flan-T5 cannot do it. I still called it from code (`PromptTemplate` + `llm.invoke`), not from a chat window. For attention I kept the mask *before* softmax. After training, `torch.triu(w_m, diagonal=1).max()` printed `0.0`, so future tokens are actually blocked. If I had masked after softmax the upper triangle would not be zero. I used that print as the check, not the assistant's description of the plot.
