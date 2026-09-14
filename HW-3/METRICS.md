# METRICS — DATA 266 HW3

SID4 = 0670 | SEED = 670 | SLICE = 670 | HP_ID = 4 | CLS_A = 0 | CLS_B = 5  
HP_ID is reported only. Two trained models: unmasked attention and causal-masked attention.

From `attention.ipynb` on kernel Python (HW1). LLM: Flan-T5-small + LangChain PromptTemplate.

## Prompt engineering (2 examples each)

Correct math answer is 12. Bank sense should be river / side of a river. Youngest is Cara.

| Technique | Ex1 task | Ex1 output | Ex1 ok? | Ex2 task | Ex2 output | Ex2 ok? | vs simpler prompt |
|-----------|----------|------------|---------|----------|------------|---------|-------------------|
| Zero-shot | 15% of 80 | 98.6 | no | river bank sense | side of a river | yes | baseline |
| Few-shot | 15% of 80 | 59 | no | river bank sense | river | yes | still wrong math; bank stays right and shorter |
| Chain-of-Thought | 15% of 80 | 80% | no | Alice/Bob/Cara youngest | Alice > Bob > Cara | partial | copied numbers; order ok but no "Cara" |
| Zero-shot CoT | 15% of 80 | answer: 15 | no | youngest | Bob | no | rambles more than zero-shot |
| Meta-prompting | write prompt then 15% of 80 | gen=`80%`; then population sentence | no | write prompt then bank | gen=`bank`; OUT=`bank` | no | never wrote a prompt |
| Tree of Thoughts | 3 ways then pick, 15% of 80 | -2.5 | no | 3 meanings of telescope sentence | echoed the sentence | no | no tree, just junk / copy |

## Attention

Dataset: assignment paragraph, word tokens, seed 670, next-token CE, d=32, 400 epochs.

| Model | mask | final loss (ep 400) | check | heatmap |
|-------|------|---------------------|-------|---------|
| unmasked | none | 0.3316 | row sums = 1 | `heatmap_unmasked.png` (from notebook) |
| causal | lower tri before softmax | 0.2093 | upper triangle max = 0.0 | `heatmap_causal.png` (from notebook) |

n tokens = 46, vocab = 40.
