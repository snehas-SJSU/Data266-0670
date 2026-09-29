# AI-Use Appendix — Sneha Singh

SID4 = 0670 | SEED = 670 | SLICE = 670 | HP_ID = 4 | CLS_A = 0 | CLS_B = 5

1. Which parts did you use an assistant for, and which did you write yourself?

I set SID4 / SEED the same way as HW1. I picked flan-t5-small, the prompt (`summarize the dialogue:` plus the dialogue, target = summary), and the two held-out dialogues (`dev_56`, `dev_357`). I ran `lora_dialogsum.ipynb` on kernel Python (HW1), checked the loss plot, and wrote the before/after note myself.

- Step 0: SID4 670, seeds, HP_ID reported only
- Data: 256-row subset, 2 processed samples
- Baseline: same 2 dialogues before LoRA
- LoRA: r=4 and r=16, alpha 16, dropout 0.05, on `q` and `v`
- Train: 2 epochs, batch 4, lr 1e-3, adapters saved
- Compare: same 2 dialogues after, plus the rank note

I used an assistant for the PEFT attach (`LoraConfig`, `TaskType.SEQ_2_SEQ_LM`, `get_peft_model`) and for masking pad tokens as `-100` in the summary labels.

2. Give one specific thing it produced that was wrong. Paste the wrong output.

The print for processed sample 2 cut the dialogue at 700 characters, so the input looked done when it was not:

```
#Person2#: Ah! I get your point. We have just what y
TARGET:
#Person1# is looking for an unfurnished apartment with a lower cost near downtown. #Person2# recommends Jinyuan apartments. #Person1# asks for its address and will see it tomorrow afternoon.
```

The target talks about Jinyuan apartments and an address. Those words were not in the printed input.

3. How did you find out? What did the failure look like?

I read sample 2 next to its target. The summary mentions Jinyuan and a viewing tomorrow. The printed dialogue stopped mid-sentence at "what y", so the pair did not match.

4. What did you change, and why does your version work?

I took off the 700-character cut and print the full dialogue. The notebook sample now includes the rest of the call (Jinyuan apartments, 19 Lingual Road, viewing tomorrow), and that matches the target.
