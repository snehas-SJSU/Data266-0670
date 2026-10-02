# AI-Use Appendix — Sneha Singh

SID4 = 0670 | SEED = 670 | SLICE = 670 | HP_ID = 4 | CLS_A = 0 | CLS_B = 5

1. Which parts did you use an assistant for, and which did you write yourself?

I set Step 0 the same way as Homework 1 (SID4 670, seed 670). I decided the split: 500 labeled images shared by A, B, and C, all 100,000 unlabeled images for rotation, 20,000 for SimCLR, ResNet-18 with no pretrained weights, temperature 0.2, and a frozen encoder during the linear probes. I ran `stl10_ssl.ipynb` on Colab GPU and copied the printed accuracies into METRICS.md and the report.

I used an assistant for the notebook skeleton: the NT-Xent batch (the other view is the positive, the diagonal is masked), the frozen linear probe, and the comparison write-up after the Colab numbers were in.

2. Give one specific thing it produced that was wrong. Paste the wrong output.

The first training loop treated every 2-tuple as (image, label). SimCLR also returns two tensors, so that loop would have run cross-entropy on the second view.

3. How did you find out? What did the failure look like?

I read the loop before Run all. Classifier batches and two-view batches both have length 2, so the branch could not tell them apart.

4. What did you change, and why does your version work?

The loop takes `contrastive=True` only for the SimCLR loader. Classifier steps still use cross-entropy. The linear probe sets the encoder to eval and blocks its gradients, so only the linear layer updates. The Colab run then printed A 0.3149, B 0.3957, C 0.4467.
