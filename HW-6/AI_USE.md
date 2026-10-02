# AI-Use Appendix — Sneha Singh

SID4 = 0670 | SEED = 670 | SLICE = 670 | HP_ID = 4 | CLS_A = 0 | CLS_B = 5

1. Which parts did you use an assistant for, and which did you write yourself?

I set Step 0 the same way as HW1. I picked STL-10, ResNet-18 with no pretrained weights, the shared 500-image list, and the three models (supervised, rotation, SimCLR). I ran `stl10_ssl.ipynb` on Colab GPU, checked the test accuracies, and wrote the comparison in the report myself.

- Step 0: SID4 670, seeds, HP_ID reported only
- Data: 500 labeled images, same list for A, B, and C
- A: end-to-end, 12 epochs
- B: rotation on all unlabeled images, then a frozen linear probe
- C: SimCLR on 20,000 images, tau 0.2, then a frozen linear probe
- Compare: test accuracy and the same 3 neighbor queries

I used an assistant for the contrastive loss (the other view is the positive, diagonal masked) and for keeping the encoder frozen during the linear probe.

2. Give one specific thing it produced that was wrong. Paste the wrong output.

The first training loop treated every 2-tuple as (image, label). SimCLR also returns two tensors, so that loop would have run cross-entropy on the second view.

3. How did you find out? What did the failure look like?

I read the loop before Run all. Classifier batches and two-view batches both have length 2, so the branch could not tell them apart.

4. What did you change, and why does your version work?

The loop takes `contrastive=True` only for the SimCLR loader. Classifier steps still use cross-entropy. The linear probe sets the encoder to eval and blocks its gradients, so only the linear layer updates. The Colab run then printed A 0.3426, B 0.4260, C 0.4980.
