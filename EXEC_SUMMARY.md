# Executive summary

**The problem.** Introspection methods let a model describe its own activations in
language. They're evaluated one concept at a time: inject an activation containing a
known concept, check whether the description recovers it. Pepper et al. report 94%
recall@1. But they train adapters on single-concept vectors and then apply them to
polysemantic residual stream activations, calling that transfer "somewhat surprising"
themselves. Nobody has measured whether a description covers everything that's present.
That matters because the concept an auditor is looking for is rarely the loudest thing
in an activation.

**Takeaways.**

- The description covers one concept, and it's whichever one dominates.
- At a 25% share the second concept is named **0 times in 240**, identical to what the
  method returns when the concept is absent entirely.
- Both concepts are named together **at most 15 times in 240** anywhere on the curve.
- An adapter with 33x more parameters is clearly better wherever the concept dominates,
  and still returns 0-1/240 below a 25% share.
- The natural language autoencoder's verbalizer is a much better interpreter and still
  fails, one threshold lower.
- I don't know why it happens. Capacity, prompting, output length, layer, and which
  concept sits in which role are all ruled out.

---

### Detection against concept share

![fig1](figures/fig1_main_curve.png)

*Both Llama adapters. Detection of the second concept against its share of the
activation, 240 descriptions per point. Grey square is the control, the same
measurement with the concept absent.*

I built activations as `normalize(a*d_A + (1-a)*d_B)` over unit-normalized SAE decoder
rows, so each concept's share is exact, and measured how often the description names
the second concept. Detection holds flat while a concept has three-quarters of the
activation or more, falls at parity, and reaches zero by a 25% share. The control is
also zero. Both adapters land in the same place below 25% despite a 33x parameter
difference.

---

### What the description actually names

![fig2](figures/fig2_what_it_names.png)

*Scalar-affine adapter. What each description named, as a share of all 240 at that
ratio.*

At a 25% share the model names the dominant concept in 233 of 240 descriptions, more
reliably than it describes a pure vector. So the composed activations are being read
correctly; the descriptions are confident and specific and cover one thing. At a 50/50
mixture it names at least one of the two concepts 211 times in 240 and both only 15
times. Which one it picks is lopsided in a way I can't explain: concept A 189 times
against concept B's 37.

---

### Cross-architecture

![fig3](figures/fig3_cross_architecture.png)

*Llama's trained adapter against the Gemma natural language autoencoder's verbalizer,
each with its own false-positive floor. 240 and 200 descriptions per point.*

The verbalizer from a natural language autoencoder, trained on a different model with
an objective that rewards keeping content in the description, is better than either
adapter at everything: 200/200 on a pure concept, and both concepts named 92 times in
200 at parity. It still finds the minority concept at a 25% share where the adapters
are at zero. But it loses it by 10%. So it's clearly the better tool and it still has
the same flaw, further down.

---

**Limitations.** All the main results use constructed activations, which aren't what
models produce on their own. Twelve concept pairs on Llama and ten on Gemma, from one
SAE each. I tested the verbalizer and not the reconstructor, so I don't know whether a
description that never mentions a concept can still be used to rebuild it.
