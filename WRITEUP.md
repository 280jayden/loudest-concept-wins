# The loudest concept wins

## The question

When an activation contains two concepts in a known proportion, does the model's
description of that activation name both, or only the dominant one?

Introspection methods take an activation, inject it into a model, and read out a natural
language description of it. SelfIE and Patchscopes established the approach. Pepper et
al. (arXiv:2602.10352) train small adapters on top of it and report 94% recall@1 against
SAE feature labels.

That evaluation is single-concept. A vector containing one known feature goes in, and the
description is checked for that feature. Recall@1 measures whether the description found
the thing that was put there. It says nothing about whether the description covers the
rest of what was present, because in the evaluation there is no rest.

Pepper et al. train on single-feature decoder directions and then apply the adapters to
residual stream activations. Their own comment on the transfer, in the discussion section
on going from monosemantic to polysemantic, is that "that this works at all is somewhat
surprising." The evidence for the transfer is that the description recovers a feature they
were checking for. A residual stream activation at that position holds many features, and
recall@1 is scored against a single labelled one. A description covering only that feature
scores the same as a description covering everything in the activation. So the transfer
leaves open how much of an activation a description covers.

I care about this because of what introspection is already being used for. SelfIE reports
that it "reveals LLM internal reasoning in cases such as making ethical decisions,
internalizing prompt injection, and recalling harmful knowledge." The natural language
autoencoder has gone further: Fraser-Taliente et al. applied it during Claude Opus 4.6's
pre-deployment audit, where it surfaced evaluation awareness the model held without
stating it, and where auditing agents using it found the root cause of a model's
misalignment without access to the training data that caused it.

A real activation holds many features at once, so a short description that names one of
them is already leaving the rest out. The audit description named evaluation awareness,
and whatever else was in that activation went unreported. So a success shows one concept can be recovered, and leaves open how much of
the activation the description accounted for.

This matters more as models get more capable. If something concerning were held at a small
share at the position an auditor reads, the description would come back the same as it
does when nothing is there.

## Result in one line

The description names one concept, and it is whichever concept dominates the activation.
When the second concept holds a quarter of the activation or less, the Llama adapters
name it 0 times in 240, which is what they return when the concept is not in the
activation at all.

---

## Setup

### Models and readouts

- Llama-3.1-8B-Instruct with the Goodfire SAE at layer 19 (65,536 latents)
- Gemma-3-12B-IT with Gemma Scope 2 at layer 32 (16,384 latents)

### Methods

Three, used as released, no retraining:

| method | trainable params | model |
|---|---|---|
| scalar affine adapter, Pepper et al. | 4,097 | Llama |
| scalar affine + rank-16, Pepper et al. | 135,169 | Llama |
| natural language autoencoder verbaliser, Fraser-Taliente et al. | full model | Gemma |

The two Llama adapters differ by 33x in parameter count, which gives a capacity
comparison inside one model. The Gemma setup differs from the Llama one in three ways: a
different base model, a different SAE for building the vectors and scoring the
descriptions, and a method trained on a different objective. The natural language
autoencoder's verbaliser is trained jointly with a reconstructor by RL, so the training
signal rewards descriptions that retain enough information to rebuild the activation. If
any of the three should carry two concepts, it is that one.

### Building the activations

There is no dataset of activations with known concept proportions, so I built one:

```
v(alpha) = normalize(alpha * d_A + (1 - alpha) * d_B)
```

over unit-normalised SAE decoder rows. Each concept's share of the activation is set by
alpha. The outer normalisation holds the injected norm constant across the sweep, so a
change in detection across alpha is not a change in injection strength.

For Llama the vector is injected at the reserved-token embedding in the prompt
`What is the meaning of "X"?`, at magnitude 1.0. That is the unit norm the adapters were
trained on and the default in the released evaluation code (`scale_values` defaults to
`[1.0]`). For Gemma the construction adds `b_dec`, because bare decoder directions are
off-distribution at layer 32, and scales by s=30000. The natural language autoencoder
prescribes its own injection norm.

Because the Gemma construction adds a bias term, alpha does not by itself determine the
ratio the two concepts end up at. I encoded the composed vectors back through the SAE and
measured the second concept's share of the two activations. Against nominal shares of
50%, 25% and 10%, the ten pairs come out at 42% to 55%, 17% to 28%, and 4% to 11%. Each
range contains its target and no two ranges overlap, so the three conditions are the
shares they claim to be. s=30000 is the smallest value in the sweep that achieved this
while keeping both concepts above the SAE's detection floor.

### Gates on what gets used

Three conditions, applied before any pair enters the sweep:

1. **Each concept must be described correctly when injected alone.** All 10 Llama
   concepts pass at hit rate 1.0. If a concept cannot be recovered from a pure vector,
   its absence from a mixture says nothing.
2. **Decoder cosine below 0.1 between the two concepts in a pair.** Two concepts pointing
   in similar directions would make "the description named A" and "the description named
   B" partly the same event.
3. **The second concept's latent must fire in the composed vector at every ratio.** This
   confirms the concept is present in the activation the method reads, so a miss is
   a property of the method and not of the construction.

All 10 Llama concepts sit in the validation split of the labelled feature set, so the
adapters never saw them in training.

Gate 2 and gate 3 removed pairs. On Gemma, 3 of 5 candidate pairs passed:
`travel x hatred_discrimination` failed the cosine gate at 0.178, and
`travel x malware` failed because the second latent read 0 at one ratio.

### Scoring

Generation scoring, following Pepper et al. The description goes back into the model, ten
short conversations are generated from it, those are run forward and SAE-encoded, and the
target latent is checked for activation above 0.3. The descriptions are never read as
text and never keyword-matched.

I used this because a keyword check is not reliable here. One example was that the concept
`programming` cleared a keyword gate 3/3 on a description about Vedic
astrology, matching only on the word "code". Under behavioural scoring the same
description scored 0.20 and failed. That pair had been detected 6/6 at the 25% ratio, so
the keyword version of the gate was inflating the result in the direction I did not want.

### Sample

12 pairs x 20 descriptions = 240 per ratio per Llama adapter. 10 pairs x 20 = 200 per
ratio on Gemma. Ratios are 100%, 75%, 50%, 25% and 10% of the activation held by the
second concept, plus a 0% control.

---

## Does the metric work

Three checks:

| check | result |
|---|---|
| concept B scored on vectors holding only concept A | 0/240 |
| concept B scored on vectors holding only concept B | 184/240 (76.7%) |
| a concept scored against random directions | 0/144 `<<RERUN AT 240 BEFORE SUBMITTING>>` |

The first two are the 0% and 100% rows of the main result table.

The metric almost never claims a concept is present when it is not. It misses a real
concept about a quarter of the time, which is consistent with the roughly 70% generation
scoring number Pepper et al. report at 70B.

Both of those matter for reading the main result. The false-positive rate near zero means
the 0/240 readings are trustworthy. The 77% recovery on a pure concept means the metric
is biased against finding concepts, so it cannot manufacture the collapse. The result is
a floor on how bad the omission is.

---

## Main result

### Scalar affine adapter, 4,097 params, 240 descriptions per ratio

| share held by concept B | A named | B named | both | neither |
|---|---|---|---|---|
| 100% | 2 (0.8%) | 184 (76.7%) | 0 | 54 |
| 75% | 5 (2.1%) | 182 (75.8%) | 0 | 53 |
| 50% | 189 (78.8%) | 37 (15.4%) | 15 | 29 |
| 25% | 233 (97.1%) | **0 (0.0%)** | 0 | 7 |
| 10% | 229 (95.4%) | **0 (0.0%)** | 0 | 11 |
| 0% (control) | 231 (96.2%) | **0 (0.0%)** | 0 | 9 |

### Scalar affine + rank-16, 135,169 params, same pairs, same sample

| share held by concept B | A named | B named | both | neither |
|---|---|---|---|---|
| 100% | 0 (0.0%) | 217 (90.4%) | 0 | 23 |
| 75% | 1 (0.4%) | 223 (92.9%) | 1 | 17 |
| 50% | 136 (56.7%) | 89 (37.1%) | 7 | 22 |
| 25% | 230 (95.8%) | 1 (0.4%) | 1 | 10 |
| 10% | 232 (96.7%) | 0 (0.0%) | 0 | 8 |
| 0% (control) | 236 (98.3%) | 0 (0.0%) | 0 | 4 |

Three things to take from these.

**The 25% row equals the control row.** At a 25% share the second concept is named 0
times in 240. With the concept removed from the activation entirely, it is named 0 times
in 240. The method returns the same answer for a concept holding a quarter of the
activation and a concept that was never there.

**The activations are being read correctly.** At a 25% share the adapter names the
dominant concept 233/240, which is a higher rate than it recovers a pure single-concept
vector (184/240). The descriptions are confident, specific, and about one thing. This is
not the method failing on a strange input.

**Capacity is not the constraint.** The rank-16 adapter is clearly better wherever the
concept dominates: 217 against 184 on a pure vector, 89 against 37 at parity. It is
equally blind below a 25% share: 1/240 and 0/240. A 33x parameter increase buys a better
reading of the dominant concept and nothing at all for the subordinate one.

### What the descriptions named

The "both" column never exceeds 15/240 (6.3%) anywhere on either curve, and that maximum
is at the 50/50 mixture. Across the whole sweep, the description is about one concept.

The role of each concept does not matter much. When the everyday concept is the one held
at a 25% share, it is named 5/240. When the safety-relevant concept is held at 25%, it is
named 0/240. That is a difference of one event on counts this low, so it is not evidence
of a direction. What it does show is no sign of role-dependent suppression.

**Every pair collapses at the same place.** The 25% number is not an average over pairs
that fail at different points. Broken out per pair, the scalar affine adapter names the
second concept 0/20 on all 12 pairs at a 25% share and 0/20 on all 12 at 10%. The rank-16
adapter is 0/20 on 11 of 12 pairs at 25%, with a single pair at 1/20, and 0/20 on all 12
at 10%. Parity is where the pairs disagree: per-pair counts there run 0, 0, 0, 0, 0, 0, 0,
1, 1, 2, 3, 10 and 20 out of 20. So the threshold is sharp and shared, and the variation
sits above it.

---

## Cross-architecture: the natural language autoencoder

The Gemma verbaliser, 10 pairs x 20 = 200 per ratio:

| share held by concept B | A named | B named | both | neither |
|---|---|---|---|---|
| 100% | 35 (17.5%) | 200 (100%) | 35 | 0 |
| 75% | 106 (53%) | 200 (100%) | 106 | 0 |
| 50% | 132 (66%) | 160 (80%) | 92 | 0 |
| 25% | 200 (100%) | 55 (27.5%) | 55 | 0 |
| 10% | 200 (100%) | 9 (4.5%) | 9 | 0 |

This method is better than the Llama adapters at every point of comparison. It
recovers a pure concept 200/200. It names both concepts 92/200 (46%) at parity, against
the adapter's 15/240 (6.3%). It never fails to name something: the "neither" column is 0
at every ratio.

It also has a false-positive rate the Llama adapters do not, so the numbers need a floor
before they can be read. Two independent estimates:

- **17.5%**: the first concept is named on 35/200 pure-second-concept rows, where it is
  genuinely absent from the activation.
- **10.9%**: 219/2000 for features belonging to neither concept in the pair.

Taking the more conservative 17.5%:

- At a **25% share**, 55/200 against a 35/200 floor. z = 2.39, p = 0.017. Real detection.
- At a **10% share**, 9/200 = 4.5%, below the floor. No detection.

So the better method holds the subordinate concept one threshold further down than
the adapters do, and then loses it. The threshold moved. It did not go away.

The same role asymmetry appears here and is much weaker: the everyday concept at a 25%
share is named 53%, the safety-relevant concept at a 25% share 27.5%. Both clear the
floor, unlike on Llama where both are at zero.

---

## What I ruled out

I tested seven alternative explanations for the collapse. All of them fail.

**Injection magnitude.** The released evaluation config uses
`scale_values = [0.5, 0.8, 1.3, 2.1, 3.4, 5.5]`, multiples of the trained unit norm.
Pure-concept recovery across those six is 19%, 75%, 81%, 97%, 97%, 67%, an inverted U
with a wide plateau. The collapse holds at all six. Restricting to the two best
magnitudes (2.1x and 3.4x) gives 97.2% / 95.8% / 30.6% at 100/75/50, and 0/72 at each of
25%, 10% and control. The strongest version of the method shows the same cliff.

**Layer.** 9 layers (4, 8, 12, 16, 19, 22, 25, 28, 31), 864 descriptions. The second
concept is named 1/864.

**Prompting.** Asking explicitly for multiple concepts helps at parity and not below:
both named goes from 7/90 to 30/96 at 50/50, and from 0/90 to 1/96 at a 25% share.

**Output length.** Raising the token budget from 30 to 100 leaves the median description
at 6 words and moves both-named from 12/216 to 9/216. The descriptions are short because
that is the trained output format, and giving them room changes nothing.

**Adapter capacity.** Covered above. 33x more parameters, same blindness below 25%.

**Refusal training suppressing the concerning concept.** At parity, neutral second
concepts are named 27/54 (50%) and concerning second concepts 23/36 (64%). The difference
runs the wrong way for this explanation.

**Semantic relatedness and co-occurrence.** The idea that concepts closer to the dominant
concept survive did not replicate on Llama. Co-occurrence frequency has a direct
counterexample: `programming x malware` survived at the lowest co-occurrence in the set
(0.0074), and `legal x ransomware` went silent at the highest (0.0594).

There is also a check on whether the information is present in the activation at all,
separately from whether the description mentions it. Concept B's SAE activation over the
strongest unrelated latent is 5.17x at parity, 1.50x at a 25% share, and 0.28x at 10%. On
a pure first-concept control, concept B sits at rank ~1607 with activation 0.000. So at
parity the concept is clearly there in the activation and clearly missing from the
description. Below parity the margin is too small for this to say anything, and I do not
claim it does.

**How readable the concept is does not predict whether it gets described.** Correlating
each pair's concept-B activation at parity against how often that pair's descriptions
named it gives r = -0.33 across the 12 pairs, which on n=12 is nothing, and what sign
there is runs the wrong way. The individual pairs make the point more plainly. In two
pairs concept B is the single strongest latent in the composed activation, ahead of
concept A, and those two pairs are described 2/20 and 0/20. The two pairs described most
often at parity, 20/20 and 10/20, have among the lowest concept-B activations in the set.
Whatever decides which concept gets named is not how strongly the concept is represented.

---

## What I do not know

**At a 50/50 mixture, the first concept is named 189/240 and the second 37/240.** Neither
concept dominates by construction. Both are gated to be present. The decoder cosine is
below 0.1, so they are not competing for the same direction. Something decides which of
two equal concepts gets described, and I have no account of it. Swapping which concept
plays which role does not remove it, and none of the seven explanations above predicts
it.

This is the loose end I would pull first with more time. The obvious candidates are
properties of the concepts rather than the mixture: how frequent the feature is in
training text, how early in the layer stack it becomes linearly readable, how many tokens
its natural description takes.

**No mechanism for the collapse.** I have measured the threshold on three methods
and ruled out seven explanations. I cannot say what the adapter is doing that produces
it.

---

## Limitations

**The activations are constructed.** Two decoder directions mixed at a set ratio is not
what a residual stream looks like at a real token position, which carries many features
with a long tail. The constructed setup is what makes the ratio exact, and that is the
tradeoff. I ran a smaller pass on 57 activations from real text and the pattern goes the
same way, with detection falling from 42% for the strongest latents to 17% for the
weakest. That pass has no false-positive control, so I do not treat it as evidence, and I
would build the control before relying on it.

**Small concept set.** 12 pairs on Llama and 10 on Gemma, from one SAE per model. The
concepts are all ones that pass a clean-description gate, which selects for features that
are easy to describe. Whether the threshold sits at 25% for a harder feature set is
untested.

**I tested the verbaliser and not the reconstructor.** The natural language autoencoder
has a decoder that rebuilds the activation from the description. If a description that
never mentions a concept still lets the reconstructor recover it, the information is in
the text in a form a human reader would not see, and the conclusion would need
qualifying. This is one run away and I did not get to it.

**One threshold, three methods.** I can say the threshold moves with method
quality because it moved once, between the Llama adapters and the natural language
autoencoder. Two points is a direction and not a curve.

**The metric misses about a quarter of real concepts.** This makes every detection number
here a floor. It cannot create the collapse, since a metric that under-detects cannot
turn a present concept into a 0/240 that matches its own control, but it does mean the
absolute rates are understated.

---

## Things I caught

Listing these because they changed results.

**A scaling bug that inverted the ratios on Gemma.** Mixtures were scaled by each
feature's median activation, which ranged from 604 to 1957 across features. That swamped
the mixing coefficient. Nominal 25% mixtures were running anywhere from 0.20 to 1.63, so
in some pairs the concept I was calling subordinate was the stronger one. Everything on
Gemma was rebuilt on unit directions and verified against the SAE encoder.

**A keyword gate passing a description that had nothing to do with the concept.** Covered
above: `programming` matched on "code" in a passage about Vedic astrology. Moving to
behavioural scoring removed that pair and made the result stricter.

**Two control files, one of them dead.** One control run described the first concept
0/90, where live conditions ran 83-92%. The injection had not taken. Both files were on
disk with similar names, and using the wrong one would have made the control look like a
detection failure.

**A circular experiment, killed before it finished.** I set up a test for whether the
omitted concept is present in the model's states while it writes the description. That
reads an SAE over text the model just generated, so it mostly measures what the text
says. The generated positions also attend back to the injection site. It could not have
answered the question.

**A number that would not reconstruct.** An earlier summary of the monitoring experiment
recorded 0/84 and 10/84. Neither denominator matches the stored data, which is 8 passages
x 12 descriptions = 96. Recomputed from the file as 1/96 and 11/96.

**An overstated claim I withdrew.** "Concept B is rank 2 of 65,536 at a 25% share" is
true and misleading. Its margin over the strongest unrelated latent is 1.01x. Rank is
doing no work at that margin. The claim is now scoped to parity, where the margin is
5.17x, and reported as a margin.

**A metric recommendation that was backwards.** I had planned to report recall over both
concepts present in the activation. Computing it shows recall *rising* as the effect gets
worse: 43%, 46%, 48% at 50/25/10. The dominant concept is named more often as the
subordinate one fades, so the aggregate improves while the thing being measured
deteriorates. The reported metric is detection of the subordinate concept specifically.

---

## Why this matters for safety

The usable claim is narrow and I want to state it without stretching it.

These methods are being proposed as a way to look inside a model and see what it is
representing. The value of that for auditing depends on the method reporting things that
are present but not dominant, because a concerning representation during benign-looking
output is not going to be the largest component of the activation.

What I measured is that there is a share below which two of these methods report a
concept at exactly the rate they report a concept that is absent. For the Llama adapters
that share is 25%. For a much better method it is somewhere between 10% and 25%. A
description that does not mention a concept is not evidence that the concept is not
there, and the amount of evidence it provides depends on a threshold that nobody has been
reporting.

I am not claiming these methods have failed at an audit. The natural language autoencoder
is genuinely good, and it found a subordinate concept where the adapters found nothing. I
am claiming the threshold exists, that it is measurable, and that it should be reported
alongside recall@1, because recall@1 on single-concept vectors cannot see it.

---

## What I would do next

1. **Run the reconstructor.** Take descriptions that never mention the second concept and
   check whether the natural language autoencoder's decoder can still rebuild it. This
   separates "the method lost the information" from "the method has the
   information and does not say it", and those have different fixes.
2. **Build the false-positive control for real activations**, then take the real-text
   result seriously. Matched activations where the concept is genuinely absent, scored
   the same way.
3. **More methods on the same axis.** Two points establish that the threshold moves.
   Placing four or five methods on the same measurement would show whether it is
   tracking method quality smoothly or sitting at a fixed place for a whole class of
   method.
4. **Attack the 50/50 asymmetry.** It is the one result with no explanation, and it is
   about which concept the model chooses to talk about, which is more interesting than
   the threshold.

---

## Sources

- SelfIE, Chen, Vondrick & Mao, arXiv:2403.10949
- Patchscopes, Ghandeharioun et al., arXiv:2401.06102
- Learning Self-Interpretation from Interpretability Artifacts, Pepper et al.,
  arXiv:2602.10352. The adapters, the injection scales, and the scoring procedure.
- Natural Language Autoencoders, Fraser-Taliente, Kantamneni, Ong, Mossing et al.,
  Transformer Circuits 2026. Model `kitft/nla-gemma3-12b-L32-av`.

Code and all result files: github.com/280jayden/loudest-concept-wins
