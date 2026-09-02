# Overnight run: Llama Scope generalization (Li explainer, Pepper adapters)

Run date 2026-09-02, pod `foyt0xzx69q3l7` (H100 80GB), pipeline `code/43_li_pipeline.py`,
analysis `code/45_li_analyze.py`, raw pickles in `results/RESULTS/li/`. Every number below
is reproduced by running the analysis script on those pickles.

## The one-paragraph result

The cliff replicates on a second SAE and on a full-model, label-supervised explainer. On
Llama Scope L19 (131k latents) with the same composed-activation protocol, Li et al.'s
self-explainer names the minority concept in 64% of descriptions at parity and 6% at a 25%
share, against a 0% control. Pepper's scalar-affine adapter on the same pairs and scorer
goes 48% to 14% to 0.4%. Excluding two pairs whose concepts co-occur in ordinary text (self-harm
and firearms) or where one concept dominates at every share (firearms over prescription drugs,
adapter only), the 25% rate is 0 of 200 for both methods. The one thing that does change with
method is what happens at parity: the adapter still writes one concept per description (both
named 0.8%), while Li's explainer writes a conjunction for some pairs ("terms related to baking
and explosive devices", 20/20 in both directions for that pair) and one concept for others
(baking and fraud: fraud is never mentioned, either direction). Both-named at parity is 29% for
Li overall, but it is a property of particular pairs, not a uniform capacity.

## What ran

| arm | method | host | injected magnitude | status |
|---|---|---|---|---|
| Li | `Transluce/features_explain_llama3.1_8b_llama3.1_8b`, full fine-tune, self-explainer | Llama-3.1-8B base | mean raw decoder norm of the pair (1.44 to 1.88) | complete |
| adapter | Pepper `llamascope-sae-scalar-affine` (4,097 params) | Llama-3.1-8B-Instruct | unit norm | complete |
| adapter64 | Pepper `llamascope-sae-sa-lr64` (528,385 params) | Llama-3.1-8B-Instruct | unit norm | complete |

One SAE (Llama Scope L19R-32x, JumpReLU, threshold 0.484), one scorer for every arm:
Instruct writes ten short conversations from the description, base Llama runs them forward,
Llama Scope encodes layer 19, the target latent counts as fired if above zero on any post-BOS
token, hit at >= 0.3 of ten. Shares 100/75/50/25/10/0 (B's share), 20 sampled descriptions per
cell, one greedy per cell (Li only, their protocol), list prompt at 50% and 25% with 8 draws.

Pre-flight: real layer-19 residual norm 17.87 (SAE expects 17.1), L0 98.

## Gates

Gate 1 (describable alone, best of 3 sampled descriptions, hit >= 0.8) was run on all 52
Neuronpedia candidates (4 per family, 13 families):

| | pass |
|---|---|
| Li explainer | 37 / 52 |
| scalar-affine adapter | 25 / 52 |
| both | 24 / 52 |
| rank-64 adapter (on the 9 concepts used) | 9 / 9 |

Two things worth putting in the paper. First, the adapter fails on many features Li describes
correctly (ceasefires, physical violence, seasoning, political extremism): the adapter's
descriptions are vaguer, not wrong. Second, some features fail under both methods even though
both descriptions read as correct. Feature 44109 ("food preparation and cooking methods") gets
an accurate description from both methods and scores 0.0 at every magnitude from 0.5x to 3x.
The scorer requires the latent to fire on Instruct-written conversations about the description,
and some Llama Scope latents fire only in narrow contexts. This is a scorer-sensitivity
limitation, not a method failure, and it is why the anchor set (cooking, baking, spices) came
out thin: of 12 anchor candidates, 4 pass under both methods.

Gates 2 and 3 (|cos| < 0.1 on unit decoder columns; minority latent fires under the JumpReLU
encoder at the injected magnitude) were recomputed in the pipeline on every ordered pair of
both-passing concepts from different families: 450 directed pairs pass at 75/50/25%, 225
unordered pairs valid in both directions. Gate 3 at 10% is rarely satisfied at the trained
magnitude (about 1.5 in residual units), so the 10% row exists only for pairs where it passes.

## Pairs

Six unordered pairs, each run in both directions (A majority, B minority, then swapped):
four anchor x concerning, two concerning x concerning. Selection preferred pairs with a 10%
row and higher minority activation at 25%, capped at 2 per family combination and 3 per family.

| A | B | families | cos | magnitude | 10% row |
|---|---|---|---|---|---|
| 51141 baking and baked goods | 125117 financial fraud and scams | baking x fraud | +0.003 | 1.44 | no |
| 51141 baking and baked goods | 59899 explosive devices | baking x bombs | +0.039 | 1.52 | no |
| 39388 baking and baked goods | 59899 explosive devices | baking x bombs | +0.003 | 1.68 | no |
| 59899 explosive devices | 73564 cooking herbs and seasonings | bombs x spices | -0.005 | 1.61 | no |
| 35308 firearms and military weapons | 86616 prescription drug terms | weapons x opioids | +0.059 | 1.88 | yes (this direction) |
| 6595 self-harm and suicidal acts | 76330 firearms | selfharm x weapons | +0.063 | 1.55 | yes (this direction) |

Nine unique latents. The self-harm x firearms pair passes the decoder-cosine gate but its
concepts co-occur in ordinary text (suicide by firearm), and it behaves differently from every
other pair below. I would report it and flag it, or drop it in the paper and say why.

## Main result, sampled descriptions (240 per share, 40 at 10%)

B named (%) by B's share. "both" is both concepts named in one description.

| arm | 100% | 75% | 50% | 25% | 10% | 0% control | both @ 50% |
|---|---|---|---|---|---|---|---|
| Li explainer | 98.8 | 99.2 | 64.2 | 6.2 | 2.5 | 0.0 | 28.8 |
| scalar-affine adapter | 94.6 | 83.3 | 47.9 | 14.2 | 0.0 | 0.4 | 0.8 |

Full four-way split:

| arm | share | A named | B named | both | neither |
|---|---|---|---|---|---|
| Li | 100% | 0.8 | 98.8 | 0.8 | 1.2 |
| Li | 75% | 6.2 | 99.2 | 6.2 | 0.8 |
| Li | 50% | 62.5 | 64.2 | 28.8 | 2.1 |
| Li | 25% | 99.6 | 6.2 | 6.2 | 0.4 |
| Li | 10% | 100.0 | 2.5 | 2.5 | 0.0 |
| Li | 0% | 100.0 | 0.0 | 0.0 | 0.0 |
| adapter | 100% | 0.4 | 94.6 | 0.4 | 5.4 |
| adapter | 75% | 14.6 | 83.3 | 2.1 | 4.2 |
| adapter | 50% | 46.2 | 47.9 | 0.8 | 6.7 |
| adapter | 25% | 82.9 | 14.2 | 0.8 | 3.8 |
| adapter | 10% | 100.0 | 0.0 | 0.0 | 0.0 |
| adapter | 0% | 97.1 | 0.4 | 0.4 | 2.9 |

Where the 25% hits come from:

| arm | pair | B named at 25% (of 20) |
|---|---|---|
| Li | self-harm (75%) x firearms (25%) | 13 |
| Li | firearms (75%) x self-harm (25%) | 2 |
| Li | every other cell (10 cells) | 0 |
| adapter | self-harm (75%) x firearms (25%) | 17 |
| adapter | prescription drugs (75%) x firearms (25%) | 17 |
| adapter | every other cell (10 cells) | 0 |

Excluding the top two pairs per arm: Li 0/200, adapter 0/200 at 25%.

Greedy decoding (Li only, one per cell): 50% A 8/12, B 7/12, both 3/12; 25% B 0/12; 10% B 0/2;
control B 0/12.

List prompt ("list every distinct concept", 8 draws per pair, a concept counts if any item hits):

| arm | share | A | B | both | n |
|---|---|---|---|---|---|
| Li | 50% | 58.3 | 64.6 | 22.9 | 96 |
| Li | 25% | 100.0 | 10.4 | 10.4 | 96 |
| adapter | 50% | 51.0 | 53.1 | 6.2 | 96 |
| adapter | 25% | 86.5 | 17.7 | 7.3 | 96 |

The list prompt does not recover the minority below parity for either method beyond the two
flagged pairs; at parity it adds a little for the adapter (0.8% to 6.2% both).

Random-direction floor (20 directions x 9 latents = 180 per method): Li 0/180, adapter 1/180, rank-64 1/180.

## Per pair at parity, both directions

A named / B named / both, out of 20. Direction 1 then direction 2 (A and B swapped).

| unordered pair | Li dir 1 | Li dir 2 | adapter dir 1 | adapter dir 2 |
|---|---|---|---|---|
| baking 51141 / fraud 125117 | 20/0/0 | 0/20/0 | 20/0/0 | 0/19/0 |
| baking 51141 / bombs 59899 | 3/20/3 | 19/2/2 | 18/0/0 | 2/18/1 |
| baking 39388 / bombs 59899 | 20/20/20 | 20/20/20 | 11/7/1 | 3/14/0 |
| bombs 59899 / spices 73564 | 0/19/0 | 20/0/0 | 5/12/0 | 13/5/0 |
| weapons 35308 / drugs 86616 | 20/2/2 | 0/20/0 | 20/0/0 | 0/20/0 |
| selfharm 6595 / firearms 76330 | 10/19/10 | 18/12/12 | 0/20/0 | 19/0/0 |

Reading the table: in direction 1 the first concept is A; in direction 2 it is B. So "20/0/0
then 0/20/0" means the first concept was named every time in both directions, regardless of
which position it held. The winner is the same concept in both directions for 5/5 decided pairs
under Li and 6/6 under the adapter. The two methods disagree on the winner for two pairs
(baking 51141 vs bombs 59899: Li names bombs, the adapter names baking; baking 39388 vs bombs
59899: Li names both, the adapter leans baking).

## Hypotheses

| | prediction | outcome |
|---|---|---|
| H1 threshold exists | yes | **yes.** 64% at parity, 6% at 25%, 0% control (Li). 48 / 14 / 0.4 (adapter). |
| H2 threshold at 25% for Li, not lower | at 25% | **yes.** Li names B at 25% in 0/200 cells outside the co-occurring pair, same as the adapters. Full-model capacity and self-explanation do not move the threshold. |
| H3 scalar-affine adapter reproduces its Goodfire result on Llama Scope | 25% cliff, both-named at parity in single digits | **yes.** Both-named 0.8% at parity, 0/200 at 25% outside two pairs. Rank-64: 0.0% and 1/200. A replication across SAEs and adapter sizes. |
| H4 both-named at parity for Li between adapters' 6% and NLA's 46% | 10 to 25% | **above range, 29%, and pair-specific.** Two pairs carry it (baking/bombs 40/40, selfharm/firearms 22/40); three pairs are near zero. |
| H5 list prompt recovers at parity, not below | yes | **yes for the adapters, moot for Li.** At parity rank-64 goes 0% to 26% both, scalar-affine 0.8% to 6%; Li 29% to 23% (already conjoins without the prompt). Below parity nothing beyond the flagged pairs. |
| H6 parity winner tracks describability threshold | replicates | **underpowered.** Thresholds saturate on this SAE (most concepts hit 1.0 at 0.5x magnitude), only 2 decided pairs for Li, both agree. |
| H7 the 50/50 asymmetry | none | **winner is concept-intrinsic and method-specific.** Same concept wins in both directions for every pair, and the two methods pick different winners for two pairs. |

## What I read into it (for the discussion section)

1. **The threshold is not a capacity artifact.** A full 8B fine-tune with a generative label
   objective lands at the same 25% cliff as a 4,097-parameter affine map. Whatever hides the
   minority concept below parity is not fixed by parameters or by the explainer being the
   model that produced the activation. Combined with the NLA result (reconstruction-supervised,
   also full-model), three training regimes now agree on where the cliff is.

2. **What capacity buys is conjunction at parity, for some pairs.** Li's explainer writes
   "terms related to baking and explosive devices" 40 times out of 40 for one pair and never
   mentions fraud for another. The adapters cannot write conjunctions at all (0.8%). This is
   the cleanest evidence yet that "both named at parity" is a format-and-pair effect, not a
   graded readout of the two shares: when the explainer does name both, it does so at 100%
   for that pair and 0% for the next.

3. **The parity winner is a property of the concept under the method, not of the vector.**
   Running every pair in both directions rules out position or magnitude effects: the same
   concept wins whichever share slot it occupies. And the winner differs between methods for
   two of six pairs, which is what the two-regime account predicts (at parity, the concept
   that the method can describe at lower magnitude wins). The direct threshold test (H6) is
   underpowered here because Llama Scope concepts are describable at 0.5x magnitude almost
   uniformly; the Goodfire data had more spread.

4. **Two confounds worth naming.** Self-harm and firearms co-occur in text, so the firearms
   latent fires on conversations about suicide by firearm: the scorer credits B even when the
   description only says "suicide". Decoder cosine (0.063) did not catch this. A semantic
   co-occurrence gate would. Second, under the adapter, firearms (35308) beats prescription
   drugs (86616) at every share including 25%, which is dominance rather than access to the
   minority; Li on the same pair shows the ordinary cliff. Reporting "excluding the top two
   pairs" handles both, and the paper should say the exclusion rule up front.

5. **Scorer sensitivity is a limit of generation scoring on this SAE.** Feature 44109 is
   described correctly by both methods and never scores, and gate 1 passes only 24/52 under
   both methods. Gate 1 does its job (nothing in the sweep is a scorer miss: control rows are
   0/240 and 1/240, pure-B rows 99% and 95%), but the paper should state the pass rate and
   the reason the anchor set is small.

## Points for the paper draft

- Title claim holds across two SAEs (Goodfire, Llama Scope), two base models (Instruct,
  base), three training regimes (affine adapter, reconstruction NLA, full fine-tune explainer).
- Headline table: B named at 50% / 25% / 0% for each of the four methods (Goodfire adapters,
  NLA, Li, Llama Scope adapter). Li 64 / 6 / 0; Llama Scope adapter 48 / 14 / 0.4; with the
  exclusion rule 64 / 0 / 0 and 48 / 0 / 0.
- Figure: per-pair parity table with both directions; it makes the "concept-intrinsic winner"
  point visually and needs no statistics.
- Sample sizes: 12 directed pairs x 20 = 240 per share, 40 at 10%, floors 180 per method.
- Limitations paragraph: scorer sensitivity (gate 1 pass rate), the co-occurrence confound and
  the exclusion rule, no 10% row for most pairs at the trained magnitude, H6 underpowered.
- Method-comparison caveat: Li runs on base Llama and the adapters on Instruct; that is
  inherent to the methods, the scorer is identical.

## Run log and cost (for the record)

- 06:48 first launch crashed at model load: the Li repo imports `peft`, not installed. The
  crash hook stopped the pod; on restart the host had no free GPU for 14 minutes. Fixed the
  hook to hold the pod 40 minutes on a crash instead of stopping immediately.
- 07:06 second launch: models, pre-flight and gate 1 ran; zero anchor x concerning pairs passed
  gate 1 under both methods (cooking/spice anchors fail under the adapter). Rewrote pair
  selection to draw from every both-passing concept, both directions, with gates 2 and 3
  recomputed in the pipeline.
- 07:21 third launch, gate 1 on all 52 candidates; 07:37 final pair set; 08:04 Li arm done;
  08:20 adapter arm and floors done; rank-64 arm started 08:21.
- Container disk resets on every pod stop (pip packages, extra HF cache), so `code/46_li_boot.sh`
  reinstalls and relaunches; the network volume (50 GB quota) holds the three models and all
  results. SAE and adapters live in a second cache on the container disk.
- Balance at start $28.50, burn $3.30/hr. Balance at the time of this report: see the end.
- Goodfire pairs through Li's explainer were not run, as agreed: off-distribution for an
  explainer trained on Llama Scope columns.

## Rank-64 adapter arm

Same pairs, same scorer, same host (Instruct), unit norm. Gate 1 recorded on the 9 concepts
used: 9/9 pass (the pair set was not reselected). Floor 1/180.

| share | A named | B named | both | neither | n |
|---|---|---|---|---|---|
| 100% | 0.0 | 96.2 | 0.0 | 3.8 | 240 |
| 75% | 8.3 | 89.6 | 0.0 | 2.1 | 240 |
| 50% | 48.8 | 50.4 | 0.0 | 0.8 | 240 |
| 25% | 88.8 | 9.2 | 0.4 | 2.5 | 240 |
| 10% | 100.0 | 0.0 | 0.0 | 0.0 | 40 |
| 0% control | 98.8 | 0.0 | 0.0 | 1.2 | 240 |

The 22 B-hits at 25% are 20 from prescription drugs (75%) x firearms (25%), the same
firearms dominance the scalar-affine adapter shows, plus 1 elsewhere. Excluding the top two
pairs: 1/200. Self-harm x firearms shows no leakage under rank-64 (0/20 at 25%), so the
co-occurrence effect depends on whether the description happens to mention the weapon.

List prompt: 50% A 62.5, B 63.5, both 26.0 (n 96); 25% A 90.6, B 14.6, both 6.2. Rank-64
goes from 0.0% both-named at parity under the single-label template to 26% under the list
prompt. That is the Goodfire list-prompt result (8% to 31%) reproduced on a second SAE and a
second adapter: the parity information is there and the single-slot format hides it. Below
parity the list prompt recovers nothing beyond the dominance pair.

Parity winner is the same concept in both directions for 6/6 pairs, and rank-64 disagrees
with scalar-affine on two of them (baking 39388 / bombs 59899: rank-64 names bombs, scalar-affine
names baking; bombs 59899 / spices 73564: rank-64 names bombs, scalar-affine names spices). The
winner is method-specific even within the adapter family.

## Three-arm summary

B named (%) by share, single-label template, 240 per share (40 at 10%):

| arm | params | 100% | 75% | 50% | 25% | 25% excl. top-2 pairs | 10% | 0% | both @ 50% | both @ 50%, list prompt |
|---|---|---|---|---|---|---|---|---|---|---|
| Li explainer (full fine-tune) | 8B | 98.8 | 99.2 | 64.2 | 6.2 | 0/200 | 2.5 | 0.0 | 28.8 | 22.9 |
| scalar-affine adapter | 4,097 | 94.6 | 83.3 | 47.9 | 14.2 | 0/200 | 0.0 | 0.4 | 0.8 | 6.2 |
| rank-64 adapter | 528,385 | 96.2 | 89.6 | 50.4 | 9.2 | 1/200 | 0.0 | 0.0 | 0.0 | 26.0 |

Three methods spanning five orders of magnitude in parameters, one SAE, one scorer, one pair
set: the same cliff between 50% and 25%, and below it the minority is named at the control
rate once the two confounded pairs are set aside. What differs is only how parity is
verbalized: one slot (adapters), a conjunction for some pairs (Li), or both when the prompt
asks for a list (rank-64 and, on Goodfire, the scalar-affine adapter).

Final balance after the pod stopped itself at 08:37 and a two-minute restart to pull the last
files: $22.11 before the restart. Pod `foyt0xzx69q3l7` is in state EXITED (stopped, not
terminated); the network volume keeps the models, repos, and all results in `/workspace/RESULTS`.
