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

Random-direction floor, rerun at 20 directions x 12 latents = 240 per method (`51_floors240.py`, `floors240.pkl`): Li 0/240, adapter 0/240, rank-64 1/240. (Original 20 x 9 run: 0, 1, 1 of 180.) Goodfire rerun at 20 x 12: scalar-affine 1/240, rank-16 0/240, at both scale 1.0 and 2.1.

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

Reading the table: in direction 1 the first concept is A; in direction 2 it is B. **Correction
to what I wrote overnight:** at parity the two directions are the same vector
(0.5 u_A + 0.5 u_B normalised is symmetric), so the two columns per method are two independent
20-draw samples of one cell, not a test of position. "Same winner in both directions" at parity
is therefore a replication across 40 draws, not evidence against a position effect. What the two
directions do test is the 75% and 25% rows, where the vectors differ (0.75 u_A + 0.25 u_B versus
0.25 u_A + 0.75 u_B): the minority is hidden whichever concept is the minority. The per-share
counts at 75/25/10% come from 12 distinct vectors x 20 draws; at 50% from 6 vectors x 40 draws.

What survives: the parity winner is stable across 40 independent draws for most pairs (20/0
then 0/20 means the same concept won all 40), and the two methods pick different winners for
two pairs (baking 51141 vs bombs 59899: Li names bombs, the adapter names baking; baking 39388
vs bombs 59899: Li names both, the adapter leans baking). So the winner is a property of the
concept under the method, not of the vector.

## Hypotheses

| | prediction | outcome |
|---|---|---|
| H1 threshold exists | yes | **yes.** 64% at parity, 6% at 25%, 0% control (Li). 48 / 14 / 0.4 (adapter). |
| H2 threshold at 25% for Li, not lower | at 25% | **yes.** Li names B at 25% in 0/200 cells outside the co-occurring pair, same as the adapters. Full-model capacity and self-explanation do not move the threshold. |
| H3 scalar-affine adapter reproduces its Goodfire result on Llama Scope | 25% cliff, both-named at parity in single digits | **yes.** Both-named 0.8% at parity, 0/200 at 25% outside two pairs. Rank-64: 0.0% and 1/200. A replication across SAEs and adapter sizes. |
| H4 both-named at parity for Li between adapters' 6% and NLA's 46% | 10 to 25% | **above range, 29%, and pair-specific.** Two pairs carry it (baking/bombs 40/40, selfharm/firearms 22/40); three pairs are near zero. |
| H5 list prompt recovers at parity, not below | yes | **yes for the adapters, moot for Li.** At parity rank-64 goes 0% to 26% both, scalar-affine 0.8% to 6%; Li 29% to 23% (already conjoins without the prompt). Below parity nothing beyond the flagged pairs. |
| H6 parity winner tracks describability threshold | replicates | **underpowered.** Thresholds saturate on this SAE (most concepts hit 1.0 at 0.5x magnitude), only 2 decided pairs for Li, both agree. |
| H7 the 50/50 asymmetry | none | **winner is concept-specific and method-specific.** Stable across 40 draws per pair; the two methods pick different winners for two pairs. The direction test is vacuous at parity (same vector), see correction above. |

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

3. **The parity winner is a property of the concept under the method.** It is stable across
   40 independent draws per pair, and it differs between methods for two of six pairs, which is
   what the two-regime account predicts (at parity, the concept that the method can describe
   at lower magnitude wins). Note the parity vector is identical in both directions, so
   direction is not a manipulation there; it is one at 75/25/10%. The direct threshold test (H6) is
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
- Figure: per-pair parity table, 40 draws per pair per method; it makes the "concept-specific,
  method-specific winner" point visually and needs no statistics.
- Sample sizes: 12 directed pairs x 20 = 240 per share, 40 at 10%, floors 240 per method (20 directions x 12 latents).
- Limitations paragraph: scorer sensitivity (gate 1 pass rate), the co-occurrence confound and
  the exclusion rule, no 10% row for most pairs at the trained magnitude, H6 underpowered.
- Method-comparison caveat: Li runs on base Llama and the adapters on Instruct; that is
  inherent to the methods, the scorer is identical.


## Quirks in the raw descriptions

Read from all 3,720 sampled descriptions plus the gate-1 outputs (`code/45_li_analyze.py`
does the tables; this section is from reading the text).

- **Style.** Li writes 16 words on average, the adapters 7. Both are templated: "references to"
  opens 79% of Li's and 67% of the adapters' descriptions. Li adds a "particularly ..." qualifier
  with invented specifics: "donuts and pastries from a specific bakery in Los Angeles",
  "bakery-related products in London and Paris, France, and Virginia", the name "Graham" for a
  spice feature, "Peace Corps" for a peace feature. These specifics are confabulated detail on
  a correct concept, and the scorer ignores them.
- **Diversity.** Li produces 17.4 unique descriptions per 20 draws; the adapters 13 to 14.5.
  Adapter outputs repeat at temperature 0.7.
- **Switching, not blending.** Following one pair across shares (baking 51141 majority, fraud
  125117 minority): Li at 100% and 75% fraud writes fraud with sub-facets (elderly victims,
  charity donations, religion); at 50% it flips to baking outright and stays there. Nothing in
  between: no "financial and culinary" hybrids. The adapter does the same but with one extra
  step: at 75% fraud it drifts to a neighbouring concept (cybercrime, phishing, hacking) that
  the fraud latent does not fire on, then flips to baking at 50%.
- **"Neither" rows.** Li has 11 of 1,240; most are correct descriptions the scorer missed
  ("references to explosives and bomb-related incidents" at 100% explosives scored neither).
  The adapters have 55 and 25; theirs are vaguer near-misses ("medical or healthcare concepts"
  for prescription drugs, "cybersecurity threats" for fraud) and one hallucination ("the name
  Helen" for spices). So the adapters' failures are semantic drift; Li's are scorer misses.
- **Conjunctions.** About 65% of Li's descriptions contain "X and Y", but nearly all are
  within-concept ("baked goods and desserts", "explosives and bomb-making"). Cross-concept
  conjunctions ("baking and explosive devices") occur for one pair only, and there in 40/40.
- **Gate-1 failure modes differ.** Where Li passes and the adapter fails, the adapter is not
  vaguer, it is wrong: "historical events" for cooking actions, "governance and policy" for a
  spice, "social media" for seasoning. Where both fail, Li is often right and the scorer misses
  (ransomware 82686 described as ransomware, scored 0.0; theft 110615 scored 0.7), and for a few
  features both methods agree on something other than the Neuronpedia label (self-harm 127060
  read as automotive recalls by both; extremism 124033 read as symbols by Li). Those labels may
  be wrong or the features polysemantic; worth a Neuronpedia look before citing them.
- **Dominance under the adapters.** For drugs (75%) x firearms (25%) the adapters write
  "firearms and their regulations" 17 to 20 times out of 20; Li writes opioid prescribing. The
  firearms latent is loud for the adapters specifically, consistent with the winner being a
  method-concept property.
- **Greedy at parity (Li).** Five of six pairs give a single-concept greedy description; the
  baking 39388 x bombs pair gives the conjunction; self-harm x firearms gives a mixture scored
  0.3 / 0.6. Greedy and sampled agree on the winner in every pair.

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

Parity winner is stable across the 40 draws for 6/6 pairs, and rank-64 disagrees
with scalar-affine on two of them (baking 39388 / bombs 59899: rank-64 names bombs, scalar-affine
names baking; bombs 59899 / spices 73564: rank-64 names bombs, scalar-affine names spices). The
winner is method-specific even within the adapter family.

## Three-arm summary

Single-label template, share is B's share, 240 per share (40 at 10%). "A named" and "B named"
each include the both-named rows.

| arm | share | A named | B named | both | neither |
|---|---|---|---|---|---|
| Li explainer (8B full fine-tune) | 100% | 0.8 | 98.8 | 0.8 | 1.2 |
| Li explainer | 75% | 6.2 | 99.2 | 6.2 | 0.8 |
| Li explainer | 50% | 62.5 | 64.2 | 28.8 | 2.1 |
| Li explainer | 25% | 99.6 | 6.2 | 6.2 | 0.4 |
| Li explainer | 10% | 100.0 | 2.5 | 2.5 | 0.0 |
| Li explainer | 0% | 100.0 | 0.0 | 0.0 | 0.0 |
| scalar-affine (4,097) | 100% | 0.4 | 94.6 | 0.4 | 5.4 |
| scalar-affine | 75% | 14.6 | 83.3 | 2.1 | 4.2 |
| scalar-affine | 50% | 46.2 | 47.9 | 0.8 | 6.7 |
| scalar-affine | 25% | 82.9 | 14.2 | 0.8 | 3.8 |
| scalar-affine | 10% | 100.0 | 0.0 | 0.0 | 0.0 |
| scalar-affine | 0% | 97.1 | 0.4 | 0.4 | 2.9 |
| rank-64 (528,385) | 100% | 0.0 | 96.2 | 0.0 | 3.8 |
| rank-64 | 75% | 8.3 | 89.6 | 0.0 | 2.1 |
| rank-64 | 50% | 48.8 | 50.4 | 0.0 | 0.8 |
| rank-64 | 25% | 88.8 | 9.2 | 0.4 | 2.5 |
| rank-64 | 10% | 100.0 | 0.0 | 0.0 | 0.0 |
| rank-64 | 0% | 98.8 | 0.0 | 0.0 | 1.2 |

B named at 25% excluding the two confounded pairs per arm: Li 0/200, scalar-affine 0/200,
rank-64 1/200. Both named at parity under the list prompt: Li 22.9, scalar-affine 6.2, rank-64 26.0.

Three methods spanning five orders of magnitude in parameters, one SAE, one scorer, one pair
set: the same cliff between 50% and 25%, and below it the minority is named at the control
rate once the two confounded pairs are set aside. What differs is only how parity is
verbalized: one slot (adapters), a conjunction for some pairs (Li), or both when the prompt
asks for a list (rank-64 and, on Goodfire, the scalar-affine adapter).

Final balance after the pod stopped itself at 08:37 and a two-minute restart to pull the last
files: $22.11 before the restart. Pod `foyt0xzx69q3l7` is in state EXITED (stopped, not
terminated); the network volume keeps the models, repos, and all results in `/workspace/RESULTS`.
