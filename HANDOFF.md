# Project handoff: concept omission in introspection methods

Factual record of what was run and what the numbers are. No interpretation.

## Question

When an activation contains two concepts in a known proportion, does the description
name both, or only the dominant one?

## Setup

**Models and readouts**
- Llama-3.1-8B-Instruct, Goodfire SAE at layer 19 (65,536 latents)
- Gemma-3-12B-IT, Gemma Scope 2 at layer 32 (16,384 latents)

**Interpreters**
- Pepper et al. (arXiv:2602.10352) released adapters, used without retraining:
  scalar affine (4,097 params) and scalar affine + rank-16 (135,169 params)
- Fraser-Taliente et al. natural language autoencoder (Transformer Circuits 2026),
  `kitft/nla-gemma3-12b-L32-av`, verbaliser + reconstructor trained jointly by RL

**Construction**
- `v(alpha) = normalize(alpha*d_A + (1-alpha)*d_B)` over unit-normalised decoder rows
- Outer normalisation keeps injected norm constant across the sweep
- Llama: injected at the reserved-token embedding in `What is the meaning of "X"?`.
  Injection magnitude = 1.0, which is the unit norm the adapter was trained on and the
  default in the released eval code (`scale_values` default `[1.0]`)
- Gemma: construction adds `b_dec` (bare decoder directions are off-distribution at
  that layer), scaled by s=30000; the NLA prescribes its own injection norm
- Gemma ratios verified against the SAE encoder: 0.73-1.21 at nominal 50%,
  0.21-0.39 at 25%, 0.04-0.12 at 10%

**Gates**
- Gate 1: concept must be described correctly when injected alone. All 10 Llama
  concepts at hit rate 1.0. All 10 are in the **validation split** of the labelled
  feature set, so the adapter never trained on them
- Gate 2: decoder cosine < 0.1 between paired concepts
- Validity: concerning latent must fire in the composed vector at every ratio

**Scoring** (generation scoring, following Pepper et al.)
Description is returned to the model, 10 short conversations generated from it, run
forward, SAE-encoded, target latent checked for activation. Threshold 0.3.
Descriptions are never read as text.

**Sample**
- Llama: 12 pairs x 20 descriptions = 240 per ratio, per adapter
- Gemma: 10 pairs x 20 = 200 per ratio
- Ratios: 100%, 75%, 50%, 25%, 10% concerning-concept share, plus 0% control

---

## Main results (Llama, trained magnitude, 240 per ratio)

### Scalar affine, 4,097 params

| B share | A named | B named | both | neither |
|---|---|---|---|---|
| 100% | 2/240 (0.8%) | 184/240 (76.7%) | 0 | 54 |
| 75% | 5/240 (2.1%) | 182/240 (75.8%) | 0 | 53 |
| 50% | 189/240 (78.8%) | 37/240 (15.4%) | 15 | 29 |
| 25% | 233/240 (97.1%) | 0/240 (0.0%) | 0 | 7 |
| 10% | 229/240 (95.4%) | 0/240 (0.0%) | 0 | 11 |
| 0% (control) | 231/240 (96.2%) | 0/240 (0.0%) | 0 | 9 |

### Rank-16, 135,169 params (same pairs, same sample)

| B share | A named | B named | both | neither |
|---|---|---|---|---|
| 100% | 0/240 (0.0%) | 217/240 (90.4%) | 0 | 23 |
| 75% | 1/240 (0.4%) | 223/240 (92.9%) | 1 | 17 |
| 50% | 136/240 (56.7%) | 89/240 (37.1%) | 7 | 22 |
| 25% | 230/240 (95.8%) | 1/240 (0.4%) | 1 | 10 |
| 10% | 232/240 (96.7%) | 0/240 (0.0%) | 0 | 8 |
| 0% (control) | 236/240 (98.3%) | 0/240 (0.0%) | 0 | 4 |

### Key derived facts
- Max "both" anywhere on either curve: 15/240 (6.3%), at the 50/50 mixture
- Role swap: anchor at a 25% share named 5/240 (2.1%); concerning concept at a 25%
  share named 0/240 (0.0%)
- At a 25% B share the model names at least one concept 233/240; at 50/50, 211/240
- Unexplained: at 50/50 the anchor is named 189/240 against the concerning
  concept's 37/240

---

## Gemma NLA (10 pairs x 20 = 200 per ratio)

| B share | A named | B named | both | neither |
|---|---|---|---|---|
| 100% | 35/200 (17.5%) | 200/200 (100%) | 35 | 0 |
| 75% | 106/200 (53%) | 200/200 (100%) | 106 | 0 |
| 50% | 132/200 (66%) | 160/200 (80%) | 92 | 0 |
| 25% | 200/200 (100%) | 55/200 (27.5%) | 55 | 0 |
| 10% | 200/200 (100%) | 9/200 (4.5%) | 9 | 0 |

**False-positive floor, two independent estimates:**
- 35/200 = 17.5%, anchor named on pure-B rows where the anchor is genuinely absent
- 219/2000 = 10.9%, features belonging to neither concept in the pair

Using the conservative 17.5% floor:
- 25% share: 55/200 vs 35/200, z = 2.39, p = 0.017. Real signal.
- 10% share: 9/200 = 4.5%, below the floor. No signal.

Metric sensitivity 24/24 (older 42-basis measurement).

**Notes**
- The NLA loses the subordinate concept one threshold below the Llama adapter:
  it holds at a 25% share and fails by 10%, where Llama already fails at 25%
- It names both concepts far more often: 92/200 (46%) at parity, against Llama's
  15/240 (6.3%)
- "neither" is 0 at every ratio, unlike Llama, so it always describes something relevant
- Same role asymmetry as Llama but much weaker: anchor at a 25% share 53%,
  concerning concept at a 25% share 27.5%. Both clear the floor
- Gemma numbers need floor-correcting before comparison with Llama, whose floor is ~0%

### Gemma pair gating
3 of 5 candidate pairs passed, giving 10 total:
- PASS: cooking x explosives (cos -0.022), legal x malware (0.056),
  travel x ransomware (-0.011)
- FAIL: travel x hatred_discrim (cos 0.178, above the 0.1 threshold)
- FAIL: travel x malware (concerning latent reads 0 at one ratio)

---

## Metric validation (Llama)

| Check | Result |
|---|---|
| Random directions | 0/144 false positives |
| Pure anchor scored for concerning concept | 0/240 |
| Pure concerning concept recovered | 184/240 (76.7%) |
| Manual agreement with human reading | 23/25 (92%) |

76.7% on a pure concept is consistent with the 70% generation scoring Pepper et al.
report at 70B.

---

## Supporting experiments (all superseded as primary, kept as robustness)

**Injection magnitude sweep.** The released eval config uses
`scale_values = [0.5, 0.8, 1.3, 2.1, 3.4, 5.5]`, which are multiples of the trained
unit norm. Pure-concept recovery by magnitude: 19%, 75%, 81%, 97%, 97%, 67%
(inverted U). The collapse holds at all six. Restricting to 2.1x and 3.4x gives
97.2% / 95.8% / 30.6% / 0/72 / 0/72 / 0/72.

**Layer sweep.** 9 layers (4, 8, 12, 16, 19, 22, 25, 28, 31), 864 descriptions.
Second concept named 1/864 (0.1%).
NOTE: this is NOT the constructed-vector setup. It reads the last-token hidden state of 8
real loaded prompts (the same PAIRED2 set as the monitoring experiment), so there is no
controlled ratio, and the outcome is a manual read of whether the description names the
concerning content rather than generation scoring. No false-positive control. Cut from the
write-up's ruled-out list for these reasons; it does not isolate layer for the collapse.

**List prompt (asking explicitly for multiple concepts).** Both named 7/90 -> 30/96
at parity; 0/90 -> 1/96 at a 25% share.

**Token budget.** max_new 30 -> 100: median description length unchanged at 6 words,
both-named 12/216 -> 9/216. The constraint is the trained output format, not the
allowance.

**Targeted probe** (5 pairs, 90 per cell). Naming the concept in the question.
Yes-rate 76% / 41% / 21% at 50/25/10% share, against a 29% pure-anchor floor.
z = 6.27 / 1.72 / -1.20, p < 0.001 / 0.086 / 0.23.
NOTE: the behavioural metric is invalid for this condition, since the question
contains the concept's vocabulary. It "detects" the concept on pure-anchor
activations 52/90 (58%). Only the yes/no answer is usable.

**Neutral vs concerning second concepts at parity.** 27/54 (50%) vs 23/36 (64%).
No evidence that refusal training suppresses the concerning concepts.

**Real (non-constructed) activations.** 57 activations from real text.
By latent strength relative to the top latent: 42% / 39% / 27% / 26% / 23% / 17%
across bins 90-100%, 60-90%, 40-60%, 25-40%, 10-25%, <10%.
Constructed-prose subset by measured ratio: 27% / 21% / 12% / 8%.
**No false-positive control exists for this arm. Do not draw conclusions from it.**

**Decodability margin.** Concept B's SAE activation over the strongest non-target
latent: 5.17x at parity, 1.50x at 25%, 0.28x at 10%. Pure-anchor control puts B at
rank ~1607 with activation 0.000. The claim only holds at parity.

**Monitoring position.** 8 matched prompt pairs, benign framing plus harmful payload,
read at the last token before the response. Harmful content named 1/96, refusal-type
language 11/96, false alarms on benign twins 0/96.
NOTE: at that position the model has already resolved to refuse, so the content may
genuinely not be present. Weak evidence, probably should be dropped.

**Untrained baseline.** Raw injection without the adapter recovers a pure concept
7/90. Fails its own endpoint check, so unusable as a control.

---

## Hypotheses tested and rejected

- **Semantic relatedness** (concepts closer to the anchor survive): did not replicate
  on Llama
- **Co-occurrence frequency**: direct counterexample. `programming x malware` survived
  at the lowest co-occurrence (0.0074); `legal x ransomware` went silent at the
  highest (0.0594)
- **Refusal training suppressing concerning concepts**: no effect (above)
- **Adapter capacity**: rejected, larger adapter is better where the concept dominates
  and equally blind below
- **Output length / token budget**: rejected
- **Layer**: rejected
- **Prompt wording**: rejected below parity

No mechanism was found.

---

## Errors caught during the project

- **`typical_act` scaling bug (Gemma).** Mixtures were scaled by each feature's median
  activation (range 604-1957), which swamped the mixing coefficient. Nominal 25% ratios
  ran 0.20 to 1.63, so the "subordinate" concept was sometimes stronger. Rebuilt on
  unit directions and verified against the encoder.
- **Keyword Gate 1 false pass.** `programming` cleared a keyword gate 3/3 on an
  explanation about Vedic astrology, matching only "code". Behavioural re-run scored
  0.20 and failed. Its pair had been detected 6/6 at the 25% ratio; removing it made
  the result stricter.
- **Dead-injection control run.** One control described the anchor 0/90 where live
  cells ran 83-92%. Two control files existed; the wrong one would have been used.
- **Circular experiment killed.** A test for whether the omitted concept is present in
  the model's states while writing the description reads an SAE over text the model
  just produced, so it largely measures what the text says. Generated positions also
  attend back to the injection site. Cut before completion.
- **Unsourceable number.** An earlier summary recorded 0/84 and 10/84 for the
  monitoring experiment. Neither denominator reconstructs from the stored data
  (8 passages x 12 descriptions = 96). Recomputed as 1/96 and 11/96.
- **Overstated rank claim.** "Concept B is rank 2 of 65,536" is true at a 25% share
  and misleading: its margin over the strongest unrelated latent is 1.01x. Report
  margin, never rank.
- **Bad metric recommendation.** Aggregate recall over both present concepts *rises*
  as the effect worsens (43% / 46% / 48% at 50/25/10%), because the dominant concept
  is named more often as the subordinate one fades. Report detection of the
  subordinate concept specifically.

---

## Files

Local: `results/RESULTS/`
- `trained_magnitude.pkl` — main Llama result, both adapters, 6 ratios, 240 each
- `gemma_rerun_descriptions.pkl` — NLA descriptions, 10 pairs x 20 x 5 ratios
- `gemma_full_scores.pkl` — NLA scores (in progress)
- `safety_sweep.pkl`, `fill_curve.pkl`, `matched_sample.pkl` — superseded swept runs
- `token_budget.pkl`, `targeted_probe*.pkl`, `layer_sweep.pkl`, `monitor_pos.pkl`,
  `constructed_real.pkl`, `external_real.pkl`, `sweep_results.pkl`, `list_results.pkl`

Scripts: `code/00_reload.py` through `code/38_llama_topup.py`.
`jclient.py` is a direct Jupyter REST/WebSocket client used to drive the remote GPU.

## Outstanding

1. Figures — none built on the 240 basis yet
2. Citations unverified except: SelfIE arXiv:2403.10949, Patchscopes arXiv:2401.06102,
   Pepper et al. arXiv:2602.10352, Superscopes arXiv:2503.02078,
   Faithful-Patchscopes arXiv:2602.00300, Li et al. arXiv:2511.08579
3. Random qualitative examples selected but not yet placed in the write-up
