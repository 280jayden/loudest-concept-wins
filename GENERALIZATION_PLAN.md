# Generalization experiment: Li et al. explainers

Everything below was checked against the released code and checkpoints on 2026-09-01.
Sources at the bottom.

## Why this target

Li, Guo, Huang, Steinhardt & Andreas, *Training Language Models to Explain Their Own
Computations* (arXiv:2511.08579), fine-tune Llama-3.1-8B to describe SAE features and
find self-explanation beats cross-model explanation. Three things make it the right
fourth method for the completeness axis:

1. **Same injection mechanism.** A 4096-d vector goes in as a continuous token at a
   reserved position in the embedding layer, no projection when dimensions match, no
   normalisation. The composed-vector protocol ports unchanged.
2. **Same base model family and layer.** Target is Llama-3.1-8B, all 32 layers, layer 19
   available. The existing pipeline needs a different SAE, not a different model.
3. **A third training regime.** Adapters are label-supervised and tiny. The NLA is
   reconstruction-supervised and full-model. Li's explainers are label-supervised and
   full-model. That separates "objective" from "capacity" in a way the first two cannot.

## Arms

One SAE, one set of gated pairs, one scorer. Two methods on it:

| arm | method | injects into | trained input | magnitude used |
|---|---|---|---|---|
| Li | `Transluce/features_explain_llama3.1_8b_llama3.1_8b`, full fine-tune | Llama-3.1-8B base | raw Llama Scope decoder columns | mean raw column norm of the pair (0.7 to 3.4, mean 1.52) |
| adapter | `keenanpepper/.../llamascope-sae-scalar-affine.safetensors`, 4,097 params | Llama-3.1-8B-Instruct | unit-normalised decoder columns | 1.0 |

Each method gets the magnitude it was trained on. The remaining difference between the
arms is base versus Instruct, which is inherent to the methods and is stated.

Self explainer only. The cross-model and Instruct-explainer checkpoints test a different
question. Rank-64 adapter (528,385 params) exists and is not run for now.

## Assets

| item | exact identifier | checked |
|---|---|---|
| explainer | `Transluce/features_explain_llama3.1_8b_llama3.1_8b` | full 16 GB model, class `ContinuousLlama` from `TransluceAI/introspective-interp`, config arch `LlamaWithIntervention` |
| adapter | `keenanpepper/selfie-adapters-llama-3.1-8b-instruct/llamascope-sae-scalar-affine.safetensors` | 19 kB, scalar affine |
| target / scorer model | `meta-llama/Llama-3.1-8B` | base |
| writer + adapter host | `meta-llama/Meta-Llama-3.1-8B-Instruct` | |
| SAE | `fnlp/Llama3_1-8B-Base-LXR-32x/Llama3_1-8B-Base-L19R-32x/checkpoints/final.safetensors` | 2.15 GB, tensors `encoder.weight [131072,4096]`, `encoder.bias`, `decoder.weight [4096,131072]`, `decoder.bias` |
| labels | Neuronpedia source `19-llamascope-res-131k` | `POST /api/explanation/search` with `layers`, `GET /api/feature/...` |

**SAE facts from `hyperparams.json` and the weights.** JumpReLU, threshold 0.484375,
dataset-wise input normalisation with average norm 17.125 (so inputs are rescaled by
64/17.125 before encoding), decoder bias on, hook `blocks.19.hook_resid_post`. Decoder
columns are **not** unit norm in the released file: mean 1.52, range 0.72 to 3.37. Li's
dataloader uses the raw column, so their trained magnitude is per-feature.

**Explainer injection, from the code.** Prompt `At layer 19, <|reserved_special_token_10|>
<|reserved_special_token_12|><|reserved_special_token_11|> encodes `, wrapped in the
tokenizer's chat template if it has one, assistant header appended. The vector replaces
the embedding at token 12 via `inputs_continuous_tokens`, no projection, no
normalisation. Their eval decodes greedily, `generate_limit` 20, stop strings
`" [END]"`, a newline, and `">>>."`.

## Gates

Same three as the main experiment. Gates 2 and 3 are CPU and run locally before the
pod exists (`42_li_local_gates.py`). Gate 1 needs the models and runs on the pod under
both methods; a pair is kept only if both concepts pass under both methods.

1. **Describable alone.** Under each method, at that method's trained magnitude, hit
   rate >= 0.8 by generation scoring, best of three descriptions. Also recorded at 0.5,
   0.7, 1.0, 1.5, 2.0, 3.0 times the magnitude for the explainer, which gives each
   concept's describability threshold.
2. **Decoder cosine** |cos(d_A, d_B)| < 0.1 on unit-normalised columns.
3. **Minority latent fires at every share** under the Llama Scope JumpReLU encoder, on
   the composed vector at the injection magnitude, after the SAE's own rescale. Measured
   locally: at the trained magnitude this passes at 75/50/25% for every pair tried and
   at 10% for some pairs and not others, so gate 3 is the filter that decides which
   pairs carry a 10% row. No rescaling, since that would move the explainer off its
   trained input. If fewer than 12 pairs survive, the count is reported.

## Construction

`v(alpha) = normalize(alpha*d_A + (1-alpha)*d_B)` over unit-normalised decoder columns,
then scaled to the arm's magnitude. Shares 100, 75, 50, 25, 10, and the 0% control.

## Scoring, one function for both arms

Description goes to Instruct, which writes ten short conversations from the same prompt
as the main experiment. Each is run forward through **base** Llama-3.1-8B, layer 19 is
encoded with Llama Scope, and the target latent counts as fired if it is above zero on
any post-BOS token. A description is a hit at >= 0.3 of the ten. Floors on this scorer:
the 0% control row, and 20 random directions x 12 latents = 240, under each method.

## Sample

12 pairs x 20 sampled descriptions per share (temperature 0.7, top-p 0.9, the sampler
used throughout), = 240 per share. Plus one greedy description per cell, their protocol.
Plus the list prompt at 50% and 25% shares, 8 draws per cell: on the adapters it
recovers both concepts at parity about a third of the time and nothing at 25%, which is
the diagnostic for format versus access.

## Hypotheses and expected results

**H1. A threshold exists.** At a 25% share the explainer names concept B at its 0%
control rate. *Expected: yes.*

**H2. The threshold sits at 25%, with the adapters, not lower with the NLA.** The
explainer was trained on one decoder column per example, the same data shape as the
adapters, with a full model's capacity. *Expected: at 25%.* If it holds at 25% like the
NLA, capacity or self-explanation does something the adapter comparison could not see.

**H3. The scalar-affine adapter reproduces its Goodfire result on Llama Scope.**
*Expected: 25% cliff, both-named at parity in single digits.* A replication across SAEs
if yes; SAE-dependence if no. Either is reported.

**H4. Both-named at parity** lands between the adapters' 6% and the NLA's 46% for the
explainer. *Expected: 10 to 25%.*

**H5. List prompt recovers at parity and not below**, for both methods. *Expected: yes.*

**H6. Parity winner tracks the describability threshold** measured in gate 1. From the
Goodfire data, r = 0.77 across 7 concepts. *Expected: replicates.*

**H7. The 50/50 asymmetry.** No prediction.

## Order on the pod

Setup (`44_li_pod_setup.sh`) -> pre-flight -> gate 1 both methods -> Li sweep and
scoring -> copy out -> adapter sweep and scoring -> copy out -> floors -> summary ->
`runpodctl stop pod`. All in `43_li_pipeline.py`, resumable, checkpointed per cell.

## Budget

H100 SXM 80 GB. Explainer 16 GB + base 16 GB + Instruct 16 GB + SAE 4 GB (fp32
encoder and decoder) + working memory, about 55 GB. Roughly 4 hours, about $14. 150 GB
volume.

## Sources

- Li, Guo, Huang, Steinhardt & Andreas, arXiv:2511.08579. Code github.com/TransluceAI/introspective-interp. Checkpoints huggingface.co/Transluce.
- Llama Scope: He et al., arXiv:2410.20526. Checkpoints huggingface.co/fnlp/Llama3_1-8B-Base-LXR-32x
- Pepper et al., arXiv:2602.10352. Adapters huggingface.co/keenanpepper/selfie-adapters-llama-3.1-8b-instruct
- Neuronpedia: neuronpedia.org/llama3.1-8b/19-llamascope-res-131k
