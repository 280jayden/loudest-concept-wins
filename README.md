# The loudest concept wins

Introspection methods describe one concept per activation. When a second concept holds
a quarter of the activation or less, it is named at the same rate as a concept that was
never in the activation at all.

MATS 12.0 application project (Neel Nanda's stream).

## Result

Llama-3.1-8B-Instruct, Goodfire SAE layer 19, adapters from Pepper et al.
240 descriptions per ratio (12 concept pairs x 20).

| second concept's share | scalar affine (4,097 params) | + rank-16 (135,169 params) |
|---|---|---|
| 100% (pure) | 184/240 | 217/240 |
| 75% | 182/240 | 223/240 |
| 50% | 37/240 | 89/240 |
| 25% | **0/240** | 1/240 |
| 10% | 0/240 | 0/240 |
| 0% (control) | **0/240** | 0/240 |

The 25% row and the control row are the same number.

Gemma-3-12B-IT with the Fraser-Taliente et al. natural language autoencoder verbaliser,
200 per ratio, holds the minority concept at a 25% share (55/200 against a 35/200
false-positive floor, p = 0.017) and loses it by 10%. A better method moves the
threshold down one step.

## Method

Activations are built rather than found:

```
v(alpha) = normalize(alpha * d_A + (1 - alpha) * d_B)
```

over unit-normalised SAE decoder rows, so each concept's share is exact. Three gates on
what gets used: each concept must be described correctly when injected alone (all ten at
hit rate 1.0), paired concepts must have decoder cosine below 0.1, and the minority
concept's latent must fire in the composed vector at every ratio. All ten concepts sit in
the validation split of the SAE's labelled features, so the adapters never trained on them.

Descriptions are scored by generation scoring, following Pepper et al.: the description
goes back to the model, ten short conversations are generated from it, those are run
forward and SAE-encoded, and the target latent is checked for activation. Descriptions are
never read as text.

## Repository

```
code/         experiment scripts, 00 through 39, run in order
results/
  RESULTS/    every result file, so numbers can be checked without a GPU
  figures/    the three figures in the write-up
HANDOFF.md    complete record: methods, numbers, controls, rejected hypotheses,
              and the errors caught during the project
```

`HANDOFF.md` is the place to start. It marks which results are usable and which are not,
including three that should not be relied on and two claims that were made and withdrawn.

## Running it

The experiment scripts drive a remote GPU through `code/jclient.py`, a small Jupyter
REST/WebSocket client. They expect three environment variables:

```
HF_TOKEN         HuggingFace token with access to Llama and Gemma
JUPYTER_URL      https://<pod>-8888.proxy.runpod.net
JUPYTER_TOKEN    the Jupyter server token
```

`code/00_reload.py` loads the model, SAE and adapters and defines the shared helpers that
later scripts assume are in the kernel namespace.

The analysis and figure scripts (`26`, `27`, `28`, `39`) run locally against the pickles
in `results/RESULTS/` and need no GPU.

## Built on

- SelfIE, Chen, Vondrick & Mao (arXiv:2403.10949)
- Patchscopes, Ghandeharioun et al. (arXiv:2401.06102)
- Learning Self-Interpretation from Interpretability Artifacts, Pepper et al.
  (arXiv:2602.10352) - the adapters, the injection scales, and the scoring procedure
- Natural Language Autoencoders, Fraser-Taliente et al. (Transformer Circuits, 2026)
