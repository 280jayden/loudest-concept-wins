# MATS project - resume notes

## What is here
- HEADLINE_NUMBERS.json  : every key result, plain JSON
- full_session_code.py   : all 92 executed cells, in order (the real record)
- core_pipeline.py       : the critical constants + two easy-to-get-wrong details
- *.pkl                  : raw results for all 8 experiments

## Persistence
/workspace = RunPod network volume -> persists across Stop/Start (incl. 19GB HF cache)
/          = container overlay -> pip packages may need reinstalling:
   pip install -r /workspace/selfie-adapters/requirements.txt
   pip install jupyterlab jupyter-collaboration jupyter-mcp-tools ipykernel jupyter-mcp-server

## Two things that are easy to get wrong
1. adapter.transform(v, normalize_input=False) -- else the injection scale is cancelled
2. compose() must re-normalize -- else 50/50 mixes are ~30% quieter than pure endpoints
   (sqrt(a^2+(1-a)^2)=0.707 at a=0.5), which mimics a masking effect

## Still open
- SA+LR sweep ~70/120 cells (resumes automatically; partial already conclusive: 0/54 at alpha=0.75)
- Optional: llamascope-sae-scalar-affine.safetensors = different SAE, same model (cross-SAE test)
- Not done: figures, write-up
