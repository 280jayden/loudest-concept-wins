"""
Gemma Scope 2 SAE, layer 32 residual stream - matches kitft/nla-gemma3-12b-L32.

Getting this working is the make-or-break for the Gemma NLA experiment: it is what
gives us clean, labelled concept directions (the thing the Qwen attempt lacked).
Download it and verify BEFORE pulling 40GB of NLA weights.
"""
import os, json, torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file

REPO = "google/gemma-scope-2-12b-it"
SAE_PATH = "resid_post_all/layer_32_width_16k_l0_big"

d = snapshot_download(REPO, allow_patterns=[f"{SAE_PATH}/*"])
base = os.path.join(d, SAE_PATH)
print("files:", os.listdir(base))
for f in os.listdir(base):
    print(f"  {f}: {os.path.getsize(os.path.join(base,f))/1e6:.1f} MB")

cfg = json.load(open(os.path.join(base, "config.json")))
print("\nconfig:", json.dumps(cfg, indent=2)[:900])

params = load_file(os.path.join(base, "params.safetensors"))
print("\nparam tensors:")
for k, v in params.items():
    print(f"  {k:16} {tuple(v.shape)}  {v.dtype}")

ex_path = os.path.join(base, "examples.safetensors")
if os.path.exists(ex_path):
    ex = load_file(ex_path)
    print("\nexamples tensors (candidate source of feature meanings):")
    for k, v in ex.items():
        print(f"  {k:24} {tuple(v.shape)}  {v.dtype}")
