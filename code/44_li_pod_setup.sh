#!/usr/bin/env bash
# Pod setup for the Llama Scope arm (Li explainer + Pepper llamascope adapter).
# Run once on a fresh RunPod pod with an 80 GB card and a /workspace volume.
#   HF_TOKEN=... bash 44_li_pod_setup.sh
set -euo pipefail
[ -z "${HF_TOKEN:-}" ] && [ -f /workspace/.hf_token ] && export HF_TOKEN="$(cat /workspace/.hf_token)"
: "${HF_TOKEN:?set HF_TOKEN or write it to /workspace/.hf_token}"

cd /workspace
mkdir -p /workspace/li /workspace/RESULTS

echo "== repos =="
[ -d selfie-adapters ]       || git clone -q https://github.com/agencyenterprise/selfie-adapters.git
[ -d introspective-interp ]  || git clone -q --depth 1 https://github.com/TransluceAI/introspective-interp.git
[ -d MATS-project ]          || git clone -q https://github.com/280jayden/loudest-concept-wins.git MATS-project
(cd MATS-project && git pull -q)

echo "== python deps =="
pip install -q -r selfie-adapters/requirements.txt
pip install -q "transformers==4.55.0" accelerate safetensors huggingface_hub   # Transluce config was saved with 4.55.0
python - <<'EOF'
import transformers, torch; print("transformers", transformers.__version__, "| torch", torch.__version__, "| cuda", torch.cuda.is_available())
EOF

echo "== hf login + downloads (~50 GB) =="
python - <<'EOF'
import os
from huggingface_hub import login, snapshot_download, hf_hub_download
login(token=os.environ["HF_TOKEN"])
for r in ["meta-llama/Llama-3.1-8B", "meta-llama/Meta-Llama-3.1-8B-Instruct",
          "Transluce/features_explain_llama3.1_8b_llama3.1_8b"]:
    p = snapshot_download(r, allow_patterns=["*.json", "*.safetensors", "tokenizer*"]); print("ok", r)
hf_hub_download("fnlp/Llama3_1-8B-Base-LXR-32x", "Llama3_1-8B-Base-L19R-32x/checkpoints/final.safetensors"); print("ok llama scope L19")
hf_hub_download("keenanpepper/selfie-adapters-llama-3.1-8b-instruct", "llamascope-sae-scalar-affine.safetensors"); print("ok adapter")
EOF

echo "== shutdown path =="
which runpodctl && echo "RUNPOD_POD_ID=${RUNPOD_POD_ID:-MISSING}" || echo "WARNING: runpodctl not found; pod will not self-stop"

echo "== candidates =="
ls -la /workspace/MATS-project/results/RESULTS/np_candidates_L19_131k.json
df -h /workspace | tail -1
echo "setup done. run:  cd /workspace && HF_TOKEN=\$HF_TOKEN LI_STOP_POD=1 nohup python MATS-project/code/43_li_pipeline.py > /workspace/li/nohup.log 2>&1 &"
