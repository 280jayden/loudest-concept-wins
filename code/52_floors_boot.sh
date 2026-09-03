#!/usr/bin/env bash
# After a pod restart: reinstall deps (container disk resets) and launch the 240-floor script.
set -euo pipefail
cd /workspace/MATS-project && git pull -q && git log --oneline -1
pip install -q -r /workspace/selfie-adapters/requirements.txt
pip install -q "transformers==4.55.0" accelerate safetensors huggingface_hub hf_transfer
python -c "import transformers, torch; print('transformers', transformers.__version__, '| cuda', torch.cuda.is_available())"
mkdir -p /workspace/li /workspace/RESULTS
cd /workspace && setsid nohup env LI_EXTRA_CACHE=/root/hf_extra python /workspace/MATS-project/code/51_floors240.py > /workspace/li/floors240.nohup 2>&1 < /dev/null &
sleep 5; pgrep -af 51_floors240 | grep -v pgrep || echo "not running"
