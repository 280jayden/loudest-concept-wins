#!/usr/bin/env bash
# Run after every pod (re)start: the container disk resets, so pip packages and the extra HF cache are gone.
# Idempotent. Then launches the pipeline (resumable) with self-stop enabled.
set -euo pipefail
cd /workspace/MATS-project && git pull -q && git log --oneline -1
pip install -q -r /workspace/selfie-adapters/requirements.txt
pip install -q "transformers==4.55.0" accelerate safetensors huggingface_hub hf_transfer peft
python -c "import transformers, torch; print('transformers', transformers.__version__, '| cuda', torch.cuda.is_available())"
mkdir -p /workspace/li /workspace/RESULTS
touch /workspace/li/KEEP                      # tell any crashed-and-holding pipeline not to stop the pod
pkill -f 43_li_pipeline.py || true
sleep 2
rm -f /workspace/li/KEEP
cd /workspace && setsid nohup env LI_STOP_POD=1 LI_EXTRA_CACHE=/root/hf_extra \
  python /workspace/MATS-project/code/43_li_pipeline.py > /workspace/li/nohup.log 2>&1 < /dev/null &
sleep 5; pgrep -af 43_li_pipeline | grep -v pgrep || echo "pipeline not running"
