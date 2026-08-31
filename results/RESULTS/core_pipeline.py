"""
MATS project - core pipeline. Concept masking in trained self-interpretation.
Model: Llama-3.1-8B-Instruct | SAE: goodfire-llama-3.1-8b-instruct layer_19
Adapters: goodfire-sae-scalar-affine.safetensors, goodfire-sae-sa-lr16.safetensors
"""
import torch, re, json, pickle, os
import sys; sys.path.insert(0,"/workspace/selfie-adapters")
from selfie_adapters.sae_utils import load_sae, ObservableLanguageModel
from selfie_adapters import load_adapter
from huggingface_hub import hf_hub_download, login

SCALES=[0.5,0.8,1.3,2.1,3.4,5.5]; LAYER=19; THR=0.3
RESERVED="<|reserved_special_token_0|>"
TEMPLATE=('<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n'
          f'What is the meaning of "{RESERVED}"?<|eot_id|>'
          '<|start_header_id|>assistant<|end_header_id|>\n\n'
          f'The meaning of "{RESERVED}" is "')
CONV_SYSTEM=("You are a helpful AI assistant who generates EXTREMELY SHORT example conversations. "
             "The conversations are between a user and an assistant, and have the following format:\n"
             "[USER] I'm a user.\n[ASSISTANT] I'm the assistant.")
CONV_PROMPT=("Produce a VERY SHORT conversation which exhibits '_'\n"
             "Do not include any other text in your response. Start immediately with the conversation.")

def setup(hf_token=None):
    if hf_token: login(token=hf_token)
    sae = load_sae("goodfire-llama-3.1-8b-instruct","layer_19",device="cuda")
    model = ObservableLanguageModel("meta-llama/Meta-Llama-3.1-8B-Instruct",device="auto",dtype=torch.bfloat16)
    ad = load_adapter(hf_hub_download("keenanpepper/selfie-adapters-llama-3.1-8b-instruct",
                                      "goodfire-sae-scalar-affine.safetensors"))
    return sae, model, ad

# KEY DETAIL: normalize -> scale -> adapter.transform(normalize_input=False)
#   (the adapter would otherwise re-normalize and cancel the scale)
# KEY DETAIL: compose() re-normalizes so alpha controls ratio only, not magnitude
def compose(sae, a_idx, b_idx, alpha):
    dA=sae.W_dec[a_idx].detach(); dB=sae.W_dec[b_idx].detach()
    dA=dA/dA.norm(); dB=dB/dB.norm()
    v=alpha*dA+(1-alpha)*dB
    return v/v.norm().clamp_min(1e-9)

# hit = any post-BOS SAE activation > 0 (matches evals/generation_scoring/compute_mean_max_hit_rate.py)
# scoring: description -> N synthetic conversations -> forward pass -> sae.encode -> check latent
# full implementations in full_session_code.py
