import torch, sys, json, re, os, pickle
sys.path.insert(0, "/workspace/selfie-adapters")
from selfie_adapters.sae_utils import load_sae, ObservableLanguageModel
from selfie_adapters import load_adapter
from huggingface_hub import hf_hub_download, login

login(token=os.environ["HF_TOKEN"])

sae = load_sae("goodfire-llama-3.1-8b-instruct", "layer_19", device="cuda")
model = ObservableLanguageModel("meta-llama/Meta-Llama-3.1-8B-Instruct",
                                device="auto", dtype=torch.bfloat16)
adapter = load_adapter(hf_hub_download(
    "keenanpepper/selfie-adapters-llama-3.1-8b-instruct",
    "goodfire-sae-scalar-affine.safetensors"))
adapter_lr = load_adapter(hf_hub_download(
    "keenanpepper/selfie-adapters-llama-3.1-8b-instruct",
    "goodfire-sae-sa-lr16.safetensors"))

# ------------------------------------------------------------------ constants
SCALES = [0.5, 0.8, 1.3, 2.1, 3.4, 5.5]
LAYER, THR = 19, 0.3
RESERVED = "<|reserved_special_token_0|>"
TEMPLATE = ('<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n'
            f'What is the meaning of "{RESERVED}"?<|eot_id|>'
            '<|start_header_id|>assistant<|end_header_id|>\n\n'
            f'The meaning of "{RESERVED}" is "')
CONV_SYSTEM = ("You are a helpful AI assistant who generates EXTREMELY SHORT example "
               "conversations. The conversations are between a user and an assistant, "
               "and have the following format:\n[USER] I'm a user.\n[ASSISTANT] I'm the assistant.")
CONV_PROMPT = ("Produce a VERY SHORT conversation which exhibits '_'\n"
               "Do not include any other text in your response. Start immediately with the conversation.")

tok, hf, DEV = model.tokenizer, model._original_model, model.device
tok.pad_token = tok.pad_token or tok.eos_token
_tt = tok(TEMPLATE, return_tensors="pt", add_special_tokens=False).to(DEV)
_inject_id = tok.convert_tokens_to_ids(RESERVED)
INJECT_POS = [i for i, t in enumerate(_tt["input_ids"][0]) if t == _inject_id]
with torch.no_grad():
    TEMPLATE_EMBEDS = hf.model.embed_tokens(_tt["input_ids"])

_d = json.load(open("/workspace/selfie-adapters/data/goodfire_8b_sae_labels.json"))
LABELS_ALL = {v["index"]: v["labels"][0] for v in _d[0]["vectors"]}
VAL = {v["index"]: v["labels"][0] for v in _d[0]["vectors"] if v["split"] == "val"}


# ------------------------------------------------------------------ pipeline
@torch.no_grad()
def generate_descriptions(vectors, scale, trained=True, adpt=None, max_new=30, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    a = adpt if adpt is not None else adapter
    v = vectors.to(DEV).float()
    if v.ndim == 1:
        v = v.unsqueeze(0)
    v = v / v.norm(dim=-1, keepdim=True).clamp_min(1e-8) * scale
    soft = a.transform(v, normalize_input=False) if trained else v
    soft = soft.to(dtype=TEMPLATE_EMBEDS.dtype, device=DEV)
    emb = TEMPLATE_EMBEDS.expand(soft.shape[0], -1, -1).clone()
    for p in INJECT_POS:
        emb[:, p, :] = soft
    attn = torch.ones(emb.shape[:2], dtype=torch.long, device=DEV)
    out = hf.generate(inputs_embeds=emb, attention_mask=attn, max_new_tokens=max_new,
                      do_sample=True, temperature=0.7, top_p=0.9,
                      pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
    r = []
    for i in range(soft.shape[0]):
        t_ = tok.decode(out[i], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        r.append(t_.rsplit('"', 1)[0] if '"' in t_ else t_)
    return r


@torch.no_grad()
def resid_at_layer(input_ids, layer=LAYER):
    return hf(input_ids=input_ids, output_hidden_states=True).hidden_states[layer + 1]


def parse_meta_conversation(text):
    parts = []
    for line in text.split("\n"):
        parts.extend(re.split(r"(\[(?:USER|ASSISTANT)\])", line))
    conv, role, buf = [], None, []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        m = re.match(r"\[(USER|ASSISTANT)\]", p)
        if m:
            if role and buf:
                cc = " ".join(buf).strip()
                if cc:
                    conv.append({"role": "user" if role == "USER" else "assistant", "content": cc})
            role, buf = m.group(1), []
        else:
            if role is None:
                role = "USER" if not conv else ("ASSISTANT" if conv[-1]["role"] == "user" else "USER")
                buf = [p]
            else:
                buf.append(p)
    if role and buf:
        cc = " ".join(buf).strip()
        if cc:
            conv.append({"role": "user" if role == "USER" else "assistant", "content": cc})
    if not conv and text.strip():
        conv = [{"role": "assistant", "content": text}]
    return conv


@torch.no_grad()
def score_label(label, latent_indices, n=10, return_texts=False):
    msgs = [{"role": "system", "content": CONV_SYSTEM},
            {"role": "user", "content": CONV_PROMPT.replace("_", label)}]
    tok.padding_side = "left"
    enc = tok.apply_chat_template([msgs] * n, tokenize=True, add_generation_prompt=True,
                                  return_tensors="pt", padding=True, return_dict=True).to(DEV)
    gen = hf.generate(**enc, max_new_tokens=100, do_sample=True, temperature=0.7, top_p=0.9,
                      pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
    texts = [tok.decode(g[enc["input_ids"].shape[1]:], skip_special_tokens=True).strip() for g in gen]
    hits = {li: 0 for li in latent_indices}
    valid = 0
    for t in texts:
        conv = parse_meta_conversation(t)
        if not conv:
            continue
        ids = tok.apply_chat_template(conv, tokenize=True, add_generation_prompt=False,
                                      return_tensors="pt", return_dict=True)["input_ids"].to(DEV)
        h = resid_at_layer(ids).to(sae.W_enc.device, sae.W_enc.dtype)
        acts = sae.encode(h)[0]
        valid += 1
        for li in latent_indices:
            if (acts[1:, li] > 0).any().item():
                hits[li] += 1
    res = {li: (hits[li] / valid if valid else 0.0) for li in latent_indices}
    return (res, texts) if return_texts else res


def compose(a_idx, b_idx, alpha):
    dA = sae.W_dec[a_idx].detach(); dB = sae.W_dec[b_idx].detach()
    dA = dA / dA.norm(); dB = dB / dB.norm()
    v = alpha * dA + (1 - alpha) * dB
    return v / v.norm().clamp_min(1e-9)


PAIRS = [("cooking x consumer-law", 12201, 16864, "neutral"),
         ("baking x legalese", 11970, 45010, "neutral"),
         ("spices x criminal-defense", 21592, 1755, "neutral"),
         ("baking x EXTREMISM", 11970, 56450, "safety"),
         ("cooking x SCAM-FRAUD", 12201, 6214, "safety")]

print("RELOAD OK | d_sae", sae.cfg.d_sae, "| d_model", model.hidden_size,
      "| adapter", adapter.get_metadata()["num_parameters"],
      "| adapter_lr", adapter_lr.get_metadata()["num_parameters"],
      "| inject_pos", INJECT_POS)
