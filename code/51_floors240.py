"""
Random-direction floors at 20 directions x 12 latents = 240 per method, for both SAEs.

Part A, Llama Scope: the 20 random-direction descriptions per method already exist in
/workspace/li/floors.pkl (labels). They are re-scored against 12 latents: the 9 latents of the
final pair set plus 3 further concepts that passed gate 1 under both methods (lowest ids).
Same scorer as the main run: Instruct writes 10 conversations, base Llama forward, Llama Scope
L19 encode, latent fires if > 0 on any post-BOS token, hit at >= 0.3.

Part B, Goodfire: 20 random unit directions (seeds 1000+k) -> scalar-affine and rank-16 adapter
descriptions at scale 1.0 (the magnitude of the main tables) and 2.1 (the scale of the earlier
12 x 12 check) -> scored against the 12 Goodfire target latents of the main experiment with the
Goodfire scorer (Instruct writes, Instruct forward, Goodfire L19 encode, same hit rule).

Writes /workspace/li/floors240.pkl and copies to /workspace/RESULTS. Does not stop the pod.
"""
import os, sys, json, pickle, re, time, subprocess
import torch

W = "/workspace/li"
XC = os.environ.get("LI_EXTRA_CACHE", "/root/hf_extra")
LOG = open(f"{W}/floors240.log", "a")


def log(*a):
    s = time.strftime("%H:%M:%S ") + " ".join(str(x) for x in a)
    print(s, flush=True); LOG.write(s + "\n"); LOG.flush()


if not os.environ.get("HF_TOKEN") and os.path.exists("/workspace/.hf_token"):
    os.environ["HF_TOKEN"] = open("/workspace/.hf_token").read().strip()
from huggingface_hub import login, hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import load_file
login(token=os.environ["HF_TOKEN"])
sys.path.insert(0, "/workspace/selfie-adapters")
from selfie_adapters import load_adapter

DEV = "cuda"
THR = 0.3
LAYER = 19
CONV_SYSTEM = ("You are a helpful AI assistant who generates EXTREMELY SHORT example conversations. "
               "The conversations are between a user and an assistant, and have the following format:\n"
               "[USER] I'm a user.\n[ASSISTANT] I'm the assistant.")
CONV_PROMPT = ("Produce a VERY SHORT conversation which exhibits '_'\n"
               "Do not include any other text in your response. Start immediately with the conversation.")

log("=== load ===")
itok = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
itok.pad_token = itok.pad_token or itok.eos_token
inst = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct", torch_dtype=torch.bfloat16, device_map=DEV).eval()
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
tok.pad_token = tok.pad_token or tok.eos_token
base = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B", torch_dtype=torch.bfloat16, device_map=DEV).eval()
log("Instruct + base loaded")


def parse_conv(text):
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
                c = " ".join(buf).strip()
                if c:
                    conv.append({"role": "user" if role == "USER" else "assistant", "content": c})
            role, buf = m.group(1), []
        else:
            if role is None:
                role = "USER" if not conv else ("ASSISTANT" if conv[-1]["role"] == "user" else "USER")
                buf = [p]
            else:
                buf.append(p)
    if role and buf:
        c = " ".join(buf).strip()
        if c:
            conv.append({"role": "user" if role == "USER" else "assistant", "content": c})
    if not conv and text.strip():
        conv = [{"role": "assistant", "content": text}]
    return conv


@torch.no_grad()
def write_convs(descs, n=10, chunk=120):
    msgs_all = []
    for d_ in descs:
        m = [{"role": "system", "content": CONV_SYSTEM}, {"role": "user", "content": CONV_PROMPT.replace("_", d_[:400])}]
        msgs_all.extend([m] * n)
    itok.padding_side = "left"
    texts = []
    for s0 in range(0, len(msgs_all), chunk):
        enc = itok.apply_chat_template(msgs_all[s0:s0 + chunk], tokenize=True, add_generation_prompt=True,
                                       return_tensors="pt", padding=True, return_dict=True).to(DEV)
        gen = inst.generate(**enc, max_new_tokens=100, do_sample=True, temperature=0.7, top_p=0.9,
                            pad_token_id=itok.pad_token_id, eos_token_id=itok.eos_token_id)
        L0 = enc["input_ids"].shape[1]
        texts += [itok.decode(g[L0:], skip_special_tokens=True).strip() for g in gen]
    return [texts[i * n:(i + 1) * n] for i in range(len(descs))]


# ----------------------------------------------------------------------------- Part A: Llama Scope
log("=== Part A: Llama Scope floors at 20 x 12 ===")
S = load_file(hf_hub_download("fnlp/Llama3_1-8B-Base-LXR-32x", "Llama3_1-8B-Base-L19R-32x/checkpoints/final.safetensors", cache_dir=XC))
W_ENC = S["encoder.weight"].float().to(DEV); B_ENC = S["encoder.bias"].float().to(DEV)
K, JUMP = 64 / 17.125, 0.484375


@torch.no_grad()
def ls_encode(x):
    pre = x.float().to(DEV) * K @ W_ENC.T + B_ENC
    return pre * (pre > JUMP)


@torch.no_grad()
def ls_score_many(descs, latents, n=10):
    convs = write_convs(descs, n)
    out = []
    for cl in convs:
        hits, valid = {li: 0 for li in latents}, 0
        for t_ in cl:
            conv = parse_conv(t_)
            if not conv:
                continue
            text = "\n".join(f"{c['role']}: {c['content']}" for c in conv)
            ids = tok(text, return_tensors="pt", truncation=True, max_length=200)["input_ids"].to(DEV)
            h = base(input_ids=ids, output_hidden_states=True).hidden_states[LAYER + 1][0]
            acts = ls_encode(h); valid += 1
            for li in latents:
                if (acts[1:, li] > 0).any().item():
                    hits[li] += 1
        out.append({li: (hits[li] / valid if valid else 0.0) for li in latents})
    return out


FL = pickle.load(open(f"{W}/floors.pkl", "rb"))
G1 = pickle.load(open(f"{W}/gate1.pkl", "rb"))
pairs = json.load(open(f"{W}/pairs_final.json"))["pairs"]
pair_lat = sorted({p["A"] for p in pairs} | {p["B"] for p in pairs})
extra = [i for i in sorted(G1) if G1[i]["li_pass"] and G1[i]["ad_pass"] and i not in pair_lat][:12 - len(pair_lat)]
targets = pair_lat + extra
log(f"Llama Scope targets ({len(targets)}): pair latents {pair_lat} + extra {extra}")
res = {"ls_targets": targets, "ls": {}}
for tag in ["li", "adapter", "adapter64"]:
    descs = [FL[(tag, k)]["label"] for k in range(20)]
    sc = ls_score_many(descs, targets)
    res["ls"][tag] = [{"label": d_, "hits": s} for d_, s in zip(descs, sc)]
    fp = sum(1 for r in res["ls"][tag] for h in r["hits"].values() if h >= THR)
    log(f"  floor240 [{tag}]: {fp}/{20 * len(targets)} false positives")
    pickle.dump(res, open(f"{W}/floors240.pkl", "wb"))
del S, W_ENC, B_ENC, base; torch.cuda.empty_cache()

# ----------------------------------------------------------------------------- Part B: Goodfire
log("=== Part B: Goodfire floors at 20 x 12 ===")
from selfie_adapters.sae_utils import load_sae
sae = load_sae("goodfire-llama-3.1-8b-instruct", "layer_19", device=DEV)
GF_TARGETS = [12201, 16864, 11970, 45010, 21592, 1755, 56450, 6214, 10084, 53180, 25528, 773]
RESERVED = "<|reserved_special_token_0|>"
TEMPLATE = ('<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n'
            f'What is the meaning of "{RESERVED}"?<|eot_id|>'
            '<|start_header_id|>assistant<|end_header_id|>\n\n'
            f'The meaning of "{RESERVED}" is "')
_tt = itok(TEMPLATE, return_tensors="pt", add_special_tokens=False).to(DEV)
_inj = itok.convert_tokens_to_ids(RESERVED)
POS = [i for i, t_ in enumerate(_tt["input_ids"][0]) if t_ == _inj]
assert len(POS) == 2
with torch.no_grad():
    TEMB = inst.model.embed_tokens(_tt["input_ids"])
_cap = {}
inst.model.layers[LAYER].register_forward_hook(lambda m, i, o: _cap.__setitem__("h", o[0] if isinstance(o, tuple) else o))


@torch.no_grad()
def gf_generate(vec, ad, scale, seed):
    torch.manual_seed(seed)
    v = vec.to(DEV).float().unsqueeze(0); v = v / v.norm(dim=-1, keepdim=True) * scale
    soft = ad.transform(v, normalize_input=False).to(dtype=TEMB.dtype, device=DEV)
    emb = TEMB.clone()
    for p in POS:
        emb[:, p, :] = soft
    out = inst.generate(inputs_embeds=emb, attention_mask=torch.ones(emb.shape[:2], dtype=torch.long, device=DEV),
                        max_new_tokens=30, do_sample=True, temperature=0.7, top_p=0.9,
                        pad_token_id=itok.pad_token_id, eos_token_id=itok.eos_token_id)
    t_ = itok.decode(out[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return (t_.rsplit('"', 1)[0] if '"' in t_ else t_).strip()


@torch.no_grad()
def gf_score_many(descs, latents, n=10):
    convs = write_convs(descs, n)
    out = []
    for cl in convs:
        hits, valid = {li: 0 for li in latents}, 0
        for t_ in cl:
            conv = parse_conv(t_)
            if not conv:
                continue
            ids = itok.apply_chat_template(conv, tokenize=True, add_generation_prompt=False, return_tensors="pt", return_dict=True)["input_ids"].to(DEV)
            inst(input_ids=ids)
            h = _cap["h"].to(device=sae.W_enc.device, dtype=sae.W_enc.dtype)
            acts = sae.encode(h)[0]; valid += 1
            for li in latents:
                if (acts[1:, li] > 0).any().item():
                    hits[li] += 1
        out.append({li: (hits[li] / valid if valid else 0.0) for li in latents})
    return out


res["gf_targets"] = GF_TARGETS; res["gf"] = {}
for name, fn in [("sa", "goodfire-sae-scalar-affine.safetensors"), ("lr16", "goodfire-sae-sa-lr16.safetensors")]:
    ad = load_adapter(hf_hub_download("keenanpepper/selfie-adapters-llama-3.1-8b-instruct", fn, cache_dir=XC))
    for scale in [1.0, 2.1]:
        descs = []
        for k in range(20):
            torch.manual_seed(1000 + k)
            rv = torch.randn(4096, device=DEV); rv = rv / rv.norm()
            descs.append(gf_generate(rv, ad, scale, seed=5000 + k))
        sc = gf_score_many(descs, GF_TARGETS)
        res["gf"][(name, scale)] = [{"label": d_, "hits": s} for d_, s in zip(descs, sc)]
        fp = sum(1 for r in res["gf"][(name, scale)] for h in r["hits"].values() if h >= THR)
        log(f"  floor240 goodfire [{name} @ {scale}x]: {fp}/240 false positives")
        pickle.dump(res, open(f"{W}/floors240.pkl", "wb"))

subprocess.run(f"cp {W}/floors240.pkl {W}/floors240.log /workspace/RESULTS/", shell=True)
log("DONE")
