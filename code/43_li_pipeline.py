"""
Llama Scope arm: Li et al. explainer and Pepper's Llama Scope adapter, one pipeline.

Resumable. Every stage checkpoints to /workspace/li/*.pkl and skips finished cells, so
the script can be restarted after any interruption and picks up where it stopped. The
last stage copies results to /workspace/RESULTS and, if LI_STOP_POD=1, stops the pod.

Run on the pod:
    HF_TOKEN=... LI_STOP_POD=1 nohup python 43_li_pipeline.py > /workspace/li/run.log 2>&1 &

Stages (in order)
  0  environment checks: runpodctl present, HF token, disk
  1  load: base Llama-3.1-8B, Instruct, Llama Scope L19, Li explainer, Pepper llamascope
     scalar-affine adapter
  2  pre-flight: SAE L0 on a real layer-19 residual; each method on one known feature;
     magnitude sweep for the explainer (also gives the describability threshold)
  3  gates 2+3 from the Neuronpedia candidates, then gate 1 under BOTH methods; keep the intersection
  4  Li sweep: 6 shares x 12 pairs x 20 sampled descriptions, plus 1 greedy per cell,
     plus the list prompt at 50% and 25%
  5  score everything with the one scorer
  6  adapter sweep, same pairs, same scorer
  7  floors: 20 random directions x 12 latents under each method
  7b rank-64 adapter (scalar-affine + low-rank, 528,385 params): gate-1 record, sweep, list, floor
  8  save, copy to RESULTS, stop pod (also on crash, via sys.excepthook)

The scorer is identical for both methods: Instruct writes ten short conversations from
the description, base Llama runs them forward, Llama Scope encodes layer 19, the target
latent counts as fired if it is > 0 on any post-BOS token; a description is a hit at
>= 0.3 of the ten.
"""
import os, sys, json, pickle, hashlib, time, random, subprocess, re
from collections import Counter
import torch

W = "/workspace/li"
os.makedirs(W, exist_ok=True)
LOG = open(f"{W}/run.log", "a")


def log(*a):
    s = time.strftime("%H:%M:%S ") + " ".join(str(x) for x in a)
    print(s, flush=True); LOG.write(s + "\n"); LOG.flush()


def ck(name):
    p = f"{W}/{name}.pkl"
    return pickle.load(open(p, "rb")) if os.path.exists(p) else {}


def save(name, obj):
    pickle.dump(obj, open(f"{W}/{name}.pkl", "wb"))


def seed_of(*p):
    return int(hashlib.md5("|".join(map(str, p)).encode()).hexdigest()[:6], 16)


# ----------------------------------------------------------------------------- 0 env
log("=== stage 0: environment ===")
if not os.environ.get("HF_TOKEN") and os.path.exists("/workspace/.hf_token"):
    os.environ["HF_TOKEN"] = open("/workspace/.hf_token").read().strip()
assert os.environ.get("HF_TOKEN"), "HF_TOKEN not set and /workspace/.hf_token missing"
POD_ID = os.environ.get("RUNPOD_POD_ID", "")
HAS_RUNPODCTL = subprocess.run("which runpodctl", shell=True, capture_output=True).returncode == 0
log(f"runpodctl present: {HAS_RUNPODCTL}   RUNPOD_POD_ID: {POD_ID or 'MISSING'}")


def stop_pod(reason):
    """Copy everything out, then STOP (never terminate) the pod if LI_STOP_POD=1."""
    subprocess.run(f"mkdir -p /workspace/RESULTS && cp {W}/*.pkl {W}/*.json {W}/run.log /workspace/RESULTS/ 2>/dev/null", shell=True)
    if os.environ.get("LI_STOP_POD") != "1":
        log(f"[{reason}] LI_STOP_POD unset; pod left running"); return
    if HAS_RUNPODCTL and POD_ID:
        log(f"[{reason}] stopping pod {POD_ID} (not terminating)"); LOG.flush()
        subprocess.run(f"runpodctl stop pod {POD_ID}", shell=True)
    else:
        log(f"[{reason}] LI_STOP_POD=1 but runpodctl or RUNPOD_POD_ID missing; pod left running")


def _hook(t, v, tb):
    import traceback
    log("UNHANDLED EXCEPTION:\n" + "".join(traceback.format_exception(t, v, tb)))
    subprocess.run(f"mkdir -p /workspace/RESULTS && cp {W}/*.pkl {W}/*.json {W}/run.log /workspace/RESULTS/ 2>/dev/null", shell=True)
    # A stopped pod may not get its GPU back. Hold the pod for 40 min so a fix can be launched
    # (the relauncher touches /workspace/li/KEEP); stop only if nobody resumed.
    log("crash: holding pod 40 min for a resume; touch /workspace/li/KEEP to keep it up")
    for _ in range(40):
        time.sleep(60)
        if os.path.exists(f"{W}/KEEP"):
            log("KEEP present; leaving pod running"); return
    stop_pod("crash")


sys.excepthook = _hook
log(subprocess.run("df -h /workspace | tail -1", shell=True, capture_output=True, text=True).stdout.strip())
log(subprocess.run("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader", shell=True,
                   capture_output=True, text=True).stdout.strip())

# gates 2 and 3 run here, on the pod, from the Neuronpedia candidate list in the repo
CANDS_JSON = os.environ.get("LI_CANDS", "/workspace/MATS-project/results/RESULTS/np_candidates_L19_131k.json")
PAIRS_FILE = f"{W}/li_pairs_candidates.json"
# the network volume (/workspace) is ~50 GB and the three models fill it; SAE + adapters go to a second cache
XC = os.environ.get("LI_EXTRA_CACHE", "/root/hf_extra")
if not os.path.exists(PAIRS_FILE):
    from huggingface_hub import hf_hub_download as _dl
    _sae = _dl("fnlp/Llama3_1-8B-Base-LXR-32x", "Llama3_1-8B-Base-L19R-32x/checkpoints/final.safetensors", cache_dir=XC)
    r = subprocess.run(f"python /workspace/MATS-project/code/42_li_local_gates.py --sae {_sae} "
                       f"--cands {CANDS_JSON} --out {PAIRS_FILE} --per_family 4",
                       shell=True, capture_output=True, text=True)
    log(r.stdout[-3000:]); assert r.returncode == 0, r.stderr[-2000:]
CAND = json.load(open(PAIRS_FILE))
log(f"gates 2+3 candidate pairs: {len(CAND['pairs'])}")

# ----------------------------------------------------------------------------- 1 load
log("=== stage 1: load ===")
from huggingface_hub import login, hf_hub_download, snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import load_file
login(token=os.environ["HF_TOKEN"])
sys.path.insert(0, "/workspace/introspective-interp")
sys.path.insert(0, "/workspace/selfie-adapters")
from model.continuous_llama import ContinuousLlama
from selfie_adapters import load_adapter

DEV = "cuda"
LAYER, THR, K = 19, 0.3, 64 / 17.125
JUMP = 0.484375

tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
tok.pad_token = tok.pad_token or tok.eos_token
base = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B", torch_dtype=torch.bfloat16,
                                            device_map=DEV).eval()
itok = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
itok.pad_token = itok.pad_token or itok.eos_token
inst = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct",
                                            torch_dtype=torch.bfloat16, device_map=DEV).eval()
log("base + instruct loaded")

sae_path = hf_hub_download("fnlp/Llama3_1-8B-Base-LXR-32x",
                           "Llama3_1-8B-Base-L19R-32x/checkpoints/final.safetensors", cache_dir=XC)
S = load_file(sae_path)
W_DEC = S["decoder.weight"].float().to(DEV)            # (4096, 131072)
W_ENC = S["encoder.weight"].float().to(DEV)            # (131072, 4096)
B_ENC = S["encoder.bias"].float().to(DEV)
NORMS = W_DEC.norm(dim=0)
log(f"Llama Scope L19 loaded; decoder col norms mean {NORMS.mean():.3f}")


@torch.no_grad()
def sae_encode(x):
    """x: (..., 4096) residual-space vectors. Returns JumpReLU activations."""
    pre = x.float().to(DEV) * K @ W_ENC.T + B_ENC
    return pre * (pre > JUMP)


def unit(i):
    return W_DEC[:, i] / NORMS[i]


SPECIAL = {"begin_continuous": tok.convert_tokens_to_ids("<|reserved_special_token_10|>"),
           "continuous_rep": tok.convert_tokens_to_ids("<|reserved_special_token_12|>"),
           "end_continuous": tok.convert_tokens_to_ids("<|reserved_special_token_11|>")}
expl = ContinuousLlama.from_pretrained("Transluce/features_explain_llama3.1_8b_llama3.1_8b",
                                       torch_dtype=torch.bfloat16, device_map=DEV,
                                       special_tokens_ids=SPECIAL, use_embed_proj=False).eval()
log("Li explainer loaded")

adapter = load_adapter(hf_hub_download("keenanpepper/selfie-adapters-llama-3.1-8b-instruct",
                                       "llamascope-sae-scalar-affine.safetensors", cache_dir=XC))
log("Pepper llamascope scalar-affine adapter loaded:", adapter.get_metadata())

# ---- prompts, exactly as each method was trained
LI_PROMPT = ("At layer 19, <|reserved_special_token_10|><|reserved_special_token_12|>"
             "<|reserved_special_token_11|> encodes ")
if tok.chat_template:
    LI_TEXT = tok.apply_chat_template([{"role": "user", "content": LI_PROMPT}], tokenize=False) \
              + "<|start_header_id|>assistant<|end_header_id|>\n\n"
    LI_IDS = tok(LI_TEXT, return_tensors="pt", add_special_tokens=False)["input_ids"].to(DEV)
else:
    LI_IDS = tok(LI_PROMPT, return_tensors="pt", add_special_tokens=True)["input_ids"].to(DEV)
assert (LI_IDS[0] == SPECIAL["continuous_rep"]).sum().item() == 1
LI_STOP = [" [END]", "\n", ">>>."]
log("Li prompt tokens:", LI_IDS.shape[1], "| chat template used:", bool(tok.chat_template))

LI_LIST_PROMPT = ("At layer 19, <|reserved_special_token_10|><|reserved_special_token_12|>"
                  "<|reserved_special_token_11|> encodes the following distinct concepts: 1.")

RESERVED = "<|reserved_special_token_0|>"
AD_TEMPLATE = ('<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n'
               f'What is the meaning of "{RESERVED}"?<|eot_id|>'
               '<|start_header_id|>assistant<|end_header_id|>\n\n'
               f'The meaning of "{RESERVED}" is "')
_tt = itok(AD_TEMPLATE, return_tensors="pt", add_special_tokens=False).to(DEV)
_inj = itok.convert_tokens_to_ids(RESERVED)
AD_POS = [i for i, t_ in enumerate(_tt["input_ids"][0]) if t_ == _inj]
with torch.no_grad():
    AD_EMB = inst.model.embed_tokens(_tt["input_ids"])
AD_LIST = ('<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n'
           f'List every distinct concept present in "{RESERVED}". Give a short numbered list, '
           'one concept per line.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'
           f'The distinct concepts present in "{RESERVED}" are:\n1.')
_lt = itok(AD_LIST, return_tensors="pt", add_special_tokens=False).to(DEV)
AD_LIST_POS = [i for i, t_ in enumerate(_lt["input_ids"][0]) if t_ == _inj]
with torch.no_grad():
    AD_LIST_EMB = inst.model.embed_tokens(_lt["input_ids"])


# ----------------------------------------------------------------------------- generators
@torch.no_grad()
def gen_li(vec, n, mag, seed=None, greedy=False, list_prompt=False, max_new=30):
    """Li explainer. vec: unit direction (4096). Injected at magnitude `mag` (residual units)."""
    if seed is not None:
        torch.manual_seed(seed)
    if list_prompt:
        text = (tok.apply_chat_template([{"role": "user", "content": LI_LIST_PROMPT}], tokenize=False)
                + "<|start_header_id|>assistant<|end_header_id|>\n\n") if tok.chat_template else LI_LIST_PROMPT
        ids = tok(text, return_tensors="pt", add_special_tokens=not bool(tok.chat_template))["input_ids"].to(DEV)
        stop, mn = ["\n\n", ">>>."], 90
    else:
        ids, stop, mn = LI_IDS, LI_STOP, max_new
    ids = ids.expand(n, -1)
    v = (vec.to(DEV).float() * mag).to(torch.bfloat16).unsqueeze(0)      # (1, 4096)
    cont = [v.clone() for _ in range(n)]
    kw = dict(do_sample=False, temperature=None, top_p=None) if greedy else \
         dict(do_sample=True, temperature=0.7, top_p=0.9)
    out = expl.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                        inputs_continuous_tokens=cont, max_new_tokens=mn,
                        pad_token_id=tok.eos_token_id, eos_token_id=tok.eos_token_id,
                        stop_strings=stop, tokenizer=tok, **kw)
    seq = out.sequences if hasattr(out, "sequences") else out
    res = []
    for r in seq:
        t_ = tok.decode(r[ids.shape[1]:], skip_special_tokens=True)
        for s_ in stop:
            t_ = t_.split(s_)[0]
        res.append(t_.strip().strip('"'))
    return res


@torch.no_grad()
def gen_adapter(vec, n, seed=None, list_prompt=False, max_new=30, ad=None, scale=1.0):
    """Pepper llamascope adapter on Instruct. Trained at unit norm -> scale 1.0 (the sweep).
    `ad` selects the adapter object (default: scalar-affine); rank-64 uses the same path.
    `scale` != 1 is used only by the describability-threshold stage."""
    if seed is not None:
        torch.manual_seed(seed)
    ad = ad if ad is not None else adapter
    v = vec.to(DEV).float().unsqueeze(0)
    v = v / v.norm(dim=-1, keepdim=True) * scale
    soft = ad.transform(v, normalize_input=False).to(dtype=AD_EMB.dtype, device=DEV)
    emb0, pos, mn = (AD_LIST_EMB, AD_LIST_POS, 90) if list_prompt else (AD_EMB, AD_POS, max_new)
    emb = emb0.expand(n, -1, -1).clone()
    for p in pos:
        emb[:, p, :] = soft
    attn = torch.ones(emb.shape[:2], dtype=torch.long, device=DEV)
    out = inst.generate(inputs_embeds=emb, attention_mask=attn, max_new_tokens=mn,
                        do_sample=True, temperature=0.7, top_p=0.9,
                        pad_token_id=itok.pad_token_id, eos_token_id=itok.eos_token_id)
    res = []
    for r in out:
        t_ = itok.decode(r, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        res.append((t_.rsplit('"', 1)[0] if '"' in t_ and not list_prompt else t_).strip())
    return res


# ----------------------------------------------------------------------------- scorer (one for all)
CONV_SYSTEM = ("You are a helpful AI assistant who generates EXTREMELY SHORT example conversations. "
               "The conversations are between a user and an assistant, and have the following format:\n"
               "[USER] I'm a user.\n[ASSISTANT] I'm the assistant.")
CONV_PROMPT = ("Produce a VERY SHORT conversation which exhibits '_'\n"
               "Do not include any other text in your response. Start immediately with the conversation.")


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
def score_many(descs, latents, n=10, chunk=120):
    """Batched scorer, one metric for every arm.
    description -> Instruct writes n conversations -> base Llama forward -> Llama Scope L19;
    the target latent counts as fired if > 0 on any post-BOS token; hit rate = fired / n valid.
    Batching only changes how many conversations share a generate call; prompts, sampling
    (T 0.7, top-p 0.9, 100 new tokens) and the forward/encode are identical per conversation."""
    out = [None] * len(descs)
    idx = []
    for i, d_ in enumerate(descs):
        if d_:
            idx.append(i)
        else:
            out[i] = {li: 0.0 for li in latents}
    msgs_all = []
    for i in idx:
        m = [{"role": "system", "content": CONV_SYSTEM},
             {"role": "user", "content": CONV_PROMPT.replace("_", descs[i][:400])}]
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
    for j, i in enumerate(idx):
        hits, valid = {li: 0 for li in latents}, 0
        for t_ in texts[j * n:(j + 1) * n]:
            conv = parse_conv(t_)
            if not conv:
                continue
            if tok.chat_template:                  # same formatting as the Goodfire arm when available
                ids = tok.apply_chat_template(conv, tokenize=True, add_generation_prompt=False,
                                              return_tensors="pt", return_dict=True)["input_ids"][:, :200].to(DEV)
            else:                                  # base tokenizer without a template: plain text, BOS added
                text = "\n".join(f"{c['role']}: {c['content']}" for c in conv)
                ids = tok(text, return_tensors="pt", truncation=True, max_length=200)["input_ids"].to(DEV)
            h = base(input_ids=ids, output_hidden_states=True).hidden_states[LAYER + 1][0]
            acts = sae_encode(h)
            valid += 1
            for li in latents:
                if (acts[1:, li] > 0).any().item():
                    hits[li] += 1
        out[i] = {li: (hits[li] / valid if valid else 0.0) for li in latents}
    return out


def score(desc, latents, n=10):
    return score_many([desc], latents, n)[0]


def compose(ia, ib, alpha):
    v = alpha * unit(ia) + (1 - alpha) * unit(ib)
    return v / v.norm()


def mag_for(ia, ib):
    return 0.5 * (float(NORMS[ia]) + float(NORMS[ib]))


# ----------------------------------------------------------------------------- 2 pre-flight
log("=== stage 2: pre-flight ===")
pf = ck("preflight")
if "sae_l0" not in pf:
    txt = "The FDA requires that all packaged foods list their ingredients in descending order by weight."
    ids = tok(txt, return_tensors="pt")["input_ids"].to(DEV)
    h = base(input_ids=ids, output_hidden_states=True).hidden_states[LAYER + 1][0]
    acts = sae_encode(h)
    pf["sae_l0"] = float((acts[1:] > 0).sum(-1).float().mean())
    pf["resid_norm"] = float(h[1:].float().norm(dim=-1).mean())
    save("preflight", pf)
log(f"real residual norm {pf['resid_norm']:.2f} (SAE expects ~17.1) | L0 {pf['sae_l0']:.1f} (expect ~50)")

feats = {int(k): v for k, v in CAND["features"].items()}
probe = sorted(feats, key=lambda i: -feats[i]["sim"])[:2]
if "known" not in pf:
    pf["known"] = {}
    for i in probe:
        m = float(NORMS[i])
        li_out = gen_li(unit(i), 3, m, seed=1)
        ad_out = gen_adapter(unit(i), 3, seed=1)
        pf["known"][i] = {"label": feats[i]["desc"], "li": li_out, "adapter": ad_out,
                          "li_score": score(li_out[0], [i])[i], "ad_score": score(ad_out[0], [i])[i]}
        log(f"feature {i} '{feats[i]['desc'][:40]}' | Li: {li_out[0]!r} (score {pf['known'][i]['li_score']:.1f}) "
            f"| adapter: {ad_out[0]!r} (score {pf['known'][i]['ad_score']:.1f})")
    save("preflight", pf)

if "mag_sweep" not in pf:                    # describability threshold of the explainer
    pf["mag_sweep"] = {}
    for i in probe:
        m0 = float(NORMS[i])
        pf["mag_sweep"][i] = {}
        for f in [0.5, 0.7, 1.0, 1.5, 2.0, 3.0]:
            d_ = gen_li(unit(i), 1, m0 * f, seed=7)[0]
            pf["mag_sweep"][i][f] = (d_, score(d_, [i])[i])
        log(f"mag sweep {i}: " + "  ".join(f"{f}x:{s:.1f}" for f, (_, s) in pf["mag_sweep"][i].items()))
    save("preflight", pf)

# ----------------------------------------------------------------------------- 3 gate 1
log("=== stage 3: gate 1 under both methods ===")
G1 = ck("gate1")
concepts = sorted(feats)                     # every Neuronpedia candidate (<= 4 per family), not only those in anchor x concerning pairs
for i in concepts:
    if i in G1:
        continue
    m = float(NORMS[i])
    li_d = gen_li(unit(i), 3, m, seed=seed_of("g1li", i))
    ad_d = gen_adapter(unit(i), 3, seed=seed_of("g1ad", i))
    li_s = [score(d_, [i])[i] for d_ in li_d]
    ad_s = [score(d_, [i])[i] for d_ in ad_d]
    G1[i] = {"li": list(zip(li_d, li_s)), "adapter": list(zip(ad_d, ad_s)),
             "li_pass": max(li_s) >= 0.8, "ad_pass": max(ad_s) >= 0.8}
    save("gate1", G1)
    log(f"g1 {i:>7} Li {max(li_s):.1f} {'PASS' if G1[i]['li_pass'] else 'fail'} | "
        f"adapter {max(ad_s):.1f} {'PASS' if G1[i]['ad_pass'] else 'fail'} | {feats[i]['desc'][:40]}")

ok = sorted(i for i in concepts if G1[i]["li_pass"] and G1[i]["ad_pass"])
log(f"gate 1: Li pass {sum(G1[i]['li_pass'] for i in concepts)}/{len(concepts)}, adapter pass "
    f"{sum(G1[i]['ad_pass'] for i in concepts)}/{len(concepts)}, both {len(ok)}/{len(concepts)}")

# Pair pool: any two concepts that pass gate 1 under BOTH methods, from different families
# (the anchor x concerning restriction left no pairs on this SAE: most cooking/spice features fail
# gate 1 under the adapter). Gates 2 and 3 are recomputed here at the injection magnitude; the 10%
# share is required only for the 10% row (pairs failing gate 3 at 10% skip that share). Each unordered
# pair runs in both directions (A majority / B minority, then swapped).
B_SHARES = [0.75, 0.5, 0.25, 0.1]


@torch.no_grad()
def gate3(ia, ib):
    m, acts = mag_for(ia, ib), {}
    for bs in B_SHARES:
        a_ = sae_encode(compose(ia, ib, 1 - bs) * m)
        acts[f"{round(bs*100)}%"] = [round(float(a_[ia]), 3), round(float(a_[ib]), 3)]
    core = all(acts[k][0] > 0 and acts[k][1] > 0 for k in ("75%", "50%", "25%"))
    return core, acts


directed = {}
for ia in ok:
    for ib in ok:
        if ia == ib or feats[ia]["family"] == feats[ib]["family"]:
            continue
        cos = float(unit(ia) @ unit(ib))
        if abs(cos) >= 0.1:
            continue
        core, acts = gate3(ia, ib)
        if not core:
            continue
        directed[(ia, ib)] = {"A": ia, "B": ib, "A_family": feats[ia]["family"], "B_family": feats[ib]["family"],
                              "A_desc": feats[ia]["desc"], "B_desc": feats[ib]["desc"], "cos": round(cos, 4),
                              "inject_mag": round(mag_for(ia, ib), 3), "acts_A_B": acts,
                              "has_10pct": acts["10%"][0] > 0 and acts["10%"][1] > 0}
unordered = {}
for (ia, ib), p in directed.items():
    k = tuple(sorted((ia, ib)))
    if (ib, ia) in directed:                    # keep only pairs valid in both directions
        unordered[k] = min(p["acts_A_B"]["25%"][1], directed[(ib, ia)]["acts_A_B"]["25%"][1])
log(f"directed pairs passing gates 2+3 (75/50/25): {len(directed)}; unordered valid both ways: {len(unordered)}")
# up to 6 unordered pairs, strongest minority-at-25% first, at most 2 per family combination, at most 3 per family
chosen_u, per_combo, per_fam = [], Counter(), Counter()
for k in sorted(unordered, key=lambda k: -unordered[k]):
    fa, fb = feats[k[0]]["family"], feats[k[1]]["family"]
    combo = tuple(sorted((fa, fb)))
    if per_combo[combo] >= 2 or per_fam[fa] >= 3 or per_fam[fb] >= 3 or len(chosen_u) >= 6:
        continue
    chosen_u.append(k); per_combo[combo] += 1; per_fam[fa] += 1; per_fam[fb] += 1
if len(chosen_u) < 6:
    for k in sorted(unordered, key=lambda k: -unordered[k]):
        if k not in chosen_u and len(chosen_u) < 6:
            chosen_u.append(k)
chosen = [directed[(a_, b_)] for (a_, b_) in chosen_u] + [directed[(b_, a_)] for (a_, b_) in chosen_u]
PAIRS = [(f"{feats[p['A']]['desc'][:16]} x {feats[p['B']]['desc'][:22]}", p["A"], p["B"]) for p in chosen]
assert len(set(nm for nm, _, _ in PAIRS)) == len(PAIRS), "pair names collide; lengthen the name slices"
NO10 = {(p["A"], p["B"]) for p in chosen if not p["has_10pct"]}
json.dump({"pairs": chosen, "n": len(chosen), "gate1_both_pass": ok}, open(f"{W}/pairs_final.json", "w"), indent=1)
log(f"final pairs: {len(PAIRS)} directed ({len(chosen_u)} unordered x 2 directions); without a 10% row: {len(NO10)}")
for nm, a_, b_ in PAIRS:
    log(f"    {nm}   [{feats[a_]['family']} x {feats[b_]['family']}]" + ("  (no 10%)" if (a_, b_) in NO10 else ""))
assert len(chosen_u) >= 3, "too few pairs survived gates; stop and look"

# ----------------------------------------------------------------------------- 3b describability threshold
# For H6 (parity winner tracks the single-concept describability threshold): every concept in the
# final pairs, alone, at sub-trained magnitudes. Li: 0.5/0.7/1.0 x its raw norm. Adapter: 0.5/0.8 x unit
# (the Goodfire arm used gate-1 hit at scale 0.5/0.8). Three sampled descriptions per cell, best hit kept.
log("=== stage 3b: describability threshold per concept ===")
TH = ck("thresh")
for i in sorted({ia for _, ia, _ in PAIRS} | {ib for _, _, ib in PAIRS}):
    if i in TH:
        continue
    TH[i] = {"li": {}, "adapter": {}}
    for f in [0.5, 0.7, 1.0]:
        d_ = gen_li(unit(i), 3, float(NORMS[i]) * f, seed=seed_of("thli", i, f))
        TH[i]["li"][f] = list(zip(d_, [x[i] for x in score_many(d_, [i])]))
    for f in [0.5, 0.8]:
        d_ = gen_adapter(unit(i), 3, seed=seed_of("thad", i, f), scale=f)
        TH[i]["adapter"][f] = list(zip(d_, [x[i] for x in score_many(d_, [i])]))
    save("thresh", TH)
    log(f"thr {i:>7} Li " + " ".join(f"{f}x:{max(s for _, s in TH[i]['li'][f]):.1f}" for f in [0.5, 0.7, 1.0]) +
        " | adapter " + " ".join(f"{f}x:{max(s for _, s in TH[i]['adapter'][f]):.1f}" for f in [0.5, 0.8]) +
        f" | {feats[i]['desc'][:36]}")
subprocess.run(f"cp {W}/thresh.pkl {W}/gate1.pkl {W}/pairs_final.json /workspace/RESULTS/ 2>/dev/null", shell=True)

ALPHAS = [0.0, 0.25, 0.5, 0.75, 0.9, 1.0]
SHARE = {0.0: "100%", 0.25: "75%", 0.5: "50%", 0.75: "25%", 0.9: "10%", 1.0: "0% (control)"}
N_DESC = 20


# ----------------------------------------------------------------------------- 4+5 Li sweep
def sweep(tag, genfn, res):
    for al in ALPHAS:
        for nm, ia, ib in PAIRS:
            key = (tag, nm, al)
            if key in res or (al == 0.9 and (ia, ib) in NO10):
                continue
            v, m = compose(ia, ib, al), mag_for(ia, ib)
            texts = genfn(v, N_DESC, m, seed_of(tag, ia, ib, al))
            ss = score_many(texts, [ia, ib])
            rows = [{"label": t_, "hit_A": s[ia], "hit_B": s[ib]} for t_, s in zip(texts, ss)]
            res[key] = rows
            save(tag, res)
        done = [r for k, v_ in res.items() if k[0] == tag and k[2] == al for r in v_]
        named = sum(1 for r in done if r["hit_B"] >= THR)
        log(f"[{tag}] {SHARE[al]:>13}  B named {named}/{len(done)}")
    return res


def list_rows(texts, ia, ib):
    """List-prompt cells: split each draw into numbered items (max 6), score every item at
    n=6, a concept counts as recovered if any item hits it. Same rule as the Goodfire arm."""
    per = [[re.sub(r'^\s*\d+[\.\)]\s*', '', x).strip(" -*") for x in t_.split("\n") if x.strip()][:6] for t_ in texts]
    flat = [it for items in per for it in items]
    ss = score_many(flat, [ia, ib], n=6)
    rows, k = [], 0
    for t_, items in zip(texts, per):
        bA = bB = 0.0
        for _ in items:
            bA, bB = max(bA, ss[k][ia]), max(bB, ss[k][ib]); k += 1
        rows.append({"label": t_, "items": items, "hit_A": bA, "hit_B": bB})
    return rows


log("=== stage 4/5: Li sweep + scoring ===")
LI = ck("li")
LI = sweep("li", lambda v, n, m, s: gen_li(v, n, m, seed=s), LI)
for nm, ia, ib in PAIRS:                          # greedy, their protocol, one per cell
    for al in ALPHAS:
        key = ("li_greedy", nm, al)
        if key in LI or (al == 0.9 and (ia, ib) in NO10):
            continue
        t_ = gen_li(compose(ia, ib, al), 1, mag_for(ia, ib), greedy=True)[0]
        s = score(t_, [ia, ib]); LI[key] = [{"label": t_, "hit_A": s[ia], "hit_B": s[ib]}]
        save("li", LI)
for nm, ia, ib in PAIRS:                          # list prompt at 50% and 25%
    for al in [0.5, 0.75]:
        key = ("li_list", nm, al)
        if key in LI:
            continue
        LI[key] = list_rows(gen_li(compose(ia, ib, al), 8, mag_for(ia, ib), seed=seed_of("lilist", ia, ib, al),
                                   list_prompt=True), ia, ib)
        save("li", LI)
    log(f"[li_list] {nm}")
log("Li arm complete")
subprocess.run(f"mkdir -p /workspace/RESULTS && cp {W}/li.pkl {W}/gate1.pkl {W}/preflight.pkl {W}/pairs_final.json /workspace/RESULTS/", shell=True)

# ----------------------------------------------------------------------------- 6 adapter sweep
log("=== stage 6: adapter sweep (Pepper llamascope scalar-affine) ===")
AD = ck("adapter")
AD = sweep("adapter", lambda v, n, m, s: gen_adapter(v, n, seed=s), AD)
for nm, ia, ib in PAIRS:
    for al in [0.5, 0.75]:
        key = ("adapter_list", nm, al)
        if key in AD:
            continue
        AD[key] = list_rows(gen_adapter(compose(ia, ib, al), 8, seed=seed_of("adlist", ia, ib, al), list_prompt=True), ia, ib)
        save("adapter", AD)
log("adapter arm complete")
subprocess.run(f"cp {W}/adapter.pkl /workspace/RESULTS/", shell=True)

# ----------------------------------------------------------------------------- 7 floors
log("=== stage 7: random-direction floor, 20 x 12 latents = 240 per method ===")
# the 12 latents scored against: every concerning (B) latent of the final pairs first, then anchors
targets = list(dict.fromkeys([ib for _, _, ib in PAIRS] + [ia for _, ia, _ in PAIRS]))[:12]
LI_FLOOR_MAG = float(NORMS.mean())          # a random direction has no pair; use the mean trained magnitude


def floors_for(tag, genfn, FL):
    keys = [(tag, k) for k in range(20) if (tag, k) not in FL]
    if keys:
        descs = []
        for _, k in keys:
            torch.manual_seed(1000 + k)
            rv = torch.randn(4096, device=DEV); rv = rv / rv.norm()
            descs.append(genfn(rv, seed_of("floor", tag, k)))
        for key, d_, s in zip(keys, descs, score_many(descs, targets)):
            FL[key] = {"label": d_, "hits": {t_: s[t_] for t_ in targets}}
        save("floors", FL)
    fp = sum(1 for k_, v_ in FL.items() if k_[0] == tag for t_, h in v_["hits"].items() if h >= THR)
    log(f"floor [{tag}]: {fp}/{20*len(targets)} false positives")


FL = ck("floors")
floors_for("li", lambda v, s: gen_li(v, 1, LI_FLOOR_MAG, seed=s)[0], FL)
floors_for("adapter", lambda v, s: gen_adapter(v, 1, seed=s)[0], FL)
subprocess.run(f"cp {W}/floors.pkl /workspace/RESULTS/", shell=True)

# ----------------------------------------------------------------------------- 7b rank-64 adapter (extra arm)
log("=== stage 7b: Pepper llamascope scalar-affine + rank-64 adapter, same pairs, same scorer ===")
adapter64 = load_adapter(hf_hub_download("keenanpepper/selfie-adapters-llama-3.1-8b-instruct",
                                         "llamascope-sae-sa-lr64.safetensors", cache_dir=XC))
log("rank-64 adapter loaded:", adapter64.get_metadata())
G64 = ck("gate1_64")                         # recorded for audit; the pair set is NOT reselected
for i in sorted({ia for _, ia, _ in PAIRS} | {ib for _, _, ib in PAIRS}):
    if i in G64:
        continue
    d_ = gen_adapter(unit(i), 3, seed=seed_of("g1ad64", i), ad=adapter64)
    s_ = [x[i] for x in score_many(d_, [i])]
    G64[i] = {"adapter": list(zip(d_, s_)), "ad_pass": max(s_) >= 0.8}
    save("gate1_64", G64)
    log(f"g1(rank-64) {i:>7} {max(s_):.1f} {'PASS' if G64[i]['ad_pass'] else 'fail'} | {feats[i]['desc'][:40]}")
A64 = ck("adapter64")
A64 = sweep("adapter64", lambda v, n, m, s: gen_adapter(v, n, seed=s, ad=adapter64), A64)
for nm, ia, ib in PAIRS:
    for al in [0.5, 0.75]:
        key = ("adapter64_list", nm, al)
        if key in A64:
            continue
        A64[key] = list_rows(gen_adapter(compose(ia, ib, al), 8, seed=seed_of("ad64list", ia, ib, al),
                                         list_prompt=True, ad=adapter64), ia, ib)
        save("adapter64", A64)
floors_for("adapter64", lambda v, s: gen_adapter(v, 1, seed=s, ad=adapter64)[0], FL)
log("rank-64 arm complete")
subprocess.run(f"cp {W}/adapter64.pkl {W}/gate1_64.pkl {W}/floors.pkl /workspace/RESULTS/", shell=True)

# ----------------------------------------------------------------------------- 8 summarise, copy, stop
log("=== stage 8: summary ===")
summary = {"pairs": [nm for nm, _, _ in PAIRS], "n_pairs": len(PAIRS), "N_DESC": N_DESC}
for tag, res in [("li", LI), ("adapter", AD), ("adapter64", A64)]:
    summary[tag] = {}
    for al in ALPHAS:
        rows = [r for k, v_ in res.items() if k[0] == tag and k[2] == al for r in v_]
        c = Counter()
        for r in rows:
            a_, b_ = r["hit_A"] >= THR, r["hit_B"] >= THR
            c["both" if (a_ and b_) else "A" if a_ else "B" if b_ else "neither"] += 1
        summary[tag][SHARE[al]] = {"A": c["A"] + c["both"], "B": c["B"] + c["both"], "both": c["both"],
                                   "neither": c["neither"], "n": len(rows)}
        log(f"[{tag}] {SHARE[al]:>13}  A {summary[tag][SHARE[al]]['A']:>3}  B {summary[tag][SHARE[al]]['B']:>3}  "
            f"both {c['both']:>3}  neither {c['neither']:>3}  / {len(rows)}")
json.dump(summary, open(f"{W}/summary.json", "w"), indent=1)
subprocess.run(f"cp {W}/*.pkl {W}/*.json {W}/run.log /workspace/RESULTS/ 2>/dev/null", shell=True)
log("results copied to /workspace/RESULTS")

log("done")
stop_pod("done")
