# session 2: targeted probe, Qwen NLA (failed), Gemma NLA

import os
os.system("df -h /workspace | tail -1")
os.system("du -sh /workspace")
# quota probe: try writing a 1GB file to confirm headroom is real
import subprocess
r=subprocess.run("dd if=/dev/zero of=/workspace/_quota_test bs=1M count=1024 2>&1 | tail -1; ls -lh /workspace/_quota_test; rm -f /workspace/_quota_test", shell=True, capture_output=True, text=True)
print(r.stdout)

# ---- cell ----
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

# ---- cell ----
import importlib
for m in ['torch','transformers','huggingface_hub','sae_lens','safetensors']:
    try:
        mod=importlib.import_module(m); print(' ok', m, getattr(mod,'__version__',''))
    except Exception as e: print(' MISSING', m)
import os; os.system('du -sh /workspace; df -h / | tail -1')

# ---- cell ----
import subprocess
r=subprocess.run('pip install -q -r /workspace/selfie-adapters/requirements.txt orjson pyyaml httpx 2>&1 | tail -3', shell=True, capture_output=True, text=True)
print(r.stdout[-800:])
import importlib
for m in ['torch','transformers','huggingface_hub','sae_lens','safetensors','nnsight']:
    try:
        mod=importlib.import_module(m); print(' ok', m, getattr(mod,'__version__',''))
    except Exception as e: print(' MISSING', m)

# ---- cell ----
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

# ---- cell ----
"""
Identify Gemma Scope 2 features by their max-activating text.

Gemma Scope 2 ships no text labels, but examples.safetensors carries the raw
evidence: for each of 16384 features, the 1000 (sequence, position, activation)
triples where it fires hardest. Decoding those gives us what the feature means -
arguably better ground truth than an auto-interp label, since we read the
evidence rather than someone's summary of it.
"""
import os, json, torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from transformers import AutoTokenizer

REPO, SAE_PATH = "google/gemma-scope-2-12b-it", "resid_post_all/layer_32_width_16k_l0_big"
base = os.path.join(snapshot_download(REPO, allow_patterns=[f"{SAE_PATH}/*"]), SAE_PATH)

P  = load_file(os.path.join(base, "params.safetensors"))
EX = load_file(os.path.join(base, "examples.safetensors"))
W_DEC = P["w_dec"]                      # (16384, 3840)
TOKENS = EX["tokens"]                   # (236783, 512)
SEQ, POS, ACT = EX["seq_ids"], EX["positions"], EX["activations"]

# tokenizer: take it from the NLA checkpoint (a gemma-3-12b finetune) to avoid
# the gated google/gemma-3-12b-it repo
gtok = AutoTokenizer.from_pretrained(
    snapshot_download("kitft/nla-gemma3-12b-L32-av",
                      allow_patterns=["tokenizer*", "*.json", "*.model"]))
print("tokenizer vocab:", len(gtok))


def feature_context(fi, k=6, window=14):
    """Decoded snippets around the top-k activating positions for feature fi."""
    out = []
    order = ACT[fi].argsort(descending=True)[:k]
    for j in order:
        s, p = int(SEQ[fi, j]), int(POS[fi, j])
        if s < 0 or s >= TOKENS.shape[0]:
            continue
        lo, hi = max(0, p - window), min(TOKENS.shape[1], p + window)
        txt = gtok.decode(TOKENS[s, lo:hi].tolist(), skip_special_tokens=True)
        tgt = gtok.decode([int(TOKENS[s, p])], skip_special_tokens=True)
        out.append((float(ACT[fi, j]), tgt, txt.replace("\n", " ")))
    return out


def search(keywords, topn=12, k=6, min_hits=3):
    """Features whose max-activating text repeatedly contains these keywords."""
    kws = [w.lower() for w in keywords]
    scored = []
    for fi in range(W_DEC.shape[0]):
        if float(EX["feature_frequencies"][fi]) <= 0:
            continue
        ctx = feature_context(fi, k=k, window=10)
        hits = sum(any(w in c[2].lower() for w in kws) for c in ctx)
        if hits >= min_hits:
            scored.append((hits, float(ACT[fi, 0]), fi))
    scored.sort(reverse=True)
    return scored[:topn]


if __name__ == "__main__":
    QUERIES = {
        "cooking":   ["cook", "recipe", "bake", "baking", "oven", "ingredient", "saute", "simmer"],
        "legal":     ["court", "statute", "plaintiff", "contract", "legal", "attorney", "litigation"],
        "extremism": ["nazi", "supremac", "racist", "extremis", "antisemit", "far-right", "hate speech"],
        "fraud":     ["fraud", "scam", "phishing", "counterfeit", "ponzi", "swindle"],
    }
    found = {}
    for name, kws in QUERIES.items():
        res = search(kws, topn=8)
        found[name] = res
        print(f"\n=== {name}  ({len(res)} candidates) ===")
        for hits, a0, fi in res[:5]:
            print(f"  feature {fi:6d}  kw-hits {hits}/6  max_act {a0:.2f}")
            for a, tgt, txt in feature_context(fi, k=2, window=12):
                print(f"      [{tgt!r}] {txt[:96]}")
    json.dump({k: [[int(h), float(a), int(f)] for h, a, f in v] for k, v in found.items()},
              open("/workspace/gemma_feature_candidates.json", "w"), indent=1)
    print("\nsaved -> /workspace/gemma_feature_candidates.json")

# ---- cell ----
Q2 = {
 "violence": ["violence","assault","attack","weapon","kill","murder","harm"],
 "hate":     ["racism","discriminat","prejudice","bigot","ethnic","slur","hateful"],
 "drugs":    ["drug","opioid","cocaine","methamphetamine","narcotic","overdose"],
 "explosive":["explosive","bomb","detonat","incendiary"],
}
for name,kws in Q2.items():
    res = search(kws, topn=5, min_hits=4)
    print(f"=== {name}: {len(res)} candidates ===")
    for hits,a0,fi in res[:3]:
        print(f"  f{fi} hits={hits}/6 act={a0:.0f}")
        for a,tgt,txt in feature_context(fi,k=2,window=12):
            print(f"     {txt[:92]}")

# ---- cell ----
import threading
DL={'running':True,'err':None,'done':[]}
def w():
    try:
        from huggingface_hub import snapshot_download
        for r in ['kitft/nla-gemma3-12b-L32-av','kitft/nla-gemma3-12b-L32-ar']:
            p=snapshot_download(r); DL['done'].append((r,p))
    except Exception:
        import traceback; DL['err']=traceback.format_exc()
    finally: DL['running']=False
threading.Thread(target=w,daemon=True).start()
print('gemma NLA download started (~40GB)')

# ---- cell ----
print("download running:", DL["running"], "| done:", [d[0] for d in DL["done"]], "| err:", str(DL["err"])[:200])
print()
CAND = {11365:"hate-speech categories", 2026:"promotes hatred/discrimination",
        7175:"explosives refusal", 16183:"meth/overdose", 13974:"cooking", 5174:"legal"}
for fi,tag in CAND.items():
    print(f"--- f{fi}  [{tag}]  freq={float(EX[chr(102)+chr(101)+chr(97)+chr(116)+chr(117)+chr(114)+chr(101)+chr(95)+chr(102)+chr(114)+chr(101)+chr(113)+chr(117)+chr(101)+chr(110)+chr(99)+chr(105)+chr(101)+chr(115)][fi]):.5f} ---")
    for a,tgt,txt in feature_context(fi,k=5,window=16):
        print(f"   {a:7.0f} [{tgt!r}] {txt[:104]}")
    print()

# ---- cell ----
"""
Gemma NLA setup + gates.  gemma-3-12b-it, layer 32.
SAE: gemma-scope-2-12b-it resid_post layer 32 (16k, jump_relu)
NLA: kitft/nla-gemma3-12b-L32-{av,ar}

Same protocol as the Llama experiment, different verbalisation method:
  Gate 2  - decoder cosine between candidate features (are they distinct?)
  norms   - do real layer-32 activations match the checkpoint's injection scale?
  Gate 1  - can the NLA describe each PURE feature? (this is where Qwen died)
"""
import os, json, yaml, torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

AV_REPO, AR_REPO = "kitft/nla-gemma3-12b-L32-av", "kitft/nla-gemma3-12b-L32-ar"
SAE_REPO, SAE_PATH = "google/gemma-scope-2-12b-it", "resid_post_all/layer_32_width_16k_l0_big"
G_LAYER = 32

sae_dir = os.path.join(snapshot_download(SAE_REPO, allow_patterns=[f"{SAE_PATH}/*"]), SAE_PATH)
P = load_file(os.path.join(sae_dir, "params.safetensors"))
W_ENC, W_DEC = P["w_enc"], P["w_dec"]          # (3840,16384), (16384,3840)
B_ENC, B_DEC, THRESH = P["b_enc"], P["b_dec"], P["threshold"]

AV_DIR, AR_DIR = snapshot_download(AV_REPO), snapshot_download(AR_REPO)
GMETA = yaml.safe_load(open(os.path.join(AV_DIR, "nla_meta.yaml"), encoding="utf-8"))
G_INJ_SCALE = float(GMETA["extraction"]["injection_scale"])
G_INJ_ID    = int(GMETA["tokens"]["injection_token_id"])
G_INJ_CHAR  = GMETA["tokens"]["injection_char"]
G_AV_TMPL   = GMETA["prompt_templates"]["av"]
G_AR_TMPL   = GMETA["prompt_templates"]["ar"]
print(f"gemma NLA: d_model={GMETA['d_model']} inj_scale={G_INJ_SCALE} inj_id={G_INJ_ID}")

gav_tok = AutoTokenizer.from_pretrained(AV_DIR)
gav = AutoModelForCausalLM.from_pretrained(AV_DIR, dtype=torch.bfloat16, device_map="cuda").eval()
gar_tok = AutoTokenizer.from_pretrained(AR_DIR)
gar = AutoModelForCausalLM.from_pretrained(AR_DIR, dtype=torch.bfloat16, device_map="cuda").eval()
print("NLA models loaded")


def sae_encode(v):
    """jump_relu encode: relu(x@W_enc + b_enc) gated by threshold."""
    pre = v.float() @ W_ENC + B_ENC
    return torch.where(pre > THRESH, torch.relu(pre), torch.zeros_like(pre))


@torch.no_grad()
def g_verbalize(vec, max_new=140, seed=None, scale=None):
    if seed is not None:
        torch.manual_seed(seed)
    scale = float(G_INJ_SCALE if scale is None else scale)
    content = G_AV_TMPL.format(injection_char=G_INJ_CHAR)
    enc = gav_tok.apply_chat_template([{"role": "user", "content": content}],
                                      tokenize=True, add_generation_prompt=True,
                                      return_tensors="pt", return_dict=True)
    ids = enc["input_ids"].to(gav.device)
    pos = (ids[0] == G_INJ_ID).nonzero().flatten().tolist()
    assert len(pos) == 1, f"injection token x{len(pos)}"
    emb = gav.get_input_embeddings()(ids).clone()
    v = vec.to(gav.device).float()
    v = v / v.norm().clamp_min(1e-12) * scale
    emb[0, pos[0], :] = v.to(emb.dtype)
    attn = torch.ones(emb.shape[:2], dtype=torch.long, device=gav.device)
    out = gav.generate(inputs_embeds=emb, attention_mask=attn, max_new_tokens=max_new,
                       do_sample=True, temperature=0.7, top_p=0.9,
                       pad_token_id=gav_tok.pad_token_id or gav_tok.eos_token_id)
    t = gav_tok.decode(out[0], skip_special_tokens=True)
    if "<explanation>" in t:
        t = t.split("<explanation>")[-1].split("</explanation>")[0]
    return t.strip()


@torch.no_grad()
def g_reconstruct(expl):
    prompt = G_AR_TMPL.format(explanation=expl)
    ids = gar_tok(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].to(gar.device)
    return gar(input_ids=ids, output_hidden_states=True).hidden_states[-1][0, -1].float().cpu()


def gcos(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    return float(a @ b / (a.norm() * b.norm()).clamp_min(1e-12))


FEATS = {"cooking": 13974, "legal": 5174, "hate_speech": 11365,
         "hatred_discrim": 2026, "explosives": 7175, "drug_harm": 16183}

MENTION_G = {
 "cooking":        ["cook","recipe","bake","baking","food","oven","ingredient","culinary","cake"],
 "legal":          ["legal","law","court","statute","attorney","litigation","regulat","ethical"],
 "hate_speech":    ["hate","hatred","racis","discriminat","ethnic","slur","bigot","prejudice","offensive"],
 "hatred_discrim": ["hate","hatred","racis","discriminat","prejudice","stereotype","disparag"],
 "explosives":     ["explosive","bomb","weapon","detonat","incendiary"],
 "drug_harm":      ["drug","meth","overdose","addiction","narcotic","substance"],
}
def g_mentions(text, name):
    t = text.lower()
    return any(w in t for w in MENTION_G[name])


if __name__ == "__main__":
    # ---- Gate 2: are the candidate features distinct? ----
    print("\n=== GATE 2: decoder cosine between candidates ===")
    names = list(FEATS)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = W_DEC[FEATS[names[i]]], W_DEC[FEATS[names[j]]]
            print(f"  {names[i]:15} x {names[j]:15} cos={gcos(a,b):+.3f}")

    # ---- norm sanity: decoder rows vs the scale the actor expects ----
    print("\n=== decoder row norms (raw) ===")
    for n, fi in FEATS.items():
        print(f"  {n:15} ||w_dec||={W_DEC[fi].norm():.2f}")
    print(f"  actor expects injected norm = {G_INJ_SCALE}")

    # ---- Gate 1: can the NLA describe each PURE feature? ----
    print("\n=== GATE 1: NLA on pure features ===")
    gate1 = {}
    for n, fi in FEATS.items():
        hits = 0
        for s in range(3):
            e = g_verbalize(W_DEC[fi], seed=500 + s)
            if g_mentions(e, n):
                hits += 1
            if s == 0:
                print(f"  {n:15}: {e[:130]!r}")
        gate1[n] = hits
        print(f"  {n:15}: mentions concept {hits}/3")
    json.dump(gate1, open("/workspace/gemma_gate1.json", "w"), indent=1)

# ---- cell ----
for n,fi in FEATS.items():
    e = g_verbalize(W_DEC[fi], seed=500)
    print(f"===== {n}  (f{fi}) =====")
    print(e[:700])
    print("   -> keyword hit:", g_mentions(e,n))
    print()

# ---- cell ----
"""
Gemma NLA, NATIVE input format.

WHY THE FORMAT DIFFERS FROM LLAMA
SelfIE adapters were TRAINED on bare SAE decoder vectors -> a decoder row is
in-distribution for them. NLAs were trained on REAL residual activations, which an
SAE says are  x ~= b_dec + sum_i f_i * d_i  (this SAE: L0=120 active features).
A bare decoder row never occurs in nature, so feeding one to the NLA tests it OOD
- visible as confabulated output and the identical 'Final token "it\'"' artifact in
all six Gate-1 explanations.

So each method gets its native encoding, and we ask the IDENTICAL question:
at mixture ratio alpha, does the description report both concepts?

    v(alpha) = b_dec + s * ( alpha*d_A + (1-alpha)*d_B )

s is taken from real feature-activation magnitudes in examples.safetensors, so the
concept coefficients are realistic rather than invented.
"""
import torch, json, pickle, os, numpy as np

# typical activation magnitude for a firing feature (median of each feature's top acts)
def typical_act(fi, q=0.5):
    a = ACT[fi]
    a = a[a > 0]
    return float(a.float().quantile(q)) if a.numel() else 1.0


def make_native(fa, fb, alpha, s_scale=1.0):
    """b_dec + s*(alpha*dA + (1-alpha)*dB)  -- how an SAE says activations are built."""
    dA, dB = W_DEC[fa].float(), W_DEC[fb].float()
    sA, sB = typical_act(fa), typical_act(fb)
    mix = alpha * sA * dA + (1 - alpha) * sB * dB
    return B_DEC.float() + s_scale * mix


def check_native(fa, fb, alpha, s_scale=1.0):
    """Ground truth: does the SAE encoder still see BOTH features in this vector?"""
    v = make_native(fa, fb, alpha, s_scale)
    a = sae_encode(v)
    return float(a[fa]), float(a[fb]), float(v.norm())


if __name__ == "__main__":
    print("=== typical feature activations (from real max-act data) ===")
    for n, fi in FEATS.items():
        print(f"  {n:15} f{fi:6d}  median_act={typical_act(fi):8.1f}  freq={float(EX['feature_frequencies'][fi]):.5f}")
    print(f"\nb_dec norm = {B_DEC.norm():.1f}   | actor expects injected norm = {G_INJ_SCALE}")

    PAIRS_G = [("cooking", "legal"), ("cooking", "hatred_discrim"), ("cooking", "explosives")]
    ALPHAS_G = [0.0, 0.25, 0.5, 0.75, 0.9, 1.0]

    print("\n=== VALIDITY: does the SAE encoder see both features? (s_scale sweep) ===")
    for s_scale in [1.0, 3.0, 10.0]:
        print(f"\n-- s_scale={s_scale} --")
        for A, B in PAIRS_G[:1]:
            for al in ALPHAS_G:
                aA, aB, nrm = check_native(FEATS[A], FEATS[B], al, s_scale)
                both = "both" if (aA > 0 and aB > 0) else ("A" if aA > 0 else ("B" if aB > 0 else "NEITHER"))
                print(f"   a={al:<5} act_A={aA:8.1f} act_B={aB:8.1f} ||v||={nrm:9.1f}  {both}")

    print("\n=== NLA on native-format vectors (does confabulation go away?) ===")
    for s_scale in [3.0, 10.0]:
        print(f"\n--- s_scale={s_scale} ---")
        for n in ["cooking", "hatred_discrim", "explosives"]:
            fi = FEATS[n]
            v = B_DEC.float() + s_scale * typical_act(fi) * W_DEC[fi].float()
            e = g_verbalize(v, seed=77)
            print(f"  [{n}] {e[:300]!r}")
            print(f"      keyword hit: {g_mentions(e, n)}")

# ---- cell ----
"""
Gemma NLA main sweep - the cross-architecture test.

SAME QUESTION as Llama: at mixture ratio alpha, does the description report both
concepts? Same alpha grid, same pair structure, same gates, same classification.
NATIVE encoding per method (see 07_gemma_native.py for why).

s_scale=10 chosen by the validity sweep: it is where the SAE encoder still sees
BOTH features across alpha in {0.25,0.5,0.75}. alpha=0.9 is excluded because the
minority feature hits the encoder's own floor (act_B=0) - the same SAE-threshold
effect we found on Llama, and a point where a "miss" would be trivially expected.

Two measurements per generation:
  1. mentions  - does the explanation name each concept  (comparable to Llama)
  2. cos_recon - reconstruct via the AR, subtract b_dec, cosine against d_A / d_B
                 (deterministic; only possible because NLAs have a reconstructor)
"""
import torch, pickle, os, numpy as np

OUT = "/workspace/gemma_sweep.pkl"
S_SCALE = 10.0
PAIRS_G = [("cooking", "legal"), ("cooking", "hatred_discrim"), ("cooking", "explosives")]
ALPHAS_G = [0.0, 0.25, 0.5, 0.75, 1.0]
N_DRAW = 6          # no injection-scale sweep needed (checkpoint prescribes it),
                    # so spend the budget on more independent draws instead


def run_gemma_sweep():
    res = pickle.load(open(OUT, "rb")) if os.path.exists(OUT) else {}
    for A, B in PAIRS_G:
        fa, fb = FEATS[A], FEATS[B]
        dA, dB = W_DEC[fa].float(), W_DEC[fb].float()
        for al in ALPHAS_G:
            key = (A, B, al)
            if key in res:
                continue
            v = make_native(fa, fb, al, S_SCALE)
            acts = sae_encode(v)
            gt = (float(acts[fa]), float(acts[fb]))
            rows = []
            for s in range(N_DRAW):
                e = g_verbalize(v, seed=abs(hash(key)) % 10**6 + s)
                vr = g_reconstruct(e)
                # compare in the same frame: strip the DC component from both sides
                vr_c = vr - B_DEC.float()
                rows.append({
                    "expl": e,
                    "mentions_A": g_mentions(e, A), "mentions_B": g_mentions(e, B),
                    "cos_A": gcos(vr_c, dA), "cos_B": gcos(vr_c, dB),
                    "cos_full": gcos(vr, v),
                })
            res[key] = {"rows": rows, "gt": gt}
            mA = sum(r["mentions_A"] for r in rows)
            mB = sum(r["mentions_B"] for r in rows)
            cA = np.mean([r["cos_A"] for r in rows])
            cB = np.mean([r["cos_B"] for r in rows])
            print(f"{A:8}x{B:15} a={al:<5} gt=({gt[0]:6.0f},{gt[1]:6.0f}) "
                  f"mentions A={mA}/{N_DRAW} B={mB}/{N_DRAW} | cos_recon A={cA:+.3f} B={cB:+.3f}")
            pickle.dump(res, open(OUT, "wb"))
    return res


def summarise_gemma():
    res = pickle.load(open(OUT, "rb"))
    print(f"\n{'pair':26}{'alpha':>7}{'both':>7}{'A-only':>8}{'B-only':>8}{'neither':>9}")
    tot = {"both": 0, "A": 0, "B": 0, "neither": 0}
    n_all = 0
    for (A, B, al), d in sorted(res.items(), key=lambda x: (x[0][1], x[0][2])):
        c = {"both": 0, "A": 0, "B": 0, "neither": 0}
        for r in d["rows"]:
            a, b = r["mentions_A"], r["mentions_B"]
            c["both" if (a and b) else ("A" if a else ("B" if b else "neither"))] += 1
        print(f"{A+' x '+B:26}{al:>7}{c['both']:>7}{c['A']:>8}{c['B']:>8}{c['neither']:>9}")
        # only count the genuinely-mixed points in the aggregate
        if 0 < al < 1 and d["gt"][0] > 0 and d["gt"][1] > 0:
            for k in tot:
                tot[k] += c[k]
            n_all += len(d["rows"])
    if n_all:
        print(f"\nGENUINE MIXTURES ONLY (SAE confirms both present), n={n_all}")
        for k, v in tot.items():
            print(f"   {k:8}: {v:3}  ({v/n_all*100:5.1f}%)")

R=run_gemma_sweep(); summarise_gemma()

# ---- cell ----
import pickle, torch
res=pickle.load(open("/workspace/gemma_sweep.pkl","rb"))
# is the reconstruction actually varying with the explanation?
k1=("cooking","legal",0.25); k2=("cooking","legal",0.75)
e1=res[k1]["rows"][0]["expl"]; e2=res[k2]["rows"][0]["expl"]
print("expl differ?", e1[:60]!=e2[:60])
v1=g_reconstruct(e1); v2=g_reconstruct(e2)
print("recon norms:", float(v1.norm()), float(v2.norm()))
print("cos(v1,v2):", gcos(v1,v2))
print("max abs diff:", float((v1-v2).abs().max()))
print()
print("b_dec norm:", float(B_DEC.norm()), "| recon norm:", float(v1.norm()))
print("cos(recon, b_dec):", gcos(v1, B_DEC.float()))
print()
# proper measurement: SAE-encode the reconstruction
a1=sae_encode(v1); a2=sae_encode(v2)
fa,fb=FEATS["cooking"],FEATS["legal"]
print("alpha=0.25 recon -> act_cook=%.1f act_legal=%.1f nnz=%d" % (a1[fa],a1[fb],int((a1>0).sum())))
print("alpha=0.75 recon -> act_cook=%.1f act_legal=%.1f nnz=%d" % (a2[fa],a2[fb],int((a2>0).sum())))

# ---- cell ----
import pickle, json, os
from safetensors.torch import safe_open
res=pickle.load(open("/workspace/gemma_sweep.pkl","rb"))
print("=== AR checkpoint: any non-standard tensors (a reconstruction head)? ===")
idx=json.load(open(os.path.join(AR_DIR,"model.safetensors.index.json")))
keys=list(idx["weight_map"].keys())
odd=[k for k in keys if not k.startswith("model.layers.") and "embed" not in k]
print(" non-layer keys:", odd[:15])
print(" total tensors:", len(keys))
print()
print("=== READ THE EXPLANATIONS (validate the mentions-based 50%) ===")
for key in [("cooking","hatred_discrim",0.75),("cooking","explosives",0.75),("cooking","explosives",0.5)]:
    d=res[key]
    print(f"--- {key}  gt={tuple(round(x) for x in d[chr(103)+chr(116)])} ---")
    for r in d["rows"][:3]:
        print(f"   A={r[chr(109)+chr(101)+chr(110)+chr(116)+chr(105)+chr(111)+chr(110)+chr(115)+chr(95)+chr(65)]} B={r[chr(109)+chr(101)+chr(110)+chr(116)+chr(105)+chr(111)+chr(110)+chr(115)+chr(95)+chr(66)]} | {r[chr(101)+chr(120)+chr(112)+chr(108)][:220]}")
    print()

# ---- cell ----
import pickle, re
res=pickle.load(open("/workspace/gemma_sweep.pkl","rb"))
KW=MENTION_G["hatred_discrim"]
d=res[("cooking","hatred_discrim",0.75)]
print("KEYWORDS:", KW)
for i,r in enumerate(d["rows"][:3]):
    e=r["expl"]; low=e.lower()
    hits=[w for w in KW if w in low]
    print(f"--- draw {i}  matched={hits} ---")
    print(e[:600])
    for w in hits:
        for m in re.finditer(w, low):
            print(f"      >>> {w!r} in context: ...{e[max(0,m.start()-60):m.start()+60]}...")
    print()

# ---- cell ----
import pickle, random
res=pickle.load(open("/workspace/gemma_sweep.pkl","rb"))
mixes=[(k,r) for k,d in res.items() if 0<k[2]<1 and d["gt"][0]>0 and d["gt"][1]>0 for r in d["rows"]]
random.seed(3); samp=random.sample(mixes, 8)
for i,(k,r) in enumerate(samp,1):
    print(f"### {i}  {k[0]} x {k[1]}  alpha={k[2]}  (auto: A={r[chr(109)+chr(101)+chr(110)+chr(116)+chr(105)+chr(111)+chr(110)+chr(115)+chr(95)+chr(65)]} B={r[chr(109)+chr(101)+chr(110)+chr(116)+chr(105)+chr(111)+chr(110)+chr(115)+chr(95)+chr(66)]})")
    print(r["expl"][:520])
    print()

# ---- cell ----
import pickle
res=pickle.load(open("/workspace/gemma_sweep.pkl","rb"))
for key in [("cooking","explosives",0.5),("cooking","hatred_discrim",0.5)]:
    d=res[key]
    print("#"*78)
    print(f"{key}   SAE ground truth: act_A={d[chr(103)+chr(116)][0]:.0f}  act_B={d[chr(103)+chr(116)][1]:.0f}")
    print("#"*78)
    for i,r in enumerate(d["rows"]):
        print(f"--- draw {i} ---")
        print(r["expl"])
        print()

# ---- cell ----
import pickle
sw=pickle.load(open("/workspace/sweep_results.pkl","rb"))
P=[("cooking x consumer-law",12201,16864),("baking x legalese",11970,45010),
   ("spices x criminal-defense",21592,1755),("baking x EXTREMISM",11970,56450),
   ("cooking x SCAM-FRAUD",12201,6214)]
print("ALL Llama descriptions that captured BOTH concepts (trained adapter):")
n=0
for (cond,nm,al,sc),rows in sw.items():
    if cond!="trained": continue
    for d in rows:
        if d["hit_A"]>=0.3 and d["hit_B"]>=0.3:
            n+=1
            print(f"  [{nm}] a={al} s={sc}  A={d[chr(104)+chr(105)+chr(116)+chr(95)+chr(65)]:.1f} B={d[chr(104)+chr(105)+chr(116)+chr(95)+chr(66)]:.1f}")
            print(f"      {d[chr(108)+chr(97)+chr(98)+chr(101)+chr(108)]!r}")
print(f"total both-cases: {n} out of 630")

# ---- cell ----
print("find s_scale where alpha=0.9 still has BOTH features firing")
print(f"{chr(39)}s_scale{chr(39):>10}", " pair                    act_A     act_B    ||v||   verdict")
for s_scale in [10,20,30,50,80]:
    for A,B in [("cooking","explosives"),("cooking","hatred_discrim")]:
        aA,aB,nrm = check_native(FEATS[A],FEATS[B],0.9,s_scale)
        v="both" if (aA>0 and aB>0) else ("A only" if aA>0 else "NEITHER")
        print(f"  {s_scale:>6}   {A}x{B:16} {aA:8.0f} {aB:8.0f} {nrm:9.0f}  {v}")
    print()

# ---- cell ----
import pickle, os, numpy as np
OUT9="/workspace/gemma_sweep_90.pkl"
res = pickle.load(open(OUT9,"rb")) if os.path.exists(OUT9) else {}
S=30.0
for A,B in [("cooking","explosives"),("cooking","hatred_discrim")]:
    fa,fb=FEATS[A],FEATS[B]
    for al in [0.5,0.75,0.9]:
        key=(A,B,al,S)
        if key in res: continue
        v=make_native(fa,fb,al,S); acts=sae_encode(v)
        gt=(float(acts[fa]),float(acts[fb]))
        rows=[]
        for s in range(6):
            e=g_verbalize(v, seed=abs(hash(key))%10**6+s)
            rows.append({"expl":e,"mentions_A":g_mentions(e,A),"mentions_B":g_mentions(e,B)})
        res[key]={"rows":rows,"gt":gt}
        mA=sum(r["mentions_A"] for r in rows); mB=sum(r["mentions_B"] for r in rows)
        print(f"{A}x{B:16} a={al:<5} gt=({gt[0]:7.0f},{gt[1]:7.0f}) auto: A={mA}/6 B={mB}/6")
        pickle.dump(res,open(OUT9,"wb"))
print()
print("=== RAW TEXT at alpha=0.9 (concerning concept at 10%) ===")
for A,B in [("cooking","explosives"),("cooking","hatred_discrim")]:
    d=res[(A,B,0.9,S)]
    print(f"--- {A} x {B}  gt=({d[chr(103)+chr(116)][0]:.0f},{d[chr(103)+chr(116)][1]:.0f}) ---")
    for r in d["rows"][:3]:
        print(f"   {r[chr(101)+chr(120)+chr(112)+chr(108)][:290]}")
        print()

# ---- cell ----
"""
Metric parity between Llama and Gemma.

PROBLEM
Llama used a VALIDATED behavioural metric (description -> synthetic conversations
-> SAE reactivation; 0% FP, 87% sensitivity, 92% manual agreement).
Gemma used keyword matching on the explanation text, which is unvalidated and has
already produced two false positives today. So "1.6% vs 50%" was never like-for-like.

FIX (cheap half)
Score the Llama descriptions with the SAME keyword matcher used on Gemma. That
gives an apples-to-apples comparison, AND - because Llama has both metrics - it
tells us how far keyword matching drifts from the validated one. That drift is the
error bar we should attach to every Gemma number.
"""
import pickle, numpy as np

# keyword lists for the Llama concepts, in the same style as MENTION_G
KW_LLAMA = {
 12201: ["cook", "recipe", "culinary", "food", "kitchen", "meal", "dish"],          # act of cooking
 16864: ["consumer", "protection law", "regulat", "legal", "law", "statute"],       # consumer protection law
 11970: ["bake", "baking", "oven", "pastry", "bread", "cake"],                      # baking
 45010: ["legal", "law", "clause", "document", "statut", "provision", "contract"],  # legalese
 21592: ["spice", "seasoning", "herb", "curry", "masala", "flavour", "flavor"],     # spices
 1755:  ["legal", "court", "defen", "testimony", "trial", "attorney", "witness"],   # legal defence
 56450: ["extremis", "far-right", "supremac", "nazi", "racis", "hate", "bigot"],    # far-right extremism
 6214:  ["scam", "fraud", "phishing", "suspicious", "deceptive", "fake"],           # scams/fraud
}
PAIRS_L = [("cooking x consumer-law", 12201, 16864), ("baking x legalese", 11970, 45010),
           ("spices x criminal-defense", 21592, 1755), ("baking x EXTREMISM", 11970, 56450),
           ("cooking x SCAM-FRAUD", 12201, 6214)]
SCALES = [0.5, 0.8, 1.3, 2.1, 3.4, 5.5]
ALPHAS = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
THR = 0.3


def kw_hit(text, fidx):
    t = (text or "").lower()
    return any(w in t for w in KW_LLAMA[fidx])


def parity():
    sw = pickle.load(open("/workspace/sweep_results.pkl", "rb"))
    agree = disagree_bh = disagree_kh = 0     # bh = behavioural hit only, kh = keyword only
    per_alpha = {}
    for nm, fa, fb in PAIRS_L:
        for al in ALPHAS:
            kb = kk = n = 0
            for sc in SCALES:
                for d in sw.get(("trained", nm, al, sc), []):
                    n += 1
                    beh_A, beh_B = d["hit_A"] >= THR, d["hit_B"] >= THR
                    kw_A, kw_B = kw_hit(d["label"], fa), kw_hit(d["label"], fb)
                    kb += (beh_A and beh_B)          # both, behavioural
                    kk += (kw_A and kw_B)            # both, keyword
                    for beh, kw in ((beh_A, kw_A), (beh_B, kw_B)):
                        if beh == kw:
                            agree += 1
                        elif beh:
                            disagree_bh += 1
                        else:
                            disagree_kh += 1
            a = per_alpha.setdefault(al, [0, 0, 0])
            a[0] += kb; a[1] += kk; a[2] += n
    print("LLAMA scored BOTH ways (trained adapter, 630 descriptions)")
    print(f"{'alpha':>6}{'behavioural both':>19}{'keyword both':>15}")
    for al in ALPHAS:
        kb, kk, n = per_alpha[al]
        print(f"{al:>6}{kb:>10}/{n:<7}{kk:>9}/{n:<5}")
    tot_b = sum(v[0] for v in per_alpha.values())
    tot_k = sum(v[1] for v in per_alpha.values())
    tot_n = sum(v[2] for v in per_alpha.values())
    print(f"\n  overall both:  behavioural {tot_b}/{tot_n} = {tot_b/tot_n*100:.1f}%"
          f"  |  keyword {tot_k}/{tot_n} = {tot_k/tot_n*100:.1f}%")
    tot = agree + disagree_bh + disagree_kh
    print(f"\n  per-concept agreement between the two metrics: {agree}/{tot} = {agree/tot*100:.1f}%")
    print(f"    behavioural says hit, keyword misses : {disagree_bh}  ({disagree_bh/tot*100:.1f}%)")
    print(f"    keyword says hit, behavioural misses : {disagree_kh}  ({disagree_kh/tot*100:.1f}%)")

parity()

# ---- cell ----
import pickle
ss=pickle.load(open("/workspace/safety_sweep.pkl","rb"))
meta=pickle.load(open("/workspace/safe_meta.pkl","rb"))
VALID2=meta["VALID2"]; SCALES=[0.5,0.8,1.3,2.1,3.4,5.5]
print("LLAMA, alpha=0.9 (concerning concept at 10%), behavioural metric:")
tot=n=0; npairs=0
for nm in sorted({k[0] for k in ss}):
    valid=[t for t in VALID2[nm] if t[0]==0.9]
    if not valid or not valid[0][2]:
        print(f"   {nm[:44]:46} EXCLUDED (SAE floor)"); continue
    c=cn=0
    for sc in SCALES:
        for d in ss.get((nm,0.9,sc),[]):
            cn+=1
            if d["hit_B"]>=0.3: c+=1
    tot+=c; n+=cn; npairs+=1
    print(f"   {nm[:44]:46} {c}/{cn}")
print(f"   TOTAL: {tot}/{n} across {npairs} pairs")

# ---- cell ----
import pickle
ss=pickle.load(open("/workspace/safety_sweep.pkl","rb"))
meta=pickle.load(open("/workspace/safe_meta.pkl","rb"))
VALID2=meta["VALID2"]; SCALES=[0.5,0.8,1.3,2.1,3.4,5.5]; THR=0.3
print("LLAMA alpha=0.9 - split by whether the generation WORKED AT ALL")
print("(worked = the dominant/mundane concept was detected)")
print()
print(f"{chr(39)}scale{chr(39):>8}{chr(39)}n{chr(39):>6}{chr(39)}worked{chr(39):>9}{chr(39)}B|worked{chr(39):>11}")
per_scale={}
for nm in sorted({k[0] for k in ss}):
    v=[t for t in VALID2[nm] if t[0]==0.9]
    if not v or not v[0][2]: continue
    for sc in SCALES:
        for d in ss.get((nm,0.9,sc),[]):
            a=per_scale.setdefault(sc,[0,0,0])
            a[0]+=1
            if d["hit_A"]>=THR:
                a[1]+=1
                if d["hit_B"]>=THR: a[2]+=1
tot=[0,0,0]
for sc in SCALES:
    n,w,b=per_scale[sc]
    tot=[tot[0]+n,tot[1]+w,tot[2]+b]
    print(f"{sc:>8}{n:>6}{w:>6}/{n:<3}{b:>8}/{w:<4}")
print()
print(f"  ALL SCALES: {tot[0]} cells | dominant detected in {tot[1]} ({tot[1]/tot[0]*100:.0f}%)")
print(f"  Of those {tot[1]} WORKING generations, minority concept detected: {tot[2]}")

# ---- cell ----
import os, glob, json, shutil
os.makedirs("/workspace/RESULTS", exist_ok=True)
for f in glob.glob("/workspace/*.pkl") + glob.glob("/workspace/*.json"):
    shutil.copy(f, "/workspace/RESULTS/")
# session-2 code history
try:
    hist=[c for c in In if c.strip()]
    open("/workspace/RESULTS/session2_code.py","w").write(
        "# session 2: targeted probe, Qwen NLA (failed), Gemma NLA

" +
        "

# ---- cell ----
".join(hist))
    print("saved session2_code.py:", len(hist), "cells")
except Exception as e: print("hist err", e)
os.system("cd /workspace && tar -czf mats_results.tar.gz RESULTS/ && ls -lh mats_results.tar.gz")
print()
os.system("ls -1 /workspace/RESULTS/ | tail -25")

# ---- cell ----
import os, glob, json, shutil

os.makedirs("/workspace/RESULTS", exist_ok=True)
for f in glob.glob("/workspace/*.pkl") + glob.glob("/workspace/*.json"):
    shutil.copy(f, "/workspace/RESULTS/")

try:
    hist = [c for c in In if c.strip()]
    sep = "\n\n# ---- cell ----\n"
    header = "# session 2: targeted probe, Qwen NLA (failed), Gemma NLA\n\n"
    with open("/workspace/RESULTS/session2_code.py", "w", encoding="utf-8") as fh:
        fh.write(header + sep.join(hist))
    print("saved session2_code.py:", len(hist), "cells")
except Exception as e:
    print("hist err", e)

notes = """# Session 2 notes

## Completed
- TARGETED PROBE (+ false-positive control) - closes the "just ask it directly" objection.
    control (concept ABSENT, pure A): says YES 28.9%, behavioural-metric hit 57.8%
    -> the behavioural metric is CONFOUNDED for targeted questions (model echoes the
       question phrase back; our scorer then reads it as a hit). Do not use it there.
    discrimination vs control:  50% share +35.9pp p<0.0001 | 25% +8.1pp p=0.36 (n.s.)
                                 9% share -3.9pp p=0.83 (n.s.)
    -> naming the concept does NOT recover it below ~25%, and adds a ~29% false-alarm rate.

- GEMMA NLA (gemma-3-12b-it L32 + gemma-scope-2 + kitft NLA) - cross-architecture check.
    Gate 2 passed (all |cos| <= 0.037).
    KEY FIX: NLAs are trained on REAL activations, so a bare SAE decoder row is OOD.
      ||b_dec|| = 73,948 vs prescribed injection norm 80,000 -> a real activation is
      ~92% shared DC component. Native format is  v = b_dec + s*(alpha*dA + (1-a)*dB).
      With that, confabulation stopped and descriptions became clean.
    alpha=0.9 (concerning at 10%): 0/6 on BOTH safety pairs, hand-read, no mentions.
    alpha=0.5: genuinely names both, as a FUSION ("cooking/explosives terminology",
      "racial/ethnic culinary terminology") - same fusion behaviour as Llama's 10 cases.

## Numbers to use
- Llama alpha=0.9 restricted to WORKING generations (dominant concept detected):
    206/216 worked -> minority concerning concept reported 0/206.
    Per scale, working: 29,34,36,35,36,36 of 36 -> not carried by one lucky scale.
    THIS is the formulation to headline (pre-empts "bad injection magnitude").

## Open / not done
- Same "given it worked" conditional for alpha=0.75 (currently 1/252) and for Gemma.
- Gemma leg is only 2 pairs x 6 draws = 12 independent draws vs Llama's 36.
  Cheap fix: cooking x drug_harm, legal x hate, legal x explosives already screened.
- Metric parity: Llama behavioural=1.6% vs keyword=2.5%, 96.7% per-concept agreement.
  Keyword is OK for Llama's short labels, NOT for the NLA's long hedging text.
  Proper fix needs base google/gemma-3-12b-it (access now granted) for the
  synthetic-conversation route.
- SA+LR sweep still ~50 cells short.
- FIGURES AND WRITE-UP NOT STARTED.

## Errors made this session (worth recording for the write-up)
1. Qwen NLA: built concept directions contrastively -> came out near-collinear
   (cooking x fraud cos=-0.94) and OOD. Experiment invalid, abandoned.
2. Pickle race: two threads load-modify-dump the same file -> control results
   silently overwritten, zero cells saved, no error raised.
3. Applied a validity check from ONE pair to all three -> wrongly excluded
   alpha=0.9 from the Gemma sweep (cooking x hatred was valid all along).
4. Keyword metric on Gemma reported 50% both-capture; hand-reading found ~0 clean.
"""
with open("/workspace/RESULTS/SESSION2_NOTES.md", "w", encoding="utf-8") as fh:
    fh.write(notes)

os.system("cd /workspace && tar -czf mats_results.tar.gz RESULTS/ && ls -lh mats_results.tar.gz")
print()
os.system("ls -1 /workspace/RESULTS/")