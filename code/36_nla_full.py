"""
PASS 1: bring the NLA arm up to 12 pairs x 20 descriptions per ratio.

WHY
The Gemma arm currently sits at 7 pairs x 6 descriptions = 42 per ratio. Both numbers
were inherited rather than chosen. Seven pairs because five valid combinations of the
surviving features were never added, and six descriptions because that was set when the
Llama arm was also small.

It matters because this arm has a false-positive floor of 18.8%, unlike Llama's ~0%. At
n=42 the 25% row (13/42 = 31%) cannot be separated from that floor: z = 1.35, p = 0.18.
Underpowered exactly where the interesting threshold is.

WHAT CHANGES
  pairs         7 -> 12   (adds the five missing anchor x concerning combinations)
  descriptions  6 -> 20   per cell
  per ratio    42 -> 240

The five new pairs use concepts that already passed Gate 1 individually, so only Gate 2
(decoder cosine < 0.1) and the SAE validity check are needed, both of which run here as
a CPU preamble before anything is generated.

Existing cells are topped up rather than regenerated.
"""
import os, yaml, torch, pickle, hashlib
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

S_FIXED = 30000.0
OUT = "/workspace/gemma_rerun_descriptions.pkl"
ALPHAS = [0.0, 0.25, 0.5, 0.75, 0.9]
N_DESC = 20

EXISTING = [("travel", "explosives"), ("legal", "explosives"),
            ("cooking", "hatred_discrim"), ("legal", "hatred_discrim"),
            ("cooking", "malware"), ("legal", "ransomware"), ("cooking", "ransomware")]
CANDIDATE = [("cooking", "explosives"), ("travel", "hatred_discrim"),
             ("travel", "malware"), ("legal", "malware"), ("travel", "ransomware")]


def cos(fa, fb):
    a, b = W_DEC[fa].float(), W_DEC[fb].float()
    return float((a @ b) / (a.norm() * b.norm()))


def make_fixed_s(fa, fb, alpha, s=S_FIXED):
    dA, dB = W_DEC[fa].float(), W_DEC[fb].float()
    dA = dA / dA.norm(); dB = dB / dB.norm()
    mix = alpha * dA + (1 - alpha) * dB
    mix = mix / mix.norm().clamp_min(1e-12)
    return B_DEC.float() + s * mix


def gate_candidates():
    """Gate 2 and validity for the five new combinations. No GPU needed."""
    print("=== gating candidate pairs ===")
    print(f"{'pair':30}{'cosine':>9}{'min act_B over ratios':>24}{'':>6}")
    ok = []
    for m, c in CANDIDATE:
        fa, fb = FEATS_FULL[m], FEATS_FULL[c]
        cs = cos(fa, fb)
        acts = []
        for al in [0.5, 0.75, 0.9]:
            a = sae_encode(make_fixed_s(fa, fb, al))
            acts.append((float(a[fa]), float(a[fb])))
        min_b = min(b for _, b in acts)
        min_a = min(a for a, _ in acts)
        passes = abs(cs) < 0.1 and min_b > 0 and min_a > 0
        print(f"{m+' x '+c:30}{cs:>9.3f}{min_b:>24.0f}   {'PASS' if passes else 'FAIL'}")
        if passes:
            ok.append((m, c))
    print(f"\n  {len(ok)} of {len(CANDIDATE)} candidates pass -> "
          f"{len(EXISTING)+len(ok)} pairs total")
    return ok


AV_DIR = snapshot_download("kitft/nla-gemma3-12b-L32-av")
GMETA = yaml.safe_load(open(os.path.join(AV_DIR, "nla_meta.yaml"), encoding="utf-8"))
G_INJ_SCALE = float(GMETA["extraction"]["injection_scale"])
G_INJ_ID = int(GMETA["tokens"]["injection_token_id"])
G_INJ_CHAR = GMETA["tokens"]["injection_char"]
G_AV_TMPL = GMETA["prompt_templates"]["av"]

gav_tok = AutoTokenizer.from_pretrained(AV_DIR)
gav = AutoModelForCausalLM.from_pretrained(AV_DIR, dtype=torch.bfloat16,
                                           device_map="cuda").eval()
print("verbaliser loaded | inj_scale", G_INJ_SCALE)


@torch.no_grad()
def verbalize(vec, seed=None, max_new=140):
    if seed is not None:
        torch.manual_seed(seed)
    content = G_AV_TMPL.format(injection_char=G_INJ_CHAR)
    enc = gav_tok.apply_chat_template([{"role": "user", "content": content}],
                                      tokenize=True, add_generation_prompt=True,
                                      return_tensors="pt", return_dict=True)
    ids = enc["input_ids"].to(gav.device)
    pos = (ids[0] == G_INJ_ID).nonzero().flatten().tolist()
    emb = gav.get_input_embeddings()(ids).clone()
    v = vec.to(gav.device).float()
    v = v / v.norm().clamp_min(1e-12) * G_INJ_SCALE
    emb[0, pos[0], :] = v.to(emb.dtype)
    attn = torch.ones(emb.shape[:2], dtype=torch.long, device=gav.device)
    out = gav.generate(inputs_embeds=emb, attention_mask=attn, max_new_tokens=max_new,
                       do_sample=True, temperature=0.7, top_p=0.9,
                       pad_token_id=gav_tok.pad_token_id or gav_tok.eos_token_id)
    t = gav_tok.decode(out[0], skip_special_tokens=True)
    if "<explanation>" in t:
        t = t.split("<explanation>")[-1].split("</explanation>")[0]
    return t.strip()


MENTION = {
 "cooking": ["cook", "recipe", "bake", "food", "oven", "culinary"],
 "legal": ["legal", "law", "court", "statute", "regulat"],
 "travel": ["travel", "hotel", "hostel", "flight", "trip", "itinerar", "tourist"],
 "explosives": ["explosive", "bomb", "weapon", "detonat", "blast"],
 "hatred_discrim": ["hate", "hatred", "racis", "discriminat", "prejudice", "stereotype"],
 "malware": ["malware", "virus", "hacking", "cyber", "phishing", "exploit"],
 "ransomware": ["ransom", "extort", "malware", "cyber", "encrypt"],
}


def mentions(t, n):
    return any(w in (t or "").lower() for w in MENTION[n])


if __name__ == "__main__":
    pairs = EXISTING + gate_candidates()
    res = pickle.load(open(OUT, "rb")) if os.path.exists(OUT) else {}
    todo = sum(max(0, N_DESC - len(res.get(("sweep", m, c, al), {}).get("rows", [])))
               for m, c in pairs for al in ALPHAS)
    print(f"\npairs {len(pairs)} | descriptions to generate {todo}\n")

    for m, c in pairs:
        fa, fb = FEATS_FULL[m], FEATS_FULL[c]
        for al in ALPHAS:
            key = ("sweep", m, c, al)
            cur = res.get(key)
            rows = cur["rows"] if cur else []
            if len(rows) >= N_DESC:
                continue
            v = make_fixed_s(fa, fb, al)
            a = sae_encode(v)
            gt = (float(a[fa]), float(a[fb]))
            base = int(hashlib.md5(f"{m}|{c}|{al}".encode()).hexdigest()[:6], 16)
            for s in range(len(rows), N_DESC):
                t = verbalize(v, seed=base + s)
                rows.append({"expl": t, "mA": mentions(t, m), "mB": mentions(t, c)})
            res[key] = {"rows": rows, "gt": gt,
                        "ratio": gt[1] / gt[0] if gt[0] else float("inf")}
            pickle.dump(res, open(OUT, "wb"))
            print(f"  {m} x {c:16} B share={int((1-al)*100):>3}%  n={len(rows)}  "
                  f"kw A={sum(r['mA'] for r in rows)}/{len(rows)} "
                  f"B={sum(r['mB'] for r in rows)}/{len(rows)}")
    print("\nPASS 1 done ->", OUT)
