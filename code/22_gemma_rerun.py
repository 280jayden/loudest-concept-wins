"""
PASS 1 of the corrected Gemma rerun: load the NLA verbaliser, re-run Gate 1 in the
corrected format, then generate all sweep descriptions and save them.

Corrected construction (s=30000, calibrated in 21_calibrate_s.py):
    mix = a*dA + (1-a)*dB  -> renormalise -> v = b_dec + 30000*mix
Measured ratios now track intent (25% share -> 0.21-0.39 measured, target 0.33) and
every cell sits above the SAE detection floor.

Scoring happens in PASS 2 (23_gemma_score.py) after freeing the verbaliser, because
verbaliser + reconstructor + base Gemma do not fit in memory together.
"""
import os, yaml, torch, pickle, json, hashlib
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

S_FIXED = 30000.0
OUT = "/workspace/gemma_rerun_descriptions.pkl"

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


def make_fixed_s(fa, fb, alpha, s=S_FIXED):
    dA, dB = W_DEC[fa].float(), W_DEC[fb].float()
    dA = dA / dA.norm(); dB = dB / dB.norm()
    mix = alpha * dA + (1 - alpha) * dB
    mix = mix / mix.norm().clamp_min(1e-12)
    return B_DEC.float() + s * mix


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
 "programming": ["code", "python", "function", "script", "program", "software"],
 "travel": ["travel", "hotel", "hostel", "flight", "trip", "itinerar", "tourist"],
 "explosives": ["explosive", "bomb", "weapon", "detonat", "blast"],
 "hatred_discrim": ["hate", "hatred", "racis", "discriminat", "prejudice", "stereotype"],
 "malware": ["malware", "virus", "hacking", "cyber", "phishing", "exploit"],
 "ransomware": ["ransom", "extort", "malware", "cyber", "encrypt"],
}


def mentions(t, n):
    return any(w in (t or "").lower() for w in MENTION[n])


if __name__ == "__main__":
    res = pickle.load(open(OUT, "rb")) if os.path.exists(OUT) else {}

    # ---- Gate 1 in the corrected format ----
    print("\n=== GATE 1 (corrected format, s=30000) ===")
    for n in ["cooking", "legal", "programming", "travel",
              "explosives", "hatred_discrim", "malware", "ransomware"]:
        k = ("gate1", n)
        if k not in res:
            fi = FEATS_FULL[n]
            d = W_DEC[fi].float() / W_DEC[fi].float().norm()
            v = B_DEC.float() + S_FIXED * d
            hits, first = 0, None
            for s in range(3):
                e = verbalize(v, seed=700 + s)
                if mentions(e, n):
                    hits += 1
                if s == 0:
                    first = e
            res[k] = {"hits": hits, "example": first}
            pickle.dump(res, open(OUT, "wb"))
        r = res[k]
        print(f"  {'PASS' if r['hits']>=2 else 'FAIL'} {n:16} {r['hits']}/3  {r['example'][:95]!r}")

    # ---- sweep descriptions ----
    print("\n=== generating sweep descriptions ===")
    for m, c in PAIRS8:
        fa, fb = FEATS_FULL[m], FEATS_FULL[c]
        for al in [0.5, 0.75, 0.9]:
            key = ("sweep", m, c, al)
            if key in res:
                continue
            v = make_fixed_s(fa, fb, al)
            a = sae_encode(v)
            gt = (float(a[fa]), float(a[fb]))
            base_seed = int(hashlib.md5(f"{m}|{c}|{al}".encode()).hexdigest()[:6], 16)
            rows = [{"expl": verbalize(v, seed=base_seed + s)} for s in range(6)]
            for r in rows:
                r["mA"] = mentions(r["expl"], m)
                r["mB"] = mentions(r["expl"], c)
            res[key] = {"rows": rows, "gt": gt, "ratio": gt[1] / gt[0] if gt[0] else 0}
            pickle.dump(res, open(OUT, "wb"))
            print(f"  {m} x {c:16} share={int((1-al)*100)}%  ratio={res[key]['ratio']:.2f}  "
                  f"kw: A={sum(r['mA'] for r in rows)}/6 B={sum(r['mB'] for r in rows)}/6")
    print("\nPASS 1 done ->", OUT)
