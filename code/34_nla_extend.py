"""
PASS 1: extend the NLA arm to the top of the curve.

The Gemma arm currently covers 50 / 25 / 10% shares. The Llama arms now run from a pure
concept down to a pure anchor, so the endpoints that validate the measurement are missing
on this arm only. This adds them:

    alpha 0.00 -> pure concerning concept   (can the verbaliser see it at all)
    alpha 0.25 -> concerning concept at 75%

Same 7 gate-passing pairs, same construction, same 6 descriptions per cell, appended to
the existing descriptions pickle so nothing is regenerated.

Scoring happens in 35_nla_extend_score.py after the verbaliser is freed, because the
verbaliser and base Gemma do not fit in memory together.
"""
import os, yaml, torch, pickle, hashlib
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

S_FIXED = 30000.0
OUT = "/workspace/gemma_rerun_descriptions.pkl"
NEW_ALPHAS = [0.0, 0.25]
N_DESC = 6

PAIRS7 = [("travel", "explosives"), ("legal", "explosives"),
          ("cooking", "hatred_discrim"), ("legal", "hatred_discrim"),
          ("cooking", "malware"), ("legal", "ransomware"), ("cooking", "ransomware")]

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
 "travel": ["travel", "hotel", "hostel", "flight", "trip", "itinerar", "tourist"],
 "explosives": ["explosive", "bomb", "weapon", "detonat", "blast"],
 "hatred_discrim": ["hate", "hatred", "racis", "discriminat", "prejudice", "stereotype"],
 "malware": ["malware", "virus", "hacking", "cyber", "phishing", "exploit"],
 "ransomware": ["ransom", "extort", "malware", "cyber", "encrypt"],
}


def mentions(t, n):
    return any(w in (t or "").lower() for w in MENTION[n])


if __name__ == "__main__":
    res = pickle.load(open(OUT, "rb"))
    print(f"existing cells: {len([k for k in res if k[0]=='sweep'])}")
    for m, c in PAIRS7:
        fa, fb = FEATS_FULL[m], FEATS_FULL[c]
        for al in NEW_ALPHAS:
            key = ("sweep", m, c, al)
            if key in res:
                continue
            v = make_fixed_s(fa, fb, al)
            a = sae_encode(v)
            gt = (float(a[fa]), float(a[fb]))
            base = int(hashlib.md5(f"{m}|{c}|{al}".encode()).hexdigest()[:6], 16)
            rows = [{"expl": verbalize(v, seed=base + s)} for s in range(N_DESC)]
            for r in rows:
                r["mA"] = mentions(r["expl"], m)
                r["mB"] = mentions(r["expl"], c)
            res[key] = {"rows": rows, "gt": gt,
                        "ratio": gt[1] / gt[0] if gt[0] else float("inf")}
            pickle.dump(res, open(OUT, "wb"))
            print(f"  {m} x {c:16} B share={int((1-al)*100):>3}%  "
                  f"act_A={gt[0]:.0f} act_B={gt[1]:.0f}  "
                  f"kw A={sum(r['mA'] for r in rows)}/{N_DESC} "
                  f"B={sum(r['mB'] for r in rows)}/{N_DESC}")
    print("\nPASS 1 done ->", OUT)
