"""
Expand the Gemma feature pool so the cross-architecture comparison is even.

CURRENT IMBALANCE
  Llama : 7 concerning concepts x 2 mundane anchors = 14 pairs (12 valid at 10%)
  Gemma : 2 concerning concepts x 1 mundane anchor  =  2 pairs

TARGET
  ~5-6 concerning x 2 anchors = 10-12 pairs, matching Llama's structure.

We need BOTH more concerning concepts and more mundane anchors - Llama's pair count
came from having 3 anchors (cooking/baking/spices), Gemma has only cooking so far.
"""
import os, json, torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from transformers import AutoTokenizer

REPO, SAE_PATH = "google/gemma-scope-2-12b-it", "resid_post_all/layer_32_width_16k_l0_big"
base = os.path.join(snapshot_download(REPO, allow_patterns=[f"{SAE_PATH}/*"]), SAE_PATH)
P, EX = load_file(os.path.join(base, "params.safetensors")), load_file(os.path.join(base, "examples.safetensors"))
W_ENC, W_DEC = P["w_enc"], P["w_dec"]
B_ENC, B_DEC, THRESH = P["b_enc"], P["b_dec"], P["threshold"]
SEQ, POS, ACT, TOKENS = EX["seq_ids"], EX["positions"], EX["activations"], EX["tokens"]
FREQ = EX["feature_frequencies"]

gtok = AutoTokenizer.from_pretrained(
    snapshot_download("kitft/nla-gemma3-12b-L32-av",
                      allow_patterns=["tokenizer*", "*.json", "*.model"]))


def feature_context(fi, k=6, window=14):
    out = []
    for j in ACT[fi].argsort(descending=True)[:k]:
        s, p = int(SEQ[fi, j]), int(POS[fi, j])
        if s < 0 or s >= TOKENS.shape[0]:
            continue
        lo, hi = max(0, p - window), min(TOKENS.shape[1], p + window)
        out.append((float(ACT[fi, j]),
                    gtok.decode(TOKENS[s, lo:hi].tolist(), skip_special_tokens=True).replace("\n", " ")))
    return out


def search(keywords, topn=10, k=6, min_hits=4, exclude=()):
    kws = [w.lower() for w in keywords]
    exc = [w.lower() for w in exclude]
    scored = []
    for fi in range(W_DEC.shape[0]):
        if float(FREQ[fi]) <= 0:
            continue
        ctx = feature_context(fi, k=k, window=10)
        hits = sum(any(w in c[1].lower() for w in kws) for c in ctx)
        bad = sum(any(w in c[1].lower() for w in exc) for c in ctx)
        if hits >= min_hits and bad <= 1:
            scored.append((hits, float(ACT[fi, 0]), fi))
    scored.sort(reverse=True)
    return scored[:topn]


CONCERNING = {
 "weapons":     ["weapon", "firearm", "gun", "ammunition", "rifle", "shooting"],
 "cyberattack": ["hacking", "malware", "ransomware", "exploit", "phishing", "breach"],
 "selfharm":    ["self-harm", "suicide", "suicidal", "hurting yourself"],
 "csam":        ["child", "minor", "exploitation", "abuse of a child"],
 "bioweapon":   ["pathogen", "anthrax", "bioweapon", "toxin", "nerve agent"],
 "violence2":   ["assault", "attack", "murder", "kill", "violent crime"],
 "extortion":   ["blackmail", "extortion", "ransom", "coercion", "threaten"],
}
MUNDANE = {
 "sports":      ["football", "soccer", "basketball", "tournament", "athlete", "match"],
 "weather":     ["weather", "forecast", "rain", "temperature", "storm", "climate"],
 "music":       ["music", "guitar", "album", "song", "band", "melody"],
 "travel":      ["travel", "flight", "hotel", "itinerary", "destination", "tourist"],
 "gardening":   ["garden", "plant", "soil", "seed", "flower", "prune"],
 "programming": ["python", "function", "code", "variable", "debug", "compile"],
}

if __name__ == "__main__":
    found = {}
    for group, qs in [("CONCERNING", CONCERNING), ("MUNDANE", MUNDANE)]:
        print(f"\n{'='*70}\n{group}\n{'='*70}")
        for name, kws in qs.items():
            res = search(kws, topn=4)
            found[name] = res
            print(f"\n--- {name}: {len(res)} candidates ---")
            for hits, a0, fi in res[:2]:
                print(f"  f{fi:6d} hits={hits}/6 act={a0:.0f} freq={float(FREQ[fi]):.5f}")
                for a, txt in feature_context(fi, k=2, window=12):
                    print(f"      {txt[:96]}")
    json.dump({k: [[int(h), float(a), int(f)] for h, a, f in v] for k, v in found.items()},
              open("/workspace/gemma_expand_candidates.json", "w"), indent=1)
    print("\nsaved -> /workspace/gemma_expand_candidates.json")
