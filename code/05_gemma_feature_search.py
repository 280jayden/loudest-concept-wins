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
