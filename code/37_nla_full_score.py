"""
PASS 2: score the expanded NLA arm and re-measure its false-positive floor.

Run after freeing the verbaliser.

The floor matters more here than on the Llama arm. Llama's is ~0%, so almost any sample
separates a real effect from it. This SAE runs at roughly 120 active latents per token,
which puts the floor near 19%, and the 25% row sits close enough to it that the old n=42
could not tell them apart. The floor is therefore re-measured on the larger sample rather
than carried over.

Two independent estimates of it are printed:
  pure-B row      the anchor is genuinely absent, so any anchor detection is a false one
  absent-feature  descriptions scored against features not in the pair at all
"""
import os, pickle, json, torch
from collections import Counter
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download

SRC = "/workspace/gemma_rerun_descriptions.pkl"
OUT = "/workspace/gemma_full_scores.pkl"
G_LAYER, THR = 32, 0.3
ALPHAS = [0.0, 0.25, 0.5, 0.75, 0.9]
N_DESC = 20

BASE_DIR = snapshot_download("google/gemma-3-12b-it")
gb_tok = AutoTokenizer.from_pretrained(BASE_DIR)
gbase = AutoModelForCausalLM.from_pretrained(BASE_DIR, dtype=torch.bfloat16,
                                             device_map="cuda").eval()
print("base gemma loaded")

CONV_SYS_G = ("You are a helpful AI assistant who generates EXTREMELY SHORT example "
              "conversations. The conversations are between a user and an assistant, "
              "and have the following format:\n[USER] I'm a user.\n[ASSISTANT] I'm the assistant.")
CONV_PROMPT_G = ("Produce a VERY SHORT conversation which exhibits '_'\n"
                 "Do not include any other text in your response. "
                 "Start immediately with the conversation.")


@torch.no_grad()
def score_desc(desc, latents, n=10, max_new=100):
    msgs = [{"role": "system", "content": CONV_SYS_G},
            {"role": "user", "content": CONV_PROMPT_G.replace("_", desc[:400])}]
    gb_tok.padding_side = "left"
    enc = gb_tok.apply_chat_template([msgs] * n, tokenize=True, add_generation_prompt=True,
                                     return_tensors="pt", padding=True,
                                     return_dict=True).to(gbase.device)
    out = gbase.generate(**enc, max_new_tokens=max_new, do_sample=True, temperature=0.7,
                         top_p=0.9, pad_token_id=gb_tok.pad_token_id or gb_tok.eos_token_id)
    hits, valid = {li: 0 for li in latents}, 0
    for g in out:
        t = gb_tok.decode(g[enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        if not t:
            continue
        valid += 1
        ids = gb_tok(t, return_tensors="pt", truncation=True,
                     max_length=200)["input_ids"].to(gbase.device)
        hs = gbase(input_ids=ids, output_hidden_states=True).hidden_states[G_LAYER + 1][0]
        acts = sae_encode(hs.float().cpu())
        for li in latents:
            if (acts[1:, li] > 0).any().item():
                hits[li] += 1
    return {li: (hits[li] / valid if valid else 0.0) for li in latents}


if __name__ == "__main__":
    D = pickle.load(open(SRC, "rb"))
    # Only pairs that were topped up to N_DESC across every ratio. This excludes
    # programming x malware, which is still in the file from earlier runs but failed
    # behavioural Gate 1 at 0.20, and any candidate pair that failed gating in pass 1.
    cand = sorted({(k[1], k[2]) for k in D if k[0] == "sweep"})
    pairs = [p for p in cand
             if all(len(D.get(("sweep", p[0], p[1], al), {}).get("rows", [])) >= N_DESC
                    for al in ALPHAS)]
    dropped = [p for p in cand if p not in pairs]
    if dropped:
        print("excluded (not at full sample):", [f"{a} x {b}" for a, b in dropped])
    res = pickle.load(open(OUT, "rb")) if os.path.exists(OUT) else {}
    names = list(FEATS_FULL)
    print(f"pairs {len(pairs)}")

    for m, c in pairs:
        fa, fb = FEATS_FULL[m], FEATS_FULL[c]
        absent = [FEATS_FULL[o] for o in names if o not in (m, c)][:2]
        for al in ALPHAS:
            k = ("score", m, c, al)
            d = D.get(("sweep", m, c, al))
            if not d or k in res:
                continue
            res[k] = [score_desc(r["expl"], [fa, fb] + absent, n=10)
                      for r in d["rows"][:N_DESC]]
            pickle.dump(res, open(OUT, "wb"))
        print(f"  scored {m} x {c}")

    print("\nNATURAL LANGUAGE AUTOENCODER")
    print(f"{'B share':>9}{'A named':>16}{'B named':>16}{'both':>7}{'neither':>9}")
    out = {}
    for al in ALPHAS:
        rows = []
        for m, c in pairs:
            fa, fb = FEATS_FULL[m], FEATS_FULL[c]
            for r in res.get(("score", m, c, al), []):
                rows.append((r[fa] >= THR, r[fb] >= THR))
        if not rows:
            continue
        cc = Counter()
        for a, b in rows:
            cc["both" if (a and b) else "A" if a else "B" if b else "neither"] += 1
        A, B, n = cc["A"] + cc["both"], cc["B"] + cc["both"], len(rows)
        print(f"{int((1-al)*100):>8}%{A:>10}/{n:<4}{A/n*100:>4.0f}%"
              f"{B:>10}/{n:<4}{B/n*100:>4.0f}%{cc['both']:>7}{cc['neither']:>9}")
        out[f"{int((1-al)*100)}%"] = {"A": A, "B": B, "n": n,
                                     "both": cc["both"], "neither": cc["neither"]}

    # floor estimate 1: anchor detections on the pure-B row, where the anchor is absent
    pb = out.get("100%")
    if pb:
        print(f"\n  floor, anchor named on pure-B rows: {pb['A']}/{pb['n']} = "
              f"{pb['A']/pb['n']*100:.1f}%")

    # floor estimate 2: features that are in neither position of the pair
    fh = fn = 0
    for m, c in pairs:
        absent = [FEATS_FULL[o] for o in names if o not in (m, c)][:2]
        for al in ALPHAS:
            for r in res.get(("score", m, c, al), []):
                for o in absent:
                    if o in r:
                        fn += 1
                        fh += (r[o] >= THR)
    if fn:
        print(f"  floor, absent features:            {fh}/{fn} = {fh/fn*100:.1f}%")
    json.dump(out, open("/workspace/nla_full_curve_v2.json", "w"), indent=1)
