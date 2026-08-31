"""
PASS 2: score the new NLA endpoint cells with the same generation-scoring protocol.

Run after freeing the verbaliser. Appends to the existing scores pickle, then prints the
full NLA curve including the endpoints, so it can sit next to the two Llama tables.
"""
import os, pickle, json, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download

SRC = "/workspace/gemma_rerun_descriptions.pkl"
OUT = "/workspace/gemma_rerun_scores.pkl"
G_LAYER, THR = 32, 0.3
ALL_ALPHAS = [0.0, 0.25, 0.5, 0.75, 0.9]
N_DESC = 6

PAIRS7 = [("travel", "explosives"), ("legal", "explosives"),
          ("cooking", "hatred_discrim"), ("legal", "hatred_discrim"),
          ("cooking", "malware"), ("legal", "ransomware"), ("cooking", "ransomware")]

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
    from collections import Counter
    D = pickle.load(open(SRC, "rb"))
    res = pickle.load(open(OUT, "rb")) if os.path.exists(OUT) else {}

    for m, c in PAIRS7:
        fa, fb = FEATS_FULL[m], FEATS_FULL[c]
        for al in [0.0, 0.25]:
            k = ("score", m, c, al)
            if k in res:
                continue
            d = D.get(("sweep", m, c, al))
            if not d:
                continue
            res[k] = [score_desc(r["expl"], [fa, fb], n=10) for r in d["rows"][:N_DESC]]
            pickle.dump(res, open(OUT, "wb"))
            print(f"  scored {m} x {c} at B share {int((1-al)*100)}%")

    print("\nNATURAL LANGUAGE AUTOENCODER, full curve")
    print(f"{'B share':>9}{'A named':>16}{'B named':>16}{'both':>7}{'neither':>9}")
    out = {}
    for al in ALL_ALPHAS:
        rows = []
        for m, c in PAIRS7:
            fa, fb = FEATS_FULL[m], FEATS_FULL[c]
            for r in res.get(("score", m, c, al), []):
                rows.append((r[fa] >= THR, r[fb] >= THR))
        if not rows:
            continue
        cc = Counter()
        for a, b in rows:
            cc["both" if (a and b) else "A" if a else "B" if b else "neither"] += 1
        A, B, n = cc["A"] + cc["both"], cc["B"] + cc["both"], len(rows)
        print(f"{int((1-al)*100):>8}%{A:>10}/{n:<3} {A/n*100:>4.0f}%"
              f"{B:>10}/{n:<3} {B/n*100:>4.0f}%{cc['both']:>7}{cc['neither']:>9}")
        out[f"{int((1-al)*100)}%"] = {"A": A, "B": B, "n": n,
                                      "both": cc["both"], "neither": cc["neither"]}
    json.dump(out, open("/workspace/nla_full_curve.json", "w"), indent=1)
    print("\n  false-positive floor for this arm: 9/48 = 18.8%")
