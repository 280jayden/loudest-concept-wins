"""
Score Gemma with the IDENTICAL Llama protocol.

ONE protocol for both models:
    description -> 10 short synthetic conversations -> forward pass -> SAE encode
    -> did the target feature fire (any post-BOS token)?

WHY NOT SCORE THE NLA EXPLANATION TEXT DIRECTLY
It was the obvious shortcut - NLA explanations already quote example sentences, unlike
Llama's short labels. We tried it and it fails validation:

    sensitivity  96/96 = 100%      (anchor fires on its own explanation)
    spurious     81/96 = 84.4%     (an UNRELATED feature also fires)
                                   ~36% false-positive rate per feature

Cause: this SAE has L0=120 (about 120 features active per token) and NLA explanations
run 200-400 tokens, so nearly any feature fires somewhere in the text. Llama's protocol
avoids this by scoring SHORT (~50 token) generated conversations, where "did this
feature fire" is discriminative. Same reason AE Studio generate text rather than
scoring the label: the check needs short, on-topic text.

So we discard direct scoring and use the Llama protocol for both models.
"""
import pickle, json, torch

GS_PATH = "/workspace/gemma_sweep_full.pkl"
OUT = "/workspace/gemma_llama_protocol.pkl"


@torch.no_grad()
def score_gemma_llama_style(desc, latents, n=10, max_new=100):
    """Identical to Llama's score_label, but through base Gemma + the Gemma SAE."""
    msgs = [{"role": "system", "content": CONV_SYS_G},
            {"role": "user", "content": CONV_PROMPT_G.replace("_", desc[:400])}]
    gb_tok.padding_side = "left"
    enc = gb_tok.apply_chat_template([msgs] * n, tokenize=True, add_generation_prompt=True,
                                     return_tensors="pt", padding=True,
                                     return_dict=True).to(gbase.device)
    out = gbase.generate(**enc, max_new_tokens=max_new, do_sample=True, temperature=0.7,
                         top_p=0.9, pad_token_id=gb_tok.pad_token_id or gb_tok.eos_token_id)
    hits = {li: 0 for li in latents}
    valid = 0
    for g in out:
        t = gb_tok.decode(g[enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        if not t:
            continue
        valid += 1
        ids = gb_tok(t, return_tensors="pt", truncation=True, max_length=200)["input_ids"].to(gbase.device)
        hs = gbase(input_ids=ids, output_hidden_states=True).hidden_states[G_LAYER + 1][0]
        acts = sae_encode(hs.float().cpu())
        for li in latents:
            if (acts[1:, li] > 0).any().item():
                hits[li] += 1
    return {li: (hits[li] / valid if valid else 0.0) for li in latents}


def validate_protocol():
    """Same three checks we ran on Llama: sensitivity, and false positives on
       (a) a concept that is absent, (b) unrelated features."""
    print("=== VALIDATION (Llama protocol, on Gemma) ===")
    GS = pickle.load(open(GS_PATH, "rb"))
    names = list(FEATS_FULL)
    sens_h = sens_n = fp_h = fp_n = 0
    for (m, c, al), d in sorted(GS.items(), key=str):
        if not d.get("valid") or al != 0.9:
            continue                                   # anchor dominant, concerning at 10%
        fa = FEATS_FULL[m]
        absent = [FEATS_FULL[o] for o in names if o not in (m, c)][:2]
        for r in d["rows"][:3]:
            sc = score_gemma_llama_style(r["expl"], [fa] + absent, n=10)
            sens_h += (sc[fa] >= 0.3); sens_n += 1
            fp_h += sum(1 for o in absent if sc[o] >= 0.3); fp_n += len(absent)
    print(f"  sensitivity (dominant concept detected): {sens_h}/{sens_n} = {sens_h/sens_n*100:.1f}%")
    print(f"  false positive (absent feature detected): {fp_h}/{fp_n} = {fp_h/fp_n*100:.1f}%")
    return sens_h / max(sens_n, 1), fp_h / max(fp_n, 1)


def rescore_all(n_draws=6):
    """Rescore every mixture explanation with the Llama protocol."""
    GS = pickle.load(open(GS_PATH, "rb"))
    res = pickle.load(open(OUT, "rb")) if __import__("os").path.exists(OUT) else {}
    print(f"\n{'pair':30}{'alpha':>6}{'keyword':>10}{'llama-protocol':>17}")
    agg = {}
    for (m, c, al), d in sorted(GS.items(), key=str):
        if not d.get("valid"):
            continue
        key = (m, c, al)
        fa, fb = FEATS_FULL[m], FEATS_FULL[c]
        if key not in res:
            rows = []
            for r in d["rows"][:n_draws]:
                sc = score_gemma_llama_style(r["expl"], [fa, fb], n=10)
                rows.append({"hit_A": sc[fa], "hit_B": sc[fb]})
            res[key] = rows
            pickle.dump(res, open(OUT, "wb"))
        rows = res[key]
        kw = sum(1 for r in d["rows"][:n_draws] if r["mB"])
        lp = sum(1 for r in rows if r["hit_B"] >= 0.3)
        agg.setdefault(al, [0, 0, 0])
        agg[al][0] += kw; agg[al][1] += lp; agg[al][2] += len(rows)
        print(f"{m+' x '+c:30}{al:>6}{kw:>6}/{len(rows):<3}{lp:>12}/{len(rows):<3}")
    print(f"\n{'concerning share':>18}{'keyword':>14}{'llama-protocol':>18}")
    for al in sorted(agg, reverse=True):
        k, l, n = agg[al]
        print(f"{str(int((1-al)*100))+'%':>18}{k:>8}/{n:<5}{l:>12}/{n:<5}")
    json.dump({str(k): v for k, v in agg.items()},
              open("/workspace/gemma_llama_protocol_scores.json", "w"), indent=1)
    return agg


if __name__ == "__main__":
    validate_protocol()
    rescore_all()
