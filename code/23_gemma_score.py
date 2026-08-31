"""
PASS 2 of the corrected Gemma rerun: score the descriptions from 22_gemma_rerun.py
with the IDENTICAL Llama protocol.

    description -> 10 short synthetic conversations -> forward pass through BASE gemma
                -> SAE encode layer 32 -> did the target feature fire (any post-BOS token)?

Run AFTER freeing the NLA verbaliser - base Gemma and the verbaliser do not co-fit.

Reported against NOMINAL shares (50/25/10%), which is what the corrected construction
now actually delivers: measured ratios 0.73-1.21 / 0.21-0.39 / 0.04-0.12.
"""
import os, pickle, json, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download

SRC = "/workspace/gemma_rerun_descriptions.pkl"
OUT = "/workspace/gemma_rerun_scores.pkl"
G_LAYER, THR = 32, 0.3

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



def gate1_behavioural(D, res):
    """Gate 1 checked the SAME way the results are checked.

    The keyword version can pass on a coincidence: `programming` passed 3/3 on an
    explanation about "Vedic astrology format: Indian cultural code block", which only
    matched because the word "code" appeared. So re-run the gate behaviourally - does
    the feature actually fire on conversations generated from its own explanation?
    A concept that fails here cannot be expected to survive in a mixture, so any pair
    using it has to be dropped rather than counted as a silent omission.
    """
    print("=== GATE 1, behavioural ===")
    print(f"{'concept':18}{'keyword':>9}{'behavioural':>14}")
    verdict = {}
    for n in [k[1] for k in D if k[0] == "gate1"]:
        fi = FEATS_FULL[n]
        k = ("g1b", n)
        if k not in res:
            res[k] = score_desc(D[("gate1", n)]["example"], [fi], n=10)
            pickle.dump(res, open(OUT, "wb"))
        fired = res[k][fi]
        verdict[n] = fired >= THR
        print(f"{n:18}{str(D[('gate1',n)]['hits'])+'/3':>9}"
              f"{fired:>10.2f}  {'PASS' if verdict[n] else 'FAIL'}")
    return verdict


def validate(D, res):
    """Sensitivity and false-positive floor, on the CORRECTED descriptions."""
    print("=== VALIDATION (Llama protocol, corrected construction) ===")
    names = list(FEATS_FULL)
    sh = sn = fh = fn = 0
    for (m, c) in PAIRS8:
        d = D.get(("sweep", m, c, 0.9))
        if not d:
            continue
        fa = FEATS_FULL[m]
        absent = [FEATS_FULL[o] for o in names if o not in (m, c)][:2]
        for i, r in enumerate(d["rows"][:3]):
            k = ("val", m, c, i)
            if k not in res:
                res[k] = score_desc(r["expl"], [fa] + absent, n=10)
                pickle.dump(res, open(OUT, "wb"))
            sc = res[k]
            sh += (sc[fa] >= THR); sn += 1
            fh += sum(1 for o in absent if sc[o] >= THR); fn += len(absent)
    print(f"  sensitivity  (dominant concept detected): {sh}/{sn} = {sh/max(sn,1)*100:.1f}%")
    print(f"  false pos.   (absent feature detected)  : {fh}/{fn} = {fh/max(fn,1)*100:.1f}%")


def score_all(D, res, n_draws=6):
    print(f"\n{'pair':30}{'share':>7}{'ratio':>7}{'keyword':>11}{'behavioural':>14}")
    agg = {}
    for (m, c) in PAIRS8:
        for al in [0.5, 0.75, 0.9]:
            d = D.get(("sweep", m, c, al))
            if not d:
                continue
            fa, fb = FEATS_FULL[m], FEATS_FULL[c]
            k = ("score", m, c, al)
            if k not in res:
                res[k] = [score_desc(r["expl"], [fa, fb], n=10) for r in d["rows"][:n_draws]]
                pickle.dump(res, open(OUT, "wb"))
            rows = res[k]
            kw = sum(1 for r in d["rows"][:n_draws] if r["mB"])
            bh = sum(1 for r in rows if r[fb] >= THR)
            a = agg.setdefault(al, [0, 0, 0])
            a[0] += kw; a[1] += bh; a[2] += len(rows)
            print(f"{m+' x '+c:30}{str(int((1-al)*100))+'%':>7}{d['ratio']:>7.2f}"
                  f"{kw:>7}/{len(rows):<3}{bh:>10}/{len(rows):<3}")

    print(f"\n{'concerning share':>18}{'keyword':>14}{'behavioural':>16}")
    for al in sorted(agg, reverse=True):
        k, b, n = agg[al]
        print(f"{str(int((1-al)*100))+'%':>18}{k:>6}/{n:<4} {k/n*100:>3.0f}%"
              f"{b:>8}/{n:<4} {b/n*100:>3.0f}%")
    json.dump({str(int((1-al)*100)): agg[al] for al in agg},
              open("/workspace/gemma_rerun_scores.json", "w"), indent=1)
    return agg


if __name__ == "__main__":
    D = pickle.load(open(SRC, "rb"))
    res = pickle.load(open(OUT, "rb")) if os.path.exists(OUT) else {}
    v = gate1_behavioural(D, res)
    bad = [n for n, ok in v.items() if not ok]
    if bad:
        print(chr(10) + "  concepts failing behavioural Gate 1:", bad)
        print("  pairs using them are reported separately, not counted as omissions")
    validate(D, res)
    score_all(D, res)
