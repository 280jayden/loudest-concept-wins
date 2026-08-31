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
