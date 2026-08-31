"""
Validate the Gemma behavioural metric on SAVED explanations, then rescore the sweep.

Ground truth we can exploit: at alpha=1.0 the vector is PURE anchor, at alpha=0.0 PURE
concerning. So an explanation from alpha=1.0 should fire the anchor feature and NOT the
concerning one, and vice versa. That gives sensitivity and specificity without needing
the verbaliser loaded (it is unloaded to fit base Gemma in memory).

Then rescore every mixture explanation with whichever metric validates, and compare
against the keyword numbers - the gap is how wrong keyword matching was.
"""
import pickle, json, numpy as np

GS = pickle.load(open("/workspace/gemma_sweep_full.pkl", "rb"))
G90 = pickle.load(open("/workspace/gemma_sweep_90.pkl", "rb")) if __import__("os").path.exists("/workspace/gemma_sweep_90.pkl") else {}


def validate():
    """alpha=0.5..0.9 explanations: does the ANCHOR feature fire? (it should - anchor dominates)
       and how often does a RANDOM OTHER feature fire? (it should not)"""
    names = list(FEATS_FULL)
    hitA = nA = spur = nspur = 0
    print(f"{'pair':30}{'alpha':>6}{'anchor fires':>14}{'random-other fires':>20}")
    for (m, c, al), d in sorted(GS.items(), key=str):
        if not d.get("valid") or al < 0.75:
            continue                      # anchor clearly dominant at 0.75/0.9
        fa = FEATS_FULL[m]
        others = [FEATS_FULL[o] for o in names if o not in (m, c)][:4]
        a = s = 0
        for r in d["rows"][:6]:
            f = gemma_sae_fires(r["expl"], [fa] + others)
            a += (f[fa] > 0)
            s += any(f[o] > 0 for o in others)
        hitA += a; nA += min(6, len(d["rows"]))
        spur += s; nspur += min(6, len(d["rows"]))
        print(f"{m+' x '+c:30}{al:>6}{a:>10}/{min(6,len(d['rows'])):<3}{s:>16}/{min(6,len(d['rows']))}")
    print(f"\n  SENSITIVITY  anchor feature fires on its own explanation: {hitA}/{nA} = {hitA/nA*100:.1f}%")
    print(f"  SPURIOUS     an unrelated feature fires:                  {spur}/{nspur} = {spur/nspur*100:.1f}%")
    return hitA / nA, spur / nspur


def rescore():
    """Behavioural rescore of every mixture explanation, vs the keyword result."""
    print(f"\n{'pair':30}{'alpha':>6}{'keyword B':>11}{'behav B':>10}")
    agg = {}
    for (m, c, al), d in sorted(GS.items(), key=str):
        if not d.get("valid"):
            continue
        fb = FEATS_FULL[c]
        kw = sum(1 for r in d["rows"] if r["mB"])
        bh = 0
        for r in d["rows"]:
            f = gemma_sae_fires(r["expl"], [fb])
            bh += (f[fb] > 0)
            r["behav_B"] = f[fb] > 0
        n = len(d["rows"])
        agg.setdefault(al, [0, 0, 0])
        agg[al][0] += kw; agg[al][1] += bh; agg[al][2] += n
        print(f"{m+' x '+c:30}{al:>6}{kw:>7}/{n:<3}{bh:>6}/{n:<3}")
    pickle.dump(GS, open("/workspace/gemma_sweep_full.pkl", "wb"))
    print(f"\n{'concerning share':>18}{'keyword':>14}{'behavioural':>16}")
    for al in sorted(agg, reverse=True):
        k, b, n = agg[al]
        share = int((1 - al) * 100)
        print(f"{str(share)+'%':>18}{k:>8}/{n:<5}{b:>10}/{n:<5}")
    json.dump({str(k): v for k, v in agg.items()},
              open("/workspace/gemma_behavioural_scores.json", "w"), indent=1)
    return agg


if __name__ == "__main__":
    print("=== VALIDATION ===")
    sens, spur = validate()
    if sens < 0.5:
        print("\n  WARNING: sensitivity too low - direct scoring of explanation text does not")
        print("  work; fall back to generating conversations (the Llama protocol).")
    print("\n=== RESCORE ===")
    rescore()
