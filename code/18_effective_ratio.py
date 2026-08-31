"""
Detection vs EFFECTIVE activation ratio, not nominal alpha.

WHY THIS MATTERS
We set alpha (e.g. 0.9 = "concerning concept at 10%"), but alpha is a coefficient on
UNIT-NORMALISED decoder directions. Features have different natural activation
magnitudes, so the same alpha produces very different ratios in SAE activation terms:

    programming x malware @ a=0.9 -> act 8,300 vs 3,835  = ratio 0.46  (detected 6/6)
    cooking     x malware @ a=0.9 -> act 23,249 vs 4,368 = ratio 0.19  (detected 0/6)

Same nominal 10% share, same concerning feature, HIGHER absolute activation in the
silent case - but a weak anchor leaves the minority relatively louder.

So the honest x-axis is  ratio = act_B / act_A  measured by the SAE encoder, not alpha.
This also explains why the "semantic relatedness" story failed to replicate: the two
Gemma survivors both had weak anchors, which is a simpler explanation than semantics.
"""
import pickle, json, numpy as np

THR = 0.3
SCALES = [0.5, 0.8, 1.3, 2.1, 3.4, 5.5]


def llama_points():
    """(effective ratio, detected?) for every Llama safety-sweep generation."""
    ss = pickle.load(open("/workspace/safety_sweep.pkl", "rb"))
    meta = pickle.load(open("/workspace/safe_meta.pkl", "rb"))
    V = meta["VALID2"]
    pts = []
    for nm in sorted({k[0] for k in ss}):
        for al in [0.5, 0.75, 0.9]:
            v = [t for t in V[nm] if t[0] == al]
            if not v or not v[0][2]:
                continue
            # V rows are (alpha, a_fires, b_fires, b_value); recover A from the sweep meta
            gtB = v[0][3] if len(v[0]) > 3 else None
            rows = []
            for sc in SCALES:
                rows += ss.get((nm, al, sc), [])
            if not rows:
                continue
            worked = [r for r in rows if r["hit_A"] >= THR]
            if not worked:
                continue
            det = sum(1 for r in worked if r["hit_B"] >= THR)
            pts.append({"model": "llama", "pair": nm, "alpha": al,
                        "gtB": gtB, "n": len(worked), "det": det})
    return pts


def gemma_points():
    gs = pickle.load(open("/workspace/gemma_sweep_full.pkl", "rb"))
    lp = pickle.load(open("/workspace/gemma_llama_protocol.pkl", "rb"))
    pts = []
    for (m, c, al), rows in lp.items():
        d = gs.get((m, c, al))
        if not d or not d.get("valid"):
            continue
        gtA, gtB = d["gt"]
        det = sum(1 for r in rows if r["hit_B"] >= THR)
        pts.append({"model": "gemma", "pair": f"{m} x {c}", "alpha": al,
                    "ratio": gtB / gtA, "gtA": gtA, "gtB": gtB,
                    "n": len(rows), "det": det})
    return pts


if __name__ == "__main__":
    gp = gemma_points()
    print("=== GEMMA: detection vs EFFECTIVE activation ratio ===")
    print(f"{'pair':30}{'alpha':>6}{'act_A':>9}{'act_B':>8}{'ratio':>8}{'detected':>11}")
    for p in sorted(gp, key=lambda x: x["ratio"]):
        print(f"{p['pair']:30}{p['alpha']:>6}{p['gtA']:>9.0f}{p['gtB']:>8.0f}"
              f"{p['ratio']:>8.3f}{p['det']:>7}/{p['n']}")

    print("\n=== binned by effective ratio (Gemma, Llama protocol) ===")
    bins = [(0.0, 0.10), (0.10, 0.20), (0.20, 0.35), (0.35, 0.60), (0.60, 2.0)]
    print(f"{'ratio bin':>16}{'cells':>7}{'detected':>12}")
    for lo, hi in bins:
        sel = [p for p in gp if lo <= p["ratio"] < hi]
        if not sel:
            continue
        det = sum(p["det"] for p in sel); n = sum(p["n"] for p in sel)
        print(f"{f'{lo:.2f}-{hi:.2f}':>16}{len(sel):>7}{det:>7}/{n:<4} = {det/n*100:.0f}%")
    print("\n  false-positive floor for this metric on Gemma: 18.8%")
    json.dump([{k: (float(v) if isinstance(v, (int, float)) else v) for k, v in p.items()}
               for p in gp], open("/workspace/effective_ratio.json", "w"), indent=1)
