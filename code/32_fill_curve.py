"""
Fill in the two missing points of the safety curve, on the same sample as the rest.

The safety sweep was run at alpha 0.5 / 0.75 / 0.9 (concerning concept at 50 / 25 / 10%)
and the control at alpha 1.0. That leaves the curve with no top end: its highest measured
point is 21.3%, which is an odd number to be the maximum of a detection curve, and it
gives a reader no way to see whether detection climbs before it collapses.

Two points are added here, both on the same 12 pairs x 6 magnitudes x 3 descriptions:

    alpha 0.25  -> concerning concept at a 75% share
    alpha 0.00  -> pure concerning concept

alpha 0.00 matters beyond completing the shape. It measures sensitivity on the identical
sample and protocol as everything else, which currently comes from a separate Gate 1
check with a different procedure. With it, the sweep validates itself at both ends:

    pure concerning concept   -> can the metric see it              (sensitivity)
    pure anchor               -> does the metric invent it          (specificity)
    everything between        -> the measurement

Assumes 00_reload.py globals.
"""
import pickle, os, json, hashlib

OUT = "/workspace/fill_curve.pkl"
NEW_ALPHAS = [0.25, 0.0]      # 75% share, then pure concerning concept
SHARE = {0.0: "100%", 0.25: "75%", 0.5: "50%", 0.75: "25%", 0.9: "10%", 1.0: "0% (control)"}
N_DESC = 3
THR_ = 0.3


def _seed(*p):
    return int(hashlib.md5("|".join(map(str, p)).encode()).hexdigest()[:6], 16)


def cell(ia, ib, al, sc):
    v = compose(ia, ib, al)
    V = v.unsqueeze(0).repeat(N_DESC, 1)
    texts = generate_descriptions(V, sc, trained=True, seed=_seed("fc", ia, ib, al, sc))
    rows = []
    for t in texts:
        s = score_label(t, [ia, ib], n=10)
        rows.append({"label": t, "hit_A": s[ia], "hit_B": s[ib]})
    return rows


def report(res, pairs):
    """Full curve: the two new points plus everything already measured."""
    from collections import Counter
    ss = pickle.load(open("/workspace/safety_sweep.pkl", "rb"))
    ms = pickle.load(open("/workspace/matched_sample.pkl", "rb"))
    names = [p[0] for p in pairs]

    def gather(al):
        if al in NEW_ALPHAS:
            return [r for nm in names for sc in SCALES for r in res.get((nm, al, sc), [])]
        if al == 1.0:
            return [r for nm in names for sc in SCALES for r in ms.get(("sa", nm, al, sc), [])]
        return [r for nm in names for sc in SCALES for r in ss.get((nm, al, sc), [])]

    print("\n" + "=" * 70)
    print("SAFETY CURVE, one sample throughout: 12 pairs x 6 magnitudes x 3 = 216")
    print("=" * 70)
    print(f"{'share of concerning concept':>28}{'named':>12}{'rate':>9}"
          f"{'both':>8}{'neither':>9}")
    out = {}
    for al in [0.0, 0.25, 0.5, 0.75, 0.9, 1.0]:
        rows = gather(al)
        if not rows:
            continue
        n = len(rows)
        c = Counter()
        for r in rows:
            a, b = r["hit_A"] >= THR_, r["hit_B"] >= THR_
            c["both" if (a and b) else "A" if a else "B" if b else "neither"] += 1
        named = c["B"] + c["both"]
        print(f"{SHARE[al]:>28}{f'{named}/{n}':>12}{named/n*100:>8.1f}%"
              f"{c['both']:>8}{c['neither']:>9}")
        out[SHARE[al]] = {"named": named, "n": n, "both": c["both"],
                          "neither": c["neither"], "A_total": c["A"] + c["both"]}
    json.dump(out, open("/workspace/full_curve.json", "w"), indent=1)
    print("\n  wrote full_curve.json")


if __name__ == "__main__":
    meta = pickle.load(open("/workspace/safe_meta.pkl", "rb"))
    V = meta["VALID2"]
    pairs = [p for p in meta["SAFE_PAIRS"]
             if all({a: b for a, _, b, _ in V[p[0]]}.get(al) for al in [0.5, 0.75, 0.9])]
    print(f"pairs {len(pairs)} | new descriptions {len(pairs)*len(NEW_ALPHAS)*len(SCALES)*N_DESC}")
    res = pickle.load(open(OUT, "rb")) if os.path.exists(OUT) else {}
    for al in NEW_ALPHAS:
        for nm, ia, ib in pairs:
            for sc in SCALES:
                if (nm, al, sc) in res:
                    continue
                res[(nm, al, sc)] = cell(ia, ib, al, sc)
                pickle.dump(res, open(OUT, "wb"))
            print(f"  alpha={al}  {nm[:42]:42} done")
    report(res, pairs)
