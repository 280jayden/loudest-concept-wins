"""
Rerun the whole curve at the magnitude the adapter was actually trained on.

WHY
The adapter is f(h) = alpha*h + b, and every input vector was L2-normalised to unit norm
during training (agencyenterprise/selfie-adapters). The learned alpha therefore already
carries the injection magnitude. Our earlier runs passed normalize_input=False and
multiplied by a manual scale, which means the sweep was over MULTIPLES of the trained
norm: 0.5x to 5.5x.

Sweeping was defensible and showed the collapse holds at every multiple, but it costs a
magnitude axis, an aggregate that needs explaining (72.7% at a pure concept, because it
averages over multiples where injection fails), and a working-band caveat. Running at the
trained magnitude removes all of that.

scale=1.0 IS the trained condition. generate_descriptions normalises to unit norm and
multiplies by scale, so at 1.0 the adapter receives exactly what it was fitted on, and
normalize_input=False on an already-unit vector is identical to normalize_input=True.

PREDICTION (state before looking)
The earlier sweep gives 75% at 0.8x and 81% at 1.3x on a pure concept. The trained
magnitude sits between them, so the 100% row should land near 78%. If it comes back wildly
different, the account above is wrong and this run should be discarded rather than reported.

SAMPLE
The six magnitudes were also acting as a repetition axis. To keep 216 per ratio on one
condition we raise repeats from 3 to 18:
    12 pairs x 18 descriptions = 216 per ratio
Ratios 1.0 (pure anchor, control) through 0.0 (pure concerning concept) give a curve whose
endpoints validate the measurement. The rank-16 adapter is run on the mixture ratios and
the control for the capacity comparison.

Assumes 00_reload.py globals.
"""
import pickle, os, json, hashlib

OUT = "/workspace/trained_magnitude.pkl"
SCALE = 1.0
ALPHAS = [0.0, 0.25, 0.5, 0.75, 0.9, 1.0]
LR_ALPHAS = [0.5, 0.75, 0.9, 1.0]
SHARE = {0.0: "100%", 0.25: "75%", 0.5: "50%", 0.75: "25%", 0.9: "10%", 1.0: "0% (control)"}
N_DESC = 18
BATCH = 9                      # generate in two batches of 9 to bound memory


def _seed(*p):
    return int(hashlib.md5("|".join(map(str, p)).encode()).hexdigest()[:6], 16)


def cell(ia, ib, al, adpt, tag):
    """N_DESC descriptions of one (pair, ratio) at the trained magnitude."""
    v = compose(ia, ib, al)
    texts = []
    for b in range(0, N_DESC, BATCH):
        n = min(BATCH, N_DESC - b)
        V = v.unsqueeze(0).repeat(n, 1)
        texts += generate_descriptions(V, SCALE, trained=True, adpt=adpt,
                                       seed=_seed(tag, ia, ib, al, b))
    rows = []
    for t in texts:
        s = score_label(t, [ia, ib], n=10)
        rows.append({"label": t, "hit_A": s[ia], "hit_B": s[ib]})
    return rows


def run(pairs, res):
    for tag, adpt, alphas in [("sa", adapter, ALPHAS), ("lr", adapter_lr, LR_ALPHAS)]:
        for al in alphas:
            for nm, ia, ib in pairs:
                key = (tag, nm, al)
                if key in res:
                    continue
                res[key] = cell(ia, ib, al, adpt, tag)
                pickle.dump(res, open(OUT, "wb"))
            done = [r for k, v in res.items() if k[0] == tag and k[2] == al for r in v]
            named = sum(1 for r in done if r["hit_B"] >= THR)
            print(f"  [{tag}] {SHARE[al]:>13}  B named {named}/{len(done)}")
    return res


def report(res, pairs):
    from collections import Counter
    names = [p[0] for p in pairs]

    def rows(tag, al):
        return [r for nm in names for r in res.get((tag, nm, al), [])]

    print("\n" + "=" * 72)
    print(f"TRAINED MAGNITUDE (scale=1.0)   12 pairs x {N_DESC} descriptions = "
          f"{12*N_DESC} per ratio")
    print("=" * 72)
    print(f"{'share of concept B':>22}{'B named':>12}{'rate':>9}{'both':>8}{'neither':>9}")
    out = {}
    for al in ALPHAS:
        rs = rows("sa", al)
        if not rs:
            continue
        c = Counter()
        for r in rs:
            a, b = r["hit_A"] >= THR, r["hit_B"] >= THR
            c["both" if (a and b) else "A" if a else "B" if b else "neither"] += 1
        named = c["B"] + c["both"]
        print(f"{SHARE[al]:>22}{f'{named}/{len(rs)}':>12}{named/len(rs)*100:>8.1f}%"
              f"{c['both']:>8}{c['neither']:>9}")
        out[SHARE[al]] = {"named": named, "n": len(rs), "both": c["both"],
                          "neither": c["neither"]}

    print(f"\n{'share':>22}{'scalar affine':>16}{'rank-16':>12}")
    for al in LR_ALPHAS:
        sa, lr = rows("sa", al), rows("lr", al)
        if not (sa and lr):
            continue
        ns = sum(1 for r in sa if r["hit_B"] >= THR)
        nl = sum(1 for r in lr if r["hit_B"] >= THR)
        print(f"{SHARE[al]:>22}{f'{ns}/{len(sa)}':>16}{f'{nl}/{len(lr)}':>12}")

    pure = out.get("100%")
    if pure:
        print(f"\n  prediction was ~78% on the pure row; observed "
              f"{pure['named']/pure['n']*100:.1f}%")
    json.dump(out, open("/workspace/trained_magnitude.json", "w"), indent=1)


if __name__ == "__main__":
    meta = pickle.load(open("/workspace/safe_meta.pkl", "rb"))
    V = meta["VALID2"]
    pairs = [p for p in meta["SAFE_PAIRS"]
             if all({a: b for a, _, b, _ in V[p[0]]}.get(al) for al in [0.5, 0.75, 0.9])]
    total = len(pairs) * (len(ALPHAS) + len(LR_ALPHAS)) * N_DESC
    print(f"pairs {len(pairs)} | descriptions {total}")
    res = pickle.load(open(OUT, "rb")) if os.path.exists(OUT) else {}
    res = run(pairs, res)
    report(res, pairs)
