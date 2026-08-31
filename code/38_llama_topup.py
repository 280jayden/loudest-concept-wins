"""
Top the Llama arm up from 18 to 20 descriptions per cell, so both arms report 240.

The 18 was inherited: it came from 216/12, which itself came from the old design of
6 magnitudes x 3 descriptions. Nothing selected it. Raising both arms to 20 gives
12 pairs x 20 = 240 per ratio on each, and makes the sample size a stated choice
rather than an artefact of an earlier layout.

Existing rows are kept and appended to. Seeds continue from index 18 so the new
descriptions are fresh samples rather than repeats of the first eighteen.
"""
import pickle, os, hashlib, json
from collections import Counter

OUT = "/workspace/trained_magnitude.pkl"
SCALE = 1.0
ALPHAS = [0.0, 0.25, 0.5, 0.75, 0.9, 1.0]
SHARE = {0.0: "100%", 0.25: "75%", 0.5: "50%", 0.75: "25%", 0.9: "10%", 1.0: "0% (control)"}
N_DESC = 20


def _seed(*p):
    return int(hashlib.md5("|".join(map(str, p)).encode()).hexdigest()[:6], 16)


def top_up(ia, ib, al, adpt, tag, have):
    """Generate the missing descriptions for one cell."""
    need = N_DESC - have
    v = compose(ia, ib, al)
    V = v.unsqueeze(0).repeat(need, 1)
    texts = generate_descriptions(V, SCALE, trained=True, adpt=adpt,
                                  seed=_seed(tag, ia, ib, al, "topup"))
    rows = []
    for t in texts:
        s = score_label(t, [ia, ib], n=10)
        rows.append({"label": t, "hit_A": s[ia], "hit_B": s[ib]})
    return rows


if __name__ == "__main__":
    meta = pickle.load(open("/workspace/safe_meta.pkl", "rb"))
    V = meta["VALID2"]
    pairs = [p for p in meta["SAFE_PAIRS"]
             if all({a: b for a, _, b, _ in V[p[0]]}.get(al) for al in [0.5, 0.75, 0.9])]
    res = pickle.load(open(OUT, "rb"))

    todo = sum(max(0, N_DESC - len(res.get((t, nm, al), [])))
               for t in ("sa", "lr") for nm, _, _ in pairs for al in ALPHAS)
    print(f"pairs {len(pairs)} | descriptions to add {todo}")

    for tag, adpt in [("sa", adapter), ("lr", adapter_lr)]:
        for al in ALPHAS:
            for nm, ia, ib in pairs:
                key = (tag, nm, al)
                have = len(res.get(key, []))
                if have >= N_DESC:
                    continue
                res[key] = res.get(key, []) + top_up(ia, ib, al, adpt, tag, have)
                pickle.dump(res, open(OUT, "wb"))
            n = sum(len(res.get((tag, nm, al), [])) for nm, _, _ in pairs)
            print(f"  [{tag}] {SHARE[al]:>13}  n={n}")

    print("\n" + "=" * 74)
    print(f"TRAINED MAGNITUDE   12 pairs x {N_DESC} descriptions = {12*N_DESC} per ratio")
    print("=" * 74)
    print(f"{'B share':>10}{'A named':>16}{'B named':>16}{'both':>7}{'neither':>9}")
    out = {}
    for tag in ("sa", "lr"):
        print(f"  -- {'scalar affine' if tag=='sa' else 'rank-16'} --")
        for al in ALPHAS:
            rows = [r for nm, _, _ in pairs for r in res.get((tag, nm, al), [])]
            if not rows:
                continue
            c = Counter()
            for r in rows:
                a, b = r["hit_A"] >= THR, r["hit_B"] >= THR
                c["both" if (a and b) else "A" if a else "B" if b else "neither"] += 1
            A, B, n = c["A"] + c["both"], c["B"] + c["both"], len(rows)
            print(f"{SHARE[al]:>10}{A:>10}/{n:<4}{A/n*100:>4.1f}%"
                  f"{B:>10}/{n:<4}{B/n*100:>4.1f}%{c['both']:>7}{c['neither']:>9}")
            out[f"{tag}_{int((1-al)*100)}%"] = {"A": A, "B": B, "n": n,
                                                "both": c["both"], "neither": c["neither"]}
    json.dump(out, open("/workspace/trained_magnitude_240.json", "w"), indent=1)
    print("\n  wrote trained_magnitude_240.json")
