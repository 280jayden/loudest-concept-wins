"""
Does the second concept appear if the description is allowed to be longer?

MOTIVATION
At an equal mixture the interpreter names both concepts in 12/216 generations, while
the NLA verbaliser on the other architecture names both in 20/42. The obvious
difference is length: the median Llama description is 6 words, the median NLA
explanation is 93. A six-word noun phrase cannot hold two unrelated concepts, so the
omission may be a property of the output format rather than of what reached the model.

WHAT IS AND IS NOT CHANGED
Only max_new_tokens, 30 -> 100. The prompt template is left alone, because the adapter
was trained under it and replacing it moves the injection off the distribution the
adapter was fitted to - the same problem that made the untrained baseline
uninterpretable.

Note the template ends with:   The meaning of "<RESERVED>" is "
which primes a short quoted label. The model may therefore close the quote and stop at
its usual length no matter what budget it is given. That outcome is informative rather
than a failed run: it would locate the constraint in the trained output format rather
than in the token allowance.

READING THE RESULT
  descriptions get longer AND both-rate rises  -> format constraint, not representation
  descriptions get longer, both-rate flat      -> length was not the binding constraint
  descriptions stay ~6 words                   -> the template, not the budget, sets the format

alpha = 0.5 only, on the same 12 pairs and 6 magnitudes as the main experiment, so the
216 is directly comparable to the 12/216 already measured at max_new=30.

Assumes 00_reload.py globals.
"""
import pickle, os, json, hashlib

OUT = "/workspace/token_budget.pkl"
ALPHA = 0.5
MAX_NEW = 100
N_DESC = 3


def _seed(*p):
    return int(hashlib.md5("|".join(map(str, p)).encode()).hexdigest()[:6], 16)


def cell(ia, ib, sc):
    v = compose(ia, ib, ALPHA)
    V = v.unsqueeze(0).repeat(N_DESC, 1)
    texts = generate_descriptions(V, sc, trained=True, max_new=MAX_NEW,
                                  seed=_seed("tb", ia, ib, sc))
    rows = []
    for t in texts:
        s = score_label(t, [ia, ib], n=10)
        rows.append({"label": t, "hit_A": s[ia], "hit_B": s[ib],
                     "words": len(t.split())})
    return rows


def report(res):
    from collections import Counter
    rows = [r for v in res.values() for r in v]
    n = len(rows)
    c = Counter()
    for r in rows:
        a, b = r["hit_A"] >= THR, r["hit_B"] >= THR
        c["both" if (a and b) else "A" if a else "B" if b else "neither"] += 1
    w = sorted(r["words"] for r in rows)

    print("\n" + "=" * 62)
    print(f"TOKEN BUDGET TEST  alpha=0.5  max_new={MAX_NEW}   n={n}")
    print("=" * 62)
    print(f"{'':>14}{'max_new=30':>14}{'max_new=100':>14}")
    print(f"{'median words':>14}{6:>14}{w[len(w)//2]:>14}")
    print(f"{'A only':>14}{128:>14}{c['A']:>14}")
    print(f"{'B only':>14}{34:>14}{c['B']:>14}")
    print(f"{'both':>14}{12:>14}{c['both']:>14}")
    print(f"{'neither':>14}{42:>14}{c['neither']:>14}")
    print(f"{'B total':>14}{46:>14}{c['B']+c['both']:>14}")
    print(f"\n  longest descriptions produced:")
    for r in sorted(rows, key=lambda x: -x["words"])[:4]:
        print(f"    {r['words']:>3}w  {r['label'][:96]!r}")
    json.dump({"n": n, "median_words": w[len(w) // 2], "counts": dict(c),
               "baseline_max_new_30": {"A": 128, "B": 34, "both": 12, "neither": 42,
                                       "median_words": 6}},
              open("/workspace/token_budget.json", "w"), indent=1)


if __name__ == "__main__":
    meta = pickle.load(open("/workspace/safe_meta.pkl", "rb"))
    V = meta["VALID2"]
    pairs = [p for p in meta["SAFE_PAIRS"]
             if all({a: b for a, _, b, _ in V[p[0]]}.get(al) for al in [0.5, 0.75, 0.9])]
    print(f"pairs {len(pairs)} | descriptions {len(pairs)*len(SCALES)*N_DESC}")
    res = pickle.load(open(OUT, "rb")) if os.path.exists(OUT) else {}
    for nm, ia, ib in pairs:
        for sc in SCALES:
            if (nm, sc) in res:
                continue
            res[(nm, sc)] = cell(ia, ib, sc)
            pickle.dump(res, open(OUT, "wb"))
        print(f"  {nm[:44]:44} done")
    report(res)
