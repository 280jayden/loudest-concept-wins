"""
Put the main result, its false-positive control, and the adapter comparison on ONE
identical sample.

THE PROBLEM THIS FIXES
The project grew: the first sweep used 5 concept pairs, the safety sweep later used 12,
and the adapter comparison was run against the earlier 5-pair set. Every experiment is
internally sound, but three numbers that belong in one argument sat on three different
samples, which reads as six studies instead of one.

AFTER THIS RUN
    main result        12 pairs x 6 magnitudes x 3 descriptions = 216 per ratio  [have]
    control (pure A)   same 12 pairs, same 216                                   [new]
    low-rank adapter   same 12 pairs, same 216 per ratio                         [new]
    NLA                42 per ratio - different model, expected to differ

WHAT RUNS HERE
  alpha 1.0 is the pure-anchor control: concept B is genuinely absent, so any time the
  metric reports B is a false positive. It uses the same compose() path as every other
  cell, so the control is not a differently-built object - it is the same construction
  with the mixing coefficient set to its endpoint.

  The scalar-affine arm already exists for alpha 0.5/0.75/0.9 in safety_sweep.pkl, so
  only its control is generated here. The low-rank arm is generated in full.

Assumes 00_reload.py globals: compose, generate_descriptions, score_label,
adapter, adapter_lr, SCALES, THR.
"""
import pickle, os, json, hashlib

OUT = "/workspace/matched_sample.pkl"
ALPHAS = [0.5, 0.75, 0.9, 1.0]        # 1.0 = pure anchor = the control
SHARE = {0.5: "50%", 0.75: "25%", 0.9: "10%", 1.0: "absent (control)"}
N_DESC = 3


def _seed(*p):
    return int(hashlib.md5("|".join(map(str, p)).encode()).hexdigest()[:6], 16)


def cell(ia, ib, al, sc, adpt, tag):
    """Three descriptions of one (pair, ratio, magnitude), scored for both concepts."""
    v = compose(ia, ib, al)
    V = v.unsqueeze(0).repeat(N_DESC, 1)
    texts = generate_descriptions(V, sc, trained=True, adpt=adpt,
                                  seed=_seed(tag, ia, ib, al, sc))
    rows = []
    for t in texts:
        s = score_label(t, [ia, ib], n=10)
        rows.append({"label": t, "hit_A": s[ia], "hit_B": s[ib]})
    return rows


def run(pairs, res):
    plan = [("lr", adapter_lr, ALPHAS),      # full low-rank arm, incl. its control
            ("sa", adapter, [1.0])]          # scalar affine: only the control is missing
    for tag, adpt, alphas in plan:
        for nm, ia, ib in pairs:
            for al in alphas:
                for sc in SCALES:
                    key = (tag, nm, al, sc)
                    if key in res:
                        continue
                    res[key] = cell(ia, ib, al, sc, adpt, tag)
                    pickle.dump(res, open(OUT, "wb"))
            done = sum(1 for k in res if k[0] == tag and k[1] == nm)
            print(f"  [{tag}] {nm[:40]:40} {done} cells")
    return res


def report(res, pairs):
    ss = pickle.load(open("/workspace/safety_sweep.pkl", "rb"))
    names = {p[0] for p in pairs}

    def count(store, tag, al, from_sweep=False):
        hit = n = 0
        for nm in names:
            for sc in SCALES:
                rows = (store.get((nm, al, sc), []) if from_sweep
                        else store.get((tag, nm, al, sc), []))
                for r in rows:
                    n += 1
                    hit += (r["hit_B"] >= THR)
        return hit, n

    print("\n" + "=" * 66)
    print("ONE SAMPLE: 12 pairs x 6 magnitudes x 3 descriptions = 216 per ratio")
    print("=" * 66)
    print(f"{'share of concerning concept':>30}{'scalar affine':>18}{'+ rank-16':>16}")
    print(f"{'':>30}{'4,097 params':>18}{'135,169 params':>16}")
    for al in [0.5, 0.75, 0.9]:
        sa = count(ss, "sa", al, from_sweep=True)
        lr = count(res, "lr", al)
        print(f"{SHARE[al]:>30}{f'{sa[0]}/{sa[1]}':>18}{f'{lr[0]}/{lr[1]}':>16}")
    sa_c = count(res, "sa", 1.0)
    lr_c = count(res, "lr", 1.0)
    print(f"{'concept absent (control)':>30}{f'{sa_c[0]}/{sa_c[1]}':>18}{f'{lr_c[0]}/{lr_c[1]}':>16}")

    print("\n  anchor described (injection sanity, should be high):")
    for tag, store, frm in [("sa", ss, True), ("lr", res, False)]:
        h = n = 0
        for nm in names:
            for al in ([0.5, 0.75, 0.9] if frm else ALPHAS):
                for sc in SCALES:
                    rows = (store.get((nm, al, sc), []) if frm
                            else store.get((tag, nm, al, sc), []))
                    for r in rows:
                        n += 1
                        h += (r["hit_A"] >= THR)
        print(f"     {tag}: {h}/{n} = {h/max(n,1)*100:.0f}%")

    out = {"basis": "12 pairs x 6 magnitudes x 3 descriptions = 216 per ratio",
           "scalar_affine": {SHARE[a]: list(count(ss, "sa", a, True)) for a in [0.5, 0.75, 0.9]},
           "low_rank": {SHARE[a]: list(count(res, "lr", a)) for a in [0.5, 0.75, 0.9]},
           "control_scalar_affine": list(sa_c), "control_low_rank": list(lr_c)}
    json.dump(out, open("/workspace/matched_sample.json", "w"), indent=1)
    print("\n  wrote matched_sample.json")


if __name__ == "__main__":
    meta = pickle.load(open("/workspace/safe_meta.pkl", "rb"))
    V = meta["VALID2"]
    pairs = [p for p in meta["SAFE_PAIRS"]
             if all({a: b for a, _, b, _ in V[p[0]]}.get(al) for al in [0.5, 0.75, 0.9])]
    total = len(pairs) * (len(ALPHAS) + 1) * len(SCALES) * N_DESC
    print(f"pairs {len(pairs)} | new descriptions to generate ~{total}")
    res = pickle.load(open(OUT, "rb")) if os.path.exists(OUT) else {}
    res = run(pairs, res)
    report(res, pairs)
