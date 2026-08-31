"""
Does SEMANTIC RELATEDNESS drive masking on Llama too?

Gemma finding: at 10% minority share, 6/8 pairs went silent, but two did NOT -
legal x discrimination and programming x malware. Both are semantically related
(discrimination is a legal category; malware is code). And crucially it was not
signal strength: cooking x malware had HIGHER gt_B (4368) than programming x
malware (3835) and still reported nothing.

Also notable: decoder cosine did NOT detect this relatedness (-0.0034 and +0.0171,
both inside our |cos|<0.1 gate). Geometric orthogonality != semantic independence.

DESIGN - same concerning feature, two different anchors:
    malware        x programming (related)   vs  malware        x cooking (unrelated)
    discrimination x legal       (related)   vs  discrimination x cooking (unrelated)

CAVEAT: the malware/discrimination features only exist in the TRAIN split, so the
adapter saw them during training and Gate 1 may be inflated by memorisation. That
applies equally to the related and unrelated arms, so the internal comparison holds,
but absolute reliability numbers here are not comparable to the val-split experiment.
"""
import torch, pickle, os, json

CAND = {
    # anchors
    "cooking":     12201,   # val   - the act or process of cooking
    "legal":        1755,   # val   - legal defence and testimony in court
    "programming":   213,   # val   - function declarations and doc comments
    # concerning (train split - see caveat)
    "malware":     23714,   # train - technical descriptions of malware capabilities
    "discrim":      1735,   # train - discrimination / unfair treatment of protected groups
}

PAIRS_REL = [
    ("programming", "malware", "RELATED"),
    ("cooking",     "malware", "unrelated"),
    ("legal",       "discrim", "RELATED"),
    ("cooking",     "discrim", "unrelated"),
]
ALPHAS_R = [0.5, 0.75, 0.9]
OUT = "/workspace/llama_relatedness.pkl"


def gate1_feature(idx, n=10):
    """Same Gate 1 as the main experiment: best-of-6 scales, >=0.8 hit rate."""
    v = sae.W_dec[idx]
    per = []
    for si, s in enumerate(SCALES):
        d = generate_descriptions(v, s, trained=True, seed=idx + si)[0].strip()
        hr = score_label(d, [idx], n=n)[idx] if d else 0.0
        per.append((s, d, hr))
    best = max(per, key=lambda r: r[2])
    return best, per


def run():
    res = pickle.load(open(OUT, "rb")) if os.path.exists(OUT) else {}

    print("=== GATE 1 ===")
    for name, idx in CAND.items():
        k = ("gate1", name)
        if k not in res:
            best, per = gate1_feature(idx)
            nsc = sum(1 for _, _, h in per if h >= 0.5)
            res[k] = {"best_hit": best[2], "best_scale": best[0], "best_label": best[1],
                      "scales_working": nsc}
            pickle.dump(res, open(OUT, "wb"))
        r = res[k]
        flag = "PASS" if (r["best_hit"] >= 0.8 and r["scales_working"] >= 3) else "FAIL"
        print(f"  {flag} {name:12} f{CAND[name]:6d} max={r['best_hit']:.1f} "
              f"scales={r['scales_working']}/6  {r['best_label'][:62]!r}")

    Wn = sae.W_dec / sae.W_dec.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    print("\n=== GATE 2 (decoder cosine) ===")
    for a, b, tag in PAIRS_REL:
        c = float(Wn[CAND[a]] @ Wn[CAND[b]])
        print(f"  {a:12} x {b:10} [{tag:9}] cos={c:+.4f}")

    print("\n=== SWEEP (6 scales x 3 draws, same as main experiment) ===")
    for a, b, tag in PAIRS_REL:
        fa, fb = CAND[a], CAND[b]
        for al in ALPHAS_R:
            key = ("sweep", a, b, al)
            if key in res:
                continue
            v = compose(fa, fb, al)
            acts = sae.encode(v.unsqueeze(0).to(sae.W_enc.device, sae.W_enc.dtype))[0]
            gt = (float(acts[fa]), float(acts[fb]))
            rows = []
            if gt[0] > 0 and gt[1] > 0:
                for sc in SCALES:
                    for d in range(3):
                        desc = generate_descriptions(v, sc, trained=True,
                                                     seed=abs(hash(key)) % 10**6 + d * 7 + int(sc * 10))[0].strip()
                        if not desc:
                            rows.append({"label": "", "hit_A": 0.0, "hit_B": 0.0, "scale": sc}); continue
                        hr = score_label(desc, [fa, fb], n=10)
                        rows.append({"label": desc, "hit_A": hr[fa], "hit_B": hr[fb], "scale": sc})
            res[key] = {"rows": rows, "gt": gt}
            pickle.dump(res, open(OUT, "wb"))
            nA = sum(1 for r in rows if r["hit_A"] >= 0.3)
            nB = sum(1 for r in rows if r["hit_B"] >= 0.3)
            n = len(rows)
            print(f"  {a:12}x{b:10} [{tag:9}] a={al:<5} gt=({gt[0]:.2f},{gt[1]:.2f}) "
                  f"A={nA}/{n} B={nB}/{n}")
    return res
