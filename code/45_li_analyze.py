"""
Analysis of the Llama Scope arm (Li explainer, Pepper scalar-affine, Pepper rank-64).
Reads the pickles pulled from the pod into results/RESULTS/li/ and prints every table
the write-up needs; also writes results/RESULTS/li/analysis.json.

usage:  python code/45_li_analyze.py [results/RESULTS/li]
"""
import os, sys, json, pickle
from collections import Counter, defaultdict

D = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "..", "results", "RESULTS", "li")
THR = 0.3
ALPHAS = [0.0, 0.25, 0.5, 0.75, 0.9, 1.0]
SHARE = {0.0: "100%", 0.25: "75%", 0.5: "50%", 0.75: "25%", 0.9: "10%", 1.0: "0% (control)"}
ARMS = [("li", "Li explainer"), ("adapter", "Pepper scalar-affine"), ("adapter64", "Pepper rank-64")]


def load(name):
    p = os.path.join(D, name + ".pkl")
    return pickle.load(open(p, "rb")) if os.path.exists(p) else {}


def cell(rows):
    c = Counter()
    for r in rows:
        a_, b_ = r["hit_A"] >= THR, r["hit_B"] >= THR
        c["both" if (a_ and b_) else "A" if a_ else "B" if b_ else "neither"] += 1
    n = len(rows)
    return {"A": c["A"] + c["both"], "B": c["B"] + c["both"], "both": c["both"], "neither": c["neither"], "n": n}


def pct(k, n):
    return f"{100*k/n:5.1f}%" if n else "   - "


out = {}
pairs = json.load(open(os.path.join(D, "pairs_final.json")))["pairs"] if os.path.exists(os.path.join(D, "pairs_final.json")) else []
cand = json.load(open(os.path.join(D, "li_pairs_candidates.json"))) if os.path.exists(os.path.join(D, "li_pairs_candidates.json")) else {}
feats = {int(k): v for k, v in cand.get("features", {}).items()}
print(f"== pairs ({len(pairs)}) ==")
for p in pairs:
    print(f"  A {p['A']:>7} {p['A_desc'][:34]:34} | B {p['B']:>7} {p['B_desc'][:40]:40} | cos {p['cos']:+.3f} mag {p['inject_mag']:.2f} actB@10% {p['acts_A_B']['10%'][1]:.2f}")
out["pairs"] = pairs

# ---- gate 1
G1 = load("gate1")
if G1:
    print(f"\n== gate 1 ({len(G1)} concepts): describable alone, best of 3, hit >= 0.8 ==")
    n_li = sum(v["li_pass"] for v in G1.values()); n_ad = sum(v["ad_pass"] for v in G1.values())
    n_both = sum(v["li_pass"] and v["ad_pass"] for v in G1.values())
    print(f"  Li pass {n_li}/{len(G1)}   adapter pass {n_ad}/{len(G1)}   both {n_both}/{len(G1)}")
    out["gate1"] = {"n": len(G1), "li_pass": n_li, "adapter_pass": n_ad, "both": n_both}
G64 = load("gate1_64")
if G64:
    print(f"  rank-64 pass {sum(v['ad_pass'] for v in G64.values())}/{len(G64)} (recorded on the final concepts, pairs not reselected)")
    out["gate1_64"] = {"n": len(G64), "pass": sum(v["ad_pass"] for v in G64.values())}

# ---- pre-flight
PF = load("preflight")
if PF:
    print(f"\n== pre-flight ==  residual norm {PF.get('resid_norm', 0):.2f}  L0 {PF.get('sae_l0', 0):.1f}")
    for i, v in PF.get("known", {}).items():
        print(f"  {i} '{v['label'][:40]}'  Li {v['li'][0]!r} ({v['li_score']:.1f})  adapter {v['adapter'][0]!r} ({v['ad_score']:.1f})")
    for i, v in PF.get("mag_sweep", {}).items():
        print(f"  mag sweep {i}: " + "  ".join(f"{f}x:{s:.1f}" for f, (_, s) in v.items()))

# ---- main sweep tables
res = {tag: load(tag) for tag, _ in ARMS}
summary = {}
for tag, name in ARMS:
    R = res[tag]
    if not R:
        continue
    print(f"\n== {name}: sampled descriptions, 20 per pair per share ==")
    print(f"  {'share':>13}  {'A named':>9} {'B named':>9} {'both':>9} {'neither':>9}   n")
    summary[tag] = {}
    for al in ALPHAS:
        rows = [r for k, v in R.items() if isinstance(k, tuple) and k[0] == tag and k[2] == al for r in v]
        c = cell(rows); summary[tag][SHARE[al]] = c
        print(f"  {SHARE[al]:>13}  {pct(c['A'], c['n'])}   {pct(c['B'], c['n'])}   {pct(c['both'], c['n'])}   {pct(c['neither'], c['n'])}   {c['n']}")
    # greedy
    gk = tag + "_greedy"
    if any(isinstance(k, tuple) and k[0] == gk for k in R):
        print(f"  greedy (1 per cell):")
        for al in ALPHAS:
            rows = [r for k, v in R.items() if isinstance(k, tuple) and k[0] == gk and k[2] == al for r in v]
            if rows:
                c = cell(rows); summary[tag][SHARE[al] + "_greedy"] = c
                print(f"  {SHARE[al]:>13}  A {c['A']:>2}/{c['n']}  B {c['B']:>2}/{c['n']}  both {c['both']:>2}/{c['n']}")
    # list prompt
    lk = tag + "_list"
    if any(isinstance(k, tuple) and k[0] == lk for k in R):
        print(f"  list prompt (8 draws per pair, a concept counts if any item hits it):")
        for al in [0.5, 0.75]:
            rows = [r for k, v in R.items() if isinstance(k, tuple) and k[0] == lk and k[2] == al for r in v]
            if rows:
                c = cell(rows); summary[tag][SHARE[al] + "_list"] = c
                print(f"  {SHARE[al]:>13}  A {pct(c['A'], c['n'])}  B {pct(c['B'], c['n'])}  both {pct(c['both'], c['n'])}  n {c['n']}")
out["summary"] = summary

# ---- per pair at 50% and 25%
for tag, name in ARMS:
    R = res[tag]
    if not R:
        continue
    print(f"\n== {name}: per pair ==   (A named / B named / both, out of 20)")
    print(f"  {'pair':44} {'75%':>10} {'50%':>10} {'25%':>10} {'10%':>10}")
    per = {}
    for nm, ia, ib in [(p["name"], p["A"], p["B"]) for p in pairs]:
        row = []
        for al in [0.25, 0.5, 0.75, 0.9]:
            rows = R.get((tag, nm, al), [])
            c = cell(rows); row.append(f"{c['A']:>2}/{c['B']:>2}/{c['both']:>2}")
            per.setdefault(nm, {})[SHARE[al]] = c
        print(f"  {nm[:44]:44} " + " ".join(f"{x:>10}" for x in row))
    out.setdefault("per_pair", {})[tag] = per

# ---- H6: parity winner vs describability threshold
TH = load("thresh")
if TH and res["li"]:
    print("\n== H6: at parity, does the concept with the lower describability threshold win? ==")
    for tag, name in ARMS:
        R = res[tag]
        if not R:
            continue
        key = "li" if tag == "li" else "adapter"
        fs = [0.5, 0.7, 1.0] if tag == "li" else [0.5, 0.8]
        def thr_score(i):          # sum of best hit over sub-trained magnitudes: higher = easier to describe
            return sum(max(s for _, s in TH[i][key][f]) for f in fs if f in TH[i][key])
        rows_x, rows_y = [], []
        agree = tot = 0
        for p in pairs:
            nm = p["name"]
            rows = R.get((tag, nm, 0.5), [])
            if not rows or p["A"] not in TH or p["B"] not in TH:
                continue
            c = cell(rows)
            if c["A"] == c["B"]:
                continue
            winner_easier = (c["A"] > c["B"]) == (thr_score(p["A"]) > thr_score(p["B"]))
            if thr_score(p["A"]) == thr_score(p["B"]):
                continue
            agree += winner_easier; tot += 1
        print(f"  {name}: parity winner is the more-describable concept in {agree}/{tot} decided pairs")
        out.setdefault("h6", {})[tag] = {"agree": agree, "decided": tot}

# ---- floors
FL = load("floors")
if FL:
    print("\n== random-direction floor (20 directions x 12 latents) ==")
    for tag, name in ARMS:
        ks = [k for k in FL if k[0] == tag]
        if not ks:
            continue
        fp = sum(1 for k in ks for h in FL[k]["hits"].values() if h >= THR)
        n = sum(len(FL[k]["hits"]) for k in ks)
        print(f"  {name}: {fp}/{n} false positives ({100*fp/n:.1f}%)")
        out.setdefault("floors", {})[tag] = {"fp": fp, "n": n}


# ---- cross-arm comparison, direction consistency (H7), related-pair exclusion
def unordered_pairs():
    seen, out_ = set(), []
    for p in pairs:
        k = tuple(sorted((p["A"], p["B"])))
        if k not in seen:
            seen.add(k); out_.append(k)
    return out_

byid = {(p["A"], p["B"]): p for p in pairs}
print("\n== cross-arm: B named (%) by share ==")
print(f"  {'arm':22} " + " ".join(f"{SHARE[a]:>13}" for a in ALPHAS) + "   both@50%")
for tag, name in ARMS:
    if tag not in summary or not summary[tag].get("50%", {}).get("n"):
        continue
    row = [pct(summary[tag][SHARE[a]]["B"], summary[tag][SHARE[a]]["n"]) for a in ALPHAS]
    print(f"  {name:22} " + " ".join(f"{x:>13}" for x in row) + f"   {pct(summary[tag]['50%']['both'], summary[tag]['50%']['n'])}")

print("\n== parity winner consistent across the two directions? (H7) ==")
for tag, name in ARMS:
    R = res[tag]
    if not R:
        continue
    cons = tot = 0; detail = []
    for i, j in unordered_pairs():
        p1, p2 = byid.get((i, j)), byid.get((j, i))
        if not p1 or not p2:
            continue
        c1, c2 = cell(R.get((tag, p1["name"], 0.5), [])), cell(R.get((tag, p2["name"], 0.5), []))
        if not c1["n"] or not c2["n"]:
            continue
        w1 = i if c1["A"] > c1["B"] else j if c1["B"] > c1["A"] else None      # winner in direction 1 (A=i)
        w2 = j if c2["A"] > c2["B"] else i if c2["B"] > c2["A"] else None      # winner in direction 2 (A=j)
        if w1 is None or w2 is None:
            detail.append(f"{i}/{j}: tie"); continue
        tot += 1; cons += (w1 == w2)
        detail.append(f"{i}/{j}: {'same winner ' + str(w1) if w1 == w2 else 'DIFFERENT'}")
    print(f"  {name}: same concept wins in both directions for {cons}/{tot} unordered pairs   [" + "; ".join(detail) + "]")
    out.setdefault("h7", {})[tag] = {"consistent": cons, "decided": tot}

print("\n== 25% share, B named, excluding pairs whose concepts co-occur naturally (self-harm/firearms) or show one-sided dominance ==")
for tag, name in ARMS:
    R = res[tag]
    if not R:
        continue
    rows_all = [(k[1], r) for k, v in R.items() if isinstance(k, tuple) and k[0] == tag and k[2] == 0.75 for r in v]
    if not rows_all:
        continue
    per_pair = Counter(nm for nm, r in rows_all if r["hit_B"] >= THR)
    top = per_pair.most_common(2)
    excl = {nm for nm, _ in top}
    kept = [(nm, r) for nm, r in rows_all if nm not in excl]
    print(f"  {name}: top pairs {[(nm[:30], n) for nm, n in top]}; excluding them: "
          f"{sum(r['hit_B'] >= THR for _, r in kept)}/{len(kept)} B named")

json.dump(out, open(os.path.join(D, "analysis.json"), "w"), indent=1)
print("\nwrote", os.path.join(D, "analysis.json"))
