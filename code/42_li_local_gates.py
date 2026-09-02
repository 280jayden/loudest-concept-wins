"""
Gates 2 and 3 for the Llama Scope arm. CPU only, no pod.

Inputs
  --sae        path to Llama3_1-8B-Base-L19R-32x/checkpoints/final.safetensors
  --cands      np_candidates_L19_131k.json  (Neuronpedia search output: family -> [features])

What it does
  1. Builds the anchor set (cooking / baking / spices families) and the concerning set
     (everything else), taking the top candidates per family by search similarity.
  2. Gate 2: |cos(d_A, d_B)| < 0.1 on unit-normalised decoder columns.
  3. Gate 3: the minority latent fires under the Llama Scope JumpReLU encoder in the
     composed vector at every share, evaluated at the magnitude that will actually be
     injected. Li et al. trained on raw decoder columns (norms 0.7 to 3.4, mean 1.52),
     so the injected vector is  unit_mix * mean(||d_A||, ||d_B||)  and the encoder sees
     it after the SAE's own dataset-wise rescale (x * 64 / 17.125).
  4. Writes results/RESULTS/li_pairs_candidates.json with every pair that passes both,
     plus the per-share encoder activations so the choice is auditable.

Gate 1 (describable alone under each method) needs the models and runs on the pod.
"""
import json, argparse, itertools, math
import torch
from safetensors.torch import load_file

ap = argparse.ArgumentParser()
ap.add_argument("--sae", required=True)
ap.add_argument("--cands", required=True)
ap.add_argument("--out", default="../results/RESULTS/li_pairs_candidates.json")
ap.add_argument("--per_family", type=int, default=4)
a = ap.parse_args()

SHARES = [0.25, 0.5, 0.75, 0.9]          # alpha = A's share; B holds 75/50/25/10 %
COS_MAX, THR, K = 0.1, 0.484375, 64 / 17.125
ANCHOR_FAMILIES = ["cooking", "baking", "spices"]

t = load_file(a.sae)
Wd = t["decoder.weight"].float()          # (4096, 131072): column i is feature i
We = t["encoder.weight"].float()
be = t["encoder.bias"].float()
norms = Wd.norm(dim=0)
print(f"decoder column norms: mean {norms.mean():.3f} min {norms.min():.3f} max {norms.max():.3f}")


def enc(x):
    pre = We @ x + be
    return pre * (pre > THR)


cands = json.load(open(a.cands))
feats = {}
for fam, rows in cands.items():
    for r in rows[: a.per_family]:
        feats[int(r["index"])] = {"family": fam, "desc": r["desc"], "sim": r["sim"],
                                  "maxAct": r["maxAct"], "density": r["density"],
                                  "norm": float(norms[int(r["index"])])}
anchors = [i for i, f in feats.items() if f["family"] in ANCHOR_FAMILIES]
concern = [i for i, f in feats.items() if f["family"] not in ANCHOR_FAMILIES]
print(f"{len(anchors)} anchor candidates, {len(concern)} concerning candidates")

D = {i: Wd[:, i] / norms[i] for i in feats}
out = {"features": {str(i): f for i, f in feats.items()}, "pairs": []}
n_g2 = n_g3 = 0
for ia, ib in itertools.product(anchors, concern):
    cos = float(D[ia] @ D[ib])
    if abs(cos) >= COS_MAX:
        continue
    n_g2 += 1
    m = 0.5 * (float(norms[ia]) + float(norms[ib]))     # injected magnitude for this pair
    acts, ok = {}, True
    for al in SHARES:
        v = al * D[ia] + (1 - al) * D[ib]
        v = v / v.norm()
        act = enc(v * m * K)
        gA, gB = float(act[ia]), float(act[ib])
        acts[f"{round((1-al)*100)}%"] = [round(gA, 3), round(gB, 3)]
        if gA <= 0 or gB <= 0:
            ok = False
    ok_no10 = all(acts[k][0] > 0 and acts[k][1] > 0 for k in ("75%", "50%", "25%"))
    if ok_no10 and not ok:
        out.setdefault("pairs_pass_without_10pct", []).append({"A": ia, "B": ib, "B_family": feats[ib]["family"], "cos": round(cos, 4), "acts_A_B": acts})
    if not ok:
        continue
    n_g3 += 1
    out["pairs"].append({"A": ia, "B": ib, "A_family": feats[ia]["family"], "B_family": feats[ib]["family"],
                         "A_desc": feats[ia]["desc"], "B_desc": feats[ib]["desc"],
                         "cos": round(cos, 4), "inject_mag": round(m, 3), "acts_A_B": acts})
print(f"pairs passing gate 2: {n_g2}   passing gates 2+3 (minority fires at 75/50/25/10%): {n_g3}")
fam_count = {}
for p in out["pairs"]:
    fam_count[p["B_family"]] = fam_count.get(p["B_family"], 0) + 1
print("passing pairs per concerning family:", fam_count)
alt = out.get("pairs_pass_without_10pct", [])
print(f"pairs passing at 75/50/25 but NOT 10%: {len(alt)}  by family:", dict(__import__('collections').Counter(p["B_family"] for p in alt)))
print(f"\n{'A':>7} {'B':>7} {'cos':>7} {'mag':>5}  {'act_B@10%':>10}  A desc | B desc")
for p in sorted(out["pairs"], key=lambda p: -p["acts_A_B"]["10%"][1])[:30]:
    print(f"{p['A']:>7} {p['B']:>7} {p['cos']:>+7.3f} {p['inject_mag']:>5.2f}  {p['acts_A_B']['10%'][1]:>10.2f}  "
          f"{p['A_desc'][:32]} | {p['B_desc'][:40]}")
json.dump(out, open(a.out, "w"), indent=1)
print("\nwrote", a.out)
