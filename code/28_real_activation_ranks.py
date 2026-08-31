"""
The same decodability question, on REAL activations - the non-circular version.

WHY THIS EXISTS
27_decodability_ceiling.py shows that in a CONSTRUCTED mixture the omitted concept is
rank 2 of 65,536. The fair objection is that the vector was built out of those two
decoder rows, so of course the SAE recovers them.

These records come from forward passes over real text (news passages and constructed
prose), where the SAE decides on its own which latents are active and how strongly.
For every such activation we have:
   top       - the active latents, with activation values, in rank order
   descs     - 12 self-interpretation descriptions of that same activation, each scored
               for whether it caused the latent to fire

So we can ask directly: as a function of a latent's rank and its strength relative to
the top latent, how often does the model's description of that activation mention it?
No mixture is imposed; the ratios are whatever the real activation happened to contain.
"""
import pickle, json
from collections import defaultdict

THR = 0.3
SRC = [("../results/RESULTS/constructed_real.pkl", "h", "constructed prose"),
       ("../results/RESULTS/external_real.pkl", "f", "external news")]


def records():
    for path, field, label in SRC:
        try:
            store = pickle.load(open(path, "rb"))
        except FileNotFoundError:
            continue
        for key, v in store.items():
            rec = v.get(field)
            if not rec or not rec.get("top"):
                continue
            yield label, key, rec, v.get("descs", [])


def detection_for(idx, descs):
    """fraction of the 12 descriptions of this activation that named latent idx"""
    n = hit = 0
    for entry in descs:
        scores = entry[2] if len(entry) > 2 else {}
        if idx in scores:
            n += 1
            hit += (scores[idx] >= THR)
    return hit, n


if __name__ == "__main__":
    by_rank = defaultdict(lambda: [0, 0])
    by_ratio = defaultdict(lambda: [0, 0])
    BINS = [(0.90, 1.01, "90-100% of top"), (0.60, 0.90, "60-90%"),
            (0.40, 0.60, "40-60%"), (0.25, 0.40, "25-40%"),
            (0.10, 0.25, "10-25%"), (0.0, 0.10, "<10%")]
    n_rec = 0
    for label, key, rec, descs in records():
        if not descs:
            continue
        n_rec += 1
        top = rec["top"]
        top1 = top[0][1]
        for r, (idx, act) in enumerate(((t[0], t[1]) for t in top), start=1):
            hit, n = detection_for(idx, descs)
            if n == 0:
                continue
            by_rank[min(r, 6)][0] += hit; by_rank[min(r, 6)][1] += n
            ratio = act / top1 if top1 else 0
            for lo, hi, name in BINS:
                if lo <= ratio < hi:
                    by_ratio[name][0] += hit; by_ratio[name][1] += n
                    break

    print(f"real activations analysed: {n_rec}\n")
    print("=== how often the description names a latent, BY RANK ===")
    print(f"{'rank in the activation':>24}{'named':>14}")
    for r in sorted(by_rank):
        h, n = by_rank[r]
        lab = f"{r}" if r < 6 else "6+"
        print(f"{lab:>24}{h:>7}/{n:<5} {h/n*100:>3.0f}%")

    print("\n=== ...and BY STRENGTH RELATIVE TO THE TOP LATENT ===")
    print(f"{'relative strength':>24}{'named':>14}")
    for lo, hi, name in BINS:
        if name not in by_ratio:
            continue
        h, n = by_ratio[name]
        print(f"{name:>24}{h:>7}/{n:<5} {h/n*100:>3.0f}%")

    json.dump({"by_rank": {str(k): v for k, v in by_rank.items()},
               "by_relative_strength": dict(by_ratio), "n_activations": n_rec},
              open("../results/RESULTS/real_activation_ranks.json", "w"), indent=1)
    print("\nwrote real_activation_ranks.json")
