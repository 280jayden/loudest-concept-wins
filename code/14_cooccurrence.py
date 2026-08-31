"""
Behavioural relatedness: do two features FIRE ON THE SAME TEXT?

Decoder cosine did not capture semantic relatedness - it passed legal x discrimination
(-0.0034) and programming x malware (+0.0171) as "distinct". Plausible reason: an SAE is
trained to DECOMPOSE, so shared structure between two related concepts tends to be
factored into its own feature, leaving each decoder row encoding what is DISTINCTIVE.
Low cosine is partly what the objective produces, not evidence of semantic independence.

Co-occurrence is the natural alternative: legal and discrimination should fire on the
same documents constantly (discrimination law); cooking and malware essentially never.
examples.safetensors stores, for every feature, the 1000 sequences it fires hardest on -
so this is a lookup, no GPU needed.

If co-occurrence predicts which minority concepts survive at 10% share, the relatedness
account is rescued with a proper measure. If it does not, the account dies cleanly.
"""
import torch, json

def cooc(fi, fj):
    """Jaccard overlap of the sequence sets the two features fire on."""
    a = set(int(x) for x in SEQ[fi].tolist() if x >= 0)
    b = set(int(x) for x in SEQ[fj].tolist() if x >= 0)
    if not a or not b:
        return 0.0, 0
    inter = len(a & b)
    return inter / len(a | b), inter


def dcos(fi, fj):
    a, b = W_DEC[fi].float(), W_DEC[fj].float()
    return float(a @ b / (a.norm() * b.norm()).clamp_min(1e-12))


if __name__ == "__main__":
    # the 8 Gemma pairs, tagged by whether the minority SURVIVED at 10% share
    PAIRS_TAG = [
        ("travel",      "explosives",     0),   # 0/6 at 10%
        ("legal",       "explosives",     0),
        ("cooking",     "hatred_discrim", 0),
        ("legal",       "hatred_discrim", 1),   # 6/6 - survived
        ("cooking",     "malware",        0),
        ("programming", "malware",        1),   # 6/6 - survived
        ("legal",       "ransomware",     0),
        ("cooking",     "ransomware",     0),
    ]
    print(f"{'pair':34}{'survived':>10}{'dec cos':>10}{'co-occur':>11}{'shared seqs':>13}")
    rows = []
    for m, c, surv in PAIRS_TAG:
        fi, fj = FEATS_FULL[m], FEATS_FULL[c]
        j, inter = cooc(fi, fj)
        d = dcos(fi, fj)
        rows.append((surv, j, d, f"{m} x {c}"))
        print(f"{m+' x '+c:34}{surv:>10}{d:>+10.4f}{j:>11.4f}{inter:>13}")

    surv = [r[1] for r in rows if r[0] == 1]
    died = [r[1] for r in rows if r[0] == 0]
    print(f"\n  survived (n={len(surv)}): mean co-occurrence {sum(surv)/len(surv):.4f}")
    print(f"  silent   (n={len(died)}): mean co-occurrence {sum(died)/len(died):.4f}")
    scos = [abs(r[2]) for r in rows if r[0] == 1]
    dcos_ = [abs(r[2]) for r in rows if r[0] == 0]
    print(f"\n  survived: mean |decoder cos| {sum(scos)/len(scos):.4f}")
    print(f"  silent  : mean |decoder cos| {sum(dcos_)/len(dcos_):.4f}")
    print("\n  -> if co-occurrence separates the groups but decoder cosine does not,")
    print("     co-occurrence is the right relatedness measure and our Gate 2 was wrong.")
    json.dump([[int(s), float(j), float(d), n] for s, j, d, n in rows],
              open("/workspace/gemma_cooccurrence.json", "w"), indent=1)
