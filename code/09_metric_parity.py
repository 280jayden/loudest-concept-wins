"""
Metric parity between Llama and Gemma.

PROBLEM
Llama used a VALIDATED behavioural metric (description -> synthetic conversations
-> SAE reactivation; 0% FP, 87% sensitivity, 92% manual agreement).
Gemma used keyword matching on the explanation text, which is unvalidated and has
already produced two false positives today. So "1.6% vs 50%" was never like-for-like.

FIX (cheap half)
Score the Llama descriptions with the SAME keyword matcher used on Gemma. That
gives an apples-to-apples comparison, AND - because Llama has both metrics - it
tells us how far keyword matching drifts from the validated one. That drift is the
error bar we should attach to every Gemma number.
"""
import pickle, numpy as np

# keyword lists for the Llama concepts, in the same style as MENTION_G
KW_LLAMA = {
 12201: ["cook", "recipe", "culinary", "food", "kitchen", "meal", "dish"],          # act of cooking
 16864: ["consumer", "protection law", "regulat", "legal", "law", "statute"],       # consumer protection law
 11970: ["bake", "baking", "oven", "pastry", "bread", "cake"],                      # baking
 45010: ["legal", "law", "clause", "document", "statut", "provision", "contract"],  # legalese
 21592: ["spice", "seasoning", "herb", "curry", "masala", "flavour", "flavor"],     # spices
 1755:  ["legal", "court", "defen", "testimony", "trial", "attorney", "witness"],   # legal defence
 56450: ["extremis", "far-right", "supremac", "nazi", "racis", "hate", "bigot"],    # far-right extremism
 6214:  ["scam", "fraud", "phishing", "suspicious", "deceptive", "fake"],           # scams/fraud
}
PAIRS_L = [("cooking x consumer-law", 12201, 16864), ("baking x legalese", 11970, 45010),
           ("spices x criminal-defense", 21592, 1755), ("baking x EXTREMISM", 11970, 56450),
           ("cooking x SCAM-FRAUD", 12201, 6214)]
SCALES = [0.5, 0.8, 1.3, 2.1, 3.4, 5.5]
ALPHAS = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
THR = 0.3


def kw_hit(text, fidx):
    t = (text or "").lower()
    return any(w in t for w in KW_LLAMA[fidx])


def parity():
    sw = pickle.load(open("/workspace/sweep_results.pkl", "rb"))
    agree = disagree_bh = disagree_kh = 0     # bh = behavioural hit only, kh = keyword only
    per_alpha = {}
    for nm, fa, fb in PAIRS_L:
        for al in ALPHAS:
            kb = kk = n = 0
            for sc in SCALES:
                for d in sw.get(("trained", nm, al, sc), []):
                    n += 1
                    beh_A, beh_B = d["hit_A"] >= THR, d["hit_B"] >= THR
                    kw_A, kw_B = kw_hit(d["label"], fa), kw_hit(d["label"], fb)
                    kb += (beh_A and beh_B)          # both, behavioural
                    kk += (kw_A and kw_B)            # both, keyword
                    for beh, kw in ((beh_A, kw_A), (beh_B, kw_B)):
                        if beh == kw:
                            agree += 1
                        elif beh:
                            disagree_bh += 1
                        else:
                            disagree_kh += 1
            a = per_alpha.setdefault(al, [0, 0, 0])
            a[0] += kb; a[1] += kk; a[2] += n
    print("LLAMA scored BOTH ways (trained adapter, 630 descriptions)")
    print(f"{'alpha':>6}{'behavioural both':>19}{'keyword both':>15}")
    for al in ALPHAS:
        kb, kk, n = per_alpha[al]
        print(f"{al:>6}{kb:>10}/{n:<7}{kk:>9}/{n:<5}")
    tot_b = sum(v[0] for v in per_alpha.values())
    tot_k = sum(v[1] for v in per_alpha.values())
    tot_n = sum(v[2] for v in per_alpha.values())
    print(f"\n  overall both:  behavioural {tot_b}/{tot_n} = {tot_b/tot_n*100:.1f}%"
          f"  |  keyword {tot_k}/{tot_n} = {tot_k/tot_n*100:.1f}%")
    tot = agree + disagree_bh + disagree_kh
    print(f"\n  per-concept agreement between the two metrics: {agree}/{tot} = {agree/tot*100:.1f}%")
    print(f"    behavioural says hit, keyword misses : {disagree_bh}  ({disagree_bh/tot*100:.1f}%)")
    print(f"    keyword says hit, behavioural misses : {disagree_kh}  ({disagree_kh/tot*100:.1f}%)")
