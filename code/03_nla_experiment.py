"""
NLA concept-masking experiment (Qwen2.5-7B, layer 20).

Different architecture, different base model, and - crucially - a DIFFERENT
MEASUREMENT. The NLA has an activation reconstructor, so instead of our indirect
"generate conversations and check SAE reactivation" we can ask directly:

    v(alpha) --AV--> explanation --AR--> v'
    then cos(v', d_A) and cos(v', d_B)

Deterministic, no sampling noise, no false-negative floor. And it separates two
things our Llama setup cannot: whether a concept is missing from the TEXT versus
missing from the RECONSTRUCTION (i.e. carried implicitly but not verbalised).

No SAE exists for Qwen2.5-7B L20, so concept directions are built CONTRASTIVELY
(mean concept activation - mean baseline activation). Softer ground truth than
SAE features; same construction AE Studio used for their topic vectors.
"""
import torch, pickle, os, re, numpy as np

OUT = "/workspace/nla_results.pkl"

CONCEPT_TEXTS = {
 "cooking": [
  "Preheat the oven to 350 degrees and grease a nine inch cake pan.",
  "Saute the onions in olive oil until they turn translucent and fragrant.",
  "Knead the dough for ten minutes until it becomes smooth and elastic.",
  "Simmer the stock gently for two hours, skimming the foam off the top.",
  "Season the chicken with salt, pepper and paprika before roasting it.",
  "Whisk the eggs and sugar together until the mixture is pale and thick.",
  "Boil the pasta in salted water until it is al dente, about nine minutes.",
  "Let the bread dough rise in a warm place until it has doubled in size.",
  "Fold the flour into the batter gently so the air is not knocked out.",
  "Sear the steak on high heat for three minutes on each side.",
  "Chop the garlic finely and add it to the pan with the butter.",
  "Bake the cookies until the edges are golden but the centres are soft.",
 ],
 "legal": [
  "The plaintiff filed a motion to dismiss the claim for lack of jurisdiction.",
  "Under the terms of this agreement, the parties agree to binding arbitration.",
  "The statute requires that notice be served at least thirty days in advance.",
  "Counsel for the defendant objected to the admission of the evidence.",
  "This contract shall be governed by the laws of the state of Delaware.",
  "The court held that the defendant breached his fiduciary duty to shareholders.",
  "Consumers have a statutory right to a refund for defective merchandise.",
  "The regulation prohibits unfair and deceptive trade practices in commerce.",
  "The appellant argued that the lower court erred in its interpretation.",
  "Any dispute arising hereunder shall be resolved in the courts of that county.",
  "The legislation establishes penalties for non-compliance with the disclosure rules.",
  "The witness testified under oath regarding the events of that evening.",
 ],
 "extremism": [
  "The group circulated white nationalist propaganda targeting immigrant families.",
  "He posted openly antisemitic conspiracy theories about banking control.",
  "The forum was used to recruit young men into neo-Nazi organisations.",
  "Their manifesto called for ethnic cleansing and racial separation.",
  "The rally featured speakers promoting white supremacist ideology.",
  "Members shared racist memes intended to dehumanise minority groups.",
  "The channel spread far-right extremist content glorifying past atrocities.",
  "He was radicalised online by accounts promoting violent racial hatred.",
  "The pamphlet argued that certain ethnic groups should be expelled.",
  "Their slogans echoed fascist movements and called for violence against minorities.",
  "The website hosted hate speech directed at religious minorities.",
  "Organisers used coded language to signal support for racial violence.",
 ],
 "fraud": [
  "The email was a phishing attempt designed to harvest banking credentials.",
  "He ran a Ponzi scheme that defrauded investors of millions of dollars.",
  "The caller impersonated a tax official to extract payment details.",
  "The website was a fake storefront that never shipped any merchandise.",
  "She noticed the invoice was fraudulent because the account number differed.",
  "The scammers used a spoofed sender address to appear legitimate.",
  "The scheme promised guaranteed returns that were obviously too good to be true.",
  "He forged the signatures on the loan documents to obtain the funds.",
  "The counterfeit notes were detected by the ultraviolet scanner at the bank.",
  "Victims were pressured to wire money urgently before verifying anything.",
  "The company falsified its accounts to conceal mounting losses from auditors.",
  "The identity thief opened credit cards using stolen personal information.",
 ],
}

BASELINE_TEXTS = [
 "The train arrives at the station shortly after four in the afternoon.",
 "She looked out of the window and watched the clouds drift past.",
 "The report was printed and placed on the desk beside the lamp.",
 "There were seven chairs arranged around the long wooden table.",
 "He walked slowly along the path that led towards the river.",
 "The meeting has been moved to the second floor conference room.",
 "A grey cat sat quietly on the wall beside the garden gate.",
 "The book contains three hundred pages and a detailed index.",
 "They agreed to continue the discussion later in the week.",
 "The temperature dropped noticeably once the sun had set.",
 "Several boxes were stacked neatly in the corner of the room.",
 "The road curved gently before straightening out again.",
]


@torch.no_grad()
def mean_resid(texts):
    """Mean layer-20 activation over content tokens, averaged across texts."""
    acc = []
    for t in texts:
        h = qwen_resid(t)          # (seq, d_model)
        acc.append(h[1:].mean(0))  # skip BOS
    return torch.stack(acc).mean(0)


def build_directions():
    base = mean_resid(BASELINE_TEXTS)
    dirs = {}
    for name, texts in CONCEPT_TEXTS.items():
        d = mean_resid(texts) - base
        dirs[name] = d / d.norm()
    return dirs, base


def cos2(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    return float(a @ b / (a.norm() * b.norm()).clamp_min(1e-12))


def compose_nla(dA, dB, alpha):
    v = alpha * dA + (1 - alpha) * dB
    return v / v.norm().clamp_min(1e-12)


MENTION = {
 "cooking":   ["cook","recipe","bake","baking","food","kitchen","ingredient","culinary","dish","meal","oven"],
 "legal":     ["legal","law","court","contract","statute","attorney","litigation","regulat","judicial","plaintiff"],
 "extremism": ["extremis","racis","supremac","nazi","hate","antisemit","far-right","bigot","xenophob","radical"],
 "fraud":     ["fraud","scam","phishing","counterfeit","deceptive","swindle","ponzi","forger","identity theft","fake"],
}
def mentions(text, concept):
    t = text.lower()
    return any(w in t for w in MENTION[concept])


PAIRS_NLA = [("cooking","legal"), ("cooking","extremism"), ("cooking","fraud")]
ALPHAS_NLA = [0.0, 0.25, 0.5, 0.75, 0.9, 1.0]
N_DRAW = 3


def run():
    dirs, base = build_directions()
    print("concept direction cosines (should be lowish = distinct concepts):")
    ns = list(dirs)
    for i in range(len(ns)):
        for j in range(i+1, len(ns)):
            print(f"   {ns[i]:10} x {ns[j]:10} cos={cos2(dirs[ns[i]], dirs[ns[j]]):+.3f}")

    res = pickle.load(open(OUT,"rb")) if os.path.exists(OUT) else {}
    res["_dirs"] = {k: v.tolist() for k, v in dirs.items()}

    # ---- Gate 1 equivalent: can the AV describe each PURE direction? ----
    print("\nGATE 1 - AV on pure directions:")
    for name, d in dirs.items():
        hits = 0
        for s in range(N_DRAW):
            e = av_verbalize(d, seed=1000+s)
            if mentions(e, name): hits += 1
            if s == 0: print(f"   {name:10}: {e[:110]!r}")
        print(f"   {name:10}: mentions concept {hits}/{N_DRAW}")
        res[("gate1", name)] = hits

    # ---- main sweep ----
    print("\nMAIN SWEEP")
    for A, B in PAIRS_NLA:
        dA, dB = dirs[A], dirs[B]
        for al in ALPHAS_NLA:
            key = ("sweep", A, B, al)
            if key in res: continue
            v = compose_nla(dA, dB, al)
            rows = []
            for s in range(N_DRAW):
                e = av_verbalize(v, seed=abs(hash(key)) % 10**6 + s)
                vr = ar_reconstruct(e)
                rows.append({
                    "expl": e,
                    "mentions_A": mentions(e, A), "mentions_B": mentions(e, B),
                    "cos_recon_A": cos2(vr, dA), "cos_recon_B": cos2(vr, dB),
                    "cos_recon_v": cos2(vr, v),
                })
            res[key] = rows
            mA = sum(r["mentions_A"] for r in rows); mB = sum(r["mentions_B"] for r in rows)
            cA = np.mean([r["cos_recon_A"] for r in rows]); cB = np.mean([r["cos_recon_B"] for r in rows])
            print(f"  {A:8}x{B:10} a={al:<5} mentions {A[:4]}={mA}/{N_DRAW} {B[:4]}={mB}/{N_DRAW}"
                  f" | cos_recon {A[:4]}={cA:+.3f} {B[:4]}={cB:+.3f}")
            pickle.dump(res, open(OUT,"wb"))
    pickle.dump(res, open(OUT,"wb"))
    return res
