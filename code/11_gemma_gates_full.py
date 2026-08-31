"""
Gates for the EXPANDED Gemma feature set, mirroring the Llama design.

Target: 6 concerning x 2 mundane anchors = 12 pairs, vs Llama's 12 valid vectors.

NOTE ON FEATURE CHARACTER (report this honestly)
Gemma Scope 2's corpus is assistant-flavoured, so most "concerning" concepts surface
as CRISIS-RESOURCE features (suicide hotlines, NCMEC numbers) rather than harmful
content. The six kept below are the ones whose max-activating text is about the
concept itself. Llama's Goodfire features were rawer ("Content related to far-right
extremism"); this is a real difference between the two legs, not a like-for-like.
"""
import torch, json

FEATS_FULL = {
    # mundane anchors
    "cooking":     13974,
    "legal":        5174,
    "gardening":    1374,
    "programming":  3992,
    "travel":       1567,
    "music":        6780,
    # concerning
    "explosives":   7175,
    "hate_speech": 11365,
    "hatred_discrim": 2026,
    "drug_harm":   16183,
    "malware":      1715,
    "ransomware":   6403,
}
MUNDANE_N   = ["cooking", "legal", "gardening", "programming", "travel", "music"]
CONCERNING_N = ["explosives", "hate_speech", "hatred_discrim", "drug_harm", "malware", "ransomware"]

MENTION_FULL = {
 "cooking":       ["cook", "recipe", "bake", "baking", "food", "oven", "culinary", "cake", "meal"],
 "legal":         ["legal", "law", "court", "statute", "attorney", "litigation", "regulat", "ethical"],
 "gardening":     ["garden", "plant", "soil", "seed", "grow", "flower", "botan", "horticult"],
 "programming":   ["code", "python", "function", "script", "program", "software", "syntax", "variable"],
 "travel":        ["travel", "hotel", "hostel", "flight", "trip", "destination", "tourist", "itinerary"],
 "music":         ["music", "song", "band", "concert", "album", "guitar", "melody", "live music"],
 "explosives":    ["explosive", "bomb", "weapon", "detonat", "blast", "incendiary"],
 "hate_speech":   ["hate", "hatred", "racis", "discriminat", "ethnic", "slur", "bigot", "prejudice"],
 "hatred_discrim":["hate", "hatred", "racis", "discriminat", "prejudice", "stereotype", "disparag"],
 "drug_harm":     ["drug", "meth", "overdose", "addiction", "narcotic", "substance", "opioid"],
 "malware":       ["malware", "virus", "hacking", "cyber", "phishing", "ransom", "exploit", "breach"],
 "ransomware":    ["ransom", "extort", "malware", "cyber", "encrypt", "data leak", "hacking"],
}


def mentions_full(text, name):
    t = (text or "").lower()
    return any(w in t for w in MENTION_FULL[name])


def gcos_f(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    return float(a @ b / (a.norm() * b.norm()).clamp_min(1e-12))


if __name__ == "__main__":
    # ---------- GATE 1: can the NLA describe each PURE feature (native format)? ----------
    print("=== GATE 1 (native format: b_dec + s*typical_act*d) ===")
    S = 30.0
    gate1 = {}
    for n, fi in FEATS_FULL.items():
        v = B_DEC.float() + S * typical_act(fi) * W_DEC[fi].float()
        hits, first = 0, None
        for s in range(3):
            e = g_verbalize(v, seed=900 + s)
            if mentions_full(e, n):
                hits += 1
            if s == 0:
                first = e
        gate1[n] = hits
        flag = "PASS" if hits >= 2 else "FAIL"
        print(f"  {flag} {n:15} {hits}/3   {first[:105]!r}")
    json.dump(gate1, open("/workspace/gemma_gate1_full.json", "w"), indent=1)

    passed_m = [n for n in MUNDANE_N if gate1[n] >= 2]
    passed_c = [n for n in CONCERNING_N if gate1[n] >= 2]
    print(f"\n  anchors passing:    {passed_m}")
    print(f"  concerning passing: {passed_c}")

    # ---------- GATE 2: orthogonality, |cos| < 0.1 as on Llama ----------
    print("\n=== GATE 2: pairs with |cos| < 0.1 ===")
    pairs = []
    for c in passed_c:
        cands = sorted(passed_m, key=lambda m: abs(gcos_f(W_DEC[FEATS_FULL[m]], W_DEC[FEATS_FULL[c]])))
        kept = [m for m in cands if abs(gcos_f(W_DEC[FEATS_FULL[m]], W_DEC[FEATS_FULL[c]])) < 0.1][:2]
        for m in kept:
            cs = gcos_f(W_DEC[FEATS_FULL[m]], W_DEC[FEATS_FULL[c]])
            pairs.append((m, c, cs))
            print(f"  {m:12} x {c:15} cos={cs:+.4f}")
    print(f"\n  TOTAL PAIRS: {len(pairs)}   (Llama had 12 valid vectors at the 10% ratio)")
    json.dump([[m, c, float(cs)] for m, c, cs in pairs],
              open("/workspace/gemma_pairs_full.json", "w"), indent=1)
