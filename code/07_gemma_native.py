"""
Gemma NLA, NATIVE input format.

WHY THE FORMAT DIFFERS FROM LLAMA
SelfIE adapters were TRAINED on bare SAE decoder vectors -> a decoder row is
in-distribution for them. NLAs were trained on REAL residual activations, which an
SAE says are  x ~= b_dec + sum_i f_i * d_i  (this SAE: L0=120 active features).
A bare decoder row never occurs in nature, so feeding one to the NLA tests it OOD
- visible as confabulated output and the identical 'Final token "it\'"' artifact in
all six Gate-1 explanations.

So each method gets its native encoding, and we ask the IDENTICAL question:
at mixture ratio alpha, does the description report both concepts?

    v(alpha) = b_dec + s * ( alpha*d_A + (1-alpha)*d_B )

s is taken from real feature-activation magnitudes in examples.safetensors, so the
concept coefficients are realistic rather than invented.
"""
import torch, json, pickle, os, numpy as np

# typical activation magnitude for a firing feature (median of each feature's top acts)
def typical_act(fi, q=0.5):
    a = ACT[fi]
    a = a[a > 0]
    return float(a.float().quantile(q)) if a.numel() else 1.0


def make_native(fa, fb, alpha, s_scale=1.0):
    """b_dec + s*(alpha*dA + (1-alpha)*dB)  -- how an SAE says activations are built."""
    dA, dB = W_DEC[fa].float(), W_DEC[fb].float()
    sA, sB = typical_act(fa), typical_act(fb)
    mix = alpha * sA * dA + (1 - alpha) * sB * dB
    return B_DEC.float() + s_scale * mix


def check_native(fa, fb, alpha, s_scale=1.0):
    """Ground truth: does the SAE encoder still see BOTH features in this vector?"""
    v = make_native(fa, fb, alpha, s_scale)
    a = sae_encode(v)
    return float(a[fa]), float(a[fb]), float(v.norm())


if __name__ == "__main__":
    print("=== typical feature activations (from real max-act data) ===")
    for n, fi in FEATS.items():
        print(f"  {n:15} f{fi:6d}  median_act={typical_act(fi):8.1f}  freq={float(EX['feature_frequencies'][fi]):.5f}")
    print(f"\nb_dec norm = {B_DEC.norm():.1f}   | actor expects injected norm = {G_INJ_SCALE}")

    PAIRS_G = [("cooking", "legal"), ("cooking", "hatred_discrim"), ("cooking", "explosives")]
    ALPHAS_G = [0.0, 0.25, 0.5, 0.75, 0.9, 1.0]

    print("\n=== VALIDITY: does the SAE encoder see both features? (s_scale sweep) ===")
    for s_scale in [1.0, 3.0, 10.0]:
        print(f"\n-- s_scale={s_scale} --")
        for A, B in PAIRS_G[:1]:
            for al in ALPHAS_G:
                aA, aB, nrm = check_native(FEATS[A], FEATS[B], al, s_scale)
                both = "both" if (aA > 0 and aB > 0) else ("A" if aA > 0 else ("B" if aB > 0 else "NEITHER"))
                print(f"   a={al:<5} act_A={aA:8.1f} act_B={aB:8.1f} ||v||={nrm:9.1f}  {both}")

    print("\n=== NLA on native-format vectors (does confabulation go away?) ===")
    for s_scale in [3.0, 10.0]:
        print(f"\n--- s_scale={s_scale} ---")
        for n in ["cooking", "hatred_discrim", "explosives"]:
            fi = FEATS[n]
            v = B_DEC.float() + s_scale * typical_act(fi) * W_DEC[fi].float()
            e = g_verbalize(v, seed=77)
            print(f"  [{n}] {e[:300]!r}")
            print(f"      keyword hit: {g_mentions(e, n)}")
