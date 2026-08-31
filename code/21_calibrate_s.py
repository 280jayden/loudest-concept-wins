"""
Find an s that keeps the INTENDED mixture ratios AND keeps both concepts above the
SAE's detection floor.

Constraint 1 (ratio fidelity): at nominal 25% concerning, measured act_conc/act_mund
should sit near 0.33; at 10%, near 0.11. And the concerning concept must never exceed
the mundane one at those shares.

Constraint 2 (above floor): act_conc > 0 in every cell, otherwise the SAE cannot see
the concept and the cell is untestable.

Raising s scales BOTH concepts, so the ratio is preserved while weak features clear
the JumpReLU threshold. We sweep s and report the first value satisfying both.
"""
import torch

TARGET = {0.5: 1.00, 0.75: 0.33, 0.9: 0.11}      # nominal ratios implied by alpha


def make_fixed_s(fa, fb, alpha, s):
    dA, dB = W_DEC[fa].float(), W_DEC[fb].float()
    dA = dA / dA.norm(); dB = dB / dB.norm()
    mix = alpha * dA + (1 - alpha) * dB
    mix = mix / mix.norm().clamp_min(1e-12)
    return B_DEC.float() + s * mix


def evaluate(s):
    """Return (all_above_floor, max_ratio_error, rows)."""
    rows, ok, err = [], True, 0.0
    for m, c in PAIRS8:
        fa, fb = FEATS_FULL[m], FEATS_FULL[c]
        for al in [0.5, 0.75, 0.9]:
            v = make_fixed_s(fa, fb, al, s)
            a = sae_encode(v)
            gA, gB = float(a[fa]), float(a[fb])
            if gA <= 0 or gB <= 0:
                ok = False
            r = gB / gA if gA > 0 else float("inf")
            if al >= 0.75 and gA > 0:
                err = max(err, abs(r - TARGET[al]))
            rows.append((m, c, al, gA, gB, r))
    return ok, err, rows


if __name__ == "__main__":
    print(f"{'s':>9}{'all above floor':>18}{'max ratio error':>18}{'||v||':>10}")
    best = None
    for s in [6000, 10000, 15000, 20000, 30000, 45000, 60000, 80000]:
        ok, err, rows = evaluate(s)
        nrm = float((B_DEC.float() + s * (W_DEC[FEATS_FULL['cooking']].float() /
                     W_DEC[FEATS_FULL['cooking']].float().norm())).norm())
        print(f"{s:>9}{str(ok):>18}{err:>18.3f}{nrm:>10.0f}")
        if ok and best is None:
            best = s
    print(f"\n  first s satisfying both constraints: {best}")

    if best:
        ok, err, rows = evaluate(best)
        print(f"\n=== s={best}: intended vs measured ratios ===")
        print(f"{'pair':30}{'nominal':>9}{'target':>8}{'measured':>10}{'act_mund':>10}{'act_conc':>10}")
        for m, c, al, gA, gB, r in rows:
            share = f"{int((1-al)*100)}%"
            print(f"{m+' x '+c:30}{share:>9}{TARGET[al]:>8.2f}{r:>10.2f}{gA:>10.0f}{gB:>10.0f}")
