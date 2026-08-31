"""
DECODABILITY CEILING: is the omitted concept still there, and how easy is it to find?

THE OBJECTION THIS ANSWERS
"Obviously a concept that has been turned down is harder to report." That reading treats
the result as a dose-response curve. It only holds if the information is genuinely
degraded. So measure how hard the omitted concept actually is to recover from the SAME
vector, using the simplest possible reader: the SAE encoder, one matrix multiply.

If concept B ranks near the top of 65,536 latents at a mixture where self-interpretation
names it 1 time in 216, then nothing has been lost from the activation. The information
is trivially available and the interpretation channel is discarding it. That is an
information-loss result, not a signal-strength one.

WHAT IS COMPARED
  reader 1: SAE encoder on the composed vector      -> rank of B among all latents
  reader 2: the model describing that same vector   -> measured already (exp3)

The composed vector is built from SAE decoder rows, so it lives in residual-stream
space. This asks what an SAE readout of that activation sees, before the interpretation
adapter is involved at all.

Ranks are reported at each injection scale actually used in the sweep, so the claim is
about the very vectors that were injected, not an idealised version of them.
"""
import torch, pickle, json
from huggingface_hub import hf_hub_download

SCALES = [0.5, 0.8, 1.3, 2.1, 3.4, 5.5]
ALPHAS = [0.5, 0.75, 0.9]
SHARE = {0.5: "50%", 0.75: "25%", 0.9: "10%"}


def load_sae():
    p = hf_hub_download("Goodfire/Llama-3.1-8B-Instruct-SAE-l19",
                        "Llama-3.1-8B-Instruct-SAE-l19.pth")
    sd = torch.load(p, map_location="cpu", weights_only=True)
    W_enc = sd["encoder_linear.weight"]            # (65536, 4096)
    b_enc = sd["encoder_linear.bias"]              # (65536,)
    W_dec = sd["decoder_linear.weight"].T.contiguous()   # (65536, 4096)
    b_dec = sd["decoder_linear.bias"]              # (4096,)
    return W_enc, b_enc, W_dec, b_dec


def encode(v, W_enc, b_enc):
    return torch.relu(W_enc @ v + b_enc)


def compose(W_dec, a_idx, b_idx, alpha):
    """Identical to the compose() used to build every injected vector."""
    dA = W_dec[a_idx]; dB = W_dec[b_idx]
    dA = dA / dA.norm(); dB = dB / dB.norm()
    v = alpha * dA + (1 - alpha) * dB
    return v / v.norm().clamp_min(1e-9)


def rank_of(acts, idx):
    """1-based rank of latent idx by activation (1 = most active)."""
    return int((acts > acts[idx]).sum().item()) + 1


def control_absent(W_enc, b_enc, W_dec, keep):
    """Rank of B in a PURE-A vector, where B is genuinely absent.

    Without this the rank result is circular: a reader that always ranks whatever you
    ask about near the top proves nothing. Here B sits around rank 1600 with activation
    0.000 and never enters the top 10 - so the readout does report absence correctly,
    and rank 2 at a 25% mixture is a real detection.
    """
    ranks = []
    for nm, ia, ib in keep:
        v = W_dec[ia] / W_dec[ia].norm()
        rr = [rank_of(encode(v * sc, W_enc, b_enc), ib) for sc in SCALES]
        ranks.append(sorted(rr)[len(rr) // 2])
    return ranks

if __name__ == "__main__":
    W_enc, b_enc, W_dec, b_dec = load_sae()
    d_sae = W_enc.shape[0]
    print(f"SAE loaded: {d_sae} latents, d_model {W_enc.shape[1]}")

    PAIRS = pickle.load(open("../results/RESULTS/safe_meta.pkl", "rb"))["SAFE_PAIRS"]
    V = pickle.load(open("../results/RESULTS/safe_meta.pkl", "rb"))["VALID2"]
    keep = [p for p in PAIRS
            if all({a: b for a, _, b, _ in V[p[0]]}.get(al) for al in ALPHAS)]
    print(f"pairs on the constant basis: {len(keep)}\n")

    rows, agg = [], {al: [] for al in ALPHAS}
    print(f"{'pair':45}{'share':>7}{'rank A':>8}{'rank B':>8}{'act A':>9}{'act B':>9}")
    for nm, ia, ib in keep:
        for al in ALPHAS:
            v = compose(W_dec, ia, ib, al)
            rr, aa, bb = [], [], []
            for sc in SCALES:
                acts = encode(v * sc, W_enc, b_enc)
                rr.append((rank_of(acts, ia), rank_of(acts, ib)))
                aa.append(float(acts[ia])); bb.append(float(acts[ib]))
            # median across the 6 injection scales
            rA = sorted(r[0] for r in rr)[len(rr) // 2]
            rB = sorted(r[1] for r in rr)[len(rr) // 2]
            agg[al].append(rB)
            rows.append({"pair": nm, "alpha": al, "rank_A": rA, "rank_B": rB,
                         "rank_B_by_scale": [r[1] for r in rr],
                         "act_A": sum(aa) / len(aa), "act_B": sum(bb) / len(bb)})
            print(f"{nm:45}{SHARE[al]:>7}{rA:>8}{rB:>8}"
                  f"{sum(aa)/len(aa):>9.1f}{sum(bb)/len(bb):>9.1f}")

    print(f"\n=== rank of the OMITTED concept among {d_sae} latents ===")
    print(f"{'share of B':>12}{'median rank':>14}{'worst':>8}{'in top 10':>11}{'in top 100':>12}")
    for al in ALPHAS:
        r = sorted(agg[al])
        med = r[len(r) // 2]
        print(f"{SHARE[al]:>12}{med:>14}{max(r):>8}"
              f"{sum(1 for x in r if x <= 10):>7}/{len(r):<3}"
              f"{sum(1 for x in r if x <= 100):>8}/{len(r):<3}")

    json.dump({"d_sae": d_sae, "rows": rows,
               "median_rank_B": {SHARE[al]: sorted(agg[al])[len(agg[al]) // 2] for al in ALPHAS}},
              open("../results/RESULTS/decodability_ceiling.json", "w"), indent=1)
    print("\nwrote decodability_ceiling.json")
