"""
STEP 1 (free, no GPU): does removing the per-feature typical_act scaling make the
Gemma mixture ratios match the nominal shares?

BROKEN (what we ran):
    mix = a*typical_act(A)*dA + (1-a)*typical_act(B)*dB
    typical activations differ hugely between features (~4k vs ~15k), so they swamp a.
    Result: nominal "25% concerning" produced actual ratios from 0.20 to 1.63 - in
    programming x malware the "minority" concept was 1.6x STRONGER than the dominant one.

FIXED (matches Llama):
    mix = a*dA + (1-a)*dB      -> renormalise -> v = b_dec + s*mix

b_dec stays: it is what fixed the confabulation problem (a real activation is ~92% DC).
Only the per-feature scaling goes.

Pass condition: ratios at a=0.75 should sit near Llama's 0.14-0.38 band, and the
concerning concept must never exceed the mundane one.
"""
import os, json, torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file

REPO, SAE_PATH = "google/gemma-scope-2-12b-it", "resid_post_all/layer_32_width_16k_l0_big"
base = os.path.join(snapshot_download(REPO, allow_patterns=[f"{SAE_PATH}/*"]), SAE_PATH)
P = load_file(os.path.join(base, "params.safetensors"))
EX = load_file(os.path.join(base, "examples.safetensors"))
W_ENC, W_DEC = P["w_enc"], P["w_dec"]
B_ENC, B_DEC, THRESH = P["b_enc"], P["b_dec"], P["threshold"]
ACT = EX["activations"]

FEATS_FULL = {"cooking": 13974, "legal": 5174, "gardening": 1374, "programming": 3992,
              "travel": 1567, "music": 6780, "explosives": 7175, "hate_speech": 11365,
              "hatred_discrim": 2026, "drug_harm": 16183, "malware": 1715, "ransomware": 6403}
PAIRS8 = [("travel", "explosives"), ("legal", "explosives"), ("cooking", "hatred_discrim"),
          ("legal", "hatred_discrim"), ("cooking", "malware"), ("programming", "malware"),
          ("legal", "ransomware"), ("cooking", "ransomware")]


def sae_encode(v):
    pre = v.float() @ W_ENC + B_ENC
    return torch.where(pre > THRESH, torch.relu(pre), torch.zeros_like(pre))


def typical_act(fi, q=0.5):
    a = ACT[fi]; a = a[a > 0]
    return float(a.float().quantile(q)) if a.numel() else 1.0


def make_broken(fa, fb, alpha, s=30.0):
    dA, dB = W_DEC[fa].float(), W_DEC[fb].float()
    mix = alpha * typical_act(fa) * dA + (1 - alpha) * typical_act(fb) * dB
    return B_DEC.float() + s * mix


def make_fixed(fa, fb, alpha, s=None):
    """Unit directions, renormalised - exactly Llama's construction, plus b_dec."""
    dA, dB = W_DEC[fa].float(), W_DEC[fb].float()
    dA = dA / dA.norm(); dB = dB / dB.norm()
    mix = alpha * dA + (1 - alpha) * dB
    mix = mix / mix.norm().clamp_min(1e-12)
    if s is None:                      # match the norm the actor expects (80000)
        s = 80000.0 - float(B_DEC.norm())
    return B_DEC.float() + s * mix


if __name__ == "__main__":
    print("typical_act values (the source of the distortion):")
    for n, fi in FEATS_FULL.items():
        print(f"   {n:16} {typical_act(fi):9.1f}")

    for tag, fn in [("BROKEN (typical_act scaling)", make_broken),
                    ("FIXED  (unit directions)", make_fixed)]:
        print(f"\n{'='*74}\n{tag}\n{'='*74}")
        print(f"{'pair':30}{'alpha':>6}{'act_mund':>10}{'act_conc':>10}{'ratio':>8}{'||v||':>10}")
        bad = 0
        for m, c in PAIRS8:
            for al in [0.5, 0.75, 0.9]:
                v = fn(FEATS_FULL[m], FEATS_FULL[c], al)
                a = sae_encode(v)
                gA, gB = float(a[FEATS_FULL[m]]), float(a[FEATS_FULL[c]])
                r = gB / gA if gA > 0 else float("inf")
                flag = ""
                if al >= 0.75 and (r > 0.5 or gA == 0):
                    flag = "  <-- minority not actually minor"; bad += 1
                print(f"{m+' x '+c:30}{al:>6}{gA:>10.0f}{gB:>10.0f}{r:>8.2f}{float(v.norm()):>10.0f}{flag}")
        print(f"\n   cells where the 'minority' concept is not actually minor: {bad}")
