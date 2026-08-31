"""
Behavioural metric for Gemma, matching the Llama protocol.

LLAMA (AE Studio's generation scoring):
    description -> 10 synthetic conversations -> forward pass -> SAE encode -> feature fires?
    Conversations are needed because a LABEL ("Baking-related activities") does not
    reliably activate the feature it names - the feature fires on INSTANCES.

NLA explanations are different: they already quote instances, e.g.
    "The sentence 'Air frying is a cooking method that focuses on heat...' signals..."
so we may be able to score the explanation text DIRECTLY. Two candidate metrics:

    (A) direct   : explanation text        -> base Gemma -> SAE encode
    (B) generated: explanation -> conversations -> base Gemma -> SAE encode   [Llama-style]

VALIDATE BEFORE USING (this is the step that was missing):
  sensitivity - on Gate-1 explanations of PURE features, does the right feature fire?
  specificity - do OTHER features' explanations spuriously fire it?
Report both, exactly as we did for Llama (87% / 0%).
"""
import torch, pickle, json
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download

BASE_DIR = snapshot_download("google/gemma-3-12b-it")
gb_tok = AutoTokenizer.from_pretrained(BASE_DIR)
gbase = AutoModelForCausalLM.from_pretrained(BASE_DIR, dtype=torch.bfloat16,
                                             device_map="cuda").eval()
G_LAYER = 32
print("base gemma loaded")


@torch.no_grad()
def gemma_sae_fires(text, latents, max_tokens=400):
    """Run text through BASE gemma, SAE-encode layer 32, return max activation per latent."""
    ids = gb_tok(text, return_tensors="pt", truncation=True,
                 max_length=max_tokens)["input_ids"].to(gbase.device)
    hs = gbase(input_ids=ids, output_hidden_states=True).hidden_states[G_LAYER + 1][0]
    acts = sae_encode(hs.float().cpu())          # (seq, 16384)
    return {li: float(acts[1:, li].max()) for li in latents}   # skip BOS, as on Llama


CONV_SYS_G = ("You are a helpful AI assistant who generates EXTREMELY SHORT example "
              "conversations. The conversations are between a user and an assistant, "
              "and have the following format:\n[USER] I'm a user.\n[ASSISTANT] I'm the assistant.")
CONV_PROMPT_G = ("Produce a VERY SHORT conversation which exhibits '_'\n"
                 "Do not include any other text in your response. "
                 "Start immediately with the conversation.")


@torch.no_grad()
def gemma_generated_score(desc, latents, n=10):
    """Llama-style: description -> n conversations -> SAE encode each."""
    msgs = [{"role": "system", "content": CONV_SYS_G},
            {"role": "user", "content": CONV_PROMPT_G.replace("_", desc[:400])}]
    gb_tok.padding_side = "left"
    enc = gb_tok.apply_chat_template([msgs] * n, tokenize=True, add_generation_prompt=True,
                                     return_tensors="pt", padding=True,
                                     return_dict=True).to(gbase.device)
    out = gbase.generate(**enc, max_new_tokens=100, do_sample=True, temperature=0.7,
                         top_p=0.9, pad_token_id=gb_tok.pad_token_id or gb_tok.eos_token_id)
    hits = {li: 0 for li in latents}
    for g in out:
        t = gb_tok.decode(g[enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        if not t:
            continue
        f = gemma_sae_fires(t, latents)
        for li in latents:
            if f[li] > 0:
                hits[li] += 1
    return {li: hits[li] / n for li in latents}


if __name__ == "__main__":
    # ---- VALIDATION on pure features (we know the ground truth) ----
    S = 30.0
    names = list(FEATS_FULL)
    print("\n=== metric validation: explanation of a PURE feature ===")
    print(f"{'feature':16}{'direct: own':>13}{'direct: others max':>21}{'generated: own':>16}")
    sens_d = sens_g = spec_fail = n = 0
    for nm in names:
        fi = FEATS_FULL[nm]
        v = B_DEC.float() + S * typical_act(fi) * W_DEC[fi].float()
        e = g_verbalize(v, seed=4242)
        others = [FEATS_FULL[o] for o in names if o != nm]
        f_direct = gemma_sae_fires(e, [fi] + others)
        own_d = f_direct[fi]
        oth_max = max(f_direct[o] for o in others)
        gen = gemma_generated_score(e, [fi], n=10)[fi]
        n += 1
        sens_d += (own_d > 0)
        sens_g += (gen >= 0.3)
        spec_fail += (oth_max > own_d)
        print(f"{nm:16}{own_d:>13.1f}{oth_max:>21.1f}{gen:>16.1f}")
    print(f"\n  direct    - own feature fires for {sens_d}/{n} features")
    print(f"  generated - own feature >=3/10 for {sens_g}/{n} features")
    print(f"  cases where ANOTHER feature outfired the target: {spec_fail}/{n}")
