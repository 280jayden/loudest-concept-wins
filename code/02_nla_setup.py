"""
NLA setup: Qwen2.5-7B-Instruct, layer 20.

NOTE ON THEIR CODE
`nla_inference.load_nla_config` breaks under transformers>=5: it does
`enumerate(apply_chat_template(..., tokenize=True))`, which used to yield token
ids but now yields BatchEncoding KEYS ('input_ids','attention_mask'), so the
injection-token assert fires with "appears 0x". The checkpoints are fine - the
tokenizer encodes the injection char correctly. So we read nla_meta.yaml
ourselves and do the injection with plain HF (same pattern we use for SelfIE).

BUILT-IN CORRECTNESS CHECK
Their docs: cos ~0.9 = good decode for clean positions, 0.5 mediocre, 0.0
orthogonal; and a wrong injection norm sends the vector OOD -> CJK gibberish.
So we round-trip a REAL activation first and only proceed if cos is sane.
"""
import sys, os, yaml, torch
sys.path.insert(0, "/workspace/natural_language_autoencoders")
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

AV_REPO, AR_REPO = "kitft/nla-qwen2.5-7b-L20-av", "kitft/nla-qwen2.5-7b-L20-ar"
BASE, NLA_LAYER = "Qwen/Qwen2.5-7B-Instruct", 20

AV_DIR, AR_DIR = snapshot_download(AV_REPO), snapshot_download(AR_REPO)
META = yaml.safe_load(open(os.path.join(AV_DIR, "nla_meta.yaml"), encoding="utf-8"))
AR_META = yaml.safe_load(open(os.path.join(AR_DIR, "nla_meta.yaml"), encoding="utf-8"))

D_MODEL   = META["d_model"]
INJ_SCALE = float(META["extraction"]["injection_scale"])
INJ_ID    = int(META["tokens"]["injection_token_id"])
INJ_CHAR  = META["tokens"]["injection_char"]
AV_TMPL   = META["prompt_templates"]["av"]
AR_TMPL   = META["prompt_templates"]["ar"]
MSE_SCALE = float(AR_META["extraction"]["mse_scale"])
print(f"d_model={D_MODEL} inj_scale={INJ_SCALE} inj_id={INJ_ID} mse_scale={MSE_SCALE:.2f}")

qtok = AutoTokenizer.from_pretrained(AV_DIR)
av = AutoModelForCausalLM.from_pretrained(AV_DIR, dtype=torch.bfloat16, device_map="cuda").eval()

# /workspace volume is quota-capped (~50GB, already ~43GB used by Llama+AV+AR+SAE).
# Base Qwen goes on the CONTAINER disk instead - ephemeral, but it is only a
# re-downloadable base model. All RESULTS stay on the persistent volume.
TMP_CACHE = "/root/qwen_cache"
os.makedirs(TMP_CACHE, exist_ok=True)
qbase_tok = AutoTokenizer.from_pretrained(BASE, cache_dir=TMP_CACHE)
qbase = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16,
                                             device_map="cuda", cache_dir=TMP_CACHE).eval()

# AR: truncated backbone (K=layer+1 blocks, final LN -> Identity) + recon head
ar_tok = AutoTokenizer.from_pretrained(AR_DIR)
ar = AutoModelForCausalLM.from_pretrained(AR_DIR, dtype=torch.bfloat16, device_map="cuda").eval()
print("models loaded")


@torch.no_grad()
def qwen_resid(text, layer=NLA_LAYER):
    ids = qbase_tok(text, return_tensors="pt").to(qbase.device)
    return qbase(**ids, output_hidden_states=True).hidden_states[layer + 1][0].float().cpu()


@torch.no_grad()
def av_verbalize(vec, max_new=140, seed=None, scale=None):
    """activation -> explanation text (local HF reimplementation of their actor)"""
    if seed is not None:
        torch.manual_seed(seed)
    scale = float(INJ_SCALE if scale is None else scale)
    content = AV_TMPL.format(injection_char=INJ_CHAR)
    enc = qtok.apply_chat_template([{"role": "user", "content": content}],
                                   tokenize=True, add_generation_prompt=True,
                                   return_tensors="pt", return_dict=True)
    ids = enc["input_ids"].to(av.device)
    pos = (ids[0] == INJ_ID).nonzero().flatten().tolist()
    assert len(pos) == 1, f"injection token found {len(pos)}x (expected 1)"
    emb = av.get_input_embeddings()(ids).clone()
    v = vec.to(av.device).float()
    v = v / v.norm().clamp_min(1e-12) * scale
    emb[0, pos[0], :] = v.to(emb.dtype)
    attn = torch.ones(emb.shape[:2], dtype=torch.long, device=av.device)
    out = av.generate(inputs_embeds=emb, attention_mask=attn, max_new_tokens=max_new,
                      do_sample=True, temperature=0.7, top_p=0.9,
                      pad_token_id=qtok.pad_token_id or qtok.eos_token_id)
    txt = qtok.decode(out[0], skip_special_tokens=True)
    if "<explanation>" in txt:
        txt = txt.split("<explanation>")[-1].split("</explanation>")[0]
    return txt.strip()


@torch.no_grad()
def ar_reconstruct(explanation):
    """explanation text -> predicted activation (AR hidden state at final position)"""
    prompt = AR_TMPL.format(explanation=explanation)
    ids = ar_tok(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].to(ar.device)
    hs = ar(input_ids=ids, output_hidden_states=True).hidden_states
    return hs[-1][0, -1].float().cpu()


def cos(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    return float(a @ b / (a.norm() * b.norm()).clamp_min(1e-12))


if __name__ == "__main__":
    TEXT = ("The FDA requires that all packaged foods list their ingredients in "
            "descending order by weight, and allergens must be declared clearly.")
    h = qwen_resid(TEXT)
    v_real = h[len(h) // 2]
    print(f"\nreal activation norm={v_real.norm():.1f} (actor trained at {INJ_SCALE})")
    for s in range(2):
        e = av_verbalize(v_real, seed=s)
        r = ar_reconstruct(e)
        print(f"\n[seed {s}] cos={cos(r, v_real):.3f}")
        print("  EXPL:", repr(e[:220]))
    print("\nreference: ~0.9 good | 0.5 mediocre | 0.0 orthogonal")
    print("CJK gibberish in EXPL => injection norm wrong / OOD")
