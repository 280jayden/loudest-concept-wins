"""
Gemma NLA setup + gates.  gemma-3-12b-it, layer 32.
SAE: gemma-scope-2-12b-it resid_post layer 32 (16k, jump_relu)
NLA: kitft/nla-gemma3-12b-L32-{av,ar}

Same protocol as the Llama experiment, different verbalisation method:
  Gate 2  - decoder cosine between candidate features (are they distinct?)
  norms   - do real layer-32 activations match the checkpoint's injection scale?
  Gate 1  - can the NLA describe each PURE feature? (this is where Qwen died)
"""
import os, json, yaml, torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

AV_REPO, AR_REPO = "kitft/nla-gemma3-12b-L32-av", "kitft/nla-gemma3-12b-L32-ar"
SAE_REPO, SAE_PATH = "google/gemma-scope-2-12b-it", "resid_post_all/layer_32_width_16k_l0_big"
G_LAYER = 32

sae_dir = os.path.join(snapshot_download(SAE_REPO, allow_patterns=[f"{SAE_PATH}/*"]), SAE_PATH)
P = load_file(os.path.join(sae_dir, "params.safetensors"))
W_ENC, W_DEC = P["w_enc"], P["w_dec"]          # (3840,16384), (16384,3840)
B_ENC, B_DEC, THRESH = P["b_enc"], P["b_dec"], P["threshold"]

AV_DIR, AR_DIR = snapshot_download(AV_REPO), snapshot_download(AR_REPO)
GMETA = yaml.safe_load(open(os.path.join(AV_DIR, "nla_meta.yaml"), encoding="utf-8"))
G_INJ_SCALE = float(GMETA["extraction"]["injection_scale"])
G_INJ_ID    = int(GMETA["tokens"]["injection_token_id"])
G_INJ_CHAR  = GMETA["tokens"]["injection_char"]
G_AV_TMPL   = GMETA["prompt_templates"]["av"]
G_AR_TMPL   = GMETA["prompt_templates"]["ar"]
print(f"gemma NLA: d_model={GMETA['d_model']} inj_scale={G_INJ_SCALE} inj_id={G_INJ_ID}")

gav_tok = AutoTokenizer.from_pretrained(AV_DIR)
gav = AutoModelForCausalLM.from_pretrained(AV_DIR, dtype=torch.bfloat16, device_map="cuda").eval()
gar_tok = AutoTokenizer.from_pretrained(AR_DIR)
gar = AutoModelForCausalLM.from_pretrained(AR_DIR, dtype=torch.bfloat16, device_map="cuda").eval()
print("NLA models loaded")


def sae_encode(v):
    """jump_relu encode: relu(x@W_enc + b_enc) gated by threshold."""
    pre = v.float() @ W_ENC + B_ENC
    return torch.where(pre > THRESH, torch.relu(pre), torch.zeros_like(pre))


@torch.no_grad()
def g_verbalize(vec, max_new=140, seed=None, scale=None):
    if seed is not None:
        torch.manual_seed(seed)
    scale = float(G_INJ_SCALE if scale is None else scale)
    content = G_AV_TMPL.format(injection_char=G_INJ_CHAR)
    enc = gav_tok.apply_chat_template([{"role": "user", "content": content}],
                                      tokenize=True, add_generation_prompt=True,
                                      return_tensors="pt", return_dict=True)
    ids = enc["input_ids"].to(gav.device)
    pos = (ids[0] == G_INJ_ID).nonzero().flatten().tolist()
    assert len(pos) == 1, f"injection token x{len(pos)}"
    emb = gav.get_input_embeddings()(ids).clone()
    v = vec.to(gav.device).float()
    v = v / v.norm().clamp_min(1e-12) * scale
    emb[0, pos[0], :] = v.to(emb.dtype)
    attn = torch.ones(emb.shape[:2], dtype=torch.long, device=gav.device)
    out = gav.generate(inputs_embeds=emb, attention_mask=attn, max_new_tokens=max_new,
                       do_sample=True, temperature=0.7, top_p=0.9,
                       pad_token_id=gav_tok.pad_token_id or gav_tok.eos_token_id)
    t = gav_tok.decode(out[0], skip_special_tokens=True)
    if "<explanation>" in t:
        t = t.split("<explanation>")[-1].split("</explanation>")[0]
    return t.strip()


@torch.no_grad()
def g_reconstruct(expl):
    prompt = G_AR_TMPL.format(explanation=expl)
    ids = gar_tok(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].to(gar.device)
    return gar(input_ids=ids, output_hidden_states=True).hidden_states[-1][0, -1].float().cpu()


def gcos(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    return float(a @ b / (a.norm() * b.norm()).clamp_min(1e-12))


FEATS = {"cooking": 13974, "legal": 5174, "hate_speech": 11365,
         "hatred_discrim": 2026, "explosives": 7175, "drug_harm": 16183}

MENTION_G = {
 "cooking":        ["cook","recipe","bake","baking","food","oven","ingredient","culinary","cake"],
 "legal":          ["legal","law","court","statute","attorney","litigation","regulat","ethical"],
 "hate_speech":    ["hate","hatred","racis","discriminat","ethnic","slur","bigot","prejudice","offensive"],
 "hatred_discrim": ["hate","hatred","racis","discriminat","prejudice","stereotype","disparag"],
 "explosives":     ["explosive","bomb","weapon","detonat","incendiary"],
 "drug_harm":      ["drug","meth","overdose","addiction","narcotic","substance"],
}
def g_mentions(text, name):
    t = text.lower()
    return any(w in t for w in MENTION_G[name])


if __name__ == "__main__":
    # ---- Gate 2: are the candidate features distinct? ----
    print("\n=== GATE 2: decoder cosine between candidates ===")
    names = list(FEATS)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = W_DEC[FEATS[names[i]]], W_DEC[FEATS[names[j]]]
            print(f"  {names[i]:15} x {names[j]:15} cos={gcos(a,b):+.3f}")

    # ---- norm sanity: decoder rows vs the scale the actor expects ----
    print("\n=== decoder row norms (raw) ===")
    for n, fi in FEATS.items():
        print(f"  {n:15} ||w_dec||={W_DEC[fi].norm():.2f}")
    print(f"  actor expects injected norm = {G_INJ_SCALE}")

    # ---- Gate 1: can the NLA describe each PURE feature? ----
    print("\n=== GATE 1: NLA on pure features ===")
    gate1 = {}
    for n, fi in FEATS.items():
        hits = 0
        for s in range(3):
            e = g_verbalize(W_DEC[fi], seed=500 + s)
            if g_mentions(e, n):
                hits += 1
            if s == 0:
                print(f"  {n:15}: {e[:130]!r}")
        gate1[n] = hits
        print(f"  {n:15}: mentions concept {hits}/3")
    json.dump(gate1, open("/workspace/gemma_gate1.json", "w"), indent=1)
