"""
IS THE OMITTED CONCEPT PRESENT IN THE MODEL WHILE IT WRITES THE DESCRIPTION?

THE CLAIM THIS TESTS
Everything so far shows the description omits concept B. That is consistent with two
very different stories:
    (a) B never reaches the interpretation forward pass  -> nothing to report
    (b) B is active in the model's own states while it writes a description that
        does not mention it                              -> the loss is in the REPORT
Only (b) supports "the information is available and the interpretation channel discards
it", which is the version that matters for monitoring.

METHOD (paired - both measures on the SAME generation)
  1. inject the mixture, generate a 30-token description
  2. re-run a forward pass over [injected prompt + the tokens it just generated],
     capture layer 19, SAE-encode ONLY the generated positions
        -> internal_B: did latent B fire while the model was composing its answer?
  3. score that same description the usual behavioural way
        -> described_B: does the description actually convey B?
  The cell of interest is internal_B AND NOT described_B.

TWO CONTROLS, BOTH NECESSARY
  pure-A control: same procedure with a pure-A vector, where B is genuinely absent.
     Gives the false-positive rate for internal_B. Without it, "B fired somewhere in
     30 tokens" is not evidence of anything.
  A-sanity: internal_A should be high everywhere; if it is not, injection is broken
     (this is exactly how the dead-injection run in targeted_probe.pkl was caught).

REPORT MARGIN, NOT JUST "FIRED"
A latent firing at all is a weak fact - the composed-vector analysis showed concept B
can rank 2nd of 65,536 while sitting 1.01x above unrelated background latents. So we
also record the strongest NON-target activation and report B/background. A result only
counts if B clears the background and the pure-A floor.

HONEST LIMIT ON INTERPRETATION
Generated positions attend back to the injection position, so B being readable there
does not prove the model has "internalised" B independently of the injected vector.
The defensible claim is narrower and still sufficient: the concept is present and
linearly readable in the residual stream at the positions where the model is composing
its answer, and the answer does not mention it. Do not write "the model knows and hides
it" - write "the information is available at the point of reporting and is not reported".

Assumes 00_reload.py globals: hf, tok, sae, adapter, compose, score_label,
TEMPLATE_EMBEDS, INJECT_POS, LAYER, THR, DEV.
"""
import torch, pickle, os, json, hashlib

OUT = "/workspace/internal_vs_reported.pkl"
SCALES_I = [0.5, 0.8, 1.3, 2.1, 3.4, 5.5]   # all six, matching exp3 exactly
ALPHAS = [0.5, 0.75, 0.9]
SHARE = {0.5: "50%", 0.75: "25%", 0.9: "10%"}
N_DESC = 3                     # 12 pairs x 6 scales x 3 = 216 PER RATIO, matching exp3


def _seed(*parts):
    return int(hashlib.md5("|".join(map(str, parts)).encode()).hexdigest()[:6], 16)


@torch.no_grad()
def generate_and_capture(vec, scale, ia, ib, max_new=30, seed=None):
    """Generate one description and read the residual stream at the generated positions."""
    if seed is not None:
        torch.manual_seed(seed)
    v = vec.to(DEV).float().unsqueeze(0)
    v = v / v.norm(dim=-1, keepdim=True).clamp_min(1e-8) * scale
    soft = adapter.transform(v, normalize_input=False).to(TEMPLATE_EMBEDS.dtype)
    emb = TEMPLATE_EMBEDS.clone()
    for p in INJECT_POS:
        emb[:, p, :] = soft
    attn = torch.ones(emb.shape[:2], dtype=torch.long, device=DEV)
    gen = hf.generate(inputs_embeds=emb, attention_mask=attn, max_new_tokens=max_new,
                      do_sample=True, temperature=0.7, top_p=0.9,
                      pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
    text = tok.decode(gen[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
    text = text.rsplit('"', 1)[0] if '"' in text else text

    # forward over prompt + what it generated, so the captured states are the ones the
    # model actually had while producing this text
    gen_emb = hf.model.embed_tokens(gen)
    full = torch.cat([emb, gen_emb.to(emb.dtype)], dim=1)
    attn2 = torch.ones(full.shape[:2], dtype=torch.long, device=DEV)
    hs = hf(inputs_embeds=full, attention_mask=attn2,
            output_hidden_states=True).hidden_states[LAYER + 1]
    gen_states = hs[0, emb.shape[1]:, :]                     # generated positions only
    acts = sae.encode(gen_states.to(sae.W_enc.device, sae.W_enc.dtype).unsqueeze(0))[0]

    a_col, b_col = acts[:, ia].float(), acts[:, ib].float()
    masked = acts.float().clone()
    masked[:, ia] = -1
    masked[:, ib] = -1
    return {
        "text": text.strip(),
        "internal_A": float(a_col.max()), "internal_B": float(b_col.max()),
        "internal_bg": float(masked.max()),
        "B_positions": int((b_col > 0).sum()), "n_positions": int(b_col.numel()),
    }


def run(pairs, res):
    for nm, ia, ib in pairs:
        for al in ALPHAS:
            for sc in SCALES_I:
                key = (nm, al, sc)
                if key in res:
                    continue
                v = compose(ia, ib, al)
                rows = []
                for d in range(N_DESC):
                    r = generate_and_capture(v, sc, ia, ib, seed=_seed(nm, al, sc, d))
                    s = score_label(r["text"], [ia, ib], n=10)
                    r["described_A"], r["described_B"] = s[ia], s[ib]
                    rows.append(r)
                res[key] = rows
                pickle.dump(res, open(OUT, "wb"))
                print(f"  {nm[:38]:38} {SHARE[al]:>4} s={sc}  "
                      f"internalB {sum(x['internal_B'] > 0 for x in rows)}/{len(rows)}  "
                      f"describedB {sum(x['described_B'] >= THR for x in rows)}/{len(rows)}")
    return res


def run_control(pairs, res):
    """pure A: B genuinely absent -> internal_B here is the false-positive floor."""
    print("\n  control (pure A)...")
    for nm, ia, ib in pairs:
        for sc in SCALES_I:
            key = ("CTRL", nm, sc)
            if key in res:
                continue
            d = sae.W_dec[ia].detach()
            v = d / d.norm()
            rows = [generate_and_capture(v, sc, ia, ib, seed=_seed("ctrl", nm, sc, i))
                    for i in range(N_DESC)]
            res[key] = rows
            pickle.dump(res, open(OUT, "wb"))
    return res


def report(res):
    live = {k: v for k, v in res.items() if k[0] != "CTRL"}
    print("\n=== internal presence vs what was reported ===")
    print(f"{'share':>7}{'A internal':>13}{'B internal':>13}{'B described':>14}"
          f"{'internal & silent':>19}")
    for al in ALPHAS:
        rows = [r for k, rs in live.items() if k[1] == al for r in rs]
        if not rows:
            continue
        n = len(rows)
        ai = sum(r["internal_A"] > 0 for r in rows)
        bi = sum(r["internal_B"] > 0 for r in rows)
        bd = sum(r["described_B"] >= THR for r in rows)
        gap = sum((r["internal_B"] > 0) and (r["described_B"] < THR) for r in rows)
        print(f"{SHARE[al]:>7}{ai:>8}/{n:<4}{bi:>8}/{n:<4}{bd:>9}/{n:<4}"
              f"{gap:>13}/{n:<4} = {gap/n*100:.0f}%")

    ctrl = [r for k, rs in res.items() if k[0] == "CTRL" for r in rs]
    if ctrl:
        cb = sum(r["internal_B"] > 0 for r in ctrl)
        print(f"\n  CONTROL (pure A, B absent): B internal {cb}/{len(ctrl)} = "
              f"{cb/len(ctrl)*100:.0f}%   <- floor; the rows above must clear this")
        cm = sorted(r["internal_B"] / r["internal_bg"] for r in ctrl if r["internal_bg"] > 0)
        if cm:
            print(f"  CONTROL B/background (median): {cm[len(cm)//2]:.2f}")

    print("\n  margin: B vs strongest non-target latent (median)")
    for al in ALPHAS:
        rows = [r for k, rs in live.items() if k[1] == al for r in rs]
        m = sorted(r["internal_B"] / r["internal_bg"] for r in rows if r["internal_bg"] > 0)
        if m:
            print(f"    {SHARE[al]:>4}  B/background = {m[len(m)//2]:.2f}")

    json.dump({"n_generations": sum(len(v) for v in live.values()),
               "control_n": len(ctrl)},
              open("/workspace/internal_vs_reported.json", "w"), indent=1)


if __name__ == "__main__":
    meta = pickle.load(open("/workspace/safe_meta.pkl", "rb"))
    V = meta["VALID2"]
    pairs = [p for p in meta["SAFE_PAIRS"]
             if all({a: b for a, _, b, _ in V[p[0]]}.get(al) for al in ALPHAS)]
    print(f"pairs: {len(pairs)}  cells: {len(pairs)*len(ALPHAS)*len(SCALES_I)}  "
          f"generations: {len(pairs)*len(ALPHAS)*len(SCALES_I)*N_DESC}")
    res = pickle.load(open(OUT, "rb")) if os.path.exists(OUT) else {}
    res = run(pairs, res)
    res = run_control(pairs, res)
    report(res)
