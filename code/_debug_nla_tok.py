import os, json, torch
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download

AV_DIR = snapshot_download("kitft/nla-qwen2.5-7b-L20-av")
print("files:", sorted(os.listdir(AV_DIR)))

CH = "㈎"   # the injection char
WANT = 149705

for kw in ({"trust_remote_code": True}, {}):
    print(f"\n--- AutoTokenizer(**{kw}) ---")
    try:
        t = AutoTokenizer.from_pretrained(AV_DIR, **kw)
    except Exception as e:
        print("  load failed:", e); continue
    print("  vocab_size:", t.vocab_size, "| len(tokenizer):", len(t))
    enc = t.encode(CH, add_special_tokens=False)
    print(f"  encode({CH!r}) -> {enc}")
    print("  decode(149705) ->", repr(t.decode([WANT])))
    added = getattr(t, "added_tokens_decoder", {})
    hit = {k: str(v) for k, v in added.items() if k >= 149000}
    print("  added tokens >=149000:", list(hit.items())[:6])

# what does added_tokens.json actually say?
p = os.path.join(AV_DIR, "added_tokens.json")
if os.path.exists(p):
    d = json.load(open(p, encoding="utf-8"))
    inv = {v: k for k, v in d.items()}
    print("\nadded_tokens.json entries near 149705:",
          {k: v for k, v in d.items() if 149700 <= v <= 149710})
    print("  id 149705 ->", repr(inv.get(149705)))
else:
    print("\nNO added_tokens.json in AV dir")
