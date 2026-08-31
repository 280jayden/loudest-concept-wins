import subprocess, re, os

SRC = "/workspace/natural_language_autoencoders/nla_inference.py"
print("=== file size ===")
print(os.path.getsize(SRC), "bytes")

txt = open(SRC, encoding="utf-8").read()

print("\n=== function/class definitions ===")
for m in re.finditer(r"^(def |class |async def ).*", txt, re.M):
    print(" ", m.group(0)[:110])

print("\n=== lines mentioning key mechanics ===")
KEYS = ["injection_scale", "hidden_states", "extraction_layer", "reconstruct",
        "summary", "explanation", "output_hidden_states", "mse", "cosine",
        "injection_token", "norm"]
for i, line in enumerate(txt.split("\n"), 1):
    low = line.lower()
    if any(k in low for k in KEYS):
        print(f"{i:5}: {line.strip()[:130]}")
