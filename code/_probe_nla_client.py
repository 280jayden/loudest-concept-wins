import re
txt = open("/workspace/natural_language_autoencoders/nla_inference.py", encoding="utf-8").read()

# does the actor require sglang, or is there a local HF path?
print("=== backend mentions ===")
for i, line in enumerate(txt.split("\n"), 1):
    low = line.lower()
    if any(k in low for k in ["sglang", "vllm", "openai", "requests.post", "local", "backend",
                              "automodel", "from_pretrained"]):
        print(f"{i:5}: {line.strip()[:120]}")

print("\n=== class NLAClient signature + init ===")
m = re.search(r"class NLAClient.*?(?=\nclass |\Z)", txt, re.S)
if m:
    body = m.group(0)
    print(body[:2600])
