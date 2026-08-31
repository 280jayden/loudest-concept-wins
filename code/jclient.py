"""
Minimal Jupyter kernel client - drives the RunPod kernel over the REST + WebSocket
API. Fallback for when the jupyter MCP server isn't loaded in the session.

usage:  python jclient.py "print(1+1)"
        python jclient.py --file some_script.py
        python jclient.py --list
"""
import os
import json, sys, uuid, time, argparse
import requests, websocket

BASE  = os.environ["JUPYTER_URL"]      # e.g. https://<pod>-8888.proxy.runpod.net
TOKEN = os.environ["JUPYTER_TOKEN"]
WS    = BASE.replace("https://", "wss://")
H     = {"Authorization": f"token {TOKEN}"}


def list_kernels():
    r = requests.get(f"{BASE}/api/kernels", headers=H, timeout=30)
    r.raise_for_status()
    return r.json()


def pick_kernel(prefer_busy=False):
    """Reuse the kernel that already holds the model in memory (the biggest win)."""
    ks = list_kernels()
    if not ks:
        r = requests.post(f"{BASE}/api/kernels", headers=H, timeout=60)
        r.raise_for_status()
        return r.json()["id"]
    ks.sort(key=lambda k: k.get("last_activity", ""), reverse=True)
    return ks[0]["id"]


def execute(code, kernel_id=None, timeout=600, quiet=False):
    kernel_id = kernel_id or pick_kernel()
    url = f"{WS}/api/kernels/{kernel_id}/channels?token={TOKEN}"
    ws = websocket.create_connection(url, timeout=timeout)
    msg_id = uuid.uuid4().hex
    ws.send(json.dumps({
        "header": {"msg_id": msg_id, "username": "c", "session": uuid.uuid4().hex,
                   "msg_type": "execute_request", "version": "5.3"},
        "parent_header": {}, "metadata": {},
        "content": {"code": code, "silent": False, "store_history": True,
                    "user_expressions": {}, "allow_stdin": False, "stop_on_error": True},
        "channel": "shell",
    }))
    out, t0 = [], time.time()
    try:
        while time.time() - t0 < timeout:
            ws.settimeout(max(1, timeout - (time.time() - t0)))
            try:
                m = json.loads(ws.recv())
            except websocket.WebSocketTimeoutException:
                break
            if m.get("parent_header", {}).get("msg_id") != msg_id:
                continue
            t, c = m["msg_type"], m.get("content", {})
            if t == "stream":
                out.append(c.get("text", ""))
            elif t in ("execute_result", "display_data"):
                out.append(c.get("data", {}).get("text/plain", ""))
            elif t == "error":
                out.append("\n".join(c.get("traceback", [])))
            elif t == "status" and c.get("execution_state") == "idle":
                break
    finally:
        ws.close()
    text = "".join(out)
    if not quiet:
        print(text)
    return text


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("code", nargs="?")
    ap.add_argument("--file")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--kernel")
    ap.add_argument("--timeout", type=int, default=600)
    a = ap.parse_args()
    if a.list:
        for k in list_kernels():
            print(f"{k['id']}  {k['name']}  {k['execution_state']}  {k.get('last_activity','')}")
        sys.exit()
    src = open(a.file, encoding="utf-8").read() if a.file else a.code
    execute(src, kernel_id=a.kernel, timeout=a.timeout)
