#!/usr/bin/env python3
"""TTFT and true decode rate for every family this lane built, from the server's OWN timings.

Every tok/s figure I have quoted so far was wall-clock around the HTTP request, so it folded prefill into decode. The
/v1/completions response carries `timings: {prompt_ms, predicted_ms}`; TTFT is `prompt_ms` and the decode rate is
`completion_tokens / (predicted_ms/1000)`. One npu-hold ticket measures all three families back to back.

Runs INSIDE the ticket: serve -> 2 requests -> kill, for each family, sequentially, so the device never holds two servers.
"""
import json, os, signal, subprocess, sys, time

BIN = os.path.expanduser("~/wt/zha-main/build-npu/1bit")
PORT = 18099
PROMPT = "The first person to walk on the Moon was"

FAMILIES = [
    ("2048x3072 (Llama-3.2-1B 1.7.1)", "llama32-1b-rai171-b8",  "kernels-2048-3072-b8"),
    ("2048x6144 (OLMo-1B)",           "olmo1b-b8-container",   "kernels-2048-6144-b8"),
    ("2048x2560 (Qwen2.5-3B)",        "qwen25-3b-b8-container", "kernels-2048-2560-b8"),
]

def wait_ready(proc, log):
    for _ in range(120):
        if proc.poll() is not None:
            return False
        r = subprocess.run(["curl", "-sf", "--max-time", "2", f"http://127.0.0.1:{PORT}/v1/health"],
                           capture_output=True)
        if r.returncode == 0:
            return True
        time.sleep(1)
    return False

print(f"{'family':32s} {'run':>3s} {'TTFT ms':>8s} {'decode tok/s':>12s} {'tokens':>6s}  text")
for name, container, kernels in FAMILIES:
    env = dict(os.environ,
               ONEBIT_NPU_KERNELS=os.path.expanduser(f"~/ogaload/oga13/{kernels}"))
    log = open(f"/home/bcloud/.cache/scratch-pi-rlk/timing-{kernels}.log", "w")
    proc = subprocess.Popen([BIN, "serve", "-m", os.path.expanduser(f"~/scratch-packs/{container}"),
                             "--device", "npu", "--port", str(PORT), "--host", "127.0.0.1",
                             "--ctx-size", "2048"], env=env, stdout=log, stderr=subprocess.STDOUT)
    try:
        if not wait_ready(proc, log):
            print(f"{name:32s} SERVE FAILED (see /home/bcloud/.cache/scratch-pi-rlk/timing-{kernels}.log)")
            continue
        for run in (1, 2):
            body = json.dumps({"prompt": PROMPT, "max_tokens": 32, "temperature": 0})
            r = subprocess.run(["curl", "-s", "--max-time", "300", f"http://127.0.0.1:{PORT}/v1/completions",
                                "-H", "Content-Type: application/json", "-d", body],
                               capture_output=True, text=True)
            try:
                d = json.loads(r.stdout)
            except Exception:
                print(f"{name:32s} {run:3d}  not JSON: {r.stdout[:120]}"); continue
            if "choices" not in d:
                print(f"{name:32s} {run:3d}  error: {json.dumps(d.get('error'))[:120]}"); continue
            t = d.get("timings") or {}
            ttft = t.get("prompt_ms")
            pred = t.get("predicted_ms")
            ct = (d.get("usage") or {}).get("completion_tokens") or 0
            rate = (ct / (pred / 1000.0)) if pred else None
            txt = (d["choices"][0].get("text") or "").strip().replace("\n", " ")[:58]
            print(f"{name:32s} {run:3d} "
                  f"{(ttft if ttft is not None else -1):8.1f} "
                  f"{(rate if rate else -1):12.2f} {ct:6d}  {txt}")
    finally:
        proc.send_signal(signal.SIGTERM)
        try: proc.wait(timeout=20)
        except subprocess.TimeoutExpired: proc.kill()
        log.close()
        time.sleep(3)
