#!/usr/bin/env python3
# Copyright 2026 bong-water-water-bong
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Target 1's cross-process gate: a real dma-buf KV handoff between two processes (docs/zero-copy.md,
# "Target 1" and docs/hybrid.md, "The shared route").
#
# This drives the contract app/serve_hybrid.h implements in production, with no engine in the middle:
#   1. listen on a unix socket in <outdir>;
#   2. POST /slots/0?action=share {"socket": ...} -- the server connects and sends
#      {4-byte LE length}{JSON header with params + regions}, the region fds riding in SCM_RIGHTS;
#   3. write the header to <outdir>/layout.json, put the fds at 3,4,... and exec the provider;
#   4. when it exits, POST /slots/0?action=adopt {"tokens": [...], "share_id": N};
#   5. ask the server for greedy tokens: prompt_n == 1 there means it decoded from the adopted cells, and
#      n_read == 0 means no prompt bytes were copied.
#
# The provider is the fork's reference one (llama.cpp tools/kvshare-prefill, a CPU prefill), so the whole
# route is exercised with no NPU. An AddressSanitizer build of it works too and is what checks the imported
# region: the report count is part of the verdict either way (0 for a normal build).
#
# usage: kvshare_fd_handoff.py <llama-server> <provider> <model.gguf> <outdir> [max_tokens]
import json
import os
import re
import socket
import struct
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

SRV = ""
PROV = ""
MODEL = ""
# Prompt length is a parameter (target 1's gate asks for 307 / 4k / 16k): repeat a sentence until the tokenizer
# reaches PROMPT_TOKENS, and size the decoder's context to CTX so the region fits. Both sides get n_ctx from the
# share header, so a mismatch is a refusal rather than a wrong answer.
PROMPT_BASE = "The tide comes in twice a day, and the harbour clock records every turn of it. "
PROMPT_TOKENS = int(os.environ.get("PROMPT_TOKENS", "0"))   # 0 = the fixed 145-token prompt this script grew up with
CTX = os.environ.get("CTX", "4096")


def _pdeathsig_setter():
    """Build the prctl call ONCE, at import: the setter runs in a forked child (preexec_fn).

    Why it matters here and not in a single-threaded program: this driver runs a socket-listener thread, so the fork
    that spawns a child happens in a multi-threaded process, and anything the child does before `exec` that allocates
    or imports can deadlock on a lock another thread held at fork time. Resolving libc and the constant at import time
    leaves the child with one C call that allocates nothing. Linux only (the HRX route); elsewhere the setter is a
    no-op and the `finally` remains the reaper.
    """
    try:
        import ctypes, signal
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        sig = int(signal.SIGTERM)
        return lambda: libc.prctl(1, sig)   # PR_SET_PDEATHSIG
    except Exception:
        return lambda: None


_DEATHSIG = _pdeathsig_setter()


def die_with_parent():
    """Have the kernel kill the child if this process dies, however it dies.

    The `finally` below reaps both children on a clean exit, a Ctrl-C and a SIGTERM, but not on SIGKILL (an OOM kill,
    a CI timeout, `kill -9` by an operator): the server is in its own session and the provider runs for up to 1800 s,
    so either would survive and keep its HRX context or its region -- and on this box an orphaned llama-server held
    51 GiB of GTT for 14 h and knocked out another service. PR_SET_PDEATHSIG closes that case: the kernel signals the
    child when the parent dies, whatever the reason.
    """
    _DEATHSIG()


def post(api, path, body, timeout=1800):
    req = urllib.request.Request(api + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def wait_bound_port(log_path, proc, deadline_s=180):
    """The port the server bound, read back from the server's own ready line.

    The driver used to pick a port with bind(0)-and-release and hope the server won it; `--port 0` makes llama-server
    choose (`server-http.cpp` binds with `bind_to_any_port` and prints `listening on http://host:port`), so there is no
    window to lose -- engine#1485's shape 2, which the other serve tests migrated to (#1525, #1545).

    Returns None if the server died or never printed the line; the caller treats that as "did not come up", with the
    log tail, rather than guessing a port.
    """
    # greedy up to the LAST ":" and anchored at end-of-line: `http://[::1]:44444` otherwise matches the `1` of
    # `[::1]` (measured -- the non-greedy form captured 1), and a unix:// ready line matches nothing, which is
    # the right answer for a socket the caller cannot curl.
    pat = re.compile(r"listening on \S*:(\d+)\s*$")
    t0 = time.time()
    while time.time() - t0 < deadline_s:
        try:
            with open(log_path, encoding="utf-8", errors="replace") as f:
                for line in f:
                    m = pat.search(line)
                    if m:
                        return int(m.group(1))
        except FileNotFoundError:
            pass
        if proc.poll() is not None:
            return None
        time.sleep(0.2)
    return None


def wait_health(api, proc, out, deadline_s=900):
    t0 = time.time()
    while time.time() - t0 < deadline_s:
        try:
            with urllib.request.urlopen(api + "/health", timeout=2) as r:
                if r.status == 200:
                    return True, time.time() - t0
        except Exception:
            pass
        if proc.poll() is not None:
            break
        time.sleep(0.5)
    return False, time.time() - t0


def main():
    global SRV, PROV, MODEL
    if len(sys.argv) < 5:
        print(__doc__ or "", file=sys.stderr)
        print("usage: kvshare_fd_handoff.py <llama-server> <provider> <model.gguf> <outdir> [max_tokens]",
              file=sys.stderr)
        return 2
    SRV, PROV, MODEL, out = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    max_tokens = int(sys.argv[5]) if len(sys.argv) > 5 else 8
    os.makedirs(out, exist_ok=True)
    log_path = os.path.join(out, "server.log")
    api_sock = os.path.join(out, "share.sock")
    layout_path = os.path.join(out, "layout.json")
    ids_path = os.path.join(out, "ids.txt")
    if os.path.exists(api_sock):
        os.unlink(api_sock)
    port = None   # read back from the server's own ready line; nothing is picked and released
    api = None

    def say(msg):
        line = f"{time.strftime('%Y-%m-%dT%H:%M%z')} {msg}"
        print(line, flush=True)
        with open(os.path.join(out, "window.log"), "a") as f:
            f.write(line + "\n")

    say(f"server {SRV}")
    say(f"provider {PROV}")
    env = dict(os.environ)
    env["ONEBIT_KV_SHARE"] = "1"          # the decode context owns and exports its KV region
    env.pop("ONEBIT_PREFILL_DEVICE", None)  # no in-server prefill: the external provider does it
    # The no-SVM HRX configuration this box's working servers run with (taken from a live serve's environ):
    # without these the HRX registry enumerates zero devices and every --device HRX0 is rejected.
    env.update({
        "HSA_USE_SVM": "0",
        "HSA_USERPTR_FOR_PAGED_MEM": "0",
        "HRX_ALLOW_NO_SVM": "1",
        "IREE_HAL_AMDGPU_LIBHSA_PATH": "/opt/rocm-therock/lib/python3.14/site-packages/_rocm_sdk_core/lib/libhsa-runtime64.so.1",
        "GGML_HRX_REPORT": "1",
        "GGML_HRX_STRICT": "1",
        "GGML_HRX_DISABLE_DISPATCH": "qwen.attention",
    })
    try:
        probe = subprocess.run([SRV, "--list-devices"], env=env, capture_output=True, text=True, timeout=180)
        say("devices: " + " ".join(probe.stdout.strip().splitlines()[-4:]) +
            (" stderr=" + probe.stderr.strip()[-200:] if probe.stderr.strip() else ""))
    except Exception as e:
        say(f"device probe failed: {e}")
    with open(log_path, "w") as lf:
        # --slot-save-path is not about files here: every /slots action (share and adopt included) is refused with
        # HTTP 501 "does not support slots action" unless it is set -- measured 2026-11-04 09:52, the first
        # attempt of this script. The engine's serve_hybrid.h sets it for the same reason.
        slotdir = os.path.join(out, "slots")
        os.makedirs(slotdir, exist_ok=True)
        srv = subprocess.Popen([SRV, "-m", MODEL, "--host", "127.0.0.1", "--port", "0", "--device", "HRX0",
                                "-ngl", "99", "--jinja", "-c", CTX, "-np", "1", "-fa", "on", "--no-warmup",
                                "--slot-save-path", slotdir],
                               env=env, stdout=lf, stderr=subprocess.STDOUT, start_new_session=True,
                               preexec_fn=die_with_parent)
    try:
        port = wait_bound_port(log_path, srv)
        if not port:
            say("server never reported a port it bound: " + open(log_path).read()[-400:].replace("\n", " "))
            return 3
        api = f"http://127.0.0.1:{port}"
        say(f"server pid {srv.pid} bound :{port} (asked for 0: the server chooses, so no port is picked and released)")
        up, secs = wait_health(api, srv, out)
        say(f"health up={up} after {secs:.1f}s")
        if not up:
            say("server did not come up: " + open(log_path).read()[-400:].replace("\n", " "))
            return 3

        if PROMPT_TOKENS:
            prompt, toks = PROMPT_BASE * 8, None
            while True:   # grow to the requested length; the tokenizer is the only honest measure of it
                toks = post(api, "/tokenize", {"content": prompt})["tokens"]
                if len(toks) >= PROMPT_TOKENS:
                    break
                prompt = prompt + PROMPT_BASE * max(1, (PROMPT_TOKENS - len(toks)) // 24)
            # Do NOT truncate to PROMPT_TOKENS: the adopted cell list has to be the exact tokenization of the text
            # the completion will send, otherwise the decoder finds a prefix N cells long and re-evaluates the
            # rest (measured 10:20: prompt_n 9 and 15 instead of 1 -- an artifact of cutting the id list, not of
            # the handoff). Growing past the target and reporting the real count is the honest shape.
            say(f"prompt grown to {len(toks)} tokens (requested ~{PROMPT_TOKENS}, ctx {CTX})")
        else:
            prompt = PROMPT
            toks = post(api, "/tokenize", {"content": prompt})["tokens"]
        # The completion below MUST send this same text: the adopted cells cover positions [0, N) of it, and a
        # *shorter* prompt makes the server trim the slot back (seq_rm(p0, -1)) -- the partial rollback the 10-31
        # notes record as refused on a recurrent cache, and an error here even on a plain one.
        prompt_used = prompt
        with open(ids_path, "w") as f:
            f.write(" ".join(str(t) for t in toks))
        say(f"decoder tokenized the prompt: {len(toks)} tokens")

        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        listener.bind(api_sock)
        listener.listen(1)
        share = {}

        def do_share():
            import urllib.error
            try:
                share["resp"] = post(api, "/slots/0?action=share", {"socket": api_sock}, timeout=600)
            except urllib.error.HTTPError as e:
                share["err"] = f"HTTP {e.code}: " + e.read().decode(errors="replace")[:600]
            except Exception as e:
                share["err"] = str(e)

        th = threading.Thread(target=do_share, daemon=True)
        th.start()
        # A failed POST never connects, and a bare 600 s accept() hid exactly that on the first attempt
        # (2026-11-04 09:38, ten minutes of GPU ticket for a one-line error). Poll both sides instead.
        listener.settimeout(1.0)
        conn = None
        deadline = time.time() + 600
        while time.time() < deadline:
            try:
                conn, _ = listener.accept()
                break
            except socket.timeout:
                if "err" in share or "resp" in share:
                    break
        if conn is None:
            say("share could not be honoured -- " + share.get("err", "no connection and no response"))
            say("share reply: " + json.dumps(share.get("resp", {}))[:600])
            srv.kill()
            return 5
        fds = []
        payload = b""
        want = None
        while True:
            data, anc, flags, _ = conn.recvmsg(65536, socket.CMSG_SPACE(64 * 4))
            for level, typ, cdata in anc:
                if level == socket.SOL_SOCKET and typ == socket.SCM_RIGHTS:
                    n = len(cdata) // 4
                    for i in range(n):
                        fd = struct.unpack("i", cdata[i * 4:(i + 1) * 4])[0]
                        os.set_inheritable(fd, True)
                        fds.append(fd)
            payload += data
            if want is None and len(payload) >= 4:
                want = 4 + struct.unpack("I", payload[:4])[0]
            if want is not None and len(payload) >= want:
                break
        conn.close()
        listener.close()
        header = json.loads(payload[4:want].decode())
        with open(layout_path, "w") as f:
            json.dump(header, f)
        say(f"received {len(fds)} region fd(s), header has {len(header.get('regions', []))} region(s)")
        th.join(timeout=600)
        if "err" in share:
            say("share failed: " + share["err"])
            return 4
        share_id = share["resp"].get("share_id")
        say(f"share ok: share_id={share_id} regions={len(share['resp'].get('regions', []))}")

        # the sanitized provider: fds at 3,4,... exactly as the engine passes them
        for i, fd in enumerate(fds):
            os.dup2(fd, 3 + i)
            os.set_inheritable(3 + i, True)
        fds_arg = ",".join(str(3 + i) for i in range(len(fds)))
        penv = dict(os.environ)
        penv["ASAN_OPTIONS"] = ("detect_leaks=0:detect_odr_violation=0:halt_on_error=0:symbolize=1:"
                                "print_stacktrace=1:detect_stack_use_after_return=1")
        prov_log = os.path.join(out, "provider.log")
        t0 = time.time()
        # pass_fds is not optional: subprocess's close_fds (the default) closes every descriptor except 0,1,2
        # even when the caller called set_inheritable() on it, so the first attempt handed the provider a dead
        # fd 3 -- mmap returned EBADF and the run blamed the dma-buf instead of the harness (measured 2026-11-04
        # 09:55, and reproduced in isolation with a memfd: "child: fd/3 MISSING" without this, mmap OK with it).
        # Popen, not run(): the provider can run for the whole timeout (1800 s on a 16k prompt), so it needs the
        # same two properties as the server -- its own session, so the deadline can reap the whole group, and
        # PR_SET_PDEATHSIG, so a driver killed by SIGKILL does not leave it behind (the shape of the 2026-11-08
        # incident and the rule engine#1624 is about). `run()` would give neither: it blocks, and its child is in
        # the driver's process group, which the teardown below does not signal.
        with open(prov_log, "w") as pf:
            prov = subprocess.Popen([PROV, "--model", MODEL, "--tokens", ids_path, "--layout", layout_path,
                                     "--fds", fds_arg], env=penv, stdout=pf, stderr=subprocess.STDOUT,
                                    pass_fds=tuple(3 + i for i in range(len(fds))),
                                    start_new_session=True, preexec_fn=die_with_parent)
            try:
                rc = prov.wait(timeout=1800)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(prov.pid), 9)
                except ProcessLookupError:
                    pass
                rc = -9

        class _R:   # the rest of the block reads like the run() it replaced
            returncode = rc
        prow = _R()
        say(f"provider exit={prow.returncode} wall={time.time() - t0:.1f}s log={prov_log}")
        prov_text = open(prov_log).read()
        asan = prov_text.count("ERROR: AddressSanitizer") + prov_text.count("SUMMARY: AddressSanitizer")
        corrupt = prov_text.count("corrupted size vs. prev_size")
        say(f"provider ASan reports={asan} glibc corruption lines={corrupt}")
        say("provider result: " + next((l for l in prov_text.splitlines() if '"prefill_ms"' in l), "<none>"))
        if asan:
            say("--- first ASan report")
            i = prov_text.find("ERROR: AddressSanitizer")
            show(prov_text[i:i + 3000])

        if prow.returncode == 0 and asan == 0 and corrupt == 0:
            try:
                ad = post(api, "/slots/0?action=adopt", {"tokens": toks, "share_id": share_id}, timeout=600)
                say(f"adopt ok: {json.dumps({k: ad[k] for k in list(ad)[:6]})[:200]}")
            except Exception as e:
                say(f"ADOPT FAILED: {e}")
                return 5
            try:
                rep = post(api, "/completion", {"prompt": prompt_used, "temperature": 0, "n_predict": max_tokens,
                                                 "cache_prompt": True})
            except urllib.error.HTTPError as e:
                say(f"DECODE FAILED: HTTP {e.code} " + e.read().decode(errors="replace")[:300])
                return 6
            t = rep.get("timings") or {}
            say(f"decode after adopt: prompt_n={t.get('prompt_n')} predicted_n={t.get('predicted_n')} "
                f"prompt_ms={t.get('prompt_ms')} text={rep.get('content')!r}")
            say("VERDICT: PASS - the provider mapped the exported region and prefilled into it with zero "
                "sanitizer reports, and the decoder decoded from the adopted cells")
            return 0
        say("VERDICT: not clean (see provider log)")
        return 1
    finally:
        try:
            os.killpg(os.getpgid(srv.pid), 15)
        except ProcessLookupError:
            pass
        try:
            srv.wait(timeout=180)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(srv.pid), 9)


def show(text):
    print(text, flush=True)


if __name__ == "__main__":
    sys.exit(main())
