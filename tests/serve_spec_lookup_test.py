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
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""forward-serve --spec-lookup returns the plain loop's reply, token for token (engine#1781).

usage: tests/serve_spec_lookup_test.py <1bit binary>        (needs numpy)

The speculative path (app/forward_serve.cpp, `spec`) decodes a greedy request through mtp::run_from() with the
prompt-lookup drafter. It is only allowed where it cannot change a token, and this holds it to that end to end: the
two-layer qwen3 container tests/batch_check_state_ci.py fabricates, a 256-token byte-level tokenizer written here, and
`1bit forward-serve` in host-GEMM mode (NPU_INFER_HOST_GEMM=1: no device is opened, no lock is taken) started twice,
plain, with --spec-lookup 4, and with --spec-draft on a DFlash package written here (tests/dflash_draft_test.cpp's
synthetic layout, sized to the target). The same id prompts go to both: several budgets, stop sequences cut out of the plain
replies (so they do end their replies), and one seeded sampled request. Text, finish_reason and completion_tokens
must be equal; the spec server must take the speculative path on exactly the greedy requests and draft (its `[spec]`
lines).

A dev-box row: forward-serve is built only with ONEBIT_NPU. Exits 77 (skip) without numpy.
"""
import importlib.util
import json
import os
import socket
import struct
import subprocess
import sys
import tempfile
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
TOOL = os.path.join(HERE, "..", "tools", "safetensors_to_q4nx.py")


def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


def write_tokenizer(d):
    m = bytes_to_unicode()
    vocab = {m[b]: b for b in range(256)}
    # The two ChatML specials, so the chat route (the one that serves logprobs) takes requests on this container.
    added = [{"id": 256 + i, "content": t, "single_word": False, "lstrip": False, "rstrip": False, "normalized": False,
              "special": True} for i, t in enumerate(("<|im_start|>", "<|im_end|>"))]
    tok = {"version": "1.0", "added_tokens": added,
           "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True, "use_regex": True},
           "decoder": {"type": "ByteLevel"},
           "model": {"type": "BPE", "vocab": vocab, "merges": []}}
    with open(os.path.join(d, "tokenizer.json"), "w") as f:
        json.dump(tok, f)


# A generation-1 DFlash package for the fabricated target, after tests/dflash_draft_test.cpp's synthetic one: the same
# formulas and layout (q4_1 tile grids of 5120 B, a BF16 fusion in 32-row blocks), sized to this target (hidden 256,
# taps on its layers 0 and 1).
def to_bf16(f):
    return struct.unpack("<I", struct.pack("<f", f))[0] >> 16


def code_of(salt, n, k):
    return (n * 7 + k * 3 + salt) & 15


def scale_of(salt, n, g):
    return 0.0078125 * (1 + ((n + g + salt) % 4))


def min_of(salt, n, g):
    return -0.0625 * (1 + ((n * 3 + g + salt) % 5))


def encode_projection(salt, N, K):
    out = bytearray()
    for tr in range(N // 32):
        for tc in range(K // 256):
            tile = bytearray(5120)
            for g in range(8):
                for r in range(32):
                    n, grp = tr * 32 + r, tc * 8 + g
                    struct.pack_into("<H", tile, 2 * (g * 32 + r), to_bf16(scale_of(salt, n, grp)))
                    struct.pack_into("<H", tile, 512 + 2 * (g * 32 + r), to_bf16(min_of(salt, n, grp)))
            for r in range(32):
                half, rr = r // 16, r % 16
                for c in range(256):
                    g, j = c // 32, c % 32
                    code = code_of(salt, tr * 32 + r, tc * 256 + c)
                    at = 1024 + (half * 8 + g) * 256 + 8 * j + rr // 2
                    tile[at] = (tile[at] & 0xF0) | code if rr % 2 == 0 else (tile[at] & 0x0F) | (code << 4)
            out += tile
    return bytes(out)


def write_draft(d, hidden=256, inter=512, layers=2, heads=4, kv_heads=2, head_dim=64, block=8, taps=(0, 1)):
    os.makedirs(d)
    t = []
    t.append(("dflash.embedding_scale", "F32", [1], struct.pack("<f", 16.0)))
    t.append(("dflash.rms_eps", "F32", [1], struct.pack("<f", 1e-6)))
    t.append(("dflash.rope_theta", "F32", [1], struct.pack("<f", 1e6)))
    t.append(("dflash.block_size", "I32", [1], struct.pack("<i", block)))
    t.append(("dflash.sliding_window", "I32", [1], struct.pack("<i", 1024)))
    t.append(("dflash.target_layer_ids", "I32", [len(taps)], struct.pack(f"<{len(taps)}i", *taps)))
    fused = len(taps) * hidden
    fc = bytearray()
    for a in range(hidden // 32):
        for k in range(fused):
            for r in range(32):
                fc += struct.pack("<H", to_bf16(0.03125 * (((a * 32 + r) * 5 + k * 11) % 64 - 32)))
    t.append(("model.fc.weight", "BF16", [hidden // 32, fused, 32], bytes(fc)))

    def gain(name, salt, width):
        t.append((name, "BF16", [width], b"".join(struct.pack("<H", to_bf16(1.0 + 0.125 * ((i + salt) % 8)))
                                                  for i in range(width))))

    def proj(name, salt, N, K):
        b = encode_projection(salt, N, K)
        t.append((name, "I8", [len(b) // 5120, 5120], b))

    gain("model.hidden_norm.weight", 1, hidden)
    q, kv = heads * head_dim, kv_heads * head_dim
    for l in range(layers):
        p, sl = f"model.layers.{l}.", 10 * (l + 1)
        gain(p + "input_layernorm.weight", sl + 1, hidden)
        gain(p + "post_attention_layernorm.weight", sl + 2, hidden)
        gain(p + "self_attn.k_norm.weight", sl + 3, head_dim)
        gain(p + "self_attn.q_norm.weight", sl + 4, head_dim)
        proj(p + "self_attn.q_proj.weight", sl + 1, q, hidden)
        proj(p + "self_attn.k_proj.weight", sl + 2, kv, hidden)
        proj(p + "self_attn.v_proj.weight", sl + 3, kv, hidden)
        proj(p + "self_attn.o_proj.weight", sl + 4, hidden, q)
        proj(p + "mlp.gate_proj.weight", sl + 5, inter, hidden)
        proj(p + "mlp.up_proj.weight", sl + 6, inter, hidden)
        proj(p + "mlp.down_proj.weight", sl + 7, hidden, inter)
    gain("model.norm.weight", 2, hidden)
    cfg = {"model_type": "dflash", "architectures": ["DFlashDraftModel"], "base_model": "synthetic",
           "draft_weight": "dflash_weight.q4nx", "block_size": block, "mask_token_id": 4, "target_layer_ids": list(taps),
           "num_hidden_layers": layers, "hidden_size": hidden, "intermediate_size": inter, "num_attention_heads": heads,
           "num_key_value_heads": kv_heads, "head_dim": head_dim, "rms_norm_eps": 1e-06, "rope_theta": 1000000.0,
           "sliding_window": 1024, "embedding_scale": 16.0, "causal": False}
    with open(os.path.join(d, "config.json"), "w") as f:
        json.dump(cfg, f)
    header, payload = {"__metadata__": {}}, bytearray()
    for name, dtype, shape, b in t:
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [len(payload), len(payload) + len(b)]}
        payload += b
    h = json.dumps(header, separators=(",", ":")).encode()
    with open(os.path.join(d, "dflash_weight.q4nx"), "wb") as f:
        f.write(struct.pack("<Q", len(h)) + h + bytes(payload))


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def post(port, body):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)


def post_chat(port, body):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)


def lp_of(j):
    """A chat reply's text, finish and logprobs as (token, logprob, [(token, logprob)...]) per entry."""
    c = j["choices"][0]
    ents = (c.get("logprobs") or {}).get("content") or []
    return (c["message"]["content"], c["finish_reason"],
            [(e["token"], e["logprob"], [(t["token"], t["logprob"]) for t in e["top_logprobs"]]) for e in ents])


def lp_close(a, b, tol=1e-5):
    """Same text, finish and tokens; every logprob within tol (the verify rows and the plain steps are one forward)."""
    if a[0] != b[0] or a[1] != b[1] or len(a[2]) != len(b[2]):
        return False, float("inf")
    worst = 0.0
    for (ta, la, xa), (tb, lb, xb) in zip(a[2], b[2]):
        if ta != tb or [t for t, _ in xa] != [t for t, _ in xb]:
            return False, float("inf")
        worst = max([worst, abs(la - lb)] + [abs(p - q) for (_, p), (_, q) in zip(xa, xb)])
    return worst <= tol, worst


def start(binary, model, port, extra, log):
    env = dict(os.environ, NPU_INFER_HOST_GEMM="1", OMP_WAIT_POLICY="PASSIVE")
    p = subprocess.Popen([binary, "forward-serve", "-m", os.path.join(model, "model.q4nx"), "--kernels",
                          os.path.join(model, "kernels"), "-p", str(port)] + extra,
                         env=env, stdout=log, stderr=subprocess.STDOUT)
    for _ in range(600):
        if p.poll() is not None:
            return p
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1).read()
            return p
        except Exception:
            time.sleep(0.1)
    return p


def main():
    if len(sys.argv) != 2:
        print(__doc__.strip().splitlines()[2])
        return 2
    try:
        import numpy as np
    except ImportError:
        print("SKIP: numpy is not installed")
        return 77
    spec = importlib.util.spec_from_file_location("bcs", os.path.join(HERE, "batch_check_state_ci.py"))
    bcs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bcs)
    # The head is wider than the tokenizer (512 rows against 256 ids), as on Qwen3 (151,936 against 151,669): the verify
    # batch's rows are the head's width while the loop reads them at the tokenizer's, and only a wider head shows that.
    bcs.VOCAB = 512
    fails = 0
    with tempfile.TemporaryDirectory(prefix="serve_spec_lookup.", dir=bcs.tmp_root()) as tmp:
        src, out = os.path.join(tmp, "hf"), os.path.join(tmp, "q4nx")
        bcs.make_model(src, np, "qwen3")
        r = subprocess.run([sys.executable, TOOL, "pack", src, out], capture_output=True, text=True)
        if r.returncode != 0:
            print(f"FAIL  packing the fabricated container: {r.stderr[-600:]}")
            return 1
        write_tokenizer(out)
        prompts = [[40, 41, 42, 43, 40, 41, 42, 43, 40, 41], list(range(60, 90)), [7, 7, 7, 7, 7, 7], [200, 3, 200, 3, 200]]
        base = [{"prompt": p, "max_tokens": n, "temperature": 0} for p in prompts for n in (1, 5, 24)]
        # Constrained (greedy): each draft is checked in the grammar state after the drafts before it, and the reply must
        # be the plain server's. The first grammar forces "b" after "a" and "d" after "c", so on a prompt of those
        # pairs the lookup's drafts are often the forced token and are kept.
        ab = list(b"abcdabcdabcdab")
        grammar = 'root ::= ("ab" | "cd" | "e")+'
        constrained = [{"prompt": ab, "max_tokens": 24, "temperature": 0, "grammar": grammar},
                       {"prompt": prompts[1], "max_tokens": 24, "temperature": 0,
                        "response_format": {"type": "json_object"}}]
        base += constrained
        # A sampled request (seeded, so both servers draw the same tokens) must stay on the plain loop.
        sampled = [{"prompt": prompts[1], "max_tokens": 16, "temperature": 0.8, "seed": 7}]
        # --spec-sample's own requests: sampled, seeded, one on a repeating prompt so the lookup has drafts to offer.
        # Each is sent twice; the distribution itself is held by tests/mtp_sampled_test.cpp, this holds the wiring.
        sample_cases = [sampled[0], {"prompt": prompts[0], "max_tokens": 24, "temperature": 0.8, "seed": 11}]
        # Chat requests with logprobs (top 3): greedy, so every speculative server must report what plain reports.
        chats = [{"messages": [{"role": "user", "content": q}], "max_tokens": 24, "temperature": 0, "logprobs": True,
                  "top_logprobs": 3} for q in ("abcabcabcabc", "\x07" * 6)]
        # (the second keeps drafts under the lookup: 7 of 13 when written, so accepted rows report logprobs too)
        chat_sampled = {"messages": [{"role": "user", "content": "abcabcabcabc"}], "max_tokens": 24,
                        "temperature": 0.8, "seed": 5, "logprobs": True, "top_logprobs": 3}
        replies, logs, cases, sampled_replies = {}, {}, [], []
        draft = os.path.join(tmp, "draft")
        write_draft(draft)
        dump = os.path.join(tmp, "dump")
        os.makedirs(dump)
        for name, extra in (("plain", ["--dump-logits", dump]), ("spec", ["--spec-lookup", "4"]),
                            ("dflash", ["--spec-draft", draft]), ("sample", ["--spec-lookup", "4", "--spec-sample"])):
            port = free_port()
            logp = os.path.join(tmp, name + ".log")
            with open(logp, "w") as log:
                p = start(sys.argv[1], out, port, extra, log)
                try:
                    if p.poll() is not None:
                        print(f"FAIL  {name} server exited: " + open(logp).read()[-800:])
                        return 1
                    if name == "plain":
                        got = [post(port, b) for b in base]
                        # Stop sequences that occur: a piece of the plain reply itself, taken a few characters in, so
                        # each one cuts its reply mid-way (finish "stop") and the hold-back runs on both paths.
                        stops = []
                        for j, b in zip(got, base):
                            t = j["choices"][0]["text"]
                            # printable ASCII only: a replacement character (a split UTF-8 byte) never matches
                            at = next((i for i in range(3, len(t) - 1) if t[i:i + 2].isascii() and t[i:i + 2].isprintable()), -1)
                            if b["max_tokens"] == 24 and at >= 0:
                                stops.append({"prompt": b["prompt"], "max_tokens": 24, "temperature": 0,
                                              "stop": [t[at:at + 2]]})
                        cases = base + stops + sampled
                    replies[name + "_chat"] = [lp_of(post_chat(port, b)) for b in chats]
                    if name == "sample":
                        replies["sample_chat_sampled"] = lp_of(post_chat(port, chat_sampled))
                    replies[name] = []
                    for b in cases:
                        j = post(port, b)
                        c = j["choices"][0]
                        replies[name].append((c["text"], c["finish_reason"], j["usage"]["completion_tokens"]))
                    if name == "sample":
                        j = post(port, {"prompt": ab, "max_tokens": 24, "temperature": 0.9, "seed": 4, "grammar": grammar})
                        replies["sample_grammar"] = j["choices"][0]["text"]
                        # Near-greedy: at temperature 0.001 the distribution is the argmax's to within e^-10 on these
                        # logits, so the draft check must refuse what greedy refuses and the text is plain greedy's.
                        cold = [dict(b, temperature=0.001, seed=3) for b in base if b["max_tokens"] == 24 and b not in constrained]
                        replies["cold"] = []
                        for b in cold:
                            j = post(port, b)
                            c = j["choices"][0]
                            replies["cold"].append((c["text"], c["finish_reason"], j["usage"]["completion_tokens"]))
                        for b in sample_cases + sample_cases:
                            j = post(port, b)
                            c = j["choices"][0]
                            sampled_replies.append((c["text"], c["finish_reason"], j["usage"]["completion_tokens"]))
                finally:
                    p.terminate()
                    try:
                        p.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        p.kill()
                        p.wait()
            logs[name] = open(logp).read()
        for arm in ("spec", "dflash", "sample"):
            for b, a, c in zip(cases, replies["plain"], replies[arm]):
                if arm == "sample" and b["temperature"]:
                    continue   # with --spec-sample a sampled reply is a different draw; checked below
                ok = a == c
                fails += not ok
                what = (f"prompt {prompts.index(b['prompt']) if b['prompt'] in prompts else 'ab'} max_tokens {b['max_tokens']}"
                        + (" grammar" if "grammar" in b else " json_object" if "response_format" in b else "")
                        + (f" stop {b['stop']!r}" if "stop" in b else "")
                        + (f" temperature {b['temperature']}" if b["temperature"] else ""))
                print(f"{'PASS' if ok else 'FAIL'}  {arm}: {what}: {c[2]} tokens, finish {c[1]}"
                      + ("" if ok else f" (plain {a!r}, {arm} {c!r})"))
        for arm in ("spec", "dflash", "sample"):
            for i, (a, c) in enumerate(zip(replies["plain_chat"], replies[arm + "_chat"])):
                ok, worst = lp_close(a, c)
                ok = ok and len(a[2]) >= 2
                fails += not ok
                print(f"{'PASS' if ok else 'FAIL'}  {arm}: chat {i} with top_logprobs 3: {len(c[2])} entries equal plain's "
                      f"(worst logprob difference {worst:.2e})" + ("" if ok else f" (plain {a!r}, {arm} {c!r})"))
        # Plain's logprobs against an independent softmax over the tokenizer's 258 ids, from the rows plain dumped (its
        # chat requests follow the base ones: step_<len(base) + 1>_* and the next). The arms above hold every speculative server to plain,
        # so this holds them all; plain against the others cannot see a defect in the helper they share (engine#2272).
        n_ids = 258
        for i, a in enumerate(replies["plain_chat"]):
            ok, worst, shift = len(a[2]) >= 2, 0.0, 0.0
            for n, (_, lp, top) in enumerate(a[2]):
                step = os.path.join(dump, f"step_{len(base) + i + 1}_{n}.bin")
                row = np.fromfile(step, dtype=np.float32).astype(np.float64)
                ok = ok and row.size == bcs.VOCAB and len(top) == 3   # the row is the head's width: padding in it
                if not ok:
                    break
                ids = row[:n_ids]
                ls = ids - (ids.max() + np.log(np.exp(ids - ids.max()).sum()))
                best = sorted(range(n_ids), key=lambda t: (-ids[t], t))[:3]   # greedy's pick, then the tie rule
                worst = max([worst, abs(lp - ls[best[0]])] + [abs(q - ls[t]) for (_, q), t in zip(top, best)])
                wide = row[best[0]] - (row.max() + np.log(np.exp(row - row.max()).sum()))
                shift = max(shift, abs(wide - ls[best[0]]))
            # shift: what normalising over the head's width would have reported instead; it must be visible here
            ok = ok and worst <= 1e-4 and shift > 1e-3
            fails += not ok
            print(f"{'PASS' if ok else 'FAIL'}  plain: chat {i}: {len(a[2])} logprobs and top_logprobs equal a softmax "
                  f"over the {n_ids} tokenizer ids of the dumped {bcs.VOCAB}-wide rows (worst {worst:.1e}; the head's "
                  f"width would shift them by up to {shift:.2e})")
        # The chat requests are each server's first: their [spec] lines lead the log. Accepted drafts must be among them,
        # or the rows that report a kept draft's logprobs were never exercised.
        chat_lines = [l for l in logs["spec"].splitlines() if l.startswith("[spec] lookup ")][:len(chats)]
        kept = sum(int(l.split(" drafts accepted")[0].split(", ")[-1].split(" of ")[0]) for l in chat_lines)
        ok = kept > 0
        fails += not ok
        print(f"{'PASS' if ok else 'FAIL'}  spec: the chat requests with logprobs kept {kept} drafts")
        import re
        g_lines = [l for l in logs["spec"].splitlines() if l.startswith("[spec] lookup grammar ")]
        g_kept = sum(int(l.split(" drafts accepted")[0].split(", ")[-1].split(" of ")[0]) for l in g_lines)
        ok = len(g_lines) == len(constrained) and g_kept > 0
        fails += not ok
        print(f"{'PASS' if ok else 'FAIL'}  spec: {len(g_lines)} of {len(constrained)} constrained requests took the "
              f"speculative path and kept {g_kept} drafts")
        sg = replies["sample_grammar"]
        ok = bool(re.fullmatch(r"(ab|cd|e)+", sg)) and any(l.startswith("[spec] lookup grammar ")
                                                            for l in logs["sample"].splitlines())
        fails += not ok
        print(f"{'PASS' if ok else 'FAIL'}  sample: a sampled constrained reply speculates and stays in its grammar ({sg!r})")
        # Unset, --spec-fallback is the drafter's default: 2 tokens per round for the lookup, 3 for a draft model.
        for arm, want in (("spec", 2), ("dflash", 3), ("sample", 2)):
            line = f"speculative decode falls back to plain below {want} tokens per round (default)"
            ok = line in logs[arm]
            fails += not ok
            print(f"{'PASS' if ok else 'FAIL'}  {arm}: unset --spec-fallback is the drafter's default ({want})")
        sc = replies["sample_chat_sampled"]
        ok = len(sc[2]) >= 2 and all(l <= 0 and len(x) == 3 and x[0][1] >= l for _, l, x in sc[2])
        fails += not ok
        print(f"{'PASS' if ok else 'FAIL'}  sample: a sampled chat request with logprobs reports {len(sc[2])} entries, each "
              "a log-probability at most its row's best")
        n_stop = sum(1 for b, c in zip(cases, replies["spec"]) if "stop" in b and c[1] == "stop")
        ok = n_stop >= 2
        fails += not ok
        print(f"{'PASS' if ok else 'FAIL'}  {n_stop} stop-sequence requests ended on their stop sequence")
        greedy = sum(1 for b in cases if not b["temperature"]) + len(chats)
        for arm, kind in (("spec", "lookup"), ("dflash", "dflash")):
            spec_lines = [l for l in logs[arm].splitlines() if l.startswith(f"[spec] {kind} ")]
            drafted = sum(int(l.split(" of ")[1].split()[0]) for l in spec_lines)
            accepted = sum(int(l.split(" drafts accepted")[0].split(", ")[-1].split(" of ")[0]) for l in spec_lines)
            # Every request of two or more tokens drafts: a sink that kept the previous request's rows, or missed this
            # one's, would make the head refuse (the text stays right, so only this line can see it).
            # (The lookup has nothing to propose when no earlier text matches, so this holds the DFlash arm only.)
            dry = [l for l in spec_lines if arm == "dflash" and int(l.split(" tokens:")[0].split()[-1]) >= 2
                   and " of 0 drafts" in l]
            ok = len(spec_lines) == greedy and drafted > 0 and not dry and "[spec]" not in logs["plain"]
            fails += not ok
            print(f"{'PASS' if ok else 'FAIL'}  {arm}: the speculative path ({kind}) on {len(spec_lines)} requests, the "
                  f"{greedy} greedy ones ({accepted} of {drafted} drafts accepted, {len(dry)} multi-token requests without a draft); the sampled one "
                  f"and the plain server never")
        greedy24 = [r for b, r in zip(cases, replies["plain"]) if not b["temperature"] and b["max_tokens"] == 24
                    and "stop" not in b and b not in constrained]
        for i, (a, c) in enumerate(zip(greedy24, replies["cold"])):
            ok = a == c
            fails += not ok
            print(f"{'PASS' if ok else 'FAIL'}  sample: prompt {i} at temperature 0.001 is plain greedy's reply"
                  + ("" if ok else f" (plain {a!r}, sampled {c!r})"))
        # --spec-sample: greedy requests above equal plain's; the sampled ones speculate, keep their budget, and are
        # reproducible from their seed.
        n = len(sample_cases)
        lines = [l for l in logs["sample"].splitlines() if l.startswith("[spec] lookup sampled ")]
        drafted = sum(int(l.split(" of ")[1].split()[0]) for l in lines)
        accepted = sum(int(l.split(" drafts accepted")[0].split(", ")[-1].split(" of ")[0]) for l in lines)
        same = sampled_replies[:n] == sampled_replies[n:]
        full = all(r[2] == b["max_tokens"] or r[1] == "stop" for r, b in zip(sampled_replies, sample_cases + sample_cases))
        want = 2 * n + sum(1 for b in cases if b["temperature"]) + len(replies["cold"]) + 1   # + the sampled chat   # its sampled requests plus the shared sampled case
        ok = len(lines) == want and drafted > 0 and same and full
        fails += not ok
        print(f"{'PASS' if ok else 'FAIL'}  sample: {len(lines)} of {want} sampled requests took the speculative path "
              f"({accepted} of {drafted} drafts accepted), the same seed gave the same reply: {same}, budgets kept: {full}")
    print("ALL PASS" if not fails else f"FAILED {fails}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
