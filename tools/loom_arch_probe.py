#!/usr/bin/env python3
# Copyright 2026 bong-water-water-bong
# SPDX-License-Identifier: Apache-2.0
"""loom_arch_probe.py -- does the Loom port starter support this GGUF's architecture, and what would it need?

engine#3221: on 2026-11-21 a `weaver port` of GLM-4.7-Flash ran for 1056 s, retried three times and ended with
"the agent finished without a WEAVER_RESULT line". The cause was settled in seconds once the metadata was read:
`general.architecture = deepseek2`, an MLA model, and the starter's `starter/gen_llm.py` has no MLA path. The
port attempt could not have succeeded, and nothing said so before it started.

This answers that before the attempt: read the GGUF header, compare the architecture against the starter's
supported list, and print the parameters a starter entry would need. No model download (it reads the header
only), no device, no weaver.

Exit: 0 supported, 1 not supported by the starter, 2 cannot tell (unreadable or missing architecture key).
"""
import argparse
import json
import struct
import sys

# starter/loom-port-llm/starter/{build.py,attn_rules.py}, read 2026-11-21. A name here means the generator has an
# attention shape for it; it does NOT mean any particular checkpoint of that family ports cleanly.
SUPPORTED = ("llama", "qwen2", "qwen3", "qwen35", "gemma3", "gemma4", "phi3", "glm")

# What a starter entry needs for the families whose shape is not plain attention. Printed so the answer is
# actionable rather than "unsupported".
NEEDS = {
    "deepseek2": "MLA: q_lora_rank, kv_lora_rank, key_length (kv_lora + rope), key_length_mla/value_length_mla, "
                 "head_count_kv = 1, plus routed-expert params (expert_count/used/shared/group, group_used)",
}

_T = {0: ("<B", 1), 1: ("<b", 1), 2: ("<H", 2), 3: ("<h", 2), 4: ("<I", 4), 5: ("<i", 4), 6: ("<f", 4),
      7: ("<?", 1), 10: ("<Q", 8), 11: ("<q", 8), 12: ("<d", 8)}


def read_kv(path):
    """{key: value} for every scalar/string kv the header holds, or None when the file is not a readable GGUF."""
    try:
        with open(path, "rb") as f:
            if f.read(4) != b"GGUF":
                return None
            _version, = struct.unpack("<I", f.read(4))
            _tensors, n_kv = struct.unpack("<QQ", f.read(16))

            def rstr():
                n, = struct.unpack("<Q", f.read(8))
                return f.read(n).decode("utf-8", "replace")

            out = {}
            for _ in range(n_kv):
                key = rstr()
                kind, = struct.unpack("<I", f.read(4))
                if kind == 8:
                    out[key] = rstr()
                elif kind == 9:
                    atype, = struct.unpack("<I", f.read(4))
                    count, = struct.unpack("<Q", f.read(8))
                    if atype == 8:
                        for _ in range(count):
                            rstr()
                    else:
                        f.read(_T[atype][1] * count)
                elif kind in _T:
                    out[key] = struct.unpack(_T[kind][0], f.read(_T[kind][1]))[0]
                else:
                    return None
            return out
    except (OSError, struct.error, KeyError):
        return None


def supported_from_starter(root):
    """Architecture names the starter's sources mention, or None when the tree is unreadable.

    The constant below is a snapshot, and a snapshot that drifts re-creates the failure this tool exists to prevent in
    the opposite direction: a *supported* architecture reported unsupported. So the list can be read from the source it
    is a claim about - `starter/*.py` mentions each architecture it handles by name (engine#3221 review).
    """
    import pathlib
    import re
    found = set()
    try:
        files = list(pathlib.Path(root).glob("*.py"))
    except OSError:
        return None
    if not files:
        return None
    pattern = re.compile(r"[\"']([a-z][a-z0-9_]{1,15})[\"']")
    for path in files:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for name in pattern.findall(text):
            if name in SUPPORTED:
                found.add(name)
    return found or None


def probe(kv, supported=SUPPORTED, source="built-in list"):
    """(exit status, lines). Pure: the tests below feed it synthetic kv dicts."""
    if kv is None:
        return 2, ["loom_arch_probe: not a readable GGUF - cannot tell, and that is not a pass"]
    arch = kv.get("general.architecture")
    if not arch:
        return 2, ["loom_arch_probe: no general.architecture - cannot tell, and that is not a pass"]
    if arch in supported:
        return 0, [f"loom_arch_probe: {arch} is in the port starter's supported list ({source})"]
    lines = [f"loom_arch_probe: {arch} is NOT in the port starter ({', '.join(sorted(supported))}; {source})",
             "  a `weaver port` of this checkpoint cannot succeed as the starter stands"]
    need = NEEDS.get(arch)
    if need:
        lines.append(f"  it would need: {need}")
    params = sorted(k for k in kv if k.startswith(arch + "."))
    if params:
        lines.append(f"  the GGUF carries {len(params)} {arch}.* key(s), so the parameters a starter entry needs are "
                     f"in the file: {', '.join(params[:8])}{' …' if len(params) > 8 else ''}")
    return 1, lines


_SYNTH = ("GGUF", 3, 0, 3)   # magic, version, 0 tensors, 3 kv


def _gguf_bytes(pairs):
    out = [b"GGUF", struct.pack("<I", 3), struct.pack("<QQ", 0, len(pairs))]
    for key, value in pairs:
        out.append(struct.pack("<Q", len(key)) + key.encode())
        if isinstance(value, str):
            out.append(struct.pack("<I", 8) + struct.pack("<Q", len(value)) + value.encode())
        else:
            out.append(struct.pack("<I", 4) + struct.pack("<I", value))
    return b"".join(out)


def self_test():
    import tempfile
    ok = 0
    bad = 0

    def arm(name, data, expect_rc, needle=None):
        nonlocal ok, bad
        with tempfile.NamedTemporaryFile(suffix=".gguf") as f:
            f.write(data)
            f.flush()
            rc, lines = probe(read_kv(f.name))
        text = "\n".join(lines)
        good = rc == expect_rc and (needle is None or needle in text)
        ok, bad = (ok + 1, bad) if good else (ok, bad + 1)
        print(f"{'ok  ' if good else 'FAIL'} {name}: rc {rc} (want {expect_rc})")

    arm("a supported architecture passes", _gguf_bytes([("general.architecture", "qwen3")]), 0)
    arm("an unsupported one fails and names what it needs",
        _gguf_bytes([("general.architecture", "deepseek2")]), 1, "MLA")
    arm("an unsupported one lists the keys the file already carries",
        _gguf_bytes([("general.architecture", "deepseek2"), ("deepseek2.attention.q_lora_rank", 768)]), 1,
        "q_lora_rank")
    arm("a file that is not a GGUF is cannot-tell, not a pass", b"NOPE" + b"\0" * 32, 2)
    arm("a GGUF with no architecture key is cannot-tell",
        _gguf_bytes([("general.name", "x")]), 2)
    print(f"loom_arch_probe self-test: {ok + bad} arms, {ok} ok, {bad} failed")
    return 0 if bad == 0 else 1


def main():
    parser = argparse.ArgumentParser(description="Does the Loom port starter support this GGUF's architecture?")
    parser.add_argument("gguf", nargs="?", help="path to a .gguf")
    parser.add_argument("--starter", help="the starter directory to read the supported list from (default: built-in)")
    parser.add_argument("--self-test", action="store_true", help="run the arms and exit")
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    if not args.gguf:
        parser.error("a GGUF path is required (or --self-test)")
    supported, source = SUPPORTED, "built-in list"
    if args.starter:
        derived = supported_from_starter(args.starter)
        if derived is None:
            print(f"loom_arch_probe: cannot read the starter at {args.starter} - cannot tell, and that is not a pass")
            return 2
        supported, source = derived, f"read from {args.starter}"
    rc, lines = probe(read_kv(args.gguf), supported, source)
    print("\n".join(lines))
    return rc


if __name__ == "__main__":
    sys.exit(main())
