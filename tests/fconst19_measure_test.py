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
"""The 1.9 measurement tool's own paths, on synthetic packs.

`tools/fconst19_measure.py` shipped a crash on its refusal path because the only pair it was ever run on
verified on the FIRST candidate, so the "trying next" line never executed. These fixtures execute it: the
first candidate always fails structurally, so a regression there is a failing test rather than a NameError
found by whoever next measures a real pack.

The fixtures are built from the rule itself -- a 1.8-block, a header, a rotated bundle -- which is the one
thing a synthetic test can do that a real pack cannot: it can be WRONG on purpose.

Three cases:
  * a correct pair with a decoy structure first      -> prints the table line, and the decoy line executes
  * a pair that differs by a few hundred codes       -> accepted, residual reported (the tolerance branch)
  * a pair whose residual is far above the tolerance -> refused, with the number

Usage: tests/fconst19_measure_test.py   (no arguments; writes to a temp dir and cleans up)
"""
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
TOOL = os.path.join(HERE, "..", "tools", "fconst19_measure.py")

# The tool's own sizes are part of its contract: it probes 4 KB of payload head and it will not consider a
# tile below 4 KB, so the fixtures have to be at least that big to exercise it at all. Hidden 4096 gives an
# 8,256-byte prefix, which is also what makes a `hidden` field in config.json plausible.
HIDDEN, HEADS, LAYERS = 4096, 2, 8         # 8 layers => ~198 KB per file, so one stray byte is 5e-6 of it
HDR_LEN = HEADS * 4                         # 8 bytes: 00 00 7f ff per head
PREFIX = 64 + 2 * HIDDEN                    # the block envelope: [eps/pad][norm] then the bundle
N_TILES, TILE, ROT = 2, 8192, 1             # a 2-tile bundle rotated by 1
BUNDLE = N_TILES * TILE
HEADER_OFF = PREFIX + BUNDLE + 64           # bundle, then a little verbatim span, then the header
STRIDE_8 = HEADER_OFF + HDR_LEN + 128       # header, then the rest of the block


def block_18(seed):
    """One 1.8 layer block.

    Deterministic but NOT periodic: a ramp (`i * 7 & 0xFF`) repeats every 256 bytes, and a rotation by a
    multiple of 256 then hides completely -- the first version of this fixture did exactly that and the tool
    (correctly) saw no divergence until the header. Pseudo-random bytes have no such alignment.
    """
    return random.Random(seed).randbytes(STRIDE_8)


def block_19(seed):
    """The same block in the 1.9 form: header inserted, bundle rotated by ROT."""
    a = block_18(seed)
    out = bytearray(a[:PREFIX])
    for i in range(N_TILES):                       # stored tile j holds canonical (j + ROT) mod n
        j = (i + ROT) % N_TILES
        out += a[PREFIX + j * TILE:PREFIX + (j + 1) * TILE]
    out += a[PREFIX + BUNDLE:HEADER_OFF]
    out += bytes.fromhex("00007fff") * HEADS
    out += a[HEADER_OFF:]
    return bytes(out)


def write_pack(root, name, token_bytes, declared_18=True):
    d = os.path.join(root, name)
    os.makedirs(d, exist_ok=True)
    json.dump({"hidden_size": HIDDEN, "num_attention_heads": HEADS, "num_hidden_layers": LAYERS},
              open(os.path.join(d, "config.json"), "w"))
    stem = "dd_metastate_Llm_Token_Token_aietok" if declared_18 else "dd_metastate_Llm_Token_aie2_token_modellayers"
    open(os.path.join(d, stem + ".fconst"), "wb").write(token_bytes)
    return d


def run(pairs, extra=()):
    p = subprocess.run([sys.executable, TOOL, "--pack-1-8", pairs[0], "--pack-1-9", pairs[1], *extra],
                       capture_output=True, text=True)
    return p.returncode, p.stdout, p.stderr


def case_exact(root):
    a = b"".join(block_18(L) for L in range(LAYERS)) + b"TAIL" * 16
    b = b"".join(block_19(L) for L in range(LAYERS)) + b"TAIL" * 16
    p18 = write_pack(root, "p18", a)
    p19 = write_pack(root, "p19", b, declared_18=False)
    rc, out, err = run((p18, p19), ("--probe-wrong",))
    ok = rc == 0 and "byte for byte" in out and f"{{{HIDDEN}, {HEADS}, {LAYERS}" in out
    # the decoy/perturbation lines prove the refusal-and-continue path executed, which is what shipped broken
    ok = ok and "wrong rule" in err
    return ok, f"rc={rc} out={out.strip()[:90]!r} err_has_probe={'wrong rule' in err}"


def case_tolerance(root):
    a = b"".join(block_18(L) for L in range(LAYERS)) + b"TAIL" * 16
    b = bytearray(b"".join(block_19(L) for L in range(LAYERS)) + b"TAIL" * 16)
    # One stray code, which is what a fine-tune's twin looks like at this scale: 1 B of ~198 KB is 5e-6, inside
    # the 1e-5 tolerance, where the real pairs measured 5.6e-7 (Qwen3-1.7B) to 2.7e-6 (Llama-3.2-1B).
    # It goes in the VERBATIM region after the header, not in the bundle: a byte inside the first 4 KB of the
    # bundle is what the tool's displacement probe reads, so mutating there breaks discovery instead of testing
    # the tolerance -- a different failure, and the distinction cost a fixture iteration to see.
    # In a LATER layer, because discovery inspects block 0 only -- which is exactly why real fine-tune pairs
    # pass: Qwen3-1.7B's 535 differing bytes are all in layer 24, so block 0 is byte-exact and the residual is
    # reported from the blocks after it. Mutating block 0 instead breaks the tail check and tests discovery.
    later = STRIDE_8 + HDR_LEN                   # the header of layer 1 in the 1.9 layout
    b[later + HDR_LEN + 10] ^= 0x11
    p18 = write_pack(root, "t18", a)
    p19 = write_pack(root, "t19", bytes(b), declared_18=False)
    rc, out, err = run((p18, p19))
    ok = rc == 0 and "bytes differ" in out and "byte for byte" not in out
    return ok, f"rc={rc} out={out.strip()[:100]!r}"


def case_refuse(root):
    a = b"".join(block_18(L) for L in range(LAYERS)) + b"TAIL" * 16
    # A long run of the verbatim region rewritten: the structure still fits, the residual does not, so this
    # exercises the refusal the tolerance exists to make safe rather than the discovery refusal above it.
    b = bytearray(b"".join(block_19(L) for L in range(LAYERS)) + b"TAIL" * 16)
    later = STRIDE_8 + HDR_LEN
    for i in range(4096):
        b[later + HDR_LEN + i] ^= 0xA5
    p18 = write_pack(root, "r18", a)
    p19 = write_pack(root, "r19", bytes(b), declared_18=False)
    rc, out, err = run((p18, p19))
    ok = rc != 0 and ("refuse" in err.lower() or "no candidate" in err.lower())
    return ok, f"rc={rc} err={err.strip().splitlines()[-1][:90] if err.strip() else ''!r}"



def case_check_row(root):
    """engine#2010: a row must BE the measurement. `prefix` is the first byte where the twins diverge, so a row
    whose prefix came from `64 + 2*hidden + 4,352` (or from a neighbour's shape) is wrong in a way no geometry
    check sees. The guard is `--check-row <pack_import.cpp>`; here the source file is a two-line stand-in."""
    a = b"".join(block_18(L) for L in range(LAYERS)) + b"TAIL" * 16
    b = b"".join(block_19(L) for L in range(LAYERS)) + b"TAIL" * 16
    p18 = write_pack(root, "p18", a)
    p19 = write_pack(root, "p19", b, declared_18=False)
    rc, out, err = run((p18, p19), ("--json",))
    if rc != 0:
        return False, f"the pair itself did not measure: rc={rc} {err.strip()[:80]}"
    row = json.loads([l for l in out.splitlines() if l.startswith("{")][0])
    good = (f"    {{{row['hidden']}, {row['heads']}, {row['layers']}, {row['prefix']}, {row['n_tiles']}, "
            f"{row['rot']}, {row['tile']}}},")   # the struct's order: prefix, n, rot, tile
    src = os.path.join(root, "good.cpp")
    with open(src, "w") as f:
        f.write("constexpr Fconst19Rule kFconst19Rules[] = {\n" + good + "\n};\n")
    rc1, out1, err1 = run((p18, p19), ("--check-row", src))
    # and a row whose prefix is off by one tile-length, which is what a derived prefix looks like
    bad = good.replace(f", {row['prefix']},", f", {row['prefix'] + 64},", 1)
    src2 = os.path.join(root, "bad.cpp")
    with open(src2, "w") as f:
        f.write("constexpr Fconst19Rule kFconst19Rules[] = {\n" + bad + "\n};\n")
    rc2, out2, err2 = run((p18, p19), ("--check-row", src2))
    ok = rc1 == 0 and rc2 == 1 and "is exactly what this pair measures" in (out1 + err1) \
         and "MISMATCH" in (out2 + err2) and f"prefix {row['prefix'] + 64} vs measured" in (out2 + err2)
    return ok, f"match rc={rc1}; mismatch rc={rc2}; detail={(out2 + err2).strip()[-90:]!r}"


def main():
    root = tempfile.mkdtemp(prefix="fconst19-test.")
    try:
        results = [("exact pair (and the refusal path executes)", case_exact(root)),
                   ("a few hundred codes differ: accepted with residual", case_tolerance(root)),
                   ("a scrambled tile: refused", case_refuse(root)),
                   ("--check-row: the table's row must be this pair's measurement", case_check_row(root))]
    finally:
        shutil.rmtree(root, ignore_errors=True)
    bad = [n for n, (ok, _) in results if not ok]
    for n, (ok, detail) in results:
        print(f"{'ok  ' if ok else 'FAIL'} {n}" + ("" if ok else f"  [{detail}]"))
    if bad:
        print(f"\n{len(bad)} of {len(results)} cases failed")
        return 1
    print(f"\nall {len(results)} cases pass")
    return 0


if __name__ == "__main__":
    sys.exit(main())
