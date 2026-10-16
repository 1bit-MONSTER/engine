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
"""Extract every JIT-cache kernel variant, disassemble it, and report the per-variant instruction profile.

The Oct-5 kvnan bisection logs show the same Loom source and the same numerical result (checksum 1.190266)
from variants that either race (bad_runs 2-249 per 250) or do not (bad_runs 0). So the difference has to be
in the emitted code. This dumps each variant's hsaco to /tmp/fa-variants/<tag>.hsaco and its disassembly to
<tag>.s, then prints an instruction profile so a racing variant can be diffed against a clean one.
"""
import os
import pathlib
import re
import struct
import subprocess
import sys

OBJDUMP = "/home/bcloud/.cache/xdna-km/llvm19/llvm-objdump"
TARGET = "gfx1151"
ROOTS = [pathlib.Path.home() / ".cache" / "kvnan"]
OUT = pathlib.Path("/tmp/fa-variants")

TAGS = ["b0", "b1", "b2", "b3", "c0", "c1", "c2", "c3", "e0", "e1", "e2", "e3",
        "s0", "s1", "s2", "jit190", "jit190c", "jit190d", "jit190e", "jit190p",
        "jit190q", "jit190r", "jit190w", "jit192", "jith336", "jitmf"]


def extract(blob: bytes):
    idx = blob.find(b"\x7fELF")
    if idx < 8:
        return None
    (n,) = struct.unpack_from("<Q", blob, idx - 8)
    if n <= 0 or idx + n > len(blob) + 64:
        return None
    return blob[idx : idx + n]


def profile(text: str):
    def c(pattern: str) -> int:
        return len(re.findall(pattern, text))

    return {
        "gl_inv": c(r"buffer_gl[01]_inv"),
        "barrier": c(r"\bs_barrier\b"),
        "waitcnt": c(r"\bs_waitcnt\b"),
        "vscnt": c(r"\bs_waitcnt_vscnt\b"),
        "vmcnt": c(r"\bs_waitcnt_vmcnt\b"),
        "lgkmcnt": c(r"\bs_waitcnt_lgkmcnt\b"),
        "ds_write": c(r"\bds_write"),
        "ds_read": c(r"\bds_read"),
        "global_store": c(r"\bglobal_store"),
        "global_load": c(r"\bglobal_load"),
        "exp": c(r"\bexp\b"),
        "lines": text.count("\n"),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for root in ROOTS:
        for tag in TAGS:
            d = root / tag
            if not d.is_dir():
                continue
            files = [f for f in d.iterdir() if f.is_file()]
            if len(files) != 1:
                continue
            blob = files[0].read_bytes()
            hsaco = extract(blob)
            if hsaco is None:
                continue
            (OUT / f"{tag}.hsaco").write_bytes(hsaco)
            res = subprocess.run([OBJDUMP, "-d", "--mcpu=" + TARGET, str(OUT / f"{tag}.hsaco")],
                                 capture_output=True, text=True, timeout=180)
            (OUT / f"{tag}.s").write_text(res.stdout)
            rows.append((tag, profile(res.stdout)))
    keys = ["gl_inv", "barrier", "waitcnt", "vscnt", "vmcnt", "lgkmcnt", "ds_write", "ds_read",
            "global_store", "global_load", "exp", "lines"]
    print(f"{'tag':10s} " + " ".join(f"{k:>12s}" for k in keys))
    for tag, p in rows:
        print(f"{tag:10s} " + " ".join(f"{p[k]:12d}" for k in keys))
    print()
    print(f"hsaco + .s written to {OUT}")


if __name__ == "__main__":
    sys.exit(main())
