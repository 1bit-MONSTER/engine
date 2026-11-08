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
"""How many HIP code objects a shared library carries, by the property and not by a string (engine#1443).

`make-binary-drop.sh` used to assert `strings libggml-hrx.so | grep -c amdgcn-amd-amdhsa`, which tests a *clang
offload-bundle header* rather than the property it names. A HIP fat binary built the ordinary way embeds its code
objects as ELF images with `e_machine == EM_AMDGPU` (224) and contains no bundle header at all, so the shipped c31
library — 51 code objects — failed the check, while a build that happened to have a bundle header passed. Measured on
four real libraries on one box:

    c31 payload lib   10,864,208 B   53 ELF headers   51 EM_AMDGPU   (0 `amdgcn-amd-amdhsa` strings -> old check: FAIL)
    tprs-hrx-154       5,513,432 B    3 ELF headers    2 EM_AMDGPU   (2 strings -> old check: pass)
    hrx-a5271d9       11,427,720 B   52 ELF headers   50 EM_AMDGPU   (0 strings -> old check: FAIL)
    canonical          5,129,432 B    1 ELF header     0 EM_AMDGPU   (a build with no code objects: correctly refused)

What this does *not* answer: whether the **private** kernels are in the build. A tree configured `ONEBIT_HRX=ON` with
`ONEBIT_GPU_PRIVATE` empty still embeds AMDGPU code objects -- measured on this box, that library is 5,615,208 B with 5
ELF headers and **4** `EM_AMDGPU` images, so it passes this tool while carrying none of the private addon. "Are *our*
kernels in it" is the `nm -D ... ggml_hrx_hip_addon_register` test beside this one in `make-binary-drop.sh`; that symbol
separated all four libraries above (yes, no, yes, ...) where the code-object count did not. Keep both: this one says "the
file carries HIP code at all", the addon symbol says "and it is the kernels the drop promises".

usage: code_objects.py <file> [--min N] [--quiet]
exit:  0 when the file carries at least N (default 1) AMDGPU code objects, 1 when it carries fewer, 2 on a read error
"""
from __future__ import annotations

import argparse
import struct
import sys

EM_AMDGPU = 224          # include/elf/amdgpu.h, e_machine for an AMDGPU code object
ELF_MAGIC = b"\x7fELF"
EHDR_MIN = 20            # magic (4) + class/data/version/osabi/abiversion/pad (12) + e_type (2) + e_machine (2)


def count(path: str) -> int:
    """Embedded ELF headers whose e_machine is EM_AMDGPU.

    The file's own header counts too if it is one (a bare code object), which is what makes this usable on a .hsaco
    as well as on a fat .so. A short or truncated header is skipped rather than raising: a packaging check must not
    crash on a cut-off file, it must report zero code objects.
    """
    data = open(path, "rb").read()
    n, at = 0, data.find(ELF_MAGIC)
    while at != -1:
        if at + EHDR_MIN <= len(data):
            (machine,) = struct.unpack_from("<H", data, at + 18)
            if machine == EM_AMDGPU:
                n += 1
        at = data.find(ELF_MAGIC, at + len(ELF_MAGIC))
    return n


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("file")
    ap.add_argument("--min", type=int, default=1, help="how many code objects the file must carry (default 1)")
    ap.add_argument("--quiet", action="store_true", help="say nothing on success")
    a = ap.parse_args(argv)
    try:
        n = count(a.file)
    except OSError as exc:
        print(f"{a.file}: {exc}", file=sys.stderr)
        return 2
    if n < a.min:
        print(f"{a.file}: {n} AMDGPU code object(s), needs {a.min}: no embedded ELF with e_machine EM_AMDGPU (224). "
              f"A HIP fat binary embeds its code objects as such ELF images; 'amdgcn-amd-amdhsa' is a clang "
              f"offload-bundle header and is absent from a normally-built library (engine#1443).", file=sys.stderr)
        return 1
    if not a.quiet:
        print(n)
    return 0


if __name__ == "__main__":
    sys.exit(main())
