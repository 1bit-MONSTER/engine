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
"""Add a per-wave NaN mask to the tensor fingerprint, so region and lane patterns are distinguishable.

Run from the instrumented tree root (the fingerprint tree, ~/wt/llama-315copy), alongside patch-fp-sync.py.

WHY IT IS NEEDED, and why counting cannot substitute: the fingerprint reports `nan` (a count) and `first` (the
index of the first NaN). Those two numbers are **structurally unable** to say whether the NaNs are one region
or a wave-interleaved pattern — `first + nan <= extent` holds either way
(dev/315/315-campaign/evidence/315-corruption-is-wave-granular.md, second correction). A 32-bit mask over the
first 2048 elements (32 waves of 64) resolves it in one line: a region is a contiguous run of set bits, a lane
pattern is interleaved.

Gated by GGML_HRX_TENSOR_FP_WAVEMASK, so it is inert unless asked for and adds one fprintf per printed line.
Bit w is set iff wave w (elements [64w, 64w+64)) contains at least one NaN.

NOT ESTABLISHED by this script: that it compiles. Verify with a build AND check the artefact for the env
string before trusting a run — an env var whose instrument is not in the binary is silently inert (the
record's defect #13, and the reason this campaign was re-aimed at a different tree).
"""
from __future__ import annotations

import pathlib
import sys

REL = "ggml/src/ggml-hrx/ggml-hrx.cpp"

MARKER = "GGML_HRX_TENSOR_FP_WAVEMASK"

A_DECL_OLD = "                    int64_t       first_nan = -1;\n"
A_DECL_NEW = ("                    int64_t       first_nan = -1;\n"
              "                    // engine#315: a per-wave NaN mask over the first n_embd(2048) elements, i.e. 32\n"
              "                    // waves of 64. A count cannot separate one corrupted region from a wave-interleaved\n"
              "                    // pattern; this mask can: a region is a contiguous run of set bits.\n"
              "                    uint32_t      wavemask = 0;\n")

A_NAN_OLD = """                        if (std::isnan(f[j])) {
                            ++nan;
                            if (first_nan < 0) {
                                first_nan = (int64_t) j;
                            }
                        } else {"""

A_NAN_NEW = """                        if (std::isnan(f[j])) {
                            ++nan;
                            if (first_nan < 0) {
                                first_nan = (int64_t) j;
                            }
                            {   // set the bit for this element's wave (64 elements each), first 32 waves only
                                const int64_t wave = (int64_t) j / 64;
                                if (wave < 32) wavemask |= (uint32_t) (1u << wave);
                            }
                        } else {"""

A_PRINT_OLD = """                    std::fprintf(stderr, "[hrx-finite-fp] call=%d name=%s finite=%zu hash_finite=%016llx nan=%d\\n",
                                 fp_calls, node->name, finite_count, (unsigned long long) hash_finite, nan);"""

A_PRINT_NEW = """                    std::fprintf(stderr, "[hrx-finite-fp] call=%d name=%s finite=%zu hash_finite=%016llx nan=%d\\n",
                                 fp_calls, node->name, finite_count, (unsigned long long) hash_finite, nan);
                    if (std::getenv("GGML_HRX_TENSOR_FP_WAVEMASK") != nullptr) {
                        // 32 bits, bit w = wave w (elements [64w,64w+64)) holds >=1 NaN. Contiguous run of set
                        // bits => one corrupted region; interleaved => a lane/wave pattern.
                        std::fprintf(stderr, "[hrx-wavemask] call=%d name=%s nan=%d wavemask=0x%08x\\n",
                                     fp_calls, node->name, nan, (unsigned) wavemask);
                    }"""


def main() -> int:
    path = pathlib.Path(REL)
    if not path.exists():
        print(f"FAIL: {REL} not found; run from the fingerprint tree root (~/wt/llama-315copy)")
        return 1
    text = path.read_text()
    if MARKER in text:
        print("already patched")
        return 0
    for anchor, label in ((A_DECL_OLD, "decl"), (A_NAN_OLD, "nan-branch"), (A_PRINT_OLD, "print")):
        n = text.count(anchor)
        if n != 1:
            print(f"FAIL: anchor {label} found {n} times, expected 1 — nothing written")
            return 1
    text = text.replace(A_DECL_OLD, A_DECL_NEW, 1)
    text = text.replace(A_NAN_OLD, A_NAN_NEW, 1)
    text = text.replace(A_PRINT_OLD, A_PRINT_NEW, 1)
    if text.count("{") != text.count("}"):
        print(f"FAIL: brace imbalance ({text.count('{')} vs {text.count('}')}) — nothing written")
        return 1
    path.write_text(text)
    print(f"patched {path}")
    print(f"  wavemask occurrences: {text.count('wavemask')}")
    print("  NOTE: anchors applied and braces balance; compilation is NOT established. Build it AND check the")
    print("        artefact for GGML_HRX_TENSOR_FP_WAVEMASK before trusting a run.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
