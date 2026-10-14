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
"""Add the DEVICE-buffer producer probe to graph-executor.cpp, per
docs/315-campaign/evidence/315-device-buffer-probe-proposal.md.

Run on the box, inside the instrumented llama tree (the default is ~/wt/llama-315probe). Every anchor is
asserted to occur exactly once, so a silent no-op is impossible — the same discipline as tools/patch-*.py.

WHY A PATCH SCRIPT AND NOT JUST THE CODE: the change cannot be compile-tested here. This script verifies the
anchors and the idempotency marker, which are the parts that CAN be verified offline; that a build then
compiles is explicitly NOT established by running it. Do not report the instrument as working until a build
has succeeded and an arm has produced an `armed`/`anomaly` line.

WHAT IT CHANGES (3 functions, ~40 lines) and why each is needed:

  producer_probe_target   accept non-host tensors when GGML_HRX_PRODUCER_DEVICE is set, selected by an exact
                          row-multiple (bytes % 8192 == 0) so the residual family is chosen without needing T.
  producer_probe_fill     for a device target, upload a finite pattern from a host staging buffer with
                          ggml_backend_tensor_set. The existing raw node->data write is a HOST pointer; on a
                          device tensor it would write through a device address from the CPU.
  producer_probe_scan     for a device target, download with ggml_backend_tensor_get after the drained
                          stream and classify the staging buffer. Same reason, on the read side.

The payload in device mode is FINITE (0xDEADBEEF), not the NaN sentinel: this probe fills a live layer input,
so a NaN fill would propagate and manufacture the very symptom under study.
"""
from __future__ import annotations

import pathlib
import sys

REL = "ggml/src/ggml-hrx/runtime/graph-executor.cpp"

MARKER = "GGML_HRX_PRODUCER_DEVICE"

A_TARGET_OLD = """bool producer_probe_target(const ggml_tensor * t) {
    if (t == nullptr || t->data == nullptr) return false;
    if (t->buffer == nullptr || !ggml_backend_buffer_is_host(t->buffer)) return false;
"""

A_TARGET_NEW = """// engine#315: device mode. The host-only probe cannot reach a layer residual (hrx-producer prints host=0
// for every layer node), so this admits device tensors too. The payload is FINITE here, not the NaN
// sentinel: this fills a tensor that is also an op's input, so a NaN fill would propagate and manufacture
// the symptom under study.
bool producer_probe_device_enabled() {
    const char * v = std::getenv("GGML_HRX_PRODUCER_DEVICE");
    return v != nullptr && *v != '\\0' && *v != '0';
}

bool producer_probe_target(const ggml_tensor * t) {
    if (t == nullptr || t->data == nullptr) return false;
    if (producer_probe_device_enabled()) {
        // Exact row-multiple: an integer number of n_embd(2048)-wide rows, so the residual family is
        // selected without needing to know T. Verified against the graph: the layer residuals are the
        // device tensors whose byte size is an exact multiple of 8192.
        const size_t dbytes = ggml_nbytes(t);
        return dbytes >= 8192u && dbytes % 8192u == 0u;
    }
    if (t->buffer == nullptr || !ggml_backend_buffer_is_host(t->buffer)) return false;
"""

A_FILL_OLD = """        uint32_t * words = static_cast<uint32_t *>(node->data);
        const size_t n = ggml_nbytes(node) / sizeof(uint32_t);
        for (size_t w = 0; w < n; ++w) words[w] = kProbeSentinel;
"""

A_FILL_NEW = """        const size_t n = ggml_nbytes(node) / sizeof(uint32_t);
        if (producer_probe_device_enabled()) {
            // Device target: node->data is not CPU-addressable, so upload from a host staging buffer.
            static std::vector<uint32_t> staging;
            staging.assign(n, kProbeFinite);
            ggml_backend_tensor_set(const_cast<ggml_tensor *>(node), staging.data(), 0, ggml_nbytes(node));
        } else {
            uint32_t * words = static_cast<uint32_t *>(node->data);
            for (size_t w = 0; w < n; ++w) words[w] = kProbeSentinel;
        }
"""

A_SCAN_OLD = """        const float *  fp = static_cast<const float *>(node->data);
"""

A_SCAN_NEW = """        // Device target: download through the backend instead of dereferencing node->data, which is a
        // device address. The stream is already drained above, so this reads completed writes. NOTE: `n`
        // is declared AFTER this point in the original, so it must not be referenced here.
        static std::vector<float> down;
        const float *            fp = nullptr;
        if (producer_probe_device_enabled()) {
            const size_t dfloats = ggml_nbytes(node) / sizeof(float);
            down.assign(dfloats, 0.0f);
            ggml_backend_tensor_get(const_cast<ggml_tensor *>(node), down.data(), 0, ggml_nbytes(node));
            fp = down.data();
        } else {
            fp = static_cast<const float *>(node->data);
        }
"""

# The finite payload constant goes next to the existing NaN sentinel so the two are read together. The anchor
# includes the sentinel's TRAILING COMMENT deliberately: matching only the declaration prefix left that
# comment stranded on the end of the new line, which compiles but reads as a mangled double comment.
A_SENT_OLD = "constexpr uint32_t kProbeSentinel = 0x7FC0DEADu;  // payload NaN, distinguishable from a plain 0x7FC00000"
A_SENT_NEW = ("constexpr uint32_t kProbeSentinel = 0x7FC0DEADu;  // payload NaN, distinguishable from a plain 0x7FC00000\n"
              "constexpr uint32_t kProbeFinite   = 0xDEADBEEFu;  // engine#315 device mode: FINITE, never a NaN")


def apply(text: str, old: str, new: str, label: str) -> str:
    n = text.count(old)
    if n != 1:
        print(f"FAIL: anchor {label} found {n} times, expected 1")
        sys.exit(1)
    return text.replace(old, new)


def main() -> int:
    path = pathlib.Path(REL)
    if not path.exists():
        print(f"FAIL: {REL} not found; run from the instrumented tree root")
        return 1
    text = path.read_text()
    if MARKER in text:
        print("already patched")
        return 0
    # A pre-check on every anchor before any replacement, so a partial patch cannot be written.
    for old, label in ((A_SENT_OLD, "sentinel-const"), (A_TARGET_OLD, "target"),
                       (A_FILL_OLD, "fill"), (A_SCAN_OLD, "scan")):
        if text.count(old) != 1:
            print(f"FAIL: anchor {label} found {text.count(old)} times, expected 1 — nothing written")
            return 1
    text = apply(text, A_SENT_OLD, A_SENT_NEW, "sentinel-const")
    text = apply(text, A_TARGET_OLD, A_TARGET_NEW, "target")
    text = apply(text, A_FILL_OLD, A_FILL_NEW, "fill")
    text = apply(text, A_SCAN_OLD, A_SCAN_NEW, "scan")
    # Balance check: a cheap structural guard against a truncated or mis-spliced replacement. Also require
    # <vector> to be reachable, since the staging buffers use std::vector.
    if text.count("{") != text.count("}"):
        print(f"FAIL: brace imbalance after patch ({text.count('{')} vs {text.count('}')}) — nothing written")
        return 1
    if "#include <vector>" not in text and "<vector>" not in text.split("namespace")[0]:
        print("WARN: <vector> not found before the namespace; the patch needs it for the staging buffers.")
        print("      Add the include before building.")
    path.write_text(text)
    print(f"patched {path}")
    print(f"  device-enable branch: {text.count(MARKER)} occurrence(s)")
    print("  NOTE: this only proves the anchors applied and braces balance. It does NOT prove it compiles —")
    print("        build it before claiming the instrument works.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
