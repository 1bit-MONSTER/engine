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
"""Add the A'/B' coherence toggle for the host staging download buffer, per
docs/315-campaign/evidence/315-candidate-host-staging-coherence.md.

Run on the box from the instrumented tree root (default ~/wt/llama-315probe). Anchors are asserted to occur
exactly once and a pre-check runs over ALL of them before anything is written, so a partial or silent patch is
impossible — the same discipline as tools/patch-*.py.

WHY THIS IS ENV-GATED AND NOT TWO BUILDS: arm A' (unchanged) and arm B' (bit added) must differ in exactly one
thing. Gating on GGML_HRX_STAGING_COHERENT makes ONE binary serve both arms, so the A/B cannot be confounded by
a build difference — the failure mode that would make a null result uninterpretable.

WHAT IT CHANGES AND WHY: allocate_mapped_host_staging_buffer allocates the buffer the device writes
(hrx_stream_copy_buffer) and the host later reads (a plain memcpy at command-program-executor.cpp:129) with no
invalidate, and its params request HOST_LOCAL | DEVICE_VISIBLE WITHOUT HOST_COHERENT. The same tree calls
coherence "a requirement ... not a preference" for an analogous device-writes/host-reads buffer because "the
read path never invalidates". Arm B' adds the bit; arm A' leaves everything as shipped.

NOT ESTABLISHED BY THIS SCRIPT, and it must not be claimed: that it compiles, and that the achieved mapping was
ever non-coherent in the first place (the HIP backend may grant coherence to host-located memory regardless —
see the falsifier in the document). The script only applies the toggle.
"""
from __future__ import annotations

import pathlib
import sys

REL = "ggml/src/ggml-hrx/runtime/host-memory.cpp"

MARKER = "GGML_HRX_STAGING_COHERENT"

HELPER_ANCHOR = "Status allocate_mapped_host_staging_buffer(hrx_device_t device, size_t size, hrx_buffer_t & buffer, void *& host_data) {"

HELPER = """// engine#315 A'/B' toggle. The staging buffer this function returns is written by the device
// (hrx_stream_copy_buffer in command-program-executor.cpp:1099) and read by the host with a plain memcpy at
// :129 with no invalidate, yet it is requested without HOST_COHERENT. The same tree calls coherence "a
// requirement here and not a preference" for an analogous device-writes/host-reads buffer because "the read
// path never invalidates" (event_timestamp_pool.c:136-146). GGML_HRX_STAGING_COHERENT=1 selects arm B'.
static bool staging_coherent_enabled() {
    const char * v = std::getenv("GGML_HRX_STAGING_COHERENT");
    return v != nullptr && *v != '\\0' && *v != '0';
}

Status allocate_mapped_host_staging_buffer(hrx_device_t device, size_t size, hrx_buffer_t & buffer, void *& host_data) {"""

PARAMS_OLD = """    hrx_buffer_params_t params = {
        HRX_MEMORY_TYPE_HOST_LOCAL | HRX_MEMORY_TYPE_DEVICE_VISIBLE,
        HRX_MEMORY_ACCESS_ALL,"""

PARAMS_NEW = """    hrx_memory_type_t staging_type = HRX_MEMORY_TYPE_HOST_LOCAL | HRX_MEMORY_TYPE_DEVICE_VISIBLE;
    if (staging_coherent_enabled()) {
        staging_type |= HRX_MEMORY_TYPE_HOST_COHERENT;
    }
    hrx_buffer_params_t params = {
        staging_type,
        HRX_MEMORY_ACCESS_ALL,"""


def main() -> int:
    path = pathlib.Path(REL)
    if not path.exists():
        print(f"FAIL: {REL} not found; run from the instrumented tree root")
        return 1
    text = path.read_text()
    if MARKER in text:
        print("already patched")
        return 0
    for anchor, label in ((HELPER_ANCHOR, "helper-anchor"), (PARAMS_OLD, "params")):
        n = text.count(anchor)
        if n != 1:
            print(f"FAIL: anchor {label} found {n} times, expected 1 — nothing written")
            return 1
    text = text.replace(HELPER_ANCHOR, HELPER, 1)
    text = text.replace(PARAMS_OLD, PARAMS_NEW, 1)
    if text.count("{") != text.count("}"):
        print(f"FAIL: brace imbalance ({text.count('{')} vs {text.count('}')}) — nothing written")
        return 1
    if "HRX_MEMORY_TYPE_HOST_COHERENT" not in text:
        print("FAIL: coherence symbol absent after patch — nothing written")
        return 1
    path.write_text(text)
    print(f"patched {path}")
    print(f"  toggle occurrences: {text.count(MARKER)}")
    print("  NOTE: anchors applied and braces balance; compilation is NOT established. Build it before use.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
