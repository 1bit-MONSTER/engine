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
#
# engine#315: the id probe reads an i32 tensor that, by construction, must hold ranks in [0, ne0) - and it
# returns float-shaped values instead (evidence: 315-idprobe-result.md). ggml_argsort allocates a FRESH i32
# tensor (ggml.c:5347) and ggml_argsort_top_k views it, so after the graph executes that buffer must hold
# ranks. It does not, in faulting or clean arms. Two candidate explanations, and they are separable:
#
#   (a) the read path returns memory that is not the tensor's contents. buffer_get does a plain memcpy and
#       RETURNS when the buffer is host-visible (context->base != GGML_HRX_FAKE_PTR_BASE); it never reaches
#       hrx_synchronous_d2h. If that mapping is non-coherent, the memcpy can return a stale or reused
#       occupant - which is exactly what "float probabilities in an i32 rank tensor" looks like.
#   (b) the argsort op did not write that buffer at all.
#
# This patch separates them by reading the SAME tensor twice, in the same line of output: once the way the
# probe has been reading it (plain memcpy) and once through hrx_synchronous_d2h, which is the synchronising
# path buffer_get skips for host-visible buffers. Both reads are printed, tagged `hrx-ids-fp` (unchanged, so
# prior logs stay comparable) and `hrx-ids-sync` (new).
#
# Readout, fixed here before the run:
#   - memcpy floats and sync ranks in [0, ne0)  -> (a) confirmed: the read path was the defect, the tensor is
#                                                  fine, and the id family becomes testable.
#   - both float-shaped                         -> (b): the argsort output genuinely is not in that buffer.
#   - both ranks                                -> the earlier float reading did not reproduce; the probe is
#                                                  then re-read against this run only.
#
# Diagnostic only: no criterion changes, no arm from it is eligible.
#
# Idempotent: refuses to apply twice.
import pathlib
import sys

SRC = pathlib.Path.home() / "wt/llama-315copy/ggml/src/ggml-hrx/ggml-hrx.cpp"

REPORT_HELPER = r'''
// engine#315: print the shape of an i32 index list - min/max, how many entries fall outside [0, ne0), and how
// many repeat within a row. Shared by the plain-memcpy and the synchronising read so the two lines are
// computed identically and can be compared directly.
static void hrx_ids_report(const char * tag, int call, const char * name, size_t ibytes, int64_t ine0,
                           const int32_t * ids) {
    const size_t icount = ibytes / sizeof(int32_t);
    const size_t irows  = ine0 > 0 ? icount / static_cast<size_t>(ine0) : 0;
    size_t       oor = 0, dup = 0;
    int32_t      imin = 0, imax = 0;
    for (size_t j = 0; j < icount; ++j) {
        const int32_t v = ids[j];
        if (j == 0 || v < imin) { imin = v; }
        if (j == 0 || v > imax) { imax = v; }
        if (v < 0 || v >= static_cast<int32_t>(ine0)) { ++oor; }
    }
    for (size_t r = 0; r < irows; ++r) {
        for (size_t a = 0; a < static_cast<size_t>(ine0); ++a) {
            for (size_t b = a + 1; b < static_cast<size_t>(ine0); ++b) {
                if (ids[r * static_cast<size_t>(ine0) + a] == ids[r * static_cast<size_t>(ine0) + b]) { ++dup; }
            }
        }
    }
    std::fprintf(stderr,
                 "[%s] call=%d name=%s bytes=%zu count=%zu ne0=%lld rows=%zu "
                 "min=%d max=%d oor=%zu dup_pairs=%zu first8=[",
                 tag, call, name, ibytes, icount, (long long) ine0, irows, imin, imax, oor, dup);
    for (size_t j = 0; j < icount && j < 8; ++j) {
        std::fprintf(stderr, "%s%d", j == 0 ? "" : " ", (int) ids[j]);
    }
    std::fprintf(stderr, "]\n");
    std::fflush(stderr);
}
'''

HERO_ANCHOR = "static bool hrx_fp_name_listed("

OLD_BLOCK = '''                        const size_t         ibytes = ggml_nbytes(node);
                        std::vector<uint8_t> ihost(ibytes);
                        ggml_backend_tensor_get(node, ihost.data(), 0, ibytes);
                        const int32_t * ids = reinterpret_cast<const int32_t *>(ihost.data());
                        const size_t    icount = ibytes / sizeof(int32_t);
                        const int64_t   ine0   = node->ne[0] > 0 ? node->ne[0] : 1;
                        const size_t    irows  = ine0 > 0 ? icount / static_cast<size_t>(ine0) : 0;
                        size_t          oor = 0, dup = 0;
                        int32_t         imin = 0, imax = 0;
                        for (size_t j = 0; j < icount; ++j) {
                            const int32_t v = ids[j];
                            if (j == 0 || v < imin) { imin = v; }
                            if (j == 0 || v > imax) { imax = v; }
                            if (v < 0 || v >= static_cast<int32_t>(ine0)) { ++oor; }
                        }
                        for (size_t r = 0; r < irows; ++r) {
                            for (size_t a = 0; a < static_cast<size_t>(ine0); ++a) {
                                for (size_t b = a + 1; b < static_cast<size_t>(ine0); ++b) {
                                    if (ids[r * static_cast<size_t>(ine0) + a] == ids[r * static_cast<size_t>(ine0) + b]) { ++dup; }
                                }
                            }
                        }
                        std::fprintf(stderr,
                                     "[hrx-ids-fp] call=%d name=%s bytes=%zu count=%zu ne0=%lld rows=%zu "
                                     "min=%d max=%d oor=%zu dup_pairs=%zu first8=[",
                                     fp_calls, node->name, ibytes, icount, (long long) ine0, irows, imin, imax, oor, dup);
                        for (size_t j = 0; j < icount && j < 8; ++j) {
                            std::fprintf(stderr, "%s%d", j == 0 ? "" : " ", (int) ids[j]);
                        }
                        std::fprintf(stderr, "]\\n");
                        std::fflush(stderr);
                        continue;'''

NEW_BLOCK = '''                        const size_t ibytes = ggml_nbytes(node);
                        const int64_t ine0  = node->ne[0] > 0 ? node->ne[0] : 1;

                        // (1) the read the probe has always done: plain memcpy for a host-visible buffer.
                        std::vector<uint8_t> ihost(ibytes);
                        ggml_backend_tensor_get(node, ihost.data(), 0, ibytes);
                        hrx_ids_report("hrx-ids-fp", fp_calls, node->name, ibytes, ine0,
                                       reinterpret_cast<const int32_t *>(ihost.data()));

                        // (2) the same tensor through the synchronising path that buffer_get skips for a
                        // host-visible buffer. If (1) and (2) disagree, the read path is what (1) reported.
                        ggml_backend_buffer_t node_buffer =
                            node->view_src != nullptr ? node->view_src->buffer : node->buffer;
                        if (node_buffer != nullptr) {
                            auto *       nctx = buffer_context(node_buffer);
                            const size_t noff = tensor_offset(nctx, node);
                            std::vector<uint8_t> isync(ibytes);
                            if (HRX_CHECK(hrx_synchronous_d2h(nctx->device->device, nctx->buffer, noff,
                                                              isync.data(), ibytes))) {
                                hrx_ids_report("hrx-ids-sync", fp_calls, node->name, ibytes, ine0,
                                               reinterpret_cast<const int32_t *>(isync.data()));
                            }
                        }
                        continue;'''


def main() -> int:
    if not SRC.exists():
        print(f"missing {SRC}", file=sys.stderr)
        return 2
    text = SRC.read_text()
    if "hrx-ids-sync" in text:
        print("already applied (hrx-ids-sync present)")
        return 0
    if OLD_BLOCK not in text:
        print("anchor block not found - the id instrument is not in the expected state", file=sys.stderr)
        return 3
    if HERO_ANCHOR not in text:
        print("helper anchor not found", file=sys.stderr)
        return 4
    text = text.replace(HERO_ANCHOR, REPORT_HELPER + "\n" + HERO_ANCHOR, 1)
    text = text.replace(OLD_BLOCK, NEW_BLOCK, 1)
    SRC.write_text(text)
    print(f"patched {SRC}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
