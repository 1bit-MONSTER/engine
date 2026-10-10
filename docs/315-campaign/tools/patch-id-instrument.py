#!/usr/bin/env python3
# Copyright 2026 bong-water-water-bong
# SPDX-License-Identifier: Apache-2.0
#
# engine#315: patch the tensor fingerprint in ~/wt/llama-315copy so the next probe can answer the question
# 315row-p1c left open. Three changes, all inside the GGML_HRX_TENSOR_FP block:
#
#   1. ANCHORED name matching. The current filter is std::strstr(want, node->name), i.e. it looks for the
#      node name *inside* the comma-separated list, so a list entry "ffn_moe_weights-41" also matches a node
#      named "ffn_moe_weights-4". Today's coverage was layers 4 and 41 only, which is a probe defect.
#   2. i32 ids diagnostics. ffn_moe_topk / ffn_moe_argsort are i32; the instrument reads every tensor as f32,
#      so the ids - the one tensor the fault sits on - have never been looked at. For an i32 tensor print the
#      first 8 values, the count outside [0, ne0), the number of repeated values per row, and min/max.
#      Rationale: the argsort's rank computation stops being a bijection when an input row holds two NaNs,
#      which leaves a stale id in the top-k view (evidence/315-id-producing-kernel.md).
#   3. A finite-subset hash for f32 tensors, to settle whether the identical ffn_moe_weights hashes at layers
#      4 and 41 mean an aliased read or just all-NaN content.
#
# Idempotent: refuses to apply twice.
import hashlib
import pathlib
import re
import sys

SRC = pathlib.Path.home() / "wt/llama-315copy/ggml/src/ggml-hrx/ggml-hrx.cpp"

HELPER = r'''
// engine#315: anchored match of a node name against the comma-separated GGML_HRX_TENSOR_FP list, plus a
// finite-subset hash and an i32 ids dump. The previous filter was std::strstr(list, name), which matches a
// list entry "ffn_moe_weights-41" against a node named "ffn_moe_weights-4"; that is a probe defect, not a
// property of the fault.
static bool hrx_fp_name_listed(const char * list, const char * name) {
    if (list == nullptr || name == nullptr || name[0] == '\0') {
        return false;
    }
    const size_t name_len = std::strlen(name);
    const char * p        = list;
    while (*p != '\0') {
        const char * comma = std::strchr(p, ',');
        const size_t len   = comma != nullptr ? static_cast<size_t>(comma - p) : std::strlen(p);
        if (len == name_len && std::strncmp(p, name, name_len) == 0) {
            return true;
        }
        if (comma == nullptr) {
            break;
        }
        p = comma + 1;
    }
    return false;
}
'''

OLD_FILTER = '''                    if (tail_n == 0 && (node->name[0] == '\\0' || std::strstr(want, node->name) == nullptr)) {
                        continue;
                    }'''

NEW_FILTER = '''                    if (tail_n == 0 && !hrx_fp_name_listed(want, node->name)) {
                        continue;
                    }
                    if (tail_n == 0 && node->type == GGML_TYPE_I32) {
                        // engine#315: the ids. Read as i32 and report the shape of the index list, not a
                        // NaN count - an i32 tensor has no NaN, and the question is whether the list is a
                        // clean permutation with every entry inside [0, ne0).
                        const size_t         ibytes = ggml_nbytes(node);
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
                        continue;
                    }'''

OLD_HASH_DECL = '''                    int64_t       first_nan = -1;
                    uint64_t      hash = 1469598103934665603ull;'''
NEW_HASH_DECL = '''                    int64_t       first_nan = -1;
                    uint64_t      hash = 1469598103934665603ull;
                    // engine#315: a second hash over the finite subset only, so two tensors that share a
                    // NaN payload cannot look identical when their finite parts differ.
                    uint64_t      hash_finite = 1469598103934665603ull;
                    size_t        finite_count = 0;'''

OLD_HASH_BODY = '''                        if (std::isnan(f[j])) {
                            ++nan;
                            if (first_nan < 0) {
                                first_nan = (int64_t) j;
                            }
                        }
                    }'''
NEW_HASH_BODY = '''                        if (std::isnan(f[j])) {
                            ++nan;
                            if (first_nan < 0) {
                                first_nan = (int64_t) j;
                            }
                        } else {
                            hash_finite = (hash_finite ^ bits) * 1099511628211ull;
                            ++finite_count;
                        }
                    }'''

OLD_PRINT = '''                                 (long long) (first_nan >= 0 ? first_nan / ne0 : -1), (unsigned long long) hash);'''
NEW_PRINT = '''                                 (long long) (first_nan >= 0 ? first_nan / ne0 : -1), (unsigned long long) hash);
                    std::fprintf(stderr, "[hrx-finite-fp] call=%d name=%s finite=%zu hash_finite=%016llx nan=%d\\n",
                                 fp_calls, node->name, finite_count, (unsigned long long) hash_finite, nan);
                    std::fflush(stderr);'''


def main() -> int:
    text = SRC.read_text()
    if "hrx_fp_name_listed" in text:
        print("already patched")
        return 0
    before = hashlib.sha256(text.encode()).hexdigest()[:16]
    for old, new, label in (
        (OLD_FILTER, NEW_FILTER, "anchored filter + i32 ids dump"),
        (OLD_HASH_DECL, NEW_HASH_DECL, "finite hash decl"),
        (OLD_HASH_BODY, NEW_HASH_BODY, "finite hash body"),
        (OLD_PRINT, NEW_PRINT, "finite hash print"),
    ):
        if text.count(old) != 1:
            print(f"FAIL: {label}: found {text.count(old)} matches, expected 1")
            return 1
        text = text.replace(old, new)
    anchor = "static enum ggml_status graph_compute(ggml_backend_t backend, ggml_cgraph * graph) {"
    if text.count(anchor) != 1:
        print("FAIL: graph_compute anchor not unique")
        return 1
    text = text.replace(anchor, HELPER + "\n" + anchor)
    SRC.write_text(text)
    after = hashlib.sha256(text.encode()).hexdigest()[:16]
    print(f"patched {SRC}  {before} -> {after}")
    for marker in ("hrx_fp_name_listed", "hrx-ids-fp", "hrx-finite-fp"):
        print(f"  {marker}: {text.count(marker)} occurrences")
    return 0


if __name__ == "__main__":
    sys.exit(main())
