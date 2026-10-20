<!--
Copyright 2026 bong-water-water-bong
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->
<!-- Copyright 2026 bong-water-water-bong
     SPDX-License-Identifier: Apache-2.0

     engine#315: the position-selecting fingerprint diff, verbatim, generated from the live
     ~/wt/llama-315copy tree with tools/gen-nan-dump.py and verified to apply with
     `git apply --check -p1`. Not built yet: it is the instrument for the next diagnostic arm,
     and building it would change the binary that produced section 23's evidence. -->

# 315-nan-dump

Extends the per-tensor fingerprint (`GGML_HRX_TENSOR_FP`, section 23) in two ways, so the stage
that manufactures the NaN can be found without guessing tensor names:

- `GGML_HRX_TENSOR_FP=tail:<N>` selects the last `N` graph nodes **by position** instead of by name.
  The head-end pipeline (last layer, final norm, LM head) is the graph's tail, and the producer is
  not yet named, so position is the only selector that cannot silently match nothing.
- `GGML_HRX_TENSOR_FP_NANONLY=1` prints only the lines where `nan > 0`. The device-to-host read is
  still paid for every selected node - only the log shrinks - which is what makes a position sweep
  readable rather than a flood.

Each line now also carries `op=%s` (`ggml_op_name`), so a fingerprint identifies the *op* as well as
the tensor, which is what "name the compute stage" needs.

```diff
--- a/ggml/src/ggml-hrx/ggml-hrx.cpp
+++ b/ggml/src/ggml-hrx/ggml-hrx.cpp
@@ -582,9 +582,23 @@
             const int    fp_limit = lim_env != nullptr ? std::atoi(lim_env) : 1000000;
             static int   fp_calls = 0;
             if (fp_calls < fp_limit) {
-                for (int i = 0; i < graph->n_nodes; ++i) {
+                // engine#315: GGML_HRX_TENSOR_FP also accepts "tail:<N>", which selects the last N nodes
+                // of the graph by position instead of by name. The stage that manufactures the NaN is not
+                // known yet, so position is the only way to walk the head-end pipeline without guessing
+                // tensor names; N bounds the extra device->host traffic per graph. GGML_HRX_TENSOR_FP_NANONLY
+                // keeps only the NaN-bearing lines, which is what makes a position sweep readable.
+                int tail_n = 0;
+                if (std::strncmp(want, "tail:", 5) == 0) {
+                    tail_n = std::atoi(want + 5);
+                }
+                const bool fp_nan_only = std::getenv("GGML_HRX_TENSOR_FP_NANONLY") != nullptr;
+                const int  fp_first    = tail_n > 0 && graph->n_nodes > tail_n ? graph->n_nodes - tail_n : 0;
+                for (int i = fp_first; i < graph->n_nodes; ++i) {
                     ggml_tensor * node = graph->nodes[i];
-                    if (node == nullptr || node->name[0] == '\0' || std::strstr(want, node->name) == nullptr) {
+                    if (node == nullptr) {
+                        continue;
+                    }
+                    if (tail_n == 0 && (node->name[0] == '\0' || std::strstr(want, node->name) == nullptr)) {
                         continue;
                     }
                     const size_t         bytes = ggml_nbytes(node);
@@ -602,9 +616,12 @@
                             ++nan;
                         }
                     }
+                    if (fp_nan_only && nan == 0) {
+                        continue;
+                    }
                     std::fprintf(stderr,
-                                 "[hrx-tensor-fp] call=%d name=%s bytes=%zu nan=%d of %zu hash=%016llx\n",
-                                 fp_calls, node->name, bytes, nan, n, (unsigned long long) hash);
+                                 "[hrx-tensor-fp] call=%d name=%s op=%s bytes=%zu nan=%d of %zu hash=%016llx\n",
+                                 fp_calls, node->name, ggml_op_name(node->op), bytes, nan, n, (unsigned long long) hash);
                     std::fflush(stderr);
                 }
                 ++fp_calls;
```
