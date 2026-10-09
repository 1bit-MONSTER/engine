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

     engine#315 evidence: the 315-coherence-fix diff, verbatim. It is stored as Markdown because the repository's
     CI requires a copyright notice on every file and a raw .patch cannot carry one. -->

# 315-coherence-fix

```diff
diff --git a/ggml/src/ggml-hrx/ggml-hrx.cpp b/ggml/src/ggml-hrx/ggml-hrx.cpp
index 8090e3b..1acfdb5 100644
--- a/ggml/src/ggml-hrx/ggml-hrx.cpp
+++ b/ggml/src/ggml-hrx/ggml-hrx.cpp
@@ -309,9 +309,13 @@ static ggml_backend_buffer_t buffer_alloc(ggml_backend_buffer_type_t buft, size_
     // transfer memory: DEVICE_VISIBLE permits handle-based stream copies without implying direct device access.
     if (host_visible) {
         memory_type = HRX_MEMORY_TYPE_HOST_LOCAL | HRX_MEMORY_TYPE_DEVICE_VISIBLE;
-        if (direct_host_binding) {
-            memory_type |= HRX_MEMORY_TYPE_HOST_COHERENT;
-        }
+        // engine#315: a host-visible buffer is read directly by the CPU — the generic
+        // ggml_backend_tensor_get_async short-circuits for host buffers, so no backend copy runs and
+        // nothing in the transfer path can order the device's write against that read. Coherence must
+        // therefore not be conditional on the direct-binding knob: with that knob off (the default) the
+        // reader can observe unwritten bytes, which is the observed whole-vocabulary NaN. Same memory
+        // class, one unconditional bit, no new allocation path and no configuration change.
+        memory_type |= HRX_MEMORY_TYPE_HOST_COHERENT;
     }
     hrx_buffer_params_t params = {
         memory_type,
```
