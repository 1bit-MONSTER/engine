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

     engine#315 evidence, stored as Markdown because the repository's CI requires a copyright notice on
     every file and a raw .txt log cannot carry one. Verbatim output. -->

## The HRX test suite at the pin's kernel

```
# engine#315: the HRX test suite at the pin's kernel - full baseline, 9 runnable tests, all green
# run 2026-10-09T22:55-03:00 on strixhalo, build dir ~/wt/v315-build (fork at 26330cc20; ggml/src/ggml-hrx identical to the pin e44c9d01a)
# env: IREE_HAL_AMDGPU_LIBHSA_PATH=/opt/rocm-therock/lib/python3.14/site-packages/_rocm_sdk_core/lib/libhsa-runtime64.so.1 LD_LIBRARY_PATH=<build>/bin
# each process run under systemd-run --user --scope -p MemoryMax=4G so a runaway cannot OOM the foreign session holding the box
# two tests do NOT compile at this revision (pre-existing, unrelated to the masked-V extension):
#   test-hrx-ops.cpp: fatal error: dispatch_registration/common/dispatch-activation-publication.h file not found
#   test-hrx-loom-jit.cpp: API drift vs the HRX revision in use (KernelSpecialization members)

Running 5 HRX buffer test case(s) with 1 worker process(es)
[PASS]    1/5 backend_buffer                                                                              216 ms
[PASS]    2/5 host_buffer                                                                                 158 ms
[PASS]    3/5 host_transfer                                                                               171 ms
[PASS]    4/5 host_staging                                                                                160 ms
[PASS]    5/5 host_weight_cache                                                                           144 ms

HRX buffer test summary: 5 passed, 0 failed, 5 total, 852 ms
MUL_MAT    mxfp4 k=2880 tokens=2 rows=0  loom_libs:ggml_mul_mat_f32_f32_wmma          nmse=1.26e-05 OK
MUL_MAT    mxfp4 k=2880 tokens=3 rows=0  loom_libs:ggml_mul_mat_f32_f32_wmma          nmse=1.44e-05 OK
MUL_MAT    mxfp4 k=2880 tokens=4 rows=0  loom_libs:ggml_mul_mat_f32_f32_wmma          nmse=1.63e-05 OK
MUL_MAT    mxfp4 k=2880 tokens=8 rows=0  loom_libs:ggml_mul_mat_f32_f32_wmma          nmse=1.52e-05 OK
MUL_MAT_ID mxfp4 k=2880 tokens=1 rows=1  loom_libs:ggml_mul_mat_id_decode_f32_wave64  nmse=1.56e-05 OK
MUL_MAT_ID mxfp4 k=2880 tokens=2 rows=1  loom_libs:ggml_mul_mat_id_decode_f32_wave64  nmse=1.24e-05 OK
MUL_MAT_ID mxfp4 k=2880 tokens=1 rows=4  loom_libs:ggml_mul_mat_id_decode_f32_wave64  nmse=1.58e-05 OK
MUL_MAT_ID mxfp4 k=2880 tokens=2 rows=4  loom_libs:ggml_mul_mat_id_decode_f32_wave64  nmse=1.43e-05 OK
MUL_MAT    mxfp4 k=1152 tokens=2 rows=0  loom_libs:ggml_mul_mat_f32_f32_wmma          nmse=1.5e-05 OK
MUL_MAT    mxfp4 k=1152 tokens=3 rows=0  loom_libs:ggml_mul_mat_f32_f32_wmma          nmse=1.45e-05 OK
MUL_MAT    mxfp4 k=1152 tokens=4 rows=0  loom_libs:ggml_mul_mat_f32_f32_wmma          nmse=1.33e-05 OK
MUL_MAT    mxfp4 k=1152 tokens=8 rows=0  loom_libs:ggml_mul_mat_f32_f32_wmma          nmse=1.42e-05 OK
MUL_MAT_ID mxfp4 k=1152 tokens=1 rows=1  loom_libs:ggml_mul_mat_id_decode_f32_wave64  nmse=1.36e-05 OK
MUL_MAT_ID mxfp4 k=1152 tokens=2 rows=1  loom_libs:ggml_mul_mat_id_decode_f32_wave64  nmse=1.23e-05 OK
MUL_MAT_ID mxfp4 k=1152 tokens=1 rows=4  loom_libs:ggml_mul_mat_id_decode_f32_wave64  nmse=1.4e-05 OK
MUL_MAT_ID mxfp4 k=1152 tokens=2 rows=4  loom_libs:ggml_mul_mat_id_decode_f32_wave64  nmse=1.3e-05 OK
MUL_MAT    mxfp4 k=2048 tokens=2 rows=0  loom_libs:ggml_kquant_mul_mat_decode_tokens_f32 nmse=1.16e-05 OK
MUL_MAT    mxfp4 k=2048 tokens=3 rows=0  loom_libs:ggml_kquant_mul_mat_decode_tokens_f32 nmse=1.21e-05 OK
MUL_MAT    mxfp4 k=2048 tokens=4 rows=0  loom_libs:ggml_kquant_mul_mat_decode_tokens_f32 nmse=1.44e-05 OK
MUL_MAT    mxfp4 k=2048 tokens=8 rows=0  loom_libs:ggml_kquant_mul_mat_decode_tokens_f32 nmse=1.31e-05 OK
MUL_MAT_ID mxfp4 k=2048 tokens=1 rows=1  loom_libs:ggml_mul_mat_id_decode_f32_wave64  nmse=1.49e-05 OK
MUL_MAT_ID mxfp4 k=2048 tokens=2 rows=1  loom_libs:ggml_mul_mat_id_decode_f32_wave64  nmse=1.36e-05 OK
MUL_MAT_ID mxfp4 k=2048 tokens=1 rows=4  loom_libs:ggml_mul_mat_id_decode_f32_wave64  nmse=1.52e-05 OK
MUL_MAT_ID mxfp4 k=2048 tokens=2 rows=4  loom_libs:ggml_mul_mat_id_decode_f32_wave64  nmse=1.42e-05 OK
MUL_MAT    q8_0  k=2880 tokens=2 rows=0  loom_libs:ggml_mul_mat_f32_f32_wmma          nmse=2.07e-05 OK
MUL_MAT    q8_0  k=2880 tokens=3 rows=0  loom_libs:ggml_mul_mat_f32_f32_wmma          nmse=1.68e-05 OK
MUL_MAT    q8_0  k=2880 tokens=4 rows=0  loom_libs:ggml_mul_mat_f32_f32_wmma          nmse=1.45e-05 OK
MUL_MAT    q8_0  k=2880 tokens=8 rows=0  loom_libs:ggml_mul_mat_f32_f32_wmma          nmse=1.29e-05 OK
MUL_MAT_ID q8_0  k=2880 tokens=1 rows=1  loom_libs:ggml_mul_mat_id_decode_f32_wave64  nmse=1.47e-05 OK
MUL_MAT_ID q8_0  k=2880 tokens=2 rows=1  loom_libs:ggml_mul_mat_id_decode_f32_wave64  nmse=1.44e-05 OK
MUL_MAT_ID q8_0  k=2880 tokens=1 rows=4  loom_libs:ggml_mul_mat_id_decode_f32_wave64  nmse=1.36e-05 OK
MUL_MAT_ID q8_0  k=2880 tokens=2 rows=4  loom_libs:ggml_mul_mat_id_decode_f32_wave64  nmse=1.51e-05 OK
MUL_MAT    q8_0  k=1152 tokens=2 rows=0  loom_libs:ggml_mul_mat_f32_f32_wmma          nmse=1.43e-05 OK
MUL_MAT    q8_0  k=1152 tokens=3 rows=0  loom_libs:ggml_mul_mat_f32_f32_wmma          nmse=1.42e-05 OK
MUL_MAT    q8_0  k=1152 tokens=4 rows=0  loom_libs:ggml_mul_mat_f32_f32_wmma          nmse=1.33e-05 OK
MUL_MAT    q8_0  k=1152 tokens=8 rows=0  loom_libs:ggml_mul_mat_f32_f32_wmma          nmse=1.5e-05 OK
MUL_MAT_ID q8_0  k=1152 tokens=1 rows=1  loom_libs:ggml_mul_mat_id_decode_f32_wave64  nmse=1.25e-05 OK
MUL_MAT_ID q8_0  k=1152 tokens=2 rows=1  loom_libs:ggml_mul_mat_id_decode_f32_wave64  nmse=1.48e-05 OK
MUL_MAT_ID q8_0  k=1152 tokens=1 rows=4  loom_libs:ggml_mul_mat_id_decode_f32_wave64  nmse=1.74e-05 OK
MUL_MAT_ID q8_0  k=1152 tokens=2 rows=4  loom_libs:ggml_mul_mat_id_decode_f32_wave64  nmse=1.4e-05 OK
MUL_MAT    q8_0  k=2048 tokens=2 rows=0  loom_libs:ggml_kquant_mul_mat_decode_tokens_f32 nmse=1.35e-05 OK
MUL_MAT    q8_0  k=2048 tokens=3 rows=0  loom_libs:ggml_kquant_mul_mat_decode_tokens_f32 nmse=1.49e-05 OK
MUL_MAT    q8_0  k=2048 tokens=4 rows=0  loom_libs:ggml_kquant_mul_mat_decode_tokens_f32 nmse=1.25e-05 OK
MUL_MAT    q8_0  k=2048 tokens=8 rows=0  loom_libs:ggml_kquant_mul_mat_decode_tokens_f32 nmse=1.56e-05 OK
MUL_MAT_ID q8_0  k=2048 tokens=1 rows=1  loom_libs:ggml_mul_mat_id_decode_f32_wave64  nmse=1.37e-05 OK
MUL_MAT_ID q8_0  k=2048 tokens=2 rows=1  loom_libs:ggml_mul_mat_id_decode_f32_wave64  nmse=1.49e-05 OK
MUL_MAT_ID q8_0  k=2048 tokens=1 rows=4  loom_libs:ggml_mul_mat_id_decode_f32_wave64  nmse=1.71e-05 OK
MUL_MAT_ID q8_0  k=2048 tokens=2 rows=4  loom_libs:ggml_mul_mat_id_decode_f32_wave64  nmse=1.41e-05 OK
MUL_MAT    q4_0  k=2880 tokens=2 rows=0  loom_libs:ggml_mul_mat_f32_f32_wmma          nmse=1.36e-05 OK
MUL_MAT    q4_0  k=2880 tokens=3 rows=0  loom_libs:ggml_mul_mat_f32_f32_wmma          nmse=1.53e-05 OK
MUL_MAT    q4_0  k=2880 tokens=4 rows=0  loom_libs:ggml_mul_mat_f32_f32_wmma          nmse=1.49e-05 OK
MUL_MAT    q4_0  k=2880 tokens=8 rows=0  loom_libs:ggml_mul_mat_f32_f32_wmma          nmse=1.36e-05 OK
MUL_MAT_ID q4_0  k=2880 tokens=1 rows=1  loom_libs:ggml_mul_mat_id_decode_f32_wave64  nmse=1.48e-05 OK
MUL_MAT_ID q4_0  k=2880 tokens=2 rows=1  loom_libs:ggml_mul_mat_id_decode_f32_wave64  nmse=1.46e-05 OK
MUL_MAT_ID q4_0  k=2880 tokens=1 rows=4  loom_libs:ggml_mul_mat_id_decode_f32_wave64  nmse=1.42e-05 OK
MUL_MAT_ID q4_0  k=2880 tokens=2 rows=4  loom_libs:ggml_mul_mat_id_decode_f32_wave64  nmse=1.24e-05 OK
MUL_MAT    q4_0  k=1152 tokens=2 rows=0  loom_libs:ggml_mul_mat_f32_f32_wmma          nmse=1.45e-05 OK
MUL_MAT    q4_0  k=1152 tokens=3 rows=0  loom_libs:ggml_mul_mat_f32_f32_wmma          nmse=1.21e-05 OK
MUL_MAT    q4_0  k=1152 tokens=4 rows=0  loom_libs:ggml_mul_mat_f32_f32_wmma          nmse=1.58e-05 OK
MUL_MAT    q4_0  k=1152 tokens=8 rows=0  loom_libs:ggml_mul_mat_f32_f32_wmma          nmse=1.38e-05 OK
MUL_MAT_ID q4_0  k=1152 tokens=1 rows=1  loom_libs:ggml_mul_mat_id_decode_f32_wave64  nmse=1.5e-05 OK
MUL_MAT_ID q4_0  k=1152 tokens=2 rows=1  loom_libs:ggml_mul_mat_id_decode_f32_wave64  nmse=1.42e-05 OK
MUL_MAT_ID q4_0  k=1152 tokens=1 rows=4  loom_libs:ggml_mul_mat_id_decode_f32_wave64  nmse=1.36e-05 OK
MUL_MAT_ID q4_0  k=1152 tokens=2 rows=4  loom_libs:ggml_mul_mat_id_decode_f32_wave64  nmse=1.6e-05 OK
MUL_MAT    q4_0  k=2048 tokens=2 rows=0  loom_libs:ggml_mul_mat_f32_f32_wmma          nmse=1.66e-05 OK
MUL_MAT    q4_0  k=2048 tokens=3 rows=0  loom_libs:ggml_mul_mat_f32_f32_wmma          nmse=1.27e-05 OK
MUL_MAT    q4_0  k=2048 tokens=4 rows=0  loom_libs:ggml_mul_mat_f32_f32_wmma          nmse=1.47e-05 OK
MUL_MAT    q4_0  k=2048 tokens=8 rows=0  loom_libs:ggml_mul_mat_f32_f32_wmma          nmse=1.25e-05 OK
MUL_MAT_ID q4_0  k=2048 tokens=1 rows=1  loom_libs:ggml_mul_mat_id_decode_f32_wave64  nmse=1.29e-05 OK
MUL_MAT_ID q4_0  k=2048 tokens=2 rows=1  loom_libs:ggml_mul_mat_id_decode_f32_wave64  nmse=1.38e-05 OK
MUL_MAT_ID q4_0  k=2048 tokens=1 rows=4  loom_libs:ggml_mul_mat_id_decode_f32_wave64  nmse=1.49e-05 OK
MUL_MAT_ID q4_0  k=2048 tokens=2 rows=4  loom_libs:ggml_mul_mat_id_decode_f32_wave64  nmse=1.3e-05 OK
test-hrx-decode-stride: 72 cases, 0 failures
repack: repack tensor  with mxfp4_8x8
repack: repack tensor  with mxfp4_8x8
repack: repack tensor  with mxfp4_8x8
  placement: MUL_MAT_ID=CPU ADD_ID=CPU MUL_MAT_ID=CPU ADD_ID=CPU SWIGLU_OAI=CPU MUL_MAT_ID=CPU ADD_ID=CPU
experts=CPU_REPACK tokens= 1 nmse=1.51e-15 guard=ok OK
repack: repack tensor  with mxfp4_8x8
repack: repack tensor  with mxfp4_8x8
repack: repack tensor  with mxfp4_8x8
  placement: MUL_MAT_ID=CPU ADD_ID=CPU MUL_MAT_ID=CPU ADD_ID=CPU SWIGLU_OAI=CPU MUL_MAT_ID=CPU ADD_ID=CPU
experts=CPU_REPACK tokens= 3 nmse=1.34e-15 guard=ok OK
repack: repack tensor  with mxfp4_8x8
repack: repack tensor  with mxfp4_8x8
repack: repack tensor  with mxfp4_8x8
  placement: MUL_MAT_ID=CPU ADD_ID=CPU MUL_MAT_ID=CPU ADD_ID=CPU SWIGLU_OAI=CPU MUL_MAT_ID=CPU ADD_ID=CPU
experts=CPU_REPACK tokens=12 nmse=1.54e-15 guard=ok OK
repack: repack tensor  with mxfp4_8x8
repack: repack tensor  with mxfp4_8x8
repack: repack tensor  with mxfp4_8x8
  placement: MUL_MAT_ID=CPU ADD_ID=CPU MUL_MAT_ID=CPU ADD_ID=CPU SWIGLU_OAI=CPU MUL_MAT_ID=CPU ADD_ID=CPU
experts=CPU_REPACK tokens=40 nmse=1.38e-15 guard=ok OK
  placement: MUL_MAT_ID=HRX ADD_ID=HRX MUL_MAT_ID=HRX ADD_ID=HRX SWIGLU_OAI=HRX MUL_MAT_ID=HRX ADD_ID=HRX
experts=CPU        tokens= 1 nmse=9.47e-05 guard=ok OK
  placement: MUL_MAT_ID=HRX ADD_ID=HRX MUL_MAT_ID=HRX ADD_ID=HRX SWIGLU_OAI=HRX MUL_MAT_ID=HRX ADD_ID=HRX
experts=CPU        tokens= 3 nmse=0.000107 guard=ok OK
  placement: MUL_MAT_ID=HRX ADD_ID=HRX MUL_MAT_ID=HRX ADD_ID=HRX SWIGLU_OAI=HRX MUL_MAT_ID=HRX ADD_ID=HRX
experts=CPU        tokens=12 nmse=0.000101 guard=ok OK
  placement: MUL_MAT_ID=HRX ADD_ID=HRX MUL_MAT_ID=HRX ADD_ID=HRX SWIGLU_OAI=HRX MUL_MAT_ID=HRX ADD_ID=HRX
experts=CPU        tokens=40 nmse=0.000101 guard=ok OK
test-hrx-moe-split: all cases OK
mxfp4 k= 2880 tokens= 1  loom_libs:ggml_mul_mat_id_decode_f32_wave64      nmse=1.57e-05  OK
mxfp4 k= 2880 tokens= 7  loom_libs:ggml_mul_mat_id_f32_f32_wmma           nmse=1.34e-05  OK
mxfp4 k= 2880 tokens=40  loom_libs:ggml_mul_mat_id_f32_f32_wmma           nmse=1.41e-05  OK
mxfp4 k= 1152 tokens= 1  loom_libs:ggml_mul_mat_id_decode_f32_wave64      nmse=1.54e-05  OK
mxfp4 k= 1152 tokens= 7  loom_libs:ggml_mul_mat_id_f32_f32_wmma           nmse=1.61e-05  OK
mxfp4 k= 1152 tokens=40  loom_libs:ggml_mul_mat_id_f32_f32_wmma           nmse=1.42e-05  OK
mxfp4 k= 2816 tokens= 1  loom_libs:ggml_mul_mat_id_decode_f32_wave64      nmse=1.46e-05  OK
mxfp4 k= 2816 tokens= 7  loom_libs:ggml_mul_mat_id_f32_f32_wmma           nmse=1.44e-05  OK
mxfp4 k= 2816 tokens=40  loom_libs:ggml_mul_mat_id_f32_f32_wmma           nmse=1.45e-05  OK
q8_0  k= 2880 tokens= 1  loom_libs:ggml_mul_mat_id_decode_f32_wave64      nmse=1.43e-05  OK
q8_0  k= 2880 tokens= 7  loom_libs:ggml_mul_mat_id_f32_f32_wmma           nmse=1.5e-05  OK
q8_0  k= 2880 tokens=40  loom_libs:ggml_mul_mat_id_f32_f32_wmma           nmse=1.43e-05  OK
q8_0  k= 1152 tokens= 1  loom_libs:ggml_mul_mat_id_decode_f32_wave64      nmse=1.29e-05  OK
q8_0  k= 1152 tokens= 7  loom_libs:ggml_mul_mat_id_f32_f32_wmma           nmse=1.5e-05  OK
q8_0  k= 1152 tokens=40  loom_libs:ggml_mul_mat_id_f32_f32_wmma           nmse=1.46e-05  OK
q8_0  k= 2816 tokens= 1  loom_libs:ggml_mul_mat_id_decode_f32_wave64      nmse=1.38e-05  OK
q8_0  k= 2816 tokens= 7  loom_libs:ggml_mul_mat_id_f32_f32_wmma           nmse=1.5e-05  OK
q8_0  k= 2816 tokens=40  loom_libs:ggml_mul_mat_id_f32_f32_wmma           nmse=1.44e-05  OK
q4_0  k= 2880 tokens= 1  loom_libs:ggml_mul_mat_id_decode_f32_wave64      nmse=1.26e-05  OK
q4_0  k= 2880 tokens= 7  loom_libs:ggml_mul_mat_id_f32_f32_wmma           nmse=1.34e-05  OK
q4_0  k= 2880 tokens=40  loom_libs:ggml_mul_mat_id_f32_f32_wmma           nmse=1.43e-05  OK
q4_0  k= 1152 tokens= 1  loom_libs:ggml_mul_mat_id_decode_f32_wave64      nmse=1.34e-05  OK
q4_0  k= 1152 tokens= 7  loom_libs:ggml_mul_mat_id_f32_f32_wmma           nmse=1.41e-05  OK
q4_0  k= 1152 tokens=40  loom_libs:ggml_mul_mat_id_f32_f32_wmma           nmse=1.38e-05  OK
q4_0  k= 2816 tokens= 1  loom_libs:ggml_mul_mat_id_decode_f32_wave64      nmse=1.33e-05  OK
q4_0  k= 2816 tokens= 7  loom_libs:ggml_mul_mat_id_f32_f32_wmma           nmse=1.4e-05  OK
q4_0  k= 2816 tokens=40  loom_libs:ggml_mul_mat_id_f32_f32_wmma           nmse=1.41e-05  OK
test-hrx-mul-mat-id-k32: all cases OK
test-hrx-attention-sink: disjoint output accepted, output aliasing the query refused
n=64    rows=1     rep 0: max rel error 9.69e-08 ok
n=64    rows=1     rep 1: max rel error 9.69e-08 ok
n=64    rows=1     rep 2: max rel error 9.69e-08 ok
n=64    rows=1     rep 3: max rel error 9.69e-08 ok
n=64    rows=7     rep 0: max rel error 9.74e-08 ok
n=64    rows=7     rep 1: max rel error 9.74e-08 ok
n=64    rows=7     rep 2: max rel error 9.74e-08 ok
n=64    rows=7     rep 3: max rel error 9.74e-08 ok
n=1024  rows=1     rep 0: max rel error 1.70e-07 ok
n=1024  rows=1     rep 1: max rel error 1.70e-07 ok
n=1024  rows=1     rep 2: max rel error 1.70e-07 ok
n=1024  rows=1     rep 3: max rel error 1.70e-07 ok
n=1024  rows=17    rep 0: max rel error 2.12e-07 ok
n=1024  rows=17    rep 1: max rel error 2.12e-07 ok
n=1024  rows=17    rep 2: max rel error 2.12e-07 ok
n=1024  rows=17    rep 3: max rel error 2.12e-07 ok
n=1024  rows=8704  rep 0: max rel error 0.00e+00 ok
n=1024  rows=8704  rep 1: max rel error 0.00e+00 ok
n=1024  rows=8704  rep 2: max rel error 0.00e+00 ok
n=1024  rows=8704  rep 3: max rel error 0.00e+00 ok
n=4096  rows=3     rep 0: max rel error 0.00e+00 ok
n=4096  rows=3     rep 1: max rel error 0.00e+00 ok
n=4096  rows=3     rep 2: max rel error 0.00e+00 ok
n=4096  rows=3     rep 3: max rel error 0.00e+00 ok
test-hrx-hadamard: PASS
dispatch: loom_libs:ggml_get_rows_f32
test-hrx-mxfp4: 7 rows x 256 values bit-exact (448 values with a subnormal scale flushed to zero)
```
