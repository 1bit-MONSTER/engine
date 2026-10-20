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

# GLM never dispatches the only kernel that changed between the clean bound and the pin

This closes the dilemma section 52 left open, with a command and its output rather than an argument.

## The two facts that meet

**Fact 1 (section 52).** The whole HRX backend differs by exactly one file between the recorded clean
lower bound `f2099e9b7` and the pin the in-regime arms run (`e44c9d01a`):

```sh
ssh strixhalo 'cd ~/wt/llama-315copy && git diff --stat f2099e9b7..HEAD -- ggml/src/ggml-hrx/'
#  .../flash_attention_decode_split_f32_f16_wmma.loom | 104 ++++++++++++++++++++-
#  1 file changed, 102 insertions(+), 2 deletions(-)
```

That file is the decode-split flash-attention kernel, and section 52 therefore kept it as "the only
candidate for *what changed*".

**Fact 2 (this measurement).** No in-regime arm ever dispatches it. Counting the kernel the HRX
dispatcher names across every arm log on the box:

```sh
$ grep -rho "ggml_flash_attention[a-z0-9_]*" ~/wt/rt-det/*/server.log | sort | uniq -c | sort -rn
# (no such token — the logs name the kernel with a `common.` prefix and a mode suffix)

$ grep -rh "flash_attention" ~/wt/rt-det/*/server.log | head -2
ggml_hrx dispatch: common.flash_attention_f32_f16_wmma (single)
ggml_hrx dispatch: common.flash_attention_f32_f16_wmma (single)

$ grep -rl "decode_split" ~/wt/rt-det/*/server.log | wc -l
0
```

**944 arm logs, 60 of which name a flash-attention kernel, and every one of the 60 names
`common.flash_attention_f32_f16_wmma`. None names `decode_split`.** The decode path GLM actually runs
is the *non-split* FA kernel; the changed file is the *split* one.

## What that means

The only difference between the configuration recorded as clean and the configuration recorded as
faulting is **not on GLM's execution path**. So the pin move cannot have introduced the fault, and
section 52's dilemma ("either older than the clean bound, or triggered by the one thing that did
change") resolves to its **first** branch: the defect is older than the clean bound, and the "clean"
label was a statement about **rate, not absence** — which is what section 52 itself suspected and now
has direct evidence for.

This is a *constraint*, and it is falsifiable in one line: **any arm log containing `decode_split` as
a dispatched kernel falsifies it.**

## The kernel cannot even be instantiated at GLM's head geometry

As a second, independent line, the MLA race harness section 52 named as the cheap separating test was
built and run at GLM's real geometry. It does not reach a race, because the changed kernel is rejected
by Loom's own target check:

```sh
$ ./glm-mla-race-fa 1 2048 32 300 1
HRX Loom JIT Loom compilation failed failed: Loom compilation failed
  diagnostic[0] TARGET/059: target 'amdgpu-rdna3-5' export
  'ggml_flash_attention_decode_split_f32_f16_wmma_next_q8' config 'amdgpu.rdna3_5.core' rejected
  'index.assume' in '@ggml_flash_attention_decode_split_f32_f16_wmma_next_q8': predicate 'range' on
  'query_heads_per_key_value_head0' requires range [1, 16] but known facts prove range [20, 20]
```

GLM-4.7-Flash is MLA with **20 query heads over 1 latent KV head**, and the split kernel asserts
`query_heads_per_key_value_head ∈ [1, 16]`. At this model's geometry that predicate is false, so the
kernel cannot compile at all — consistent with fact 2, and independently confirming section 38's
warning that the sweeps which cleared this kernel were run at shapes that are **not** this model's
geometry.

## Honest bounds

* The harness's *selector* differs from the real model's: the synthetic graph at `q=20, kv=1` selects
  the split kernel, while the real model at the same nominal head counts selects the non-split one.
  What the harness shows is that the split kernel is uninstantiable at `q/kv = 20`; it is **fact 2**
  (from the arm logs), not the harness, that establishes which kernel GLM dispatches.
* The harness and build are new this session (`~/wt/glm-mla-race-fa`, built against
  `~/wt/315fp-build/bin` and the `~/wt/llama-315copy` headers, both at the pin). The source is
  `tools/glm-mla-race.cpp` with the MLA de-absorption `MUL_MAT` removed, because that node is
  unsupported by this HRX build and aborted the original harness before it could run — which is why
  the recorded version was "compiled and never run".
* 60 of 944 logs naming a FA kernel is a small fraction because most arms' logs carry no dispatch
  lines at all; the 60 are the ones that do, and the claim is about those.
