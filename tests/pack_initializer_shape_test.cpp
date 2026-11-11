// Copyright 2026 bong-water-water-bong
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// pack_initializer_shape_test -- the shape rule for a 1.8 external-data initializer (engine#1503). No ONNX parser, no
// pack, no device: the rule is a function of (name, byte length, element size, the model's declared sizes), so every arm
// is a hand-computed value. The refusals matter more than the accepts -- a shape inferred wrongly imports cleanly and
// serves garbage, so the cases that must NOT resolve are the ones worth pinning.
#include "pack_initializer_shape.h"

#include <cstdio>
#include <vector>

using onebit::npu::InitShape;
using onebit::npu::resolve_init_shape;

static int failures = 0;
static void check(bool ok, const char* what) {
    std::printf("%s  %s\n", ok ? "ok  " : "FAIL", what);
    if (!ok) ++failures;
}

int main() {
    // The measured pack: Llama-3.2-1B 1.8 hybrid, hidden 2048, IM 8192, vocab 128256, bf16 elements.
    const std::vector<int64_t> known{2048, 8192, 128256};
    const int bf16 = 2;

    // (i) the refusal this reader exists to fix: a layernorm weight is 2048 bf16 = 4096 B, one of the known sizes.
    {
        const InitShape s = resolve_init_shape("model.layers.0.input_layernorm.weight", 2048 * bf16, bf16, known);
        check(s.ok && s.ndim == 1 && s.d0 == 2048, "(i) a layernorm weight resolves to [2048]");
    }
    // (ii) a 2-D weight whose dimension is unambiguous: 2048 x 3 elements -- only 2048 divides it among the known
    //     sizes, so the shape is decided. (2048 x 8192 would NOT be unambiguous: both are declared sizes, which is arm
    //     (v) below, and my first version of this arm asserted the opposite and was wrong.)
    {
        const InitShape s = resolve_init_shape("model.layers.0.mlp.down_proj.weight", 2048 * 3LL * bf16, bf16, known);
        check(s.ok && s.ndim == 2 && s.d0 == 2048 && s.d1 == 3, "(ii) a weight only one declared size divides resolves");
    }
    // (iii) a length that is not a whole number of elements is refused, and says so with both numbers.
    {
        const InitShape s = resolve_init_shape("model.layers.0.mlp.down_proj.weight", 4097, bf16, known);
        check(!s.ok && s.why.find("4097") != std::string::npos && s.why.find("2-byte") != std::string::npos,
              "(iii) an odd byte length is refused, naming the length and the element size");
    }
    // (iv) a norm whose length matches none of the declared sizes is refused: it is not a norm of this model.
    {
        const InitShape s = resolve_init_shape("model.layers.0.input_layernorm.weight", 3000 * bf16, bf16, known);
        check(!s.ok && s.why.find("matches none") != std::string::npos, "(iv) a norm of an undeclared size is refused");
    }
    // (v) the case the rule exists for: a weight consistent with TWO declared dimensions must refuse, not pick one.
    //     2048*8192 elements is divisible by both 2048 and 8192, so a weight of that length is ambiguous.
    {
        const InitShape s = resolve_init_shape("model.layers.0.some.weight", 2048 * 8192LL * bf16, bf16, known);
        check(!s.ok && s.why.find("several declared dimensions") != std::string::npos,
              "(v) a weight consistent with two dimensions is REFUSED as ambiguous, naming them");
    }
    // (vi) a weight with no dividing declared dimension is refused.
    {
        const InitShape s = resolve_init_shape("model.layers.0.some.weight", 1000 * bf16, bf16, known);
        check(!s.ok && s.why.find("no declared dimension") != std::string::npos,
              "(vi) a weight with no dividing dimension is refused");
    }
    // (vii) a bias is 1-D by name and takes a known size; a bias of an undeclared size is refused like a norm.
    {
        const InitShape a = resolve_init_shape("model.layers.0.attn.q_proj.bias", 2048 * bf16, bf16, known);
        const InitShape b = resolve_init_shape("model.layers.0.attn.q_proj.bias", 777 * bf16, bf16, known);
        check(a.ok && a.ndim == 1 && a.d0 == 2048 && !b.ok, "(vii) a bias is 1-D like a norm, and refuses an undeclared size");
    }
    // (viii) a non-positive length or element size is refused rather than divided by zero.
    {
        const InitShape a = resolve_init_shape("x.norm.weight", 0, bf16, known);
        const InitShape b = resolve_init_shape("x.norm.weight", 4096, 0, known);
        check(!a.ok && !b.ok, "(viii) a zero length or zero element size is refused, not divided by");
    }


    // ── the external_data reference (the other half of reading a 1.8 initializer) ────────────────────────────────────
    using onebit::npu::resolve_external_ref;
    // (ix) the ordinary form: a location, an offset and a length.
    {
        const auto r = resolve_external_ref({"location", "model.onnx.data", "offset", "542113792", "length", "4096"});
        check(r.ok && r.location == "model.onnx.data" && r.offset == 542113792 && r.length == 4096,
              "(ix) location/offset/length resolve to the reference the tensor names");
    }
    // (x) a missing length is a refusal, because the default would otherwise be zero bytes read silently.
    {
        const auto r = resolve_external_ref({"location", "model.onnx.data", "offset", "0"});
        check(!r.ok && r.why.find("no \"length\"") != std::string::npos,
              "(x) no length entry is refused rather than read as zero");
    }
    // (xi) a duplicate key is refused: the protobuf allows it and taking the last would be a silent pick.
    {
        const auto r = resolve_external_ref({"offset", "0", "offset", "8", "length", "16"});
        check(!r.ok && r.why.find("duplicate") != std::string::npos, "(xi) a duplicate offset is refused");
    }
    // (xii) a non-numeric or negative-looking value is refused, naming the key and the value.
    {
        const auto a = resolve_external_ref({"offset", "-8", "length", "16"});
        const auto b = resolve_external_ref({"offset", "0", "length", "4x"});
        check(!a.ok && a.why.find("offset") != std::string::npos && !b.ok && b.why.find("length") != std::string::npos,
              "(xii) a signed offset and a non-numeric length are both refused, naming the key");
    }
    // (xiii) an odd key/value count is refused; unknown keys are ignored, as the protobuf permits.
    {
        const auto a = resolve_external_ref({"offset", "0", "length"});
        const auto b = resolve_external_ref({"offset", "0", "length", "16", "checksum", "deadbeef"});
        check(!a.ok && b.ok && b.length == 16, "(xiii) an odd list is refused and an unknown key is ignored");
    }

    if (failures == 0) std::printf("pack initializer shape: 13 arms pass\n");
    return failures ? 1 : 0;
}
