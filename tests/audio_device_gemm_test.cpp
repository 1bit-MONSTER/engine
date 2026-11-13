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
// tests/audio_device_gemm_test.cpp -- the decision layer around one encoder projection on the device (npu/forward/include/
// audio_device_gemm.h, engine#1783), driven against a FAKE target: no device, no container, no kernels.
//
// The three things it pins are the three the real path gets wrong silently: the target must be asked for the STAGED width
// (N768 is named N1024), the caller must get its OWN width back (de-padded per row, not a staged stride), and every refusal
// (wrong output size, too many rows, no gemv, an unnamed projection, a target failure) must say why by name.
#include <cstdio>
#include <string>
#include <vector>
#include "audio_device_gemm.h"

using namespace onebit::audio;
namespace t = ::npu::target;

static int failures = 0;
static void check(bool ok, const std::string& what) {
    std::printf("%s %s\n", ok ? "ok  " : "FAIL", what.c_str());
    if (!ok) ++failures;
}

struct FakeTarget : t::NpuTarget {
    t::Caps c;
    bool supports_gemv = true;
    bool fail = false;
    size_t wrote = 0;            // what the fake claims to have written; 0 = "rows x staged_n"
    t::GemvArgs seen{};
    const char* name() const override { return "fake"; }
    bool open(const onebit::npu::plan::Plan&, const std::string&, const char*, std::string&) override { return true; }
    bool load_block(int, int, std::string&) override { return true; }
    t::Caps caps() const override { return c; }
    bool supports(t::Op op) const override { return op == t::Op::gemv && supports_gemv; }
    bool gemv(const t::GemvArgs& a, std::vector<float>& out, std::string& err) override {
        seen = a;
        if (fail) { err = "kernel not resident"; return false; }
        const size_t n = wrote ? wrote : size_t(a.M) * size_t(a.N);
        out.assign(n, 0.0f);
        for (size_t i = 0; i < n; ++i) out[i] = float(i);   // recognisable pattern
        return true;
    }
};

int main() {
    FakeTarget ft;
    ft.c.max_rows = 16;
    ft.c.prefill_rows = 128;
    std::vector<float> x(128 * 768, 1.0f);
    std::vector<float> out;
    std::string why;

    // 1. the de-pad: O at k768 n768 is named N1024, so the fake writes 128x1024 and the caller must get 128x768
    DeviceProjection p;
    p.role = 1; p.layer = 3; p.rows = 128; p.k = 768; p.n = 768; p.x = x.data(); p.tag = "O";
    check(device_projection(ft, p, &out, &why), "a staged O projection runs");
    check(ft.seen.N == 1024 && ft.seen.K == 768, "the target is asked for the STAGED width (N1024), not the model's 768");
    check(ft.seen.M == 128 && ft.seen.rows, "a 128-row launch is the wide body");
    check(out.size() == size_t(128) * 768, "the caller gets its own width back (128 x 768), de-padded");
    check(out[1] == 1.0f && out[767] == 767.0f && out[768] == 1024.0f,
          "the de-pad drops the padding column: row 1 starts at staged column 1024");

    // 2. a target that writes the wrong size is refused, not repaired
    ft.wrote = 128 * 768;
    check(!device_projection(ft, p, &out, &why) && why.find("the target and the plan disagree") != std::string::npos,
          "an output that is not rows x staged_n is refused by name");
    ft.wrote = 0;

    // 3. more rows than the target launches in one call is refused with both numbers
    p.rows = 129;
    check(!device_projection(ft, p, &out, &why) && why.find("prefill_rows=128") != std::string::npos,
          "rows beyond the prefill launch is refused and names the limit");
    p.rows = 128;

    // 4. a target without gemv is refused, naming the target
    ft.supports_gemv = false;
    check(!device_projection(ft, p, &out, &why) && why.find("\"fake\"") != std::string::npos,
          "a target without gemv is refused, naming it");
    ft.supports_gemv = true;

    // 5. a target failure keeps the target's own words
    ft.fail = true;
    check(!device_projection(ft, p, &out, &why) && why == "kernel not resident", "a target failure keeps its reason");
    ft.fail = false;

    // 6. an unnamed projection with no tag is refused (the log must be answerable)
    p.tag = nullptr; p.k = 777; p.n = 5;
    check(!device_projection(ft, p, &out, &why) && why.find("needs a tag") != std::string::npos,
          "an unnamed shape with no tag is refused");

    // 7. the decode tier: 16 rows is NOT the wide body
    p.k = 768; p.n = 768; p.rows = 16; p.tag = "O";
    check(device_projection(ft, p, &out, &why) && !ft.seen.rows, "a 16-row launch is the per-row body");

    std::printf(failures ? "audio device gemm: %d FAILED\n" : "audio device gemm: all cases ok\n", failures);
    return failures ? 1 : 0;
}
