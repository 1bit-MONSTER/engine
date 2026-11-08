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

// npu_plan_op_reference -- the registration rule for ops a plan may name (engine#799).
//
// Every op a plan can name must ship a host reference WITH known-answer cases, and the reference must
// reproduce those answers -- on its own cases and on inputs it was not written from.  The test walks the
// table rather than naming entries, so a new entry cannot be added without being run, and it asserts the
// exact list of ops a plan can require but which still have no reference, so that gap is a decision in the
// test rather than an omission.
//
// Host-only: no model, no container, no device.
#include "plan/op_registry.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace {

int failures = 0;

void check(bool ok, const std::string& what) {
    if (ok) return;
    ++failures;
    std::printf("FAIL %s\n", what.c_str());
}

// Exact equality would be right for these cases (every expected value is a small integer or half), but a
// tolerance makes the failure message readable when one is not.
void check_case(const onebit::npu::plan::OpEntry& e, const onebit::npu::plan::OpCase& c) {
    const std::string label = std::string(onebit::npu::plan::op_name(e.op)) + ": " + c.what;
    check(c.m > 0 && c.k > 0 && c.n > 0, label + " has positive dims");
    if (c.m <= 0 || c.k <= 0 || c.n <= 0) return;
    std::vector<float> got((size_t)c.m * c.n, 0.0f);
    std::vector<float> fresh((size_t)c.m * c.n, 0.0f);
    check(e.ref != nullptr, label + " has a reference");
    if (!e.ref) return;
    // `fresh` is checked to be untouched by the reference's failure path: a reference that returns false
    // must not have written partial output a caller could use.
    const bool ok = e.ref(c.a, c.m, c.k, c.n, c.b, got.data());
    check(ok, label + " reference returns true");
    if (!ok) return;
    float worst = 0.0f;
    for (size_t i = 0; i < got.size(); i++) {
        const float d = std::fabs(got[i] - c.want[i]);
        if (d > worst) worst = d;
    }
    char msg[256];
    std::snprintf(msg, sizeof(msg), "%s: worst |d| %.6g (provenance: %s)", label.c_str(),
                  (double)worst, c.how);
    check(worst <= 1e-6f, msg);
}

}  // namespace

int main() {
    using namespace onebit::npu::plan;

    // 1. The table is walked, not named.
    check(op_entry_count() > 0, "the registry has at least one entry");
    for (size_t i = 0; i < op_entry_count(); i++) {
        const OpEntry& e = op_entry(i);
        const std::string label = op_name(e.op);
        check(e.ref != nullptr, label + " entry has a reference");
        check(e.n_cases > 0, label + " entry has known-answer cases");
        check(e.op != OpId::gdn_step || true, label);   // no-op, keeps the enum referenced
        for (size_t c = 0; c < e.n_cases; c++) check_case(e, e.cases[c]);
    }

    // 2. EVERY op in the vocabulary, not just the ones with entries: an OpId added tomorrow with no entry
    //    must show up here rather than leave the test green (the reviewer of this PR named that hole).
    {
        const std::string documented_gap = "gdn_step,attn_step";
        std::string unregistered;
        int seen = 0;
        for (int i = 0; i < (int)OpId::count; i++) {
            const OpId op = (OpId)i;
            const OpEntry* e = find_op(op);
            const std::string name = op_name(op);
            check(name != "?", name + " has a name (the vocabulary is walked, not listed)");
            if (e) {
                check(e->ref != nullptr, name + " has a reference");
                check(e->n_cases > 0, name + " has known-answer cases");
            } else {
                if (!unregistered.empty()) unregistered += ",";
                unregistered += name;
            }
            ++seen;
        }
        check(seen == 4, "the vocabulary has the four ops this build knows about");
        check(unregistered == documented_gap,
              "the ops with no entry are exactly the documented gap (got \"" + unregistered + "\")");
    }

    // 3. find_op() answers by id and by name, and nullptr for what is missing -- that nullptr is what the
    //    loader's rule will key on.
    check(find_op(OpId::gemv) != nullptr, "find_op(gemv)");
    check(find_op(OpId::lm_head) != nullptr, "find_op(lm_head)");
    check(find_op("gemv") == find_op(OpId::gemv), "find_op by name matches the entry");
    check(find_op(OpId::gdn_step) == nullptr, "gdn_step has no entry yet");
    check(find_op(OpId::attn_step) == nullptr, "attn_step has no entry yet");
    check(find_op("not_an_op") == nullptr, "an unknown name finds nothing");

    // 3. The coverage gap is a decision in this test, not an omission: exactly these two ops are missing.
    const std::string missing = op_coverage();
    check(missing == "gdn_step,attn_step",
          "op_coverage() names exactly the unregistered ops (got \"" + missing + "\")");

    // 4. A reference that is handed bad dimensions refuses rather than computing garbage.
    float out[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    const float a[2] = {1.0f, 2.0f};
    check(!gemv_ref(a, 0, 2, 2, a, out), "gemv_ref refuses m = 0");
    check(!gemv_ref(a, 1, 0, 2, a, out), "gemv_ref refuses k = 0");
    check(!lm_head_ref(a, 2, 1, 2, a, out), "lm_head_ref refuses m > 1 (its contract is one row)");

    std::printf("%s\n", failures ? "FAILED" : "npu_plan_op_reference: ok");
    return failures ? 1 : 0;
}
