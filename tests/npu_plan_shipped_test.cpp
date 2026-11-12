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

// engine#799: the plans checked in under configs/plans/ are reference artefacts, and nothing loaded them.
//
// `docs/npu-plan.md` says four plans are checked in "one per model directory on the build box", and that
// `tools/npu_plan.py <dir> --check` proves a directory's copy still matches - which is true, and which needs the
// model directory. CI has no packs, so the shipped plans had no gate at all: a hand edit, or a generator change,
// could leave one refused by `Plan::load()` and the first evidence would be a lane loading it on the box.
//
// This loads each shipped plan through the SAME loader the forward uses, so the refusal vocabulary is the real
// one (PlanError, "1bit-npu-plan/1"), not a JSON schema written twice. It needs no model, no device and no pack.
//
// What it does NOT prove: that a plan still matches its model directory (that is `--check`, and it needs the
// pack), or that a plan is *right* - only that it is one the engine can read.

#include "plan/plan.h"

#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

namespace {

int bad = 0;

void check(bool ok, const std::string& what) {
    std::printf("%s %s\n", ok ? "ok  " : "FAIL", what.c_str());
    if (!ok) ++bad;
}

}  // namespace

int main(int argc, char** argv) {
    namespace fs = std::filesystem;
    const fs::path dir = argc > 1 ? fs::path(argv[1]) : fs::path("configs/plans");
    std::vector<fs::path> plans;
    std::error_code ec;
    for (const auto& e : fs::directory_iterator(dir, ec)) {
        if (ec) break;
        if (e.is_regular_file() && e.path().extension() == ".json") plans.push_back(e.path());
    }
    if (!plans.empty()) std::sort(plans.begin(), plans.end());

    // An empty directory is not a pass: it means the path is wrong (a test run from a build dir), and a suite that
    // passed over zero plans would be the vacuous green this whole test exists to prevent.
    check(!ec && !plans.empty(),
          "the shipped plans directory holds at least one .json (" + dir.string() + (ec ? ": " + ec.message() : "") + ")");
    if (plans.empty()) { std::printf("npu_plan_shipped: FAIL\n"); return 1; }

    for (const auto& p : plans) {
        const std::string name = p.filename().string();
        try {
            const auto plan = onebit::npu::plan::Plan::load(p.string());
            check(plan.layers > 0, name + ": loads through Plan::load() with " + std::to_string(plan.layers) + " layer(s)");
            // The loader sets `format` to kFormat when it accepts the file; asserting it here means a future
            // format bump has to touch this test as well, rather than silently accepting the old plans.
            check(plan.format == onebit::npu::plan::kFormat, name + ": carries the format this build reads");
        } catch (const std::exception& e) {
            check(false, name + ": REFUSED by Plan::load(): " + e.what());
        }
    }

    std::printf("npu_plan_shipped: %s\n", bad ? "FAIL" : "ok");
    return bad ? 1 : 0;
}
