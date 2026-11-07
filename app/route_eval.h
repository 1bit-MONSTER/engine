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


// `1bit route --laya-models FILE --eval CASES` and `--dump-encoder CASES --out DIR`: a model table's
// routing measured over labelled states, and the decision head's input written out for a head-only
// fine-tune (docs/laya-models.md, "Measuring and fine-tuning the router").
#pragma once

#include <string>

namespace onebit {

struct RouteEvalOptions {
    std::string laya_model;   // --laya-model (resolved by laya_checkpoint)
    std::string table;        // --laya-models
    std::string eval_cases;   // --eval
    std::string dump_cases;   // --dump-encoder
    std::string out_dir;      // --out (dump)
    std::string per_case;     // --per-case FILE (eval): one JSON line per case
    bool json = false;        // --json (eval)
    bool rotate = false;      // --rotate (dump): every rotation of the options, labels moved with them
};

int run_route_eval(const RouteEvalOptions& o);

}  // namespace onebit
