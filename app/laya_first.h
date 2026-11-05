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

// `1bit laya-first`: the routing table that ships (docs/laya-first.md §4) and the
// `collection.router` policy it desugars into. The table is the product surface — class ->
// {model, hf_repo, file, device, ctx} — and the policy is the mechanism Lemonade already
// understands (one rule per class, `route_to` that class's model), so the engine, the validate
// endpoint and the decision format are untouched.
#pragma once

namespace onebit {

int run_laya_first(int argc, char** argv);

}  // namespace onebit
