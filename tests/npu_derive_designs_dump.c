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

// npu_derive_designs_dump -- print the design set the engine derives for a set of model dims.
//
//   npu_derive_designs_dump H NH NKV HD IM fused_gu [attn_output_gate] [tag]
//
// One line per design: `ROLE M K N elf`, with the unique role name (q4nx_design_role_name()), the
// geometry and the artifact.  It exists so the plan generator can be checked against the derivation
// instead of against a second copy of the rules: `tests/npu_plan_designs_parity_test.py` runs this
// binary and tools/npu_plan.py on the same dims and requires the two sets to be identical, which is
// the only way a plan's `designs` section (engine#803 A2a) can be trusted to name the kernels the
// container's weights were written for.
//
// No model, no device: the derivation is pure arithmetic over the dims.
#include "q4nx_pack.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char** argv) {
    if (argc != 7 && argc != 8 && argc != 9) {
        fprintf(stderr, "usage: %s H NH NKV HD IM fused_gu [attn_output_gate] [tag]\n", argv[0]);
        return 2;
    }
    Q4nxModelDims d;
    memset(&d, 0, sizeof(d));
    d.H = atoi(argv[1]);
    d.NH = atoi(argv[2]);
    d.NKV = atoi(argv[3]);
    d.HD = atoi(argv[4]);
    d.IM = atoi(argv[5]);
    d.fused_gu = atoi(argv[6]);
    d.attn_output_gate = argc > 7 ? atoi(argv[7]) : 0;
    d.M = Q4NX_DESIGN_M_PREFILL;
    /* The forward rewrites every artifact name with "_b8" for a b8 container (the blocked 8x8 tile
     * layout the kernels carry); the derivation itself does not know the layout, so the dump applies
     * the same rewrite when asked. Absent tag = the rm/pb16i/t2s families, whose prefill names are the
     * untagged int8 ones. */
    const char* tag = argc > 8 ? argv[8] : "";

    Q4nxDesignGeom designs[Q4NX_MAX_DESIGNS];
    memset(designs, 0, sizeof(designs));
    const int n = q4nx_derive_designs(&d, designs);
    if (n <= 0) {
        fprintf(stderr, "derive failed for H=%d NH=%d NKV=%d HD=%d IM=%d fused_gu=%d\n", d.H, d.NH,
                d.NKV, d.HD, d.IM, d.fused_gu);
        return 1;
    }
    for (int i = 0; i < n; i++) {
        char name[Q4NX_ELF_NAME_MAX];
        snprintf(name, sizeof(name), "%s", designs[i].elf_name);
        if (strcmp(tag, "_b8") == 0 || strcmp(tag, "b8") == 0) {
            const size_t len = strlen(name);
            if (len > 4 && strcmp(name + len - 4, ".elf") == 0) {
                char tagged[Q4NX_ELF_NAME_MAX];
                snprintf(tagged, sizeof(tagged), "%.*s_b8.elf", (int)(len - 4), name);
                snprintf(name, sizeof(name), "%s", tagged);
            }
        }
        printf("%s %d %d %d %s\n", q4nx_design_role_name(designs[i].role), designs[i].M, designs[i].K,
               designs[i].N, name);
    }
    return 0;
}
