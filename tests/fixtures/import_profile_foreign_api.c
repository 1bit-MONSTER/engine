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

// The shape of libamdf's XDNA memory profile that tests/npu_import_contract.sh has to read, reduced to the
// lines that decide it (the pinned source is third_party/hrx-system, memory_profile.c). This is the
// advertising pin: the ordinal-1 (import) branch sets FOREIGN_API on its dma-buf-fd entry.
//
// Three shapes here are the whole reason the scan is not a grep: the ordinal-0 branch advertises the flag
// too (so a whole-file search cannot tell a narrowed import class from a create class), and the ordinal-1
// branch CLEARS the flag on the opaque-fd entry it clones (so a scan that ignores negated uses reads that
// clear as an advertisement -- engine#1283's review).
//
// Synthetic on purpose: a copy of the third-party file would drag its notice and its licence into this tree
// for a four-line question. The indentation and the two-entry clone are the original's.
  if (memory_profile_ordinal == 0) {
    profile.external_memory_support_count = 2;
    profile.external_memory_support[0] = (amdf_external_memory_support_t){
        .type = AMDF_EXTERNAL_MEMORY_TYPE_DMA_BUF_FD,
        .flags = AMDF_EXTERNAL_MEMORY_SUPPORT_FLAG_EXPORT |
                 AMDF_EXTERNAL_MEMORY_SUPPORT_FLAG_CROSS_PROCESS |
                 AMDF_EXTERNAL_MEMORY_SUPPORT_FLAG_FOREIGN_API,
    };
    profile.external_memory_support[1] = profile.external_memory_support[0];
    profile.external_memory_support[1].type =
        AMDF_EXTERNAL_MEMORY_TYPE_OPAQUE_FD;
    profile.external_memory_support[1].flags &=
        ~AMDF_EXTERNAL_MEMORY_SUPPORT_FLAG_FOREIGN_API;
  } else if (memory_profile_ordinal == 1) {
    profile.roles =
        AMDF_MEMORY_PROFILE_ROLE_IMPORT | AMDF_MEMORY_PROFILE_ROLE_HOST_MAP;
    profile.external_memory_support_count = 2;
    profile.external_memory_support[0] = (amdf_external_memory_support_t){
        .type = AMDF_EXTERNAL_MEMORY_TYPE_DMA_BUF_FD,
        .flags = AMDF_EXTERNAL_MEMORY_SUPPORT_FLAG_IMPORT |
                 AMDF_EXTERNAL_MEMORY_SUPPORT_FLAG_SOURCE_OFFSET |
                 AMDF_EXTERNAL_MEMORY_SUPPORT_FLAG_CROSS_PROCESS |
                 AMDF_EXTERNAL_MEMORY_SUPPORT_FLAG_FOREIGN_API,
        .source_offset_alignment = 1,
        .byte_length_alignment = 1,
    };
    profile.external_memory_support[1] = profile.external_memory_support[0];
    profile.external_memory_support[1].type =
        AMDF_EXTERNAL_MEMORY_TYPE_OPAQUE_FD;
    profile.external_memory_support[1].provenance =
        (amdf_external_memory_provenance_t)
            AMDF_LINUX_DMA_BUF_DIRECT_HOST_PROVENANCE;
    profile.external_memory_support[1].flags &=
        ~AMDF_EXTERNAL_MEMORY_SUPPORT_FLAG_FOREIGN_API;
  } else {
    profile.roles =
        AMDF_MEMORY_PROFILE_ROLE_REGISTER | AMDF_MEMORY_PROFILE_ROLE_HOST_MAP;
  }
