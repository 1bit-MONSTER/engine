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
// batch_width.h -- how many rows one launch may carry (engine#1900).
//
// The forward chunks a batched pass at this width, and the activation BO holds exactly this many rows: an over-wide
// launch reads and poisons past the BO, which is memory corruption rather than a slow path (engine#1372). So the
// aggregate over a design set must be the MINIMUM of the widths actually loaded, never the first one that happens to
// be populated -- a 16/16/128 set returning 128 corrupts memory on the two narrow roles, while returning 16 is only
// narrower than it needs to be.
//
// Pure and header-only on purpose: the aggregation is the part that can be wrong in the dangerous direction, so it is
// pinned by a host test with no device, no ELF and no runtime.
#pragma once

#include <string>

namespace onebit {
namespace npu {

// The smallest positive width in `widths`, or `fallback` when none is positive.
//
// `widths` may contain 0 for "no design at this role", which is the ordinary state of a design array -- the old code
// returned the FIRST populated entry, so a zero-width slot simply moved the answer to whichever role came next.
inline int batch_width_min(const int* widths, int n, int fallback) {
    int w = 0;
    if (widths)
        for (int i = 0; i < n; ++i) {
            const int v = widths[i];
            if (v > 0 && (w == 0 || v < w)) w = v;
        }
    return w > 0 ? w : (fallback > 0 ? fallback : 1);
}

// --- a wide family beside the 16-row one (engine#1990) ---------------------------------------------------------
//
// A sealed set may carry the same roles twice: built for `narrow` rows (decode and short chunks) and for `wide` rows
// (prefill chunks). The three questions below are the ones that can be wrong in the dangerous direction, so they
// live here with the minimum above and are pinned by the same host test.

// Is every role that has a narrow kernel also covered by a wide kernel of exactly `wide` rows?  The chunk width is
// one number for the whole pass, so a single role without a wide kernel means the wide family cannot be used at
// all: its launches would carry more rows than that role's buffers hold. `wide` must exceed every narrow width.
inline bool wide_family_complete(const int* narrow_widths, const int* wide_widths, int n, int wide) {
    if (!narrow_widths || !wide_widths || n <= 0 || wide <= 0) return false;
    int roles = 0;
    for (int i = 0; i < n; ++i) {
        if (narrow_widths[i] <= 0) {
            if (wide_widths[i] > 0) return false;   // a wide kernel for a role the set does not run
            continue;
        }
        if (wide_widths[i] != wide || wide <= narrow_widths[i]) return false;
        ++roles;
    }
    return roles > 0;
}

// The chunk width of a pass: the wide family's rows when one was adopted (0 = none), else the narrow minimum.
inline int chunk_rows(int narrow, int wide) { return wide > narrow ? wide : narrow; }

// Does a launch of `rows` rows go to the wide family?  Only when it does not fit the narrow one and does fit the
// wide one: a wide launch costs more than a narrow one whatever it carries, so a decode step never takes it, and a
// launch past `wide` rows is never routed here (the caller refuses it by name).
inline bool wide_takes(int rows, int narrow, int wide) { return wide > narrow && rows > narrow && rows <= wide; }

// --- a narrow family for decode (engine: decode device wait) -----------------------------------------------------
//
// The mirror of the wide family: the same roles built for FEWER rows than the 16-row family, for the launches
// that carry only a few rows (a decode step carries one). Complete means every role with a 16-row kernel has a
// narrow kernel of exactly `narrow` rows, and `narrow` is below every 16-row width; a role without one would
// silently stay on its 16-row kernel, which is safe but is not what the log would claim, so it is all or none.
inline bool narrow_family_complete(const int* base_widths, const int* narrow_widths, int n, int narrow) {
    if (!base_widths || !narrow_widths || n <= 0 || narrow <= 0) return false;
    int roles = 0;
    for (int i = 0; i < n; ++i) {
        if (base_widths[i] <= 0) {
            if (narrow_widths[i] > 0) return false;
            continue;
        }
        if (narrow_widths[i] != narrow || narrow >= base_widths[i]) return false;
        ++roles;
    }
    return roles > 0;
}

// Does a launch of `rows` rows go to the narrow family?  Only when it fits: 1..narrow rows. One row more and the
// launch would write past the narrow kernel's activation buffer, so the bound is the kernel's own row count.
inline bool narrow_takes(int rows, int narrow) { return narrow > 0 && rows >= 1 && rows <= narrow; }

// Which file is the wide sealed ELF, if any?  (The narrow family's file is chosen by the same rule.)  The environment wins over the pack so a set can be tried, or switched
// off, without editing a pack: `env` "0" means no wide family even when the pack names one, any other non-empty
// value is the path as given. Otherwise the pack's name is used: as given when absolute, else inside `kernels_dir`
// (where the 16-row sealed ELF lives). Empty answer = no wide family. `env` may be null.
inline std::string wide_elf_path(const char* env, const std::string& pack_name, const std::string& kernels_dir) {
    if (env && *env) return (env[0] == '0' && env[1] == 0) ? std::string() : std::string(env);
    if (pack_name.empty()) return std::string();
    if (pack_name[0] == '/' || kernels_dir.empty()) return pack_name;
    return kernels_dir + "/" + pack_name;
}

}  // namespace npu
}  // namespace onebit
