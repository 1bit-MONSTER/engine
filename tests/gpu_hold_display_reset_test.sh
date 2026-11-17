#!/usr/bin/env bash
# Copyright 2026 bong-water-water-bong
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# gpu_hold_display_reset_test.sh: a DRM vblank reset is not a compute fault (engine#2803).
#
# Measured 2026-11-16: three GPU resets in one boot, each preceded by `WARNING: drivers/gpu/drm/drm_vblank.c:{380,1329}`
# on a kworker and followed by `[drm] device wedged, but no recovery needed`, while the GPU sat idle between tickets. One
# of them refused an `hw_gate` take after a 794 s queue wait, so a pin bump carrying a test-only fork commit read as
# "the box is faulting". `GPU_FAULT_RE` catches them with `amdgpu.*(reset|hang)`, which cannot see the warning above it,
# so the classification looks at the burst.
#
# Arms, all from the real log lines:
#   1. a vblank/wedged burst with no compute marker does NOT count as a compute fault;
#   2. the same burst WITH a gfxhub/MES/ring-timeout line beside it DOES (the conservatism that keeps this a guard);
#   3. a plain compute reset with no display marker DOES;
#   4. a line with neither marker keeps the refusal (unknown stays refused).
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/.." && pwd)
ok=0
bad=0
chk() { if [ "$2" = "$1" ]; then printf 'ok   %s\n' "$3"; ok=$((ok + 1)); else printf 'FAIL %s (want %s got %s)\n' "$3" "$1" "$2"; bad=$((bad + 1)); fi; }

# Extract the shipped helper and its two patterns rather than restating them: an arm that copies the rule is a second
# implementation to keep in sync (the reason llama.cpp#2762's arms call the real function).
eval "$(sed -n "/^GPU_FAULT_DISPLAY_RE=/,/^}/p" "$ROOT/scripts/gpu-hold" | grep -vE '^GPU_FAULT_RE=|^GPU_FAULT_SKIP_RE=')"

VBLANK='Nov 17 00:10:22 strixhalo kernel: WARNING: drivers/gpu/drm/drm_vblank.c:1329 at drm_crtc_wait_one_vblank+0x181/0x210 [drm], CPU#4: kworker/4
Nov 17 00:10:22 strixhalo kernel: amdgpu 0000:c5:00.0: GPU reset(3) succeeded!
Nov 17 00:10:22 strixhalo kernel: amdgpu 0000:c5:00.0: [drm] device wedged, but no recovery needed'
COMPUTE='Nov 17 00:10:20 strixhalo kernel: amdgpu 0000:c5:00.0: ring gfx_0.0.0 timeout, signaled seq=1, emitted seq=2
Nov 17 00:10:22 strixhalo kernel: amdgpu 0000:c5:00.0: GPU reset(3) succeeded!'
MES='Nov 17 00:10:20 strixhalo kernel: amdgpu 0000:c5:00.0: MES failed to respond to msg=REMOVE_QUEUE'
BAREFOOT='Nov 17 00:10:22 strixhalo kernel: amdgpu 0000:c5:00.0: GPU reset(3) succeeded!'

printf '%s\n' "$VBLANK" | _fault_is_compute > /dev/null && chk 0 1 "a DRM vblank reset burst is NOT a compute fault" || chk 0 0 "a DRM vblank reset burst is NOT a compute fault"
printf '%s\n%s\n' "$VBLANK" "$COMPUTE" | _fault_is_compute > /dev/null && chk 0 0 "...but with a ring timeout beside it, it IS (the refusal stands)" || chk 0 1 "...but with a ring timeout beside it, it IS (the refusal stands)"
printf '%s\n' "$COMPUTE" | _fault_is_compute > /dev/null && chk 0 0 "a compute reset with no display marker IS a compute fault" || chk 0 1 "a compute reset with no display marker IS a compute fault"
printf '%s\n%s\n' "$VBLANK" "$MES" | _fault_is_compute > /dev/null && chk 0 0 "...and a MES line beside a vblank burst also keeps the refusal" || chk 0 1 "...and a MES line beside a vblank burst also keeps the refusal"
printf '%s\n' "$BAREFOOT" | _fault_is_compute > /dev/null && chk 0 0 "a reset with neither marker stays refused (unknown is conservative)" || chk 0 1 "a reset with neither marker stays refused (unknown is conservative)"

printf '\n%s ok, %s failed\n' "$ok" "$bad"
[ "$bad" -eq 0 ]
