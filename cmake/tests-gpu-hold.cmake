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
# The `gpu_hold_*` ctest registrations (engine#1696).
#
# They live here rather than in CMakeLists.txt because every test-adding PR edits the same contiguous block of
# registrations there, so any two of them conflicted by construction: three rebases went through that region in one hour
# on 2026-11-09, none of them a logical conflict, all resolved the same mechanical "keep both". One family per file makes
# a new `gpu_hold_*` test a one-line addition in its own file.
#
# This file is included from CMakeLists.txt, where the registrations used to be. Two readers key on that filename and
# both were widened in the same commit: tools/ci/ci_targets_check.py follows `include(cmake/tests-*.cmake)` so the
# add_test ratchet still sees these, and tools/ci/hrx_build_scope.py treats `cmake/tests-*.cmake` like CMakeLists.txt so
# a registration-only PR here still classifies as `configure-only` instead of paying a full ONEBIT_HRX compile.

# engine#1621: a hold that ends must end everything it started. Three orphans in two days were all GRANDCHILDREN
# of a ticket (ticket → shell → runner → server): `pkill -P` reaches one level, so the process holding the memory
# outlived the ticket. Drives scripts/gpu-hold with a wrapper → mid → work chain and asserts the deepest one dies.
add_test(NAME gpu_hold_tree COMMAND bash ${CMAKE_SOURCE_DIR}/tests/gpu_hold_tree_test.sh)

# engine#1650: a hold ends itself when the box runs out of memory, before the kernel picks a victim — the 11-08/09
# failure where an unbudgeted GPU run took the box to 0.7 GiB and the OOM killer chose the PR-agent LLM.
add_test(NAME gpu_hold_mem_abort COMMAND bash ${CMAKE_SOURCE_DIR}/tests/gpu_hold_mem_abort_test.sh ${CMAKE_SOURCE_DIR}/scripts/gpu-hold)

# engine#1635: a queued ticket that is removed (reaped, or deleted by another lane) used to leave its owner waiting for a
# queue position that no longer existed -- the full -w, 5400 s by default, looking exactly like "queued behind a busy box".
# Drives scripts/gpu-hold on a private queue: the ticket is deleted mid-wait, and separately rewritten to a dead pid so a
# second real gpu-hold reaps it and leaves the note the victim quotes. No device, no build; skips (77) without the script.
add_test(NAME gpu_hold_ticket_lost COMMAND bash ${CMAKE_SOURCE_DIR}/tests/gpu_hold_ticket_lost_test.sh)

# engine #1504: the GPU hold queue is the FLEET's when the shared state is installed, with no environment to say
# so - the failure it guards against is silent (a caller under another account taking its own token). Two arms:
# fleet state present -> the claim uses the fleet's queue and token; absent -> the old ~/GPU-* paths, so a box
# that has not been set up is unchanged. No GPU: the claim holds the token for three seconds and runs `true`.
add_test(NAME gpu_hold_queue COMMAND ${CMAKE_SOURCE_DIR}/tests/gpu_hold_queue_test.sh)
# engine#1639: the half of the survivor bound that a CPU-only box can drive. The bound itself fires only for a payload root
# wedged in uninterruptible sleep, which user space cannot create, so this covers the other half: a holder that released with
# a survivor must leave a record, and the NEXT holder must warn about it and clear it.
add_test(NAME gpu_hold_survivors COMMAND bash ${CMAKE_SOURCE_DIR}/tests/gpu_hold_survivors_test.sh)
add_test(NAME gpu_hold_group
         COMMAND bash ${CMAKE_SOURCE_DIR}/tests/gpu_hold_group_test.sh)
# engine#1609 follow-up: the resolution has THREE directory states and `gpu_hold_queue` above covers two of them
# (fleet state fully installed, and absent). The third -- the hold directory exists but gpu-queue does not, which is
# what a partially-run install or any `mkdir` of /var/lib/1bit-fleet leaves -- is where the queue and the token used to
# resolve under DIFFERENT bases, so two callers could each hold a token and both run GPU work. Asserts the two
# resolve under one base in all three states, plus that the script's two predicates are textually identical.
add_test(NAME gpu_hold_hold_dir COMMAND bash ${CMAKE_SOURCE_DIR}/tests/gpu_hold_hold_dir_test.sh ${CMAKE_SOURCE_DIR}/scripts/gpu-hold)
