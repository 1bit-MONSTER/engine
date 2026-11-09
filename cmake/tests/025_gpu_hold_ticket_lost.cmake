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
# engine#1635: a queued ticket that is removed (reaped, or deleted by another lane) used to leave its owner waiting for a
# queue position that no longer existed -- the full -w, 5400 s by default, looking exactly like "queued behind a busy box".
# Drives scripts/gpu-hold on a private queue: the ticket is deleted mid-wait, and separately rewritten to a dead pid so a
# second real gpu-hold reaps it and leaves the note the victim quotes. No device, no build; skips (77) without the script.
add_test(NAME gpu_hold_ticket_lost COMMAND bash ${CMAKE_SOURCE_DIR}/tests/gpu_hold_ticket_lost_test.sh)
