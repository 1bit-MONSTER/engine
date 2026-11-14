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
# engine#2416: `tools/ci/hip_f16_audit.py` had a pin gate (engine#2411) and a self-test, and **nothing invoked the audit** -- `grep -rn
# hip_f16_audit` found one `--self-test` row and no caller, so engine#2406's "audit those on a pin bump" stayed a thing a person had to
# remember. That is how engine#2409 read *"0 narrowing(s)"* on files that had six casts.
#
# This row is the invoker. The audit needs a gpu-kernels checkout and the engine checks out none, so on a runner without one the
# wrapper exits 77 with a message that says the audit did NOT run -- loud, because a quiet skip is the failure mode. `SKIP_RETURN_CODE`
# is what makes 77 a *skip* rather than a red: without it CTest reports the row as failed, and a row that goes red for the absence of a
# private tree is a red people learn to ignore.
add_python_test(hip_f16_audit_pinned ${CMAKE_SOURCE_DIR}/tools/ci/hip_f16_audit_pinned.py --root ${CMAKE_SOURCE_DIR})
set_tests_properties(hip_f16_audit_pinned PROPERTIES SKIP_RETURN_CODE 77)
