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
# engine#2456. `tests_referenced_check` was satisfied by a *mention*, so these four tools' `--self-test` arms counted as run while
# nothing ran them: their rows name a test that exercises the tool's logic (`tests/<tool>_test.py`), and the tool's own name appeared
# only in the comments beside those rows. Under the narrowed rule they are real debt, and each now has the row it claimed.
#
# The spellings differ and that is the tools', not a typo: `gsm8k_eval.py` and `mtp_depth_pool.py` implement `--selftest` (the
# hyphenated form is an argument error on the first and a *filename* on the second), while `q4nx_diff.py` and `seal_manifest.py` take
# `--self-test`. All four were measured green at these spellings before the rows were written.
add_python_test(gsm8k_eval_self ${CMAKE_SOURCE_DIR}/tools/gsm8k_eval.py --selftest)
add_python_test(mtp_depth_pool_self ${CMAKE_SOURCE_DIR}/tools/mtp_depth_pool.py --selftest)
add_python_test(q4nx_diff_self ${CMAKE_SOURCE_DIR}/tools/q4nx_diff.py --self-test)
add_python_test(seal_manifest_self ${CMAKE_SOURCE_DIR}/tools/seal_manifest.py --self-test)

# engine#2478: the census that turns the NVMe IO_PAGE_FAULT question into a command (same address set across boots =
# one stale translation; disjoint = re-allocation behind a live descriptor). Its arms are synthetic journals, so the
# row runs anywhere -- the device is only needed for the input, never for the check.
add_python_test(iommu_fault_census ${CMAKE_SOURCE_DIR}/tools/diag/iommu_fault_census.py --self-test)

# engine#2478 review: four more tools carried a self-test that no row ran -- the same debt engine#2456 found for the
# four above, found by `tools/ci/tool_selftest_wired_check.py` (which now fails when the list grows again). Each was
# run at the spelling below before the row was written: `template_bom.py` and `pin_watch.py` and `split_ceiling.py`
# take `--self-test`, `moe_q4nx_ref.py` takes the bare word.
add_python_test(template_bom_self ${CMAKE_SOURCE_DIR}/tools/template_bom.py --self-test)
add_python_test(moe_q4nx_ref_self ${CMAKE_SOURCE_DIR}/tools/moe_q4nx_ref.py selftest)
add_python_test(pin_watch_self ${CMAKE_SOURCE_DIR}/tools/pin_watch.py --self-test)
add_python_test(split_ceiling_self ${CMAKE_SOURCE_DIR}/tools/split_ceiling.py --self-test)
