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
# Python tool self-tests (engine#1979). Every row here drives a `tools/*.py` or `tests/*.py` self-test through
# `add_python_test()`, defined in CMakeLists.txt above this include. They live in their own file because they are the rows
# every lane appends one of, and they were the block that went stale first -- two lanes appending to different registration
# files cannot conflict. `cmake/tests-gpu-hold.cmake` is the same idea for a target-bearing family, and both CI guards
# already treat `cmake/tests-*.cmake` as registration files, so nothing in the checks needs to know this file exists.
# engine#1864: the ordering rule itself, checked on the file (no configure, no build): every ${Python3_EXECUTABLE} use has
# to follow the find_package above. The nine that did not ran through their exec bit, and a mode-only rewrite of one
# script turned the row into BAD_COMMAND.
add_python_test(python_test_order_check ${CMAKE_SOURCE_DIR}/tests/python_test_order_check.py --check --root ${CMAKE_SOURCE_DIR})
add_python_test(python_test_order_check_self ${CMAKE_SOURCE_DIR}/tests/python_test_order_check.py --self-test)
add_python_test(ci_targets ${CMAKE_SOURCE_DIR}/tools/ci/ci_targets_check.py --check --root ${CMAKE_SOURCE_DIR})
add_python_test(ci_targets_self ${CMAKE_SOURCE_DIR}/tools/ci/ci_targets_check.py --self-test)
# engine#803 gate, second half: the comparator that turns two --dump-logits directories into the KLD verdict the
# P4 gate asks for. No device and no model, so the gate is one command when a ticket is free; the self-test pins
# the four cases a KLD implementation gets wrong (identical -> exactly 0, a shift is positive and directional, a
# uniform shift stays 0, and an asymmetric file set is refused by name).
add_python_test(logits_kld_self ${CMAKE_SOURCE_DIR}/tools/logits_kld.py --self-test)
# engine#2018: this one was in tools/ci/unreferenced_py_tests.txt as a baseline entry, and it should not have been: the
# file is portable and passes (worst-case |sum| 11.18e6 inside int32 at K=11008, 4 arms), it just never had a runner. A
# baseline records a test as never-run, which for a passing test is the wrong outcome -- so it is registered and the line
# is removed, which is the shrink the baseline's own comment asks for.
add_python_test(moe_expert_chunks_accum_test ${CMAKE_SOURCE_DIR}/tests/moe_expert_chunks_accum_test.py)
# engine#1820: the other half of ci_targets_check -- a standalone shell test that nothing references cannot fail.
# `need` is every tests/*.sh, `have` is every file that mentions its basename somewhere that can run it, and
# tools/ci/unreferenced_tests.txt carries the deliberate exclusions WITH their reasons, so the list can shrink.
add_python_test(tests_referenced ${CMAKE_SOURCE_DIR}/tools/ci/tests_referenced_check.py --check --root ${CMAKE_SOURCE_DIR})
add_python_test(tests_referenced_self ${CMAKE_SOURCE_DIR}/tools/ci/tests_referenced_check.py --self-test)
# engine#1901: five tools ship a --self-test that no runner named, so the arms existed, were maintained beside the logic they
# check, and never executed. All five run device-free and fast (58-277 ms measured), so they are rows rather than allow-list
# excuses, and tools/ci/tests_referenced_check.py now counts tools/**/*.py that declare one and are named by no RUNNER
# (CMakeLists/cmake/workflows -- a citation in another tool's source is not a runner: that is the 5-vs-2 difference).
# Registered through add_python_test (engine#1864), so they run THROUGH the interpreter: my first version used the raw
# `${Python3_EXECUTABLE}` form above the find_package and the repository's own ordering guard caught it by name --
# tools/mtp_head_ref.py is the one of the five without an exec bit, and it became BAD_COMMAND under ctest.
# engine#1962: the narrowing-body enumeration, as a ratchet. The corpus is a submodule, so the check
# skips (rc 77) where third_party/llama.cpp is not checked out; the baseline file is in the engine tree,
# so the recorded set is reviewable everywhere.
add_python_test(tool_selftest_hrx_f16_narrowing ${CMAKE_SOURCE_DIR}/tools/ci/hrx_f16_narrowing.py --self-test)
add_test(NAME hrx_f16_narrowing_check COMMAND ${Python3_EXECUTABLE}
         ${CMAKE_SOURCE_DIR}/tools/ci/hrx_f16_narrowing.py --check --root ${CMAKE_SOURCE_DIR})
set_tests_properties(hrx_f16_narrowing_check PROPERTIES SKIP_RETURN_CODE 77)
add_python_test(tool_selftest_batched_repeat ${CMAKE_SOURCE_DIR}/tools/batched_repeat.py --selftest)
add_python_test(tool_selftest_logprob_agree ${CMAKE_SOURCE_DIR}/tools/logprob_agree.py --selftest)
add_python_test(tool_selftest_mtp_head_ref ${CMAKE_SOURCE_DIR}/tools/mtp_head_ref.py --selftest)
add_python_test(tool_selftest_spec_decode_acceptance ${CMAKE_SOURCE_DIR}/tools/spec_decode_acceptance.py --selftest)
add_python_test(tool_selftest_whisper_encoder_baseline ${CMAKE_SOURCE_DIR}/tools/whisper_encoder_baseline.py --self-test)
# engine#1780: the DFlash draft package's gate before a loader -- tile-grid sizes from the config, tile-header sanity and
# the container's scalars against config.json, on a synthetic package with eight negative controls. Stdlib only.
add_python_test(tool_selftest_dflash_draft_info ${CMAKE_SOURCE_DIR}/tools/dflash_draft_info.py --selftest)
# engine#1750: what a TDR recovery did, counted from the kernel log (failed context restarts among them). The self-test
# parses lines in the driver's own formats for three module generations. Stdlib only, no device.
add_python_test(tool_selftest_amdxdna_tdr_report ${CMAKE_SOURCE_DIR}/tools/amdxdna_tdr_report.py --self-test)
# engine#2013: the qwen35 (gated linear attention) GGUF -> Q4NX converter. Its own selftest (tile round-trips, the
# value-head permutation), and a pack + verify of a small qwen35 GGUF the test writes itself, with the three inputs it
# has to refuse. numpy only: no gguf package, no model, no device.
add_python_test(tool_selftest_qwen35_to_q4nx ${CMAKE_SOURCE_DIR}/tools/qwen35_to_q4nx.py selftest)
add_python_test(qwen35_to_q4nx ${CMAKE_SOURCE_DIR}/tests/qwen35_to_q4nx_test.py)
