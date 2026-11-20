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
# engine#2858: a llama-bench run whose HRX runtime failed to initialise still prints a
# full results table labelled `backend HRX`, from whatever path it fell back to -- measured
# on the shipped bench tree on a box it was not built on: `pp512 11.22` on 2026-11-18 and
# `73.61` on the same tree a day later, with the runtime saying "did not initialise …
# registers no devices" in BOTH logs. So neither number was an HRX measurement, and the
# table is the only artefact anybody keeps. `tools/ci/hrx_bench_backend_check.py` reads a
# kept log and says whether its backend initialised: 0 a table from an initialised
# backend, 1 the log says it did not, 2 cannot tell (an empty or table-less log is not a
# pass). The self-test is the ctest row; the useful mode is a lane command on a log,
# exactly like `ctest_skips.py --log`.
add_python_test(hrx_bench_backend_self ${CMAKE_SOURCE_DIR}/tools/ci/hrx_bench_backend_check.py --self-test)
