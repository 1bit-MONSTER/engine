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
if(ONEBIT_HRX)
if(ONEBIT_SERVE_TEST_GGUF)
        # Target 1 with real dma-bufs (docs/zero-copy.md, "Target 1"): the decoder exports its KV region, a
        # separate provider process receives the fds over SCM_RIGHTS and prefills into them, and the decoder
        # adopts the cells with no data movement. Registered here, after include(cmake/hrx.cmake), because
        # ONEBIT_HRX_SERVER is only defined there -- reading it earlier drops the argument on a first configure
        # and shifts the rest. Needs the fork's reference provider, which this build does not produce: pass
        # ONEBIT_KVSHARE_PREFILL or the test skips (77), as it does without a gfx1151.
        add_test(NAME hybrid_kvshare_e2e
                 COMMAND ${CMAKE_SOURCE_DIR}/tests/hybrid_kvshare_e2e.sh "${ONEBIT_HRX_SERVER}" "${ONEBIT_SERVE_TEST_GGUF}" "${ONEBIT_KVSHARE_PREFILL}")
        set_tests_properties(hybrid_kvshare_e2e PROPERTIES SKIP_RETURN_CODE 77)
endif()
endif()
