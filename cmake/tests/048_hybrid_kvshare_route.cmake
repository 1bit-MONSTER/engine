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
# target 1's shared route (ONEBIT_KV_SHARE=1): the fake backend speaks ?action=share/adopt and the fake
# provider fails unless it was handed one readable fd per region -- so the socket, the inherited fds, the
# share_id and the fallback are all pinned with no model, no GPU and no dma-buf. Needs no exporter, so CI can
# run it (the GPU gate in ~/wt/zc-probes cannot).
add_test(NAME hybrid_kvshare_route COMMAND ${CMAKE_SOURCE_DIR}/tests/hybrid_kvshare_route.sh $<TARGET_FILE:onebit>)
