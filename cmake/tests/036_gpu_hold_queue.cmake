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
# engine #1504: the GPU hold queue is the FLEET's when the shared state is installed, with no environment to say
# so - the failure it guards against is silent (a caller under another account taking its own token). Two arms:
# fleet state present -> the claim uses the fleet's queue and token; absent -> the old ~/GPU-* paths, so a box
# that has not been set up is unchanged. No GPU: the claim holds the token for three seconds and runs `true`.
add_test(NAME gpu_hold_queue COMMAND ${CMAKE_SOURCE_DIR}/tests/gpu_hold_queue_test.sh)
