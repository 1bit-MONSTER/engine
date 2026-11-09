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
# engine #574: the signed manifest of a sealed NPU kernel set. tools/seal_manifest.py writes it (sha256
# of the ciphertext, Ed25519 over the manifest) and checks it; the test is a throwaway set in a temp
# dir, so no key, no device and no build: the four refusals are a set with no manifest, a file the
# manifest does not list, a ciphertext byte flipped after signing, and a manifest signed by another key.
# the handoff driver must not orphan its server on SIGKILL (PR_SET_PDEATHSIG); no GPU, no model
add_test(NAME kvshare_orphan COMMAND ${CMAKE_SOURCE_DIR}/tests/kvshare_orphan_test.sh)
