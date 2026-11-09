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
# engine #1279: the version a package is built with (scripts/pack-version.sh) -- a release tag as
# it stands, a candidate mapped into the ISO week of its own date with a suffix that sorts below that
# release, and the ordering a candidate needs: previous release < candidate < its release. The
# ordering is the row dpkg, rpm and pacman had to agree on; no container, no build, dpkg only.
add_test(NAME pack_version COMMAND ${CMAKE_SOURCE_DIR}/tests/pack_version_test.sh)
