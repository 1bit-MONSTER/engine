#!/usr/bin/env bash
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
# lemond-onebit [lemond args]
#
# The launcher in the Lemonade + onebit release package (docs/releases.md): it starts the
# package's lemond with the package's engine as the onebit backend, so nothing has to be on PATH.
#   lemonade-onebit-<tag>/lemond-onebit    this script (scripts/lemond-onebit.sh)
#   lemonade-onebit-<tag>/bin/            lemond, lemonade and Lemonade's resources
#   lemonade-onebit-<tag>/1bit/            the engine: 1bit and its backends (scripts/package-linux.sh)
set -euo pipefail
here=$(cd "$(dirname "$(readlink -f "$0")")" && pwd)
export LEMONADE_ONEBIT_BIN=${LEMONADE_ONEBIT_BIN:-$here/1bit/1bit}
# the NPU engine loads sealed kernels only, always: no opt-out (the release 1bit refuses plain
# kernel sets on its own too)
export ONEBIT_KCRYPT_REQUIRE_SEALED=1
# Authenticity, engine #574: a sealed set also carries manifest.json (every file it ships, with the
# sha256 of its ciphertext) and manifest.sig (Ed25519 over the manifest by the release signing key),
# and the loader verifies that signature with the release public key it was built with. The switch
# for it must be ONEBIT_KCRYPT_REQUIRE_MANIFEST=1, read by the same private NPU runtime code that
# reads ONEBIT_KCRYPT_REQUIRE_SEALED above (that loader is not in this repository). It is NOT set
# here yet: a public switch nothing reads is worse than none - #548 was closed for shipping exactly
# that - so this line lands with the pin bump that makes the loader honour it, together with the
# release public key's path. Until then the packaging and os/mini/release-check.sh require the
# manifest and verify the signature over it; the loader-side requirement is the open half.
export PATH=$here/bin:$PATH
exec "$here/bin/lemond" "$@"
