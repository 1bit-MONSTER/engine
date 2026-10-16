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
export PATH=$here/bin:$PATH
exec "$here/bin/lemond" "$@"
