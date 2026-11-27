#!/bin/sh
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
# engine#3528: /etc/profile.d/92onboard-loom gives a login shell the same Loom route Lemonade's serves have
# (ONEBIT_LOOM_CARDS = the boot-assembled /data/loom-cards, ONEBIT_LOOM_SERVE = the payload's loom-serve), only when
# those exist, and never over a value already set. Runs a copy with the two appliance paths pointed into a temp dir.
set -u
here=$(cd "$(dirname "$0")/.." && pwd)
f=$here/os/v3/puppy/embed/etc/profile.d/92onboard-loom
t=$(mktemp -d) || exit 1
bad=0
trap 'rm -rf "${t:?}"' EXIT
check() { if [ "$1" = 0 ]; then echo "ok   $2"; else echo "FAIL $2"; bad=$((bad + 1)); fi; }
mkdir -p "$t/data/loom-cards" "$t/opt/loom"; printf '#!/bin/sh\n' > "$t/opt/loom/loom-serve"; chmod +x "$t/opt/loom/loom-serve"
sed -e "s|/data/loom-cards|$t/data/loom-cards|g" -e "s|/opt/1bit-package/loom/loom-serve|$t/opt/loom/loom-serve|g" "$f" > "$t/p"
grep -q "$t/data/loom-cards" "$t/p" && grep -q "$t/opt/loom/loom-serve" "$t/p"; check $? "the copy points at the test's paths"
o=$(env -i sh -c '. "$1"; echo "$ONEBIT_LOOM_CARDS|$ONEBIT_LOOM_SERVE"' _ "$t/p")
[ "$o" = "$t/data/loom-cards|$t/opt/loom/loom-serve" ]; check $? "on the appliance: both are exported (as 10lemonade does for lemond)"
o=$(env -i ONEBIT_LOOM_CARDS=/mine ONEBIT_LOOM_SERVE=/my/ls sh -c '. "$1"; echo "$ONEBIT_LOOM_CARDS|$ONEBIT_LOOM_SERVE"' _ "$t/p")
[ "$o" = "/mine|/my/ls" ]; check $? "a value already set wins"
rm -rf "${t:?}/data" "${t:?}/opt"
o=$(env -i sh -c '. "$1"; echo "${ONEBIT_LOOM_CARDS:-none}|${ONEBIT_LOOM_SERVE:-none}"' _ "$t/p"); r=$?
[ "$o" = "none|none" ] && [ $r = 0 ]; check $? "elsewhere (no /data/loom-cards, no payload loom-serve): nothing set, and sourcing succeeds"
o=$(env -i sh -c 'set -e; . "$1"; echo sourced' _ "$t/p"); [ "$o" = sourced ]; check $? "sourcing under set -e does not abort a login shell"

echo "loom_profile_test: $([ $bad = 0 ] && echo ok || echo FAIL)"
[ $bad = 0 ]
