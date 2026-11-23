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
# engine#3350 rc1: os/v3/puppy/stage-loom.sh puts the pinned hrx-loom server and the pinned Loom cards into a payload,
# and embed/usr/local/bin/1bit-loom-cards assembles a card on the data partition with its weights linked into the
# Hugging Face cache. Runs both on FAKES (a wheel built here, a two-file card), so no network, no 20 MB, no root, no
# device. Also checks the shipped loom-cards.pin is complete.
set -u
here=$(cd "$(dirname "$0")/.." && pwd)
stage=$here/os/v3/puppy/stage-loom.sh
boot=$here/os/v3/puppy/embed/usr/local/bin/1bit-loom-cards
t=$(mktemp -d) || exit 1
bad=0
trap 'rm -rf "${t:?}"' EXIT
check() { if [ "$1" = 0 ]; then echo "ok   $2"; else echo "FAIL $2"; bad=$((bad + 1)); fi; }
command -v python3 > /dev/null || { echo "loom_payload_test: needs python3"; exit 1; }

# a fake wheel: the three files stage-loom.sh keeps, one it must leave out, and the licence
python3 -I - "$t/w.whl" << 'PY'
import sys, zipfile
z = zipfile.ZipFile(sys.argv[1], "w")
z.writestr("hrx_loom-9.9.9.data/scripts/loom-serve", "#!/bin/sh\necho serve\n")
z.writestr("hrx_loom-9.9.9.data/scripts/libiree.so", "iree")
z.writestr("hrx_loom-9.9.9.data/scripts/libloomc.so", "loomc")
z.writestr("hrx_loom-9.9.9.data/scripts/loom-compile", "not shipped")
z.writestr("hrx_loom-9.9.9.dist-info/licenses/LICENSE", "Apache-2.0 WITH LLVM-exception\n")
z.close()
PY
python3 -I -c 'import zipfile,hashlib,sys; print(hashlib.sha256(zipfile.ZipFile(sys.argv[1]).read("hrx_loom-9.9.9.data/scripts/loom-serve")).hexdigest())' "$t/w.whl" > "$t/ssha"
printf 'version=9.9.9\nwheel=w.whl\nwheel_sha256=%s\nloom_serve_sha256=%s\n' "$(sha256sum "$t/w.whl" | cut -c1-64)" "$(cat "$t/ssha")" > "$t/loom.config"

# a fake card: model.json names two weight files by @checkpoint origin, the lock pins that checkpoint
c=$t/cardsrc/fake-loom
mkdir -p "$c/program"
printf '%s\n' '{"variants":{"gguf":{"services":{"chat":{"parameters":{"model":{"files":[{"path":"parameter/m.gguf","origin":"@checkpoint/m.gguf"}]},"draft":{"files":[{"path":"parameter/d.gguf","origin":"@checkpoint/d.gguf"}]}}}}}}}' > "$c/model.json"
printf '%s\n' '{"origins":{"checkpoint":{"requested":{"kind":"hf","repository":"org/Model-GGUF","revision":"abc123"},"resolved":{"revision":"abc123"}}},"files":[{"origin":"@checkpoint/m.gguf","size":1,"sha256":"1111111111111111111111111111111111111111111111111111111111111111"},{"origin":"@checkpoint/d.gguf","size":1,"sha256":"2222222222222222222222222222222222222222222222222222222222222222"}]}' > "$c/model.json.lock"
printf 'prog\n' > "$c/program/llm.loom"
{ echo "card fake-loom org/fake-loom r1"
  for f in model.json model.json.lock program/llm.loom; do echo "$(sha256sum "$c/$f" | cut -c1-64) $(stat -c %s "$c/$f") $f"; done; } > "$t/cards.pin"

P=$t/payload; mkdir -p "$P"
ONEBIT_LOOM_PIN=$t/loom.config ONEBIT_LOOM_CARDS_PIN=$t/cards.pin sh "$stage" "$P" --wheel "$t/w.whl" --cards-from "$t/cardsrc" 2> "$t/err"
check $? "stage-loom stages the pinned wheel and card (exit 0)"
[ -x "$P/loom/loom-serve" ] && [ -f "$P/loom/libiree.so" ] && [ -f "$P/loom/libloomc.so" ] && [ -f "$P/loom/LICENSE" ] && [ ! -e "$P/loom/loom-compile" ]
check $? "loom/ holds loom-serve, its two libraries and the LICENSE, and not the rest of the wheel"
cmp -s "$c/program/llm.loom" "$P/loom-cards/fake-loom/program/llm.loom" && [ ! -e "$P/loom-cards/fake-loom/parameter" ]
check $? "the card's files are staged byte for byte, without parameter/ (the weights)"
[ -z "$(find "$P/loom" "$P/loom-cards" -type l)" ] && [ -z "$(find "$P/loom" "$P/loom-cards" \( -perm -0020 -o -perm -0002 \) ! -type l)" ]
check $? "the payload holds no link (pack_outside_links) and nothing group/other-writable"

printf 'x' >> "$c/program/llm.loom"; P2=$t/p2; mkdir -p "$P2"
ONEBIT_LOOM_PIN=$t/loom.config ONEBIT_LOOM_CARDS_PIN=$t/cards.pin sh "$stage" "$P2" --wheel "$t/w.whl" --cards-from "$t/cardsrc" 2> "$t/err"
[ $? = 2 ] && grep -q "not its pinned sha256/size" "$t/err"; check $? "a card file that is not its pin is refused (exit 2)"
printf 'prog\n' > "$c/program/llm.loom"
sed 's/^wheel_sha256=.*/wheel_sha256=0000000000000000000000000000000000000000000000000000000000000000/' "$t/loom.config" > "$t/bad.config"
ONEBIT_LOOM_PIN=$t/bad.config ONEBIT_LOOM_CARDS_PIN=$t/cards.pin sh "$stage" "$P2" --wheel "$t/w.whl" --cards-from "$t/cardsrc" 2> "$t/err"
[ $? = 2 ] && grep -q "not the pinned wheel" "$t/err"; check $? "a wheel that is not the pin is refused (exit 2)"

# the boot helper
D=$t/data/loom-cards; HF=$t/hf
sh "$boot" "$P/loom-cards" "$D" "$HF" 2> "$t/err"; check $? "1bit-loom-cards assembles the card (exit 0)"
[ "$(readlink "$D/fake-loom/parameter/m.gguf")" = "$HF/hub/models--org--Model-GGUF/snapshots/abc123/m.gguf" ] &&
    [ "$(readlink "$D/fake-loom/parameter/d.gguf")" = "$HF/hub/models--org--Model-GGUF/snapshots/abc123/d.gguf" ]
check $? "before any pull, parameter/ links point at the lock's snapshot (model and draft)"
mkdir -p "$HF/hub/models--org--Model-GGUF/refs"; echo def456 > "$HF/hub/models--org--Model-GGUF/refs/main"
sh "$boot" "$P/loom-cards" "$D" "$HF" 2> /dev/null
[ "$(readlink "$D/fake-loom/parameter/m.gguf")" = "$HF/hub/models--org--Model-GGUF/snapshots/def456/m.gguf" ]
check $? "after a pull, they follow the revision refs/main names (the catalogue pulls the repo head)"
[ "$(readlink "$D/fake-loom/program/llm.loom")" = "$P/loom-cards/fake-loom/program/llm.loom" ]
check $? "the card's own files are links to the payload's copy"
touch "$D/fake-loom/stale"; sh "$boot" "$P/loom-cards" "$D" "$HF" 2> /dev/null
[ ! -e "$D/fake-loom/stale" ]; check $? "a rebuild leaves nothing an earlier boot left in the card"
printf '%s\n' '{"origins":{}}' > "$t/nolock.json"; cp -r "$P/loom-cards" "$t/broken"; chmod -R u+w "$t/broken"; cp "$t/nolock.json" "$t/broken/fake-loom/model.json.lock"
sh "$boot" "$t/broken" "$t/data2" "$HF" 2> "$t/err"; [ $? = 1 ] && grep -q "could not link its weights" "$t/err" && [ ! -e "$t/data2/fake-loom" ]
check $? "a card whose weights' origin is not pinned is refused, and not half-built"
sh "$boot" "$t/none" "$t/data3" "$HF"; check $? "no payload cards is not an error"

# the shipped pin
pin=$here/os/v3/puppy/loom-cards.pin
ok=0
grep -qE '^card [A-Za-z0-9._-]+ [A-Za-z0-9._-]+/[A-Za-z0-9._-]+ [0-9a-f]{40}$' "$pin" || ok=1
for f in model.json model.json.lock; do grep -qE "^[0-9a-f]{64} [0-9]+ $f\$" "$pin" || ok=1; done
check $ok "the shipped loom-cards.pin names each card at a 40-hex revision, with model.json and its lock"

echo "loom_payload_test: $([ $bad = 0 ] && echo ok || echo FAIL)"
[ $bad = 0 ]
