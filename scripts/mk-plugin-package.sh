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
# mk-plugin-package.sh <1bit-bundle-dir> <out-dir> <version>
#
# Packs the engine as a SEPARATE package that plugs into a stock Lemonade (no fork, no patch):
# <out-dir>/1bit-engine_<version>_amd64.deb and <out-dir>/1bit-engine-<version>-linux-x86_64.tar.zst.
#
# <1bit-bundle-dir> is the 1bit/ directory of a Lemonade + onebit package (it carries 1bit, the HRX
# llama.cpp build with ggml-hrx, and the sealed NPU kernel sets). The package adds the two things a
# Lemonade on a plain distribution needs and the bundle does not carry: the HSA runtime the HRX
# backend dlopens (it searches $ORIGIN, so it lands beside libhrx.so), and a `llama-server` shim that
# sets the engine's own HRX environment before exec'ing the real binary. The shim is the whole
# plug-in contract: Lemonade's llamacpp-hrx recipe resolves the name `llama-server` inside the
# directory named by `hrx_bin`, so pointing `hrx_bin` at <prefix>/lemonade-bin is enough.
#
# Userspace only: no kernel module, no Python at runtime (scripts/weekly.sh's package checks).
#
# LIBHSA overrides the HSA runtime source; the default is the system's libhsa-runtime64.so.1.
set -euo pipefail

[ $# -ge 3 ] || { sed -n '18,34p' "$0"; exit 2; }
bundle=$(cd "$1" && pwd) out=$2 ver=$3
[ -x "$bundle/1bit" ] || { echo "not a 1bit bundle: $bundle (no 1bit binary)" >&2; exit 1; }
[ -d "$bundle/hrx/llama/bin" ] || { echo "not a 1bit bundle: $bundle (no hrx/llama/bin)" >&2; exit 1; }

libhsa=${LIBHSA:-}
if [ -z "$libhsa" ]; then
    for c in "$bundle/hrx/llama/bin/libhsa-runtime64.so.1" \
             /usr/lib/x86_64-linux-gnu/libhsa-runtime64.so.1 \
             /usr/lib64/libhsa-runtime64.so.1; do
        [ -f "$c" ] && { libhsa=$c; break; }
    done
fi
[ -n "$libhsa" ] && [ -f "$libhsa" ] || {
    echo "no libhsa-runtime64.so.1 found; set LIBHSA=<path> (the HRX backend dlopens it)" >&2; exit 1; }

pkg=1bit-engine
root=$(mktemp -d)/root
trap 'rm -rf "$(dirname "$root")"' EXIT
mkdir -p "$root/opt/1bit/engine" "$root/usr/share/doc/$pkg" "$root/usr/share/1bit" \
         "$root/usr/bin" "$root/usr/lib/systemd/system/lemond.service.d" "$out"

# the payload: 1bit, its backends, the sealed NPU kernel sets, the licences
cp -a "$bundle/1bit" "$bundle/hrx" "$bundle/npu" "$bundle/lemonade-hrx" "$bundle/NOTICE" \
      "$bundle/NOTICE-BINARY" "$bundle/LICENSE" "$bundle/README.txt" "$root/opt/1bit/engine/"
chmod 0755 "$root/opt/1bit/engine/1bit"
install -m 0644 "$libhsa" "$root/opt/1bit/engine/hrx/llama/bin/libhsa-runtime64.so.1"
ln -s /opt/1bit/engine/1bit "$root/usr/bin/1bit-engine"

# the shim: the name the recipe resolves inside hrx_bin's directory. It is the bundle's own
# lemonade-hrx wrapper, which runs `1bit serve --print-backend-env` (serve's architecture and memory
# checks, and the environment serve gives an HRX backend: strict placement, the HSA runtime path,
# the CWSR save-area trio, ternary decode) and then execs the engine's llama-server on Lemonade's
# argv, with serve's defaults for what the argv left out (-ngl 99, --cache-reuse 256, -kvu above one
# slot). Lemonade's own build_server_environment() hands a backend only GGML_DISABLE_VULKAN=1, which
# is why the engine's wrapper and not our binary is what the recipe must start (engine#715).
[ -x "$root/opt/1bit/engine/lemonade-hrx" ] || {
    echo "the bundle has no lemonade-hrx wrapper; it is the shim this package installs" >&2; exit 1; }
mkdir -p "$root/opt/1bit/engine/lemonade-bin"
ln -s ../lemonade-hrx "$root/opt/1bit/engine/lemonade-bin/llama-server"

# wiring for an installed Lemonade: the unit drop-in (env form) and the config fragment (key form)
cat > "$root/usr/lib/systemd/system/lemond.service.d/10-1bit-engine.conf" <<'UNIT'
# 1bit-engine: run stock Lemonade's llamacpp-hrx recipe on this engine's llama-server.
# Only applies to a lemond started from this unit; the config key form (see the README) is equivalent.
[Service]
Environment=LEMONADE_HRX_HRX_BIN=/opt/1bit/engine/lemonade-bin
UNIT
cat > "$root/usr/share/1bit/lemonade-config-fragment.json" <<'FRAG'
{"hrx": {"hrx_bin": "/opt/1bit/engine/lemonade-bin"}}
FRAG

cat > "$root/usr/share/doc/$pkg/README.md" <<'DOC'
# 1bit-engine

The 1bit engine (HRX GPU kernels through llama.cpp's ggml-hrx backend, sealed NPU kernel sets) and
the shim that lets a **stock** Lemonade run it: no fork, no patch, nothing added upstream.

Userspace only: no kernel modules, no Python at runtime.

## Verify the payload

    1bit-engine --version
    /opt/1bit/engine/1bit serve --help | head -3

## Use it from an installed Lemonade

Either form is enough (they are equivalent):

    LEMONADE_HRX_HRX_BIN=/opt/1bit/engine/lemonade-bin          # environment for lemond
    {"hrx": {"hrx_bin": "/opt/1bit/engine/lemonade-bin"}}       # Lemonade's config.json

Restart lemond, then load any `*-HRX` model: Lemonade execs
`/opt/1bit/engine/lemonade-bin/llama-server` with
`-m <gguf> --ctx-size N --device HRX0 --port P --jinja --metrics --parallel 1`. That name is a link to
this package's `lemonade-hrx`, which asks `1bit serve` for the backend environment and serve's checks
before exec'ing the engine, so Lemonade's single `GGML_DISABLE_VULKAN=1` is not the whole story.

`docs/install-any-distro.md` in the engine repo has the host prerequisites per device and the
verification steps.
DOC
cp "$bundle/NOTICE" "$bundle/LICENSE" "$root/usr/share/doc/$pkg/"

size=$(du -sk --exclude=DEBIAN "$root" | cut -f1)
mkdir -p "$root/DEBIAN"
cat > "$root/DEBIAN/control" <<CTRL
Package: $pkg
Version: $ver
Architecture: amd64
Maintainer: bong-water-water-bong <bong-water-water-bong@1bit.gg>
Installed-Size: $size
Section: misc
Priority: optional
Depends: libc6 (>= 2.35), libstdc++6, libgomp1, libssl3t64 | libssl3, zlib1g, libzstd1, libpcre2-8-0
Recommends: lemonade-server
Description: 1bit engine - HRX GPU + sealed NPU kernels, plug-in for stock Lemonade
 The 1bit engine binary, its HRX llama.cpp build with the ggml-hrx backend and the HSA runtime
 library it dlopens, the sealed NPU kernel sets, and a shim that makes a stock Lemonade (its
 llamacpp-hrx recipe) run this engine with no patch and no fork.
 .
 Userspace only: no kernel modules, no Python at runtime.
CTRL

dpkg-deb --root-owner-group --build "$root" "$out/${pkg}_${ver}_amd64.deb" >/dev/null
tar -C "$root" --exclude=DEBIAN -I 'zstd -19 -T4' -cf "$out/${pkg}-${ver}-linux-x86_64.tar.zst" \
    opt usr/bin usr/share/1bit usr/share/doc usr/lib/systemd
echo "$out/${pkg}_${ver}_amd64.deb"
echo "$out/${pkg}-${ver}-linux-x86_64.tar.zst"
