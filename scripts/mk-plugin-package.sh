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
cp -a "$bundle/1bit" "$bundle/hrx" "$bundle/npu" "$bundle/NOTICE" "$bundle/NOTICE-BINARY" \
      "$bundle/LICENSE" "$bundle/README.txt" "$root/opt/1bit/engine/"
chmod 0755 "$root/opt/1bit/engine/1bit"
install -m 0644 "$libhsa" "$root/opt/1bit/engine/hrx/llama/bin/libhsa-runtime64.so.1"
ln -s /opt/1bit/engine/1bit "$root/usr/bin/1bit-engine"

# the shim: stock Lemonade's llamacpp-hrx recipe hands its backend only GGML_DISABLE_VULKAN=1, so
# the engine's HRX environment is set here. Without the trio the HRX kernels take the ROCr SVM path
# and a nightly libhsa page-faults (engine#715). A value the caller set wins.
mkdir -p "$root/opt/1bit/engine/lemonade-bin"
cat > "$root/opt/1bit/engine/lemonade-bin/llama-server" <<'SHIM'
#!/bin/sh
HSA_USE_SVM=${HSA_USE_SVM:-0}
HSA_USERPTR_FOR_PAGED_MEM=${HSA_USERPTR_FOR_PAGED_MEM:-0}
HRX_ALLOW_NO_SVM=${HRX_ALLOW_NO_SVM:-1}
GGML_HRX_STRICT=${GGML_HRX_STRICT:-1}
export HSA_USE_SVM HSA_USERPTR_FOR_PAGED_MEM HRX_ALLOW_NO_SVM GGML_HRX_STRICT
# the engine's own llama.cpp default for HRX: every layer on the device
case " $* " in *" -ngl "*) ;; *) set -- "$@" -ngl 99 ;; esac
exec "$(dirname "$0")/../hrx/llama/bin/llama-server" "$@"
SHIM
chmod 0755 "$root/opt/1bit/engine/lemonade-bin/llama-server"

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
`-m <gguf> --ctx-size N --device HRX0 --port P --jinja --metrics --parallel 1`.

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
