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
# Packs the engine as a SEPARATE package that plugs into a stock Lemonade (no fork, no patch), as a
# native package for every family (scripts/pack-distro.sh), out of ONE payload tree:
#   <out-dir>/1bit-engine_<version>_amd64.deb            Debian, Ubuntu
#   <out-dir>/1bit-engine-<version>-1.x86_64.rpm         Fedora, openSUSE Tumbleweed
#   <out-dir>/1bit-engine-<version>-1-x86_64.pkg.tar.zst Arch Linux
#   <out-dir>/1bit-engine-<version>-linux-x86_64.tar.zst everything else
# The four install the same files at the same paths, which is why they are built from the same tree:
# only the wrapping differs.
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

# usage(): the header comment as prose, anchored on the comment block rather than a line range -- two of these ranges had
# drifted past the end of their header and printed code to whoever asked for help (engine#2129).
usage() { awk 'NR>1 && /^#/ {sub(/^# ?/,""); print; next} NR>1 {exit}' "$0"; }

[ $# -ge 3 ] || { usage; exit 2; }
bundle=$(cd "$1" && pwd) out=$2 ver=$3
[ -x "$bundle/1bit" ] || { echo "not a 1bit bundle: $bundle (no 1bit binary)" >&2; exit 1; }
[ -d "$bundle/hrx/llama/bin" ] || { echo "not a 1bit bundle: $bundle (no hrx/llama/bin)" >&2; exit 1; }

# Only a runtime that registers gfx1151 will do. The distro's libhsa-runtime64 rejects the
# HSA_AMD_AGENT_INFO_PM4_EMULATION probe on gfx1151, so HRX then registers no device (docs/hrx.md:932-941) —
# and a clean `ldd` says nothing about that probe. So the bundle's own runtime is the default and there is no
# distro fallback; a bundle without one has to name a TheRock/gfx1151 runtime explicitly.
libhsa=${LIBHSA:-$bundle/hrx/llama/bin/libhsa-runtime64.so.1}
[ -f "$libhsa" ] || {
    echo "no libhsa-runtime64.so.1 at $libhsa; set LIBHSA=<the TheRock/gfx1151 runtime>" >&2
    echo "the distro's ROCr is not a substitute: it rejects the PM4-emulation probe on gfx1151 and HRX gets no device" >&2
    exit 1; }

pkg=1bit-engine
root=$(mktemp -d)/root
trap 'rm -rf "$(dirname "$root")"' EXIT
mkdir -p "$root/opt/1bit/engine" "$root/usr/share/doc/$pkg" "$root/usr/share/1bit" \
         "$root/usr/bin" "$root/usr/lib/systemd/system/lemond.service.d" "$out"

# the payload: 1bit, its backends, the sealed NPU kernel sets, the licences. LICENSE-BINARY travels with
# NOTICE-BINARY because LICENSE-BINARY section 2 ("every copy is accompanied by this license and
# NOTICE-BINARY") covers the HRX/NPU binaries in this payload, and package-linux.sh ships both in the
# bundle for the same reason.
cp -a "$bundle/1bit" "$bundle/hrx" "$bundle/npu" "$bundle/lemonade-hrx" "$bundle/NOTICE" \
      "$bundle/NOTICE-BINARY" "$bundle/LICENSE" "$bundle/LICENSE-BINARY" "$bundle/README.txt" \
      "$root/opt/1bit/engine/"
chmod 0755 "$root/opt/1bit/engine/1bit"

# The GGUF-on-NPU path is public helper scripts, not code compiled into 1bit: the binary looks for them
# BESIDE ITSELF (app/built_path.h:44-56 -> packaged_path("scripts/repack_gguf.py") at app/serve.cpp:2482,
# packaged_path("tools/gguf_to_q4nx.py") at app/serve.cpp:2589). Without them `1bit serve -m model.gguf
# --device npu` fails with "Q4NX repack script not found" even with python3 and the converter installed, so
# the payload keeps the two directories package-linux.sh stages (scripts/package-linux.sh:98-105).
for d in "$bundle/scripts" "$bundle/tools"; do
    [ -d "$d" ] || { echo "not a 1bit bundle: $d is missing (the NPU helper scripts travel beside 1bit)" >&2; exit 1; }
done
cp -a "$bundle/scripts" "$bundle/tools" "$root/opt/1bit/engine/"

# The HSA runtime is NOT a standalone file. A TheRock libhsa lists TheRock's own libraries in NEEDED
# (librocm_sysdeps_elf.so.1, librocm_sysdeps_drm.so.2, librocm_sysdeps_numa.so.1, librocm_sysdeps_z*…,
# librocprofiler-register.so.0) and carries RUNPATH=$ORIGIN, so the set has to land FLAT beside it or the
# backend's dlopen fails and HRX registers no device. Measured 2026-11-02 on the appliance's own libhsa
# (sha256 f6ef689d…): shipped as one file in /opt/1bit/libhsa it answered "librocm_sysdeps_elf.so.1: cannot
# open shared object file", and the board booted with a GPU it could not use. ldd is the arbiter here: the
# private siblings are copied in, and anything still unresolved refuses the build.
hsa_dir=$root/opt/1bit/engine/hrx/llama/bin
install -m 0644 "$libhsa" "$hsa_dir/libhsa-runtime64.so.1"
srcdir=$(cd "$(dirname "$libhsa")" && pwd)
libdirs=$srcdir:$srcdir/rocm_sysdeps/lib:$srcdir/../../_rocm_sdk_core/lib:$srcdir/../../_rocm_sdk_core/lib/rocm_sysdeps/lib:${LD_LIBRARY_PATH:-}
copied=0
for so in $(LD_LIBRARY_PATH=$libdirs ldd "$libhsa" | awk '/=> \// {print $3}' | sort -u); do
    case "$so" in /lib/* | /usr/lib/*) continue ;; esac   # the distro's own: dpkg-deb Depends, not ours to ship
    install -m 0644 "$so" "$hsa_dir/$(basename "$so")"
    copied=$((copied + 1))
done
# The ldd closure above is not the whole set: libhsa OPENS the aqlprofile agent by name at run time
# (libhsa-amd-aqlprofile64.so.1, the profiling interface), so ldd cannot see it and a payload built from
# ldd alone ships a runtime that reaches for it wherever the machine happens to offer it - which is how
# engine #1411's c22 install ended up mapping 64 files out of a builder's toolchain. It ships beside the
# runtime with the rest, and the build refuses if it is not where the runtime was found.
for pat in libhsa-amd-aqlprofile64.so*; do
    aq=$(ls "$srcdir"/$pat 2>/dev/null | head -1)
    [ -n "$aq" ] || { echo "no $pat beside $libhsa: the HSA runtime opens the aqlprofile agent by name at run time, so a payload without it resolves it from wherever the machine has one (engine #1411)" >&2; exit 1; }
    install -m 0644 "$aq" "$hsa_dir/$(basename "$aq")"
    copied=$((copied + 1))
done
missing=$(LD_LIBRARY_PATH=$hsa_dir ldd "$hsa_dir/libhsa-runtime64.so.1" | awk '/not found/ {print $1}' | tr '\n' ' ')
[ -z "${missing// /}" ] || {
    echo "libhsa needs $missing and this build could not stage it beside the runtime;
pass LIBHSA=<a runtime whose sibling libraries are next to it> (the release-gated TheRock set)" >&2
    exit 1; }
echo "   HSA runtime: $(basename "$libhsa") + $copied sibling libraries, ldd clean"
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

# What the .deb declares about the package. The .rpm and the Arch package declare their own in their
# recipes (packaging/rpm/1bit-engine.spec, packaging/arch/PKGBUILD): the dependency names are each
# family's, and a maintainer of one family should not have to read this script to find them.
export PKG_NAME=$pkg PKG_VERSION=$ver
export PKG_SUMMARY='1bit engine: HRX GPU kernels and the sealed NPU kernel sets, plug-in for stock Lemonade'
export PKG_DESC="The 1bit engine binary, its HRX llama.cpp build with the ggml-hrx backend and the HSA runtime
library it dlopens, the sealed NPU kernel sets, and a shim that makes a stock Lemonade (its
llamacpp-hrx recipe) run this engine with no patch and no fork.

Userspace only: no kernel modules, no Python at runtime."
export PKG_MAINTAINER='bong-water-water-bong <bong-water-water-bong@1bit.gg>'
# The floor is what the payload really has, not a guess: readelf -V on its 1bit reads
# sqrtf@GLIBC_2.43, so on an older distribution the .deb would install and then refuse to load
# (docs/install-any-distro.md, "Ubuntu 26.04"). The .rpm gets the same floor from rpm itself.
export PKG_DEB_DEPENDS='libc6 (>= 2.43), libstdc++6, libgomp1, libssl3t64 | libssl3, zlib1g, libzstd1, libpcre2-8-0'
export PKG_DEB_RECOMMENDS='lemonade-server'

for fmt in deb rpm arch tarball; do "$(cd "$(dirname "$0")" && pwd)/pack-distro.sh" "$fmt" "$root" "$out"; done
