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
# build.sh <compat> <kernel-dir> <out-dir> [woof-ce-dir]
#
# Build a 1bit Puppy from woof-CE in a throwaway container: no host package is installed, no host device
# is touched, the container is capped (12 GB memory, 4 CPUs) and nothing survives it except <out-dir>.
#
#   <compat>      resolute64 (Ubuntu 26.04) - the only one that can run the release: our `1bit` and
#                 `lemond` need GLIBC_2.43 (measured), and noble64's Puppy has 2.39
#   <kernel-dir>  the directory os/v3/puppy/mk-huge-kernel.sh wrote: huge-<kver>.tar.zst
#   <out-dir>     where the built .iso and .sfs land
#   [woof-ce-dir] the woof-CE clone (default $HOME/wt/woof-CE)
#
# LIBHSA_DIR=<dir> installs the WHOLE release-gated HSA runtime set at /opt/rocm-therock/lib: the pinned
# libhsa-runtime64.so.1 plus every sibling it needs at load time (the TheRock `librocm_sysdeps_*` set and
# librocprofiler-register.so.0). Cycle 13's board boot measured why the set and not one file (engine#715 +
# the 2 Nov hardware run):
#   * nothing bakes an HSA path into `1bit` by default (engine#1459 / #1460): `ONEBIT_HRX_LIBHSA` is a CMake
#     cache value the build fills by globbing ONEBIT_HRX_TOOLCHAIN, kept as build metadata because mkimage reads
#     it to decide which runtime to SHIP; `-DONEBIT_HRX_LIBHSA_BAKE=<path>` is the explicit opt-in that compiles
#     one in. app/serve.cpp:201 resolves in the order --hrx-libhsa, a baked path if one was asked for, then a
#     recursive scan of /opt/rocm-therock - so on the appliance the scan is the route;
#   * the pinned libhsa carries `RUNPATH=$ORIGIN` and NEEDED librocm_sysdeps_{elf,drm,drm_amdgpu,numa} +
#     librocprofiler-register, so a lone file cannot load even when handed to the loader: measured `ldd` = 4
#     `not found`. With the release kit's own directory on the path: 0 not found.
# So the set is installed flat, /opt/1bit/libhsa becomes a symlink to it (v2's path and the identity row keep
# working), and the build refuses a set that is incomplete or that `ldd` cannot resolve - the silent-absence
# class this lane keeps paying for (an image with no HRX runtime looks exactly like an image whose HRX runtime
# nobody loaded). Without LIBHSA_DIR the build warns, as before.
#
# PAYLOAD=<dir> embeds the release package (TPUPPY-RELEASE U3/U4): the unpacked
# lemonade-onebit-<tag>/ tree, whose lemond-onebit launcher resolves bin/ and 1bit/ BESIDE ITSELF. It is
# copied to /opt/1bit-package, /etc/init.d/10lemonade from os/v3/puppy/embed starts it at boot (Puppy's
# rc.services runs every executable /etc/init.d script), and the build refuses an image whose SFS set lacks
# the engine or the launcher. Without PAYLOAD the image is the plain Puppy base, as before.
#
# Two things the woof-CE build assumes that are not obvious, both measured here:
#   * merge2out refuses to run unless $GITHUB_ACTIONS is set or the host is a running Puppy - CI sets it,
#     so a container build has to as well;
#   * `losetup` needs a /dev/loopN node, and the container only has 0-7: this script mknods them, because
#     this box keeps ~20 loop devices busy (snaps, other lanes) and the free one is numbered above 7;
#   * `3builddistro` takes the kernel from `woof-out_*/kernel-kit/output/` and otherwise downloads a stock
#     Ubuntu kernel. Dropping our huge tarball there is what makes the image boots OUR kernel.
#
# Inside the container: merge2out -> kernel into kernel-kit/output -> 0setup (package indexes) ->
# 1download (the Ubuntu packages) -> 2createpackages (Puppy's split into puppy/zdrv/fdrv/bdrv SFS) ->
# 3builddistro release (the SFS files and the bootable ISO).
set -euo pipefail
[ $# -ge 3 ] || { sed -n '/^# build.sh/,/^# Inside the container/p' "$0" | sed 's/^# \{0,1\}//'; exit 1; }
compat=$1 kernel=$2 out=$3
# /tmp survives a session, not a reboot: the default clone lives under $HOME (measured 2026-11-01, the
# first re-run after a restart stopped at "no woof-CE at /tmp/woof-ce")
woof=${4:-$HOME/wt/woof-CE}
# The build tree must live on a real filesystem that allows device nodes, for two measured reasons:
#   * 3builddistro mounts an overlay for its petbuild chroots and overlay-on-overlay is refused ("mount:
#     ... bad superblock on petbuild", the chroot has no bash, the build stops at busybox) - a container's
#     own rootfs is overlay, so the build root is a host directory bind-mounted in;
#   * debootstrap refuses a target on a nodev mount ("cannot create ... test-dev-null ... mounted with
#     noexec or nodev"), and /tmp on this box is exactly that: tmpfs, nodev.
# So the default is under $HOME on the root filesystem, and a build root that cannot host the build is
# refused before 20 minutes of downloads.
build_root=${BUILD_ROOT:-$HOME/wt/1bit-puppy-build}
mkdir -p "$build_root"; build_root=$(cd "$build_root" && pwd)
mnt_opts=$(findmnt -T "$build_root" -no OPTIONS 2>/dev/null || true)
case ",$mnt_opts," in
*,nodev,* | *,noexec,*)
    printf 'build: REFUSED: %s is mounted %s; the build needs device nodes and executables there\n' \
        "$build_root" "$mnt_opts" >&2
    exit 1
    ;;
esac
[ -f "$woof/merge2out" ] || { echo "build: no woof-CE at $woof - clone https://github.com/puppylinux-woof-CE/woof-CE there, or pass the directory as the 4th argument" >&2; exit 1; }
# The same trap one level in: $HOME/wt/woof-CE is a symlink into the build root on this box, and container-build.sh
# starts with `rm -rf /work/woof-CE; cp -a /woof /work/woof-CE` - so a /woof that RESOLVES inside the build root
# is the directory the copy is writing to, and those two lines empty the clone they are copying from. Measured on
# 2026-11-03 (cycle-17 build): an empty root-owned woof-CE and a failure three lines later at "no such file
# woof-code/3builddistro", which reads like an upstream change and is really a self-copy.
woof_real=$(cd "$woof" 2>/dev/null && pwd -P) || { echo "build: cannot resolve $woof" >&2; exit 1; }
case "$woof_real/" in
"$build_root"/*)
    printf 'build: REFUSED: %s resolves to %s, inside the build root %s; the container would copy that directory into\n' \
        "$woof" "$woof_real" "$build_root" >&2
    printf 'build:           itself and empty it. Pass a woof-CE clone that lives outside the build root (4th argument).\n' >&2
    exit 1 ;;
esac
kernel=$(cd "$kernel" && pwd)
ls "$kernel"/huge-*.tar.* >/dev/null 2>&1 || {
    echo "build: no huge-*.tar.* in $kernel (run mk-huge-kernel.sh first)" >&2; exit 1; }
mkdir -p "$out"; out=$(cd "$out" && pwd)
recipe=$(cd "$(dirname "$0")" && pwd)   # os/v3/puppy, mounted at /recipe: packages-extra lives there
payload=""
if [ -n "${PAYLOAD:-}" ]; then
    payload=$(cd "$PAYLOAD" 2>/dev/null && pwd) || { echo "build: PAYLOAD=$PAYLOAD does not exist" >&2; exit 1; }
    [ -x "$payload/lemond-onebit" ] && [ -x "$payload/1bit/1bit" ] || {
        echo "build: $payload is not the release package: want lemond-onebit and 1bit/1bit in it" >&2; exit 1; }
fi
libhsa_dir=""
if [ -n "${LIBHSA_DIR:-}" ]; then
    libhsa_dir=$(cd "$LIBHSA_DIR" 2>/dev/null && pwd) || { echo "build: LIBHSA_DIR=$LIBHSA_DIR does not exist" >&2; exit 1; }
    if [ -n "${LIBHSA:-}" ]; then
        echo "build: LIBHSA is the cycle-13 spelling; a lone libhsa cannot load (its \$ORIGIN siblings are needed)" >&2
        echo "build: pass LIBHSA_DIR=<dir> holding libhsa-runtime64.so.1 and the librocm_sysdeps set" >&2
        exit 1
    fi
fi
# Every name the pinned libhsa NEEDs, plus the sysdeps' own deps. ldd on the box: all four of the first group
# are "not found" with the file alone.
libhsa_set="libhsa-runtime64.so.1 librocm_sysdeps_elf.so.1 librocm_sysdeps_drm.so.2 \
librocm_sysdeps_drm_amdgpu.so.1 librocm_sysdeps_numa.so.1 librocprofiler-register.so.0 \
librocm_sysdeps_z.so.1 librocm_sysdeps_zstd.so.1 librocm_sysdeps_liblzma.so.5 librocm_sysdeps_bz2.so"
if [ -n "$libhsa_dir" ]; then
    miss=""
    for f in $libhsa_set; do [ -f "$libhsa_dir/$f" ] || miss="$miss $f"; done
    [ -z "$miss" ] || {
        echo "build: REFUSED: LIBHSA_DIR=$libhsa_dir is missing:$miss" >&2
        echo "build:         (the release kit's 1bit-llama directory holds the full set beside its libhsa)" >&2
        exit 1
    }
fi
# The release public key that verifies the sealed NPU sets' manifests, and the engine scripts the U4/U6 rows
# need inside the image. All public material: the private seal key never enters the image or the container.
sealpub=${SEALPUB:-$HOME/.config/1bit-npu-key/seal.pub}
[ -f "$sealpub" ] || {
    echo "build: no release seal public key at $sealpub (tools/seal_manifest.py keygen writes one)" >&2; exit 1; }
repo=$(cd "$recipe/../../.." && pwd)   # the engine tree: os/mini/release-check.sh, tools/seal_manifest.py
[ -f "$repo/os/mini/release-check.sh" ] || { echo "build: $repo does not look like the engine tree" >&2; exit 1; }
free_gb=$(free -g | awk '/^Mem:/{print $7}')
[ "${free_gb:-0}" -ge 16 ] || echo "build: WARNING: ${free_gb} GB available; the container is capped at 12 GB" >&2

# The console TUI (#921) is a Rust binary built off this box by os/v3/puppy/console-tui/build-static.sh: no Rust
# toolchain in this container, so it is handed in. Two things are needed and BOTH were missing until cycle 19 -
# docker forwards only what `-e` names, and the file has to be mounted to exist inside: a host CONSOLE_TUI_BIN
# therefore reached container-build.sh as the empty string and the image silently came out with the dialog menu
# (measured: cycle 19's first build printed "note: CONSOLE_TUI_BIN is not set" with the variable set on the host).
tui_bin=${CONSOLE_TUI_BIN:-}
if [ -n "$tui_bin" ]; then
    [ -f "$tui_bin" ] || { echo "build: REFUSED: CONSOLE_TUI_BIN=$tui_bin does not exist" >&2; exit 1; }
    tui_bin=$(cd "$(dirname "$tui_bin")" && pwd)/$(basename "$tui_bin")
    # container-build.sh installs and verifies the TUI inside its PAYLOAD block (it is part of the embedded
    # appliance, not of a bare Puppy rootfs), so a build with CONSOLE_TUI_BIN and no PAYLOAD would accept the
    # binary here and then skip it in silence - the same failure this file just fixed one layer up. Refuse it.
    if [ -z "${payload:-}" ]; then
        echo "build: REFUSED: CONSOLE_TUI_BIN=$tui_bin needs PAYLOAD - the console TUI is installed in the" >&2
        echo "  embedded-package block of container-build.sh, so a plain-Puppy build would silently omit it" >&2
        exit 1
    fi
    echo "build: console TUI $tui_bin ($(sha256sum "$tui_bin" | cut -c1-16))"
fi

echo "build: woof-CE $woof, kernel $kernel, out $out, build root $build_root ($mnt_opts)"
# The container side is its own file (os/v3/puppy/container-build.sh, mounted at /recipe) since 2026-11-02:
# it was a 300-line single-quoted `bash -c '...'` string here, where one apostrophe in a comment ended the
# quote and bash reported a syntax error a hundred lines away - five builds in this lane, two of them in
# cycle 13 - and where shellcheck could not check it at all. Parsing it here costs a second and catches a
# broken recipe before the image pull and the 20-minute downloads.
[ -f "$recipe/container-build.sh" ] || { echo "build: no $recipe/container-build.sh" >&2; exit 1; }
bash -n "$recipe/container-build.sh" || { echo "build: REFUSED: $recipe/container-build.sh does not parse" >&2; exit 1; }
exec docker run --rm --privileged --memory=12g --cpus=4 \
    -v "$woof":/woof:ro -v "$build_root":/work -v "$kernel":/kernels:ro -v "$out":/out \
    -v "$recipe":/recipe:ro -v "$repo":/repo:ro -v "$sealpub":/seal.pub:ro \
    ${payload:+-v "$payload":/payload:ro} \
    ${payload:+-e PAYLOAD=/payload -e PAYLOAD_NAME="$(basename "$payload")"} \
    ${libhsa_dir:+-v "$libhsa_dir":/libhsa-runtime:ro} \
    ${libhsa_dir:+-e LIBHSA_DIR=/libhsa-runtime} \
    ${tui_bin:+-v "$tui_bin":/console-tui:ro} \
    ${tui_bin:+-e CONSOLE_TUI_BIN=/console-tui} \
    --device /dev/loop-control:/dev/loop-control \
    -e COMPAT="$compat" ubuntu:24.04 bash /recipe/container-build.sh
