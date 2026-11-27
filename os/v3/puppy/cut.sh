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
# cut.sh <engine build dir> <tag> <out dir>
#
# Cuts a 1bit OS release image from a FRESH engine build (engine#3287). Run it from the engine tree that
# configured <engine build dir>; it refuses a build dir whose 1bit was built from another commit.
#
#   1. the engine package, from <engine build dir>, now        -> <out>/engine-pkg/1bit-<tag>-linux-x86_64.tar.zst
#   2. Lemonade (+ web app) at this tree's gitlink, now          -> <out>/lemonade
#   3. the payload tree, assembled from 1 + 2, now               -> <out>/lemonade-onebit-<tag>/
#   4. the image: WOOF=<woof-CE> KERNEL_DIR=<dir> builds the Puppy base with build.sh (PAYLOAD embedded, the
#      reproducible path); without WOOF, BASE=<osfiles dir> takes a released base (vmlinuz, initrd.gz and the four
#      SFS) and patches the payload + this tree's OS files into its puppy SFS with patch-sfs.sh - the base layers
#      are then NOT rebuilt, and BUILT-FROM.txt says so. Either way mkimage.sh writes <out>/image/1bit-os-puppy.img
#   5. image-probe.sh opens the image's OWN filesystem and must PASS against the payload tree of step 3
#   6. archive-image.sh keeps that exact image + its sha256 + the probe log + BUILT-FROM.txt under
#      $ONEBIT_IMAGE_ARCHIVE/<tag>/ (default ~/release-archive), and prints the path to write the stick from.
#
# What this fixes, each from the c34/c35 nights: every step-1..3 directory is removed and rebuilt in this run, so a
# re-roll cannot leave a stale staged lemonade-onebit-<tag>/ behind (one c34 image carried the previous engine);
# the pre-write probe exits non-zero on a mismatch AND when it cannot look (it used to print "skipped" and let the
# write proceed); and the tested image is archived by the cut itself (the v2026.47 one was lost with its build host).
#
# Env: ESP=<dir with BOOTX64.EFI grubx64.efi mmx64.efi> (the signed shim chain mkimage.sh embeds; required),
#      BASE=<osfiles dir> | WOOF=<woof-CE dir> KERNEL_DIR=<mk-huge-kernel out> [LIBHSA_DIR=<HSA runtime set>],
#      LEMONADE=<an already-built lemonade dir> (skip step 2; its lemond must report this tree's lemonade pin),
#      NODE_TARBALL=<the pinned Node.js tarball> (BASE cut: use a local copy instead of downloading pin.txt's node_url),
#      LOOM_WHEEL=<the pinned hrx-loom wheel>, LOOM_CARDS_FROM=<dir of the pinned cards> (local copies for stage-loom.sh),
#      ONEBIT_IMAGE_ARCHIVE=<archive root>, NO_ARCHIVE=1 (a dry cut: probe, no archive),
#      RELEASE_PUB=<the release public key: 32 raw bytes or 64 hex> (BASE cut: baked as /etc/1bit/release-pub, the key
#      the in-system updater verifies SHA256SUMS.sig with, tools/release_sign.py; without it the image carries no key
#      and the updater refuses every download).
# Needs passwordless sudo for the root steps (patch-sfs, base-modes, mkimage), as the release host has.
# Nothing here writes a stick, tags, uploads or publishes.
set -euo pipefail
prog=$(basename "$0")
say() { echo "[$(date '+%H:%M:%S')] $*"; }
die() { echo "$prog: $*" >&2; exit 1; }
usage() { awk 'NR>1 && /^#/ {sub(/^# ?/,""); print; next} NR>1 {exit}' "$0" | sed -n '/^cut.sh /,$p'; }
[ $# -eq 3 ] || { usage >&2; exit 2; }
here=$(cd "$(dirname "$0")" && pwd)
T=$(cd "$here/../../.." && pwd)
build=$(cd "$1" && pwd) || die "no build dir $1"
tag=$2
[[ "$tag" =~ ^[A-Za-z0-9._-]+$ ]] || die "tag '$tag' is not a plain name"
mkdir -p "$3"; out=$(cd "$3" && pwd)
logs=$out/logs; mkdir -p "$logs"
[ -n "${ESP:-}" ] && [ -f "$ESP/BOOTX64.EFI" ] && [ -f "$ESP/grubx64.efi" ] && [ -f "$ESP/mmx64.efi" ] ||
    die "ESP=<dir> must hold BOOTX64.EFI, grubx64.efi and mmx64.efi (the signed shim chain)"
if [ -n "${WOOF:-}" ]; then
    [ -n "${KERNEL_DIR:-}" ] || die "WOOF needs KERNEL_DIR=<the directory mk-huge-kernel.sh wrote>"
    mode=woof
else
    [ -n "${BASE:-}" ] || die "name the base: WOOF=<woof-CE> KERNEL_DIR=<dir> (rebuild) or BASE=<osfiles dir> (patch)"
    for f in vmlinuz initrd.gz; do [ -f "$BASE/$f" ] || die "BASE=$BASE has no $f"; done
    for l in puppy zdrv fdrv bdrv; do ls "$BASE"/${l}_*.sfs > /dev/null 2>&1 || die "BASE=$BASE has no ${l}_*.sfs"; done
    mode=base
fi
sudo -n true 2> /dev/null || die "needs passwordless sudo (patch-sfs, base-modes, mkimage run as root)"
head=$(git -C "$T" rev-parse HEAD)
say "cut $tag from engine $head (${mode} mode), build $build, out $out"

# ── 1. the engine package, from the build, now ────────────────────────────────────────────────────────────────
[ -x "$build/1bit" ] || die "$build has no 1bit (build the engine first)"
built=$("$build/1bit" version 2> /dev/null | sed -n 's/.*(engine \([0-9a-f]*\) .*/\1/p')
[ -n "$built" ] && [ "${head:0:${#built}}" = "$built" ] ||
    die "$build/1bit was built from engine '${built:-unknown}', this tree is ${head:0:12}: rebuild before cutting"
[ -z "$(git -C "$T" status --porcelain --untracked-files=no -- . ':!third_party')" ] ||
    die "the engine tree has uncommitted changes: a cut is of a commit"
rm -rf "$out/engine-pkg"
"$T/scripts/package-linux.sh" "$build" "$out/engine-pkg" "$tag" > "$logs/package-linux.log" 2>&1 ||
    { grep -v 'PHDR segment' "$logs/package-linux.log" | tail -20; die "package-linux failed"; }
pkg=$out/engine-pkg/1bit-$tag-linux-x86_64.tar.zst
[ -f "$pkg" ] || die "package-linux wrote no $pkg"
say "  engine package $(basename "$pkg") $(stat -c %s "$pkg") B"

# ── 2. Lemonade at this tree's pin, now ─────────────────────────────────────────────────────────────────────────
lpin=$(git -C "$T" ls-tree HEAD third_party/lemonade | awk '{print $3}')
if [ -n "${LEMONADE:-}" ]; then
    lem=$(cd "$LEMONADE" && pwd)
    say "  lemonade: taking the build at $lem"
else
    lem=$out/lemonade
    [ "$(git -C "$T/third_party/lemonade" rev-parse HEAD 2> /dev/null || true)" = "$lpin" ] ||
        die "third_party/lemonade is not at the gitlink ${lpin:0:12} (git submodule update --init third_party/lemonade)"
    # lemond must not link libwebsockets dynamically (the appliance has none): hide the system .pc so its own
    # vendored copy is built static - what the tested c35 payload was built with
    pc=$out/pkgconf; mkdir -p "$pc"
    printf '#!/bin/sh\nfor a in "$@"; do case "$a" in libwebsockets*) exit 1 ;; esac; done\nexec /usr/bin/pkg-config "$@"\n' > "$pc/pkg-config"
    chmod +x "$pc/pkg-config"
    rm -rf "$lem"
    PATH=$pc:$PATH PKG_CONFIG=$pc/pkg-config LEMONADE_WEB_APP=1 nice "$T/scripts/build-lemonade.sh" "$lem" \
        > "$logs/lemonade-build.log" 2>&1 || { tail -25 "$logs/lemonade-build.log"; die "lemonade build failed"; }
fi
[ -x "$lem/bin/lemond" ] || die "$lem has no bin/lemond"
! ldd "$lem/bin/lemond" 2> /dev/null | grep -q websockets || die "$lem/bin/lemond links libwebsockets dynamically"

# ── 3. the payload tree, now ────────────────────────────────────────────────────────────────────────────────────
P=$out/lemonade-onebit-$tag
sudo -n rm -rf "$P"; mkdir -p "$P/1bit"
cp -a "$lem/bin" "$P/"
zstd -dc "$pkg" | tar -C "$P/1bit" --strip-components=1 -xf - || die "cannot unpack $pkg into the payload"
install -m 0755 "$T/scripts/lemond-onebit.sh" "$P/lemond-onebit"
plem=$("$P/lemond-onebit" --version 2> /dev/null | sed -n 's/^lemond version //p' | head -1)
[ -n "$plem" ] || die "the payload's lemond-onebit --version printed no version"
case "$plem" in *"${lpin:0:8}"*) ;; *) die "the payload's lemond is $plem, the tree pins lemonade ${lpin:0:8}" ;; esac
# engine#3350 rc1: the Loom route's server (the hrx-loom wheel config/loom.config pins) and the Loom cards
# os/v3/puppy/loom-cards.pin lists, each file sha256-checked; the cards' weights stay out (1bit-loom-cards links them on
# /data at boot). LOOM_WHEEL=<file> / LOOM_CARDS_FROM=<dir> use local copies instead of downloading.
sh "$here/stage-loom.sh" "$P" ${LOOM_WHEEL:+--wheel "$LOOM_WHEEL"} ${LOOM_CARDS_FROM:+--cards-from "$LOOM_CARDS_FROM"} \
    2> "$logs/stage-loom.log" || { cat "$logs/stage-loom.log"; die "stage-loom.sh failed"; }
say "  $(tail -1 "$logs/stage-loom.log")"
# The RUNPATH gate covers the WHOLE payload, so it runs after every step that adds binaries, the Loom ones included.
# (#3377 staged them after this check: a future hrx-loom wheel or source build with a build-tree RUNPATH would ship.)
bad=$(find "$P" -type f -exec sh -c 'readelf -d "$1" 2>/dev/null | grep -E "R(UN)?PATH" | grep -v "\[\$ORIGIN\]"' _ {} \; | wc -l)
[ "$bad" = 0 ] || die "the payload has $bad RUNPATH(s) other than \$ORIGIN"
# the rest of scripts/weekly.sh stage_package's payload steps, in its order, on THIS copy (a LEMONADE= dir is never
# written): the catalogue offers only the NPU models whose kernel set the payload carries (engine#3287: v2026.47 listed
# Qwen3-1.7B-NPU-1bit with no set), links out of the tree (the web app's KaTeX fonts) become copies, then the modes
# fold. Both helpers are idempotent, so a LEMONADE= dir that already had them applied comes out the same.
python3 "$T/tools/npu_catalogue_filter.py" "$P/bin/resources/server_models.json" "$P/1bit/npu/kernels" \
    "$T/config/npu-catalogue-sets.json" > "$logs/npu-catalogue-filter.log" 2>&1 ||
    { cat "$logs/npu-catalogue-filter.log"; die "npu_catalogue_filter failed"; }
say "  $(tail -1 "$logs/npu-catalogue-filter.log")"
bash -c 'source "$1/scripts/pack-elf.sh"; pack_localize_links "$2"' _ "$T" "$P/bin" || die "pack_localize_links failed"
out_links=$(bash -c 'source "$1/scripts/pack-elf.sh"; pack_outside_links "$2"' _ "$T" "$P")
[ -z "$out_links" ] || die "the payload has links that leave it: $out_links"
bash -c 'source "$1/scripts/pack-elf.sh"; pack_normalise_modes "$2"' _ "$T" "$P"
[ "$(find "$P" \( -perm -0020 -o -perm -0002 \) ! -type l | wc -l)" = 0 ] || die "the payload has group/other-writable paths"
(cd "$P" && find . -type f -print0 | sort -z | xargs -0 sha256sum) > "$out/payload.sha256"
esha=$(sha256sum "$P/1bit/1bit" | cut -c1-64)
say "  payload $P: engine $esha, lemond $plem, $(wc -l < "$out/payload.sha256") files"

# ── 4. the image ────────────────────────────────────────────────────────────────────────────────────────────────
O=$out/osfiles; sudo -n rm -rf "$O" "$out/image"; mkdir -p "$O" "$out/image"
if [ "$mode" = woof ]; then
    say "== Puppy base from woof-CE (build.sh, PAYLOAD embedded)"
    sudo -n chown -R root:root "$P"
    PAYLOAD=$P ${LIBHSA_DIR:+LIBHSA_DIR=$LIBHSA_DIR} bash "$here/build.sh" resolute64 "$KERNEL_DIR" "$O" "$WOOF" \
        > "$logs/build.log" 2>&1 || { tail -25 "$logs/build.log"; die "build.sh failed"; }
    basedesc="woof-CE $(git -C "$WOOF" rev-parse --short=12 HEAD 2> /dev/null || echo '?') (rebuilt)"
else
    say "== Puppy base from $BASE (patched; base layers not rebuilt)"
    bsfs=$(ls "$BASE"/puppy_*.sfs | head -1); sfsname=$(basename "$bsfs")
    F=$out/osfiles-src; sudo -n rm -rf "$F"; mkdir -p "$F"
    M=$out/sfs.manifest
    printf 'opt/1bit-package\t%s\n' "$P" > "$M"
    # the OS files container-build.sh installs from this tree (keep in step with its install lines)
    (cd "$here/embed" && find . -type f | sed 's|^\./||' | sort) | while read -r f; do
        printf '%s\t%s\n' "$f" "$here/embed/$f"; done > "$M.os"
    while read -r dst src; do printf '%s\t%s\n' "$dst" "$T/$src"; done >> "$M.os" <<'EOF'
usr/local/bin/1bit-release-check os/mini/release-check.sh
usr/local/bin/1bit-selftest os/mini/selftest.sh
bin/1bit-xhci-capture tools/xhci_boot_capture.sh
usr/local/bin/1bit-install os/v3/puppy/1bit-install.sh
usr/sbin/1bit-board os/mini/board.sh
etc/1bit/base-modes-keep os/v3/puppy/base-modes-keep
usr/lib/1bit/seal_manifest.py tools/seal_manifest.py
usr/lib/1bit/release_sign.py tools/release_sign.py
usr/lib/1bit/merge_accounts.py tools/merge_accounts.py
EOF
    # generated from the payload, as container-build.sh does: the catalogue list and the identity
    python3 "$here/catalogue-manifest.py" "$P/bin/resources/server_models.json" "$out/onebit-models.txt" ||
        die "could not write onebit-models.txt from the payload catalogue"
    printf 'etc/1bit/onebit-models.txt\t%s\n' "$out/onebit-models.txt" >> "$M.os"
    # OS-UPDATER item 3: the release PUBLIC key the in-system updater verifies SHA256SUMS.sig with (tools/release_sign.py),
    # as one line of hex. It is a public key: nothing secret enters the image.
    if [ -n "${RELEASE_PUB:-}" ]; then
        python3 -I - "$RELEASE_PUB" "$out/release-pub" << 'PY' || die "RELEASE_PUB=$RELEASE_PUB is not a 32-byte Ed25519 public key (raw or hex)"
import sys
d = open(sys.argv[1], "rb").read(200)
k = d if len(d) == 32 else bytes.fromhex(d.decode("ascii").strip())
if len(k) != 32:
    raise SystemExit(1)
open(sys.argv[2], "w").write(k.hex() + "\n")
PY
        printf 'etc/1bit/release-pub\t%s\n' "$out/release-pub" >> "$M.os"
        say "  release public key baked: $(cut -c1-16 "$out/release-pub")... (/etc/1bit/release-pub)"
    else
        say "  NOTE: no RELEASE_PUB: this image carries no release public key, so its updater refuses every download"
    fi
    # the base's rc.shutdown with the 1bit-shutdown-watch hook, as container-build.sh does (engine#3287)
    sudo -n unsquashfs -q -cat "$bsfs" etc/rc.d/rc.shutdown > "$out/rc.shutdown" || die "the base SFS has no etc/rc.d/rc.shutdown"
    sh "$here/hook-rc-shutdown.sh" "$out/rc.shutdown" || die "could not hook 1bit-shutdown-watch into rc.shutdown"
    chmod 755 "$out/rc.shutdown"
    printf 'etc/rc.d/rc.shutdown\t%s\n' "$out/rc.shutdown" >> "$M.os"
    # the base's /etc/group without the desktop user in the root-equivalent groups (disk, kmem, ...), as container-build.sh does
    sudo -n unsquashfs -q -cat "$bsfs" etc/group > "$out/group" || die "the base SFS has no etc/group"
    sh "$here/harden-groups.sh" "$out/group" || die "could not take the non-root users out of the root-equivalent groups"
    chmod 644 "$out/group"
    printf 'etc/group\t%s\n' "$out/group" >> "$M.os"
    # the base's /etc/shadow with root's password field '*': the base ships the distribution's default root password,
    # which is public (su / sudo-with-the-root-password = root for any process). Mode 600 here; the image keeps the
    # base file's own 0600 root:root (patch-sfs.sh installs a replacement with the target's mode)
    ( umask 077; sudo -n unsquashfs -q -cat "$bsfs" etc/shadow > "$out/shadow" ) || die "the base SFS has no etc/shadow"
    sh "$here/lock-root-password.sh" "$out/shadow" || die "could not replace the default root password"
    chmod 600 "$out/shadow"
    printf 'etc/shadow\t%s\n' "$out/shadow" >> "$M.os"
    baseid=$(sudo -n unsquashfs -q -cat "$bsfs" etc/1bit/release-identity) || die "the base SFS has no etc/1bit/release-identity"
    hsa=$(echo "$baseid" | grep '^libhsa_sha256=') || die "the base identity names no libhsa_sha256"
    { echo "payload=lemonade-onebit-$tag"; echo "payload_lemond=$plem"; echo "engine_sha256=$esha"
      echo "$hsa"; echo "built=$(date -u +%FT%TZ)"; } > "$out/release-identity"
    printf 'etc/1bit/release-identity\t%s\n' "$out/release-identity" >> "$M.os"
    # the base's rc.country, with the RTC read as UTC, as container-build.sh does (engine#3308). rc.country is a base
    # file, not an embed/ one: without this a BASE cut ships the base's `set_hwclock_type --hctosys` under
    # HWCLOCKTIME=localtime, every boot runs 4 h ahead on an RTC that holds UTC, and 1bit-logs prunes the real boots
    sudo -n unsquashfs -q -cat "$bsfs" etc/rc.d/rc.country > "$out/rc.country" || die "the base SFS has no etc/rc.d/rc.country"
    sed -i -e "s|/usr/sbin/set_hwclock_type -q --hctosys|/usr/sbin/set_hwclock_type -q utc --hctosys|" \
        -e "s|/usr/sbin/set_hwclock_type cli --hctosys|/usr/sbin/set_hwclock_type cli utc --hctosys|" "$out/rc.country"
    grep -q '/usr/sbin/set_hwclock_type -q utc --hctosys' "$out/rc.country" &&
        ! grep -qE 'set_hwclock_type [a-z-]* *--hctosys' "$out/rc.country" ||
        die "REFUSED: rc.country still reads the RTC as local time (engine#3308)"
    chmod 755 "$out/rc.country"
    printf 'etc/rc.d/rc.country\t%s\n' "$out/rc.country" >> "$M.os"
    # stage root-owned with the install mode (755 executable, else 644)
    while IFS=$'\t' read -r dst src; do
        [ -f "$src" ] || die "manifest source $src is missing"
        if [ -x "$src" ]; then md=755; else md=644; fi
        sudo -n install -D -m $md -o root -g root "$src" "$F/$dst"
        printf '%s\t%s\n' "$dst" "$F/$dst"
    done < "$M.os" > "$M.staged"
    # Node.js + npm (engine#3310), as container-build.sh stages them: the pinned release, sha256-checked, with its exec
    # wrappers. stage-node.sh --manifest prints the patch-sfs rows (the tree, and the wrappers with their mode). A base
    # cut before #3311 has no Node.js, and without these rows the image fails release-check's `node` row.
    # NODE_TARBALL=<file> uses a local copy of the pinned tarball instead of downloading it (it must match the pin).
    N=$out/node-root; sudo -n rm -rf "$N"; mkdir -p "$N"
    sh "$here/stage-node.sh" "$N" ${NODE_TARBALL:+--tarball "$NODE_TARBALL"} --manifest > "$out/node.manifest" \
        2> "$logs/stage-node.log" || { cat "$logs/stage-node.log" >&2; die "Node.js could not be staged (stage-node.sh)"; }
    [ "$(wc -l < "$out/node.manifest")" -ge 2 ] || die "stage-node.sh printed no manifest rows"
    sudo -n chown -R root:root "$N"
    cat "$out/node.manifest" >> "$M.staged"
    # patch-sfs refuses a target whose bytes the base already has: keep only what differs (a tree is compared whole)
    X=$out/base-x; sudo -n rm -rf "$X"; cut -f1 "$M.staged" > "$out/os-paths.txt"
    sudo -n unsquashfs -q -n -d "$X" -ef "$out/os-paths.txt" "$bsfs" > /dev/null 2>&1 || true
    same=0
    while IFS=$'\t' read -r dst src md; do
        if [ -d "$src" ]; then
            sudo -n diff -r --no-dereference -q "$src" "$X/$dst" > /dev/null 2>&1 && { same=$((same + 1)); continue; }
        elif sudo -n cmp -s "$src" "$X/$dst"; then same=$((same + 1)); continue; fi
        printf '%s\t%s%s\n' "$dst" "$src" "${md:+$'\t'$md}" >> "$M"
    done < "$M.staged"
    sudo -n rm -rf "$X"; rm -f "$M.os" "$M.staged"
    say "  $same OS file(s) already identical in the base; patching $(($(wc -l < "$M") - 1)) + the payload"
    sudo -n chown -R root:root "$P"
    sudo -n bash "$here/patch-sfs.sh" --replace --keep-modes "$here/base-modes-keep" --manifest "$M" "$bsfs" "$O/$sfsname" \
        > "$logs/patch-sfs.log" 2>&1 || { tail -20 "$logs/patch-sfs.log"; die "patch-sfs failed"; }
    # the other layers get the engine#1474 normalisation (go-w only, no byte changes), each its own compression
    for l in bdrv fdrv zdrv; do
        src=$(ls "$BASE"/${l}_*.sfs | head -1); t=$out/layer-$l; sudo -n rm -rf "$t"
        sudo -n unsquashfs -q -n -d "$t" "$src"
        sudo -n bash "$T/scripts/base-modes.sh" "$t" --keep "$here/base-modes-keep" > "$logs/base-modes-$l.log" 2>&1 ||
            die "base-modes failed on $l"
        # a layer that carries its own account files gets the main SFS's treatment too: the overlay reads the FIRST
        # lower layer that has a file, and bdrv sits ABOVE the main SFS (lowerdir=/pup_b:/pup_ro2:/pup_f:/pup_z) with
        # its own passwd/group/shadow, so locking root only in the main SFS (#3552) left the base default in effect
        if sudo -n test -f "$t/etc/shadow"; then
            sudo -n sh "$here/lock-root-password.sh" "$t/etc/shadow" || die "could not replace the default root password in $l"
            say "  $l: root has no usable password in its own etc/shadow"
        fi
        if sudo -n test -f "$t/etc/group"; then
            sudo -n sh "$here/harden-groups.sh" "$t/etc/group" || die "could not harden $l's etc/group"
        fi
        case $l in zdrv) opts="-comp xz -b 128K" ;; *) opts="-comp zstd -Xcompression-level 19 -b 256K" ;; esac
        # shellcheck disable=SC2086
        sudo -n mksquashfs "$t" "$O/$(basename "$src")" $opts -no-exports -no-xattrs -noappend -quiet > /dev/null
        sudo -n rm -rf "$t"
    done
    # every layer of the image that carries an etc/shadow must ship root with no usable password (never printed)
    for f in "$O"/*.sfs; do
        rf=$(sudo -n unsquashfs -q -cat "$f" etc/shadow 2> /dev/null | awk -F: '$1 == "root" { print ($2 == "" ? "E" : substr($2, 1, 1)); exit }')   # E: empty = no password at all
        case $rf in ''|'*'|'!') ;; *) die "$(basename "$f") ships root with a usable password in etc/shadow" ;; esac
    done
    cp "$BASE/vmlinuz" "$BASE/initrd.gz" "$O/"
    sudo -n chown -R "$(id -u):$(id -g)" "$O"
    basedesc="patched from $BASE ($(sha256sum "$bsfs" | cut -c1-16) $sfsname; base layers not rebuilt)"
fi
say "== mkimage"
sudo -n env SHIM="$ESP/BOOTX64.EFI" GRUB="$ESP/grubx64.efi" MOK_MANAGER="$ESP/mmx64.efi" \
    bash "$here/mkimage.sh" "$O" "$out/image" > "$logs/mkimage.log" 2>&1 || { tail -20 "$logs/mkimage.log"; die "mkimage failed"; }
I=$out/image/1bit-os-puppy.img
isha=$(sha256sum "$I" | cut -c1-64)
say "  image $I $(stat -c %s "$I") B sha256 $isha"

# ── 5. the pre-write probe: the image's own filesystem, against the payload just built ──────────────────────────
say "== image-probe"
PROBE_WORK=$out/probe-work bash "$here/image-probe.sh" "$I" --payload "$P" --tag "$tag" 2>&1 | tee "$out/probe.log"
prc=${PIPESTATUS[0]}
[ "$prc" = 0 ] || die "the pre-write probe did not pass (exit $prc): this image must not be written"

{ echo "tag=$tag"; echo "engine=$head"; echo "lemonade=$lpin"; echo "lemond=$plem"
  echo "engine_sha256=$esha"; echo "image_sha256=$isha"; echo "base=$basedesc"; echo "cut=$(date -u +%FT%TZ)"; } > "$out/BUILT-FROM.txt"

# ── 6. keep the exact image ─────────────────────────────────────────────────────────────────────────────────────
if [ "${NO_ARCHIVE:-0}" = 1 ]; then
    say "NO_ARCHIVE=1: not archived. Image $I (sha256 $isha) - a dry cut; do not ship it"
    exit 0
fi
a=$(ARCHIVE_EXTRA="$out/probe.log $out/BUILT-FROM.txt $out/payload.sha256" bash "$here/archive-image.sh" "$I" "$tag") ||
    die "archiving failed: the image is not kept, so it is not a release image"
say "CUT $tag: $a (sha256 $isha). Write the stick FROM THIS PATH; re-check any time with:"
say "  os/v3/puppy/image-probe.sh $a --engine-sha $esha --tag $tag"
