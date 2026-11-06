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
# patch-sfs.sh <src.sfs> <path-in-sfs> <replacement> <out.sfs>
# patch-sfs.sh --manifest <manifest> <src.sfs> <out.sfs>
#
# The manifest form is the same tool for a cycle that changes SEVERAL paths (c29: six - the console menu and TUI, the
# release check, the installer, the selftest the image never shipped, the lemonade service's models dir, and the whole
# embedded payload tree). One line per path, `<path-in-sfs><TAB><host-source>`, `#` and blank lines ignored. A source may
# be a DIRECTORY (the payload roll: `opt/1bit-package<TAB>/home/.../lemonade-onebit-rc-...`), and a target the source SFS
# does not have is an ADDITION rather than a refusal, because a new file is a normal thing for a cycle to carry.
# A third, optional field is the MODE for a path the SFS does not have yet, and it is a fact about the image rather than
# about the file it came from: the repo keeps `os/mini/selftest.sh` 0775 and the build installs it 0755, so an addition
# that inherited the source mode shipped a group-writable executable in /usr/local/bin and looked fine in every log.
#
# Everything the single-file form proves is still proven, one target at a time: the trees must differ only inside the
# declared set (and each declared target must actually differ), the repack must unpack to the tree that was packed, and
# the metadata comparison covers the declared set in both directions. Additionally the packed SFS is refused if ANY
# entry has uid or gid >= 1000: replacing a 160 MB payload tree as root is exactly where a host uid rides into an image.
#
# A cycle that changes ONE file inside the appliance's puppy SFS (c28: /usr/local/bin/1bit-release-check, engine
# #1177) does not need a 20-minute woof-CE rebuild - and a rebuild would NOT be a smaller change: it re-downloads
# the Ubuntu package set, so every byte that happened to drift that day would ride into a demo image. This
# unpacks the shipped SFS, replaces that one file, and repacks it with woof-CE's own options (its out-dir's
# SFSCOMP, measured: `-comp zstd -Xcompression-level 19 -b 256K -no-exports -no-xattrs`).
#
# The proof is the point, and it is strict - nothing here is trusted because the command exited 0:
#   * the source is unpacked to <work>/src, hard-linked to <work>/new (so the copy costs no space);
#   * `diff -r` <work>/src <work>/new must name exactly ONE file: the one that was replaced;
#   * the packed SFS is unpacked again and `diff -r` <work>/new <work>/verify must be EMPTY, so the repack lost
#     nothing and carries the tree that was packed;
#   * mode/owner/type/size are compared explicitly, in both directions, because `diff -r` compares NAMES AND CONTENTS
#     ONLY: measured on a two-file SFS, a tree whose `sub/` went 0700 -> 0777 and `sub/su` 0644 -> 0755 diffs clean. The
#     rootfs carries 17 setuid/setgid entries `fix-ownership.sh` restores, so that gap is the whole point of running as
#     root. The replacement also INHERITS the source file's mode, so patching a 04755 target cannot silently un-setuid it.
# Any of those failing is a non-zero exit and no output file.
#
# Run as root (uid/gid/setuid bits are part of the artifact), like mkimage.sh, and in $HOME - /tmp here is
# tmpfs and an unpacked rootfs is 1.5 GB.
set -euo pipefail
prog=$(basename "$0")
usage() { sed -n "/^# $prog /,/^# Run as root/p" "$0" | sed 's/^# \{0,1\}//'; }
[ $# -eq 4 ] || { usage >&2; exit 1; }
manifest=''
if [ "${1:-}" = --manifest ]; then manifest=$2; src=$3; out=$4; target=''; repl=''; else src=$1 target=$2 repl=$3 out=$4; fi
targets=(); repls=(); modes=()
if [ -n "$manifest" ]; then
    [ -f "$manifest" ] || { echo "$prog: no manifest at $manifest" >&2; exit 1; }
    while IFS=$'\t' read -r t r m; do
        case ${t:-} in ''|'#'*) continue;; esac
        [ -n "${r:-}" ] || { echo "$prog: manifest line has a path and no source (TAB-separated): $t" >&2; exit 1; }
        case $t in /*|*..*) echo "$prog: manifest path must be relative and must not climb: $t" >&2; exit 1;; esac
        case ${m:-} in ''|[0-7][0-7][0-7]|[0-7][0-7][0-7][0-7]) ;; *) echo "$prog: manifest mode for $t is not octal: $m" >&2; exit 1;; esac
        targets+=("$t"); repls+=("$r"); modes+=("${m:-}")
    done < "$manifest"
    [ ${#targets[@]} -gt 0 ] || { echo "$prog: $manifest declares no paths" >&2; exit 1; }
else
    targets=("$target"); repls=("$repl"); modes=("")
fi
for f in "$src" "${repls[@]}"; do [ -f "$f" ] || [ -d "$f" ] || { echo "$prog: no $f" >&2; exit 1; }; done
[ "$(id -u)" = 0 ] || { echo "$prog: run as root (uid/gid and setuid bits are part of the artifact)" >&2; exit 1; }
case $out in /*) ;; *) echo "$prog: <out.sfs> must be an absolute path" >&2; exit 1;; esac
[ -e "$out" ] && { echo "$prog: $out exists; refusing to overwrite a packed SFS" >&2; exit 1; }
for t in unsquashfs mksquashfs diff sha256sum; do command -v "$t" >/dev/null || { echo "$prog: no $t" >&2; exit 1; }; done

work=$(mktemp -d "${SFS_WORK:-$HOME/wt/sfs-patch}/run.XXXXXX" 2>/dev/null || true)
[ -n "$work" ] || { mkdir -p "${SFS_WORK:-$HOME/wt/sfs-patch}"; work=$(mktemp -d "${SFS_WORK:-$HOME/wt/sfs-patch}/run.XXXXXX"); }
trap 'rm -rf "$work"' EXIT
# The unpacked rootfs is ~1.5 GB and is unpacked twice; a tmpfs work dir would fill RAM instead of dying.
case $(df --output=fstype "$work" | tail -1) in tmpfs|ramfs) echo "$prog: $work is on tmpfs; set SFS_WORK to a real filesystem" >&2; exit 1;; esac

echo "$prog: source $src ($(stat -c%s "$src") B, sha256 $(sha256sum "$src" | cut -c1-16)...)"
echo "$prog: unpacking to $work/src"
unsquashfs -q -n -d "$work/src" "$src" > "$work/unsquashfs.log" 2>&1 ||
    { echo "$prog: unsquashfs failed (see $work/unsquashfs.log)"; cat "$work/unsquashfs.log" >&2; exit 1; }
# What the source SFS must and must not have, checked before anything is copied: a file target must be a regular file or
# absent (absent = an addition), a directory target must be a directory or absent.
for i in "${!targets[@]}"; do
    t=${targets[$i]} r=${repls[$i]}
    if [ -d "$r" ]; then
        [ ! -e "$work/src/$t" ] || [ -d "$work/src/$t" ] ||
            { echo "$prog: $t is a file in $src and the manifest gives a directory" >&2; exit 1; }
    elif [ -e "$work/src/$t" ] && [ ! -f "$work/src/$t" ]; then
        echo "$prog: $t is not a regular file in $src" >&2; exit 1
    fi
done

cp -al "$work/src" "$work/new"          # hard links: the 55,664 unmodified paths cost nothing twice
for i in "${!targets[@]}"; do
    t=${targets[$i]} r=${repls[$i]}
    if [ -d "$r" ]; then
        rm -rf "$work/new/$t"           # break the link into the source tree, and the directories below it
        mkdir -p "$(dirname "$work/new/$t")"
        cp -a "$r" "$work/new/$t"
        # A host uid in the image is the c15f finding (a non-root-writable /etc = root code), so the copy is root's and
        # the packed SFS is refused below if any entry still is not.
        chown -R root:root "$work/new/$t"
        echo "$prog: $t (tree, $(find "$r" | wc -l) paths, mode $(stat -c%a "$r")) -> root:root"
    elif [ -e "$work/src/$t" ]; then
        old_sha=$(sha256sum "$work/src/$t" | cut -d' ' -f1)
        new_sha=$(sha256sum "$r" | cut -d' ' -f1)
        [ "$old_sha" != "$new_sha" ] || { echo "$prog: $t in $src already has these bytes ($new_sha); nothing to patch" >&2; exit 1; }
        rm -f "$work/new/$t"             # break that link, so the source tree keeps the old bytes
        install -m "$(stat -c%a "$work/src/$t")" -o root -g root "$r" "$work/new/$t"   # inherit the mode
        echo "$prog: $t $old_sha -> $new_sha"
    else
        m=${modes[$i]:-$(stat -c%a "$r")}   # the declared mode, or the source's when the manifest does not state one
        install -D -m "$m" -o root -g root "$r" "$work/new/$t"
        echo "$prog: $t (new, $(stat -c%s "$r") B, mode $m)"
    fi
done

# A line counts as declared if it names a target, or names both the target's directory and its own name: `diff -rq`
# writes an addition or a removal as `Only in <dir>: <name>`, where the whole path never appears.
matches_target() { case $1 in *"$2"*) return 0;; esac; case $1 in *"$(dirname "$2")"*"$(basename "$2")"*) return 0;; esac; return 1; }
declared() { local t; for t in "${targets[@]}"; do matches_target "$1" "$t" && return 0; done; return 1; }
declared_in() { local l; while IFS= read -r l; do [ -n "$l" ] || continue; declared "$l" || printf '%s\n' "$l"; done; }

changed=$(diff -rq --no-dereference "$work/src" "$work/new" || true)
n=$(printf '%s\n' "$changed" | grep -c . || true)
printf '%s\n' "$changed" | sed -n '1,5p'
[ "$n" -ge 1 ] || { echo "$prog: REFUSED: the patched tree is identical to the source tree" >&2; exit 1; }
undeclared=$(printf '%s\n' "$changed" | declared_in || true)
[ -z "$undeclared" ] || {
    echo "$prog: REFUSED: $n path(s) differ and these are outside the declared set:" >&2
    printf '%s\n' "$undeclared" | sed -n '1,10p' >&2; rm -f "$out"; exit 1; }
if [ -z "$manifest" ]; then
    [ "$n" = 1 ] || { echo "$prog: REFUSED: $n paths differ between the source tree and the patched tree, not 1" >&2; exit 1; }
    case $changed in *"$target"*) ;; *) echo "$prog: REFUSED: the one difference is not $target" >&2; exit 1;; esac
else
    # Every declared target has to be one of the differences: a manifest row whose bytes are already in the image is a
    # claim the artifact does not carry, and a cut that silently ships nothing is the failure mode this tool exists for.
    for i in "${!targets[@]}"; do
        t=${targets[$i]}
        hit=$(printf '%s\n' "$changed" | while IFS= read -r l; do [ -n "$l" ] && matches_target "$l" "$t" && { echo y; break; }; done)
        [ -n "$hit" ] || { echo "$prog: REFUSED: $t is declared but the patched tree does not differ there" >&2; rm -f "$out"; exit 1; }
    done
    echo "$prog: $n path line(s) differ from the source; all of them inside the declared set (${#targets[@]} targets)"
fi

echo "$prog: packing with woof-CE's options (-comp zstd -Xcompression-level 19 -b 256K -no-exports -no-xattrs)"
mksquashfs "$work/new" "$out" -comp zstd -Xcompression-level 19 -b 256K -no-exports -no-xattrs -noappend \
    -processors "${SFS_JOBS:-$(nproc)}" > "$work/mksquashfs.log" 2>&1 ||
    { echo "$prog: mksquashfs failed:" >&2; tail -5 "$work/mksquashfs.log" >&2; rm -f "$out"; exit 1; }

echo "$prog: verifying the packed SFS unpacks to the tree that was packed"
unsquashfs -q -n -d "$work/verify" "$out" > "$work/verify.log" 2>&1 ||
    { echo "$prog: the new SFS does not unpack:" >&2; cat "$work/verify.log" >&2; rm -f "$out"; exit 1; }
back=$(diff -rq --no-dereference "$work/new" "$work/verify" || true)
[ -z "$back" ] || { echo "$prog: REFUSED: the repack is not the tree that was packed:"
                    printf '%s\n' "$back" | sed -n '1,10p' >&2; rm -f "$out"; exit 1; }
for i in "${!targets[@]}"; do
    t=${targets[$i]} r=${repls[$i]}
    [ -d "$r" ] && continue
    [ "$(sha256sum "$work/verify/$t" | cut -d' ' -f1)" = "$(sha256sum "$r" | cut -d' ' -f1)" ] ||
        { echo "$prog: REFUSED: the SFS does not carry $t's bytes" >&2; rm -f "$out"; exit 1; }
done

# Ownership of the whole packed image, not just of what was written here: `fix-ownership.sh` restores 17 setuid/setgid
# entries and the build path already refuses a uid >= 1000 anywhere, so a patch path that writes as root must too.
badown=$(unsquashfs -lln "$out" 2>/dev/null | awk '{ split($2, a, "/"); if (a[1]+0 >= 1000 || a[2]+0 >= 1000) print }' | sed -n '1,20p')
[ -z "$badown" ] || {
    echo "$prog: REFUSED: entries owned by uid or gid >= 1000 rode into the packed SFS:" >&2
    printf '%s\n' "$badown" >&2; rm -f "$out"; exit 1; }
echo "$prog: ownership: no entry in the packed SFS has uid or gid >= 1000"

# Metadata, which `diff -r` cannot see: mode, owner, type and size over both trees, in both directions. A directory's
# SIZE is normalised away: on the host filesystem it is an allocation artifact of the order `cp -al` created the entries
# in (measured: `usr/share/icons/PMaterial/64/actions` 16,384 vs 12,288 B for two trees with identical contents), and it
# says nothing about the artifact - the first run of this check refused a cut for exactly that, which is why the field is
# '-' for a directory and only a regular file's size is compared.
tree_meta() { ( cd "$1" && find . -printf '%m %u %g %y %s %p\n' | awk '{ if ($4 == "d") $5 = "-"; print }' | sort ); }
tree_meta "$work/src" > "$work/meta-src"; tree_meta "$work/new" > "$work/meta-new"
dmeta=$(diff "$work/meta-src" "$work/meta-new" || true)
dn=$(printf '%s\n' "$dmeta" | grep -c '^[<>]' || true)
if [ -z "$manifest" ]; then
    dbad=$(printf '%s\n' "$dmeta" | grep '^[<>]' | grep -vc "\./$target\$" || true)
    [ "$dbad" = 0 ] && [ "$dn" -le 2 ] || {
        echo "$prog: REFUSED: mode/owner/type/size differ in $dbad path(s) besides $target:" >&2
        printf '%s\n' "$dmeta" | grep '^[<>]' | grep -v "\./$target\$" | sed -n '1,8p' >&2; rm -f "$out"; exit 1; }
else
    mbabad=$(printf '%s\n' "$dmeta" | grep '^[<>]' | cut -c3- | declared_in || true)
    [ -z "$mbabad" ] || {
        echo "$prog: REFUSED: mode/owner/type/size differ outside the declared set:" >&2
        printf '%s\n' "$mbabad" | sed -n '1,8p' >&2; rm -f "$out"; exit 1; }
fi
tree_meta "$work/verify" > "$work/meta-verify"
vmeta=$(diff "$work/meta-new" "$work/meta-verify" || true)
[ -z "$vmeta" ] || {
    echo "$prog: REFUSED: the repack's modes/owners/types/sizes are not the packed tree's:" >&2
    printf '%s\n' "$vmeta" | sed -n '1,8p' >&2; rm -f "$out"; exit 1; }
echo "$prog: metadata: $dn line(s) differ from the source, all inside the declared set; repack matches it exactly"

echo "$prog: OK  $out  ($(stat -c%s "$out") B, sha256 $(sha256sum "$out" | cut -c1-16)..., $(find "$work/new" | wc -l) paths)"
if [ -z "$manifest" ]; then
    echo "$prog: source was $(stat -c%s "$src") B; one file differs: $target"
else
    echo "$prog: source was $(stat -c%s "$src") B; ${#targets[@]} declared paths differ: ${targets[*]}"
fi
