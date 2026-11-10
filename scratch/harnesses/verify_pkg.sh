#!/usr/bin/env bash
# Verify a package tarball the way a user's audit would, from the archive rather than from the staging tree.
#
# Scratch harness (not engine tree). The point: #1361/#1435 normalise modes at STAGING time, and the only thing that
# proves the fix survives into the bytes a user downloads is to unpack the artifact and look. It also proves the
# assert added in package-linux.sh cannot false-fail a correct roll -- a gate that reds a working build is worse than
# the defect it prevents (engine#1352's own warning).
#
# usage: verify_pkg.sh <tarball.tar.zst>
set -uo pipefail

pkg=${1:?usage: verify_pkg.sh <tarball.tar.zst>}
[ -f "$pkg" ] || { echo "no such tarball: $pkg" >&2; exit 2; }
. "$(dirname "$0")/../rlk-drop/scripts/pack-elf.sh"

# THE AUTHORITATIVE NUMBER COMES FROM THE ARCHIVE, NOT FROM AN EXTRACTION.  `tar -x` creates a top-level directory
# that the archive may not carry explicitly (the lemonade fold's archive stores `…/bin/` as its first member, so the
# parent is synthesized on extract) -- and it does so with the EXTRACTOR's umask.  That invented a phantom `. 775`
# root for a payload whose archive has no writable entries at all.  Count group/other-write bits in `tar -tv` output;
# positions: mode[1]=type, mode[6]=group-w, mode[9]=other-w; symlinks are skipped (their mode is not used).
arch_bad=$(tar --zstd -tvf "$pkg" 2>/dev/null | awk '{t=substr($1,1,1); if (t!="l" && (substr($1,6,1)=="w"||substr($1,9,1)=="w")) c++} END{print c+0}')
arch_root=$(tar --zstd -tvf "$pkg" 2>/dev/null | head -1 | awk '{n=split($NF,a,"/"); print ($NF ~ /\/$/ && n==2) ? $1 : "none (synthesized on extract)"}')
echo "archive-level writable entries: $arch_bad   (explicit top-level entry: $arch_root)"

w=$(mktemp -d); trap 'rm -rf "$w"' EXIT
tar --zstd -C "$w" -xf "$pkg" || { echo "cannot unpack $pkg" >&2; exit 1; }
top=$(find "$w" -maxdepth 1 -mindepth 1 -type d | head -1)
[ -n "$top" ] || { echo "archive has no top-level directory" >&2; exit 1; }

echo "archive: $(basename "$pkg")  ($(stat -c%s "$pkg") B)"
echo "top: $(basename "$top")"
echo "files: $(find "$top" -type f | wc -l)   symlinks: $(find "$top" -type l | wc -l)   dirs: $(find "$top" -type d | wc -l)"
echo
echo "mode histogram (files):"
find "$top" -type f -printf '%m\n' | sort | uniq -c | sort -rn | sed 's/^/  /'
echo
bad=$(pack_writable_files "$top")
if [ -z "$bad" ]; then
    echo "group/other-writable: NONE  <- the property #1352 asked for"
else
    echo "group/other-writable:"
    echo "$bad" | sed 's/^/  /'
fi
echo
echo "symlinks still links (sample):"
find "$top" -type l -printf '  %P -> %l\n' | head -3
echo
# An executable file that lost its bit is the regression a mode pass could cause; count both ways.
echo "executable files: $(find "$top" -type f -perm -u+x | wc -l)"
find "$top" -type f -perm -u+x -printf '  %P\n' | head -6
[ -z "$bad" ] || exit 1
