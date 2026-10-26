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
# mkrepo.sh <debs-dir> <repo-dir> [gnupg-home] [key-id]
#
# Lays the debs out as an apt repository and signs it: pool/main/<letter>/<source>/<deb>, the
# Packages index, Release, and InRelease/Release.gpg from the archive key. With no key given the
# Release file is produced unsigned and the script says so - useful for a local run, never for
# publishing. The result is a directory of static files: publish it to 1bit.gg/apt as it is.
set -euo pipefail
[ $# -ge 2 ] || { sed -n '21,27p' "$0"; exit 2; }
debs=$1 repo=$2 gnupg=${3:-} keyid=${4:-}
[ -d "$debs" ] || { echo "mkrepo: no $debs"; exit 2; }
dist=$repo/dists/stable
mkdir -p "$dist/main/binary-amd64" "$repo/pool/main"
# each deb into pool/main/<first letter of its source name>/<source>/, which is what apt's index
# records as its location - a deb dropped anywhere else still installs, but the layout stays boring
for d in "$debs"/*.deb; do
    [ -e "$d" ] || continue
    pkg=$(dpkg-deb -f "$d" Package)
    src=${pkg#1bit-}
    src=${src:-appliance}
    mkdir -p "$repo/pool/main/${src:0:1}/1bit-$src"
    cp -f "$d" "$repo/pool/main/${src:0:1}/1bit-$src/"
done
( cd "$repo" && apt-ftparchive packages pool > "$dist/main/binary-amd64/Packages" )
gzip -9 -kf "$dist/main/binary-amd64/Packages"
( cd "$repo" && apt-ftparchive \
    -o APT::FTPArchive::Release::Origin="1bit" \
    -o APT::FTPArchive::Release::Label="1bit OS" \
    -o APT::FTPArchive::Release::Suite="stable" \
    -o APT::FTPArchive::Release::Codename="stable" \
    -o APT::FTPArchive::Release::Architectures="amd64" \
    -o APT::FTPArchive::Release::Components="main" \
    -o APT::FTPArchive::Release::Description="1bit OS, Lemonade Server appliance - packages" \
    release "$dist" > "$dist/Release" )
rm -f "$dist/Release.gpg" "$dist/InRelease"
if [ -n "$gnupg" ] && [ -n "$keyid" ]; then
    [ -d "$gnupg" ] || { echo "mkrepo: no GNUPGHOME at $gnupg"; exit 2; }
    GNUPGHOME=$gnupg gpg --batch --yes -u "$keyid" --detach-sign -o "$dist/Release.gpg" "$dist/Release"
    GNUPGHOME=$gnupg gpg --batch --yes -u "$keyid" --clearsign -o "$dist/InRelease" "$dist/Release"
    echo "mkrepo: signed with $keyid"
else
    echo "mkrepo: WARNING - Release is UNSIGNED (no key given); fine for a local run, never publish it" >&2
fi
echo "mkrepo: $(ls "$repo"/pool/main/*/*/*.deb | wc -l) packages, $(wc -c < "$dist/Release") bytes of Release"
