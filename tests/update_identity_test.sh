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
# Security review (cl-kernel via the orchestrator, 24 Nov): --latest must install only the release it ASKED for.
# GitHub's tag_name is unsigned, so an old genuine signed image re-published under a newer tag would verify. The
# install reads the tag INSIDE the verified image (etc/1bit/release-identity in its puppy_*.sfs) and
# update_identity_check refuses: no identity, a tag other than the one asked for, not strictly newer than the running
# release, an -rc on the stable channel. Sourced with ONEBIT_UPDATE_LIB=1: no root, no disk.
set -u
here=$(cd "$(dirname "$0")/.." && pwd)
u=$here/os/v3/puppy/embed/usr/local/bin/1bit-update
t=$(mktemp -d) || exit 1
bad=0
trap 'rm -rf "${t:?}"' EXIT
check() { if [ "$1" = 0 ]; then echo "ok   $2"; else echo "FAIL $2"; bad=$((bad + 1)); fi; }
ic() { ( ONEBIT_UPDATE_LIB=1 LOG=/dev/null; . "$u"; update_identity_check "$@" ) > "$t/why" 2>&1; }
gt() { ( ONEBIT_UPDATE_LIB=1 LOG=/dev/null; . "$u"; update_ver_gt "$@" ); }
ic v2026.50 v2026.50 v2026.48 stable; check $? "the release asked for, newer than running, on stable: installed"
ic v2026.50-rc1 v2026.50-rc1 v2026.48 test; check $? "a release candidate on the test channel: installed"
ic v2026.40 v2026.50 v2026.48 stable; [ $? = 1 ] && grep -q "re-published older image" "$t/why"
check $? "an old signed image re-published under a newer tag (image v2026.40, asked v2026.50): refused"
ic v2026.48 v2026.48 v2026.48 stable; [ $? = 1 ] && grep -q "not newer than the running" "$t/why"; check $? "the running release again: refused"
ic v2026.47 v2026.47 v2026.48 stable; [ $? = 1 ] && grep -q "not newer than the running" "$t/why"; check $? "an older release (a rollback): refused"
ic v2026.50-rc1 v2026.50-rc1 v2026.48 stable; [ $? = 1 ] && grep -q "release candidate and the channel is stable" "$t/why"
check $? "an -rc on the stable channel: refused"
ic '' v2026.50 v2026.48 stable; [ $? = 1 ] && grep -q "carries no release identity" "$t/why"; check $? "an image with no identity: refused"
ic v2026.50 v2026.50 '' stable; [ $? = 1 ]; check $? "no running identity to compare with: refused"
gt v2026.49 v2026.49-rc1 && gt v2026.49-rc2 v2026.49-rc1 && gt v2026.50 v2026.49 && ! gt v2026.49-rc1 v2026.49 && ! gt v2026.49 v2026.49 && ! gt junk v2026.1
check $? "version order: final > its rcs, rc2 > rc1, a later week > an earlier one; equal and junk are not newer"
# the install's read: the identity out of a real puppy_*.sfs, with the same unsquashfs line
if command -v mksquashfs > /dev/null 2>&1 && command -v unsquashfs > /dev/null 2>&1; then
    mkdir -p "$t/root/etc/1bit"; printf 'payload=lemonade-onebit-v2026.50\npayload_lemond=x\n' > "$t/root/etc/1bit/release-identity"
    mksquashfs "$t/root" "$t/puppy_test.sfs" -quiet -noappend > /dev/null 2>&1
    it=$(unsquashfs -q -cat "$t/puppy_test.sfs" etc/1bit/release-identity 2> /dev/null | sed -n 's/^payload=lemonade-onebit-//p' | head -1)
    [ "$it" = v2026.50 ]; check $? "the tag is read out of the image's puppy_*.sfs ($it)"
else
    echo "info no mksquashfs/unsquashfs here: the sfs read is covered by the board run, not this machine"
fi
grep -q '^expect_tag= expect_ch= expect_run=' "$u"; check $? "expect_tag starts empty in every run (never taken from the environment)"

echo "update_identity_test: $([ $bad = 0 ] && echo ok || echo FAIL)"
[ $bad = 0 ]
