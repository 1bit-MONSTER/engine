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
# build-static.sh - build `1bit-console-tui` as ONE static binary and say where it is.
#
# The image build does not carry a Rust toolchain, so this runs here and its output is handed to
# container-build.sh as CONSOLE_TUI_BIN:
#
#   os/v3/puppy/console-tui/build-static.sh
#   CONSOLE_TUI_BIN=$PWD/os/v3/puppy/console-tui/target/x86_64-unknown-linux-musl/release/1bit-console-tui \
#       bash os/v3/puppy/build.sh resolute64
#
# musl-static and not glibc-dynamic: the appliance is a Puppy image whose glibc may move, and a console that
# cannot start because of a loader is a dead screen. The musl target links with the bundled rust-lld, so no
# musl-gcc is needed. `--selftest` proves the binary answers without a terminal.
#
# Exit: 0 built and static, 1 the toolchain or the resulting binary is not what the image needs, or the tree has
# uncommitted changes in this crate - in that last case the hash it prints is marked `-dirty`, because a hash from a
# dirty tree is not the artifact of any commit and cannot be quoted in a hand-off (engine#1326).
set -eu

cd "$(dirname "$0")"
TARGET=x86_64-unknown-linux-musl
OUT=target/$TARGET/release/1bit-console-tui

if ! command -v cargo > /dev/null 2>&1; then
    echo "build-static: no cargo in PATH (rustup default stable, then: rustup target add $TARGET)" >&2
    exit 1
fi
if ! rustup target list --installed 2> /dev/null | grep -qx "$TARGET"; then
    echo "build-static: the $TARGET target is not installed (rustup target add $TARGET)" >&2
    exit 1
fi

echo "build-static: cargo test (the probe parsers and the two console renders)"
cargo test --quiet
echo "build-static: cargo build --release --target $TARGET"
cargo build --release --target "$TARGET"

# "one static binary, no runtime deps" is the requirement, so check rather than trust: a dynamically linked build
# here would be a console that fails on a box whose loader path differs, which is the class of defect this image
# has already paid for once (libhsa).
if readelf -l "$OUT" 2> /dev/null | grep -q INTERP; then
    echo "build-static: REFUSED: $OUT is dynamically linked (needs an interpreter)" >&2
    exit 1
fi
if ldd "$OUT" 2>&1 | grep -q "=>"; then
    echo "build-static: REFUSED: $OUT has shared-library dependencies" >&2
    exit 1
fi

"$OUT" --version > /dev/null
"$OUT" --snapshot 80x25 --fixture > /dev/null

# An artifact hash is quoted in cycle hand-offs and pinned as CONSOLE_TUI_BIN, so it has to belong to a COMMIT.
# A dirty tree produces a hash nobody else can reproduce - and not only by a little: rustc embeds `file:line` in
# panic messages, so `cargo fmt` re-wrapping a line moved this binary by 24 bytes (engine#1326, where the hash of a
# pre-`cargo fmt` tree was quoted as if it were #1323's head, and the release lane could not tell it apart from a
# real difference). The check is scoped to this crate: those are the only files that can change these bytes, and
# the engine tree is big enough that an unrelated edit elsewhere must not block a console build.
hash=$(sha256sum "$OUT" | cut -c1-16)
bytes=$(stat -c %s "$OUT")
head=$(git rev-parse --short HEAD 2> /dev/null || echo unknown)
dirty=$(git status --porcelain -- . 2> /dev/null || true)
if [ -n "$dirty" ]; then
    echo "build-static: DIRTY $hash-dirty $bytes bytes - NOT $head's artifact: $(printf '%s\n' "$dirty" | wc -l | tr -d ' ') uncommitted path(s) under $PWD" >&2
    echo "build-static: commit them and run this again to get a hash that names a commit (the binary below is fine to test with)" >&2
    echo "$PWD/$OUT"
    exit 1
fi

echo "build-static: ok $hash $bytes bytes (HEAD $head, tree clean)"
echo "$PWD/$OUT"
