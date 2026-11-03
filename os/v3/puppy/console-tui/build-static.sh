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
# Exit: 0 built and static, 1 the toolchain or the resulting binary is not what the image needs.
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

echo "build-static: ok $(sha256sum "$OUT" | cut -c1-16) $(stat -c %s "$OUT") bytes"
echo "$PWD/$OUT"
