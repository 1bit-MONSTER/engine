#!/usr/bin/env python3
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
#
# engine#3442 review (cl-hyperloom): verify_image must check the signature over, and take the digest from, the SAME
# SHA256SUMS bytes. An open() that serves the genuine signed bytes on the first read and an attacker's digest on any
# later one must not get a forged image through, and SHA256SUMS must be read exactly once. Also RFC 8032 5.1.3: a
# point whose y is not below p is refused. Usage: release_sign_toctou_check.py <tools dir>.
import builtins, hashlib, os, sys, tempfile
sys.path.insert(0, sys.argv[1])
import release_sign as rs, seal_manifest as ed
bad = 0
def want(name, cond):
    global bad
    print(("ok   " if cond else "FAIL ") + name); bad += 0 if cond else 1
d = tempfile.mkdtemp(prefix="rs-toctou.")
seed = bytes.fromhex("9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60")
pub = ed.public_key(seed)
good = os.urandom(1 << 16); evil = os.urandom(1 << 16)
img = os.path.join(d, "1bit-os-vT.img")
open(img, "wb").write(evil)                                   # the file on disk is the ATTACKER's image
sums = os.path.join(d, "SHA256SUMS"); sig = os.path.join(d, "SHA256SUMS.sig")
genuine = ("%s  1bit-os-vT.img\n" % hashlib.sha256(good).hexdigest()).encode()
forged = ("%s  1bit-os-vT.img\n" % hashlib.sha256(evil).hexdigest()).encode()
open(sums, "wb").write(genuine); open(sig, "w").write(rs.sign(sums, seed, "test-v1"))
real_open, reads = builtins.open, []
def swapping_open(path, mode="r", *a, **k):                   # the genuine signed bytes first, the forged ones after
    if os.path.abspath(str(path)) == sums and "b" in mode:
        reads.append(1)
        open_ = real_open(path + (".genuine" if len(reads) == 1 else ".forged"), mode, *a, **k)
        return open_
    return real_open(path, mode, *a, **k)
real_open(sums + ".genuine", "wb").write(genuine); real_open(sums + ".forged", "wb").write(forged)
builtins.open = swapping_open
try:
    ok, why = rs.verify_image(img, sums, sig, pub)
finally:
    builtins.open = real_open
want("a SHA256SUMS swapped between reads (genuine signed bytes first, the attacker's digest after) does NOT pass "
     "verify_image (%s)" % why[:70], not ok)
want("verify_image reads SHA256SUMS exactly once (read %d time(s))" % len(reads), len(reads) == 1)
# _decode: RFC 8032 5.1.3 - y >= p is not a point
p = 2 ** 255 - 19
for y, label in ((p, "y = p"), (p + 1, "y = p + 1 (aliases y = 1)")):
    try:
        ed._decode(int.to_bytes(y, 32, "little")); refused = False
    except ValueError:
        refused = True
    want("_decode refuses a non-canonical point, %s" % label, refused)
msg = b"m"; s = ed.sign(seed, msg)
ry = int.from_bytes(s[:32], "little") & ((1 << 255) - 1)
if ry + p < 1 << 255:   # a non-canonical alias of R exists only for small y; build it when it does
    r_alias = int.to_bytes((ry + p) | (int.from_bytes(s[:32], "little") & (1 << 255)), 32, "little")
    want("a signature whose R is re-encoded non-canonically does not verify", not ed.verify(pub, msg, r_alias + s[32:]))
want("the genuine signature still verifies", ed.verify(pub, msg, s))
sys.exit(1 if bad else 0)
