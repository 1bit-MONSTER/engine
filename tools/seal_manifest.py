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
"""Signed manifests for the sealed NPU kernel sets (engine #574).

A sealed set is `<set>/layout.json` plus `<set>/layer1_*/layer1.elf`, and every kernel is a
`1BPKCRYP` ciphertext file. Sealing buys confidentiality and nothing else: a set re-sealed by
another key, or a set swapped for another one that also decrypts, looks no different to the
engine beyond "the key check passed". #574 adds authenticity *on top of* sealing (#548 was
closed as not planned for replacing sealing instead of adding to it): `manifest.json` lists
every file the set ships with the sha256 of its **ciphertext**, and `manifest.sig` is an
Ed25519 signature over the exact bytes of `manifest.json`, so covering the manifest covers the
ciphertext files through their digests. The loader - the private NPU runtime, which holds only
the release public key - verifies the signature; this tool is the public half: it writes the
manifest and signs it, and checks a set against it holding only the public key.

Python 3 standard library only, deliberately. Not every machine that has to check a set can be
trusted to have a crypto module or openssl: the OS image is busybox plus the release, and
`os/mini/release-check.sh` has to prove the stick's sets against the same rule. So the Ed25519
below is the RFC 8032 formulation over a 32-byte seed, signing and verifying, and nothing else.

  seal_manifest.py keygen <priv> <pub>
      A throwaway key pair, for tests and for a set nobody ships. The release signing key stays
      on the release machine in ONEBIT_NPU_KEY_DIR, beside the kernel key, and never enters a
      package.

  seal_manifest.py write <set-dir> --key <priv>
      Writes <set-dir>/manifest.json and <set-dir>/manifest.sig over every other file in the set.

  seal_manifest.py check <set-dir> [--pub <pub>]
      The rule that needs no secret: every file the set ships is listed with the sha256 of its
      bytes, the manifest lists nothing the set does not ship, and the set has a manifest at all.
      With --pub it also verifies manifest.sig - `verify` is the same thing, spelled out.

  seal_manifest.py verify <set-dir> --pub <pub>
      check, and manifest.sig must verify with the release public key.

Formats, fixed so a shell can read them too (`os/mini/release-check.sh` does, in busybox sh):

  manifest.json   one file per line, keys in this order, files sorted by path:
                      {"path": "layer1_1/layer1.elf", "size": 4096, "sha256": "<64 hex>"}
                  The manifest never lists itself or manifest.sig.
  private key     64 hex characters: the 32-byte Ed25519 seed, one line.
  public key      64 hex characters: the 32-byte Ed25519 public key, one line.
  manifest.sig    128 hex characters: the 64-byte signature, one line.

Nothing here seals or unseals anything, and nothing here mints a release: the signature is only
as good as the public key the loader was built with, which the private runtime owns.
"""

import hashlib
import json
import os
import sys

FORMAT = 1
SELF = ("manifest.json", "manifest.sig")
MAGIC = b"1BPKCRYP"


# ── Ed25519 (RFC 8032), 32-byte seed in, 64-byte signature out ────────────────────────────────
_P = 2 ** 255 - 19
_L = 2 ** 252 + 27742317777372353535851937790883648493
_D = -121665 * pow(121666, _P - 2, _P) % _P
_I = pow(2, (_P - 1) // 4, _P)


def _x_recover(y, sign):
    x2 = (y * y - 1) * pow(_D * y * y + 1, _P - 2, _P) % _P
    if x2 == 0:
        if sign:
            raise ValueError("not a curve point")
        return 0
    x = pow(x2, (_P + 3) // 8, _P)
    if (x * x - x2) % _P:
        x = x * _I % _P
    if (x * x - x2) % _P:
        raise ValueError("not a curve point")
    if (x & 1) != sign:
        x = _P - x
    return x


_GY = 4 * pow(5, _P - 2, _P) % _P
_G = (_x_recover(_GY, 0), _GY, 1, _x_recover(_GY, 0) * _GY % _P)


def _add(a, b):
    x1, y1, z1, t1 = a
    x2, y2, z2, t2 = b
    p = (y1 - x1) * (y2 - x2) % _P
    q = (y1 + x1) * (y2 + x2) % _P
    r = t1 * 2 * _D * t2 % _P
    s = z1 * 2 * z2 % _P
    return ((q - p) * (s - r) % _P, (s + r) * (q + p) % _P,
            (s - r) * (s + r) % _P, (q - p) * (q + p) % _P)


def _mul(k, point):
    out = (0, 1, 1, 0)
    while k:
        if k & 1:
            out = _add(out, point)
        point = _add(point, point)
        k >>= 1
    return out


def _encode(point):
    x, y, z, _ = point
    zi = pow(z, _P - 2, _P)
    x, y = x * zi % _P, y * zi % _P
    return int.to_bytes(y | ((x & 1) << 255), 32, "little")


def _decode(data):
    if len(data) != 32:
        raise ValueError("a point is 32 bytes")
    y = int.from_bytes(data, "little") & ((1 << 255) - 1)
    sign = data[31] >> 7
    x = _x_recover(y, sign)
    return (x, y, 1, x * y % _P)


def _clamp(seed):
    h = hashlib.sha512(seed).digest()
    a = int.from_bytes(h[:32], "little")
    a &= (1 << 254) - 8
    a |= 1 << 254
    return a, h[32:]


def public_key(seed):
    return _encode(_mul(_clamp(seed)[0], _G))


def sign(seed, msg):
    a, prefix = _clamp(seed)
    pub = _encode(_mul(a, _G))
    r = int.from_bytes(hashlib.sha512(prefix + msg).digest(), "little") % _L
    big_r = _encode(_mul(r, _G))
    k = int.from_bytes(hashlib.sha512(big_r + pub + msg).digest(), "little") % _L
    return big_r + int.to_bytes((r + k * a) % _L, 32, "little")


def verify(pub, msg, sig):
    """True when `sig` is `msg`'s signature by `pub`, cofactorless, as RFC 8032 allows."""
    if len(pub) != 32 or len(sig) != 64:
        return False
    s = int.from_bytes(sig[32:], "little")
    if s >= _L:
        return False
    try:
        a, big_r = _decode(pub), _decode(sig[:32])
    except ValueError:
        return False
    k = int.from_bytes(hashlib.sha512(sig[:32] + pub + msg).digest(), "little") % _L
    return _encode(_mul(s, _G)) == _encode(_add(big_r, _mul(k, a)))


# ── keys on disk: one line of hex each, so a C++ loader reads them as bytes and nothing else ────
def _read_hex(path, n, what):
    try:
        text = open(path, "r", encoding="ascii").read()
    except OSError as e:
        raise SystemExit("seal_manifest: cannot read the %s key %s: %s" % (what, path, e))
    text = text.strip()
    if len(text) != 2 * n:
        raise SystemExit("seal_manifest: the %s key %s is %d hex characters, want %d (%d bytes): "
                         "keygen writes that form" % (what, path, len(text), 2 * n, n))
    try:
        return bytes.fromhex(text)
    except ValueError:
        raise SystemExit("seal_manifest: the %s key %s is not hex" % (what, path))


def _write_hex(path, data):
    with open(path, "w", encoding="ascii") as f:
        f.write(data.hex() + "\n")


# ── the set and its manifest ──────────────────────────────────────────────────────────────────
def _members(setdir):
    """Every file the set ships, as paths relative to it, manifest files excluded."""
    out = []
    for root, dirs, files in os.walk(setdir):
        dirs.sort()
        for name in sorted(files):
            if root == setdir and name in SELF:
                continue
            out.append(os.path.relpath(os.path.join(root, name), setdir))
    return sorted(out)


def _digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write(setdir, key_path):
    """Write manifest.json over the set's ciphertext + manifest.sig, signed with `key_path`."""
    if not os.path.isdir(setdir):
        raise SystemExit("seal_manifest: no such set directory: %s" % setdir)
    seed = _read_hex(key_path, 32, "private")
    entries = []
    for rel in _members(setdir):
        full = os.path.join(setdir, rel)
        entries.append('    {"path": %s, "size": %d, "sha256": "%s"}'
                       % (json.dumps(rel), os.path.getsize(full), _digest(full)))
    body = ('{\n  "format": %d,\n  "set": %s,\n  "files": [\n%s%s\n  ]\n}\n'
            % (FORMAT, json.dumps(os.path.basename(os.path.abspath(setdir))),
               ",\n".join(entries), "" if entries else ""))
    manifest = os.path.join(setdir, "manifest.json")
    with open(manifest, "w", encoding="utf-8") as f:
        f.write(body)
    sig = sign(seed, body.encode("utf-8"))
    _write_hex(os.path.join(setdir, "manifest.sig"), sig)
    print("ok   %s: %d file(s) signed, manifest %d bytes, sha256 %s"
          % (os.path.basename(setdir), len(entries), len(body),
             hashlib.sha256(body.encode("utf-8")).hexdigest()[:16]))
    return 0


def check(setdir, pub=None):
    """The manifest rule. `pub` also makes manifest.sig mandatory and verifies it."""
    problems = []
    if not os.path.isdir(setdir):
        print("FAIL no such set directory: %s" % setdir)
        return 1
    name = os.path.basename(os.path.abspath(setdir))
    manifest = os.path.join(setdir, "manifest.json")
    try:
        with open(manifest, "rb") as f:
            raw = f.read()
    except OSError:
        print("FAIL %s: no manifest.json (the set was never signed: seal_manifest.py write)"
              % name)
        return 1
    try:
        data = json.loads(raw.decode("utf-8"))
        fmt = data["format"]
        listed = data["files"]
        declared = data.get("set")
        assert isinstance(listed, list)
    except Exception as e:
        print("FAIL %s: manifest.json does not read as a format %d manifest (%s)" % (name, FORMAT, e))
        return 1
    if fmt != FORMAT:
        problems.append("manifest.json says format %r, this build reads format %d" % (fmt, FORMAT))
    if declared != name:
        problems.append("manifest.json names the set %r but it is in %r (a set moved onto "
                        "another one's name)" % (declared, name))
    seen = set()
    for ent in listed:
        try:
            rel, size, sha = ent["path"], int(ent["size"]), ent["sha256"]
        except Exception:
            problems.append("manifest.json has an entry that is not a "
                            "{path, size, sha256} record: %r" % (ent,))
            continue
        if rel in SELF:
            problems.append("manifest.json lists itself (%s)" % rel)
            continue
        if rel in seen:
            problems.append("manifest.json lists %s twice" % rel)
            continue
        seen.add(rel)
        full = os.path.join(setdir, rel)
        if not os.path.isfile(full):
            problems.append("%s is listed in the manifest but is not in the set" % rel)
            continue
        if os.path.getsize(full) != size:
            problems.append("%s is %d bytes, the manifest says %d"
                            % (rel, os.path.getsize(full), size))
            continue
        got = _digest(full)
        if got != sha:
            problems.append("%s has sha256 %s, the manifest says %s (its bytes are not the ones "
                            "that were signed)" % (rel, got, sha))
    for rel in _members(setdir):
        if rel not in seen:
            problems.append("%s ships in the set but is not in the manifest "
                            "(nothing vouches for it)" % rel)
    sig_path = os.path.join(setdir, "manifest.sig")
    if pub is not None:
        if not os.path.isfile(sig_path):
            problems.append("no manifest.sig: the manifest is not signed")
        else:
            try:
                sig = bytes.fromhex(open(sig_path, encoding="ascii").read().strip())
            except (OSError, ValueError):
                sig = b""
            if not verify(pub, raw, sig):
                problems.append("manifest.sig does not verify with the release public key "
                                "(signed by another key, or the manifest changed after signing)")
    for p in problems:
        print("FAIL %s: %s" % (name, p))
    if not problems:
        print("ok   %s: %d file(s), each listed with the sha256 of its ciphertext%s"
              % (name, len(seen), ", manifest.sig verified" if pub is not None else ""))
    return 1 if problems else 0


# ── a set that ships no key at all: the self-test's own fixture ────────────────────────────────
def self_test():
    import tempfile
    tmp = tempfile.mkdtemp(prefix="seal-manifest-selftest.")
    seed = bytes.fromhex("9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60")
    want_pub = "d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a"
    want_sig = ("e5564300c360ac729086e2cc806e828a84877f1eb8e5d974d873e06522490155"
                "5fb8821590a33bacc61e39701cf9b46bd25bf5f0595bbe24655141438e7a100b")
    bad = 0

    def want(name, got, expect):
        nonlocal bad
        if got != expect:
            print("FAIL self-test: %s is %s, want %s" % (name, got, expect))
            bad = 1
        else:
            print("ok   self-test: %s" % name)

    want("RFC 8032 test 1 public key", public_key(seed).hex(), want_pub)
    want("RFC 8032 test 1 signature over the empty message", sign(seed, b"").hex(), want_sig)
    want("RFC 8032 test 1 verifies", verify(bytes.fromhex(want_pub), b"", bytes.fromhex(want_sig)),
         True)
    want("RFC 8032 test 1 rejects another message",
         verify(bytes.fromhex(want_pub), b"x", bytes.fromhex(want_sig)), False)
    want("a signature by a different seed does not verify",
         verify(bytes.fromhex(want_pub), b"", sign(b"\x01" * 32, b"")), False)
    # and the manifest round trip, so the shapes above are exercised and not just the curve
    setdir = os.path.join(tmp, "demo-set")
    os.makedirs(os.path.join(setdir, "layer1_1"))
    open(os.path.join(setdir, "layout.json"), "w").write('{"layers": 1}\n')
    open(os.path.join(setdir, "layer1_1", "layer1.elf"), "wb").write(MAGIC + b"\x00" * 64)
    _write_hex(os.path.join(tmp, "priv"), seed)
    _write_hex(os.path.join(tmp, "pub"), bytes.fromhex(want_pub))
    write(setdir, os.path.join(tmp, "priv"))
    want("the signed set checks", check(setdir, bytes.fromhex(want_pub)), 0)
    open(os.path.join(setdir, "layer1_1", "layer1.elf"), "r+b").write(MAGIC + b"\x01" * 64)
    # the failure is the point of the last case, so keep its FAIL lines out of this transcript
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        got = check(setdir, bytes.fromhex(want_pub))
    want("a flipped byte fails the check", got, 1)
    print("seal_manifest self-test %s" % ("PASS" if not bad else "FAIL"))
    return bad


def main(argv):
    usage = ("usage: seal_manifest.py write <set-dir> --key <priv> | "
             "check <set-dir> [--pub <pub>] | verify <set-dir> --pub <pub> | "
             "keygen <priv> <pub> | --self-test")
    args = list(argv[1:])
    if args[:1] == ["--self-test"]:
        return self_test()
    opts, rest = {}, []
    while args:
        a = args.pop(0)
        if a in ("--key", "--pub"):
            if not args:
                raise SystemExit("seal_manifest: %s needs a path\n%s" % (a, usage))
            opts[a[2:]] = args.pop(0)
        elif a in ("-h", "--help"):
            print(usage)
            return 0
        else:
            rest.append(a)
    if not rest:
        raise SystemExit(usage)
    cmd, rest = rest[0], rest[1:]
    if cmd == "keygen":
        if len(rest) != 2:
            raise SystemExit(usage)
        seed = os.urandom(32)
        _write_hex(rest[0], seed)
        _write_hex(rest[1], public_key(seed))
        print("ok   key pair written: %s (private) and %s (public), %s"
              % (rest[0], rest[1], public_key(seed).hex()))
        return 0
    if len(rest) != 1:
        raise SystemExit(usage)
    setdir = rest[0]
    if cmd == "write":
        if "key" not in opts:
            raise SystemExit("seal_manifest: write needs --key <private key>\n" + usage)
        return write(setdir, opts["key"])
    if cmd == "check":
        pub = _read_hex(opts["pub"], 32, "public") if "pub" in opts else None
        return check(setdir, pub)
    if cmd == "verify":
        if "pub" not in opts:
            raise SystemExit("seal_manifest: verify needs --pub <release public key>\n" + usage)
        return check(setdir, _read_hex(opts["pub"], 32, "public"))
    raise SystemExit(usage)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
