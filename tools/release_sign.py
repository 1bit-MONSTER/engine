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
"""Signed release checksums for the 1bit OS in-system updater (briefs OS-UPDATER item 3).

A release publishes its image and SHA256SUMS; checksums alone prove only that a download is the file the same server
listed, which is not enough for an OS that installs its own updates. So the release flow signs SHA256SUMS with the
release key and publishes SHA256SUMS.sig beside it, and the box verifies that signature with the public key baked
into its image (/etc/1bit/release-pub) BEFORE it trusts any digest in the file. The image is covered through its
sha256 line, so the signature is over a few hundred bytes, never over the 9 GB image (Ed25519 signs the whole message:
signing or verifying the image itself would read it into memory).

The signed message is a fixed domain prefix plus the exact bytes of SHA256SUMS:
    b"1bit-os release SHA256SUMS v1\\n" + <SHA256SUMS bytes>
The prefix keeps a release signature from ever being read as another document's (the NPU kernel manifests are signed
with the same kind of key and start "npumanifest 2"), so reusing a key across the two cannot turn one into the other.

Python 3 standard library only, like tools/seal_manifest.py, whose RFC 8032 Ed25519 this uses: the box verifies with
the python3 that 1bit OS ships and nothing else.

  release_sign.py sign <SHA256SUMS> --key <seed> --key-id <id> [--out <SHA256SUMS.sig>]
      Signs. <seed> is the 32-byte Ed25519 seed, raw (as kmanifest's npu.sec) or one line of hex (seal_manifest's
      keygen). It is read in place and never printed or written anywhere; only the public key and the signature are.
  release_sign.py verify <SHA256SUMS> <SHA256SUMS.sig> --pub <pub>
      Exit 0 when the signature is the release key's over these exact bytes; 1 otherwise (a refusal says why).
  release_sign.py verify-image <image> <SHA256SUMS> <SHA256SUMS.sig> --pub <pub>
      verify, then the image's sha256 (streamed, 1 MiB at a time) must equal the SHA256SUMS line that names its file.
  release_sign.py pubkey <seed>
      The seed's public key in hex (tests and the one-time setup; the release public key is not secret).
  release_sign.py --self-test

The .sig file is four lines of text: `1bit-release-sig 1`, `key <id>`, `pub <hex>`, `sig <hex>`. The `pub` line is
informational: verify uses the key it is GIVEN, and refuses a file whose `pub` names another key, so a signature by
some other key cannot pass by bringing its own public key along.
"""
import hashlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import seal_manifest as _ed  # noqa: E402  (RFC 8032 Ed25519, standard library only)

DOMAIN = b"1bit-os release SHA256SUMS v1\n"
SIG_MAGIC = "1bit-release-sig 1"


def _die(msg, code=1):
    print("release_sign: " + msg, file=sys.stderr)
    raise SystemExit(code)


def _read_key(path, what):
    """A 32-byte key file: raw bytes, or one line of hex. Its bytes are returned and never echoed."""
    try:
        with open(path, "rb") as f:
            data = f.read(200)
    except OSError as e:
        _die("cannot read the %s key %s: %s" % (what, path, e.strerror))
    if len(data) == 32:
        return data
    try:
        text = data.decode("ascii").strip()
        if len(text) == 64:
            return bytes.fromhex(text)
    except (UnicodeDecodeError, ValueError):
        pass
    _die("the %s key %s is neither 32 raw bytes nor 64 hex characters" % (what, path))


def _message(sums_path):
    try:
        with open(sums_path, "rb") as f:
            sums = f.read()
    except OSError as e:
        _die("cannot read %s: %s" % (sums_path, e.strerror))
    if not sums or len(sums) > 1 << 20:
        _die("%s is %s: a SHA256SUMS is a few lines" % (sums_path, "empty" if not sums else "over 1 MiB"))
    return DOMAIN + sums, sums


def sign(sums_path, seed, key_id):
    msg, _ = _message(sums_path)
    if not key_id or any(c.isspace() for c in key_id):
        _die("--key-id must be one word (it names the key in the .sig file)")
    pub = _ed.public_key(seed)
    sig = _ed.sign(seed, msg)
    if not _ed.verify(pub, msg, sig):
        _die("the signature does not verify against its own key (refusing to write it)")
    return "%s\nkey %s\npub %s\nsig %s\n" % (SIG_MAGIC, key_id, pub.hex(), sig.hex())


def _parse_sig(sig_path):
    try:
        with open(sig_path, "r", encoding="ascii") as f:
            lines = f.read(4096).splitlines()
    except (OSError, UnicodeDecodeError) as e:
        return None, "cannot read %s (%s)" % (sig_path, getattr(e, "strerror", None) or e)
    if not lines or lines[0].strip() != SIG_MAGIC:
        return None, "%s is not a 1bit release signature (no '%s' line)" % (sig_path, SIG_MAGIC)
    fields = {}
    for ln in lines[1:]:
        k, _, v = ln.strip().partition(" ")
        if k in ("key", "pub", "sig") and k not in fields:
            fields[k] = v.strip()
    try:
        sig = bytes.fromhex(fields.get("sig", ""))
        pub = bytes.fromhex(fields.get("pub", "")) if fields.get("pub") else None
    except ValueError:
        return None, "%s has a sig or pub line that is not hex" % sig_path
    if len(sig) != 64:
        return None, "%s carries no 64-byte signature" % sig_path
    return {"key": fields.get("key", "?"), "pub": pub, "sig": sig}, None


def verify(sums_path, sig_path, pub):
    """(True, key id) or (False, why)."""
    s, why = _parse_sig(sig_path)
    if s is None:
        return False, why
    if s["pub"] is not None and s["pub"] != pub:
        return False, "%s was signed by key %s (%s...), not by this release key (%s...)" % (
            sig_path, s["key"], s["pub"].hex()[:16], pub.hex()[:16])
    msg, _ = _message(sums_path)
    if not _ed.verify(pub, msg, s["sig"]):
        return False, "the signature in %s does not match %s: the file was changed, or signed by another key" % (
            sig_path, sums_path)
    return True, s["key"]


def _sums_entry(sums, name):
    for ln in sums.decode("utf-8", "replace").splitlines():
        parts = ln.strip().split(None, 1)
        if len(parts) == 2:
            digest, f = parts[0].lower(), parts[1].lstrip("*").strip()
            if os.path.basename(f) == name and len(digest) == 64:
                return digest
    return None


def verify_image(image, sums_path, sig_path, pub):
    ok, why = verify(sums_path, sig_path, pub)
    if not ok:
        return False, why
    _, sums = _message(sums_path)
    want = _sums_entry(sums, os.path.basename(image))
    if want is None:
        return False, "the signed %s has no line for %s" % (sums_path, os.path.basename(image))
    h = hashlib.sha256()
    try:
        with open(image, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    except OSError as e:
        return False, "cannot read %s: %s" % (image, e.strerror)
    if h.hexdigest() != want:
        return False, "%s has sha256 %s..., the signed SHA256SUMS says %s...: not the released image" % (
            image, h.hexdigest()[:16], want[:16])
    return True, "%s matches the signed SHA256SUMS (key %s)" % (os.path.basename(image), why)


def self_test():
    import tempfile
    bad = 0

    def want(name, cond):
        nonlocal bad
        print(("ok   " if cond else "FAIL ") + name)
        bad += 0 if cond else 1

    d = tempfile.mkdtemp(prefix="release-sign-selftest.")
    seed = bytes.fromhex("9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60")  # RFC 8032 test 1
    other = bytes(range(32))
    pub = _ed.public_key(seed)
    img = os.path.join(d, "1bit-os-test.img")
    with open(img, "wb") as f:
        f.write(os.urandom(3 << 20))
    digest = hashlib.sha256(open(img, "rb").read()).hexdigest()
    sums = os.path.join(d, "SHA256SUMS")
    with open(sums, "w") as f:
        f.write("%s  1bit-os-test.img\n" % digest)
    sig = os.path.join(d, "SHA256SUMS.sig")
    with open(sig, "w") as f:
        f.write(sign(sums, seed, "test-v1"))
    want("RFC 8032 key: the public key is the RFC's",
         pub.hex() == "d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a")
    want("a signed SHA256SUMS verifies", verify(sums, sig, pub)[0])
    want("the image verifies through its signed line", verify_image(img, sums, sig, pub)[0])
    want("the signature is over the DOMAIN-prefixed bytes, not the bare file",
         not _ed.verify(pub, open(sums, "rb").read(), bytes.fromhex(_parse_sig(sig)[0]["sig"].hex())))
    with open(sums, "a") as f:
        f.write("%s  extra.img\n" % ("0" * 64))
    want("a SHA256SUMS changed after signing is refused", not verify(sums, sig, pub)[0])
    with open(sums, "w") as f:
        f.write("%s  1bit-os-test.img\n" % digest)
    with open(img, "r+b") as f:
        f.seek(12345)
        f.write(b"\x00\x01\x02")
    want("a tampered image is refused (its sha256 is not the signed one)", not verify_image(img, sums, sig, pub)[0])
    want("another release key is refused", not verify(sums, sig, _ed.public_key(other))[0])
    forged = os.path.join(d, "forged.sig")
    with open(forged, "w") as f:
        f.write(sign(sums, other, "evil"))
    ok, why = verify(sums, forged, pub)
    want("a signature that brings its own public key is refused (%s)" % why[:40], not ok)
    with open(forged, "w") as f:
        f.write("not a signature\n")
    want("a file that is not a release signature is refused", not verify(sums, forged, pub)[0])
    want("an image with no line in the signed SHA256SUMS is refused",
         not verify_image(os.path.join(d, "SHA256SUMS.sig"), sums, sig, pub)[0])
    for p in os.listdir(d):
        os.unlink(os.path.join(d, p))
    os.rmdir(d)
    print("release_sign self-test: %s" % ("ok" if bad == 0 else "%d FAIL" % bad))
    return 0 if bad == 0 else 1


def main(argv):
    if argv[:1] == ["--self-test"]:
        return self_test()
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    cmd, rest = argv[0], argv[1:]

    def opt(name):
        if name in rest:
            i = rest.index(name)
            if i + 1 < len(rest):
                v = rest[i + 1]
                del rest[i:i + 2]
                return v
            _die("%s needs a value" % name, 2)
        return None

    if cmd == "sign":
        key, key_id, out = opt("--key"), opt("--key-id"), opt("--out")
        if len(rest) != 1 or not key or not key_id:
            _die("usage: sign <SHA256SUMS> --key <seed> --key-id <id> [--out <SHA256SUMS.sig>]", 2)
        text = sign(rest[0], _read_key(key, "signing"), key_id)
        out = out or rest[0] + ".sig"
        tmp = out + ".tmp"
        with open(tmp, "w", encoding="ascii") as f:
            f.write(text)
        os.replace(tmp, out)
        print("release_sign: signed %s -> %s (key %s, pub %s)" % (rest[0], out, key_id, text.split("pub ")[1][:16] + "..."))
        return 0
    if cmd in ("verify", "verify-image"):
        pub = opt("--pub")
        need = 2 if cmd == "verify" else 3
        if len(rest) != need or not pub:
            _die("usage: %s %s --pub <pub>" % (cmd, "<SHA256SUMS> <SHA256SUMS.sig>" if cmd == "verify"
                                               else "<image> <SHA256SUMS> <SHA256SUMS.sig>"), 2)
        p = _read_key(pub, "release public")
        ok, why = verify(*rest, p) if cmd == "verify" else verify_image(*rest, p)
        print(("release_sign: OK " + (why if cmd == "verify-image" else "signed by key " + why)) if ok
              else ("release_sign: REFUSED " + why), file=sys.stdout if ok else sys.stderr)
        return 0 if ok else 1
    if cmd == "pubkey":
        if len(rest) != 1:
            _die("usage: pubkey <seed>", 2)
        print(_ed.public_key(_read_key(rest[0], "signing")).hex())
        return 0
    _die("unknown command %r (sign, verify, verify-image, pubkey, --self-test)" % cmd, 2)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
