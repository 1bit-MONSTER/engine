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
"""The two NPU kernel-set schemes a release package can carry, and the checks each one needs.

The packager (`scripts/package-linux.sh`) and the weekly checks (`scripts/weekly.sh`, package_checks) were written
against one scheme, the **sealed** one (engine #574): `<set>/layout.json`, every kernel a `1BPKCRYP` ciphertext at
`<set>/layer1_*/layer1.elf`, `manifest.json` a format-1 JSON listing each ciphertext's sha256, and `manifest.sig`
Ed25519 over the manifest. `tools/seal_manifest.py` writes and checks that pair.

The kernel sets this release ships use a **second** scheme: `<set>/{<id>/layer1.elf, manifest, manifest.sig}` with a
signed manifest whose first line is `npumanifest 2` and whose body is one `<sha256> <size> <path>` line per kernel.
`manifest.json` carries the same bytes there, so the old check read binary where it expected JSON:

    scripts/package-linux.sh:  FAIL 1024x3072x16x8: manifest.json does not read as a format 1 manifest
                                     ('utf-8' codec can't decode byte 0x80 in position 1091)

and package_checks additionally demanded `1BPKCRYP` of every kernel, which the new sets never carry.

This tool owns the difference, so both callers ask one question instead of two divergent ones:

    npu_set_scheme.py detect <set>             sealed1 | kmanifest2 | unknown
    npu_set_scheme.py stage  <src> <dst>       copy exactly the files that scheme ships
    npu_set_scheme.py check  <dir> [--pub K]   verify the staged set (exit 0 ok, 1 bad, 2 cannot tell)

`check` verifies **digests**, not authenticity, for the npumanifest 2 scheme: the signature is Ed25519 over the
manifest and the engine verifies it at load time with the release public key compiled in. Digest verification is exact
and needs no crypto library, which is the half the packager can do on a machine that has only the tarball — and a check
that cannot verify a signature must not report "signed".

Python 3 standard library only, like tools/seal_manifest.py: the release machine is the OS image's busybox plus the
package, and a check that needs a module the image does not carry is a check nobody runs.
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

SEALED_MAGIC = b"1BPKCRYP"
KMANIFEST_MAGIC = "npumanifest 2"
SIG_BYTES = 64          # Ed25519
LAYOUT = "layout.json"
SIG = "manifest.sig"


def _read_head(path: Path, n: int = 256) -> bytes:
    try:
        with open(path, "rb") as f:
            return f.read(n)
    except OSError:
        return b""


def _is_hex(name: str) -> bool:
    return len(name) in (2, 4, 6, 8) and all(c in "0123456789abcdef" for c in name)


def detect(set_dir: Path) -> str:
    """`sealed1`, `kmanifest2`, or `unknown`.

    The MANIFEST decides, not layout.json: the signed-manifest scheme ships no layout.json (the engine reads the layout
    the manifest names), so requiring one here reported "unknown" for exactly the sets this release carries.
    """
    head = _read_head(set_dir / "manifest")
    if head.startswith(KMANIFEST_MAGIC.encode()):
        return "kmanifest2"
    if not head:
        head = _read_head(set_dir / "manifest.json")
        if head.startswith(KMANIFEST_MAGIC.encode()):
            return "kmanifest2"
    manifest_json = set_dir / "manifest.json"
    if manifest_json.is_file():
        try:
            doc = json.loads(manifest_json.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            doc = None
        if isinstance(doc, dict) and "files" in doc:
            return "sealed1"
    return "unknown"


def _kmanifest_files(set_dir: Path) -> tuple[list[tuple[str, int, str]], list[str]]:
    """(sha256, size, path) per file the kmanifest lists, plus the header lines it carried."""
    path = set_dir / "manifest"
    if not path.is_file():
        path = set_dir / "manifest.json"
    header, entries = [], []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        # A digest line is `<sha256> <size> <path>`, optionally introduced by the word `file` (the writer emits
        # "file <sha256> <size> <path>"); everything else is header ("npumanifest 2", "set …", "key …", "layout …").
        parts = line.split()
        if parts and parts[0] == "file":
            parts = parts[1:]
        if len(parts) == 3 and len(parts[0]) == 64 and parts[1].isdigit() and all(c in "0123456789abcdef" for c in parts[0]):
            entries.append((parts[0], int(parts[1]), parts[2]))
        else:
            header.append(line)
    return entries, header


def stage(src: Path, dst: Path) -> int:
    scheme = detect(src)
    if scheme == "unknown":
        print(f"npu_set_scheme: {src} is not a kernel set (no layout.json plus a manifest)", file=sys.stderr)
        return 2
    dst.mkdir(parents=True, exist_ok=True)
    files = 0

    def take(rel: str) -> None:
        nonlocal files
        s, d = src / rel, dst / rel
        if s.is_file():
            d.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(s, d)
            files += 1

    if (src / LAYOUT).is_file():
        take(LAYOUT)
    if scheme == "sealed1":
        for name in ("manifest.json", SIG):
            take(name)
        for layer in sorted(src.iterdir()):
            if layer.is_dir() and layer.name.startswith("layer1_") and (layer / "layer1.elf").is_file():
                take(f"{layer.name}/layer1.elf")
    else:
        for name in ("manifest", "manifest.json", SIG):
            take(name)
        for layer in sorted(src.iterdir()):
            if layer.is_dir() and _is_hex(layer.name) and (layer / "layer1.elf").is_file():
                take(f"{layer.name}/layer1.elf")
                # each kernel carries its own signed manifest beside it; the set-level pair covers the set
                for name in ("manifest", "manifest.sig"):
                    take(f"{layer.name}/{name}")
    print(f"npu_set_scheme: staged {files} file(s) from a {scheme} set into {dst}")
    return 0


def check(set_dir: Path, pub: str | None = None) -> int:
    scheme = detect(set_dir)
    if scheme == "unknown":
        print(f"npu_set_scheme: {set_dir}: cannot tell the scheme (no layout.json plus a manifest)", file=sys.stderr)
        return 2
    problems: list[str] = []
    if scheme == "sealed1":
        # The sealed scheme has an owner already: tools/seal_manifest.py writes and checks the pair, and its
        # format-1 manifest covers the ciphertext digests. Delegate rather than reimplement it.
        manifest = set_dir / "manifest.json"
        try:
            doc = json.loads(manifest.read_text(encoding="utf-8"))
            listed = doc.get("files", {})
        except (OSError, ValueError) as exc:
            print(f"npu_set_scheme: {set_dir}: manifest.json does not read as a format 1 manifest ({exc})", file=sys.stderr)
            return 1
        # engine #574's writer emits a LIST of {path, size, sha256}; an older draft used a {path: sha256} map. Both are
        # format 1, so both are read rather than one being declared wrong by the checker that must accept the other.
        entries = listed if isinstance(listed, list) else [dict(v, path=k) if isinstance(v, dict) else {"path": k, "sha256": v}
                                                          for k, v in listed.items()]
        for entry in entries:
            if not isinstance(entry, dict) or not entry.get("path"):
                problems.append("a manifest entry has no path")
                continue
            rel = entry["path"]
            want = entry.get("sha256")
            target = set_dir / rel
            if not target.is_file():
                problems.append(f"listed file is missing: {rel}")
                continue
            data = target.read_bytes()
            digest = hashlib.sha256(data).hexdigest()
            if want and digest != want:
                problems.append(f"digest mismatch: {rel}")
            if "size" in entry and len(data) != entry["size"]:
                problems.append(f"size mismatch: {rel} ({len(data)} != {entry['size']})")
            head = _read_head(target, len(SEALED_MAGIC))
            if head != SEALED_MAGIC:
                problems.append(f"not sealed (no {SEALED_MAGIC.decode()} magic): {rel}")
        if not (set_dir / SIG).is_file():
            problems.append(f"no {SIG}: the manifest is not signed")
        if pub and not problems:
            problems.append("the sealed scheme's signature is verified by tools/seal_manifest.py verify, not here")
    else:
        entries, header = _kmanifest_files(set_dir)
        if not entries:
            problems.append("the kmanifest lists no files (want '<sha256> <size> <path>' lines)")
        if not any(line.startswith("set ") for line in header):
            problems.append("the kmanifest has no 'set <shape>' header line")
        seen = set()
        for digest, size, rel in entries:
            seen.add(rel)
            target = set_dir / rel
            if not target.is_file():
                problems.append(f"listed file is missing: {rel}")
                continue
            data = target.read_bytes()
            if len(data) != size:
                problems.append(f"size mismatch: {rel} ({len(data)} != {size})")
            if hashlib.sha256(data).hexdigest() != digest:
                problems.append(f"digest mismatch: {rel}")
            if not rel.endswith("/layer1.elf"):
                problems.append(f"unexpected file in the manifest (kernels are <hex>/layer1.elf): {rel}")
            elif _read_head(target, 4) != b"\x7fELF":
                problems.append(f"not an ELF: {rel}")
        sig = set_dir / SIG
        if not sig.is_file():
            problems.append(f"no {SIG}: the manifest is not signed")
        elif sig.stat().st_size != SIG_BYTES:
            problems.append(f"{SIG} is {sig.stat().st_size} B, not an Ed25519 signature ({SIG_BYTES} B)")
        # Per-kernel manifests: each kernel directory that carries one must carry its signature too, and the kernel it
        # names must be the file beside it (the set manifest covers the set; these cover a single kernel).
        for kern in sorted(p for p in set_dir.iterdir() if p.is_dir() and _is_hex(p.name)):
            kman = kern / "manifest"
            if not kman.is_file():
                continue
            ksig = kern / SIG
            if not ksig.is_file():
                problems.append(f"kernel {kern.name}: has a manifest but no {SIG}")
            elif ksig.stat().st_size != SIG_BYTES:
                problems.append(f"kernel {kern.name}: {SIG} is {ksig.stat().st_size} B, not an Ed25519 signature")
            kentries, _ = _kmanifest_files(kern)
            elf = kern / "layer1.elf"
            if not kentries:
                problems.append(f"kernel {kern.name}: its manifest lists no files")
            elif elf.is_file():
                digest = hashlib.sha256(elf.read_bytes()).hexdigest()
                if not any(d == digest and rel.endswith("layer1.elf") for d, _sz, rel in kentries):
                    problems.append(f"kernel {kern.name}: its manifest does not list this layer1.elf's digest")
        # Exact allowlist: what the scheme ships is what goes in the package.
        try:
            for entry in sorted(set_dir.rglob("*")):
                if entry.is_dir():
                    continue
                rel = entry.relative_to(set_dir).as_posix()
                if rel in (LAYOUT, "manifest", "manifest.json", SIG) or rel in seen:
                    continue
                parts = rel.split("/")
                if len(parts) == 2 and _is_hex(parts[0]) and parts[1] in ("manifest", SIG):
                    continue  # the kernel's own signed manifest, shipped beside it
                if len(parts) == 2 and _is_hex(parts[0]) and parts[1] in ("manifest", SIG):
                    continue
                problems.append(f"unexpected file in the set (it ships layout.json + manifest + manifest.sig + "
                                f"<hex>/layer1.elf only): {rel}")
        except OSError as exc:
            problems.append(f"cannot list {set_dir} ({exc})")
    if problems:
        for p in problems:
            print(f"npu_set_scheme: {set_dir.name}: {p}", file=sys.stderr)
        return 1
    n = len(_kmanifest_files(set_dir)[0]) if scheme == "kmanifest2" else "?"
    extra = "" if scheme == "kmanifest2" else ""
    print(f"npu_set_scheme: {set_dir.name}: {scheme} set, {n} kernel(s), digests match"
          f"{', signed manifest present (the engine verifies the signature at load)' if scheme != 'sealed1' else ''}")
    return 0


def self_test() -> int:
    """Hermetic arms for the two schemes and the three failure shapes (engine#3237).

    Every arm builds its kernel set from bytes in a temp dir: no device, no release key, no network. The arms exist
    because this tool decides which NPU kernel set an image ships and had none -- measured before this commit:
    `--self-test` was not a verb, so nothing that runs without a staged set and a public key verified it, while every
    other release-path tool carried arms.
    """
    import pathlib
    import tempfile
    ok = fail = 0

    def arm(name, want, got):
        nonlocal ok, fail
        if want == got:
            ok += 1
            print(f"  ok   {name}")
        else:
            fail += 1
            print(f"  FAIL {name}: got {got!r}, want {want!r}")

    def kset(root: pathlib.Path, *, magic=KMANIFEST_MAGIC, entries=None, sig_bytes=SIG_BYTES,
             set_header="set 1024x3072x16x8", elf=b"\x7fELFkernel", extra_files=None, write_elf=True):
        """A kernel set built from bytes: one hex kernel dir, a kmanifest naming it, and its signature."""
        root.mkdir(parents=True, exist_ok=True)
        kern = root / "65"
        kern.mkdir(exist_ok=True)
        if write_elf:
            (kern / "layer1.elf").write_bytes(elf)
        digest = hashlib.sha256(elf).hexdigest()
        lines = [magic]
        if set_header:
            lines.append(set_header)
        for sha, size, rel in (entries if entries is not None else [(digest, len(elf), "65/layer1.elf")]):
            lines.append(f"{sha} {size} {rel}")
        (root / "manifest").write_text("\n".join(lines) + "\n", encoding="utf-8")
        (root / SIG).write_bytes(b"\x00" * sig_bytes)
        for name, data in (extra_files or {}).items():
            (root / name).write_bytes(data)
        return root

    with tempfile.TemporaryDirectory() as td:
        base = pathlib.Path(td)

        # the positive control: a set that is right in every particular must be rc 0, or every arm below is vacuous
        good = kset(base / "good")
        arm("a well-formed kmanifest2 set checks clean", 0, check(good))
        arm("…and its scheme is detected from the manifest HEAD, not layout.json", "kmanifest2", detect(good))

        # engine#3237's actual failure family: the manifest names bytes the set does not carry, or carries different ones.
        # Each fixture is built so that ONE thing is wrong: a set that also tripped the exact-allowlist would pass this
        # arm for the wrong reason, which is the defect this self-test exists to prevent in other tools.
        import contextlib
        import io

        def problems_of(d):
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                rc = check(d)
            return rc, err.getvalue()

        # the manifest lists a kernel and the set carries no kernel at all: the allowlist cannot fire, so "missing" is
        # the only problem, and the arm can actually distinguish it from a digest or size defect
        missing = base / "missing"
        kset(missing, entries=[("0" * 64, 8, "65/gone.elf")], write_elf=False)
        rc, err = problems_of(missing)
        arm("a manifest naming a file the set does not contain is a finding", 1, rc)
        arm("…and that is the ONLY problem (the arm cannot pass for another reason)", True,
            "listed file is missing" in err and "unexpected file" not in err)

        rc, err = problems_of(kset(base / "wrong", entries=[("0" * 64, len(b"\x7fELFkernel"), "65/layer1.elf")]))
        arm("a digest that does not match the bytes is a finding", 1, rc)
        arm("…and the message names the digest, not the file", True,
            "digest mismatch" in err and "missing" not in err)

        sized = kset(base / "sized", entries=[(hashlib.sha256(b"\x7fELFkernel").hexdigest(), 11, "65/layer1.elf")])
        arm("a size that does not match the bytes is a finding", 1, check(sized))

        unsigned = kset(base / "unsigned")
        (unsigned / SIG).unlink()
        arm("a set with no signature is a finding", 1, check(unsigned))

        short_sig = kset(base / "shortsig", sig_bytes=32)
        arm("a signature that is not 64 B is a finding", 1, check(short_sig))

        # cannot-tell: nothing to judge must not read as either pass or finding
        empty = base / "empty"
        empty.mkdir()
        arm("a directory with no manifest is cannot tell", 2, check(empty))

        # the sealed scheme is detected by its format-1 manifest.json and delegated, not reimplemented
        sealed = base / "sealed"
        sealed.mkdir()
        (sealed / "manifest.json").write_text(json.dumps({"files": {}}), encoding="utf-8")
        arm("a format-1 manifest.json is the sealed scheme", "sealed1", detect(sealed))

        # engine#3237 review: the sealed path carries its OWN digest comparison (`if want and digest != want`), and the
        # detection arm above pins the scheme, not a tampered file -- measured: neutering that comparison left 13 ok /
        # 0 failed. A fixture that satisfies every other rule and differs only in the bytes makes it discriminating.
        def format1_set(root: pathlib.Path, *, tampered: bool):
            root.mkdir(parents=True, exist_ok=True)
            good = SEALED_MAGIC + b"ciphertext"
            (root / "65").mkdir(exist_ok=True)
            (root / "65" / "layer1.elf").write_bytes(good if not tampered else SEALED_MAGIC + b"ciphertexT")
            (root / "manifest.json").write_text(json.dumps({"files": [
                {"path": "65/layer1.elf", "size": len(good), "sha256": hashlib.sha256(good).hexdigest()}]}),
                encoding="utf-8")
            (root / SIG).write_bytes(b"\x00" * SIG_BYTES)
            return root

        clean1 = format1_set(base / "sealed-clean", tampered=False)
        arm("a well-formed format-1 set checks clean", 0, check(clean1))
        rc, err = problems_of(format1_set(base / "sealed-tampered", tampered=True))
        arm("a format-1 set whose bytes changed after signing is a finding", 1, rc)
        arm("…and it is the DIGEST that names it, not a missing file or signature", True,
            "digest mismatch" in err and "missing" not in err and SIG not in err)

        # engine#3237 review, second pass: the PER-KERNEL manifests (each <hex>/ carries its own) are a third family of
        # checks, and they had no arm either -- a set whose kernel manifest disagrees with the kernel beside it is the
        # shape a re-sealed kernel produces when only the set-level pair is regenerated.
        def per_kernel_set(root: pathlib.Path, *, kdigest=None, ksig=True):
            root = kset(root)
            kern = root / "65"
            (kern / "manifest").write_text(
                f"{KMANIFEST_MAGIC}\nset 1024x3072x16x8\n"
                f"{kdigest or hashlib.sha256(b'\x7fELFkernel').hexdigest()} 10 layer1.elf\n", encoding="utf-8")
            if ksig:
                (kern / SIG).write_bytes(b"\x00" * SIG_BYTES)
            return root

        arm("a kernel whose own manifest agrees with it checks clean", 0, check(per_kernel_set(base / "kern-ok")))
        rc, err = problems_of(per_kernel_set(base / "kern-baddigest", kdigest="0" * 64))
        arm("a kernel manifest that disagrees with the kernel beside it is a finding", 1, rc)
        arm("…and it names the kernel's manifest, not the set's", True, "does not list this layer1.elf" in err)
        rc, err = problems_of(per_kernel_set(base / "kern-nosig", ksig=False))
        arm("a kernel manifest with no signature beside it is a finding", 1, rc)
        arm("…and it names the missing signature for that kernel", True, "has a manifest but no" in err)

        # the writer emits `file <sha> <size> <path>`; both spellings must parse or the check reads an empty manifest
        prefixed = kset(base / "prefixed")
        (prefixed / "manifest").write_text(
            f"{KMANIFEST_MAGIC}\nset 1024x3072x16x8\nfile {hashlib.sha256(b'\x7fELFkernel').hexdigest()} "
            f"{len(b'\x7fELFkernel')} 65/layer1.elf\n", encoding="utf-8")
        arm("the writer's `file <sha> <size> <path>` form parses", 1, len(_kmanifest_files(prefixed)[0]))
        arm("…and that set checks clean", 0, check(prefixed))

    print(f"npu_set_scheme self-test: {ok} ok, {fail} failed")
    return 0 if fail == 0 else 1



def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    # `--self-test` is a verb of its own: the subparsers are `required=True`, and the arms must run on a box with no
    # kernel set at all (ci.yml's own row), so it is read before argparse sees the subcommand.
    if argv[:1] == ["--self-test"]:
        return self_test()
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("detect")
    d.add_argument("set")
    s = sub.add_parser("stage")
    s.add_argument("src")
    s.add_argument("dst")
    c = sub.add_parser("check")
    c.add_argument("dir")
    c.add_argument("--pub", default=None, help="release public key (sealed scheme: verify with tools/seal_manifest.py)")
    a = ap.parse_args(argv)
    if a.cmd == "detect":
        print(detect(Path(a.set)))
        return 0
    if a.cmd == "stage":
        return stage(Path(a.src), Path(a.dst))
    return check(Path(a.dir), a.pub)


if __name__ == "__main__":
    sys.exit(main())
