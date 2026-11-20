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
"""hrx_embedded_objects_check.py — the HIP code objects a built libggml-hrx.so embeds are the ones its sources name.

engine#2935: an HRX build can succeed and the server then refuse every model at load ("N HIP kernel registration(s)
do not match their compiled code objects", "backend 0 did not become ready"). The refusal is correct; it is only
late. Everything it compares is in the build tree when the build ends, and none of it needs a device, so this runs
the comparison there and fails the BUILD.

Four generated artefacts sit between the kernel sources and the server, and each can be older than the one before:

    <add-on>/kernels/hrx_kernel_ids.h            which kernel ids each code object must define (private add-on only)
    hip-code-objects/<stem>.<target>.hsaco       one compiled code object per kernel file
    ggml-hrx-hip-code-objects.inc                the table of those objects that the library compiles in
    bin/libggml-hrx.so                           what the server loads

Arms, in the order the artefacts are made:

  objects   every row of the .inc names a .hsaco that exists and has the row's digest (a stale .inc)
  library   the bytes of every such .hsaco are in libggml-hrx.so (a library linked from another .inc, or copied in)
  ids       add-on builds only: every (id, code object) row of the add-on's kernels/hrx_kernel_ids.h is a kernel
            symbol of that code object, and no such object defines a kernel id the header does not have (a .hsaco
            compiled from another add-on revision). A code object the build does not embed is skipped, as the
            library's own load-time check skips it.
  register  with --registration: load the library and let it build its dispatch registries, which runs the add-on's
            own load-time check (every REGISTERED kernel against its code object's arguments) with no device. This
            is the arm that sees a dispatcher naming a kernel nothing compiled; a mismatch aborts the child process.

Only ids, code-object stems and counts are printed. The ids are one-way and are in the shipped binaries already;
kernel names are not printed unless --names is given (the register arm relays the library's own messages, which
spell them), so the output can go in a public log.

  tools/ci/hrx_embedded_objects_check.py --tree BUILD [--addon DIR] [--registration] [--names]
  tools/ci/hrx_embedded_objects_check.py --self-test

BUILD is the HRX llama.cpp build tree (<engine build>/hrx/llama) or the engine build directory above it. The add-on
directory is read from the tree's CMakeCache.txt (GGML_HRX_HIP_ADDON_DIR) unless --addon gives it.

Exit: 0 every arm that could run matched; 1 a mismatch (named); 2 cannot tell (an artefact is missing or unreadable:
"nothing was checked" is not "all clean"); 77 BUILD holds no HRX tree at all (the ctest skip convention).
"""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

SKIPPED = 77
SUBDIR = Path("ggml/src/ggml-hrx")
INC_NAME = "ggml-hrx-hip-code-objects.inc"
INC_ROW_RE = re.compile(r'\{ "([^"]+)", "([^"]+)", kHipCodeObject\d+, sizeof\(kHipCodeObject\d+\), "([0-9a-f]{16})" \}')
ID_ROW_RE = re.compile(r'\{\s*0x[0-9a-fA-F]+ULL,\s*"(k_[0-9a-f]{12})",\s*"([^"]+)"\s*\}')
ID_RE = re.compile(r"^k_[0-9a-f]{12}$")
# ggml::hrx::hip_eager_op_declared(ggml_op): builds the dispatch registries on first use (hip/hip-capabilities.h)
REGISTRY_SYMBOL = "_ZN4ggml3hrx21hip_eager_op_declaredE7ggml_op"
ABI_OK_RE = re.compile(r"HIP add-on ABI check: (\d+) kernels match their code objects; (\d+) kernel ids resolve")
NAME_RE = re.compile(r"hip kernel \S+ \(")


def kernel_ids_of(data: bytes) -> set[str] | None:
    """The kernel-descriptor symbols (`<id>.kd`) of a raw ELF64 code object, or None when it is not one."""
    if len(data) < 64 or data[:4] != b"\x7fELF" or data[4] != 2 or data[5] != 1:
        return None
    shoff, = struct.unpack_from("<Q", data, 0x28)
    shentsize, shnum = struct.unpack_from("<HH", data, 0x3A)
    if shentsize < 64 or shoff + shentsize * shnum > len(data):
        return None
    sections = []
    for i in range(shnum):
        _name, sh_type, _flags, _addr, off, size, link, _info, _align, entsize = struct.unpack_from(
            "<IIQQQQIIQQ", data, shoff + i * shentsize)
        sections.append((sh_type, off, size, link, entsize))
    ids: set[str] = set()
    for sh_type, off, size, link, entsize in sections:
        if sh_type != 2 or entsize < 24 or link >= len(sections) or off + size > len(data):  # SHT_SYMTAB
            continue
        _t, str_off, str_size, _l, _e = sections[link]
        strtab = data[str_off:str_off + str_size]
        for at in range(off, off + size, entsize):
            name_at, = struct.unpack_from("<I", data, at)
            end = strtab.find(b"\0", name_at)
            name = strtab[name_at:end if end >= 0 else len(strtab)].decode("ascii", "replace")
            if name.endswith(".kd") and ID_RE.match(name[:-3]):
                ids.add(name[:-3])
    return ids


def resolve_tree(build: Path) -> Path | None:
    for candidate in (build, build / "hrx" / "llama"):
        if (candidate / SUBDIR).is_dir():
            return candidate
    return None


def cached_addon(tree: Path) -> Path | None:
    try:
        text = (tree / "CMakeCache.txt").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    m = re.search(r"^GGML_HRX_HIP_ADDON_DIR:[A-Z]+=(.*)$", text, re.M)
    return Path(m.group(1)) if m and m.group(1).strip() else None


def find_library(tree: Path) -> Path | None:
    for rel in ("bin/libggml-hrx.so", "ggml/src/ggml-hrx/libggml-hrx.so", "ggml/src/libggml-hrx.so"):
        if (tree / rel).exists():
            return (tree / rel).resolve()
    return None


def check_tree(tree: Path, addon: Path | None, out=print) -> int:
    """The objects, library and ids arms. Returns 0, 1 or 2."""
    problems: list[str] = []
    inc_path = tree / SUBDIR / INC_NAME
    try:
        rows = INC_ROW_RE.findall(inc_path.read_text(encoding="ascii", errors="replace"))
    except OSError as error:
        out(f"hrx_embedded_objects_check: cannot read {inc_path}: {error.strerror}")
        return 2
    if not rows:
        out(f"hrx_embedded_objects_check: {inc_path} has no code-object rows: nothing to check is not a clean tree")
        return 2
    library = find_library(tree)
    if library is None:
        out(f"hrx_embedded_objects_check: no libggml-hrx.so under {tree}: the library arm cannot run")
        return 2
    blob = library.read_bytes()

    objects: dict[str, list[tuple[str, bytes]]] = {}  # stem -> [(target, data)]
    for stem, target, digest in rows:
        path = tree / SUBDIR / "hip-code-objects" / f"{stem}.{target}.hsaco"
        try:
            data = path.read_bytes()
        except OSError:
            problems.append(f"objects: {stem}.{target}: the .inc embeds it and the .hsaco is gone")
            continue
        have = hashlib.sha256(data).hexdigest()[:16]
        if have != digest:
            problems.append(f"objects: {stem}.{target}: the .inc embeds {digest}, the .hsaco on disk is {have} "
                            f"({INC_NAME} is older than the code object)")
        if blob.find(data) < 0:
            problems.append(f"library: {stem}.{target}: {library.name} does not contain the .hsaco on disk ({have})")
        objects.setdefault(stem, []).append((target, data))

    id_rows = 0
    id_note = "no add-on configured, ids arm not applicable"
    if addon is not None:
        header = addon / "kernels" / "hrx_kernel_ids.h"
        try:
            pairs = ID_ROW_RE.findall(header.read_text(encoding="utf-8", errors="replace"))
        except OSError as error:
            out(f"hrx_embedded_objects_check: the tree was configured with the add-on {addon} and its "
                f"kernels/hrx_kernel_ids.h cannot be read ({error.strerror}): the ids arm cannot run")
            return 2
        if not pairs:
            out(f"hrx_embedded_objects_check: {header} has no (id, code object) rows: the ids arm cannot run")
            return 2
        wanted: dict[str, set[str]] = {}
        for kernel_id, stem in pairs:
            wanted.setdefault(stem, set()).add(kernel_id)
        embedded = [stem for stem in wanted if stem in objects]
        if not embedded:
            out(f"hrx_embedded_objects_check: none of the {len(wanted)} code objects {header.name} names is embedded: "
                "an add-on build that embeds no add-on kernel is not a clean tree")
            return 2
        for stem in sorted(embedded):
            for target, data in objects[stem]:
                have_ids = kernel_ids_of(data)
                if have_ids is None:
                    problems.append(f"ids: {stem}.{target}: not a raw ELF64 code object")
                    continue
                id_rows += len(wanted[stem])
                missing = sorted(wanted[stem] - have_ids)
                extra = sorted(have_ids - wanted[stem])
                if missing:
                    problems.append(f"ids: {stem}.{target}: {len(missing)} of {len(wanted[stem])} ids the add-on "
                                    f"names are not kernels of the code object ({' '.join(missing[:8])}"
                                    f"{' ...' if len(missing) > 8 else ''})")
                if extra:
                    problems.append(f"ids: {stem}.{target}: {len(extra)} kernels of the code object have no id in "
                                    f"the add-on ({' '.join(extra[:8])}{' ...' if len(extra) > 8 else ''})")
        id_note = f"{id_rows} add-on ids in {len(embedded)} code objects"

    if problems:
        for line in problems:
            out(f"hrx_embedded_objects_check: {line}")
        out(f"hrx_embedded_objects_check: FAIL {tree}: {len(problems)} mismatch(es). The embedded HIP code objects are "
            "not the ones this tree's sources name; a server built from it refuses models at load (engine#2935). "
            f"Delete {SUBDIR}/hip-code-objects/*.hsaco and {SUBDIR}/{INC_NAME} in the tree and rebuild.")
        return 1
    out(f"ok   hrx_embedded_objects_check: {tree}: {len(rows)} code objects match the .inc and {library.name}; {id_note}")
    return 0


def check_registration(tree: Path, expect_addon: bool, names: bool, out=print) -> int:
    """The register arm: the library's own load-time check, in a child process, without a device."""
    library = find_library(tree)
    if library is None:
        out(f"hrx_embedded_objects_check: no libggml-hrx.so under {tree}: the register arm cannot run")
        return 2
    child = ("import ctypes, sys\n"
             "lib = ctypes.CDLL(sys.argv[1], mode=ctypes.RTLD_GLOBAL)\n"
             f"fn = getattr(lib, {REGISTRY_SYMBOL!r}, None)\n"
             "if fn is None: sys.exit(3)\n"
             "fn.argtypes = [ctypes.c_int]; fn.restype = ctypes.c_bool\n"
             "fn(0)\n")
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = os.pathsep.join(filter(None, [str(library.parent), env.get("LD_LIBRARY_PATH", "")]))
    try:
        p = subprocess.run([sys.executable, "-c", child, str(library)], capture_output=True, text=True,
                           env=env, timeout=300)
    except (OSError, subprocess.TimeoutExpired) as error:
        out(f"hrx_embedded_objects_check: the register arm could not run: {error}")
        return 2
    return judge_registration(p.returncode, p.stdout + p.stderr, expect_addon, names, out)


def judge_registration(rc: int, log: str, expect_addon: bool, names: bool, out=print) -> int:
    """Pure half of the register arm (self-tested): the child's exit and log -> 0, 1 or 2."""
    mismatches = [line for line in log.splitlines()
                  if "HIP add-on ABI mismatch" in line or "HIP add-on kernel id" in line]
    if rc != 0 and (mismatches or "do not match their compiled code objects" in log
                    or "do not resolve to a symbol" in log):
        for line in mismatches[:40]:
            line = line[line.find("ggml_hrx:"):] if "ggml_hrx:" in line else line
            out("hrx_embedded_objects_check: register: " + (line if names else NAME_RE.sub("hip kernel <name> (", line)))
        out(f"hrx_embedded_objects_check: FAIL register: {len(mismatches)} registration(s) do not match the embedded "
            "code objects; this library aborts when a server loads a model (engine#2935)"
            + ("" if names else "; --names prints the kernel names"))
        return 1
    if rc == 3:
        out("hrx_embedded_objects_check: the library does not export the registry entry point: the register arm cannot run")
        return 2
    if rc != 0:
        tail = " | ".join(log.strip().splitlines()[-3:])
        out(f"hrx_embedded_objects_check: the register arm's child ended with {rc} before a verdict: {tail}")
        return 2
    m = ABI_OK_RE.search(log)
    if m:
        out(f"ok   hrx_embedded_objects_check: register: {m.group(1)} registered kernels match their code objects, "
            f"{m.group(2)} ids resolve (the library's own check, no device)")
        return 0
    if expect_addon:
        out("hrx_embedded_objects_check: the registries were built and the add-on's check did not report: "
            "a check that said nothing is not a pass")
        return 2
    out("ok   hrx_embedded_objects_check: register: registries built; no add-on configured, so no add-on check to run")
    return 0


def run(build: Path, addon: Path | None, registration: bool, names: bool, out=print) -> int:
    tree = resolve_tree(build)
    if tree is None:
        out(f"SKIPPED: hrx_embedded_objects_check found no HRX llama.cpp tree under {build} "
            f"(no {SUBDIR}); nothing was checked")
        return SKIPPED
    if addon is None:
        addon = cached_addon(tree)
    rc = check_tree(tree, addon, out)
    if registration and rc == 0:
        rc = check_registration(tree, addon is not None, names, out)
    return rc


# ---- self-test -------------------------------------------------------------------------------------------------------

def _fake_object(ids: list[str], pad: bytes = b"") -> bytes:
    """A minimal ELF64 with one .symtab naming `<id>.kd` for each id (what kernel_ids_of reads, nothing more)."""
    strtab = b"\0" + b"".join(i.encode() + b".kd\0" for i in ids) + pad
    symtab = b"\0" * 24
    at = 1
    for i in ids:
        symtab += struct.pack("<IBBHQQ", at, 0x11, 0, 1, 0, 64)
        at += len(i) + 4
    body = strtab + symtab
    shoff = 64 + len(body)
    header = b"\x7fELF\x02\x01\x01" + b"\0" * 9 + struct.pack("<HHIQQQIHHHHHH", 3, 224, 1, 0, 0, shoff, 0, 64, 0, 0, 64, 3, 0)
    null = b"\0" * 64
    sh_str = struct.pack("<IIQQQQIIQQ", 0, 3, 0, 0, 64, len(strtab), 0, 0, 1, 0)
    sh_sym = struct.pack("<IIQQQQIIQQ", 0, 2, 0, 0, 64 + len(strtab), len(symtab), 1, 1, 8, 24)
    return header + body + null + sh_str + sh_sym


def _fake_tree(root: Path, objects: dict[str, bytes], *, inc_from: dict[str, bytes] | None = None,
               lib_from: dict[str, bytes] | None = None, header: dict[str, list[str]] | None = None) -> Path:
    tree = root / "hrx" / "llama"
    obj_dir = tree / SUBDIR / "hip-code-objects"
    obj_dir.mkdir(parents=True)
    (tree / "bin").mkdir()
    for stem, data in objects.items():
        (obj_dir / f"{stem}.gfx0000.hsaco").write_bytes(data)
    inc = "".join(f'    {{ "{stem}", "gfx0000", kHipCodeObject{n}, sizeof(kHipCodeObject{n}), '
                  f'"{hashlib.sha256(data).hexdigest()[:16]}" }},\n'
                  for n, (stem, data) in enumerate((inc_from if inc_from is not None else objects).items()))
    (tree / SUBDIR / INC_NAME).write_text(inc)
    (tree / "bin" / "libggml-hrx.so").write_bytes(
        b"LIB" + b"".join((lib_from if lib_from is not None else objects).values()) + b"END")
    cache = ""
    if header is not None:
        addon = root / "addon"
        (addon / "kernels").mkdir(parents=True)
        (addon / "kernels" / "hrx_kernel_ids.h").write_text(
            "".join(f'    {{ 0x{n:016x}ULL, "{i}", "{stem}" }},\n'
                    for n, (stem, i) in enumerate((s, i) for s, ids in header.items() for i in ids)))
        cache = f"GGML_HRX_HIP_ADDON_DIR:PATH={addon}\n"
    (tree / "CMakeCache.txt").write_text(cache)
    return root


def self_test() -> int:
    a, b, c = "k_aaaaaaaaaaaa", "k_bbbbbbbbbbbb", "k_cccccccccccc"
    one, two, old = _fake_object([a, b]), _fake_object([c]), _fake_object([a])
    plain = _fake_object([], pad=b"not built through the id scheme\0")
    cases = [
        # (name, kwargs for _fake_tree, expected rc, text the verdict must carry)
        ("clean add-on tree", dict(objects={"x": one, "y": two, "core": plain}, header={"x": [a, b], "y": [c]}), 0, "3 add-on ids"),
        ("clean tree with no add-on", dict(objects={"core": plain}), 0, "not applicable"),
        ("the .inc is older than a rebuilt code object", dict(objects={"x": one}, inc_from={"x": old}, header={"x": [a, b]}), 1, "objects: x"),
        ("the library was linked from other objects", dict(objects={"x": one}, lib_from={"x": old}, header={"x": [a, b]}), 1, "library: x"),
        ("a code object compiled from an older add-on revision", dict(objects={"x": old}, header={"x": [a, b]}), 1, "1 of 2 ids"),
        ("a code object with a kernel the add-on has no id for", dict(objects={"x": one}, header={"x": [a]}), 1, "have no id"),
        ("an id whose code object is not embedded is skipped", dict(objects={"x": one}, header={"x": [a, b], "z": [c]}), 0, "2 add-on ids"),
        ("an add-on build that embeds no add-on object", dict(objects={"core": plain}, header={"x": [a]}), 2, "not a clean tree"),
        ("the .inc lists nothing", dict(objects={}), 2, "nothing to check"),
    ]
    failures = 0
    with tempfile.TemporaryDirectory() as tmp:
        for n, (name, kwargs, expect, needle) in enumerate(cases):
            lines: list[str] = []
            rc = run(_fake_tree(Path(tmp) / str(n), **kwargs), None, False, False, lines.append)
            ok = rc == expect and needle in "\n".join(lines)
            failures += not ok
            print(f"{'ok  ' if ok else 'FAIL'} {name}: rc {rc} (expect {expect})" + ("" if ok else " :: " + " | ".join(lines)))
        # no HRX tree at all is a skip, and the add-on's header missing is "cannot tell", not a pass
        lines = []
        rc = run(Path(tmp) / "nothing-here", None, False, False, lines.append)
        ok = rc == SKIPPED
        failures += not ok
        print(f"{'ok  ' if ok else 'FAIL'} no HRX tree: rc {rc} (expect {SKIPPED})")
        lines = []
        rc = run(_fake_tree(Path(tmp) / "gone", objects={"x": one}), Path(tmp) / "no-such-addon", False, False, lines.append)
        ok = rc == 2
        failures += not ok
        print(f"{'ok  ' if ok else 'FAIL'} the configured add-on is gone: rc {rc} (expect 2)")
    abort = ("E ggml_hrx: HIP add-on ABI mismatch: hip kernel some_kernel (x, gfx0000): the code object has no kernel "
             "of that name\nggml_hrx: 1 HIP kernel registration(s) do not match their compiled code objects (see above)")
    verdicts = [
        ("register: the library's check passed", (0, "I ggml_hrx: HIP add-on ABI check: 5 kernels match their code objects; 7 kernel ids resolve (9 us)", True, False), 0, "5 registered"),
        ("register: a mismatch aborts, names withheld", (-6, abort, True, False), 1, "hip kernel <name> ("),
        ("register: --names relays them", (-6, abort, True, True), 1, "some_kernel"),
        ("register: an add-on build whose check said nothing", (0, "", True, False), 2, "not a pass"),
        ("register: no add-on, nothing to report", (0, "", False, False), 0, "no add-on configured"),
        ("register: the child died of something else", (-11, "", True, False), 2, "before a verdict"),
        ("register: no entry point in the library", (3, "", True, False), 2, "cannot run"),
    ]
    for name, args, expect, needle in verdicts:
        lines = []
        rc = judge_registration(*args, lines.append)
        ok = rc == expect and needle in "\n".join(lines) and (args[3] or "some_kernel" not in "\n".join(lines))
        failures += not ok
        print(f"{'ok  ' if ok else 'FAIL'} {name}: rc {rc} (expect {expect})" + ("" if ok else " :: " + " | ".join(lines)))
    total = len(cases) + 2 + len(verdicts)
    print(f"hrx_embedded_objects_check self-test: {total} case(s), {failures} failure(s)")
    return 1 if failures else 0


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tree", metavar="BUILD", help="the HRX llama.cpp build tree, or the engine build directory")
    ap.add_argument("--addon", metavar="DIR", help="the add-on directory (default: GGML_HRX_HIP_ADDON_DIR of the tree)")
    ap.add_argument("--registration", action="store_true", help="also run the library's own load-time check, without a device")
    ap.add_argument("--names", action="store_true", help="relay kernel names in the register arm's messages")
    ap.add_argument("--self-test", action="store_true", help="exercise every verdict on fixtures; needs no build")
    args = ap.parse_args(argv)
    if args.self_test:
        return self_test()
    if not args.tree:
        ap.error("--tree BUILD is required (or --self-test)")
    return run(Path(args.tree), Path(args.addon) if args.addon else None, args.registration, args.names)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
