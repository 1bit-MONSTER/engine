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
"""The release notes' Source code section (tools/os_sources.py -> tools/weekly_changes.py).

Renders the section from a fixture sources.json (tests/data/os_sources) and checks its rows,
checks that weekly_changes.py writes the same notes as before when --os-sources is absent
(and those notes plus the section when it is given), and checks the pieces of os_sources.py
that decide what is GPL: the copyright-file classifier, the merged-/usr path aliases, and,
where dpkg is present, a real lookup of BusyBox at its image path /bin/busybox.

No network: weekly_changes' git and GitHub calls are replaced with fixed answers.
"""
import datetime as dt
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import os_sources as osrc  # noqa: E402
import weekly_changes as wc  # noqa: E402

FIXTURE = os.path.join(ROOT, "tests/data/os_sources/sources.json")
FAILS = []


def check(cond, what):
    print(("ok   " if cond else "FAIL ") + what)
    if not cond:
        FAILS.append(what)


def notes(mod, extra=()):
    """the Markdown weekly_changes.main() writes, with git and GitHub answered from here"""
    def run(*cmd):
        if "log" in cmd:
            return "abc1234\tserve: a fix (#12)\tbong\ndef5678\tBump llama.cpp to f5b7f4a (#13)\tbot\n"
        if "rev-parse" in cmd:
            return "abc1234abc1234abc1234abc1234abc1234abc12\n"
        raise AssertionError(cmd)
    mod.run = run
    mod.commit_date = lambda src, rev: dt.datetime(2026, 9, 27, tzinfo=dt.timezone.utc)
    mod.submodules = lambda src, rev: {}
    mod.pins = lambda src, rev, mods: []
    mod.models = lambda src, old, new, after: {"architectures": [], "lemonade": [], "huggingface": []}
    with tempfile.TemporaryDirectory() as d:
        bumps = os.path.join(d, "bumps.json")
        json.dump([{"number": 14, "title": "Bump zinc", "result": "held", "why": "tests failed"}], open(bumps, "w"))
        md = os.path.join(d, "notes.md")
        sys.argv = ["weekly_changes.py", "--src", d, "--from", "a", "--to", "b", "--tag", "v2026.40",
                    "--previous", "v2026.39", "--bumps", bumps, "--json", os.path.join(d, "c.json"),
                    "--md", md, *extra]
        mod.main()
        return open(md).read()


# what weekly_changes.py wrote for this input before --os-sources existed (origin/main 0baf286)
BEFORE = """Weekly build of 1bit engine at the pins below, since v2026.39.

## Held back

- #14 Bump zinc: tests failed

## What moved upstream

No pin moved this week.

## Engine

- serve: a fix (#12)

## Packages

| file | what |
|---|---|
| `1bit-v2026.40-linux-x86_64.tar.zst` | 1bit and its backends (Vulkan, HRX, lean, ZINC, ONNX, DwarfStar, NPU with XRT) |
| `lemonade-onebit-v2026.40-linux-x86_64.tar.zst` | Lemonade (lemond + CLI) with the onebit recipe |
| `1bit-v2026.40-windows-x64.zip` | 1bit.exe with Vulkan and ONNX backends |
| `1bit-os-v2026.40.img.zst`, `.efi` | 1bit OS: boot the engine from a USB stick |
| `SHA256SUMS`, `changes.json` | checksums; this list as data |

Licenses: the engine is Apache-2.0 (LICENSE, NOTICE in each package); each backend keeps its own license. 1bit OS also carries GPL programs (the Linux kernel, BusyBox) as built on the release machine; their complete source is available on request for three years from this release: open an issue at https://github.com/1bit-MONSTER/engine/issues.
"""


def test_section():
    src = json.load(open(FIXTURE))
    md = "\n".join(wc.source_section(src))
    rows = [l for l in md.splitlines() if l.startswith("| ") and not l.startswith("| Component")]
    check(md.startswith("## Source code\n"), "section starts with its heading")
    check("| Linux kernel | GPL-2.0 | 7.3-rc4, `third_party/linux` at `72d3fcf802c4`, built as `7.3.0-rc4-1bit` "
          "with `config/kernel/strixhalo.config` + `strixhalo.fragment` (`scripts/build-kernel.sh`) | "
          "[torvalds/linux@72d3fcf](https://github.com/torvalds/linux/tree/72d3fcf802c45d00b300f25b848a93c3a2bd7c7e), "
          "config in this repository |" in rows, "kernel row: the pin, the release, the config")
    check("| BusyBox | GPL-2.0 | Ubuntu `busybox` 1:1.37.0-7ubuntu1 (BusyBox v1.37.0) | "
          "`apt-get source busybox=1:1.37.0-7ubuntu1` · [busybox.net](https://busybox.net/downloads/) |" in rows,
          "BusyBox row: apt-get source and upstream link")
    check("| GNU C Library (libc6) | LGPL-2.1+ | Ubuntu `glibc` 2.43-2ubuntu2.4 | `apt-get source glibc=2.43-2ubuntu2.4` |"
          in rows, "glibc row")
    check(any(r.startswith("| GCC runtime (libgcc-s1, libstdc++6) | GPL-3.0 with the GCC Runtime Library Exception |")
              for r in rows), "GCC row names the runtime libraries, not gcc-16-base")
    check(any(r.startswith("| util-linux libraries (libblkid1, libmount1, libuuid1) | LGPL-2.1+ / BSD-3-Clause |")
              for r in rows), "util-linux row")
    check(any("amdxdna" in r and "GPL v2" in r and "amd/xdna-driver@995ecbb" in r for r in rows),
          "out-of-tree amdxdna row points at the xdna-driver pin")
    check(any(r.startswith("| EFI stub (systemd-boot-efi)") and "`apt-get source systemd=259.5-0ubuntu3.4`" in r
              for r in rows), "EFI stub row (LGPL, part of the .efi)")
    check("| linux-firmware | redistributable binary-only (see each WHENCE entry) | Ubuntu `linux-firmware` "
          "20260319.git217ca6e4 | `apt-get source linux-firmware=20260319.git217ca6e4` · "
          "[linux-firmware](https://git.kernel.org/pub/scm/linux/kernel/git/firmware/linux-firmware.git) |" in rows,
          "firmware row")
    check(not any("Mesa" in r or "dropbear" in r for r in rows), "permissive packages stay out of the table")
    perm = next((l for l in md.splitlines() if l.startswith("The other components")), "")
    check("Mesa (MIT)" in perm and "dropbear (MIT)" in perm and "XRT (Apache-2.0" in perm,
          "permissive list: Mesa, Dropbear, XRT")
    check("Ubuntu 26.04 source packages" in perm, "permissive list names the distribution release")
    check("**Written offer.** For at least three years from this release" in md and
          'https://github.com/1bit-MONSTER/engine/issues with the title "Source request" and the release tag.' in md,
          "written offer: three years, Source request issue")
    order = [r.split(" | ")[0] for r in rows]
    check(order[0] == "| Linux kernel" and order[-1].startswith("| linux-firmware"), "kernel first, firmware last")


def test_notes():
    without = notes(wc)
    check(without == BEFORE, "notes without --os-sources are byte-identical to before")
    section = "\n".join(wc.source_section(json.load(open(FIXTURE))))
    check(notes(wc, ["--os-sources", FIXTURE]) == without + "\n" + section,
          "with --os-sources: the same notes, then the Source code section")


def test_classify():
    dep5 = "Format: https://www.debian.org/doc/packaging-manuals/copyright-format/1.0/\n\n"
    cases = [
        ("Files: *\nCopyright: x\nLicense: LGPL-2.1+\n\nFiles: debian/*\nLicense: GPL-2+\n", "lgpl"),
        ("Files: *\nLicense: MIT\n\nFiles: debian/*\nLicense: GPL-2+\n", "permissive"),
        ("Files: *\nLicense: BSD-3-clause or GPL-2\n", "permissive"),
        ("Files: *\nLicense: GPL-2+ or LGPL-3+\n", "lgpl"),
        ("Files: *\nLicense: GPL-3+ with Bison exception\n", "gpl"),
        ("Files: src/*\nLicense: Apache-2.0\n\nFiles: lib/*\nLicense: LGPL-2.1+\n", "lgpl"),
    ]
    for text, want in cases:
        got, _ = osrc.classify(dep5 + text)
        label = " / ".join(l for l in text.splitlines() if l.startswith("License:"))
        check(got == want, f"classify {label!r} -> {want} (got {got})")
    check(osrc.classify("This package is under the GNU General Public License v2.\n")[0] == "gpl", "free-form GPL")
    check(osrc.classify("GNU Lesser General Public License\n")[0] == "lgpl", "free-form LGPL")
    check(osrc.classify("Permission is hereby granted, free of charge\n")[0] == "permissive", "free-form MIT")
    check(osrc.classify(None)[0] == "unknown", "no copyright file -> unknown")
    check("/usr/bin/busybox" in osrc.aliases("/bin/busybox"), "/bin/busybox is looked up as /usr/bin/busybox")
    check("/lib/x86_64-linux-gnu/libc.so.6" in osrc.aliases("/usr/lib/x86_64-linux-gnu/libc.so.6"), "/usr/lib -> /lib")
    check(osrc.aliases("/opt/xilinx/xrt/lib/x") == ["/opt/xilinx/xrt/lib/x"], "no alias outside merged dirs")
    check(osrc.kernel_version("7.3.0-rc4-1bit") == "7.3-rc4", "kernel 7.3.0-rc4-1bit -> 7.3-rc4")
    check(osrc.kernel_version("7.0.12-34-generic") == "7.0.12", "kernel 7.0.12-34-generic -> 7.0.12")


def test_collect():
    """BusyBox copied to /bin/busybox, as mkimage.sh does, is found by its /usr/bin owner"""
    if not shutil.which("dpkg") or subprocess.run(["dpkg", "-S", "/usr/bin/busybox"], capture_output=True).returncode:
        print("skip collect: no dpkg-owned /usr/bin/busybox on this machine")
        return
    with tempfile.TemporaryDirectory() as root:
        os.makedirs(os.path.join(root, "bin"))
        shutil.copy("/usr/bin/busybox", os.path.join(root, "bin/busybox"))
        os.symlink("busybox", os.path.join(root, "bin/sh"))
        with open(os.path.join(root, "init"), "w") as f:
            f.write("#!/bin/sh\n")
        data = osrc.collect(root, "7.3.0-rc4-1bit", ROOT, None, None, None, None)
    bb = [p for p in data["packages"] if p["source"] == "busybox"]
    check(len(bb) == 1 and bb[0]["class"] == "gpl" and bb[0]["files"] == 1, "busybox: one file, GPL, symlinks not counted")
    check((data["busybox"] or {}).get("source") == "busybox" and data["busybox"].get("upstream_version"),
          "busybox record: package and upstream version")
    check(data["unowned"] == [{"path": "/init", "component": "other"}], "/init owned by no package")
    check(data["kernel"]["from_pin"] and data["kernel"]["version"] == "7.3-rc4", "kernel record from the release")


if __name__ == "__main__":
    test_section()
    test_notes()
    test_classify()
    test_collect()
    if FAILS:
        print(f"{len(FAILS)} failed")
        sys.exit(1)
    print("all passed")
