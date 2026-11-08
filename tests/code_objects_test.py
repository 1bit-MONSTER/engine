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
"""tools/code_objects.py against crafted files (engine#1443).

The bug this covers: `make-binary-drop.sh` refused a real release build because it grepped for `amdgcn-amd-amdhsa`, a
clang offload-bundle string that a normally-built HIP fat binary does not contain -- the shipped c31 library carries 51
code objects and zero such strings. So the cases below are the ones that separate the property from the string:

  1. a fat library whose own header is x86-64 and which embeds two EM_AMDGPU ELF images -> 2 code objects, passes;
  2. embedded ELF images of another machine (x86-64) -> 0, refused: the check must not be "count ELF magics";
  3. only the file's own header -> 0, refused (the 5.1 MB build on this box, which really has no code objects);
  4. a bare code object (the file itself is EM_AMDGPU) -> 1, passes;
  5. a truncated header at the end -> not a code object, and no crash;
  6. `--min N` is honoured.

No ROCm, no GPU and no big file is needed: 64-byte fixtures, built here.
"""
import os
import struct
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
TOOL = os.path.join(HERE, "..", "tools", "code_objects.py")
EM_AMDGPU, EM_X86_64 = 224, 62
failures = 0


def check(ok, what):
    global failures
    if ok:
        print(f"ok   {what}")
    else:
        failures += 1
        print(f"FAIL {what}")


def elf(machine, pad=44):
    """A minimal ELF header: the magic, the e_type/e_machine pair at offset 16/18, then filler."""
    return b"\x7fELF\x02\x01\x01\x00" + b"\x00" * 8 + struct.pack("<HH", 1, machine) + b"\x00" * pad


def run(path, *args):
    r = subprocess.run([sys.executable, TOOL, path, *args], capture_output=True, text=True)
    return r.returncode, r.stdout.strip(), r.stderr


with tempfile.TemporaryDirectory() as tmp:
    fat = os.path.join(tmp, "fat.so")
    open(fat, "wb").write(elf(EM_X86_64) + os.urandom(64) + elf(EM_AMDGPU) + os.urandom(64) + elf(EM_AMDGPU))
    rc, out, err = run(fat)
    check(rc == 0 and out == "2", f"two embedded EM_AMDGPU images -> 2, passes (rc {rc}, out {out!r})")

    x86 = os.path.join(tmp, "x86.so")
    open(x86, "wb").write(elf(EM_X86_64) + os.urandom(32) + elf(EM_X86_64) + elf(EM_X86_64))
    rc, out, err = run(x86)
    check(rc == 1 and "EM_AMDGPU" in err, "ELF images of another machine are not code objects (refused, names why)")

    header_only = os.path.join(tmp, "plain.so")
    open(header_only, "wb").write(elf(EM_X86_64))
    rc, _, _ = run(header_only)
    check(rc == 1, "a library with only its own header is refused")

    bare = os.path.join(tmp, "obj.hsaco")
    open(bare, "wb").write(elf(EM_AMDGPU))
    rc, out, _ = run(bare)
    check(rc == 0 and out == "1", "a bare code object passes")

    trunc = os.path.join(tmp, "trunc.so")
    open(trunc, "wb").write(elf(EM_X86_64) + b"\x7fELF\x02\x01")   # a cut-off header at the end
    rc, out, _ = run(trunc)
    check(rc == 1 and out == "", "a truncated trailing header is skipped, not counted and not a crash")

    rc, _, _ = run(fat, "--min", "3")
    check(rc == 1, "--min 3 refuses a file with 2 code objects")
    rc, out, _ = run(fat, "--min", "2", "--quiet")
    check(rc == 0 and out == "", "--min 2 with --quiet passes silently")

    rc, _, err = run(os.path.join(tmp, "nope.so"))
    check(rc == 2, "an unreadable file is exit 2, not a false verdict")

# and the real thing, when this box has one: the shipped c31 library, which the old string check refused
REAL = "/home/bcloud/wt/payload-c26c/lemonade-onebit-rc-2026-11-05-c26c/1bit/hrx/llama/bin/libggml-hrx.so.0.18.0"
if os.path.exists(REAL):
    rc, out, _ = run(REAL)
    check(rc == 0 and int(out) >= 40, f"the shipped c31 library passes with {out} code objects (the case in #1443)")
else:
    print("skip the real c31 library (not on this box)")

print(f"code_objects_test: {failures} failure(s)")
sys.exit(1 if failures else 0)
