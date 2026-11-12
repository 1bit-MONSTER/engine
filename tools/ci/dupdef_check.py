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
"""dupdef_check.py -- a shadowed top-level definition silently disables the one it shadows.

A Python module may define `def self_test()` twice. The second wins, the first is dead, and nothing
warns: the file still imports, the surviving function still passes, and the arms the first one
carried can never fail because they never run. `tools/hrx_f16_narrowing_check.py` shipped exactly
that -- two `self_test()` bodies, the scan arms in the shadowed one -- which is a check that cannot
fail wearing the costume of two checks.

Usage:
    dupdef_check.py --check [--root DIR ...]
    dupdef_check.py --self-test
"""

import ast
import pathlib
import sys

# Vendored and generated trees are not ours to police; a duplicate there is upstream's business.
SKIP_DIR_NAMES = {'.git', 'build', 'node_modules', '__pycache__', 'third_party', '.venv'}
DEFAULT_ROOTS = ('tools', 'scripts')


def duplicates(path):
    """[(name, [lineno, ...])] for top-level def/class names defined more than once."""
    try:
        tree = ast.parse(path.read_text(errors='replace'))
    except (SyntaxError, OSError):
        # A file we cannot parse is not this check's finding; whoever owns syntax reporting says so.
        return []
    seen = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            seen.setdefault((node.name, type(node).__name__), []).append(node.lineno)
    return [(name, lines) for (name, _kind), lines in sorted(seen.items()) if len(lines) > 1]


def files_under(roots):
    for root in roots:
        p = pathlib.Path(root)
        if p.is_file() and p.suffix == '.py':
            yield p
            continue
        for f in sorted(p.rglob('*.py')):
            if not any(part in SKIP_DIR_NAMES for part in f.parts):
                yield f


def main(argv):
    if '--self-test' in argv:
        return self_test()
    if '--check' not in argv:
        print(__doc__.strip().splitlines()[-1].strip())
        print('usage: dupdef_check.py --check [--root DIR ...] | --self-test')
        return 2
    roots = []
    for i, a in enumerate(argv):
        if a == '--root' and i + 1 < len(argv):
            roots.append(argv[i + 1])
    roots = roots or list(DEFAULT_ROOTS)

    bad = 0
    n = 0
    for f in files_under(roots):
        if f.name == 'dupdef_check.py':
            continue                      # its own fixtures define duplicates on purpose
        n += 1
        for name, lines in duplicates(f):
            bad += 1
            print(f"dupdef: {f}: `{name}` is defined {len(lines)} times (lines {', '.join(map(str, lines))}) "
                  f"-- everything but the last is dead, and its assertions never run")
    if bad:
        print(f"dupdef_check: {bad} shadowed definition(s) in {n} file(s)")
        return 1
    print(f"dupdef_check: no shadowed top-level definition in {n} file(s)")
    return 0


def self_test():
    import tempfile
    rc = 0

    def chk(want, got, what):
        nonlocal rc
        if want == got:
            print(f"ok   {what}")
        else:
            print(f"FAIL {what}: want {want}, got {got}")
            rc = 1

    with tempfile.TemporaryDirectory() as d:
        p = pathlib.Path(d, 'shadowed.py')
        p.write_text("def self_test():\n    return 1\n\ndef other():\n    return 2\n\ndef self_test():\n    return 3\n")
        chk([('self_test', [1, 7])], duplicates(p), 'two top-level defs of one name are reported with both lines')

        q = pathlib.Path(d, 'indented.py')
        q.write_text("class A:\n    def f(self):\n        return 1\n    def f(self):\n        return 2\n")
        chk([], duplicates(q), 'a method defined twice inside a class is not a top-level shadow')

        r = pathlib.Path(d, 'once.py')
        r.write_text("def a():\n    pass\n\ndef b():\n    pass\n")
        chk([], duplicates(r), 'distinct names are not a finding')

        s = pathlib.Path(d, 'broken.py')
        s.write_text("def a(:\n")
        chk([], duplicates(s), 'an unparseable file is not this check\'s finding (absence of a result, not a pass)')

        # The defect this check was written for, reproduced: the surviving def passes, the shadowed one never runs.
        t = pathlib.Path(d, 'hrx_shaped.py')
        t.write_text("def self_test():\n    check('scan arm')\n\ndef check(x):\n    pass\n\ndef self_test():\n    pass\n")
        chk([('self_test', [1, 7])], duplicates(t), 'the hrx_f16_narrowing_check.py shape (a shadowed self_test) is caught')

    print(f"dupdef_check --self-test: {'ok' if rc == 0 else 'FAIL'}")
    return rc


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
