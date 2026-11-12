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
"""tool_help_check.py -- `--help` must not be a traceback.

Four tools in this tree indexed sys.argv directly, so `--help` arrived as a filename: one died with
IndexError, three with `FileNotFoundError: '--help'` (two of those on
`--help/dd_metastate_Llm_Token_Token_modellayers.0…`, which is what a path join with the flag looks
like). A user asking a tool what it does got a stack trace, and nothing in CI noticed because nothing
runs `--help`.

The rule this checks is deliberately narrow: **no traceback**. A tool that prints a usage line and
exits 2 is fine -- that is a style choice, and several tools here use it. A traceback is not.

Usage:
    tool_help_check.py --check [--root DIR]
    tool_help_check.py --self-test
"""

import ast
import hashlib
import os
import pathlib
import subprocess
import sys

SKIP_DIR_NAMES = {'.git', 'build', 'node_modules', '__pycache__', 'third_party', '.venv'}
# Scanned root is tools/ only. `tests/` fixtures are not asked for help, and `os/licenses/` holds
# vendored third-party sources -- a traceback in someone else's example file is not our defect.
TIMEOUT_S = 30


def is_cli(path):
    """A tool that has a `__main__` block and reads sys.argv. A pure module is not a CLI, so silence
    on --help from one is not a defect -- and three of the six silent files under tools/ are exactly
    that (tools/q4nx_pack.py, tools/q4nx_t2s.py, tools/laya_finetune/head.py have no entry point)."""
    try:
        tree = ast.parse(pathlib.Path(path).read_text(errors='replace'))
    except (SyntaxError, OSError):
        return False
    has_main = False
    for node in ast.walk(tree):
        if isinstance(node, ast.If):
            src = ast.dump(node.test)
            if '__name__' in src and '__main__' in src:
                has_main = True
    return has_main and 'sys.argv' in pathlib.Path(path).read_text(errors='replace')


def tree_state(root, cwd):
    """A digest of the tree a tool could write into, and the untracked paths inside it.

    `git status` when the cwd is inside a work tree -- a tool writes relative to the cwd, not to the
    directory it lives in, so scanning `root` alone misses the exact defect this is for: the two
    `oga_*` tools wrote a file named `--help` (172 KB of JSON) into the repo ROOT and a directory
    `--help/dd_metastate_...` beside it. git reports both as untracked, and it costs one process
    instead of hashing every path.
    """
    try:
        # git in the SAME directory the tool runs in -- a write lands relative to that cwd, and asking
        # git from the checker's own cwd reports the wrong tree (the self-test arm caught this).
        r = subprocess.run(['git', 'status', '--porcelain', '--untracked-files=all'],
                           cwd=str(cwd), capture_output=True, text=True, timeout=30)
        if r.returncode == 0:
            untracked = {line[3:] for line in r.stdout.splitlines() if line.startswith('?? ')}
            return 'git:' + hashlib.sha256(r.stdout.encode()).hexdigest()[:16], untracked
    except (OSError, subprocess.TimeoutExpired):
        pass
    h = hashlib.sha256()
    scan = pathlib.Path(cwd)
    paths = set()
    for f in sorted(scan.rglob('*')):
        rel = str(f.relative_to(scan))
        if '.git' in rel.split('/'):
            continue
        paths.add(rel)
        h.update(rel.encode())
        if f.is_file():
            try:
                h.update(str(f.stat().st_size).encode())
            except OSError:
                h.update(b'?')
    return 'tree:' + h.hexdigest()[:16], paths


def remove_created(cwd, created):
    """Delete what a misbehaving tool wrote, deepest first, and only inside the tree it was run in.

    A checker that reports "this tool writes a file while answering --help" and then leaves that file in the
    developer's tree is only half a check: the next `git status` blames them, and a second run reports the same
    tool against a tree that is now dirty for a reason nobody chose.
    """
    scan = pathlib.Path(cwd)
    for rel in sorted(created, key=lambda r: r.count('/'), reverse=True):
        p = scan / rel
        try:
            if p.is_file() or p.is_symlink():
                p.unlink()
            elif p.is_dir():
                p.rmdir()          # only succeeds when the tool's own contents are gone/empty
        except OSError:
            pass


def tools_under(root):
    for f in sorted(pathlib.Path(root).rglob('*.py')):
        if any(part in SKIP_DIR_NAMES for part in f.parts):
            continue
        if f.name == pathlib.Path(__file__).name:
            continue
        yield f


def crashes(path, root, cwd=None):
    """(crashed, detail) for `python3 <path> --help`. A timeout is not a crash: it is a tool we cannot
    ask cheaply, and reporting it as broken would be a claim we did not measure."""
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
    # Absolute, because cwd is the scan root: a relative path would resolve to <root>/<path> and
    # python would exit 2 with "can't open file" and NO traceback, which this function would have
    # read as a clean run. That false pass is the defect this check exists to catch, in its own body.
    cwd = pathlib.Path.cwd() if cwd is None else cwd
    script = pathlib.Path(path).resolve()
    try:
        # cwd is where the caller stands, not the scan root: these tools are documented to run from
        # the repo root, and running them from tools/ changes their answer (check_pins.py's
        # `git config -f .gitmodules` fails there and it looked unaskable rather than silent).
        p = subprocess.run([sys.executable, str(script), '--help'], cwd=str(cwd), env=env,
                           capture_output=True, text=True, timeout=TIMEOUT_S)
    except subprocess.TimeoutExpired:
        return ('skip', f'timed out after {TIMEOUT_S}s')
    err = p.stderr or ''
    if "can't open file" in err:
        # The script we globbed could not be started. That is never a pass and never a skip: it means
        # this checker handed python a path it cannot open, i.e. a defect in here. Kept distinct from
        # a crash so an arm can tell "asked and crashed" from "never asked", which a two-valued
        # result cannot (the first version of this arm passed with the bug reintroduced).
        return ('internal', 'could not be started: ' + err.strip().splitlines()[-1][:80])
    if 'ModuleNotFoundError' in err or 'ImportError' in err:
        # A tool whose dependency is not installed on this box cannot be asked for help. That is
        # absence of a result, not a defect: reporting it would be a claim we did not measure.
        return ('skip', 'a dependency is not installed here')
    if 'Traceback (most recent call last)' in err:
        last = [ln for ln in err.strip().splitlines() if ln.strip()][-1].strip()
        # The subject is "`--help` was treated as an argument", so the *evidence* has to be the flag
        # appearing as a path or value. A tool that dies for another reason -- a missing dependency,
        # a data file that is not in the tree, a git call at import -- cannot answer --help here and
        # is not this check's finding. Attributing those to --help would be a claim the measurement
        # does not support (each of the three was misattributed before this clause existed).
        if '--help' in last or "'-h'" in last:
            return ('fail', last[:120])
        return ('skip', 'cannot answer here: ' + last[:100])
    if p.returncode == 0 and not (p.stdout or '').strip() and not err.strip():
        # Asked a CLI for help and it said nothing, successfully. A crash at least tells you it broke;
        # this tells you nothing at all and looks like a clean run -- the shape check_pins.py had
        # (--help became a base ref, no diff, exit 0).
        return ('silent', 'exited 0 and printed nothing')
    return ('clean', f'rc {p.returncode}')


def check(root):
    bad, skipped, n = [], [], 0
    for f in tools_under(root):
        rel = f.relative_to(root)
        n += 1
        before, before_paths = tree_state(root, pathlib.Path.cwd())
        state, detail = crashes(f, root)
        after, after_paths = tree_state(root, pathlib.Path.cwd())
        if before != after:
            remove_created(pathlib.Path.cwd(), after_paths - before_paths)
            # A tool that writes while answering `--help` is a defect whether or not it also printed
            # something: the file it wrote is named after an argument, and nothing will clean it up.
            bad.append((rel, f'wrote into the tree while answering --help '
                             f'({before} -> {after})'))
            continue
        if state == 'silent':
            if is_cli(f):
                bad.append((rel, detail))
            else:
                skipped.append((rel, 'not a CLI (no __main__), so --help is not its question'))
            continue
        if state == 'skip':
            skipped.append((rel, detail))
        elif state in ('fail', 'internal'):
            bad.append((rel, detail))
    for rel, detail in bad:
        print(f"tool_help: {rel} -- `--help` raises: {detail}")
    for rel, detail in skipped:
        print(f"tool_help: {rel} -- not asked ({detail})")
    if bad:
        print(f"tool_help_check: {len(bad)} tool(s) crash on --help, of {n - len(skipped)} asked "
              f"({len(skipped)} not asked)")
        return 1
    print(f"tool_help_check: no tool crashes on --help ({n - len(skipped)} asked, {len(skipped)} not asked)")
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
        root = pathlib.Path(d)
        (root / 'crashes.py').write_text("import sys\nopen(sys.argv[1]).read()\n")
        (root / 'clean.py').write_text("import sys\nprint('usage: clean.py <x>')\nsys.exit(2)\n")
        (root / 'helps.py').write_text("print('usage: helps.py')\n")
        chk(('fail', 'clean'), (crashes(root / 'crashes.py', root)[0], crashes(root / 'clean.py', root)[0]),
            'a traceback where the flag is the opened path is a crash; a usage line with rc 2 is not')
        chk('clean', crashes(root / 'helps.py', root)[0], 'a tool that prints usage and exits 0 is clean')

        # The false pass that this checker shipped with: cwd is the scan root, so a RELATIVE path
        # resolved to <root>/<path>, python exited 2 with "can't open file" and no traceback, and the
        # tool was counted as clean without ever being asked. Resolved to absolute now.
        sub = root / 'sub'
        sub.mkdir()
        (sub / 'dirs.py').write_text("import sys\nopen(sys.argv[1]).read()\n")
        # The exact shape that produced the false green: the glob yields paths prefixed by the scan
        # root while cwd IS the scan root, so <root>/<path> is <root>/<root>/... . The arm has to
        # reproduce that (a relative path with cwd = root), not pass an absolute one -- the first
        # version of this arm did the latter and passed with the bug reintroduced.
        cwd0 = os.getcwd()
        os.chdir(root.parent)
        try:
            rel = pathlib.Path(root.name) / 'sub' / 'dirs.py'
            state = crashes(rel, root)[0]
        finally:
            os.chdir(cwd0)
        chk('fail', state,
            'a path relative to the invocation cwd, scanned with cwd=root, is still asked '
            '(with the bug it is INTERNAL: a different state, which fails this arm)')

        (root / 'silentcli.py').write_text(
            "import sys\nif __name__ == '__main__':\n    _ = sys.argv[1:]\n")
        (root / 'module.py').write_text("print('imported, not run')\n")
        chk('silent', crashes(root / 'silentcli.py', root)[0],
            'a CLI that exits 0 on --help without printing is silent, not clean')
        chk((True, False), (is_cli(root / 'silentcli.py'), is_cli(root / 'module.py')),
            'a file with __main__ and sys.argv is a CLI; a plain module is not')

        (root / 'dep.py').write_text("import not_a_real_module_xyz\n")
        # The side-effect rule, in the shape that produced it: a tool whose argv[1] is an output path.
        (root / 'writer.py').write_text(
            "import sys, pathlib\nif __name__ == '__main__':\n"
            "    pathlib.Path(sys.argv[1]).write_text('x')\n")
        before, before_paths = tree_state(root, root)
        crashes(root / 'writer.py', root, cwd=root)
        after, after_paths = tree_state(root, root)
        chk(False, before == after,
            'a tool that writes its argument while answering --help changes the tree digest')
        # And the check undoes the write: the file the tool created is gone, and so is its digest change.
        remove_created(root, after_paths - before_paths)
        chk(False, (root / '--help').exists(),
            'the file such a tool wrote is removed, so the tree is left as it was found')
        chk(True, before == tree_state(root, root)[0],
            'after the removal the tree digest is the one from before the run')

        chk('skip', crashes(root / 'dep.py', root)[0], 'a missing dependency is not asked, not a defect')

        (root / 'other.py').write_text("open('data-that-is-not-here.bin').read()\n")
        chk('skip', crashes(root / 'other.py', root)[0],
            'a crash whose traceback never names the flag is not this check\'s finding')
    print(f"tool_help_check --self-test: {'ok' if rc == 0 else 'FAIL'}")
    return rc


def main(argv):
    if '--self-test' in argv:
        return self_test()
    if '--check' not in argv:
        print(__doc__.strip().splitlines()[-1].strip())
        print('usage: tool_help_check.py --check [--root DIR] | --self-test')
        return 2
    root = '.'
    for i, a in enumerate(argv):
        if a == '--root' and i + 1 < len(argv):
            root = argv[i + 1]
    return check(root)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
