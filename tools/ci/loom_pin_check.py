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
"""loom_pin_check.py -- the Loom toolchain pin says what the cards were built with, and the tree agrees.

`config/loom.config` records three facts that are only true together (LOOM-GPU-SWITCH.md, work item L5): the
`hrx-system` commit the shipped Loom cards were built against, the `hrx_loom` version that commit declares in
`loom/packaging/python/version.json`, and the `hrx-loom-library` catalog commit that same file names. Each is a
`sha=`/`version=` line in a config file, and config files do not fail when they drift -- a pin that names a commit
whose version.json says something else builds a toolchain nobody tested, and the drift is invisible until a card
fails to compile against it.

This is the cheap half: read the three, read the version file at the recorded commit, and say whether they agree.
Nothing is fetched. Without a local clone of the repo it answers **cannot tell (2)**, never a pass -- the pin is a
claim about a commit this box does not have, which is exactly the state a reviewer needs named.

Also reported, not judged: whether the recorded commit is an **ancestor of the engine's `third_party/hrx-system`
gitlink**. Today it is not (measured 2026-11-21), which means the engine tree cannot build this toolchain and it has
to come from a separate clone; that is the state LOOM-GPU-SWITCH.md condition 1 leaves the fallback route in.

`--loom-serve PATH|auto` also checks the **file that gets launched**, because none of the three facts above can:
`1bit serve` launches `loom-serve` by PATH, and the binary cannot say which toolchain it is (`--version` prints
"development (commit unknown)" at 0.2.0 and is rejected outright by 0.1.0), while package metadata answers "what does
interpreter X have installed" -- two different questions when an activated venv puts another `loom-serve` ahead of the
pin's one. So the pin records that binary's sha256 and this compares it.

Usage:
    loom_pin_check.py --check [--source DIR] [--python PYTHON] [--config PATH] [--loom-serve PATH|auto]
    loom_pin_check.py --self-test

Exit: 0 the pin agrees, 1 a finding, 2 cannot tell (no clone, or no installed package to compare).
"""

import argparse
import json
import pathlib
import re
import shutil
import subprocess
import sys

DEFAULTS = ('~/amd-oss/loom-fun/hrx-system', '~/amd-oss/hrx-system', '~/.cache/scratch-cl-infra/hrxs')


def read_config(path):
    out = {}
    for line in pathlib.Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        if '=' in line:
            k, v = line.split('=', 1)
            out[k.strip()] = v.strip()
    return out


def git(source, *args):
    return subprocess.run(['git', '-C', str(source), *args], capture_output=True, text=True, check=False)


def find_source(explicit):
    """The first clone that actually contains the recorded commit; None when none does."""
    candidates = [explicit] if explicit else list(DEFAULTS)
    for c in candidates:
        d = pathlib.Path(c).expanduser()
        if not d.is_dir():
            continue
        if (d / '.git').exists() or (d / 'HEAD').exists():
            yield d


def version_at(source, sha):
    """(dict, None) for the version.json at `sha`, or (None, reason)."""
    ls = git(source, 'ls-tree', '--name-only', sha, 'loom/packaging/python/version.json')
    if ls.returncode != 0:
        return None, f'{sha[:10]} is not a commit in {source}'
    if not ls.stdout.strip():
        return None, f'{sha[:10]} carries no loom/packaging/python/version.json (a runtime pin, not a toolchain pin)'
    show = git(source, 'show', f'{sha}:loom/packaging/python/version.json')
    if show.returncode != 0:
        return None, f'cannot read the version file at {sha[:10]} in {source}'
    try:
        return json.loads(show.stdout), None
    except json.JSONDecodeError as e:
        return None, f'the version file at {sha[:10]} is not JSON: {e}'


def installed_version(python):
    if not python:
        return None, 'no interpreter given (--python), so the installed package is not compared'
    exe = shutil.which(python) or (python if pathlib.Path(python).exists() else None)
    if not exe:
        return None, f'{python} is not an interpreter here'
    p = subprocess.run([exe, '-c',
                        'import importlib.metadata as m\ntry:\n print(m.version("hrx_loom"))\nexcept Exception:\n raise SystemExit(3)'],
                       capture_output=True, text=True, check=False)
    if p.returncode == 3:
        return None, f'{python} has no hrx_loom installed'
    if p.returncode != 0:
        return None, f'{python} could not be queried: {(p.stderr or "").strip()[:120]}'
    return p.stdout.strip(), None


def check(cfg_path, source=None, python=None, wheel=None, loom_serve=None, installed=True):
    findings, notes = [], []
    cfg = read_config(cfg_path)
    for key in ('repo', 'sha', 'package', 'version', 'version_file', 'model_catalog_repo', 'model_catalog_sha'):
        if not cfg.get(key):
            findings.append(f'{cfg_path} has no {key}= line')
    if findings:
        for f in findings:
            print(f'loom_pin: {f}', file=sys.stderr)
        return 1
    sha, version, cat = cfg['sha'], cfg['version'], cfg['model_catalog_sha']
    if not re.fullmatch(r'[0-9a-f]{40}', sha):
        print(f'loom_pin: sha= is not a 40-hex commit: {sha!r}', file=sys.stderr)
        return 1

    src = None
    for cand in find_source(source):
        v, why = version_at(cand, sha)
        if v is not None:
            src, data = cand, v
            break
        notes.append(f'{cand}: {why}')
    else:
        for n in notes:
            print(f'loom_pin: {n}', file=sys.stderr)
        print(f'loom_pin: no local clone carries {sha[:10]}; the pin is a claim this box cannot verify -- '
              f'clone {cfg["repo"]} and pass --source, then re-run (cannot tell, and that is not a pass)',
              file=sys.stderr)
        return 2

    have = data.get('package-version')
    if have != version:
        findings.append(f'{cfg_path} says version={version} but {sha[:10]} declares package-version={have!r}')
    catalog = data.get('model-catalog') or ''
    cat_ref = cfg.get('model_catalog_ref')
    if not (catalog.endswith('@' + cat) or (cat_ref and catalog.endswith('@' + cat_ref))):
        findings.append(f'{cfg_path} says model_catalog_sha={cat[:12]}'
                        f'{" / model_catalog_ref=" + cat_ref if cat_ref else ""}'
                        f' but {sha[:10]} names {catalog or "<none>"}')
    notes.append(f'{cfg["package"]} {have} at {sha[:10]} in {src}')

    gitlink = git(src, 'rev-parse', 'HEAD').stdout.strip()
    inside = git(src, 'merge-base', '--is-ancestor', sha, 'HEAD').returncode == 0 if gitlink else False
    notes.append(f'{sha[:10]} is {"an ancestor of" if inside else "NOT an ancestor of"} the local clone head {gitlink[:10]}')
    have_inst, why = installed_version(python)
    if not installed:
        notes.append('this box\'s installed package was NOT compared (--no-installed): the pin is a claim about the tree')
    elif have_inst is None:
        notes.append(f'installed package not compared: {why}')
    elif have_inst != version:
        findings.append(f'the installed hrx_loom is {have_inst}, the pin says {version}')
    else:
        notes.append(f'installed hrx_loom {have_inst} matches the pin')

    if wheel:
        import hashlib
        want = cfg.get('wheel_sha256')
        if not want:
            findings.append(f'{cfg_path} records no wheel_sha256, so {wheel} cannot be checked against the shipped artefact')
        elif not pathlib.Path(wheel).is_file():
            findings.append(f'no wheel at {wheel}')
        else:
            got = hashlib.sha256(pathlib.Path(wheel).read_bytes()).hexdigest()
            if got != want:
                findings.append(f'{wheel} has sha256 {got[:16]}…, the pin says {want[:16]}…')
            else:
                notes.append(f'{pathlib.Path(wheel).name} sha256 matches the pin')

    # The launchable binary (see the docstring): the one fact none of the above can establish.
    binpath = None
    if loom_serve == 'auto':
        binpath = shutil.which('loom-serve')
        if not binpath:
            findings.append('--loom-serve auto: no loom-serve on PATH')
    elif loom_serve:
        binpath = loom_serve
    if binpath:
        import hashlib
        want = cfg.get('loom_serve_sha256')
        p = pathlib.Path(binpath).expanduser()
        if not p.is_file():
            findings.append(f'no loom-serve at {binpath}')
        elif not want:
            findings.append(f'{cfg_path} records no loom_serve_sha256, so {binpath} cannot be checked against the pin')
        else:
            got = hashlib.sha256(p.read_bytes()).hexdigest()
            if got != want:
                findings.append(f'{binpath} has sha256 {got[:16]}…, the pin says {want[:16]}…'
                                f'{" (a venv ahead of it on PATH?)" if ".venv" in str(p) else ""}')
            else:
                notes.append(f'{binpath} sha256 matches the pin')
    else:
        notes.append('the launchable loom-serve was not identified (--loom-serve PATH|auto), so nothing here says '
                     'which binary a route would start')
    for n in notes:
        print(f'loom_pin: {n}')
    for f in findings:
        print(f'loom_pin: {f}', file=sys.stderr)
    if findings:
        return 1
    print(f'loom_pin: {cfg["package"]} {version} at {sha[:10]} and catalog {cat[:12]} agree')
    return 0


def self_test():
    import tempfile
    ok = [0, 0]

    def chk(cond, what):
        ok[0 if cond else 1] += 1
        print(('ok   ' if cond else 'FAIL ') + what)

    tmp = pathlib.Path(tempfile.mkdtemp())
    src = tmp / 'src'
    src.mkdir()
    subprocess.run(['git', '-C', str(src), 'init', '-q'], check=True)
    for k, v in (('user.email', 't@e'), ('user.name', 't')):
        subprocess.run(['git', '-C', str(src), 'config', k, v], check=True)
    vf = src / 'loom/packaging/python'
    vf.mkdir(parents=True)
    (vf / 'version.json').write_text(json.dumps(
        {'package-version': '0.2.0.dev0', 'model-catalog': 'github:ROCm/hrx-loom-library@dfd67f4ab2333b648274a3fff97baad464878436'}))
    subprocess.run(['git', '-C', str(src), 'add', '-A'], check=True)
    subprocess.run(['git', '-C', str(src), 'commit', '-qm', 'v'], check=True)
    sha = subprocess.run(['git', '-C', str(src), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()

    def cfg(**over):
        base = dict(repo='ROCm/hrx-system', sha=sha, package='hrx_loom', version='0.2.0.dev0',
                    version_file='loom/packaging/python/version.json',
                    model_catalog_repo='ROCm/hrx-loom-library',
                    model_catalog_sha='dfd67f4ab2333b648274a3fff97baad464878436')
        base.update(over)
        p = tmp / f'c{len(list(tmp.glob("c*")))}.config'
        p.write_text(''.join(f'{k}={v}\n' for k, v in base.items()))
        return p

    chk(check(cfg(), source=str(src)) == 0, 'a config that agrees with its commit is exit 0')
    chk(check(cfg(version='0.1.0'), source=str(src)) == 1, 'a version the commit does not declare is a finding')
    chk(check(cfg(model_catalog_sha='0' * 40), source=str(src)) == 1, 'a catalog pin the commit does not name is a finding')
    chk(check(cfg(sha='f' * 40), source=str(src)) == 2, 'a commit the clone does not have is cannot-tell, not a pass')
    chk(check(cfg(), source=str(tmp / 'nope')) == 2, 'no clone at all is cannot-tell')
    chk(check(cfg(sha='abc'), source=str(src)) == 1, 'a short sha is a finding (the shape check that config_pin_check also does)')

    # a RELEASE names its catalog by tag, not by sha (`model-catalog: github:ROCm/hrx-loom-library@v0.2.0`): the ref form must
    # be accepted, or every release pin would report a finding it does not have.
    (vf / 'version.json').write_text(json.dumps({'package-version': '0.2.0', 'model-catalog': 'github:ROCm/hrx-loom-library@v0.2.0'}))
    subprocess.run(['git', '-C', str(src), 'commit', '-qam', 'release'], check=True)
    rsha = subprocess.run(['git', '-C', str(src), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    chk(check(cfg(sha=rsha, version='0.2.0', model_catalog_ref='v0.2.0'), source=str(src)) == 0,
        'a catalog named by TAG is accepted when model_catalog_ref names it')
    chk(check(cfg(sha=rsha, version='0.2.0', model_catalog_ref='v0.1.0'), source=str(src)) == 1,
        '...and a ref the release does not name is still a finding')

    # The shipped ARTEFACT: a release is a wheel with a hash, not just a version string.
    whl = tmp / 'hrx_loom-0.2.0-py3-none-any.whl'
    whl.write_bytes(b'PK\x03\x04 not really a wheel, but its hash is what the pin names')
    import hashlib
    digest = hashlib.sha256(whl.read_bytes()).hexdigest()
    chk(check(cfg(sha=rsha, version='0.2.0', model_catalog_ref='v0.2.0', wheel_sha256=digest), source=str(src), wheel=str(whl)) == 0,
        'a wheel whose sha256 is the pinned one is accepted')
    chk(check(cfg(sha=rsha, version='0.2.0', model_catalog_ref='v0.2.0', wheel_sha256='0' * 64), source=str(src), wheel=str(whl)) == 1,
        '...and a wheel with any other sha256 is a finding')

    # The launchable binary: the artefact a route starts. Same "identify the file" reasoning as the wheel, and the same
    # two ways to get it wrong (an unpinned binary, and a binary that is not the pinned one).
    binf = tmp / 'loom-serve'
    binf.write_bytes(b'\x7fELF not really, but its hash is what the pin names')
    bdig = hashlib.sha256(binf.read_bytes()).hexdigest()
    rel = dict(sha=rsha, version='0.2.0', model_catalog_ref='v0.2.0')
    chk(check(cfg(loom_serve_sha256=bdig, **rel), source=str(src), loom_serve=str(binf)) == 0,
        'a loom-serve whose sha256 is the pinned one is accepted')
    chk(check(cfg(loom_serve_sha256='0' * 64, **rel), source=str(src), loom_serve=str(binf)) == 1,
        '...and any other loom-serve binary is a finding')
    chk(check(cfg(**rel), source=str(src), loom_serve=str(binf)) == 1,
        'a binary pin missing from the config is a finding, not a silence')
    chk(check(cfg(loom_serve_sha256=bdig, **rel), source=str(src), loom_serve=str(tmp / 'nope')) == 1,
        'a loom-serve that is not there is a finding')
    chk(check(cfg(loom_serve_sha256=bdig, **rel), source=str(src)) == 0,
        'not identifying a binary is a note, not a finding (a runner has none installed)')
    chk(check(cfg(loom_serve_sha256=bdig, **rel), source=str(src), python=str(tmp / 'no-such-python'), installed=False) == 0,
        '--no-installed never consults this box, so a broken or foreign interpreter cannot make the row fail')
    print(f'loom_pin_check self-test: {ok[0]} ok, {ok[1]} failed')
    return 0 if ok[1] == 0 else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--check', action='store_true')
    ap.add_argument('--self-test', action='store_true')
    ap.add_argument('--config', default=str(pathlib.Path(__file__).resolve().parents[2] / 'config/loom.config'))
    ap.add_argument('--source', default=None, help='a clone of the repo in config/loom.config')
    ap.add_argument('--python', default=sys.executable, help='interpreter to ask for the installed hrx_loom')
    ap.add_argument('--wheel', default=None, help='a hrx_loom wheel whose sha256 must match config/loom.config')
    ap.add_argument('--no-installed', dest='installed', action='store_false',
                    help='check the pin\'s facts about the TREE only; do not compare this box\'s installed package '
                         '(a pre-PR row must not fail a lane for another lane\'s venv)')
    ap.add_argument('--loom-serve', default=None, metavar='PATH|auto',
                    help='the loom-serve binary a route would launch (auto = the first on PATH); its sha256 must match '
                         'config/loom.config, because the binary cannot report its own toolchain')
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test()
    if not a.check:
        ap.error('one of --check or --self-test')
    return check(a.config, a.source, a.python, a.wheel, a.loom_serve, a.installed)


if __name__ == '__main__':
    sys.exit(main())
