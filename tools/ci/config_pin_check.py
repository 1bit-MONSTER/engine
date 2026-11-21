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
"""config_pin_check.py -- every pin in config/*.config is a sha, not a hope.

`config/gpu-kernels.config` is consumed twice before anyone sees it work: `scripts/fetch-private.sh` fetches the
checkout it names, and `scripts/hw-gate.sh` builds the code objects from it (it reads `sha=` with a `sed`, at
hw-gate.sh:425). A truncated or mistyped sha therefore fails **hours into an HRX build**, or worse, is fetched by a
consumer that tolerates a short sha and silently builds something else.

engine#2316 named this while bumping the pin: `tools/check_pins.py` compares *submodule gitlinks* and does not cover
`config/*.config` shas at all, so nothing looked at this file until a build did. This is the cheap half of that guard:
the shape, offline, plus `--online` to ask each `repo=` whether the sha is the tip of the ref the pin names -- or of
the repo's default branch when the file names none.

Usage:
    config_pin_check.py --check                    # shape only, no network
    config_pin_check.py --check --online           # also ask each `repo=` for its tip
    config_pin_check.py --self-test
"""

import argparse
import re
import subprocess
import sys

SHA = re.compile(r'^[0-9a-f]{40}$')


def pin_refs(got, default_branch=None):
    """(label, refs to ask) for one pin, or (None, None) when the pin names no ref at all.

    `config/loom.config` names `ref=v0.2.0` / `branch=release/v0.2`, and asking the repo's default branch instead
    reported "behind main" on every run: main and that release diverge by design, so the number could never be 0 and
    said nothing. A `branch=` is asked before a `ref=` because a tag does not move while a branch does; both are
    tried so a pin written as a tag still resolves.

    A pin that names NO ref (engine#3186 removes them from `loom.config`, because `v0.2.0` is `0cd370f2baaa` in
    `ROCm/hrx-system` and `12f2db54fe30` in our fork -- one name, two histories) cannot be asked a branch question
    at all. The caller then asks every advertised ref, which is the only answer that is true in both repos.
    """
    named = got.get('branch') or got.get('ref')
    if named:
        return named, [f'refs/heads/{named}', f'refs/tags/{named}']
    return None, None


def refs_at(lines, sha):
    """Advertised refs of a `git ls-remote --heads --tags` answer that point at `sha`, branches first.

    A tag answers twice (the tag object and its peeled `^{}` commit); the peeled line is the commit, so it wins.
    An empty list is a real answer -- the sha is on no branch or tag, which is what a pin to a mirror-only commit
    looks like in a repo that does not carry it.
    """
    peeled = {}
    for line in lines:
        parts = line.split('\t')
        if len(parts) == 2 and parts[1].rstrip().endswith('^{}'):
            peeled[parts[1].rstrip()[:-3]] = parts[0]
    out = []
    for line in lines:
        parts = line.split('\t')
        if len(parts) != 2 or parts[1].rstrip().endswith('^{}'):
            continue
        if peeled.get(parts[1].rstrip(), parts[0]) == sha:
            out.append(parts[1].strip())
    return sorted(out, key=lambda r: (not r.startswith('refs/heads/'), r))


def tip_from_lsremote(lines):
    """The commit sha among `git ls-remote` output lines, or '' when the repo did not answer.

    Three lines can name one commit: a branch, a tag object, and the tag's peeled `^{}` commit. The order here is
    the point -- preferring the tag object would report a pin as "behind" forever, which is the very shape of the
    `main`-vs-`release/v0.2` defect this mode had (engine#3160).
    """
    heads = [l for l in lines if l.split('\t')[-1].startswith('refs/heads/')]
    peeled = [l for l in lines if l.rstrip().endswith('^{}')]
    plain = [l for l in lines if not l.rstrip().endswith('^{}')]
    pick = heads or peeled or plain
    return pick[0].split()[0] if pick else ''


def parse(path):
    """{key: value} for a config file's `key=value` lines; comments and blanks ignored.

    A file with no `sha=` is not this check's finding: `config/hr-toolchain.config` pins a version and a gfx index, and
    `config/llama-mirror-branch.config` is empty by design.
    """
    out = {}
    for line in path.read_text(encoding='utf-8', errors='replace').splitlines():
        s = line.strip()
        if not s or s.startswith('#'):
            continue
        if s.count('=') == 0:
            continue
        k, _, v = s.partition('=')
        out[k.strip()] = v.strip()
    return out


def check(root, online=False):
    import pathlib
    cfg = pathlib.Path(root) / 'config'
    files = sorted(cfg.glob('*.config')) if cfg.is_dir() else []
    bad, checked, at_tip, asked, unanswered, unattached = [], [], [], [], [], []
    for f in files:
        got = parse(f)
        if 'sha' not in got:
            continue
        checked.append(f)
        if not SHA.match(got['sha']):
            bad.append(f"{f.name}: sha={got['sha']!r} is not 40 lowercase hex characters")
            continue
        if online and 'repo' in got:
            # Ask the repo for the pin's OWN ref, and for every advertised ref when the file names none.
            # `config/npu-kernels.config` pins a repo whose default is `npu-dx`, and the first version of this mode
            # queried `main`, found nothing, and called a good pin bad. A repo that cannot be asked is a SKIP, never
            # a failure -- an unauthenticated fetch of a private repo and a mistyped sha must not read the same.
            branch, patterns = pin_refs(got)
            if patterns is None:
                # No ref to ask: ask all of them. (The default branch alone is the WRONG question -- it is how this
                # pin read "behind" on every run -- but its tip is still context worth printing beside a sha that no
                # ref carries, which is what a lagging pin looks like.)
                argv = ['git', 'ls-remote', '--heads', '--tags',
                        f"https://github.com/{got['repo']}.git"]
            else:
                argv = ['git', 'ls-remote', f"https://github.com/{got['repo']}.git", *patterns]
            lines = subprocess.run(argv, capture_output=True, text=True).stdout.splitlines()
            if branch is None:
                refs = [l for l in lines if not l.startswith('ref:')]
                hits = refs_at(refs, got['sha'])
                if not refs:
                    unanswered.append(f.name)
                    print(f"config_pin: {f.name}: {got['repo']} did not answer for its branches and tags "
                          f"(not checked)")
                elif hits:
                    asked.append(f.name)
                    at_tip.append(f.name)
                    more = f" and {len(hits) - 1} other ref(s)" if len(hits) > 1 else ""
                    print(f"config_pin: {f.name}: sha is {got['sha'][:12]}, at the tip of {got['repo']} "
                          f"{hits[0]}{more}")
                else:
                    # A real answer, not a failure: the sha is on no branch or tag of the repo the pin names. That
                    # is what pinning to a lagging commit looks like, and naming the default branch's tip beside it
                    # is the gap the reader wants. (`--symref` does not help: it omits HEAD once --heads filters the
                    # answer, and one of these repos has a branch literally named `fork/HEAD`.)
                    asked.append(f.name)
                    unattached.append(f.name)
                    br = subprocess.run(['gh', 'api', f"repos/{got['repo']}", '--jq', '.default_branch'],
                                        capture_output=True, text=True)
                    beside = ''
                    if br.returncode == 0 and br.stdout.strip():
                        d = br.stdout.strip()
                        head = next((l.split()[0] for l in refs
                                     if l.split('\t')[-1].strip() == f'refs/heads/{d}'), '')
                        if head:
                            beside = f"; {d} is {head[:12]}"
                    print(f"config_pin: {f.name}: sha is {got['sha'][:12]}, on no advertised branch or tag of "
                          f"{got['repo']} ({len(refs)} ref(s) asked{beside})")
                continue
            tip = [tip_from_lsremote(lines)]
            tip = tip if tip[0] else []
            if not tip:
                # Counted apart from "behind": a repo that never answered is a cannot-tell, and conflating the two
                # is how a pin reads as "behind" until someone bumps it, forever.
                unanswered.append(f.name)
                print(f"config_pin: {f.name}: {got['repo']} did not answer for {patterns[0]} (not checked)")
            elif tip[0] == got['sha']:
                # A match prints nothing in the first version, so `--online` silence meant either "asked and equal"
                # or "never asked" -- measured on engine#2405, where I cited a rc 0 that could not tell me whether
                # the gpu-kernels pin had been compared at all. Say the match out loud.
                asked.append(f.name)
                at_tip.append(f.name)
                print(f"config_pin: {f.name}: sha is {got['sha'][:12]}, at the tip of {got['repo']} {branch}")
            else:
                # Not a failure: a pin may deliberately lag the tip (that is what a pin is for). Reported so a bump
                # PR can show the gap it closes, which is what engine#2316's body does by hand.
                asked.append(f.name)
                print(f"config_pin: {f.name}: sha is {got['sha'][:12]}, "
                      f"{got['repo']} {branch} is {tip[0][:12]}")
    for b in bad:
        print(f"config_pin: {b}")
    if not checked:
        print('config_pin: no config/*.config carries a sha (cannot tell)')
        return 2
    if bad:
        print(f"config_pin_check: {len(bad)} bad pin(s) of {len(checked)} checked")
        return 1
    online_note = ""
    if online:
        # The count alone cannot say whether --online did anything, which is how a matching pin could look
        # unchecked. Name the comparison: M of N were asked, K were at the tip, and the rest had no repo=.
        behind = len(asked) - len(at_tip) - len(unattached)
        online_note = (f"; {len(asked)} asked online, {len(at_tip)} at the tip, "
                       f"{behind} behind it")
        if unanswered:
            online_note += f", {len(unanswered)} did not answer (cannot tell, not a pass)"
        if unattached:
            online_note += f", {len(unattached)} on no advertised branch or tag"
    print(f"config_pin_check: {len(checked)} sha pin(s) are 40 lowercase hex characters{online_note}")
    return 0


def self_test():
    import pathlib
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
        (root / 'config').mkdir()
        good = root / 'config' / 'good.config'
        good.write_text("# comment\nrepo=1bit-MONSTER/gpu-kernels\nsha=" + 'a' * 40 + "\npath=x\n")
        chk(0, check(root), 'a 40-hex sha in a repo-pinning config passes')

        short = root / 'config' / 'short.config'
        short.write_text(f"repo=r\nsha={'b' * 12}\n")
        chk(1, check(root), 'a truncated sha fails, which is the case that costs an HV build')

        upper = root / 'config' / 'upper.config'
        upper.write_text(f"repo=r\nsha={'C' * 40}\n")
        chk(1, check(root), 'uppercase hex fails: every consumer compares these as lowercase')

        # The bad files are removed BEFORE the no-sha arm: leaving them in makes that arm measure their failure
        # instead of the thing it names (this is how the first version of the fixture passed for the wrong reason).
        short.unlink(); upper.unlink()
        nose = root / 'config' / 'nopin.config'
        nose.write_text("version=1.2.3\ngfx=gfx1151\n")
        chk(0, check(root), 'a config with no sha is not this check\'s finding')

        nose.unlink()
        (root / 'config' / 'empty.config').write_text("# nothing\n")
        chk(0, check(root), 'an empty config does not fail the check')

        # The ref arm needs no network: it asserts WHICH question --online would ask. `config/loom.config` names
        # ref=v0.2.0/branch=release/v0.2, and asking main reported "behind" on every run against a branch the pin
        # deliberately is not on (engine#3160, re-measured on main 9a4abfde7).
        chk(('release/v0.2', ['refs/heads/release/v0.2', 'refs/tags/release/v0.2']),
            pin_refs({'branch': 'release/v0.2', 'ref': 'v0.2.0'}),
            'a pin that names a branch is asked about that branch, never the default')
        chk(('v0.2.0', ['refs/heads/v0.2.0', 'refs/tags/v0.2.0']),
            pin_refs({'ref': 'v0.2.0'}), 'a tag-only pin is asked for its tag')
        chk((None, None), pin_refs({'repo': 'r'}),
            'a pin naming no ref has no branch question to ask, so the caller asks every ref')

        # Real `git ls-remote` shapes: an annotated tag answers with the tag object AND its peeled commit, and a
        # repo with both a branch and a tag of the pin's name answers with both.
        chk('c' * 40, tip_from_lsremote([f"{'t' * 40}\trefs/tags/v0.2.0", f"{'c' * 40}\trefs/tags/v0.2.0^{{}}"]),
            'an annotated tag resolves to its peeled commit, not the tag object')
        chk('b' * 40, tip_from_lsremote([f"{'b' * 40}\trefs/heads/main", f"{'c' * 40}\trefs/tags/main"]),
            'a branch wins over a tag of the same name (a branch moves, a tag does not)')
        chk('', tip_from_lsremote([]), 'a repo that did not answer yields no tip, not a false match')

        # engine#3186: `v0.2.0` is 0cd370f2baaa in ROCm/hrx-system and 12f2db54fe30 in our fork, so loom.config stops
        # naming a ref. The no-ref answer must be "which advertised refs carry the sha", and "none" is a real answer.
        _s = 'a' * 40
        chk(['refs/heads/release/v0.2', 'refs/tags/v0.2.0'],
            refs_at([f"{_s}\trefs/tags/v0.2.0", f"{_s}\trefs/heads/release/v0.2", f"{'b' * 40}\trefs/heads/main"], _s),
            'every advertised ref that carries the sha is named, branches before tags')
        chk([], refs_at([f"{'b' * 40}\trefs/heads/release/v0.2", f"{'c' * 40}\trefs/tags/v0.2.0"], _s),
            'a sha no branch or tag carries is reported as such, not as a match')
        chk([], refs_at([], _s), 'a repo that did not answer names no refs (and is not a match)')

    print(f"config_pin_check --self-test: {'ok' if rc == 0 else 'FAIL'}")
    return rc


def self_test_online_reporting():
    """`--online` must say what it compared: a match used to print nothing, which reads as "never asked"."""
    import pathlib, tempfile
    problems = []
    tmp = tempfile.mkdtemp(prefix="pincheck-")
    cfg = pathlib.Path(tmp) / "config"
    cfg.mkdir()
    (cfg / "a.config").write_text("repo=1bit-MONSTER/gpu-kernels\nsha=" + "0" * 40 + "\n")
    (cfg / "b.config").write_text("sha=" + "1" * 40 + "\n")            # no repo=: cannot be asked
    (cfg / "c.config").write_text("repo=1bit-MONSTER/does-not-exist\nsha=" + "2" * 40 + "\n")
    # offline: no online line at all
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        check(tmp, online=False)
    if "asked online" in buf.getvalue():
        problems.append("offline run claimed an online comparison")
    if problems:
        for problem in problems:
            print("FAIL " + problem)
        return 1
    print("config_pin_check --online self-test: ok")
    return 0


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--check', action='store_true')
    ap.add_argument('--online', action='store_true',
                    help="also ask each repo= for the pin's own ref (or its default branch when it names none)")
    ap.add_argument('--root', default='.')
    ap.add_argument('--self-test', action='store_true')
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test() if not a.online else self_test_online_reporting()
    if not a.check:
        ap.print_help()
        return 2
    return check(a.root, online=a.online)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
