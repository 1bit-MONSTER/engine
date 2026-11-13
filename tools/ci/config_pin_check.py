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
the shape, offline, plus `--online` to ask each repo whether the sha is the advertised tip.

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
    bad, checked = [], []
    for f in files:
        got = parse(f)
        if 'sha' not in got:
            continue
        checked.append(f)
        if not SHA.match(got['sha']):
            bad.append(f"{f.name}: sha={got['sha']!r} is not 40 lowercase hex characters")
            continue
        if online and 'repo' in got:
            # Ask the repo for its OWN default branch: `config/npu-kernels.config` pins a repo whose default is
            # `npu-dx`, and the first version of this mode queried `main`, found nothing, and called a good pin bad.
            # A repo that cannot be asked is a SKIP, never a failure -- an unauthenticated fetch of a private repo and
            # a mistyped sha must not read the same.
            br = subprocess.run(['gh', 'api', f"repos/{got['repo']}", '--jq', '.default_branch'],
                                capture_output=True, text=True)
            if br.returncode != 0 or not br.stdout.strip():
                print(f"config_pin: {f.name}: cannot ask {got['repo']} for its default branch "
                      f"(not checked, not a pass)")
                continue
            branch = br.stdout.strip()
            tip = subprocess.run(['git', 'ls-remote', f"https://github.com/{got['repo']}.git",
                                  f'refs/heads/{branch}'], capture_output=True, text=True).stdout.split()
            if not tip:
                print(f"config_pin: {f.name}: {got['repo']} did not answer for refs/heads/{branch} (not checked)")
            elif tip[0] != got['sha']:
                # Not a failure: a pin may deliberately lag the tip (that is what a pin is for). Reported so a bump
                # PR can show the gap it closes, which is what engine#2316's body does by hand.
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
    print(f"config_pin_check: {len(checked)} sha pin(s) are 40 lowercase hex characters")
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

    print(f"config_pin_check --self-test: {'ok' if rc == 0 else 'FAIL'}")
    return rc


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--check', action='store_true')
    ap.add_argument('--online', action='store_true', help='also ask each repo= for refs/heads/main')
    ap.add_argument('--root', default='.')
    ap.add_argument('--self-test', action='store_true')
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test()
    if not a.check:
        ap.print_help()
        return 2
    return check(a.root, online=a.online)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
