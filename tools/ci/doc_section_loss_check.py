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
"""doc_section_loss_check.py -- which sections a branch drops that its base still has.

`registration_loss_check` covers what a branch removes from the *build*: a ctest row whose target vanished, a
`add_test` that a rebase resolved away. The same lost update happens to **documentation**, and nothing looked at it:

  engine#2183, at `407c0c5`: `docs/npu.md` is 3,914 lines on main and 3,358 on the branch, and the merge result is
  missing `### Sentence embeddings from a q4nx encoder (Embedding-Gemma-300M-NPU2, engine#1786)` -- 40 lines that
  had merged **that morning** in #2314. The rebase had resolved `docs/npu.md` by taking the branch's side.

Three decisions that keep this from crying wolf, each from a false positive seen today:

  * **The set is repo-wide, not per file.** A section moved from one document to another is not a loss; a section
    that left the tree is. (Per-file comparison would report every split and every rename of a file.)
  * **A heading is identified by its TEXT, not its level.** Promoting `### Foo` to `## Foo` is an edit, not a
    loss, and a check that compared `"### Foo"` would say otherwise.
  * **It reports and does not judge.** A branch may mean to delete a section; the tool names it and the reader
    decides -- the same contract `registration_loss_check` has.

Usage:
    doc_section_loss_check.py --base origin/main
    doc_section_loss_check.py --base origin/main --head origin/some-branch
    doc_section_loss_check.py --self-test
"""

import argparse
import json
import pathlib
import re
import subprocess
import sys

HEADING = re.compile(r'^\s{0,3}(#{2,4})\s+(.*?)\s*#*\s*$')


def headings_at(rev, root='.', spans=None):
    """{heading text} over every markdown file in the tree at `rev`.

    When `spans` is a dict it is filled with {heading text: (first line, last line)} so a lost section can be
    reported with its SIZE. Without that, "2 sections missing" reads as a corner case: on engine#2183 the two
    headings covered 39 and 63 lines, and the other 592 deleted lines of that file belonged to sections whose
    headings both ends of the hole still had.

    Read from the revision, not the working tree: a branch that *deleted* a document would otherwise have that
    document enumerated from disk and its sections counted as present.
    """
    files = subprocess.run(['git', '-C', root, 'ls-tree', '-r', '--name-only', rev],
                           capture_output=True, text=True).stdout.split()
    out = {}
    for f in files:
        if not f.endswith('.md'):
            continue
        txt = subprocess.run(['git', '-C', root, 'show', f'{rev}:{f}'],
                             capture_output=True, text=True).stdout
        lines = txt.splitlines()
        for i, line in enumerate(lines):
            m = HEADING.match(line)
            if not m:
                continue
            text = m.group(2).strip()
            if text not in out:
                out[text] = f
                if spans is not None:
                    level = len(m.group(1))
                    end = len(lines)
                    for j in range(i + 1, len(lines)):
                        nxt = HEADING.match(lines[j])
                        if nxt and len(nxt.group(1)) <= level:
                            end = j
                            break
                    spans[text] = (i + 1, end, level)
    return out


def check(root, base, head):
    # Compare against the MERGE BASE, not against `base` itself. A branch that forked before other lanes landed
    # their sections shows those as missing when measured against today's main -- the same false positive the
    # registration check names in its own header, and the one that flagged 3 sections here when only 1 was lost.
    mb = subprocess.run(['git', '-C', root, 'merge-base', base, head],
                        capture_output=True, text=True).stdout.strip()
    if not mb:
        print(f'doc_section_loss: cannot find a merge base for {base} and {head} - cannot tell, and that is not a pass')
        return 2
    spans = {}
    b, h = headings_at(mb, root, spans), headings_at(head, root)
    base = mb[:12]
    if not b:
        print(f'doc_section_loss: cannot read any heading at {base} - cannot tell, and that is not a pass')
        return 2
    lost = sorted(set(b) - set(h))
    for t in lost:
        first, last, level = spans.get(t, (0, 0, 0))
        size = last - first + 1
        print(f'doc_section_loss: "{t}" is in {b[t]} at {base} and nowhere at {head} '
              f'(lines {first}-{last}, {size} line(s), {"#" * level})')
    if lost:
        total = sum((spans.get(t, (0, 0, 0))[1] - spans.get(t, (0, 0, 0))[0] + 1) for t in lost)
        print(f'doc_section_loss: {len(lost)} of {len(b)} section(s) at {base} are missing at {head}, {total} line(s) '
              f'- either this branch means to delete them (say so in the commit) or a merge dropped them')
        print('doc_section_loss: this counts SECTIONS, not content: a hole whose two ends kept their headings is '
              'invisible here, so read `git diff --stat <merge-base>..HEAD -- <file>` too')
        return 1
    print(f'doc_section_loss: all {len(b)} section(s) at {base} are present at {head}')
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
        r = pathlib.Path(d)

        def repo(files):
            subprocess.run(['git', 'init', '-q'], cwd=r, check=True)
            for k, v in files.items():
                p = r / k
                p.parent.mkdir(parents=True, exist_ok=True)
                if v is None:
                    if p.exists():
                        p.unlink()
                    continue
                p.write_text(v)
            subprocess.run(['git', 'add', '-A'], cwd=r, check=True)
            subprocess.run(['git', '-c', 'user.email=t@e', '-c', 'user.name=t', 'commit', '-qm', 'm'],
                           cwd=r, check=True)
            return subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=r, capture_output=True,
                                  text=True).stdout.strip()

        a = repo({'d.md': '## Keep\n\n### Lost\n\n## Also\n'})
        b = repo({'d.md': '## Keep\n\n## Also\n'})
        chk(1, check(r, a, b), 'a section the branch removed is a finding')

        c = repo({'d.md': '## Keep\n\n## Also\n', 'e.md': '### Lost\n'})
        chk(0, check(r, a, c), 'a section MOVED to another document is not a loss')

        d2 = repo({'d.md': '## Keep\n\n# Lost\n\n## Also\n'})
        chk(0, check(r, a, d2), 'a section whose LEVEL changed is an edit, not a loss')

        chk(2, check(r, 'does-not-exist', b), 'an unreadable base is rc 2, not a pass')

        # The size of a lost section is the reason this reports at all: on engine#2183 the two headings covered 39
        # and 63 lines, so "2 sections" undersold 104 lines. An arm pins the arithmetic, not just the count.
        wider = repo({'d.md': '## Keep\n\n### Lost\n' + ('body\n' * 7) + '## Also\n'})
        spans = {}
        headings_at(wider, str(r), spans)
        chk((3, 10, 3), spans.get('Lost'), 'a lost section reports its line span and level')

    print(f"doc_section_loss_check --self-test: {'ok' if rc == 0 else 'FAIL'}")
    return rc


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--base', help='the revision this branch should still contain')
    ap.add_argument('--head', default='HEAD')
    ap.add_argument('--root', default='.')
    ap.add_argument('--self-test', action='store_true')
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test()
    if not a.base:
        ap.error('--base is required')
    return check(a.root, a.base, a.head)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
