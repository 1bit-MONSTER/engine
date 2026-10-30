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
"""Every scheduled bump workflow must report its own failures (ctest bump_workflow_report).

bump-hrx failed on every scheduled run from 2026-10-01 to 2026-10-05 and bump-ds4 from
2026-10-21 to 2026-10-30; both froze their pin while the job went red at the same time
every day and told no one. The step that fixes that is only as good as it is uniform, so
this test drives the reporting step of every bump workflow against a stubbed `gh`:

  * the workflow has the step, last, with `if: failure()`, and `issues: write`;
  * with no open issue it runs `gh issue create` once, with that workflow's own title;
  * with the issue already open it runs `gh issue comment <number>` once, and creates
    nothing - one thread per failing workflow, not one issue a day;
  * no two workflows share a title, so their threads cannot be merged by accident.

No network and no yaml module: the step is lifted out of the workflow text (CI installs
only numpy and gguf, so PyYAML is not available here), and `gh` is a shell stub that
records its arguments.
"""
import os
import re
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKFLOWS = ["bump-ds4", "bump-laya", "bump-lemonade", "bump-tokenizers", "bump-xdna"]
STEP_NAME = "Report a failed sync so it cannot break silently"
failures = []


def check(what, cond, detail=""):
    if cond:
        print(f"ok   {what}")
    else:
        print(f"FAIL {what}" + (f": {detail}" if detail else ""))
        failures.append(what)


def block_scalar(lines, start):
    """The de-indented body of the `run: |` block that begins at or after line index `start`."""
    i = start
    while i < len(lines) and "run: |" not in lines[i]:
        i += 1
    if i == len(lines):
        return None
    indent = len(lines[i]) - len(lines[i].lstrip())
    body = []
    for line in lines[i + 1:]:
        if line.strip() and (len(line) - len(line.lstrip())) <= indent:
            break
        body.append(line)
    while body and not body[-1].strip():
        body.pop()
    return "\n".join(b[4 * (indent // 2 + 1):] if False else b for b in body)


def step_script(text):
    """(script, title) of the reporting step, or (None, None) when the workflow lacks it."""
    lines = text.splitlines()
    at = next((i for i, l in enumerate(lines) if f"- name: {STEP_NAME}" in l), None)
    if at is None:
        return None, None
    head = "\n".join(lines[at:at + 8])
    if "if: failure()" not in head:
        return None, None
    script = block_scalar(lines, at)
    m = re.search(r'^ *title="([^"]+)"', script or "", re.M)
    return script, (m.group(1) if m else None)


def run_step(script, existing):
    """Run the step's shell with a stubbed `gh`; return the calls it made."""
    with tempfile.TemporaryDirectory() as tmp:
        log = os.path.join(tmp, "calls")
        stub = os.path.join(tmp, "gh")
        with open(stub, "w") as fh:
            fh.write("#!/bin/sh\n")
            fh.write(f'printf "%s\\n" "$*" >> "{log}"\n')
            fh.write('case "$*" in\n')
            # `gh issue list … --jq …` prints the open issue number, or nothing
            fh.write(f'  "issue list"*) [ -n "{existing}" ] && echo "{existing}" ;;\n')
            fh.write("esac\nexit 0\n")
        os.chmod(stub, 0o700)   # owner-only: the stub is run by this user, no one else needs it
        env = dict(os.environ, PATH=tmp + os.pathsep + os.environ.get("PATH", ""),
                   GITHUB_RUN_ID="1234567890", GITHUB_REPOSITORY="1bit-MONSTER/engine")
        p = subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True)
        calls = []
        if os.path.exists(log):
            calls = [l for l in open(log).read().splitlines() if l.strip()]
        return p.returncode, calls, p.stderr


titles = {}
for wf in WORKFLOWS:
    path = os.path.join(ROOT, ".github", "workflows", f"{wf}.yml")
    if not os.path.exists(path):
        check(f"{wf}: workflow exists", False, path)
        continue
    text = open(path).read()
    script, title = step_script(text)
    check(f"{wf}: has the '{STEP_NAME}' step with if: failure()", script is not None)
    if script is None:
        continue
    check(f"{wf}: grants issues: write", re.search(r"^permissions:\n(?:  .*\n)*?  issues: write", text, re.M) is not None)
    check(f"{wf}: declares a title", title is not None, "no title= line in the step")
    if title:
        titles.setdefault(title, []).append(wf)

    # the shell has to parse before anything else is believed
    p = subprocess.run(["bash", "-n"], input=script, text=True, capture_output=True)
    check(f"{wf}: the step's shell parses", p.returncode == 0, p.stderr.strip()[:120])

    rc, calls, err = run_step(script, existing="")
    creates = [c for c in calls if c.startswith("issue create")]
    comments = [c for c in calls if c.startswith("issue comment")]
    check(f"{wf}: no open issue -> exactly one 'gh issue create'", rc == 0 and len(creates) == 1 and not comments,
          f"rc={rc} calls={calls} stderr={err.strip()[:80]}")
    if creates and title:
        check(f"{wf}: creates the issue with its own title", f'--title {title}' in creates[0], creates[0][:160])
    check(f"{wf}: the body names the failing run", any("actions/runs/1234567890" in c for c in creates + comments))

    rc, calls, err = run_step(script, existing="42")
    creates = [c for c in calls if c.startswith("issue create")]
    comments = [c for c in calls if c.startswith("issue comment")]
    check(f"{wf}: issue already open -> one comment on #42, nothing created",
          rc == 0 and not creates and len(comments) == 1 and " 42 " in comments[0] + " ",
          f"rc={rc} calls={calls} stderr={err.strip()[:80]}")

dupes = {t: w for t, w in titles.items() if len(w) > 1}
check("every workflow reports under its own title", not dupes, str(dupes))

if failures:
    print(f"\n{len(failures)} check(s) failed")
    sys.exit(1)
print("\nall checks passed")
