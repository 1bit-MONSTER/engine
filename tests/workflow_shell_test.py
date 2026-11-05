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
"""A workflow's shell must parse, and the pin bump step must open one PR per moved repo.

pin-watch.yml's "Open one pin bump per changed repo" step was committed with an unterminated quote
(`git commit -m "pin: $repo at $newsha""`). Nothing executed that step until a pin actually moved, so a
syntax error that kills the whole step - and with it every bump PR - sat on main looking green: the
scheduled run at 05:43 UTC would have been its first reader.

Two guarantees, both offline (CI installs no PyYAML, so the blocks are lifted out of the text):

  * every `run:` in every workflow parses under `bash -n` (the class, not just this instance);
  * the pin bump step, driven against a local bare origin, a stubbed `gh` and a stubbed `python3`, opens
    exactly one PR per moved repo, leaves a repo that went upstream to a human, and fails the run when an
    upstream could not be read at all - an unread upstream is not a clean day.
"""
import json
import os
import re
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKFLOWS = os.path.join(ROOT, ".github", "workflows")
STEP = "Open one pin bump per changed repo"
failures = []


def check(what, cond, detail=""):
    if cond:
        print(f"ok   {what}")
    else:
        print(f"FAIL {what}" + (f": {detail}" if detail else ""))
        failures.append(what)


def blocks(text):
    """[(line number, shell)] for every `run: |` block and every inline `run:` scalar."""
    lines = text.splitlines()
    out, i = [], 0
    while i < len(lines):
        m = re.match(r"^(\s*)(?:- )?run: (\||>[-+]?|)\s*$", lines[i])
        inline = re.match(r"^(\s*)(?:- )?run: (?!\||>)(\S.*)$", lines[i])
        if m:
            indent, body, j = len(m.group(1)), [], i + 1
            while j < len(lines) and (not lines[j].strip() or len(lines[j]) - len(lines[j].lstrip()) > indent):
                body.append(lines[j])
                j += 1
            while body and not body[-1].strip():
                body.pop()
            cut = min((len(b) - len(b.lstrip()) for b in body if b.strip()), default=0)
            out.append((i + 1, "\n".join(b[cut:] for b in body) + "\n"))
            i = j
        elif inline:
            out.append((i + 1, inline.group(2)))
            i += 1
        else:
            i += 1
    return out


def step_script(text, name):
    """The de-indented shell of the named step, or None."""
    lines = text.splitlines()
    at = next((i for i, l in enumerate(lines) if f"- name: {name}" in l), None)
    if at is None:
        return None
    found = blocks("\n".join(lines[at:]))
    return found[0][1] if found else None


def sandbox(tmp, report):
    """A bare origin plus a working clone carrying a pins file and the report the check step writes."""
    origin, work = os.path.join(tmp, "origin.git"), os.path.join(tmp, "work")
    subprocess.run(["git", "init", "-q", "--bare", origin], check=True)
    subprocess.run(["git", "clone", "-q", origin, work], check=True, capture_output=True)
    for cmd in (["config", "user.email", "t@example.com"], ["config", "user.name", "t"]):
        subprocess.run(["git", "-C", work] + cmd, check=True)
    os.makedirs(os.path.join(work, "models"))
    os.makedirs(os.path.join(work, "tools"))
    pins = {"version": 1, "pins": {}}
    for repo in ("ok/one", "ok/two", "human/gone", "broken/x"):
        pins["pins"][repo] = {"commit": "a" * 40, "pinned": "2026-11-02", "used_for": ["t"], "files": {}}
    with open(os.path.join(work, "models", "pins.json"), "w") as fh:
        json.dump(pins, fh, indent=1)
    open(os.path.join(work, "tools", "pin_watch.py"), "w").close()
    with open(os.path.join(work, "pin-report.md"), "w") as fh:
        fh.write("# Model pins: 4 of 4 moved (2026-11-05)\n\n")
        for c in report:
            fh.write(f"## `{c['repo']}`\n\n{c['body']}\n")
    subprocess.run(["git", "-C", work, "add", "-A"], check=True)
    subprocess.run(["git", "-C", work, "commit", "-qm", "init"], check=True)
    subprocess.run(["git", "-C", work, "branch", "-M", "main"], check=True)
    subprocess.run(["git", "-C", work, "push", "-q", "-u", "origin", "main"], check=True)

    stub = os.path.join(tmp, "stub")
    os.makedirs(stub)
    log = os.path.join(tmp, "gh.log")
    with open(os.path.join(stub, "gh"), "w") as fh:
        fh.write(f'#!/bin/sh\nprintf "%s\\n" "$*" >> "{log}"\necho https://example/pr/1\n')
    # `python3` stands in for the tool's two network calls; the step's own `python3 -c` still runs for real
    with open(os.path.join(stub, "python3"), "w") as fh:
        fh.write(f'''#!/bin/sh
REAL={sys.executable}
if [ "$1" = tools/pin_watch.py ] && [ "$2" = --bump ]; then
  case "$3" in
    human/gone) echo "pin_watch: $3: gone upstream (404)" >&2; exit 4 ;;
    broken/x)   echo "pin_watch: $3: could not read the Hub" >&2; exit 1 ;;
  esac
  "$REAL" -c 'import json,sys; p="models/pins.json"; d=json.load(open(p)); d["pins"][sys.argv[1]]["commit"]="b"*40; json.dump(d,open(p,"w"),indent=1)' "$3" || exit 1
  exit 0
fi
if [ "$1" = tools/pin_watch.py ] && [ "$2" = --pr-body ]; then
  echo "body for $3"; exit 0
fi
exec "$REAL" "$@"
''')
    for f in ("gh", "python3"):
        os.chmod(os.path.join(stub, f), 0o700)
    return work, stub, log


def drive(script, work, stub, log):
    env = dict(os.environ, PATH=stub + os.pathsep + os.environ.get("PATH", ""), GH_TOKEN="test")
    p = subprocess.run(["bash", "-c", script], cwd=work, env=env, capture_output=True, text=True)
    calls = open(log).read().splitlines() if os.path.exists(log) else []
    heads = subprocess.run(["git", "-C", work, "ls-remote", "--heads", "origin"],
                           capture_output=True, text=True).stdout.split()
    return p, [c for c in calls if c.strip()], heads


# 1. every workflow's shell parses - the class the pin-watch step was caught by
for name in sorted(os.listdir(WORKFLOWS)):
    if not name.endswith(".yml"):
        continue
    text = open(os.path.join(WORKFLOWS, name)).read()
    found = blocks(text)
    bad = []
    for line, shell in found:
        p = subprocess.run(["bash", "-n"], input=shell, text=True, capture_output=True)
        if p.returncode != 0:
            bad.append(f"line {line}: {p.stderr.strip().splitlines()[-1]}")
    check(f"{name}: {len(found)} run block(s) parse", not bad, "; ".join(bad)[:200])

# 2. the pin bump step: one PR per moved repo, a gone repo left to a human, an unread one fatal
text = open(os.path.join(WORKFLOWS, "pin-watch.yml")).read()
script = step_script(text, STEP)
check("pin-watch: has the pin bump step", script is not None)
if script:
    p = subprocess.run(["bash", "-n"], input=script, text=True, capture_output=True)
    check("pin-watch: the pin bump step's shell parses", p.returncode == 0, p.stderr.strip()[:200])

    report = [{"repo": "ok/one", "body": "`a` -> `b`"},
              {"repo": "human/gone", "body": "**Gone upstream** — the Hub answers `404`."},
              {"repo": "ok/two", "body": "`a` -> `b`"}]
    with tempfile.TemporaryDirectory() as tmp:
        work, stub, log = sandbox(tmp, report)
        p, calls, heads = drive(script, work, stub, log)
        creates = [c for c in calls if c.startswith("pr create")]
        check("pin-watch: a gone repo does not fail the run", p.returncode == 0, p.stderr.strip()[-200:])
        check("pin-watch: one 'gh pr create' per moved repo", len(creates) == 2, str(calls))
        check("pin-watch: the gone repo is warned about, not bumped",
              "human/gone needs a human decision" in p.stdout + p.stderr)
        branches = sorted(h.split("refs/heads/")[-1] for h in heads if "refs/heads/pin-watch/" in h)
        check("pin-watch: one branch per moved repo, none for the gone one", len(branches) == 2, str(branches))
        for repo in ("ok/one", "ok/two"):
            slug = repo.replace("/", "-").replace(".", "-")
            branch = next((b for b in branches if b.startswith(f"pin-watch/{slug}-")), None)
            check(f"pin-watch: {repo} has a pushed branch at main", branch is not None, str(branches))
            if branch is None:
                continue
            check(f"pin-watch: {repo}'s PR names that branch",
                  any(c.startswith("pr create") and f"--head {branch}" in c for c in creates), str(creates))
            msg = subprocess.run(["git", "-C", work, "log", "-1", "--format=%s", branch],
                                 capture_output=True, text=True).stdout.strip()
            check(f"pin-watch: {repo}'s commit says what it pins", msg == f"pin: {repo} at bbbbbbbb", msg)
            files = subprocess.run(["git", "-C", work, "show", "--name-only", "--format=", branch],
                                   capture_output=True, text=True).stdout.split()
            check(f"pin-watch: {repo}'s commit touches only the pins file", files == ["models/pins.json"], str(files))

    report = [{"repo": "broken/x", "body": "`a` -> `b`"}]
    with tempfile.TemporaryDirectory() as tmp:
        work, stub, log = sandbox(tmp, report)
        p, calls, heads = drive(script, work, stub, log)
        check("pin-watch: an unread upstream fails the run instead of a green no-PR day",
              p.returncode != 0 and "could not be bumped" in p.stdout + p.stderr, f"rc={p.returncode}")
        check("pin-watch: and opens nothing", not [c for c in calls if c.startswith("pr create")], str(calls))

if failures:
    print(f"\n{len(failures)} check(s) failed")
    sys.exit(1)
print("\nall checks passed")
