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
"""`tools/copyright.py --check` must see a file that is not staged yet.

The tool enumerated `git ls-files`, so an untracked file was invisible to it — and a brand-new file
is untracked exactly when a lane runs the check before `git add`. That is how
`onnx/tests/ort_run_generic.sh` passed `--check` locally and failed CI's copyright step on the same
head (PR #1406): a green from a check that could not see the file most likely to be wrong.

Each case runs the real tool in a scratch git repository, so it tests the enumeration and not a copy
of it. Ignored files must stay out: a build artifact needs no notice.
"""
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
TOOL = os.path.join(HERE, os.pardir, "tools", "copyright.py")
# Data, not this file's header: NOTICE is the string the cases below assert the tool against, which is why
# the second Copyright+SPDX pair here is legitimate (engine#1547).
NOTICE = ("# Copyright 2026 bong-water-water-bong\n"
          "# SPDX-License-Identifier: Apache-2.0\n"
          "#\n"
          "# Licensed under the Apache License, Version 2.0 (the \"License\");\n"
          "# you may not use this file except in compliance with the License.\n"
          "# You may obtain a copy of the License at\n"
          "#\n"
          "#     http://www.apache.org/licenses/LICENSE-2.0\n"
          "#\n"
          "# Unless required by applicable law or agreed to in writing, software\n"
          "# distributed under the License is distributed on an \"AS IS\" BASIS,\n"
          "# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.\n"
          "# See the License for the specific language governing permissions and\n"
          "# limitations under the License.\n")
FAIL = []


def check(ok, what, why=""):
    print(("ok   " if ok else "FAIL ") + what + (("  -- " + why) if not ok and why else ""))
    if not ok:
        FAIL.append(what)


def run(repo, *args):
    return subprocess.run([sys.executable, os.path.abspath(TOOL), *args], cwd=repo,
                          capture_output=True, text=True)


def git(repo, *args):
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


with tempfile.TemporaryDirectory() as repo:
    git(repo, "init", "-q")
    git(repo, "config", "user.email", "t@example.invalid")
    git(repo, "config", "user.name", "test")
    # a tracked file WITH its notice: the baseline must stay green
    with open(os.path.join(repo, "ok.py"), "w") as f:
        f.write(NOTICE + "print(1)\n")
    git(repo, "add", "ok.py")
    git(repo, "commit", "-q", "-m", "baseline")
    rc = run(repo, "--check")
    check(rc.returncode == 0, "a tracked file with its notice passes", rc.stdout + rc.stderr)

    # THE CASE: a new file without a notice, NOT staged — this used to pass
    with open(os.path.join(repo, "new.sh"), "w") as f:
        f.write("#!/usr/bin/env bash\necho hi\n")
    rc = run(repo, "--check")
    check(rc.returncode != 0,
          "an UNTRACKED file without a notice FAILS the check (the bug this fixes)",
          f"rc {rc.returncode}: {rc.stdout.strip()}")
    check("new.sh" in (rc.stdout + rc.stderr),
          "and the failure names it", rc.stdout + rc.stderr)

    # --fix must reach it too, and then the check is green
    rc = run(repo, "--fix")
    check(rc.returncode == 0 and "new.sh" in rc.stdout, "--fix adds the notice to the untracked file",
          rc.stdout + rc.stderr)
    check(NOTICE.splitlines()[3] in open(os.path.join(repo, "new.sh")).read(),
          "the notice it added is the full Apache block, not just SPDX")
    rc = run(repo, "--check")
    check(rc.returncode == 0, "and the check is green afterwards", rc.stdout + rc.stderr)

    # an ignored file is not source: it must not need a notice. The .gitignore itself is source (the
    # repo's own carries the notice and is tracked), so it gets one here too.
    with open(os.path.join(repo, ".gitignore"), "w") as f:
        f.write(NOTICE + "build/\n")
    git(repo, "add", ".gitignore")
    git(repo, "commit", "-q", "-m", "ignore build/")
    os.mkdir(os.path.join(repo, "build"))
    with open(os.path.join(repo, "build", "artifact.txt"), "w") as f:
        f.write("no notice here\n")
    rc = run(repo, "--check")
    check(rc.returncode == 0, "an ignored file is left out", rc.stdout + rc.stderr)

    # A notice pasted twice in the header. This shape shipped in #1535 and #1544 because has_notice() only
    # asked whether the notice was PRESENT; #1547 lists the 20 files it left behind.
    with open(os.path.join(repo, "twice.py"), "w") as f:
        f.write(NOTICE + "\n# Copyright 2026 bong-water-water-bong\n# SPDX-License-Identifier: Apache-2.0\nprint(2)\n")
    rc = run(repo, "--check")
    check(rc.returncode != 0, "a notice duplicated inside the header window FAILS the check",
          f"rc {rc.returncode}: {rc.stdout.strip()}")
    check("twice.py" in (rc.stdout + rc.stderr), "and the failure names the file", rc.stdout + rc.stderr)

    # resolve it the way the tool's message says (delete the second copy) so the next case starts clean
    with open(os.path.join(repo, "twice.py"), "w") as f:
        f.write(NOTICE + "print(2)\n")

    # ... but a second occurrence that is DATA, further down, is legitimate: a heredoc or template that emits
    # the notice into a generated file. Three files in the tree are exactly this, and they must stay green.
    with open(os.path.join(repo, "emits.sh"), "w") as f:
        f.write(NOTICE + "".join(f"# line {i}\n" for i in range(20))
                + "cat > out.h <<'EOF'\n# Copyright 2026 bong-water-water-bong\n"
                  "# SPDX-License-Identifier: Apache-2.0\nEOF\n")
    rc = run(repo, "--check")
    check(rc.returncode == 0, "a second occurrence past the window (emitted notice) stays green",
          rc.stdout + rc.stderr)

print(f"copyright_check_untracked_test: {len(FAIL)} failure(s)")
sys.exit(1 if FAIL else 0)
