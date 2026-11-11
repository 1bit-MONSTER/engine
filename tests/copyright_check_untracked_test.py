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

    # engine#2175: a SHORT header (Copyright + SPDX, no licence block) is not a "missing copyright notice".
    # The message has to name the shape, or a reader hunts for a notice that is on line 1 -- measured on
    # #2175/#2173 whose audio headers kept only the two lines while #2133 carried the full block.
    short = NOTICE.splitlines()[:2]
    with open(os.path.join(repo, "short.h"), "w") as f:
        f.write("\n".join(short) + "\n// short.h -- the file's own description\n")
    git(repo, "add", "short.h")
    rc = run(repo, "--check")
    check(rc.returncode != 0, "a SHORT header still fails (the licence block is required)", rc.stdout)
    check("copyright + SPDX present but the licence block is not" in rc.stdout,
          "and the failure names the SHAPE, not a missing notice", rc.stdout + rc.stderr)
    check("missing copyright notice: short.h" not in rc.stdout,
          "and does NOT claim the notice is missing", rc.stdout + rc.stderr)
    # engine#2175: --fix must REPLACE the stub, not stack the block above it. Before this it left both, so the
    # file came out with the Copyright/SPDX pair twice, the duplicate rule fired, and --check stayed red -- the
    # tool's own remedy could not fix the shape it had just named.
    rc = run(repo, "--fix")
    check(rc.returncode == 0, "--fix repairs a short header", rc.stdout + rc.stderr)
    got = open(os.path.join(repo, "short.h")).read()
    check(got.count("SPDX-License-Identifier") == 1,
          "and exactly one SPDX line survives the repair", f"{got.count('SPDX-License-Identifier')} found")
    check("Licensed under the Apache License, Version 2.0" in got,
          "and the full licence block is there", got[:200])
    check("the file's own description" in got, "and the file's own header text is kept", got)
    rc = run(repo, "--check")
    check(rc.returncode == 0, "and --check is green after the repair", rc.stdout + rc.stderr)
    # engine#2175 review: a script's short header sits BEHIND its `#!`. The first version of strip_short_header()
    # looked at line 1 only, so the stub was not found, the block went in after the shebang, and the file ended up
    # with two notices -- the same defect one file type over.
    with open(os.path.join(repo, "short.sh"), "w") as f:
        f.write("#!/usr/bin/env bash\n" + NOTICE.splitlines()[0] + "\n" + NOTICE.splitlines()[1]
                + "\n# short.sh -- a script\necho hi\n")
    git(repo, "add", "short.sh")
    rc = run(repo, "--fix")
    check(rc.returncode == 0, "--fix repairs a short header behind a shebang", rc.stdout + rc.stderr)
    got = open(os.path.join(repo, "short.sh")).read()
    check(got.startswith("#!/usr/bin/env bash\n"),
          "the shebang stays on line 1", got.splitlines()[0])
    check(got.count("SPDX-License-Identifier") == 1,
          "and exactly one SPDX line in the script too", f"{got.count('SPDX-License-Identifier')} found")
    check("short.sh -- a script" in got, "and the script's own comment survives", got)
    rc = run(repo, "--check")
    check(rc.returncode == 0, "and --check is green for the script", rc.stdout + rc.stderr)
    os.unlink(os.path.join(repo, "short.sh"))
    git(repo, "add", "-A", "short.sh")
    os.unlink(os.path.join(repo, "short.h"))
    git(repo, "add", "-A", "short.h")

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

    # engine#1564: an untracked file whose comment style the tool does not know is debris, not repo content —
    # run output a probe dropped (`.hrx123/logs/*.jsonl`), a scratch file. Skipping it is what stops the gate
    # from reding a developer clone while `main` is clean; a TRACKED file of the same name is still an error.
    with open(os.path.join(repo, "probe-dump.jsonl"), "w") as f:
        f.write('{"x": 1}\n')
    rc = run(repo, "--check")
    check(rc.returncode == 0,
          "an UNTRACKED file with no known comment style is skipped, not failed (engine#1564)",
          rc.stdout + rc.stderr)
    git(repo, "add", "probe-dump.jsonl")
    git(repo, "commit", "-q", "-m", "track a jsonl")
    rc = run(repo, "--check")
    check(rc.returncode != 0, "the same file TRACKED is still an error: it would ship without a notice",
          rc.stdout + rc.stderr)
    check("no comment style known" in (rc.stdout + rc.stderr), "and it is named as an unknown style",
          rc.stdout + rc.stderr)

    # A build tree's generated sources carry no notice by design, and `.h` IS a known style, so before this
    # an untracked `build*/generated/*.h` went straight to missing[] -- 36 of them in a developer clone with a
    # build dir, while `main` was clean. Untracked + under a build dir is skipped; TRACKED is still an error.
    git(repo, "rm", "-q", "--cached", "probe-dump.jsonl")
    git(repo, "commit", "-q", "-m", "drop the tracked jsonl")
    os.makedirs(os.path.join(repo, "build-1336", "generated"))
    with open(os.path.join(repo, "build-1336", "generated", "recipes_default.h"), "w") as f:
        f.write("static const int k = 1;\n")
    rc = run(repo, "--check")
    check(rc.returncode == 0,
          "an UNTRACKED generated source in a build tree is skipped, not failed",
          rc.stdout + rc.stderr)
    git(repo, "add", "-f", "build-1336/generated/recipes_default.h")
    git(repo, "commit", "-q", "-m", "track a generated header")
    rc = run(repo, "--check")
    check(rc.returncode != 0, "the same generated source TRACKED is still an error: it is repo content now",
          rc.stdout + rc.stderr)

# --fix must not delete blank lines (#2183 review). The licence block's own blank comment lines reduce to the empty
# string, and canonical_notice_line() treated "" as a notice line. So add_notice() deleted every blank or bare `//`
# line in the header window: on #2183's 13 two-liner headers --fix gave +143 -68, dropping the separator after the
# description, the blank lines between #include groups and the one after `namespace {`. And repair_notice()'s
# doubled-block dedupe dropped every blank line after the first in the WHOLE file. Both cases compare the written
# file with the exact expected text, so a single lost line fails them.
CPP_NOTICE = "".join(("//" + line[1:]) for line in NOTICE.splitlines(keepends=True))
CPP_BODY = ("//\n// x.cpp -- what this file is for.\n//\n// A second paragraph.\n\n"
            "#include <cstdio>\n#include <string>\n\n#include \"x.h\"\n\nnamespace {\n\nint g = 0;\n\n}  // namespace\n")
with tempfile.TemporaryDirectory() as repo:
    git(repo, "init", "-q")
    path = os.path.join(repo, "twoliner.cpp")
    with open(path, "w") as f:
        f.write("// Copyright 2026 bong-water-water-bong\n// SPDX-License-Identifier: Apache-2.0\n" + CPP_BODY)
    run(repo, "--fix")
    with open(path) as f:
        fixed = f.read()
    check(fixed == CPP_NOTICE + CPP_BODY,
          "--fix on a two-liner header keeps every blank and bare `//` line below it",
          "lost %d blank line(s)" % (CPP_BODY.count("\n\n") - fixed.count("\n\n")))
    rc = run(repo, "--check")
    check(rc.returncode == 0, "...and the file it wrote passes --check", rc.stdout + rc.stderr)

with tempfile.TemporaryDirectory() as repo:
    git(repo, "init", "-q")
    path = os.path.join(repo, "nearmiss.cpp")
    # blank lines well past the 20-line header window: the repair path reads the whole file
    long_body = CPP_BODY + "".join("int v%d = %d;\n" % (i, i) for i in range(30)) + "\nint z;\n\nint w;\n"
    with open(path, "w") as f:
        f.write(CPP_NOTICE + "// Copyright 2026 bong-water-bong\n// SPDX-License-Identifier: Apache-2.0\n" + long_body)
    run(repo, "--fix")
    with open(path) as f:
        fixed = f.read()
    check(fixed == CPP_NOTICE + long_body,
          "--fix repairing a near-miss holder keeps every blank line in the file, past the header window too",
          "lost %d blank line(s)" % (long_body.count("\n\n") - fixed[len(CPP_NOTICE):].count("\n\n")))
    rc = run(repo, "--check")
    check(rc.returncode == 0, "...and the repaired file passes --check", rc.stdout + rc.stderr)

print(f"copyright_check_untracked_test: {len(FAIL)} failure(s)")
sys.exit(1 if FAIL else 0)
