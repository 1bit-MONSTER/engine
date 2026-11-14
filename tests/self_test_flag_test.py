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
"""self_test_flag_test.py -- the `--self-test` flag must survive being ignored.

engine#2445 review: the flag's whole job is to *return the child's status*, so the failure
worth pinning is the quiet one. `subprocess.run([...])` without `.returncode` exits 0
whatever the arms said, and every row that calls the tool would stay green while the arms
went red. Equally, a flag that disappears reads as an argparse error (rc 2) rather than as
"no self-test".

Both are measured the same way: a copy of the tool in a scratch tree, pointed at a test
file that exits with a chosen status. Nothing here imports the tools, so the arms can move
without this file following them.

Exit: 0 the flag reports the child's status, 1 it does not.
"""
import os
import shutil
import subprocess
import sys
import tempfile

TOOLS = ("registry_loss_check", "branch_stale_check")


def tool_root(tmp, tool):
    """A scratch tree shaped like the repo: tools/ci/<tool>.py next to tests/."""
    src = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools", "ci", tool + ".py")
    shutil.copyfile(src, os.path.join(tmp, "tools", "ci", tool + ".py"))
    return os.path.join(tmp, "tools", "ci", tool + ".py")


def run_self_test(tmp, tool, child_rc):
    """Run the copied tool's --self-test against a test file that exits `child_rc`."""
    test_path = os.path.join(tmp, "tests", tool + "_test.py")
    with open(test_path, "w") as f:
        f.write("import sys\nsys.exit(%d)\n" % child_rc)
    return subprocess.run([sys.executable, os.path.join(tmp, "tools", "ci", tool + ".py"), "--self-test"],
                          capture_output=True, text=True)


def main():
    fails = 0

    def check(label, ok, detail=""):
        nonlocal fails
        if ok:
            print("ok   " + label)
        else:
            print("FAIL " + label + ("  [" + detail + "]" if detail else ""))
            fails += 1

    for tool in TOOLS:
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "tools", "ci"))
            os.makedirs(os.path.join(tmp, "tests"))
            tool_root(tmp, tool)

            # (1)+(2) THE ARMS THAT MATTER, and they are checked as a PAIR: the tool's status must
            # equal the child's, so the two runs have to differ. Checking "child 1 -> rc != 0" alone
            # is not enough -- a dropped `.returncode` falls through to the diff logic, which exits 1
            # too, and that arm would pass for the wrong reason (measured: the first version of this
            # file reported the fallout on the passing arm instead of naming the cause).
            r_fail = run_self_test(tmp, tool, 1)
            r_pass = run_self_test(tmp, tool, 0)
            check("%s --self-test reports a passing child as 0 (rc %d, want 0)" % (tool, r_pass.returncode),
                  r_pass.returncode == 0, (r_pass.stderr or r_pass.stdout).strip()[-160:])
            check("%s --self-test distinguishes a failing child (rc %d vs %d, want != and != 0)"
                  % (tool, r_fail.returncode, r_pass.returncode),
                  r_fail.returncode != 0 and r_fail.returncode != r_pass.returncode,
                  (r_fail.stderr or r_fail.stdout).strip()[-160:])

            # (3) a removed flag is an argparse error, not a silently passing row. This is what
            # the old rows could not see: they invoked the test file and never touched the flag.
            r = subprocess.run([sys.executable, os.path.join(tmp, "tools", "ci", tool + ".py"), "--nonsense"],
                               capture_output=True, text=True)
            check("%s rejects an unknown flag with rc 2 rather than 0 (rc %d)" % (tool, r.returncode),
                  r.returncode == 2, (r.stderr or "").strip()[-160:])

    print("self_test_flag: %s" % ("ok" if fails == 0 else "%d failure(s)" % fails))
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
