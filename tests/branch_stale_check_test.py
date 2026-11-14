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
"""branch_stale_check_test.py -- the stale-branch shape, built in throwaway repositories.

A file the base advances after the branch was cut, which the branch never touched. The merge keeps the base's copy
(engine#2410), so the check names it as behind without failing; it fails on a conflict, stays quiet when the
branch does touch the file, and refuses rather than passes when a ref cannot be read.
"""

import os
import shutil
import subprocess
import sys
import tempfile
import unittest

CHECK = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools", "ci",
                     "branch_stale_check.py")


def git(cwd, *args):
    return subprocess.run(("git",) + args, cwd=cwd, capture_output=True, text=True,
                          env={**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
                               "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"})


class BranchStaleCheckTest(unittest.TestCase):
    def setUp(self):
        self.repo = tempfile.mkdtemp(prefix="stale-")
        git(self.repo, "init", "-q", "-b", "main")
        self.write("shared.txt", "v1\n")
        self.write("other.txt", "other v1\n")
        commit(self.repo, "base")
        git(self.repo, "checkout", "-q", "-b", "feature")
        self.write("feature.txt", "the branch's own work\n")
        commit(self.repo, "the branch does its own thing")
        self.feature_before_base_moves = git(self.repo, "rev-parse", "HEAD").stdout.strip()
        git(self.repo, "checkout", "-q", "main")
        self.write("shared.txt", "v2 - merged while the branch was open\n")
        commit(self.repo, "main advances a file the branch never touched")
        self.main = git(self.repo, "rev-parse", "HEAD").stdout.strip()

    def tearDown(self):
        shutil.rmtree(self.repo, ignore_errors=True)

    def write(self, rel, text):
        with open(os.path.join(self.repo, rel), "w", encoding="utf-8") as handle:
            handle.write(text)

    def run_check(self, base, head):
        return subprocess.run((sys.executable, CHECK, "--base", base, "--head", head), cwd=self.repo,
                              capture_output=True, text=True)

    def test_a_file_the_base_advanced_is_named_as_behind_not_failed(self):
        result = self.run_check(self.main, self.feature_before_base_moves)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("note branch_stale_check: shared.txt - behind the base (the merge keeps the base's copy)",
                      result.stdout)
        self.assertNotIn("FAIL", result.stdout)
        self.assertIn("rebase before trusting them", result.stdout)

    def test_a_squash_merge_keeps_the_base_copy_of_a_path_the_branch_never_changed(self):
        # engine#2410: the premise the check rests on, measured rather than asserted.
        git(self.repo, "checkout", "-q", "main")
        merged = git(self.repo, "merge", "--squash", "feature")
        self.assertEqual(merged.returncode, 0, merged.stdout + merged.stderr)
        with open(os.path.join(self.repo, "shared.txt"), encoding="utf-8") as handle:
            self.assertEqual(handle.read(), "v2 - merged while the branch was open\n")

    def test_a_conflict_with_the_base_fails(self):
        git(self.repo, "checkout", "-q", "feature")
        self.write("other.txt", "the branch's other\n")
        commit(self.repo, "the branch edits other.txt")
        head = git(self.repo, "rev-parse", "HEAD").stdout.strip()
        git(self.repo, "checkout", "-q", "main")
        self.write("other.txt", "the base's other\n")
        commit(self.repo, "the base edits other.txt too")
        base = git(self.repo, "rev-parse", "HEAD").stdout.strip()
        result = self.run_check(base, head)
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("FAIL branch_stale_check: other.txt - the merge with the base conflicts here", result.stdout)

    def test_the_branchs_own_file_is_not_reported(self):
        # `feature.txt` does not exist on the base at all: the branch adds it and that must never be a failure.
        result = self.run_check(self.main, self.feature_before_base_moves)
        self.assertNotIn("feature.txt", result.stdout)

    def test_a_branch_that_changed_the_file_itself_is_not_reported(self):
        # The branch edits the same file, so a difference there is its intent, not staleness.
        git(self.repo, "checkout", "-q", "feature")
        self.write("shared.txt", "v2 - merged while the branch was open\n")
        commit(self.repo, "the branch rebases its own copy")
        head = git(self.repo, "rev-parse", "HEAD").stdout.strip()
        result = self.run_check(self.main, head)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertNotIn("FAIL", result.stdout)

    def test_a_file_the_base_added_and_the_branch_predates_is_not_a_failure(self):
        # The branch cannot have removed a file it never saw: a merge supplies it. This is the arm that must NOT
        # fire, because a check that fails on the harmless case teaches readers to ignore it.
        git(self.repo, "checkout", "-q", "main")
        self.write("added-after-the-fork.txt", "new on the base\n")
        commit(self.repo, "the base adds a file")
        base = git(self.repo, "rev-parse", "HEAD").stdout.strip()
        result = self.run_check(base, self.feature_before_base_moves)
        self.assertIn("the base also added", result.stdout)
        self.assertNotIn("added-after-the-fork.txt - the branch carries", result.stdout)

    def test_a_dirty_tree_is_named_because_the_check_reads_commits(self):
        # It reads `git show HEAD:...`, so an uncommitted stale copy is invisible. It must say so.
        git(self.repo, "checkout", "-q", "main")
        self.write("shared.txt", "an uncommitted edit\n")
        result = self.run_check(self.main, "HEAD")
        self.assertIn("uncommitted path(s) are NOT included", result.stdout)

    def test_an_unreadable_ref_refuses_instead_of_passing(self):
        result = self.run_check(self.main, "no-such-ref")
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertIn("cannot be read - cannot tell, and that is not a pass", result.stderr)


def commit(cwd, message):
    git(cwd, "add", "-A")
    assert git(cwd, "commit", "-q", "-m", message).returncode == 0


if __name__ == "__main__":
    unittest.main()
