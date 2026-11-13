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
"""registry_loss_check_test.py -- the three arms, each proven on a real pair of commits.

Each arm builds a throwaway repository, so the check is exercised against git rather than against a mock, and the
failing materials are the ones it is meant to catch: a class removed from `registry/significant.json`, a
`coverage` counter removed from `registry/census.json`, and a class that a `tests/` file stops mentioning.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

CHECK = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools", "ci",
                     "registry_loss_check.py")


def git(cwd, *args):
    return subprocess.run(("git",) + args, cwd=cwd, capture_output=True, text=True,
                          env={**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
                               "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"})


def commit(cwd, message):
    git(cwd, "add", "-A")
    assert git(cwd, "commit", "-q", "-m", message).returncode == 0


class RegistryLossCheckTest(unittest.TestCase):
    def setUp(self):
        self.repo = tempfile.mkdtemp(prefix="rlc-")
        git(self.repo, "init", "-q", "-b", "main")
        os.makedirs(os.path.join(self.repo, "registry"))
        os.makedirs(os.path.join(self.repo, "tests"))
        self.write("registry/significant.json",
                   json.dumps({"classes": {"Skylar2ForCausalLM": {"model_type": "skylar2"},
                                           "OmoikaneForCausalLM": {"model_type": "omoikane"}}}, indent=1))
        self.write("registry/census.json",
                   json.dumps({"date": "2026-11-13",
                               "coverage": {"models": 437432, "with_architecture": 344820},
                               "raw": {"counts": {"Skylar2ForCausalLM": 1}}}, indent=1))
        self.write("tests/census_watch_test.py", "SKYLAR = 'Skylar2ForCausalLM'\n")
        commit(self.repo, "base")
        self.base = git(self.repo, "rev-parse", "HEAD").stdout.strip()

    def tearDown(self):
        shutil.rmtree(self.repo, ignore_errors=True)

    def write(self, rel, text):
        path = os.path.join(self.repo, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(text)

    def run_check(self, base, head):
        return subprocess.run((sys.executable, CHECK, "--base", base, "--head", head), cwd=self.repo,
                              capture_output=True, text=True)

    def test_a_class_that_the_head_lacks_fails_and_names_it(self):
        self.write("registry/significant.json", json.dumps({"classes": {"OmoikaneForCausalLM": {}}}, indent=1))
        commit(self.repo, "drop a class")
        result = self.run_check(self.base, "HEAD")
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("class 'Skylar2ForCausalLM' is on the base and not at the head", result.stdout)

    def test_a_coverage_counter_that_the_head_lacks_fails(self):
        census = json.loads(open(os.path.join(self.repo, "registry/census.json"), encoding="utf-8").read())
        del census["coverage"]["models"]
        self.write("registry/census.json", json.dumps(census, indent=1))
        commit(self.repo, "drop a counter")
        result = self.run_check(self.base, "HEAD")
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("census counter coverage/models (437432) is gone", result.stdout)

    def test_a_class_no_test_mentions_any_more_fails(self):
        self.write("registry/significant.json", json.dumps({"classes": {"OmoikaneForCausalLM": {}}}, indent=1))
        self.write("tests/census_watch_test.py", "SKYLAR = ''\n")
        commit(self.repo, "drop the class and its test reference")
        result = self.run_check(self.base, "HEAD")
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("Skylar2ForCausalLM: 1 mention(s) under tests/ on the base, 0 at the head", result.stdout)

    def test_a_census_value_that_moved_is_not_a_failure(self):
        # The census is rebuilt, so a branch carrying a different number has not lost anything.
        census = json.loads(open(os.path.join(self.repo, "registry/census.json"), encoding="utf-8").read())
        census["coverage"]["models"] = 400001
        self.write("registry/census.json", json.dumps(census, indent=1))
        commit(self.repo, "older census")
        result = self.run_check(self.base, "HEAD")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_a_ref_that_cannot_be_read_refuses_instead_of_passing(self):
        result = self.run_check(self.base, "no-such-ref")
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertIn("cannot be read - cannot tell, and that is not a pass", result.stderr)


if __name__ == "__main__":
    unittest.main()
