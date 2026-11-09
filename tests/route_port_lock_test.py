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
"""engine#1070: every test that starts a server holds the `route_ports` lock.

`ctest -j` runs the suite concurrently. A route test picks its loopback port by binding :0,
closing the socket and then letting the child bind that number; between the close and the child's
bind another route test can be handed the same port. That is how pm_route failed with EADDRINUSE on
PR #1065 and passed on a re-run of the same commit (#1070). CMakeLists.txt serialises those tests
with `RESOURCE_LOCK route_ports`.

This test keeps that list from drifting away from the tests it protects: it reads the configured
test list and its properties (`ctest -N --show-only=json-v1`), finds every test whose command runs
a tests/*.sh that binds a loopback port, and requires the lock on it. A new route test that forgets
the lock fails here by name, instead of flaking a later PR.

usage: tests/route_port_lock_test.py path/to/ctest build-dir
"""
import json
import os
import re
import subprocess
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTS_DIR = os.path.join(ROOT, "tests")

CTEST = sys.argv[1] if len(sys.argv) > 1 else "ctest"
BUILD_DIR = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "build")

# The shapes a route script binds a loopback port in: the freeport helper, the :0 bind it makes,
# and the fixed 43xxx literals print_backend/lemonade_hrx/template_pin start their servers on.
BIND_RE = re.compile(r'freeport|free_port|bind\(\s*\("127\.0\.0\.1",\s*0\)\s*\)|--port\s+43\d{3}\b')


def configured_tests(ctest, build_dir):
    out = subprocess.run([ctest, "--test-dir", build_dir, "-N", "--show-only=json-v1"],
                         capture_output=True, text=True, check=True).stdout
    return json.loads(out).get("tests", [])


def route_script(test):
    """The tests/*.sh a configured test runs, or None for a test that is not a script."""
    for arg in test.get("command", []):
        if isinstance(arg, str) and arg.endswith(".sh") \
                and os.path.dirname(os.path.abspath(arg)) == TESTS_DIR:
            return arg
    return None


def binds_a_port(script):
    with open(script, encoding="utf-8", errors="replace") as f:
        return bool(BIND_RE.search(f.read()))


def lock_of(test):
    for prop in test.get("properties", []):
        if prop.get("name") == "RESOURCE_LOCK":
            value = prop.get("value")
            return value[0] if isinstance(value, list) and value else value
    return None


class RoutePortLockTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tests = configured_tests(CTEST, BUILD_DIR)

    def census(self):
        found = {}
        for test in self.tests:
            script = route_script(test)
            if script and binds_a_port(script):
                found[test["name"]] = script
        return found

    def test_the_census_finds_the_route_tests(self):
        # A regex that stopped matching would pass the lock check vacuously; name what must be found.
        found = self.census()
        # recipes_route is no longer here on purpose: it runs `serve --port 0` since engine #1485, so it picks nothing
        # and releases nothing and the lock protects it from nothing. The names below are the ones that still choose
        # their port before launching.
        for name in ("pm_route", "print_backend", "smoke_serve"):
            self.assertIn(name, found, f"{name} starts a server; the census must find it")
        # 5 in a build with no model passed; a model-enabled build configures more (hybrid_e2e,
        # laya_route_e2e, laya_models_e2e, pm_laya_route, hrx_cpu_weight_parity and serve_e2e_<dev>), so the floor is
        # the number the CI build has, not the number a developer's build may have. It counted 15 before engine #1485
        # moved fifteen scripts to `serve --port 0`; the rest of the old
        # list was `bench_selftest` (binds through tools/bench.py, pinned by its own case below).
        self.assertGreaterEqual(len(found), 5,
                                "fewer route tests found than the suite has: " + ", ".join(sorted(found)))

    def test_indirect_server_tests_hold_the_lock_too(self):
        # tools/bench.py starts its servers with its own free_port under this test's name, so the
        # script text has no bind for the census to find; #1070 still requires the lock, and this is
        # where that is pinned rather than left to the list's comment.
        by_name = {test["name"]: test for test in self.tests}
        self.assertIn("bench_selftest", by_name, "bench_selftest is a test in every configuration")
        self.assertEqual(lock_of(by_name["bench_selftest"]), "route_ports")
        # engine #1485: the same shape one hop away, found by counting files rather than trusting the
        # list's comment. tests/hybrid_kvshare_e2e.sh runs tests/kvshare_fd_handoff.py, whose
        # free_port() binds :0, reads the number and closes before the server binds it - a pick and a
        # release, in the driver rather than in the script the census reads. Configured only when the
        # kvshare driver's inputs exist, so it is asserted only then.
        if "hybrid_kvshare_e2e" in by_name:
            self.assertEqual(lock_of(by_name["hybrid_kvshare_e2e"]), "route_ports",
                             "hybrid_kvshare_e2e picks its port in kvshare_fd_handoff.py; it needs the route lock")

    def test_every_route_test_holds_the_route_port_lock(self):
        missing = []
        for name, script in sorted(self.census().items()):
            test = next(t for t in self.tests if t["name"] == name)
            if lock_of(test) != "route_ports":
                missing.append(f"{name} ({os.path.basename(script)})")
        self.assertEqual(missing, [],
                         "these tests start a loopback server without RESOURCE_LOCK route_ports: "
                         + ", ".join(missing))


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]])
