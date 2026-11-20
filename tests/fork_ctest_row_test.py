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
"""The five states of tools/ci/fork_ctest_known_fail.sh, on fixtures (engine#2848 review).

The row was merged with none of its paths armed: the body verified them by hand, and the two
`cannot tell` paths are what decide whether a PR is red. Three of the states are skips and one is a
refusal, so a wrapper that turned everything into a skip, or everything into a failure, would pass a
row that only checked "it ran":

    no opt-in                              -> 77   a plain `ctest` on a lane box must not start the suite
    opt-in, ratchet tool absent            -> 77   the tool lands separately (engine#2825)
    opt-in, tool present, no build tree    -> 77
    opt-in, build tree, no git in ROOT     -> 77   an exported source tree, not a stale configure
    opt-in, build tree, submodule absent   -> 2    cannot tell
    opt-in, build tree, submodule stale    -> 2    cannot tell, and the message names both shas

Every fixture stops before `ctest` runs: the skip branches and the two refusals are all upstream of
the suite, which is the property that makes this hermetic.

Herds nothing: no device, no network, no fork checkout, ~0.1 s.
"""
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile

REPO = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = REPO / "tools" / "ci" / "fork_ctest_known_fail.sh"
PIN = "d1111111111111111111111111111111111111aa"   # what ROOT's HEAD says the submodule is


def git(cwd, *args):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True,
                   env={**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
                        "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"})


_cases = 0


def fixture(tmp, *, tool=True, build_tree=True, git_root=True, submodule="absent", stale=False, real_ratchet=False, stamp=None):
    """A ROOT laid out the way the script expects to find itself, plus an HRX_BUILD."""
    global _cases
    _cases += 1
    case = tmp / f"c{_cases}"                     # one directory per fixture: ROOT and HRX_BUILD both
    root = case / "root"
    (root / "tools" / "ci").mkdir(parents=True)
    shutil.copy2(SCRIPT, root / "tools" / "ci" / SCRIPT.name)
    if real_ratchet:                                  # the tool as it ships, so the row's 0/1 is real
        for name in ("ctest_known_fail.py", "ctest_known_fail.txt"):
            shutil.copy2(REPO / "tools" / "ci" / name, root / "tools" / "ci" / name)
    elif tool:
        (root / "tools" / "ci" / "ctest_known_fail.py").write_text("# stub\n")
        (root / "tools" / "ci" / "ctest_known_fail.txt").write_text("# stub\n")
    build = case / "build"
    if build_tree:
        (build / "hrx" / "llama").mkdir(parents=True)
    if stamp is not None:
        (build / "bench-stamp.txt").write_text(f"engine=deadbeef0000\nllama_cpp={stamp}\nhost=box\n")

    if git_root:
        git(root, "init", "-q", ".")
        if submodule in ("absent", "stale"):
            git(root, "update-index", "--add", "--cacheinfo", f"160000,{PIN},third_party/llama.cpp")
            git(root, "commit", "-qm", "pin")          # the gitlink is in HEAD, as in a real tree
        if submodule == "present":
            (root / "third_party" / "llama.cpp").mkdir(parents=True)
            git(root / "third_party" / "llama.cpp", "init", "-q", ".")
            git(root / "third_party" / "llama.cpp", "commit", "-q", "--allow-empty", "-m", "fork")
        if stale:
            (root / "third_party" / "llama.cpp").mkdir(parents=True, exist_ok=True)
            git(root / "third_party" / "llama.cpp", "init", "-q", ".")
            git(root / "third_party" / "llama.cpp", "commit", "-q", "--allow-empty", "-m", "old")
    return root, build


def head_of(repo):
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, check=True,
                          capture_output=True, text=True).stdout.strip()


def stub_ctest(case, body):
    """A `ctest` that writes `body` and fails, the way a suite with known failures does."""
    bin_dir = case / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    (bin_dir / "log.txt").write_text(body)
    ctest = bin_dir / "ctest"
    ctest.write_text(f"#!/bin/sh\ncat {bin_dir / 'log.txt'}\nexit 8\n")
    ctest.chmod(0o755)
    return str(bin_dir)


def run(root, build, extra_env=None):
    env = {k: v for k, v in os.environ.items() if k not in ("ONEBIT_FORK_CTEST", "HRX_BUILD")}
    env["HRX_BUILD"] = str(build)
    env.update(extra_env or {})
    if "PATH" in env and "stub" in (extra_env or {}):
        env["PATH"] = env.pop("stub") + os.pathsep + env["PATH"]
    return subprocess.run([str(root / "tools" / "ci" / SCRIPT.name)], capture_output=True, text=True, env=env)


def main():
    fails = 0

    def ok(cond, what, detail=""):
        nonlocal fails
        print(("ok   " if cond else "FAIL ") + what + ("" if cond else f"  [{detail}]"))
        fails += 0 if cond else 1

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = pathlib.Path(tmpdir)

        # 1. No opt-in. This is the state a plain `ctest` on a lane box is in, and it is also the
        #    shebang arm: the file runs as an executable, so a `#!/usr/bin/env python3` on an sh
        #    script is a SyntaxError here rather than a silent pass.
        root, build = fixture(tmp)
        p = run(root, build)
        ok(p.returncode == 77, "a plain `ctest` skips (77) instead of starting the suite", f"rc {p.returncode}: {p.stderr[-200:]}")
        ok("ONEBIT_FORK_CTEST" in p.stdout, "the skip names the opt-in variable", p.stdout.strip())
        ok("SyntaxError" not in p.stderr and "SyntaxError" not in p.stdout,
           "the file executes under its own shebang (not through python3)", (p.stderr or "")[-200:])

        # 2. Opted in, but the ratchet tool is not on the tree (the state before engine#2825).
        root, build = fixture(tmp, tool=False)
        p = run(root, build, {"ONEBIT_FORK_CTEST": "1"})
        ok(p.returncode == 77 and "ctest_known_fail.py" in p.stdout,
           "opt-in + tool absent skips (77) and names the tool", f"rc {p.returncode}: {p.stdout.strip()}")

        # 3. Opted in, tool present, no fork build tree.
        root, build = fixture(tmp, build_tree=False)
        p = run(root, build, {"ONEBIT_FORK_CTEST": "1"})
        ok(p.returncode == 77 and "no fork build tree" in p.stdout,
           "opt-in + no build tree skips (77) and says so", f"rc {p.returncode}: {p.stdout.strip()}")

        # 4. Opted in, build tree, ROOT is not a git tree (an exported tarball). This used to print a
        #    raw `fatal: not a git repository`, then "the tree pins " with an empty sha, then advice to
        #    reconfigure a build tree -- for a source tree that has no build-tree problem.
        root, build = fixture(tmp, git_root=False)
        p = run(root, build, {"ONEBIT_FORK_CTEST": "1"})
        ok(p.returncode == 77 and "no git tree" in p.stdout,
           "opt-in + no git in ROOT skips (77) with a reason of its own", f"rc {p.returncode}: {p.stdout.strip()} {p.stderr.strip()}")
        ok("reconfigure" not in p.stderr and "reconfigure" not in p.stdout,
           "...and does not advise reconfiguring a build tree", (p.stdout + p.stderr).strip())

        # 5. Opt-in, build tree, the submodule is not checked out: cannot tell, and that is a refusal.
        root, build = fixture(tmp, submodule="absent")
        p = run(root, build, {"ONEBIT_FORK_CTEST": "1"})
        ok(p.returncode == 2 and "not checked out" in p.stderr,
           "opt-in + submodule absent refuses (2)", f"rc {p.returncode}: {p.stderr.strip()}")

        # 6. Opt-in, build tree, the submodule is at a different commit: cannot tell, and both shas
        #    are named -- an empty WANT here would read as "reconfigure" with no pin to compare to.
        root, build = fixture(tmp, submodule="absent", stale=True)
        p = run(root, build, {"ONEBIT_FORK_CTEST": "1"})
        ok(p.returncode == 2 and PIN in p.stderr and "the tree pins" in p.stderr,
           "opt-in + stale submodule refuses (2) and names the pinned sha", f"rc {p.returncode}: {p.stderr.strip()}")

        # 7-8. Opt-in, build tree, submodule AT the pin, and a stub `ctest`: the row now reaches its two real
        #    verdicts. These are the only paths that decide pass-vs-fail, and they were the least armed of all --
        #    the log shape is ctest's (FAILED block, then the summary) because that is what the ratchet parses.
        case = tmp / "real"
        case.mkdir()
        root, build = fixture(case, submodule="present", real_ratchet=True)
        at = head_of(root / "third_party" / "llama.cpp")
        (build / "bench-stamp.txt").write_text(f"engine=deadbeef0000\nllama_cpp={at[:12]}\n")
        git(root, "update-index", "--add", "--cacheinfo", f"160000,{at},third_party/llama.cpp")
        git(root, "commit", "-qm", "pin at head")      # WANT == GOT: the submodule matches the pin
        known = "test-kv-zero"
        body = ("The following tests FAILED:\n"
                f"\t 19 - {known} (Subprocess aborted)\n"
                "Errors while running CTest\n"
                "99% tests passed, 1 tests failed out of 112\n")
        stub = stub_ctest(case, body)
        p = run(root, build, {"ONEBIT_FORK_CTEST": "1", "stub": stub})
        ok(p.returncode == 0, "opt-in + matching submodule + a known failure exits 0 (the ratchet answered)",
           f"rc {p.returncode}: {p.stdout.strip()[-300:]} {p.stderr.strip()[-300:]}")

        case2 = tmp / "real-new"
        case2.mkdir()
        root2, build2 = fixture(case2, submodule="present", real_ratchet=True)
        at2 = head_of(root2 / "third_party" / "llama.cpp")
        (build2 / "bench-stamp.txt").write_text(f"engine=deadbeef0000\nllama_cpp={at2[:12]}\n")
        git(root2, "update-index", "--add", "--cacheinfo", f"160000,{at2},third_party/llama.cpp")
        git(root2, "commit", "-qm", "pin at head")
        body2 = ("The following tests FAILED:\n"
                 "\t 20 - test-brand-new (SEGV)\n"
                 "Errors while running CTest\n"
                 "99% tests passed, 1 tests failed out of 112\n")
        p = run(root2, build2, {"ONEBIT_FORK_CTEST": "1", "stub": stub_ctest(case2, body2)})
        ok(p.returncode == 1 and "test-brand-new" in p.stderr,
           "opt-in + matching submodule + a NEW failure exits 1 and names it",
           f"rc {p.returncode}: {p.stderr.strip()[-300:]}")

        # 11. THE SOURCE TREE IS NOT THE TREE THAT RUNS. ROOT's submodule matches the pin, but the build tree
        #     under HRX_BUILD is a copy that may have been configured from anything. Measured on 2026-11-20: with
        #     ROOT aligned to 8d26add2c and a different-pin copy in HRX_BUILD, the row reported 11 NEW failures and
        #     11 now-passing, none of them about the pinned source. No stamp -> cannot tell.
        case3 = tmp / "no-stamp"
        case3.mkdir()
        root3, build3 = fixture(case3, submodule="present", real_ratchet=True)
        at3 = head_of(root3 / "third_party" / "llama.cpp")
        git(root3, "update-index", "--add", "--cacheinfo", f"160000,{at3},third_party/llama.cpp")
        git(root3, "commit", "-qm", "pin at head")
        p = run(root3, build3, {"ONEBIT_FORK_CTEST": "1", "stub": stub_ctest(case3, body)})
        ok(p.returncode == 2 and "bench-stamp.txt" in p.stderr,
           "a build tree with no stamp is 'cannot tell' (2), not a comparison",
           f"rc {p.returncode}: {p.stderr.strip()[-220:]}")

        # 12. ...and a stamp that names a DIFFERENT fork is refused by name, with both values printed.
        case4 = tmp / "wrong-stamp"
        case4.mkdir()
        root4, build4 = fixture(case4, submodule="present", real_ratchet=True, stamp="0123456789ab")
        at4 = head_of(root4 / "third_party" / "llama.cpp")
        git(root4, "update-index", "--add", "--cacheinfo", f"160000,{at4},third_party/llama.cpp")
        git(root4, "commit", "-qm", "pin at head")
        p = run(root4, build4, {"ONEBIT_FORK_CTEST": "1", "stub": stub_ctest(case4, body)})
        ok(p.returncode == 2 and "0123456789ab" in p.stderr and at4[:12] in p.stderr,
           "a stamp naming another fork is refused, and the message names both revisions",
           f"rc {p.returncode}: {p.stderr.strip()[-240:]}")

    print("fork_ctest_row: ok" if not fails else f"fork_ctest_row: {fails} failure(s)")
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
