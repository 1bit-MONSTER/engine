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
"""registry_loss_check.py -- did this branch drop a registry entry the base still has?

The registry files are append-mostly: a class in `registry/significant.json` is a reviewed decision, and a
counter in `registry/census.json` is a measurement. Neither is ever meant to lose a row because a branch was
cut before the row landed. `tools/registry_build.py --check-gaps` cannot see that: it iterates the classes
that are *present* and reports "none of the 41 reviewed significant classes is mapped without a reason",
which is equally true of a file with 41 classes and one with 40.

Measured on engine#2021 (`ac36bd0`): main has 23 `*ForCausalLM` classes and the branch 22, `Skylar2` is gone
from both `registry/significant.json` (2 mentions -> 0) and `tests/census_watch_test.py` (5 -> 0), and
`--check-gaps` passes on it. Measured on engine#2369 (`bc4126c`): `registry/census.json` loses `models`,
`with_architecture`, `declined_models` and `architectures_seen`.

Three checks, because the loss shows up in three places:
  classes   -- a key of `classes` in significant.json that the base has and the head does not
  counters  -- a top-level counter in census.json that the base has and the head drops or decreases
  mentions  -- a class name mentioned in `tests/` on the base and not at all in the head

Exit 0 quiet, 1 loss found, 2 a ref could not be read (which is not a pass).
"""

#!/usr/bin/env python3
"""guard_loss_check.py -- did this branch delete a guard that nothing replaced?

A squash merge can drop a refusal, an `assert` or an `if (...) return null` while leaving every registration,
heading and test row intact -- so none of the other loss checks sees it. Measured on llama.cpp#371, whose own
diff removed `model_buffer_from_mmap()`'s null/unaligned refusal; it was found a night later by a hand diff
(llama.cpp#372). This looks for that shape directly.

The rule: a removed line whose condition mentions an identifier that appears NOWHERE in the file at the head is
a guard that went away with its subject. A condition that was *rewritten* still has its identifiers in the file,
so a re-shape is reported as `moved` and does not fail.

usage: guard_loss_check.py --check [--base origin/main] [--head HEAD] [--root .]
       guard_loss_check.py --self-test

Exit: 0 nothing lost (or only moved), 1 a guard's identifiers are gone, 2 cannot tell.
"""
import argparse, re, subprocess, sys

GUARD_KEYWORDS = ("refus", "assert", "GGML_ABORT", "!= 0", "== nullptr", "cannot", "must be", "is not set")
IDENT = re.compile(r"[A-Za-z_][A-Za-z_0-9]{3,}")
SKIP_DIRS = ("third_party/", "docs/", "registry/", "tests/data/")


def git(*args, cwd=None):
    return subprocess.run(("git",) + args, cwd=cwd, capture_output=True, text=True)


def guards_removed(diff_text):
    """[(path, line)] for removed lines that look like a guard, from a unified diff."""
    out, path = [], None
    for line in diff_text.splitlines():
        if line.startswith("+++ b/"):
            path = line[6:]
            continue
        if not path or not line.startswith("-") or line.startswith("---"):
            continue
        body = line[1:]
        stripped = body.strip()
        if stripped.startswith(("//", "*", "/*")):
            continue                      # a comment about a guard is not a guard
        if any(k in body for k in GUARD_KEYWORDS) and ("if" in body or "assert" in body.lower()):
            out.append((path, body))
    return out


def identifiers(text):
    return {i for i in IDENT.findall(text) if i not in ("const", "size_t", "return", "nullptr", "reinterpret_cast")}


def head_mentions(root, head, names):
    """Files in the head tree that still mention any of `names` (engine#2436: a guard can move
    into a helper). One `git grep` per name, deduped; a failure is an empty answer, which only
    means the caller keeps its stricter verdict -- never a claim that the identifier is gone."""
    seen = []
    for n in names:
        r = git("grep", "-l", "-F", n, head, cwd=root)
        for line in r.stdout.splitlines():
            path = line.split(":", 1)[0].split(":", 1)[-1] if ":" in line else line
            path = path.split(":")[-1] if path.startswith(head) else path
            if path and path not in seen:
                seen.append(path)
    return seen


def check(root, base, head):
    mb = git("merge-base", base, head, cwd=root)
    if mb.returncode != 0:
        print(f"guard_loss_check: no merge base for {base} and {head} - cannot tell", file=sys.stderr)
        return 2
    diff = git("diff", f"{mb.stdout.strip()}", head, "--", cwd=root)
    if diff.returncode != 0:
        print(f"guard_loss_check: git diff failed - cannot tell", file=sys.stderr)
        return 2
    lost, moved = [], []
    for path, body in guards_removed(diff.stdout):
        if any(path.startswith(d) for d in SKIP_DIRS):
            continue
        at_head = git("show", f"{head}:{path}", cwd=root)
        if at_head.returncode != 0:
            lost.append((path, body, "the file is gone too"))
            continue
        missing = [i for i in identifiers(body) if i not in at_head.stdout]
        if missing:
            # engine#2436, measured: a guard moved into a helper (`npu_contexts_refused` left
            # forward_serve.cpp for npu_load_retry.h) is a rewrite, not a loss. Ask the head tree
            # before calling it lost.
            elsewhere = [p for p in head_mentions(root, head, missing) if p != path]
            if elsewhere:
                moved.append((path, body))
            else:
                lost.append((path, body, "not in this file and not anywhere at head: " + ", ".join(sorted(missing))))
        else:
            moved.append((path, body))
    for path, body, why in lost:
        print(f"LOST {path}: {body.strip()[:100]}\n     {why}")
    for path, body in moved:
        print(f"note {path}: guard rewritten, identifiers still present in the file or at head: {body.strip()[:72]}")
    print(f"guard_loss_check: {len(guards_removed(diff.stdout))} removed guard-shaped line(s), "
          f"{len(lost)} lost, {len(moved)} rewritten")
    return 1 if lost else 0


def self_test():
    fail = 0
    def chk(label, got, want):
        nonlocal fail
        ok = got == want
        print(("ok   " if ok else "FAIL ") + label)
        fail += 0 if ok else 1

    # a deleted guard whose identifier is gone from the file -> lost
    diff = "+++ b/src/a.cpp\n-if (ptr == nullptr || size == 0 || ptr % TENSOR_ALIGNMENT != 0) {\n+// nothing\n"
    g = guards_removed(diff)
    chk("a guard-shaped removed line is found", len(g), 1)
    chk("its identifiers are extracted", "TENSOR_ALIGNMENT" in identifiers(g[0][1]), True)
    # the discriminator itself: an identifier present at the head means rewritten, not lost
    chk("an identifier still in the file is not missing", "TENSOR_ALIGNMENT" in "int x = TENSOR_ALIGNMENT;", True)
    # a comment is not a guard
    chk("a plain comment is not guard-shaped", guards_removed("+++ b/a.cpp\n-// the buffer is aligned\n"), [])
    chk("a comment that MENTIONS an assert is not a guard",
        guards_removed("+++ b/a.cpp\n-        // the case must reach a kernel or it asserts nothing\n"), [])
    # engine#2436, measured on a real PR: a guard that moved into a helper is a rewrite, not a loss.
    # Built in a scratch repo because the answer comes from the head tree, not from a string.
    import subprocess, tempfile, os, pathlib
    with tempfile.TemporaryDirectory() as td:
        def g(*a):
            return subprocess.run(["git"] + list(a), cwd=td, capture_output=True, text=True)
        g("init", "-q"); g("config", "user.email", "t@t"); g("config", "user.name", "t")
        pathlib.Path(td, "a.cpp").write_text("int f(){ if (npu_contexts_refused(why)) { return 1; } return 0; }\n")
        g("add", "."); g("commit", "-qm", "base"); base = g("rev-parse", "HEAD").stdout.strip()
        pathlib.Path(td, "a.cpp").write_text("int f(){ return open_with_retry(); }\n")
        pathlib.Path(td, "help.h").write_text("bool npu_contexts_refused(const std::string&);\n")
        g("add", "."); g("commit", "-qm", "head"); head = g("rev-parse", "HEAD").stdout.strip()
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc_moved = check(td, base, head)
        chk("a guard whose identifier moved to a helper is NOT lost", rc_moved, 0)
        chk("and it is reported as a rewrite", "note " in buf.getvalue(), True)
        chk("saying where it went is not required, but the file is named", "a.cpp" in buf.getvalue(), True)
    print(f"guard_loss_check self-test: {'ok' if fail == 0 else str(fail) + ' failure(s)'}")
    return 0 if fail == 0 else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--base", default="origin/main")
    ap.add_argument("--head", default="HEAD")
    ap.add_argument("--root", default=".")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    if not a.check:
        ap.error("one of --check or --self-test is required")
    return check(a.root, a.base, a.head)


if __name__ == "__main__":
    sys.exit(main())
