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

A move only explains a removal when the identifier that turns up elsewhere is RARE (<= 5 files at
head). `size` appears in most files of a C++ tree and the #371 guard carries it, so an unbounded
search would report that deletion as "rewritten" - measured on 60f744628, rc 0 instead of 1.

Known blind spots, named rather than implied: GUARD_KEYWORDS is a heuristic, so a refusal phrased
outside it is invisible; and SKIP_DIRS includes tests/, so a guard deleted from a test file is not
reported (test asserts are usually fixtures, not the product's refusals - a judgement, not a proof).

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


# A name that appears in more than this many files at head carries no evidence that the guard
# moved: `size` lives in most files of a C++ tree, so finding it "elsewhere" would excuse a real
# loss (engine#2429's CHANGES, 15:26Z, on the very #371 diff this tool was built from).
RARE_MAX = 5


def head_mentions(root, head, names):
    """{identifier: [paths at head that mention it]} -- engine#2436: a guard can move into a helper.

    `git grep -l -F n <rev>` prints `<rev>:<path>`, so the *path* is after the first colon and the
    rev must be matched exactly (the first version kept the rev and could not name the destination).
    A grep that fails yields no paths for that name, which keeps the caller's stricter verdict: this
    may under-report a move, never claim an identifier is gone.
    """
    out = {}
    for n in names:
        r = git("grep", "-l", "-F", n, head, cwd=root)
        paths = []
        for line in r.stdout.splitlines():
            head_rev, sep, path = line.partition(":")
            path = path if sep and head_rev == head else line
            if path and path not in paths:
                paths.append(path)
        out[n] = paths
    return out


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
            # before calling it lost -- but only a RARE identifier can answer, or `size` in any
            # other file would excuse the #371 alignment guard this tool exists to catch (15:26Z).
            mentions = head_mentions(root, head, missing)
            rare = [i for i in missing if 0 < len(mentions.get(i, [])) <= RARE_MAX]
            elsewhere = sorted({p for i in rare for p in mentions[i] if p != path})
            if elsewhere:
                moved.append((path, body, elsewhere))
            else:
                lost.append((path, body, "not in this file and not anywhere at head: " + ", ".join(sorted(missing))))
        else:
            moved.append((path, body))
    for path, body, why in lost:
        print(f"LOST {path}: {body.strip()[:100]}\n     {why}")
    for path, body, elsewhere in moved:
        dest = f" -> {' '.join(elsewhere[:3])}" if elsewhere else ""
        print(f"note {path}: guard rewritten, identifiers still present in the file or at head: "
              f"{body.strip()[:64]}{dest}")
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
    # engine#2429 CHANGES #3: nothing drove check() to the branch that returns 1. Build the case.
    import subprocess, tempfile, pathlib, io, contextlib

    def scratch(files_before, files_after):
        td = tempfile.mkdtemp()
        def g(*a):
            return subprocess.run(["git"] + list(a), cwd=td, capture_output=True, text=True)
        g("init", "-q"); g("config", "user.email", "t@t"); g("config", "user.name", "t")
        for name, text in files_before.items():
            pathlib.Path(td, name).parent.mkdir(parents=True, exist_ok=True)
            pathlib.Path(td, name).write_text(text)
        g("add", "-A"); g("commit", "-qm", "base")
        for name, text in files_after.items():
            pathlib.Path(td, name).write_text(text)
        g("add", "-A"); g("commit", "-qm", "head")
        return td

    # 1. a guard deleted outright, nothing else changed -> LOST, rc 1
    td = scratch({"src/a.cpp": "int f(void*p,size_t n){ if (p == nullptr || n % TENSOR_ALIGNMENT != 0) { return -1; } return 0; }\n"},
                 {"src/a.cpp": "int f(void*p,size_t n){ return 0; }\n"})
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc_lost = check(td, "HEAD^", "HEAD")
    chk("check() reports a deleted guard as LOST (the branch that returns 1)", rc_lost, 1)
    chk("and it names the identifiers that are gone",
        "TENSOR_ALIGNMENT" in buf.getvalue(), True)

    # 2. THE GAP (their 15:26Z): the #371 shape, where `size` appears in every file of the tree.
    #    `size` must not excuse the loss -- only a RARE identifier can answer "it moved".
    common = "struct S { size_t size; };\nint g(S s){ return (int)s.size; }\n"
    # the guard itself carries `size`, and `size` is in every other file: that is the #371 shape
    td2 = scratch({"src/a.cpp": "int f(void*p,size_t size){ if (size == 0 || reinterpret_cast<uintptr_t>(p) % TENSOR_ALIGNMENT != 0) { return -1; } return 0; }\n",
                   "src/b.cpp": common, "src/c.cpp": common, "src/d.cpp": common,
                   "src/e.cpp": common, "src/f.cpp": common, "src/g.cpp": common},
                  # the head file must not mention any of them either, or `size` never enters
                  # `missing` and the rarity rule is never consulted (this arm was vacuous once)
                  {"src/a.cpp": "int f(void){ return 0; }\n"})
    buf2 = io.StringIO()
    with contextlib.redirect_stdout(buf2):
        rc_ubiq = check(td2, "HEAD^", "HEAD")
    chk("an identifier that is in every file cannot excuse a loss", rc_ubiq, 1)
    chk("  ... and the report still names the guard", "LOST" in buf2.getvalue(), True)

    # 3. the helper move still reads as a rewrite, and the note names the destination
    td3 = scratch({"src/a.cpp": "int f(void*p,size_t n){ if (n % TENSOR_ALIGNMENT != 0) { return -1; } return 0; }\n"},
                  {"src/a.cpp": "int f(void*p,size_t n){ return open_with_retry(); }\n",
                   "src/help.h": "bool need_alignment(size_t TENSOR_ALIGNMENT);\n"})
    buf3 = io.StringIO()
    with contextlib.redirect_stdout(buf3):
        rc_move = check(td3, "HEAD^", "HEAD")
    chk("a rare identifier found in another file is a rewrite, not a loss", rc_move, 0)
    chk("and the note names where it went", "src/help.h" in buf3.getvalue(), True)

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
