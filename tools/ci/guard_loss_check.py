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

Diagnostics are reported, not gated (engine#2761). A removed line that reports rather than refuses
(`fprintf(stderr, ...)`, `log "::warning::..."`, `LOG(WARNING)`) is a `WARN-LOST` line in the output
and counts in the summary, but it does not redden a default run: most deleted diagnostics are
deliberate debug prints, and a check that fails those is a check lanes learn to skip. `--strict-warnings`
turns them into rc 1 for the PR class where the message IS the product (gpu-hold's queue messages,
which is how #2825's bundled removal read). Measured before this family existed: 0 of 3 removed
diagnostic lines were even candidates.

Known blind spots, named rather than implied: GUARD_KEYWORDS and WARN_KEYWORDS are heuristics, so a
guard or a diagnostic phrased outside them is invisible; SKIP_DIRS includes tests/, so a guard deleted
from a test file is not reported (test asserts are usually fixtures, not the product's refusals - a
judgement, not a proof); and a branch that predates a fix on main reverts it at merge without any diff
here ever containing the line, which is a merge-tree shape this tool cannot see by construction.

Exit: 0 nothing lost (or only moved), 1 a guard's identifiers are gone, 2 cannot tell.
"""
import argparse, re, subprocess, sys

GUARD_KEYWORDS = ("refus", "assert", "GGML_ABORT", "!= 0", "== nullptr", "cannot", "must be", "is not set",
                  "throw ")   # engine#2946: `throw std::runtime_error(...)` is this tree's refusal idiom
# engine#2761: a *diagnostic* is not a refusal, but losing one is not nothing either. Measured on
# #2825, where a bundled hunk removed the gpu-hold wait-time re-check -- two lines (the `if` and the
# `log "::warning::..."` it guarded) -- and this tool found **0 of 3** removed diagnostic lines,
# because a removal only becomes a candidate when it carries a refusal keyword. A warning that goes
# away is a lane running without the sentence that told it what it was about to do.
WARN_KEYWORDS = ("::warning::", "fprintf(stderr", "stderr", "warn(", "LOG(WARNING", "warning:")
IDENT = re.compile(r"[A-Za-z_][A-Za-z_0-9]{3,}")
# engine#2451: `tests/`, not only `tests/data/`. A test's `REQUIRE`/`CHECK` is a fixture, not the
# product's refusal - and the heuristic cannot see that style anyway (REQUIRE is not a guard keyword),
# so skipping tests/ loses nothing it could catch while removing the case that would redden a PR for
# rewriting an assertion on purpose (measured: 8 of 72 LOST hits over 55 fork commits were in tests/,
# every one a deliberate assertion rewrite; the engine, where the row runs, was 0 over 11 heads).
SKIP_DIRS = ("third_party/", "docs/", "registry/", "tests/")


def git(*args, cwd=None):
    return subprocess.run(("git",) + args, cwd=cwd, capture_output=True, text=True)


def removed_guards(diff_text):
    """[(path, line, kind)] for removed guard-shaped lines, kind in {"refusal", "warning"}."""
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
        # engine#2946: the `if`/`assert` half used to gate the keyword half, so adding "throw " alone would have
        # changed nothing -- a route-level refusal is a bare `throw std::runtime_error(...)` with no `if` on the line.
        # A throw terminates the path by itself, so it satisfies the gate. A rethrow (`throw e;`) is counted too; the
        # cost is 0 removed `throw ` lines across the last 30 commits touching app/ and npu/, and a rethrow inside a
        # catch block is the only shape that noise could take.
        if any(k in body for k in GUARD_KEYWORDS) and (
                "throw " in body or "if" in body or "assert" in body.lower()):
            out.append((path, body, "refusal"))
        elif any(k in body for k in WARN_KEYWORDS):
            # not a refusal (nothing is decided by it), but a diagnostic that left the file
            out.append((path, body, "warning"))
    return out


def guards_removed(diff_text):
    """[(path, line)] for the refusal-shaped removals -- the ones that gate a run."""
    return [(p, b) for p, b, kind in removed_guards(diff_text) if kind == "refusal"]


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


def check(root, base, head, strict_warnings=False):
    mb = git("merge-base", base, head, cwd=root)
    if mb.returncode != 0:
        print(f"guard_loss_check: no merge base for {base} and {head} - cannot tell", file=sys.stderr)
        return 2
    diff = git("diff", f"{mb.stdout.strip()}", head, "--", cwd=root)
    if diff.returncode != 0:
        print(f"guard_loss_check: git diff failed - cannot tell", file=sys.stderr)
        return 2
    lost, moved, warn_lost, warn_moved = [], [], [], []
    for path, body, kind in removed_guards(diff.stdout):
        if any(path.startswith(d) for d in SKIP_DIRS):
            continue
        sink_lost, sink_moved = (lost, moved) if kind == "refusal" else (warn_lost, warn_moved)
        at_head = git("show", f"{head}:{path}", cwd=root)
        if at_head.returncode != 0:
            sink_lost.append((path, body, "the file is gone too"))
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
                sink_moved.append((path, body, elsewhere))
            else:
                sink_lost.append((path, body, "not in this file and not anywhere at head: " + ", ".join(sorted(missing))))
        else:
            # every `moved` entry is a 3-tuple: the print loop unpacks a destination list. This path
            # (guard rewritten in place, identifiers still present) appended 2-tuples and crashed the
            # tool - and Python's traceback exits 1, which a runner reads as "guard LOST". Found by
            # running the row below against a diff that rewrites a guard.
            sink_moved.append((path, body, []))
    for path, body, why in lost:
        print(f"LOST {path}: {body.strip()[:100]}\n     {why}")
    for path, body, why in warn_lost:
        print(f"WARN-LOST {path}: {body.strip()[:100]}\n     {why}")
    for path, body, elsewhere in moved + warn_moved:
        dest = f" -> {' '.join(elsewhere[:3])}" if elsewhere else ""
        print(f"note {path}: guard rewritten, identifiers still present in the file or at head: "
              f"{body.strip()[:64]}{dest}")
    # A refusal's removal changes what the program does; a diagnostic's removal changes what a lane
    # can see it doing. Both are reported, but only the first reddens a default run: 60 % of the
    # diagnostic lines deleted in this repo's history are deliberate (debug prints, a message that
    # moved), and a check that fails those is a check lanes learn to skip. `--strict-warnings`
    # exists for the PR class where a warning IS the product -- gpu-hold's queue messages.
    total = len(removed_guards(diff.stdout))
    print(f"guard_loss_check: {total} removed guard-shaped line(s), "
          f"{len(lost)} lost, {len(moved)} rewritten, {len(warn_lost)} diagnostic(s) lost, "
          f"{len(warn_moved)} diagnostic(s) rewritten")
    return 1 if (lost or (strict_warnings and warn_lost)) else 0


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
    # engine#2946: `throw std::runtime_error(...)` is this tree's refusal idiom in app/ and npu/, and it carries no
    # `if` on the line -- so the keyword alone was not enough, the gate had to let a throw through as well.
    thrown = guards_removed('+++ b/app/forward_serve.cpp\n-            throw std::runtime_error("logprobs: '
                            '/v1/chat/completions only");\n')
    chk("a removed throw is a lost refusal", len(thrown), 1)
    chk("and it keeps its own line", "logprobs" in thrown[0][1], True)
    chk("an identifier that merely starts with throw is not a refusal",
        guards_removed("+++ b/a.cpp\n-int thrower_count = 0;\n"), [])
    chk("a comment about a throw is still not a guard",
        guards_removed('+++ b/a.cpp\n-        // throw std::runtime_error("no");\n'), [])
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

    # 0. THE CRASH ARM: a guard REWRITTEN in place, identifiers still present -> note, rc 0, no unpack
    #    error. This path appended a 2-tuple into a list the printer unpacks as 3, so the tool raised
    #    ValueError and exited 1: a crash wearing the LOST exit code.
    td0 = scratch({"src/a.cpp": "int f(void*p,size_t n){ if (n % TENSOR_ALIGNMENT != 0) { return -1; } return 0; }\n"},
                  {"src/a.cpp": "int f(void*p,size_t TENSOR_ALIGNMENT){ if (TENSOR_ALIGNMENT == 0) { return -2; } return 0; }\n"})
    buf0 = io.StringIO()
    with contextlib.redirect_stdout(buf0):
        rc_rw = check(td0, "HEAD^", "HEAD")
    chk("a guard rewritten with its identifiers present is a note, not a crash", rc_rw, 0)
    chk("  ... and it is reported as rewritten", "rewritten" in buf0.getvalue(), True)

    # 0b. THE SKIP ARM (engine#2451): a guard deleted under tests/ is not reported, and the same guard
    #     under src/ is - the docstring and SKIP_DIRS must say the same thing, and the skip must not be
    #     so wide that it silences product code.
    guard = "int f(void*p,size_t n){ if (p == nullptr || n % TENSOR_ALIGNMENT != 0) { return -1; } return 0; }\n"
    gone  = "int f(void*p,size_t n){ return 0; }\n"
    td_t = scratch({"tests/a.cpp": guard}, {"tests/a.cpp": gone})
    buf_t = io.StringIO()
    with contextlib.redirect_stdout(buf_t):
        rc_test = check(td_t, "HEAD^", "HEAD")
    chk("a guard deleted under tests/ is skipped (fixtures are not refusals)", rc_test, 0)
    td_s = scratch({"src/a.cpp": guard}, {"src/a.cpp": gone})
    buf_s = io.StringIO()
    with contextlib.redirect_stdout(buf_s):
        rc_src = check(td_s, "HEAD^", "HEAD")
    chk("the same guard under src/ is still reported", rc_src, 1)

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

    # 4. THE DIAGNOSTIC FAMILY (engine#2761 / #2825): a removed warning is reported but does not gate a
    #    default run, and --strict-warnings turns it into rc 1 for the PR class where the message IS the
    #    product. The corpus is the real shape: #2825's bundled hunk removed gpu-hold's wait-time re-check
    #    -- `log "::warning::queued while ..."` -- and before this family existed the tool found 0 of the
    #    3 removed diagnostic lines, because a removal only became a candidate with a refusal keyword.
    chk("a removed ::warning:: line is a candidate at all",
        len(removed_guards("+++ b/scripts/gpu-hold\n-    log \"::warning::queued while a fault is in the log\"\n")), 1)
    chk("and it is classified as a warning, not a refusal",
        [k for _p, _b, k in removed_guards("+++ b/scripts/gpu-hold\n-    log \"::warning::x\"\n")], ["warning"])
    chk("the refusal contract is unchanged: guards_removed() still returns only refusals",
        guards_removed("+++ b/s.sh\n-    log \"::warning::x\"\n"), [])
    chk("a comment about a warning is not a removed warning",
        removed_guards("+++ b/a.cpp\n-// stderr is not used here\n"), [])
    # a line that BOTH refuses and reports is a refusal: removing it changes what the program does, so
    # the refusal branch wins (measured on this arm's first draft, whose `== nullptr` made it rc 1)
    chk("a line that refuses AND reports is a refusal, not a warning",
        [k for _p, _b, k in removed_guards("+++ b/a.cpp\n-  if (p == nullptr) { fprintf(stderr, \"warning: null\"); }\n")], ["refusal"])
    warn_guard = "int f(void*p,size_t NEEDS_WARN){ if (NEEDS_WARN == 0x2) { fprintf(stderr, \"warning: null\"); } return 0; }\n"
    td4 = scratch({"src/a.cpp": warn_guard}, {"src/a.cpp": "int f(void*p,size_t NEEDS_WARN){ return 0; }\n"})
    buf4 = io.StringIO()
    with contextlib.redirect_stdout(buf4):
        rc_warn = check(td4, "HEAD^", "HEAD")
    chk("a removed diagnostic whose identifiers are gone does NOT red a default run", rc_warn, 0)
    chk("  ... but it is in the output and named",
        "WARN-LOST" in buf4.getvalue() and "src/a.cpp" in buf4.getvalue(), True)
    buf4b = io.StringIO()
    with contextlib.redirect_stdout(buf4b):
        rc_warn_strict = check(td4, "HEAD^", "HEAD", strict_warnings=True)
    chk("--strict-warnings makes the same removal rc 1", rc_warn_strict, 1)
    # and a refusal in the same tree is still rc 1 by default: adding the family must not have moved the gate
    td5 = scratch({"src/a.cpp": "int f(void*p,size_t n){ if (n % TENSOR_ALIGNMENT != 0) { return -1; } return 0; }\n"},
                  {"src/a.cpp": "int f(void*p,size_t n){ return 0; }\n"})
    buf5 = io.StringIO()
    with contextlib.redirect_stdout(buf5):
        rc_refusal = check(td5, "HEAD^", "HEAD")
    chk("a refusal alone is still rc 1 without the new flag", rc_refusal, 1)

    print(f"guard_loss_check self-test: {'ok' if fail == 0 else str(fail) + ' failure(s)'}")
    return 0 if fail == 0 else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--base", default="origin/main")
    ap.add_argument("--head", default="HEAD")
    ap.add_argument("--root", default=".")
    ap.add_argument("--strict-warnings", action="store_true",
                    help="rc 1 when a removed diagnostic's identifiers are gone too (the gpu-hold "
                         "queue-message class, engine#2761/#2825)")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    if not a.check:
        ap.error("one of --check or --self-test is required")
    return check(a.root, a.base, a.head, a.strict_warnings)


if __name__ == "__main__":
    sys.exit(main())
