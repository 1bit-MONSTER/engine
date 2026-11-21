#!/usr/bin/env bash
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
#
# pre-pr-checks.sh [--base REF] [--list] [--tree-only] [--base-optional] [--fetch-base]
#
# `docs/pre-pr-checks.md` (engine#2311) lists the host-side checks to run before asking for a merge; `--list`
# prints the current set, which is the only count worth trusting -- this comment used to say "six" and was
# still saying it ten checks later, because nothing re-derives a number written in prose.
# Asking a lane to type them by hand, two of which need an argument, is how one gets skipped: three
# stale-baseline incidents in two days, each a check that exists and was not run.
#
# This runs every row `--list` prints and prints ONE line per check, so the answer is per-check rather than "the paste
# looked empty". A check that cannot run says `skip` and why -- a skipped check is not a pass, and the
# summary keeps the count so a green line cannot hide one (the reason `tools/ci/tmp_path_check.py`
# prints its excluded files instead of excluding them silently).
#
# Deliberate: a NEW file must be `git add`ed first. Three of these rows read the index or the tracked
# tree, so an unadded file is invisible to them and the run reads green -- the same trap that made
# `tools/copyright.py --check` skip untracked files by design.
set -uo pipefail

# usage(): the header comment as prose, anchored on the comment block rather than a line range -- two
# scripts in this repo drifted past the end of their header and printed code to whoever asked for
# help (engine#2129).
usage() { awk 'NR>1 && /^#/ {sub(/^# ?/,""); print; next} NR>1 {exit}' "$0"; }

base=origin/main
strict=""
base_optional=""
fetch_base=""
# engine#3116: ONE place that maps a row's exit code to a verdict line, so the table can be asserted without running the
# whole list -- and so the runner cannot drift from what it promises. The two "could not look" codes are different
# events with the same meaning for the reader:
#   77  the repo's "cannot run here" (ctest SKIP_RETURN_CODE)                        -> skip
#   2   cannot tell: the tool refused to judge (shallow clone, no merge base, an      -> skip, printed as `cannot-tell`
#       unreadable ref). Four rows speak this code and #3107 made it common: on a
#       shallow tree they used to land in the FAIL branch and read as a verdict on the PR.
#   0   ok (with the submodule_pin_check vacuity case handled by the caller)         -> ok
#   *   FAIL
# A row's 2 is NOT this script's own usage error: `--base` that does not resolve exits 2 *before* the loop (see below),
# so mapping a row's code here cannot mask a caller's mistake.
classify() {  # <rc> <name> <last line> <whole output> -> prints the verdict line, sets BUCKET
    # engine#3116 review (cl-review): this used to *echo* the bucket, and the loop called it as
    # `case "$(classify ...)" in` -- which captures the verdict LINE **and** the bucket as one two-line string, so no
    # case arm matched and every row fell into `*)` = FAIL. Measured on this head: `--base origin/main` printed
    # "16 failed, 0 ok, 1 skipped". The bucket is a variable now, which cannot be captured by accident, and the arms
    # below assert it directly (the previous arms called classify() and read its last line -- they tested the function
    # and never the wiring, which is why they all stayed green while the runner failed every row).
    BUCKET=""            # set on every path (`set -u` is on: an arm that reads it before a call must not abort the run)
    case "$1" in
        77) printf 'skip  %-42s %s\n' "$2" "$3"; BUCKET=skip ;;
        2)
            # EXCEPT argparse: a wrong flag in a row here also exits 2, and a broken row must stay red. Matched on the
            # tool's own usage-error shape, which is the one thing `2` means that is NOT "cannot tell" (cl-infra's catch
            # on #3117). Everything else that exits 2 refused to judge -- shallow clone, no merge base, unreadable ref.
            case "$4" in
                *usage:*error:*) printf 'FAIL  %-42s %s\n' "$2" "$3"; BUCKET=fail ;;
                # engine#3116 review: print `cannot-tell`, not `skip`. The summary counts this bucket separately
                # ("N could not be told"), so a `skip` label left the column and the summary disagreeing -- six `skip`
                # lines above "4 skipped, 2 could not be told", which is precisely the distinction this change exists to
                # make visible. It also makes `grep '^cannot-tell'` answer "which rows could not look?" directly.
                *)               printf 'cannot-tell  %-34s %s\n' "$2" "$3"; BUCKET=cannot-tell ;;
            esac ;;
        0)  printf 'ok    %-42s %s\n' "$2" "$3"; BUCKET=ok ;;
        *)  printf 'FAIL  %-42s %s\n' "$2" "$3"; BUCKET=fail ;;
    esac
}

list_only=0
tree_only=0

# engine#3116 review (cl-review): the loop's half is a function too, because "the arms tested classify() and never the
# wiring" is exactly how a two-line return shipped: `case "$(classify …)" in` captured the verdict line *and* the bucket,
# no arm matched, and every row fell into `*)` = FAIL (measured: "16 failed, 0 ok, 1 skipped"). judge_row() decides and
# counts in one place; the arms below call it and assert the counters, not just the word.
judge_row() {  # <rc> <name> <last> <whole output> -> prints the verdict line, sets BUCKET, bumps one counter
    classify "$1" "$2" "$3" "$4"
    case "$BUCKET" in
        ok) ok=$((ok + 1)) ;;
        skip) skipped=$((skipped + 1)) ;;
        cannot-tell) untold=$((untold + 1)) ;;
        *) failed=$((failed + 1)); return 1 ;;
    esac
    return 0
}

# engine#3116: the mapping table, asserted directly. Five codes, one line each, and the two "could not look" codes are
# told apart in the bucket even though both print as a skip -- so a later edit that folds 2 back into FAIL fails here.
if [ "${1:-}" = "--self-test" ]; then
    fails=0
    # the counters judge_row() bumps (`set -u`: an arm reads them before the run's own initialisation, which is later)
    ok=0; failed=0; skipped=0; untold=0
    arm() {  # <rc> <want-bucket> <want-word-in-line>
        local got line
        # engine#3116 review: call it the way the loop does -- NO command substitution -- and assert the variable.
        # The old arms captured stdout and read the last line, which is what let a two-line return ship.
        local words tf
        # NOT `words=$(classify ...)`: that is a subshell, so BUCKET would be set in the child and be unbound here --
        # which is the same mistake one layer down from the one this fix is about. Call it in this shell, capture its
        # stdout through a file.
        tf=$(mktemp)
        classify "$1" "stub_row" "stub: something happened" "${4:-stub: something happened}" > "$tf"
        got=${BUCKET:-}
        words=$(cat "$tf"); rm -f "$tf"
        # ... and the wiring is asserted with it: judge_row() must bump the counter for this bucket, which is the half
        # the old arms never touched (they called classify() and read its stdout).
        # ... and WHICH counter moves is the assertion, not just that one did: a bucket that fell through to `*` still
        # bumps a counter, which is how the two-line capture failed every row while looking counted.
        local b_ok=$ok b_sk=$skipped b_un=$untold b_fa=$failed want
        case "$2" in ok) want=ok ;; skip) want=skipped ;; cannot-tell) want=untold ;; *) want=failed ;; esac
        judge_row "$1" "stub_row" "${words}" "${4:-stub: something happened}" >/dev/null
        if [ $((ok - b_ok)) -ne "$([ "$want" = ok ] && echo 1 || echo 0)" ] ||
           [ $((skipped - b_sk)) -ne "$([ "$want" = skipped ] && echo 1 || echo 0)" ] ||
           [ $((untold - b_un)) -ne "$([ "$want" = untold ] && echo 1 || echo 0)" ] ||
           [ $((failed - b_fa)) -ne "$([ "$want" = failed ] && echo 1 || echo 0)" ]; then
            echo "  [FAIL] rc $1: judge_row counted it as ok+$((ok-b_ok)) skipped+$((skipped-b_sk)) untold+$((untold-b_un)) failed+$((failed-b_fa)), want $want+1"
            fails=$((fails + 1))
        fi
        if [ "$got" = "$2" ] && case "$words" in *"$3"*) true ;; *) false ;; esac; then
            echo "  [PASS] rc $1 -> $got"
        else
            echo "  [FAIL] rc $1 -> '$got' ('$words'), want bucket $2 with '$3'"
            fails=$((fails + 1))
        fi
    }
    arm 0  ok          "ok    stub_row"
    arm 77 skip        "skip  stub_row"
    # ... and the expected LABEL moves with it: this arm used to pin "skip  stub_row", i.e. the defect, in the test
    # written to catch it (engine#3116 review).
    arm 2  cannot-tell "cannot-tell  stub_row"
    # ... and a row that exits 2 because its OWN invocation is wrong stays red (argparse), which is the one 2 that is
    # not "cannot tell": a broken row must not read as a skip (cl-infra's catch on #3117).
    classify 2 "stub_row" "stub.py: error: the following arguments are required: --base" "usage: stub.py [-h] --check\nstub.py: error: the following arguments are required: --base" >/dev/null
    usage_line=${BUCKET:-}
    if [ "$usage_line" = "fail" ]; then echo "  [PASS] rc 2 with an argparse usage error -> fail"; else
        echo "  [FAIL] rc 2 with an argparse usage error -> '$usage_line', want fail"; fails=$((fails + 1)); fi
    arm 1  fail        "FAIL  stub_row"
    arm 130 fail       "FAIL  stub_row"
    # engine#3131: `--tree-only` must drop EXACTLY the open-PR rows and keep the rest in the same order. Asserted by
    # running the list twice rather than by reading the arrays: the defect this guards is a filter that silently drops
    # one row too many (a gate row stops running, and the summary still says green).
    full=$(bash "$0" --list 2>/dev/null | sed 's|^python3 ||')
    tree=$(bash "$0" --list --tree-only 2>/dev/null | sed 's|^python3 ||')
    nfull=$(printf '%s\n' "$full" | grep -c .)
    ntree=$(printf '%s\n' "$tree" | grep -c .)
    if [ "$nfull" = 17 ] && [ "$ntree" = 15 ]; then
        echo "  [PASS] --list $nfull rows, --list --tree-only $ntree"
    else
        echo "  [FAIL] --list $nfull rows, --list --tree-only $ntree (want 17 / 15)"
        fails=$((fails + 1))
    fi
    if [ "$(printf '%s\n' "$full" | grep -cE 'duplicate_fixes_check|pr_branch_stale_check')" = 2 ] &&
       [ "$(printf '%s\n' "$tree" | grep -cE 'duplicate_fixes_check|pr_branch_stale_check')" = 0 ] &&
       diff <(printf '%s\n' "$full" | grep -vE 'duplicate_fixes_check|pr_branch_stale_check') <(printf '%s\n' "$tree") >/dev/null; then
        echo "  [PASS] --tree-only drops exactly the two open-PR rows and keeps the other 15 in order"
    else
        echo "  [FAIL] --tree-only did not drop exactly the two open-PR rows (or reordered the rest)"
        fails=$((fails + 1))
    fi
    echo "pre-pr-checks self-test: $((8-fails))/8 arms passed"
    [ "$fails" -eq 0 ]
    exit $?
fi

while [ $# -gt 0 ]; do
    case "$1" in
        --base) base=${2:-}; shift 2 || { usage >&2; exit 2; } ;;
        --base=*) base=${1#--base=}; shift ;;
        --list) list_only=1; shift ;;
        --tree-only) tree_only=1; shift ;;
        --base-optional) base_optional=1; shift ;;
        --fetch-base) fetch_base=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "pre-pr-checks: unknown argument '$1'" >&2; usage >&2; exit 2 ;;
    esac
done
[ -n "$base" ] || { echo "pre-pr-checks: --base needs a ref" >&2; exit 2; }

# Run from the toplevel: the rows name `tools/...` and `config/...` relative to it, so a caller
# in scripts/ got twelve `can't open file` lines and rc 1 with every row red for the wrong reason.
top=$(git rev-parse --show-toplevel 2>/dev/null) || { echo "pre-pr-checks: not inside a git work tree" >&2; exit 2; }
cd "$top" || { echo "pre-pr-checks: cannot enter $top" >&2; exit 2; }

# ORDER MATTERS (engine#3150 review): this block must run BEFORE the `touches`/`strict` decision below, because that
# decision is computed from `$base`. Placed after it, the one run that resolves `origin/main` still printed *"cannot diff
# origin/main...HEAD, so the gpu-hold --strict-warnings check is OFF for this run"* -- i.e. the answer to the arm that
# guards `scripts/gpu-hold` depended on WHEN the base arrived rather than on the diff, which is the class this suite
# exists to remove and it bites hardest on the PRs that touch gpu-hold (engine#2902's own case).
# `--fetch-base` (engine#3131): the second half of the same story. `--base-optional` reports the four merge-base rows as
# cannot-tell, which is honest and is not a measurement -- and they are the rows that caught last week's incidents. In a
# default-depth checkout the history is one `git fetch --unshallow` away, measured on this repo: +33 MB and 3.89 s took
# the suite from `8 green, 4 could not be told` to `12 green, 0 could not be told`. So the caller can ask for the fetch.
#
# It fetches; it does not judge. If the fetch fails, nothing here changes the answer -- #3106's shallow guard still
# refuses those four rows with its own sentence, and `--base-optional` still reports them as cannot-tell rather than as
# passes. `--filter=blob:none` is deliberately NOT used: measured at +15 MB but 9.8 s, i.e. six seconds spent to save
# eighteen megabytes on a runner where wall time is the scarce resource.
if [ "$list_only" = 0 ] && [ -n "$fetch_base" ] && ! git rev-parse --verify -q "$base^{commit}" >/dev/null; then
    if [ "$(git rev-parse --is-shallow-repository 2>/dev/null)" = true ]; then
        printf 'pre-pr-checks: --fetch-base: %s does not resolve in this shallow clone, fetching history so the four loss rows can measure\n' "$base" >&2
        if git fetch --unshallow --no-tags origin >/dev/null 2>&1; then
            printf 'pre-pr-checks: --fetch-base: fetched (measured on engine: +33 MB, 3.89 s)\n' >&2
        else
            printf 'pre-pr-checks: --fetch-base: the fetch failed, so the loss rows stay cannot-tell (nothing here turns that into a pass)\n' >&2
        fi
    else
        printf 'pre-pr-checks: --fetch-base: %s does not resolve and this clone is already full, so a fetch would not help\n' "$base" >&2
    fi
fi

# engine#2902: a removed diagnostic only reddens where the message IS the product. gpu-hold's queue
# messages are what every lane reads when a ticket is released unheld (a bundled PR removed exactly that
# in #2825), so a diff touching that script is checked with --strict-warnings; every other diff keeps the
# default, where a deleted debug print is not a loss. This runs AFTER `--base` is parsed and after the
# chdir to the toplevel: the first version ran before both, so `--base <ref>` was silently ignored (the
# decision was made against origin/main) and a --help paid for a git call.
if ! touches=$(git diff --name-only "$base"...HEAD 2>/dev/null); then
    echo "pre-pr-checks: cannot diff $base...HEAD, so the gpu-hold --strict-warnings check is OFF for this run" >&2
    touches=""
fi
case "$touches" in *scripts/gpu-hold*) strict=" --strict-warnings" ;; esac

# The checks, in the order docs/pre-pr-checks.md lists them: cheapest and most common first, the two
# that need a revision last.
checks=(
    "tools/copyright.py --check"
    # engine#2344: NOTICE is hand-maintained, so a dependency compiled into the binary can arrive with no
    # licence line; this is the check that caught it. Registered as a ctest row but not run here until now.
    "tools/ci/notice_covers_deps.py --check"
    # engine#2807's detector, run at handover: EIGHT same-day collisions (#2796/97/98, #2845/44/43, #2920/21,
    # #2927/29/32, #2942/43, #2919/26, #2944/45, #2949/52) each cost a review and a merge slot, and the tool
    # that finds them existed all day -- nobody ran it.
    #
    # ADVISORY, not a gate row, and the reason is the tool's own false-positive direction: it flags any two open
    # PRs naming one issue, and a legitimate HALF (a fix plus its arm, #2797/#2798) is indistinguishable from a
    # duplicate without the allow-list or the `Refs`+overlap rule that `duplicate_fixes_allow.txt` was written
    # for -- neither is on main yet. A hard row would red the case I have spent the day arguing must pass, so it
    # prints the finding for the lane to read and stays green; make it hard the day the exemption lands.
    "tools/ci/duplicate_fixes_check.py --gate"
    "tools/ci/tests_referenced_check.py"
    "tools/ci/ci_targets_check.py --check"
    "tools/ci/tmp_path_check.py --check"
    "tools/ci/conflict_marker_check.py --check"
    "tools/ci/submodule_pin_check.py --check"
    # engine#2445: the tree must CONFIGURE, not just parse as text. A rebase resolution left one line
    # ending `…_test.py)ms)` and every check above stayed green -- they read files, run Python and diff
    # refs, and none asks CMake to read CMakeLists.txt the way CI's Configure step does. That step caught
    # it in 63s; this is the same read, run here.
    "tools/ci/cmake_configure_check.py --check"
    "tools/ci/registration_loss_check.py --check --base $base"
    "tools/ci/registry_loss_check.py --base $base"
    "tools/ci/branch_stale_check.py --base $base"
    # engine#2429: a refusing/asserting condition deleted in the diff with nothing replacing it. Calibrated on 15
    # open PR heads (one spurious hit, fixed by the head-tree search) and on the real llama.cpp#371 diff; exit 2 when
    # it cannot read the refs, so "cannot tell" is never a pass.
    "tools/ci/guard_loss_check.py --check --base $base$strict"
    # engine#2416: the HIP add-on's f16 audit over $GPU_KERNELS_DIR (or third_party/gpu-kernels) at the pin; with no
    # tree it exits 77 and is listed as `skip`, never `ok`
    "tools/ci/hip_f16_audit.py --find ."
    # gpu-kernels#99 put check-kernel-ids.py --self-test inside tools/check.sh, and gpu-kernels#102 says that
    # repo has no CI -- so it ran when someone remembered. This runs it from the repo that does have CI, over the
    # same discovery as the audit above; 77 when there is no tree, so a plain checkout is a skip, not a failure.
    "tools/ci/kernel_id_selftest.py --check --find ."
    # engine#2478 review: a self-test that nothing runs rots silently. 72 tools carry one; ten were named by
    # neither this list nor CMakeLists.txt (four of them found by hand, five more only after comment lines were
    # excluded from the reference text, and the checker itself). Wired in CMake, and this row keeps the next one
    # from going quiet.
    "tools/ci/tool_selftest_wired_check.py"
    # engine#3057: `pr_branch_stale_check` shipped in #2996 with a ctest row for its SELF-TEST and nothing that ran the
    # check itself, so two PRs tonight (`#3052`, `#3055`) displayed 84 files and `+5993/-373` -- main's work, not theirs
    # -- and nothing said a word. It is NOT the same question as `branch_stale_check.py` above: that one catches the
    # two-dot READING of a stale branch (a path the branch never touched looks reverted), this one asks whether the
    # MERGE deletes lines main gained since the branch was cut. DELETIONS are the signal, not distance.
    #
    # ADVISORY, not a gate row, and the reason is scope: it scans the OPEN-PR SET, so a hard row here would block one
    # lane's pre-PR run for another lane's stale branch. The hard gate belongs where a single PR's branch is known --
    # a CI job on the PR, which needs a `--pr <n>` mode the tool does not have yet (follow-up, engine#3057). rc 2 is
    # cannot-tell (unreadable list, truncated page, unfetchable ref) and becomes a skip: a gate that cannot look must
    # not look green (engine#2932).
    "tools/ci/pr_branch_stale_check.py --advisory"
)

# The page and this list drift apart, and it has happened twice in one night: the page said the runner "runs the first six"
# after a seventh was added (its count sentence contradicting its own table three lines down), and the table's order disagreed
# with this array at three positions while THIS file's comment above claims the order is the page's. Both were found by hand, by
# a lane that happened to compare two lists. So the comparison is now part of the run: whoever edits one list and not the other
# gets a refusal naming the difference, instead of a page that describes a tool it no longer matches.
page="${PRE_PR_CHECKS_PAGE:-docs/pre-pr-checks.md}"
if [ -f "$page" ]; then
    want=$(sed -n '/^checks=(/,/^)/p' "$0" | sed -n 's/^[[:space:]]*"\([a-zA-Z0-9_\/\.]*\).*/\1/p')
    have=$(sed -n '/^| command | red means |/,/^$/p' "$page" | sed -n 's/^| `python3 \([a-zA-Z0-9_\/\.]*\).*/\1/p')
    if [ "$want" != "$have" ]; then
        echo "pre-pr-checks: the page and this runner disagree about WHICH checks run, or their order:" >&2
        diff <(printf '%s\n' "$want") <(printf '%s\n' "$have") | sed 's/^/  /' >&2
        echo "  runner: $0   page: $page   (set PRE_PR_CHECKS_PAGE to point elsewhere)" >&2
        exit 2
    fi
else
    echo "pre-pr-checks: no $page to cross-check against; the page/runner agreement is unchecked" >&2
fi

# engine#3131: two rows answer a question about the whole OPEN-PR QUEUE, not about this tree. They cannot gate a shared
# CI job: run there, one lane's duplicate or one lane's behind branch reds every author's run. Measured 2026-11-21:
# `duplicate_fixes_check --gate` was red for roughly an hour while #3106 and #3116 each had several claiming PRs, and
# every lane's pre-PR run printed `FAIL duplicate_fixes_check: one of these should close` for a pair it had nothing to do
# with. `--tree-only` (used by the CI job) drops exactly these two, with a named skip so the summary still says what did
# not run. A list, not a heuristic: a row added later must be classified on purpose.
open_pr_rows=(
    "tools/ci/duplicate_fixes_check.py --gate"
    "tools/ci/pr_branch_stale_check.py --advisory"
)
tree_only_skipped=0
if [ "$tree_only" = 1 ]; then
    kept=()
    for c in "${checks[@]}"; do
        is_open_pr=0
        for o in "${open_pr_rows[@]}"; do [ "$c" = "$o" ] && is_open_pr=1; done
        if [ "$is_open_pr" = 1 ]; then
            # the skip line belongs to a RUN, not to --list (a caller parsing `--list --tree-only` must see only the
            # rows that will run; the count of dropped rows is in the summary of a real run)
            [ "$list_only" = 0 ] && printf 'skip  %-42s --tree-only: this row reads the open-PR set, so it is not a gate on this tree\n' "${c%% *}"
            tree_only_skipped=$((tree_only_skipped + 1))
            continue
        fi
        kept+=("$c")
    done
    checks=("${kept[@]}")
fi

if [ "$list_only" = 1 ]; then
    for c in "${checks[@]}"; do echo "python3 $c"; done
    exit 0
fi

# A `--base` that does not resolve is a USAGE error, not a skip: the caller asked for a comparison that
# cannot happen, and a run that prints "all 5 check(s) green, 1 skipped" then exits 0 is the shape this
# script exists to stop (review of engine#2313). A skip is for a check that cannot run in this
# environment; an argument that cannot be resolved is the caller's to fix.
#
# `--base-optional` (engine#3131) is the one exception, and it has to be ASKED for: an automated run in a checkout that
# has no history -- `actions/checkout` at its default depth fetches ONE ref, measured here: a `--depth 1` clone of a
# single branch has `refs/remotes/origin/<branch>` and nothing else, so neither `origin/main` nor `HEAD~1` resolves --
# cannot be a usage error, because the caller did not mistype anything and there is no ref it could have passed instead.
# Those four rows are then reported as `cannot-tell` with their own line and counted in the summary, so the run still
# executes the eleven rows that read files and the index instead of refusing to run any of them.
base_missing=""
if ! git rev-parse --verify -q "$base^{commit}" >/dev/null; then
    if [ -n "$base_optional" ]; then
        base_missing=1
    else
        echo "pre-pr-checks: --base '$base' does not resolve to a commit here" >&2
        echo "  the registration check compares against it; pass a ref that exists (default: origin/main)" >&2
        echo "  (or --base-optional for an automated run in a checkout with no history)" >&2
        exit 2
    fi
fi

ok=0; failed=0; skipped=0; untold=0; detail=""
tree_note=""
[ "$tree_only_skipped" -gt 0 ] && tree_note=", $tree_only_skipped open-PR row(s) not run (--tree-only)"

# The rows that name --base in `checks` are the only ones an absent base makes unanswerable. They are counted and
# printed HERE rather than run and classified, because running them would print a tool's own cannot-tell sentence about
# a ref this script already knows is absent -- two lines saying the same thing in different words.
if [ -n "$base_missing" ]; then
    kept=()
    for c in "${checks[@]}"; do
        case "$c" in
            *" --base $base"*)
                printf 'cannot-tell  %-34s --base %s does not resolve in this checkout, so this row cannot look\n' \
                    "${c%% *}" "$base"
                untold=$((untold + 1)) ;;
            *) kept+=("$c") ;;
        esac
    done
    checks=("${kept[@]}")
fi

for c in "${checks[@]}"; do
    name=${c%% *}
    out=$(python3 $c 2>&1); rc=$?
    last=$(printf '%s\n' "$out" | grep -v '^$' | tail -1)
    if [ $rc -eq 0 ]; then
        # `submodule_pin_check --check` exits 0 having compared NOTHING when submodules are declared but
        # none is checked out -- `0 submodule(s) at their pinned commit, N not initialised` -- which would
        # print `ok` for a check that ran no comparison, the shape this script exists to stop. Matched on
        # the tool's own summary rather than re-deriving the condition, because the tool is the thing that
        # knows. Its OTHER "nothing to compare" case, `no submodules reported - cannot tell, and that is not
        # a pass`, exits 2 and is now a cannot-tell skip (engine#3116): the tool already refuses to call that
        # a pass, and this runner's job is to not call it a verdict on the PR either.
        # `--require-initialised` is where a release build turns this one into a failure.
        case "$last" in
            "submodule_pin: 0 submodule(s) at their pinned commit, "*)
                printf 'skip  %-42s nothing compared: %s\n' "$name" "${last#submodule_pin: }"
                skipped=$((skipped + 1)); continue ;;
        esac
    fi
    # engine#3116: classify() is the ONLY place a row's exit code becomes a verdict, so the table the self-test asserts
    # is the table the run uses. It used to be an `if 77 / elif 0 / else FAIL` chain here, which is why 2 fell through to
    # FAIL and a shallow clone read as a verdict on the PR.
    # engine#3116: judge_row() is the only place a row is decided and counted, and it sets BUCKET -- nothing here
    # captures stdout, which is what shipped a two-line return and failed every row.
    if ! judge_row "$rc" "$name" "$last" "$out"; then
        # The whole output, not its last line: the case that matters is a checker whose summary is generic and whose
        # detail line names the commit or file (`submodule_pin_check` prints "NOT the commit this tree pins" above a
        # summary that only counts), and the last line sends the reader to the wrong place.
        detail="$detail$name:"$'\n'"$out"$'\n'
    fi
done

echo
if [ $failed -gt 0 ]; then
    echo "pre-pr-checks: $failed failed, $ok ok, $skipped skipped, $untold could not be told${tree_note}"
    printf '%s' "$detail"
    exit 1
fi
echo "pre-pr-checks: all $ok check(s) green, $skipped skipped, $untold could not be told${tree_note}"
exit 0
