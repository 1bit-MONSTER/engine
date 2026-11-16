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
"""The KLD half of the HRX hardware gate (scripts/hw-gate.sh, docs/hw-gate.md).

The base logits are too big for git (hundreds of MB per model), so they stay on the gate box
under $HW_GATE_STORE/kld and tests/golden/hw-gate/manifest.json records what they are: their
sha256, a fingerprint of the model file and of the text they were computed from, the pins, and
the thresholds a run is judged by.

  hw_gate_manifest.py set    MANIFEST NAME BASE MODEL TEXT   after --record
  hw_gate_manifest.py verify MANIFEST NAME BASE MODEL TEXT   exit 1 if any of the three changed
  hw_gate_manifest.py judge  MANIFEST NAME LOG               parse llama-perplexity, exit 1 over a threshold
  hw_gate_manifest.py memcheck MODEL|- LLAMA_CPP_DIR         tools/mem_budget.py pre-check before a GPU step
  hw_gate_manifest.py placement LOG                          ggml-hrx's placement report, exit 1 on a node off HRX
"""
import hashlib
import json
import os
import re
import subprocess
import sys

# Defaults for a newly recorded model. A rerun at the recording pin gives KLD 0 (HRX is
# deterministic for a fixed graph); the thresholds leave room for a kernel that legitimately
# changes rounding and still catch a wrong kernel by orders of magnitude (docs/hw-gate.md).
DEFAULT_LIMITS = {"max_mean_kld": 0.01, "max_kld_p99": 0.2, "min_same_top_p": 95.0}


def sha256(path, limit=None):
    h = hashlib.sha256()
    left = limit
    with open(path, "rb") as f:
        while True:
            n = 1 << 20 if left is None else min(1 << 20, left)
            if n == 0:
                break
            b = f.read(n)
            if not b:
                break
            h.update(b)
            if left is not None:
                left -= len(b)
    return h.hexdigest()


def model_fingerprint(path):
    # the first 64 MiB holds the GGUF header, metadata and the first tensors; hashing 17 GB per run is too slow
    return {"file": os.path.basename(path), "size": os.path.getsize(path), "sha256_first_64MiB": sha256(path, 64 << 20)}


def pins(root):
    """The two submodule pins, or None for a submodule whose pin could not be read.

    This used to write the string "unknown" into the manifest's `recorded_at`, and NOTHING consumes that field
    (`grep -rn recorded_at` finds only this writer), so a `git ls-tree` failure produced an artifact that still
    looked like a record of what ran. `recorded_at` is a name that promises commit ids: an unreadable pin is
    recorded as **null** and said out loud, rather than as a value a reader would have to know is a placeholder.
    """
    out = {}
    for sub in ("llama.cpp", "hrx-system"):
        try:
            line = subprocess.run(["git", "-C", root, "ls-tree", "HEAD", "third_party/" + sub],
                                  capture_output=True, text=True, check=True).stdout.split()
            out[sub] = line[2]
        except (subprocess.CalledProcessError, IndexError) as exc:
            print(f"hw_gate_manifest: could not read the pin of third_party/{sub} - recording null "
                  f"({type(exc).__name__})", file=sys.stderr)
            out[sub] = None
    return out


ABOUT = 'HRX hardware gate KLD bases (docs/hw-gate.md). The .kld files live on the gate box, not in git; this file pins what they are. `recorded_at` names the tree each base was measured at: it is provenance, and it may name a commit that is no longer reachable from the current pin (engine#2643) -- `tools/hw_gate_manifest.py judge` prints how the golden\'s tree relates to the tested one on the verdict line, and says "cannot place" rather than guessing when it cannot.'   # engine#2643: one string for the template and the shipped manifest, so a reader of either learns what recorded_at can and cannot mean.

def load(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"about": ABOUT, "kld": {}}


def cmd_set(manifest, name, base, model, text):
    m = load(manifest)
    old = m["kld"].get(name, {})
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    m["kld"][name] = {
        "base_sha256": sha256(base), "base_bytes": os.path.getsize(base),
        "model": model_fingerprint(model), "text": {"file": os.path.basename(text), "sha256": sha256(text)},
        "args": "-c 512 --chunks 8 -b 512 -ub 512 --device HRX0 -ngl 99",
        "recorded_at": pins(root), "limits": old.get("limits", dict(DEFAULT_LIMITS)),
    }
    with open(manifest, "w") as f:
        json.dump(m, f, indent=1, sort_keys=True)
        f.write("\n")
    return 0


def cmd_verify(manifest, name, base, model, text):
    e = load(manifest)["kld"].get(name)
    if e is None:
        print(f"no KLD golden for {name} in {manifest}: record one with scripts/hw-gate.sh --record")
        return 1
    if not os.path.exists(base):
        print(f"base logits missing: {base} (record them at the release pin, docs/hw-gate.md)")
        return 1
    if os.path.getsize(base) != e["base_bytes"] or sha256(base) != e["base_sha256"]:
        print(f"base logits {base} do not match the manifest sha256")
        return 1
    if model_fingerprint(model) != e["model"]:
        print(f"model file {model} is not the one the base was recorded with")
        return 1
    if sha256(text) != e["text"]["sha256"]:
        print(f"text {text} is not the one the base was recorded with")
        return 1
    return 0


def stat(log, label):
    m = re.search(r"^" + re.escape(label) + r"\s*:\s*([-+0-9.eE]+)", log, re.M)
    return float(m.group(1)) if m else None


def _git(repo, *args):
    p = subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True)
    return p.returncode, p.stdout.strip()


def provenance(recorded, tested, repo):
    """Where the golden's llama.cpp commit sits relative to the tested one, for the verdict line (engine#2643).

    `recorded_at` is the only record of which tree a KLD base came from, and a verdict that compares today's run
    against a base from another tree has to say so on the line that reports it. "cannot place" is never folded into
    a relation: a clone that holds a commit without its parents (a --depth fetch) answers "no common ancestor" for
    commits that do share one, which is how #2643 first read an ancestor as a disjoint history.
    """
    short = lambda c: c[:12] if c else "null"
    head = f"golden llama.cpp {short(recorded)}, tested {short(tested)}"
    if not recorded or not tested:
        return f"{head}: cannot place (a null pin)"
    if recorded == tested:
        return f"{head}: same commit"
    for c in (recorded, tested):
        if _git(repo, "cat-file", "-e", c + "^{commit}")[0] != 0:
            return f"{head}: cannot place ({short(c)} is not in {repo})"
    if _git(repo, "merge-base", "--is-ancestor", recorded, tested)[0] == 0:
        n = _git(repo, "rev-list", "--count", f"{recorded}..{tested}")[1]
        return f"{head}: golden is {n} commits behind the tested tree"
    if _git(repo, "merge-base", "--is-ancestor", tested, recorded)[0] == 0:
        n = _git(repo, "rev-list", "--count", f"{tested}..{recorded}")[1]
        return f"{head}: golden is {n} commits AHEAD of the tested tree"
    rc, base = _git(repo, "merge-base", recorded, tested)
    if rc != 0 or not base:
        shallow = _git(repo, "rev-parse", "--is-shallow-repository")[1] or "?"
        return (f"{head}: cannot place (no common history in this clone; shallow={shallow}"
                + ("; fetch the history with `git -C <llama.cpp> fetch --unshallow`"
                   if shallow == "true" else "") + ")")
    a = _git(repo, "rev-list", "--count", f"{base}..{recorded}")[1]
    b = _git(repo, "rev-list", "--count", f"{base}..{tested}")[1]
    return f"{head}: diverged at {short(base)} (golden +{a}, tested +{b})"


def cmd_judge_exit(where, bad):
    """The judge's exit from the two things that decide it: the provenance relation and the KLD limits.

    An unplaceable base is 2 (INCOMPLETE) even when every number is inside the limits: the numbers were not
    compared against a base this clone can place, so the run could not judge the gate (engine#2643).
    """
    if "cannot place" in where:
        return 2
    return 1 if bad else 0


def cmd_judge(manifest, name, logpath):
    entry = load(manifest)["kld"].get(name, {})
    limits = entry.get("limits", DEFAULT_LIMITS)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    where = provenance((entry.get("recorded_at") or {}).get("llama.cpp"), pins(root).get("llama.cpp"),
                       os.path.join(root, "third_party", "llama.cpp"))
    with open(logpath, errors="replace") as f:
        log = f.read()
    mean = stat(log, "Mean    KLD")
    p99 = stat(log, "99.0%   KLD")
    top = stat(log, "Same top p")
    if mean is None or top is None:
        print("no KL divergence statistics in the llama-perplexity log")
        return 1
    bad = []
    if mean != mean or mean > limits["max_mean_kld"]:
        bad.append(f"mean KLD {mean:.6f} > {limits['max_mean_kld']}")
    if p99 is not None and (p99 != p99 or p99 > limits["max_kld_p99"]):
        bad.append(f"KLD p99 {p99:.5f} > {limits['max_kld_p99']}")
    if top < limits["min_same_top_p"]:
        bad.append(f"same top token {top:.2f}% < {limits['min_same_top_p']}%")
    msg = f"mean KLD {mean:.6f}, p99 {p99 if p99 is not None else float('nan'):.5f}, same top {top:.2f}%"
    print(msg + ("; " + "; ".join(bad) if bad else "") + f" [{where}]")
    return cmd_judge_exit(where, bad)


# ggml-hrx's placement observer (GGML_HRX_REPORT=2, engine #390), one line per graph:
# "placement_observer: placement: 512 nodes, 34 off HRX (MUL_MAT x 30, GET_ROWS x 4), 7 graph splits"
# engine#2450: the engine now CLASSIFIES an off-HRX node as by-design (a weight in the loader's file mapping, or an
# unaligned-matmul input with no HRX dense schedule) or unexplained, and only the second is a misplacement -- but the
# placement line printed neither count, so this row failed on nodes the engine itself documents as intended. The two
# fields are OPTIONAL here so the row keeps working against a binary that does not print them yet: present -> decide on
# `unexplained`, absent -> the old rule (any node off HRX fails). That keeps this change mergeable before the fork's.
PLACEMENT = re.compile(r"placement: (\d+) nodes, (\d+) off HRX \(([^)]*)\)"
                       r"(?:, (\d+) by design, (\d+) unexplained)?, (\d+) graph splits")
STRICT_ABORT = re.compile(r"GGML_HRX_STRICT=1: (.*)")


def cmd_placement(logpath):
    """The gate's placement check: every graph the backend scheduled while the gate ran (load,
    warmup, the kat prompts and the needle) must be all on HRX. No report at all is a failure too:
    the server was started with GGML_HRX_REPORT=2 and -lv 4, so silence means the check saw nothing."""
    with open(logpath, errors="replace") as f:
        log = f.read()
    graphs = [(int(n), int(off), ops, int(splits), (int(byd) if byd else None),
               (int(unex) if unex else None))
              for n, off, ops, byd, unex, splits in PLACEMENT.findall(log)]
    abort = STRICT_ABORT.search(log)
    if abort:
        print("GGML_HRX_STRICT refused: " + abort.group(1).strip()[:300])
        return 1
    if not graphs:
        print("no placement report in the server log (GGML_HRX_REPORT=2, -lv 4 were passed)")
        return 1
    largest = max(g[0] for g in graphs)
    # Which number decides: `unexplained` when the engine printed it, the raw off-HRX count when it did not.
    classified = any(g[5] is not None for g in graphs)
    if classified:
        # A log can hold both shapes (an older line and a newer one from the same run); an unclassified graph falls
        # back to its raw count so the comparison never mixes None with int.
        worst = max(graphs, key=lambda g: (g[5] if g[5] is not None else g[1], g[1]))
        unexplained, by_design = (worst[5] if worst[5] is not None else worst[1]), (worst[4] or 0)
    else:
        worst = max(graphs, key=lambda g: g[1])
        unexplained, by_design = worst[1], 0
    if unexplained > 0:
        bad = sum(1 for g in graphs if (g[5] if g[5] is not None else g[1]) > 0)
        print(f"{worst[1]} of {worst[0]} nodes off HRX ({worst[2]}), {worst[3]} graph splits"
              + (f", {by_design} by design, {unexplained} unexplained" if classified else "")
              + f"; {bad} of {len(graphs)} graphs had misplaced nodes")
        return 1
    if classified and by_design > 0:
        print(f"{by_design} off-HRX node(s) BY DESIGN and none unexplained, in {len(graphs)} graphs "
              f"(largest {largest} nodes); the engine classifies mapped-weight and unaligned-matmul nodes as intended")
        return 0
    print(f"all on HRX in {len(graphs)} graphs (largest {largest} nodes)")
    return 0


FAULT_ZONE_GIB = 95.0   # engine#315: faults only seen at GTT >= 95 GiB
OPS_HEADROOM_GIB = 6.0  # test-backend-ops allocates a few GiB at most


def cmd_memcheck(model, llama_dir):
    """Exit 1 when the step would take the box into the #315 zone or past MemAvailable.
    The estimator failing is not a refusal: scripts/mem-guard.sh still guards the step."""
    tool = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mem_budget.py")
    g = float(1 << 30)
    if model == "-":
        p = subprocess.run([sys.executable, tool, "box", "--json"], capture_output=True, text=True)
        try:
            led = json.loads(p.stdout)
        except ValueError:
            # Not a refusal, by this function's contract: mem-guard.sh still guards the step. But say so - a
            # pre-check that fails silently reads as "ok" in the log, and "I could not measure" is not "there is
            # 0 GiB in use" (tools/ci's own doc_section_loss/lane_syntax rule, one level up).
            print("memory pre-check: could not read tools/mem_budget.py box output - not refusing "
                  "(scripts/mem-guard.sh still guards this step)", file=sys.stderr)
            return 0
        used = led.get("gtt_used", 0) / g
        avail = led["mem_available"] / g
        if used + OPS_HEADROOM_GIB >= FAULT_ZONE_GIB or avail < 16:
            print(f"memory pre-check: GTT used {used:.1f} GiB, MemAvailable {avail:.1f} GiB: too close to "
                  f"the engine#315 zone, not now", file=sys.stderr)
            return 1
        print(f"memory pre-check: GTT used {used:.1f} GiB, MemAvailable {avail:.1f} GiB: ok", file=sys.stderr)
        return 0
    p = subprocess.run([sys.executable, tool, "model", model, "-c", "8192", "--no-fit-params",
                        "--llama-cpp", llama_dir, "--json"], capture_output=True, text=True)
    try:
        r = json.loads(p.stdout)
    except ValueError:
        print(f"memory pre-check: could not read tools/mem_budget.py model output for "
              f"{os.path.basename(model)} - not refusing (scripts/mem-guard.sh still guards this step)",
              file=sys.stderr)
        return 0
    msg = (f"memory pre-check: {os.path.basename(model)} needs {r['need'] / g:.1f} GiB, GTT after load "
           f"{r['gtt_used_after'] / g:.1f} GiB, MemAvailable {r['mem_available'] / g:.1f} GiB")
    if not r["fits"] or r["fault_zone_after"]:
        print(msg + ": not now", file=sys.stderr)
        return 1
    print(msg + ": ok", file=sys.stderr)
    return 0


def cmd_selftest():
    """Arms for provenance(), which tells four easy-to-conflate cases apart and shipped with none (engine#2643).

    Hermetic: throwaway repos in a temp dir, no network, no gate, no device. The case that matters most is the
    last one -- a clone that holds a commit WITHOUT its parents answers "no common history" for two commits that
    share one, which is exactly how #2643 first read an ancestor as a disjoint history.
    """
    import pathlib
    import tempfile
    import shutil

    def run(repo, *a):
        return subprocess.run(["git", "-C", repo, *a], capture_output=True, text=True)

    def commit(repo, name):
        (pathlib.Path(repo) / name).write_text(name)
        run(repo, "add", "-A")
        run(repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", name)
        return run(repo, "rev-parse", "HEAD").stdout.strip()

    fails = 0

    def check(got, want, what):
        nonlocal fails
        if isinstance(got, bool):
            ok = got is want
        elif isinstance(want, str):
            ok = want in got
        else:
            ok = got == want
        print(("ok   " if ok else "FAIL ") + what + ("" if ok else f"\n  want {want!r}\n  got  {got!r}"))
        if not ok:
            fails += 1

    tmp = tempfile.mkdtemp(prefix="prov-")
    try:
        repo = os.path.join(tmp, "r")
        os.makedirs(repo)
        run(repo, "init", "-q", "-b", "main")
        c1 = commit(repo, "one")
        c2 = commit(repo, "two")
        c3 = commit(repo, "three")

        check(provenance(c1, c1, repo), "same commit", "the same commit is named, not diffed")
        check(provenance(None, c3, repo), "cannot place (a null pin)", "a null recorded pin is refused")
        check(provenance(c1, None, repo), "cannot place (a null pin)", "a null tested pin is refused")
        check(provenance("0" * 40, c3, repo), "cannot place (", "an absent object is refused, not related")
        check(provenance(c1, c3, repo), "golden is 2 commits behind", "an ancestor is counted, exactly")
        # engine#2643: a base this clone cannot place is INCOMPLETE (2), not a PASS and not a FAIL -- and the
        # signal is the word the line already carries, so the two cannot drift apart.
        if "cannot place" not in provenance("0" * 40, c3, repo):
            print("FAIL an absent object did not say 'cannot place'"); fails += 1
        if cmd_judge_exit(provenance("0" * 40, c3, repo), bad=False) != 2:
            print("FAIL an unplaceable base judged 0 instead of 2 (INCOMPLETE)"); fails += 1
        if cmd_judge_exit(provenance(c3, c3, repo), bad=False) != 0:
            print("FAIL a placeable, in-limits base did not judge 0"); fails += 1
        if cmd_judge_exit(provenance(c3, c3, repo), bad=True) != 1:
            print("FAIL a placeable base outside the limits did not judge 1"); fails += 1
        check(provenance(c3, c1, repo), "golden is 2 commits AHEAD", "the reverse direction is distinct")
        behind = provenance(c1, c3, repo)
        check("behind" in behind and "AHEAD" not in behind, True, "the two directions never both appear")

        # diverged: a branch from c2 that is not on main's tip
        run(repo, "checkout", "-q", "-b", "side", c2)
        side = commit(repo, "side-one")
        run(repo, "checkout", "-q", "main")
        d = provenance(side, c3, repo)
        check(d, "diverged at", "two branches off one base are diverged, not related")
        check("golden +1, tested +1" in d, True, "the divergence names both counts")

        # the case #2643 got wrong: a SHALLOW clone holds the tip and none of its parents
        shallow = os.path.join(tmp, "shallow")
        subprocess.run(["git", "clone", "-q", "--depth", "1", "file://" + repo, shallow],
                       capture_output=True, text=True)
        got = provenance(c1, c3, shallow)
        check(got, "cannot place (", "a shallow clone cannot place an older commit")
        check("no common history" not in got, True,
              "a shallow clone does NOT answer 'no common history' -- that was the #2643 misread")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print(f"hw_gate_manifest self-test: {fails} failure(s)")
    return 1 if fails else 0


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    cmd, args = argv[1], argv[2:]
    if cmd == "set" and len(args) == 5:
        return cmd_set(*args)
    if cmd == "verify" and len(args) == 5:
        return cmd_verify(*args)
    if cmd == "memcheck" and len(args) == 2:
        return cmd_memcheck(*args)
    if cmd == "judge" and len(args) == 3:
        return cmd_judge(*args)
    if cmd == "selftest" and not args:
        return cmd_selftest()
    if cmd == "placement" and len(args) == 1:
        return cmd_placement(*args)
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
