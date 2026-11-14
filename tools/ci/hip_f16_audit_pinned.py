#!/usr/bin/env python3
"""hip_f16_audit_pinned.py -- run the add-on f16 audit against the tree the engine pins.

`tools/ci/hip_f16_audit.py` enforces the two f16 properties `hrx_f16_narrowing.py` enforces, over the HIP add-on's
kernels, which the Loom ratchet cannot see. engine#2406 put *"audit those on a pin bump"* on the ratchet's green
line, and engine#2416 found that **nothing ran it**: the tool had a `--self-test` row and no caller, because the
engine checks out no gpu-kernels tree (submodules are hrx-system, llama.cpp, xdna-driver). So the audit was done
by hand twice on 2026-11-14, on 1,210 new `.hip` lines and on `q41_decode.hip`.

This is that caller.

    GPU_KERNELS_DIR=<gpu-kernels checkout>   (or the default third_party/gpu-kernels)

**Absent**, it exits **77** -- the convention `hrx_f16_narrowing.py` already uses for a missing `.loom` corpus --
and says in one line that the audit did NOT run. A quiet skip would be the same failure the audit exists to
prevent: a green row that reads as "checked" when nothing was checked.

**Present**, it delegates with `--config-root`, so engine#2411's pin gate applies and an audit of the wrong commit is
refused rather than reported clean; engine#2413's `--expect-sha` overrides the pin when the commit under audit is a
candidate that has not merged yet.
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

TOOL = Path(__file__).with_name("hip_f16_audit.py")
REPO = Path(__file__).resolve().parent.parent.parent
DEFAULT_DIR = REPO / "third_party" / "gpu-kernels"


def kernels_dir():
    """The gpu-kernels checkout to audit, or None if this tree has none."""
    override = os.environ.get("GPU_KERNELS_DIR")
    root = Path(override) if override else DEFAULT_DIR
    return root if root.is_dir() else None


def run(root=None, config_root=None, expect_sha=None, tool=None):
    """(exit code, stdout+stderr) for one audit. Split out so --self-test can drive both branches."""
    root = Path(root) if root else kernels_dir()
    if root is None or not Path(root).is_dir():
        shown = root if root is not None else DEFAULT_DIR
        return 77, (f"SKIPPED: no gpu-kernels checkout at {shown} -- the add-on f16 audit did NOT run "
                    f"(set GPU_KERNELS_DIR, or clone the pinned tree)\n")
    cmd = [sys.executable, str(tool or TOOL), "--dir", str(root)]
    if expect_sha:
        cmd += ["--expect-sha", expect_sha]
    elif config_root:
        cmd += ["--config-root", str(config_root)]
    done = subprocess.run(cmd, capture_output=True, text=True)
    return done.returncode, done.stdout + done.stderr


def self_test() -> int:
    problems = []

    def chk(want, got, what):
        if want != got:
            problems.append(f"{what}: want {want!r}, got {got!r}")

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        # no checkout: 77, and the message names the consequence rather than reading as a pass
        code, out = run(root=tmp / "absent")
        chk(77, code, "an absent gpu-kernels tree is a skip")
        chk(True, "did NOT run" in out, "the skip says the audit did not run")
        chk(True, "GPU_KERNELS_DIR" in out, "the skip says how to give it a tree")

        # a present checkout with a matching pin: the audit runs and reports
        kernels = tmp / "kernels" / "addons" / "hip-hrx" / "kernels"
        kernels.mkdir(parents=True)
        (kernels / "format.hip").write_text("__half h = __float2half_rn(d);\n")
        engine = tmp / "engine"
        (engine / "config").mkdir(parents=True)
        for cmd in (["git", "init", "-q"], ["git", "add", "-A"],
                    ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "--allow-empty", "-m", "x"]):
            subprocess.run(cmd, cwd=tmp / "kernels", capture_output=True)
        at = subprocess.run(["git", "rev-parse", "HEAD"], cwd=tmp / "kernels",
                            capture_output=True, text=True).stdout.strip()
        (engine / "config" / "gpu-kernels.config").write_text(f"sha={at}\n")
        code, out = run(root=tmp / "kernels", config_root=engine)
        chk(0, code, "a tree at the pinned commit audits")
        chk(True, "narrowing" in out, "and the audit reports its counts")

        # the pin gate still applies through the wrapper: a different pin is refused, not reported clean
        (engine / "config" / "gpu-kernels.config").write_text("sha=" + "f" * 40 + "\n")
        code, out = run(root=tmp / "kernels", config_root=engine)
        chk(2, code, "a tree at another commit is refused")
        chk(True, "says" in out or "pin" in out, "the refusal names the mismatch")

        # and --expect-sha overrides the pin, so a candidate commit can be audited and still asserted
        code, out = run(root=tmp / "kernels", expect_sha=at[:12])
        chk(0, code, "--expect-sha audits a candidate commit")

        # THE PATH CI TAKES: no arguments, GPU_KERNELS_DIR set, this repo's own config as the pin. The gate has to
        # apply here -- a wrapper that skips it audits any commit and reports clean, which is the bug above.
        saved = os.environ.get("GPU_KERNELS_DIR")
        os.environ["GPU_KERNELS_DIR"] = str(tmp / "kernels")
        try:
            import contextlib
            import io
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                code = main([])
            out = buf.getvalue()
        finally:
            if saved is None:
                os.environ.pop("GPU_KERNELS_DIR", None)
            else:
                os.environ["GPU_KERNELS_DIR"] = saved
        chk(2, code, "the default path audits nothing when the tree is not at this repo's pin")
        chk(True, "pin" in out or "expect-sha" in out, "and it says why")

    if problems:
        for problem in problems:
            print("FAIL " + problem)
        return 1
    print("hip_f16_audit_pinned self-test: ok")
    return 0


def main(argv) -> int:
    if "--self-test" in argv:
        return self_test()
    if any(a in ("-h", "--help") for a in argv):
        print(__doc__)
        return 0
    expect = None
    if "--expect-sha" in argv:
        expect = argv[argv.index("--expect-sha") + 1]
    # The DEFAULT path must apply the pin gate. The first version of this wrapper passed neither --expect-sha nor
    # --config-root unless the caller asked, so it audited happily at any commit -- engine#2411's gate existed and
    # was bypassed by the only path CI takes. Its self-test passed because it called run(config_root=...) directly
    # and never exercised this wiring; the case below does.
    code, out = run(expect_sha=expect, config_root=None if expect else REPO)
    sys.stdout.write(out)
    return code


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
