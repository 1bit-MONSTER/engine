#!/bin/bash
# Copyright 2026 bong-water-water-bong
# SPDX-License-Identifier: Apache-2.0
#
# npu_syntax_check.sh -- compile-check the NPU lane's own translation units, which CI never parses (engine#2330).
#
# Why this exists: `app/forward_serve.cpp` is compiled **only under ONEBIT_NPU**, and CI configures without the private
# NPU runtime (`ONEBIT_NPU_RT_LIB` / `ONEBIT_NPU_RT_DIR`, docs/npu.md "The NPU runtime"). So a green CI `build` says
# nothing about that file, and on 11-13 a brace-short body in it landed on `main` while every check-run was success.
# It was found by a lane compiling the file locally -- which is the thing this script automates.
#
#   scripts/npu_syntax_check.sh --build-dir <configured ONEBIT_NPU build> [file ...]
#
# It reads that build's `compile_commands.json` -- the same flags, defines and include paths the real build uses -- and runs
# the compiler with `-fsyntax-only`: no link, no objects, no device, no runtime library, no private submodule fetch. With no
# `--build-dir` it SKIPS (77): a syntax check against guessed flags would be worse than none.
#
# Not a substitute for building: it cannot catch a missing symbol (that is a link error) or a wrong-but-legal type. It catches
# exactly the class that landed: a file that does not parse.
set -uo pipefail
build=""
files=()
while [ $# -gt 0 ]; do
    case "$1" in
        --build-dir) build=${2:-}; shift 2 ;;
        -h|--help) sed -n '2,18p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) files+=("$1"); shift ;;
    esac
done
if [ -z "$build" ]; then
    echo "SKIP: no --build-dir (a configured ONEBIT_NPU build); a syntax check against guessed flags would be worse than none"
    exit 77
fi
cc_json="$build/compile_commands.json"
if [ ! -f "$cc_json" ]; then
    echo "SKIP: $cc_json is absent (configure with -DCMAKE_EXPORT_COMPILE_COMMANDS=ON)"
    exit 77
fi

python3 - "$cc_json" "${files[@]}" <<'PY'
import json, os, shlex, subprocess, sys

cc, want = sys.argv[1], sys.argv[2:]
db = json.load(open(cc))
entries = []
for e in db:
    f = e['file']
    if 'ONEBIT_NPU' not in e.get('command', ''):
        continue                      # a unit that is not in the NPU lane: CI already compiles it
    if want and not any(f.endswith(w) for w in want):
        continue
    entries.append(e)
if not entries:
    print('SKIP: no ONEBIT_NPU translation units in that build (was it configured -DONEBIT_NPU=ON?)')
    sys.exit(77)

bad = 0
for e in sorted(entries, key=lambda x: x['file']):
    cmd = e['command']
    # -fsyntax-only, output to /dev/null: same parse, no object, no link
    cmd = cmd.replace(' -o ' + e['output'], ' -o /dev/null') if e.get('output') else cmd
    cmd = cmd.replace(' -c ', ' -fsyntax-only -c ', 1)
    r = subprocess.run(shlex.split(cmd), capture_output=True, text=True, cwd=e.get('directory'))
    rel = os.path.relpath(e['file'], e.get('directory') or '.')
    if r.returncode == 0:
        print(f'ok   {rel}')
        continue
    bad += 1
    errs = [l for l in r.stderr.splitlines() if 'error:' in l]
    print(f'FAIL {rel}: {len(errs)} error(s)')
    for l in errs[:5]:
        print('     ' + l.strip())
print(f'npu_syntax_check: {len(entries) - bad} of {len(entries)} NPU-lane unit(s) parse')
sys.exit(1 if bad else 0)
PY
