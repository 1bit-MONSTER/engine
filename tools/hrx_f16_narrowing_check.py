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
#
# hrx_f16_narrowing_check.py -- the three ways this corpus loses a value to half precision, enumerated.
#
# Why this exists: the same defect -- a value that exceeds the f16 range becoming inf and then NaN -- was found
# five times by asking the question again by hand (engine #1539, #1962, #2127, #2186). A hand grep is what produced
# a count of 25 when the answer was 52 ("vector.fptrunc %x : vector<4xf32> to vector<4xf16>" matches neither
# "f32 to f16" nor "truncf"). NOTE, from engine#2220: `truncf` is trunc()-toward-zero on f32, NOT a narrowing --
# `unary_f32_apply.loom` has bare `truncf` lines whose target is `f32`, and a ratchet that counts them flags a file
# that cannot narrow. This tool therefore matches `fptrunc` only. The list has to be a query, and it has to cover
# all three shapes:
#
#   1. INPUT       a value loaded from an f32 view, truncated to f16, with no 2^-6 prescale
#                  (hrx_f16_prescale_check.py's original class -- engine #1962/#2127)
#   2. PUBLISH     a COMPUTED f32 value truncated to f16 (silu(gate) * up into an output plane), with no prescale
#                  -- no load in sight, so class 1 cannot see it (engine #2186 item 1)
#   3. ACCUMULATOR an f16 accumulator updated by a multiply: there is no conversion instruction at all, because the
#                  accumulator IS f16, so no truncation scan of any kind can see it (engine #2186 item 1)
#
# Reachability, so a reader can argue with the report: class 1 and 2 overflow when the truncated value exceeds the
# f16 rounding boundary, 65520 (the midpoint between the largest finite half, 65504, and the next representable
# half, 65536 -- note 65505 does NOT overflow, it rounds down to 65504). Class 3 overflows when |a*b| > 65504, i.e.
# when both factors exceed ~181 (181^2 = 32761, 256^2 = 65536).
#
# usage: hrx_f16_narrowing_check.py <corpus-root> [--self-test]
# Exit: 0 always for a report (this is an inventory, not a gate); --self-test is the gate.
import hashlib, re, sys, pathlib

NARROW = re.compile(r'fptrunc\s+(%[A-Za-z0-9_]+)|fptrunc\s+%([A-Za-z0-9_]+)')
DEF = re.compile(r'\s*(%[A-Za-z0-9_]+)\s*=\s*(.*)$')
PRESCALE = re.compile(r'0\.015625|prescale')
MULF = re.compile(r'vector\.mulf\s+(%[A-Za-z0-9_]+)\s*,\s*(%[A-Za-z0-9_]+)')
CONST = re.compile(r'\s*(%[A-Za-z0-9_]+)\s*=\s*vector\.constant\s+([0-9.eE+-]+)')


def is_prescale(v, lines, upto):
    """A value is a 2^-6 prescale if it is a vector.constant 0.015625, or a mulf by one."""
    for j in range(upto, max(-1, upto - 400), -1):
        d = DEF.match(lines[j])
        if not d or d.group(1) != v:
            continue
        rhs = d.group(2)
        if PRESCALE.search(rhs):
            return True
        m = MULF.search(rhs)
        if m:
            # both operands: whichever one is a 0.015625 constant makes this a prescale
            for other in (m.group(1), m.group(2)):
                if other == v:
                    continue
                for k in range(j, max(-1, j - 200), -1):
                    c = CONST.match(lines[k])
                    if c and c.group(1) == other and c.group(2).startswith('0.015625'):
                        return True
        return False
    return False
# Only a VECTOR load of a f32 view is an activation tile. A scalar `view.load … -> f16` is a block scale or
# sub-scale (`%d`, `%s`) -- format-bounded, never a 9.1e4 model activation -- so it is not this defect's business.
# (False positives matter here: `%d` appeared three times in the first run and each one costs a reader a look.)
LOAD = re.compile(r'(vector\.load[\w.]*)\s+"?(%[A-Za-z0-9_]+)[^\n]*vector<[0-9]+xf32>')
SCALAR_LOAD = re.compile(r'(view\.load)\s+"?(%[A-Za-z0-9_]+)[^\n]*->\s*f16')
# engine #2186 item 1: a COMPUTED f32 value truncated to f16 -- `silu(gate) * up` published to an output plane.
# The checker above only sees truncations of a *load*, so that class was invisible. Distinguished from a weight
# dequant by provenance: a dequantised weight multiplies integer codes (an uitofp/sitofp appears within the window), a
# published activation is pure float arithmetic on loaded or accumulated values.
INT_TO_FLOAT = re.compile(r'(uitofp|sitofp)')
COMPUTED = re.compile(r'vector\.(mulf|addf|subf|divf|maximumf|minimumf)\s+%')


def classify(path, lines):
    """-> list of (lineno, operand, how) ; how in {prescaled, unguarded-input, other}"""
    out = []
    for i, line in enumerate(lines):
        m = NARROW.search(line)
        if not m:
            continue
        operand = m.group(1) or ('%' + m.group(2))
        # find the definition of the truncated value, inside the enclosing function
        how = 'other'
        for j in range(i, max(-1, i - 400), -1):
            d = DEF.match(lines[j])
            if not d or d.group(1) != operand:
                continue
            rhs = d.group(2)
            if is_prescale(operand, lines, i):
                how = 'prescaled'          # the truncation operand is a 2^-6 prescale result
            elif LOAD.search(rhs):
                how = 'unguarded-input'    # truncating a loaded (activation) value directly
            elif COMPUTED.search(rhs) and not any(INT_TO_FLOAT.search(l) for l in lines[max(0, i - 40):i]):
                how = 'publish'            # a computed float value published to f16, no int->float (not a dequant)
            break
        # a prescale applied to the *operand's own source* in the lines just above also counts
        if how == 'unguarded-input' and any(PRESCALE.search(l) for l in lines[max(0, i - 12):i]):
            how = 'prescaled-nearby'
        out.append((i + 1, operand, how))
    return out



# --- class 3: f16 accumulators ---------------------------------------------------------------------------------
FRAGMENT_F16 = re.compile(r'(%[A-Za-z0-9_]+)\s*=\s*vector\.fragment<init>[^\n]*vector<(\d+)xf16>')
LOOP_F16 = re.compile(r'scf\.for[^\n]*->\s*\(vector<\d+>xf16|scf\.for[^\n]*\((%[A-Za-z0-9_]+)\s*=\s*%[A-Za-z0-9_]+\s*:\s*vector<(\d+)xf16>')
MUL = re.compile(r'vector\.mulf\s+(%[A-Za-z0-9_]+)\s*,\s*(%[A-Za-z0-9_]+)')
# The update is not always a scalar mulf: a WMMA/contract accumulation is a template.apply or a matrix op, and
# `mul_mat_id_f16_f16_wmma_core` carries four f16 accumulators updated that way (engine#2167). A mulf-only scan
# reported them as `none` even with a wide enough window.
WMMA = re.compile(r'template\.apply<@[^>]*wmma[^>]*>|vector\.(fma|contract|dot|outerproduct)\b|wmma\.(mma|load|store)')
ADD = re.compile(r'vector\.addf\s+(%[A-Za-z0-9_]+)\s*,\s*(%[A-Za-z0-9_]+)')


# How far past an accumulator's initialisation to look for its update. 120 was too short, and the failure was silent
# in the direction that matters: `mul_mat_id_f16_f16_wmma_core` carries four f16 accumulators whose update sits ~250
# lines below the `vector.fragment<init>` (a long Q4_K/Q6 loop body), so they were reported as `none` -- the bucket a
# reader skims past -- while engine#2167 is exactly about those four. Measured on that file: 120 -> ['none'] x4,
# 250 -> ['mul'] x4. The report prints this number so nobody has to trust the window blindly.
WINDOW = 250


def scan(path, lines):
    """-> list of (lineno, accname, width, kind, update_instruction)"""
    out = []
    for i, line in enumerate(lines):
        m = FRAGMENT_F16.search(line)
        if not m:
            continue
        acc, width = m.group(1), int(m.group(2))
        # the updates are the float ops that follow, until the loop closes (a bounded window is enough for a report)
        window = lines[i:i + WINDOW]
        kind, upd = 'none', ''
        for l in window:
            m = WMMA.search(l)
            if m:
                kind, upd = 'mul', m.group(0)[:44]
                break
            if MUL.search(l):
                kind, upd = 'mul', 'vector.mulf'
                break
        if kind == 'none':
            for l in window:
                if ADD.search(l):
                    kind, upd = 'add', 'vector.addf'
                    break
        # engine#2167 asked WHICH instruction updates the accumulator, and the answer is not always a mulf -- a
        # WMMA/contract accumulate is a template.apply, so reporting only "mul" hides which one to fix.
        out.append((i + 1, acc, width, kind, upd))
    return out


def self_test():
    rc = 0
    def chk(want, got, what):
        nonlocal rc
        if want == got:
            print(f"ok   {what}")
        else:
            print(f"FAIL {what}: want {want}, got {got}")
            rc = 1

    # class 1: an activation loaded from a f32 view and truncated with nothing in between
    raw = ["  %loaded = vector.load %v[%a] : view<8xf32> -> vector<4xf32>",
           "  %c = vector.fptrunc %loaded : vector<4xf32> to vector<4xf16>"]
    chk('unguarded-input', classify('x', raw)[0][2], 'class 1: a loaded activation truncated with no prescale')
    pre = ["  %k = vector.constant 0.015625 : vector<4xf32>",
           "  %p = vector.mulf %loaded, %k : vector<4xf32>",
           "  %c = vector.fptrunc %p : vector<4xf32> to vector<4xf16>"]
    chk('prescaled', classify('x', pre)[0][2], 'class 1 control: a 2^-6 prescale result is not flagged')
    near = ["  %k = vector.constant 0.015625 : vector<4xf32>",
            "  %loaded = vector.load %v[%a] : view<8xf32> -> vector<4xf32>",
            "  %c = vector.fptrunc %loaded : vector<4xf32> to vector<4xf16>"]
    chk('prescaled-nearby', classify('x', near)[0][2], 'class 1: a prescale declared just above counts')
    # a scalar block scale is not an activation (the %d false positive that cost a reader a look)
    scale = ["  %d = vector.extf %d_f16 : f16 to f32",
             "  %c = vector.fptrunc %d : vector<4xf32> to vector<4xf16>"]
    chk('other', classify('x', scale)[0][2], 'class 1 control: a scalar block scale is not an activation')
    # a dequantised weight carries an int->float conversion in its window
    w = ["  %codes = vector.uitofp %raw : vector<4xi8> to vector<4xf32>",
         "  %values_f32 = vector.mulf %centered, %s : vector<4xf32>",
         "  %c = vector.fptrunc %values_f32 : vector<4xf32> to vector<4xf16>"]
    chk('other', classify('x', w)[0][2], 'class 1 control: a dequantised weight is not an activation')
    # class 2: a computed value published to f16, pure float arithmetic, no int->float
    pub = ["  %gate = vector.mulf %g, %silu : vector<16xf32>",
           "  %product = vector.mulf %gate, %up : vector<16xf32>",
           "  %out = vector.fptrunc %product : vector<16xf32> to vector<16xf16>"]
    chk('publish', classify('x', pub)[0][2], 'class 2: a computed activation published to f16')
    # class 3: an f16 accumulator updated by a multiply, versus adds only, versus f32
    mul_acc = ["  %init = vector.fragment<init> %zero shape [%m, %n] : vector<16xf16>",
               "    %next = vector.mulf %block_acc, %k : vector<16xf16>"]
    chk('mul', scan('x', mul_acc)[0][3], 'class 3: a mul-updated f16 accumulator (the #2186 item-1 shape)')
    chk('vector.mulf', scan('x', mul_acc)[0][4], 'class 3: the update instruction is named, not just classified')
    wmma_acc = ["  %init = vector.fragment<init> %zero shape [%m, %n] : vector<8xf16>",
                "    %next = template.apply<@ggml.some_wmma_core>(%block_acc) : (vector<8xf16>) -> (vector<8xf16>)"]
    chk('mul', scan('x', wmma_acc)[0][3], 'class 3: a WMMA/contract update is a multiply (engine#2167)')
    add_acc = ["  %init = vector.fragment<init> %zero shape [%m, %n] : vector<16xf16>",
               "    %next = vector.addf %block_acc, %k : vector<16xf16>"]
    chk('add', scan('x', add_acc)[0][3], 'class 3: an add-only f16 accumulator is reported separately')
    chk(0, len(scan('x', ["  %init = vector.fragment<init> %zero shape [%m, %n] : vector<16xf32>"])),
        'class 3 control: an f32 accumulator is not this defect')
    far = ["  %init = vector.fragment<init> %zero shape [%m, %n] : vector<16xf16>"] + ["  // pad"] * 200 + \
          ["    %next = vector.mulf %block_acc, %k : vector<16xf16>"]
    chk('mul', scan('x', far)[0][3], 'class 3: an update 200 lines below the init is still found')
    # The corpus is part of the result: same bytes -> same fingerprint, a changed file -> a different one.
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        r = pathlib.Path(d)
        (r / 'a.loom').write_text('one\n'); (r / 'sub').mkdir(); (r / 'sub' / 'b.loom').write_text('two\n')
        fp1 = fingerprint(r)
        chk(True, fp1.endswith('(2 .loom file(s))'), 'the fingerprint counts the .loom files it read')
        chk(fp1, fingerprint(r), 'the same corpus fingerprints the same way twice')
        (r / 'sub' / 'b.loom').write_text('two changed\n')
        chk(False, fp1 == fingerprint(r), 'a one-byte change to one file changes the fingerprint')
        (r / 'a.loom').rename(r / 'sub' / 'a.loom')
        chk(False, fp1 == fingerprint(r), 'a file moved to another path changes it too (the path is hashed)')
    print(f"self-test: {'ok' if rc == 0 else 'FAIL'}")
    return rc


def fingerprint(root):
    """sha256 over the corpus: sorted (relative path, file digest), so a moved file changes it too."""
    h = hashlib.sha256()
    n = 0
    for p in sorted(root.rglob('*.loom')):
        h.update(str(p.relative_to(root)).encode())
        h.update(hashlib.sha256(p.read_bytes()).digest())
        n += 1
    return f"{h.hexdigest()[:16]} ({n} .loom file(s))"


def main(argv):
    if '--self-test' in argv:
        return self_test()
    if len(argv) < 2:
        print("usage: hrx_f16_narrowing_check.py <corpus-root>")
        return 2
    root = pathlib.Path(argv[1])
    # The corpus is part of the result. Two runs that disagree are only comparable if the tree they
    # read is named and fingerprinted, and "it found 9" is unattributable without both.
    inputs, pubs, accs = [], [], []
    for p in sorted(root.rglob('*.loom')):
        lines = p.read_text(errors='replace').splitlines()
        for ln, operand, how in classify(p, lines):
            if how == 'unguarded-input':
                inputs.append(f"{p.relative_to(root)}:{ln}  {operand}")
            elif how == 'publish':
                pubs.append(f"{p.relative_to(root)}:{ln}  {operand}")
        for ln, acc, width, kind, upd in scan(p, lines):
            accs.append((kind, f"{p.relative_to(root)}:{ln}  {acc}  vector<{width}xf16>  [{upd or '-'}]"))
    print(f"CORPUS       {root.resolve()}")
    print(f"FINGERPRINT  {fingerprint(root)}")
    for r in inputs:
        print(f"INPUT        {r}")
    for r in pubs:
        print(f"PUBLISH      {r}")
    for kind, r in accs:
        if kind == 'mul':
            print(f"ACCUMULATOR  {r}")
    print()
    print(f"class 1  inputs truncated without a prescale          : {len(inputs)}   (overflow above 65520)")
    print(f"class 2  computed values published to f16 unguarded   : {len(pubs)}   (overflow above 65520)")
    print(f"class 3  f16 accumulators updated by a multiply       : {sum(1 for k, _ in accs if k == 'mul')}   (overflow on |a*b| > 65504)")
    print(f"         f16 accumulators add-only / no-op-window     : {sum(1 for k, _ in accs if k == 'add')} / {sum(1 for k, _ in accs if k == 'none')}   (window: {WINDOW} lines)")
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
