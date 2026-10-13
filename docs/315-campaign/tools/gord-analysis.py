#!/usr/bin/env python3
"""engine#315: read the graph-reorder A/B (arm A vs arm B) from the two consoles.
Computes the pre-registered branches: B clean while A faults (supported), B~=A (refuted),
B faults A clean (contradicted), both clean (uninformative)."""
import re, sys, collections

def read(path):
    cells = collections.defaultdict(lambda: collections.Counter())
    try:
        txt = open(path, errors='replace').read()
    except OSError:
        return None
    for line in txt.splitlines():
        m = re.search(r'arm ([qec]): (OK|FAULT|OOR|INVALID)', line)
        if m:
            cells[m.group(1)][m.group(2)] += 1
    return cells

def summary(name, path):
    c = read(path)
    if c is None:
        print(f"{name}: no console at {path}")
        return None
    e_ok, e_fa = c['e']['OK'], c['e']['FAULT']
    print(f"{name}: cell q OK={c['q']['OK']} FAULT={c['q']['FAULT']} | "
          f"cell e OK={e_ok} FAULT={e_fa} OOR={c['e']['OOR']} INVALID={c['e']['INVALID']} | "
          f"cell c OK={c['c']['OK']} FAULT={c['c']['FAULT']}")
    return (e_ok, e_fa)

A = summary("arm A (reorder ON )", "docs/315-campaign/evidence/.gord1a.txt")
B = summary("arm B (reorder OFF)", "docs/315-campaign/evidence/.gord1b.txt")
if A and B:
    a_ok, a_fa = A; b_ok, b_fa = B
    a_rate = a_fa/(a_ok+a_fa) if a_ok+a_fa else None
    b_rate = b_fa/(b_ok+b_fa) if b_ok+b_fa else None
    print()
    print(f"cell-e fault rate: A={a_rate:.0%} ({a_fa}/{a_ok+a_fa})  B={b_rate:.0%} ({b_fa}/{b_ok+b_fa})")
    if a_fa+a_ok and b_fa+b_ok:
        if b_fa == 0 and a_fa > 0:
            print("READING: B clean while A faults -> H-gord SUPPORTED (reorder is on the causal path)")
        elif b_ok == 0 and a_ok == 0:
            print("READING: both fault in every cell-e arm -> B ~= A -> H-gord REFUTED (lever closes)")
        elif b_fa > 0 and a_fa == 0:
            print("READING: B faults, A clean -> CONTRADICTED in direction; report as such")
        else:
            print("READING: mixed -> report the observed counts; do not force a branch")
    else:
        print("READING: one arm has zero scored cell-e arms -> UNINFORMATIVE")
