# moe_placement fixtures

`flash-next-decode.placement.json` is the **real output of the shipped writer** for a real trace, kept
here so the reader is tested against something the pipeline actually produces rather than only against
tables built by hand in the test.

## What it is

Qwen3.8-Flash-Next (48 routed layers, 512 experts, top-10) decode trace, 2 device groups:

```
$ python3 tools/moe_placement.py <trace> -g 2 -e 512 -o flash-next-decode.placement.json
moe_placement: 48 layer(s), 2 group(s); one-device share 52.3% over 10560 decode steps IN-SAMPLE
               (contiguous control 0.2%); held out 18.9% over 5280 steps
```

| property | value |
|---|---|
| schema | `1bit/moe-placement/1` |
| groups | 2 |
| layers | 48, keyed by decimal layer number |
| per layer | every expert id `0..511` exactly once (48 of 48 layers are permutations) |
| `share_in_sample` | 0.523 |
| `share_held_out` | **0.1892** (above the reader's 15 % floor) |
| `held_out_steps` | 5280 |
| per-layer `unseen` | 138 of 512 experts never fire in a 220-step trace — informational; unseen experts are still assigned a group, which is why the permutation rule holds |

## Why it is a fixture and not a shipped placement

The numbers are a property of **this trace**, not of the model: a different prompt mix moves the
one-device share (the issue's table shows 44–60 % across three traces and two models). So this is a
regression fixture for the writer→reader path, not a recommendation to place a model this way.

## Regenerating

The writer is deterministic for a given trace and seed; regenerate with the command above. A new fixture
must keep `share_held_out >= 0.15` or the reader refuses it, which is the check the fixture exists to
exercise.

## See also

- `tools/moe_placement.py` — the writer (CI-registered via `add_python_test(moe_placement … --self-test)`).
- `app/moe_placement.h` — the reader, and the only authority on whether a table is usable.
- `docs/moe-streaming.md` — what consumes a table, and why the two-engine splitter is the next step.
