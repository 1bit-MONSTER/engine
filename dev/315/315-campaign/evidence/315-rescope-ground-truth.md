<!--
Copyright 2026 bong-water-water-bong
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->
<!--
Copyright 2026 bong-water-water-bong
SPDX-License-Identifier: Apache-2.0
-->

# Re-scope ground truth: the barrier-emission surface is ALREADY in scope

Written to support the objective's re-scope, because the blocked reason said the remaining surface was
"the compiler's emission of execution-barrier scopes … out of scope". That is **half right and half
wrong**, and the difference decides the re-scope.

## What is out of scope

The `loom-libs/ops/*.loom` kernel **DSL** sources and their external compiler toolchain
(`~/amd-oss/hyperloom-toolchain`, the TheRock/loom trees) are not named in the boundaries. An expedition
into the loom compiler proper remains out of scope.

## What is IN scope, and is where the §33 question actually lives

The boundaries explicitly include "the 1bit-MONSTER/llama.cpp **ggml-hrx** side". The hazard/ordering
machinery that §33 left as the one standing question is **inside ggml-hrx**, in the fork at the pin
(`e44c9d01a`):

    ggml/src/ggml-hrx/runtime/graph-record-order.cpp     <- the byte-range hazard recorder
    ggml/src/ggml-hrx/runtime/graph-record-order.h
    ggml/src/ggml-hrx/runtime/graph-program-cache.cpp    <- retains recorded programs
    ggml/src/ggml-hrx/dispatch/command-plan*.{cpp,h}     <- plan/metadata
    ggml/src/ggml-hrx/kernel-corpus/kernels/loom-libs/ops/*.loom   (59 files, in-repo)

So the question "does the recorder record every producer/consumer pair it should" is answerable **without
leaving scope**: the recorder is `RangeHazards` in `graph-record-order.cpp`.

## The concrete surface, read at the pin

`RangeHazards` is a per-buffer `std::map<offset, Segment>` where `Segment{end, last_writer, readers}`:

- `collect(buffer, offset, length, writes, deps)`: for every segment overlapping `[offset, offset+length)`,
  push `last_writer` as a dependency, and — only when the current command writes — push all `readers`;
- `update(command, buffer, offset, length, writes)`: on a write, set `last_writer = command` **and
  `readers.clear()`**; on a read, append the command to `readers` (deduplicating only the last entry);
- `count_barriers(deps, order)`: "one before a command with a dependency recorded since the last barrier".

Three properties are worth a bounded check, because each could drop a pair:

1. **Write clears readers.** After a write, prior readers are forgotten — correct for write-after-read, but
   it means a *later* reader depends only on the last writer, never on an earlier producer whose write was
   superseded in a different range. Whether a range-split interaction can lose a pair here is a bounded
   code question.
2. **`collect` and `update` are separate calls**; correctness depends on their ordering relative to barrier
   insertion. If `update` runs before `collect` for the same command, a self-dependency or a dropped pair is
   possible.
3. **`kMaxTransientExtent = 16 MiB`** (`:29`) — the file names a transient-extent cap. What happens to
   hazards for buffers/spans beyond it is unread and is a candidate for a silently-dropped dependency.

## Why this is the right re-scope target

- It is **in scope** by the objective's own words.
- It is **falsifiable and bounded**: read the recorder's call sites, establish the collect/update/barrier
  order, and determine whether a producer/consumer pair can be omitted — no GPU, no regime, no model.
- It has a **discriminating test**: if a pair can be dropped, construct the command sequence that drops it
  and show the missing barrier; if it cannot, criterion 1's cause question is answerable negatively within
  scope, which is itself a result.
- It connects to the **trigger**: a missing dependency is exactly the kind of ordering gap that a
  preemption/eviction window turns from benign overlap into a stale read.

## What this does NOT claim

It does not claim the recorder has a bug. It claims the recorder is the in-scope surface where the
remaining question lives, that its three properties above are the specific things to check, and that this
is a bounded read rather than a toolchain expedition. The blocked reason's "out of scope" conclusion was
correct about the loom DSL and wrong about ggml-hrx.
