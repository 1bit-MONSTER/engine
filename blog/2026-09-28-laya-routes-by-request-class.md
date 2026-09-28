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
tags: laya, routing, serve

# Laya routing: ask the model what it knows

The engine can run the same model on several devices of one machine: Vulkan and HRX on the
Radeon, ROCm, ZINC, and the NPU for its own model format. Someone has to pick one for each
request. Step 4 of the port gave that job to
[Laya](https://github.com/NandhaKishorM/laya), a small decision model that answers typed
questions about a piece of text in one forward pass, with a calibrated confidence. This week we
wired it into `1bit serve`, and the first thing we learned was which question to ask it.

## Asking for a device

The first version asked Laya directly: "which device should run this request?", with the
device names as the answers. On eight varied requests (code, a greeting, a translation, a
contract summary, an explanation, arithmetic, a refactor, a haiku), it picked the same device
every time:

| Candidates | Picks |
|---|---|
| `vulkan,hrx,zinc` | `zinc` 8 of 8 |
| `npu,hrx,vulkan,zinc` | `npu` 4, `zinc` 4 |

That is not a flaw in Laya. It knows a great deal about text and nothing about this machine,
so it had no basis for the answer. We wrote an RFC to change the question, and it was accepted.

## Asking about the request

Now Laya answers a question it can answer: what kind of request is this? Code, prose (an
explanation, story or plan), a short answer, or work on a long pasted document. A separate
policy file, filled only from measured results, maps each class to a device.

Long documents turned out to be a question of size more than meaning. Laya often read a long
pasted log or contract as code, and those were also its slowest decisions, because they fill its
whole 512-token window. So any request of 1,024 characters or more is a long document without
asking the model, and Laya chooses among the other three.

We measured it on 200 hand-labelled requests, 50 per class:

| Setup | Accuracy | code | prose | short | long doc | per decision |
|---|---|---|---|---|---|---|
| **typed-decisions checkpoint, size gate** | **95.5%** | 100% | 82% | 100% | 100% | 588 ms |
| root checkpoint, size gate | 92.0% | 94% | 74% | 100% | 100% | 401 ms |
| typed-decisions, four-way question | 80.5% | 100% | 62% | 90% | 70% | 1,739 ms |
| root, four-way question | 64.0% | 98% | 30% | 96% | 32% | 1,379 ms |

The size gate took accuracy from 80.5% to 95.5% and cut the decision time by two thirds. Prose
is still the class Laya misses most, reading some of it as a short answer. The confidence is
useful: the least confident quarter of the answers is right 84% of the time, the rest 98-100%.
Below a confidence of 0.16 a request takes the default device. A test now fails if accuracy
drops under 90%.

## What the policy says today

Every class goes to Vulkan. That is a measured result, not a placeholder. Vulkan decodes
fastest on Strix Halo for every model file we have measured. The one per-class difference we
found is the drafter for speculative decoding. On Qwen3.8-27B, short replies decode faster with
the MTP head, and code much faster with DFlash2:

| Class | DFlash2 | MTP |
|---|---|---|
| code | 45.7 tok/s | 31.7 |
| prose | 28.3 | 25.8 |
| short | 17.8 | **28.2** |

But llama-server fixes the drafter when it starts. We checked whether a request can change it
or switch it off, and it cannot: `speculative.type`, `speculative.n_max` and
`speculative.p_min` in a request leave the number of drafted tokens unchanged. Sending short
replies to MTP would take a second server holding a second copy of the 27B model, about 16 GB,
to save about 0.3 s on a 16-token reply. The policy moves a class only when it has a net gain,
so that row stays on Vulkan for now.

## What it costs

Laya decides once per conversation, from its opening messages, and later turns reuse the
answer. Measured through `1bit serve --device auto`, Qwen3.8-27B with DFlash2:

| | 1-token reply, first turn | later turn | decode code / prose / short |
|---|---|---|---|
| without Laya | 498-518 ms | 713-849 ms | 50.5-50.7 / 26.7-26.9 / 19.0-19.2 |
| with `--laya` | 1,008-1,048 ms | 703-767 ms | 50.6 / 26.7 / 18.9-19.1 |

The first turn pays about half a second. After that, nothing changes. Routing is therefore
opt-in: `--device auto` still means Vulkan unless you pass `--laya`.

## Try it

```sh
scripts/fetch-laya.sh                                    # the pinned checkpoint
1bit serve -m model.gguf --device auto --laya
1bit route --laya-model ~/.local/share/1bit/laya --classify --state "Write a Rust function that sums a slice."
# code 0.44 vulkan
```

Every reply carries an `X-1bit-Route` header with the class, the confidence and the device, and
the server logs each decision. That traffic is what the next policy rows will come from, and
the details are in [docs/laya.md](../docs/laya.md).
