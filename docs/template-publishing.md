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
# Publishing a model's chat template

A GGUF's `tokenizer.chat_template` is a Jinja2 program that the llama.cpp-family servers run on
every request. It can rewrite the system context, add an instruction, gate behaviour on a word in the
user's message or hand an agent a URL — with no change to a single weight, and with nothing in the
usual hub or model-card checks looking at it. Pillar Security's template-backdoor work and the
"inference-time backdoors via chat templates" paper both go through that seam; the paper's own
demonstration drops an accuracy from 90 % to 15 % from a template alone. So on the publishing side a
template is not decoration next to the weights, it is part of the model, and it gets the same
treatment as a weight file: named, hashed, listed, and checked before the model is served.

## The rule

1. **Model card**: every model card we publish carries the template's sha256 and where it came from
   (the source model's `tokenizer_config.json` / `chat_template.jinja`, or its GGUF). A template we
   wrote ourselves says so, and says why.
2. **`SHA256SUMS` covers it**: next to the per-file lines, the template's own digest, so a checker can
   verify the template without parsing a GGUF.
3. **`UPLOAD-READY.md` gains the row**: no publish without the two values above, checked at the
   revision we converted from.
4. **The served side checks it**: `1bit serve` names the template's sha256 at every load and compares
   it with `config/template-pins.json` ([serve.md](serve.md#chat-template---template-pin---no-template-pin---template-pin-strict)),
   and the pull path checks it before a pulled model is ever registered or served
   ([laya-first.md](laya-first.md), "A row is not served with an unexpected template").

Rule 1-3 apply to the model we publish; rule 4 applies to every model we run, ours or not.

## The snippets

The card gets a row in its front-matter table, next to the quantisation and the file list:

```markdown
| chat template | sha256 `b55ecfbbbcbb238a011465bfa139366db18790103f9205060cf2855a3f51b30f` — `Zyphra/ZAYA1-8B` `tokenizer_config.json` @ `main` (2026-11-05), unmodified |
```

`SHA256SUMS` gets one line per template, keyed to the file it belongs to — the existing per-file lines
stay exactly as they are:

```
b55ecfbbbcbb238a011465bfa139366db18790103f9205060cf2855a3f51b30f  zaya1-8b-Q4_K_M.gguf#tokenizer.chat_template
```

`UPLOAD-READY.md` gets a row in its checklist table:

```markdown
| chat template | sha256 `b55ecfbb…b30f` = `Zyphra/ZAYA1-8B@main` `tokenizer_config.json` [identical / vendor / ours: why] | recorded 2026-11-05 |
```

Three words decide the last cell, and they are the only three that exist:
**identical** (byte-identical to the source model's), **vendor** (the difference is the upstream
quantizer's, and the re-host is byte-identical to their file), **ours** (we authored it — the card says
what it does and why).

## Getting the two values (verified on this box, 2026-11-05)

```bash
# the template's sha256, straight from the GGUF's metadata header (no tensors are read)
python3 - <<'EOF'
import hashlib, gguf
r = gguf.GGUFReader("zaya1-8b-Q4_K_M.gguf")
t = r.fields.get("tokenizer.chat_template")
print(hashlib.sha256(bytes(t.parts[-1])).hexdigest() if t is not None else "none")
EOF
# → b55ecfbbbcbb238a011465bfa139366db18790103f9205060cf2855a3f51b30f
```

The source's own template, from the revision we converted from:

```bash
python3 -c 'import json,hashlib,sys; d=json.load(open(sys.argv[1])); print(hashlib.sha256(d["chat_template"].encode()).hexdigest())' \
  tokenizer_config.json          # or: sha256sum chat_template.jinja
```

A Q4NX / NPU container has no GGUF: its template is the `chat_template` string in
`tokenizer_config.json`, and the digest is over that string (the same `bash` line above; no `gguf`).

Two traps, both measured here:

- **A sha is per file, not per model.** `Qwen3-4B` has one template in our published
  `1bit-MONSTER/Qwen3-4B-GGUF` (`57f1fd00f0…`, Qwen's own GGUF) and a different one in the unsloth-style
  quant the box serves as `~/models/Qwen3-4B-Q4_K_M.gguf` (`5da44855ab…`). Record the sha you measured,
  on the file you published, and name the source it matched.
- **The generation prompt is part of it.** A template that differs only in how it opens the assistant
  turn still changes what the model sees; that is a difference, not a detail.

## Proposal: the chat template as a signed part (CycloneDX)

The reason a template escapes review is that it has no name, no digest and no place in a dependency
graph: it is a string inside a file. Give it the three things a component has and every existing tool
starts working on it. In CycloneDX 1.6 terms, the template is a `data` component of the model's own
BOM:

```json
{
  "type": "data",
  "bom-ref": "pkg:huggingface/1bit-MONSTER/ZAYA1-8B-GGUF/zaya1-8b-Q4_K_M.gguf?chat-template",
  "name": "tokenizer.chat_template",
  "hashes": [{"alg": "SHA-256", "content": "b55ecfbbbcbb238a011465bfa139366db18790103f9205060cf2855a3f51b30f"}],
  "properties": [
    {"name": "onebit:template:source", "value": "Zyphra/ZAYA1-8B@main tokenizer_config.json"},
    {"name": "onebit:template:relation", "value": "identical"},
    {"name": "onebit:template:parent", "value": "pkg:huggingface/1bit-MONSTER/ZAYA1-8B-GGUF/zaya1-8b-Q4_K_M.gguf@<file sha256>"}
  ]
}
```

What that buys, concretely:

- a pin in `config/template-pins.json` becomes a dependency check rather than a hand-kept table: the
  expected digest is a BOM entry, and `1bit serve`'s line is the runtime that enforces it;
- a re-host whose vendor changed the template (the unsloth `gpt-oss` rewrite, bartowski's DeepSeek
  generation prompt, TheBloke's Mistral system-message handling) shows up as a **component difference**
  between two BOMs, which is the form a reviewer already knows how to read;
- signing the BOM signs the template, so "the weights are the same file and the template is not"
  stops being an invisible substitution.

What it does not buy, and must not be sold as: a digest says *which* template, never whether it is
harmless. The pin is only as good as the revision it was recorded from, and a first-time pull of an
unknown model has no ancestor to compare with — that case is why the engine logs the sha at every load
instead of only checking a pin.

No upstream filing, no vendor contact: this is a proposal for our own BOM output
(`tools/` and the release packages) and for the cards we publish.
