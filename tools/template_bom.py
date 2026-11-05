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
"""The chat template as a CycloneDX component, so a BOM can sign and diff it.

A `tokenizer.chat_template` is a Jinja program the llama.cpp-family backends run on every request, and
it has no name, no digest and no place in a dependency graph — which is why it escapes review. Give it
the shape of a component and the tools that already read a BOM start covering it: the digest goes in
`hashes`, where it came from goes in `properties`, and "the weights are the same file and the template
is not" becomes a component difference instead of an invisible substitution
(docs/template-publishing.md, "Proposal: the chat template as a signed part").

usage: tools/template_bom.py [--bom | --sums] [--self-test] PATH...

  PATH    a .gguf file, or a model directory (Q4NX/NPU container) — the template is read from the GGUF's
          metadata header, or from the directory's tokenizer_config.json `chat_template`, else
          `chat_template.jinja` beside it. Only the header is read; the tensors never are.
  --bom   emit one CycloneDX 1.6 document with a component per PATH; without it, one component per line
          (JSON Lines), so it composes with the other tools.
  --sums  emit the `SHA256SUMS` lines instead — `<sha256>  <file>#tokenizer.chat_template` for a `.gguf`,
          `<dir>/<source-file>#chat_template` for a container — the row a published file needs
          (docs/template-publishing.md). 'none' models print nothing, so the file stays clean.
  --self-test  build two tiny fixtures (a GGUF with a template, a directory with chat_template.jinja) and
          check the components this tool derives from them.

Exit codes: 0 all paths read, 1 a path could not be read, 3 --self-test failed.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile

SPEC_VERSION = "1.6"


def read_template(path: str) -> tuple[str | None, str | None]:
    """(template text, where it came from) — `(None, None)` when the model carries none."""
    if os.path.isdir(path):
        cfg = os.path.join(path, "tokenizer_config.json")
        if os.path.exists(cfg):
            try:
                with open(cfg, encoding="utf-8") as f:
                    t = json.load(f).get("chat_template")
                if isinstance(t, str) and t:
                    return t, "tokenizer_config.json"
            except Exception:
                pass
        jinja = os.path.join(path, "chat_template.jinja")
        if os.path.exists(jinja):
            with open(jinja, encoding="utf-8") as f:
                return f.read(), "chat_template.jinja"
        return None, None
    import gguf  # only for a .gguf: keeps --self-test's directory case dependency-free

    reader = gguf.GGUFReader(path)
    field = reader.fields.get("tokenizer.chat_template")
    if field is None:
        return None, None
    return bytes(field.parts[-1]).decode("utf-8", "replace"), "GGUF metadata (tokenizer.chat_template)"


def component(path: str) -> dict:
    tpl, source = read_template(path)
    ref = "pkg:generic/" + os.path.basename(os.path.abspath(path.rstrip("/"))) + "?chat-template"
    props = [
        {"name": "onebit:template:path", "value": os.path.abspath(path)},
        {"name": "onebit:template:source-file", "value": source or "none"},
    ]
    if tpl is None:
        return {
            "type": "data",
            "bom-ref": ref,
            "name": "tokenizer.chat_template",
            "description": "no chat template in this model (a base model, an mmproj, or a container without one)",
            "properties": props,
        }
    props.insert(1, {"name": "onebit:template:bytes", "value": str(len(tpl.encode()))})
    return {
        "type": "data",
        "bom-ref": ref,
        "name": "tokenizer.chat_template",
        "hashes": [{"alg": "SHA-256", "content": hashlib.sha256(tpl.encode()).hexdigest()}],
        "properties": props,
    }


def sums_line(path: str) -> str | None:
    """The `SHA256SUMS` line for PATH, or None when the model carries no template."""
    tpl, source = read_template(path)
    if tpl is None:
        return None
    sha = hashlib.sha256(tpl.encode()).hexdigest()
    if os.path.isdir(path):
        return f"{sha}  {os.path.join(path, os.path.basename(source)) if source else path}#chat_template"
    return f"{sha}  {os.path.basename(path)}#tokenizer.chat_template"


def self_test() -> int:
    import struct

    failures = 0

    def expect(what: str, got, want) -> None:
        ok = got == want
        print(f"{'ok  ' if ok else 'FAIL'} {what}: {got!r}")
        nonlocal failures
        failures += 0 if ok else 1

    tpl = "{% for m in messages %}{{ '<s>' ~ m['role'] }}{% endfor %}<s>assistant\\n"
    with tempfile.TemporaryDirectory() as d:
        # a header-only GGUF carrying the template, written the way a real one stores it
        def s(x: str) -> bytes:
            b = x.encode()
            return struct.pack("<Q", len(b)) + b

        kv = s("general.architecture") + struct.pack("<I", 8) + s("zamba2")
        kv += s("tokenizer.chat_template") + struct.pack("<I", 8) + s(tpl)
        gguf_path = os.path.join(d, "tpl.gguf")
        with open(gguf_path, "wb") as f:
            f.write(b"GGUF" + struct.pack("<IQQ", 3, 0, 2) + kv)
        c = component(gguf_path)
        expect("gguf: sha256", c["hashes"][0]["content"], hashlib.sha256(tpl.encode()).hexdigest())
        expect("gguf: source", next(p["value"] for p in c["properties"] if p["name"] == "onebit:template:source-file"),
               "GGUF metadata (tokenizer.chat_template)")
        expect("gguf: bytes", next(p["value"] for p in c["properties"] if p["name"] == "onebit:template:bytes"),
               str(len(tpl.encode())))

        # a container directory with only chat_template.jinja (the shape our OGA pack imports have)
        cont = os.path.join(d, "cont")
        os.makedirs(cont)
        with open(os.path.join(cont, "tokenizer_config.json"), "w", encoding="utf-8") as f:
            json.dump({"tokenizer_class": "Qwen2Tokenizer"}, f)
        with open(os.path.join(cont, "chat_template.jinja"), "w", encoding="utf-8") as f:
            f.write(tpl)
        c = component(cont)
        expect("dir: sha256", c["hashes"][0]["content"], hashlib.sha256(tpl.encode()).hexdigest())
        expect("dir: source", next(p["value"] for p in c["properties"] if p["name"] == "onebit:template:source-file"),
               "chat_template.jinja")

        # a model that carries no template: a component with no hash and a description that says why
        bare = os.path.join(d, "bare")
        os.makedirs(bare)
        with open(os.path.join(bare, "tokenizer_config.json"), "w", encoding="utf-8") as f:
            json.dump({"tokenizer_class": "LlamaTokenizer"}, f)
        c = component(bare)
        expect("no template: no hashes", "hashes" in c, False)
        expect("no template: source", next(p["value"] for p in c["properties"] if p["name"] == "onebit:template:source-file"),
               "none")
        expect("gguf: sums key", sums_line(gguf_path).split("  ")[1], "tpl.gguf#tokenizer.chat_template")
        expect("dir: sums key", sums_line(cont).split("  ")[1].endswith("cont/chat_template.jinja#chat_template"), True)
        expect("no template: no sums line", sums_line(bare), None)
    return 3 if failures else 0


def main(argv: list[str]) -> int:
    if "--self-test" in argv:
        return self_test()
    as_bom = "--bom" in argv
    as_sums = "--sums" in argv
    paths = [a for a in argv if not a.startswith("-")]
    if not paths:
        print(__doc__.strip().splitlines()[0], file=sys.stderr)
        print("usage: tools/template_bom.py [--bom | --sums] [--self-test] PATH...", file=sys.stderr)
        return 2
    comps, failed = [], False
    for p in paths:
        try:
            comps.append(component(p))
        except Exception as e:
            print(f"template_bom: {p}: {e}", file=sys.stderr)
            failed = True
    if as_sums:
        for p in paths:
            try:
                line = sums_line(p)
            except Exception:
                line = None
            if line:
                print(line)
    elif as_bom:
        print(json.dumps({"bomFormat": "CycloneDX", "specVersion": SPEC_VERSION, "version": 1,
                          "components": comps}, indent=1))
    else:
        for c in comps:
            print(json.dumps(c, sort_keys=True))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
