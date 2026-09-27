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
# GGUF -> Q4NX repack: runs the FLM_Q4NX_Converter (in its own directory, its
# configs/ are relative) and then writes the Hugging Face config.json the
# model-generic forward reads. The converter emits only model.q4nx +
# tokenizer.json; config.json is derived here from the GGUF metadata.
#
# Usage: python3 repack_gguf.py <model.gguf> <output_dir> [converter_dir]
import json
import os
import subprocess
import sys

# GGUF general.architecture -> HF model_type (the forward's model_type).
ARCH_TO_MODEL_TYPE = {
    "qwen2": "qwen2",
    "qwen3": "qwen3",
    "qwen3moe": "qwen3moe",
    "qwen3_5_moe": "qwen3_5_moe",
    "qwen35moe": "qwen3_5_moe",
    "llama": "llama",
    "deepseek2": "deepseek2",
    "minicpm": "minicpm",
    "gemma2": "gemma2",
    "gemma3": "gemma3",
}

# Arches whose attention projections carry a bias (the GGUF has no bias flag).
ATTENTION_BIAS_ARCHES = {"qwen2"}


def _field(reader, name):
    for f in reader.fields.values():
        if f.name == name:
            return f
    return None


def _num(reader, name, default):
    f = _field(reader, name)
    if f is None:
        return default
    v = f.contents()
    if isinstance(v, (tuple, list)):
        v = v[-1]
    if hasattr(v, "item"):
        v = v.item()
    return v


def write_config(gguf_path, out_dir, arch, vocab_size):
    from gguf import GGUFReader

    reader = GGUFReader(gguf_path)
    p = arch
    embedding = int(_num(reader, f"{p}.embedding_length", 0))
    heads = int(_num(reader, f"{p}.attention.head_count", 0))
    nkv = int(_num(reader, f"{p}.attention.head_count_kv", heads))
    head_dim = int(_num(reader, f"{p}.attention.key_length", embedding // heads if heads else 0))
    if not head_dim and heads:
        head_dim = embedding // heads

    # A separate output.weight tensor (not just token_embd) means the LM head
    # is untied. The forward defaults to tied when the field is absent, which
    # would use embed_tokens as the head and produce garbage.
    has_output = any(getattr(t, "name", "") == "output.weight" for t in reader.tensors)

    cfg = {
        "model_type": ARCH_TO_MODEL_TYPE.get(arch, arch),
        "hidden_size": embedding,
        "num_hidden_layers": int(_num(reader, f"{p}.block_count", 0)),
        "num_attention_heads": heads,
        "num_key_value_heads": nkv,
        "head_dim": head_dim,
        "intermediate_size": int(_num(reader, f"{p}.feed_forward_length", 0)),
        "vocab_size": vocab_size,
        "rms_norm_eps": float(_num(reader, f"{p}.attention.layer_norm_rms_epsilon", 1e-6)),
        "rope_theta": float(_num(reader, f"{p}.rope.freq_base", 10000.0)),
        "tie_word_embeddings": not has_output,
        "hidden_act": str(_num(reader, f"{p}.activation_function", "")).strip() or "silu",
    }
    # MiniCPM4 architecture scales.  The forward must apply them: the embedding
    # is scaled by 12.0 and every residual branch by scale_depth/sqrt(NL)
    # (0.2475 for 32 layers); without them the hidden state runs at the wrong
    # magnitude and the logits explode (absmax ~136 instead of ~30), decoding to
    # punctuation soup instead of ' Paris'.  logit_scale is a constant on the
    # output and does not move the argmax, but carry it for completeness.
    for _k in ("embedding_scale", "residual_scale", "logit_scale"):
        _v = _num(reader, f"{p}.{_k}", None)
        if _v is not None:
            cfg[_k] = float(_v)
    eos = _num(reader, "tokenizer.ggml.eos_token_id", None)
    bos = _num(reader, "tokenizer.ggml.bos_token_id", None)
    if bos is not None:
        cfg["bos_token_id"] = int(bos)
    if eos is not None:
        cfg["eos_token_id"] = int(eos)
    # A chat template that opens with {{- bos_token }} (MiniCPM5 and other llama-class
    # models) needs the BOS id prepended to the rendered prompt. The forward builds a
    # plain ChatML prompt, which never carries it, and those models derail without it.
    tf = _field(reader, "tokenizer.chat_template")
    template = tf.contents() if tf is not None else None
    if isinstance(template, bytes):
        template = template.decode("utf-8", "replace")
    if isinstance(template, str) and template.lstrip().startswith("{{- bos_token }}"):
        cfg["add_bos_token"] = 1
    if arch in ATTENTION_BIAS_ARCHES:
        cfg["attention_bias"] = True
    # LongRoPE (MiniCPM4): the GGUF ships rope_factors_short.weight; carry the
    # short factor through config.json so the forward can apply it.
    for t in reader.tensors:
        if getattr(t, "name", "") == "rope_factors_short.weight":
            import numpy as np
            sf = np.frombuffer(t.data, dtype=np.float32).tolist()
            cfg["rope_scaling"] = {"rope_type": "longrope", "short_factor": sf}
            break
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    return cfg


def main():
    if len(sys.argv) < 3:
        print("usage: repack_gguf.py <model.gguf> <output_dir> [converter_dir]")
        return 2
    gguf, out_dir = sys.argv[1], sys.argv[2]
    # The Q4NX converter is not vendored in this repo. Resolve it in order:
    # ONEBIT_Q4NX_CONVERTER, a sibling checkout next to the engine, the
    # 1bit-MONSTER checkout that carries it upstream, then a staging checkout.
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(here, os.pardir, "third_party", "FLM_Q4NX_Converter"),
        os.path.join(here, os.pardir, os.pardir, "1bit-MONSTER", "third_party", "FLM_Q4NX_Converter"),
        "/home/bcloud/wt/twostream-main/third_party/FLM_Q4NX_Converter",
    ]
    default_converter = os.environ.get("ONEBIT_Q4NX_CONVERTER") or next(
        (os.path.abspath(p) for p in candidates if os.path.exists(os.path.join(p, "convert.py"))), "")
    converter_dir = os.path.abspath(sys.argv[3] if len(sys.argv) > 3 else default_converter)

    os.makedirs(out_dir, exist_ok=True)
    convert_py = os.path.join(converter_dir, "convert.py")
    if not os.path.exists(convert_py):
        print(f"converter not found: {convert_py}")
        print("set ONEBIT_Q4NX_CONVERTER to an FLM_Q4NX_Converter checkout (convert.py + configs/)")
        return 1

    # The converter resolves configs/<arch>.json relative to its CWD.
    r = subprocess.run([sys.executable, convert_py, gguf, out_dir],
                       cwd=converter_dir)
    if r.returncode != 0 or not os.path.exists(os.path.join(out_dir, "model.q4nx")):
        print("converter failed")
        return 1

    # Vocab size from the Q4NX the converter wrote. The converter pads the
    # vocab rows to a multiple of the 32-row Q4NX tile (MiniCPM4-8B: 73448 ->
    # 73472); the forward must dequantize with the PADDED size or it fails.
    vocab_size = 0
    q4nx = os.path.join(out_dir, "model.q4nx")
    if os.path.exists(q4nx):
        import struct as _struct
        with open(q4nx, "rb") as f:
            hsz = _struct.unpack("<Q", f.read(8))[0]
            d = json.loads(f.read(hsz))
        emb = d.get("model.embed_tokens.weight") or d.get("lm_head.weight")
        if emb and len(emb.get("shape", [])) == 2:
            vocab_size = emb["shape"][0]

    # Arch from the GGUF metadata.
    from gguf import GGUFReader
    reader = GGUFReader(gguf)
    arch_field = _field(reader, "general.architecture")
    arch = arch_field.contents() if arch_field else ""
    if isinstance(arch, bytes):
        arch = arch.decode()
    arch = str(arch)

    cfg = write_config(gguf, out_dir, arch, vocab_size)
    print(f"config.json: {cfg}")
    print("repack complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
