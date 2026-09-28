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
        # The GGUF arch (e.g. qwen35moe) differs from the HF model_type
        # (qwen3_5_moe); the stamp check re-resolves the converter by arch, so
        # carry it explicitly.
        "arch": arch,
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

    # MLA (deepseek2, e.g. GLM-4.7-Flash).  A deepseek2 config's head_dim /
    # key_length describe the KV LATENT, not a per-head width, so the generic
    # values above are wrong for the forward: it needs
    #   head_dim         = value_length_mla  (per-head VALUE width)
    #                      so the O design derives K = NH*head_dim = NH*v_head,
    #   intermediate_size = expert_feed_forward_length (per-EXPERT width) so the
    #                      G/U/D designs ARE the expert MLP,
    # plus the latent geometry itself.  Set mla=1 and let the forward branch.
    if p == "deepseek2":
        q_lora = int(_num(reader, f"{p}.attention.q_lora_rank", 0))
        kv_lora = int(_num(reader, f"{p}.attention.kv_lora_rank", 0))
        rope_dim = int(_num(reader, f"{p}.rope.dimension_count", 0))
        k_mla = int(_num(reader, f"{p}.attention.key_length_mla", 0))
        v_mla = int(_num(reader, f"{p}.attention.value_length_mla", head_dim))
        cfg.update({
            "mla": 1,
            "head_dim": v_mla,
            "intermediate_size": int(_num(reader, f"{p}.expert_feed_forward_length", 0)),
            "q_lora_rank": q_lora,
            "kv_lora_rank": kv_lora,
            "rope_dim": rope_dim,
            "qk_nope_head_dim": max(k_mla - rope_dim, 0),
            "v_head_dim": v_mla,
            "dense_intermediate_size": int(_num(reader, f"{p}.feed_forward_length", 0)),
            "n_expert": int(_num(reader, f"{p}.expert_count", 0)),
            "top_k": int(_num(reader, f"{p}.expert_used_count", 0)),
            "expert_shared": int(_num(reader, f"{p}.expert_shared_count", 0)),
            "leading_dense_block_count": int(_num(reader, f"{p}.leading_dense_block_count", 0)),
            "expert_weights_scale": float(_num(reader, f"{p}.expert_weights_scale", 1.0)),
            "expert_weights_norm": int(_num(reader, f"{p}.expert_weights_norm", 0)),
            # llama.cpp's deepseek2: 1 = softmax, 2 = SIGMOID.  Getting this
            # wrong still produces plausible output with a wrong argmax.
            "expert_gating_func": int(_num(reader, f"{p}.expert_gating_func", 1)),
        })
    # Qwen MoE (qwen3moe / qwen35moe).  Same MoE meanings as deepseek2 above but
    # different GGUF field names, no MLA, and all layers are MoE (no dense leading
    # block).  The per-EXPERT intermediate must replace the dense
    # feed_forward_length (5472 for the 30B) so the G/U/D designs ARE the expert
    # MLP; the forward's n_expert/top_k/expert gating come from these keys.
    if p in ("qwen3moe", "qwen35moe"):
        cfg.update({
            "intermediate_size": int(_num(reader, f"{p}.expert_feed_forward_length", 0)),
            "n_expert": int(_num(reader, f"{p}.expert_count", 0)),
            "top_k": int(_num(reader, f"{p}.expert_used_count", 0)),
            "expert_shared": int(_num(reader, f"{p}.expert_shared_feed_forward_length", 0)),
            "leading_dense_block_count": 0,
            "expert_weights_norm": int(_num(reader, f"{p}.expert_weights_norm", 1)),
            "expert_weights_scale": float(_num(reader, f"{p}.expert_weights_scale", 1.0)),
            "expert_gating_func": 1,   # Qwen MoE gating = softmax
        })
        if p == "qwen35moe":
            # GDN (linear attention) geometry + the layer-type list the forward
            # dispatches on.  The HF config names these linear_*; the GGUF carries
            # them under ssm.*.  attn_output_gate makes the q_proj emit q|gate
            # (q_rows doubles), which the QKV design must know.
            def _ssm(key, dflt=0):
                return int(_num(reader, f"{p}.ssm.{key}", dflt))
            _vhd = _ssm("state_size")
            _inner = _ssm("inner_size")
            _fai = int(_num(reader, f"{p}.full_attention_interval", 4)) or 4
            _nl = int(cfg["num_hidden_layers"])
            cfg.update({
                "linear_num_key_heads": _ssm("group_count"),
                "linear_key_head_dim": _vhd,
                "linear_value_head_dim": _vhd,
                "linear_num_value_heads": (_inner // _vhd) if _vhd else 0,
                "linear_conv_kernel_dim": _ssm("conv_kernel", 4),
                "attn_output_gate": 1,
                "layer_types": ["full_attention" if (i % _fai == _fai - 1) else "linear_attention"
                                for i in range(_nl)],
            })
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
    # ONEBIT_Q4NX_CONVERTER, the canonical npu-infer/model-generic checkout,
    # a sibling checkout next to the engine, the 1bit-MONSTER checkout that
    # carries it upstream, then a staging checkout.
    #
    # The canonical ("npu-infer/model-generic") converter is preferred because
    # it is the one whose output the model-generic forward was validated
    # against: it carries the minicpm module + config, the LongRoPE work, the
    # MiniCPM4 scale fixes, and the llama-arch q/k interleaved->paired reorder
    # applied for EVERY arch "llama". The 1bit-MONSTER checkout still gates that
    # reorder on `freq_base <= 20000`, which is false for MiniCPM5-1B
    # (freq_base 5e6), so it repacks q/k unreordered and the forward decodes
    # " the" instead of " Paris". Its output matches the published
    # MiniCPM5-1B-NPU2 model.q4nx byte for byte; the gated one does not.
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(here, os.pardir, os.pardir, "np-model-generic", "third_party", "FLM_Q4NX_Converter"),
        os.path.join(here, os.pardir, "third_party", "FLM_Q4NX_Converter"),
        os.path.join(here, os.pardir, os.pardir, "1bit-MONSTER", "third_party", "FLM_Q4NX_Converter"),
        os.path.join(here, os.pardir, os.pardir, "1bit-MONSTER-iso-build", "third_party", "FLM_Q4NX_Converter"),
        "/home/bcloud/wt/twostream-main/third_party/FLM_Q4NX_Converter",
    ]
    default_converter = os.environ.get("ONEBIT_Q4NX_CONVERTER") or next(
        (os.path.abspath(p) for p in candidates if os.path.exists(os.path.join(p, "convert.py"))), "")
    converter_dir = os.path.abspath(sys.argv[3] if len(sys.argv) > 3 else default_converter)

    def pick_converter(arch):
        # Prefer a checkout that has a module for THIS arch.  The canonical
        # np-model-generic converter has no deepseek2 module, while the
        # iso-build checkout does -- it is what built the GLM-4.7-Flash Q4NX the
        # forward's MLA+MoE path was validated against (" Paris, France.").
        if arch:
            for p in candidates:
                ap = os.path.abspath(p)
                if os.path.exists(os.path.join(ap, "convert.py")) and \
                   os.path.exists(os.path.join(ap, "q4nx", "models", str(arch) + ".py")):
                    return ap
        return None

    # Provenance stamp.  The dir cache cannot tell which converter produced a dir
    # (`model.q4nx` existence is all it checks), and the ¬declared-vs-carried scan
    # cannot either, so the pair is: the scan (what was carried) plus this stamp
    # (which converter did the carrying).  `--check-stamp <dir>` re-resolves the
    # converter and compares -- exit 0 current, 2 cannot tell (caller fails open),
    # 1 stale (caller discards and rebuilds).
    def converter_commit():
        try:
            r = subprocess.run(["git", "-C", converter_dir, "rev-parse", "--short", "HEAD"],
                               capture_output=True, text=True, timeout=10)
            return r.stdout.strip() if r.returncode == 0 else "(unknown)"
        except Exception:
            return "(unknown)"

    if sys.argv[1] == "--check-stamp":
        stamp_path = os.path.join(sys.argv[2], "repack-stamp.txt")
        if not os.path.exists(stamp_path):
            print(f"stamp: none at {stamp_path} (dir predates the stamp)")
            return 1
        got = dict(l.rstrip("\n").split("=", 1) for l in open(stamp_path) if "=" in l)
        # Re-resolve with the directory's own arch so a dir whose arch needs a
        # different converter keeps comparing against the right one.
        try:
            _c = json.load(open(os.path.join(sys.argv[2], "config.json")))
            _a = _c.get("arch") or _c.get("model_type", "")
        except Exception:
            _a = ""
        if not os.environ.get("ONEBIT_Q4NX_CONVERTER"):
            _picked = pick_converter(_a)
            if _picked:
                converter_dir = _picked
        if not os.path.exists(os.path.join(converter_dir, "convert.py")):
            print(f"stamp: cannot tell, converter unresolved ('{converter_dir}')")
            return 2
        for key, want in (("converter_path", converter_dir), ("converter_commit", converter_commit()),
                          # The NPU forward reads Q4_1 tiles; a dir built before the
                          # repack forced Q4_1 (e.g. qwen2's old Q4_0 config) has no
                          # such key and must be rebuilt.
                          ("q4nx_default_tensor_type", "Q4_1")):
            if got.get(key) != want:
                print(f"stamp: stale, {key}='{got.get(key)}' != '{want}'")
                return 1
        print("stamp: current")
        return 0

    os.makedirs(out_dir, exist_ok=True)
    convert_py = os.path.join(converter_dir, "convert.py")
    if not os.path.exists(convert_py):
        print(f"converter not found: {convert_py}")
        print("set ONEBIT_Q4NX_CONVERTER to an FLM_Q4NX_Converter checkout (convert.py + configs/)")
        return 1

    # The converter resolves configs/<arch>.json relative to its CWD.
    #
    # The NPU forward reads Q4_1 tiles (5120 B: 512 B bf16 scales + 512 B bf16
    # zeros + 4096 B packed int4).  A converter config that asks for Q4_0 packs
    # no zero half, and the forward then decodes the whole tensor wrong -- not a
    # rounding error, garbage (Qwen2.5-7B-Instruct answered a CJK token instead
    # of ' Paris'; the same GGUF repacked with Q4_1 is byte-identical to the
    # published FastFlowLM dir).  qwen2.json and lfm2.json shipped Q4_0 while
    # every other arch shipped Q4_1.  Rather than trust a moving external
    # checkout, run the converter from a patched copy whenever this arch's
    # config asks for anything but Q4_1.  The provenance stamp still records the
    # resolved converter and its commit, so the dir cache stays stable.
    run_dir = converter_dir
    from gguf import GGUFReader as _GGUFReader
    _r = _GGUFReader(gguf)
    _af = _field(_r, "general.architecture")
    _arch = _af.contents() if _af else ""
    if isinstance(_arch, bytes):
        _arch = _arch.decode()
    _arch = str(_arch)
    if not os.environ.get("ONEBIT_Q4NX_CONVERTER") and len(sys.argv) <= 3:
        _picked = pick_converter(_arch)
        if _picked and _picked != converter_dir:
            print(f"repack: {_arch} has no module in {converter_dir}; using {_picked}")
            converter_dir = _picked
            run_dir = converter_dir
    _cfg = os.path.join(converter_dir, "configs", _arch + ".json")
    if _arch and os.path.exists(_cfg):
        try:
            with open(_cfg) as f:
                _c = json.load(f)
        except Exception:
            _c = {}
        _t = _c.get("default_tensor_type")
        # deepseek2: the converter's k_b "MLA math grid" transpose (its commit
        # 62594563f) is INVERTED.  The dequantised per-head slice is [512][192]
        # (kv_lora x nope); packed AS-IS it is the 12 tiles/head the forward reads
        # (`kb_rows_ = qk_nope_ = 192`), while transposing it to [192][512] gives
        # 16 tiles/head and the forward then segfaults.  Measured directly against
        # the converter's own _pack_2d_float, and against the working Q4NX the
        # GLM capture used.  Neutralise the gate in the patched copy.
        _kb_patch = False
        _kb_file = os.path.join(converter_dir, "q4nx", "models", _arch + ".py")
        if _arch == "deepseek2" and os.path.exists(_kb_file):
            try:
                if 'transpose_slice = "k_b_proj" in nm' in open(_kb_file).read():
                    _kb_patch = True
            except Exception:
                pass
        if (_t and _t != "Q4_1") or _kb_patch:
            import shutil as _shutil, tempfile as _tempfile
            run_dir = _tempfile.mkdtemp(prefix="q4nx-converter-")
            _shutil.copytree(converter_dir, run_dir, dirs_exist_ok=True)
            if _t and _t != "Q4_1":
                _c["default_tensor_type"] = "Q4_1"
                with open(os.path.join(run_dir, "configs", _arch + ".json"), "w") as f:
                    json.dump(_c, f, indent=4)
                print(f"repack: {_arch} config asks for default_tensor_type={_t}; the NPU "
                      f"forward reads Q4_1 tiles, so the converter runs from a patched "
                      f"copy ({run_dir})")
            if _kb_patch:
                _p = os.path.join(run_dir, "q4nx", "models", _arch + ".py")
                _s2 = open(_p).read().replace(
                    'transpose_slice = "k_b_proj" in nm',
                    'transpose_slice = False  # engine: k_b packs as-is -> 12 tiles/head')
                with open(_p, "w") as f:
                    f.write(_s2)
                print(f"repack: {_arch} k_b transpose is inverted (gives 16 tiles/head, the "
                      f"forward reads 12); neutralised in the patched copy ({run_dir})")
    r = subprocess.run([sys.executable, os.path.join(run_dir, "convert.py"), gguf, out_dir],
                       cwd=run_dir)
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

    # Record the provenance the cache-hit gate needs (see --check-stamp above).
    with open(os.path.join(out_dir, "repack-stamp.txt"), "w") as f:
        f.write(f"converter_path={converter_dir}\n")
        f.write(f"converter_commit={converter_commit()}\n")
        f.write(f"arch={arch}\n")
        f.write(f"model_type={cfg.get('model_type', '')}\n")
        f.write("q4nx_default_tensor_type=Q4_1\n")
        f.write(f"repack_script={os.path.abspath(__file__)}\n")

    # Enforce the scale-key guard before the dir is cached (and before anyone
    # trusts it).  MiniCPM4-8B decoded garbage because embedding_scale /
    # residual_scale / logit_scale were declared by the GGUF and dropped here,
    # and no argmax-level check could see it.  Two tiers, deliberately:
    #   FAIL on the known-critical list (proven factors the forward must apply);
    #   WARN (never fail) on any other scale-ish key, because the next arch's
    #     factor will have an unlisted name, while some scale-ish keys are
    #     legitimately not carried (deepseek2.expert_weights_scale is a routing
    #     scale, not a logit scale).  Mirrors scripts/check_repack_config.py.
    _critical = ("embedding_scale", "residual_scale", "logit_scale", "scale_emb",
                 "scale_depth", "dim_model_base", "final_logit_softcapping",
                 "attn_logit_softcapping")
    _missing, _warned = [], []
    for _f in reader.fields.values():
        _tail = _f.name.split(".")[-1]
        _low = _tail.lower()
        if not (_tail in _critical or any(h in _low for h in ("scale", "softcap", "mup"))):
            continue
        _v = _f.contents()
        if isinstance(_v, (list, tuple)):
            _v = _v[-1]
        try:
            _v = float(_v)
        except (TypeError, ValueError):
            continue
        if cfg.get(_tail) is None:
            (_missing if _tail in _critical else _warned).append((_tail, _v))
    for _k, _v in _warned:
        print(f"[WARN] the GGUF declares {arch}.{_k} = {_v} and config.json does not "
              f"carry it; scale-ish but not known-critical -- check whether the "
              f"forward needs it")
    if _missing:
        for _k, _v in _missing:
            print(f"[FAIL] the GGUF declares {arch}.{_k} = {_v} but config.json drops it; "
                  f"the forward would not apply it")
        return 1
    print("[OK] every known-critical scale-like key the GGUF declares is carried in config.json")
    print(f"config.json: {cfg}")
    print("repack complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
