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
# Usage: python3 repack_gguf.py <model.gguf> <output_dir> [converter_dir] [--skip-verify]
#        python3 repack_gguf.py --check-stamp <dir>      (converter still current?)
#        python3 repack_gguf.py --verify-source <dir> <model.gguf>  (does this dir decode to its GGUF?)
#        python3 repack_gguf.py --tile-check <dir> <model.gguf>     (the same check under its old name)
import json
import os
import re
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


def tile_order_policy(rc, line):
    """(what the stamp may claim, is the repack fatal) from tools/q4nx_diff.py's verdict.

    Why this exists here and not in the plan generator: `tools/npu_plan.py` writes
    `container.tile_order = "engine-1"` for every plan it generates (engine#1130 carries that field
    and refuses an id it does not implement at load time), so the load-time guard cannot catch a
    container whose tiles are a DIFFERENT order -- the declaration is unconditional.  A converter
    that writes another order repacks cleanly and decodes as noise at HTTP 200 with nothing in the
    log (measured: FLM_Q4NX_Converter `d1d5232`, engine#1110).  The repack is the only step holding
    both the source GGUF and the tiles, so the claim is either measured true here or the repack
    fails; a directory nobody could check says `unverified` instead of claiming `engine-1`.
    """
    if rc == 0 and line.startswith("matches layout ours"):
        return "engine-1", False          # the order npu/forward/src/q4nx.cpp decodes
    if rc == 2:
        return "unverified", False        # nothing comparable: say so, do not claim a layout
    return "", True                      # another layout, or none: these tiles are not these weights


def tile_order_id(line):
    """The layout id a `q4nx_diff` verdict line names, for the stamp: `flm`, `ours`, or `unknown`.

    Only used on the refusing path (a stamp that says `engine-1` is written as that word), so the id
    exists to tell `tools/npu_plan.py` *which* order was measured -- any id but `engine-1` stops it
    writing the claim the repack just disproved.
    """
    m = re.match(r"matches layout ([A-Za-z0-9_-]+)", line)
    return m.group(1) if m else "unknown"


def tile_check_exit(rc, line):
    """Exit code for `--tile-check`: 0 ours, 1 another order or none, 2 nothing compared.

    The checker's own rc is not this: `verdict()` returns 0 for ANY matching layout, including the
    one this engine does not decode, so the CLI goes through the policy above instead of passing it
    through (engine#1110 review, blocker 1).
    """
    order, fatal = tile_order_policy(rc, line)
    if fatal:
        return 1
    return 2 if order == "unverified" else 0


def verify_source(gguf, out_dir):
    """(rc, line) verifying a converted container against the GGUF it was packed from (engine#1109).

    This is the same measurement `tools/q4nx_diff.py decode --layout both` prints, run on the directory
    the converter just wrote: decode it under each tile layout and compare every comparable tensor with
    the GGUF's own dequantized weight.  A converter checkout of the wrong VERSION packs the same values
    permuted inside the 5120-B chunk, decodes as noise at HTTP 200 with nothing in the log, and neither
    its path nor its commit can tell -- measured corr +0.0325 / max|err| 5.5 against the right checkout's
    corr +0.9994 / 2.9e-02.  So the repack compares the container with its source instead of trusting
    the name.  rc 0 one layout matched within the requant tolerance, 1 none did, 2 nothing comparable.

    Imported rather than reimplemented so the repack, `--verify-source`, and the tool an operator runs
    by hand cannot disagree; the verdict's tolerance, ratio and correlation are that file's, and its
    logic is what tests/q4nx_diff_grid_test.py and `q4nx_diff.py --self-test` already cover.
    """
    tools = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "tools")
    sys.path.insert(0, os.path.abspath(tools))
    try:
        import q4nx_diff as _D
    except Exception as e:                     # no numpy, no gguf-py, a pruned package
        return 2, f"tile check unavailable ({e})"

    class _Args:
        pass

    a = _Args()
    a.q4nx, a.gguf, a.tensor = out_dir, gguf, None
    results = {}
    for layout in sorted(_D.DECODERS):
        try:
            results[layout] = _D._decode_pass(a, layout, quiet=True)
        except Exception as e:
            return 2, f"tile check unavailable ({e})"
    return _D.verdict(results)


# The name `--tile-check` used before the check was named for what it proves: it is not a declaration
# about tile order, it is the container decoded against its source (engine#1109).  Kept so existing
# notes and the npu_plan refusal message keep working.
tile_order_check = verify_source


def converter_note(model_dir):
    """'path @ commit' from a directory's repack stamp, or '' -- who built these tiles.

    Used on the failure path so a refusal names the converter that produced the container, which is
    the one fact `repack-stamp.txt` has and the user needs (engine#1109).
    """
    try:
        with open(os.path.join(model_dir, "repack-stamp.txt")) as f:
            fields = dict(line.rstrip("\n").split("=", 1) for line in f if "=" in line)
    except OSError:
        return ""
    return "%s @ %s" % (fields.get("converter_path", "?"), fields.get("converter_commit", "?"))


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
    # `--skip-verify` is a flag on the repack form, not a subcommand: the verification decodes the whole
    # container against the GGUF and costs a GGUF pass, so the tight repack loops the payload builds run
    # in can opt out (engine#1109).  Opting out is recorded in the stamp as unverified, never as engine-1.
    skip_verify = "--skip-verify" in sys.argv[1:]
    argv = [a for a in sys.argv[1:] if a != "--skip-verify"]
    if len(argv) < 2:
        print("usage: repack_gguf.py <model.gguf> <output_dir> [converter_dir] [--skip-verify]")
        print("       repack_gguf.py --check-stamp <dir>")
        print("       repack_gguf.py --verify-source <dir> <model.gguf>")
        print("       repack_gguf.py --tile-check <dir> <model.gguf>")
        return 2
    gguf, out_dir = argv[0], argv[1]
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
    #
    # The canonical checkout is looked up next to the engine checkout AND in the
    # home directory: an installed engine (~/.local/share/1bit-engine) or a
    # checkout anywhere but ~ has no sibling np-model-generic, and fell through to
    # the staging checkout below, an older converter without the reorder
    # (MiniCPM5-1B then answered " the", #224).
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(here, os.pardir, os.pardir, "np-model-generic", "third_party", "FLM_Q4NX_Converter"),
        os.path.join(os.path.expanduser("~"), "np-model-generic", "third_party", "FLM_Q4NX_Converter"),
        os.path.join(here, os.pardir, "third_party", "FLM_Q4NX_Converter"),
        os.path.join(here, os.pardir, os.pardir, "1bit-MONSTER", "third_party", "FLM_Q4NX_Converter"),
        os.path.join(here, os.pardir, os.pardir, "1bit-MONSTER-iso-build", "third_party", "FLM_Q4NX_Converter"),
        "/home/bcloud/wt/twostream-main/third_party/FLM_Q4NX_Converter",
    ]
    default_converter = os.environ.get("ONEBIT_Q4NX_CONVERTER") or next(
        (os.path.abspath(p) for p in candidates if os.path.exists(os.path.join(p, "convert.py"))), "")
    converter_dir = os.path.abspath(argv[2] if len(argv) > 2 else default_converter)

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

    if argv[0] == "--check-stamp":
        stamp_path = os.path.join(argv[1], "repack-stamp.txt")
        if not os.path.exists(stamp_path):
            print(f"stamp: none at {stamp_path} (dir predates the stamp)")
            return 1
        got = dict(l.rstrip("\n").split("=", 1) for l in open(stamp_path) if "=" in l)
        # A directory the repack REFUSED must never be reused: the refusing path leaves model.q4nx on
        # disk, so before this a cache hit reported `stamp: current` and served the container the repack
        # had just rejected -- the wrong-converter container this issue is about, still served after a
        # guard refused it (engine#1109).  Stale is the answer no matter what the other keys say.
        if got.get("refused"):
            print(f"stamp: stale, the repack refused this directory "
                  f"({got.get('refused')}: {got.get('tile_check', '')})")
            return 1
        # Re-resolve with the directory's own arch so a dir whose arch needs a
        # different converter keeps comparing against the right one.
        try:
            _c = json.load(open(os.path.join(argv[1], "config.json")))
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
        # A stamp that never measured the container against its source is the same defect one step
        # earlier, and a rebuild is the only way it becomes verified.  `unverified` is "cannot tell"
        # (2): serve reuses it WITH a warning, and re-checking it would re-run the measurement that
        # just failed on every start.
        order = got.get("tile_order")
        if order is None:
            print("stamp: stale, no measured tile_order (dir predates the source check)")
            return 1
        if order == "unverified":
            print(f"stamp: WARNING: this container was never verified against its source "
                  f"({got.get('tile_check', 'no reason recorded')}); reusing it UNVERIFIED")
            return 2
        if order != "engine-1":
            print(f"stamp: stale, tile_order='{order}' is not the order this engine decodes "
                  f"({got.get('tile_check', '')})")
            return 1
        print("stamp: current")
        return 0

    # `--verify-source <dir> <model.gguf>`: the check the repack runs after the converter returns, on
    # a directory that already exists -- a published container, one built by another converter, or one
    # a cache is about to serve (engine#1109).  It decodes the container against the GGUF it was packed
    # from, under both tile layouts, so a wrong converter checkout reads as a named tensor with its
    # correlation instead of a 200.  Exit codes: 0 verified in the engine-1 order, 1 another order or
    # none, 2 nothing comparable.
    if argv[0] == "--verify-source":
        if len(argv) < 3:
            print("usage: repack_gguf.py --verify-source <dir> <model.gguf>")
            return 2
        rc, line = verify_source(argv[2], argv[1])
        exit_rc = tile_check_exit(rc, line)
        if exit_rc == 0:
            print(f"verify-source: matches its source in the engine-1 order: {line}")
            return 0
        if exit_rc == 2:
            print(f"verify-source: WARNING: cannot verify the container against {argv[2]}: {line}")
            return 2
        print(f"verify-source: FAIL: {line}")
        print(f"verify-source: {argv[1]} does not decode to {argv[2]}")
        note = converter_note(argv[1])
        if note:
            print(f"verify-source: built by {note}")
        return 1

    # `--tile-check`: the same check under the name #1110 introduced.  Kept as a separate spelling so
    # scripts written against its exact output keep working; new callers should say --verify-source.
    if argv[0] == "--tile-check":
        if len(argv) < 3:
            print("usage: repack_gguf.py --tile-check <dir> <model.gguf>")
            return 2
        rc, line = verify_source(argv[2], argv[1])
        print(line)
        return tile_check_exit(rc, line)

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
    if not os.environ.get("ONEBIT_Q4NX_CONVERTER") and len(argv) <= 2:
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
        # deepseek2: converters from before 5fc52149b transpose k_b's per-head
        # slice, which packs 16 tiles/head where the forward reads 12
        # (kb_rows_ = qk_nope_ = 192) and segfaults. Refuse them instead of
        # rewriting their source.
        _kb_file = os.path.join(converter_dir, "q4nx", "models", _arch + ".py")
        if _arch == "deepseek2" and os.path.exists(_kb_file):
            with open(_kb_file) as f:
                if 'transpose_slice = "k_b_proj" in nm' in f.read():
                    print(f"repack: {_kb_file} transposes k_b (16 tiles/head; the "
                          f"forward reads 12). Update the converter to 5fc52149b or later.")
                    return 1
        if _t and _t != "Q4_1":
            import shutil as _shutil, tempfile as _tempfile
            run_dir = _tempfile.mkdtemp(prefix="q4nx-converter-")
            _shutil.copytree(converter_dir, run_dir, dirs_exist_ok=True)
            _c["default_tensor_type"] = "Q4_1"
            with open(os.path.join(run_dir, "configs", _arch + ".json"), "w") as f:
                json.dump(_c, f, indent=4)
            print(f"repack: {_arch} config asks for default_tensor_type={_t}; the NPU "
                  f"forward reads Q4_1 tiles, so the converter runs from a patched "
                  f"copy ({run_dir})")
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

    # The container is verified against the GGUF it came from before the directory is cached, for the
    # same reason the scale scan below is: a converter checkout of the wrong version is silent all the
    # way to a 200 (engine#1109), and only the source can tell (its own commit cannot).  Cost is one
    # dequantize of every comparable tensor per layout, and `--skip-verify` opts out for the tight
    # repack loops the payload builds run in; a skipped dir is stamped unverified, never engine-1.
    # Paid once per dir because a cache hit re-checks metadata only (docs/npu.md, "Declared scales are
    # checked").
    if skip_verify:
        _tile_rc, _tile_line = 2, "skipped (--skip-verify)"
    else:
        _tile_rc, _tile_line = verify_source(gguf, out_dir)
    tile_order, _tile_fatal = tile_order_policy(_tile_rc, _tile_line)
    if _tile_fatal:
        print(f"repack: verify-source FAIL: {_tile_line}")
        print(f"repack: converter: {converter_dir} @ {converter_commit()}")
        print("repack: the converter wrote q4_1 tiles in an order this engine does not decode -- refusing "
              "the directory.  Point --converter / ONEBIT_Q4NX_CONVERTER at a checkout that writes the "
              "engine-1 order (engine#1110; measured good: FLM_Q4NX_Converter 81be119e).")
        # The refusal leaves the tiles on disk, and `tools/npu_plan.py` writes container.tile_order for
        # whatever directory it is pointed at -- so the measurement has to survive it, or one plan run
        # hands the loader the "engine-1" claim this check just disproved (measured: npu_plan.py on a
        # refused dir wrote tile_order="engine-1", and the loader accepts that).  Stamp the verdict.
        with open(os.path.join(out_dir, "repack-stamp.txt"), "w") as f:
            f.write(f"converter_path={converter_dir}\n")
            f.write(f"converter_commit={converter_commit()}\n")
            f.write(f"tile_order={tile_order_id(_tile_line)}\n")
            f.write(f"tile_check={_tile_line}\n")
            f.write("refused=tile order\n")
        return 1
    if tile_order == "unverified":
        print(f"repack: WARNING: the container could not be verified against its source ({_tile_line}); "
              f"the stamp records that instead of claiming the engine-1 order")
    else:
        print(f"repack: the container is verified against {gguf}: {_tile_line}")

    # Record the provenance the cache-hit gate needs (see --check-stamp above).
    with open(os.path.join(out_dir, "repack-stamp.txt"), "w") as f:
        f.write(f"converter_path={converter_dir}\n")
        f.write(f"converter_commit={converter_commit()}\n")
        f.write(f"arch={arch}\n")
        f.write(f"model_type={cfg.get('model_type', '')}\n")
        f.write("q4nx_default_tensor_type=Q4_1\n")
        f.write(f"tile_order={tile_order}\n")
        f.write(f"tile_check={_tile_line}\n")
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
