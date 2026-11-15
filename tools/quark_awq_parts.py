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
"""Run AMD Quark (0.13) weight-only PTQ on a Hugging Face checkpoint and write the parts tools/quark_to_q4nx.py reads.

usage: tools/quark_awq_parts.py HF_DIR OUT.safetensors [--algo awq] [--scheme int4_wo_32] [--n 128] [--seqlen 512]

Needs Quark and PyTorch in its own environment (docs/npu.md "Building an AWQ pack"); nothing in CI runs it.
It drives Quark's Python API rather than `quark-cli torch-llm-ptq`, because on Quark 0.13 the CLI cannot do this
on CPU (measured, engine QUARK-EVAL 2026-11-15): there is no built-in qwen3 AWQ config, the CPU device-map check
refuses `--device cpu`, and `--multi_device` calibrates on meta tensors.  The model is loaded fully on CPU, the
qwen3 template gets the qwen2 AWQ scaling groups (Qwen3's q/k norms come after the projections, so the pairs
are the same), and the frozen state dict is written with the zero points as F32 so a numpy reader takes it.

The default scheme is int4_wo_32: symmetric (zero = -8 s, exact in Q4NX's bf16 zero) and group 32 (the q4_1
group).  uint4_* converts only with tools/quark_to_q4nx.py --allow-uint4; see that tool for why.
Cost on the strixhalo CPU (4 threads under thermal-run): about 3.1 h for Qwen3-1.7B AWQ at 128 x 512.
Calibration data is a real cost RTN does not have: every AWQ pack needs this run once, then ships like any pack.

Not run in CI (it needs torch and Quark). Its gate is a by-hand end-to-end run on a tiny random Qwen3 into
tools/quark_to_q4nx.py (docs/npu.md "An AWQ pack from AMD Quark"); re-run that whenever the Quark version moves.
The parts file records the Quark version in its metadata for that reason.
"""
import argparse
import time

# The qwen2 scaling groups, registered for qwen3 (Quark 0.13 has none). AWQ pairs an op with the linear(s) it
# feeds and folds the scale into that op, so the pairs depend only on which op feeds which linear. Qwen3 adds
# q_norm / k_norm AFTER q_proj / k_proj, which leaves input_layernorm -> q/k/v, v -> o, post_attention_layernorm
# -> gate/up and up -> down unchanged. check_pairs() refuses a model whose layer does not have these modules,
# so a future layout change fails here instead of quantizing the wrong pairs.
QWEN3_AWQ = {
    "name": "awq",
    "model_decoder_layers": "model.layers",
    "scaling_layers": [
        {"prev_op": "input_layernorm", "layers": ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
         "inp": "self_attn.q_proj", "module2inspect": "self_attn"},
        {"prev_op": "self_attn.v_proj", "layers": ["self_attn.o_proj"], "inp": "self_attn.o_proj"},
        {"prev_op": "post_attention_layernorm", "layers": ["mlp.gate_proj", "mlp.up_proj"],
         "inp": "mlp.gate_proj", "module2inspect": "mlp"},
        {"prev_op": "mlp.up_proj", "layers": ["mlp.down_proj"], "inp": "mlp.down_proj"},
    ],
}


def check_pairs(model, cfg):
    """Every op a scaling group names must exist in the first decoder layer (and the norm must be a norm)."""
    layer = model.get_submodule(cfg["model_decoder_layers"])[0]
    names = dict(layer.named_modules())
    for g in cfg["scaling_layers"]:
        for n in [g["prev_op"], *g["layers"], g["inp"]]:
            if n not in names:
                raise SystemExit(f"AWQ pair {g['prev_op']} -> {g['layers']}: the layer has no module {n!r}; "
                                 "this model's layout differs from qwen2/qwen3, so the scaling groups do not apply")
        if g["prev_op"].endswith("layernorm") and "norm" not in type(names[g["prev_op"]]).__name__.lower():
            raise SystemExit(f"{g['prev_op']} is a {type(names[g['prev_op']]).__name__}, not a norm")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("hf")
    p.add_argument("out")
    p.add_argument("--algo", default="awq")
    p.add_argument("--scheme", default="int4_wo_32")
    p.add_argument("--n", type=int, default=128)
    p.add_argument("--seqlen", type=int, default=512)
    p.add_argument("--dataset", default="pileval")
    a = p.parse_args()

    import torch
    from safetensors.torch import save_file
    from transformers import AutoModelForCausalLM

    from quark.experimental.cli.register_template import create_template_from_json_dict
    from quark.torch import LLMTemplate, ModelQuantizer
    from quark.torch.utils.llm import get_calib_dataloader, get_tokenizer, preprocess_for_quantization

    t0 = time.time()
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_pretrained(a.hf, dtype=torch.bfloat16, device_map=None).eval()
    mt = model.config.model_type
    if mt == "qwen3" and a.algo == "awq":
        check_pairs(model, QWEN3_AWQ)
        LLMTemplate.register_template(create_template_from_json_dict({
            "model_type": "qwen3", "kv_layers_name": ["*k_proj", "*v_proj"], "q_layer_name": "*q_proj",
            "exclude_layers_name": ["lm_head"], "algorithm_configs": {"awq": QWEN3_AWQ}}))
    preprocess_for_quantization(model)
    tok = get_tokenizer(a.hf, max_seq_len=a.seqlen, model_type=mt)
    calib = get_calib_dataloader(dataset_name=a.dataset, tokenizer=tok, batch_size=1, num_calib_data=a.n,
                                 seqlen=a.seqlen, device="cpu")
    q = ModelQuantizer(LLMTemplate.get(mt).get_config(scheme=a.scheme, algorithm=a.algo), False)
    model = q.freeze(q.quantize_model(model, calib))
    parts = {}
    for k, v in model.state_dict().items():
        v = v.detach().cpu().contiguous()
        if k.endswith("._weight_quantizer.zero_point"):
            v = v.to(torch.float32)
        elif "_quantizer" in k and not k.endswith("._weight_quantizer.scale"):
            continue
        parts[k] = v
    import quark
    save_file(parts, a.out, metadata={"quark_scheme": a.scheme, "quark_algo": a.algo, "source": a.hf,
                                      "quark_version": str(getattr(quark, "__version__", "unknown"))})
    print(f"{len(parts)} tensors -> {a.out} in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
