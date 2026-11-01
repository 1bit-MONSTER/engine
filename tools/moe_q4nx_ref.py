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
"""moe_q4nx_ref: a CPU mixture-of-experts forward over a Q4NX container, and its tokenizer.

This is the host reference a MoE decode path is judged against. It reads the container the
NPU route reads (a Q4NX model directory: model.q4nx + config.json, docs/npu.md) and nothing
else, so a disagreement with the engine is the container's or the kernel's, never a second
copy of the weights. The routed experts are gathered exactly as the architecture says: a
float32 softmax over the router logits, the top-k experts, the normalised top-k weights
(Qwen3MoE `norm_topk_prob`), then that many SwiGLU experts summed.

  moe_q4nx_ref.py tokenizer <model.gguf> <out dir>
        build tokenizer.json + tokenizer_config.json from the GGUF's own vocabulary and chat
        template, so a packed directory is servable and this reference tokenises the way the
        engine does (nothing is downloaded, nothing is copied from HF).

  moe_q4nx_ref.py serve <model dir> [--port 18110] [--max-tokens 64]
        an OpenAI-compatible /v1/models + /v1/chat/completions with logprobs, greedy, so
        tools/logprob_agree.py can compare it against another route.

  moe_q4nx_ref.py logprobs <model dir> --prompt TEXT [--max-tokens 64]
        the same forward, one prompt, printed as JSON (no server).

  moe_q4nx_ref.py selftest
        the numerics a bug would silently pass: softmax/top-k normalisation, RoPE, and a
        synthetic container whose routers pick known experts.

Weights stay as packed and are decoded per tensor when first used: the dense tensors
(attention, lm_head, router, embedding) are cached in float32 (~5 GB for a 30B-A3B), the
routed experts are decoded per token from the mapped file, which is the same streaming the
NPU and GPU routes do.
"""
import argparse
import json
import struct
import sys
import threading
import time

import numpy as np

from q4nx_pack import TILE, from_bf16, untile


# --- the container ---------------------------------------------------------------------------

class Q4nx:
    """A Q4NX model directory: the tensor table, the mapped data and config.json."""

    def __init__(self, path):
        self.dir = path
        with open(path + "/model.q4nx", "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            self.table = json.loads(f.read(n))
            self.base = 8 + n
        self.mm = np.memmap(path + "/model.q4nx", dtype=np.uint8, mode="r")
        with open(path + "/config.json") as f:
            c = json.load(f)
        self.cfg = c
        self.hidden = c["hidden_size"]
        self.layers = c["num_hidden_layers"]
        self.heads = c["num_attention_heads"]
        self.kv_heads = c.get("num_key_value_heads", self.heads)
        self.head_dim = c.get("head_dim", self.hidden // self.heads)
        self.rope_theta = c.get("rope_theta", 10000.0)
        self.eps = c.get("rms_norm_eps", 1e-6)
        self.experts = c.get("num_experts", 0)
        self.top_k = c.get("num_experts_per_tok", 0)
        self.norm_topk = c.get("norm_topk_prob", True)
        self.moe_meta = self.table.get("__metadata__", {}).get("moe", {})
        self._cache = {}

    # a tensor's raw bytes, as stored
    def raw(self, name):
        lo, hi = self.table[name]["data_offsets"]
        return np.asarray(self.mm[self.base + lo:self.base + hi])

    def shape(self, name):
        return self.table[name]["shape"]

    def bf16(self, name):
        """A BF16 tensor as float32, its stored shape."""
        t = self.table[name]
        w = from_bf16(self.raw(name).view(np.uint16))
        return w.reshape(t["shape"]) if t["dtype"] == "BF16" else w

    def weight(self, name):
        """A tile tensor as float32 [rows, cols], chunks of rows at a time (peak memory)."""
        if name in self._cache:
            return self._cache[name]
        t = self.table[name]
        if t["dtype"] == "BF16":
            w = self.bf16(name).astype(np.float32)
        else:
            rt, kt = t["shape"][0], t["shape"][1]
            rows, cols = rt * 32, kt * 256
            raw = self.raw(name).reshape(rt, kt, TILE)
            w = np.empty((rows, cols), np.float32)
            step = max(1, 4096 // 32)  # 4096 rows at a time: a 151936-row lm_head is 1.2 GB f32
            for r0 in range(0, rt, step):
                r1 = min(rt, r0 + step)
                w[r0 * 32:r1 * 32] = untile(raw[r0:r1].reshape(-1, TILE), (r1 - r0) * 32, cols)
        self._cache[name] = w
        return w

    def experts_weight(self, name, e):
        """Expert e of a routed stack as float32 [rows, cols]; not cached (streamed)."""
        t = self.table[name]
        rt, kt = t["shape"][0], t["shape"][1]
        info = self.moe_meta["tensors"][name]
        lo = info["offsets"][e]
        rows = info["rows"]
        rte = rt // info["experts"]
        raw = np.asarray(self.mm[self.base + lo:self.base + lo + info["bytes"]]).reshape(rte, kt, TILE)
        w = np.empty((rows, kt * 256), np.float32)
        w[:] = untile(raw.reshape(-1, TILE), rows, kt * 256)
        return w

    def layer_name(self, layer, suffix):
        return f"model.layers.{layer}.{suffix}"


# --- the forward -----------------------------------------------------------------------------

def rmsnorm(x, w, eps):
    return (x / np.sqrt((x * x).mean(-1, keepdims=True) + eps)) * w


def silu(x):
    return x / (1.0 + np.exp(-x))


def softmax(x):
    e = np.exp(x - x.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def rope_tables(pos, dim, theta):
    inv = theta ** (-np.arange(0, dim, 2, dtype=np.float64) / dim)
    ang = np.outer(np.asarray(pos, np.float64), inv)  # [T, dim/2]
    return np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)


def rope(x, cos, sin):
    """x [T, heads, dim] -> rotated in place-free copy; rotate_half on the last axis."""
    x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    c, s = cos[:, None, :], sin[:, None, :]
    return np.concatenate([x1 * c - x2 * s, x2 * c + x1 * s], axis=-1).astype(np.float32)


class KVCache:
    def __init__(self, layers, kv_heads, head_dim):
        self.k = [np.zeros((0, kv_heads, head_dim), np.float32) for _ in range(layers)]
        self.v = [np.zeros((0, kv_heads, head_dim), np.float32) for _ in range(layers)]

    def append(self, layer, k, v):
        self.k[layer] = np.concatenate([self.k[layer], k], axis=0)
        self.v[layer] = np.concatenate([self.v[layer], v], axis=0)

    def length(self):
        return self.k[0].shape[0]


class Forward:
    """Greedy decode over the container; `step` takes tokens and returns next-token logprobs."""

    def __init__(self, q):
        self.q = q
        self.embed = q.bf16("model.embed_tokens.weight").astype(np.float32)
        self.norm = q.weight("model.norm.weight")
        self.lm_head = None  # built on first use (1.2 GB for a 151936 x 2048 head)
        self.kv = KVCache(q.layers, q.kv_heads, q.head_dim)
        self.pos = 0
        self.capture_layer = -1   # set to L to keep layer L's MoE input row
        self.capture_stop = False
        self.capture = None
        self.captured = None

    def reset(self):
        self.kv = KVCache(self.q.layers, self.q.kv_heads, self.q.head_dim)
        self.pos = 0

    def logits(self, tokens):
        q = self.q
        x = self.embed[np.asarray(tokens, np.int64)].astype(np.float32)  # [T, H]
        T = x.shape[0]
        cos, sin = rope_tables(np.arange(self.pos, self.pos + T), q.head_dim, q.rope_theta)
        gqa = q.heads // q.kv_heads
        for L in range(q.layers):
            self._layer = L
            # a kernel gate needs the MoE block's input, not the block's output
            self.capture = L if L == self.capture_layer else None
            h = rmsnorm(x, q.weight(q.layer_name(L, "input_layernorm.weight")), q.eps)
            shape = (T, q.heads, q.head_dim)
            qv = (h @ q.weight(q.layer_name(L, "self_attn.q_proj.weight")).T).reshape(shape)
            k = (h @ q.weight(q.layer_name(L, "self_attn.k_proj.weight")).T).reshape(
                (T, q.kv_heads, q.head_dim))
            v = (h @ q.weight(q.layer_name(L, "self_attn.v_proj.weight")).T).reshape(
                (T, q.kv_heads, q.head_dim))
            qv = rope(rmsnorm(qv, q.weight(q.layer_name(L, "self_attn.q_norm.weight")), q.eps),
                      cos, sin)
            k = rope(rmsnorm(k, q.weight(q.layer_name(L, "self_attn.k_norm.weight")), q.eps),
                     cos, sin)
            self.kv.append(L, k, v)
            K, V = self.kv.k[L], self.kv.v[L]  # [P, kv_heads, dim]
            K = np.repeat(K, gqa, axis=1)
            V = np.repeat(V, gqa, axis=1)
            scores = np.einsum("thd,phd->htp", qv, K) / np.sqrt(q.head_dim)
            # causal mask over the absolute positions this call covers
            P = K.shape[0]
            mask = np.arange(P)[None, :] > (self.pos + np.arange(T))[:, None]
            scores = np.where(mask[None], -np.inf, scores)
            probs = softmax(scores)  # [heads, T, P]
            attn = np.einsum("htp,phd->thd", probs, V).reshape(T, q.heads * q.head_dim)
            x = x + attn @ q.weight(q.layer_name(L, "self_attn.o_proj.weight")).T
            h = rmsnorm(x, q.weight(q.layer_name(L, "post_attention_layernorm.weight")), q.eps)
            if self.capture is not None:
                self.captured = h.copy()
                if self.capture_stop:   # a gate needs one layer's input, not the logits
                    return None
            if q.experts:
                x = x + self.moe(h)
            else:
                g = h @ q.weight(q.layer_name(L, "mlp.gate_proj.weight")).T
                u = h @ q.weight(q.layer_name(L, "mlp.up_proj.weight")).T
                x = x + (silu(g) * u) @ q.weight(q.layer_name(L, "mlp.down_proj.weight")).T
        self.pos += T
        if self.lm_head is None:
            self.lm_head = q.weight("lm_head.weight") if q.table.get("lm_head.weight") else \
                self.embed
        hn = rmsnorm(x, self.norm, q.eps)
        return hn @ self.lm_head.T  # [T, V]

    def moe(self, h):
        """The router exactly as the architecture says, then the k chosen experts summed."""
        q = self.q
        y = np.zeros_like(h)
        gate_name = self._lname("mlp.gate.weight")
        for i in range(h.shape[0]):
            xi = h[i]
            logits = q.weight(gate_name) @ xi
            p = softmax(logits.astype(np.float64)).astype(np.float32)
            top = np.argsort(-p, kind="stable")[:q.top_k]
            w = p[top]
            if q.norm_topk:
                w = w / w.sum()
            for e, we in zip(top, w):
                gate = q.experts_weight(self._lname("mlp.experts.gate_proj.weight"), e)
                up = q.experts_weight(self._lname("mlp.experts.up_proj.weight"), e)
                down = q.experts_weight(self._lname("mlp.experts.down_proj.weight"), e)
                act = silu(gate @ xi) * (up @ xi)
                y[i] += we * (down @ act)
        return y

    _layer = 0

    def _lname(self, suffix):
        return f"model.layers.{self._layer}.{suffix}"

    def step(self, tokens):
        """Feed tokens; the log-probabilities of every vocabulary token after the last one."""
        lg = self.logits(tokens)[-1].astype(np.float64)
        m = lg.max()
        return lg - m - np.log(np.exp(lg - m).sum())


# --- the same forward over the GGUF's own weights (no container) ------------------------------

class GgufWeights:
    """The container interface, backed by the GGUF itself: f32 from the original blocks.

    This exists to measure ONE thing: how much of the container's distance from HRX is the
    container's own quantisation. Same forward code, same tokenizer, different weights — so a
    difference between this and the container is the repack, and nothing else.
    """

    def __init__(self, path):
        import gguf
        self.r = gguf.GGUFReader(path)
        self.by_name = {t.name: t for t in self.r.tensors}
        import gguf_to_q4nx as G
        self.to_gguf = {}   # q4nx name -> gguf name
        for t in self.r.tensors:
            try:
                self.to_gguf[G.hf_name(t.name)] = t.name
            except SystemExit:
                pass
        arch = self.r.fields["general.architecture"].contents()
        a = lambda k, d=None: self.r.fields.get(f"{arch}.{k}").contents() if f"{arch}.{k}" in self.r.fields else d  # noqa: E731
        self.hidden = a("embedding_length")
        self.layers = a("block_count")
        self.heads = a("attention.head_count")
        self.kv_heads = a("attention.head_count_kv", self.heads)
        self.head_dim = a("attention.key_length", self.hidden // self.heads)
        self.rope_theta = float(a("rope.freq_base"))
        self.eps = float(a("attention.layer_norm_rms_epsilon"))
        self.experts = a("expert_count", 0)
        self.top_k = a("expert_used_count", 0)
        self.norm_topk = True
        self.moe_meta = {}
        self.table = {n: {} for n in self.to_gguf}   # same keys Forward asks for
        self._cache = {}

    def layer_name(self, layer, suffix):
        return f"model.layers.{layer}.{suffix}"

    def _tensor(self, name):
        t = self.by_name[self.to_gguf[name]]
        return t

    def _rowbytes(self, t):
        import gguf
        bs, ts = gguf.GGML_QUANT_SIZES[t.tensor_type]
        return int(t.shape[0]) // bs * ts

    def _deq_rows(self, t, r0, n):
        """Rows [r0, r0+n) of a 2-D or 3-D ggml tensor, dequantised to f32 [n, cols]."""
        import gguf
        if len(t.shape) == 1:   # a norm: one row, so it broadcasts over the activations
            return np.asarray(gguf.quants.dequantize(t.data, t.tensor_type),
                              np.float32).reshape(1, int(t.shape[0]))
        K = int(t.shape[0])
        rb = self._rowbytes(t)
        raw = np.asarray(t.data).reshape(-1)[r0 * rb:(r0 + n) * rb]
        w = gguf.quants.dequantize(raw, t.tensor_type)
        return np.asarray(w, np.float32).reshape(n, K)

    def weight(self, name):
        if name in self._cache:
            return self._cache[name]
        t = self._tensor(name)
        w = self._deq_rows(t, 0, int(t.shape[1]) if len(t.shape) > 1 else 1)
        self._cache[name] = w
        return w

    def bf16(self, name):
        t = self._tensor(name)
        return np.asarray(self._deq_rows(t, 0, int(t.shape[1]) if len(t.shape) > 1 else 1),
                          np.float32)

    def experts_weight(self, name, e):
        t = self._tensor(name)
        N = int(t.shape[1])
        return self._deq_rows(t, e * N, N)

    def close(self):
        self._cache.clear()


def topk(lp, k):
    idx = np.argpartition(-lp, k)[:k]
    idx = idx[np.argsort(-lp[idx])]
    return idx, lp[idx]


# --- chat, tokenizer, serving ----------------------------------------------------------------

def build_tokenizer(gguf_path, out_dir):
    """tokenizer.json + tokenizer_config.json from the GGUF's own vocabulary and template."""
    import gguf
    import tokenizers

    r = gguf.GGUFReader(gguf_path)
    toks = r.fields["tokenizer.ggml.tokens"].contents()
    types = r.fields["tokenizer.ggml.token_type"].contents()
    merges = [m.split(" ") for m in r.fields["tokenizer.ggml.merges"].contents()]
    # 1 = NORMAL (the BPE vocabulary itself), 2/3/4/5 = added (unknowns, control, user, unused)
    vocab = {t: i for i, (t, ty) in enumerate(zip(toks, types)) if ty == 1}
    added = [{"id": i, "content": t, "single_word": False, "lstrip": False, "rstrip": False,
              "normalized": False, "special": ty in (2, 3, 4)}
             for i, (t, ty) in enumerate(zip(toks, types)) if ty != 1]
    # Qwen2/Qwen3 pre-tokenisation: the byte-level BPE pattern these vocabularies were built for.
    pattern = ("(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| "
               "?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+")
    tj = {
        "version": "1.0", "truncation": None, "padding": None, "added_tokens": added,
        "normalizer": {"type": "NFC"},
        "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
            {"type": "Split", "pattern": {"Regex": pattern}, "behavior": "Isolated", "invert": False},
            {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": False, "use_regex": False}]},
        "post_processor": {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": False,
                           "use_regex": False},
        "decoder": {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": False,
                    "use_regex": False},
        "model": {"type": "BPE", "dropout": None, "unk_token": None,
                  "continuing_subword_prefix": None, "end_of_word_suffix": None,
                  "fuse_unk": False, "byte_fallback": False, "ignore_merges": False,
                  "vocab": vocab, "merges": merges},
    }
    # the mapping must be a bijection and total, or every token id after the first clash is wrong
    assert len(vocab) + len(added) == len(toks), (len(vocab), len(added), len(toks))
    tokenizers.Tokenizer.from_str(json.dumps(tj))
    if out_dir:
        with open(out_dir + "/tokenizer.json", "w") as f:
            json.dump(tj, f)
        chat = r.fields.get("tokenizer.chat_template")
        cfg = {
            "tokenizer_class": "Qwen2Tokenizer",
            "chat_template": chat.contents() if chat else None,
            "eos_token": toks[types.index(3)] if 3 in types else None,
            "add_bos_token": False,
            "clean_up_tokenization_spaces": False,
        }
        with open(out_dir + "/tokenizer_config.json", "w") as f:
            json.dump(cfg, f, indent=2)
        print(f"wrote {out_dir}/tokenizer.json ({len(vocab)} bpe + {len(added)} added tokens) "
              f"and tokenizer_config.json")
    return tj


class Chat:
    """The model directory's tokenizer + chat template, rendered as the engine renders it."""

    def __init__(self, model_dir, gguf_template=None):
        import tokenizers
        with open(model_dir + "/tokenizer.json") as f:
            self.tok = tokenizers.Tokenizer.from_str(f.read())
        tmpl = gguf_template
        path = model_dir + "/tokenizer_config.json"
        if tmpl is None:
            with open(path) as f:
                tmpl = json.load(f).get("chat_template")
        self.tmpl = tmpl
        self._tf = {}
        try:
            import jinja2
            from jinja2.sandbox import SandboxedEnvironment
            # The chat template arrives with the model directory, i.e. it is data from a
            # downloaded file, so it runs in jinja's sandbox: a template cannot reach Python
            # internals. Autoescape is declared rather than left defaulted, and it is off on
            # purpose: the rendered text is a token sequence (<|im_start|>user ...), not HTML,
            # so escaping would corrupt every marker. select_autoescape() keeps that explicit.
            env = SandboxedEnvironment(
                autoescape=jinja2.select_autoescape(default_for_string=False, default=False),
                trim_blocks=True, lstrip_blocks=True,
                extensions=["jinja2.ext.loopcontrols"])
            self._tf[tmpl] = env.from_string(tmpl)
        except ImportError:
            pass

    def render(self, prompt, enable_thinking=False):
        if self.tmpl and self._tf:
            return self._tf[self.tmpl].render(
                messages=[{"role": "user", "content": prompt}], add_generation_prompt=True,
                enable_thinking=enable_thinking, tools=None)
        return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

    def encode(self, text):
        return self.tok.encode(text).ids

    def decode(self, ids):
        return self.tok.decode(ids, skip_special_tokens=False)


class Ref:
    """Greedy chat completion with logprobs, one request at a time."""

    def __init__(self, model_dir, max_tokens=64, top_logprobs=5):
        self.q = Q4nx(model_dir)
        self.fw = Forward(self.q)
        self.chat = Chat(model_dir)
        self.max_tokens = max_tokens
        self.top_logprobs = top_logprobs
        self.lock = threading.Lock()

    def eos(self):
        e = self.q.cfg.get("eos_token_id")
        ids = set(e if isinstance(e, list) else [e] if e is not None else [])
        return ids | {151643}  # <|endoftext|>: llama.cpp treats it as EOG too

    def complete(self, prompt, max_tokens=None, top_logprobs=None):
        max_tokens = max_tokens or self.max_tokens
        top_logprobs = top_logprobs or self.top_logprobs
        text = self.chat.render(prompt)
        ids = self.chat.encode(text)
        t0 = time.time()
        with self.lock:
            self.fw.reset()
            lp = self.fw.step(ids)
            out, toks, eos = [], [], self.eos()
            for _ in range(max_tokens):
                nxt = int(np.argmax(lp))
                idx, lps = topk(lp, top_logprobs)
                # an engine serves float32 log-probs; a JSON double would carry more
                # precision than any route can, which shows up as "not float32" downstream
                toks.append({
                    "id": nxt, "token": self.chat.decode([nxt]),
                    "lp": float(np.float32(lp[nxt])),
                    "top": [{"id": int(i), "token": self.chat.decode([int(i)]),
                             "lp": float(np.float32(v))}
                            for i, v in zip(idx, lps)],
                })
                out.append(nxt)
                if nxt in eos:
                    break
                lp = self.fw.step([nxt])
        return {
            "text": self.chat.decode(out),
            "tokens": toks,
            "secs": round(time.time() - t0, 3),
            "prompt_tokens": len(ids),
        }


def forced(ref, prompt, max_tokens, url, model):
    """Score a route's own greedy continuation with this forward, position by position.

    The served routes stop at end-of-turn, so a comparison built only on positions where both
    routes happened to emit the same token is a handful of tokens per prompt. Here the route's
    generated ids are teacher-forced through this forward, so every position the route produced
    is compared - same token, two distributions - which is the number a weight or KV difference
    shows up in.
    """
    import urllib.error
    import urllib.request
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "temperature": 0, "top_k": 1, "seed": 1234, "max_tokens": max_tokens,
            "logprobs": True, "top_logprobs": 1, "cache_prompt": False, "ignore_eos": True,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions",
                                data=json.dumps(body).encode(),
                                headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=1200) as r:
            res = json.loads(r.read())
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        return {"prompt": prompt, "error": f"{type(e).__name__}: {e}"}
    toks = ((res.get("choices") or [{}])[0].get("logprobs") or {}).get("content") or []
    if not toks:
        return {"prompt": prompt, "error": "route reported no logprobs"}
    ids = [t["id"] for t in toks]
    prompt_ids = ref.chat.encode(ref.chat.render(prompt))
    with ref.lock:
        ref.fw.reset()
        lp = ref.fw.step(prompt_ids)
        rows = []
        for t, nxt in zip(toks, ids):
            mine, theirs = float(lp[nxt]), float(t["logprob"])
            mine = float(np.float32(mine))  # what a serving route would report
            rows.append({"id": nxt, "token": ref.chat.decode([nxt]), "route_lp": theirs,
                         "ref_lp": mine, "d": abs(mine - theirs),
                         "ref_top1": int(np.argmax(lp)),
                         "route_top1": int(t["top"][0]["id"]) if t.get("top") else nxt})
            lp = ref.fw.step([nxt])
    ds = [r["d"] for r in rows]
    return {"prompt": prompt, "positions": len(rows), "tokens": rows,
            "max_abs_dlogp": max(ds), "mean_abs_dlogp": sum(ds) / len(ds),
            "top1_agree": sum(r["ref_top1"] == r["route_top1"] for r in rows) / len(rows)}


PROMPTS = [
    "What is the capital of France? Answer in one word.",
    "What is 17 * 23? Answer with the number only.",
    "Name the three primary colours.",
    "Complete this sentence: The quick brown fox",
    "List the first five prime numbers, comma separated.",
    "Translate 'good morning' into Spanish.",
    "What does CPU stand for?",
    "Write one sentence about the sea.",
]


def agree(args):
    """8 prompts x N tokens against a served route, every position teacher-forced."""
    ref = Ref(args.model, args.max_tokens, 1)
    out = []
    for p in (args.prompt or PROMPTS):
        r = forced(ref, p, args.max_tokens, args.route, args.route_model)
        if "error" in r:
            print(f"ERROR {p}: {r['error']}", flush=True)
            out.append(r)
            continue
        print(f"{r['positions']:3d} pos  max|dlogp| {r['max_abs_dlogp']:.4e}  "
              f"mean|d| {r['mean_abs_dlogp']:.4e}  top1 {r['top1_agree']:.3f}  {p}", flush=True)
        out.append(r)
    good = [r for r in out if "error" not in r]
    if good:
        pos = sum(r["positions"] for r in good)
        print(f"\n{len(good)}/{len(out)} prompts, {pos} positions teacher-forced: "
              f"max|dlogp| {max(r['max_abs_dlogp'] for r in good):.4e}, "
              f"mean|dlogp| {sum(r['mean_abs_dlogp'] * r['positions'] for r in good) / pos:.4e}, "
              f"top-1 {sum(r['top1_agree'] * r['positions'] for r in good) / pos:.4f}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=1)
    return 0 if len(good) == len(out) else 1


def score(fw, prompt_ids, cont_ids):
    """Feed the prompt, then score and feed each continuation id: one step per position."""
    lp = fw.step(prompt_ids)
    out = []
    for nxt in cont_ids:
        out.append(float(np.float32(lp[nxt])))
        lp = fw.step([nxt])
    return out


def vs_gguf(args):
    """Container vs the GGUF's own f32 weights vs the served route, on one prompt."""
    import gc
    import urllib.request

    model = args.model
    prompt = args.prompt
    chat = Chat(model)
    body = {"model": args.route_model, "messages": [{"role": "user", "content": prompt}],
            "temperature": 0, "top_k": 1, "seed": 1234, "max_tokens": args.max_tokens,
            "logprobs": True, "top_logprobs": 5, "cache_prompt": False, "ignore_eos": True,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(args.route.rstrip("/") + "/v1/chat/completions",
                                data=json.dumps(body).encode(),
                                headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        res = json.loads(r.read())
    toks = ((res.get("choices") or [{}])[0].get("logprobs") or {}).get("content") or []
    ids = [t["id"] for t in toks]
    route_lp = [float(np.float32(t["logprob"])) for t in toks]
    route_top1 = [t["top"][0]["id"] if t.get("top") else t["id"] for t in toks]
    prompt_ids = chat.encode(chat.render(prompt))
    print(f"prompt ids {len(prompt_ids)}, route ids {len(ids)}", flush=True)

    lps, top1 = {}, {}
    want = {"container": Q4nx, "gguf": lambda m: GgufWeights(args.gguf)}
    for label in (("container", "gguf") if args.only == "both" else (args.only,)):
        weights = want[label](model)
        fw = Forward(weights)
        fw.reset()
        t0 = time.time()
        lp = fw.step(prompt_ids)
        lps[label], top1[label] = [], []
        for nxt in ids:
            lps[label].append(float(np.float32(lp[nxt])))
            top1[label].append(int(np.argmax(lp)))
            lp = fw.step([nxt])
        print(f"{label}: {len(lps[label])} positions in {time.time()-t0:.0f}s", flush=True)
        del fw, weights
        gc.collect()

    def stats(a, b):
        d = [abs(x - y) for x, y in zip(a, b)]
        return (max(d) if d else 0.0, sum(d) / len(d) if d else 0.0)

    ref = dict(lps)
    ref["route"] = route_lp
    out = {"prompt": prompt, "positions": len(ids), "per_position": ref}
    pairs = [(a, b) for a in lps for b in ref if a != b]
    for a, b in pairs:
        ma, me = stats(ref[a], ref[b])
        out[f"{a}_vs_{b}"] = (ma, me)
        print(f"{a:9s} vs {b:6s} max|dlogp| {ma:.4e}  mean|dlogp| {me:.4e}")
    for label in lps:
        agree = sum(x == y for x, y in zip(top1[label], route_top1)) / len(route_top1)
        out[f"{label}_top1_vs_route"] = agree
        print(f"{label:9s} argmax == route's top-1: {agree:.3f}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f)
    return 0


def serve(args):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    ref = Ref(args.model, args.max_tokens, args.top_logprobs)
    print(f"moe_q4nx_ref: {args.model}: {ref.q.layers} layers, "
          f"{ref.q.experts} experts top-{ref.q.top_k}, vocab "
          f"{ref.q.shape('model.embed_tokens.weight')[0]}", flush=True)

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            if self.path.rstrip("/") in ("/v1/models", "/models"):
                self.send_json({"object": "list", "data": [{"id": args.model_id, "object": "model"}]})
            elif self.path.rstrip("/") in ("/health", ""):
                self.send_json({"status": "ok"})
            else:
                self.send_error(404)

        def do_POST(self):
            if not self.path.startswith("/v1/chat/completions"):
                return self.send_error(404)
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(n) or b"{}")
            msg = (body.get("messages") or [{}])[-1].get("content") or ""
            kw = (body.get("chat_template_kwargs") or {})
            r = ref.complete(msg, body.get("max_tokens"),
                             body.get("top_logprobs"))
            content = [{"id": t["id"], "token": t["token"], "logprob": t["lp"],
                        "bytes": None, "top_logprobs": [{"id": x["id"], "token": x["token"],
                                                         "logprob": x["lp"], "bytes": None}
                                                        for x in t["top"]]} for t in r["tokens"]]
            self.send_json({
                "id": "chatcmpl-ref", "object": "chat.completion", "model": args.model_id,
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": r["text"]},
                             "logprobs": {"content": content}}],
                "usage": {"prompt_tokens": r["prompt_tokens"], "completion_tokens":
                          len(r["tokens"]), "total_tokens": r["prompt_tokens"] + len(r["tokens"])},
            })

        def send_json(self, obj):
            b = json.dumps(obj).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

    srv = ThreadingHTTPServer(("127.0.0.1", args.port), H)
    print(f"serving on http://127.0.0.1:{args.port}", flush=True)
    srv.serve_forever()


def selftest():
    """The numerics a bug would silently pass, then a real container if one is given."""
    x = np.array([0.5, -1.0, 2.0], np.float32)
    p = softmax(x.astype(np.float64))
    assert abs(float(p.sum()) - 1.0) < 1e-12
    # top-k of a known router: the two largest, renormalised to sum to 1
    assert list(np.argsort(-p, kind="stable")[:2]) == [2, 0]
    w = p[[2, 0]] / p[[2, 0]].sum()
    assert abs(float(w.sum()) - 1.0) < 1e-12
    # RoPE is a rotation: |x| is preserved and position 0 is the identity
    rng = np.random.default_rng(0)
    v = rng.standard_normal((1, 1, 8)).astype(np.float32)
    c, s = rope_tables([0], 8, 1e7)
    assert np.allclose(rope(v, c, s), v)
    c, s = rope_tables([5], 8, 1e7)
    assert np.allclose(np.linalg.norm(rope(v, c, s)), np.linalg.norm(v), atol=1e-6)
    # rmsnorm scales by its weight and leaves a unit-rms vector with unit gain
    h = rng.standard_normal((1, 16)).astype(np.float32)
    assert abs(float(np.sqrt((rmsnorm(h, np.ones(16, np.float32), 0.0) ** 2).mean())) - 1.0) < 1e-6
    print("ok   softmax / top-k renormalisation / RoPE / rmsnorm")
    print("moe_q4nx_ref selftest: ok")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("tokenizer", help="build tokenizer.json + tokenizer_config.json from a GGUF")
    t.add_argument("gguf")
    t.add_argument("out")
    s = sub.add_parser("serve")
    s.add_argument("model")
    s.add_argument("--port", type=int, default=18110)
    s.add_argument("--max-tokens", type=int, default=64)
    s.add_argument("--top-logprobs", type=int, default=5)
    s.add_argument("--model-id", default="moe-q4nx-ref")
    lg = sub.add_parser("logprobs")
    lg.add_argument("model")
    lg.add_argument("--prompt", required=True)
    lg.add_argument("--max-tokens", type=int, default=64)
    lg.add_argument("--top-logprobs", type=int, default=5)
    sub.add_parser("selftest")
    ag = sub.add_parser("agree", help="teacher-forced logprob agreement against a served route")
    ag.add_argument("model")
    ag.add_argument("--route", required=True, help="the served OpenAI endpoint to compare with")
    ag.add_argument("--route-model", default="qwen3-coder")
    ag.add_argument("--max-tokens", type=int, default=64)
    ag.add_argument("--prompt", action="append", help="override the built-in 8 prompts")
    ag.add_argument("--out")
    v = sub.add_parser("vs-gguf", help="container vs the GGUF's own weights, on the route's tokens")
    v.add_argument("model")
    v.add_argument("--gguf", required=True)
    v.add_argument("--prompt", required=True)
    v.add_argument("--route", default="http://127.0.0.1:8090")
    v.add_argument("--route-model", default="qwen3-coder")
    v.add_argument("--max-tokens", type=int, default=64)
    v.add_argument("--only", choices=("both", "container", "gguf"), default="both",
                   help="which weights to run (gguf alone answers 'is the container the gap?')")
    v.add_argument("--out")
    a = p.parse_args()
    if a.cmd == "vs-gguf":
        return vs_gguf(a)
    if a.cmd == "tokenizer":
        build_tokenizer(a.gguf, a.out)
        return 0
    if a.cmd == "selftest":
        selftest()
        return 0
    if a.cmd == "serve":
        serve(a)
        return 0
    if a.cmd == "agree":
        return agree(a)
    ref = Ref(a.model, a.max_tokens, a.top_logprobs)
    r = ref.complete(a.prompt)
    print(json.dumps(r, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
