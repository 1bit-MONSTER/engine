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
"""mtp_head_ref.py -- host-side reference for the MTP draft head (deliverable 3's oracle).

What it is: the Qwen3.8 (== Bonsai-2) multi-token-prediction head is ONE full-attention decoder layer with an
input-fusion GEMM in front.  Given the base model's last hidden state and the embedding of the accepted token it
produces the draft logits of the next token.  This file is the CPU reference of that step: the number the NPU head
(pb16i family, M=16) is compared against, and the piece that catches a misread fusion order -- which would draft
garbage with no crash anywhere.

Written from the container facts (npu-re/docs/FACTS-flm-q4nx-container-and-mtp-head.md sections 2-4), not from
FastFlowLM code: nothing of theirs is linked, copied or shipped.  Pure NumPy; no NPU, no XRT, no device.

Three things live here:

  1. `decode_mode0_tile()` -- the 4736-byte "I8" head tile -> (32 rows x 256 K) float32, per the stated mode-0
     formula (u8 sc|mn tables, 4-bit nibbles, bf16 d|dmin; w = q4*S + M, ADDITIVE min).  This is a CROSS-CHECK for
     the converter, not a path we ship.
  2. `pb16i_plan_from_mode0()` / `pb16i_plan()` / `gemv_pb16i()` -- the arithmetic our integer kernels do (signed
     int4 codes x per-32-K-group int16 magnitudes, int32 accumulate per group, one host rounding).  The source
     tile's additive per-group min cannot be produced by that kernel, but it is constant per (group, column), so
     the host recovers it exactly from the activations' group sums (eight numbers per GEMV):
         y[n] = sum_g ms_g[n] * P_g[n]  +  sum_g (M_g[n] + 8*S_g[n]) * A_g ,   A_g = sum_{k in g} a[k]
     the +8 folding the unsigned->signed code shift.  So the head converts EXACTLY (to int16 mantissa + one host
     rounding) rather than lossily; `pb16i_plan()` is only the fallback for weights that arrive as floats.
  3. `head_step()` -- one draft step in the model's geometry (fc over [embedding | hidden], q and the output gate
     out of one tensor, QK-norm, partial rope 64 of 256, GQA 24/4, gated attention output, SwiGLU), returning
     logits and the new hidden state.

    python3 tools/mtp_head_ref.py --selftest        # all of the above checked; no device, no weights needed
"""
import sys

import numpy as np

# Geometry: the shipped container's config.json and the head descriptor agree on every value.
GEOM = dict(H=5120, IM=17408, HD=256, NH=24, NKV=4, ROT=64, THETA=1e7, EPS=1e-6)
G = 32          # K values per scale group (both formats)
TILE = 4736     # bytes of one mode-0 ("I8") head tile: 256 K x 32 columns


# ---------------------------------------------------------------- helpers
def bf16_to_f32(u16):
    """bf16 (raw uint16 or array of them) -> float32: zero-extend and shift, as the device does."""
    return (np.asarray(u16, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def f32_to_bf16_bits(v):
    u = np.asarray(v, dtype=np.float32).view(np.uint32)
    return ((u + 0x8000) >> 16).astype(np.uint16)


def rms(x, w, eps=None):
    """RMSNorm over the last axis with a weight vector."""
    eps = GEOM["EPS"] if eps is None else eps
    x = np.asarray(x, dtype=np.float32)
    return (x / np.sqrt((x * x).mean(axis=-1, keepdims=True) + eps)) * w


def silu(x):
    return x / (1.0 + np.exp(-x))


def rope_partial(x, positions, rot=None, theta=None):
    """HF-convention RoPE on the FIRST `rot` channels of the last axis; the rest passes through.

    Half-split pairing: channel i pairs with i + rot/2, angle = pos * theta^(-2i/rot), i < rot/2.
    x: (..., T, HD) float32; positions: (T,).
    """
    rot = GEOM["ROT"] if rot is None else rot
    theta = GEOM["THETA"] if theta is None else theta
    x = np.array(x, dtype=np.float32, copy=True)
    half = rot // 2
    inv = (1.0 / (theta ** (np.arange(half, dtype=np.float32) / half))).astype(np.float32)
    for t, pos in enumerate(np.atleast_1d(positions)):
        ang = (float(pos) * inv).astype(np.float32)
        c, s = np.cos(ang), np.sin(ang)
        a, b = x[..., t, :half].copy(), x[..., t, half:rot].copy()
        x[..., t, :half] = a * c - b * s
        x[..., t, half:rot] = b * c + a * s
    return x


# ------------------------------------------------ 1. the source tile -> weights (cross-check)
def decode_mode0_tile(tile):
    """4736-byte head tile -> w[n][k] (32 rows n x 256 K k) float32.

    Tile = u8 sc[8][32] | u8 mn[8][32] | 4096 B nibbles | bf16 d[32] | bf16 dmin[32]
      S[g,n] = sc[g,n]*d[n];  M[g,n] = mn[g,n]*dmin[n];  w(n,k) = q4*S[k/32,n] + M[k/32,n]
      nibble byte index = k*16 + n/2, low nibble = even n, high = odd n
    """
    t = np.frombuffer(tile, dtype=np.uint8)
    if t.size != TILE:
        raise ValueError(f"mode-0 tile must be {TILE} B, got {t.size}")
    sc = t[0x000:0x100].reshape(8, 32).astype(np.float32)
    mn = t[0x100:0x200].reshape(8, 32).astype(np.float32)
    nib = t[0x200:0x1200]
    d = bf16_to_f32(t[0x1200:0x1240].view(np.uint16))
    dmin = bf16_to_f32(t[0x1240:0x1280].view(np.uint16))
    S = sc * d[None, :]                                   # (8, 32)
    M = mn * dmin[None, :]
    q4 = np.empty((256, 32), dtype=np.float32)            # [k][n]
    q4[:, 0::2] = (nib & 0x0F).astype(np.float32).reshape(256, 16)
    q4[:, 1::2] = (nib >> 4).astype(np.float32).reshape(256, 16)
    g_of_k = np.arange(256) // G
    return np.ascontiguousarray((q4 * S[g_of_k, :] + M[g_of_k, :]).T)   # (32, 256)


def mode0_parts(tile):
    """The tile's four tables, decoded: S (8,32), M (8,32), q4 (256,32) unsigned, d, dmin."""
    t = np.frombuffer(tile, dtype=np.uint8)
    sc = t[0x000:0x100].reshape(8, 32).astype(np.float32)
    mn = t[0x100:0x200].reshape(8, 32).astype(np.float32)
    nib = t[0x200:0x1200]
    d = bf16_to_f32(t[0x1200:0x1240].view(np.uint16))
    dmin = bf16_to_f32(t[0x1240:0x1280].view(np.uint16))
    q4 = np.empty((256, 32), dtype=np.uint8)
    q4[:, 0::2] = (nib & 0x0F).reshape(256, 16)
    q4[:, 1::2] = (nib >> 4).reshape(256, 16)
    return dict(S=sc * d[None, :], M=mn * dmin[None, :], q4=q4, d=d, dmin=dmin, sc=sc, mn=mn)


# ------------------------------------------------ 2. our integer kernels
def pb16i_plan_from_mode0(tile):
    """EXACT plan: keep the source's codes and scales, fold the additive min into a host-side table.

    codes  = q4 - 8                  (unsigned 0..15 -> signed -8..7, exactly representable by the kernel's int4)
    ms     = S / 2^e                 (int16 mantissa; the only lossy step, bounded by 2^-15 relative)
    add    = M + 8*S                 (host table: the source's additive min plus the code-shift correction)
    Returns dict(codes (32,256) int8, ms (8,32) int16, e float, add (8,32) float32).
    """
    p = mode0_parts(tile)
    S, M, q4 = p["S"], p["M"], p["q4"]
    smax = float(np.abs(S).max())
    e = float(np.floor(np.log2(smax)) - 10) if smax > 0 else 0.0     # ~11 bits of mantissa headroom
    ms = np.clip(np.round(S / (2.0 ** e)), -32768, 32767).astype(np.int32)
    # The kernel's int32 partials must not overflow: |P| <= 8 * 32 * |A|max, guaranteed by the int8 activations.
    codes = (q4.astype(np.int32) - 8).T.astype(np.int8)               # (32, 256)
    add = (M + 8.0 * S).astype(np.float32)                           # (8, 32)
    return dict(codes=codes, ms=ms.astype(np.int16), e=e, add=add)


def pb16i_plan(w):
    """FALLBACK plan for weights that only exist as floats (re-quantisation, NOT exact -- measure it).

    Same contract as above; the per-32-group scale is the group's absmax/7 and the additive table holds the
    fitted residual mean.  The 4-bit grid is the loss here, not our format: a requantised copy of an int4 source
    is never the same function, which is why the head is converted from the source codes and not from floats.
    """
    w = np.asarray(w, dtype=np.float32)
    n_rows, K = w.shape
    wg = w.reshape(n_rows, K // G, G)
    s = np.abs(wg).max(axis=2) / 7.0                                  # (n, g)
    s = np.where(s > 0, s, 1e-30)
    smax = float(s.max())
    e = float(np.floor(np.log2(smax)) - 10)
    ms = np.clip(np.round(s.T / (2.0 ** e)), -32768, 32767).astype(np.int16)     # (g, n)
    deq = ms.astype(np.float32) * (2.0 ** e)
    codes = np.zeros((n_rows, K), dtype=np.int8)
    add = np.zeros((K // G, n_rows), dtype=np.float32)
    for g in range(K // G):
        q = np.clip(np.round(wg[:, g, :] / deq[g, :, None]), -8, 7)
        codes[:, g * G:(g + 1) * G] = q.astype(np.int8)
        add[g, :] = (wg[:, g, :] - q * deq[g, :, None]).mean(axis=1)
    return dict(codes=codes, ms=ms, e=e, add=add)


def gemv_pb16i(a, plan):
    """One GEMV as the device computes it (per-group int32 partials, int16 magnitudes), plus the host term."""
    a = np.asarray(a, dtype=np.float32)
    K = a.size
    n_rows = plan["codes"].shape[0]
    assert K == plan["codes"].shape[1]
    y = np.zeros(n_rows, dtype=np.float64)
    for g in range(K // G):
        a_g = a[g * G:(g + 1) * G].astype(np.float64)
        P = plan["codes"][:, g * G:(g + 1) * G].astype(np.float64) @ a_g        # int32 on the device
        y += P * (plan["ms"][g, :].astype(np.float64) * (2.0 ** plan["e"]))
        y += a_g.sum() * plan["add"][g, :].astype(np.float64)                   # host group-sum term
    return y.astype(np.float32)


# ------------------------------------------------ 3. the head, one step
def head_step(hidden, tok_emb, W, cache=None, pos=None, g=GEOM):
    """One MTP draft step: (logits|None, new_hidden, cache).

    hidden : (H,)   base model's last hidden state after token t
    tok_emb: (H,)   base container's embedding row of token t
    W      : 'fc' (H,2H), 'q' (2DQ,H), 'k' (KV,H), 'v' (KV,H), 'o' (H,DQ), 'gate','up' (IM,H), 'down' (H,IM),
             'pre_fc_emb','pre_fc_hid','ln_in','ln_post','norm' (H,), 'q_norm','k_norm' (HD,),
             optional 'lm_head' (V,H) -- absent means the caller scores the hidden state some other way.
    cache  : {'k': (NKV,T,HD), 'v': (NKV,T,HD)} of earlier steps, or None
    """
    H, HD, NH, NKV, IM = g["H"], g["HD"], g["NH"], g["NKV"], g["IM"]
    DQ, KV = NH * HD, NKV * HD
    hidden = np.asarray(hidden, dtype=np.float32)
    tok_emb = np.asarray(tok_emb, dtype=np.float32)

    # input fusion: [embedding block | hidden block], each RMS-normed by its own pre_fc_norm
    u = np.concatenate([rms(tok_emb, W["pre_fc_emb"]), rms(hidden, W["pre_fc_hid"])]).astype(np.float32)
    y = (W["fc"] @ u).astype(np.float32)

    hn = rms(y, W["ln_in"])
    qg = (W["q"] @ hn).astype(np.float32)
    q, gate = qg[:DQ].reshape(NH, HD), qg[DQ:].reshape(NH, HD)     # q first, then the output gate
    k = (W["k"] @ hn).astype(np.float32).reshape(NKV, HD)
    v = (W["v"] @ hn).astype(np.float32).reshape(NKV, HD)
    q = rms(q, W["q_norm"])                                       # QK-norm per head over head_dim
    k = rms(k, W["k_norm"])
    if pos is None:
        pos = 0 if cache is None else cache["k"].shape[1]
    q = rope_partial(q[:, None, :], [pos], g["ROT"], g["THETA"])[:, 0, :]
    k = rope_partial(k[:, None, :], [pos], g["ROT"], g["THETA"])[:, 0, :]
    kc = k[:, None, :] if cache is None else np.concatenate([cache["k"], k[:, None, :]], axis=1)
    vc = v[:, None, :] if cache is None else np.concatenate([cache["v"], v[:, None, :]], axis=1)

    scale = np.float32(1.0 / np.sqrt(HD))                          # 1/16 at head_dim 256 (descriptor: 0.0625)
    rep = max(1, NH // NKV)
    out = np.empty((NH, HD), dtype=np.float32)
    for h in range(NH):
        kh = h // rep
        s = (kc[kh] @ q[h]) * scale
        p = np.exp(s - s.max())
        p /= p.sum()
        out[h] = p.astype(np.float32) @ vc[kh]
    out = (out.reshape(DQ) * (1.0 / (1.0 + np.exp(-gate.reshape(DQ))))).astype(np.float32)
    y = (y + W["o"] @ out).astype(np.float32)

    hp = rms(y, W["ln_post"])
    y = (y + W["down"] @ (silu((W["gate"] @ hp).astype(np.float32)) *
                          (W["up"] @ hp).astype(np.float32))).astype(np.float32)

    yf = rms(y, W["norm"])
    logits = None if "lm_head" not in W else (W["lm_head"] @ yf).astype(np.float32)
    return logits, y, {"k": kc, "v": vc}


def naive_head_step(hidden, tok_emb, W, cache=None, pos=None, g=GEOM):
    """The same step written straight from the math, sharing no structure with head_step (self-test only)."""
    H, HD, NH, NKV, IM = g["H"], g["HD"], g["NH"], g["NKV"], g["IM"]
    DQ, KV = NH * HD, NKV * HD
    rn = lambda x, w: x / np.sqrt(np.mean(np.square(x)) + g["EPS"]) * w
    e = rn(tok_emb, W["pre_fc_emb"])
    h = rn(hidden, W["pre_fc_hid"])
    x = W["fc"] @ np.concatenate([e, h])
    n1 = rn(x, W["ln_in"])
    qq = (W["q"] @ n1)[:DQ]
    gg = (W["q"] @ n1)[DQ:]
    kk = W["k"] @ n1
    vv = W["v"] @ n1
    qs = np.stack([rn(qq[i * HD:(i + 1) * HD], W["q_norm"]) for i in range(NH)])
    ks = np.stack([rn(kk[i * HD:(i + 1) * HD], W["k_norm"]) for i in range(NKV)])
    vs = np.stack([vv[i * HD:(i + 1) * HD] for i in range(NKV)])
    if pos is None:
        pos = 0 if cache is None else cache["k"].shape[1]
    qs = rope_partial(qs[:, None, :], [pos], g["ROT"], g["THETA"])[:, 0]
    ks = rope_partial(ks[:, None, :], [pos], g["ROT"], g["THETA"])[:, 0]
    K = ks[:, None, :] if cache is None else np.concatenate([cache["k"], ks[:, None, :]], 1)
    V = vs[:, None, :] if cache is None else np.concatenate([cache["v"], vs[:, None, :]], 1)
    ao = []
    for i in range(NH):
        j = i * NKV // NH
        s = (K[j] @ qs[i]) / np.sqrt(HD)
        p = np.exp(s - s.max())
        p /= p.sum()
        ao.append(p @ V[j])
    x = x + W["o"] @ (np.concatenate(ao) / (1.0 + np.exp(-gg)))
    n2 = rn(x, W["ln_post"])
    x = x + W["down"] @ (silu(W["gate"] @ n2) * (W["up"] @ n2))
    yf = rn(x, W["norm"])
    logits = None if "lm_head" not in W else W["lm_head"] @ yf
    return logits, x, {"k": K, "v": V}


# ------------------------------------------------ self-test
def _rand(*shape, scale=0.1, seed=0):
    return (np.random.default_rng(seed).standard_normal(shape) * scale).astype(np.float32)


def _make_tile(seed=0):
    """A synthetic mode-0 tile (random tables) -- lets the decoder and the exact plan be checked with no weights."""
    rng = np.random.default_rng(seed)
    sc = rng.integers(0, 256, size=(8, 32)).astype(np.uint8)
    mn = rng.integers(0, 256, size=(8, 32)).astype(np.uint8)
    q4 = rng.integers(0, 16, size=(256, 32)).astype(np.uint8)
    d = (rng.standard_normal(32) * 0.02).astype(np.float32)
    dmin = (rng.standard_normal(32) * 0.002).astype(np.float32)
    t = bytearray(TILE)
    t[0x000:0x100] = sc.tobytes()
    t[0x100:0x200] = mn.tobytes()
    nib = (q4[:, 0::2] | (q4[:, 1::2] << 4)).astype(np.uint8)
    t[0x200:0x1200] = nib.tobytes()
    t[0x1200:0x1240] = f32_to_bf16_bits(d).tobytes()
    t[0x1240:0x1280] = f32_to_bf16_bits(dmin).tobytes()
    return bytes(t)


def _micro(seed=0):
    """A tiny head with the same structure, and the geometry dict that goes with it."""
    g = dict(H=8, IM=8, HD=4, NH=2, NKV=1, ROT=2, THETA=1e7, EPS=1e-6)
    H, HD, NH, NKV = g["H"], g["HD"], g["NH"], g["NKV"]
    DQ, KV, IM = NH * HD, NKV * HD, g["IM"]
    W = {
        "fc": _rand(H, 2 * H, seed=seed + 1), "q": _rand(2 * DQ, H, seed=seed + 2),
        "k": _rand(KV, H, seed=seed + 3), "v": _rand(KV, H, seed=seed + 4),
        "o": _rand(H, DQ, seed=seed + 5), "gate": _rand(IM, H, seed=seed + 6),
        "up": _rand(IM, H, seed=seed + 7), "down": _rand(H, IM, seed=seed + 8),
        "lm_head": _rand(13, H, seed=seed + 9),
        "pre_fc_emb": (1 + _rand(H, seed=seed + 10)).astype(np.float32),
        "pre_fc_hid": (1 + _rand(H, seed=seed + 11)).astype(np.float32),
        "ln_in": (1 + _rand(H, seed=seed + 12)).astype(np.float32),
        "ln_post": (1 + _rand(H, seed=seed + 13)).astype(np.float32),
        "norm": (1 + _rand(H, seed=seed + 14)).astype(np.float32),
        "q_norm": (1 + _rand(HD, seed=seed + 15)).astype(np.float32),
        "k_norm": (1 + _rand(HD, seed=seed + 16)).astype(np.float32),
    }
    return W, _rand(H, seed=seed + 17), _rand(H, seed=seed + 18), g


def selftest():
    ok = True

    # (a) the tile formula, both directions: build a tile from known tables, decode, compare with the formula
    tile = _make_tile(seed=7)
    p = mode0_parts(tile)
    w = decode_mode0_tile(tile)
    g_of_k = np.arange(256) // G
    ref = (p["q4"].astype(np.float32) * p["S"][g_of_k, :] + p["M"][g_of_k, :]).T
    err = float(np.abs(w - ref).max())
    print(f"[a] mode-0 tile decode vs its own formula: max|diff| = {err:.3e}  {'OK' if err == 0 else 'FAIL'}")
    ok &= err == 0

    # (b) the EXACT conversion path: source codes/scales kept, additive min recovered by the host group sums
    a = _rand(256, seed=11)
    plan = pb16i_plan_from_mode0(tile)
    y_int = gemv_pb16i(a, plan)
    y_f32 = (w.astype(np.float64) @ a.astype(np.float64))
    rel = float(np.abs(y_int - y_f32).max() / (np.abs(y_f32).max() + 1e-30))
    print(f"[b] exact plan (source int4 codes + int16 ms + host add-term) vs float GEMV: rel max err = {rel:.3e}  "
          f"{'OK' if rel < 5e-3 else 'FAIL'}")
    ok &= rel < 5e-3

    # (c) the fallback requantisation: what it costs, measured rather than assumed
    wr = _rand(32, 256, scale=0.05, seed=12)
    plan_r = pb16i_plan(wr)
    y_r = gemv_pb16i(a, plan_r)
    y_t = wr.astype(np.float64) @ a.astype(np.float64)
    rms_err = float(np.sqrt(np.mean((y_r - y_t) ** 2)) / (np.sqrt(np.mean(y_t ** 2)) + 1e-30))
    # The 4-bit grid itself is the cost here (a Gaussian draws ~8-10 % RMS at 4 bits); the point of the test is
    # that (b) does not pay it, so the threshold is loose and the number is reported, not tuned.
    print(f"[c] requantisation fallback (int4 grid, not the conversion path): rel RMS err = {rms_err:.3e}  "
          f"{'OK' if rms_err < 1.5e-1 else 'FAIL'}   <- why the head is converted from the source codes")
    ok &= rms_err < 1.5e-1

    # (d) structured step vs naive step on a micro head, then a second step against the kv cache
    W, hid, emb, g = _micro()
    l1, h1, c1 = head_step(hid, emb, W, pos=0, g=g)
    l2, h2, c2 = naive_head_step(hid, emb, W, pos=0, g=g)
    d1, d2 = float(np.abs(l1 - l2).max()), float(np.abs(h1 - h2).max())
    print(f"[d] head_step vs naive (step 1): logits {d1:.3e}, hidden {d2:.3e}  {'OK' if max(d1, d2) < 1e-5 else 'FAIL'}")
    ok &= max(d1, d2) < 1e-5
    l1, h1, c1 = head_step(h1, emb, W, cache=c1, g=g)
    l2, h2, c2 = naive_head_step(h2, emb, W, cache=c2, g=g)
    d3 = float(np.abs(l1 - l2).max())
    print(f"[e] head_step vs naive (step 2, kv cache): logits {d3:.3e}  {'OK' if d3 < 1e-5 else 'FAIL'}")
    ok &= d3 < 1e-5

    # (f) shipped geometry, printed so a run against the real artifacts is self-documenting
    H, IM, HD, NH, NKV = GEOM["H"], GEOM["IM"], GEOM["HD"], GEOM["NH"], GEOM["NKV"]
    print(f"[f] shipped geometry: H={H} IM={IM} HD={HD} NH={NH} NKV={NKV} rope={GEOM['ROT']}/{HD} "
          f"-> q {2*NH*HD} k/v {NKV*HD} o-in {NH*HD} attn-scale 1/{int(np.sqrt(HD))} | fc in {2*H}")
    print("all checks passed" if ok else "SOME CHECKS FAILED")
    return ok


def main(argv):
    if "--selftest" in argv:
        return 0 if selftest() else 1
    print(__doc__)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
