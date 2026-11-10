// Copyright 2026 bong-water-water-bong
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// `1bit npu-pack`: turn a Ryzen AI OGA NPU model pack (`*_rai_*_npu_*`, the
// `model.onnx` + `model.onnx.data` + `dd_metastate_*.fconst` + `genai_config.json`
// set AMD publishes) into the engine's Q4NX model directory. The pack's weights
// and its tokenizer/config contract are the model; every line of this reader is
// written from the documented facts in
// amd-oss/npu-re: docs/FACTS-oga-pack-weight-layout.md (pack-side layout) and
// docs/npu.md here (container side). No vendor library, wheel or compiled
// program is called, linked or shipped.
//
// Two commands:
//   npu-pack info   <pack dir>                 what the pack says, as JSON
//   npu-pack import <pack dir> -o <out dir>    write model.q4nx + config.json
//
// A pack the importer cannot represent is refused by name, with the reason and
// the tensor, and nothing is written: a container that silently carries 1 % of
// a model is worse than no container. `--allow-incomplete` writes what is
// representable and records the rest in the container's manifest, so the gap is
// visible to whatever loads it next.
#pragma once

#include <nlohmann/json.hpp>

#include <cstdint>
#include <string>

#include <nlohmann/json.hpp>
#include <map>
#include <vector>

namespace onebit::npu::pack_import {

// Is [offset, offset+length) inside a mapped file of `size` bytes? Both comparisons are overflow-proof —
// each is against a value already smaller than `size` — so a crafted `model.onnx` cannot wrap them.
// (`offset + length > size` does wrap: offset = UINT64_MAX - 4, length = 8 sums to 3.)
inline bool ext_range_ok(uint64_t size, uint64_t offset, uint64_t length) {
    return offset <= size && length <= size - offset;
}

// a*b without wrapping. Every read whose length comes from the file's own metadata (a shape product, N*K/block,
// N*(K/block)*block/2) is computed through this, so a crafted model.onnx cannot wrap a length into a small
// allocation and then fill it with the declared count.
inline bool mul_ok(uint64_t a, uint64_t b, uint64_t& out) {
    if (a != 0 && b > UINT64_MAX / a) return false;
    out = a * b;
    return true;
}

// The model geometry the container's config.json needs.
struct Dims {
    int hidden = 0, layers = 0, heads = 0, kv_heads = 0, head_dim = 0, intermediate = 0, vocab = 0;
    float rope_theta = 0;
    // config.json's rms_norm_eps, and the two bytes it is stored as in the .fconst's block headers.
    // engine#734: the layer-block locator looked for bf16 1e-6 (86 35) only, which is what the Qwen
    // packs declare; Llama-3.2-1B declares 1e-5 (28 37) and so had no anchors at all.
    float rms_norm_eps = 1e-6f;
    unsigned short eps_bits = 0x3586;
    std::string model_type;
    std::vector<int> eos;
    // config.json's `use_qk_norm` (engine#1889). Hunyuan-7B declares it and puts 32x128 + 8x128 per-head q/k
    // gains in its block; nothing read this key, so the importer decided `qk_norm` from layout arithmetic alone,
    // landed on false, and the pack imported cleanly with the norms silently omitted -- the forward then runs
    // without normalising q/k, which produces deterministic gibberish rather than a refusal. Read here so the
    // decision has the pack's own statement as one of its inputs.
    bool use_qk_norm = false;
};

// Why a pack that DECLARES per-head q/k gains cannot be imported, or "" when it can (engine#1889).
//
// `declared` is config.json's `use_qk_norm`; `found` says the block layout located the two sections (knorm_off /
// qnorm_off) or the graph carries them. The three cases are deliberately separate: declared-and-not-found is a
// REFUSAL (importing without the gains is the silent-garbage path), found-and-not-declared is allowed only with the
// measurement behind it (Qwen3 packs ship the sections without the key), and neither is the ordinary case. Exposed
// so the three can be pinned without a 4 GB pack.
// The gap is `[64 B zeros][q_norm bf16 [head_dim]][64 B zeros][k_norm bf16 [head_dim]]`: q first (engine#1934, measured
// against the q_norm/k_norm initializers in Hunyuan-7B's own model.onnx).
struct QkGainOffsets {
    int64_t k_norm = 0, q_norm = 0;  // 0,0 = the gap is not the gains' shape
};
QkGainOffsets qk_gain_offsets(int64_t gap_start, int64_t head_dim, int64_t gap_bytes);
std::string qk_norm_refusal(bool declared, bool found);

// A hybrid pack's embedding table lives in its own graph (`embedding.onnx` + `.data`, gathered with
// GatherBlockQuantized). Returns [V, H] f32 row-major, empty when the pack has no such component, with `why` naming the
// reason. Exposed so the dequantisation can be tested against known codes and scales instead of only on a 3 GB pack.
std::vector<float> read_embedding_component(const std::string& pack_dir, std::string* why);

// One tensor the pack describes, and where its values would come from.
struct Planned {
    std::string name;     // Q4NX tensor name (model.layers.N.<...>.weight)
    std::string source;   // "model.onnx.data", "fconst:Llm_Token_Token", "fconst:Llm_Prefill", "config.json"
    std::string dtype;    // "q4_1" tiles, or "BF16"
    std::vector<int64_t> shape;
    std::string refusal;  // empty when importable; otherwise the named reason
    // What the pack's own cache/*_meta.json declares for this tensor (attrs N/K and the const
    // tensor's byte size), when it declares it. The pack stating its own geometry is a stronger
    // check on our plan than our own arithmetic, and it is what engine#734 needs to generalise
    // the .fconst offsets beyond hidden 896.
    int64_t declared_n = 0, declared_k = 0;
    std::string declared_check;  // empty when it agrees, else what disagrees
};

struct Plan {
    std::string dir;
    // The MatMulNBits block size along K the pack declares for its LM HEAD, when that is not the 128 this importer's
    // .fconst tile geometry assumes. Non-zero means the tail's head tiles cannot be walked (engine#1503): a tied pack
    // takes the head from its embedding instead, an untied one is refused by name.
    int64_t head_block = 0;
    std::string pack;  // *_rai_1.8.0_npu_4K, from the directory name
    Dims dims;
    std::vector<Planned> tensors;
    // The attention payload's geometry, MEASURED from the .fconst rather than assumed: how many 64-row
    // tiles follow the block's first epsilon header, and the reason when that does not match the shape
    // this decoder was established on. Zero tiles plus a note is what the projections' refusal cites.
    // (engine#734's lesson: `info` must report the measurement, not the plan.)
    int64_t attn_tiles = 0;
    std::string attn_note;
    // Where the four projections actually sit, once the section count is known (engine#979). k/q/v are
    // always in the block's first payload at K = hidden. o_proj has a payload of its own whose K is
    // q_proj's output width -- the same payload as q/k/v on the Qwen2/Llama families, a separate one
    // behind the two per-head norms on the Qwen3 families, where the pack itself declares
    // `o_proj N=2560 K=4096` (q_out, not hidden). o_off is relative to the block start.
    int64_t o_off = 0;
    int o_k = 0;
    // Per-head QK RMSNorm gains (Qwen3 and later): their offsets inside the block, 0 when the family has
    // none. Identified and verified byte-exact against the public Qwen3-1.7B/4B checkpoints (0 of 128
    // values differ on either pack, and the swap differs on 127-128 of 128).
    int64_t knorm_off = 0, qnorm_off = 0;
    int64_t mlp_off = 0;  // the MLP section starts just past the block's post-attention norm
    // What the pack's *graph* carries, reported so `npu-pack info` can answer a pack's genre without a
    // third-party parser: the rai_1.3/1.6/1.7.1 line ships its 4-bit weights as ordinary ONNX initializers, while
    // 1.8 and R1-Distill ship placeholders and keep the weights in model.bin (pi-inferzo, 2026-11-08).
    // graph_matmuls < 0 means "no graph was read", which is why it is not just a count.
    int64_t graph_matmuls = -1;      // MatMulNBits nodes in the main graph
    int64_t graph_real_weights = 0;  // of those, how many point at an initializer with dims (i.e. real weights)
    int64_t graph_external = 0;      // initializers that declare external data (location + offset)
    int64_t graph_external_bytes = 0;// size on disk of the external file they name, 0 when it is not in the pack
    // `com.ryzenai:MatMulNBitsBf` ops, and node inputs that resolved to a `*.qweight.packed` sibling behind a
    // `*.qweight.empty` placeholder. Both exist because a walker that matches only the Microsoft op name, or only
    // the node's own initializer, reports a weightless graph for the 1.9 external-data packs (engine#1503).
    int64_t graph_ryzenai_matmuls = 0;
    int64_t graph_packed_siblings = 0;
    int64_t graph_weightless_matmuls = 0;   // dropped: placeholder + packed sibling both byte-less (engine#1671)
    // How many of the graph's 4-bit block-128 weights satisfy length == N x (K/128 + 1) x 68 (the per-row stride the
    // .fconst tile also carries): a measurement, so `info` can say "201 of 201" or "3 of 7" instead of guessing.
    int64_t graph_stride_ok = 0, graph_stride_of = 0;
    std::map<int64_t, int64_t> graph_blocks;                 // block size -> how many ops declare it
    std::vector<std::vector<int64_t>> graph_weight_dims;     // up to 4 distinct [N, K/block, bytes] shapes
    // The MLP section's geometry, likewise measured: gate and up share one fused 4,288-byte-subvolume
    // tensor of `mlp_gu_units` x 128 columns, then a 64-byte parameter subvolume, then down_proj in
    // 2,176-byte subvolumes. All of it has to add up to the block stride exactly, which is the check
    // that keeps this from being an assumption about one pack.
    int64_t mlp_gu_units = 0, mlp_down_groups = 0;
    std::string mlp_note;

    std::vector<std::string> refused() const;
    std::string manifest() const;  // JSON: format, layout version, source, geometry, tensors
};

// The multi-component (gemma-4 1.9, `type: compositemodel`) packs keep the text tower's geometry in
// model_config.json -> config.model_attributes.text_config, while config.json is `{}` and genai_config.json's
// decoder block stops at head_size. Fills ONLY the keys `cfg` lacks and returns how many it added, so the caller
// can name the source; exposed for a unit test that needs no ONNX graph.
int fill_from_model_config(const std::string& pack_dir, nlohmann::json& cfg);

// The same value derived from the graph's own node attributes (MatMulNBits/MatMulNBitsBf K and N) when the
// initializers carry no dims -- the 1.9 external-data genre. Returns 0 when no FFN pair agrees, or when two
// widths do, so the caller refuses instead of guessing. Exported for the unit test, which needs no pack.
// Takes a PATH because it parses the graph: it is cheap only because plan() calls it once. If it ever moves into a
// per-tensor or per-layer path, that parse is paid per call -- keep it to one call per plan.
int intermediate_size_from_graph(const std::string& model_path, int64_t hidden);

// Where a pack keeps its Token .fconst: the root for a flat pack, the decoder component `genai_config.json` names for a
// multi-component one (`type: compositemodel`), whose geometry files stay at the root. Exported for the unit test.
std::string find_token_fconst_exported(const std::string& pack_dir);

// Reads the pack (model.onnx, its external data, both .fconst headers, config.json).
// Throws only when the pack is structurally unreadable.
Plan plan(const std::string& pack_dir);

// Whether the canonical Token .fconst already on the scratch path holds exactly the bytes this run would write,
// given the length the canonical form must have (engine#1662). True means the ~291 MB rewrite can be skipped. Split
// out of canonical_token_fconst() so the decision has a test that needs no 1.9 pack: a truncated copy, a missing file
// and a directory on the scratch path must all come back false.
bool canonical_copy_reusable(const std::string& work, int64_t expected_len);

// Removes the canonical Token .fconst copies in `dir` that nothing will reuse (engine#1770). Reuse writes a new copy when the
// source changes rather than overwriting, so without this the scratch directory grows one multi-hundred-MB file per change; on
// this box 699 files / 8.7 GiB of tmpfs. Three rules:
//   * share `prefix` (one pack path and geometry) but are not `keep_name` -- normal generations of a superseded source;
//   * equal `legacy_name` -- the generation before the source identity was part of the name
//     (`1bit-fconst19-<tag>-<h>-<heads>-<layers>.fconst`), which `prefix` cannot match because it was written without the version
//     suffix, so those files were invisible to this sweep and stayed resident (three of them, 10.67 GiB, on 2026-11-10);
//   * start with `1bit-fconst19` and are older than `max_age_seconds` (0 disables) -- a pack nobody imports again. A write-triggered
//     sweep only reaches the geometry being written, so the abandoned tail needs an age rule.
// Exposed for the same reason as canonical_copy_reusable(): the rule it applies is what can be wrong, and a test can prove it on a
// directory of empty files with no pack at all.
void evict_stale_canonical_copies(const std::string& dir, const std::string& keep_name, const std::string& prefix,
                                  const std::string& legacy_name  = std::string(),
                                  int64_t            max_age_seconds = 0);

// The text artifacts a runnable container needs beside model.q4nx, and which of them the pack
// actually carries. A pack fetched weight by weight can be weights-only (engine#1024; AMD's repos
// do publish the text, and how much of it varies per pack), and an importer that says nothing about
// it writes a container that only fails later, in another command, as "not an NPU model
// directory" -- which reads like our bug.
struct TextArtifacts {
    std::vector<std::string> present, missing;  // both in the importer's own file order
    bool tokenizer = false;                     // tokenizer.json present: what `1bit serve` opens
};
TextArtifacts scan_text_artifacts(const std::string& pack_dir);
// engine#1751 follow-up: the one text-artifact copier both import paths use. Copies every kTextFiles entry the pack has
// into out_dir and returns how many; a file the pack has that cannot be copied throws, naming it (a skipped file is not
// a failure -- packs legitimately ship subsets). Exported so a test can drive the failure path without a pack.
int copy_text_artifacts_exported(const std::string& pack_dir, const std::string& out_dir);

// engine#1740 (W1a): where the 1.9 Token .fconst keeps its output layer. The arithmetic is exposed because it is the
// half that can be wrong silently -- a one-tile error reads a plausible region and only a correlation against the
// embedding notices -- so it is pinned by a host test with no pack, no device and no fixture.
struct TailHeadPlacement {
    int64_t first = -1;   // earliest tile-aligned candidate (== tail_off + env)
    int64_t last = -1;    // candidate the trailing zero run implies
    int64_t rows = 0;
    int64_t zeros = 0;
    int64_t have = 0;
    int64_t want = 0;
    bool exact = false;   // have == want: the head ends the file
    std::string why;      // non-empty when no candidate exists
};
TailHeadPlacement plan_tail_head(int64_t bytes, int64_t tail_off, int64_t env, int64_t rows, int64_t tile,
                                 int64_t zeros);

// The search half: a synthetic tied head (built through the tile encoder) is located, then displaced by one tile and
// still located, then handed over as untied where padding must refuse and an exact tail must be taken unverified.
// Returns false with *why set. No pack, no device, no fixture.
bool tail_head_selftest(std::string* why);

struct Options {
    bool allow_incomplete = false;  // write what is representable and name the rest
    // The device weight layout the container will declare (`q4nx_weight_layout`). Empty means the
    // default row-major family, in which case the key is left out entirely -- as every container the
    // importer has written before this option existed. "b8" is the one layout a bare flag can
    // express: it needs no version and no sidecar image directory (see weight_layout_key).
    std::string weight_layout;
    // A kernel set to link into the container as `<out>/npu` (`--kernels`). A model directory carries
    // its own NPU kernels in `npu/`, and until now filling that in after an import was a hand-made
    // symlink: the ONE step between `npu-pack import` and `1bit serve -m <dir> --device npu`, done by
    // whoever read the docs closely enough. Empty means "don't", as every earlier import behaved.
    std::string kernels;
};

// Why a layout name cannot be declared by `--weight-layout`, or "" when it can. Only "b8" and the
// default "rm" qualify: t2s and pb16i also need `q4nx_weight_layout_version` and/or a sidecar image
// directory (`q4nx_pb16i_dir`), which an import cannot invent, so offering them here would produce a
// container that refuses to load.
std::string weight_layout_refusal(const std::string& want);

// Why a directory cannot be linked in as this container's kernel set, or "" when it can. `layout` is the
// same value `--weight-layout` took ("b8" or empty), because the container and the kernel set have to
// agree on the family: a container declaring "b8" needs `_b8` ELFs (the forward looks for exactly those
// names), and a container with no layout key needs the plain row-major ones. A set of the wrong family
// links fine and then fails at load with "no ELF for K=.. N=.." -- the kind of error that reads like a
// broken container, so it is refused here, by name, before anything is written.
std::string kernel_set_refusal(const std::vector<std::string>& file_names, const std::string& layout);

struct Report {
    std::string out_dir;
    int tensors = 0;
    long long tiles = 0;
    std::vector<std::string> refused;

    // Self-check 1: the 4-bit decoder against the pack's own independent data. With
    // tie_word_embeddings the pack's lm_head must reproduce its fp16 embedding, so the
    // decode is checked against the embedding, and two wrong decoders are run as controls.
    // An untied pack has no such reference and the numbers are left at 0 with
    // selfcheck_reference false -- a wrong number would be worse than none.
    bool selfcheck_reference = false;
    // Whether that reference is a real one: a tied pack's head must reproduce its embedding, so the correlation is a
    // check; an untied pack's head and embedding are separate matrices by construction, and the same number is
    // meaningless -- reported as such rather than left looking like a failed import.
    bool selfcheck_tied = false;
    double lmhead_corr = 0, lmhead_rel = 0, ctrl_corr_nibble = 0, ctrl_corr_nozp = 0;
    // The tail (.fconst) path has no MatMulNBits bytes to mis-decode, so its control is an ALIGNMENT one: the same
    // tiles, the same decoder, rows matched against the neighbouring embedding row. A correct decode correlates ~1
    // with the embedding; if a row-misaligned head still did, the check would be measuring nothing. -1 = not run.
    double ctrl_corr_rowshift = -1;

    // Self-check 2: the written container, read back, against the pack's dequantised values. With
    // the fp32 q4_1f form (engine#1185) q4nx_rel and q4nx_max_abs are 0 and zero_exact_pct is 100:
    // the bf16 floor described here before was the GGUF path's, not the import's.
    double q4nx_rel = 0, q4nx_max_abs = 0, zero_exact_pct = 0;
    long long codes_exact = 0, codes_total = 0;

    // The .fconst layer-block geometry, measured from the RMSNorm epsilon anchors rather than assumed
    // (engine#734): the constants measured on one pack are functions of hidden size and of each
    // projection's N and K, so they only work for that pack. stride/second_header are 0 when the
    // anchors could not be resolved, with fconst_note saying why.
    int64_t stride = 0, second_header = 0;
    std::string fconst_note;
    int norm_anchors_ok = 0, norm_anchors_total = 0;
    // The final norm is the last tensor the .fconst tail carries. `final_norm_crosscheck` is only
    // meaningful when both facts below hold, so they are reported separately rather than inferred from it:
    // false means "no agreeing duplicate", which also covers "no duplicate to compare" and "the norm was
    // refused", and those are different things to say about an import.
    bool final_norm_imported = false;      // model.norm.weight was written into the container
    // Why lm_head was synthesised rather than read (engine#1503: the pack's head is at a block this importer cannot
    // walk and the pack is tied). Empty means it was read, as usual. Carried on the Report because the config JSON is
    // built after the .fconst block that discovers this.
    std::string head_synthesized_note;
    bool prefill_tail_seen = false;        // a second copy was readable in the prefill .fconst
    bool final_norm_crosscheck = false;    // ... and it is byte-identical to the Token_Token copy

    // Self-check 3: the attention tiles' own round trip. Decoding a shipped tile and re-encoding it has to
    // give the same 34,816 bytes back, every field at once (quants, zero points, scales and the writer's
    // bias slots) -- so it fails if any index, nibble order or field offset in decode_tile() is wrong.
    long long attn_tiles_ok = 0, attn_tiles_total = 0;
    int attn_projections = 0, attn_bias_tensors = 0;
    double attn_zero_inexact_pct = 0;  // per-32-group parameters not stored exactly (0 with the fp32 form)

    // Self-check 4: the same round trip over the MLP section -- every 128-column gate/up unit and every
    // 64-column down_proj group decoded and re-encoded, which has to give the shipped bytes back.
    long long mlp_groups_ok = 0, mlp_groups_total = 0;
    int mlp_projections = 0;
    double mlp_zero_inexact_pct = 0;
};

Report import_pack(const std::string& pack_dir, const std::string& out_dir, const Options&);

// `1bit npu-pack <info|import> ...`; argv[0] is the subcommand.
int run(int argc, char** argv);

// Exposed for the tests: a decoded MatMulNBits tensor as (codes, scale, zero) per
// 32-column group, ready to tile.
struct Q4 {
    int N = 0, K = 0, block = 0;
    std::vector<uint8_t> codes;  // [N, K], 0..15
    std::vector<float> scale;    // [N, K/32]
    std::vector<float> zero;     // [N, K/32]
};
// The 1.9 external-data tile (engine#1503): the `.fconst` tile's subvolume written densely per tensor.
//
// Measured on `amd/phi-4_rai_1.9.0_npu` (`eager-mm.onnx.data`, 9.74 GB): every `MatMulNBitsBf` weight is a 1-D byte
// blob whose length is `N x (K/128 + 1) x 68` -- i.e. `N/64` tiles of `2*(K/128 + 1)` subvolumes of 2,176 B. Inside a
// subvolume: 2,048 B of 4-bit codes, 64 B of int4 zero points (four per row, one per 32-column group), 64 B of bf16
// scales (one per row). The `k = 0` subvolume pair of every tile is the writer's bias pair, zero on a pack whose
// matmuls carry no bias. `decode_external_tile` returns the whole matrix as a `Q4` (block 32: the 128-column scale is
// repeated over its four groups and each group keeps its own zero point, which is what the pack states).
constexpr int kExtSubvBytes = 2176;   // one subvolume: 2,048 B quants + 64 B int4 zero points + 64 B bf16 scales
constexpr int kExtSubvRows = 32;
constexpr int kExtTileRows = 64;      // two subvolumes per tile per K block
// Which way one row's 128 code bytes lie inside the 2,048-byte quants region. The default is the .fconst tile's
// verified formula; the alternative is here so the known-answer run can settle it without touching addressing.
constexpr bool kExtQuantsRowMajor = false;

// `row_major` defaults to kExtQuantsRowMajor (the shipped choice); it is a parameter only so the test can hold BOTH
// orderings to their own spec -- a flip of the constant must not quietly change what the other branch means.
// Turn a decoded external-data tile into the container's q4_1 parts. The two conventions differ in what zero
// means: the container dequantises `w = code*d + m` (m is the group's MIN, npu/q4nx.h), while the pack's is
// `w = (code - zp)*scale` (zp is an int4 CODE offset). So the mapping is `d = scale`, `m = -zp*scale`, codes
// unchanged -- and the obvious alternative, dequantise then re-quantise with the container's own quantiser, is
// measurably NOT the same weights: on a real phi-4 tile it reproduced the pack's codes on only 2,826 of 8,192
// entries and moved weights by 2.5 % of the tensor's range (engine#1503). `m_inexact` reports how many group
// offsets were not bf16-exact (86.3 % are, measured), because that rounding is the only loss this path has.
Q4 container_from_external_q4(const Q4& pack, long long* m_inexact = nullptr);

// The container role a matmul plays, read off the graph's own tensor name: `...qkv_proj...` -> "QKV",
// `o_proj` -> "O", `gate_proj` -> "G", `up_proj` -> "U", `down_proj` -> "D", `lm_head` -> "HEAD". These are the
// roles the design/tile writer already speaks, so the external-data genre's writer is a mapping and not a second
// vocabulary. nullptr when the name is one this importer has not measured: reported BY NAME rather than skipped,
// because a projection nothing maps is a weight that would silently not be written (engine#1503).
const char* external_role_of(const std::string& tensor_name);

// Where a pack projection's rows go in the container: the container stores `self_attn.q_proj.weight` and its
// siblings separately (only a GDN layer has a fused `linear_attn.qkv_proj.weight`), while the pack stores q/k/v as
// ONE matmul -- phi-4's is N 7680 with 40 heads x 128 (q 5120) and 10 kv heads x 128 (k 1280, v 1280). So a role
// is one container tensor except "QKV", which is three with row ranges taken from the geometry the pack states.
// Names are the forward's own (`self_attn.{q,k,v,o}_proj.weight`, `mlp.{gate,up,down}_proj.weight`,
// `lm_head.weight`), read from npu/forward/src/q4nx_forward.cpp rather than invented.
struct ContainerTensor {
    std::string suffix;   // relative to `model.layers.<L>.`, or a model-level name for HEAD
    int row_lo = 0, row_hi = 0;   // rows of the pack's matmul this tensor takes (its own N when split is 1)
};
// False + `err` for a role with no container home (the caller must then refuse rather than drop the weights).
bool external_container_tensors(const std::string& role, int nh, int nkv, int hd, std::vector<ContainerTensor>& out,
                                std::string& err);

Q4 decode_external_tile(const uint8_t* data, size_t bytes, int N, int K, std::string& err,
                        bool row_major = kExtQuantsRowMajor);
Q4 decode_matmulnbits(const uint8_t* qweight, const float* scales, const uint8_t* zeros,
                      int N, int K, int block, bool swap_nibbles = false, bool drop_zero = false);

// One 64-row tile of the .fconst attention payload, as the pack stores it. Established byte-exactly
// (amd-oss/npu-re docs/FACTS-oga-pack-weight-layout.md §7.13, §7.15): the tile is 2*(K/128)+2 subvolumes
// of 2,176 B = one pair per (column half, K block of 128) plus the pair holding the writer's bias input.
// Codes and zero points are stored as int4 (two's complement); `scale` is bf16 per (row, K block).
// Both narrow fields pair with rows the same way -- byte `i` of a subvolume carries rows 2i and 2i+1
// (low nibble = even row) -- so a per-row bug in one of them cannot hide behind the other, and a
// decode->encode round trip cannot see the pairing at all (engine#849).
//
// The same subvolume also carries the MLP section at a second width: `rows` is the group's column
// count, which fixes the subvolume's own width (rows/2 = 32 columns at 2,176 B, 64 at 4,288 B).
struct TileW {
    int k = 0;                  // the projection's K
    int rows = 64;              // columns covered: 64 in the attention payload and down_proj, 128 where
                                // gate and up are fused into one 4,288-byte subvolume
    std::vector<uint8_t> code;   // [rows][k] stored nibbles
    std::vector<float> scale;    // [rows][k/128], widened from bf16 (exact, and bf16 again on the way back)
    std::vector<uint8_t> zp;     // [rows][k/128] stored zero points
    std::vector<uint16_t> bias;  // [rows] bf16
};
// Bytes in one group of that layout: 2*(k/128) + 2 subvolumes, each 64*(rows/2) + 64 + rows bytes.
// 0 when k is not a positive multiple of 128 or rows is not a positive multiple of 64, i.e. when the
// payload cannot be groups of this shape.
int64_t group_bytes(int64_t k, int rows);
// Byte count below is the 64-row tile: the attention payload's and down_proj's shape.
// The layer-block geometry of a `.fconst`, measured from its epsilon anchors (engine#734). The anchors
// are the pack's OWN bf16 rms_norm_eps, so `eps_bits` comes from its config.json via
// `bf16_bits_of()`; the default is bf16 1e-6, which is what the Qwen family declares.
struct FconstGeom {
    int64_t stride = 0;   // bytes per layer block
    int64_t h1 = 0;       // offset of the block's SECOND header (the post-attention one only when a block
                          // carries two; on a 4-header block it opens the k_norm section -- use
                          // post_attn_hdr() for the post-attention norm and the MLP section)
    int period = 0;       // headers per block
    int blocks = 0;       // block starts seen
    // Every in-block header offset after the block's first (so hdr[0] == h1), relative to the block
    // start. These are the block's section boundaries: 1 header on the Qwen2/Llama packs (q/k/v/o share
    // one payload) and 3 on the Qwen3 packs (q/k/v, then one per-head QK norm each, then o_proj), so
    // which layout a pack uses is measured from the pack rather than guessed from a version.
    std::vector<int64_t> hdr;
    // Where the block's post-attention section starts: `h1` on a 2-header block (Qwen2.5/Llama), the LAST
    // header on a 4-header block (Qwen3, where the second and third open the per-head QK norms). Both the
    // post_attention_layernorm weight and the first byte of the MLP section are read from here, so this is
    // the one rule the two readers share -- reading `h1` unconditionally imported the k_norm section's
    // padding as post_attention_layernorm on every Qwen3 pack (engine#1003).
    int64_t post_attn_hdr() const { return hdr.empty() ? h1 : hdr.back(); }
    std::string why;      // non-empty when the geometry could not be established
};
unsigned short bf16_bits_of(float v);

// engine#1650: config.json may declare an OPTIONAL field as JSON null (AMD's 1.9
// Mistral-7B-Instruct-v0.2 pack writes `"head_dim": null`), which must read as "absent" and take
// the default rather than throw. Exported so tests/npu_pack_import_test.cpp can pin it without a
// pack; `source` names where the geometry came from, for the refusal messages.
Dims read_dims_exported(const nlohmann::json& c, const std::string& source);
// engine#1503: what the graph walker RECORDED, without exposing its model type. A three-input MatMulNBits (no qzeros,
// weights spelled `<proj>.MatMul.weight_Q4`) used to be dropped entirely, which is why Qwen3-VL-4B imported one tensor.
// Returns the matmul count; `without_zeros` counts those that carry no qzeros tensor (the implicit-zero-point case), and
// the last three report the first matmul's own K/N/block attributes.
int parse_onnx_matmul_census(const std::string& path, int* matmuls, int* without_zeros, int* K, int* N, int* block);
// engine#1638: the geometry as read, so a test can check WHICH source a number came from.
nlohmann::json load_geometry_exported(const std::string& pack_dir, std::string* source);
FconstGeom locate_fconst(const std::string& path, unsigned short eps_bits = 0x3586);
// engine#1831: the in-block header offsets the epsilon anchors support at a known stride, sorted -- every offset that
// recurs `min_repeat` or more times (offset 0, the block start, excluded). Kept as a list, not a mode: a Qwen3 block has
// three equally frequent inner headers.
std::vector<int64_t> fconst_block_headers(const std::vector<int64_t>& pos, int64_t stride, int min_repeat);
// engine#1831: set `hdr`/`h1`/`period` from that list the way locate_fconst() does -- one inner header is a 2-header block,
// three is a 4-header block. Returns false, and changes nothing, for any other count.
bool fconst_set_block_headers(FconstGeom& g, const std::vector<int64_t>& inner);

// One measured Ryzen AI 1.9 Token .fconst block shape (engine#1503). Every field is measured against the 1.8 pack of
// the same model by tools/fconst19_measure.py; none of it is derivable from the pack, which is why a pack whose shape
// is not in the table is refused rather than decoded.
struct Fconst19Rule {
    int hidden, heads, layers;
    int64_t prefix;   // bytes before the q/k/v bundle
    int n_tiles, rot;
    int64_t tile;
};

// The row for a geometry, or nullptr with `why` explaining which of the two reasons applies: the geometry has never
// been measured, or it has been measured MORE THAN ONCE with different shapes (two models can share hidden/heads/layers
// and still cut the payload differently -- Qwen3-1.7B and Llama-3.2-1B are both 2,048/32 heads and use 3 x 1,183,744
// against 5 x 591,872). A wrong row gets the ROTATION wrong, which leaves the prefix and the epsilon anchors untouched,
// so it reads as a correct parse and answers quiet nonsense; that is why an ambiguous table refuses instead of picking.
const Fconst19Rule* select_fconst19_rule(int hidden, int heads, int layers, std::string* why);
// The measured 1.9 rule table itself, with its row count, so a test can pin every row against the numbers that were
// measured. Two failure modes this exists for, both seen on this table: a wrong digit in a row (which rotates a pack's
// weights into the wrong projections while every anchor still lines up), and a row that a merge silently drops (the
// h4096 row was lost twice that way, which un-measured a whole model class with nothing failing).
const Fconst19Rule* fconst19_rules(int* count);
// The preference order for the 1.9 canonical .fconst scratch copy (engine#1801): the explicit override, then the on-disk cache
// directory, then the temp directory and the cwd. Exposed so a test can pin the order -- the bug that shipped once was a cache
// directory that was never created, so every pack silently fell through to tmpfs.
std::vector<std::string> fconst_scratch_dirs();
// True when `dir` is, or can be made into, a directory the scratch copy may be written to. Exposed with the order above because the
// bug #1801 shipped once was exactly this: the cache directory existed only in the comment.
bool fconst_scratch_ready(const std::string& dir);
// The same decision over an explicit table, so a test can hand it the one shape that decides correctness and that the
// real table must never contain: two rows for one geometry with different shapes.
const Fconst19Rule* select_fconst19_rule_from(const Fconst19Rule* rules, int count, int hidden, int heads, int layers,
                                              std::string* why);

int64_t attn_tile_bytes(int64_t k);
// Pearson correlation between a quantised head and a bf16 embedding table (engine#1690; the self-check's own statistic,
// exported so a test can pin the buffer width it decodes).
/* The embedding table the self-check compares a head against, with its WIDTH in the value (engine#1721).
 *
 * `embed_bytes` was a bare `const uint8_t*` that meant the pack's own f16 bytes and then -- from #1661, silently --
 * the container's bf16 copy, and four readers went on decoding F16. Nothing objected at compile time, and the
 * result was not a crash but a plausible number: corr 0.885220 / rel_err 0.987 on a tied pack whose container
 * scores 0.999999. So the format travels with the pointer, the default is what the importer actually holds
 * (bf16), and a reader that wants F16 has to say so. */
struct EmbedTable {
    const uint8_t* p = nullptr;
    enum class Fmt { F16, BF16 } fmt = Fmt::BF16;
    int H = 0;                       // row stride in ELEMENTS, both formats are 2 bytes
    bool ok() const { return p != nullptr && H > 0; }
};
// The only decode: bf16 or f16 per t.fmt. out-of-range rows still read (callers clamp), as before.
double embed_at(const EmbedTable& t, int row, int k);

double corr_bf16(const Q4& q, const uint8_t* embed_bf16, int H, int rows);
// Where a layer block's attention payload starts: the block's 64-byte header, then its
// input_layernorm weight as bf16 [hidden]. So it is `64 + 2 x hidden`, not a constant. It reads 1,856 on
// the 0.5B pack (hidden 896) and 4,160 on Llama-3.2-1B (hidden 2,048); reading it as the constant is what
// made every pack with a different hidden look like "whole tiles plus a stray run" (engine#979).
int64_t attn_payload_off(int64_t hidden);
TileW decode_tile(const uint8_t* tile, int k, int rows = 64);
// The exact inverse. Used only as the importer's self-check: a tile that does not survive
// encode(decode(t)) == t is a tile whose fields we do not understand, and it is refused by name.
std::vector<uint8_t> encode_tile(const TileW& t);
// A projection assembled from consecutive tiles, in the form the container's writer takes.
Q4 q_from_tiles(const std::vector<TileW>& tiles);
// gate and up are ONE tensor in the .fconst, interleaved in 32-column chunks, so a 128-column group holds
// 64 gate rows and 64 up rows: this takes one of the two as a 64-row tile (rows 0..31 from the group's
// first column half, rows 32..63 from the second).
TileW gu_half(const TileW& group, bool up);

// Q4NX q4_1 tiles of a Q4 tensor: [N/32 * K/256, 5120] bytes, plus the parameters
// that were not exactly representable in bf16 (counted, not fixed up). This is the form every
// GGUF-derived container holds (npu/q4nx.h q4_1), and the one the device streams.
void tile_q4_1(const Q4& q, std::vector<uint8_t>& out, long long* zero_inexact = nullptr);

// The same grid in the imported container's fp32 form: [N/32 * K/256, 6144] bytes (npu/q4nx.h
// q4_1f). Scale and zero are stored as f32, so the pack's own values survive bit for bit -- the
// difference engine#1185 is about: an OGA pack's zero is -zp*scale, an arbitrary f32, and the
// 5120-byte form rounded it (2.7-3.3e-3 relative on ~57 % of groups). `param_inexact` counts the
// groups whose stored bytes are not the source float; it is 0 unless the writer is wrong.
void tile_q4_1_f32(const Q4& q, std::vector<uint8_t>& out, long long* param_inexact = nullptr);

}  // namespace onebit::npu::pack_import
