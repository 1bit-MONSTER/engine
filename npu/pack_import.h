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

#include <cstring>
#include <memory>
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

// The one-header attention section of a pack whose q_proj is WIDER than hidden (Hunyuan-0.5B: hidden 1024, 16 heads x
// head_dim 128, so q_out = 2048): [q,k,v tiles at K=hidden][gap][o_proj tiles at K=q_out]. `payload` is the bytes from
// `attn_off` to the block's header. `ok` is false when the remainder is not a norms-sized gap (negative, or larger than
// two gains plus 256 B), which is a layout this reader has not measured. When the gap has the gains' shape
// (qk_gain_offsets()), `q_norm`/`k_norm` are their offsets; otherwise both are 0 and the section is still read
// (engine#1889: this branch placed o_proj and never named the gains, so a pack that declares use_qk_norm was refused
// with "gains are not where the block layout says" although they sit in the gap it had just computed).
struct WideQAttnSection {
    bool ok = false;
    int64_t gap = 0, o_off = 0, q_norm = 0, k_norm = 0;
};
// Why one block's gap is NOT `[64 B zeros][bf16 [head_dim] gain][64 B zeros][bf16 [head_dim] gain]`, or "" when it is.
// `gap` points at 2 * (64 + head_dim*2) bytes. The offsets come from the block arithmetic, which is the same number
// for every layer; this is the per-layer witness that the bytes there are gains: both headers all zero, and each gain
// finite and not all zero. A gain window displaced by a section (320 B at head_dim 128) lands on a header or on tile
// bytes and fails one of these. The import refuses the layer by name on a non-empty answer (review of engine#2063).
std::string qk_gap_block_why(const uint8_t* gap, int64_t head_dim);
WideQAttnSection wide_q_attn_section(int64_t attn_off, int64_t payload, int64_t qkv_tiles, int64_t tile_bytes,
                                     int64_t o_tiles, int64_t o_tile_bytes, int64_t head_dim);

// How many rows a head read from the .fconst tail occupies in the container. The pack stores the head in whole 64-row
// tiles, so a vocabulary that is not a multiple of 64 is carried with the pack's own padding rows (Hunyuan: 120,818 ->
// 120,832) while `vocab_size` and the plan's tensor shape stay the vocabulary. A head that is a graph initializer is
// stored as it is: on main nothing pads it, and one whose row count is not a multiple of 32 is refused by the tile
// writer ("q4_1 tiles need N % 32 == 0"). If graph-head padding lands (engine#1992 was closed unmerged), its quantum
// belongs here. This is the number a container's lm_head tile count follows from.
int64_t head_stored_rows(int64_t vocab, bool from_fconst_tail);

// Why a pack's two layout witnesses cannot be reconciled, or "" when they can (engine#1985).
//
// `vote_hdrs` is the number of in-block headers the epsilon-anchor vote recorded (0 when it could not win); `declared_qk`
// is config.json's `use_qk_norm`. A pack that states per-head q/k gains keeps them in the two sections between q/k/v and
// o_proj, which a 2-header block does not have -- so a ONE-header vote on such a pack is self-contradictory, and engine#1985
// is what trusting it costs: `Qwen3-1.7B` read as a 2-header block, `qk_norm` flipped true -> false, and every attention
// projection was refused (280 gaps) where yesterday's main imported the same pack with 0.
//
// Neither witness wins: the vote measured where anchors repeat, and the declaration states what the block must contain.
// The reader refuses and names both, because picking either one silently is the defect. Exposed so the three cases can be
// pinned without a 2.6 GB pack.
std::string header_witness_refusal(int vote_hdrs, bool declared_qk);

// The first tile of k_proj, q_proj and v_proj inside the attention payload, for n_kv k/v tiles and n_q q tiles
// (64 rows each). `q_first` false is k, q, v; true is q, k, v (engine#1889, the Hunyuan block). v is last either way.
// Split out so the order is pinned by a test instead of living in an initializer list.
struct AttnFirstTiles {
    int k = 0, q = 0, v = 0;
};
AttnFirstTiles attn_first_tiles(int n_kv, int n_q, bool q_first);

// A hybrid pack's embedding table lives in its own graph (`embedding.onnx` + `.data`, gathered with
// GatherBlockQuantized). Returns [V, H] f32 row-major, empty when the pack has no such component, with `why` naming the
// reason. Exposed so the dequantisation can be tested against known codes and scales instead of only on a 3 GB pack.
std::vector<float> read_embedding_component(const std::string& pack_dir, std::string* why);

// The per-layer half of a multi-component pack's embedding (engine#1725). gemma-4 components carry, beside the token
// table, `embed_tokens_per_layer.weight` (262144 x 8960 on e2b = layers x head_dim) and a 256-wide projection norm;
// the decoder consumes them as `per_layer_inputs`, so a container without them gives the decoder less than it asks for.
// Read separately from the token table because they have different shapes and different consumers: the token table
// becomes the head oracle, this becomes an input. Empty (with `why`) when the component has no per-layer table.
struct EmbedPerLayer {
    // [V][layers*P] f32, row-major (embed_tokens_per_layer.weight) -- NOT copied. It is 9.4 GiB on gemma-4-e2b, so the
    // reader hands out a pointer into the component's mapped external data and the importer converts it a band at a
    // time. `keep` owns whatever that pointer lives in (the mapping, or a copy of an inline initializer) and unmaps it
    // when the last copy of this struct goes.
    const uint8_t* table_f32 = nullptr;
    std::shared_ptr<const void> keep;
    std::vector<float> norm;    // [P]                       (per_layer_projection_norm.weight, shared by every layer)
    // The CONTEXT projection: the component carries it quantised as
    // `per_layer_model_projection.weight_t_Q4` [layers*P][H] plus scales and packed zero points. The loader wants it
    // dequantised, one [P][H] matrix per layer (`model.per_layer_model_proj.weight_layer{l}`), so it is dequantised here
    // -- without it the per-layer input is `16*table/sqrt(2)`, plausible and wrong at every layer (engine#1725).
    std::vector<float> model_proj;   // [layers][P][H]
    int64_t          V = 0, width = 0, P = 0, H = 0, NL = 0;
    // `ok()` is what the reader guarantees: the table (bounds-checked against its data) and the projection norm.
    // `complete()` additionally requires the context projection, which the PLE contract needs -- a component with
    // `ok()` but not `complete()` imports the table and will be refused by the loader on
    // `per_layer_model_proj.weight_layer0`, which is the honest place to refuse (the importer does not know whether the
    // decoder declares per_layer_inputs).
    bool ok() const {
        return V > 0 && width > 0 && table_f32 != nullptr && (!norm.empty() ? norm.size() == size_t(P) : true);
    }
    bool complete() const { return ok() && P > 0 && H > 0 && NL > 0 && model_proj.size() == size_t(NL) * size_t(P) * size_t(H); }
    // one value of the table, for tests and spot checks
    float at(int64_t row, int64_t col) const {
        float f;
        std::memcpy(&f, table_f32 + (size_t(row) * size_t(width) + size_t(col)) * 4, 4);
        return f;
    }
};
// `hidden` is the decoder's hidden size (the context projection's K); it comes from the pack's geometry, not from
// the component, which is why the caller passes it. Parses the component graph and maps its external data ONCE; the
// table's bytes are not read.
EmbedPerLayer read_embedding_per_layer(const std::string& pack_dir, int64_t hidden, std::string* why);

// One band of the per-layer token table as BF16 (round to nearest even, the same `f32_to_bf16` every other tensor
// gets), straight from the mapping `meta` holds: no graph parse, no file open, nothing allocated. The importer
// streams the whole table through this in 32-row bands (573 KB at a time on gemma-4-e2b) while the container is
// written, so the table is never held -- not as f32, not as BF16 (engine#1725).
// Returns the rows written, or -1 with `why` set. `out` must hold `rows * width * 2` bytes.
int64_t read_per_layer_table_band(const EmbedPerLayer& meta, int64_t first_row, int64_t rows, uint8_t* out,
                                  std::string* why);

// One tensor the pack describes, and where its values would come from.
struct ContainerPayloadAnomaly {
    std::string name;
    int64_t row_bytes = 0;
    int64_t implied_k = 0;
};
struct ContainerWriteItem {
    std::string field;    // the forward field (npu/forward/include/model.h)
    std::string source;   // ONE container tensor that fills it -- a representative, not the only one
    std::string shape;    // its shape+dtype text, as reported by container_role_shapes()
    int64_t count = 0;    // how many tensors carry this role (e.g. one per layer), so a writer does not read
                          // `source` as "the only one"
};
struct ContainerWritePlan {
    std::vector<ContainerWriteItem> items;
    std::vector<std::string> ambiguous;   // "<field>: <role> appears in N shapes"
    std::vector<std::string> no_source;    // fields the table knows whose role is absent from this pack
};
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
    // engine#1503: every MatMulNBits the pack declares is block 32 and at least one of them is a layer projection, and
    // the pack has a Token .fconst. Such a pack is read with the all-block-32 layout (Block32Layout below) from the
    // raw .fconst; the block-128 walk and the canonical copy are not used. `block32_note` is why that file does not
    // fit the layout (empty when it does), and is the refusal every .fconst tensor of the plan then carries.
    bool block32_layers = false;
    std::string block32_note;
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
    // Where the 2-header families' gap `[64 B zeros][q_norm][64 B zeros][k_norm]` starts in the block, 0 when the gains
    // are not in that form. The import checks the form in EVERY layer (qk_gap_block_why()) before it writes a gain.
    int64_t qk_gap_off = 0;
    // The order of the three projections in the block's first payload. Every family measured before Hunyuan writes
    // k, q, v (identified on Qwen2.5-0.5B by the tiles' bias subvolumes). The Hunyuan block -- q/k/v at K=hidden, the
    // packed per-head gains, o_proj -- writes q, k, v (engine#1889): with k,q,v the container's k_proj was q_proj's
    // first rows, and the model answered filler words; with q,k,v it answers. See attn_first_tiles().
    bool attn_q_first = false;
    int64_t mlp_off = 0;  // the MLP section starts just past the block's post-attention norm
    // What the pack's *graph* carries, reported so `npu-pack info` can answer a pack's genre without a
    // third-party parser: the rai_1.3/1.6/1.7.1 line ships its 4-bit weights as ordinary ONNX initializers, while
    // 1.8 and R1-Distill ship placeholders and keep the weights in model.bin (pi-inferzo, 2026-11-08).
    // graph_matmuls < 0 means "no graph was read", which is why it is not just a count.
    // engine#1786 (item F1): a FastFlowLM NPU2 model source is a safetensors container (`model.q4nx`) with no graph,
    // so the two census fields above stay -1. `container_tensors` is -1 for every pack that is not that genre; when it
    // is not, it is the container's tensor count and the dtype map says what the reader found. Reported rather than
    // refused at the first byte, because "cannot open model.onnx" says nothing about a pack that has no graph by design.
    int64_t container_tensors = -1;
    int64_t container_payload_bytes = 0;
    std::map<std::string, int64_t> container_dtypes;
    std::string container_note;   // what the container genre still needs, empty for a non-container pack
    // engine#1788: MiniCPM-V's vision tower is a SECOND container beside model.q4nx (vision_weight.q4nx, 1.14 GB /
    // 347 tensors on the 35B). Reported, not interpreted: the vision tower has its own reader (npu/vision_pack.cpp),
    // so this is for a human reading `info`, not for a plan -- and 347 tensors absent from it is how a pack looks
    // importable when half of it has never been looked at. -1 when the pack has no such file.
    int64_t container_vision_tensors = -1;
    int64_t container_vision_payload_bytes = 0;
    int64_t container_vision_roles_named = -1;
    std::map<std::string, int64_t> container_vision_role_counts;
    // engine#1788, 3c: every I8 row whose implied K is none of the widths the geometry declares. Reported for the
    // writer's benefit -- a row that implies a K the model does not have is a re-tile, not a copy.
    std::map<int64_t, int64_t> container_payload_rows;          // row_bytes -> how many I8 tensors use it
    // engine#1788, 3c: per role, the shapes the container carries -- the fact the writer's shape matching needs.
    std::map<std::string, std::map<std::string, int64_t>> container_role_shapes;
    ContainerWritePlan container_write_plan;   // role -> forward field -> source tensor (slice 3c step 5)
    std::vector<ContainerPayloadAnomaly> container_payload_anomalies;
    // engine#1786 (item F1), slice 2: WHERE the container's geometry came from, and what it could not reach.
    // A bare number in `dims` cannot say whether the pack declared it or a file beside it did, and those two are free to
    // disagree (13 of the 29 packs measured carry a `dims` block; all 29 carry config.json). Empty for a non-container pack.
    std::string container_geometry_source;          // "dims", "config.json", or empty when neither answered
    std::vector<std::string> container_geometry_missing;   // fields no source could fill, by name
    std::vector<std::string> container_geometry_inferred;  // fields derived from another field, by name
    // engine#1786 (item F1), slice 3b: what the pack's names say it carries, and what nothing recognised. `unassigned`
    // is the list that matters -- it means "nothing knows this tensor", so an empty one is the statement that the role
    // map covers the pack. The SSM and MTP families are recognised (roles `ssm` / `mtp`), so a hybrid pack and a draft
    // head do not masquerade as unknown tensors.
    std::map<std::string, int> container_role_counts;
    std::vector<std::string> container_roles_unassigned;
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
    // Of those, the ones outside the decode `If` branch: what plan() refuses on. The total above stays for `info`.
    int64_t graph_weightless_matmuls_decode = 0;
    // The `If` branch taken when input_ids has length 1, and how that was decided (read from the condition, or assumed).
    std::string graph_decode_branch;
    std::string graph_branch_note;
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

// A FastFlowLM NPU2 model source (engine#1786, item F1) is a safetensors container named `model.q4nx`: an 8-byte
// little-endian header length, that many bytes of JSON mapping tensor name -> {dtype, shape, data_offsets}, then the
// payload. There is no graph and no `model.onnx`, which is why the importer above refuses such a pack before reading a
// byte of it. The tensor NAMES are the model's own HF names, so this reader is also the work list: what it reports is
// what a container-genre plan has to map.
//
// Header only: no weights are read, no device is opened. Exported because it is the half that can be wrong silently
// (an off-by-one header length reports an empty inventory and looks like "a pack with no tensors").
struct Q4nxTensor {
    std::string name;
    std::string dtype;
    std::vector<int64_t> shape;
    int64_t begin = 0;      // data_offsets[0]
    int64_t end = 0;        // data_offsets[1]
};
struct Q4nxContainer {
    int64_t header_bytes = 0;
    int64_t payload_bytes = 0;
    std::vector<Q4nxTensor> tensors;   // in header order
    // The safetensors `__metadata__` object, kept because it is where this genre states its own geometry (engine#1786).
    // Three quarters of the packs measured on this box carry a `dims` block with hidden/intermediate/layers/heads/vocab;
    // two carry no metadata at all (`d3-edge`, `d3-flash`), which is why the reader keeps it as data and does not
    // interpret it here.
    nlohmann::json meta;
    std::string why;                        // non-empty when the container could not be read; tensors is then empty
};
Q4nxContainer read_q4nx_container(const std::string& path);

// engine#1786 (item F1): the container genre's geometry, from the two places a pack can state it, in order of
// authority -- the container's own `dims` block, then config.json. `source` names which one answered, so a plan can
// never present a declared number and a file's number as the same kind of fact; `missing` names the fields neither
// could fill, and `inferred` the ones derived from other fields (head_dim from hidden/heads).
//
// Pure: no file is opened, so the three genres the census measured -- 13 self-describing, 14 gguf-converted, 2 with
// no metadata at all -- can all be pinned without a pack, which is the point of extracting it from `plan()`.
struct ContainerGeometry {
    Dims dims;
    std::string source;                 // "dims", "config.json", or empty when neither answered
    std::vector<std::string> missing;   // fields no source could fill, by name
    std::vector<std::string> inferred;  // fields derived from another field, by name
};
ContainerGeometry container_geometry(const nlohmann::json& meta, const nlohmann::json& config);

// engine#1786 (item F1), slice 3a: a container's ROLES, from tensor names alone -- the question a plan must answer
// before it can map a tensor to a projection.
//
// Measured on this box (engine#1786 comment of 2026-11-11, headers only, 30 packs): **21 of 30 classify fully by name
// alone**, and the entire residue is two families -- `model.layers.N.linear_attn.ssm_*` (the SSM-hybrid attention) and
// `model.mtp.*` (MTP draft heads, which are not models). Both HF and the alternate spellings resolve (`q_proj` and
// `attn_q`/`wq`), and so do the biases. So the rule below is a RULE, not a per-pack table -- which is why it is worth
// having before the SSM forward exists.
//
// What a role does NOT establish: the byte layout of the tensor, or whether it is stored transposed. A name says which
// projection; `graph_weights` (present on 13 of the packs) and the pack's own `dims` are what check the rest.
struct ContainerRoles {
    std::map<std::string, std::string> role_of;   // tensor name -> role
    std::vector<std::string> unassigned;          // names no role matched, in header order
    std::map<std::string, int> counts;            // role -> number of tensors
};
ContainerRoles container_roles(const std::vector<Q4nxTensor>& tensors);

// engine#1786 (item F1) / engine#1788, slice 3c step 2: what each role fills in the forward, as a CHECK rather than a
// mapping the writer can quietly get wrong. The role vocabulary is not a safe key on its own -- `gate_proj` and
// `self_attn_gate_proj` are different tensors that one needle used to share, and a container can carry both -- so two
// roles that name the same destination field are refused instead of being written in some order that happens to work.
// Returns the forward's field name (npu/forward/include/model.h), or "" when the role is not a per-layer weight the
// forward's loader reads: fused pairs (`qkv_proj`, `gate_up_proj`) are consumed by a fused path, `ssm`/`mtp`/`vision`
// are their own containers or their own code, and `shared_expert_gate` is read only by the MoE block.
const char* container_role_field(const std::string& role);
// engine#1788, slice 3c step 3: does a container tensor's payload agree with the geometry the pack declares? An I8 row
// in this genre is `2*K + 640` bytes (K the projection's input width, 640 the fixed per-group scale/zero overhead) --
// measured on both MiniCPM-V packs on this box (35B: uniform 4736 with K = H = 2048; 1B: 4736 and 8704, K = 2048/4032).
// A row that implies a K the model's geometry does not contain is a re-tile the writer must not copy blind, so it is
// reported by name rather than written.

// `plausible_k` is NOT used to judge: running it against the two packs on this box produced false positives (both
// containers' `lm_head` rows imply K = 4032, a width no geometry field of either model declares, and 37 of the 1B's
// rows with it). So the check reports what it can prove and stops there:
//   - `container_payload_rows`  the distribution of row widths, which is the fact a writer needs (one width = a copy,
//                               several = a re-tile per shape);
//   - `container_payload_anomalies`  only the rows that cannot be one of these at all (shorter than the 640-byte
//                               overhead, or an odd remainder), with implied_k = -1.
std::map<int64_t, int64_t> container_payload_rows(const std::vector<Q4nxTensor>& tensors);
// engine#1788, slice 3c step 4: per role, the shapes a container actually carries. The writer's job is "for this
// destination field, are the source shapes what that field holds?", and after the plausible-K attempt produced false
// positives on both packs on this box, the useful thing to hand it is the FACT: which shapes each role appears in.
// A role with one shape is a mapping; a role with several is a decision the writer must make per shape.
std::map<std::string, std::map<std::string, int64_t>> container_role_shapes(const std::vector<Q4nxTensor>& tensors);
// engine#1788, slice 3c step 5: the writer's mapping, as a plan it can be wrong about. For every role that names a
// forward field, which source tensor supplies it and in what shape; a role present in two shapes is AMBIGUOUS (the
// writer must decide, not pick), and a field the table knows whose role is absent is listed separately rather than
// called missing -- whether a model needs `up_proj` or `up_exps` is a per-model question, not a table one.
ContainerWritePlan container_write_plan(const std::vector<Q4nxTensor>& tensors);
std::vector<ContainerPayloadAnomaly> container_payload_anomalies(const std::vector<Q4nxTensor>& tensors);
// The destination fields claimed by more than one role in `counts` (each with the roles that claim it). Empty is the
// only state a writer may start from.
std::vector<std::string> container_role_field_conflicts(const std::map<std::string, int>& counts);

// Reads the pack (model.onnx, its external data, both .fconst headers, config.json).
// Throws only when the pack is structurally unreadable.
// engine#1889 (review of #2063): plan()'s wide-q branch (q_out != hidden), as one function so its effect on the plan is
// testable without a fabricated 1.9 pack. When the section was read (`wq.ok`) and q is wider than hidden, it places the
// q/k/v run, o_proj and the gains, and sets attn_q_first when the gap held the gains: that block is the Hunyuan writer's,
// whose first payload is q, k, v (read as k, q, v, the container's k_proj was q_proj's first rows, correlation 0.01).
// Returns false and leaves `p` untouched otherwise.
bool apply_wide_q_section(Plan& p, const WideQAttnSection& wq, int64_t n_kv, int64_t n_q, int q_out, int64_t hidden);
Plan plan(const std::string& pack_dir);

// Whether the canonical Token .fconst already on the scratch path holds exactly the bytes this run would write,
// given the length the canonical form must have (engine#1662). True means the ~291 MB rewrite can be skipped. Split
// out of canonical_token_fconst() so the decision has a test that needs no 1.9 pack: a truncated copy, a missing file
// and a directory on the scratch path must all come back false.
bool canonical_copy_reusable(const std::string& work, int64_t expected_len);

// engine#1985: how the canonical copy reaches `work`. It is written under canonical_copy_temp_name(work) -- a private name
// in the same directory -- and renamed into place, so a concurrent import of the same pack finds no copy or a complete one,
// never one that is being written. Writing in place let a second import truncate the file the first was about to read.
// Throws std::runtime_error when the copy cannot be written or moved; the temporary file is removed on either failure.
// Exposed so the rule has an arm that needs no pack (threads writing one path while a reader checks every complete-length
// file it finds), and so the temporary name's two properties are pinned: a pack's sweep prefix never matches it, and the
// age rule does.
std::string canonical_copy_temp_name(const std::string& work);
void write_canonical_copy(const std::string& work, const uint8_t* data, size_t len);

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

// engine#1907: apply evict_stale_canonical_copies() to EVERY scratch candidate, not only the one being written. Since #1801 the
// write lands in the on-disk cache, so a sweep of that directory alone never reached /tmp again, and the copies older binaries
// had left in tmpfs (17.0 GiB of RAM, measured 2026-11-10) stayed until reboot. `written_dir` is swept even if it is not yet a
// directory as seen here; every other entry is swept only when it already exists, and is never created.
void sweep_fconst_scratch_dirs(const std::vector<std::string>& dirs, const std::string& written_dir,
                               const std::string& keep_name, const std::string& prefix, const std::string& legacy_name,
                               int64_t max_age_seconds);

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

// The tied-head fit (engine#1503); what each field means is stated at fit_head() below.
struct HeadFit {
    double rel = 0, scale = 1, rms_ratio = 1, floor = 0;
};
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
    // engine#1503: tied, correlated -- but MIS-SCALED. Correlation alone passes a head that is a scaled copy of the
    // embedding: two packs measured on this box report corr 0.9915/0.9961 with rel_err 0.095/0.136 and serve garbage
    // ("myself" forever), while the pack that serves correctly reports rel_err 0.000000. A pack in this state imports
    // cleanly and answers nonsense, so it is refused rather than reported.
    bool selfcheck_scaled = false;
    double lmhead_corr = 0, lmhead_rel = 0, ctrl_corr_nibble = 0, ctrl_corr_nozp = 0;
    HeadFit head_fit;   // engine#1503: scale, rms ratio and the 4-bit noise floor beside lmhead_rel
    std::string model_type;   // the pack's, so the CLI can refuse a known-bad architecture by name
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
    // engine#1740: set when a block-32 head was READ from the .fconst tail (never together with the note above)
    std::string head_read_note;
    // engine#1725: the per-layer embedding block the container carries (0 = none): row width per layer, and rows
    int64_t per_layer_P = 0, per_layer_V = 0;
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

    // engine#1503: the all-block-32 path. Units read (640-byte-strip units plus gate/up units, head included). There
    // is no decode->encode round trip on this path, so the two counters above stay 0 for such a pack.
    long long block32_units = 0;
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
// The 1.9 block-32 tail head (engine#1740). A pack whose `/lm_head/MatMul_Q4` is declared `block_size 32` keeps the
// head in the Token .fconst tail, after the same `64 + 2*hidden` envelope every other 1.9 tail opens with, but NOT in
// the block-128 tile: it is written in 64-row units of `2,560 + 2*(K/32)*640` bytes.
//
// Measured on `Qwen-2.5_1.5B_Instruct_rai_1.9.0_npu` (tied, V 151,936, K 1,536; unit 64,000 B, 2,374 units, then
// 110,208,000 B of zeros to EOF) against the pack's own `model.embed_tokens`:
//     unit   : 2,560 B of zeros, then 2*(K/32) strips of 640 B
//     strip s: K block `s/2` (32 columns), rows `32*(s%2) .. +31` of the unit
//       +0   512 B of 4-bit codes, low nibble first; code position p (0..1023) is row `(p/256)*8 + p%8`, column
//            `(p/8)%32` -- four 8-row bands, each written column by column
//       +512  16 B of zero points, one nibble per row (0x8 everywhere on the measured pack)
//       +528  48 B of zeros
//       +576  32 bf16 scales, one per row; SIGNED -- the measured writer stores `-m/8` where m is the block's
//            largest-magnitude weight, so about half of them are negative
//     w = (code - zero_point) * scale
// Every one of the 3,072 scales of a unit equals `bf16(-m/8)` of its embedding block exactly, and the decoded rows
// correlate 0.990 .. 0.997 with the embedding (4-bit), which is what fixed the field positions. The result is returned
// with positive scales (a negative one is folded into its codes: `15 - code`, `15 - zp`), as a `Q4` at block 32, rows
// padded to whole units the way the block-128 tail head is.
//
// What is checked, because a wrong region read with confidence is the failure this path exists to avoid: the unit
// count is `ceil(V/64)` and has to fit; every unit's 2,560 B and every strip's 48 B are zero; every scale is finite;
// and everything after the last unit is zero. Any of those failing is a refusal with the offset in it.
constexpr int64_t kB32UnitRows = 64, kB32UnitHeader = 2560, kB32StripBytes = 640;
inline int64_t block32_unit_bytes(int64_t K) { return kB32UnitHeader + 2 * (K / 32) * kB32StripBytes; }
struct Block32Head {
    bool ok = false;
    Q4 q;                 // N = units x 64 (>= V), K, block 32; value = code*scale + zero, scale > 0
    int64_t units = 0;
    int64_t bytes = 0;    // units x block32_unit_bytes(K)
    int64_t padding = 0;  // zero bytes after the last unit
    std::string why;      // set when !ok
};
// `p` points at the first unit (tail_off + envelope), `avail` is the bytes from there to the end of the file, and
// `file_off` is where `p` sits in the file (only used to put an offset in a refusal).
Block32Head read_block32_head(const uint8_t* p, int64_t avail, int64_t V, int64_t K, int64_t file_off = 0);

// The same units for any matrix (engine#1503): `ceil(rows/64)` units of `block32_unit_bytes(K)` starting at `p`. This
// is the head reader without its tail rule (nothing is required of the bytes after the last unit), and it is what the
// all-block-32 layer block's q/k/v/o/down are read with. `what` names the tensor in a refusal. Same checks: the bytes
// fit, every unit header and strip gap is zero, every scale is finite; same normalisation (positive scales).
struct Block32Units {
    bool ok = false;
    Q4 q;                 // N = units x 64 (>= rows), K, block 32
    int64_t units = 0;
    int64_t bytes = 0;    // units x block32_unit_bytes(K)
    std::string why;      // set when !ok
};
Block32Units read_block32_units(const uint8_t* p, int64_t avail, int64_t rows, int64_t K, const char* what = "head",
                                int64_t file_off = 0);

// The gate/up unit of an all-block-32 pack (engine#1503): gate_proj and up_proj share one run of 64-row units, each
// `4,864 + 2*(K/32)*1,216` bytes.
//
// Measured on `SmolLM2-135M-Instruct_rai_1.9.0_npu` (K 576, intermediate 1,536: 24 units of 48,640 B per layer)
// against the public checkpoint's gate_proj / up_proj:
//     unit   : 4,864 B of zeros, then 2*(K/32) strips of 1,216 B
//     strip s: K block `s/2` (32 columns), 64 row SLOTS: slots 0..31 are GATE rows `32*(s%2) .. +31` of the unit,
//              slots 32..63 are UP rows with the same numbers
//       +0    1,024 B of 4-bit codes, low nibble first; code position p (0..2047) is slot `(p/256)*8 + p%8`, column
//             `(p/8)%32` -- the head strip's order over eight 8-slot bands instead of four
//       +1024 32 B of zero points, one nibble per slot
//       +1056 32 B of zeros
//       +1088 64 bf16 scales, one per slot, signed
// The code order is the measured part (0.98 .. 0.996 per 32-code block against the reference). That zero point and
// scale are indexed by slot is ASSUMED by analogy with the 640-byte strip; the import of the measured pack is what
// tests it (a container gate/up that still correlates with the checkpoint after the zero points and scales are
// applied). Returned as two `Q4`s with the same normalisation and the same refusals as the unit reader.
constexpr int64_t kB32MlpUnitHeader = 4864, kB32MlpStripBytes = 1216;
inline int64_t block32_mlp_unit_bytes(int64_t K) { return kB32MlpUnitHeader + 2 * (K / 32) * kB32MlpStripBytes; }
struct Block32GateUp {
    bool ok = false;
    Q4 gate, up;          // each N = units x 64, K, block 32
    int64_t units = 0;
    int64_t bytes = 0;    // units x block32_mlp_unit_bytes(K)
    std::string why;      // set when !ok
};
Block32GateUp read_block32_gate_up(const uint8_t* p, int64_t avail, int64_t rows, int64_t K, int64_t file_off = 0);

// The layer block of a pack whose EVERY MatMulNBits is block 32 (engine#1503), as offsets from the block's start.
// Read from the RAW 1.9 Token .fconst (not the canonical 1.8-form copy). With H = hidden, Nq = heads*head_dim,
// Nkv = kv_heads*head_dim, IM = intermediate, unit = block32_unit_bytes, mlp_unit = block32_mlp_unit_bytes:
//     +0        64 + 2*H   envelope: bf16 rms_norm_eps, zeros to 64, input_layernorm bf16[H]
//     q_off     (Nq/64)  x unit(H)       q_proj [Nq][H]
//     k_off     (Nkv/64) x unit(H)       k_proj [Nkv][H]
//     v_off     (Nkv/64) x unit(H)       v_proj [Nkv][H]
//     sink_off  4*heads bytes            not a weight; skipped (int32 0xFF7F0000 x heads on the measured pack)
//     o_off     (H/64)   x unit(Nq)      o_proj [H][Nq]
//     post_off  64 + 2*H   envelope: eps, zeros, post_attention_layernorm bf16[H]
//     gu_off    (IM/64)  x mlp_unit(H)   gate_proj + up_proj, [IM][H] each
//     hdr_off   64 bytes                 skipped, content not interpreted (01 then 63 zeros on the measured pack)
//     down_off  (H/64)   x unit(IM)      down_proj [H][IM]
//     stride
// and the tail at `layers x stride`: the same envelope with the final norm, `ceil(V/64)` head units at K = H, zeros.
// Measured on SmolLM2-135M-Instruct 1.9 only (stride 2,360,292; file 90,470,776 B), where Nq == H -- so o_proj's K
// being Nq rather than H is the declared shape, not a measurement. There is no bias field in this layout.
struct Block32Layout {
    std::string why;      // non-empty: this geometry does not fit the layout (a width that is not whole units)
    int64_t env = 0, q_off = 0, k_off = 0, v_off = 0, sink_off = 0, o_off = 0, post_off = 0, gu_off = 0, hdr_off = 0,
            down_off = 0, stride = 0;
    int64_t tail_off = 0, head_off = 0, head_units = 0;
    int64_t min_size = 0; // head_off + head_units x unit(H); the file is this plus zero padding
};
Block32Layout block32_layout(const Dims& d);
// Empty when a mapped file fits `lay`: it is at least `min_size` long, and every envelope (two per layer and the
// tail's) opens with the pack's epsilon followed by zeros up to the norm. Otherwise the reason, with the offset. The
// zero padding after the head is the head reader's check.
std::string block32_file_refusal(const uint8_t* map, int64_t size, const Dims& d, const Block32Layout& lay);

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
// engine#1990: a head whose row count is not a multiple of the 32-row tile quantum (CodeLlama: 32,016 rows) cannot be
// held by the q4_1 grid, so pad it with zero rows that decode to zero and leave the container's vocab_size alone. Exposed
// because the property -- padding is invisible to the vocabulary -- is what a test can pin without a pack.
void pad_rows_to_tile_quantum(Q4& q, int vocab);

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

// engine#1503: the tied-head rule, as ONE predicate so the four call sites cannot drift: the head must reproduce the
// embedding (corr) AND not be a scaled copy of it (rel_err). The constant is the measured gap between the two states --
// 0.000000 for the pack that serves, 0.095/0.136 for the two that do not -- so 0.02 sits far from both.
inline constexpr double kSelfcheckRelMax = 0.02;
bool selfcheck_head_ok(double corr, double rel);   // the same rule with no fit: scale 1, no noise floor
// engine#1503: the rule on what it claims. A head passes when it correlates (kTieCorr), is NOT scaled (scale and
// rms_ratio within kHeadScaleTol of 1), and its rel error is no worse than kHeadFloorFactor x the embedding's own 4-bit
// noise floor at the head's block (and never refused below kSelfcheckRelMax, so a head that IS its embedding passes).
inline constexpr double kHeadScaleTol = 0.05;
inline constexpr double kHeadFloorFactor = 1.5;
bool selfcheck_head_ok(double corr, const HeadFit& fit);
// A model_type this route is MEASURED to serve garbage on, with the reason; nullptr for every other (engine#1503).
const char* known_bad_architecture(const std::string& model_type);

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

// engine#1503: what the tied-head self-check measures, beyond the correlation. `rel` alone cannot tell a scaled head
// from a correct 4-bit head whose embedding is kept at higher precision: both score ~0.09 (measured: Qwen2-1.5B 1.9,
// tail head block 32, rel 0.094316; gemma-2-2b AWQ hybrid, rel 0.094874), while the pack 0.02 was calibrated on
// scores 0 only because its embedding IS the dequantised head. So the fit also carries:
//   scale     -- the least-squares a in e ~= a * head. A scaled copy shows it directly; the two above are 0.994 / 0.992.
//   rms_ratio -- rms(head) / rms(embedding), the same question without the fit (1.0018 / 1.0038).
//   floor     -- the rel error of the EMBEDDING quantised the way the head is (asymmetric 4-bit, min/max per `block`
//                columns): what a correct head costs. The two above are 1.12x / 1.07x of theirs (0.083865 / 0.088985).
HeadFit fit_head(const Q4& q, const EmbedTable& embed, int rows, int block);

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
