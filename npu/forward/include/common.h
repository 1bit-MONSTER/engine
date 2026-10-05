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
// Copyright (c) 2026 bong-water-water-bong
#ifndef NPU_INFER_COMMON_H
#define NPU_INFER_COMMON_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NPU_MAX_CTX 16

// Model configuration for Qwen3-0.6B (NPU-verified values)
typedef struct {
    int32_t hidden_size;
    int32_t num_layers;
    int32_t num_attention_heads;
    int32_t num_key_value_heads;
    int32_t intermediate_size;
    int32_t head_dim;
    int32_t vocab_size;
    int32_t max_position_embeddings;
    int32_t max_seq_len;
    float rms_norm_eps;
    bool tie_word_embeddings;
    
    // NPU-specific blocked format constants
    uint32_t npu_block_cols;   // 1024 — each BO holds this many columns
    uint32_t npu_block_rows;   // 256 — each BO holds this many rows
    uint32_t npu_weight_bo_size; // 1048576 — 1MB per weight BO
    uint32_t npu_activation_bo_size; // 1048576 — 1MB per activation BO
    uint32_t npu_kv_cache_bo_size; // 134217728 — 128MB per KV cache BO (UNIT: allocation,
                                   // a capacity CEILING. It is deliberately larger than the
                                   // layout below needs; do not read it as a token count.)
} ModelConfig;

static const ModelConfig QWEN3_0_6B_CONFIG = {
    .hidden_size = 1024,
    .num_layers = 28,
    .num_attention_heads = 16,
    .num_key_value_heads = 8,
    .intermediate_size = 3072,
    .head_dim = 128,
    .vocab_size = 151936,
    .max_position_embeddings = 40960,
    .max_seq_len = 4096,
    .rms_norm_eps = 1e-6f,
    .tie_word_embeddings = true,
    // NPU blocked format
    .npu_block_cols = 1024,
    .npu_block_rows = 256,
    .npu_weight_bo_size = 1048576,
    .npu_activation_bo_size = 1048576,
    .npu_kv_cache_bo_size = 134217728,
};

// Block quantization constants for Q4NX format
#define GROUP_SIZE 32
#define MAX_GROUPS 32768

// Logging
#define LOG_INFO(fmt, ...)  fprintf(stderr, "[INFO]  " fmt "\n", ##__VA_ARGS__)
#define LOG_ERROR(fmt, ...) fprintf(stderr, "[ERROR] " fmt "\n", ##__VA_ARGS__)
#define LOG_DEBUG(fmt, ...) fprintf(stderr, "[DEBUG] " fmt "\n", ##__VA_ARGS__)
#define LOG_WARNING(fmt, ...) fprintf(stderr, "[WARN]  " fmt "\n", ##__VA_ARGS__)

#endif // NPU_INFER_COMMON_H
