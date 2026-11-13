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

// The embedding role served in-process (engine#1311, step 3).
//
// `1bit serve` has always served an embedding model by launching a llama-server in `--embedding` role and
// proxying /v1/embeddings to it. A 1-bit encoder container (`<dir>/model.embed`, written by
// tools/trit_embed_pack.py) is not a GGUF and has no llama-server: its forward is the engine's own
// (app/embed_model.cpp), on the host or on the NPU through the engine's int8 GEMM. So for that container the
// same flags mean "serve it here":
//
//   --embedding -m <container> --device npu|cpu      the model itself is the embedding model
//   --embed <container> --device npu|cpu -m <gguf>   an embedding model beside a chat model (RAG)
//
// and /v1/embeddings answers from this process, with the OpenAI shape and a string or an array of strings as
// `input`. The reply is byte-shaped like llama-server's (object/list/data/embedding/usage) so a client that
// talks to the RAG route today needs no change.
#pragma once

#include <memory>
#include <string>
#include <vector>

namespace onebit::embed_serve {

// True when `path` is a 1-bit encoder container rather than a GGUF or a q4nx directory.
bool container(const std::string& path);

// engine#1786: true when `path` is a q4nx directory whose config.json declares `use_bidirectional_attention: true`
// (FastFlowLM's Embedding-Gemma-300M-NPU2): an encoder the per-op forward runs in one non-causal pass, pooled
// (Q4nxNpuForward::embed_pooled). Served by the same flags and the same /v1/embeddings surface. NPU builds only.
bool forward_encoder(const std::string& path);

class Backend {
public:
    // `device` is "npu" or "cpu". Loading reads the container and the vocabulary; on "npu" the ternary
    // linears go through the engine's int8 GEMM (npu/embed/embed_npu.h), and a device failure is reported
    // rather than silently answered on the host.
    bool load(const std::string& dir, const std::string& device, std::string& why);

    const std::string& id() const { return id_; }
    const std::string& device() const { return device_; }
    int dim() const { return dim_; }
    int max_tokens() const;
    long requests() const { return requests_; }

    // One OpenAI /v1/embeddings request: body in, reply body out, `status` set. On failure `status` is 4xx/5xx
    // and the body is an OpenAI error object.
    std::string embeddings(const std::string& body, int& status);

    // /v1/models and /health, in the shapes the rest of serve already speaks.
    std::string models_json() const;
    std::string health_json(bool ready) const;

private:
    std::string id_, device_, why_;
    int dim_ = 0;
    long requests_ = 0;
    struct Impl;
    std::shared_ptr<Impl> impl_;
};

// The whole serve process for `--embedding -m <container>`: one in-process encoder, the OpenAI surface
// (/v1/embeddings, /embeddings, /v1/models, /health, /v1/health) and nothing else. Blocks until stopped.
int serve_only(const std::string& model_dir, const std::string& host, int port, const std::string& device,
               const std::string& alias);

}  // namespace onebit::embed_serve
