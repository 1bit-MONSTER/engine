#pragma once
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
// mlx — MLX models on Apple Silicon (Metal) through lemon-mlx-engine's server.
//
// Local delta (1bit engine; see UPSTREAM.md). The executor is the `server` binary
// of lemon-mlx-engine (fork bong-water-water-bong/lemon-mlx-engine), which builds
// MLX's Metal backend on macOS and speaks the OpenAI chat/completions API. It is
// not a downloadable Lemonade artifact, so its path comes from the `mlx_bin`
// option (config key `mlx.mlx_bin`), then $LEMONADE_MLX_SERVER, then
// `lemon-mlx-server` on PATH. A model's checkpoint is its Hugging Face id (for
// example mlx-community/Qwen3-0.6B-4bit); the engine downloads it itself.
#include "lemon/backends/backend_descriptor.h"

#include <nlohmann/json.hpp>

namespace lemon {
namespace backends {
namespace mlx {

inline const BackendDescriptor descriptor = {
    /*recipe*/          "mlx",
    /*display_name*/    "MLX (Apple Silicon, lemon-mlx-engine)",
    /*binary*/          "",       // not a Lemonade download: see mlx_bin
    /*config_section*/  "mlx",
    /*default_device*/  DEVICE_GPU,
    /*slot_policy*/     SlotPolicy::Standard,
    /*selectable_backend*/ false,
    /*uses_ctx_size*/   false,    // the MLX server takes no --ctx-size
    /*dynamic_models*/  false,
    /*options*/ {
        {"mlx_bin", "--mlx-bin", "", "PATH",
         "Path to lemon-mlx-engine's server binary", "MLX Options"},
        {"mlx_args", "--mlx-args", "", "ARGS",
         "Extra arguments appended to the MLX server command line", "MLX Options"},
    },
    /*support*/ {
        {"metal", {"macos"}, {{"metal", {}}}, "Apple Silicon GPU"},
    },
    /*supported_modes*/ {"chat"},
    /*required_checkpoints*/ {"main"},
    /*default_capabilities*/ {},
    /*experimental*/    true,
    /*web_display_name*/ "",
    /*rocm_channels*/   {},
    /*exposes_prometheus_metrics*/ false,
    /*rocm_requires_cwsr_fix*/ false,
    /*version_policy*/  VersionPolicy::AtLeast,
    /*self_manages_downloads*/ true,  // the MLX server fetches the checkpoint from Hugging Face
    /*takes_args*/      true,
    /*arg_variants*/    {},
    /*bin_variants*/    {},
    /*config_extra*/    nlohmann::json::object(),
};

}  // namespace mlx
}  // namespace backends
}  // namespace lemon
