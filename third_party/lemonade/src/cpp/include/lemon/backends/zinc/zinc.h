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
//
// zinc: GGUF models through ZINC's server (upstream zolotukhin/zinc).
//
// Local delta (1bit engine; see UPSTREAM.md). The executor is the `zinc` binary
// that the engine builds from third_party/zinc (scripts/build-zinc.sh). One build
// has one GPU backend (Vulkan, ROCm, CUDA or Metal); CUDA is the engine's route to
// NVIDIA GPUs. zinc is not a downloadable Lemonade artifact, so its path comes
// from the `zinc_bin` option (config key `zinc.zinc_bin`), then
// $LEMONADE_ZINC_SERVER, then `zinc` on PATH. Checkpoints are GGUF files that
// Lemonade downloads and resolves exactly as it does for llamacpp.
#include "lemon/backends/backend_descriptor.h"

#include <nlohmann/json.hpp>

namespace lemon {
namespace backends {
namespace zinc {

inline const BackendDescriptor descriptor = {
    /*recipe*/          "zinc",
    /*display_name*/    "ZINC (Vulkan, ROCm, CUDA, Metal)",
    /*binary*/          "",       // not a Lemonade download: see zinc_bin
    /*config_section*/  "zinc",
    /*default_device*/  DEVICE_GPU,
    /*slot_policy*/     SlotPolicy::Standard,
    /*selectable_backend*/ false,
    /*uses_ctx_size*/   true,     // passed to zinc as -c
    /*dynamic_models*/  false,
    /*options*/ {
        {"zinc_bin", "--zinc-bin", "", "PATH",
         "Path to the zinc binary (scripts/build-zinc.sh)", "ZINC Options"},
        {"zinc_args", "--zinc-args", "", "ARGS",
         "Extra arguments appended to the zinc command line", "ZINC Options"},
    },
    /*support*/ {
        {"vulkan", {"linux"}, {{"amd_gpu", {}}}, "AMD GPUs through Vulkan (RADV)"},
        {"rocm", {"linux"}, {{"amd_gpu", {}}}, "AMD GPUs through ROCm (Qwen3.5/3.6 models)"},
        {"cuda", {"windows", "linux"}, {{"nvidia_gpu", {"sm_89", "sm_120"}}}, "NVIDIA Ada and Blackwell GPUs"},
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
    /*self_manages_downloads*/ false,  // Lemonade downloads the GGUF, as for llamacpp
    /*takes_args*/      true,
    /*arg_variants*/    {},
    /*bin_variants*/    {},
    /*config_extra*/    nlohmann::json::object(),
};

}  // namespace zinc
}  // namespace backends
}  // namespace lemon
