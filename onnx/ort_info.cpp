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
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// Unless required by applicable law or agreed to in writing, software
// limitations under the License.

// ort_info.cpp — `1bit ort-info`: prove the linked ONNX Runtime is the one we
// pinned, and dump the execution-provider roster the host will partition to.
//
// The roster is the load-bearing part: registration order is strict priority
// with no tie-break, so the list printed here is the partition order a session
// will use. CPU is always last.
//
// No session, no model: this is the cheapest possible proof that the library
// loads, the OrtApi slot layout matches the compile-time header, and the
// plugin-EP registration surface exists (GetEpDevices, ORT_API_VERSION >= 22).

#include "onnxruntime_c_api.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <dlfcn.h>

namespace {

const OrtApi* g_api = nullptr;

void report(OrtStatus* status, const char* what) {
    if (!status) return;
    std::fprintf(stderr, "ort-info: %s: %s\n", what, g_api->GetErrorMessage(status));
    g_api->ReleaseStatus(status);
}

const char* device_type_name(OrtHardwareDeviceType t) {
    switch (t) {
        case OrtHardwareDeviceType_CPU: return "CPU";
        case OrtHardwareDeviceType_GPU: return "GPU";
        case OrtHardwareDeviceType_NPU: return "NPU";
    }
    return "?";
}

}  // namespace

int onebit_ort_info_main(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "-h") || !std::strcmp(argv[i], "--help")) {
            std::printf("usage: 1bit ort-info\n\n"
                        "Prints the linked ONNX Runtime version, the OrtApi version it\n"
                        "negotiates, and the execution-provider roster in partition order.\n");
            return 0;
        }
    }

    const OrtApiBase* base = OrtGetApiBase();
    if (!base) {
        std::fprintf(stderr, "ort-info: OrtGetApiBase() returned null — libonnxruntime is not loaded\n");
        return 1;
    }

    // GetApi() is the version negotiation: a null return means the header we
    // compiled against is newer than the library that is actually loaded.
    g_api = base->GetApi(ORT_API_VERSION);
    if (!g_api) {
        std::fprintf(stderr, "ort-info: libonnxruntime %s cannot serve OrtApi v%d (header too new)\n",
                     base->GetVersionString(), ORT_API_VERSION);
        return 1;
    }

    const char* ort_version = base->GetVersionString();

    std::printf("onebit ort-info\n");
    std::printf("  ortv    : %s\n", ort_version ? ort_version : "(unknown)");
    std::printf("  api     : v%d (compile-time ORT_API_VERSION, negotiated above)\n", ORT_API_VERSION);

    Dl_info dl {};
    if (dladdr(const_cast<void*>(static_cast<const void*>(base)), &dl) && dl.dli_fname)
        std::printf("  lib     : %s\n", dl.dli_fname);

    // Compiled-in providers (the ones the .so was built with).
    char** providers = nullptr;
    int nproviders = 0;
    if (OrtStatus* st = g_api->GetAvailableProviders(&providers, &nproviders)) {
        report(st, "GetAvailableProviders");
    } else {
        std::printf("  builtin : %d\n", nproviders);
        for (int i = 0; i < nproviders; ++i) std::printf("    [%d] %s\n", i, providers[i]);
        g_api->ReleaseAvailableProviders(providers, nproviders);
    }

    OrtEnv* env = nullptr;
    if (OrtStatus* st = g_api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "onebit-ort-info", &env)) {
        report(st, "CreateEnv");
        return 1;
    }

    const OrtEpDevice* const* devices = nullptr;
    size_t ndevices = 0;
    if (OrtStatus* st = g_api->GetEpDevices(env, &devices, &ndevices)) {
        report(st, "GetEpDevices");
        g_api->ReleaseEnv(env);
        return 1;
    }

    std::printf("  eproster: %zu (partition order; first EP gets the node, no tie-break)\n", ndevices);
    for (size_t i = 0; i < ndevices; ++i) {
        const OrtEpDevice* d = devices[i];
        const char* ep = g_api->EpDevice_EpName(d);
        const char* vendor = g_api->EpDevice_EpVendor(d);
        const OrtHardwareDevice* hw = g_api->EpDevice_Device(d);
        std::printf("    [%zu] %-22s vendor=%-10s device=%s\n", i, ep ? ep : "(unnamed)",
                    vendor ? vendor : "-", hw ? device_type_name(g_api->HardwareDevice_Type(hw)) : "?");
    }

    g_api->ReleaseEnv(env);

    if (ORT_API_VERSION < 22) {
        std::fprintf(stderr, "ort-info: ORT_API_VERSION < 22 — the plugin-EP ABI is not available\n");
        return 1;
    }
    return 0;
}
