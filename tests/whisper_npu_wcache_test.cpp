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
// tests/whisper_npu_wcache_test.cpp -- engine#1783 / #3036 review: the packed-weight cache's identity and refusals,
// device-free (a fake target holds the "images").
//  1. two encoder graphs in ONE directory get different identities and different files;
//  2. a file restores only into the identity it was written for, and refuses another graph's;
//  3. a graph rewritten with new bytes at the same name, size and mtime (what `cp -a`/`touch -r` produce) changes the
//     identity, so the old file is refused, not restored;
//  4. a kernel file rewritten the same way changes it too;
//  5. the directory keeps at most kKeep cache files.

#include <sys/stat.h>
#include <sys/time.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <string>
#include <vector>

#include "whisper_npu_wcache.h"

namespace t = ::npu::target;
using onebit::audio::NpuAudioDesign;
using onebit::audio::NpuAudioGemm;

static int failures = 0;
static void check(bool ok, const std::string& what) {
    std::printf("%s %s\n", ok ? "ok  " : "FAIL", what.c_str());
    if (!ok) ++failures;
}

struct FakeTarget : t::NpuTarget {
    std::vector<NpuAudioDesign> d;
    std::map<int, std::vector<float>> cache;
    const char* name() const override { return "fake"; }
    bool open(const onebit::npu::plan::Plan&, const std::string&, const char*, std::string&) override { return true; }
    bool load_block(int, int, std::string&) override { return true; }
    t::Caps caps() const override { return t::Caps{}; }
    bool gemv(const t::GemvArgs& a, std::vector<float>& out, std::string&) override {
        std::vector<float>* w = nullptr;
        if (a.Bmat != nullptr) w = &(cache[a.wkey] = std::vector<float>(a.Bmat, a.Bmat + size_t(a.K) * size_t(a.N)));
        else w = &cache.at(a.wkey);
        out.assign(size_t(a.M) * size_t(a.N), 0.0f);
        for (int m = 0; m < a.M; ++m)
            for (int k = 0; k < a.K; ++k)
                for (int n = 0; n < a.N; ++n)
                    out[size_t(m) * a.N + n] += (k < a.a_k ? a.x[size_t(m) * a.a_k + k] : 0.0f) * (*w)[size_t(k) * a.N + n];
        return true;
    }
    bool export_weight(int, int, int wkey, std::vector<uint8_t>* img, std::vector<float>* sc) const override {
        const auto it = cache.find(wkey);
        if (it == cache.end()) return false;
        const auto* p = reinterpret_cast<const uint8_t*>(it->second.data());
        img->assign(p, p + it->second.size() * sizeof(float));
        sc->assign(1, 1.0f);
        return true;
    }
    bool import_weight(int, int, int wkey, size_t bytes, const std::function<bool(void*)>& fill, const std::vector<float>&,
                       std::string& err) override {
        std::vector<float> w(bytes / sizeof(float));
        if (!fill(w.data())) { err = "fill"; return false; }
        cache[wkey] = std::move(w);
        return true;
    }
};

static void write_file(const std::string& path, const std::string& bytes) {
    std::FILE* f = onebit_fopen_private_wb(path.c_str());
    if (f == nullptr) { std::printf("FAIL cannot write %s\n", path.c_str()); std::exit(1); }
    std::fwrite(bytes.data(), 1, bytes.size(), f);
    std::fclose(f);
}

// rewrite `path` with `bytes` (same size) and restore its old mtime, as `cp -a` / `touch -r` would leave it
static void rewrite_keep_mtime(const std::string& path, const std::string& bytes) {
    struct stat st {};
    ::stat(path.c_str(), &st);
    write_file(path, bytes);
    const struct timespec ts[2] = {st.st_atim, st.st_mtim};
    ::utimensat(AT_FDCWD, path.c_str(), ts, 0);
}

int main() {
    char tmpl[] = "/tmp/onebit-wcache-test.XXXXXX";
    const char* root = ::mkdtemp(tmpl);
    if (root == nullptr) { std::printf("FAIL mkdtemp\n"); return 1; }
    const std::string R = root, G = R + "/pack", K = R + "/kernels", C = R + "/cache";
    ::mkdir(G.c_str(), 0700);
    ::mkdir(K.c_str(), 0700);
    std::string a(4096, 'a'), b(4096, 'b');
    write_file(G + "/g1.onnx", a);
    write_file(G + "/g2.onnx", b);
    write_file(K + "/sealed_i8_b8.elf", std::string(1000, 'k'));

    const std::string id1 = whisper_npu::wcache_identity(G + "/g1.onnx", K, 8);
    const std::string id2 = whisper_npu::wcache_identity(G + "/g2.onnx", K, 8);
    check(id1 != id2, "two graphs in one directory have different identities");
    check(whisper_npu::wcache_path(C, id1) != whisper_npu::wcache_path(C, id2), "and different cache files");
    check(whisper_npu::wcache_identity(G + "/g1.onnx", K, 8) == id1, "the identity is stable across calls");
    check(whisper_npu::content_hash(G + "/g1.onnx", 1) == whisper_npu::content_hash(G + "/g1.onnx", 7),
          "the content hash does not depend on the thread count");

    // fill one weight through a fake device and save it under id1
    FakeTarget ft;
    ft.d = {{8, 256, 512}};
    NpuAudioGemm g(ft, ft.d);
    std::vector<float> A(8 * 200, 0.5f), B(200 * 300, 0.25f), Cv;
    std::string why;
    check(g.gemm("w", A.data(), 8, 200, B.data(), 300, &Cv, &why), "a weight is packed: " + why);
    int nw = 0;
    const std::string p1 = whisper_npu::wcache_path(C, id1);
    check(whisper_npu::wcache_save(C, p1, id1, g, &nw, &why) && nw == 1, "the cache is written: " + why);

    {   // restores into the same identity
        FakeTarget f2;
        f2.d = ft.d;
        NpuAudioGemm g2(f2, f2.d);
        why.clear();
        check(whisper_npu::wcache_load(p1, id1, g2, &why) == 1 && g2.cached("w"), "the same identity restores it");
    }
    {   // another graph's identity is refused even when pointed at this file
        FakeTarget f2;
        f2.d = ft.d;
        NpuAudioGemm g2(f2, f2.d);
        why.clear();
        check(whisper_npu::wcache_load(p1, id2, g2, &why) == 0 && !g2.cached("w") && !why.empty(),
              "another graph's identity is refused: " + why);
    }
    // the graph rewritten with new bytes at the same name, size and mtime: a new identity, and the old file is refused
    rewrite_keep_mtime(G + "/g1.onnx", std::string(4096, 'c'));
    const std::string id1b = whisper_npu::wcache_identity(G + "/g1.onnx", K, 8);
    check(id1b != id1, "a graph rewritten at the same size and mtime changes the identity");
    {
        FakeTarget f2;
        f2.d = ft.d;
        NpuAudioGemm g2(f2, f2.d);
        why.clear();
        check(whisper_npu::wcache_load(p1, id1b, g2, &why) == 0 && !g2.cached("w"),
              "so the stale file is refused, not restored: " + why);
        check(whisper_npu::wcache_load(whisper_npu::wcache_path(C, id1b), id1b, g2, &why) == 0,
              "and the new identity has no file yet");
    }
    rewrite_keep_mtime(K + "/sealed_i8_b8.elf", std::string(1000, 'K'));
    check(whisper_npu::wcache_identity(G + "/g1.onnx", K, 8) != id1b, "a kernel file rewritten the same way changes it too");

    // at most kKeep files stay in the directory
    for (int i = 0; i < 6; ++i) {
        const std::string id = id1 + "#" + std::to_string(i);
        whisper_npu::wcache_save(C, whisper_npu::wcache_path(C, id), id, g, &nw, &why);
        struct timespec ts = {};
        ts.tv_nsec = 2000000;
        nanosleep(&ts, nullptr);
    }
    int left = 0;
    if (DIR* d = opendir(C.c_str())) {
        while (dirent* e = readdir(d)) {
            const std::string nm = e->d_name;
            if (nm.size() > 3 && nm.compare(nm.size() - 3, 3, ".wc") == 0) ++left;
        }
        closedir(d);
    }
    check(left == int(whisper_npu::kKeep), "the directory keeps " + std::to_string(whisper_npu::kKeep) + " cache files (has " +
                                               std::to_string(left) + ")");

    std::string cmd = "rm -rf '" + R + "'";
    if (std::system(cmd.c_str()) != 0) std::printf("note: could not remove %s\n", R.c_str());
    std::printf("whisper npu wcache: %s\n", failures == 0 ? "all pass" : "FAILED");
    return failures == 0 ? 0 : 1;
}
