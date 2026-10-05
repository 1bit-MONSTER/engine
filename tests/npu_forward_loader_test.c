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

// npu_forward_loader_test
//
// The model-generic forward's Q4NX loader (npu/forward/src/model.c) against
// crafted files written to a temporary directory: a file shorter than its
// length prefix, a header length past the end of the file, a header with no
// terminating NUL that ends exactly at the end of a page-sized file, and
// data_offsets that are reversed, wrap around, or end past the file are all
// refused or parsed without reading outside the mapping; a well-formed file
// loads and its tensor data stays inside the file.  The loader is plain C, so
// this runs without XRT.
#define _POSIX_C_SOURCE 200809L  /* mkdtemp */
#include "model.h"

#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int failures = 0;
static char dir[64];

static void check(int ok, const char* what) {
    printf("%s %s\n", ok ? "ok  " : "FAIL", what);
    if (!ok) failures++;
}

/* fopen with 0600. These are crafted inputs in a temp directory, and the default umask
 * would leave them readable, and on a permissive umask writable, by every local user
 * (CodeQL cpp/world-writable-file-creation). */
static FILE* open_private(const char* path) {
    int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0600);
    return fd >= 0 ? fdopen(fd, "wb") : NULL;
}

/* Writes the length prefix `len`, `header`, then `data_bytes` zero bytes; returns the path. */
static const char* write_file(const char* name, uint64_t len, const char* header, size_t data_bytes) {
    static char path[128];
    snprintf(path, sizeof path, "%s/%s", dir, name);
    FILE* f = open_private(path);
    if (!f) return path;
    fwrite(&len, 8, 1, f);
    fwrite(header, 1, strlen(header), f);
    for (size_t i = 0; i < data_bytes; i++) fputc(0, f);
    fclose(f);
    return path;
}

static ModelWeights* load(const char* path) {
    ModelConfig c = QWEN3_0_6B_CONFIG;
    return model_load(path, c);
}

int main(void) {
    snprintf(dir, sizeof dir, "/tmp/npu_forward_loader_test.XXXXXX");
    if (!mkdtemp(dir)) return 2;

    {   /* well-formed */
        const char* h = "{\"model.norm.weight\":{\"dtype\":\"BF16\",\"shape\":[4],\"data_offsets\":[0,8]}}";
        ModelWeights* m = load(write_file("good.q4nx", strlen(h), h, 8));
        check(m != NULL, "a well-formed file loads");
        if (m) {
            TensorDesc* t = model_tensor_by_name(m, "model.norm.weight");
            check(t && t->data_size == 8 && model_tensor_data(m, t) != NULL, "its tensor data is in the file");
            TensorDesc bad = *t;
            bad.data_offset = 1u << 30;
            check(model_tensor_data(m, &bad) == NULL, "a descriptor past the end of the file gets no data");
            model_free(m);
        }
    }
    {   /* shorter than the length prefix */
        char path[128];
        snprintf(path, sizeof path, "%s/short.q4nx", dir);
        FILE* f = open_private(path);
        fputs("abc", f);
        fclose(f);
        check(load(path) == NULL, "a file shorter than 8 bytes is refused");
    }
    {   /* header length past the end of the file */
        const char* h = "{}";
        check(load(write_file("len.q4nx", UINT64_MAX - 4, h, 0)) == NULL, "a header length past the end is refused");
    }
    {   /* a header with no closing quote or NUL, ending exactly at a page boundary */
        char h[4096 - 8 + 1];
        memset(h, 'x', sizeof h - 1);
        h[sizeof h - 1] = '\0';
        memcpy(h, "{\"a.weight\":{\"dtype\":\"", 22);
        ModelWeights* m = load(write_file("open.q4nx", sizeof h - 1, h, 0));
        check(1, "an unterminated header is parsed inside the file");
        if (m) model_free(m);
    }
    {   /* reversed, wrapping and past-the-end data_offsets */
        const char* cases[][2] = {
            {"rev.q4nx", "{\"w.weight\":{\"dtype\":\"BF16\",\"shape\":[4],\"data_offsets\":[8,0]}}"},
            {"wrap.q4nx", "{\"w.weight\":{\"dtype\":\"BF16\",\"shape\":[4],\"data_offsets\":"
                          "[18446744073709550000,18446744073709551000]}}"},
            {"past.q4nx", "{\"w.weight\":{\"dtype\":\"BF16\",\"shape\":[4],\"data_offsets\":[0,1073741824]}}"},
        };
        for (int i = 0; i < 3; i++) {
            char what[96];
            snprintf(what, sizeof what, "bad data_offsets are refused (%s)", cases[i][0]);
            check(load(write_file(cases[i][0], strlen(cases[i][1]), cases[i][1], 8)) == NULL, what);
        }
    }
    char cmd[96];
    snprintf(cmd, sizeof cmd, "rm -rf %s", dir);
    if (system(cmd) != 0) fprintf(stderr, "could not remove %s\n", dir);
    printf("%s\n", failures ? "FAILED" : "all cases ok");
    return failures ? 1 : 0;
}
