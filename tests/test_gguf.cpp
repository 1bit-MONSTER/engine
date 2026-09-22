#include <cstdio>
#include <string>
#include <variant>

#include "check.h"
#include "gguf.h"
#include "gguf_writer.h"

using namespace onebit;

static void test_roundtrip() {
    GgufWriter w;
    w.str("general.architecture", "qwen3");
    w.u32("qwen3.block_count", 2);
    w.i32("signed", -7);
    w.f32("qwen3.rope.freq_base", 1000000.0f);
    w.str_array("tokenizer.ggml.tokens", {"a", "bb", ""});
    w.tensor_f32("x.weight", {3, 2}, {1, 2, 3, 4, 5, 6});
    w.tensor_f32("y.weight", {1}, {9});
    std::string path = GgufWriter::write_temp(w.build(), "roundtrip");

    auto f = GgufFile::open(path);
    REQUIRE(f.has_value());
    CHECK(f->version() == 3);
    CHECK(f->get_string("general.architecture") == "qwen3");
    CHECK(f->get_uint("qwen3.block_count") == 2u);
    CHECK(!f->get_uint("signed").has_value());  // negative: not representable
    CHECK(f->get_float("qwen3.rope.freq_base") == 1000000.0);
    CHECK(!f->get_string("missing").has_value());
    const GgufArray* toks = f->get_array("tokenizer.ggml.tokens");
    REQUIRE(toks && toks->size() == 3);
    CHECK(std::get<std::string>((*toks)[1].v) == "bb");

    const GgufTensor* x = f->tensor("x.weight");
    REQUIRE(x);
    CHECK(x->type == GgmlType::F32);
    CHECK(x->n_elements() == 6);
    CHECK((x->ne == std::vector<uint64_t>{3, 2}));
    CHECK(reinterpret_cast<const float*>(x->data)[5] == 6.0f);
    const GgufTensor* y = f->tensor("y.weight");
    REQUIRE(y);
    CHECK(reinterpret_cast<uintptr_t>(y->data) % 32 == 0);
    CHECK(reinterpret_cast<const float*>(y->data)[0] == 9.0f);
    CHECK(f->tensor("z.weight") == nullptr);
    std::remove(path.c_str());
}

static void test_rejects_bad_files() {
    GgufWriter w;
    w.tensor_f32("x.weight", {4}, {1, 2, 3, 4});
    auto good = w.build();

    auto bad_magic = good;
    bad_magic[0] = 'X';
    std::string p1 = GgufWriter::write_temp(bad_magic, "badmagic");
    CHECK(!GgufFile::open(p1).has_value());

    auto truncated = good;
    truncated.resize(truncated.size() - 20);  // 16 data bytes + 16 padding: cut into the data
    std::string p2 = GgufWriter::write_temp(truncated, "truncated");
    CHECK(!GgufFile::open(p2).has_value());

    auto header_only = good;
    header_only.resize(20);
    std::string p3 = GgufWriter::write_temp(header_only, "header");
    CHECK(!GgufFile::open(p3).has_value());

    CHECK(!GgufFile::open("/nonexistent/file.gguf").has_value());
    std::remove(p1.c_str());
    std::remove(p2.c_str());
    std::remove(p3.c_str());
}

int main() {
    test_roundtrip();
    test_rejects_bad_files();
    return test_result("test_gguf");
}
