// gguf.h: read-only GGUF v2/v3 reader over a memory-mapped file.
//
// Format reference: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md
// Metadata values are decoded into GgufValue; tensor data is never copied, and
// GgufTensor::data points into the mapping, which lives as long as the GgufFile.
#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace onebit {

// ggml tensor types this engine knows by id. Values match ggml.h.
enum class GgmlType : uint32_t {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    Q8_K = 15,
    BF16 = 30,
};

std::string_view ggml_type_name(GgmlType t);

struct GgufValue;
using GgufArray = std::vector<GgufValue>;

struct GgufValue {
    std::variant<uint64_t, int64_t, double, bool, std::string, GgufArray> v;
};

struct GgufTensor {
    std::string name;
    GgmlType type;
    std::vector<uint64_t> ne;  // ne[0] is the contiguous (fastest) dimension
    uint64_t offset = 0;       // relative to the start of the data section
    const std::byte* data = nullptr;

    uint64_t n_elements() const;
};

class GgufFile {
public:
    static std::expected<GgufFile, std::string> open(const std::string& path);

    GgufFile(GgufFile&& other) noexcept;
    GgufFile& operator=(GgufFile&& other) noexcept;
    GgufFile(const GgufFile&) = delete;
    GgufFile& operator=(const GgufFile&) = delete;
    ~GgufFile();

    uint32_t version() const { return version_; }
    const std::map<std::string, GgufValue>& metadata() const { return kv_; }
    const std::vector<GgufTensor>& tensors() const { return tensors_; }

    const GgufValue* find(std::string_view key) const;
    const GgufTensor* tensor(std::string_view name) const;

    // Typed accessors. Integer getters accept any integer metadata type that
    // fits; they return nullopt when the key is missing or has another type.
    std::optional<uint64_t> get_uint(std::string_view key) const;
    std::optional<double> get_float(std::string_view key) const;
    std::optional<std::string> get_string(std::string_view key) const;
    const GgufArray* get_array(std::string_view key) const;

private:
    GgufFile() = default;
    void release();

    void* map_ = nullptr;
    size_t size_ = 0;
    uint32_t version_ = 0;
    std::map<std::string, GgufValue> kv_;
    std::vector<GgufTensor> tensors_;
    std::map<std::string, size_t, std::less<>> by_name_;
};

}  // namespace onebit
