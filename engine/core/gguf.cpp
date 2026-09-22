#include "gguf.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstring>
#include <format>
#include <utility>

namespace onebit {

std::string_view ggml_type_name(GgmlType t) {
    switch (t) {
        case GgmlType::F32: return "F32";
        case GgmlType::F16: return "F16";
        case GgmlType::Q4_0: return "Q4_0";
        case GgmlType::Q4_1: return "Q4_1";
        case GgmlType::Q5_0: return "Q5_0";
        case GgmlType::Q5_1: return "Q5_1";
        case GgmlType::Q8_0: return "Q8_0";
        case GgmlType::Q8_1: return "Q8_1";
        case GgmlType::Q2_K: return "Q2_K";
        case GgmlType::Q3_K: return "Q3_K";
        case GgmlType::Q4_K: return "Q4_K";
        case GgmlType::Q5_K: return "Q5_K";
        case GgmlType::Q6_K: return "Q6_K";
        case GgmlType::Q8_K: return "Q8_K";
        case GgmlType::BF16: return "BF16";
    }
    return "unknown";
}

uint64_t GgufTensor::n_elements() const {
    uint64_t n = 1;
    for (uint64_t d : ne) n *= d;
    return n;
}

namespace {

constexpr uint32_t kMagic = 0x46554747;  // "GGUF" little-endian
constexpr uint64_t kDefaultAlignment = 32;
// Guards against corrupt headers asking for absurd allocations.
constexpr uint64_t kMaxCount = uint64_t{1} << 32;
constexpr uint32_t kMaxDims = 4;

enum ValueType : uint32_t {
    UINT8 = 0, INT8 = 1, UINT16 = 2, INT16 = 3, UINT32 = 4, INT32 = 5,
    FLOAT32 = 6, BOOL = 7, STRING = 8, ARRAY = 9, UINT64 = 10, INT64 = 11, FLOAT64 = 12,
};

class Cursor {
public:
    Cursor(const std::byte* base, size_t size) : base_(base), size_(size) {}

    size_t pos() const { return pos_; }

    template <typename T>
    std::expected<T, std::string> read() {
        if (size_ - pos_ < sizeof(T)) return std::unexpected(truncated());
        T out;
        std::memcpy(&out, base_ + pos_, sizeof(T));
        pos_ += sizeof(T);
        return out;
    }

    std::expected<std::string, std::string> read_string() {
        auto len = read<uint64_t>();
        if (!len) return std::unexpected(len.error());
        if (size_ - pos_ < *len) return std::unexpected(truncated());
        std::string s(reinterpret_cast<const char*>(base_ + pos_), *len);
        pos_ += *len;
        return s;
    }

    std::expected<GgufValue, std::string> read_value(uint32_t type, int depth = 0) {
        switch (type) {
            case UINT8: return uint_value<uint8_t>();
            case UINT16: return uint_value<uint16_t>();
            case UINT32: return uint_value<uint32_t>();
            case UINT64: return uint_value<uint64_t>();
            case INT8: return int_value<int8_t>();
            case INT16: return int_value<int16_t>();
            case INT32: return int_value<int32_t>();
            case INT64: return int_value<int64_t>();
            case FLOAT32: return float_value<float>();
            case FLOAT64: return float_value<double>();
            case BOOL: {
                auto b = read<uint8_t>();
                if (!b) return std::unexpected(b.error());
                return GgufValue{*b != 0};
            }
            case STRING: {
                auto s = read_string();
                if (!s) return std::unexpected(s.error());
                return GgufValue{std::move(*s)};
            }
            case ARRAY: {
                if (depth > 0) return std::unexpected(std::string("nested GGUF arrays are not supported"));
                auto elem_type = read<uint32_t>();
                if (!elem_type) return std::unexpected(elem_type.error());
                auto count = read<uint64_t>();
                if (!count) return std::unexpected(count.error());
                if (*count > kMaxCount) return std::unexpected(std::string("GGUF array too large"));
                GgufArray arr;
                arr.reserve(*count);
                for (uint64_t i = 0; i < *count; ++i) {
                    auto v = read_value(*elem_type, depth + 1);
                    if (!v) return std::unexpected(v.error());
                    arr.push_back(std::move(*v));
                }
                return GgufValue{std::move(arr)};
            }
        }
        return std::unexpected(std::format("unknown GGUF value type {}", type));
    }

private:
    template <typename T>
    std::expected<GgufValue, std::string> uint_value() {
        auto x = read<T>();
        if (!x) return std::unexpected(x.error());
        return GgufValue{uint64_t{*x}};
    }
    template <typename T>
    std::expected<GgufValue, std::string> int_value() {
        auto x = read<T>();
        if (!x) return std::unexpected(x.error());
        return GgufValue{int64_t{*x}};
    }
    template <typename T>
    std::expected<GgufValue, std::string> float_value() {
        auto x = read<T>();
        if (!x) return std::unexpected(x.error());
        return GgufValue{double{*x}};
    }
    std::string truncated() const { return std::format("GGUF truncated at byte {}", pos_); }

    const std::byte* base_;
    size_t size_;
    size_t pos_ = 0;
};

// Bytes occupied by n elements of type t, or nullopt for types whose block
// size this reader does not know (the tensor is still listed, just unsized).
std::optional<uint64_t> type_bytes(GgmlType t, uint64_t n) {
    auto blocks = [&](uint64_t block, uint64_t bytes) -> std::optional<uint64_t> {
        if (n % block != 0) return std::nullopt;
        return n / block * bytes;
    };
    switch (t) {
        case GgmlType::F32: return n * 4;
        case GgmlType::F16:
        case GgmlType::BF16: return n * 2;
        case GgmlType::Q4_0: return blocks(32, 18);
        case GgmlType::Q4_1: return blocks(32, 20);
        case GgmlType::Q5_0: return blocks(32, 22);
        case GgmlType::Q5_1: return blocks(32, 24);
        case GgmlType::Q8_0: return blocks(32, 34);
        case GgmlType::Q8_1: return blocks(32, 36);
        case GgmlType::Q2_K: return blocks(256, 84);
        case GgmlType::Q3_K: return blocks(256, 110);
        case GgmlType::Q4_K: return blocks(256, 144);
        case GgmlType::Q5_K: return blocks(256, 176);
        case GgmlType::Q6_K: return blocks(256, 210);
        case GgmlType::Q8_K: return blocks(256, 292);
    }
    return std::nullopt;
}

}  // namespace

std::expected<GgufFile, std::string> GgufFile::open(const std::string& path) {
    int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) return std::unexpected(std::format("cannot open {}: {}", path, std::strerror(errno)));
    struct stat st {};
    if (fstat(fd, &st) != 0) {
        ::close(fd);
        return std::unexpected(std::format("cannot stat {}", path));
    }
    GgufFile f;
    f.size_ = static_cast<size_t>(st.st_size);
    void* m = mmap(nullptr, f.size_, PROT_READ, MAP_PRIVATE, fd, 0);
    ::close(fd);
    if (m == MAP_FAILED) return std::unexpected(std::format("cannot mmap {}", path));
    f.map_ = m;

    const auto* base = static_cast<const std::byte*>(m);
    Cursor c(base, f.size_);
    auto fail = [&](std::string msg) { return std::unexpected(path + ": " + msg); };

    auto magic = c.read<uint32_t>();
    if (!magic) return fail(magic.error());
    if (*magic != kMagic) return fail("not a GGUF file (bad magic)");
    auto version = c.read<uint32_t>();
    if (!version) return fail(version.error());
    if (*version != 2 && *version != 3) return fail(std::format("unsupported GGUF version {}", *version));
    f.version_ = *version;

    auto n_tensors = c.read<uint64_t>();
    if (!n_tensors) return fail(n_tensors.error());
    auto n_kv = c.read<uint64_t>();
    if (!n_kv) return fail(n_kv.error());
    if (*n_tensors > kMaxCount || *n_kv > kMaxCount) return fail("header counts out of range");

    for (uint64_t i = 0; i < *n_kv; ++i) {
        auto key = c.read_string();
        if (!key) return fail(key.error());
        auto type = c.read<uint32_t>();
        if (!type) return fail(type.error());
        auto value = c.read_value(*type);
        if (!value) return fail(std::format("key {}: {}", *key, value.error()));
        f.kv_.insert_or_assign(std::move(*key), std::move(*value));
    }

    f.tensors_.reserve(*n_tensors);
    for (uint64_t i = 0; i < *n_tensors; ++i) {
        GgufTensor t;
        auto name = c.read_string();
        if (!name) return fail(name.error());
        t.name = std::move(*name);
        auto n_dims = c.read<uint32_t>();
        if (!n_dims) return fail(n_dims.error());
        if (*n_dims == 0 || *n_dims > kMaxDims) return fail(std::format("tensor {}: bad rank {}", t.name, *n_dims));
        for (uint32_t d = 0; d < *n_dims; ++d) {
            auto dim = c.read<uint64_t>();
            if (!dim) return fail(dim.error());
            t.ne.push_back(*dim);
        }
        auto type = c.read<uint32_t>();
        if (!type) return fail(type.error());
        t.type = static_cast<GgmlType>(*type);
        auto offset = c.read<uint64_t>();
        if (!offset) return fail(offset.error());
        t.offset = *offset;
        f.tensors_.push_back(std::move(t));
    }

    uint64_t alignment = f.get_uint("general.alignment").value_or(kDefaultAlignment);
    if (alignment == 0 || (alignment & (alignment - 1)) != 0) return fail("general.alignment is not a power of two");
    uint64_t data_start = (c.pos() + alignment - 1) / alignment * alignment;

    for (size_t i = 0; i < f.tensors_.size(); ++i) {
        GgufTensor& t = f.tensors_[i];
        uint64_t begin = data_start + t.offset;
        auto bytes = type_bytes(t.type, t.n_elements());
        if (!bytes) return fail(std::format("tensor {}: unsupported type {} or bad shape", t.name,
                                            static_cast<uint32_t>(t.type)));
        if (begin > f.size_ || *bytes > f.size_ - begin) return fail(std::format("tensor {} extends past end of file", t.name));
        t.data = base + begin;
        if (!f.by_name_.emplace(t.name, i).second) return fail(std::format("duplicate tensor {}", t.name));
    }
    return f;
}

GgufFile::GgufFile(GgufFile&& o) noexcept { *this = std::move(o); }

GgufFile& GgufFile::operator=(GgufFile&& o) noexcept {
    if (this != &o) {
        release();
        map_ = std::exchange(o.map_, nullptr);
        size_ = std::exchange(o.size_, 0);
        version_ = o.version_;
        kv_ = std::move(o.kv_);
        tensors_ = std::move(o.tensors_);
        by_name_ = std::move(o.by_name_);
    }
    return *this;
}

GgufFile::~GgufFile() { release(); }

void GgufFile::release() {
    if (map_) munmap(map_, size_);
    map_ = nullptr;
    size_ = 0;
}

const GgufValue* GgufFile::find(std::string_view key) const {
    auto it = kv_.find(std::string(key));
    return it == kv_.end() ? nullptr : &it->second;
}

const GgufTensor* GgufFile::tensor(std::string_view name) const {
    auto it = by_name_.find(name);
    return it == by_name_.end() ? nullptr : &tensors_[it->second];
}

std::optional<uint64_t> GgufFile::get_uint(std::string_view key) const {
    const GgufValue* v = find(key);
    if (!v) return std::nullopt;
    if (auto* u = std::get_if<uint64_t>(&v->v)) return *u;
    if (auto* i = std::get_if<int64_t>(&v->v); i && *i >= 0) return static_cast<uint64_t>(*i);
    return std::nullopt;
}

std::optional<double> GgufFile::get_float(std::string_view key) const {
    const GgufValue* v = find(key);
    if (!v) return std::nullopt;
    if (auto* d = std::get_if<double>(&v->v)) return *d;
    return std::nullopt;
}

std::optional<std::string> GgufFile::get_string(std::string_view key) const {
    const GgufValue* v = find(key);
    if (!v) return std::nullopt;
    if (auto* s = std::get_if<std::string>(&v->v)) return *s;
    return std::nullopt;
}

const GgufArray* GgufFile::get_array(std::string_view key) const {
    const GgufValue* v = find(key);
    return v ? std::get_if<GgufArray>(&v->v) : nullptr;
}

}  // namespace onebit
