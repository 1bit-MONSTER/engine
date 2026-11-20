#pragma once

// [F] GGUF loader: metadata (every key, typed) and the tensor table with format and shape, read
// from the header alone; tensor bytes are read on demand with pread (never mmapped: design D5,
// the weights V2 hands a device are copied into driver-owned buffers, not registered user pages).
//
// The tensor table (shards, offsets, sizes, format check, crafted-file bounds) is the engine's
// existing reader, onebit::moe::GgufIndex (moe/gguf_index.h, covered by ctest moe_gguf_index);
// this file adds the typed metadata the architecture library needs.

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace onebit::v2 {

struct MetaValue {
    enum class Kind { Int, Float, Bool, Str, IntArray, FloatArray, StrArray, OtherArray } kind = Kind::Int;
    int64_t i = 0;
    double f = 0;
    bool b = false;
    std::string s;
    std::vector<int64_t> ia;
    std::vector<double> fa;
    std::vector<std::string> sa;
};

struct TensorInfo {
    std::string name;
    uint32_t fmt = 0;           // ggml type id (formats.h)
    std::vector<uint64_t> ne;   // ne[0] is the contiguous (input) dimension
    int file = 0;               // shard index
    uint64_t offset = 0;        // absolute offset of the data in that shard
    uint64_t bytes = 0;
    uint64_t n_elements() const;
    std::string shape_str() const;  // "1024x151936"
};

class GgufFile {
public:
    // Throws std::runtime_error on an unreadable, truncated or crafted file.
    static GgufFile open(const std::string& path);

    const std::string& path() const { return files_.front(); }
    const std::vector<std::string>& files() const { return files_; }
    uint64_t file_bytes() const { return file_bytes_; }   // all shards
    uint32_t version() const { return version_; }
    const std::string& arch() const { return arch_; }

    const std::map<std::string, MetaValue>& meta() const { return meta_; }
    const MetaValue* find(const std::string& key) const;
    int64_t get_int(const std::string& key, int64_t dflt) const;
    double get_float(const std::string& key, double dflt) const;   // ints convert
    std::string get_str(const std::string& key, const std::string& dflt = "") const;
    // "<arch>.<key>", e.g. arch_int("block_count", 0)
    int64_t arch_int(const std::string& key, int64_t dflt) const { return get_int(arch_ + "." + key, dflt); }
    double arch_float(const std::string& key, double dflt) const { return get_float(arch_ + "." + key, dflt); }

    const std::vector<TensorInfo>& tensors() const { return tensors_; }
    const TensorInfo* tensor(const std::string& name) const;
    // Reads a tensor's bytes (exactly t.bytes) into dst with pread; throws on a short read.
    void read(const TensorInfo& t, void* dst) const;

private:
    std::vector<std::string> files_;
    uint64_t file_bytes_ = 0;
    uint32_t version_ = 0;
    std::string arch_;
    std::map<std::string, MetaValue> meta_;
    std::vector<TensorInfo> tensors_;
    std::map<std::string, size_t> by_name_;
};

}  // namespace onebit::v2
