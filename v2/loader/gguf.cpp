#include "gguf.h"

#include "formats.h"
#include "gguf_index.h"   // moe/: the engine's GGUF tensor-table reader

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <unistd.h>

namespace onebit::v2 {

namespace {

// A forward-only reader over the first shard's header. Every length and count is checked
// against the bytes left in the file (as moe/gguf_index.cpp does), so a crafted header ends in an
// error, not a huge allocation.
struct MetaReader {
    FILE* f = nullptr;
    uint64_t size = 0, pos = 0;
    uint64_t left() const { return pos > size ? 0 : size - pos; }
    void raw(void* p, size_t n) {
        if (n > left() || std::fread(p, 1, n, f) != n) throw std::runtime_error("truncated GGUF header");
        pos += n;
    }
    template <class T> T get() {
        T v;
        raw(&v, sizeof v);
        return v;
    }
    std::string str(uint64_t max = uint64_t(1) << 24) {
        const uint64_t n = get<uint64_t>();
        if (n > left() || n > max) throw std::runtime_error("bad GGUF string length");
        std::string s(static_cast<size_t>(n), '\0');
        raw(s.data(), s.size());
        return s;
    }
};

constexpr size_t kScalarSize[] = {1, 1, 2, 2, 4, 4, 4, 1, 0, 0, 8, 8, 8};

bool is_int_type(uint32_t t) { return t <= 5 || t == 10 || t == 11; }
bool is_float_type(uint32_t t) { return t == 6 || t == 12; }

int64_t read_int(MetaReader& r, uint32_t t) {
    switch (t) {
        case 0: return r.get<uint8_t>();
        case 1: return r.get<int8_t>();
        case 2: return r.get<uint16_t>();
        case 3: return r.get<int16_t>();
        case 4: return r.get<uint32_t>();
        case 5: return r.get<int32_t>();
        case 10: return int64_t(r.get<uint64_t>());
        case 11: return r.get<int64_t>();
        default: throw std::runtime_error("bad GGUF int type");
    }
}

double read_float(MetaReader& r, uint32_t t) { return t == 6 ? double(r.get<float>()) : r.get<double>(); }

MetaValue read_value(MetaReader& r, uint32_t type) {
    MetaValue v;
    if (type == 8) {
        v.kind = MetaValue::Kind::Str;
        v.s = r.str();
    } else if (type == 7) {
        v.kind = MetaValue::Kind::Bool;
        v.b = r.get<uint8_t>() != 0;
    } else if (is_int_type(type)) {
        v.kind = MetaValue::Kind::Int;
        v.i = read_int(r, type);
    } else if (is_float_type(type)) {
        v.kind = MetaValue::Kind::Float;
        v.f = read_float(r, type);
    } else if (type == 9) {
        const uint32_t et = r.get<uint32_t>();
        const uint64_t n = r.get<uint64_t>();
        if (et == 8) {
            if (n > r.left() / 8) throw std::runtime_error("bad GGUF array length");
            v.kind = MetaValue::Kind::StrArray;
            v.sa.reserve(static_cast<size_t>(n));
            for (uint64_t i = 0; i < n; ++i) v.sa.push_back(r.str(uint64_t(1) << 20));
        } else if (et < 13 && kScalarSize[et] != 0) {
            if (n > r.left() / kScalarSize[et]) throw std::runtime_error("bad GGUF array length");
            if (is_int_type(et) || et == 7) {
                v.kind = MetaValue::Kind::IntArray;
                v.ia.reserve(static_cast<size_t>(n));
                for (uint64_t i = 0; i < n; ++i) v.ia.push_back(et == 7 ? int64_t(r.get<uint8_t>()) : read_int(r, et));
            } else {
                v.kind = MetaValue::Kind::FloatArray;
                v.fa.reserve(static_cast<size_t>(n));
                for (uint64_t i = 0; i < n; ++i) v.fa.push_back(read_float(r, et));
            }
        } else {
            throw std::runtime_error("bad GGUF array type");
        }
    } else {
        throw std::runtime_error("bad GGUF value type " + std::to_string(type));
    }
    return v;
}

}  // namespace

uint64_t TensorInfo::n_elements() const {
    uint64_t n = 1;
    for (uint64_t v : ne) n *= v;
    return n;
}

std::string TensorInfo::shape_str() const {
    std::string s;
    for (size_t i = 0; i < ne.size(); ++i) s += (i ? "x" : "") + std::to_string(ne[i]);
    return s;
}

GgufFile GgufFile::open(const std::string& path) {
    GgufFile g;
    // tensor table: the engine's existing reader (shards, offsets, sizes, crafted-file checks)
    const moe::GgufIndex idx = moe::GgufIndex::open(path);
    g.files_ = idx.files;
    for (const auto& f : g.files_) {
        std::error_code ec;
        const auto n = std::filesystem::file_size(f, ec);
        if (ec) throw std::runtime_error("cannot stat " + f);
        g.file_bytes_ += n;
    }
    for (const auto& t : idx.tensors) {
        TensorInfo ti;
        ti.name = t.name;
        ti.fmt = t.type;
        ti.ne = t.ne;
        ti.file = t.file;
        ti.offset = t.offset;
        ti.bytes = t.bytes;
        const uint64_t want = fmt_bytes(ti.fmt, ti.n_elements());
        if (want != ti.bytes) throw std::runtime_error("tensor " + ti.name + ": " + std::to_string(ti.bytes) + " bytes for " + fmt_name(ti.fmt) + " " + ti.shape_str());
        std::error_code ec;
        const uint64_t fsz = std::filesystem::file_size(g.files_[size_t(ti.file)], ec);
        if (ec || ti.offset > fsz || ti.bytes > fsz - ti.offset) throw std::runtime_error("tensor " + ti.name + " runs past the end of its file");
        if (g.by_name_.count(ti.name)) throw std::runtime_error("duplicate tensor " + ti.name);
        g.by_name_[ti.name] = g.tensors_.size();
        g.tensors_.push_back(std::move(ti));
    }

    // metadata: the first shard holds it
    std::unique_ptr<FILE, int (*)(FILE*)> f(std::fopen(path.c_str(), "rb"), &std::fclose);
    if (!f) throw std::runtime_error("cannot open " + path);
    std::setvbuf(f.get(), nullptr, _IOFBF, 1 << 20);
    MetaReader r{f.get()};
    {
        std::error_code ec;
        r.size = std::filesystem::file_size(path, ec);
        if (ec) throw std::runtime_error("cannot stat " + path);
    }
    if (r.get<uint32_t>() != 0x46554747u) throw std::runtime_error(path + " is not a GGUF file");
    g.version_ = r.get<uint32_t>();
    if (g.version_ < 2) throw std::runtime_error("GGUF v1 is not supported");
    (void) r.get<uint64_t>();  // tensor count: taken from the index above
    const uint64_t n_kv = r.get<uint64_t>();
    if (n_kv > r.left() / 12) throw std::runtime_error("bad GGUF header counts");
    for (uint64_t i = 0; i < n_kv; ++i) {
        std::string key = r.str(uint64_t(1) << 16);
        const uint32_t type = r.get<uint32_t>();
        g.meta_[std::move(key)] = read_value(r, type);
    }
    g.arch_ = g.get_str("general.architecture");
    return g;
}

const MetaValue* GgufFile::find(const std::string& key) const {
    auto it = meta_.find(key);
    return it == meta_.end() ? nullptr : &it->second;
}

int64_t GgufFile::get_int(const std::string& key, int64_t dflt) const {
    const MetaValue* v = find(key);
    if (!v) return dflt;
    if (v->kind == MetaValue::Kind::Int) return v->i;
    if (v->kind == MetaValue::Kind::Bool) return v->b;
    return dflt;
}

double GgufFile::get_float(const std::string& key, double dflt) const {
    const MetaValue* v = find(key);
    if (!v) return dflt;
    if (v->kind == MetaValue::Kind::Float) return v->f;
    if (v->kind == MetaValue::Kind::Int) return double(v->i);
    return dflt;
}

std::string GgufFile::get_str(const std::string& key, const std::string& dflt) const {
    const MetaValue* v = find(key);
    return v && v->kind == MetaValue::Kind::Str ? v->s : dflt;
}

const TensorInfo* GgufFile::tensor(const std::string& name) const {
    auto it = by_name_.find(name);
    return it == by_name_.end() ? nullptr : &tensors_[it->second];
}

void GgufFile::read(const TensorInfo& t, void* dst) const {
    const int fd = ::open(files_[size_t(t.file)].c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) throw std::runtime_error("cannot open " + files_[size_t(t.file)]);
    auto* p = static_cast<char*>(dst);
    uint64_t done = 0;
    while (done < t.bytes) {
        const ssize_t n = ::pread(fd, p + done, size_t(std::min<uint64_t>(t.bytes - done, uint64_t(1) << 30)), off_t(t.offset + done));
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) {
            ::close(fd);
            throw std::runtime_error("short read of tensor " + t.name);
        }
        done += uint64_t(n);
    }
    ::close(fd);
}

}  // namespace onebit::v2
