#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#if defined(_WIN32)
#  define WIN32_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#endif

namespace marigold_native {

enum class TensorDType : std::uint8_t {
    f16,
    f32
};

struct TensorData {
    const std::byte* bytes = nullptr;
    TensorDType dtype = TensorDType::f32;

    float operator[](std::uint64_t index) const;
};

struct TensorView {
    TensorData data{};
    std::array<std::uint64_t, 4> dimensions{};
    std::uint32_t rank = 0;
    std::uint64_t elements = 0;
};

class SafeTensors {
public:
    explicit SafeTensors(const std::string& path_utf8);
    SafeTensors(const SafeTensors&) = delete;
    SafeTensors& operator=(const SafeTensors&) = delete;
    ~SafeTensors();

    const TensorView& tensor(std::string_view name) const;
    bool contains(std::string_view name) const;
    std::size_t tensor_count() const { return tensors_.size(); }
    const std::vector<std::string>& tensor_names() const {
        return names_;
    }

private:
    void close() noexcept;

#if defined(_WIN32)
    HANDLE file_ = INVALID_HANDLE_VALUE;
    HANDLE mapping_ = nullptr;
#else
    int file_descriptor_ = -1;
#endif
    const std::byte* view_ = nullptr;
    std::uint64_t size_ = 0;
    std::vector<std::string> names_;
    std::unordered_map<std::string, TensorView> tensors_;
    std::unordered_map<std::string, std::string> aliases_;
};

}  // namespace marigold_native
