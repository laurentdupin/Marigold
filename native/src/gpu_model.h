#pragma once

#include "safetensors.h"
#include "vulkan.h"
#include <inferbridge/native_harness_precision.h>

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <unordered_map>

namespace marigold_native {

struct GpuTensor {
    VulkanBuffer buffer;
    VulkanBuffer half_buffer;
    VulkanBuffer winograd_buffer;
    VulkanBuffer int8_buffer;
    VulkanBuffer int8_scales;
    std::array<std::uint64_t, 4> dimensions{};
    std::uint32_t rank = 0;
    std::uint64_t elements = 0;
};

class GpuModel {
public:
    GpuModel(const SafeTensors& model, VulkanContext& context);

    const GpuTensor& tensor(std::string_view name) const;
    void retain_transformer_precision(bool half_weight);
    void retain_dpt_precision(bool half_weight);
    void discard_winograd();
    std::size_t tensor_count() const { return tensors_.size(); }
    bool uses_int8_weights() const {
        return precision_ == inferbridge::native::Precision::int8;
    }

private:
    VulkanContext& context_;
    std::unordered_map<std::string_view, GpuTensor> tensors_;
    inferbridge::native::Precision precision_ =
        inferbridge::native::Precision::automatic;
};

}  // namespace marigold_native
