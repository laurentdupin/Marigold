#pragma once

#include "gpu_model.h"
#include "operators.h"
#include "prompt_cache.h"

#include <string>
#include <memory>

namespace marigold_native {

struct GpuImage {
    VulkanBuffer buffer;
    std::uint32_t channels = 0;
    std::uint32_t height = 0;
    std::uint32_t width = 0;
};

class MarigoldGpuGraph {
public:
    MarigoldGpuGraph(
        VulkanContext& context, GpuModel& unet, GpuModel& vae,
        VulkanOperators& operators, const TokenTensor& prompt,
        bool full_v1);
    ~MarigoldGpuGraph();
    MarigoldGpuGraph(const MarigoldGpuGraph&) = delete;
    MarigoldGpuGraph& operator=(const MarigoldGpuGraph&) = delete;

    VulkanBuffer infer_device(
        VulkanBuffer rgb, std::uint32_t width, std::uint32_t height,
        VulkanBuffer target_noise);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

VulkanBuffer marigold_infer_gpu(
    VulkanContext& context,
    GpuModel& unet,
    GpuModel& vae,
    VulkanOperators& operators,
    const TokenTensor& prompt,
    const float* rgb,
    std::uint32_t width,
    std::uint32_t height,
    const float* target_noise,
    bool full_v1);

GpuImage marigold_vae_encode_gpu(
    VulkanContext& context, GpuModel& vae, VulkanOperators& operators,
    const float* normalized_rgb_nchw, std::uint32_t width,
    std::uint32_t height);
GpuImage marigold_unet_gpu(
    VulkanContext& context, GpuModel& unet, VulkanOperators& operators,
    const TokenTensor& prompt, const float* sample_nchw,
    std::uint32_t width, std::uint32_t height,
    std::uint32_t timestep = 999u);
GpuImage marigold_vae_decode_gpu(
    VulkanContext& context, GpuModel& vae, VulkanOperators& operators,
    const float* latent_nchw, std::uint32_t width, std::uint32_t height);
GpuImage marigold_spatial_attention_gpu(
    VulkanContext& context, GpuModel& vae, VulkanOperators& operators,
    const float* input_nchw, std::uint32_t channels,
    std::uint32_t width, std::uint32_t height,
    const std::string& prefix);

}  // namespace marigold_native
