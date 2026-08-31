#pragma once

#include "model_bundle.h"
#include "transformer_cpu.h"
#include "external_gpu.h"

#include <cstdint>
#include <memory>
#include <string>

namespace marigold_native {

class MetalExecutor {
public:
    MetalExecutor(
        const ModelBundle& model,
        const TokenTensor& prompt,
        bool full_v1);
    ~MetalExecutor();
    MetalExecutor(const MetalExecutor&) = delete;
    MetalExecutor& operator=(const MetalExecutor&) = delete;

    void set_cache_path(const std::string& cache_path);

    ImageTensor infer(
        const float* rgb,
        std::uint32_t width,
        std::uint32_t height,
        const float* target_noise);
    std::shared_ptr<ExternalJob> submit_texture(
        const ExternalTextureRequest& request);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace marigold_native
