#pragma once
#include "external_gpu.h"
#include "marigold_native.h"
#include <memory>
#include <string>
namespace marigold_native {
std::shared_ptr<ExternalGpu> create_metal_external_gpu(
    marigold_context* context, const std::string& cache_path);
}
