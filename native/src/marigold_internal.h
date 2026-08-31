#pragma once
#include "external_gpu.h"
#include "marigold_native.h"
#include <memory>
namespace marigold_native {
std::shared_ptr<ExternalGpu> create_metal_external_gpu(
    marigold_context* context);
}
