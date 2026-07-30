#pragma once

#include "transformer_cpu.h"

namespace marigold_native {

ImageTensor unet_predict(
    const SafeTensors& model,
    const ImageTensor& sample,
    std::uint32_t timestep,
    const TokenTensor& prompt);

}  // namespace marigold_native
