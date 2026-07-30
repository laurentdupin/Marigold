#pragma once

#include "tensor_cpu.h"

#include <cstdint>
#include <string>
#include <vector>

namespace marigold_native {

struct TokenTensor {
    std::uint32_t tokens = 0;
    std::uint32_t dimensions = 0;
    std::vector<float> values;
};

TokenTensor linear(
    const SafeTensors& model,
    const TokenTensor& input,
    const std::string& weight,
    const std::string& bias = {});
TokenTensor transformer2d(
    const SafeTensors& model,
    const ImageTensor& input,
    const TokenTensor& context,
    const std::string& prefix,
    std::uint32_t heads,
    ImageTensor& output);

}  // namespace marigold_native
