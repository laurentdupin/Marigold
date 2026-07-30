#pragma once

#include "safetensors.h"

#include <cstdint>
#include <string>
#include <vector>

namespace marigold_native {

struct ImageTensor {
    std::uint32_t channels = 0;
    std::uint32_t height = 0;
    std::uint32_t width = 0;
    std::vector<float> values;
};

ImageTensor conv2d(
    const SafeTensors& model,
    const ImageTensor& input,
    const std::string& weight,
    const std::string& bias,
    std::uint32_t stride = 1,
    std::uint32_t pad_before = 1,
    std::uint32_t pad_after = 1);

void group_norm(
    const SafeTensors& model,
    ImageTensor& tensor,
    const std::string& weight,
    const std::string& bias,
    float epsilon = 1.0e-6f);

void silu(ImageTensor& tensor);
void add_in_place(ImageTensor& destination, const ImageTensor& source);
ImageTensor nearest_upsample_2x(const ImageTensor& input);
ImageTensor nearest_upsample(
    const ImageTensor& input,
    std::uint32_t output_height,
    std::uint32_t output_width);
ImageTensor resnet(
    const SafeTensors& model,
    const ImageTensor& input,
    const std::string& prefix);
ImageTensor spatial_attention(
    const SafeTensors& model,
    const ImageTensor& input,
    const std::string& prefix);

}  // namespace marigold_native
