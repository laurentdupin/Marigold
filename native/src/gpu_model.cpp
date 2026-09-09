#include "gpu_model.h"
#include <inferbridge/linux_model_execution_policy.h>
#include "inferbridge/native_harness_precision.h"
#include "inferbridge/native_harness_vulkan_initialization.h"

#include <array>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace marigold_native {
namespace {

std::uint16_t float_to_half(float input) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &input, sizeof(bits));
    const std::uint32_t sign = (bits >> 16) & 0x8000u;
    const std::uint32_t exponent = (bits >> 23) & 0xffu;
    std::uint32_t mantissa = bits & 0x7fffffu;
    if (exponent == 0xffu) {
        return static_cast<std::uint16_t>(
            sign | (mantissa != 0 ? 0x7e00u : 0x7c00u));
    }
    const int half_exponent = static_cast<int>(exponent) - 127 + 15;
    if (half_exponent >= 31) {
        return static_cast<std::uint16_t>(sign | 0x7c00u);
    }
    if (half_exponent <= 0) {
        if (half_exponent < -10) {
            return static_cast<std::uint16_t>(sign);
        }
        mantissa |= 0x800000u;
        const std::uint32_t shift =
            static_cast<std::uint32_t>(14 - half_exponent);
        std::uint32_t rounded = mantissa >> shift;
        const std::uint32_t remainder = mantissa & ((1u << shift) - 1u);
        const std::uint32_t halfway = 1u << (shift - 1u);
        if (remainder > halfway ||
            (remainder == halfway && (rounded & 1u) != 0)) {
            ++rounded;
        }
        return static_cast<std::uint16_t>(sign | rounded);
    }
    std::uint32_t rounded_mantissa = mantissa >> 13;
    const std::uint32_t remainder = mantissa & 0x1fffu;
    if (remainder > 0x1000u ||
        (remainder == 0x1000u && (rounded_mantissa & 1u) != 0)) {
        ++rounded_mantissa;
        if (rounded_mantissa == 0x400u) {
            rounded_mantissa = 0;
            if (half_exponent + 1 >= 31) {
                return static_cast<std::uint16_t>(sign | 0x7c00u);
            }
            return static_cast<std::uint16_t>(
                sign | ((half_exponent + 1) << 10));
        }
    }
    return static_cast<std::uint16_t>(
        sign | (static_cast<std::uint32_t>(half_exponent) << 10) |
        rounded_mantissa);
}

}  // namespace

GpuModel::GpuModel(const SafeTensors& model, VulkanContext& context)
    : context_(context) {
    precision_ = inferbridge::native::require_supported_precision(
        inferbridge::native::requested_precision(),
        {context.subgroup_size() == 32u,
         context.supports_packed_int8_dot()},
        context.subgroup_size() == 32u
            ? inferbridge::native::Precision::fp16
            : inferbridge::native::Precision::fp32);
    const bool half_weights = precision_ == inferbridge::native::Precision::fp16;
    tensors_.reserve(model.tensor_count());
    inferbridge::native_harness::batch_vulkan_initialization_uploads(
        context, model.tensor_names(),
        [&model](std::string_view name) {
            return model.tensor(name).elements * sizeof(float);
        },
        [&](std::string_view name) {
        const TensorView& source = model.tensor(name);
        if (source.elements >
            std::numeric_limits<std::size_t>::max() / sizeof(float)) {
            throw std::runtime_error(
                "model tensor is too large for this process: " +
                std::string(name));
        }
        const std::size_t bytes =
            static_cast<std::size_t>(source.elements) * sizeof(float);
        const bool packed_only = inferbridge::native_harness::linux_packed_weight_only(
            half_weights, context.subgroup_size(), source.rank, source.dimensions);
        GpuTensor destination{
            packed_only ? VulkanBuffer{} : context.create_device_buffer(bytes),
            {},
            {},
            {},
            {},
            source.dimensions,
            source.rank,
            source.elements,
        };
        std::vector<float> converted(
            static_cast<std::size_t>(source.elements));
        for (std::uint64_t index = 0; index < source.elements; ++index) {
            converted[static_cast<std::size_t>(index)] = source.data[index];
        }
        if (!packed_only) context.upload(destination.buffer, converted.data(), bytes);
        if (precision_ == inferbridge::native::Precision::int8 &&
            source.rank == 2 && source.dimensions[1] % 4u == 0u) {
            const auto quantized = inferbridge::native::quantize_int8_rows(
                converted.data(), static_cast<std::size_t>(source.dimensions[0]),
                static_cast<std::size_t>(source.dimensions[1]));
            destination.int8_buffer = context.create_device_buffer(
                quantized.packed.size() * sizeof(std::uint32_t));
            destination.int8_scales = context.create_device_buffer(
                quantized.scales.size() * sizeof(float));
            context.upload(destination.int8_buffer, quantized.packed.data(),
                quantized.packed.size() * sizeof(std::uint32_t));
            context.upload(destination.int8_scales, quantized.scales.data(),
                quantized.scales.size() * sizeof(float));
        }
        if (context.subgroup_size() == 64 &&
            source.rank == 4 &&
            source.dimensions[2] == 3 &&
            source.dimensions[3] == 3) {
            const std::size_t output_channels =
                static_cast<std::size_t>(source.dimensions[0]);
            const std::size_t input_channels =
                static_cast<std::size_t>(source.dimensions[1]);
            std::vector<float> transformed(
                output_channels * input_channels * 16);
            for (std::size_t output = 0;
                 output < output_channels;
                 ++output) {
                for (std::size_t input = 0;
                     input < input_channels;
                     ++input) {
                    const float* kernel = converted.data() +
                        (output * input_channels + input) * 9;
                    float temporary[4][3];
                    for (std::size_t column = 0; column < 3; ++column) {
                        temporary[0][column] = kernel[column];
                        temporary[1][column] = 0.5f * (
                            kernel[column] + kernel[3 + column] +
                            kernel[6 + column]);
                        temporary[2][column] = 0.5f * (
                            kernel[column] - kernel[3 + column] +
                            kernel[6 + column]);
                        temporary[3][column] = kernel[6 + column];
                    }
                    float* destination_values =
                        transformed.data() +
                        (output * input_channels + input) * 16;
                    for (std::size_t row = 0; row < 4; ++row) {
                        destination_values[row * 4] = temporary[row][0];
                        destination_values[row * 4 + 1] = 0.5f * (
                            temporary[row][0] + temporary[row][1] +
                            temporary[row][2]);
                        destination_values[row * 4 + 2] = 0.5f * (
                            temporary[row][0] - temporary[row][1] +
                            temporary[row][2]);
                        destination_values[row * 4 + 3] =
                            temporary[row][2];
                    }
                }
            }
            destination.winograd_buffer = context.create_device_buffer(
                transformed.size() * sizeof(float));
            context.upload(
                destination.winograd_buffer,
                transformed.data(),
                transformed.size() * sizeof(float));
        }
        if (half_weights && context.subgroup_size() == 32 && source.rank == 2 &&
            source.dimensions[1] % 4 == 0) {
            const std::uint64_t outputs = source.dimensions[0];
            const std::uint64_t inputs = source.dimensions[1];
            std::vector<std::uint32_t> packed(
                static_cast<std::size_t>((source.elements + 1) / 2), 0);
            for (std::uint64_t output = 0; output < outputs; ++output) {
                for (std::uint64_t input = 0; input < inputs; ++input) {
                    const std::uint64_t packed_index =
                        ((input / 4) * outputs + output) * 4 + input % 4;
                    packed[static_cast<std::size_t>(packed_index / 2)] |=
                        static_cast<std::uint32_t>(float_to_half(
                            converted[static_cast<std::size_t>(
                                output * inputs + input)])) <<
                        ((packed_index & 1u) * 16u);
                }
            }
            destination.half_buffer = context.create_device_buffer(
                packed.size() * sizeof(std::uint32_t));
            context.upload(
                destination.half_buffer, packed.data(),
                packed.size() * sizeof(std::uint32_t));
            context.discard(destination.buffer);
        } else if (half_weights && context.subgroup_size() == 32 &&
            source.rank == 4 && source.dimensions[2] == 3 &&
            source.dimensions[3] == 3) {
            const std::uint64_t output_channels = source.dimensions[0];
            const std::uint64_t input_channels = source.dimensions[1];
            std::vector<std::uint32_t> packed(
                static_cast<std::size_t>((source.elements + 1) / 2), 0);
            for (std::uint64_t output = 0;
                 output < output_channels; ++output) {
                for (std::uint64_t input = 0;
                     input < input_channels; ++input) {
                    for (std::uint64_t kernel = 0; kernel < 9; ++kernel) {
                        const std::uint64_t source_index =
                            (output * input_channels + input) * 9 + kernel;
                        const std::uint64_t packed_index =
                            (input * 9 + kernel) * output_channels + output;
                        packed[static_cast<std::size_t>(packed_index / 2)] |=
                            static_cast<std::uint32_t>(float_to_half(
                                converted[static_cast<std::size_t>(
                                    source_index)])) <<
                            ((packed_index & 1u) * 16u);
                    }
                }
            }
            destination.half_buffer = context.create_device_buffer(
                packed.size() * sizeof(std::uint32_t));
            context.upload(
                destination.half_buffer, packed.data(),
                packed.size() * sizeof(std::uint32_t));
        }
        if (!tensors_.emplace(name, std::move(destination)).second) {
            throw std::runtime_error(
                "duplicate GPU tensor name: " + std::string(name));
        }
        }, 64ull * 1024ull * 1024ull, context.subgroup_size() == 32u);
}

const GpuTensor& GpuModel::tensor(std::string_view name) const {
    const auto found = tensors_.find(name);
    if (found == tensors_.end()) {
        throw std::runtime_error(
            "GPU model is missing tensor: " + std::string(name));
    }
    return found->second;
}

void GpuModel::retain_transformer_precision(bool half_weight) {
    for (auto& entry : tensors_) {
        const std::string_view name = entry.first;
        if (name.rfind("pretrained.blocks.", 0) != 0 ||
            name.size() < 7 ||
            name.substr(name.size() - 7) != ".weight" ||
            (name.find(".attn.") == std::string_view::npos &&
             name.find(".mlp.") == std::string_view::npos)) {
            continue;
        }
        GpuTensor& tensor = entry.second;
        context_.discard(
            half_weight ? tensor.buffer : tensor.half_buffer);
    }
}

void GpuModel::retain_dpt_precision(bool half_weight) {
    for (auto& entry : tensors_) {
        const std::string_view name = entry.first;
        if (name.rfind("depth_head.", 0) != 0 ||
            name.size() < 7 ||
            name.substr(name.size() - 7) != ".weight") {
            continue;
        }
        GpuTensor& tensor = entry.second;
        context_.discard(
            half_weight ? tensor.buffer : tensor.half_buffer);
    }
}

void GpuModel::discard_winograd() {
    for (auto& entry : tensors_) {
        context_.discard(entry.second.winograd_buffer);
    }
}

}  // namespace marigold_native
