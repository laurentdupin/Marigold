#include <inferbridge/linux_model_execution_policy.h>
#include "operators.h"

#include "add_scaled_spv.h"
#include "add_spv.h"
#include "add_position_spv.h"
#include "bilinear_align_true_spv.h"
#include "bilinear_align_true_image_spv.h"
#include "bmm_spv.h"
#include "bmm_score_half_spv.h"
#include "bmm_value_half_spv.h"
#include "conv2d_spv.h"
#include "conv2d_pointwise_gemm_spv.h"
#include "conv2d8_spv.h"
#include "conv2d_half_spv.h"
#include "conv2d8_half_spv.h"
#include "conv2d8_tiled_spv.h"
#include "conv2d8_stride2_tiled_spv.h"
#include "conv2d8_stride2_tiled_half_spv.h"
#include "conv2d8_tiled16x8_spv.h"
#include "conv2d8_tiled16x8_half_spv.h"
#include "conv2d_winograd_spv.h"
#include "conv_transpose_nonoverlap_spv.h"
#include "conv_transpose_nonoverlap_half_spv.h"
#include "gelu_spv.h"
#include "layer_norm_spv.h"
#include "linear_spv.h"
#include "linear16_spv.h"
#include "linear_half_spv.h"
#include "linear16_half_spv.h"
#include "linear_vec8_spv.h"
#include "linear_vec8_half_spv.h"
#include "linear_int8_tiled_spv.h"
#include "quantize_rows_int8_spv.h"
#include "inferbridge/native_harness_precision.h"
#include "prepare_tokens_spv.h"
#include "position_bicubic_spv.h"
#include "project_tokens_spv.h"
#include "project_tokens_half_spv.h"
#include "relu_spv.h"
#include "softmax_lastdim_spv.h"
#include "softmax_lastdim_half_spv.h"
#include "group_norm_spv.h"
#include "group_norm_silu_spv.h"
#include "silu_spv.h"
#include "nearest_spv.h"
#include "concatenate_spv.h"
#include "add_channel_spv.h"
#include "nchw_tokens_spv.h"
#include "geglu_spv.h"
#include "attention_scores_spv.h"
#include "attention_values_spv.h"
#include "attention_bmm_spv.h"
#include "preprocess_rgb_spv.h"
#include "preprocess_texture_spv.h"
#include "seeded_noise_spv.h"
#include "posterior_sample_spv.h"
#include "scale_values_spv.h"
#include "depth_output_spv.h"
#include "depth_minimum_spv.h"
#include "normalize_depth_spv.h"
#include "depth_to_image_spv.h"
#include "scheduler_target_spv.h"
#include "ddim_step_spv.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>

namespace marigold_native {
namespace {

std::uint32_t divide_up(std::uint32_t value, std::uint32_t divisor) {
    return (value + divisor - 1) / divisor;
}

// Separate-input attention tiles retained by Lotus 19b40d5 / 247de48.
// Use a distinct pipeline so the existing fused-QKV BMM contract is unchanged.
struct AttentionBmmParameters {
    std::uint32_t rows, columns, inner, batches;
    std::uint32_t weight_transposed, output_token_major, qkv_embedding;
    std::uint32_t input_qkv_query, weight_qkv_kind, qkv_heads, qkv_tokens;
    float input_scale;
};
static_assert(sizeof(AttentionBmmParameters) == 48);

void require_bytes(
    const VulkanBuffer& buffer,
    std::uint64_t elements,
    const char* name) {
    if (elements > std::numeric_limits<std::uint64_t>::max() / sizeof(float) ||
        buffer.size() < elements * sizeof(float)) {
        throw std::invalid_argument(
            std::string(name) + " Vulkan buffer is too small");
    }
}

void require_half_elements(
    const VulkanBuffer& buffer,
    std::uint64_t elements,
    const char* name) {
    const std::uint64_t words = (elements + 1) / 2;
    if (words > std::numeric_limits<std::uint64_t>::max() /
            sizeof(std::uint32_t) ||
        buffer.size() < words * sizeof(std::uint32_t)) {
        throw std::invalid_argument(
            std::string(name) + " packed-half Vulkan buffer is too small");
    }
}

}  // namespace

VulkanOperators::VulkanOperators(VulkanContext& context)
    : context_(context),
      linear_(context.create_pipeline(
          marigold_linear_spv, marigold_linear_spv_size, 4, 12)),
      linear16_(context.create_pipeline(
          marigold_linear16_spv, marigold_linear16_spv_size, 4, 12)),
      linear_half_(context.create_pipeline(
          marigold_linear_half_spv, marigold_linear_half_spv_size, 4, 12)),
      linear16_half_(context.create_pipeline(
          marigold_linear16_half_spv,
          marigold_linear16_half_spv_size,
          4,
          12)),
      linear_vec8_(context.create_pipeline(
          marigold_linear_vec8_spv,
          marigold_linear_vec8_spv_size,
          4,
          12)),
      linear_vec8_half_(context.create_pipeline(
          marigold_linear_vec8_half_spv,
          marigold_linear_vec8_half_spv_size, 4, 12)),
      quantize_rows_int8_(
          context.supports_packed_int8_dot() &&
              inferbridge::native::requested_precision() ==
                  inferbridge::native::Precision::int8
          ? context.create_pipeline(marigold_quantize_rows_int8_spv,
                marigold_quantize_rows_int8_spv_size,
              {
                                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                            },
                            {
                                VK_ACCESS_SHADER_READ_BIT,
                                VK_ACCESS_SHADER_WRITE_BIT,
                                VK_ACCESS_SHADER_WRITE_BIT,
                            },
                            4)
          : VulkanPipeline{}),
      linear_int8_tiled_(
          context.supports_packed_int8_dot() &&
              inferbridge::native::requested_precision() ==
                  inferbridge::native::Precision::int8
          ? context.create_pipeline(marigold_linear_int8_tiled_spv,
                marigold_linear_int8_tiled_spv_size, 6, 28)
          : VulkanPipeline{}),
      gelu_(context.create_pipeline(
          marigold_gelu_spv, marigold_gelu_spv_size, 2, 4)),
      layer_norm_(context.create_pipeline(
          marigold_layer_norm_spv, marigold_layer_norm_spv_size, 4, 12)),
      add_scaled_(context.create_pipeline(
          marigold_add_scaled_spv, marigold_add_scaled_spv_size, 4, 8)),
      bmm_(context.create_pipeline(
          marigold_bmm_spv, marigold_bmm_spv_size, 3, 44)),
      bmm_score_half_(context.create_pipeline(
          marigold_bmm_score_half_spv,
          marigold_bmm_score_half_spv_size,
          2,
          8)),
      bmm_value_half_(context.create_pipeline(
          marigold_bmm_value_half_spv,
          marigold_bmm_value_half_spv_size,
          3,
          8)),
      softmax_lastdim_(context.create_pipeline(
          marigold_softmax_lastdim_spv,
          marigold_softmax_lastdim_spv_size,
          2,
          8)),
      softmax_lastdim_half_(context.create_pipeline(
          marigold_softmax_lastdim_half_spv,
          marigold_softmax_lastdim_half_spv_size,
          {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
          {
              VK_ACCESS_SHADER_READ_BIT |
              VK_ACCESS_SHADER_WRITE_BIT,
          },
          8)),
      prepare_tokens_(context.create_pipeline(
          marigold_prepare_tokens_spv,
          marigold_prepare_tokens_spv_size,
          5,
          24)),
      position_bicubic_(context.create_pipeline(
          marigold_position_bicubic_spv,
          marigold_position_bicubic_spv_size,
          {
              VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
              VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
              VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
              VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
              VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
          },
          0)),
      add_position_(context.create_pipeline(
          marigold_add_position_spv,
          marigold_add_position_spv_size,
          4,
          20)),
      add_(context.create_pipeline(
          marigold_add_spv, marigold_add_spv_size, 3, 4)),
      project_tokens_(context.create_pipeline(
          marigold_project_tokens_spv,
          marigold_project_tokens_spv_size,
          4,
          20)),
      project_tokens_half_(context.create_pipeline(
          marigold_project_tokens_half_spv,
          marigold_project_tokens_half_spv_size,
          4,
          20)),
      conv2d_(context.create_pipeline(
          marigold_conv2d_spv, marigold_conv2d_spv_size, 4, 48)),
      conv2d_pointwise_gemm_(context.create_pipeline(
          marigold_conv2d_pointwise_gemm_spv,
          marigold_conv2d_pointwise_gemm_spv_size, 4, 48)),
      conv2d8_(context.create_pipeline(
          marigold_conv2d8_spv, marigold_conv2d8_spv_size, 4, 48)),
      conv2d_half_(context.create_pipeline(
          marigold_conv2d_half_spv, marigold_conv2d_half_spv_size, 4, 48)),
      conv2d8_half_(context.create_pipeline(
          marigold_conv2d8_half_spv,
          marigold_conv2d8_half_spv_size,
          4,
          48)),
      conv2d8_tiled_(context.create_pipeline(
          marigold_conv2d8_tiled_spv,
          marigold_conv2d8_tiled_spv_size,
          4,
          48)),
      conv2d8_stride2_tiled_(context.create_pipeline(
          marigold_conv2d8_stride2_tiled_spv,
          marigold_conv2d8_stride2_tiled_spv_size,
          4,
          48)),
      conv2d8_stride2_tiled_half_(context.create_pipeline(
          marigold_conv2d8_stride2_tiled_half_spv,
          marigold_conv2d8_stride2_tiled_half_spv_size, 4, 48)),
      conv2d8_tiled16x8_(context.create_pipeline(
          marigold_conv2d8_tiled16x8_spv,
          marigold_conv2d8_tiled16x8_spv_size,
          4,
          48)),
      conv2d8_tiled16x8_half_(context.create_pipeline(
          marigold_conv2d8_tiled16x8_half_spv,
          marigold_conv2d8_tiled16x8_half_spv_size, 4, 48)),
      conv2d_winograd_(context.create_pipeline(
          marigold_conv2d_winograd_spv,
          marigold_conv2d_winograd_spv_size,
          4,
          48)),
      conv_transpose_nonoverlap_(context.create_pipeline(
          marigold_conv_transpose_nonoverlap_spv,
          marigold_conv_transpose_nonoverlap_spv_size,
          4,
          24)),
      conv_transpose_nonoverlap_half_(context.create_pipeline(
          marigold_conv_transpose_nonoverlap_half_spv,
          marigold_conv_transpose_nonoverlap_half_spv_size,
          4,
          24)),
      bilinear_align_true_(context.create_pipeline(
          marigold_bilinear_align_true_spv,
          marigold_bilinear_align_true_spv_size,
          2,
          24)),
      bilinear_align_true_image_(context.create_pipeline(
          marigold_bilinear_align_true_image_spv,
          marigold_bilinear_align_true_image_spv_size,
          {
              VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
              VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          },
          16)),
      relu_(context.create_pipeline(
          marigold_relu_spv, marigold_relu_spv_size, 2, 4)),
      group_norm_(context.create_pipeline(
          marigold_group_norm_spv, marigold_group_norm_spv_size, 4, 16)),
      group_norm_silu_(context.create_pipeline(
          marigold_group_norm_silu_spv,
          marigold_group_norm_silu_spv_size, 4, 16)),
      silu_(context.create_pipeline(
          marigold_silu_spv, marigold_silu_spv_size, 1, 4)),
      nearest_(context.create_pipeline(
          marigold_nearest_spv, marigold_nearest_spv_size, 2, 20)),
      concatenate_(context.create_pipeline(
          marigold_concatenate_spv, marigold_concatenate_spv_size, 3, 8)),
      add_channel_(context.create_pipeline(
          marigold_add_channel_spv, marigold_add_channel_spv_size, 2, 8)),
      nchw_tokens_(context.create_pipeline(
          marigold_nchw_tokens_spv, marigold_nchw_tokens_spv_size, 2, 12)),
      geglu_(context.create_pipeline(
          marigold_geglu_spv, marigold_geglu_spv_size, 2, 8)),
      attention_scores_(context.create_pipeline(
          marigold_attention_scores_spv,
          marigold_attention_scores_spv_size, 3, 16)),
      attention_values_(context.create_pipeline(
          marigold_attention_values_spv,
          marigold_attention_values_spv_size, 3, 16)),
      attention_scores_tiled_(context.create_pipeline(
          marigold_attention_bmm_spv,
          marigold_attention_bmm_spv_size, 3, 48)),
      attention_values_tiled_(context.create_pipeline(
          marigold_attention_bmm_spv,
          marigold_attention_bmm_spv_size, 3, 48)),
      preprocess_rgb_(context.create_pipeline(
          marigold_preprocess_rgb_spv, marigold_preprocess_rgb_spv_size, 2, 8)),
      preprocess_texture_(context.create_pipeline(
          marigold_preprocess_texture_spv,
          marigold_preprocess_texture_spv_size,
          {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
           VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}, 16)),
      seeded_noise_(context.create_pipeline(
          marigold_seeded_noise_spv, marigold_seeded_noise_spv_size, 1, 12)),
      posterior_sample_(context.create_pipeline(
          marigold_posterior_sample_spv,
          marigold_posterior_sample_spv_size, 3, 4)),
      scale_values_(context.create_pipeline(
          marigold_scale_values_spv, marigold_scale_values_spv_size, 1, 8)),
      depth_output_(context.create_pipeline(
          marigold_depth_output_spv, marigold_depth_output_spv_size, 2, 16)),
      depth_minimum_(context.create_pipeline(
          marigold_depth_minimum_spv, marigold_depth_minimum_spv_size, 2, 4)),
      normalize_depth_(context.create_pipeline(
          marigold_normalize_depth_spv, marigold_normalize_depth_spv_size, 2, 4)),
      depth_to_image_(context.create_pipeline(
          marigold_depth_to_image_spv, marigold_depth_to_image_spv_size,
          {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
           VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}, 8)),
      scheduler_target_(context.create_pipeline(
          marigold_scheduler_target_spv,
          marigold_scheduler_target_spv_size, 2, 4)),
      ddim_step_(context.create_pipeline(
          marigold_ddim_step_spv,
          marigold_ddim_step_spv_size, 3, 12)) {
    linear_.set_debug_name("linear");
    linear16_.set_debug_name("linear16");
    linear_half_.set_debug_name("linear_half");
    linear16_half_.set_debug_name("linear16_half");
    linear_vec8_.set_debug_name("linear_vec8");
    linear_vec8_half_.set_debug_name("linear_vec8_half");
    if (context.supports_packed_int8_dot() &&
        inferbridge::native::requested_precision() ==
            inferbridge::native::Precision::int8) {
        quantize_rows_int8_.set_debug_name("quantize_rows_int8");
        linear_int8_tiled_.set_debug_name("linear_int8_tiled");
    }
    gelu_.set_debug_name("gelu");
    layer_norm_.set_debug_name("layer_norm");
    add_scaled_.set_debug_name("add_scaled");
    bmm_.set_debug_name("bmm");
    bmm_score_half_.set_debug_name("bmm_score_half");
    bmm_value_half_.set_debug_name("bmm_value_half");
    softmax_lastdim_.set_debug_name("softmax_lastdim");
    softmax_lastdim_half_.set_debug_name(
        "softmax_lastdim_half");
    prepare_tokens_.set_debug_name("prepare_tokens");
    position_bicubic_.set_debug_name("position_bicubic");
    add_position_.set_debug_name("add_position");
    add_.set_debug_name("add");
    project_tokens_.set_debug_name("project_tokens");
    project_tokens_half_.set_debug_name(
        "project_tokens_half");
    conv2d_.set_debug_name("conv2d");
    conv2d_pointwise_gemm_.set_debug_name("conv2d_pointwise_gemm");
    conv2d8_.set_debug_name("conv2d8");
    conv2d_half_.set_debug_name("conv2d_half");
    conv2d8_half_.set_debug_name("conv2d8_half");
    conv2d8_tiled_.set_debug_name("conv2d8_tiled");
    conv2d8_stride2_tiled_.set_debug_name(
        "conv2d8_stride2_tiled");
    conv2d8_stride2_tiled_half_.set_debug_name(
        "conv2d8_stride2_tiled_half");
    conv2d8_tiled16x8_.set_debug_name("conv2d8_tiled16x8");
    conv2d8_tiled16x8_half_.set_debug_name(
        "conv2d8_tiled16x8_half");
    conv2d_winograd_.set_debug_name("conv2d_winograd");
    conv_transpose_nonoverlap_.set_debug_name(
        "conv_transpose_nonoverlap");
    conv_transpose_nonoverlap_half_.set_debug_name(
        "conv_transpose_nonoverlap_half");
    bilinear_align_true_.set_debug_name(
        "bilinear_align_true");
    bilinear_align_true_image_.set_debug_name(
        "bilinear_align_true_image");
    relu_.set_debug_name("relu");
    group_norm_.set_debug_name("group_norm");
    group_norm_silu_.set_debug_name("group_norm_silu");
    silu_.set_debug_name("silu");
    nearest_.set_debug_name("nearest");
    concatenate_.set_debug_name("concatenate");
    add_channel_.set_debug_name("add_channel");
    nchw_tokens_.set_debug_name("nchw_tokens");
    geglu_.set_debug_name("geglu");
    attention_scores_.set_debug_name("attention_scores");
    attention_values_.set_debug_name("attention_values");
    attention_scores_tiled_.set_debug_name("attention_scores_tiled");
    attention_values_tiled_.set_debug_name("attention_values_tiled");
    // INT8 quantization can amplify the tiled reduction's rounding differences.
    // Keep reduced-precision modes on the scalar path; only FP32 is qualified.
    const char* attention_tiling = std::getenv("MARIGOLD_ATTENTION_TILING");
    attention_tiling_ = attention_tiling != nullptr &&
        std::string(attention_tiling) == "1" &&
        inferbridge::native::requested_precision() ==
            inferbridge::native::Precision::fp32;
    const char* attention_trace = std::getenv("MARIGOLD_ATTENTION_TRACE");
    attention_trace_ = attention_trace != nullptr &&
        std::string(attention_trace) == "1";
    preprocess_rgb_.set_debug_name("preprocess_rgb");
    preprocess_texture_.set_debug_name("preprocess_texture");
    seeded_noise_.set_debug_name("seeded_noise");
    posterior_sample_.set_debug_name("posterior_sample");
    scale_values_.set_debug_name("scale_values");
    depth_output_.set_debug_name("depth_output");
    depth_minimum_.set_debug_name("depth_minimum");
    normalize_depth_.set_debug_name("normalize_depth");
    depth_to_image_.set_debug_name("depth_to_image");
    scheduler_target_.set_debug_name("scheduler_target");
    ddim_step_.set_debug_name("ddim_step");
}

#include <inferbridge/vulkan/operators/vulkan_operators_linear_int8_variant_2.h>

#include <inferbridge/vulkan/operators/vulkan_operators_linear_variant_2.h>

#include <inferbridge/vulkan/operators/vulkan_operators_layer_norm_variant_1.h>

#include <inferbridge/vulkan/operators/vulkan_operators_add_scaled.h>

#include <inferbridge/vulkan/operators/vulkan_operators_attention_head64_variant_1.h>

#include <inferbridge/vulkan/operators/vulkan_operators_prepare_tokens_variant_1.h>

#include <inferbridge/vulkan/operators/vulkan_operators_project_tokens_variant_1.h>

#include <inferbridge/vulkan/operators/vulkan_operators_conv2d_variant_1.h>

void VulkanOperators::conv2d_asymmetric(
    VulkanBuffer& output, const VulkanBuffer& input,
    const VulkanBuffer& weight, const VulkanBuffer& bias,
    std::uint32_t input_width, std::uint32_t input_height,
    std::uint32_t input_channels, std::uint32_t output_channels,
    std::uint32_t kernel, std::uint32_t stride,
    std::uint32_t pad_before, std::uint32_t pad_after,
    bool has_bias, bool winograd, bool half_weight) {
    const std::uint32_t output_width =
        (input_width + pad_before + pad_after - kernel) / stride + 1;
    const std::uint32_t output_height =
        (input_height + pad_before + pad_after - kernel) / stride + 1;
    require_bytes(
        input, std::uint64_t(input_width) * input_height * input_channels,
        "convolution input");
    const std::uint64_t weight_elements =
        std::uint64_t(output_channels) * input_channels * kernel * kernel;
    if (half_weight) {
        require_half_elements(weight, weight_elements, "convolution weight");
    } else {
        require_bytes(weight, weight_elements, "convolution weight");
    }
    require_bytes(bias, has_bias ? output_channels : 1, "convolution bias");
    require_bytes(
        output, std::uint64_t(output_width) * output_height *
            output_channels, "convolution output");
    struct Parameters {
        std::uint32_t input_width, input_height, input_channels;
        std::uint32_t output_width, output_height, output_channels;
        std::uint32_t kernel, stride;
        std::int32_t padding;
        std::uint32_t has_bias, batches, output_channel_blocks;
    };
    const bool tiled =
        !winograd && kernel == 3 && stride == 1 && pad_before == 1 &&
        pad_after == 1 && input_width == output_width &&
        input_height == output_height &&
        inferbridge::native_harness::linux_tiled_convolution_supported(
            context_.subgroup_size(), half_weight);
    const bool stride2_tiled =
        kernel == 3 && stride == 2 &&
        inferbridge::native_harness::linux_tiled_convolution_supported(
            context_.subgroup_size(), half_weight);
    const bool pointwise =
        !winograd && kernel == 1 && stride == 1 &&
        pad_before == 0 && pad_after == 0 &&
        input_width == output_width && input_height == output_height;
    const std::uint32_t blocks =
        divide_up(output_channels, winograd ? 4 :
            ((tiled || stride2_tiled) ? 8 : 4));
    const Parameters parameters{
        input_width, input_height, input_channels,
        output_width, output_height, output_channels,
        kernel, stride, static_cast<std::int32_t>(pad_before),
        has_bias ? 1u : 0u, 1u, blocks};
    context_.dispatch(
        pointwise ? conv2d_pointwise_gemm_ :
        (winograd ? conv2d_winograd_ :
        (tiled ? (half_weight
            ? conv2d8_tiled16x8_half_ : conv2d8_tiled16x8_) :
        (stride2_tiled ? (half_weight
            ? conv2d8_stride2_tiled_half_
            : conv2d8_stride2_tiled_) : conv2d_))),
        {&output, &input, &weight, &bias},
        &parameters, sizeof(parameters),
        pointwise
            ? divide_up(output_width * output_height, 64)
            : (winograd
            ? divide_up(divide_up(output_width, 2), 8)
            : divide_up(output_width, tiled ? 16 : 8)),
        pointwise
            ? divide_up(output_channels, 64)
            : (winograd
            ? divide_up(divide_up(output_height, 2), 8)
            : divide_up(output_height, 8)),
        pointwise ? 1 : blocks);
}

#include <inferbridge/vulkan/operators/vulkan_operators_conv_transpose_nonoverlap_variant_2.h>

#include <inferbridge/vulkan/operators/vulkan_operators_bilinear_align_true_variant_1.h>

#include <inferbridge/vulkan/operators/vulkan_operators_bilinear_align_true_image.h>

#include <inferbridge/vulkan/operators/vulkan_operators_relu.h>

#include <inferbridge/vulkan/operators/vulkan_operators_add.h>

void VulkanOperators::group_norm(
    VulkanBuffer& output, const VulkanBuffer& input,
    const VulkanBuffer& scale,
    const VulkanBuffer& bias, std::uint32_t channels,
    std::uint32_t spatial, float epsilon, bool silu) {
    if (channels == 0 || channels % 32 != 0 || spatial == 0) {
        throw std::invalid_argument("invalid group normalization dimensions");
    }
    require_bytes(output, std::uint64_t(channels) * spatial, "group output");
    require_bytes(input, std::uint64_t(channels) * spatial, "group input");
    require_bytes(scale, channels, "group scale");
    require_bytes(bias, channels, "group bias");
    struct Parameters {
        std::uint32_t channels, spatial, groups;
        float epsilon;
    } parameters{channels, spatial, 32, epsilon};
    context_.dispatch(
        silu ? group_norm_silu_ : group_norm_,
        {&output, &input, &scale, &bias},
        &parameters, sizeof(parameters), 32);
}

#include <inferbridge/vulkan/operators/vulkan_operators_silu.h>

#include <inferbridge/vulkan/operators/vulkan_operators_nearest.h>

#include <inferbridge/vulkan/operators/vulkan_operators_concatenate.h>

#include <inferbridge/vulkan/operators/vulkan_operators_add_channel.h>

#include <inferbridge/vulkan/operators/vulkan_operators_nchw_tokens.h>

#include <inferbridge/vulkan/operators/vulkan_operators_geglu.h>

bool VulkanOperators::tiled_attention_selected(
    std::uint32_t queries, std::uint32_t keys, std::uint32_t heads) const {
    // Leave the two-token prompt and other small/asymmetric multi-head cases
    // on their scalar path. Tail guards in the tile support odd dimensions.
    return attention_tiling_ && queries >= 32u && keys >= 64u &&
        (heads == 1u || queries == keys);
}

void VulkanOperators::attention_separate(
    VulkanBuffer& output, const VulkanBuffer& query,
    const VulkanBuffer& key, const VulkanBuffer& value,
    std::uint32_t queries, std::uint32_t keys,
    std::uint32_t heads, std::uint32_t head_dimensions,
    VulkanBuffer* probability_output) {
    if (!queries || !keys || !heads || !head_dimensions ||
        std::uint64_t(heads) * head_dimensions >
            std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument("invalid separate attention dimensions");
    }
    const std::uint32_t dimensions = heads * head_dimensions;
    require_bytes(query, std::uint64_t(queries) * dimensions, "query");
    require_bytes(key, std::uint64_t(keys) * dimensions, "key");
    require_bytes(value, std::uint64_t(keys) * dimensions, "value");
    require_bytes(output, std::uint64_t(queries) * dimensions, "attention");
    VulkanBuffer owned_scores;
    if (probability_output == nullptr) {
        owned_scores = context_.create_device_buffer(
            std::uint64_t(heads) * queries * keys * sizeof(float));
        probability_output = &owned_scores;
    }
    require_bytes(*probability_output,
        std::uint64_t(heads) * queries * keys, "attention probabilities");
    VulkanBuffer& scores = *probability_output;
    const bool tiled = tiled_attention_selected(queries, keys, heads);
    if (attention_trace_) {
        const std::string shape = std::to_string(queries) + "/" +
            std::to_string(keys) + "/" + std::to_string(heads) + "/" +
            std::to_string(head_dimensions);
        if (traced_attention_shapes_.insert(shape).second) {
            std::fprintf(stderr,
                "MARIGOLD_ATTENTION_PATH queries=%u keys=%u heads=%u head_dimensions=%u selected=%s\n",
                queries, keys, heads, head_dimensions, tiled ? "tiled" : "scalar");
        }
    }
    if (tiled) {
        const bool single_head = heads == 1u;
        AttentionBmmParameters score_parameters{
            queries, keys, head_dimensions, heads, 1u, 0u,
            single_head ? 0u : dimensions, single_head ? 0u : 2u,
            single_head ? 0u : 3u, single_head ? 0u : heads,
            single_head ? 0u : queries,
            1.0f / std::sqrt(static_cast<float>(head_dimensions))};
        context_.dispatch(
            attention_scores_tiled_, {&scores, &query, &key},
            &score_parameters, sizeof(score_parameters),
            divide_up(keys, 32u), divide_up(queries, 64u), heads);
        struct SoftmaxParameters {
            std::uint32_t rows, columns;
        } softmax{heads * queries, keys};
        context_.dispatch(
            softmax_lastdim_, {&scores, &scores},
            &softmax, sizeof(softmax), softmax.rows);
        AttentionBmmParameters value_parameters{
            queries, head_dimensions, keys, heads, 0u,
            single_head ? 0u : 1u, single_head ? 0u : dimensions,
            0u, single_head ? 0u : 3u, single_head ? 0u : heads,
            single_head ? 0u : queries, 1.0f};
        context_.dispatch(
            attention_values_tiled_, {&output, &scores, &value},
            &value_parameters, sizeof(value_parameters),
            divide_up(head_dimensions, 32u), divide_up(queries, 64u), heads);
        return;
    }
    struct Parameters {
        std::uint32_t queries, keys, heads, head_dimensions;
    } parameters{queries, keys, heads, head_dimensions};
    context_.dispatch(
        attention_scores_, {&scores, &query, &key},
        &parameters, sizeof(parameters), divide_up(keys, 64),
        queries * heads);
    struct SoftmaxParameters {
        std::uint32_t rows, columns;
    } softmax{heads * queries, keys};
    context_.dispatch(
        softmax_lastdim_, {&scores, &scores},
        &softmax, sizeof(softmax), softmax.rows);
    context_.dispatch(
        attention_values_, {&output, &scores, &value},
        &parameters, sizeof(parameters),
        divide_up(head_dimensions, 64), queries * heads);
}

#include <inferbridge/vulkan/operators/vulkan_operators_preprocess_rgb.h>

void VulkanOperators::preprocess_texture(
    VulkanBuffer& output, const VulkanImage& input,
    std::uint32_t source_width, std::uint32_t source_height,
    std::uint32_t target_width, std::uint32_t target_height) {
    require_bytes(output, std::uint64_t(target_width) * target_height * 3,
                  "texture preprocessing output");
    struct Parameters { std::uint32_t source_width, source_height, target_width, target_height; }
        parameters{source_width, source_height, target_width, target_height};
    context_.dispatch_image_to_buffer(
        preprocess_texture_, input, output, &parameters, sizeof(parameters),
        divide_up(target_width, 16), divide_up(target_height, 16));
}

void VulkanOperators::seeded_noise(
    VulkanBuffer& noise, std::uint32_t count, std::uint64_t seed) {
    require_bytes(noise, count, "seeded noise");
    struct Parameters { std::uint32_t count, seed_low, seed_high; }
        parameters{count, static_cast<std::uint32_t>(seed),
                   static_cast<std::uint32_t>(seed >> 32u)};
    context_.dispatch(seeded_noise_, {&noise}, &parameters, sizeof(parameters),
                      divide_up(count, 256));
}

#include <inferbridge/vulkan/operators/vulkan_operators_posterior_sample.h>

#include <inferbridge/vulkan/operators/vulkan_operators_scale_values.h>

#include <inferbridge/vulkan/operators/vulkan_operators_depth_output.h>

void VulkanOperators::normalize_depth(
    VulkanBuffer& depth, std::uint32_t count) {
    require_bytes(depth, count, "depth normalization");
    VulkanBuffer range = context_.create_device_buffer(sizeof(float));
    context_.dispatch(depth_minimum_, {&depth, &range}, &count, sizeof(count), 1);
    context_.dispatch(normalize_depth_, {&depth, &range}, &count, sizeof(count),
                      divide_up(count, 256));
}

#include <inferbridge/vulkan/operators/vulkan_operators_depth_to_image.h>

void VulkanOperators::scheduler_target(
    VulkanBuffer& prediction, const VulkanBuffer& noise,
    std::uint32_t count) {
    require_bytes(prediction, count, "scheduler prediction");
    require_bytes(noise, count, "scheduler noise");
    context_.dispatch(
        scheduler_target_, {&prediction, &noise}, &count, sizeof(count),
        divide_up(count, 256));
}

void VulkanOperators::ddim_step(
    VulkanBuffer& output,
    const VulkanBuffer& prediction,
    const VulkanBuffer& sample,
    std::uint32_t count,
    float alpha,
    float previous_alpha) {
    require_bytes(output, count, "DDIM output");
    require_bytes(prediction, count, "DDIM prediction");
    require_bytes(sample, count, "DDIM sample");
    struct Parameters {
        std::uint32_t count;
        float alpha;
        float previous_alpha;
    } parameters{count, alpha, previous_alpha};
    context_.dispatch(
        ddim_step_, {&output, &prediction, &sample},
        &parameters, sizeof(parameters), divide_up(count, 256));
}

}  // namespace marigold_native
