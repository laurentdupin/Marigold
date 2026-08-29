#pragma once

#include "vulkan.h"

#include <cstdint>

namespace marigold_native {

class VulkanOperators {
public:
    explicit VulkanOperators(VulkanContext& context);

    void linear(
        VulkanBuffer& output,
        const VulkanBuffer& input,
        const VulkanBuffer& weight,
        const VulkanBuffer& bias,
        std::uint32_t rows,
        std::uint32_t input_columns,
        std::uint32_t output_columns,
        bool gelu,
        bool block16 = false,
        bool half_weight = false);
    void linear_int8(
        VulkanBuffer& output, const VulkanBuffer& input,
        const VulkanBuffer& packed_weight,
        const VulkanBuffer& weight_scales, const VulkanBuffer& bias,
        std::uint32_t rows, std::uint32_t input_columns,
        std::uint32_t output_columns, bool gelu = false);

    void layer_norm(
        VulkanBuffer& output,
        const VulkanBuffer& input,
        const VulkanBuffer& weight,
        const VulkanBuffer& bias,
        std::uint32_t rows,
        std::uint32_t columns,
        float epsilon);

    void add_scaled(
        VulkanBuffer& output,
        const VulkanBuffer& residual,
        const VulkanBuffer& addend,
        const VulkanBuffer& scale,
        std::uint32_t count,
        std::uint32_t columns);

    void attention_head64(
        VulkanBuffer& output,
        const VulkanBuffer& qkv,
        std::uint32_t tokens,
        std::uint32_t heads,
        VulkanBuffer* score_scratch = nullptr,
        bool half_scores = false,
        std::uint32_t batches = 1);

    void prepare_tokens(
        VulkanBuffer& output,
        const VulkanBuffer& image,
        const VulkanBuffer& patch_weight,
        const VulkanBuffer& patch_bias,
        const VulkanBuffer& class_token,
        const VulkanBuffer& position,
        std::uint32_t input_width,
        std::uint32_t input_height,
        std::uint32_t embedding,
        std::uint32_t batches = 1);

    void project_tokens(
        VulkanBuffer& output,
        const VulkanBuffer& tokens,
        const VulkanBuffer& weight,
        const VulkanBuffer& bias,
        std::uint32_t width,
        std::uint32_t height,
        std::uint32_t embedding,
        std::uint32_t output_channels,
        bool half_weight = false,
        std::uint32_t batches = 1);

    void conv2d(
        VulkanBuffer& output,
        const VulkanBuffer& input,
        const VulkanBuffer& weight,
        const VulkanBuffer& bias,
        std::uint32_t input_width,
        std::uint32_t input_height,
        std::uint32_t input_channels,
        std::uint32_t output_channels,
        std::uint32_t kernel,
        std::uint32_t stride,
        std::uint32_t padding,
        bool has_bias,
        bool block8 = false,
        bool half_weight = false,
        std::uint32_t batches = 1);
    void conv2d_asymmetric(
        VulkanBuffer& output, const VulkanBuffer& input,
        const VulkanBuffer& weight, const VulkanBuffer& bias,
        std::uint32_t input_width, std::uint32_t input_height,
        std::uint32_t input_channels, std::uint32_t output_channels,
        std::uint32_t kernel, std::uint32_t stride,
        std::uint32_t pad_before, std::uint32_t pad_after,
        bool has_bias, bool winograd = false,
        bool half_weight = false);

    void conv_transpose_nonoverlap(
        VulkanBuffer& output,
        const VulkanBuffer& input,
        const VulkanBuffer& weight,
        const VulkanBuffer& bias,
        std::uint32_t input_width,
        std::uint32_t input_height,
        std::uint32_t input_channels,
        std::uint32_t output_channels,
        std::uint32_t kernel,
        bool half_weight = false,
        std::uint32_t batches = 1);

    void bilinear_align_true(
        VulkanBuffer& output,
        const VulkanBuffer& input,
        std::uint32_t input_width,
        std::uint32_t input_height,
        std::uint32_t output_width,
        std::uint32_t output_height,
        std::uint32_t channels,
        std::uint32_t batches = 1);
    void bilinear_align_true_image(
        VulkanImage& output,
        const VulkanBuffer& input,
        std::uint32_t input_width,
        std::uint32_t input_height,
        std::uint32_t output_width,
        std::uint32_t output_height);

    void relu(VulkanBuffer& output, const VulkanBuffer& input, std::uint32_t count);

    void add(
        VulkanBuffer& output,
        const VulkanBuffer& left,
        const VulkanBuffer& right,
        std::uint32_t count);
    void group_norm(
        VulkanBuffer& values, const VulkanBuffer& scale,
        const VulkanBuffer& bias, std::uint32_t channels,
        std::uint32_t spatial, float epsilon, bool silu = false);
    void silu(VulkanBuffer& values, std::uint32_t count);
    void nearest(
        VulkanBuffer& output, const VulkanBuffer& input,
        std::uint32_t input_width, std::uint32_t input_height,
        std::uint32_t output_width, std::uint32_t output_height,
        std::uint32_t channels);
    void concatenate(
        VulkanBuffer& output, const VulkanBuffer& left,
        const VulkanBuffer& right, std::uint32_t left_count,
        std::uint32_t right_count);
    void add_channel(
        VulkanBuffer& values, const VulkanBuffer& channel,
        std::uint32_t channels, std::uint32_t spatial);
    void nchw_tokens(
        VulkanBuffer& output, const VulkanBuffer& input,
        std::uint32_t tokens, std::uint32_t channels, bool reverse);
    void geglu(
        VulkanBuffer& output, const VulkanBuffer& input,
        std::uint32_t rows, std::uint32_t dimensions);
    void attention_separate(
        VulkanBuffer& output, const VulkanBuffer& query,
        const VulkanBuffer& key, const VulkanBuffer& value,
        std::uint32_t queries, std::uint32_t keys,
        std::uint32_t heads, std::uint32_t head_dimensions = 64);
    void preprocess_rgb(
        VulkanBuffer& output, const VulkanBuffer& input,
        std::uint32_t width, std::uint32_t height);
    void preprocess_texture(
        VulkanBuffer& output, const VulkanImage& input,
        std::uint32_t source_width, std::uint32_t source_height,
        std::uint32_t target_width, std::uint32_t target_height);
    void seeded_noise(
        VulkanBuffer& noise, std::uint32_t count, std::uint64_t seed);
    void posterior_sample(
        VulkanBuffer& output, const VulkanBuffer& posterior,
        const VulkanBuffer& noise, std::uint32_t count);
    void scale_values(
        VulkanBuffer& values, std::uint32_t count, float scale);
    void depth_output(
        VulkanBuffer& output, const VulkanBuffer& decoded,
        std::uint32_t source_width, std::uint32_t source_height,
        std::uint32_t target_width, std::uint32_t target_height);
    void normalize_depth(VulkanBuffer& depth, std::uint32_t count);
    void depth_to_image(
        VulkanImage& output, const VulkanBuffer& depth,
        std::uint32_t width, std::uint32_t height);
    void scheduler_target(
        VulkanBuffer& prediction, const VulkanBuffer& noise,
        std::uint32_t count);
    void ddim_step(
        VulkanBuffer& output, const VulkanBuffer& prediction,
        const VulkanBuffer& sample, std::uint32_t count,
        float alpha, float previous_alpha);

private:
    VulkanContext& context_;
    VulkanPipeline linear_;
    VulkanPipeline linear16_;
    VulkanPipeline linear_half_;
    VulkanPipeline linear16_half_;
    VulkanPipeline linear_vec8_;
    VulkanPipeline linear_vec8_half_;
    VulkanPipeline quantize_rows_int8_;
    VulkanPipeline linear_int8_tiled_;
    VulkanPipeline gelu_;
    VulkanPipeline layer_norm_;
    VulkanPipeline add_scaled_;
    VulkanPipeline bmm_;
    VulkanPipeline bmm_score_half_;
    VulkanPipeline bmm_value_half_;
    VulkanPipeline softmax_lastdim_;
    VulkanPipeline softmax_lastdim_half_;
    VulkanPipeline prepare_tokens_;
    VulkanPipeline position_bicubic_;
    VulkanPipeline add_position_;
    VulkanPipeline add_;
    VulkanPipeline project_tokens_;
    VulkanPipeline project_tokens_half_;
    VulkanPipeline conv2d_;
    VulkanPipeline conv2d_pointwise_gemm_;
    VulkanPipeline conv2d8_;
    VulkanPipeline conv2d_half_;
    VulkanPipeline conv2d8_half_;
    VulkanPipeline conv2d8_tiled_;
    VulkanPipeline conv2d8_stride2_tiled_;
    VulkanPipeline conv2d8_stride2_tiled_half_;
    VulkanPipeline conv2d8_tiled16x8_;
    VulkanPipeline conv2d8_tiled16x8_half_;
    VulkanPipeline conv2d_winograd_;
    VulkanPipeline conv_transpose_nonoverlap_;
    VulkanPipeline conv_transpose_nonoverlap_half_;
    VulkanPipeline bilinear_align_true_;
    VulkanPipeline bilinear_align_true_image_;
    VulkanPipeline relu_;
    VulkanPipeline group_norm_;
    VulkanPipeline group_norm_silu_;
    VulkanPipeline silu_;
    VulkanPipeline nearest_;
    VulkanPipeline concatenate_;
    VulkanPipeline add_channel_;
    VulkanPipeline nchw_tokens_;
    VulkanPipeline geglu_;
    VulkanPipeline attention_scores_;
    VulkanPipeline attention_values_;
    VulkanPipeline preprocess_rgb_;
    VulkanPipeline preprocess_texture_;
    VulkanPipeline seeded_noise_;
    VulkanPipeline posterior_sample_;
    VulkanPipeline scale_values_;
    VulkanPipeline depth_output_;
    VulkanPipeline depth_minimum_;
    VulkanPipeline normalize_depth_;
    VulkanPipeline depth_to_image_;
    VulkanPipeline scheduler_target_;
    VulkanPipeline ddim_step_;
};

}  // namespace marigold_native
