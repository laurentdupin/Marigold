#include "vae_cpu.h"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>

namespace marigold_native {
namespace {

ImageTensor mid_block(
    const SafeTensors& model,
    ImageTensor hidden,
    const std::string& prefix) {
    hidden = resnet(model, hidden, prefix + ".resnets.0");
    hidden = spatial_attention(
        model, hidden, prefix + ".attentions.0");
    return resnet(model, hidden, prefix + ".resnets.1");
}

}  // namespace

Posterior vae_encode(
    const SafeTensors& model,
    const ImageTensor& rgb) {
    ImageTensor hidden = conv2d(
        model, rgb, "encoder.conv_in.weight", "encoder.conv_in.bias");
    for (std::uint32_t block = 0; block < 4; ++block) {
        for (std::uint32_t layer = 0; layer < 2; ++layer) {
            hidden = resnet(
                model, hidden,
                "encoder.down_blocks." + std::to_string(block) +
                    ".resnets." + std::to_string(layer));
        }
        if (block != 3) {
            const std::string prefix =
                "encoder.down_blocks." + std::to_string(block) +
                ".downsamplers.0.conv";
            hidden = conv2d(
                model, hidden, prefix + ".weight", prefix + ".bias",
                2, 0, 1);
        }
    }
    hidden = mid_block(model, std::move(hidden), "encoder.mid_block");
    group_norm(
        model, hidden, "encoder.conv_norm_out.weight",
        "encoder.conv_norm_out.bias");
    silu(hidden);
    hidden = conv2d(
        model, hidden, "encoder.conv_out.weight",
        "encoder.conv_out.bias");
    hidden = conv2d(
        model, hidden, "quant_conv.weight", "quant_conv.bias",
        1, 0, 0);
    if (hidden.channels != 8) {
        throw std::runtime_error("Marigold VAE posterior must have 8 channels");
    }

    Posterior result;
    result.mean = {4, hidden.height, hidden.width, {}};
    result.log_variance = {4, hidden.height, hidden.width, {}};
    const std::uint64_t plane =
        std::uint64_t(hidden.height) * hidden.width;
    result.mean.values.assign(
        hidden.values.begin(), hidden.values.begin() + 4 * plane);
    result.log_variance.values.assign(
        hidden.values.begin() + 4 * plane, hidden.values.end());
    for (float& value : result.log_variance.values) {
        value = std::max(-30.0f, std::min(20.0f, value));
    }
    return result;
}

ImageTensor vae_decode(
    const SafeTensors& model,
    const ImageTensor& latent) {
    ImageTensor hidden = conv2d(
        model, latent, "post_quant_conv.weight",
        "post_quant_conv.bias", 1, 0, 0);
    hidden = conv2d(
        model, hidden, "decoder.conv_in.weight",
        "decoder.conv_in.bias");
    hidden = mid_block(model, std::move(hidden), "decoder.mid_block");

    for (std::uint32_t block = 0; block < 4; ++block) {
        for (std::uint32_t layer = 0; layer < 3; ++layer) {
            hidden = resnet(
                model, hidden,
                "decoder.up_blocks." + std::to_string(block) +
                    ".resnets." + std::to_string(layer));
        }
        if (block != 3) {
            hidden = nearest_upsample_2x(hidden);
            const std::string prefix =
                "decoder.up_blocks." + std::to_string(block) +
                ".upsamplers.0.conv";
            hidden = conv2d(
                model, hidden, prefix + ".weight", prefix + ".bias");
        }
    }
    group_norm(
        model, hidden, "decoder.conv_norm_out.weight",
        "decoder.conv_norm_out.bias");
    silu(hidden);
    return conv2d(
        model, hidden, "decoder.conv_out.weight",
        "decoder.conv_out.bias");
}

}  // namespace marigold_native
