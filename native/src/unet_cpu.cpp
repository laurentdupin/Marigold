#include "unet_cpu.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>

namespace marigold_native {
namespace {

TokenTensor activation(TokenTensor value) {
    for (float& element : value.values) {
        element = element / (1.0f + std::exp(-element));
    }
    return value;
}

TokenTensor timestep_embedding(std::uint32_t timestep) {
    constexpr std::uint32_t dimensions = 320;
    constexpr std::uint32_t half = dimensions / 2;
    TokenTensor embedding{
        1, dimensions, std::vector<float>(dimensions)};
    for (std::uint32_t i = 0; i < half; ++i) {
        const float frequency = std::exp(
            -std::log(10000.0f) * static_cast<float>(i) / half);
        const float argument = timestep * frequency;
        embedding.values[i] = std::cos(argument);
        embedding.values[half + i] = std::sin(argument);
    }
    return embedding;
}

TokenTensor embedding_mlp(
    const SafeTensors& model,
    const TokenTensor& input,
    const std::string& prefix) {
    TokenTensor hidden = linear(
        model, input, prefix + ".linear_1.weight",
        prefix + ".linear_1.bias");
    hidden = activation(std::move(hidden));
    return linear(
        model, hidden, prefix + ".linear_2.weight",
        prefix + ".linear_2.bias");
}

ImageTensor concatenate(
    const ImageTensor& left,
    const ImageTensor& right) {
    if (left.height != right.height || left.width != right.width) {
        throw std::runtime_error("Marigold skip shape mismatch");
    }
    ImageTensor output{
        left.channels + right.channels, left.height, left.width, {}};
    output.values.reserve(left.values.size() + right.values.size());
    output.values.insert(
        output.values.end(), left.values.begin(), left.values.end());
    output.values.insert(
        output.values.end(), right.values.begin(), right.values.end());
    return output;
}

ImageTensor unet_resnet(
    const SafeTensors& model,
    const ImageTensor& input,
    const TokenTensor& time,
    const std::string& prefix) {
    ImageTensor hidden = input;
    group_norm(
        model, hidden, prefix + ".norm1.weight",
        prefix + ".norm1.bias", 1.0e-5f);
    silu(hidden);
    hidden = conv2d(
        model, hidden, prefix + ".conv1.weight",
        prefix + ".conv1.bias");

    TokenTensor time_hidden = activation(time);
    time_hidden = linear(
        model, time_hidden, prefix + ".time_emb_proj.weight",
        prefix + ".time_emb_proj.bias");
    if (time_hidden.dimensions != hidden.channels) {
        throw std::runtime_error("Marigold time projection shape mismatch");
    }
    const std::uint64_t plane =
        std::uint64_t(hidden.height) * hidden.width;
    for (std::uint32_t channel = 0;
         channel < hidden.channels; ++channel) {
        float* values =
            hidden.values.data() + std::uint64_t(channel) * plane;
        for (std::uint64_t i = 0; i < plane; ++i) {
            values[i] += time_hidden.values[channel];
        }
    }
    group_norm(
        model, hidden, prefix + ".norm2.weight",
        prefix + ".norm2.bias", 1.0e-5f);
    silu(hidden);
    hidden = conv2d(
        model, hidden, prefix + ".conv2.weight",
        prefix + ".conv2.bias");

    ImageTensor residual = input;
    if (model.contains(prefix + ".conv_shortcut.weight")) {
        residual = conv2d(
            model, input, prefix + ".conv_shortcut.weight",
            prefix + ".conv_shortcut.bias", 1, 0, 0);
    }
    add_in_place(hidden, residual);
    return hidden;
}

ImageTensor apply_transformer(
    const SafeTensors& model,
    const ImageTensor& hidden,
    const TokenTensor& prompt,
    const std::string& prefix,
    std::uint32_t heads) {
    ImageTensor output;
    transformer2d(model, hidden, prompt, prefix, heads, output);
    return output;
}

}  // namespace

ImageTensor unet_predict(
    const SafeTensors& model,
    const ImageTensor& sample,
    std::uint32_t timestep,
    const TokenTensor& prompt) {
    if (sample.channels != 8 || prompt.dimensions != 1024) {
        throw std::invalid_argument("invalid Marigold UNet input");
    }
    TokenTensor time = embedding_mlp(
        model, timestep_embedding(timestep), "time_embedding");

    ImageTensor hidden = conv2d(
        model, sample, "conv_in.weight", "conv_in.bias");
    std::vector<ImageTensor> skips;
    skips.push_back(hidden);
    constexpr std::uint32_t down_heads[3] = {5, 10, 20};
    for (std::uint32_t block = 0; block < 4; ++block) {
        for (std::uint32_t layer = 0; layer < 2; ++layer) {
            const std::string root =
                "down_blocks." + std::to_string(block);
            hidden = unet_resnet(
                model, hidden, time,
                root + ".resnets." + std::to_string(layer));
            if (block < 3) {
                hidden = apply_transformer(
                    model, hidden, prompt,
                    root + ".attentions." + std::to_string(layer),
                    down_heads[block]);
            }
            skips.push_back(hidden);
        }
        if (block != 3) {
            const std::string prefix =
                "down_blocks." + std::to_string(block) +
                ".downsamplers.0.conv";
            hidden = conv2d(
                model, hidden, prefix + ".weight", prefix + ".bias",
                2, 1, 1);
            skips.push_back(hidden);
        }
    }

    hidden = unet_resnet(
        model, hidden, time, "mid_block.resnets.0");
    hidden = apply_transformer(
        model, hidden, prompt, "mid_block.attentions.0", 20);
    hidden = unet_resnet(
        model, hidden, time, "mid_block.resnets.1");

    constexpr std::uint32_t up_heads[4] = {0, 20, 10, 5};
    for (std::uint32_t block = 0; block < 4; ++block) {
        for (std::uint32_t layer = 0; layer < 3; ++layer) {
            if (skips.empty()) {
                throw std::runtime_error("Marigold skip stack underflow");
            }
            ImageTensor skip = std::move(skips.back());
            skips.pop_back();
            hidden = concatenate(hidden, skip);
            const std::string root =
                "up_blocks." + std::to_string(block);
            hidden = unet_resnet(
                model, hidden, time,
                root + ".resnets." + std::to_string(layer));
            if (block != 0) {
                hidden = apply_transformer(
                    model, hidden, prompt,
                    root + ".attentions." + std::to_string(layer),
                    up_heads[block]);
            }
        }
        if (block != 3) {
            if (skips.empty()) {
                throw std::runtime_error(
                    "Marigold skip stack lacks upsample target");
            }
            hidden = nearest_upsample(
                hidden, skips.back().height, skips.back().width);
            const std::string prefix =
                "up_blocks." + std::to_string(block) +
                ".upsamplers.0.conv";
            hidden = conv2d(
                model, hidden, prefix + ".weight", prefix + ".bias");
        }
    }
    if (!skips.empty()) {
        throw std::runtime_error("Marigold skip stack was not consumed");
    }
    group_norm(
        model, hidden, "conv_norm_out.weight",
        "conv_norm_out.bias", 1.0e-5f);
    silu(hidden);
    return conv2d(
        model, hidden, "conv_out.weight", "conv_out.bias");
}

}  // namespace marigold_native
