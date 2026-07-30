#include "marigold_gpu.h"

#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace marigold_native {
namespace {

struct GpuTokens {
    VulkanBuffer buffer;
    std::uint32_t tokens = 0;
    std::uint32_t dimensions = 0;
};

std::uint64_t elements(const GpuImage& image) {
    return std::uint64_t(image.channels) * image.height * image.width;
}

const GpuTensor& tensor(const GpuModel& model, const std::string& name) {
    return model.tensor(name);
}

class Graph {
public:
    Graph(
        VulkanContext& context, GpuModel& unet, GpuModel& vae,
        VulkanOperators& operators, const TokenTensor& prompt)
        : context_(context), unet_(unet), vae_(vae), operators_(operators),
          zero_bias_(context.create_device_buffer(16384 * sizeof(float))) {
        std::vector<float> zeros(16384, 0.0f);
        context_.upload(
            zero_bias_, zeros.data(), zeros.size() * sizeof(float));
        if (!prompt.values.empty()) {
            prompt_.tokens = prompt.tokens;
            prompt_.dimensions = prompt.dimensions;
            prompt_.buffer = context_.create_device_buffer(
                prompt.values.size() * sizeof(float));
            context_.upload(
                prompt_.buffer, prompt.values.data(),
                prompt.values.size() * sizeof(float));
        }
    }

    GpuImage run(
        const float* rgb, std::uint32_t width, std::uint32_t height,
        const float* target_noise) {
        const std::uint32_t latent_width = width / 8;
        const std::uint32_t latent_height = height / 8;
        const std::uint32_t latent_count =
            4 * latent_width * latent_height;
        VulkanBuffer host_rgb = context_.create_device_buffer(
            std::uint64_t(width) * height * 3 * sizeof(float));
        context_.upload(
            host_rgb, rgb,
            std::uint64_t(width) * height * 3 * sizeof(float));
        GpuImage image{
            context_.create_device_buffer(
                std::uint64_t(width) * height * 3 * sizeof(float)),
            3, height, width};
        operators_.preprocess_rgb(image.buffer, host_rgb, width, height);
        GpuImage posterior = vae_encode(std::move(image));
        operators_.scale_values(
            posterior.buffer, latent_count, 0.18215f);
        VulkanBuffer noise_buffer =
            context_.create_device_buffer(latent_count * sizeof(float));
        context_.upload(
            noise_buffer, target_noise, latent_count * sizeof(float));
        GpuImage sample{
            context_.create_device_buffer(
                std::uint64_t(latent_count) * 2 * sizeof(float)),
            8, latent_height, latent_width};
        operators_.concatenate(
            sample.buffer, posterior.buffer, noise_buffer,
            latent_count, latent_count);
        GpuImage prediction = unet_predict(std::move(sample));
        operators_.scheduler_target(
            prediction.buffer, noise_buffer, latent_count);
        return vae_decode(std::move(prediction));
    }

    GpuImage test_encode(GpuImage&& image) {
        return vae_encode(std::move(image));
    }
    GpuImage test_predict(GpuImage&& sample) {
        return unet_predict(std::move(sample));
    }
    GpuImage test_decode(GpuImage&& latent) {
        return vae_decode(std::move(latent));
    }
    GpuImage test_spatial_attention(
        GpuImage&& image, const std::string& prefix) {
        return spatial_attention(std::move(image), prefix);
    }

private:
    GpuTokens linear(
        GpuModel& model, GpuTokens&& input,
        const std::string& weight_name,
        const std::string& bias_name = {}) {
        const GpuTensor& kernel = tensor(model, weight_name);
        const std::uint32_t output_dimensions =
            static_cast<std::uint32_t>(kernel.dimensions[0]);
        GpuTokens output{
            context_.create_device_buffer(
                std::uint64_t(input.tokens) * output_dimensions *
                sizeof(float)),
            input.tokens, output_dimensions};
        operators_.linear(
            output.buffer, input.buffer, kernel.buffer,
            bias_name.empty() ? zero_bias_ : tensor(model, bias_name).buffer,
            input.tokens, input.dimensions, output_dimensions, false);
        return output;
    }

    GpuImage conv(
        GpuModel& model, GpuImage&& input,
        const std::string& weight_name, const std::string& bias_name,
        std::uint32_t stride = 1, std::uint32_t pad_before = 1,
        std::uint32_t pad_after = 1) {
        const GpuTensor& kernel = tensor(model, weight_name);
        const std::uint32_t output_channels =
            static_cast<std::uint32_t>(kernel.dimensions[0]);
        const std::uint32_t kernel_size =
            static_cast<std::uint32_t>(kernel.dimensions[2]);
        const std::uint32_t output_width =
            (input.width + pad_before + pad_after - kernel_size) /
                stride + 1;
        const std::uint32_t output_height =
            (input.height + pad_before + pad_after - kernel_size) /
                stride + 1;
        GpuImage output{
            context_.create_device_buffer(
                std::uint64_t(output_channels) * output_width *
                output_height * sizeof(float)),
            output_channels, output_height, output_width};
        operators_.conv2d_asymmetric(
            output.buffer, input.buffer, kernel.buffer,
            bias_name.empty() ? zero_bias_ : tensor(model, bias_name).buffer,
            input.width, input.height, input.channels, output_channels,
            kernel_size, stride, pad_before, pad_after,
            !bias_name.empty());
        return output;
    }

    void group_norm(
        GpuModel& model, GpuImage& image,
        const std::string& weight_name, const std::string& bias_name,
        float epsilon = 1.0e-6f) {
        operators_.group_norm(
            image.buffer, tensor(model, weight_name).buffer,
            tensor(model, bias_name).buffer, image.channels,
            image.width * image.height, epsilon);
    }

    GpuImage copy_image(const GpuImage& input) {
        GpuImage output{
            context_.create_device_buffer(elements(input) * sizeof(float)),
            input.channels, input.height, input.width};
        context_.copy(
            output.buffer, 0, input.buffer, 0,
            elements(input) * sizeof(float));
        return output;
    }

    GpuImage nearest(
        GpuImage&& input, std::uint32_t height, std::uint32_t width) {
        GpuImage output{
            context_.create_device_buffer(
                std::uint64_t(input.channels) * height * width *
                sizeof(float)),
            input.channels, height, width};
        operators_.nearest(
            output.buffer, input.buffer, input.width, input.height,
            width, height, input.channels);
        return output;
    }

    GpuImage concatenate(GpuImage&& left, GpuImage&& right) {
        if (left.width != right.width || left.height != right.height) {
            throw std::runtime_error("Marigold GPU skip shape mismatch");
        }
        GpuImage output{
            context_.create_device_buffer(
                (elements(left) + elements(right)) * sizeof(float)),
            left.channels + right.channels, left.height, left.width};
        operators_.concatenate(
            output.buffer, left.buffer, right.buffer,
            static_cast<std::uint32_t>(elements(left)),
            static_cast<std::uint32_t>(elements(right)));
        return output;
    }

    GpuImage vae_resnet(
        GpuImage&& input, const std::string& prefix) {
        GpuImage result;
        context_.batch([&] {
        GpuImage hidden = copy_image(input);
        group_norm(
            vae_, hidden, prefix + ".norm1.weight",
            prefix + ".norm1.bias");
        operators_.silu(
            hidden.buffer, static_cast<std::uint32_t>(elements(hidden)));
        hidden = conv(
            vae_, std::move(hidden), prefix + ".conv1.weight",
            prefix + ".conv1.bias");
        group_norm(
            vae_, hidden, prefix + ".norm2.weight",
            prefix + ".norm2.bias");
        operators_.silu(
            hidden.buffer, static_cast<std::uint32_t>(elements(hidden)));
        hidden = conv(
            vae_, std::move(hidden), prefix + ".conv2.weight",
            prefix + ".conv2.bias");
        GpuImage residual = tensor_exists(
            vae_, prefix + ".conv_shortcut.weight")
            ? conv(
                vae_, copy_image(input), prefix + ".conv_shortcut.weight",
                prefix + ".conv_shortcut.bias", 1, 0, 0)
            : copy_image(input);
        operators_.add(
            hidden.buffer, hidden.buffer, residual.buffer,
            static_cast<std::uint32_t>(elements(hidden)));
        result = std::move(hidden);
        });
        return result;
    }

    bool tensor_exists(const GpuModel& model, const std::string& name) {
        try {
            (void)model.tensor(name);
            return true;
        } catch (const std::exception&) {
            return false;
        }
    }

    GpuTokens image_to_tokens(const GpuImage& image) {
        GpuTokens output{
            context_.create_device_buffer(elements(image) * sizeof(float)),
            image.width * image.height, image.channels};
        operators_.nchw_tokens(
            output.buffer, image.buffer, output.tokens,
            output.dimensions, false);
        return output;
    }

    GpuImage tokens_to_image(
        GpuTokens&& tokens, std::uint32_t height, std::uint32_t width) {
        GpuImage output{
            context_.create_device_buffer(
                std::uint64_t(tokens.tokens) * tokens.dimensions *
                sizeof(float)),
            tokens.dimensions, height, width};
        operators_.nchw_tokens(
            output.buffer, tokens.buffer, tokens.tokens,
            tokens.dimensions, true);
        return output;
    }

    GpuTokens layer_norm(
        GpuModel& model, const GpuTokens& input,
        const std::string& prefix) {
        GpuTokens output{
            context_.create_device_buffer(
                std::uint64_t(input.tokens) * input.dimensions *
                sizeof(float)),
            input.tokens, input.dimensions};
        operators_.layer_norm(
            output.buffer, input.buffer,
            tensor(model, prefix + ".weight").buffer,
            tensor(model, prefix + ".bias").buffer,
            input.tokens, input.dimensions, 1.0e-5f);
        return output;
    }

    GpuTokens attention(
        GpuModel& model, const GpuTokens& query_input,
        const GpuTokens& key_value_input, const std::string& prefix,
        std::uint32_t heads) {
        auto clone_tokens = [&](const GpuTokens& source) {
            GpuTokens result{
                context_.create_device_buffer(
                    std::uint64_t(source.tokens) * source.dimensions *
                    sizeof(float)),
                source.tokens, source.dimensions};
            context_.copy(
                result.buffer, 0, source.buffer, 0,
                std::uint64_t(source.tokens) * source.dimensions *
                sizeof(float));
            return result;
        };
        GpuTokens q = linear(
            model, clone_tokens(query_input), prefix + ".to_q.weight",
            tensor_exists(model, prefix + ".to_q.bias")
                ? prefix + ".to_q.bias" : std::string{});
        GpuTokens k = linear(
            model, clone_tokens(key_value_input), prefix + ".to_k.weight",
            tensor_exists(model, prefix + ".to_k.bias")
                ? prefix + ".to_k.bias" : std::string{});
        GpuTokens v = linear(
            model, clone_tokens(key_value_input), prefix + ".to_v.weight",
            tensor_exists(model, prefix + ".to_v.bias")
                ? prefix + ".to_v.bias" : std::string{});
        GpuTokens attended{
            context_.create_device_buffer(
                std::uint64_t(q.tokens) * q.dimensions * sizeof(float)),
            q.tokens, q.dimensions};
        operators_.attention_separate(
            attended.buffer, q.buffer, k.buffer, v.buffer,
            q.tokens, k.tokens, heads, q.dimensions / heads);
        return linear(
            model, std::move(attended), prefix + ".to_out.0.weight",
            prefix + ".to_out.0.bias");
    }

    void add_tokens(GpuTokens& destination, const GpuTokens& source) {
        operators_.add(
            destination.buffer, destination.buffer, source.buffer,
            destination.tokens * destination.dimensions);
    }

    GpuImage spatial_attention(
        GpuImage&& input, const std::string& prefix) {
        GpuImage result;
        context_.batch([&] {
        GpuImage normalized = copy_image(input);
        group_norm(
            vae_, normalized, prefix + ".group_norm.weight",
            prefix + ".group_norm.bias");
        GpuTokens tokens = image_to_tokens(normalized);
        GpuTokens attended = attention(
            vae_, tokens, tokens, prefix, 1);
        GpuImage output = tokens_to_image(
            std::move(attended), input.height, input.width);
        operators_.add(
            output.buffer, output.buffer, input.buffer,
            static_cast<std::uint32_t>(elements(input)));
        result = std::move(output);
        });
        return result;
    }

    GpuImage vae_mid(GpuImage&& hidden, const std::string& prefix) {
        hidden = vae_resnet(
            std::move(hidden), prefix + ".resnets.0");
        hidden = spatial_attention(
            std::move(hidden), prefix + ".attentions.0");
        return vae_resnet(
            std::move(hidden), prefix + ".resnets.1");
    }

    GpuImage vae_encode(GpuImage&& rgb) {
        GpuImage hidden = conv(
            vae_, std::move(rgb), "encoder.conv_in.weight",
            "encoder.conv_in.bias");
        for (std::uint32_t block = 0; block < 4; ++block) {
            for (std::uint32_t layer = 0; layer < 2; ++layer) {
                hidden = vae_resnet(
                    std::move(hidden),
                    "encoder.down_blocks." + std::to_string(block) +
                    ".resnets." + std::to_string(layer));
            }
            if (block != 3) {
                const std::string prefix =
                    "encoder.down_blocks." + std::to_string(block) +
                    ".downsamplers.0.conv";
                hidden = conv(
                    vae_, std::move(hidden), prefix + ".weight",
                    prefix + ".bias", 2, 0, 1);
            }
        }
        hidden = vae_mid(std::move(hidden), "encoder.mid_block");
        group_norm(
            vae_, hidden, "encoder.conv_norm_out.weight",
            "encoder.conv_norm_out.bias");
        operators_.silu(
            hidden.buffer, static_cast<std::uint32_t>(elements(hidden)));
        hidden = conv(
            vae_, std::move(hidden), "encoder.conv_out.weight",
            "encoder.conv_out.bias");
        return conv(
            vae_, std::move(hidden), "quant_conv.weight",
            "quant_conv.bias", 1, 0, 0);
    }

    GpuImage vae_decode(GpuImage&& latent) {
        GpuImage hidden = conv(
            vae_, std::move(latent), "post_quant_conv.weight",
            "post_quant_conv.bias", 1, 0, 0);
        hidden = conv(
            vae_, std::move(hidden), "decoder.conv_in.weight",
            "decoder.conv_in.bias");
        hidden = vae_mid(std::move(hidden), "decoder.mid_block");
        for (std::uint32_t block = 0; block < 4; ++block) {
            for (std::uint32_t layer = 0; layer < 3; ++layer) {
                hidden = vae_resnet(
                    std::move(hidden),
                    "decoder.up_blocks." + std::to_string(block) +
                    ".resnets." + std::to_string(layer));
            }
            if (block != 3) {
                hidden = nearest(
                    std::move(hidden), hidden.height * 2, hidden.width * 2);
                const std::string prefix =
                    "decoder.up_blocks." + std::to_string(block) +
                    ".upsamplers.0.conv";
                hidden = conv(
                    vae_, std::move(hidden), prefix + ".weight",
                    prefix + ".bias");
            }
        }
        group_norm(
            vae_, hidden, "decoder.conv_norm_out.weight",
            "decoder.conv_norm_out.bias");
        operators_.silu(
            hidden.buffer, static_cast<std::uint32_t>(elements(hidden)));
        return conv(
            vae_, std::move(hidden), "decoder.conv_out.weight",
            "decoder.conv_out.bias");
    }

    GpuTokens embedding_mlp(
        GpuTokens&& input, const std::string& prefix) {
        GpuTokens hidden = linear(
            unet_, std::move(input), prefix + ".linear_1.weight",
            prefix + ".linear_1.bias");
        operators_.silu(
            hidden.buffer, hidden.tokens * hidden.dimensions);
        return linear(
            unet_, std::move(hidden), prefix + ".linear_2.weight",
            prefix + ".linear_2.bias");
    }

    GpuTokens time_embedding() {
        std::vector<float> values(320);
        for (std::uint32_t i = 0; i < 160; ++i) {
            const float frequency = std::exp(
                -std::log(10000.0f) * static_cast<float>(i) / 160.0f);
            values[i] = std::cos(999.0f * frequency);
            values[160 + i] = std::sin(999.0f * frequency);
        }
        GpuTokens timestep{
            context_.create_device_buffer(values.size() * sizeof(float)),
            1, 320};
        context_.upload(
            timestep.buffer, values.data(), values.size() * sizeof(float));
        GpuTokens time = embedding_mlp(
            std::move(timestep), "time_embedding");
        return time;
    }

    GpuImage unet_resnet(
        GpuImage&& input, const GpuTokens& time,
        const std::string& prefix) {
        GpuImage result;
        context_.batch([&] {
        GpuImage hidden = copy_image(input);
        group_norm(
            unet_, hidden, prefix + ".norm1.weight",
            prefix + ".norm1.bias", 1.0e-5f);
        operators_.silu(
            hidden.buffer, static_cast<std::uint32_t>(elements(hidden)));
        hidden = conv(
            unet_, std::move(hidden), prefix + ".conv1.weight",
            prefix + ".conv1.bias");
        GpuTokens activated{
            context_.create_device_buffer(
                std::uint64_t(time.tokens) * time.dimensions * sizeof(float)),
            time.tokens, time.dimensions};
        context_.copy(
            activated.buffer, 0, time.buffer, 0,
            std::uint64_t(time.tokens) * time.dimensions * sizeof(float));
        operators_.silu(
            activated.buffer, activated.tokens * activated.dimensions);
        GpuTokens projected = linear(
            unet_, std::move(activated), prefix + ".time_emb_proj.weight",
            prefix + ".time_emb_proj.bias");
        operators_.add_channel(
            hidden.buffer, projected.buffer, hidden.channels,
            hidden.width * hidden.height);
        group_norm(
            unet_, hidden, prefix + ".norm2.weight",
            prefix + ".norm2.bias", 1.0e-5f);
        operators_.silu(
            hidden.buffer, static_cast<std::uint32_t>(elements(hidden)));
        hidden = conv(
            unet_, std::move(hidden), prefix + ".conv2.weight",
            prefix + ".conv2.bias");
        GpuImage residual = tensor_exists(
            unet_, prefix + ".conv_shortcut.weight")
            ? conv(
                unet_, copy_image(input), prefix + ".conv_shortcut.weight",
                prefix + ".conv_shortcut.bias", 1, 0, 0)
            : copy_image(input);
        operators_.add(
            hidden.buffer, hidden.buffer, residual.buffer,
            static_cast<std::uint32_t>(elements(hidden)));
        result = std::move(hidden);
        });
        return result;
    }

    GpuImage transformer(
        GpuImage&& input, const std::string& prefix,
        std::uint32_t heads) {
        GpuImage result;
        context_.batch([&] {
        GpuImage normalized = copy_image(input);
        group_norm(
            unet_, normalized, prefix + ".norm.weight",
            prefix + ".norm.bias");
        GpuTokens tokens = image_to_tokens(normalized);
        tokens = linear(
            unet_, std::move(tokens), prefix + ".proj_in.weight",
            prefix + ".proj_in.bias");
        const std::string block = prefix + ".transformer_blocks.0";
        GpuTokens residual{
            context_.create_device_buffer(
                std::uint64_t(tokens.tokens) * tokens.dimensions *
                sizeof(float)),
            tokens.tokens, tokens.dimensions};
        auto save_residual = [&] {
            context_.copy(
                residual.buffer, 0, tokens.buffer, 0,
                std::uint64_t(tokens.tokens) * tokens.dimensions *
                sizeof(float));
        };
        save_residual();
        GpuTokens norm = layer_norm(unet_, tokens, block + ".norm1");
        GpuTokens update = attention(
            unet_, norm, norm, block + ".attn1", heads);
        add_tokens(update, residual);
        tokens = std::move(update);
        save_residual();
        norm = layer_norm(unet_, tokens, block + ".norm2");
        update = attention(
            unet_, norm, prompt_, block + ".attn2", heads);
        add_tokens(update, residual);
        tokens = std::move(update);
        save_residual();
        norm = layer_norm(unet_, tokens, block + ".norm3");
        GpuTokens projected = linear(
            unet_, std::move(norm), block + ".ff.net.0.proj.weight",
            block + ".ff.net.0.proj.bias");
        GpuTokens gated{
            context_.create_device_buffer(
                std::uint64_t(projected.tokens) *
                (projected.dimensions / 2) * sizeof(float)),
            projected.tokens, projected.dimensions / 2};
        operators_.geglu(
            gated.buffer, projected.buffer, gated.tokens, gated.dimensions);
        update = linear(
            unet_, std::move(gated), block + ".ff.net.2.weight",
            block + ".ff.net.2.bias");
        add_tokens(update, residual);
        tokens = linear(
            unet_, std::move(update), prefix + ".proj_out.weight",
            prefix + ".proj_out.bias");
        GpuImage output = tokens_to_image(
            std::move(tokens), input.height, input.width);
        operators_.add(
            output.buffer, output.buffer, input.buffer,
            static_cast<std::uint32_t>(elements(input)));
        result = std::move(output);
        });
        return result;
    }

    GpuImage unet_predict(GpuImage&& sample) {
        GpuTokens time = time_embedding();
        GpuImage hidden = conv(
            unet_, std::move(sample), "conv_in.weight", "conv_in.bias");
        std::vector<GpuImage> skips;
        skips.push_back(copy_image(hidden));
        const std::uint32_t down_heads[3] = {5, 10, 20};
        for (std::uint32_t block = 0; block < 4; ++block) {
            for (std::uint32_t layer = 0; layer < 2; ++layer) {
                const std::string root =
                    "down_blocks." + std::to_string(block);
                hidden = unet_resnet(
                    std::move(hidden), time,
                    root + ".resnets." + std::to_string(layer));
                if (block < 3) {
                    hidden = transformer(
                        std::move(hidden),
                        root + ".attentions." + std::to_string(layer),
                        down_heads[block]);
                }
                skips.push_back(copy_image(hidden));
            }
            if (block != 3) {
                const std::string prefix =
                    "down_blocks." + std::to_string(block) +
                    ".downsamplers.0.conv";
                hidden = conv(
                    unet_, std::move(hidden), prefix + ".weight",
                    prefix + ".bias", 2, 1, 1);
                skips.push_back(copy_image(hidden));
            }
        }
        hidden = unet_resnet(
            std::move(hidden), time, "mid_block.resnets.0");
        hidden = transformer(
            std::move(hidden), "mid_block.attentions.0", 20);
        hidden = unet_resnet(
            std::move(hidden), time, "mid_block.resnets.1");
        const std::uint32_t up_heads[4] = {0, 20, 10, 5};
        for (std::uint32_t block = 0; block < 4; ++block) {
            for (std::uint32_t layer = 0; layer < 3; ++layer) {
                GpuImage skip = std::move(skips.back());
                skips.pop_back();
                hidden = concatenate(std::move(hidden), std::move(skip));
                const std::string root =
                    "up_blocks." + std::to_string(block);
                hidden = unet_resnet(
                    std::move(hidden), time,
                    root + ".resnets." + std::to_string(layer));
                if (block != 0) {
                    hidden = transformer(
                        std::move(hidden),
                        root + ".attentions." + std::to_string(layer),
                        up_heads[block]);
                }
            }
            if (block != 3) {
                hidden = nearest(
                    std::move(hidden),
                    skips.back().height, skips.back().width);
                const std::string prefix =
                    "up_blocks." + std::to_string(block) +
                    ".upsamplers.0.conv";
                hidden = conv(
                    unet_, std::move(hidden), prefix + ".weight",
                    prefix + ".bias");
            }
        }
        group_norm(
            unet_, hidden, "conv_norm_out.weight",
            "conv_norm_out.bias", 1.0e-5f);
        operators_.silu(
            hidden.buffer, static_cast<std::uint32_t>(elements(hidden)));
        return conv(
            unet_, std::move(hidden), "conv_out.weight", "conv_out.bias");
    }

    VulkanContext& context_;
    GpuModel& unet_;
    GpuModel& vae_;
    VulkanOperators& operators_;
    VulkanBuffer zero_bias_;
    GpuTokens prompt_;
};
}

VulkanBuffer marigold_infer_gpu(
    VulkanContext& context, GpuModel& unet, GpuModel& vae,
    VulkanOperators& operators, const TokenTensor& prompt,
    const float* rgb, std::uint32_t width, std::uint32_t height,
    const float* target_noise) {
    Graph graph(context, unet, vae, operators, prompt);
    GpuImage decoded = graph.run(
        rgb, width, height, target_noise);
    VulkanBuffer depth = context.create_device_buffer(
        std::uint64_t(width) * height * sizeof(float));
    operators.depth_output(
        depth, decoded.buffer, decoded.width, decoded.height,
        width, height);
    return depth;
}

GpuImage marigold_vae_encode_gpu(
    VulkanContext& context, GpuModel& vae, VulkanOperators& operators,
    const float* input, std::uint32_t width, std::uint32_t height) {
    TokenTensor empty;
    Graph graph(context, vae, vae, operators, empty);
    GpuImage image{
        context.create_device_buffer(
            std::uint64_t(3) * width * height * sizeof(float)),
        3, height, width};
    context.upload(
        image.buffer, input,
        std::uint64_t(3) * width * height * sizeof(float));
    return graph.test_encode(std::move(image));
}

GpuImage marigold_unet_gpu(
    VulkanContext& context, GpuModel& unet, VulkanOperators& operators,
    const TokenTensor& prompt, const float* input,
    std::uint32_t width, std::uint32_t height) {
    Graph graph(context, unet, unet, operators, prompt);
    GpuImage sample{
        context.create_device_buffer(
            std::uint64_t(8) * width * height * sizeof(float)),
        8, height, width};
    context.upload(
        sample.buffer, input,
        std::uint64_t(8) * width * height * sizeof(float));
    return graph.test_predict(std::move(sample));
}

GpuImage marigold_vae_decode_gpu(
    VulkanContext& context, GpuModel& vae, VulkanOperators& operators,
    const float* input, std::uint32_t width, std::uint32_t height) {
    TokenTensor empty;
    Graph graph(context, vae, vae, operators, empty);
    GpuImage latent{
        context.create_device_buffer(
            std::uint64_t(4) * width * height * sizeof(float)),
        4, height, width};
    context.upload(
        latent.buffer, input,
        std::uint64_t(4) * width * height * sizeof(float));
    return graph.test_decode(std::move(latent));
}

GpuImage marigold_spatial_attention_gpu(
    VulkanContext& context, GpuModel& vae, VulkanOperators& operators,
    const float* input, std::uint32_t channels,
    std::uint32_t width, std::uint32_t height,
    const std::string& prefix) {
    TokenTensor empty;
    Graph graph(context, vae, vae, operators, empty);
    GpuImage image{
        context.create_device_buffer(
            std::uint64_t(channels) * width * height * sizeof(float)),
        channels, height, width};
    context.upload(
        image.buffer, input,
        std::uint64_t(channels) * width * height * sizeof(float));
    return graph.test_spatial_attention(std::move(image), prefix);
}

}  // namespace marigold_native
