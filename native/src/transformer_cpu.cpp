#include "transformer_cpu.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <thread>

namespace marigold_native {
namespace {

template <typename Function>
void parallel_for(std::uint32_t tasks, Function function) {
    const std::uint32_t workers = std::min(
        tasks,
        std::max(1u, std::min(16u, std::thread::hardware_concurrency())));
    std::vector<std::thread> threads;
    threads.reserve(workers);
    for (std::uint32_t worker = 0; worker < workers; ++worker) {
        const std::uint32_t begin =
            static_cast<std::uint32_t>(
                std::uint64_t(tasks) * worker / workers);
        const std::uint32_t end =
            static_cast<std::uint32_t>(
                std::uint64_t(tasks) * (worker + 1) / workers);
        threads.emplace_back(function, begin, end);
    }
    for (std::thread& thread : threads) {
        thread.join();
    }
}

void add(TokenTensor& destination, const TokenTensor& source) {
    if (destination.tokens != source.tokens ||
        destination.dimensions != source.dimensions) {
        throw std::runtime_error("Marigold token residual shape mismatch");
    }
    parallel_for(
        static_cast<std::uint32_t>(destination.values.size()),
        [&](std::uint32_t begin, std::uint32_t end) {
            for (std::uint32_t i = begin; i < end; ++i) {
                destination.values[i] += source.values[i];
            }
        });
}

TokenTensor layer_norm(
    const SafeTensors& model,
    const TokenTensor& input,
    const std::string& prefix) {
    const TensorView& weight = model.tensor(prefix + ".weight");
    const TensorView& bias = model.tensor(prefix + ".bias");
    if (weight.rank != 1 ||
        weight.dimensions[0] != input.dimensions ||
        bias.rank != 1 ||
        bias.dimensions[0] != input.dimensions) {
        throw std::runtime_error("Marigold layer norm shape mismatch");
    }
    TokenTensor output = input;
    parallel_for(input.tokens, [&](std::uint32_t begin, std::uint32_t end) {
        for (std::uint32_t token = begin; token < end; ++token) {
            const float* source =
                input.values.data() + std::uint64_t(token) * input.dimensions;
            float* destination =
                output.values.data() + std::uint64_t(token) * input.dimensions;
            double sum = 0.0;
            double square_sum = 0.0;
            for (std::uint32_t d = 0; d < input.dimensions; ++d) {
                const double value = source[d];
                sum += value;
                square_sum += value * value;
            }
            const double mean = sum / input.dimensions;
            const float inverse_std = static_cast<float>(
                1.0 / std::sqrt(
                    square_sum / input.dimensions - mean * mean + 1.0e-5));
            for (std::uint32_t d = 0; d < input.dimensions; ++d) {
                destination[d] =
                    (source[d] - static_cast<float>(mean)) *
                        inverse_std * weight.data[d] +
                    bias.data[d];
            }
        }
    });
    return output;
}

TokenTensor attention(
    const SafeTensors& model,
    const TokenTensor& query_input,
    const TokenTensor& key_value_input,
    const std::string& prefix,
    std::uint32_t heads) {
    TokenTensor q = linear(
        model, query_input, prefix + ".to_q.weight");
    TokenTensor k = linear(
        model, key_value_input, prefix + ".to_k.weight");
    TokenTensor v = linear(
        model, key_value_input, prefix + ".to_v.weight");
    if (q.dimensions != k.dimensions ||
        q.dimensions != v.dimensions ||
        q.dimensions % heads != 0) {
        throw std::runtime_error("Marigold attention head mismatch");
    }
    const std::uint32_t head_dimensions = q.dimensions / heads;
    TokenTensor attended{
        q.tokens, q.dimensions,
        std::vector<float>(
            std::uint64_t(q.tokens) * q.dimensions)};
    const float scale =
        1.0f / std::sqrt(static_cast<float>(head_dimensions));
    parallel_for(q.tokens * heads, [&](std::uint32_t begin, std::uint32_t end) {
        std::vector<float> scores(k.tokens);
        for (std::uint32_t task = begin; task < end; ++task) {
            const std::uint32_t query = task / heads;
            const std::uint32_t head = task % heads;
            const std::uint32_t offset = head * head_dimensions;
            float maximum = -INFINITY;
            for (std::uint32_t key = 0; key < k.tokens; ++key) {
                float score = 0.0f;
                for (std::uint32_t d = 0; d < head_dimensions; ++d) {
                    score += q.values[
                                 std::uint64_t(query) * q.dimensions +
                                 offset + d] *
                        k.values[
                            std::uint64_t(key) * k.dimensions + offset + d];
                }
                score *= scale;
                scores[key] = score;
                maximum = std::max(maximum, score);
            }
            float denominator = 0.0f;
            for (float& score : scores) {
                score = std::exp(score - maximum);
                denominator += score;
            }
            for (std::uint32_t d = 0; d < head_dimensions; ++d) {
                float value = 0.0f;
                for (std::uint32_t key = 0; key < k.tokens; ++key) {
                    value += scores[key] / denominator *
                        v.values[
                            std::uint64_t(key) * v.dimensions +
                            offset + d];
                }
                attended.values[
                    std::uint64_t(query) * attended.dimensions +
                    offset + d] = value;
            }
        }
    });
    return linear(
        model, attended, prefix + ".to_out.0.weight",
        prefix + ".to_out.0.bias");
}

TokenTensor feed_forward(
    const SafeTensors& model,
    const TokenTensor& input,
    const std::string& prefix) {
    TokenTensor projected = linear(
        model, input, prefix + ".net.0.proj.weight",
        prefix + ".net.0.proj.bias");
    if (projected.dimensions % 2 != 0) {
        throw std::runtime_error("Marigold GEGLU dimension mismatch");
    }
    const std::uint32_t dimensions = projected.dimensions / 2;
    TokenTensor gated{
        projected.tokens, dimensions,
        std::vector<float>(
            std::uint64_t(projected.tokens) * dimensions)};
    parallel_for(
        projected.tokens,
        [&](std::uint32_t begin, std::uint32_t end) {
            constexpr float inverse_sqrt_two =
                0.70710678118654752440f;
            for (std::uint32_t token = begin; token < end; ++token) {
                const float* source = projected.values.data() +
                    std::uint64_t(token) * projected.dimensions;
                float* destination = gated.values.data() +
                    std::uint64_t(token) * dimensions;
                for (std::uint32_t d = 0; d < dimensions; ++d) {
                    const float gate = source[dimensions + d];
                    const float gelu =
                        0.5f * gate *
                        (1.0f + std::erf(gate * inverse_sqrt_two));
                    destination[d] = source[d] * gelu;
                }
            }
        });
    return linear(
        model, gated, prefix + ".net.2.weight",
        prefix + ".net.2.bias");
}

}  // namespace

TokenTensor linear(
    const SafeTensors& model,
    const TokenTensor& input,
    const std::string& weight_name,
    const std::string& bias_name) {
    const TensorView& weight = model.tensor(weight_name);
    if (weight.rank != 2 ||
        weight.dimensions[1] != input.dimensions) {
        throw std::runtime_error(
            "Marigold linear shape mismatch: " + weight_name);
    }
    const std::uint32_t output_dimensions =
        static_cast<std::uint32_t>(weight.dimensions[0]);
    TensorData bias{};
    bool have_bias = false;
    if (!bias_name.empty()) {
        const TensorView& bias_view = model.tensor(bias_name);
        if (bias_view.rank != 1 ||
            bias_view.dimensions[0] != output_dimensions) {
            throw std::runtime_error(
                "Marigold linear bias mismatch: " + bias_name);
        }
        bias = bias_view.data;
        have_bias = true;
    }
    TokenTensor output{
        input.tokens, output_dimensions,
        std::vector<float>(
            std::uint64_t(input.tokens) * output_dimensions)};
    parallel_for(input.tokens, [&](std::uint32_t begin, std::uint32_t end) {
        for (std::uint32_t token = begin; token < end; ++token) {
            const float* source =
                input.values.data() + std::uint64_t(token) * input.dimensions;
            float* destination = output.values.data() +
                std::uint64_t(token) * output_dimensions;
            for (std::uint32_t out = 0; out < output_dimensions; ++out) {
                float sum = have_bias ? bias[out] : 0.0f;
                for (std::uint32_t in = 0; in < input.dimensions; ++in) {
                    sum += source[in] * weight.data[
                        std::uint64_t(out) * input.dimensions + in];
                }
                destination[out] = sum;
            }
        }
    });
    return output;
}

TokenTensor transformer2d(
    const SafeTensors& model,
    const ImageTensor& input,
    const TokenTensor& context,
    const std::string& prefix,
    std::uint32_t heads,
    ImageTensor& output) {
    ImageTensor normalized = input;
    group_norm(
        model, normalized, prefix + ".norm.weight",
        prefix + ".norm.bias", 1.0e-6f);
    const std::uint32_t token_count = input.height * input.width;
    TokenTensor tokens{
        token_count, input.channels,
        std::vector<float>(
            std::uint64_t(token_count) * input.channels)};
    for (std::uint32_t token = 0; token < token_count; ++token) {
        for (std::uint32_t channel = 0;
             channel < input.channels; ++channel) {
            tokens.values[
                std::uint64_t(token) * input.channels + channel] =
                normalized.values[
                    std::uint64_t(channel) * token_count + token];
        }
    }
    tokens = linear(
        model, tokens, prefix + ".proj_in.weight",
        prefix + ".proj_in.bias");
    const std::string block = prefix + ".transformer_blocks.0";
    TokenTensor residual = tokens;
    TokenTensor normalized_tokens =
        layer_norm(model, tokens, block + ".norm1");
    TokenTensor update = attention(
        model, normalized_tokens, normalized_tokens,
        block + ".attn1", heads);
    add(update, residual);
    tokens = std::move(update);

    residual = tokens;
    normalized_tokens = layer_norm(model, tokens, block + ".norm2");
    update = attention(
        model, normalized_tokens, context,
        block + ".attn2", heads);
    add(update, residual);
    tokens = std::move(update);

    residual = tokens;
    normalized_tokens = layer_norm(model, tokens, block + ".norm3");
    update = feed_forward(model, normalized_tokens, block + ".ff");
    add(update, residual);
    tokens = linear(
        model, update, prefix + ".proj_out.weight",
        prefix + ".proj_out.bias");

    output = input;
    for (std::uint32_t token = 0; token < token_count; ++token) {
        for (std::uint32_t channel = 0;
             channel < input.channels; ++channel) {
            output.values[
                std::uint64_t(channel) * token_count + token] +=
                tokens.values[
                    std::uint64_t(token) * input.channels + channel];
        }
    }
    return tokens;
}

}  // namespace marigold_native
