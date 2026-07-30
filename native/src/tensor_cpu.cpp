#include "tensor_cpu.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <thread>

namespace marigold_native {
namespace {

std::uint64_t elements(const ImageTensor& value) {
    return std::uint64_t(value.channels) * value.height * value.width;
}

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

const TensorView& vector_tensor(
    const SafeTensors& model,
    const std::string& name,
    std::uint32_t count) {
    const TensorView& tensor = model.tensor(name);
    if (tensor.rank != 1 || tensor.dimensions[0] != count) {
        throw std::runtime_error("Marigold vector shape mismatch: " + name);
    }
    return tensor;
}

}  // namespace

ImageTensor conv2d(
    const SafeTensors& model,
    const ImageTensor& input,
    const std::string& weight_name,
    const std::string& bias_name,
    std::uint32_t stride,
    std::uint32_t pad_before,
    std::uint32_t pad_after) {
    const TensorView& weight = model.tensor(weight_name);
    if (weight.rank != 4 ||
        weight.dimensions[1] != input.channels ||
        weight.dimensions[2] != weight.dimensions[3]) {
        throw std::runtime_error(
            "Marigold convolution shape mismatch: " + weight_name);
    }
    const std::uint32_t output_channels =
        static_cast<std::uint32_t>(weight.dimensions[0]);
    const std::uint32_t kernel =
        static_cast<std::uint32_t>(weight.dimensions[2]);
    if (input.height + pad_before + pad_after < kernel ||
        input.width + pad_before + pad_after < kernel) {
        throw std::runtime_error("Marigold convolution input is too small");
    }
    ImageTensor output{
        output_channels,
        (input.height + pad_before + pad_after - kernel) / stride + 1,
        (input.width + pad_before + pad_after - kernel) / stride + 1,
        {}};
    output.values.resize(static_cast<std::size_t>(elements(output)));
    TensorData bias{};
    bool have_bias = false;
    if (!bias_name.empty()) {
        bias = vector_tensor(model, bias_name, output_channels).data;
        have_bias = true;
    }

    parallel_for(
        output_channels,
        [&](std::uint32_t begin, std::uint32_t end) {
            for (std::uint32_t oc = begin; oc < end; ++oc) {
                float* output_plane = output.values.data() +
                    std::uint64_t(oc) * output.height * output.width;
                std::fill_n(
                    output_plane,
                    std::uint64_t(output.height) * output.width,
                    have_bias ? bias[oc] : 0.0f);
                for (std::uint32_t ic = 0; ic < input.channels; ++ic) {
                    const float* input_plane = input.values.data() +
                        std::uint64_t(ic) * input.height * input.width;
                    for (std::uint32_t ky = 0; ky < kernel; ++ky) {
                        for (std::uint32_t kx = 0; kx < kernel; ++kx) {
                            const float coefficient = weight.data[
                                ((std::uint64_t(oc) * input.channels + ic) *
                                     kernel +
                                 ky) *
                                    kernel +
                                kx];
                            for (std::uint32_t oy = 0;
                                 oy < output.height; ++oy) {
                                const std::int64_t iy =
                                    std::int64_t(oy) * stride + ky -
                                    pad_before;
                                if (iy < 0 ||
                                    iy >= std::int64_t(input.height)) {
                                    continue;
                                }
                                float* destination = output_plane +
                                    std::uint64_t(oy) * output.width;
                                const float* source = input_plane +
                                    std::uint64_t(iy) * input.width;
                                for (std::uint32_t ox = 0;
                                     ox < output.width; ++ox) {
                                    const std::int64_t ix =
                                        std::int64_t(ox) * stride + kx -
                                        pad_before;
                                    if (ix >= 0 &&
                                        ix < std::int64_t(input.width)) {
                                        destination[ox] +=
                                            source[ix] * coefficient;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        });
    return output;
}

void group_norm(
    const SafeTensors& model,
    ImageTensor& tensor,
    const std::string& weight_name,
    const std::string& bias_name,
    float epsilon) {
    constexpr std::uint32_t groups = 32;
    if (tensor.channels % groups != 0) {
        throw std::runtime_error("Marigold group norm channel mismatch");
    }
    const TensorView& weight =
        vector_tensor(model, weight_name, tensor.channels);
    const TensorView& bias =
        vector_tensor(model, bias_name, tensor.channels);
    const std::uint32_t channels_per_group = tensor.channels / groups;
    const std::uint64_t plane =
        std::uint64_t(tensor.height) * tensor.width;
    const std::uint64_t group_elements = channels_per_group * plane;
    parallel_for(groups, [&](std::uint32_t begin, std::uint32_t end) {
        for (std::uint32_t group = begin; group < end; ++group) {
            const float* group_begin = tensor.values.data() +
                std::uint64_t(group) * group_elements;
            double sum = 0.0;
            double square_sum = 0.0;
            for (std::uint64_t i = 0; i < group_elements; ++i) {
                const double value = group_begin[i];
                sum += value;
                square_sum += value * value;
            }
            const double mean = sum / group_elements;
            const float inverse_std = static_cast<float>(
                1.0 / std::sqrt(
                    square_sum / group_elements - mean * mean + epsilon));
            for (std::uint32_t local = 0;
                 local < channels_per_group; ++local) {
                const std::uint32_t channel =
                    group * channels_per_group + local;
                float* values = tensor.values.data() +
                    std::uint64_t(channel) * plane;
                for (std::uint64_t i = 0; i < plane; ++i) {
                    values[i] =
                        (values[i] - static_cast<float>(mean)) *
                            inverse_std * weight.data[channel] +
                        bias.data[channel];
                }
            }
        }
    });
}

void silu(ImageTensor& tensor) {
    parallel_for(
        static_cast<std::uint32_t>(tensor.values.size()),
        [&](std::uint32_t begin, std::uint32_t end) {
            for (std::uint32_t i = begin; i < end; ++i) {
                const float value = tensor.values[i];
                tensor.values[i] = value / (1.0f + std::exp(-value));
            }
        });
}

void add_in_place(ImageTensor& destination, const ImageTensor& source) {
    if (destination.channels != source.channels ||
        destination.height != source.height ||
        destination.width != source.width) {
        throw std::runtime_error("Marigold residual shape mismatch");
    }
    parallel_for(
        static_cast<std::uint32_t>(destination.values.size()),
        [&](std::uint32_t begin, std::uint32_t end) {
            for (std::uint32_t i = begin; i < end; ++i) {
                destination.values[i] += source.values[i];
            }
        });
}

ImageTensor nearest_upsample_2x(const ImageTensor& input) {
    return nearest_upsample(
        input, input.height * 2, input.width * 2);
}

ImageTensor nearest_upsample(
    const ImageTensor& input,
    std::uint32_t output_height,
    std::uint32_t output_width) {
    ImageTensor output{
        input.channels, output_height, output_width, {}};
    output.values.resize(static_cast<std::size_t>(elements(output)));
    parallel_for(
        input.channels,
        [&](std::uint32_t begin, std::uint32_t end) {
            for (std::uint32_t channel = begin; channel < end; ++channel) {
                for (std::uint32_t y = 0; y < output.height; ++y) {
                    for (std::uint32_t x = 0; x < output.width; ++x) {
                        output.values[
                            (std::uint64_t(channel) * output.height + y) *
                                output.width +
                            x] =
                            input.values[
                                (std::uint64_t(channel) * input.height +
                                 std::uint64_t(y) * input.height /
                                     output.height) *
                                    input.width +
                                std::uint64_t(x) * input.width /
                                    output.width];
                    }
                }
            }
        });
    return output;
}

ImageTensor resnet(
    const SafeTensors& model,
    const ImageTensor& input,
    const std::string& prefix) {
    ImageTensor hidden = input;
    group_norm(
        model, hidden, prefix + ".norm1.weight",
        prefix + ".norm1.bias");
    silu(hidden);
    hidden = conv2d(
        model, hidden, prefix + ".conv1.weight",
        prefix + ".conv1.bias");
    group_norm(
        model, hidden, prefix + ".norm2.weight",
        prefix + ".norm2.bias");
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

ImageTensor spatial_attention(
    const SafeTensors& model,
    const ImageTensor& input,
    const std::string& prefix) {
    ImageTensor normalized = input;
    group_norm(
        model, normalized, prefix + ".group_norm.weight",
        prefix + ".group_norm.bias");
    const std::uint32_t channels = input.channels;
    const std::uint32_t tokens = input.height * input.width;
    std::vector<float> q(std::uint64_t(tokens) * channels);
    std::vector<float> k(q.size());
    std::vector<float> v(q.size());
    const auto project = [&](const char* suffix, std::vector<float>& out) {
        const std::string base = prefix + suffix;
        const TensorView& weight = model.tensor(base + ".weight");
        const TensorView& bias = model.tensor(base + ".bias");
        if (weight.rank != 2 ||
            weight.dimensions[0] != channels ||
            weight.dimensions[1] != channels) {
            throw std::runtime_error(
                "Marigold attention projection shape mismatch: " + base);
        }
        parallel_for(tokens, [&](std::uint32_t begin, std::uint32_t end) {
            for (std::uint32_t token = begin; token < end; ++token) {
                for (std::uint32_t oc = 0; oc < channels; ++oc) {
                    float sum = bias.data[oc];
                    for (std::uint32_t ic = 0; ic < channels; ++ic) {
                        sum += normalized.values[
                                   std::uint64_t(ic) * tokens + token] *
                            weight.data[
                                std::uint64_t(oc) * channels + ic];
                    }
                    out[std::uint64_t(token) * channels + oc] = sum;
                }
            }
        });
    };
    project(".to_q", q);
    project(".to_k", k);
    project(".to_v", v);

    std::vector<float> attended(q.size());
    const float scale = 1.0f / std::sqrt(static_cast<float>(channels));
    parallel_for(tokens, [&](std::uint32_t begin, std::uint32_t end) {
        std::vector<float> scores(tokens);
        for (std::uint32_t query = begin; query < end; ++query) {
            float maximum = -INFINITY;
            for (std::uint32_t key = 0; key < tokens; ++key) {
                float score = 0.0f;
                for (std::uint32_t c = 0; c < channels; ++c) {
                    score += q[std::uint64_t(query) * channels + c] *
                        k[std::uint64_t(key) * channels + c];
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
            for (std::uint32_t c = 0; c < channels; ++c) {
                float value = 0.0f;
                for (std::uint32_t key = 0; key < tokens; ++key) {
                    value += scores[key] / denominator *
                        v[std::uint64_t(key) * channels + c];
                }
                attended[std::uint64_t(query) * channels + c] = value;
            }
        }
    });

    const TensorView& output_weight =
        model.tensor(prefix + ".to_out.0.weight");
    const TensorView& output_bias =
        model.tensor(prefix + ".to_out.0.bias");
    ImageTensor output{
        channels, input.height, input.width,
        std::vector<float>(static_cast<std::size_t>(elements(input)))};
    parallel_for(tokens, [&](std::uint32_t begin, std::uint32_t end) {
        for (std::uint32_t token = begin; token < end; ++token) {
            for (std::uint32_t oc = 0; oc < channels; ++oc) {
                float sum = output_bias.data[oc];
                for (std::uint32_t ic = 0; ic < channels; ++ic) {
                    sum += attended[
                               std::uint64_t(token) * channels + ic] *
                        output_weight.data[
                            std::uint64_t(oc) * channels + ic];
                }
                output.values[std::uint64_t(oc) * tokens + token] = sum;
            }
        }
    });
    add_in_place(output, input);
    return output;
}

}  // namespace marigold_native
