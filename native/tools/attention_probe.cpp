#include "operators.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {
void environment(const char* name, const char* value) {
#if defined(_WIN32)
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}
void save(const std::string& path, const std::vector<float>& data) {
    std::ofstream stream(path, std::ios::binary);
    stream.write(reinterpret_cast<const char*>(data.data()),
        static_cast<std::streamsize>(data.size() * sizeof(float)));
    if (!stream) throw std::runtime_error("cannot write " + path);
}
}

int main(int argc, char** argv) {
    if (argc != 10 && argc != 11) {
        std::cerr << "usage: marigold_attention_probe device tiled(0|1) queries keys heads head_dimensions warmups iterations output-prefix [profile(0|1)]\n";
        return 2;
    }
    try {
        const auto number = [&](int index) {
            const auto value = std::stoull(argv[index]);
            if (value > std::numeric_limits<std::uint32_t>::max())
                throw std::invalid_argument("integer argument is too large");
            return static_cast<std::uint32_t>(value);
        };
        const auto device = number(1), mode = number(2);
        const auto queries = number(3), keys = number(4), heads = number(5);
        const auto channels = number(6), warmups = number(7), iterations = number(8);
        if (mode > 1 || !queries || !keys || !heads || !channels || !iterations)
            throw std::invalid_argument("invalid attention probe arguments");
        environment("MARIGOLD_ATTENTION_TILING", mode ? "1" : "0");
        environment("MARIGOLD_ATTENTION_TRACE", "1");
        environment("MARIGOLD_VULKAN_PROFILE", argc == 11 ? argv[10] : "0");
        // The operator being tested uses FP32 activations in every model mode.
        environment("INFERBRIDGE_PRECISION", "fp32");
        const std::size_t dimensions = std::size_t(heads) * channels;
        const std::size_t score_count = std::size_t(heads) * queries * keys;
        if (dimensions * std::max(queries, keys) > 100000000u || score_count > 100000000u)
            throw std::invalid_argument("fixture exceeds the probe's memory bound");
        std::vector<float> query(queries * dimensions), key(keys * dimensions), value(key.size());
        for (std::size_t i = 0; i < query.size(); ++i)
            query[i] = std::sin(static_cast<float>(i % 997) * 0.037f) * 0.7f;
        for (std::size_t i = 0; i < key.size(); ++i) {
            key[i] = std::cos(static_cast<float>(i % 991) * 0.031f) * 0.6f;
            value[i] = std::sin(static_cast<float>(i % 983) * 0.023f);
        }
        marigold_native::VulkanContext context(device);
        marigold_native::VulkanOperators operators(context);
        const auto upload = [&](const std::vector<float>& data) {
            auto buffer = context.create_device_buffer(data.size() * sizeof(float));
            context.upload(buffer, data.data(), data.size() * sizeof(float));
            return buffer;
        };
        auto q = upload(query), k = upload(key), v = upload(value);
        auto output = context.create_device_buffer(query.size() * sizeof(float));
        auto probabilities = context.create_device_buffer(score_count * sizeof(float));
        std::vector<double> samples;
        for (std::uint32_t iteration = 0; iteration < warmups + iterations; ++iteration) {
            const auto begin = std::chrono::steady_clock::now();
            // Preserve normal operator batching for timing. The existing
            // profile option deliberately unbatches these three dispatches,
            // so profiled wall times are diagnostic only.
            context.batch([&] {
                operators.attention_separate(output, q, k, v, queries, keys,
                    heads, channels, &probabilities);
            });
            if (iteration >= warmups) samples.push_back(
                std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - begin).count());
        }
        std::vector<float> actual(query.size()), scores(score_count);
        context.download(output, actual.data(), actual.size() * sizeof(float));
        context.download(probabilities, scores.data(), scores.size() * sizeof(float));
        double maximum_output = 0, maximum_probability = 0;
        // Small/tail fixtures get an independent double-precision reference;
        // large production shapes are compared against the saved scalar run.
        const bool reference = score_count * std::uint64_t(channels) <= 20000000u;
        if (reference) {
            std::vector<double> row(keys);
            for (std::uint32_t head = 0; head < heads; ++head) {
                for (std::uint32_t token = 0; token < queries; ++token) {
                    for (std::uint32_t j = 0; j < keys; ++j) {
                        double sum = 0;
                        for (std::uint32_t c = 0; c < channels; ++c)
                            sum += double(query[token * dimensions + head * channels + c]) *
                                key[j * dimensions + head * channels + c];
                        row[j] = sum / std::sqrt(double(channels));
                    }
                    const double maximum = *std::max_element(row.begin(), row.end());
                    double denominator = 0;
                    for (double& score : row) { score = std::exp(score - maximum); denominator += score; }
                    for (std::uint32_t j = 0; j < keys; ++j) {
                        row[j] /= denominator;
                        maximum_probability = std::max(maximum_probability,
                            std::abs(row[j] - scores[(std::size_t(head) * queries + token) * keys + j]));
                    }
                    for (std::uint32_t c = 0; c < channels; ++c) {
                        double sum = 0;
                        for (std::uint32_t j = 0; j < keys; ++j)
                            sum += row[j] * value[j * dimensions + head * channels + c];
                        maximum_output = std::max(maximum_output,
                            std::abs(sum - actual[token * dimensions + head * channels + c]));
                    }
                }
            }
        }
        for (float item : actual) if (!std::isfinite(item)) throw std::runtime_error("nonfinite output");
        for (float item : scores) if (!std::isfinite(item)) throw std::runtime_error("nonfinite probability");
        const std::filesystem::path prefix(argv[9]);
        if (!prefix.parent_path().empty()) std::filesystem::create_directories(prefix.parent_path());
        save(prefix.string() + "-attention.bin", actual);
        save(prefix.string() + "-probabilities.bin", scores);
        std::cout << "device=" << context.device_name()
                  << "\nselected=" << (operators.tiled_attention_selected(queries, keys, heads) ? "tiled" : "scalar")
                  << "\nqueries=" << queries << "\nkeys=" << keys << "\nheads=" << heads
                  << "\nhead_dimensions=" << channels << "\ncpu_reference=" << reference
                  << "\nmaximum_probability_error=" << maximum_probability
                  << "\nmaximum_output_error=" << maximum_output << "\nsamples_ms=";
        for (double sample : samples) std::cout << sample << ',';
        std::sort(samples.begin(), samples.end());
        std::cout << "\nmedian_ms=" << samples[samples.size() / 2] << '\n';
        return reference && (maximum_probability > 0.00002 || maximum_output > 0.00002) ? 3 : 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
