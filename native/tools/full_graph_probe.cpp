#include "marigold_native.h"

#include <algorithm>
#include <cmath>
#include <chrono>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::vector<float> load(const std::string& path, std::size_t count) {
    std::ifstream stream(path, std::ios::binary);
    std::vector<float> values(count);
    stream.read(
        reinterpret_cast<char*>(values.data()),
        static_cast<std::streamsize>(count * sizeof(float)));
    if (!stream || stream.peek() != std::char_traits<char>::eof()) {
        throw std::runtime_error("invalid fixture: " + path);
    }
    return values;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 5 || argc > 7) {
        std::cerr
            << "usage: marigold_full_graph_probe "
               "snapshot derived-vae prompt-cache fixture-dir "
               "[vulkan-device [iterations]]\n";
        return 2;
    }
    marigold_context* context = nullptr;
    const std::string snapshot = argv[1];
    const marigold_model_variant variant =
        snapshot.find("marigold-v1-0") != std::string::npos &&
        snapshot.find("lcm") == std::string::npos ?
            MARIGOLD_MODEL_FULL_V1 : MARIGOLD_MODEL_LCM_V1;
    const int create_status = argc >= 6
        ? marigold_create_vulkan_variant(
            argv[1], argv[2], argv[3], variant,
            static_cast<std::uint32_t>(std::stoul(argv[5])), &context)
        : marigold_create_variant(
            argv[1], argv[2], argv[3], variant, &context);
    if (create_status != MARIGOLD_OK) {
        std::cerr << marigold_last_error() << "\n";
        return 1;
    }
    try {
        constexpr std::uint32_t size = 64;
        const std::string root = argv[4];
        const std::vector<float> nchw =
            load(root + "/rgb.bin", 3 * size * size);
        std::vector<float> rgb(3 * size * size);
        for (std::uint32_t y = 0; y < size; ++y) {
            for (std::uint32_t x = 0; x < size; ++x) {
                for (std::uint32_t c = 0; c < 3; ++c) {
                    rgb[(std::uint64_t(y) * size + x) * 3 + c] =
                        nchw[
                            (std::uint64_t(c) * size + y) * size + x] *
                            0.5f +
                        0.5f;
                }
            }
        }
        const std::vector<float> noise =
            load(root + "/target_noise.bin", 4 * 8 * 8);
        std::vector<float> depth(size * size);
        const std::uint32_t iterations =
            argc == 7 ? static_cast<std::uint32_t>(std::stoul(argv[6])) : 1;
        std::vector<double> samples;
        for (std::uint32_t iteration = 0; iteration < iterations; ++iteration) {
            const auto start = std::chrono::steady_clock::now();
            const int status = marigold_infer_rgb_f32_with_noise(
                context, rgb.data(), size, size, noise.data(), depth.data());
            samples.push_back(std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - start).count());
            if (status != MARIGOLD_OK) {
                throw std::runtime_error(marigold_last_error());
            }
        }
        std::sort(samples.begin(), samples.end());
        const std::vector<float> reference =
            load(root + "/depth.bin", size * size);
        double error = 0.0;
        double magnitude = 0.0;
        float maximum = 0.0f;
        for (std::size_t i = 0; i < reference.size(); ++i) {
            const float difference = std::abs(depth[i] - reference[i]);
            error += difference;
            magnitude += std::abs(reference[i]);
            maximum = std::max(maximum, difference);
        }
        const double relative = error / magnitude;
        std::cout << "relative_l1=" << relative
                  << "\nmaximum_absolute=" << maximum
                  << "\nmedian_ms=" << samples[samples.size() / 2] << "\n";
        marigold_destroy(context);
        return relative <= 0.01 ? 0 : 3;
    } catch (const std::exception& error) {
        std::cerr << error.what() << "\n";
        marigold_destroy(context);
        return 1;
    }
}
