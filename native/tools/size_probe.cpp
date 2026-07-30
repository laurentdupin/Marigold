#include "marigold_native.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 6) {
        std::cerr
            << "usage: marigold_size_probe "
               "snapshot derived-vae prompt-cache width height\n";
        return 2;
    }
    const std::uint32_t width =
        static_cast<std::uint32_t>(std::stoul(argv[4]));
    const std::uint32_t height =
        static_cast<std::uint32_t>(std::stoul(argv[5]));
    std::vector<float> rgb(std::uint64_t(width) * height * 3);
    for (std::uint32_t y = 0; y < height; ++y) {
        for (std::uint32_t x = 0; x < width; ++x) {
            rgb[(std::uint64_t(y) * width + x) * 3] =
                std::sin(x * 0.071f) * 0.5f + 0.5f;
            rgb[(std::uint64_t(y) * width + x) * 3 + 1] =
                std::cos(y * 0.053f) * 0.5f + 0.5f;
            rgb[(std::uint64_t(y) * width + x) * 3 + 2] =
                std::sin((x + y) * 0.037f) * 0.5f + 0.5f;
        }
    }
    std::vector<float> depth(std::uint64_t(width) * height);
    marigold_context* context = nullptr;
    int status =
        marigold_create(argv[1], argv[2], argv[3], &context);
    if (status == MARIGOLD_OK) {
        status = marigold_infer_rgb_f32(
            context, rgb.data(), width, height, 23, depth.data());
    }
    if (status != MARIGOLD_OK) {
        std::cerr << marigold_last_error() << "\n";
        marigold_destroy(context);
        return 1;
    }
    double sum = 0.0;
    for (float value : depth) {
        if (!std::isfinite(value) || value < 0.0f || value > 1.0f) {
            marigold_destroy(context);
            return 3;
        }
        sum += value;
    }
    std::cout << "width=" << width << "\nheight=" << height
              << "\nmean=" << sum / depth.size() << "\n";
    marigold_destroy(context);
    return 0;
}
