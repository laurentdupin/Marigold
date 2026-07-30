#include "model_bundle.h"
#include "vae_cpu.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>
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

void compare(
    const char* label,
    const std::vector<float>& actual,
    const std::vector<float>& reference) {
    double error = 0.0;
    double magnitude = 0.0;
    float maximum = 0.0f;
    for (std::size_t i = 0; i < reference.size(); ++i) {
        const float difference = std::abs(actual[i] - reference[i]);
        error += difference;
        magnitude += std::abs(reference[i]);
        maximum = std::max(maximum, difference);
    }
    const double relative = error / magnitude;
    std::cout << label << "_relative_l1=" << relative
              << "\n" << label << "_maximum_absolute=" << maximum << "\n";
    if (relative > 0.01) {
        throw std::runtime_error(
            std::string(label) + " exceeded the one-percent gate");
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr
            << "usage: marigold_vae_probe "
               "snapshot-root derived-vae fixture-dir\n";
        return 2;
    }
    try {
        const marigold_native::ModelBundle bundle(argv[1], argv[2]);
        const std::string fixture = argv[3];
        marigold_native::ImageTensor rgb{
            3, 64, 64, load(fixture + "/rgb.bin", 3 * 64 * 64)};
        marigold_native::Posterior posterior =
            marigold_native::vae_encode(bundle.vae(), rgb);
        for (float& value : posterior.mean.values) {
            value *= 0.18215f;
        }
        compare(
            "rgb_latent", posterior.mean.values,
            load(fixture + "/rgb_latent.bin", 4 * 8 * 8));

        marigold_native::ImageTensor latent{
            4, 8, 8,
            load(fixture + "/target_latent.bin", 4 * 8 * 8)};
        for (float& value : latent.values) {
            value /= 0.18215f;
        }
        const marigold_native::ImageTensor decoded =
            marigold_native::vae_decode(bundle.vae(), latent);
        compare(
            "decoded", decoded.values,
            load(fixture + "/decoded.bin", 3 * 64 * 64));
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << "\n";
        return 1;
    }
}
