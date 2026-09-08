#include "marigold_gpu.h"
#include "model_bundle.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
std::vector<float> load(const std::filesystem::path& path, std::size_t count) {
    std::ifstream stream(path, std::ios::binary);
    std::vector<float> data(count);
    stream.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(count * sizeof(float)));
    if (!stream || stream.peek() != std::char_traits<char>::eof())
        throw std::runtime_error("invalid fixture: " + path.string());
    return data;
}
void save(const std::filesystem::path& path, const std::vector<float>& data) {
    std::ofstream stream(path, std::ios::binary);
    stream.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size() * sizeof(float)));
    if (!stream) throw std::runtime_error("cannot write " + path.string());
}
void compare(const char* name, const std::vector<float>& actual, const std::vector<float>& reference) {
    if (actual.size() != reference.size()) throw std::runtime_error("fixture shape mismatch");
    double error = 0, magnitude = 0, maximum = 0;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        if (!std::isfinite(actual[i])) throw std::runtime_error("nonfinite fixture output");
        const double difference = std::abs(double(actual[i]) - reference[i]);
        error += difference; magnitude += std::abs(reference[i]);
        maximum = std::max(maximum, difference);
    }
    const double relative = error / std::max(magnitude, 1e-30);
    std::cout << name << "_relative_l1=" << relative << '\n'
              << name << "_maximum_absolute=" << maximum << '\n';
    if (relative > 0.01) throw std::runtime_error(std::string(name) + " exceeded existing one-percent gate");
}
}

int main(int argc, char** argv) {
    if (argc != 8 && argc != 10) {
        std::cerr << "usage: marigold_gpu_fixture_probe snapshot derived-vae fixture-dir device tiled(0|1) full-v1(0|1) output-dir [synthetic-width synthetic-height]\n";
        return 2;
    }
    try {
        const unsigned mode = std::stoul(argv[5]), full = std::stoul(argv[6]);
        if (mode > 1 || full > 1) throw std::invalid_argument("mode and variant must be 0 or 1");
#if defined(_WIN32)
        _putenv_s("MARIGOLD_ATTENTION_TILING", mode ? "1" : "0");
#else
        setenv("MARIGOLD_ATTENTION_TILING", mode ? "1" : "0", 1);
#endif
        const std::filesystem::path fixture(argv[3]), output(argv[7]);
        std::filesystem::create_directories(output);
        const bool synthetic = argc == 10;
        const std::uint32_t width = synthetic ? std::stoul(argv[8]) : 64u;
        const std::uint32_t height = synthetic ? std::stoul(argv[9]) : 64u;
        if (width < 8 || height < 8 || width > 2048 || height > 2048)
            throw std::invalid_argument("fixture dimensions out of range");
        const std::size_t pixels = std::size_t(width) * height;
        const std::size_t latent_count = std::size_t(4) * (width / 8) * (height / 8);
        std::vector<float> normalized(3 * pixels), noise(latent_count), rgb(3 * pixels);
        if (!synthetic) {
            normalized = load(fixture / "rgb.bin", normalized.size());
            noise = load(fixture / "target_noise.bin", noise.size());
        } else {
            for (std::size_t i = 0; i < normalized.size(); ++i)
                normalized[i] = std::sin(float(i % 991) * 0.013f);
            for (std::size_t i = 0; i < noise.size(); ++i)
                noise[i] = std::cos(float(i % 983) * 0.071f);
        }
        for (std::size_t i = 0; i < pixels; ++i)
            for (std::size_t c = 0; c < 3; ++c)
                rgb[i * 3 + c] = normalized[c * pixels + i] * 0.5f + 0.5f;
        marigold_native::TokenTensor prompt{2u, 1024u, load(fixture / "prompt.bin", 2048)};
        marigold_native::VulkanContext context(std::stoul(argv[4]));
        marigold_native::VulkanOperators operators(context);
        marigold_native::ModelBundle bundle(argv[1], argv[2]);
        marigold_native::GpuModel unet(bundle.unet(), context), vae(bundle.vae(), context);
        const auto download = [&](const marigold_native::VulkanBuffer& buffer, std::size_t count) {
            std::vector<float> values(count);
            context.download(buffer, values.data(), count * sizeof(float));
            return values;
        };
        if (!synthetic) {
            auto encoded = marigold_native::marigold_vae_encode_gpu(context, vae, operators, normalized.data(), width, height);
            // Encoder returns concatenated mean/logvar; only mean is used.
            auto latent = download(encoded.buffer, latent_count);
            for (float& item : latent) item *= 0.18215f;
            save(output / "rgb_latent.bin", latent);
            compare("rgb_latent", latent, load(fixture / "rgb_latent.bin", latent_count));
            auto sample = load(fixture / "rgb_latent.bin", latent_count);
            sample.insert(sample.end(), noise.begin(), noise.end());
            auto prediction = marigold_native::marigold_unet_gpu(context, unet, operators, prompt,
                sample.data(), width / 8, height / 8, full ? 901u : 999u);
            auto predicted = download(prediction.buffer, latent_count);
            save(output / (full ? "unet_t901.bin" : "unet_t999.bin"), predicted);
            // Full fixture stores the LAST DDIM prediction, not first t=901.
            // Its first-step UNet dump is a scalar/tiled pair comparison.
            if (!full) compare("unet", predicted, load(fixture / "unet_prediction.bin", latent_count));
            auto target = load(fixture / "target_latent.bin", latent_count);
            for (float& item : target) item /= 0.18215f;
            auto decoded = marigold_native::marigold_vae_decode_gpu(context, vae, operators,
                target.data(), width / 8, height / 8);
            auto values = download(decoded.buffer, 3 * pixels);
            save(output / "decoded.bin", values);
            compare("decoded", values, load(fixture / "decoded.bin", values.size()));
        }
        auto depth_buffer = marigold_native::marigold_infer_gpu(context, unet, vae, operators,
            prompt, rgb.data(), width, height, noise.data(), full != 0);
        auto depth = download(depth_buffer, pixels);
        save(output / "depth.bin", depth);
        for (float item : depth)
            if (!std::isfinite(item) || item < 0 || item > 1) throw std::runtime_error("invalid depth");
        if (!synthetic) compare("depth", depth, load(fixture / "depth.bin", pixels));
        std::cout << "device=" << context.device_name() << "\nwidth=" << width
                  << "\nheight=" << height << "\nfull_v1=" << full << "\nexplicit_noise=1\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
