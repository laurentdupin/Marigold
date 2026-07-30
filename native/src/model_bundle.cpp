#include "model_bundle.h"

#include <filesystem>
#include <stdexcept>

namespace marigold_native {
namespace {

std::string component(
    const std::string& root,
    const char* directory,
    const char* file) {
    const std::filesystem::path path =
        std::filesystem::u8path(root) / directory / file;
    if (!std::filesystem::is_regular_file(path)) {
        throw std::runtime_error(
            "missing Marigold model component: " + path.u8string());
    }
    return path.u8string();
}

std::string file(const std::string& path_utf8) {
    const std::filesystem::path path =
        std::filesystem::u8path(path_utf8);
    if (!std::filesystem::is_regular_file(path)) {
        throw std::runtime_error(
            "missing derived Marigold VAE: " + path.u8string());
    }
    return path.u8string();
}

}  // namespace

ModelBundle::ModelBundle(
    const std::string& root,
    const std::string& derived_vae,
    bool load_text_encoder)
    : unet_(std::make_unique<SafeTensors>(component(
          root, "unet",
          "diffusion_pytorch_model.fp16.safetensors"))),
      vae_(std::make_unique<SafeTensors>(file(derived_vae))) {
    if (load_text_encoder) {
        text_encoder_ = std::make_unique<SafeTensors>(component(
            root, "text_encoder", "model.fp16.safetensors"));
    }
}

const SafeTensors& ModelBundle::text_encoder() const {
    if (!text_encoder_) {
        throw std::runtime_error("Marigold text encoder was not loaded");
    }
    return *text_encoder_;
}

}  // namespace marigold_native
