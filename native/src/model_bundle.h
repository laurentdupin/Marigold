#pragma once

#include "safetensors.h"

#include <memory>
#include <string>

namespace marigold_native {

class ModelBundle {
public:
    ModelBundle(
        const std::string& snapshot_root_utf8,
        const std::string& derived_vae_utf8,
        bool load_text_encoder = false);

    const SafeTensors& unet() const { return *unet_; }
    const SafeTensors& vae() const { return *vae_; }
    const SafeTensors& text_encoder() const;

private:
    std::unique_ptr<SafeTensors> unet_;
    std::unique_ptr<SafeTensors> vae_;
    std::unique_ptr<SafeTensors> text_encoder_;
};

}  // namespace marigold_native
