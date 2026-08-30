#include "marigold_native.h"

int main(void) {
    marigold_context* context = 0;
    uint32_t width = 0;
    uint32_t height = 0;
    if (marigold_native_abi_version() != MARIGOLD_NATIVE_ABI_VERSION) {
        return 1;
    }
    if (marigold_create(0, 0, 0, &context) != MARIGOLD_INVALID_ARGUMENT) {
        return 2;
    }
    if (marigold_inferbridge_image_shape(
            53, 41, &width, &height) != MARIGOLD_OK ||
        width != 344 || height != 264) {
        return 3;
    }
    if (marigold_inferbridge_image_shape(
            53, 0, &width, &height) != MARIGOLD_INVALID_ARGUMENT) {
        return 4;
    }
    if (marigold_create_variant(
            "snapshot", "vae", "prompt",
            (marigold_model_variant)99, &context) !=
        MARIGOLD_INVALID_ARGUMENT) {
        return 5;
    }
    marigold_destroy(context);
    return 0;
}
