#include "marigold_native.h"

#include "model_bundle.h"
#include "prompt_cache.h"
#include "unet_cpu.h"
#include "vae_cpu.h"
#if defined(MARIGOLD_WITH_VULKAN)
#include "gpu_model.h"
#include "marigold_gpu.h"
#include "operators.h"
#include "vulkan.h"
#endif

#include <algorithm>
#include <cmath>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

struct marigold_context {
    std::unique_ptr<marigold_native::ModelBundle> model;
    marigold_native::TokenTensor prompt;
#if defined(MARIGOLD_WITH_VULKAN)
    std::unique_ptr<marigold_native::VulkanContext> vulkan;
    std::unique_ptr<marigold_native::GpuModel> gpu_unet;
    std::unique_ptr<marigold_native::GpuModel> gpu_vae;
    std::unique_ptr<marigold_native::VulkanOperators> operators;
#endif
};

namespace {

thread_local std::string last_error;

int fail(int code, const char* message) {
    last_error = message;
    return code;
}

int fail(int code, const std::exception& error) {
    last_error = error.what();
    return code;
}

marigold_native::ImageTensor infer(
    marigold_context& context,
    const float* rgb,
    std::uint32_t width,
    std::uint32_t height,
    const float* target_noise) {
    if (width < 8 || height < 8) {
        throw std::invalid_argument("Marigold input dimensions are too small");
    }
    marigold_native::ImageTensor image{
        3, height, width,
        std::vector<float>(std::uint64_t(3) * height * width)};
    for (std::uint32_t y = 0; y < height; ++y) {
        for (std::uint32_t x = 0; x < width; ++x) {
            for (std::uint32_t c = 0; c < 3; ++c) {
                image.values[
                    (std::uint64_t(c) * height + y) * width + x] =
                    rgb[(std::uint64_t(y) * width + x) * 3 + c] *
                        2.0f -
                    1.0f;
            }
        }
    }
    marigold_native::Posterior posterior =
        marigold_native::vae_encode(context.model->vae(), image);
    for (float& value : posterior.mean.values) {
        value *= 0.18215f;
    }
    const std::uint64_t latent_elements = posterior.mean.values.size();
    marigold_native::ImageTensor sample{
        8, posterior.mean.height, posterior.mean.width, {}};
    sample.values.reserve(static_cast<std::size_t>(latent_elements * 2));
    sample.values.insert(
        sample.values.end(),
        posterior.mean.values.begin(), posterior.mean.values.end());
    sample.values.insert(
        sample.values.end(),
        target_noise, target_noise + latent_elements);
    const marigold_native::ImageTensor prediction =
        marigold_native::unet_predict(
            context.model->unet(), sample, 999, context.prompt);

    constexpr float alpha = 0.00466009508818388f;
    constexpr float c_skip = 2.505007534736592e-9f;
    constexpr float c_out = 1.0f;
    const float sqrt_alpha = std::sqrt(alpha);
    const float sqrt_beta = std::sqrt(1.0f - alpha);
    marigold_native::ImageTensor target{
        4, posterior.mean.height, posterior.mean.width,
        std::vector<float>(static_cast<std::size_t>(latent_elements))};
    for (std::uint64_t i = 0; i < latent_elements; ++i) {
        const float original =
            sqrt_alpha * target_noise[i] -
            sqrt_beta * prediction.values[i];
        target.values[i] =
            c_out * original + c_skip * target_noise[i];
        target.values[i] /= 0.18215f;
    }
    return marigold_native::vae_decode(context.model->vae(), target);
}

void output_depth(
    const marigold_native::ImageTensor& decoded,
    std::uint32_t target_width,
    std::uint32_t target_height,
    float* output) {
    for (std::uint32_t y = 0; y < target_height; ++y) {
        const std::uint32_t source_y = std::min(
            decoded.height - 1,
            static_cast<std::uint32_t>(
                std::uint64_t(y) * decoded.height / target_height));
        for (std::uint32_t x = 0; x < target_width; ++x) {
            const std::uint32_t source_x = std::min(
                decoded.width - 1,
                static_cast<std::uint32_t>(
                    std::uint64_t(x) * decoded.width / target_width));
            float sum = 0.0f;
            for (std::uint32_t c = 0; c < 3; ++c) {
                const float raw = decoded.values[
                    (std::uint64_t(c) * decoded.height + source_y) *
                        decoded.width +
                    source_x];
                sum += (std::clamp(raw, -1.0f, 1.0f) + 1.0f) * 0.5f;
            }
            output[std::uint64_t(y) * target_width + x] = sum / 3.0f;
        }
    }
}

}  // namespace

extern "C" {

std::uint32_t marigold_native_abi_version(void) {
    return MARIGOLD_NATIVE_ABI_VERSION;
}

const char* marigold_last_error(void) {
    return last_error.c_str();
}

int marigold_create(
    const char* snapshot,
    const char* derived_vae,
    const char* prompt_cache,
    marigold_context** output) {
    if (!snapshot || !derived_vae || !prompt_cache || !output) {
        return fail(
            MARIGOLD_INVALID_ARGUMENT,
            "invalid Marigold create argument");
    }
    *output = nullptr;
    try {
        auto context = std::make_unique<marigold_context>();
        context->model =
            std::make_unique<marigold_native::ModelBundle>(
                snapshot, derived_vae);
        context->prompt =
            marigold_native::load_empty_prompt_cache(prompt_cache);
        *output = context.release();
        last_error.clear();
        return MARIGOLD_OK;
    } catch (const std::exception& error) {
        return fail(MARIGOLD_MODEL_ERROR, error);
    }
}

int marigold_create_vulkan(
    const char* snapshot,
    const char* derived_vae,
    const char* prompt_cache,
    std::uint32_t device_index,
    marigold_context** output) {
    if (!snapshot || !derived_vae || !prompt_cache || !output) {
        return fail(
            MARIGOLD_INVALID_ARGUMENT,
            "invalid Marigold create argument");
    }
    *output = nullptr;
#if !defined(MARIGOLD_WITH_VULKAN)
    (void)device_index;
    return fail(
        MARIGOLD_RUNTIME_ERROR, "this DLL was built without Vulkan");
#else
    try {
        auto context = std::make_unique<marigold_context>();
        context->model =
            std::make_unique<marigold_native::ModelBundle>(
                snapshot, derived_vae);
        context->prompt =
            marigold_native::load_empty_prompt_cache(prompt_cache);
        context->vulkan =
            std::make_unique<marigold_native::VulkanContext>(device_index);
        context->gpu_unet = std::make_unique<marigold_native::GpuModel>(
            context->model->unet(), *context->vulkan);
        context->gpu_vae = std::make_unique<marigold_native::GpuModel>(
            context->model->vae(), *context->vulkan);
        context->operators =
            std::make_unique<marigold_native::VulkanOperators>(
                *context->vulkan);
        *output = context.release();
        last_error.clear();
        return MARIGOLD_OK;
    } catch (const std::exception& error) {
        return fail(MARIGOLD_MODEL_ERROR, error);
    }
#endif
}

void marigold_destroy(marigold_context* context) {
    delete context;
}

int marigold_infer_rgb_f32_with_noise(
    marigold_context* context,
    const float* rgb,
    std::uint32_t width,
    std::uint32_t height,
    const float* target_noise,
    float* depth) {
    if (!context || !rgb || !target_noise || !depth) {
        return fail(
            MARIGOLD_INVALID_ARGUMENT,
            "invalid Marigold inference argument");
    }
    try {
#if defined(MARIGOLD_WITH_VULKAN)
        if (context->vulkan) {
            marigold_native::VulkanBuffer output =
                marigold_native::marigold_infer_gpu(
                    *context->vulkan, *context->gpu_unet,
                    *context->gpu_vae, *context->operators,
                    context->prompt, rgb, width, height, target_noise);
            context->vulkan->download(
                output, depth,
                std::uint64_t(width) * height * sizeof(float));
            last_error.clear();
            return MARIGOLD_OK;
        }
#endif
        output_depth(
            infer(*context, rgb, width, height, target_noise),
            width, height, depth);
        last_error.clear();
        return MARIGOLD_OK;
    } catch (const std::exception& error) {
        return fail(MARIGOLD_RUNTIME_ERROR, error);
    }
}

int marigold_infer_rgb_f32(
    marigold_context* context,
    const float* rgb,
    std::uint32_t width,
    std::uint32_t height,
    std::uint64_t seed,
    float* depth) {
    if (!context || !rgb || !depth || width < 8 || height < 8) {
        return fail(
            MARIGOLD_INVALID_ARGUMENT,
            "invalid Marigold inference argument");
    }
    const std::uint64_t count =
        std::uint64_t(4) * (width / 8) * (height / 8);
    std::mt19937_64 generator(seed);
    std::normal_distribution<float> normal;
    std::vector<float> noise(static_cast<std::size_t>(count));
    for (float& value : noise) {
        value = normal(generator);
    }
    return marigold_infer_rgb_f32_with_noise(
        context, rgb, width, height, noise.data(), depth);
}

}  // extern "C"
