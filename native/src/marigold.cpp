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

void inferbridge_shape(
    std::uint32_t width,
    std::uint32_t height,
    std::uint32_t& processing_width,
    std::uint32_t& processing_height) {
    if (width == 0 || height == 0) {
        throw std::invalid_argument(
            "Marigold image dimensions must be non-zero");
    }
    constexpr std::uint32_t processing_resolution = 768;
    const double scale = std::min(
        static_cast<double>(processing_resolution) / width,
        static_cast<double>(processing_resolution) / height);
    processing_width = std::max(
        1u, static_cast<std::uint32_t>(static_cast<double>(width) * scale));
    processing_height = std::max(
        1u, static_cast<std::uint32_t>(static_cast<double>(height) * scale));
}

struct FilterTap {
    std::uint32_t index;
    float weight;
};

std::vector<std::vector<FilterTap>> bilinear_taps(
    std::uint32_t input_size,
    std::uint32_t output_size) {
    std::vector<std::vector<FilterTap>> taps(output_size);
    const double scale = static_cast<double>(input_size) / output_size;
    const double support = std::max(1.0, scale);
    const double inverse_scale = scale >= 1.0 ? 1.0 / scale : 1.0;
    for (std::uint32_t output = 0; output < output_size; ++output) {
        // This is the coordinate convention used by PyTorch's antialiased
        // upsample kernel (and ultimately Pillow): pixel centres are at
        // half-integers and the filter is truncated, then renormalized, at
        // image boundaries.
        const double center = scale * (output + 0.5);
        const std::int64_t begin = std::max<std::int64_t>(
            static_cast<std::int64_t>(center - support + 0.5), 0);
        const std::int64_t end = std::min<std::int64_t>(
            static_cast<std::int64_t>(center + support + 0.5), input_size);
        std::vector<FilterTap>& row = taps[output];
        for (std::int64_t input = begin; input < end; ++input) {
            const double distance =
                std::abs((input - center + 0.5) * inverse_scale);
            const double raw_weight = std::max(0.0, 1.0 - distance);
            if (raw_weight == 0.0) {
                continue;
            }
            row.push_back({
                static_cast<std::uint32_t>(input),
                static_cast<float>(raw_weight)});
        }
        float sum = 0.0f;
        for (const FilterTap& tap : row) {
            sum += tap.weight;
        }
        for (FilterTap& tap : row) {
            tap.weight /= sum;
        }
    }
    return taps;
}

std::vector<float> preprocess_bgra(
    const std::uint8_t* bgra,
    std::uint32_t width,
    std::uint32_t height,
    std::uint32_t row_stride,
    std::uint32_t processing_width,
    std::uint32_t processing_height) {
    const auto horizontal = bilinear_taps(width, processing_width);
    const auto vertical = bilinear_taps(height, processing_height);
    std::vector<float> temporary(
        std::uint64_t(height) * processing_width * 3);
    for (std::uint32_t y = 0; y < height; ++y) {
        const std::uint8_t* row = bgra + std::uint64_t(y) * row_stride;
        for (std::uint32_t x = 0; x < processing_width; ++x) {
            for (std::uint32_t c = 0; c < 3; ++c) {
                float value = 0.0f;
                for (const FilterTap& tap : horizontal[x]) {
                    // cv2 COLOR_BGR2RGB reverses the first three BGRA bytes.
                    value += tap.weight * static_cast<float>(
                        row[std::uint64_t(tap.index) * 4 + (2 - c)]);
                }
                temporary[
                    (std::uint64_t(y) * processing_width + x) * 3 + c] =
                    value / 255.0f;
            }
        }
    }
    std::vector<float> output(
        std::uint64_t(processing_height) * processing_width * 3);
    for (std::uint32_t y = 0; y < processing_height; ++y) {
        for (std::uint32_t x = 0; x < processing_width; ++x) {
            for (std::uint32_t c = 0; c < 3; ++c) {
                float value = 0.0f;
                for (const FilterTap& tap : vertical[y]) {
                    value += tap.weight * temporary[
                        (std::uint64_t(tap.index) * processing_width + x) *
                            3 +
                        c];
                }
                output[
                    (std::uint64_t(y) * processing_width + x) * 3 + c] =
                    value;
            }
        }
    }
    return output;
}

void inferbridge_normalize(
    float* depth,
    std::uint32_t width,
    std::uint32_t height) {
    const std::uint64_t count = std::uint64_t(width) * height;
    float minimum = depth[0];
    for (std::uint64_t i = 1; i < count; ++i) {
        minimum = std::min(minimum, depth[i]);
    }
    const float denominator = 1.0f - minimum;
    for (std::uint64_t i = 0; i < count; ++i) {
        depth[i] =
            denominator > 0.0f ? (depth[i] - minimum) / denominator : 0.0f;
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

int marigold_inferbridge_image_shape(
    std::uint32_t source_width,
    std::uint32_t source_height,
    std::uint32_t* processing_width,
    std::uint32_t* processing_height) {
    if (!processing_width || !processing_height) {
        return fail(
            MARIGOLD_INVALID_ARGUMENT, "invalid Marigold shape argument");
    }
    try {
        inferbridge_shape(
            source_width, source_height,
            *processing_width, *processing_height);
        last_error.clear();
        return MARIGOLD_OK;
    } catch (const std::exception& error) {
        return fail(MARIGOLD_INVALID_ARGUMENT, error);
    }
}

int marigold_infer_bgra8_f32_with_noise(
    marigold_context* context,
    const std::uint8_t* bgra,
    std::uint32_t width,
    std::uint32_t height,
    std::uint32_t row_stride_bytes,
    const float* target_noise,
    float* depth) {
    if (!context || !bgra || !target_noise || !depth ||
        width == 0 || height == 0 ||
        row_stride_bytes < std::uint64_t(width) * 4) {
        return fail(
            MARIGOLD_INVALID_ARGUMENT,
            "invalid Marigold BGRA inference argument");
    }
    try {
        std::uint32_t processing_width = 0;
        std::uint32_t processing_height = 0;
        inferbridge_shape(
            width, height, processing_width, processing_height);
        std::vector<float> rgb = preprocess_bgra(
            bgra, width, height, row_stride_bytes,
            processing_width, processing_height);
        const int result = marigold_infer_rgb_f32_with_noise(
            context, rgb.data(), processing_width, processing_height,
            target_noise, depth);
        if (result != MARIGOLD_OK) {
            return result;
        }
        inferbridge_normalize(depth, processing_width, processing_height);
        last_error.clear();
        return MARIGOLD_OK;
    } catch (const std::exception& error) {
        return fail(MARIGOLD_RUNTIME_ERROR, error);
    }
}

int marigold_infer_bgra8_f32(
    marigold_context* context,
    const std::uint8_t* bgra,
    std::uint32_t width,
    std::uint32_t height,
    std::uint32_t row_stride_bytes,
    std::uint64_t seed,
    float* depth) {
    std::uint32_t processing_width = 0;
    std::uint32_t processing_height = 0;
    const int shape_result = marigold_inferbridge_image_shape(
        width, height, &processing_width, &processing_height);
    if (shape_result != MARIGOLD_OK) {
        return shape_result;
    }
    const std::uint64_t count =
        std::uint64_t(4) * (processing_width / 8) *
        (processing_height / 8);
    std::mt19937_64 generator(seed);
    std::normal_distribution<float> normal;
    std::vector<float> noise(static_cast<std::size_t>(count));
    for (float& value : noise) {
        value = normal(generator);
    }
    return marigold_infer_bgra8_f32_with_noise(
        context, bgra, width, height, row_stride_bytes,
        noise.data(), depth);
}

}  // extern "C"
