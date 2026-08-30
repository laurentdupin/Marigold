#ifndef MARIGOLD_NATIVE_H
#define MARIGOLD_NATIVE_H

#include <stdint.h>

#if defined(_WIN32)
#  if defined(MARIGOLD_NATIVE_BUILD)
#    define MARIGOLD_API __declspec(dllexport)
#  else
#    define MARIGOLD_API __declspec(dllimport)
#  endif
#else
#  define MARIGOLD_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define MARIGOLD_NATIVE_ABI_VERSION 4u

typedef struct marigold_context marigold_context;

enum {
    MARIGOLD_OK = 0,
    MARIGOLD_INVALID_ARGUMENT = 1,
    MARIGOLD_MODEL_ERROR = 2,
    MARIGOLD_RUNTIME_ERROR = 3
};

typedef enum marigold_model_variant {
    MARIGOLD_MODEL_LCM_V1 = 0,
    MARIGOLD_MODEL_FULL_V1 = 1
} marigold_model_variant;

MARIGOLD_API uint32_t marigold_native_abi_version(void);
MARIGOLD_API const char* marigold_last_error(void);
MARIGOLD_API int marigold_get_transfer_counters(
    uint64_t* upload_bytes, uint64_t* download_bytes);

MARIGOLD_API int marigold_create(
    const char* snapshot_root_utf8,
    const char* derived_vae_safetensors_utf8,
    const char* empty_prompt_cache_utf8,
    marigold_context** output);
MARIGOLD_API int marigold_create_vulkan(
    const char* snapshot_root_utf8,
    const char* derived_vae_safetensors_utf8,
    const char* empty_prompt_cache_utf8,
    uint32_t device_index,
    marigold_context** output);
MARIGOLD_API int marigold_create_variant(
    const char* snapshot_root_utf8,
    const char* vae_safetensors_utf8,
    const char* empty_prompt_cache_utf8,
    marigold_model_variant variant,
    marigold_context** output);
MARIGOLD_API int marigold_create_vulkan_variant(
    const char* snapshot_root_utf8,
    const char* vae_safetensors_utf8,
    const char* empty_prompt_cache_utf8,
    marigold_model_variant variant,
    uint32_t device_index,
    marigold_context** output);
MARIGOLD_API void marigold_destroy(marigold_context* context);

/*
 * RGB is interleaved float32 in [0,1]. Depth contains width*height float32
 * values in [0,1]. Explicit noise is NCHW with
 * 4*floor(width/8)*floor(height/8) values.
 */
MARIGOLD_API int marigold_infer_rgb_f32_with_noise(
    marigold_context* context,
    const float* rgb,
    uint32_t width,
    uint32_t height,
    const float* target_noise,
    float* depth);

MARIGOLD_API int marigold_infer_rgb_f32(
    marigold_context* context,
    const float* rgb,
    uint32_t width,
    uint32_t height,
    uint64_t seed,
    float* depth);

/*
 * InferBridge image contract. The source is BGRA8. The Python harness first
 * converts BGR to RGB, then resizes the longest edge to 384 by default with
 * antialiased bilinear filtering. INFERBRIDGE_DIFFUSION_LONG_EDGE can select
 * 256 through 1024. Its match_input_res=false result therefore has the
 * processing dimensions returned by marigold_inferbridge_image_shape.
 */
MARIGOLD_API int marigold_inferbridge_image_shape(
    uint32_t source_width,
    uint32_t source_height,
    uint32_t* processing_width,
    uint32_t* processing_height);

MARIGOLD_API int marigold_infer_bgra8_f32_with_noise(
    marigold_context* context,
    const uint8_t* bgra,
    uint32_t width,
    uint32_t height,
    uint32_t row_stride_bytes,
    const float* target_noise,
    float* depth);

MARIGOLD_API int marigold_infer_bgra8_f32(
    marigold_context* context,
    const uint8_t* bgra,
    uint32_t width,
    uint32_t height,
    uint32_t row_stride_bytes,
    uint64_t seed,
    float* depth);

#ifdef __cplusplus
}
#endif

#endif
