#include "inferbridge_harness.h"
#include "marigold_native.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace {

ibrh_string_view view(const std::string& value) {
    return {value.data(), value.size()};
}

bool check(bool condition, const char* message) {
    if (!condition) std::cerr << message << '\n';
    return condition;
}

}  // namespace

int main() {
    const char* path = std::getenv("MARIGOLD_SNAPSHOT");
    const char* vae = std::getenv("MARIGOLD_VAE_MODEL");
    const char* prompt = std::getenv("MARIGOLD_PROMPT_CACHE");
    const char* checkpoint = std::getenv("MARIGOLD_CHECKPOINT");
    if (path == nullptr || *path == '\0' ||
        vae == nullptr || *vae == '\0' ||
        prompt == nullptr || *prompt == '\0') {
        std::cout <<
            "MARIGOLD_SNAPSHOT/VAE/PROMPT is not set; skipping\n";
        return 77;
    }
    ibrh_api api{};
    if (!check(
            ibrh_get_api(
                IBRH_CURRENT_API_VERSION, sizeof(api), &api) == IBRH_OK,
            "API negotiation failed"))
        return 1;

    const std::string backend = "native";
    const std::string device = "{\"index\":0}";
    ibrh_runtime_create_request runtime_request{
        sizeof(runtime_request), IBRH_CURRENT_API_VERSION,
        view(backend), view(device), {}, nullptr, nullptr};
    ibrh_runtime* runtime = nullptr;
    if (!check(
            api.runtime_create(
                sizeof(runtime_request), &runtime_request, &runtime) ==
                IBRH_OK &&
                runtime != nullptr,
            "runtime creation failed"))
        return 2;

    const std::string model_path = path;
    const std::string parameters =
        std::string("{\"Checkpoint\":\"") +
        (checkpoint == nullptr ?
            "prs-eth/marigold-lcm-v1-0" : checkpoint) +
        "\",\"VaeModel\":\"" + vae +
        "\",\"PromptCache\":\"" + prompt + "\"}";
    ibrh_model_load_request load_request{
        sizeof(load_request), IBRH_CURRENT_API_VERSION,
        view(model_path), view(parameters)};
    ibrh_model* model = nullptr;
    if (!check(
            api.model_load(
                runtime, sizeof(load_request), &load_request, &model) ==
                IBRH_OK &&
                model != nullptr,
            "model load failed")) {
        api.runtime_destroy(runtime);
        return 3;
    }

    constexpr uint32_t width = 53u;
    constexpr uint32_t height = 41u;
    uint32_t output_width = 0u;
    uint32_t output_height = 0u;
    if (!check(
            marigold_inferbridge_image_shape(
                width, height, &output_width, &output_height) == MARIGOLD_OK,
            "output shape failed"))
        return 4;
    std::vector<uint8_t> pixels(width * height * 4u);
    for (uint32_t y = 0u; y < height; ++y) {
        for (uint32_t x = 0u; x < width; ++x) {
            const size_t index =
                (static_cast<size_t>(y) * width + x) * 4u;
            pixels[index] = static_cast<uint8_t>((x * 7u + y) & 255u);
            pixels[index + 1u] =
                static_cast<uint8_t>((x + y * 11u) & 255u);
            pixels[index + 2u] =
                static_cast<uint8_t>((x * 3u + y * 5u) & 255u);
            pixels[index + 3u] = 255u;
        }
    }
    ibrh_resource input{};
    input.struct_size = sizeof(input);
    input.api_version = IBRH_CURRENT_API_VERSION;
    input.domain = IBRH_RESOURCE_DOMAIN_HOST;
    input.kind = IBRH_RESOURCE_KIND_IMAGE_2D;
    input.access = IBRH_RESOURCE_ACCESS_READ;
    input.pixel_format = IBRH_PIXEL_BGRA8;
    input.width = width;
    input.height = height;
    input.depth = 1u;
    input.row_stride_bytes = width * 4u;
    input.byte_size = pixels.size();
    input.native_handle_type = IBRH_NATIVE_HANDLE_HOST_POINTER;
    input.native_handle = static_cast<uint64_t>(
        reinterpret_cast<uintptr_t>(pixels.data()));
    std::vector<float> depth(
        static_cast<size_t>(output_width) * output_height);
    ibrh_resource output{};
    output.struct_size = sizeof(output);
    output.api_version = IBRH_CURRENT_API_VERSION;
    output.domain = IBRH_RESOURCE_DOMAIN_HOST;
    output.kind = IBRH_RESOURCE_KIND_IMAGE_2D;
    output.access = IBRH_RESOURCE_ACCESS_WRITE;
    output.pixel_format = IBRH_PIXEL_DEPTH_FLOAT32;
    output.width = output_width;
    output.height = output_height;
    output.depth = 1u;
    output.row_stride_bytes = output_width * sizeof(float);
    output.byte_size = depth.size() * sizeof(float);
    output.native_handle_type = IBRH_NATIVE_HANDLE_HOST_POINTER;
    output.native_handle = static_cast<uint64_t>(
        reinterpret_cast<uintptr_t>(depth.data()));
    ibrh_synchronization no_synchronization{};
    no_synchronization.struct_size = sizeof(no_synchronization);
    no_synchronization.api_version = IBRH_CURRENT_API_VERSION;
    no_synchronization.kind = IBRH_SYNC_NONE;
    ibrh_transfer_binding input_binding{
        sizeof(input_binding), IBRH_CURRENT_API_VERSION,
        input, no_synchronization};
    ibrh_transfer_binding output_binding{
        sizeof(output_binding), IBRH_CURRENT_API_VERSION,
        output, no_synchronization};
    ibrh_submit_request submit_request{
        sizeof(submit_request), IBRH_CURRENT_API_VERSION,
        &input_binding, 1u, &output_binding, 1u,
        123456u, 987654321u, {}};
    ibrh_job* job = nullptr;
    if (!check(
            api.submit(
                model, sizeof(submit_request), &submit_request, &job) ==
                IBRH_OK &&
                job != nullptr,
            "host submission failed")) {
        api.model_unload(model);
        api.runtime_destroy(runtime);
        return 4;
    }

    ibrh_job_status status{};
    if (!check(
            api.job_poll(job, sizeof(status), &status) == IBRH_OK &&
                status.state == IBRH_JOB_COMPLETE &&
                status.output_count == 1u &&
                status.source_frame_id == submit_request.source_frame_id,
            "job status/correlation failed"))
        return 5;

    api.job_release(job);
    job = nullptr;
    float minimum = 1.0f;
    float maximum = 0.0f;
    for (float value : depth) {
        minimum = std::min(minimum, value);
        maximum = std::max(maximum, value);
    }
    if (!check(
            minimum == 0.0f && maximum > 0.0f && maximum <= 1.0f,
            "output normalization failed"))
        return 8;

    api.model_unload(model);
    api.runtime_destroy(runtime);
    std::cout << "InferBridge host transfer lifecycle passed\n";
    return 0;
}
