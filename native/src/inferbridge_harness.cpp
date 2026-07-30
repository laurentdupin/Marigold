#include "inferbridge_harness.h"

#include "marigold_native.h"

#include <atomic>
#include <cstdint>
#include <cstring>
#include <limits>
#include <mutex>
#include <new>
#include <string>
#include <vector>

struct ibrh_runtime {
    std::string error;
    int32_t vulkan_device_index = 0;
};

struct ibrh_model {
    ibrh_runtime* runtime = nullptr;
    marigold_context* context = nullptr;
    std::mutex submit_mutex;
};

struct ibrh_job {
    std::atomic<uint32_t> references{1u};
    uint64_t source_frame_id = 0u;
    uint64_t timestamp_ns = 0u;
    uint32_t width = 0u;
    uint32_t height = 0u;
    std::vector<float> depth;
};

struct ibrh_output_lease {
    ibrh_job* job = nullptr;
};

namespace {

thread_local std::string g_last_error;
constexpr char kHarnessId[] = "inferbridge.marigold.native";
constexpr char kHarnessVersion[] = "1.0.0";

ibrh_result fail(
    ibrh_runtime* runtime, ibrh_result result, const std::string& message) {
    g_last_error = message;
    if (runtime != nullptr) runtime->error = message;
    return result;
}

std::string copy_string(ibrh_string_view value) {
    return value.size == 0u ? std::string() :
        std::string(value.data, value.size);
}

bool valid_string(ibrh_string_view value) {
    return value.data != nullptr && value.size != 0u &&
        std::memchr(value.data, '\0', value.size) == nullptr;
}

bool json_string(
    const std::string& json, const std::string& key, std::string& value) {
    const std::string marker = "\"" + key + "\"";
    size_t position = json.find(marker);
    if (position == std::string::npos) return false;
    position = json.find(':', position + marker.size());
    if (position == std::string::npos) return false;
    position = json.find_first_not_of(" \t\r\n", position + 1u);
    if (position == std::string::npos || json[position] != '"') return false;
    const size_t end = json.find('"', position + 1u);
    if (end == std::string::npos) return false;
    value = json.substr(position + 1u, end - position - 1u);
    return true;
}

bool json_uint64(
    const std::string& json, const std::string& key, uint64_t& value) {
    const std::string marker = "\"" + key + "\"";
    size_t position = json.find(marker);
    if (position == std::string::npos) return false;
    position = json.find(':', position + marker.size());
    if (position == std::string::npos) return false;
    position = json.find_first_not_of(" \t\r\n", position + 1u);
    if (position == std::string::npos) return false;
    if (json[position] == '"') ++position;
    size_t end = position;
    while (end < json.size() && json[end] >= '0' && json[end] <= '9') ++end;
    if (end == position) return false;
    uint64_t parsed = 0u;
    for (size_t index = position; index < end; ++index) {
        const uint64_t digit =
            static_cast<uint64_t>(json[index] - '0');
        if (parsed >
            (std::numeric_limits<uint64_t>::max() - digit) / 10u)
            return false;
        parsed = parsed * 10u + digit;
    }
    value = parsed;
    return true;
}

marigold_model_variant model_variant(const std::string& parameters) {
    std::string checkpoint;
    (void)json_string(parameters, "Checkpoint", checkpoint);
    return checkpoint.find("marigold-v1-0") != std::string::npos &&
            checkpoint.find("lcm") == std::string::npos ?
        MARIGOLD_MODEL_FULL_V1 : MARIGOLD_MODEL_LCM_V1;
}

ibrh_result status_result(int status) {
    switch (status) {
        case MARIGOLD_OK: return IBRH_OK;
        case MARIGOLD_INVALID_ARGUMENT:
            return IBRH_ERROR_INVALID_ARGUMENT;
        case MARIGOLD_MODEL_ERROR:
        case MARIGOLD_RUNTIME_ERROR:
        default:
            return IBRH_ERROR_INTERNAL;
    }
}

void retain_job(ibrh_job* job) {
    (void)job->references.fetch_add(1u);
}

void release_job(ibrh_job* job) {
    if (job != nullptr && job->references.fetch_sub(1u) == 1u) delete job;
}

ibrh_result IBRH_CALL query_capabilities(
    size_t capabilities_size, ibrh_capabilities* capabilities) {
    if (capabilities == nullptr) return IBRH_ERROR_INVALID_ARGUMENT;
    if (capabilities_size < sizeof(*capabilities))
        return IBRH_ERROR_STRUCT_TOO_SMALL;
    *capabilities = {};
    capabilities->struct_size = sizeof(*capabilities);
    capabilities->api_version = IBRH_CURRENT_API_VERSION;
    capabilities->flags = IBRH_CAP_HOST_MEMORY;
    capabilities->input_domain_mask =
        1ull << IBRH_RESOURCE_DOMAIN_HOST;
    capabilities->output_domain_mask =
        1ull << IBRH_RESOURCE_DOMAIN_HOST;
    capabilities->maximum_inputs = 1u;
    capabilities->maximum_outputs = 1u;
    capabilities->maximum_in_flight_jobs = 1u;
    capabilities->harness_id = {kHarnessId, sizeof(kHarnessId) - 1u};
    capabilities->harness_version = {
        kHarnessVersion, sizeof(kHarnessVersion) - 1u};
    return IBRH_OK;
}

ibrh_result IBRH_CALL runtime_create(
    size_t request_size, const ibrh_runtime_create_request* request,
    ibrh_runtime** output) {
    if (request == nullptr || output == nullptr)
        return IBRH_ERROR_INVALID_ARGUMENT;
    *output = nullptr;
    if (request_size < sizeof(*request) ||
        request->struct_size < sizeof(*request))
        return IBRH_ERROR_STRUCT_TOO_SMALL;
    auto* runtime = new (std::nothrow) ibrh_runtime();
    if (runtime == nullptr) return IBRH_ERROR_INTERNAL;
    const std::string device = copy_string(request->requested_device_json);
    uint64_t index = 0u;
    if (json_uint64(device, "index", index)) {
        if (index > static_cast<uint64_t>(std::numeric_limits<int32_t>::max())) {
            delete runtime;
            return fail(
                nullptr, IBRH_ERROR_INVALID_ARGUMENT,
                "Marigold requested device index is out of range");
        }
        runtime->vulkan_device_index = static_cast<int32_t>(index);
    }
    std::string luid_text;
    if (!json_uint64(device, "index", index) &&
        json_string(device, "luid", luid_text) && !luid_text.empty()) {
        delete runtime;
        return fail(
            nullptr, IBRH_ERROR_UNSUPPORTED_CAPABILITY,
            "Marigold requires a Vulkan device index when a LUID is requested");
    }
    *output = runtime;
    return IBRH_OK;
}

void IBRH_CALL runtime_destroy(ibrh_runtime* runtime) {
    delete runtime;
}

ibrh_result IBRH_CALL model_load(
    ibrh_runtime* runtime, size_t request_size,
    const ibrh_model_load_request* request, ibrh_model** output) {
    if (runtime == nullptr || request == nullptr || output == nullptr)
        return IBRH_ERROR_INVALID_ARGUMENT;
    *output = nullptr;
    if (request_size < sizeof(*request) ||
        request->struct_size < sizeof(*request))
        return IBRH_ERROR_STRUCT_TOO_SMALL;
    if (!valid_string(request->model_path))
        return fail(
            runtime, IBRH_ERROR_INVALID_ARGUMENT,
            "Marigold model path is missing");
    const std::string path = copy_string(request->model_path);
    const std::string parameters = copy_string(request->parameters_json);
    std::string vae_model;
    std::string prompt_cache;
    if (!json_string(parameters, "VaeModel", vae_model) ||
        vae_model.empty() ||
        !json_string(parameters, "PromptCache", prompt_cache) ||
        prompt_cache.empty()) {
        return fail(
            runtime, IBRH_ERROR_INVALID_ARGUMENT,
            "Marigold VaeModel and PromptCache paths are required");
    }
    auto* model = new (std::nothrow) ibrh_model();
    if (model == nullptr) return IBRH_ERROR_INTERNAL;
    model->runtime = runtime;
    const int status =
        marigold_create_vulkan_variant(
            path.c_str(),
            vae_model.c_str(),
            prompt_cache.c_str(),
            model_variant(parameters),
            static_cast<uint32_t>(runtime->vulkan_device_index),
            &model->context);
    if (status != MARIGOLD_OK) {
        const std::string message =
            std::string("Marigold model load failed: ") + marigold_last_error();
        delete model;
        return fail(runtime, status_result(status), message);
    }
    *output = model;
    return IBRH_OK;
}

void IBRH_CALL model_unload(ibrh_model* model) {
    if (model == nullptr) return;
    marigold_destroy(model->context);
    delete model;
}

ibrh_result IBRH_CALL submit(
    ibrh_model* model, size_t request_size,
    const ibrh_submit_request* request, ibrh_job** output) {
    if (model == nullptr || request == nullptr || output == nullptr)
        return IBRH_ERROR_INVALID_ARGUMENT;
    *output = nullptr;
    if (request_size < sizeof(*request) ||
        request->struct_size < sizeof(*request))
        return IBRH_ERROR_STRUCT_TOO_SMALL;
    if (request->input_count != 1u || request->inputs == nullptr)
        return fail(
            model->runtime, IBRH_ERROR_INVALID_ARGUMENT,
            "Marigold requires exactly one BGRA8 input");
    if (request->synchronization_count != 0u)
        return fail(
            model->runtime, IBRH_ERROR_UNSUPPORTED_CAPABILITY,
            "Marigold host harness does not accept external synchronization");
    const ibrh_resource& input = request->inputs[0];
    if (input.struct_size < sizeof(input))
        return IBRH_ERROR_STRUCT_TOO_SMALL;
    if (input.domain != IBRH_RESOURCE_DOMAIN_HOST ||
        input.kind != IBRH_RESOURCE_KIND_IMAGE_2D ||
        input.native_handle_type != IBRH_NATIVE_HANDLE_HOST_POINTER ||
        input.pixel_format != IBRH_PIXEL_BGRA8 ||
        input.native_handle == 0u || input.width == 0u ||
        input.height == 0u || input.width > UINT32_MAX / 4u ||
        input.row_stride_bytes < input.width * 4u ||
        input.byte_offset > input.byte_size ||
        input.byte_size - input.byte_offset <
            static_cast<uint64_t>(input.row_stride_bytes) * input.height) {
        return fail(
            model->runtime, IBRH_ERROR_UNSUPPORTED_CAPABILITY,
            "Marigold harness requires a valid host BGRA8 image");
    }
    uint32_t output_width = 0u;
    uint32_t output_height = 0u;
    const int shape_status = marigold_inferbridge_image_shape(
        input.width,
        input.height,
        &output_width,
        &output_height);
    if (shape_status != MARIGOLD_OK)
        return fail(
            model->runtime, status_result(shape_status),
            marigold_last_error());
    uint64_t seed = 12345u;
    const std::string submit_parameters =
        copy_string(request->parameters_json);
    if (submit_parameters.find("\"Seed\"") != std::string::npos &&
        !json_uint64(submit_parameters, "Seed", seed))
        return fail(
            model->runtime, IBRH_ERROR_INVALID_ARGUMENT,
            "Marigold Seed must be an unsigned integer");
    auto* job = new (std::nothrow) ibrh_job();
    if (job == nullptr) return IBRH_ERROR_INTERNAL;
    job->source_frame_id = request->source_frame_id;
    job->timestamp_ns = request->timestamp_ns;
    job->width = static_cast<uint32_t>(output_width);
    job->height = static_cast<uint32_t>(output_height);
    try {
        job->depth.resize(
            static_cast<size_t>(job->width) * job->height);
    } catch (...) {
        delete job;
        return IBRH_ERROR_INTERNAL;
    }
    const auto* bgra = reinterpret_cast<const uint8_t*>(
        static_cast<uintptr_t>(input.native_handle)) + input.byte_offset;
    {
        std::lock_guard<std::mutex> lock(model->submit_mutex);
        const int status = marigold_infer_bgra8_f32(
            model->context,
            bgra,
            input.width,
            input.height,
            input.row_stride_bytes,
            seed,
            job->depth.data());
        if (status != MARIGOLD_OK) {
            const std::string message =
                std::string("Marigold inference failed: ") +
                marigold_last_error();
            delete job;
            return fail(model->runtime, status_result(status), message);
        }
    }
    *output = job;
    return IBRH_OK;
}

ibrh_result IBRH_CALL job_poll(
    const ibrh_job* job, size_t status_size, ibrh_job_status* status) {
    if (job == nullptr || status == nullptr)
        return IBRH_ERROR_INVALID_ARGUMENT;
    if (status_size < sizeof(*status)) return IBRH_ERROR_STRUCT_TOO_SMALL;
    *status = {};
    status->struct_size = sizeof(*status);
    status->state = IBRH_JOB_COMPLETE;
    status->output_count = 1u;
    status->source_frame_id = job->source_frame_id;
    return IBRH_OK;
}

ibrh_result IBRH_CALL job_cancel(ibrh_job* job) {
    return job == nullptr ?
        IBRH_ERROR_INVALID_ARGUMENT : IBRH_ERROR_INVALID_STATE;
}

void IBRH_CALL job_release(ibrh_job* job) {
    release_job(job);
}

ibrh_result IBRH_CALL output_acquire(
    ibrh_job* job, uint32_t output_index, size_t descriptor_size,
    ibrh_output_descriptor* descriptor, ibrh_output_lease** output) {
    if (job == nullptr || descriptor == nullptr || output == nullptr)
        return IBRH_ERROR_INVALID_ARGUMENT;
    *output = nullptr;
    if (descriptor_size < sizeof(*descriptor))
        return IBRH_ERROR_STRUCT_TOO_SMALL;
    if (output_index != 0u) return IBRH_ERROR_NOT_FOUND;
    auto* lease = new (std::nothrow) ibrh_output_lease();
    if (lease == nullptr) return IBRH_ERROR_INTERNAL;
    retain_job(job);
    lease->job = job;
    *descriptor = {};
    descriptor->struct_size = sizeof(*descriptor);
    descriptor->api_version = IBRH_CURRENT_API_VERSION;
    descriptor->output_index = 0u;
    descriptor->payload_type = IBRH_PIXEL_DEPTH_FLOAT32;
    descriptor->source_frame_id = job->source_frame_id;
    descriptor->timestamp_ns = job->timestamp_ns;
    descriptor->resource.struct_size = sizeof(descriptor->resource);
    descriptor->resource.api_version = IBRH_CURRENT_API_VERSION;
    descriptor->resource.domain = IBRH_RESOURCE_DOMAIN_HOST;
    descriptor->resource.kind = IBRH_RESOURCE_KIND_IMAGE_2D;
    descriptor->resource.access = IBRH_RESOURCE_ACCESS_READ;
    descriptor->resource.pixel_format = IBRH_PIXEL_DEPTH_FLOAT32;
    descriptor->resource.width = job->width;
    descriptor->resource.height = job->height;
    descriptor->resource.depth = 1u;
    descriptor->resource.row_stride_bytes = job->width * sizeof(float);
    descriptor->resource.byte_size = job->depth.size() * sizeof(float);
    descriptor->resource.native_handle_type =
        IBRH_NATIVE_HANDLE_HOST_POINTER;
    descriptor->resource.native_handle = static_cast<uint64_t>(
        reinterpret_cast<uintptr_t>(job->depth.data()));
    *output = lease;
    return IBRH_OK;
}

void IBRH_CALL output_release(ibrh_output_lease* lease) {
    if (lease == nullptr) return;
    release_job(lease->job);
    delete lease;
}

ibrh_result IBRH_CALL get_last_error(
    const void* object, char* destination, size_t destination_size,
    size_t* required_size) {
    const auto* runtime = static_cast<const ibrh_runtime*>(object);
    const std::string& message =
        runtime != nullptr && !runtime->error.empty() ?
        runtime->error : g_last_error;
    const size_t required = message.size() + 1u;
    if (required_size != nullptr) *required_size = required;
    if (destination == nullptr || destination_size < required)
        return IBRH_ERROR_STRUCT_TOO_SMALL;
    std::memcpy(destination, message.c_str(), required);
    return IBRH_OK;
}

}  // namespace

extern "C" IBRH_API ibrh_result IBRH_CALL ibrh_get_api(
    uint32_t requested_api_version, size_t api_size, ibrh_api* api) {
    if (api == nullptr) return IBRH_ERROR_INVALID_ARGUMENT;
    if (api_size < sizeof(*api)) return IBRH_ERROR_STRUCT_TOO_SMALL;
    if ((requested_api_version >> 16u) != IBRH_API_VERSION_MAJOR)
        return IBRH_ERROR_UNSUPPORTED_API;
    *api = {};
    api->struct_size = sizeof(*api);
    api->api_version = IBRH_CURRENT_API_VERSION;
    api->query_capabilities = query_capabilities;
    api->runtime_create = runtime_create;
    api->runtime_destroy = runtime_destroy;
    api->model_load = model_load;
    api->model_unload = model_unload;
    api->submit = submit;
    api->job_poll = job_poll;
    api->job_cancel = job_cancel;
    api->job_release = job_release;
    api->output_acquire = output_acquire;
    api->output_release = output_release;
    api->get_last_error = get_last_error;
    return IBRH_OK;
}
