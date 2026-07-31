
#include "inferbridge_harness.h"

#include "marigold_native.h"
#if defined(MARIGOLD_WITH_VULKAN)
#include "external_gpu.h"
#endif

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <deque>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif

class MarigoldGpuWorker;
struct MarigoldGpuAdmission;

struct ibrh_runtime {
    std::string error;
    int32_t vulkan_device_index = 0;
    uint64_t adapter_luid = 0u;
};

struct ibrh_model {
    ibrh_runtime* runtime = nullptr;
    marigold_context* context = nullptr;
    std::string model_path;
    std::string prompt_cache;
#if defined(MARIGOLD_WITH_VULKAN) && defined(_WIN32)
    std::shared_ptr<marigold_native::ExternalGpu> external_gpu;
    std::shared_ptr<MarigoldGpuWorker> gpu_worker;
    std::shared_ptr<std::atomic<uint32_t>> gpu_admissions =
        std::make_shared<std::atomic<uint32_t>>(0u);
#endif
    std::atomic<uint64_t> next_seed{1u};
    std::mutex submit_mutex;
};

struct ibrh_job {
    std::atomic<uint32_t> references{1u};
    uint64_t source_frame_id = 0u;
    uint64_t timestamp_ns = 0u;
    uint32_t width = 0u;
    uint32_t height = 0u;
    std::vector<float> depth;
#if defined(MARIGOLD_WITH_VULKAN) && defined(_WIN32)
    mutable std::mutex gpu_mutex;
    std::shared_ptr<marigold_native::ExternalJob> gpu_job;
    std::shared_ptr<MarigoldGpuAdmission> gpu_admission;
    std::weak_ptr<MarigoldGpuWorker> gpu_worker;
    std::atomic<uint32_t> gpu_state{IBRH_JOB_COMPLETE};
    std::atomic<bool> cancel_requested{false};
    std::string gpu_error;
    uintptr_t input_texture_handle = 0u;
    uintptr_t input_fence_handle = 0u;
    uint64_t input_fence_value = 0u;
    uint64_t seed = 0u;
    bool rgba = false;
    ~ibrh_job() {
        gpu_job.reset();
        gpu_admission.reset();
        if (input_texture_handle)
            CloseHandle(reinterpret_cast<HANDLE>(input_texture_handle));
        if (input_fence_handle)
            CloseHandle(reinterpret_cast<HANDLE>(input_fence_handle));
    }
#endif
};

struct ibrh_output_lease {
    ibrh_job* job = nullptr;
#if defined(MARIGOLD_WITH_VULKAN) && defined(_WIN32)
    std::shared_ptr<marigold_native::ExternalJob> gpu_job;
    std::shared_ptr<MarigoldGpuAdmission> gpu_admission;
#endif
};

namespace {

thread_local std::string g_last_error;
constexpr char kHarnessId[] = "inferbridge.marigold.native";
constexpr char kHarnessVersion[] = "1.1.0";

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

bool parse_luid(const std::string& value, uint64_t& result) {
    if (value.size() != 16u) return false;
    const auto nibble = [](char c) -> int {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'a' && c <= 'f') return c - 'a' + 10;
        if (c >= 'A' && c <= 'F') return c - 'A' + 10;
        return -1;
    };
    uint8_t bytes[8]{};
    for (size_t i = 0; i < 8u; ++i) {
        const int high = nibble(value[i * 2u]);
        const int low = nibble(value[i * 2u + 1u]);
        if (high < 0 || low < 0) return false;
        bytes[i] = static_cast<uint8_t>((high << 4) | low);
    }
    std::memcpy(&result, bytes, sizeof(result));
    return true;
}

bool device_index_for_luid(uint64_t luid, int32_t& device_index) {
#if defined(MARIGOLD_WITH_VULKAN) && defined(_WIN32)
    for (int32_t index = 0; index < 32; ++index) {
        try {
            const auto capabilities = marigold_native::probe_external_gpu(index);
            if (capabilities.available && capabilities.adapter_luid == luid) {
                device_index = index;
                return true;
            }
        } catch (...) {
            if (index == 0) return false;
            break;
        }
    }
#else
    (void)luid;
    (void)device_index;
#endif
    return false;
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

}  // namespace

#if defined(MARIGOLD_WITH_VULKAN) && defined(_WIN32)
void close_gpu_input_handles(ibrh_job& job) noexcept {
    const auto texture = reinterpret_cast<HANDLE>(
        std::exchange(job.input_texture_handle, 0u));
    const auto fence = reinterpret_cast<HANDLE>(
        std::exchange(job.input_fence_handle, 0u));
    if (texture) CloseHandle(texture);
    if (fence) CloseHandle(fence);
}

struct MarigoldGpuAdmission {
    explicit MarigoldGpuAdmission(std::shared_ptr<std::atomic<uint32_t>> value)
        : count(std::move(value)) {}
    ~MarigoldGpuAdmission() { count->fetch_sub(1u); }
    std::shared_ptr<std::atomic<uint32_t>> count;
};

class MarigoldGpuWorker {
public:
    explicit MarigoldGpuWorker(std::shared_ptr<marigold_native::ExternalGpu> gpu)
        : gpu_(std::move(gpu)), thread_([this] { run(); }) {}
    ~MarigoldGpuWorker() { stop(); }
    void enqueue(ibrh_job* job) {
        retain_job(job);
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (stopping_) {
                release_job(job);
                throw std::runtime_error("Marigold GPU worker is stopping");
            }
            queue_.push_back(job);
        }
        condition_.notify_one();
    }
    bool cancel_queued(ibrh_job* job) noexcept {
        bool removed = false;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            const auto found = std::find(queue_.begin(), queue_.end(), job);
            if (found != queue_.end()) {
                queue_.erase(found);
                removed = true;
            }
        }
        if (removed) {
            job->gpu_state.store(IBRH_JOB_CANCELLED);
            close_gpu_input_handles(*job);
            release_job(job);
        }
        return removed;
    }
    void stop() noexcept {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopping_ = true;
        }
        condition_.notify_all();
        if (thread_.joinable()) thread_.join();
    }
private:
    void run() noexcept {
        for (;;) {
            ibrh_job* job = nullptr;
            bool stopping = false;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                condition_.wait(lock, [&] { return stopping_ || !queue_.empty(); });
                if (queue_.empty()) {
                    if (stopping_) return;
                    continue;
                }
                job = queue_.front();
                queue_.pop_front();
                stopping = stopping_;
            }
            if (stopping || job->cancel_requested.load()) {
                job->gpu_state.store(IBRH_JOB_CANCELLED);
                close_gpu_input_handles(*job);
                release_job(job);
                continue;
            }
            try {
                auto native = gpu_->submit_texture({
                    job->input_texture_handle, job->width, job->height,
                    job->rgba, job->input_fence_handle,
                    job->input_fence_value, job->seed,
                    job->source_frame_id, job->timestamp_ns});
                close_gpu_input_handles(*job);
                if (job->cancel_requested.load()) native->cancel();
                {
                    std::lock_guard<std::mutex> lock(job->gpu_mutex);
                    job->gpu_job = std::move(native);
                }
                job->gpu_state.store(job->cancel_requested.load() ?
                    IBRH_JOB_CANCELLED : IBRH_JOB_RUNNING);
            } catch (const std::exception& error) {
                close_gpu_input_handles(*job);
                {
                    std::lock_guard<std::mutex> lock(job->gpu_mutex);
                    job->gpu_error = error.what();
                }
                job->gpu_state.store(job->cancel_requested.load() ?
                    IBRH_JOB_CANCELLED : IBRH_JOB_FAILED);
            }
            release_job(job);
        }
    }
    std::shared_ptr<marigold_native::ExternalGpu> gpu_;
    std::mutex mutex_;
    std::condition_variable condition_;
    std::deque<ibrh_job*> queue_;
    bool stopping_ = false;
    std::thread thread_;
};
#else
struct MarigoldGpuAdmission {};
#endif

namespace {

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
#if defined(MARIGOLD_WITH_VULKAN) && defined(_WIN32)
    try {
        if (marigold_native::probe_external_gpu(0u).available) {
            capabilities->flags |= IBRH_CAP_ASYNC_SUBMIT |
                IBRH_CAP_CANCELLATION | IBRH_CAP_GPU_RESOURCES |
                IBRH_CAP_EXTERNAL_SYNCHRONIZATION |
                IBRH_CAP_GPU_RESIDENT_OUTPUT;
            capabilities->input_domain_mask |=
                1ull << IBRH_RESOURCE_DOMAIN_D3D12;
            capabilities->output_domain_mask |=
                1ull << IBRH_RESOURCE_DOMAIN_D3D12;
            capabilities->synchronization_mask =
                1ull << IBRH_SYNC_D3D12_FENCE;
            capabilities->maximum_in_flight_jobs = 3u;
        }
    } catch (...) {}
#endif
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
    if (json_string(device, "luid", luid_text) && !luid_text.empty()) {
        uint64_t luid = 0u;
        if (!parse_luid(luid_text, luid) ||
            !device_index_for_luid(luid, runtime->vulkan_device_index)) {
            delete runtime;
            return fail(nullptr, IBRH_ERROR_UNSUPPORTED_CAPABILITY,
                        "Marigold could not match the requested GPU LUID");
        }
        runtime->adapter_luid = luid;
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
    if (!json_string(parameters, "VaeModel", vae_model) || vae_model.empty() ||
        !json_string(parameters, "PromptCache", prompt_cache) ||
        prompt_cache.empty()) {
        return fail(
            runtime, IBRH_ERROR_INVALID_ARGUMENT,
            "Marigold VaeModel and PromptCache paths are required");
    }
    auto* model = new (std::nothrow) ibrh_model();
    if (model == nullptr) return IBRH_ERROR_INTERNAL;
    model->runtime = runtime;
    model->model_path = path;
    model->prompt_cache = prompt_cache;
#if defined(MARIGOLD_WITH_VULKAN) && defined(_WIN32)
    if (runtime->adapter_luid != 0u) {
        try {
            model->external_gpu = marigold_native::create_external_gpu(
                path, vae_model, prompt_cache,
                model_variant(parameters) == MARIGOLD_MODEL_FULL_V1,
                static_cast<uint32_t>(runtime->vulkan_device_index));
            const auto capabilities = model->external_gpu->capabilities();
            if (!capabilities.available ||
                capabilities.adapter_luid != runtime->adapter_luid)
                throw std::runtime_error(
                    "Marigold loaded on a GPU other than the requested LUID");
            model->gpu_worker = std::make_shared<MarigoldGpuWorker>(
                model->external_gpu);
        } catch (const std::exception& error) {
            delete model;
            return fail(runtime, IBRH_ERROR_UNSUPPORTED_CAPABILITY,
                        error.what());
        }
    } else
#endif
    {
        const int status = marigold_create_vulkan_variant(
            path.c_str(), vae_model.c_str(), prompt_cache.c_str(),
            model_variant(parameters),
            static_cast<uint32_t>(runtime->vulkan_device_index),
            &model->context);
        if (status != MARIGOLD_OK) {
            const std::string message =
                std::string("Marigold model load failed: ") + marigold_last_error();
            delete model;
            return fail(runtime, status_result(status), message);
        }
    }
    *output = model;
    return IBRH_OK;
}

void IBRH_CALL model_unload(ibrh_model* model) {
    if (model == nullptr) return;
#if defined(MARIGOLD_WITH_VULKAN) && defined(_WIN32)
    if (model->gpu_worker) model->gpu_worker->stop();
    model->gpu_worker.reset();
    model->external_gpu.reset();
#endif
    if (model->context) marigold_destroy(model->context);
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
    const ibrh_resource& input = request->inputs[0];
    if (input.struct_size < sizeof(input))
        return IBRH_ERROR_STRUCT_TOO_SMALL;
    uint64_t seed =
        request->source_frame_id != 0u ? request->source_frame_id :
        request->timestamp_ns != 0u ? request->timestamp_ns :
        model->next_seed.fetch_add(1u);
    const std::string submit_parameters = copy_string(request->parameters_json);
    if (submit_parameters.find("\"Seed\"") != std::string::npos &&
        !json_uint64(submit_parameters, "Seed", seed))
        return fail(model->runtime, IBRH_ERROR_INVALID_ARGUMENT,
                    "Marigold Seed must be an unsigned integer");
#if defined(MARIGOLD_WITH_VULKAN) && defined(_WIN32)
    if (input.domain == IBRH_RESOURCE_DOMAIN_D3D12 &&
        input.kind == IBRH_RESOURCE_KIND_IMAGE_2D &&
        input.native_handle_type == IBRH_NATIVE_HANDLE_WIN32_SHARED) {
        if (!model->gpu_worker)
            return fail(model->runtime, IBRH_ERROR_UNSUPPORTED_CAPABILITY,
                        "Marigold GPU model was not loaded for external input");
        if (input.pixel_format != IBRH_PIXEL_BGRA8 ||
            !input.native_handle || !input.width || !input.height)
            return fail(model->runtime, IBRH_ERROR_INVALID_ARGUMENT,
                        "Marigold external texture descriptor is invalid");
        const ibrh_synchronization* wait = nullptr;
        for (uint32_t i = 0u; i < request->synchronization_count; ++i) {
            const auto& candidate = request->synchronizations[i];
            if (candidate.struct_size < sizeof(candidate))
                return IBRH_ERROR_STRUCT_TOO_SMALL;
            if (candidate.kind == IBRH_SYNC_D3D12_FENCE &&
                candidate.operation == IBRH_SYNC_WAIT &&
                candidate.native_handle_type == IBRH_NATIVE_HANDLE_WIN32_SHARED) {
                if (wait) return IBRH_ERROR_INVALID_ARGUMENT;
                wait = &candidate;
            }
        }
        if (!wait || !wait->native_handle)
            return fail(model->runtime, IBRH_ERROR_INVALID_ARGUMENT,
                        "Marigold external input requires one D3D12 wait fence");
        HANDLE texture_copy = nullptr;
        HANDLE fence_copy = nullptr;
        const HANDLE process = GetCurrentProcess();
        if (!DuplicateHandle(process, reinterpret_cast<HANDLE>(input.native_handle),
                             process, &texture_copy, 0, FALSE,
                             DUPLICATE_SAME_ACCESS) ||
            !DuplicateHandle(process, reinterpret_cast<HANDLE>(wait->native_handle),
                             process, &fence_copy, 0, FALSE,
                             DUPLICATE_SAME_ACCESS)) {
            if (texture_copy) CloseHandle(texture_copy);
            if (fence_copy) CloseHandle(fence_copy);
            return IBRH_ERROR_INVALID_ARGUMENT;
        }
        uint32_t admitted = model->gpu_admissions->load();
        while (admitted < 3u &&
               !model->gpu_admissions->compare_exchange_weak(
                   admitted, admitted + 1u)) {}
        if (admitted >= 3u) {
            CloseHandle(texture_copy);
            CloseHandle(fence_copy);
            return IBRH_ERROR_INVALID_STATE;
        }
        auto* job = new (std::nothrow) ibrh_job();
        if (!job) {
            model->gpu_admissions->fetch_sub(1u);
            CloseHandle(texture_copy);
            CloseHandle(fence_copy);
            return IBRH_ERROR_INTERNAL;
        }
        try {
            job->gpu_admission = std::make_shared<MarigoldGpuAdmission>(
                model->gpu_admissions);
        } catch (...) {
            model->gpu_admissions->fetch_sub(1u);
            delete job;
            CloseHandle(texture_copy);
            CloseHandle(fence_copy);
            return IBRH_ERROR_INTERNAL;
        }
        job->input_texture_handle = reinterpret_cast<uintptr_t>(texture_copy);
        job->input_fence_handle = reinterpret_cast<uintptr_t>(fence_copy);
        job->input_fence_value = wait->value;
        job->source_frame_id = request->source_frame_id;
        job->timestamp_ns = request->timestamp_ns;
        job->width = input.width;
        job->height = input.height;
        job->seed = seed;
        job->rgba = false;
        job->gpu_state.store(IBRH_JOB_QUEUED);
        try {
            job->gpu_worker = model->gpu_worker;
            model->gpu_worker->enqueue(job);
        } catch (...) {
            delete job;
            return IBRH_ERROR_INTERNAL;
        }
        *output = job;
        return IBRH_OK;
    }
#endif
    if (request->synchronization_count != 0u)
        return fail(model->runtime, IBRH_ERROR_UNSUPPORTED_CAPABILITY,
                    "Marigold host harness does not accept external synchronization");
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
    auto* job = new (std::nothrow) ibrh_job();
    if (job == nullptr) return IBRH_ERROR_INTERNAL;
    job->source_frame_id = request->source_frame_id;
    job->timestamp_ns = request->timestamp_ns;
    if (marigold_inferbridge_image_shape(
            input.width, input.height, &job->width, &job->height) != MARIGOLD_OK) {
        delete job;
        return fail(model->runtime, IBRH_ERROR_INVALID_ARGUMENT,
                    marigold_last_error());
    }
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
                std::string("Marigold inference failed: ") + marigold_last_error();
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
#if defined(MARIGOLD_WITH_VULKAN) && defined(_WIN32)
    if (job->gpu_admission) {
        std::shared_ptr<marigold_native::ExternalJob> gpu_job;
        {
            std::lock_guard<std::mutex> lock(job->gpu_mutex);
            gpu_job = job->gpu_job;
        }
        if (!gpu_job) status->state = job->gpu_state.load();
        else switch (gpu_job->state()) {
            case marigold_native::ExternalJobState::running:
                status->state = IBRH_JOB_RUNNING; break;
            case marigold_native::ExternalJobState::complete:
                status->state = IBRH_JOB_COMPLETE; break;
            case marigold_native::ExternalJobState::cancelled:
                status->state = IBRH_JOB_CANCELLED; break;
        }
    } else
#endif
    status->state = IBRH_JOB_COMPLETE;
    status->output_count = 1u;
    status->source_frame_id = job->source_frame_id;
    return IBRH_OK;
}

ibrh_result IBRH_CALL job_cancel(ibrh_job* job) {
    if (!job) return IBRH_ERROR_INVALID_ARGUMENT;
#if defined(MARIGOLD_WITH_VULKAN) && defined(_WIN32)
    job->cancel_requested.store(true);
    if (auto worker = job->gpu_worker.lock();
        worker && worker->cancel_queued(job)) return IBRH_OK;
    std::shared_ptr<marigold_native::ExternalJob> gpu_job;
    {
        std::lock_guard<std::mutex> lock(job->gpu_mutex);
        gpu_job = job->gpu_job;
    }
    if (gpu_job) {
        gpu_job->cancel();
        job->gpu_state.store(IBRH_JOB_CANCELLED);
        return IBRH_OK;
    }
    if (job->gpu_admission) {
        job->gpu_state.store(IBRH_JOB_CANCELLED);
        return IBRH_OK;
    }
#endif
    return IBRH_ERROR_INVALID_STATE;
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
#if defined(MARIGOLD_WITH_VULKAN) && defined(_WIN32)
    if (job->gpu_admission) {
        std::shared_ptr<marigold_native::ExternalJob> gpu_job;
        {
            std::lock_guard<std::mutex> lock(job->gpu_mutex);
            gpu_job = job->gpu_job;
        }
        if (!gpu_job) {
            delete lease;
            const uint32_t state = job->gpu_state.load();
            if (state == IBRH_JOB_CANCELLED) return IBRH_ERROR_CANCELLED;
            if (state == IBRH_JOB_FAILED) return IBRH_ERROR_INTERNAL;
            return IBRH_ERROR_INVALID_STATE;
        }
        marigold_native::ExternalTextureOutput native{};
        try { native = gpu_job->output(); }
        catch (...) { delete lease; return IBRH_ERROR_CANCELLED; }
        lease->gpu_job = std::move(gpu_job);
        lease->gpu_admission = job->gpu_admission;
        *descriptor = {};
        descriptor->struct_size = sizeof(*descriptor);
        descriptor->api_version = IBRH_CURRENT_API_VERSION;
        descriptor->output_index = output_index;
        descriptor->payload_type = IBRH_PIXEL_DEPTH_FLOAT32;
        descriptor->source_frame_id = native.source_frame_id;
        descriptor->timestamp_ns = native.timestamp_ns;
        descriptor->resource.struct_size = sizeof(descriptor->resource);
        descriptor->resource.api_version = IBRH_CURRENT_API_VERSION;
        descriptor->resource.domain = IBRH_RESOURCE_DOMAIN_D3D12;
        descriptor->resource.kind = IBRH_RESOURCE_KIND_IMAGE_2D;
        descriptor->resource.access = IBRH_RESOURCE_ACCESS_READ;
        descriptor->resource.pixel_format = IBRH_PIXEL_DEPTH_FLOAT32;
        descriptor->resource.width = native.width;
        descriptor->resource.height = native.height;
        descriptor->resource.depth = 1u;
        descriptor->resource.row_stride_bytes = native.width * sizeof(float);
        descriptor->resource.byte_size =
            static_cast<uint64_t>(native.width) * native.height * sizeof(float);
        descriptor->resource.native_handle_type = IBRH_NATIVE_HANDLE_WIN32_SHARED;
        descriptor->resource.native_handle = native.shared_texture_handle;
        descriptor->ready.struct_size = sizeof(descriptor->ready);
        descriptor->ready.api_version = IBRH_CURRENT_API_VERSION;
        descriptor->ready.kind = IBRH_SYNC_D3D12_FENCE;
        descriptor->ready.operation = IBRH_SYNC_WAIT;
        descriptor->ready.native_handle_type = IBRH_NATIVE_HANDLE_WIN32_SHARED;
        descriptor->ready.native_handle = native.ready_fence_handle;
        descriptor->ready.value = native.ready_fence_value;
        *output = lease;
        return IBRH_OK;
    }
#endif
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
#if defined(MARIGOLD_WITH_VULKAN) && defined(_WIN32)
    lease->gpu_job.reset();
    lease->gpu_admission.reset();
#endif
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
