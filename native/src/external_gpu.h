
#pragma once

#include <cstdint>
#include <memory>
#include <string>

namespace marigold_native {

struct ExternalGpuCapabilities {
    bool available = false;
    std::uint64_t adapter_luid = 0u;
    std::uint32_t maximum_in_flight_jobs = 0u;
};

struct ExternalTextureRequest {
    std::uintptr_t shared_texture_handle = 0u;
    std::uint32_t width = 0u;
    std::uint32_t height = 0u;
    bool rgba = false;
    std::uintptr_t wait_fence_handle = 0u;
    std::uint64_t wait_fence_value = 0u;
    std::uint64_t seed = 0u;
    std::uint64_t source_frame_id = 0u;
    std::uint64_t timestamp_ns = 0u;
};

struct ExternalTextureOutput {
    std::uintptr_t shared_texture_handle = 0u;
    std::uint32_t width = 0u;
    std::uint32_t height = 0u;
    std::uintptr_t ready_fence_handle = 0u;
    std::uint64_t ready_fence_value = 0u;
    std::uint64_t source_frame_id = 0u;
    std::uint64_t timestamp_ns = 0u;
};

enum class ExternalJobState { running, complete, cancelled };

class ExternalJob {
public:
    virtual ~ExternalJob() = default;
    virtual ExternalJobState state() const = 0;
    virtual void cancel() = 0;
    virtual ExternalTextureOutput output() const = 0;
};

class ExternalGpu : public std::enable_shared_from_this<ExternalGpu> {
public:
    virtual ~ExternalGpu() = default;
    virtual ExternalGpuCapabilities capabilities() const = 0;
    virtual std::shared_ptr<ExternalJob> submit_texture(
        const ExternalTextureRequest& request) = 0;
    virtual void transfer_counters(
        std::uint64_t& upload_bytes,
        std::uint64_t& download_bytes) const = 0;
};

std::shared_ptr<ExternalGpu> create_external_gpu(
    const std::string& snapshot_root,
    const std::string& vae_model,
    const std::string& prompt_cache,
    bool full_v1, std::uint32_t device_index);
ExternalGpuCapabilities probe_external_gpu(std::uint32_t device_index);

}  // namespace marigold_native

