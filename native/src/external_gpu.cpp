
#include "external_gpu.h"

#include "gpu_model.h"
#include "marigold_gpu.h"
#include "model_bundle.h"
#include "operators.h"
#include "prompt_cache.h"
#include "vulkan.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <utility>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>
#endif

namespace marigold_native {
namespace {

void inferbridge_shape(
    std::uint32_t width, std::uint32_t height,
    std::uint32_t& processing_width,
    std::uint32_t& processing_height) {
    if (width == 0u || height == 0u)
        throw std::invalid_argument("Marigold image dimensions are empty");
    constexpr std::uint32_t resolution = 768u;
    const double scale = std::min(
        static_cast<double>(resolution) / width,
        static_cast<double>(resolution) / height);
    processing_width = std::max(
        1u, static_cast<std::uint32_t>(width * scale));
    processing_height = std::max(
        1u, static_cast<std::uint32_t>(height * scale));
}

#if defined(_WIN32)
using Microsoft::WRL::ComPtr;
constexpr std::uint32_t kGpuSlotCount = 3u;

void check_hresult(HRESULT result, const char* operation) {
    if (FAILED(result))
        throw std::runtime_error(
            std::string(operation) + " failed with HRESULT " +
            std::to_string(static_cast<long>(result)));
}

ComPtr<ID3D12Device> matching_d3d12_device(std::uint64_t luid) {
    ComPtr<IDXGIFactory6> factory;
    check_hresult(CreateDXGIFactory2(0, IID_PPV_ARGS(&factory)),
                  "CreateDXGIFactory2");
    for (UINT index = 0;; ++index) {
        ComPtr<IDXGIAdapter1> adapter;
        const HRESULT result = factory->EnumAdapters1(index, &adapter);
        if (result == DXGI_ERROR_NOT_FOUND) break;
        check_hresult(result, "EnumAdapters1");
        DXGI_ADAPTER_DESC1 description{};
        check_hresult(adapter->GetDesc1(&description), "GetDesc1");
        std::uint64_t candidate = 0u;
        std::memcpy(&candidate, &description.AdapterLuid, sizeof(candidate));
        if (candidate != luid) continue;
        ComPtr<ID3D12Device> device;
        check_hresult(D3D12CreateDevice(
            adapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device)),
            "D3D12CreateDevice");
        return device;
    }
    return {};
}

struct SharedOutput {
    ComPtr<ID3D12Resource> resource;
    ComPtr<ID3D12Fence> fence;
    HANDLE resource_handle = nullptr;
    HANDLE fence_handle = nullptr;
    ~SharedOutput() {
        if (resource_handle) CloseHandle(resource_handle);
        if (fence_handle) CloseHandle(fence_handle);
    }
    SharedOutput() = default;
    SharedOutput(const SharedOutput&) = delete;
    SharedOutput& operator=(const SharedOutput&) = delete;
    SharedOutput(SharedOutput&& other) noexcept
        : resource(std::move(other.resource)), fence(std::move(other.fence)),
          resource_handle(std::exchange(other.resource_handle, nullptr)),
          fence_handle(std::exchange(other.fence_handle, nullptr)) {}
    SharedOutput& operator=(SharedOutput&& other) noexcept {
        if (this != &other) {
            if (resource_handle) CloseHandle(resource_handle);
            if (fence_handle) CloseHandle(fence_handle);
            resource = std::move(other.resource);
            fence = std::move(other.fence);
            resource_handle = std::exchange(other.resource_handle, nullptr);
            fence_handle = std::exchange(other.fence_handle, nullptr);
        }
        return *this;
    }
};

struct GpuSlot {
    std::atomic<bool> occupied{false};
    SharedOutput shared;
    std::uint32_t width = 0u;
    std::uint32_t height = 0u;
    std::uint64_t fence_value = 0u;
};

SharedOutput create_shared_output(
    ID3D12Device* device, std::uint32_t width, std::uint32_t height) {
    const D3D12_HEAP_PROPERTIES heap{
        D3D12_HEAP_TYPE_DEFAULT, D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
        D3D12_MEMORY_POOL_UNKNOWN, 1, 1};
    const D3D12_RESOURCE_DESC description{
        D3D12_RESOURCE_DIMENSION_TEXTURE2D, 0, width, height, 1, 1,
        DXGI_FORMAT_R32_FLOAT, {1, 0}, D3D12_TEXTURE_LAYOUT_UNKNOWN,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS};
    SharedOutput output;
    check_hresult(device->CreateCommittedResource(
        &heap, D3D12_HEAP_FLAG_SHARED, &description,
        D3D12_RESOURCE_STATE_COMMON, nullptr,
        IID_PPV_ARGS(&output.resource)), "CreateCommittedResource(Marigold output)");
    check_hresult(device->CreateFence(
        0, D3D12_FENCE_FLAG_SHARED, IID_PPV_ARGS(&output.fence)),
        "CreateFence(Marigold output)");
    check_hresult(device->CreateSharedHandle(
        output.resource.Get(), nullptr, GENERIC_ALL, nullptr,
        &output.resource_handle), "CreateSharedHandle(Marigold output)");
    check_hresult(device->CreateSharedHandle(
        output.fence.Get(), nullptr, GENERIC_ALL, nullptr,
        &output.fence_handle), "CreateSharedHandle(Marigold fence)");
    return output;
}

void validate_input(
    ID3D12Device* device, const ExternalTextureRequest& request) {
    ComPtr<ID3D12Resource> resource;
    check_hresult(device->OpenSharedHandle(
        reinterpret_cast<HANDLE>(request.shared_texture_handle),
        IID_PPV_ARGS(&resource)), "OpenSharedHandle(Marigold input)");
    const D3D12_RESOURCE_DESC description = resource->GetDesc();
    const DXGI_FORMAT expected = request.rgba ?
        DXGI_FORMAT_R8G8B8A8_UNORM : DXGI_FORMAT_B8G8R8A8_UNORM;
    if (description.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D ||
        description.Width != request.width ||
        description.Height != request.height ||
        description.DepthOrArraySize != 1u || description.MipLevels != 1u ||
        description.SampleDesc.Count != 1u || description.Format != expected)
        throw std::invalid_argument("shared Marigold input texture is invalid");
}

std::shared_ptr<GpuSlot> acquire_slot(
    const std::array<std::shared_ptr<GpuSlot>, kGpuSlotCount>& slots,
    std::atomic<std::uint32_t>& next) {
    const std::uint32_t first = next.fetch_add(1u) % kGpuSlotCount;
    for (std::uint32_t offset = 0u; offset < kGpuSlotCount; ++offset) {
        const auto& slot = slots[(first + offset) % kGpuSlotCount];
        bool expected = false;
        if (slot->occupied.compare_exchange_strong(expected, true)) return slot;
    }
    throw std::runtime_error("all Marigold GPU output slots are occupied");
}

VulkanImage prepare_output(
    GpuSlot& slot, ID3D12Device* device, VulkanContext& context,
    std::uint32_t width, std::uint32_t height) {
    if (!slot.shared.resource || slot.width != width || slot.height != height) {
        slot.shared = SharedOutput{};
        slot.shared = create_shared_output(device, width, height);
        slot.width = width;
        slot.height = height;
        slot.fence_value = 0u;
    }
    return context.import_d3d12_image(
        slot.shared.resource_handle, width, height, VK_FORMAT_R32_SFLOAT,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
}

class ExternalJobImpl final : public ExternalJob {
public:
    ExternalJobImpl(
        std::shared_ptr<ExternalGpu> owner, std::shared_ptr<GpuSlot> slot,
        VulkanImage input, VulkanImage output, VulkanSubmission submission,
        ExternalTextureRequest request, std::uint32_t output_width,
        std::uint32_t output_height, std::uint64_t fence_value)
        : owner_(std::move(owner)), slot_(std::move(slot)),
          input_(std::move(input)), output_(std::move(output)),
          submission_(std::move(submission)), request_(request),
          output_width_(output_width), output_height_(output_height),
          fence_value_(fence_value) {}
    ~ExternalJobImpl() override {
        try { submission_.wait(); } catch (...) {}
        submission_ = {};
        output_ = {};
        input_ = {};
        slot_->occupied.store(false);
    }
    ExternalJobState state() const override {
        if (cancelled_.load()) return ExternalJobState::cancelled;
        if (complete_.load()) return ExternalJobState::complete;
        if (!submission_.ready()) return ExternalJobState::running;
        complete_.store(true);
        return ExternalJobState::complete;
    }
    void cancel() override { cancelled_.store(true); }
    ExternalTextureOutput output() const override {
        if (cancelled_.load()) throw std::runtime_error("Marigold job cancelled");
        return {
            reinterpret_cast<std::uintptr_t>(slot_->shared.resource_handle),
            output_width_, output_height_,
            reinterpret_cast<std::uintptr_t>(slot_->shared.fence_handle),
            fence_value_, request_.source_frame_id, request_.timestamp_ns};
    }
private:
    std::shared_ptr<ExternalGpu> owner_;
    std::shared_ptr<GpuSlot> slot_;
    VulkanImage input_;
    VulkanImage output_;
    mutable VulkanSubmission submission_;
    ExternalTextureRequest request_{};
    std::uint32_t output_width_ = 0u;
    std::uint32_t output_height_ = 0u;
    std::uint64_t fence_value_ = 0u;
    std::atomic<bool> cancelled_{false};
    mutable std::atomic<bool> complete_{false};
};
#endif

class ExternalGpuImpl final : public ExternalGpu {
public:
    ExternalGpuImpl(
        const std::string& root, const std::string& vae_model,
        const std::string& prompt_cache, bool full_v1,
        std::uint32_t device_index)
        : model_(root, vae_model, false), context_(device_index),
          unet_(model_.unet(), context_), vae_(model_.vae(), context_),
          operators_(context_), prompt_(load_empty_prompt_cache(
              prompt_cache, static_cast<std::uint32_t>(
                  model_.unet().tensor("conv_in.weight").dimensions[1]))),
          graph_(context_, unet_, vae_, operators_, prompt_, full_v1)
#if defined(_WIN32)
          , d3d12_(matching_d3d12_device(context_.adapter_luid())),
          slots_{std::make_shared<GpuSlot>(), std::make_shared<GpuSlot>(),
                 std::make_shared<GpuSlot>()}
#endif
          {}

    ExternalGpuCapabilities capabilities() const override {
#if defined(_WIN32)
        const auto& value = context_.external_capabilities();
        const bool available = d3d12_ && value.d3d12_resource_import &&
            value.d3d12_fence_import &&
            value.d3d12_bgra8_sampled_image_import &&
            value.d3d12_rgba8_sampled_image_import &&
            value.d3d12_r32_storage_image_import;
        return {available, available ? context_.adapter_luid() : 0u,
                available ? kGpuSlotCount : 0u};
#else
        return {};
#endif
    }

    std::shared_ptr<ExternalJob> submit_texture(
        const ExternalTextureRequest& request) override {
#if !defined(_WIN32)
        (void)request;
        throw std::runtime_error("Marigold D3D12 interop is unavailable");
#else
        if (!capabilities().available)
            throw std::runtime_error("Marigold external GPU path is unavailable");
        if (!request.shared_texture_handle || !request.wait_fence_handle ||
            !request.width || !request.height)
            throw std::invalid_argument("invalid Marigold external GPU request");
        validate_input(d3d12_.Get(), request);
        auto slot = acquire_slot(slots_, next_slot_);
        try {
            std::lock_guard<std::mutex> lock(record_mutex_);
            std::uint32_t processing_width = 0u;
            std::uint32_t processing_height = 0u;
            inferbridge_shape(request.width, request.height,
                              processing_width, processing_height);
            VulkanImage output = prepare_output(
                *slot, d3d12_.Get(), context_, processing_width,
                processing_height);
            VulkanImage input = context_.import_d3d12_image(
                reinterpret_cast<void*>(request.shared_texture_handle),
                request.width, request.height,
                request.rgba ? VK_FORMAT_R8G8B8A8_UNORM :
                               VK_FORMAT_B8G8R8A8_UNORM,
                VK_IMAGE_USAGE_SAMPLED_BIT |
                    VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT);
            const std::uint64_t signal_value = ++slot->fence_value;
            VulkanSemaphore wait = context_.import_d3d12_fence(
                reinterpret_cast<void*>(request.wait_fence_handle),
                request.wait_fence_value);
            VulkanSemaphore signal = context_.import_d3d12_fence(
                slot->shared.fence_handle, signal_value);
            VulkanSubmission submission = context_.batch_async(
                std::move(wait), std::move(signal), [&] {
                    context_.acquire_external_image(
                        input, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                        VK_ACCESS_SHADER_READ_BIT);
                    context_.acquire_external_image(
                        output, VK_IMAGE_LAYOUT_GENERAL,
                        VK_ACCESS_SHADER_WRITE_BIT);
                    VulkanBuffer rgb = context_.create_device_buffer(
                        std::uint64_t(processing_width) * processing_height *
                        3 * sizeof(float));
                    operators_.preprocess_texture(
                        rgb, input, request.width, request.height,
                        processing_width, processing_height);
                    const std::uint32_t noise_count =
                        4u * (processing_width / 8u) *
                        (processing_height / 8u);
                    VulkanBuffer target = context_.create_device_buffer(
                        std::uint64_t(noise_count) * sizeof(float));
                    operators_.seeded_noise(target, noise_count, request.seed);
                    VulkanBuffer depth = graph_.infer_device(
                        std::move(rgb), processing_width, processing_height,
                        std::move(target));
                    operators_.normalize_depth(
                        depth, processing_width * processing_height);
                    operators_.depth_to_image(
                        output, depth, processing_width, processing_height);
                    context_.release_external_image(
                        input, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                        VK_ACCESS_SHADER_READ_BIT);
                    context_.release_external_image(
                        output, VK_IMAGE_LAYOUT_GENERAL,
                        VK_ACCESS_SHADER_WRITE_BIT);
                });
            return std::make_shared<ExternalJobImpl>(
                shared_from_this(), slot, std::move(input), std::move(output),
                std::move(submission), request, processing_width,
                processing_height, signal_value);
        } catch (...) {
            slot->occupied.store(false);
            throw;
        }
#endif
    }

    void transfer_counters(
        std::uint64_t& upload_bytes,
        std::uint64_t& download_bytes) const override {
        context_.transfer_counters(upload_bytes, download_bytes);
    }

private:
    ModelBundle model_;
    VulkanContext context_;
    GpuModel unet_;
    GpuModel vae_;
    VulkanOperators operators_;
    TokenTensor prompt_;
    MarigoldGpuGraph graph_;
#if defined(_WIN32)
    ComPtr<ID3D12Device> d3d12_;
    std::array<std::shared_ptr<GpuSlot>, kGpuSlotCount> slots_;
    std::atomic<std::uint32_t> next_slot_{0u};
    std::mutex record_mutex_;
#endif
};

}  // namespace

std::shared_ptr<ExternalGpu> create_external_gpu(
    const std::string& snapshot_root, const std::string& vae_model,
    const std::string& prompt_cache, bool full_v1,
    std::uint32_t device_index) {
    return std::make_shared<ExternalGpuImpl>(
        snapshot_root, vae_model, prompt_cache, full_v1, device_index);
}

ExternalGpuCapabilities probe_external_gpu(std::uint32_t device_index) {
#if defined(_WIN32)
    VulkanContext context(device_index);
    auto device = matching_d3d12_device(context.adapter_luid());
    const auto& value = context.external_capabilities();
    const bool available = device && value.d3d12_resource_import &&
        value.d3d12_fence_import &&
        value.d3d12_bgra8_sampled_image_import &&
        value.d3d12_rgba8_sampled_image_import &&
        value.d3d12_r32_storage_image_import;
    return {available, available ? context.adapter_luid() : 0u,
            available ? kGpuSlotCount : 0u};
#else
    (void)device_index;
    return {};
#endif
}

}  // namespace marigold_native

