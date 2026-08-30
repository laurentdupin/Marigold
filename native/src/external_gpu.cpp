#include "external_gpu.h"

#include "gpu_model.h"
#include "marigold_gpu.h"
#include "model_bundle.h"
#include "operators.h"
#include "prompt_cache.h"
#include "vulkan.h"
#include "inferbridge/native_harness_resource_lifetime.h"
#include "inferbridge/native_harness_resource_cache.h"
#include "inferbridge/native_harness_diffusion_shape.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdio>
#include <cstdlib>
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

bool stage_diagnostics_enabled() {
    const char* value = std::getenv("MARIGOLD_STAGE_DIAGNOSTICS");
    return value != nullptr && value[0] == '1' && value[1] == '\0';
}

void inferbridge_shape(
    std::uint32_t width, std::uint32_t height,
    std::uint32_t& processing_width,
    std::uint32_t& processing_height) {
    inferbridge::native_harness::fit_diffusion_shape(
        width, height, processing_width, processing_height);
}

#if defined(_WIN32)
using Microsoft::WRL::ComPtr;
constexpr std::uint32_t kMaxInFlightJobs = 3u;

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

void validate_output(
    ID3D12Device* device, const ExternalTextureRequest& request) {
    ComPtr<ID3D12Resource> resource;
    check_hresult(device->OpenSharedHandle(
        reinterpret_cast<HANDLE>(request.output_texture_handle),
        IID_PPV_ARGS(&resource)), "OpenSharedHandle(Marigold output)");
    const D3D12_RESOURCE_DESC description = resource->GetDesc();
    if (description.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D ||
        description.Width != request.output_width ||
        description.Height != request.output_height ||
        description.DepthOrArraySize != 1u || description.MipLevels != 1u ||
        description.SampleDesc.Count != 1u ||
        description.Format != DXGI_FORMAT_R32_FLOAT)
        throw std::invalid_argument("shared Marigold output texture is invalid");
}

class ExternalJobImpl final : public ExternalJob {
public:
    ExternalJobImpl(
        std::shared_ptr<ExternalGpu> owner, VulkanSubmission submission,
        inferbridge::native_harness::ResourceLifetimeDomainPtr lifetime)
        : owner_(std::move(owner)), submission_(std::move(submission)),
          lifetime_(std::move(lifetime)) {}
    ~ExternalJobImpl() override {
        inferbridge::native_harness::wait_then_retire(
            lifetime_, submission_, [] {});
    }
    ExternalJobState state() const override {
        if (cancelled_.load()) return ExternalJobState::cancelled;
        if (complete_.load()) return ExternalJobState::complete;
        if (!submission_.ready()) return ExternalJobState::running;
        complete_.store(true);
        return ExternalJobState::complete;
    }
    void cancel() override { cancelled_.store(true); }
private:
    std::shared_ptr<ExternalGpu> owner_;
    mutable VulkanSubmission submission_;
    inferbridge::native_harness::ResourceLifetimeDomainPtr lifetime_;
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
          operators_(context_),
          prompt_(load_empty_prompt_cache(prompt_cache, full_v1)),
          graph_(context_, unet_, vae_, operators_, prompt_, full_v1)
#if defined(_WIN32)
          , d3d12_(matching_d3d12_device(context_.adapter_luid()))
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
                available ? kMaxInFlightJobs : 0u};
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
            !request.output_texture_handle || !request.signal_fence_handle ||
            !request.width || !request.height ||
            !request.output_width || !request.output_height)
            throw std::invalid_argument("invalid Marigold external GPU request");
        try {
            auto lifetime_guard = lifetime_->acquire();
            std::uint32_t processing_width = 0u;
            std::uint32_t processing_height = 0u;
            inferbridge_shape(request.width, request.height,
                              processing_width, processing_height);
            if (request.output_width != processing_width ||
                request.output_height != processing_height)
                throw std::invalid_argument("Marigold output dimensions do not match its plan");
            const VkFormat input_format = request.rgba ?
                VK_FORMAT_R8G8B8A8_UNORM : VK_FORMAT_B8G8R8A8_UNORM;
            const auto input_usage = VK_IMAGE_USAGE_SAMPLED_BIT |
                VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
            const auto output_usage = VK_IMAGE_USAGE_STORAGE_BIT |
                VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
            VulkanImage& input = input_cache_.get_or_create({
                inferbridge::native_harness::stable_resource_identity(
                    request.shared_texture_handle,
                    request.shared_texture_identity), request.width,
                request.height, input_format, input_usage}, [&] {
                    validate_input(d3d12_.Get(), request);
                    return context_.import_d3d12_image(
                        reinterpret_cast<void*>(request.shared_texture_handle),
                        request.width, request.height, input_format,
                        input_usage);
                });
            VulkanImage& output = output_cache_.get_or_create({
                inferbridge::native_harness::stable_resource_identity(
                    request.output_texture_handle,
                    request.output_texture_identity), request.output_width,
                request.output_height, VK_FORMAT_R32_SFLOAT, output_usage}, [&] {
                    validate_output(d3d12_.Get(), request);
                    return context_.import_d3d12_image(
                        reinterpret_cast<void*>(request.output_texture_handle),
                        request.output_width, request.output_height,
                        VK_FORMAT_R32_SFLOAT, output_usage);
                });
            VulkanSemaphore wait = context_.import_d3d12_fence(
                reinterpret_cast<void*>(request.wait_fence_handle),
                request.wait_fence_value);
            VulkanSemaphore signal = context_.import_d3d12_fence(
                reinterpret_cast<void*>(request.signal_fence_handle),
                request.signal_fence_value);
            VulkanBuffer rgb;
            VulkanBuffer target;
            VulkanSubmission preprocessing;
            try {
                preprocessing = context_.batch_async(
                    std::move(wait), {}, [&] {
                    context_.acquire_external_image(
                        input, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                        VK_ACCESS_SHADER_READ_BIT);
                    rgb = context_.create_device_buffer(
                        std::uint64_t(processing_width) * processing_height *
                        3 * sizeof(float));
                    operators_.preprocess_texture(
                        rgb, input, request.width, request.height,
                        processing_width, processing_height);
                    const std::uint32_t noise_count =
                        4u * (processing_width / 8u) *
                        (processing_height / 8u);
                    target = context_.create_device_buffer(
                        std::uint64_t(noise_count) * sizeof(float));
                    operators_.seeded_noise(target, noise_count, request.seed);
                    context_.release_external_image(
                        input, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                        VK_ACCESS_SHADER_READ_BIT);
                    });
                preprocessing.wait();
                preprocessing = {};
                if (stage_diagnostics_enabled())
                    std::fprintf(
                        stderr, "marigold-stage: preprocess/noise complete\n");
            } catch (const std::exception& error) {
                throw std::runtime_error(
                    std::string("Marigold preprocess/noise stage failed: ") +
                    error.what());
            }
            // The graph's existing bounded batches keep the diffusion
            // intermediates device-resident while allowing completed
            // operator groups to recycle memory. This wait occurs only on
            // the persistent harness worker, never in public submit().
            VulkanBuffer depth;
            try {
                depth = graph_.infer_device(
                    std::move(rgb), processing_width, processing_height,
                    std::move(target));
                if (stage_diagnostics_enabled())
                    std::fprintf(
                        stderr, "marigold-stage: bounded graph complete\n");
            } catch (const std::exception& error) {
                throw std::runtime_error(
                    std::string("Marigold bounded diffusion/VAE stage failed: ") +
                    error.what());
            }
            try {
                operators_.normalize_depth(
                    depth, processing_width * processing_height);
                if (stage_diagnostics_enabled())
                    std::fprintf(
                        stderr, "marigold-stage: normalization complete\n");
            } catch (const std::exception& error) {
                throw std::runtime_error(
                    std::string("Marigold normalization stage failed: ") +
                    error.what());
            }
            VulkanSubmission submission;
            try {
                submission = context_.batch_async(
                    {}, std::move(signal), [&] {
                    context_.acquire_external_image(
                        output, VK_IMAGE_LAYOUT_GENERAL,
                        VK_ACCESS_SHADER_WRITE_BIT);
                    operators_.depth_to_image(
                        output, depth, processing_width, processing_height);
                    context_.release_external_image(
                        output, VK_IMAGE_LAYOUT_GENERAL,
                        VK_ACCESS_SHADER_WRITE_BIT);
                    });
                if (stage_diagnostics_enabled())
                    std::fprintf(
                        stderr, "marigold-stage: final output submitted\n");
            } catch (const std::exception& error) {
                throw std::runtime_error(
                    std::string("Marigold final R32 output stage failed: ") +
                    error.what());
            }
            return std::make_shared<ExternalJobImpl>(
                shared_from_this(), std::move(submission), lifetime_);
        } catch (...) {
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
    inferbridge::native_harness::StableResourceCache<VulkanImage>
        input_cache_;
    inferbridge::native_harness::StableResourceCache<VulkanImage>
        output_cache_;
    inferbridge::native_harness::ResourceLifetimeDomainPtr lifetime_ =
        inferbridge::native_harness::make_resource_lifetime_domain();
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
            available ? kMaxInFlightJobs : 0u};
#else
    (void)device_index;
    return {};
#endif
}

}  // namespace marigold_native
