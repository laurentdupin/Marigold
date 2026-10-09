#pragma once

#if defined(_WIN32)
#  if !defined(WIN32_LEAN_AND_MEAN)
#    define WIN32_LEAN_AND_MEAN
#  endif
#  if !defined(NOMINMAX)
#    define NOMINMAX
#  endif
#  if !defined(VK_USE_PLATFORM_WIN32_KHR)
#    define VK_USE_PLATFORM_WIN32_KHR
#  endif
#endif
#include <vulkan/vulkan.h>
#if defined(__linux__) && !defined(__ANDROID__)
#include <inferbridge/linux_capture_vulkan.h>
#include <memory>
#endif

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace marigold_native {

void global_transfer_counters(
    std::uint64_t& upload_bytes, std::uint64_t& download_bytes);

class VulkanContext;
class VulkanPipeline;
class VulkanBuffer;
class VulkanImage;

#include <inferbridge/vulkan/declarations/vulkan_deferred_buffer.h>

#include <inferbridge/vulkan/declarations/vulkan_batched_descriptor.h>

#include <inferbridge/vulkan/declarations/vulkan_dispatch_resource.h>

#include <inferbridge/vulkan/declarations/vulkan_external_capabilities_variant_1.h>

#include <inferbridge/vulkan/declarations/vulkan_semaphore.h>

#include <inferbridge/vulkan/declarations/vulkan_submission.h>

#include <inferbridge/vulkan/declarations/vulkan_buffer.h>

#include <inferbridge/vulkan/declarations/vulkan_image.h>

#include <inferbridge/vulkan/declarations/vulkan_pipeline.h>

#include <inferbridge/vulkan/declarations/vulkan_context_variant_1.h>

}  // namespace marigold_native
