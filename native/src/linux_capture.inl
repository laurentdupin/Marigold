#if defined(__linux__) && !defined(__ANDROID__) && defined(MARIGOLD_WITH_VULKAN)
#include "linux_capture.h"
#include <inferbridge/linux_capture_preprocess.h>
ibr_linux_capture_capabilities
marigold_linux_capture_capabilities(marigold_context *context) {
  return context->vulkan ? context->vulkan->linux_capture_capabilities()
                         : ibr_linux_capture_capabilities{};
}
void marigold_infer_linux_capture(
    marigold_context *context,
    const inferbridge::linux_capture::LinuxDmaBufImage &source, uint64_t seed,
    float *output) {
  if (!context->vulkan)
    throw std::runtime_error("marigold Vulkan context unavailable");
  auto &vk = *context->vulkan;
  uint32_t width = 0, height = 0;
  inferbridge_shape(source.width, source.height, width, height);
  auto input = inferbridge::linux_capture::capture_tensor(
      vk, source, width, height,
      {3, true, {.5f, .5f, .5f, 0}, {.5f, .5f, .5f, 1}});
  const uint64_t count = uint64_t(4) * (width / 8) * (height / 8);
  std::mt19937_64 generator(seed);
  std::normal_distribution<float> normal;
  const auto noise = [&] {
    std::vector<float> values(count);
    for (auto &value : values)
      value = normal(generator);
    auto buffer = vk.create_device_buffer(count * sizeof(float));
    vk.upload(buffer, values.data(), count * sizeof(float));
    return buffer;
  };
  auto first = noise();
  marigold_native::MarigoldGpuGraph graph(
      vk, *context->gpu_unet, *context->gpu_vae, *context->operators,
      context->prompt, context->variant == MARIGOLD_MODEL_FULL_V1);
  auto result =
      graph.infer_device(std::move(input), width, height, std::move(first));
  vk.download(result, output, uint64_t(width) * height * sizeof(float));
  inferbridge_normalize(output, width, height);
}
#endif
