#pragma once
#if defined(__linux__) && !defined(__ANDROID__)
#include <inferbridge/linux_capture_vulkan.h>
struct marigold_context;
ibr_linux_capture_capabilities
marigold_linux_capture_capabilities(marigold_context *);
void marigold_infer_linux_capture(
    marigold_context *, const inferbridge::linux_capture::LinuxDmaBufImage &,
    uint64_t, float *);
#endif
