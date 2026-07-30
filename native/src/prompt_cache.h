#pragma once

#include "transformer_cpu.h"

#include <string>

namespace marigold_native {

TokenTensor load_empty_prompt_cache(
    const std::string& path_utf8,
    bool full_v1);

}  // namespace marigold_native
