cmake_minimum_required(VERSION 3.21)
project(marigold_native LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(MARIGOLD_WITH_VULKAN
    "Build the dependency-free Vulkan executor" ON)

if(MARIGOLD_WITH_VULKAN)
    find_package(Vulkan REQUIRED COMPONENTS glslc)
    set(MARIGOLD_GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
    file(MAKE_DIRECTORY "${MARIGOLD_GENERATED_DIR}")
    function(marigold_compile_shader NAME)
        set(SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/${NAME}.glsl")
        set(SPIRV "${MARIGOLD_GENERATED_DIR}/${NAME}.spv")
        set(HEADER "${MARIGOLD_GENERATED_DIR}/${NAME}_spv.h")
        add_custom_command(
            OUTPUT "${SPIRV}"
            COMMAND Vulkan::glslc -fshader-stage=compute
                --target-env=vulkan1.3 --target-spv=spv1.6 -Werror
                "-I${CMAKE_CURRENT_SOURCE_DIR}/src/shaders"
                -o "${SPIRV}" "${SOURCE}"
            DEPENDS "${SOURCE}"
                "${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/indexing.h"
                "${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/conv2d_common.glsl"
                "${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/linear_common.glsl"
                "${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/project_tokens_common.glsl"
                "${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/conv_transpose_nonoverlap_common.glsl"
            VERBATIM)
        add_custom_command(
            OUTPUT "${HEADER}"
            COMMAND "${CMAKE_COMMAND}" "-DINPUT=${SPIRV}"
                "-DOUTPUT=${HEADER}" "-DVARIABLE=marigold_${NAME}_spv"
                -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_binary.cmake"
            DEPENDS "${SPIRV}" "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_binary.cmake"
            VERBATIM)
        set(MARIGOLD_SHADER_HEADERS
            ${MARIGOLD_SHADER_HEADERS} "${HEADER}" PARENT_SCOPE)
    endfunction()
    foreach(SHADER
            add_scaled add add_position bilinear_align_true
            bilinear_align_true_image bmm bmm_score_half bmm_value_half
            conv2d conv2d8 conv2d_half conv2d8_half
            conv_transpose_nonoverlap conv_transpose_nonoverlap_half
            elementwise gelu layer_norm linear linear16 linear_half
            linear16_half prepare_tokens position_bicubic project_tokens
            project_tokens_half relu softmax_lastdim softmax_lastdim_half
            group_norm silu nearest concatenate add_channel nchw_tokens
            geglu attention_scores attention_values preprocess_rgb
            posterior_sample scale_values depth_output scheduler_target
            ddim_step)
        marigold_compile_shader(${SHADER})
    endforeach()
endif()

add_library(marigold_model STATIC
    src/model_bundle.cpp
    src/prompt_cache.cpp
    src/safetensors.cpp
    src/tensor_cpu.cpp
    src/transformer_cpu.cpp
    src/unet_cpu.cpp
    src/vae_cpu.cpp
)
if(MARIGOLD_WITH_VULKAN)
    target_sources(marigold_model PRIVATE
        src/vulkan.cpp src/operators.cpp src/gpu_model.cpp
        src/marigold_gpu.cpp
        ${MARIGOLD_SHADER_HEADERS})
    target_include_directories(
        marigold_model PRIVATE "${MARIGOLD_GENERATED_DIR}")
    target_link_libraries(marigold_model PUBLIC Vulkan::Vulkan)
    target_compile_definitions(
        marigold_model PUBLIC MARIGOLD_WITH_VULKAN=1)
endif()
target_include_directories(marigold_model PUBLIC src)
set_target_properties(marigold_model PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(marigold_native SHARED
    src/marigold.cpp
    src/inferbridge_harness.cpp)
target_compile_definitions(
    marigold_native PRIVATE
        MARIGOLD_NATIVE_BUILD
        INFERBRIDGE_HARNESS_BUILDING_LIBRARY)
target_include_directories(marigold_native PUBLIC include)
target_link_libraries(marigold_native PRIVATE marigold_model)
set_target_properties(
    marigold_native PROPERTIES
    OUTPUT_NAME "marigold_native"
    WINDOWS_EXPORT_ALL_SYMBOLS OFF)

add_executable(marigold_model_probe tools/model_probe.cpp)
target_link_libraries(marigold_model_probe PRIVATE marigold_model)

add_executable(marigold_vae_probe tools/vae_probe.cpp)
target_link_libraries(marigold_vae_probe PRIVATE marigold_model)

add_executable(marigold_unet_probe tools/unet_probe.cpp)
target_link_libraries(marigold_unet_probe PRIVATE marigold_model)

add_executable(marigold_full_graph_probe tools/full_graph_probe.cpp)
target_include_directories(marigold_full_graph_probe PRIVATE include)
target_link_libraries(marigold_full_graph_probe PRIVATE marigold_native)

add_executable(marigold_size_probe tools/size_probe.cpp)
target_include_directories(marigold_size_probe PRIVATE include)
target_link_libraries(marigold_size_probe PRIVATE marigold_native)

add_executable(marigold_c_abi_smoke tests/c_abi_smoke.c)
target_include_directories(marigold_c_abi_smoke PRIVATE include)
target_link_libraries(marigold_c_abi_smoke PRIVATE marigold_native)
add_executable(
    marigold_harness_abi_smoke tests/harness_abi_smoke.c)
target_include_directories(
    marigold_harness_abi_smoke PRIVATE include)
target_link_libraries(
    marigold_harness_abi_smoke PRIVATE marigold_native)
add_executable(
    marigold_harness_full_graph tests/harness_full_graph.cpp)
target_include_directories(
    marigold_harness_full_graph PRIVATE include)
target_link_libraries(
    marigold_harness_full_graph PRIVATE marigold_native)

enable_testing()
add_test(NAME marigold_model_probe_usage COMMAND marigold_model_probe)
set_tests_properties(marigold_model_probe_usage PROPERTIES WILL_FAIL TRUE)
add_test(NAME marigold_c_abi_smoke COMMAND marigold_c_abi_smoke)
add_test(
    NAME marigold_harness_abi_smoke
    COMMAND marigold_harness_abi_smoke)
add_test(
    NAME marigold_harness_full_graph
    COMMAND marigold_harness_full_graph)
set_tests_properties(
    marigold_harness_full_graph PROPERTIES SKIP_RETURN_CODE 77)
