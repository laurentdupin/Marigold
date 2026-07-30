cmake_minimum_required(VERSION 3.21)
project(marigold_native LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(marigold_model STATIC
    src/model_bundle.cpp
    src/prompt_cache.cpp
    src/safetensors.cpp
    src/tensor_cpu.cpp
    src/transformer_cpu.cpp
    src/unet_cpu.cpp
    src/vae_cpu.cpp
)
target_include_directories(marigold_model PUBLIC src)
set_target_properties(
    marigold_model PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(marigold_native SHARED src/marigold.cpp)
target_compile_definitions(
    marigold_native PRIVATE MARIGOLD_NATIVE_BUILD)
target_include_directories(marigold_native PUBLIC include)
target_link_libraries(marigold_native PRIVATE marigold_model)
set_target_properties(
    marigold_native PROPERTIES OUTPUT_NAME "marigold_native")

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

enable_testing()
add_test(NAME marigold_model_probe_usage COMMAND marigold_model_probe)
set_tests_properties(marigold_model_probe_usage PROPERTIES WILL_FAIL TRUE)
add_test(NAME marigold_c_abi_smoke COMMAND marigold_c_abi_smoke)
