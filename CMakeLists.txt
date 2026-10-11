cmake_minimum_required(VERSION 3.20)
project(sparse_weight_compression LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

add_executable(sparse_compress
    src/main.cpp
    src/safetensors.cpp
    src/sparsify.cu
)

target_include_directories(sparse_compress PRIVATE include)
target_link_libraries(sparse_compress PRIVATE CUDA::cudart)

set_target_properties(sparse_compress PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_ARCHITECTURES 75;80;86;89
)
