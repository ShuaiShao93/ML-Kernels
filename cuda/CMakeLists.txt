cmake_minimum_required(VERSION 3.10)
project(cuda_kernels LANGUAGES CXX CUDA)

find_package(CUDA REQUIRED)
include_directories("${CUDA_INCLUDE_DIRS}")

add_library(gemm STATIC
    src/gemm.h
    src/gemm.cu
)
set_property(TARGET gemm PROPERTY CUDA_ARCHITECTURES native)

# For source view in ncu
target_compile_options(gemm PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                       --generate-line-info>)

add_executable(gemm_test
    test/gemm_test.cpp
)
target_link_libraries(gemm_test PRIVATE gemm)