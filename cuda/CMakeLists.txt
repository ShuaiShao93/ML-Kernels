cmake_minimum_required(VERSION 3.10)
project(cuda_kernels LANGUAGES CXX CUDA)

find_package(CUDA REQUIRED)
include_directories("${CUDA_INCLUDE_DIRS}")

set(CUDNN_LIBRARY_PATH "/usr/lib/x86_64-linux-gnu/libcudnn.so.8")

add_library(gemm STATIC
    src/gemm.h
    src/gemm.cu
)
set_property(TARGET gemm PROPERTY CUDA_ARCHITECTURES native)

# For source view in ncu
target_compile_options(gemm PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                       --generate-line-info>)

add_executable(gemm_test
    test/utils.h
    test/gemm_test.cpp
)
target_link_libraries(gemm_test PRIVATE gemm)
                   

add_library(conv2d STATIC
    src/conv_2d.h
    src/conv_2d.cu
)
set_property(TARGET conv2d PROPERTY CUDA_ARCHITECTURES native)

# For source view in ncu
target_compile_options(conv2d PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                        --generate-line-info>)

add_executable(conv2d_test
    test/utils.h
    test/conv2d_test.cpp
)
target_link_libraries(conv2d_test PRIVATE conv2d)


add_executable(conv3d_test
    test/utils.h
    test/conv3d_test.cpp
)
target_link_libraries(conv3d_test PRIVATE ${CUDA_LIBRARIES} ${CUDNN_LIBRARY_PATH})