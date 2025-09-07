cmake_minimum_required(VERSION 3.18)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler="/utf-8">)
set(CMAKE_GENERATOR_PLATFORM x64)
set(CMAKE_GENERATOR_TOOLSET v142 CACHE STRING "" FORCE)  #cuda11.x对应vs2019(v142)

project(Transpose LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

add_executable(${PROJECT_NAME}_cuda
    ../common/buffer.h
    transpose.cu
)

set_target_properties(${PROJECT_NAME}_cuda PROPERTIES
    CUDA_ARCHITECTURES "75" 
)

target_include_directories(${PROJECT_NAME}_cuda PUBLIC ../)
target_link_libraries(${PROJECT_NAME}_cuda
    CUDA::cudart
)

set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo") #生成带调试信息的cuda代码