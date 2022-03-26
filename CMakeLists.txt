cmake_minimum_required(VERSION 3.16)

set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
project(cuda_canny LANGUAGES CUDA CXX)

set(CMAKE_CUDA_STANDARD 14)

add_executable(cuda_canny main.cpp cpu_canny.cpp gpu_canny.cu)

set_target_properties(cuda_canny PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

include(FindOpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(cuda_canny ${OpenCV_LIBS})

if (NUM_THREADS)
    add_definitions(-DNUM_THREADS=${NUM_THREADS})
else()
    add_definitions(-DNUM_THREADS=1)
endif ()

if (BLOCK_SIZE)
    add_definitions(-DBLOCK_SIZE=${BLOCK_SIZE})
else()
    add_definitions(-DBLOCK_SIZE=10)
endif()
