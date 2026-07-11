cmake_minimum_required(VERSION 3.16)

project(cuSP_cpp_tasks LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

set(CUSP_CPP_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")
set(CUSP_CPP_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")

option(CUSP_CPP_BUILD_TASK1 "Build the task1_cpp executable" ON)
option(CUSP_CPP_BUILD_TASK2 "Build the task2_cpp executable" ON)

# FFTW3f 已替换为 common.cpp 中的纯 C++ fft_inplace() 实现，无需外部依赖

set(CUSP_CPP_CORE_SOURCES
    ${CUSP_CPP_SOURCE_DIR}/bsplines/bsplines.cpp
    ${CUSP_CPP_SOURCE_DIR}/common/common.cpp
    ${CUSP_CPP_SOURCE_DIR}/common/fftw_helpers.cpp
    ${CUSP_CPP_SOURCE_DIR}/convolution/convolution.cpp
    ${CUSP_CPP_SOURCE_DIR}/demod/demod.cpp
    ${CUSP_CPP_SOURCE_DIR}/estimation/estimation.cpp
    ${CUSP_CPP_SOURCE_DIR}/filtering/filtering.cpp
    ${CUSP_CPP_SOURCE_DIR}/peak_finding/peak_finding.cpp
    ${CUSP_CPP_SOURCE_DIR}/radartools/radartools.cpp
    ${CUSP_CPP_SOURCE_DIR}/spectral_analysis/spectral_analysis.cpp
    ${CUSP_CPP_SOURCE_DIR}/utils/utils.cpp
    ${CUSP_CPP_SOURCE_DIR}/waveforms/waveforms.cpp
    ${CUSP_CPP_SOURCE_DIR}/wavelets/wavelets.cpp
)

add_library(cusp_cpp_core STATIC ${CUSP_CPP_CORE_SOURCES})

target_include_directories(cusp_cpp_core
    PUBLIC
        ${CUSP_CPP_INCLUDE_DIR}
)

target_link_libraries(cusp_cpp_core
    PUBLIC
)

target_compile_features(cusp_cpp_core PUBLIC cxx_std_17)

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9)
    target_link_libraries(cusp_cpp_core PUBLIC stdc++fs)
endif()

if(MSVC)
    target_compile_options(cusp_cpp_core PRIVATE /W4)
else()
    target_compile_options(cusp_cpp_core PRIVATE -Wall -Wextra -Wpedantic)
endif()

function(cusp_cpp_add_task target_name source_file)
    add_executable(${target_name} ${CMAKE_CURRENT_SOURCE_DIR}/${source_file})
    target_link_libraries(${target_name} PRIVATE cusp_cpp_core)
    target_compile_features(${target_name} PRIVATE cxx_std_17)
endfunction()

if(CUSP_CPP_BUILD_TASK1)
    cusp_cpp_add_task(task1_cpp task1.cpp)
endif()

if(CUSP_CPP_BUILD_TASK2)
    cusp_cpp_add_task(task2_cpp task2.cpp)
endif()