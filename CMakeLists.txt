cmake_minimum_required(VERSION 3.16)

project(cuSP_cpp_tasks LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

option(CUSP_CPP_BUILD_TASK1 "Build the task1_cpp executable" ON)
option(CUSP_CPP_BUILD_TASK2 "Build the task2_cpp executable" ON)

set(CUSP_CPP_FFTW3F_HINT_DIRS
    /lib/x86_64-linux-gnu
    /usr/lib/x86_64-linux-gnu
    /lib
    /usr/lib
    /usr/local/lib
)

find_library(CUSP_CPP_FFTW3F_LIBRARY
    NAMES fftw3f
    HINTS ${CUSP_CPP_FFTW3F_HINT_DIRS}
)

if(NOT CUSP_CPP_FFTW3F_LIBRARY)
    find_file(CUSP_CPP_FFTW3F_LIBRARY
        NAMES libfftw3f.so.3 libfftw3f.so
        PATHS ${CUSP_CPP_FFTW3F_HINT_DIRS}
    )
endif()

if(NOT CUSP_CPP_FFTW3F_LIBRARY AND EXISTS "/lib/x86_64-linux-gnu/libfftw3f.so.3")
    set(CUSP_CPP_FFTW3F_LIBRARY "/lib/x86_64-linux-gnu/libfftw3f.so.3")
endif()

if(NOT CUSP_CPP_FFTW3F_LIBRARY)
    message(FATAL_ERROR
        "Could not find an FFTW3f library. "
        "Install libfftw3f or provide a path via -DCUSP_CPP_FFTW3F_LIBRARY=/path/to/libfftw3f.so.3"
    )
endif()

message(STATUS "Using FFTW3f library: ${CUSP_CPP_FFTW3F_LIBRARY}")

set(CUSP_CPP_CORE_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/bsplines/bsplines.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/common/common.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/common/fftw_helpers.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/convolution/convolution.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/demod/demod.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/estimation/estimation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/filtering/filtering.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/peak_finding/peak_finding.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/radartools/radartools.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/spectral_analysis/spectral_analysis.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/utils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/waveforms/waveforms.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/wavelets/wavelets.cpp
)

add_library(cusp_cpp_core STATIC ${CUSP_CPP_CORE_SOURCES})

target_include_directories(cusp_cpp_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(cusp_cpp_core
    PUBLIC
        ${CUSP_CPP_FFTW3F_LIBRARY}
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
