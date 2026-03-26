cmake_minimum_required(VERSION 3.14)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_VERBOSE_MAKEFILE OFF)

# === Platform Selection ===
# Supported: X86_64, AARCH64 (Jetson / JetPack), AARCH64_SERVER (DGX Spark / Grace CPU)
if(NOT DEFINED PLATFORM)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
        set(PLATFORM AARCH64_SERVER)
    else()
        set(PLATFORM X86_64)
    endif()
    message(STATUS "Auto-detected PLATFORM=${PLATFORM}")
endif()

if(NOT PLATFORM STREQUAL "X86_64" AND
   NOT PLATFORM STREQUAL "AARCH64" AND
   NOT PLATFORM STREQUAL "AARCH64_SERVER")
    message(FATAL_ERROR
        "PLATFORM must be X86_64, AARCH64, or AARCH64_SERVER. "
        "Windows is not supported in the TensorRT-native runtime.")
endif()

# === Runtime Version ===
if(NOT DEFINED RUNTIME_VERSION)
    message(FATAL_ERROR "Please specify: -DRUNTIME_VERSION=<version>")
endif()

# === Cross-compilation Toolchain Setup ===
# Must be set before project() so CMake picks up the right compiler.
set(CMAKE_SYSTEM_NAME Linux)

if(PLATFORM STREQUAL "X86_64")
    set(CMAKE_SYSTEM_PROCESSOR x86_64)
    set(CROSS_ROOT "/opt/x86-64-v2--glibc--stable-2022.08-1")
    set(COMPILER_PREFIX "x86_64-linux-")
    set(PLATFORM_BUILD_FLAGS "-fno-math-errno -fopenmp -march=haswell")
elseif(PLATFORM STREQUAL "AARCH64")
    set(CMAKE_SYSTEM_PROCESSOR aarch64)
    set(CROSS_ROOT "/opt/gcc-arm-11.2-2022.02-x86_64-aarch64-none-linux-gnu")
    set(COMPILER_PREFIX "aarch64-none-linux-gnu-")
    set(PLATFORM_BUILD_FLAGS "-fno-math-errno -fopenmp")
elseif(PLATFORM STREQUAL "AARCH64_SERVER")
    set(CMAKE_SYSTEM_PROCESSOR aarch64)
    set(CROSS_ROOT "/opt/arm-gnu-toolchain-13.2.Rel1-x86_64-aarch64-none-linux-gnu")
    set(COMPILER_PREFIX "aarch64-none-linux-gnu-")
    set(PLATFORM_BUILD_FLAGS "-fno-math-errno -fopenmp")
endif()

set(CMAKE_AR           ${CROSS_ROOT}/bin/${COMPILER_PREFIX}ar)
set(CMAKE_AS           ${CROSS_ROOT}/bin/${COMPILER_PREFIX}as)
set(CMAKE_LD           ${CROSS_ROOT}/bin/${COMPILER_PREFIX}ld)
set(CMAKE_RANLIB       ${CROSS_ROOT}/bin/${COMPILER_PREFIX}ranlib)
set(CMAKE_C_COMPILER   ${CROSS_ROOT}/bin/${COMPILER_PREFIX}gcc)
set(CMAKE_CXX_COMPILER ${CROSS_ROOT}/bin/${COMPILER_PREFIX}g++)
set(CMAKE_ASM_COMPILER ${CROSS_ROOT}/bin/${COMPILER_PREFIX}gcc)

# Never search the target sysroot for build tools (executables)
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

# Suppress CMake's default "-rdynamic" flag which breaks some cross-linkers
set(CMAKE_SHARED_LIBRARY_LINK_C_FLAGS " ")
set(CMAKE_SHARED_LIBRARY_LINK_CXX_FLAGS " ")

project(RuntimeLibrary C CXX)

add_definitions(-DRUNTIME_VERSION="${RUNTIME_VERSION}")

message(STATUS "PLATFORM=${PLATFORM}")
message(STATUS "CROSS_ROOT=${CROSS_ROOT}")
message(STATUS "RUNTIME_VERSION=${RUNTIME_VERSION}")

# === External dependency paths ===
# TRT_DIR: path to extracted TensorRT archive (must contain include/ and lib/)
# CUDA_DIR: path to CUDA toolkit (must contain include/ and lib/ or lib64/)
if(NOT DEFINED TRT_DIR)
    message(FATAL_ERROR "Please specify: -DTRT_DIR=<path to TensorRT directory>")
endif()
if(NOT DEFINED CUDA_DIR)
    message(FATAL_ERROR "Please specify: -DCUDA_DIR=<path to CUDA toolkit directory>")
endif()

set(TRT_INCLUDE_DIR  ${TRT_DIR}/include)
set(TRT_LIB_DIR      ${TRT_DIR}/lib)

set(CUDA_INCLUDE_DIR ${CUDA_DIR}/include)
if(EXISTS ${CUDA_DIR}/lib64)
    set(CUDA_LIB_DIR ${CUDA_DIR}/lib64)
else()
    set(CUDA_LIB_DIR ${CUDA_DIR}/lib)
endif()

message(STATUS "TRT_DIR=${TRT_DIR}")
message(STATUS "CUDA_DIR=${CUDA_DIR}")

# === Compiler flags ===
set(BUILD_FLAGS "${PLATFORM_BUILD_FLAGS}")
set(BUILD_FLAGS "${BUILD_FLAGS} -fno-common -Wall -Wextra -Wformat -Wformat-security")
set(BUILD_FLAGS "${BUILD_FLAGS} -Wno-missing-field-initializers -Wno-attribute-warning")
set(BUILD_FLAGS "${BUILD_FLAGS} -fdata-sections -ffunction-sections -fno-exceptions")

set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}   ${BUILD_FLAGS} -D_GNU_SOURCE -I${CROSS_ROOT}/include")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${BUILD_FLAGS} -D_GNU_SOURCE -I${CROSS_ROOT}/include -fexceptions")

set(CMAKE_SHARED_LINKER_FLAGS "${BUILD_FLAGS}")
set(CMAKE_MODULE_LINKER_FLAGS "${BUILD_FLAGS}")

# === Dependencies via FetchContent ===
include(FetchContent)

FetchContent_Declare(spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog
    GIT_TAG        v1.15.0
    GIT_SHALLOW    TRUE)
set(SPDLOG_BUILD_EXAMPLE OFF CACHE BOOL "" FORCE)
set(SPDLOG_BUILD_TESTS   OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(spdlog)

FetchContent_Declare(concurrentqueue
    GIT_REPOSITORY https://github.com/cameron314/concurrentqueue
    GIT_TAG        v1.0.4
    GIT_SHALLOW    TRUE)
FetchContent_MakeAvailable(concurrentqueue)

# tools repo contains c-utilities as a subdirectory
FetchContent_Declare(tools
    GIT_REPOSITORY https://github.com/OAAX-standard/tools
    GIT_TAG        main
    GIT_SHALLOW    TRUE
    SOURCE_SUBDIR  c-utilities)
FetchContent_MakeAvailable(tools)

# === Library Target ===
file(GLOB_RECURSE SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/runtime_core.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/runtime_utils.cpp)

add_library(RuntimeLibrary SHARED ${SRC})

target_link_options(RuntimeLibrary PRIVATE "-Wl,-z,noexecstack")

# AARCH64 targets use empty stub .so files for linking (real libs on target machine).
# --allow-shlib-undefined lets the linker accept stubs with no symbol definitions.
if(NOT PLATFORM STREQUAL "X86_64")
    target_link_options(RuntimeLibrary PRIVATE "-Wl,--allow-shlib-undefined")
endif()

# Build-type optimizations
if("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
    target_compile_options(RuntimeLibrary PUBLIC -Ofast -s -g)
elseif("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    target_compile_options(RuntimeLibrary PUBLIC -O0 -g3)
elseif("${CMAKE_BUILD_TYPE}" STREQUAL "RelWithDebInfo")
    target_compile_options(RuntimeLibrary PUBLIC -O2 -g)
elseif("${CMAKE_BUILD_TYPE}" STREQUAL "MinSizeRel")
    target_compile_options(RuntimeLibrary PUBLIC -Os -g)
endif()

# === Include Directories ===
target_include_directories(RuntimeLibrary PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${TRT_INCLUDE_DIR}
    ${CUDA_INCLUDE_DIR}
    ${concurrentqueue_SOURCE_DIR})

# === Link Directories ===
target_link_directories(RuntimeLibrary PUBLIC
    ${TRT_LIB_DIR}
    ${CUDA_LIB_DIR})

# === Link Libraries ===
target_link_libraries(RuntimeLibrary PUBLIC
    -Wl,--start-group
    nvinfer
    nvinfer_plugin
    cudart
    spdlog::spdlog
    c_utilities
    stdc++
    -Wl,--end-group)

# === Output ===
# $ORIGIN tells the dynamic linker to search the directory where the .so lives,
# so the bundled TRT/CUDA libs are found instead of any system-installed version.
set_target_properties(RuntimeLibrary PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY  ${CMAKE_BINARY_DIR}/bin
    RUNTIME_OUTPUT_DIRECTORY  ${CMAKE_BINARY_DIR}/bin
    INSTALL_RPATH             "$ORIGIN"
    BUILD_WITH_INSTALL_RPATH  TRUE)

# === Bundle runtime dependencies into bin/ ===
# Copy TRT and CUDA libs alongside libRuntimeLibrary.so so the output directory
# is self-contained. Uses cp -P to preserve symlink chains.
add_custom_command(TARGET RuntimeLibrary POST_BUILD
    COMMAND bash -c
        "cp -P ${TRT_LIB_DIR}/libnvinfer.so* \
               ${TRT_LIB_DIR}/libnvinfer_plugin.so* \
               ${CUDA_LIB_DIR}/libcudart.so* \
               $<TARGET_FILE_DIR:RuntimeLibrary>/"
    COMMENT "Bundling TRT and CUDA libs into bin/"
)
