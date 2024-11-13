cmake_minimum_required(VERSION 3.10.2)

project(RuntimeLibrary)

set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(DEPS_DIR ${CMAKE_CURRENT_LIST_DIR}/deps)

# Get cuda version as an option
set(PLATFORM "NONE" CACHE STRING "The target platform")
set(CUDA_VERSION "NULL" CACHE STRING "CUDA version")

######################### customize when cross-compiling ###############################################################

# set generic build target
set(GENERIC_BUILD_TARGET X86_64)

# set COMPILER_PREFIX, for example, "" for default compiler, arm-linux- , or aarch64-linux- etc for cross compilers

if(PLATFORM STREQUAL "X86_64")
        set(GENERIC_BUILD_TARGET X86_64)
        set(CROSS_ROOT "/opt/x86_64-unknown-linux-gnu-gcc-9.5.0")
        set(COMPILER_PREFIX "x86_64-unknown-linux-gnu-")
        set(SYSROOT "/opt/x86_64-unknown-linux-gnu-gcc-9.5.0/x86_64-unknown-linux-gnu/sysroot")
        # set compiler flags
        set(CMAKE_BUILD_FLAGS " -Wall -Wextra -Werror -Ofast -fno-math-errno -fopenmp -std=c99 -march=haswell")
elseif(PLATFORM STREQUAL "AARCH64")
        if(CUDA_VERSION STREQUAL "12")
                set(CROSS_ROOT "/opt/gcc-arm-11.2-2022.02-x86_64-aarch64-none-linux-gnu")
        else()
                set(CROSS_ROOT "/opt/gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu")
        endif()
        set(GENERIC_BUILD_TARGET AARCH64)
        set(COMPILER_PREFIX "aarch64-none-linux-gnu-")
        set(ADDITIONAL_LIBS "clog")
else()
        message(FATAL_ERROR "Please specify the target platform: X86_64 or AARCH64")
endif()

# for aarch64 cuda version 10, set ONNXRUNTIME_API_VERSION=11
if(PLATFORM STREQUAL "AARCH64" AND CUDA_VERSION STREQUAL "10")
        add_definitions(-DONNXRUNTIME_API_VERSION=11)
endif()

set(CUDA_BIN_FOLDER_NAME "${GENERIC_BUILD_TARGET}_CUDA_${CUDA_VERSION}")
set(CMAKE_C_COMPILER ${CROSS_ROOT}/bin/${COMPILER_PREFIX}gcc)
set(CMAKE_CXX_COMPILER ${CROSS_ROOT}/bin/${COMPILER_PREFIX}g++)
set(CMAKE_LINKER ${CROSS_ROOT}/bin/${COMPILER_PREFIX}ld)
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

#--build: the machine you are building on
#--host: the machine you are building for
#--target: the machine that GCC will produce binary for
set(BUILD X86_64-linux-gnu)
set(HOST ${GENERIC_BUILD_TARGET}-linux)
set(TARGET ${GENERIC_BUILD_TARGET})


######################### webassembly wamr settings ####################################################################

# set make system
set(CMAKE_SYSTEM_NAME Linux)

######################## webassembly wamr library build ################################################################

# Reset default linker flags
set(CMAKE_SHARED_LIBRARY_LINK_C_FLAGS " ")
set(CMAKE_SHARED_LIBRARY_LINK_CXX_FLAGS " ")

# compiling and linking flags
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--gc-sections ")

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -Wformat -Wformat-security -fdata-sections -ffunction-sections -Wl,--gc-sections")

####################### toolchain arguments ############################################################################

INCLUDE(CMakeForceCompiler)

set(CMAKE_VERBOSE_MAKEFILE ON)

set(CMAKE_BUILD_FLAGS " ${CMAKE_BUILD_FLAGS} -std=c99 -fno-common -Wall -Wextra")
set(CMAKE_BUILD_FLAGS " ${CMAKE_BUILD_FLAGS} -Wno-missing-field-initializers -fno-exceptions")
set(CMAKE_BUILD_FLAGS " ${CMAKE_BUILD_FLAGS} -Wno-attribute-warning -fdata-sections -ffunction-sections -Wl,--gc-sections ")

set(CMAKE_C_FLAGS " ${CMAKE_C_FLAGS} ${CMAKE_BUILD_FLAGS}  -D_GNU_SOURCE -I${CROSS_ROOT}/include")
set(CMAKE_CXX_FLAGS " ${CMAKE_CXX_FLAGS} ${CMAKE_BUILD_FLAGS}  -D_GNU_SOURCE -I${CROSS_ROOT}/include")
set(CMAKE_CPP_FLAGS " ${CMAKE_CPP_FLAGS} ${CMAKE_BUILD_FLAGS}  -D_GNU_SOURCE -I${CROSS_ROOT}/include")

message(STATUS "SDKTARGETSYSROOT=${SDKTARGETSYSROOT}")
message(STATUS "CROSS_ROOT=${CROSS_ROOT}")

set(CMAKE_AR ${CROSS_ROOT}/bin/${COMPILER_PREFIX}ar)
set(CMAKE_AS ${CROSS_ROOT}/bin/${COMPILER_PREFIX}as)
set(CMAKE_LD ${CROSS_ROOT}/bin/${COMPILER_PREFIX}ld)
set(CMAKE_RANLIB ${CROSS_ROOT}/bin/${COMPILER_PREFIX}ranlib)
set(CMAKE_C_COMPILER ${CROSS_ROOT}/bin/${COMPILER_PREFIX}gcc)
set(CMAKE_CXX_COMPILER ${CROSS_ROOT}/bin/${COMPILER_PREFIX}g++)
set(CMAKE_ASM_COMPILER ${CROSS_ROOT}/bin/${COMPILER_PREFIX}gcc)
set(CMAKE_NM ${CROSS_ROOT}/bin/${COMPILER_PREFIX}NM)

# this is the file system root of the target
set(CMAKE_FIND_ROOT_PATH ${SYSROOT})

# search for programs in the build host directories
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)

# for libraries and headers in the target directories
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)

######################## generic build settings ########################################################################

add_definitions(-DDAEMON_NO_CLOSE_STDIO=${DAEMON_NO_CLOSE_STDIO})
add_definitions(-DDAEMON_NAME= "${DAEMON_NAME}")
add_definitions(-DDAEMON_PID_FILE_NAME= "${DAEMON_PID_FILE_NAME}")
add_definitions(-DDAEMON_LOG_FILE_NAME= "${DAEMON_LOG_FILE_NAME}")
add_definitions(-DDAEMON_CMD_PIPE_NAME= "${DAEMON_CMD_PIPE_NAME}")
add_definitions(-DDAEMON_NO_CHDIR=${DAEMON_NO_CHDIR})
add_definitions(-DDAEMON_NO_FORK=${DAEMON_NO_FORK})

# set some basic c, c++ and linker flags
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${CMAKE_BUILD_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_BUILD_FLAGS}")
set(LINKER_FLAGS "${LINKER_FLAGS} ${CMAKE_BUILD_FLAGS}")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_BUILD_FLAGS}")
set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_BUILD_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_BUILD_FLAGS} ")

# add all source files
file(GLOB_RECURSE SRC
    ${SRC_DIR}/runtime_utils.c
    ${SRC_DIR}/runtime_core.c
    ${DEPS_DIR}/utils/queue.c
    ${DEPS_DIR}/utils/logger.c
)

get_filename_component(full_path . ABSOLUTE)

######################## main bin build settings #######################################################################

# binary target
add_library(RuntimeLibrary SHARED ${SRC})

# target include / link dirs
target_include_directories(RuntimeLibrary PUBLIC ${INCLUDE_DIR})
target_include_directories(RuntimeLibrary PUBLIC ${DEPS_DIR}/utils)

set(RE2_DIR "${DEPS_DIR}/re2")
set(ONNX_RUNTIME_DIR "${DEPS_DIR}/onnxruntime")
set(CUDA_DIR "${DEPS_DIR}/cuda")
set(CPUINFO_DIR "${DEPS_DIR}/cpuinfo")

target_link_libraries(RuntimeLibrary PUBLIC pthread atomic dl)

target_include_directories(RuntimeLibrary PUBLIC ${RE2_DIR}/include)
target_link_directories(RuntimeLibrary PUBLIC "${RE2_DIR}/${GENERIC_BUILD_TARGET}")
target_link_libraries(RuntimeLibrary PUBLIC re2)

target_include_directories(RuntimeLibrary PUBLIC ${CPUINFO_DIR}/include)
target_link_directories(RuntimeLibrary PUBLIC "${CPUINFO_DIR}/${GENERIC_BUILD_TARGET}")
target_link_libraries(RuntimeLibrary PUBLIC cpuinfo)


set(CMAKE_FIND_LIBRARY_SUFFIXES ".a")
set(BUILD_SHARED_LIBS OFF)
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_BUILD_FLAGS} ")
target_link_libraries(RuntimeLibrary PRIVATE ${LLVM_AVAILABLE_LIBS} -ldl pthread ${CGRLIB} -lgomp -lstdc++ -lm -ldl)


target_include_directories(RuntimeLibrary PUBLIC ${ONNX_RUNTIME_DIR}/${CUDA_BIN_FOLDER_NAME}/include)
target_include_directories(RuntimeLibrary PUBLIC ${ONNX_RUNTIME_DIR}/${CUDA_BIN_FOLDER_NAME}/include/onnxruntime/core/session)

target_link_libraries(RuntimeLibrary PUBLIC -Wl,-rpath,./)
target_link_libraries(RuntimeLibrary PUBLIC -Wl,--disable-new-dtags)

target_link_directories(RuntimeLibrary PUBLIC "${ONNX_RUNTIME_DIR}/${CUDA_BIN_FOLDER_NAME}")
target_link_libraries(
        RuntimeLibrary
        PUBLIC
        onnxruntime_session
        onnxruntime_optimizer
        onnxruntime_providers
        onnxruntime_util
        onnxruntime_framework
        onnxruntime_graph
        onnxruntime_mlas
        onnxruntime_common
        onnxruntime_flatbuffers
        absl_base
        absl_log_severity
        absl_malloc_internal
        absl_raw_logging_internal
        absl_spinlock_wait
        absl_throw_delegate
        absl_hashtablez_sampler
        absl_raw_hash_set
        absl_hash
        absl_city
        absl_low_level_hash
        absl_int128
        absl_exponential_biased
        absl_cord
        absl_cord_internal
        absl_cordz_functions
        absl_cordz_handle
        absl_cordz_info
        absl_strings
        absl_strings_internal
        absl_graphcycles_internal
        absl_synchronization
        absl_civil_time
        absl_time
        absl_time_zone
        absl_bad_optional_access
        absl_bad_variant_access
        flatbuffers
        nsync_cpp
        onnx
        onnx_proto
        protobuf-lite
        re2
        cpuinfo

        ${ADDITIONAL_LIBS}

        stdc++
)

add_custom_command(TARGET RuntimeLibrary POST_BUILD
        COMMAND cp ${ONNX_RUNTIME_DIR}/${CUDA_BIN_FOLDER_NAME}/*.so ${CMAKE_CURRENT_BINARY_DIR}/
)
# Copy cuDNN libs
add_custom_command(TARGET RuntimeLibrary POST_BUILD
        COMMAND cp ${CUDA_DIR}/${GENERIC_BUILD_TARGET}/cuda_${CUDA_VERSION}/libcudnn* ${CMAKE_CURRENT_BINARY_DIR}/
)
# Copy libm.so for CUDA 10
if(PLATFORM STREQUAL "AARCH64" AND CUDA_VERSION STREQUAL "10")
        add_custom_command(TARGET RuntimeLibrary POST_BUILD
        COMMAND cp ${CUDA_DIR}/${GENERIC_BUILD_TARGET}/cuda_${CUDA_VERSION}/libm.so.6 ${CMAKE_CURRENT_BINARY_DIR}/
        )
endif()
