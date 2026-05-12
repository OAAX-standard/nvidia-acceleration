cmake_minimum_required(VERSION 3.10.2)
project(NVIDIAOAAXTests)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# RUNTIME_LIB_DIR: directory that directly contains libRuntimeLibrary.so
# May be set to the extracted runtime bin/ dir or the cmake build output dir.
# If not given, search the TRT build tree.
if(NOT DEFINED RUNTIME_LIB_DIR)
    file(GLOB_RECURSE _found_so
        "${CMAKE_CURRENT_SOURCE_DIR}/../../tensorrt/runtime-library/build/*/libRuntimeLibrary.so"
    )
    if(_found_so)
        list(GET _found_so 0 _first_so)
        get_filename_component(RUNTIME_LIB_DIR "${_first_so}" DIRECTORY)
    else()
        set(RUNTIME_LIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../tensorrt/runtime-library/build")
    endif()
endif()

# Path to the runtime public headers (runtime_core.hpp)
if(NOT DEFINED RUNTIME_INCLUDE_DIR)
    set(RUNTIME_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../tensorrt/runtime-library/include")
endif()

# tensors_struct.h is bundled in tests/runtime/include/
set(TESTS_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")

message(STATUS "Runtime lib dir:     ${RUNTIME_LIB_DIR}")
message(STATUS "Runtime include dir: ${RUNTIME_INCLUDE_DIR}")

foreach(TEST_NAME simple_test inference_test)
    add_executable(${TEST_NAME} ${CMAKE_CURRENT_SOURCE_DIR}/${TEST_NAME}.cpp)
    target_include_directories(${TEST_NAME} PRIVATE
        ${RUNTIME_INCLUDE_DIR}
        ${TESTS_INCLUDE_DIR}
    )
    target_link_directories(${TEST_NAME} PRIVATE ${RUNTIME_LIB_DIR})
    target_link_libraries(${TEST_NAME} PRIVATE RuntimeLibrary)
    target_link_options(${TEST_NAME} PRIVATE "-Wl,--allow-shlib-undefined")
    set_target_properties(${TEST_NAME} PROPERTIES
        BUILD_RPATH  "$ORIGIN"
        INSTALL_RPATH "$ORIGIN"
    )
endforeach()
