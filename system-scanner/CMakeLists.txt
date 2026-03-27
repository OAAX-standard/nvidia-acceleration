cmake_minimum_required(VERSION 3.10)
project(system-scanner)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(system_scanner main.cpp scanner.cpp)

# gnu/libc-version.h is part of glibc — always available on the target Linux systems
# we support (x86_64 and aarch64 Ubuntu).
target_link_libraries(system_scanner PRIVATE)
