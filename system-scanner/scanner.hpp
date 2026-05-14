#pragma once
#include <string>

struct ScanResult {
    bool        supported         = false;
    std::string unsupported_reason;   // empty when supported

    std::string architecture;         // "x86_64" | "aarch64"
    std::string platform;             // "x86_64" | "jetson" | "sbsa" | "aarch64"
    std::string cpu_march;            // e.g. "x86-64-v3", "armv8.2-a+fp16+dotprod"
    std::string glibc_version;        // e.g. "2.35"
    int         cuda_major    = 0;    // 12, 13, …
    std::string gpu_compute_cap;      // e.g. "8.6"
    int         gpu_sm        = 0;    // e.g. 86

    std::string recommended_runtime;  // relative tarball path, empty if unsupported
};

ScanResult scan_system();
