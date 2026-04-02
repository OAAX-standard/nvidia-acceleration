#include "scanner.hpp"

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <string>
#include <utility>
#include <vector>
#include <gnu/libc-version.h>

// ---------------------------------------------------------------------------
// Internal utilities
// ---------------------------------------------------------------------------

static std::string exec(const std::string& cmd) {
    FILE* pipe = popen((cmd + " 2>/dev/null").c_str(), "r");
    if (!pipe) return "";
    char buf[256];
    std::string out;
    while (fgets(buf, sizeof(buf), pipe)) out += buf;
    pclose(pipe);
    while (!out.empty() && (out.back() == '\n' || out.back() == '\r' || out.back() == ' '))
        out.pop_back();
    return out;
}

static std::string read_text(const std::string& path) {
    std::ifstream f(path);
    if (!f) return "";
    return {(std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>()};
}

static std::string read_binary(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return "";
    return {(std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>()};
}

static bool file_exists(const std::string& path) {
    return std::ifstream(path).good();
}

// ---------------------------------------------------------------------------
// glibc
// ---------------------------------------------------------------------------

static std::string detect_glibc() {
    const char* v = gnu_get_libc_version();
    return v ? v : "";
}

static std::pair<int, int> parse_ver(const std::string& v) {
    auto d = v.find('.');
    if (d == std::string::npos) return {0, 0};
    try { return {std::stoi(v.substr(0, d)), std::stoi(v.substr(d + 1))}; }
    catch (...) { return {0, 0}; }
}

// Returns true if version a >= version b  (e.g. "2.38" >= "2.34")
static bool ver_ge(const std::string& a, const std::string& b) {
    auto [am, an] = parse_ver(a);
    auto [bm, bn] = parse_ver(b);
    return am > bm || (am == bm && an >= bn);
}

// ---------------------------------------------------------------------------
// GPU / CUDA detection
// ---------------------------------------------------------------------------

struct GpuInfo {
    bool        found      = false;
    std::string compute_cap;         // e.g. "8.6"
    int         sm         = 0;      // e.g. 86
    int         cuda_major = 0;
};

static GpuInfo detect_gpu() {
    GpuInfo info;

    // Compute capability from nvidia-smi
    std::string cap = exec("nvidia-smi --query-gpu=compute_cap --format=csv,noheader");
    if (cap.empty()) return info;

    // Use only first line (first GPU)
    auto nl = cap.find('\n');
    if (nl != std::string::npos) cap = cap.substr(0, nl);
    cap.erase(std::remove(cap.begin(), cap.end(), ' '), cap.end());
    if (cap.empty()) return info;

    info.compute_cap = cap;
    std::string sm_s = cap;
    sm_s.erase(std::remove(sm_s.begin(), sm_s.end(), '.'), sm_s.end());
    try { info.sm = std::stoi(sm_s); } catch (...) { return info; }

    // CUDA version: parse "CUDA Version: X.Y" from nvidia-smi header
    std::string smi = exec("nvidia-smi");
    auto p = smi.find("CUDA Version:");
    if (p != std::string::npos) {
        std::string sub = smi.substr(p + 13);
        size_t i = 0;
        while (i < sub.size() && sub[i] == ' ') i++;
        std::string maj;
        while (i < sub.size() && std::isdigit(sub[i])) maj += sub[i++];
        if (!maj.empty()) try { info.cuda_major = std::stoi(maj); } catch (...) {}
    }

    // Fallback: /usr/local/cuda/version.json  {"cuda":{"version":"12.x.y"}}
    if (!info.cuda_major) {
        std::string j = read_text("/usr/local/cuda/version.json");
        auto vp = j.find("\"version\"");
        if (vp != std::string::npos) {
            auto q = j.find('"', vp + 9);
            if (q != std::string::npos) {
                std::string maj;
                for (++q; q < j.size() && std::isdigit(j[q]); ++q) maj += j[q];
                if (!maj.empty()) try { info.cuda_major = std::stoi(maj); } catch (...) {}
            }
        }
    }

    // Fallback: /usr/local/cuda/version.txt  "CUDA Version X.Y.Z"
    if (!info.cuda_major) {
        std::string t = read_text("/usr/local/cuda/version.txt");
        auto vp = t.find("CUDA Version ");
        if (vp != std::string::npos) {
            std::string maj;
            for (char c : t.substr(vp + 13)) {
                if (std::isdigit(c)) maj += c; else break;
            }
            if (!maj.empty()) try { info.cuda_major = std::stoi(maj); } catch (...) {}
        }
    }

    info.found = true;
    return info;
}

// ---------------------------------------------------------------------------
// x86_64 CPU march detection
// ---------------------------------------------------------------------------

// Returns true if the given flag appears as a whole word in the cpuinfo flags line.
static bool has_flag(const std::string& line, const std::string& flag) {
    auto pos = line.find(' ' + flag);
    if (pos == std::string::npos) return false;
    size_t after = pos + 1 + flag.size();
    return after >= line.size() || line[after] == ' ' || line[after] == '\r' || line[after] == '\n';
}

static bool cpu_has_flags(const std::vector<std::string>& required) {
    std::ifstream f("/proc/cpuinfo");
    if (!f) return false;
    std::string line;
    while (std::getline(f, line)) {
        if (line.rfind("flags", 0) != 0) continue;
        for (const auto& flag : required)
            if (!has_flag(line, flag)) return false;
        return true;
    }
    return false;
}

static std::string detect_x86_march() {
    // Requirements per level (each level is a superset of the level below):
    // v4: avx512f avx512bw avx512cd avx512dq avx512vl
    // v3: avx avx2 bmi1 bmi2 f16c fma movbe xsave
    // v2: cx16 lahf_lm popcnt sse4_1 sse4_2 ssse3
    if (cpu_has_flags({"avx512f", "avx512bw", "avx512cd", "avx512dq", "avx512vl"}))
        return "x86-64-v4";
    if (cpu_has_flags({"avx", "avx2", "bmi1", "bmi2", "f16c", "fma", "movbe", "xsave"}))
        return "x86-64-v3";
    if (cpu_has_flags({"cx16", "lahf_lm", "popcnt", "sse4_1", "sse4_2", "ssse3"}))
        return "x86-64-v2";
    return "x86-64";
}

// ---------------------------------------------------------------------------
// aarch64 platform / march detection
// ---------------------------------------------------------------------------

// /proc/device-tree/compatible is null-separated; replace nulls before searching.
static bool dt_compatible_has(const std::string& needle) {
    std::string content = read_binary("/proc/device-tree/compatible");
    if (content.empty()) return false;
    std::replace(content.begin(), content.end(), '\0', '\n');
    return content.find(needle) != std::string::npos;
}

// ARM CPU parts that identify server-class Neoverse processors (SBSA).
// All are armv9-capable; earlier Neoverse (V1/N2) are armv8.4/8.5 but still
// map to the armv9-a tarball which is the closest available build.
static const std::vector<std::string> NEOVERSE_PARTS = {
    "0xd40",  // Neoverse V1  (armv8.4-a)
    "0xd49",  // Neoverse N2  (armv9.0-a)
    "0xd4f",  // Neoverse V2  (armv9.0-a)  — NVIDIA Grace (DGX Spark gen1)
    "0xd85",  // Neoverse V3AE (armv9.2-a) — NVIDIA Grace (DGX Spark B200)
    "0xd87",  // Neoverse V3  (armv9.2-a)  — NVIDIA Grace (DGX Spark B200)
};

// Returns "jetson" | "sbsa" | "aarch64"
static std::string detect_aarch64_platform() {
    // Jetson: device-tree contains "tegra", or JetPack marker file exists
    if (dt_compatible_has("tegra") || file_exists("/etc/nv_tegra_release"))
        return "jetson";

    // SBSA / Grace via device-tree (only present on DT-based systems)
    if (dt_compatible_has("nvidia,grace") || dt_compatible_has("nvidia,grace-hopper"))
        return "sbsa";

    // SBSA via CPU part number (works on both DT and ACPI systems)
    std::string cpuinfo = read_text("/proc/cpuinfo");
    for (const auto& part : NEOVERSE_PARTS)
        if (cpuinfo.find(part) != std::string::npos)
            return "sbsa";

    // SBSA fallback: ACPI-based systems (server ARM) have no /proc/device-tree/
    // but do have /sys/firmware/acpi/. Combined with ARM implementer 0x41 (Arm Ltd)
    // this is a strong signal of a server-class platform.
    if (file_exists("/sys/firmware/acpi/") && cpuinfo.find("CPU implementer\t: 0x41") != std::string::npos)
        return "sbsa";

    return "aarch64";
}

// All aarch64 builds use armv8-a; this is kept for diagnostic output only.
static std::string detect_aarch64_march(const std::string&) {
    return "armv8-a";
}

// ---------------------------------------------------------------------------
// Runtime tarball selection
// ---------------------------------------------------------------------------

struct Tarball {
    std::string platform;   // "x86_64", "jetson", "sbsa"
    std::string ubuntu;     // path label: "Ubuntu22.04", "Ubuntu24.04"
    std::string min_glibc;  // minimum glibc required by this build
    int         cuda;
};

// Catalog of published runtime tarballs.
// Add new entries here when new builds are released.
static const std::vector<Tarball> CATALOG = {
    {"x86_64", "Ubuntu22.04", "2.35", 12},
    {"x86_64", "Ubuntu22.04", "2.35", 13},
    {"jetson",  "Ubuntu22.04", "2.35", 12},
    {"sbsa",    "Ubuntu24.04", "2.38", 12},
    {"sbsa",    "Ubuntu24.04", "2.38", 13},
};

static std::string select_runtime(
    const std::string& platform, const std::string& glibc, int cuda_major)
{
    const std::string arch = (platform == "x86_64") ? "x86_64" : "aarch64";

    // Cap at the highest available CUDA major (13)
    const int cuda = std::min(cuda_major, 13);

    // Find a compatible entry: same platform, system glibc satisfies the minimum
    // Prefer the detected CUDA version; fall back to cuda 12 if not available
    for (int c : {cuda, 12}) {
        for (const auto& t : CATALOG) {
            if (t.platform == platform && t.cuda == c && ver_ge(glibc, t.min_glibc))
                return arch + "/" + t.ubuntu + "/library-cuda_" + std::to_string(c) + ".tar.gz";
        }
    }

    return "";
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

ScanResult scan_system() {
    ScanResult r;

#if defined(__x86_64__)
    r.architecture = "x86_64";
    r.platform     = "x86_64";
    r.cpu_march    = detect_x86_march();
#elif defined(__aarch64__)
    r.architecture = "aarch64";
    r.platform     = detect_aarch64_platform();
    r.cpu_march    = detect_aarch64_march(r.platform);
#else
    r.unsupported_reason =
        "Unsupported CPU architecture — only x86_64 and aarch64 are supported";
    return r;
#endif

    r.glibc_version = detect_glibc();
    if (r.glibc_version.empty()) {
        r.unsupported_reason = "Failed to detect glibc version";
        return r;
    }

    GpuInfo gpu = detect_gpu();
    if (!gpu.found) {
        r.unsupported_reason =
            "No NVIDIA GPU detected (nvidia-smi not found or failed)";
        return r;
    }

    r.gpu_compute_cap = gpu.compute_cap;
    r.gpu_sm          = gpu.sm;
    r.cuda_major      = gpu.cuda_major;

    if (gpu.sm < 75) {
        r.unsupported_reason =
            "GPU compute capability " + gpu.compute_cap + " (SM "
            + std::to_string(gpu.sm)
            + ") is below the TensorRT 10 minimum (SM 75 / Turing)";
        return r;
    }

    if (!gpu.cuda_major) {
        r.unsupported_reason = "Could not determine installed CUDA version";
        return r;
    }

    r.recommended_runtime =
        select_runtime(r.platform, r.glibc_version, r.cuda_major);

    if (r.recommended_runtime.empty()) {
        r.unsupported_reason =
            "No compatible runtime tarball found for this configuration ("
            + r.platform
            + ", glibc " + r.glibc_version
            + ", CUDA "  + std::to_string(r.cuda_major) + ")";
        return r;
    }

    r.supported = true;
    return r;
}
