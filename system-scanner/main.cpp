#include "scanner.hpp"

#include <cstring>
#include <fstream>
#include <iostream>
#include <string>

static std::string escape_json(const std::string& s) {
    std::string out;
    for (char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            default:   out += c;
        }
    }
    return out;
}

static std::string to_json(const ScanResult& r) {
    auto str = [](const std::string& v) -> std::string {
        return v.empty() ? "null" : "\"" + escape_json(v) + "\"";
    };
    auto num = [](int v) -> std::string {
        return v > 0 ? std::to_string(v) : "null";
    };

    std::string j;
    j += "{\n";
    j += "  \"supported\": "              + std::string(r.supported ? "true" : "false") + ",\n";
    j += "  \"unsupported_reason\": "     + str(r.unsupported_reason)    + ",\n";
    j += "  \"architecture\": "           + str(r.architecture)          + ",\n";
    j += "  \"platform\": "               + str(r.platform)              + ",\n";
    j += "  \"cpu_march\": "              + str(r.cpu_march)             + ",\n";
    j += "  \"glibc_version\": "          + str(r.glibc_version)         + ",\n";
    j += "  \"cuda_version_major\": "     + num(r.cuda_major)            + ",\n";
    j += "  \"gpu_compute_capability\": " + str(r.gpu_compute_cap)       + ",\n";
    j += "  \"gpu_sm\": "                 + num(r.gpu_sm)                + ",\n";
    j += "  \"recommended_runtime\": "    + str(r.recommended_runtime)   + "\n";
    j += "}\n";
    return j;
}

static void print_usage(const char* argv0) {
    std::cerr << "Usage: " << argv0 << " [--output <file>]\n"
              << "\n"
              << "Scans the system and emits a JSON report indicating whether the\n"
              << "OAAX NVIDIA TensorRT runtime is supported and which tarball to use.\n"
              << "\n"
              << "  --output <file>   Write JSON to <file> instead of stdout\n"
              << "\n"
              << "Exit code: 0 = supported, 1 = unsupported or error\n";
}

int main(int argc, char* argv[]) {
    std::string output_path;

    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        }
        if (std::strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
            output_path = argv[++i];
        }
    }

    ScanResult result = scan_system();
    std::string json  = to_json(result);

    if (output_path.empty()) {
        std::cout << json;
    } else {
        std::ofstream f(output_path);
        if (!f) {
            std::cerr << "Error: cannot write to " << output_path << "\n";
            return 1;
        }
        f << json;
        std::cerr << "Report written to " << output_path << "\n";
    }

    return result.supported ? 0 : 1;
}
