/**
 * Robustness test for the NVIDIA OAAX v2 runtime API.
 *
 * Covers the failure modes from the Windows runtime audit: a shared library
 * must never crash or terminate its host, whatever the config or environment.
 * The final check exits WITHOUT runtime_cleanup() on purpose — a crash during
 * static destruction turns this test's exit code non-zero.
 *
 * Usage: ./robustness_test
 */
#include <cstring>
#include <iostream>
#include "interface.h"

#define PASS(msg)  std::cout << "  PASS: " << (msg) << std::endl
#define FAIL(msg)  do { std::cerr << "  FAIL: " << (msg) << std::endl; runtime_cleanup(); return 1; } while (0)
#define ASSERT(cond, msg) do { if (!(cond)) FAIL(msg); } while (0)

int main() {
    std::cout << "=== NVIDIA OAAX v2 Runtime Robustness Test ===" << std::endl;

    std::cout << "\n[1] init with unwritable log_file must not kill the process" << std::endl;
    {
        const char* keys[]   = {"log_file", "log_stdout"};
        const char* values[] = {"/proc/unwritable/runtime.log", "true"};
        Config cfg{2, keys, values};
        RuntimeStatus rc = runtime_init(cfg);
        ASSERT(rc == RUNTIME_STATUS_SUCCESS, "init should fall back to console logging");
        ASSERT(runtime_cleanup() == RUNTIME_STATUS_SUCCESS, "cleanup failed");
        PASS("console-only fallback, no exit()");
    }

    std::cout << "\n[2] non-numeric log_level must return an error, not throw" << std::endl;
    {
        const char* keys[]   = {"log_level"};
        const char* values[] = {"debug"};
        Config cfg{1, keys, values};
        RuntimeStatus rc = runtime_init(cfg);
        ASSERT(rc != RUNTIME_STATUS_SUCCESS, "init should reject non-numeric log_level");
        const char* err = runtime_get_error();
        ASSERT(err && strlen(err) > 0, "error message should be retrievable");
        std::cout << "  error: " << err << std::endl;
        PASS("error status + message, no exception across the C boundary");
    }

    std::cout << "\n[3] null keys/values with non-zero length must be rejected" << std::endl;
    {
        Config cfg{2, nullptr, nullptr};
        ASSERT(runtime_init(cfg) == RUNTIME_STATUS_INVALID_ARGUMENT, "expected INVALID_ARGUMENT");
        PASS("invalid Config rejected");
    }

    std::cout << "\n[4] a failed init must not poison the next init" << std::endl;
    {
        Config good{0, nullptr, nullptr};
        RuntimeStatus rc = runtime_init(good);
        ASSERT(rc == RUNTIME_STATUS_SUCCESS, "init after failed init should succeed");
        PASS("clean init after failed attempts");
    }

    std::cout << "\n[5] exiting WITHOUT runtime_cleanup() must not crash the host" << std::endl;
    std::cout << "    (exit code of this test is the assertion)" << std::endl;
    std::cout << "\n=== All robustness tests passed ===" << std::endl;
    return 0;   // deliberately no runtime_cleanup()
}
