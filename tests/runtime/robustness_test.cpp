/**
 * Robustness test for the NVIDIA OAAX v2 runtime API.
 *
 * Covers the failure modes from the Windows runtime audit: a shared library
 * must never crash or terminate its host, whatever the config or environment.
 * The final check exits WITHOUT runtime_cleanup() on purpose — a crash during
 * static destruction turns this test's exit code non-zero.
 *
 * Usage: ./robustness_test [model.onnx]
 *
 * The model path is optional but exercises the strongest check ([5]): without
 * it, model_states/output_names stay empty for the whole test and several
 * real bugs (e.g. the get_output_names() ORT-allocator/free() mismatch that
 * crashed mediaserver.exe on Windows) never get touched.
 */
#include <cstring>
#include <iostream>
#include <string>
#include "interface.h"

#define PASS(msg)  std::cout << "  PASS: " << (msg) << std::endl
#define FAIL(msg)  do { std::cerr << "  FAIL: " << (msg) << std::endl; runtime_cleanup(); return 1; } while (0)
#define ASSERT(cond, msg) do { if (!(cond)) FAIL(msg); } while (0)

int main(int argc, char** argv) {
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

    std::cout << "\n[5] load -> cleanup -> re-init -> reload must not corrupt the heap" << std::endl;
    if (argc > 1) {
        // Runtime is already initialized (empty) from check [4]; load a real
        // model so output_names is actually populated, then cleanup, then
        // re-init and reload WITHOUT unloading the library -- the exact
        // "destroy and initialize" sequence that crashed mediaserver.exe on
        // Windows via get_output_names() releasing an ORT-allocator-owned
        // string that stop_and_clear_models() then freed with plain free().
        ModelConfig mc{argv[1], nullptr, 0, {0, nullptr, nullptr}};
        ASSERT(runtime_load_models(1, &mc) == RUNTIME_STATUS_SUCCESS,
               std::string("initial model load failed: ") + (runtime_get_error() ? runtime_get_error() : ""));
        ASSERT(runtime_cleanup() == RUNTIME_STATUS_SUCCESS, "cleanup after real load failed");

        Config good{0, nullptr, nullptr};
        ASSERT(runtime_init(good) == RUNTIME_STATUS_SUCCESS, "re-init after load+cleanup failed");
        ASSERT(runtime_load_models(1, &mc) == RUNTIME_STATUS_SUCCESS,
               std::string("reload after re-init failed: ") + (runtime_get_error() ? runtime_get_error() : ""));
        PASS("two load/cleanup/re-init cycles survived with a real model");
        // Deliberately left initialized with a model loaded (inference
        // thread running) for check [6] below -- a stronger exit-without-
        // cleanup test than an empty runtime.
    } else {
        std::cout << "  Skipped (no model path provided). Usage: " << argv[0] << " [model.onnx]" << std::endl;
    }

    std::cout << "\n[6] exiting WITHOUT runtime_cleanup() must not crash the host" << std::endl;
    std::cout << "    (exit code of this test is the assertion)" << std::endl;
    std::cout << "\n=== All robustness tests passed ===" << std::endl;
    return 0;   // deliberately no runtime_cleanup()
}
