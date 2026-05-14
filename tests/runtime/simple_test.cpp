/**
 * Smoke test for the NVIDIA OAAX v1 runtime API.
 *
 * Verifies:
 * 1. runtime_initialization() succeeds
 * 2. runtime_name() / runtime_version() return non-empty strings
 * 3. runtime_model_loading() fails gracefully for a non-existent path
 * 4. runtime_error_message() is set after failure
 * 5. Model loading + round-trip inference (optional, pass .trt path as arg)
 * 6. runtime_destruction() is idempotent
 *
 * Usage:
 *   ./simple_test [model.trt]
 */

#include <cstring>
#include <iostream>
#include <thread>
#include <chrono>

extern "C" {
#include "tensors_struct.h"
}

extern "C" {
    int  runtime_initialization();
    int  runtime_model_loading(const char* path);
    int  send_input(tensors_struct* input);
    int  receive_output(tensors_struct** output);
    int  runtime_destruction();
    const char* runtime_error_message();
    const char* runtime_version();
    const char* runtime_name();
}

#define PASS(msg)  std::cout << "  PASS: " << (msg) << std::endl
#define FAIL(msg)  do { std::cerr << "  FAIL: " << (msg) << std::endl; runtime_destruction(); return 1; } while (0)
#define ASSERT(cond, msg) do { if (!(cond)) FAIL(msg); } while (0)

static tensors_struct* make_float_input(const char* name, size_t* shape, size_t rank) {
    tensors_struct* ts = (tensors_struct*)malloc(sizeof(tensors_struct));
    ts->num_tensors = 1;

    ts->names = (char**)malloc(sizeof(char*));
    ts->names[0] = strdup(name);

    ts->data_types = (tensor_data_type*)malloc(sizeof(tensor_data_type));
    ts->data_types[0] = DATA_TYPE_FLOAT;

    ts->ranks = (size_t*)malloc(sizeof(size_t));
    ts->ranks[0] = rank;

    ts->shapes = (size_t**)malloc(sizeof(size_t*));
    ts->shapes[0] = (size_t*)malloc(rank * sizeof(size_t));
    size_t n_elems = 1;
    for (size_t i = 0; i < rank; i++) {
        ts->shapes[0][i] = shape[i];
        n_elems *= shape[i];
    }

    ts->data = (void**)malloc(sizeof(void*));
    float* buf = (float*)calloc(n_elems, sizeof(float));
    for (size_t i = 0; i < n_elems; i++) buf[i] = 0.5f;
    ts->data[0] = buf;

    return ts;
}

static void free_tensors(tensors_struct* ts) {
    if (!ts) return;
    for (size_t i = 0; i < ts->num_tensors; i++) {
        free(ts->names[i]);
        free(ts->shapes[i]);
        free(ts->data[i]);
    }
    free(ts->names);
    free(ts->data_types);
    free(ts->ranks);
    free(ts->shapes);
    free(ts->data);
    free(ts);
}

int main(int argc, char** argv) {
    std::cout << "=== NVIDIA OAAX v1 Runtime Smoke Test ===" << std::endl;

    // 1. Init
    std::cout << "\n[1] runtime_initialization()" << std::endl;
    ASSERT(runtime_initialization() == 0, "runtime_initialization failed");
    PASS("runtime initialized");

    // 2. Version / name
    std::cout << "\n[2] runtime_name() / runtime_version()" << std::endl;
    const char* name = runtime_name();
    const char* ver  = runtime_version();
    ASSERT(name && strlen(name) > 0, "name is empty");
    ASSERT(ver  && strlen(ver)  > 0, "version is empty");
    std::cout << "  name: " << name << "  version: " << ver << std::endl;
    PASS("name and version returned");

    // 3. Load non-existent model
    std::cout << "\n[3] runtime_model_loading() with bad path" << std::endl;
    ASSERT(runtime_model_loading("/nonexistent/path/model.trt") != 0, "load should fail for bad path");
    const char* err = runtime_error_message();
    ASSERT(err && strlen(err) > 0, "runtime_error_message() should be set after failure");
    std::cout << "  error: " << err << std::endl;
    PASS("bad-path load failed with error message");

    // 4. Optional: load real model and run inference
    if (argc > 1) {
        std::cout << "\n[4] runtime_model_loading(): " << argv[1] << std::endl;

        // Re-init after failed load
        runtime_destruction();
        ASSERT(runtime_initialization() == 0, "re-init failed");

        ASSERT(runtime_model_loading(argv[1]) == 0,
               std::string("model load failed: ") + (runtime_error_message() ? runtime_error_message() : ""));
        PASS("model loaded");

        std::cout << "\n[5] send_input() + receive_output()" << std::endl;
        size_t shape[] = {1, 4};
        tensors_struct* input = make_float_input("input", shape, 2);
        ASSERT(send_input(input) == 0, "send_input failed");
        PASS("send_input accepted");

        tensors_struct* output = nullptr;
        int poll = 0;
        while (receive_output(&output) != 0 && poll++ < 200) {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        ASSERT(output != nullptr, "receive_output timed out after 200 polls");
        std::cout << "  output: " << output->num_tensors << " tensor(s), polled " << poll << " times" << std::endl;
        free_tensors(output);
        PASS("round-trip inference OK");
    } else {
        std::cout << "\n[4] Skipping model load (no path provided)" << std::endl;
        std::cout << "    Usage: " << argv[0] << " <model.trt>" << std::endl;
    }

    // 6. Cleanup (idempotent)
    std::cout << "\n[6] runtime_destruction()" << std::endl;
    ASSERT(runtime_destruction() == 0, "runtime_destruction failed");
    ASSERT(runtime_destruction() == 0, "second runtime_destruction should succeed (idempotent)");
    PASS("destruction OK (idempotent)");

    std::cout << "\n=== All tests passed ===" << std::endl;
    return 0;
}
