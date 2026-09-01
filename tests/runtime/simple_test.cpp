/**
 * Smoke test for the NVIDIA OAAX v2 runtime API.
 *
 * Usage: ./simple_test [model.trt|model.onnx]
 */
#include <cstdlib>
#include <cstring>
#include <iostream>
#include "interface.h"

#define PASS(msg)  std::cout << "  PASS: " << (msg) << std::endl
#define FAIL(msg)  do { std::cerr << "  FAIL: " << (msg) << std::endl; runtime_cleanup(); return 1; } while (0)
#define ASSERT(cond, msg) do { if (!(cond)) FAIL(msg); } while (0)

static Tensors* make_float_input(const char* name, int* shape, int rank, int request_id) {
    Tensors* ts = (Tensors*)malloc(sizeof(Tensors));
    ts->id = request_id;
    ts->num_tensors = 1;
    ts->tensors = (TensorDescriptor*)malloc(sizeof(TensorDescriptor));
    TensorDescriptor& td = ts->tensors[0];
    td.name = strdup(name);
    td.data_type = DATA_TYPE_FLOAT;
    td.rank = rank;
    td.shape = (int*)malloc(rank * sizeof(int));
    size_t n_elems = 1;
    for (int i = 0; i < rank; i++) { td.shape[i] = shape[i]; n_elems *= (size_t)shape[i]; }
    td.data_size = n_elems * sizeof(float);
    float* buf = (float*)malloc(td.data_size);
    for (size_t i = 0; i < n_elems; i++) buf[i] = 0.5f;
    td.data = buf;
    return ts;
}

static void free_tensors(Tensors* ts) {
    if (!ts) return;
    for (int i = 0; i < ts->num_tensors; i++) { free(ts->tensors[i].name); free(ts->tensors[i].shape); free(ts->tensors[i].data); }
    free(ts->tensors); free(ts);
}

int main(int argc, char** argv) {
    std::cout << "=== NVIDIA OAAX v2 Runtime Smoke Test ===" << std::endl;

    std::cout << "\n[1] runtime_init()" << std::endl;
    Config cfg{0, nullptr, nullptr};
    ASSERT(runtime_init(cfg) == RUNTIME_STATUS_SUCCESS, "runtime_init failed");
    PASS("runtime initialized");

    std::cout << "\n[2] runtime_get_name() / runtime_get_version()" << std::endl;
    const char* name = runtime_get_name();
    const char* ver  = runtime_get_version();
    ASSERT(name && strlen(name) > 0, "name is empty");
    ASSERT(ver  && strlen(ver)  > 0, "version is empty");
    std::cout << "  name: " << name << "  version: " << ver << std::endl;
    PASS("name and version returned");

    std::cout << "\n[3] runtime_get_info()" << std::endl;
    const char* info = runtime_get_info();
    if (info) std::cout << "  info: " << info << std::endl;
    PASS("runtime_get_info did not crash");

    std::cout << "\n[4] runtime_load_models() with bad path" << std::endl;
    ModelConfig bad_mc{"/nonexistent/path/model.trt", nullptr, 0, {0, nullptr, nullptr}};
    ASSERT(runtime_load_models(1, &bad_mc) != RUNTIME_STATUS_SUCCESS, "load should fail for bad path");
    const char* err = runtime_get_error();
    ASSERT(err && strlen(err) > 0, "runtime_get_error() should be set after failure");
    std::cout << "  error: " << err << std::endl;
    PASS("bad-path load failed with error message");

    if (argc > 1) {
        std::cout << "\n[5] runtime_load_models(): " << argv[1] << std::endl;
        ModelConfig mc{argv[1], nullptr, 0, {0, nullptr, nullptr}};
        ASSERT(runtime_load_models(1, &mc) == RUNTIME_STATUS_SUCCESS,
               std::string("model load failed: ") + (runtime_get_error() ? runtime_get_error() : ""));
        PASS("model loaded");

        std::cout << "\n[6] runtime_enqueue_input() + runtime_retrieve_output()" << std::endl;
        int shape[] = {1, 4};
        Tensors* input = make_float_input("input", shape, 2, 42);
        ASSERT(runtime_enqueue_input(0, input) == RUNTIME_STATUS_SUCCESS, "runtime_enqueue_input failed");
        PASS("runtime_enqueue_input accepted");

        Tensors* output = nullptr;
        int out_model_id = -1;
        RuntimeStatus status = RUNTIME_STATUS_NO_OUTPUT_AVAILABLE;
        for (int poll = 0; poll < 200 && status != RUNTIME_STATUS_SUCCESS; poll++)
            status = runtime_retrieve_output(&out_model_id, &output, 50);
        ASSERT(status == RUNTIME_STATUS_SUCCESS, "runtime_retrieve_output timed out");
        ASSERT(output != nullptr, "output is null");
        ASSERT(out_model_id == 0, "model_id mismatch");
        std::cout << "  output: " << output->num_tensors << " tensor(s), model_id=" << out_model_id << std::endl;
        free_tensors(output);
        PASS("round-trip inference OK");
    } else {
        std::cout << "\n[5] Skipping model load (no path provided)" << std::endl;
        std::cout << "    Usage: " << argv[0] << " <model.trt|model.onnx>" << std::endl;
    }

    std::cout << "\n[7] runtime_cleanup()" << std::endl;
    ASSERT(runtime_cleanup() == RUNTIME_STATUS_SUCCESS, "runtime_cleanup failed");
    ASSERT(runtime_cleanup() == RUNTIME_STATUS_SUCCESS, "second runtime_cleanup should succeed (idempotent)");
    PASS("cleanup OK (idempotent)");

    std::cout << "\n=== All tests passed ===" << std::endl;
    return 0;
}
