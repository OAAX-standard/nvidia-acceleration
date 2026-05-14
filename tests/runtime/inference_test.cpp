/**
 * NVIDIA OAAX v1 runtime benchmark.
 *
 * Runs sequential inference (send → poll receive → measure latency) and
 * reports avg / min / max / p95 latency and throughput.
 *
 * Usage:
 *   ./inference_test <model.trt> [options]
 *
 * Options:
 *   --input-name  NAME     tensor name (default: input)
 *   --input-shape N,N,...  comma-separated dims (default: 1,3,640,640)
 *   --runs        N        benchmark iterations (default: 100)
 *   --warmup      N        warmup iterations (default: 10)
 *   --log-level   N        0=trace … 4=err (default: 3)
 *   --no-validate          skip output tensor sanity check
 */

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

extern "C" {
#include "tensors_struct.h"
}

extern "C" {
    int  runtime_initialization();
    int  runtime_initialization_with_args(int length, char** keys, void** values);
    int  runtime_model_loading(const char* path);
    int  send_input(tensors_struct* input);
    int  receive_output(tensors_struct** output);
    int  runtime_destruction();
    const char* runtime_error_message();
    const char* runtime_version();
    const char* runtime_name();
}

using Clock = std::chrono::steady_clock;
using Ms    = std::chrono::duration<double, std::milli>;

#define CHECK(cond, msg) \
    do { if (!(cond)) { std::cerr << "FAIL: " << (msg) << std::endl; runtime_destruction(); return 1; } } while (0)

static tensors_struct* make_input(const char* name, const std::vector<size_t>& shape) {
    tensors_struct* ts = (tensors_struct*)malloc(sizeof(tensors_struct));
    ts->num_tensors = 1;

    ts->names = (char**)malloc(sizeof(char*));
    ts->names[0] = strdup(name);

    ts->data_types = (tensor_data_type*)malloc(sizeof(tensor_data_type));
    ts->data_types[0] = DATA_TYPE_FLOAT;

    ts->ranks = (size_t*)malloc(sizeof(size_t));
    ts->ranks[0] = shape.size();

    ts->shapes = (size_t**)malloc(sizeof(size_t*));
    ts->shapes[0] = (size_t*)malloc(shape.size() * sizeof(size_t));
    size_t n_elems = 1;
    for (size_t i = 0; i < shape.size(); i++) {
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

static tensors_struct* poll_output(int max_polls = 500) {
    tensors_struct* out = nullptr;
    for (int i = 0; i < max_polls; i++) {
        if (receive_output(&out) == 0) return out;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    return nullptr;
}

static double percentile(std::vector<double> v, double p) {
    std::sort(v.begin(), v.end());
    size_t idx = (size_t)(p / 100.0 * (double)(v.size() - 1) + 0.5);
    return v[std::min(idx, v.size() - 1)];
}

static std::vector<size_t> parse_shape(const std::string& s) {
    std::vector<size_t> dims;
    std::istringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ','))
        dims.push_back((size_t)std::stoul(tok));
    return dims;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model.trt> [options]\n"
                  << "  --input-name  NAME\n"
                  << "  --input-shape N,N,...  (default: 1,3,640,640)\n"
                  << "  --runs        N        (default: 100)\n"
                  << "  --warmup      N        (default: 10)\n"
                  << "  --log-level   N        (default: 3)\n"
                  << "  --no-validate\n";
        return 1;
    }

    const char* model_path = argv[1];
    std::string input_name  = "input";
    std::string shape_str   = "1,3,640,640";
    int runs       = 100;
    int warmup     = 10;
    int log_level  = 3;
    bool validate  = true;

    for (int i = 2; i < argc; i++) {
        if      (strcmp(argv[i], "--input-name")  == 0 && i+1 < argc) input_name = argv[++i];
        else if (strcmp(argv[i], "--input-shape") == 0 && i+1 < argc) shape_str  = argv[++i];
        else if (strcmp(argv[i], "--runs")        == 0 && i+1 < argc) runs       = atoi(argv[++i]);
        else if (strcmp(argv[i], "--warmup")      == 0 && i+1 < argc) warmup     = atoi(argv[++i]);
        else if (strcmp(argv[i], "--log-level")   == 0 && i+1 < argc) log_level  = atoi(argv[++i]);
        else if (strcmp(argv[i], "--no-validate") == 0)               validate   = false;
    }

    auto shape = parse_shape(shape_str);

    std::cout << "=== NVIDIA OAAX Inference Benchmark ===" << std::endl;
    std::cout << "Model      : " << model_path << std::endl;
    std::cout << "Input      : " << input_name << " [" << shape_str << "]" << std::endl;
    std::cout << "Warmup     : " << warmup << " runs" << std::endl;
    std::cout << "Runs       : " << runs << std::endl << std::endl;

    // Init
    std::cout << "[1] Initializing runtime..." << std::endl;
    std::string ll_str = std::to_string(log_level);
    char* keys[]   = {(char*)"log_level"};
    void* values[] = {(void*)ll_str.c_str()};
    CHECK(runtime_initialization_with_args(1, keys, values) == 0, "runtime_initialization_with_args failed");
    std::cout << "  " << runtime_name() << " v" << runtime_version() << std::endl;

    // Load
    std::cout << "[2] Loading model..." << std::endl;
    auto t_load = Clock::now();
    CHECK(runtime_model_loading(model_path) == 0,
          std::string("runtime_model_loading failed: ") + (runtime_error_message() ? runtime_error_message() : ""));
    double load_ms = Ms(Clock::now() - t_load).count();
    std::cout << "  Loaded in " << load_ms << " ms" << std::endl;

    // Warmup
    std::cout << "[3] Warming up (" << warmup << " runs)..." << std::endl;
    for (int i = 0; i < warmup; i++) {
        tensors_struct* inp = make_input(input_name.c_str(), shape);
        CHECK(send_input(inp) == 0, "send_input failed during warmup");
        tensors_struct* out = poll_output();
        CHECK(out != nullptr, "receive_output timed out during warmup");
        free_tensors(out);
    }
    std::cout << "  Done" << std::endl;

    // Benchmark
    std::cout << "[4] Benchmarking (" << runs << " runs)..." << std::endl;
    std::vector<double> latencies(runs);
    auto bench_start = Clock::now();

    for (int i = 0; i < runs; i++) {
        tensors_struct* inp = make_input(input_name.c_str(), shape);
        auto t0 = Clock::now();
        CHECK(send_input(inp) == 0, "send_input failed during benchmark");
        tensors_struct* out = poll_output();
        CHECK(out != nullptr, "receive_output timed out during benchmark");
        latencies[i] = Ms(Clock::now() - t0).count();

        if (validate && i == 0) {
            CHECK(out->num_tensors > 0, "output has no tensors");
        }
        free_tensors(out);
    }

    double bench_ms = Ms(Clock::now() - bench_start).count();
    double avg = std::accumulate(latencies.begin(), latencies.end(), 0.0) / latencies.size();
    double mn  = *std::min_element(latencies.begin(), latencies.end());
    double mx  = *std::max_element(latencies.begin(), latencies.end());
    double p95 = percentile(latencies, 95.0);
    double fps = runs * 1000.0 / bench_ms;

    std::cout << "\n=== Results ===" << std::endl;
    std::cout << "  Load time  : " << load_ms << " ms" << std::endl;
    std::cout << "  Avg latency: " << avg << " ms" << std::endl;
    std::cout << "  Min latency: " << mn  << " ms" << std::endl;
    std::cout << "  Max latency: " << mx  << " ms" << std::endl;
    std::cout << "  p95 latency: " << p95 << " ms" << std::endl;
    std::cout << "  Throughput : " << fps << " img/s" << std::endl;

    // Cleanup
    std::cout << "\n[5] Cleaning up..." << std::endl;
    CHECK(runtime_destruction() == 0, "runtime_destruction failed");
    std::cout << "  Done" << std::endl;

    return 0;
}
