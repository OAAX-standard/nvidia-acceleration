/**
 * NVIDIA OAAX v2 runtime benchmark.
 *
 * Usage: ./inference_test <model.trt|model.onnx> [options]
 *
 * Options:
 *   --input-name  NAME     tensor name (default: input)
 *   --input-shape N,N,...  comma-separated dims (default: 1,3,640,640)
 *   --runs        N        benchmark iterations (default: 100)
 *   --warmup      N        warmup iterations (default: 10)
 *   --in-flight   N        number of requests in flight (default: 1)
 *   --log-level   N        0=trace ... 4=err (default: 3)
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
#include <vector>
#include "interface.h"

using Clock = std::chrono::steady_clock;
using Ms    = std::chrono::duration<double, std::milli>;

#define CHECK(cond, msg) \
    do { if (!(cond)) { std::cerr << "FAIL: " << (msg) << std::endl; runtime_cleanup(); return 1; } } while (0)

static Tensors* make_input(const char* name, const std::vector<int>& shape, int request_id) {
    Tensors* ts = (Tensors*)malloc(sizeof(Tensors));
    ts->id = request_id;
    ts->num_tensors = 1;
    ts->tensors = (TensorDescriptor*)malloc(sizeof(TensorDescriptor));
    TensorDescriptor& td = ts->tensors[0];
    td.name = strdup(name);
    td.data_type = DATA_TYPE_FLOAT;
    td.rank = (int)shape.size();
    td.shape = (int*)malloc(shape.size() * sizeof(int));
    size_t n_elems = 1;
    for (size_t i = 0; i < shape.size(); i++) { td.shape[i] = shape[i]; n_elems *= (size_t)shape[i]; }
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

static Tensors* poll_output(int timeout_per_poll_ms = 50, int max_polls = 200) {
    Tensors* out = nullptr;
    int model_id = -1;
    for (int i = 0; i < max_polls; i++)
        if (runtime_retrieve_output(&model_id, &out, timeout_per_poll_ms) == RUNTIME_STATUS_SUCCESS) return out;
    return nullptr;
}

static double percentile(std::vector<double> v, double p) {
    std::sort(v.begin(), v.end());
    size_t idx = (size_t)(p / 100.0 * (double)(v.size() - 1) + 0.5);
    return v[std::min(idx, v.size() - 1)];
}

static std::vector<int> parse_shape(const std::string& s) {
    std::vector<int> dims; std::istringstream ss(s); std::string tok;
    while (std::getline(ss, tok, ',')) dims.push_back(std::stoi(tok));
    return dims;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model.trt|model.onnx> [options]\n"
                  << "  --input-name NAME  --input-shape N,N,...  --runs N  --warmup N"
                  << "  --in-flight N  --log-level N  --no-validate\n";
        return 1;
    }
    const char* model_path = argv[1];
    std::string input_name = "input", shape_str = "1,3,640,640";
    int runs = 100, warmup = 10, log_level = 3, in_flight = 1;
    bool validate = true;
    for (int i = 2; i < argc; i++) {
        if      (strcmp(argv[i], "--input-name")  == 0 && i+1 < argc) input_name = argv[++i];
        else if (strcmp(argv[i], "--input-shape") == 0 && i+1 < argc) shape_str  = argv[++i];
        else if (strcmp(argv[i], "--runs")        == 0 && i+1 < argc) runs       = atoi(argv[++i]);
        else if (strcmp(argv[i], "--warmup")      == 0 && i+1 < argc) warmup     = atoi(argv[++i]);
        else if (strcmp(argv[i], "--in-flight")   == 0 && i+1 < argc) in_flight  = atoi(argv[++i]);
        else if (strcmp(argv[i], "--log-level")   == 0 && i+1 < argc) log_level  = atoi(argv[++i]);
        else if (strcmp(argv[i], "--no-validate") == 0)               validate   = false;
    }
    if (in_flight < 1) in_flight = 1;
    auto shape = parse_shape(shape_str);

    std::cout << "=== NVIDIA OAAX v2 Inference Benchmark ===" << std::endl;
    std::cout << "Model: " << model_path << "  Input: " << input_name << " [" << shape_str << "]" << std::endl;
    std::cout << "Warmup: " << warmup << "  Runs: " << runs << "  In-flight: " << in_flight << std::endl << std::endl;

    std::cout << "[1] Initializing runtime..." << std::endl;
    std::string ll_str = std::to_string(log_level);
    const char* keys[]   = {"log_level"};
    const char* values[] = {ll_str.c_str()};
    Config cfg{1, keys, values};
    CHECK(runtime_init(cfg) == RUNTIME_STATUS_SUCCESS, "runtime_init failed");
    std::cout << "  " << runtime_get_name() << " v" << runtime_get_version() << std::endl;

    std::cout << "[2] Loading model..." << std::endl;
    auto t_load = Clock::now();
    ModelConfig mc{model_path, nullptr, 0, {0, nullptr, nullptr}};
    CHECK(runtime_load_models(1, &mc) == RUNTIME_STATUS_SUCCESS,
          std::string("runtime_load_models failed: ") + (runtime_get_error() ? runtime_get_error() : ""));
    double load_ms = Ms(Clock::now() - t_load).count();
    std::cout << "  Loaded in " << load_ms << " ms" << std::endl;

    std::cout << "[3] Warming up (" << warmup << " runs, sequential)..." << std::endl;
    for (int i = 0; i < warmup; i++) {
        Tensors* inp = make_input(input_name.c_str(), shape, i);
        CHECK(runtime_enqueue_input(0, inp) == RUNTIME_STATUS_SUCCESS, "runtime_enqueue_input failed during warmup");
        Tensors* out = poll_output();
        CHECK(out != nullptr, "runtime_retrieve_output timed out during warmup");
        if (validate && i == 0) CHECK(out->num_tensors > 0, "output has no tensors");
        free_tensors(out);
    }
    std::cout << "  Done" << std::endl;

    std::cout << "[4] Benchmarking (" << runs << " runs, " << in_flight << " in-flight)..." << std::endl;

    // Pre-fill the pipeline
    int sent = 0;
    for (; sent < std::min(in_flight, runs); sent++) {
        Tensors* inp = make_input(input_name.c_str(), shape, sent);
        CHECK(runtime_enqueue_input(0, inp) == RUNTIME_STATUS_SUCCESS, "runtime_enqueue_input failed");
    }

    // Sliding window: dequeue one output → enqueue the next input
    std::vector<double> latencies(runs);
    auto bench_start = Clock::now();
    for (int received = 0; received < runs; received++) {
        auto t0 = Clock::now();
        Tensors* out = poll_output();
        CHECK(out != nullptr, "runtime_retrieve_output timed out");
        latencies[received] = Ms(Clock::now() - t0).count();
        free_tensors(out);

        if (sent < runs) {
            Tensors* inp = make_input(input_name.c_str(), shape, sent++);
            CHECK(runtime_enqueue_input(0, inp) == RUNTIME_STATUS_SUCCESS, "runtime_enqueue_input failed");
        }
    }
    double bench_ms = Ms(Clock::now() - bench_start).count();

    double avg = std::accumulate(latencies.begin(), latencies.end(), 0.0) / latencies.size();
    double mn  = *std::min_element(latencies.begin(), latencies.end());
    double p95 = percentile(latencies, 95.0);

    std::cout << "\n=== Results ===" << std::endl;
    std::cout << "  Load time  : " << load_ms << " ms" << std::endl;
    std::cout << "  In-flight  : " << in_flight << std::endl;
    std::cout << "  Avg latency: " << avg << " ms" << std::endl;
    std::cout << "  Min latency: " << mn  << " ms" << std::endl;
    std::cout << "  p95 latency: " << p95 << " ms" << std::endl;
    std::cout << "  Throughput : " << runs * 1000.0 / bench_ms << " img/s" << std::endl;

    std::cout << "\n[5] Cleaning up..." << std::endl;
    CHECK(runtime_cleanup() == RUNTIME_STATUS_SUCCESS, "runtime_cleanup failed");
    std::cout << "  Done" << std::endl;
    return 0;
}
