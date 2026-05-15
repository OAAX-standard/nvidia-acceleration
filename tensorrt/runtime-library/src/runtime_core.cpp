#include <atomic>
#include <chrono>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <NvInfer.h>
#include <cuda_runtime.h>
#include "concurrentqueue.h"
#include <spdlog/spdlog.h>

#include "runtime_core.hpp"
#include "runtime_utils.hpp"

using namespace std;

// TRT ILogger forwarding to spdlog
class TrtLogger : public nvinfer1::ILogger {
public:
    void log(Severity sev, const char *msg) noexcept override;
};
static TrtLogger trt_logger;

// Per-tensor info for one model
struct TensorInfo {
    string name;
    bool is_input;
    nvinfer1::Dims dims;
    nvinfer1::DataType dtype;
    size_t byte_size;
    void *gpu_buf = nullptr;
};

// Per-model runtime state
struct ModelState {
    int model_id;
    nvinfer1::ICudaEngine *engine = nullptr;
    nvinfer1::IExecutionContext *context = nullptr;
    cudaStream_t stream = nullptr;
    vector<TensorInfo> tensor_infos;
    vector<size_t> input_indices;
    vector<size_t> output_indices;
    moodycamel::ConcurrentQueue<Tensors *> input_queue;
    thread inference_thread;
    atomic<bool> stop{false};
};

static bool initialized = false;
static string last_error_msg;
static int log_level_val = spdlog::level::info;
static string log_file_val = "runtime.log";

static nvinfer1::IRuntime *trt_runtime = nullptr;
static vector<unique_ptr<ModelState>> model_states;
static moodycamel::ConcurrentQueue<Tensors *> output_queue;

shared_ptr<spdlog::logger> logger;

static void set_error(const string &msg) {
    last_error_msg = msg;
    if (logger) logger->error("{}", msg);
}

void TrtLogger::log(Severity sev, const char *msg) noexcept {
    if (!logger) return;
    switch (sev) {
    case Severity::kINTERNAL_ERROR:
    case Severity::kERROR:   logger->error("[TRT] {}", msg); break;
    case Severity::kWARNING: logger->warn("[TRT] {}", msg);  break;
    case Severity::kINFO:    logger->info("[TRT] {}", msg);  break;
    case Severity::kVERBOSE: logger->trace("[TRT] {}", msg); break;
    }
}

static void destroy_model_state(ModelState *ms) {
    if (!ms) return;
    ms->stop = true;
    if (ms->inference_thread.joinable()) ms->inference_thread.join();
    free_queue(ms->input_queue);
    for (auto &ti : ms->tensor_infos)
        if (ti.gpu_buf) { cudaFree(ti.gpu_buf); ti.gpu_buf = nullptr; }
    if (ms->stream)  { cudaStreamDestroy(ms->stream); ms->stream = nullptr; }
    if (ms->context) { delete ms->context; ms->context = nullptr; }
    if (ms->engine)  { delete ms->engine;  ms->engine  = nullptr; }
}

static void inference_thread_func(ModelState *ms)
{
    while (!ms->stop) {
        Tensors *input = nullptr;
        if (!ms->input_queue.try_dequeue(input)) {
            this_thread::sleep_for(chrono::milliseconds(5));
            continue;
        }
        logger->debug("Model {}: dequeued input id={}", ms->model_id, input->id);

        vector<void *> host_out_bufs(ms->output_indices.size(), nullptr);
        int req_id = input->id;

        try {
            if ((size_t)input->num_tensors != ms->input_indices.size())
                throw runtime_error("Input tensor count mismatch");

            for (size_t i = 0; i < ms->output_indices.size(); ++i) {
                host_out_bufs[i] = malloc(ms->tensor_infos[ms->output_indices[i]].byte_size);
                if (!host_out_bufs[i]) throw runtime_error("malloc failed for output buffer");
            }

            for (size_t i = 0; i < ms->input_indices.size(); ++i) {
                TensorInfo &ti = ms->tensor_infos[ms->input_indices[i]];
                if (cudaMemcpyAsync(ti.gpu_buf, input->tensors[i].data, ti.byte_size,
                                    cudaMemcpyHostToDevice, ms->stream) != cudaSuccess)
                    throw runtime_error("cudaMemcpyAsync H2D failed");
            }

            for (auto &ti : ms->tensor_infos)
                ms->context->setTensorAddress(ti.name.c_str(), ti.gpu_buf);

            if (!ms->context->enqueueV3(ms->stream))
                throw runtime_error("TRT enqueueV3 failed");

            for (size_t i = 0; i < ms->output_indices.size(); ++i) {
                TensorInfo &ti = ms->tensor_infos[ms->output_indices[i]];
                if (cudaMemcpyAsync(host_out_bufs[i], ti.gpu_buf, ti.byte_size,
                                    cudaMemcpyDeviceToHost, ms->stream) != cudaSuccess)
                    throw runtime_error("cudaMemcpyAsync D2H failed");
            }

            if (cudaStreamSynchronize(ms->stream) != cudaSuccess)
                throw runtime_error("cudaStreamSynchronize failed");

            deep_free_tensors(input);
            input = nullptr;

            size_t n_out = ms->output_indices.size();
            Tensors *out = (Tensors *)malloc(sizeof(Tensors));
            out->id = req_id;
            out->num_tensors = (int)n_out;
            out->tensors = (TensorDescriptor *)malloc(n_out * sizeof(TensorDescriptor));

            for (size_t i = 0; i < n_out; ++i) {
                TensorInfo &ti = ms->tensor_infos[ms->output_indices[i]];
                TensorDescriptor &td = out->tensors[i];

                td.name = strdup(ti.name.c_str());

                td.data_type = map_trt_to_oaax_type(ti.dtype);
                td.rank = ti.dims.nbDims;
                td.shape = (int *)malloc(td.rank * sizeof(int));
                for (int d = 0; d < td.rank; ++d)
                    td.shape[d] = (int)ti.dims.d[d];

                td.data_size = ti.byte_size;
                td.data = host_out_bufs[i];
                host_out_bufs[i] = nullptr;
            }

            if (!output_queue.try_enqueue(out)) {
                logger->error("Failed to enqueue output");
                deep_free_tensors(out);
            }
        } catch (const exception &e) {
            if (input) deep_free_tensors(input);
            for (void *buf : host_out_bufs) free(buf);
            logger->error("Inference error: {}", e.what());
        }
    }
}

extern "C" RuntimeStatus runtime_init(Config config)
{
    if (initialized) {
        set_error("Runtime already initialized");
        return RUNTIME_STATUS_ALREADY_INITIALIZED;
    }

    for (int i = 0; i < config.length; ++i) {
        string key = config.keys[i];
        string val = config.values[i];
        if (key == "log_level") {
            int v = stoi(val);
            log_level_val = (v >= spdlog::level::trace && v <= spdlog::level::off) ? v : spdlog::level::info;
        } else if (key == "log_file") {
            log_file_val = val;
        }
    }

    try {
        logger = initialize_logger(log_file_val, log_level_val, log_level_val, runtime_get_name());
        logger->info("Initializing TensorRT runtime");
        trt_runtime = nvinfer1::createInferRuntime(trt_logger);
        if (!trt_runtime) throw runtime_error("Failed to create TRT IRuntime");
        initialized = true;
        logger->info("Runtime initialized (log_level={})", log_level_val);
        return RUNTIME_STATUS_SUCCESS;
    } catch (const exception &e) {
        set_error(string("runtime_init failed: ") + e.what());
        return RUNTIME_STATUS_ERROR;
    }
}

extern "C" RuntimeStatus runtime_load_models(int num_models, const ModelConfig *model_configs)
{
    if (!initialized) return RUNTIME_STATUS_NOT_INITIALIZED;
    if (num_models <= 0 || !model_configs) return RUNTIME_STATUS_INVALID_ARGUMENT;

    for (auto &ms : model_states) destroy_model_state(ms.get());
    model_states.clear();
    free_queue(output_queue);

    try {
        for (int i = 0; i < num_models; ++i) {
            const ModelConfig &mc = model_configs[i];
            auto ms = make_unique<ModelState>();
            ms->model_id = i;

            vector<char> engine_data;
            if (mc.file_path) {
                ifstream file(mc.file_path, ios::binary | ios::ate);
                if (!file.is_open()) {
                    set_error(string("Cannot open engine file: ") + mc.file_path);
                    for (auto &s : model_states) destroy_model_state(s.get());
                    model_states.clear();
                    return RUNTIME_STATUS_FILE_NOT_FOUND;
                }
                size_t sz = (size_t)file.tellg();
                file.seekg(0, ios::beg);
                engine_data.resize(sz);
                file.read(engine_data.data(), (streamsize)sz);
                ms->engine = trt_runtime->deserializeCudaEngine(engine_data.data(), sz);
            } else if (mc.model_data && mc.model_size > 0) {
                ms->engine = trt_runtime->deserializeCudaEngine(mc.model_data, mc.model_size);
            } else {
                set_error("ModelConfig must have file_path or model_data");
                for (auto &s : model_states) destroy_model_state(s.get());
                model_states.clear();
                return RUNTIME_STATUS_INVALID_ARGUMENT;
            }

            if (!ms->engine) throw runtime_error("Failed to deserialize TRT engine");
            ms->context = ms->engine->createExecutionContext();
            if (!ms->context) throw runtime_error("Failed to create execution context");
            if (cudaStreamCreate(&ms->stream) != cudaSuccess) throw runtime_error("cudaStreamCreate failed");

            int nb = ms->engine->getNbIOTensors();
            ms->tensor_infos.resize((size_t)nb);
            for (int j = 0; j < nb; ++j) {
                const char *name = ms->engine->getIOTensorName(j);
                TensorInfo &ti = ms->tensor_infos[(size_t)j];
                ti.name = name;
                ti.is_input = (ms->engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT);
                ti.dims = ms->engine->getTensorShape(name);
                ti.dtype = ms->engine->getTensorDataType(name);
                ti.byte_size = compute_tensor_byte_size(ti.dims, ti.dtype);
                if (cudaMalloc(&ti.gpu_buf, ti.byte_size) != cudaSuccess)
                    throw runtime_error(string("cudaMalloc failed for tensor: ") + name);
                if (ti.is_input)
                    ms->input_indices.push_back((size_t)j);
                else
                    ms->output_indices.push_back((size_t)j);
            }

            logger->info("Model {}: {} inputs, {} outputs", i, ms->input_indices.size(), ms->output_indices.size());
            ms->inference_thread = thread(inference_thread_func, ms.get());
            model_states.push_back(move(ms));
        }
        return RUNTIME_STATUS_SUCCESS;
    } catch (const exception &e) {
        for (auto &ms : model_states) destroy_model_state(ms.get());
        model_states.clear();
        set_error(string("runtime_load_models failed: ") + e.what());
        return RUNTIME_STATUS_ERROR;
    }
}

extern "C" RuntimeStatus runtime_enqueue_input(int model_id, Tensors *input_tensors)
{
    if (!initialized) return RUNTIME_STATUS_NOT_INITIALIZED;
    if (model_id < 0 || (size_t)model_id >= model_states.size()) return RUNTIME_STATUS_INVALID_MODEL_ID;
    if (!input_tensors) return RUNTIME_STATUS_INVALID_ARGUMENT;

    input_tensors->id = model_id;

    if (!model_states[(size_t)model_id]->input_queue.try_enqueue(input_tensors)) {
        set_error("Failed to enqueue input tensors");
        return RUNTIME_STATUS_ERROR;
    }
    return RUNTIME_STATUS_SUCCESS;
}

extern "C" RuntimeStatus runtime_retrieve_output(int *model_id, Tensors **output_tensors, int timeout_ms)
{
    if (!initialized) return RUNTIME_STATUS_NOT_INITIALIZED;
    if (!model_id || !output_tensors) return RUNTIME_STATUS_INVALID_ARGUMENT;

    if (timeout_ms == 0) {
        if (!output_queue.try_dequeue(*output_tensors))
            return RUNTIME_STATUS_NO_OUTPUT_AVAILABLE;
        *model_id = (*output_tensors)->id;
        return RUNTIME_STATUS_SUCCESS;
    }

    auto start = chrono::steady_clock::now();
    while (true) {
        if (output_queue.try_dequeue(*output_tensors)) {
            *model_id = (*output_tensors)->id;
            return RUNTIME_STATUS_SUCCESS;
        }
        if (timeout_ms > 0) {
            auto elapsed = chrono::duration_cast<chrono::milliseconds>(
                chrono::steady_clock::now() - start).count();
            if (elapsed >= timeout_ms)
                return RUNTIME_STATUS_NO_OUTPUT_AVAILABLE;
        }
        this_thread::sleep_for(chrono::milliseconds(1));
    }
}

extern "C" RuntimeStatus runtime_cleanup(void)
{
    for (auto &ms : model_states) destroy_model_state(ms.get());
    model_states.clear();
    free_queue(output_queue);
    if (trt_runtime) { delete trt_runtime; trt_runtime = nullptr; }
    initialized = false;
    last_error_msg.clear();
    if (logger) {
        logger->info("Runtime cleaned up");
        destroy_logger(logger);
        logger = nullptr;
    }
    return RUNTIME_STATUS_SUCCESS;
}

extern "C" const char *runtime_get_error(void)
{
    return last_error_msg.empty() ? nullptr : last_error_msg.c_str();
}

extern "C" const char *runtime_get_version(void)
{
    return RUNTIME_VERSION;
}

extern "C" const char *runtime_get_name(void)
{
    return "OAAX NVIDIA TensorRT";
}

extern "C" const char *runtime_get_info(void)
{
    if (!initialized) return nullptr;
    static char buf[256];
    snprintf(buf, sizeof(buf),
             "{\"loaded_models\":%zu,\"requests_in_flight\":%zu}",
             model_states.size(), output_queue.size_approx());
    return buf;
}
