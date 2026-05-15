#include <atomic>
#include <chrono>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#endif

#include <onnxruntime_cxx_api.h>
#include "concurrentqueue.h"
#include <spdlog/spdlog.h>

#include "runtime_core.hpp"
#include "runtime_utils.hpp"

using namespace std;

// State
static bool initialized = false;
static string last_error_msg;

// ORT globals
static unique_ptr<Ort::Env> env;
static unique_ptr<Ort::SessionOptions> session_options;

// Per-model state
struct ModelState {
    unique_ptr<Ort::Session> session;
    vector<char *> output_names;
    moodycamel::ConcurrentQueue<Tensors *> input_queue;
    thread inference_thread;
    atomic<bool> stop{false};
    int model_id;
};

static vector<unique_ptr<ModelState>> model_states;
static moodycamel::ConcurrentQueue<Tensors *> output_queue;

// Config
static int log_level = spdlog::level::info;
static string log_file = "runtime.log";
static int num_threads = 4;

static shared_ptr<spdlog::logger> logger;

static void set_error(const string &msg) {
    last_error_msg = msg;
    if (logger) logger->error("{}", msg);
}

static void stop_and_clear_models() {
    for (auto &ms : model_states) {
        ms->stop = true;
        if (ms->inference_thread.joinable()) ms->inference_thread.join();
        for (auto n : ms->output_names) free(n);
        free_queue(ms->input_queue);
    }
    model_states.clear();
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

        try {
            vector<const char *> input_names;
            vector<Ort::Value> ort_inputs;

            for (int i = 0; i < input->num_tensors; ++i) {
                TensorDescriptor &td = input->tensors[i];
                input_names.push_back(td.name);
                vector<int64_t> shape(td.shape, td.shape + td.rank);
                ONNXTensorElementDataType dtype = map_to_ort_type(td.data_type);
                ort_inputs.push_back(Ort::Value::CreateTensor(
                    Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault),
                    td.data, td.data_size,
                    shape.data(), shape.size(), dtype));
            }

            int req_id = input->id;
            deep_free_tensors(input);
            input = nullptr;

            auto ort_outputs = ms->session->Run(
                Ort::RunOptions{nullptr},
                input_names.data(), ort_inputs.data(), ort_inputs.size(),
                ms->output_names.data(), ms->output_names.size());

            size_t n = ort_outputs.size();
            Tensors *out = (Tensors *)malloc(sizeof(Tensors));
            out->id = req_id;
            out->num_tensors = (int)n;
            out->tensors = (TensorDescriptor *)malloc(n * sizeof(TensorDescriptor));

            for (size_t i = 0; i < n; ++i) {
                TensorDescriptor &td = out->tensors[i];
                size_t name_len = strlen(ms->output_names[i]);
                td.name = (char *)malloc(name_len + 1);
                memcpy(td.name, ms->output_names[i], name_len + 1);

                auto info = ort_outputs[i].GetTensorTypeAndShapeInfo();
                auto shape = info.GetShape();
                td.rank = (int)shape.size();
                td.shape = (int *)malloc(td.rank * sizeof(int));
                for (int j = 0; j < td.rank; ++j)
                    td.shape[j] = (int)shape[j];

                td.data_type = map_to_tensors_struct_type(info.GetElementType());
                size_t elem_count = info.GetElementCount();
                td.data_size = elem_count * (size_t)get_element_byte_size(td.data_type);
                td.data = malloc(td.data_size);
                memcpy(td.data, ort_outputs[i].GetTensorMutableData<void>(), td.data_size);
            }

            if (!output_queue.try_enqueue(out)) {
                logger->error("Failed to enqueue output");
                deep_free_tensors(out);
            }
        } catch (const exception &e) {
            if (input) deep_free_tensors(input);
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
            log_level = (v >= spdlog::level::trace && v <= spdlog::level::off) ? v : spdlog::level::info;
        } else if (key == "log_file") {
            log_file = val;
        } else if (key == "num_threads") {
            num_threads = max(1, min(8, stoi(val)));
        }
    }

    try {
        logger = initialize_logger(log_file, log_level, log_level, runtime_get_name());
        logger->info("Initializing ORT runtime");
        env = make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, runtime_get_name());
        session_options = make_unique<Ort::SessionOptions>();
        session_options->SetIntraOpNumThreads(num_threads);
        OrtCUDAProviderOptions cuda_opts{};
        cuda_opts.device_id = 0;
        cuda_opts.arena_extend_strategy = 0;
        cuda_opts.gpu_mem_limit = SIZE_MAX;
        cuda_opts.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
        cuda_opts.do_copy_in_default_stream = 1;
        session_options->AppendExecutionProvider_CUDA(cuda_opts);
        session_options->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        initialized = true;
        logger->info("Runtime initialized (log_level={}, num_threads={})", log_level, num_threads);
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

    stop_and_clear_models();
    free_queue(output_queue);

    try {
        for (int i = 0; i < num_models; ++i) {
            const ModelConfig &mc = model_configs[i];
            auto ms = make_unique<ModelState>();
            ms->model_id = i;

#ifdef _WIN32
            if (mc.file_path) {
                int sz = MultiByteToWideChar(CP_UTF8, 0, mc.file_path, -1, NULL, 0);
                wstring wpath(sz, 0);
                MultiByteToWideChar(CP_UTF8, 0, mc.file_path, -1, &wpath[0], sz);
                ms->session = make_unique<Ort::Session>(*env, wpath.c_str(), *session_options);
            }
#else
            if (mc.file_path) {
                ms->session = make_unique<Ort::Session>(*env, mc.file_path, *session_options);
            } else if (mc.model_data && mc.model_size > 0) {
                ms->session = make_unique<Ort::Session>(*env, mc.model_data, mc.model_size, *session_options);
            } else {
                set_error("ModelConfig must have file_path or model_data");
                model_states.clear();
                return RUNTIME_STATUS_INVALID_ARGUMENT;
            }
#endif
            ms->output_names = get_output_names(*ms->session);
            logger->info("Model {}: loaded, {} outputs", i, ms->output_names.size());

            ms->inference_thread = thread(inference_thread_func, ms.get());
            model_states.push_back(move(ms));
        }
        return RUNTIME_STATUS_SUCCESS;
    } catch (const exception &e) {
        stop_and_clear_models();
        set_error(string("runtime_load_models failed: ") + e.what());
        return RUNTIME_STATUS_ERROR;
    }
}

extern "C" RuntimeStatus runtime_enqueue_input(int model_id, Tensors *input_tensors)
{
    if (!initialized) return RUNTIME_STATUS_NOT_INITIALIZED;
    if (model_id < 0 || (size_t)model_id >= model_states.size()) return RUNTIME_STATUS_INVALID_MODEL_ID;
    if (!input_tensors) return RUNTIME_STATUS_INVALID_ARGUMENT;

    // Store model_id in the id field so retrieve_output can return it
    input_tensors->id = model_id;

    if (!model_states[model_id]->input_queue.try_enqueue(input_tensors)) {
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
        auto elapsed = chrono::duration_cast<chrono::milliseconds>(
            chrono::steady_clock::now() - start).count();
        if (elapsed >= timeout_ms)
            return RUNTIME_STATUS_NO_OUTPUT_AVAILABLE;
        this_thread::sleep_for(chrono::milliseconds(1));
    }
}

extern "C" RuntimeStatus runtime_cleanup(void)
{
    stop_and_clear_models();
    free_queue(output_queue);
    session_options.reset();
    env.reset();
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
    return "OAAX NVIDIA OnnxRuntime";
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
