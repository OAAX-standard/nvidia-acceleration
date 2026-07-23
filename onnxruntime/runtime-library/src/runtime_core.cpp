#include <atomic>
#include <chrono>
#include <map>
#include <set>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#endif

#include <onnxruntime_cxx_api.h>
#include "concurrentqueue.h"
#include <spdlog/spdlog.h>

#include "runtime_core.hpp"
#include "runtime_utils.hpp"

using namespace std;

// State.
// Process-lifetime globals below are intentionally leaked (allocated with
// `new`, never destroyed): running their destructors during static
// destruction at process exit — after ORT/spdlog internals are already torn
// down, or with inference threads still running — crashes or deadlocks the
// HOST process whenever it exits without calling runtime_cleanup().
// runtime_cleanup() still releases everything explicitly.
static bool initialized = false;
static string &last_error_msg = *new string();

// ORT globals
static unique_ptr<Ort::Env> &env = *new unique_ptr<Ort::Env>();
static unique_ptr<Ort::SessionOptions> &session_options = *new unique_ptr<Ort::SessionOptions>();

// Per-model state
struct ModelState {
    unique_ptr<Ort::Session> session;
    vector<char *> output_names;
    moodycamel::ConcurrentQueue<Tensors *> input_queue;
    thread inference_thread;
    atomic<bool> stop{false};
    int model_id;
};

struct OutputItem { int model_id; Tensors *tensors; };

static vector<unique_ptr<ModelState>> &model_states = *new vector<unique_ptr<ModelState>>();
static moodycamel::ConcurrentQueue<OutputItem> &output_queue = *new moodycamel::ConcurrentQueue<OutputItem>();

// Config
static int log_level = spdlog::level::info;
#ifdef _WIN32
// Windows services run with an unwritable CWD (System32); default the log
// file to the temp directory instead. The PID suffix keeps host processes
// from fighting over one rotating log file.
static string default_log_file()
{
    char buf[MAX_PATH];
    DWORD n = GetTempPathA(MAX_PATH, buf);
    if (n == 0 || n >= MAX_PATH) return "runtime.log";
    return string(buf) + "oaax-runtime-" + to_string(GetCurrentProcessId()) + ".log";
}
#else
static string default_log_file() { return "runtime.log"; }
#endif
static string log_file = default_log_file();
static int num_threads = 4;
static int inter_op_threads = 0;        // 0 = ORT default
static size_t gpu_mem_limit = SIZE_MAX;
static string perf_mode_val;

static shared_ptr<spdlog::logger> &logger = *new shared_ptr<spdlog::logger>();

static void set_error(const string &msg) {
    last_error_msg = msg;
    if (logger) logger->error("{}", msg);
}

// For use inside a catch block: message of the in-flight exception.
static string current_exception_message() {
    try { throw; }
    catch (const exception &e) { return e.what(); }
    catch (...) { return "unknown exception"; }
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

static void drain_output_queue() {
    OutputItem item;
    while (output_queue.try_dequeue(item)) deep_free_tensors(item.tensors);
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
        auto t_start = chrono::steady_clock::now();

        try {
            vector<string> input_name_strs;
            vector<const char *> input_names;
            vector<Ort::Value> ort_inputs;

            for (int i = 0; i < input->num_tensors; ++i) {
                TensorDescriptor &td = input->tensors[i];
                input_name_strs.emplace_back(td.name);
                vector<int64_t> shape(td.shape, td.shape + td.rank);
                ONNXTensorElementDataType dtype = map_to_ort_type(td.data_type);
                ort_inputs.push_back(Ort::Value::CreateTensor(
                    Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault),
                    td.data, td.data_size,
                    shape.data(), shape.size(), dtype));
            }
            for (auto &s : input_name_strs) input_names.push_back(s.c_str());

            int req_id = input->id;

            auto ort_outputs = ms->session->Run(
                Ort::RunOptions{nullptr},
                input_names.data(), ort_inputs.data(), ort_inputs.size(),
                ms->output_names.data(), ms->output_names.size());

            deep_free_tensors(input);
            input = nullptr;

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

            auto elapsed_ms = chrono::duration_cast<chrono::microseconds>(
                chrono::steady_clock::now() - t_start).count() / 1000.0;
            logger->debug("Model {}: inference id={} done in {:.2f}ms", ms->model_id, req_id, elapsed_ms);

            if (!output_queue.try_enqueue({ms->model_id, out})) {
                logger->error("Failed to enqueue output");
                deep_free_tensors(out);
            }
        } catch (...) {
            if (input) deep_free_tensors(input);
            logger->error("Inference error: {}", current_exception_message());
        }
    }
}

extern "C" RuntimeStatus runtime_init(Config config)
{
    if (initialized) {
        set_error("Runtime already initialized");
        return RUNTIME_STATUS_ALREADY_INITIALIZED;
    }
    if (config.length > 0 && (!config.keys || !config.values)) {
        set_error("Config keys/values must not be null");
        return RUNTIME_STATUS_INVALID_ARGUMENT;
    }

    // A fresh init must not inherit tunables from a previous failed attempt
    num_threads = 4;
    inter_op_threads = 0;
    gpu_mem_limit = SIZE_MAX;
    perf_mode_val.clear();

    try {
        // Collect all config into a map for ordered application
        map<string, string> cfg;
        for (int i = 0; i < config.length; ++i)
            cfg[string(config.keys[i])] = string(config.values[i]);

        // Phase 1: extract log settings before logger init
        if (cfg.count("log_level")) {
            int v = stoi(cfg["log_level"]);
            log_level = (v >= spdlog::level::trace && v <= spdlog::level::off) ? v : spdlog::level::info;
        }
        if (cfg.count("log_file"))
            log_file = cfg["log_file"];
        bool log_stdout = cfg.count("log_stdout") && cfg["log_stdout"] == "true";

        logger = initialize_logger(log_file, log_level, log_level, runtime_get_name(), log_stdout);
        logger->info("Initializing ORT runtime");
        logger->debug("runtime_init: parsed {} config entries", config.length);

        // Detect hardware
        logger->debug("runtime_init: detecting hardware");
        HwProfile hw = detect_hardware();
        logger->info("Hardware: {} CPU cores, {} GPU SMs, {:.1f} GB GPU memory",
                     hw.cpu_cores, hw.gpu_sm_count, hw.gpu_total_mem / 1e9);
        if (hw.gpu_sm_count == 0)
            logger->warn("No CUDA driver or GPU detected; hardware-based defaults unavailable");

        // Phase 2: apply perf_mode (sets defaults based on hardware)
        logger->debug("runtime_init: applying perf_mode (if set)");
        if (cfg.count("perf_mode")) {
            perf_mode_val = cfg["perf_mode"];
            if (perf_mode_val != "power" && perf_mode_val != "eco") {
                logger->warn("Unknown perf_mode '{}', ignoring", perf_mode_val);
                perf_mode_val.clear();
            } else {
                if (perf_mode_val == "power") {
                    num_threads = hw.cpu_cores;
                    inter_op_threads = hw.cpu_cores;
                    gpu_mem_limit = SIZE_MAX;
                } else { // eco
                    num_threads = max(1, hw.cpu_cores / 2);
                    inter_op_threads = max(1, hw.cpu_cores / 2);
                    gpu_mem_limit = hw.gpu_total_mem > 0 ? hw.gpu_total_mem / 2 : SIZE_MAX;
                }
                logger->info("perf_mode={}: num_threads={}, inter_op_threads={}, gpu_mem_limit={:.1f} GB (auto)",
                             perf_mode_val, num_threads, inter_op_threads, gpu_mem_limit / 1e9);
            }
        }

        // Phase 3: explicit keys override perf_mode
        logger->debug("runtime_init: applying explicit config overrides");
        if (cfg.count("num_threads"))
            num_threads = max(1, min(512, stoi(cfg["num_threads"])));
        if (cfg.count("inter_op_threads"))
            inter_op_threads = max(0, min(512, stoi(cfg["inter_op_threads"])));
        if (cfg.count("gpu_mem_limit"))
            gpu_mem_limit = (size_t)stoull(cfg["gpu_mem_limit"]);

        // Warn on unknown keys
        static const set<string> known = {
            "log_level", "log_file", "log_stdout", "perf_mode",
            "num_threads", "inter_op_threads", "gpu_mem_limit"
        };
        for (auto &[k, v] : cfg)
            if (!known.count(k)) logger->warn("Unknown config key ignored: '{}'", k);

        logger->debug("runtime_init: creating Ort::Env");
        env = make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, runtime_get_name());
        logger->debug("runtime_init: creating Ort::SessionOptions (num_threads={}, inter_op_threads={})",
                      num_threads, inter_op_threads);
        session_options = make_unique<Ort::SessionOptions>();
        session_options->SetIntraOpNumThreads(num_threads);
        if (inter_op_threads > 0)
            session_options->SetInterOpNumThreads(inter_op_threads);
        logger->debug("runtime_init: configuring CUDA EP options (device_id=0, gpu_mem_limit={})", gpu_mem_limit);
        OrtCUDAProviderOptions cuda_opts{};
        cuda_opts.device_id = 0;
        cuda_opts.arena_extend_strategy = 0;
        cuda_opts.gpu_mem_limit = gpu_mem_limit;
        cuda_opts.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
        cuda_opts.do_copy_in_default_stream = 1;
        logger->debug("runtime_init: appending CUDA execution provider");
        session_options->AppendExecutionProvider_CUDA(cuda_opts);
        logger->debug("runtime_init: setting graph optimization level");
        session_options->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        initialized = true;
        logger->info("Runtime initialized (log_level={}, num_threads={}, inter_op_threads={}, gpu_mem_limit={:.1f} GB)",
                     log_level, num_threads, inter_op_threads, gpu_mem_limit / 1e9);
        return RUNTIME_STATUS_SUCCESS;
    } catch (...) {
        // Destroy any half-initialized ORT state now: leaving it to static
        // destructors at process exit crashes the host (~OrtEnv segfaults
        // after ORT internals are already torn down).
        session_options.reset();
        env.reset();
        set_error("runtime_init failed: " + current_exception_message());
        return RUNTIME_STATUS_ERROR;
    }
}

extern "C" RuntimeStatus runtime_load_models(int num_models, const ModelConfig *model_configs)
{
    if (!initialized) return RUNTIME_STATUS_NOT_INITIALIZED;
    if (num_models <= 0 || !model_configs) return RUNTIME_STATUS_INVALID_ARGUMENT;

    stop_and_clear_models();
    drain_output_queue();

    logger->debug("runtime_load_models: num_models={}", num_models);
    try {
        for (int i = 0; i < num_models; ++i) {
            const ModelConfig &mc = model_configs[i];
            logger->debug("Model {}: file_path={} model_data={} model_size={}",
                          i, mc.file_path ? mc.file_path : "(null)",
                          (const void *)mc.model_data, mc.model_size);
            auto ms = make_unique<ModelState>();
            ms->model_id = i;

#ifdef _WIN32
            if (mc.file_path) {
                logger->debug("Model {}: converting file_path to wide string", i);
                int sz = MultiByteToWideChar(CP_UTF8, 0, mc.file_path, -1, NULL, 0);
                wstring wpath(sz, 0);
                MultiByteToWideChar(CP_UTF8, 0, mc.file_path, -1, &wpath[0], sz);
                logger->debug("Model {}: creating Ort::Session from file_path (wide)", i);
                ms->session = make_unique<Ort::Session>(*env, wpath.c_str(), *session_options);
            }
#else
            if (mc.file_path) {
                logger->debug("Model {}: creating Ort::Session from file_path", i);
                ms->session = make_unique<Ort::Session>(*env, mc.file_path, *session_options);
            } else if (mc.model_data && mc.model_size > 0) {
                logger->debug("Model {}: creating Ort::Session from in-memory buffer ({} bytes)", i, mc.model_size);
                ms->session = make_unique<Ort::Session>(*env, mc.model_data, mc.model_size, *session_options);
            } else {
                set_error("ModelConfig must have file_path or model_data");
                stop_and_clear_models();
                return RUNTIME_STATUS_INVALID_ARGUMENT;
            }
#endif
            logger->debug("Model {}: session created, fetching output names", i);
            ms->output_names = get_output_names(*ms->session);
            logger->debug("Model {}: got {} output names, fetching input count", i, ms->output_names.size());
            size_t n_inputs = ms->session->GetInputCount();
            logger->info("Model {}: loaded, {} inputs, {} outputs", i, n_inputs, ms->output_names.size());

            Ort::AllocatorWithDefaultOptions alloc;
            for (size_t j = 0; j < n_inputs; ++j) {
                logger->debug("Model {}: input[{}] fetching name", i, j);
                auto name = ms->session->GetInputNameAllocated(j, alloc);
                logger->debug("Model {}: input[{}] name={}, fetching type/shape info", i, j, name.get());
                // Keep the TypeInfo alive: GetTensorTypeAndShapeInfo() returns a
                // non-owning view into memory owned by it (CastTypeInfoToTensorInfo
                // is a pointer cast, not a copy) - binding straight off a temporary
                // dangles as soon as this statement ends.
                Ort::TypeInfo type_info = ms->session->GetInputTypeInfo(j);
                auto info = type_info.GetTensorTypeAndShapeInfo();
                logger->debug("Model {}: input[{}] fetching GetShape()", i, j);
                auto shape = info.GetShape();
                logger->debug("Model {}: input[{}] shape rank={}", i, j, shape.size());
                string shape_str;
                for (size_t d = 0; d < shape.size(); ++d) {
                    if (d) shape_str += "x";
                    shape_str += shape[d] < 0 ? "?" : to_string(shape[d]);
                }
                logger->debug("  input  [{}] shape={}", name.get(), shape_str);
            }
            for (size_t j = 0; j < ms->output_names.size(); ++j) {
                logger->debug("Model {}: output[{}] ({}) fetching type/shape info", i, j, ms->output_names[j]);
                Ort::TypeInfo type_info = ms->session->GetOutputTypeInfo(j);
                auto info = type_info.GetTensorTypeAndShapeInfo();
                logger->debug("Model {}: output[{}] fetching GetShape()", i, j);
                auto shape = info.GetShape();
                logger->debug("Model {}: output[{}] shape rank={}", i, j, shape.size());
                string shape_str;
                for (size_t d = 0; d < shape.size(); ++d) {
                    if (d) shape_str += "x";
                    shape_str += shape[d] < 0 ? "?" : to_string(shape[d]);
                }
                logger->debug("  output [{}] shape={}", ms->output_names[j], shape_str);
            }

            logger->debug("Model {}: starting inference thread", i);
            ms->inference_thread = thread(inference_thread_func, ms.get());
            model_states.push_back(move(ms));
            logger->debug("Model {}: fully loaded and registered", i);
        }
        return RUNTIME_STATUS_SUCCESS;
    } catch (...) {
        stop_and_clear_models();
        set_error("runtime_load_models failed: " + current_exception_message());
        return RUNTIME_STATUS_ERROR;
    }
}

extern "C" RuntimeStatus runtime_enqueue_input(int model_id, Tensors *input_tensors)
{
    if (!initialized) return RUNTIME_STATUS_NOT_INITIALIZED;
    if (model_id < 0 || (size_t)model_id >= model_states.size()) return RUNTIME_STATUS_INVALID_MODEL_ID;
    if (!input_tensors) return RUNTIME_STATUS_INVALID_ARGUMENT;

    logger->debug("runtime_enqueue_input: model_id={} num_tensors={} id={}",
                  model_id, input_tensors->num_tensors, input_tensors->id);
    try {
        if (!model_states[model_id]->input_queue.try_enqueue(input_tensors)) {
            set_error("Failed to enqueue input tensors");
            return RUNTIME_STATUS_ERROR;
        }
        logger->debug("runtime_enqueue_input: model_id={} enqueued successfully", model_id);
        return RUNTIME_STATUS_SUCCESS;
    } catch (...) {
        set_error("runtime_enqueue_input failed: " + current_exception_message());
        return RUNTIME_STATUS_ERROR;
    }
}

extern "C" RuntimeStatus runtime_retrieve_output(int *model_id, Tensors **output_tensors, int timeout_ms)
{
    if (!initialized) return RUNTIME_STATUS_NOT_INITIALIZED;
    if (!model_id || !output_tensors) return RUNTIME_STATUS_INVALID_ARGUMENT;

    logger->debug("runtime_retrieve_output: timeout_ms={}", timeout_ms);
    try {
        OutputItem item;
        if (timeout_ms == 0) {
            if (!output_queue.try_dequeue(item)) {
                logger->debug("runtime_retrieve_output: no output available (non-blocking)");
                return RUNTIME_STATUS_NO_OUTPUT_AVAILABLE;
            }
            *model_id = item.model_id;
            *output_tensors = item.tensors;
            logger->debug("runtime_retrieve_output: got output for model_id={}", item.model_id);
            return RUNTIME_STATUS_SUCCESS;
        }

        auto start = chrono::steady_clock::now();
        while (true) {
            if (output_queue.try_dequeue(item)) {
                *model_id = item.model_id;
                *output_tensors = item.tensors;
                logger->debug("runtime_retrieve_output: got output for model_id={}", item.model_id);
                return RUNTIME_STATUS_SUCCESS;
            }
            if (timeout_ms > 0) {
                auto elapsed = chrono::duration_cast<chrono::milliseconds>(
                    chrono::steady_clock::now() - start).count();
                if (elapsed >= timeout_ms) {
                    logger->debug("runtime_retrieve_output: timed out after {} ms", elapsed);
                    return RUNTIME_STATUS_NO_OUTPUT_AVAILABLE;
                }
            }
            this_thread::sleep_for(chrono::milliseconds(1));
        }
    } catch (...) {
        set_error("runtime_retrieve_output failed: " + current_exception_message());
        return RUNTIME_STATUS_ERROR;
    }
}

extern "C" RuntimeStatus runtime_cleanup(void)
{
    RuntimeStatus status = RUNTIME_STATUS_SUCCESS;
    if (logger) logger->debug("runtime_cleanup: stopping {} model(s)", model_states.size());
    try {
        stop_and_clear_models();
        if (logger) logger->debug("runtime_cleanup: models stopped, draining output queue");
        drain_output_queue();
        if (logger) logger->debug("runtime_cleanup: output queue drained");
    } catch (...) {
        set_error("runtime_cleanup failed: " + current_exception_message());
        model_states.clear();
        status = RUNTIME_STATUS_ERROR;
    }

    // Always release ORT state and reset flags, even if teardown above threw:
    // leaving initialized=true would block re-init forever, and a leftover
    // Ort::Env crashes the host during static destruction at process exit.
    if (logger) logger->debug("runtime_cleanup: releasing ORT state");
    session_options.reset();
    env.reset();
    initialized = false;
    num_threads = 4;
    inter_op_threads = 0;
    gpu_mem_limit = SIZE_MAX;
    perf_mode_val.clear();
    log_level = spdlog::level::info;
    log_file = default_log_file();
    if (status == RUNTIME_STATUS_SUCCESS) last_error_msg.clear();
    if (logger) {
        logger->info("Runtime cleaned up");
        destroy_logger(logger);
        logger = nullptr;
    }
    return status;
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
