#include <atomic>
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

#include "runtime_core.hpp"   // pulls in tensors_struct.h with extern "C"
#include "runtime_utils.hpp"

using namespace std;

// Forward declaration
static void inference_thread_func();

// TRT logger: forwards TRT messages to spdlog
class TrtLogger : public nvinfer1::ILogger
{
public:
    void log(Severity severity, const char *msg) noexcept override;
};
static TrtLogger trt_logger;

// TRT objects
static nvinfer1::IRuntime *trt_runtime = nullptr;
static nvinfer1::ICudaEngine *engine = nullptr;
static nvinfer1::IExecutionContext *context = nullptr;
static cudaStream_t infer_stream = nullptr;

// Per-tensor info populated at model load time
struct TensorInfo {
    std::string name;
    bool is_input;
    nvinfer1::Dims dims;
    nvinfer1::DataType dtype;
    size_t byte_size;
    void *gpu_buf = nullptr;
};
static std::vector<TensorInfo> tensor_infos;
static std::vector<size_t> input_indices;   // indices into tensor_infos
static std::vector<size_t> output_indices;  // indices into tensor_infos

// Input/output queues
static moodycamel::ConcurrentQueue<tensors_struct *> input_tensors_queue;
static moodycamel::ConcurrentQueue<tensors_struct *> output_tensors_queue;

// Inference thread
static std::thread inference_thread;
static std::atomic<bool> stop_inference_thread{false};

// Logger and runtime config
std::shared_ptr<spdlog::logger> logger;
static int log_level = spdlog::level::info;
static std::string log_file = "runtime.log";

// TrtLogger implementation
void TrtLogger::log(Severity severity, const char *msg) noexcept
{
    if (!logger)
        return;
    switch (severity)
    {
    case Severity::kINTERNAL_ERROR:
    case Severity::kERROR:
        logger->error("[TRT] {}", msg);
        break;
    case Severity::kWARNING:
        logger->warn("[TRT] {}", msg);
        break;
    case Severity::kINFO:
        logger->info("[TRT] {}", msg);
        break;
    case Severity::kVERBOSE:
        logger->trace("[TRT] {}", msg);
        break;
    }
}

extern "C" int runtime_initialization_with_args(int length, char **keys, void **values)
{
    for (int i = 0; i < length; ++i)
    {
        std::string key(keys[i]);
        if (key == "log_level")
        {
            int value = std::stoi(static_cast<char *>(values[i]));
            log_level = (value >= spdlog::level::trace && value <= spdlog::level::off)
                            ? value
                            : spdlog::level::info;
        }
        else if (key == "log_file")
        {
            log_file = std::string(static_cast<char *>(values[i]));
        }
    }
    return runtime_initialization();
}

extern "C" int runtime_initialization()
{
    try
    {
        logger = initialize_logger(log_file, log_level, log_level, runtime_name());
        logger->info("Initializing TensorRT runtime");

        trt_runtime = nvinfer1::createInferRuntime(trt_logger);
        if (!trt_runtime)
            throw std::runtime_error("Failed to create TRT IRuntime");

        stop_inference_thread = false;
        inference_thread = std::thread(inference_thread_func);

        logger->info("Inference thread started");
        logger->info("Runtime arguments: log_level={}, log_file={}", log_level, log_file);
        return 0;
    }
    catch (const std::exception &e)
    {
        if (logger)
            logger->error("Initialization failed: {}", e.what());
        return -1;
    }
}

extern "C" int runtime_model_loading(const char *model_path)
{
    logger->info("Loading TRT engine from: {}", model_path);
    try
    {
        // Read engine file into memory
        std::ifstream file(model_path, std::ios::binary | std::ios::ate);
        if (!file.is_open())
            throw std::runtime_error(std::string("Cannot open engine file: ") + model_path);
        size_t file_size = static_cast<size_t>(file.tellg());
        file.seekg(0, std::ios::beg);
        std::vector<char> engine_data(file_size);
        file.read(engine_data.data(), static_cast<std::streamsize>(file_size));
        file.close();

        // Deserialize the engine
        engine = trt_runtime->deserializeCudaEngine(engine_data.data(), file_size);
        if (!engine)
            throw std::runtime_error("Failed to deserialize TRT engine");

        context = engine->createExecutionContext();
        if (!context)
            throw std::runtime_error("Failed to create TRT execution context");

        if (cudaStreamCreate(&infer_stream) != cudaSuccess)
            throw std::runtime_error("Failed to create CUDA stream");

        // Build tensor info and allocate persistent GPU buffers
        int nb_tensors = engine->getNbIOTensors();
        tensor_infos.clear();
        input_indices.clear();
        output_indices.clear();
        tensor_infos.resize(static_cast<size_t>(nb_tensors));

        for (int i = 0; i < nb_tensors; ++i)
        {
            const char *name = engine->getIOTensorName(i);
            TensorInfo &ti = tensor_infos[static_cast<size_t>(i)];
            ti.name = name;
            ti.is_input = (engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT);
            ti.dims = engine->getTensorShape(name);
            ti.dtype = engine->getTensorDataType(name);
            ti.byte_size = compute_tensor_byte_size(ti.dims, ti.dtype);

            if (cudaMalloc(&ti.gpu_buf, ti.byte_size) != cudaSuccess)
                throw std::runtime_error(std::string("cudaMalloc failed for tensor: ") + name);

            if (ti.is_input)
                input_indices.push_back(static_cast<size_t>(i));
            else
                output_indices.push_back(static_cast<size_t>(i));

            logger->info("  Tensor {}: name={} {} dtype={} byte_size={}",
                         i, name,
                         ti.is_input ? "INPUT" : "OUTPUT",
                         static_cast<int>(ti.dtype),
                         ti.byte_size);
        }

        logger->info("Engine loaded: {} inputs, {} outputs",
                     input_indices.size(), output_indices.size());
        return 0;
    }
    catch (const std::exception &e)
    {
        logger->error("Model loading failed: {}", e.what());
        return -1;
    }
}

extern "C" int send_input(tensors_struct *input_tensors)
{
    logger->debug("Enqueuing input. Queue size: ~{}", input_tensors_queue.size_approx());
    if (!input_tensors_queue.try_enqueue(input_tensors))
    {
        logger->warn("Failed to enqueue input tensors");
        return -1;
    }
    logger->trace("Input enqueued");
    return 0;
}

static void inference_thread_func()
{
    while (!stop_inference_thread)
    {
        tensors_struct *input_tensors = nullptr;
        if (!input_tensors_queue.try_dequeue(input_tensors))
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }
        logger->debug("Dequeued input, running inference");

        // Pre-allocate host output buffers so we can pipeline D2H with inference
        std::vector<void *> host_out_bufs(output_indices.size(), nullptr);

        try
        {
            if (input_tensors->num_tensors != input_indices.size())
                throw std::runtime_error("Input tensor count mismatch with engine");

            // Allocate host output buffers (needed before enqueuing D2H copies)
            for (size_t i = 0; i < output_indices.size(); ++i)
            {
                host_out_bufs[i] = malloc(tensor_infos[output_indices[i]].byte_size);
                if (!host_out_bufs[i])
                    throw std::runtime_error("malloc failed for output buffer");
            }

            // H2D: copy inputs to GPU (async, queued on infer_stream)
            for (size_t i = 0; i < input_indices.size(); ++i)
            {
                TensorInfo &ti = tensor_infos[input_indices[i]];
                if (cudaMemcpyAsync(ti.gpu_buf, input_tensors->data[i], ti.byte_size,
                                    cudaMemcpyHostToDevice, infer_stream) != cudaSuccess)
                    throw std::runtime_error("cudaMemcpyAsync H2D failed");
            }

            // Bind all tensor addresses (inputs and outputs)
            for (auto &ti : tensor_infos)
                context->setTensorAddress(ti.name.c_str(), ti.gpu_buf);

            // Enqueue inference
            if (!context->enqueueV3(infer_stream))
                throw std::runtime_error("TRT enqueueV3 failed");

            // D2H: copy outputs from GPU (async, queued after inference on same stream)
            for (size_t i = 0; i < output_indices.size(); ++i)
            {
                TensorInfo &ti = tensor_infos[output_indices[i]];
                if (cudaMemcpyAsync(host_out_bufs[i], ti.gpu_buf, ti.byte_size,
                                    cudaMemcpyDeviceToHost, infer_stream) != cudaSuccess)
                    throw std::runtime_error("cudaMemcpyAsync D2H failed");
            }

            // Wait for the full pipeline (H2D + inference + D2H) to complete
            if (cudaStreamSynchronize(infer_stream) != cudaSuccess)
                throw std::runtime_error("cudaStreamSynchronize failed");

            deep_free_tensors_struct(input_tensors);
            input_tensors = nullptr;

            // Build output tensors_struct; host_out_bufs[i] ownership transferred
            size_t n_out = output_indices.size();
            tensors_struct *output_tensors =
                static_cast<tensors_struct *>(malloc(sizeof(tensors_struct)));
            output_tensors->num_tensors = n_out;
            output_tensors->names      = static_cast<char **>(malloc(sizeof(char *) * n_out));
            output_tensors->data_types = static_cast<tensor_data_type *>(
                malloc(sizeof(tensor_data_type) * n_out));
            output_tensors->ranks      = static_cast<size_t *>(malloc(sizeof(size_t) * n_out));
            output_tensors->shapes     = static_cast<size_t **>(malloc(sizeof(size_t *) * n_out));
            output_tensors->data       = static_cast<void **>(malloc(sizeof(void *) * n_out));

            for (size_t i = 0; i < output_indices.size(); ++i)
            {
                TensorInfo &ti = tensor_infos[output_indices[i]];

                output_tensors->names[i] = static_cast<char *>(malloc(ti.name.size() + 1));
                strcpy(output_tensors->names[i], ti.name.c_str());

                output_tensors->data_types[i] = map_trt_to_oaax_type(ti.dtype);

                output_tensors->ranks[i] = static_cast<size_t>(ti.dims.nbDims);
                output_tensors->shapes[i] = static_cast<size_t *>(
                    malloc(sizeof(size_t) * static_cast<size_t>(ti.dims.nbDims)));
                for (int d = 0; d < ti.dims.nbDims; ++d)
                    output_tensors->shapes[i][static_cast<size_t>(d)] =
                        static_cast<size_t>(ti.dims.d[d]);

                output_tensors->data[i] = host_out_bufs[i]; // transfer ownership
                host_out_bufs[i] = nullptr;
            }

            if (!output_tensors_queue.try_enqueue(output_tensors))
            {
                logger->error("Failed to enqueue output tensors");
                deep_free_tensors_struct(output_tensors);
            }
            else
            {
                logger->debug("Output enqueued");
            }
        }
        catch (const std::exception &e)
        {
            logger->error("Inference error: {}", e.what());
            if (input_tensors)
                deep_free_tensors_struct(input_tensors);
            for (void *buf : host_out_bufs)
                free(buf);
        }
    }
}

extern "C" int receive_output(tensors_struct **output_tensors)
{
    logger->debug("Checking for output. Queue size: ~{}", output_tensors_queue.size_approx());
    if (output_tensors_queue.size_approx() == 0)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        return -1;
    }
    if (!output_tensors_queue.try_dequeue(*output_tensors))
    {
        logger->error("Failed to dequeue output tensors");
        return -1;
    }
    logger->debug("Output received");
    return 0;
}

extern "C" int runtime_destruction()
{
    logger->info("Destroying runtime...");

    stop_inference_thread = true;
    if (inference_thread.joinable())
        inference_thread.join();
    logger->trace("Inference thread stopped");

    free_queue(input_tensors_queue);
    free_queue(output_tensors_queue);

    // Free persistent GPU buffers
    for (auto &ti : tensor_infos)
    {
        if (ti.gpu_buf)
            cudaFree(ti.gpu_buf);
    }
    tensor_infos.clear();
    input_indices.clear();
    output_indices.clear();

    if (infer_stream)
    {
        cudaStreamDestroy(infer_stream);
        infer_stream = nullptr;
    }

    // TRT objects must be destroyed in reverse creation order
    if (context)
    {
        delete context;
        context = nullptr;
    }
    if (engine)
    {
        delete engine;
        engine = nullptr;
    }
    if (trt_runtime)
    {
        delete trt_runtime;
        trt_runtime = nullptr;
    }

    logger->debug("Runtime destroyed");
    destroy_logger(logger);
    return 0;
}

extern "C" const char *runtime_error_message()
{
    return "Check the stdout and/or log files for any error message.";
}

extern "C" const char *runtime_version()
{
    return RUNTIME_VERSION;
}

extern "C" const char *runtime_name()
{
    return "OAAX NVIDIA TensorRT Runtime";
}
