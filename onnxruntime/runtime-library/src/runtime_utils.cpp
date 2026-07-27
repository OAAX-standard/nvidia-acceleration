#include <cstring>
#include <iostream>
#include <vector>
#include <stdexcept>
#include <thread>

#include "runtime_utils.hpp"
#include <onnxruntime_cxx_api.h>

extern "C" {
#include "lib_loader.h"
}
#include "concurrentqueue.h"

#include <spdlog/async.h>
#include <spdlog/sinks/rotating_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

using namespace std;

void deep_free_tensors(Tensors *t)
{
    if (!t) return;
    for (int i = 0; i < t->num_tensors; ++i) {
        free(t->tensors[i].name);
        free(t->tensors[i].shape);
        free(t->tensors[i].data);
    }
    free(t->tensors);
    free(t);
}

void free_queue(moodycamel::ConcurrentQueue<Tensors *> &queue)
{
    Tensors *t;
    while (queue.try_dequeue(t))
        deep_free_tensors(t);
}

vector<char *> get_output_names(Ort::Session &session)
{
    Ort::AllocatorWithDefaultOptions allocator;
    size_t count = session.GetOutputCount();
    vector<char *> names;
    names.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        Ort::AllocatedStringPtr p = session.GetOutputNameAllocated(i, allocator);
        // p owns memory from ORT's allocator, which on Windows is backed by
        // _aligned_malloc/_aligned_free - freeing it with plain free() (as
        // stop_and_clear_models() does for entries in `names`) corrupts the
        // heap. Copy into a CRT-owned buffer instead and let p's destructor
        // free the ORT-owned original the correct way.
        size_t len = strlen(p.get()) + 1;
        char *name = (char *)malloc(len);
        memcpy(name, p.get(), len);
        names.emplace_back(name);
    }
    return names;
}

ONNXTensorElementDataType map_to_ort_type(TensorElementType t)
{
    switch (t) {
    case DATA_TYPE_FLOAT:   return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    case DATA_TYPE_UINT8:   return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
    case DATA_TYPE_INT8:    return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8;
    case DATA_TYPE_UINT16:  return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16;
    case DATA_TYPE_INT16:   return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16;
    case DATA_TYPE_INT32:   return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
    case DATA_TYPE_INT64:   return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
    case DATA_TYPE_BOOL:    return ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL;
    case DATA_TYPE_DOUBLE:  return ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE;
    case DATA_TYPE_UINT32:  return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32;
    case DATA_TYPE_UINT64:  return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64;
    case DATA_TYPE_FLOAT16: return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
    default:
        throw runtime_error("Unsupported TensorElementType for ORT mapping");
    }
}

TensorElementType map_to_tensors_struct_type(ONNXTensorElementDataType type)
{
    switch (type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:   return DATA_TYPE_FLOAT;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:   return DATA_TYPE_UINT8;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:    return DATA_TYPE_INT8;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:  return DATA_TYPE_UINT16;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:   return DATA_TYPE_INT16;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:   return DATA_TYPE_INT32;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:   return DATA_TYPE_INT64;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:    return DATA_TYPE_BOOL;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:  return DATA_TYPE_DOUBLE;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:  return DATA_TYPE_UINT32;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:  return DATA_TYPE_UINT64;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: return DATA_TYPE_FLOAT16;
    default:
        throw runtime_error("Unsupported ORT type for OAAX mapping");
    }
}

int get_element_byte_size(TensorElementType t)
{
    switch (t) {
    case DATA_TYPE_FLOAT:    return 4;
    case DATA_TYPE_DOUBLE:   return 8;
    case DATA_TYPE_INT8:     return 1;
    case DATA_TYPE_UINT8:    return 1;
    case DATA_TYPE_INT16:    return 2;
    case DATA_TYPE_UINT16:   return 2;
    case DATA_TYPE_FLOAT16:  return 2;
    case DATA_TYPE_BFLOAT16: return 2;
    case DATA_TYPE_INT32:    return 4;
    case DATA_TYPE_UINT32:   return 4;
    case DATA_TYPE_INT64:    return 8;
    case DATA_TYPE_UINT64:   return 8;
    case DATA_TYPE_BOOL:     return 1;
    default:                 return 1;
    }
}

shared_ptr<spdlog::logger> initialize_logger(const string &log_file,
                                              int file_level,
                                              int console_level,
                                              const string &prefix,
                                              bool log_stdout)
{
    // A shared library must never terminate its host: if the log file cannot
    // be created (e.g. Windows services run with an unwritable CWD), fall
    // back to console-only logging instead of exiting.
    vector<spdlog::sink_ptr> sinks;
    string file_sink_error;
    try {
        auto file_sink = make_shared<spdlog::sinks::rotating_file_sink_st>(
            log_file, 1024 * 1024 * 5, 3);
        file_sink->set_level(static_cast<spdlog::level::level_enum>(file_level));
        sinks.push_back(file_sink);
    } catch (const exception &ex) {
        file_sink_error = ex.what();
        // Unconditional console line: the logger warning below is subject to
        // level filtering, and losing file logging must never be silent.
        cerr << "[" << prefix << "] Failed to open log file '" << log_file
             << "' (" << file_sink_error << "); logging to console only\n";
    }

    if (log_stdout || sinks.empty()) {
        auto console_sink = make_shared<spdlog::sinks::stdout_color_sink_mt>();
        console_sink->set_level(static_cast<spdlog::level::level_enum>(console_level));
        sinks.push_back(console_sink);
    }

    // Intentionally leaked: joining the pool's worker thread during static
    // destruction at process exit can deadlock/crash the host (see the
    // leaked globals in runtime_core.cpp).
    static auto &thread_pool = *new shared_ptr<spdlog::details::thread_pool>(
        make_shared<spdlog::details::thread_pool>(8192, 1));

    auto logger = make_shared<spdlog::async_logger>(
        prefix, sinks.begin(), sinks.end(),
        thread_pool, spdlog::async_overflow_policy::overrun_oldest);

    logger->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%n] [%^%l%$] %v");
    logger->set_level(static_cast<spdlog::level::level_enum>(
        min(file_level, console_level)));
    logger->flush_on(spdlog::level::info);

    if (!file_sink_error.empty())
        logger->warn("Failed to open log file '{}' ({}); logging to console only",
                     log_file, file_sink_error);
    return logger;
}

void destroy_logger(const shared_ptr<spdlog::logger> &logger)
{
    if (logger) {
        logger->flush();
        spdlog::drop(logger->name());
    }
}

// GPU detection uses the CUDA *driver* API (nvcuda.dll / libcuda.so.1),
// loaded dynamically via c-utilities' lib_loader:
// - the driver library name is CUDA-version-independent and always in the
//   system search path, so no per-version soname lists are needed;
// - the library carries no hard CUDA dependency: on a machine without a
//   CUDA driver it still loads and detection degrades gracefully;
// - unlike cudaMemGetInfo, cuDeviceTotalMem does not create a CUDA context
//   (which would permanently reserve GPU memory in the host process);
// - only ABI-stable entry points are used (int/size_t signatures).
static void *load_cuda_driver()
{
    // Process-lifetime handle, loaded once; hosts may cycle init/cleanup.
    static void *handle = []() -> void * {
#ifdef _WIN32
        return load_dynamic_library("nvcuda.dll");
#else
        if (void *h = load_dynamic_library("libcuda.so.1")) return h;
        return load_dynamic_library("libcuda.so");
#endif
    }();
    return handle;
}

HwProfile detect_hardware()
{
    HwProfile hw{};
    hw.cpu_cores = (int)thread::hardware_concurrency();
    if (hw.cpu_cores <= 0) hw.cpu_cores = 1;

    void *cuda = load_cuda_driver();
    if (!cuda) return hw;

    using InitFn = int (*)(unsigned int flags);
    using DeviceGetFn = int (*)(int *device, int ordinal);
    using AttributeFn = int (*)(int *value, int attribute, int device);
    using TotalMemFn = int (*)(size_t *bytes, int device);
    auto cu_init = (InitFn)get_symbol_address(cuda, "cuInit");
    auto cu_device_get = (DeviceGetFn)get_symbol_address(cuda, "cuDeviceGet");
    auto cu_get_attribute = (AttributeFn)get_symbol_address(cuda, "cuDeviceGetAttribute");
    auto cu_total_mem = (TotalMemFn)get_symbol_address(cuda, "cuDeviceTotalMem_v2");
    if (!cu_init || !cu_device_get || cu_init(0) != 0) return hw;

    int device = 0;
    if (cu_device_get(&device, 0) != 0) return hw;

    const int CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16;
    int sm_count = 0;
    if (cu_get_attribute &&
        cu_get_attribute(&sm_count, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device) == 0)
        hw.gpu_sm_count = sm_count;

    size_t total_mem = 0;
    if (cu_total_mem && cu_total_mem(&total_mem, device) == 0)
        hw.gpu_total_mem = total_mem;

    return hw;
}
