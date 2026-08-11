#include <algorithm>
#include <cctype>
#include <cstring>
#include <iostream>
#include <vector>
#include <stdexcept>
#include <thread>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#endif

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

// Leaked wrapper (never destructed at process exit -- see the leaked globals
// in runtime_core.cpp), so it's always safe to explicitly reset the pool it
// points to from destroy_logger(). A fresh pool is created on every
// initialize_logger() call; destroy_logger() drops it there, which runs
// thread_pool's real destructor and joins its worker thread synchronously.
// async_logger only holds a weak_ptr to the pool, so this reference is its
// sole owner. A thread whose code lives in this DLL must be joined before
// runtime_cleanup() returns, or FreeLibrary() on Windows leaves the DLL
// pinned and its file can't be replaced/deleted. If the host never calls
// runtime_cleanup(), this wrapper (and the pool it holds) simply stays alive
// until process exit, same as today -- no join happens, no deadlock risk.
static auto &thread_pool = *new shared_ptr<spdlog::details::thread_pool>();

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

    thread_pool = make_shared<spdlog::details::thread_pool>(8192, 1);

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
    // Sole strong owner of the pool (see the comment above initialize_logger):
    // this runs ~thread_pool() now, joining its worker thread before
    // returning so runtime_cleanup() never leaves a RuntimeLibrary-resident
    // thread running past the call.
    thread_pool.reset();
}

#ifdef _WIN32
// Directory containing this DLL (resolved from an address inside it, not the
// host exe). Empty string on failure.
static string get_own_module_dir()
{
    HMODULE mod = nullptr;
    if (!GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                                GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                            (LPCSTR)&get_own_module_dir, &mod))
        return "";
    char path[MAX_PATH];
    DWORD n = GetModuleFileNameA(mod, path, MAX_PATH);
    if (n == 0 || n >= MAX_PATH) return "";
    char *slash = strrchr(path, '\\');
    if (!slash) return "";
    *slash = 0;
    return path;
}

static bool resolvable_via_search_path(const char *dll_name)
{
    char found[MAX_PATH];
    return SearchPathA(NULL, dll_name, NULL, MAX_PATH, found, NULL) > 0;
}

// cuDNN 9's cudnn64_9.dll is a thin loader shim: at cudnnCreate() time it
// LoadLibrary()s its component DLLs (cudnn_graph64_9.dll, cudnn_engines_*,
// ...) via the STANDARD search order - host exe dir, System32, PATH - which
// does not include this runtime's folder when we are hosted by another
// process (e.g. a mediaserver service). When the components can't be found,
// cuDNN abort()s the entire host process (fail-fast 0xc0000409). Prepending
// our own directory to the process PATH lets those late loads resolve; the
// host-side LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR only covers our static import
// chain and cannot reach LoadLibrary calls cuDNN makes on its own.
string prepare_dll_search_path(spdlog::logger &log)
{
    string dir = get_own_module_dir();
    if (dir.empty()) {
        log.warn("Could not resolve the runtime's own directory; "
                 "dependent DLLs may not be found");
        return "";
    }

    DWORD len = GetEnvironmentVariableA("PATH", NULL, 0);
    string path(len ? len - 1 : 0, '\0');
    if (len) GetEnvironmentVariableA("PATH", &path[0], len);

    // Idempotent across init/cleanup cycles: don't grow PATH on re-init.
    auto icontains = [](const string &hay, const string &needle) {
        auto it = search(hay.begin(), hay.end(), needle.begin(), needle.end(),
                         [](char a, char b) { return toupper((unsigned char)a) == toupper((unsigned char)b); });
        return it != hay.end();
    };
    if (!icontains(path, dir)) {
        if (!SetEnvironmentVariableA("PATH", (dir + ";" + path).c_str()))
            log.warn("Failed to prepend '{}' to PATH (error {})", dir, GetLastError());
        else
            log.info("Prepended runtime directory to PATH for cuDNN component resolution: {}", dir);
    }

    // A resolvable cudnn64_9.dll whose components are NOT resolvable would
    // abort() the host at model-load time - fail init cleanly instead.
    if (resolvable_via_search_path("cudnn64_9.dll") &&
        !resolvable_via_search_path("cudnn_graph64_9.dll"))
        return "cudnn64_9.dll is present but its component cudnn_graph64_9.dll is not "
               "findable (searched the runtime directory, system dirs and PATH). "
               "cuDNN 9 would abort() the host process at model load. Deploy the "
               "complete cuDNN package next to the runtime library.";
    return "";
}
#else
string prepare_dll_search_path(spdlog::logger &) { return ""; }
#endif

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
