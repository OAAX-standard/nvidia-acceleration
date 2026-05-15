#include <iostream>
#include <vector>
#include <stdexcept>

#include "runtime_utils.hpp"
#include <onnxruntime_cxx_api.h>
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
        names.emplace_back(p.release());
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
                                              const string &prefix)
{
    try {
        auto console_sink = make_shared<spdlog::sinks::stdout_color_sink_mt>();
        console_sink->set_level(static_cast<spdlog::level::level_enum>(console_level));

        auto file_sink = make_shared<spdlog::sinks::rotating_file_sink_st>(
            log_file, 1024 * 1024 * 5, 3);
        file_sink->set_level(static_cast<spdlog::level::level_enum>(file_level));

        static auto thread_pool = make_shared<spdlog::details::thread_pool>(8192, 1);

        auto logger = make_shared<spdlog::async_logger>(
            prefix, spdlog::sinks_init_list{console_sink, file_sink},
            thread_pool, spdlog::async_overflow_policy::overrun_oldest);

        spdlog::set_pattern("[%Y-%m-%d %H:%M:%S.%e] [" + prefix + "] [%^%l%$] %v");
        logger->set_level(static_cast<spdlog::level::level_enum>(
            min(file_level, console_level)));
        return logger;
    } catch (const spdlog::spdlog_ex &ex) {
        cerr << "Logger init failed: " << ex.what() << "\n";
        exit(EXIT_FAILURE);
    }
}

void destroy_logger(const shared_ptr<spdlog::logger> &logger)
{
    if (logger) {
        logger->flush();
        spdlog::drop(logger->name());
    }
}
