#include <iostream>
#include <stdexcept>

#include "runtime_utils.hpp"
#include <spdlog/async.h>
#include <spdlog/sinks/rotating_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>

using namespace std;

tensor_data_type map_trt_to_oaax_type(nvinfer1::DataType dtype)
{
    switch (dtype)
    {
    case nvinfer1::DataType::kFLOAT:  return DATA_TYPE_FLOAT;
    case nvinfer1::DataType::kHALF:   return DATA_TYPE_FLOAT16;
    case nvinfer1::DataType::kBF16:   return DATA_TYPE_BFLOAT16;
    case nvinfer1::DataType::kINT8:   return DATA_TYPE_INT8;
    case nvinfer1::DataType::kUINT8:  return DATA_TYPE_UINT8;
    case nvinfer1::DataType::kINT32:  return DATA_TYPE_INT32;
    case nvinfer1::DataType::kINT64:  return DATA_TYPE_INT64;
    case nvinfer1::DataType::kBOOL:   return DATA_TYPE_BOOL;
    default:
        throw std::runtime_error("Unsupported TensorRT data type");
    }
}

size_t trt_dtype_byte_size(nvinfer1::DataType dtype)
{
    switch (dtype)
    {
    case nvinfer1::DataType::kFLOAT:  return 4;
    case nvinfer1::DataType::kHALF:   return 2;
    case nvinfer1::DataType::kBF16:   return 2;
    case nvinfer1::DataType::kINT8:   return 1;
    case nvinfer1::DataType::kUINT8:  return 1;
    case nvinfer1::DataType::kINT32:  return 4;
    case nvinfer1::DataType::kINT64:  return 8;
    case nvinfer1::DataType::kBOOL:   return 1;
    default:
        throw std::runtime_error("Unsupported TensorRT data type");
    }
}

size_t compute_tensor_byte_size(const nvinfer1::Dims &dims, nvinfer1::DataType dtype)
{
    size_t count = 1;
    for (int i = 0; i < dims.nbDims; ++i)
        count *= static_cast<size_t>(dims.d[i]);
    return count * trt_dtype_byte_size(dtype);
}

void free_queue(moodycamel::ConcurrentQueue<tensors_struct *> &queue)
{
    tensors_struct *tensor;
    while (queue.try_dequeue(tensor))
        deep_free_tensors_struct(tensor);
}

shared_ptr<spdlog::logger> initialize_logger(const string &log_file,
                                             int file_level,
                                             int console_level,
                                             const string prefix)
{
    try
    {
        auto console_sink = make_shared<spdlog::sinks::stdout_color_sink_mt>();
        console_sink->set_level(static_cast<spdlog::level::level_enum>(console_level));

        auto file_sink = make_shared<spdlog::sinks::rotating_file_sink_st>(
            log_file, 1024 * 1024 * 5, 3); // 5 MB, 3 rotated files
        file_sink->set_level(static_cast<spdlog::level::level_enum>(file_level));

        static auto thread_pool = make_shared<spdlog::details::thread_pool>(8192, 1);

        auto logger = make_shared<spdlog::async_logger>(
            prefix, spdlog::sinks_init_list{console_sink, file_sink},
            thread_pool, spdlog::async_overflow_policy::overrun_oldest);

        spdlog::set_pattern("[%Y-%m-%d %H:%M:%S.%e] [" + prefix + "] [%^%l%$] %v");
        logger->set_level(static_cast<spdlog::level::level_enum>(
            std::min(file_level, console_level)));
        logger->flush_on(spdlog::level::info);

        return logger;
    }
    catch (const spdlog::spdlog_ex &ex)
    {
        cerr << "Logger initialization failed: " << ex.what() << "\n";
        exit(EXIT_FAILURE);
    }
}

void destroy_logger(shared_ptr<spdlog::logger> logger)
{
    if (logger)
    {
        logger->flush();
        spdlog::drop(logger->name());
        logger = nullptr;
    }
}
