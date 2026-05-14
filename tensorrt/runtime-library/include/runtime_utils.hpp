#ifndef RUNTIME_UTILS_HPP
#define RUNTIME_UTILS_HPP

#include "runtime_core.hpp"
#include <NvInfer.h>
#include <spdlog/spdlog.h>
#include "concurrentqueue.h"
#include <string>
#include <vector>

using namespace std;

// Map a TensorRT DataType to the OAAX tensor_data_type enum
tensor_data_type map_trt_to_oaax_type(nvinfer1::DataType dtype);

// Return the byte size of a single element for a TensorRT DataType
size_t trt_dtype_byte_size(nvinfer1::DataType dtype);

// Compute total byte size of a tensor given its dims and dtype
size_t compute_tensor_byte_size(const nvinfer1::Dims &dims, nvinfer1::DataType dtype);

shared_ptr<spdlog::logger> initialize_logger(const string &log_file,
                                             int file_level = spdlog::level::info,
                                             int console_level = spdlog::level::info,
                                             const string prefix = "OAAX");

void destroy_logger(shared_ptr<spdlog::logger> logger);

void free_queue(moodycamel::ConcurrentQueue<tensors_struct *> &queue);

#endif // RUNTIME_UTILS_HPP
