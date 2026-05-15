#ifndef RUNTIME_UTILS_HPP
#define RUNTIME_UTILS_HPP

#include "runtime_core.hpp"
#include <NvInfer.h>
#include <spdlog/spdlog.h>
#include "concurrentqueue.h"
#include <string>
#include <vector>

using namespace std;

TensorElementType map_trt_to_oaax_type(nvinfer1::DataType dtype);
size_t trt_dtype_byte_size(nvinfer1::DataType dtype);
size_t compute_tensor_byte_size(const nvinfer1::Dims &dims, nvinfer1::DataType dtype);
void deep_free_tensors(Tensors *t);
void free_queue(moodycamel::ConcurrentQueue<Tensors *> &queue);
shared_ptr<spdlog::logger> initialize_logger(const string &log_file,
                                              int file_level,
                                              int console_level,
                                              const string prefix);
void destroy_logger(shared_ptr<spdlog::logger> logger);

#endif // RUNTIME_UTILS_HPP
