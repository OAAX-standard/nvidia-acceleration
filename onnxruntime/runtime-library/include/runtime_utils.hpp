#ifndef RUNTIME_UTILS_HPP
#define RUNTIME_UTILS_HPP

#include "runtime_core.hpp"
#include <vector>
#include <onnxruntime_cxx_api.h>
#include <spdlog/spdlog.h>
#include "concurrentqueue.h"

using namespace std;

vector<char *> get_output_names(Ort::Session &session);
ONNXTensorElementDataType map_to_ort_type(TensorElementType t);
TensorElementType map_to_tensors_struct_type(ONNXTensorElementDataType type);
int get_element_byte_size(TensorElementType t);
void deep_free_tensors(Tensors *t);
void free_queue(moodycamel::ConcurrentQueue<Tensors *> &queue);
shared_ptr<spdlog::logger> initialize_logger(const string &log_file,
                                              int file_level,
                                              int console_level,
                                              const string &prefix);
void destroy_logger(const shared_ptr<spdlog::logger> &logger);

#endif // RUNTIME_UTILS_HPP
