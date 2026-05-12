#ifndef RUNTIME_CORE_HPP
#define RUNTIME_CORE_HPP

extern "C" {
#include "tensors_struct.h"
}

#define EXPOSE_FUNCTION __attribute__((visibility("default")))

extern "C" EXPOSE_FUNCTION int runtime_initialization_with_args(int length, char **keys, void **values);

extern "C" EXPOSE_FUNCTION int runtime_initialization();

extern "C" EXPOSE_FUNCTION int runtime_model_loading(const char *model_path);

extern "C" EXPOSE_FUNCTION int send_input(tensors_struct *input_tensors);

extern "C" EXPOSE_FUNCTION int receive_output(tensors_struct **output_tensors);

extern "C" EXPOSE_FUNCTION int runtime_destruction();

extern "C" EXPOSE_FUNCTION const char *runtime_error_message();

extern "C" EXPOSE_FUNCTION const char *runtime_version();

extern "C" EXPOSE_FUNCTION const char *runtime_name();

#endif // RUNTIME_CORE_HPP
