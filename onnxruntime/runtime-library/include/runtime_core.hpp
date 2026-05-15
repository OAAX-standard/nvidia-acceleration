#ifndef RUNTIME_CORE_HPP
#define RUNTIME_CORE_HPP

#include "interface.h"

#ifdef _WIN32
#define EXPOSE_FUNCTION __declspec(dllexport)
#else
#define EXPOSE_FUNCTION __attribute__((visibility("default")))
#endif

extern "C" EXPOSE_FUNCTION RuntimeStatus runtime_init(Config config);
extern "C" EXPOSE_FUNCTION RuntimeStatus runtime_load_models(int num_models, const ModelConfig *model_configs);
extern "C" EXPOSE_FUNCTION RuntimeStatus runtime_enqueue_input(int model_id, Tensors *input_tensors);
extern "C" EXPOSE_FUNCTION RuntimeStatus runtime_retrieve_output(int *model_id, Tensors **output_tensors, int timeout_ms);
extern "C" EXPOSE_FUNCTION RuntimeStatus runtime_cleanup(void);
extern "C" EXPOSE_FUNCTION const char *runtime_get_error(void);
extern "C" EXPOSE_FUNCTION const char *runtime_get_version(void);
extern "C" EXPOSE_FUNCTION const char *runtime_get_name(void);
extern "C" EXPOSE_FUNCTION const char *runtime_get_info(void);

#endif // RUNTIME_CORE_HPP
