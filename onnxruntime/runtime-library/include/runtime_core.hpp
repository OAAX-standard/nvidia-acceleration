#ifndef RUNTIME_CORE_HPP
#define RUNTIME_CORE_HPP

#include "interface.h"

// On Windows, exports are listed in src/runtime_exports.def: MSVC rejects
// adding __declspec(dllexport) to functions already declared in interface.h
// without it (error C2375).
#ifdef _WIN32
#define EXPOSE_FUNCTION
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
