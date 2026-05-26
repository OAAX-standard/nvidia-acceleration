/*
 * Minimal smoke test: dlopen libRuntimeLibrary.so and call runtime_init.
 * Does not require a model file.
 *
 * Build:
 *   gcc test_load.c -o test_load -ldl
 *
 * Usage:
 *   ./test_load /path/to/libRuntimeLibrary.so
 */
#include <stdio.h>
#include <dlfcn.h>

typedef enum { RUNTIME_STATUS_SUCCESS = 0 } RuntimeStatus;
typedef struct { int length; const char **keys; const char **values; } Config;

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "Usage: %s /path/to/libRuntimeLibrary.so\n", argv[0]);
        return 1;
    }

    /* Load the shared library */
    void *lib = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!lib) {
        fprintf(stderr, "FAIL: dlopen — %s\n", dlerror());
        return 1;
    }
    printf("[OK] dlopen\n");

    /* Resolve symbols */
    const char *(*get_name)(void)    = dlsym(lib, "runtime_get_name");
    const char *(*get_version)(void) = dlsym(lib, "runtime_get_version");
    const char *(*get_error)(void)   = dlsym(lib, "runtime_get_error");
    RuntimeStatus (*init)(Config)    = dlsym(lib, "runtime_init");
    RuntimeStatus (*cleanup)(void)   = dlsym(lib, "runtime_cleanup");

    if (!get_name || !get_version || !get_error || !init || !cleanup) {
        fprintf(stderr, "FAIL: dlsym — %s\n", dlerror());
        dlclose(lib);
        return 1;
    }
    printf("[OK] dlsym\n");

    printf("Runtime: %s\n", get_name());
    printf("Version: %s\n", get_version());

    /* Initialise — this is where the CUDA EP shared library is loaded */
    Config cfg = {0, NULL, NULL};
    if (init(cfg) != RUNTIME_STATUS_SUCCESS) {
        fprintf(stderr, "FAIL: runtime_init — %s\n", get_error());
        dlclose(lib);
        return 1;
    }
    printf("[OK] runtime_init\n");

    cleanup();
    dlclose(lib);
    printf("=== PASS ===\n");
    return 0;
}
