/*
 * windows_load_test.c — diagnostic loader for the OAAX NVIDIA ORT runtime DLL.
 *
 * Adapted from the equivalent tool in intel-acceleration (2026-07 Windows
 * loading audit). Loads RuntimeLibrary.dll step by step and prints the exact
 * Win32 error at every stage, so a load failure is never silent. Checks:
 *   1. The DLL file exists and its PE header matches the process bitness
 *   2. The MSVC runtime DLLs are resolvable
 *   3. Each bundled ONNX Runtime DLL loads from the runtime's directory
 *   4. The CUDA Toolkit DLLs (NOT bundled — must come from a system install
 *      matching the build's CUDA version) are resolvable on PATH
 *   5. The runtime DLL itself loads
 *   6. All nine OAAX v2 symbols resolve via GetProcAddress
 *   7. runtime_get_name/version/info return sane strings
 *   8. (--init) runtime_init + runtime_cleanup succeed
 *   9. (--model) full round-trip: load model, enqueue input, retrieve output
 *
 * Build (MSVC, x64 Native Tools prompt):
 *   cl /W4 windows_load_test.c
 *
 * Usage:
 *   windows_load_test.exe <path\to\RuntimeLibrary.dll> [--init] [--cuda-version N]
 *                         [--model model.onnx] [--input-name NAME] [--input-shape N,N,...]
 *
 * --model implies --init. All API calls are resolved dynamically via
 * GetProcAddress, so this tool never needs RuntimeLibrary.lib — only the DLL.
 */

#ifndef _WIN32
#error This diagnostic tool is Windows-only.
#endif

#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <string.h>
#include <windows.h>

/* ---- minimal OAAX v2 declarations (keep in sync with include/interface.h) ---- */

typedef int RuntimeStatus; /* RUNTIME_STATUS_SUCCESS == 0 */

typedef struct Config {
    int length;
    const char **keys;
    const char **values;
} Config;

typedef struct TensorDescriptor {
    char *name;
    int data_type; /* DATA_TYPE_FLOAT == 1 */
    int rank;
    int *shape;
    size_t data_size;
    void *data;
} TensorDescriptor;

typedef struct Tensors {
    int id;
    int num_tensors;
    TensorDescriptor *tensors;
} Tensors;

typedef struct ModelConfig {
    const char *file_path;
    const unsigned char *model_data;
    size_t model_size;
    Config config;
} ModelConfig;

typedef RuntimeStatus (*runtime_init_fn)(Config);
typedef RuntimeStatus (*runtime_load_models_fn)(int, const ModelConfig *);
typedef RuntimeStatus (*runtime_enqueue_input_fn)(int, Tensors *);
typedef RuntimeStatus (*runtime_retrieve_output_fn)(int *, Tensors **, int);
typedef RuntimeStatus (*runtime_cleanup_fn)(void);
typedef const char *(*get_string_fn)(void);

static int parse_shape(const char *s, int *out, int max_dims) {
    int n = 0;
    char buf[256];
    strncpy(buf, s, sizeof(buf) - 1);
    buf[sizeof(buf) - 1] = 0;
    char *tok = strtok(buf, ",");
    while (tok && n < max_dims) {
        out[n++] = atoi(tok);
        tok = strtok(NULL, ",");
    }
    return n;
}

static Tensors *make_float_input(const char *name, const int *shape, int rank) {
    Tensors *ts = (Tensors *)malloc(sizeof(Tensors));
    ts->id = 1;
    ts->num_tensors = 1;
    ts->tensors = (TensorDescriptor *)malloc(sizeof(TensorDescriptor));
    TensorDescriptor *td = &ts->tensors[0];
    size_t name_len = strlen(name) + 1;
    td->name = (char *)malloc(name_len);
    memcpy(td->name, name, name_len);
    td->data_type = 1; /* DATA_TYPE_FLOAT */
    td->rank = rank;
    td->shape = (int *)malloc((size_t)rank * sizeof(int));
    size_t n_elems = 1;
    for (int i = 0; i < rank; i++) {
        td->shape[i] = shape[i];
        n_elems *= (size_t)shape[i];
    }
    td->data_size = n_elems * sizeof(float);
    float *buf = (float *)malloc(td->data_size);
    for (size_t i = 0; i < n_elems; i++) buf[i] = 0.5f;
    td->data = buf;
    return ts;
}

/* Frees a Tensors struct returned BY THE RUNTIME (output). Never call this on
 * the input struct passed to runtime_enqueue_input — the runtime owns and
 * frees that one internally after inference. */
static void free_tensors(Tensors *ts) {
    if (!ts) return;
    for (int i = 0; i < ts->num_tensors; i++) {
        free(ts->tensors[i].name);
        free(ts->tensors[i].shape);
        free(ts->tensors[i].data);
    }
    free(ts->tensors);
    free(ts);
}

/* Cast FARPROC via void(*)(void) to avoid -Wcast-function-type warnings. */
#define GET_FN(type, module, name) ((type)(void (*)(void))GetProcAddress(module, name))

/* ------------------------------------------------------------------------ */

static void print_last_error_ex(const char *what, int with_hints) {
    DWORD err = GetLastError();
    char msg[512] = {0};
    FormatMessageA(FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS, NULL, err,
                   MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), msg, sizeof(msg) - 1, NULL);
    /* strip trailing newline that FormatMessage appends */
    size_t n = strlen(msg);
    while (n > 0 && (msg[n - 1] == '\n' || msg[n - 1] == '\r')) msg[--n] = 0;

    fprintf(stderr, "  FAILED: %s\n  GetLastError = %lu (0x%lX): %s\n", what, (unsigned long)err, (unsigned long)err,
            msg);

    if (!with_hints) return;
    switch (err) {
        case ERROR_MOD_NOT_FOUND: /* 126 */
            fprintf(stderr,
                    "  Hint: the DLL itself was found, but one of its DEPENDENCIES is\n"
                    "  missing (onnxruntime.dll, a CUDA Toolkit DLL, or the MSVC runtime).\n"
                    "  onnxruntime*.dll must sit next to RuntimeLibrary.dll; CUDA DLLs\n"
                    "  (cudart64_*, cublas64_*, cublasLt64_*, cudnn64_*) must come from a\n"
                    "  system CUDA Toolkit install on PATH matching the build's CUDA version.\n"
                    "  Run 'dumpbin /dependents RuntimeLibrary.dll' to list them.\n");
            break;
        case ERROR_PROC_NOT_FOUND: /* 127 */
            fprintf(stderr,
                    "  Hint: a dependency DLL was found but has the wrong version —\n"
                    "  an expected export is missing (e.g. a mismatched CUDA Toolkit\n"
                    "  version on PATH shadowing the one the runtime was built against).\n");
            break;
        case ERROR_BAD_EXE_FORMAT: /* 193 */
            fprintf(stderr, "  Hint: 32/64-bit mismatch between this process and the DLL.\n");
            break;
        case ERROR_DLL_INIT_FAILED: /* 1114 */
            fprintf(stderr, "  Hint: DllMain / a static initializer threw during load.\n");
            break;
        default:
            break;
    }
}

static void print_last_error(const char *what) { print_last_error_ex(what, 1); }

/* Read the PE header and return the machine type (0 on failure). */
static WORD pe_machine_type(const char *path) {
    WORD machine = 0;
    FILE *f = fopen(path, "rb");
    if (!f) return 0;
    IMAGE_DOS_HEADER dos;
    if (fread(&dos, sizeof(dos), 1, f) == 1 && dos.e_magic == IMAGE_DOS_SIGNATURE) {
        DWORD sig;
        if (fseek(f, dos.e_lfanew, SEEK_SET) == 0 && fread(&sig, sizeof(sig), 1, f) == 1 && sig == IMAGE_NT_SIGNATURE) {
            IMAGE_FILE_HEADER fh;
            if (fread(&fh, sizeof(fh), 1, f) == 1) machine = fh.Machine;
        }
    }
    fclose(f);
    return machine;
}

/* Probe a single dependency by name using the default search order. */
static void probe_dependency(const char *name) {
    HMODULE h = LoadLibraryA(name);
    if (h) {
        char where[MAX_PATH] = {0};
        GetModuleFileNameA(h, where, sizeof(where) - 1);
        printf("  OK  %-28s -> %s\n", name, where);
        FreeLibrary(h);
    } else {
        printf("  MISSING  %s\n", name);
        print_last_error_ex(name, 0);
    }
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr,
                "Usage: %s <path\\to\\RuntimeLibrary.dll> [--init] [--cuda-version N]\n"
                "       [--model model.onnx] [--input-name NAME] [--input-shape N,N,...]\n",
                argv[0]);
        return 2;
    }
    const char *dll_path = argv[1];
    int do_init = 0;
    const char *cuda_version = "12";
    const char *model_path = NULL;
    const char *input_name = "images";
    const char *input_shape_str = "1,3,640,640";
    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--init") == 0) do_init = 1;
        else if (strcmp(argv[i], "--cuda-version") == 0 && i + 1 < argc) cuda_version = argv[++i];
        else if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) model_path = argv[++i];
        else if (strcmp(argv[i], "--input-name") == 0 && i + 1 < argc) input_name = argv[++i];
        else if (strcmp(argv[i], "--input-shape") == 0 && i + 1 < argc) input_shape_str = argv[++i];
    }
    if (model_path) do_init = 1; /* a model round-trip requires an initialized runtime */

    /* Turn OS error dialogs into return codes so nothing blocks or hides. */
    SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOOPENFILEERRORBOX);

    printf("== OAAX NVIDIA ORT Windows runtime load test ==\n");
    printf("Process: %u-bit\n", (unsigned)(sizeof(void *) * 8));
    printf("Target DLL: %s\n\n", dll_path);

    /* --- 1. file exists + bitness ----------------------------------------- */
    printf("[1] Checking file and PE header\n");
    DWORD attrs = GetFileAttributesA(dll_path);
    if (attrs == INVALID_FILE_ATTRIBUTES) {
        print_last_error("GetFileAttributes (file not found?)");
        return 1;
    }
    WORD machine = pe_machine_type(dll_path);
    const char *arch = (machine == IMAGE_FILE_MACHINE_AMD64)   ? "x64"
                       : (machine == IMAGE_FILE_MACHINE_I386)  ? "x86"
                       : (machine == IMAGE_FILE_MACHINE_ARM64) ? "ARM64"
                                                               : "unknown";
    printf("  File exists. PE machine type: 0x%X (%s)\n", machine, arch);
    int process_is_64 = (sizeof(void *) == 8);
    int dll_is_64 = (machine == IMAGE_FILE_MACHINE_AMD64 || machine == IMAGE_FILE_MACHINE_ARM64);
    if (process_is_64 != dll_is_64) {
        fprintf(stderr,
                "  FATAL: this test process is %u-bit but the DLL is %s. Windows can\n"
                "  never load it — every load below will fail with error 193. This is\n"
                "  a property of the LOADING PROCESS, not the DLL:\n"
                "    - If test.exe is 32-bit: rebuild it from the 'x64 Native Tools\n"
                "      Command Prompt' (plain 'Developer Command Prompt' defaults to x86).\n"
                "    - If your real host app has the same bitness, that IS your bug —\n"
                "      a 32-bit host cannot load this x64 runtime.\n",
                (unsigned)(sizeof(void *) * 8), arch);
    }

    /* --- 2. make the DLL's own directory searchable for its dependencies -- */
    char dir[MAX_PATH] = {0};
    char full[MAX_PATH] = {0};
    GetFullPathNameA(dll_path, sizeof(full), full, NULL);
    strncpy(dir, full, sizeof(dir) - 1);
    char *slash = strrchr(dir, '\\');
    if (slash) *slash = 0;
    printf("\n[2] Adding DLL directory to search path: %s\n", dir);
    if (!SetDllDirectoryA(dir)) print_last_error("SetDllDirectory");

    /* --- 3. probe bundled dependencies individually ------------------------ */
    printf("\n[3] Probing bundled dependencies (each via its own LoadLibrary)\n");
    printf("  -- MSVC runtime --\n");
    probe_dependency("vcruntime140.dll");
    probe_dependency("vcruntime140_1.dll");
    probe_dependency("msvcp140.dll");
    probe_dependency("msvcp140_1.dll");
    printf("  -- ONNX Runtime --\n");
    probe_dependency("onnxruntime.dll");
    probe_dependency("onnxruntime_providers_shared.dll");
    /* Do NOT probe onnxruntime_providers_cuda.dll directly: a raw LoadLibrary
     * here (outside ORT's own provider-bridge sequence, which loads
     * onnxruntime_providers_shared.dll and wires up its host pointer first)
     * runs this DLL's static initializers before that wiring exists, crashing
     * with a null-pointer read in its own global provider-bridge init. Only
     * ORT's real path (AppendExecutionProvider_CUDA) loads this DLL safely. */
    printf("  SKIP onnxruntime_providers_cuda.dll (a raw LoadLibrary probe crashes"
           " it -- only ORT's own AppendExecutionProvider_CUDA path loads it safely)\n");

    /* --- 3b. CUDA Toolkit DLLs — NOT bundled, must come from a system install.
     * Per-version import sets of onnxruntime_providers_cuda.dll (dumpbin
     * /dependents): CUDA 12 needs cudart64_12 + cublas64_12 + cublasLt64_12 +
     * cufft64_11; CUDA 13 statically links cudart and needs only cublas64_13 +
     * cublasLt64_13 + cufft64_12. */
    printf("\n[3b] Probing CUDA Toolkit %s DLLs (NOT bundled — resolved via PATH,\n"
           "     loaded at runtime by onnxruntime_providers_cuda.dll, so they never\n"
           "     appear in RuntimeLibrary.dll's own import table)\n", cuda_version);
    char name_buf[64];
    if (strcmp(cuda_version, "13") != 0) { /* CUDA 13: cudart is statically linked */
        snprintf(name_buf, sizeof(name_buf), "cudart64_%s.dll", cuda_version);
        probe_dependency(name_buf);
    }
    snprintf(name_buf, sizeof(name_buf), "cublas64_%s.dll", cuda_version);
    probe_dependency(name_buf);
    snprintf(name_buf, sizeof(name_buf), "cublasLt64_%s.dll", cuda_version);
    probe_dependency(name_buf);
    probe_dependency(strcmp(cuda_version, "13") == 0 ? "cufft64_12.dll" : "cufft64_11.dll");
    probe_dependency("cudnn64_9.dll");
    printf("  If any of these are MISSING, install the matching CUDA Toolkit\n"
           "  version on this machine (or add its bin\\ dir to PATH) — the\n"
           "  runtime's own folder deliberately does not bundle these.\n");

    /* --- 3b2. cuDNN 9 component DLLs — cudnn64_9.dll is only a loader shim:
     * at cudnnCreate() time it LoadLibrary()s cudnn_graph64_9.dll and the
     * cudnn_engines_* DLLs via the STANDARD search order (host exe dir,
     * System32, PATH — this tool's SetDllDirectory also counts, but a real
     * host's LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR does NOT). If the components
     * can't be found, cuDNN abort()s the whole host process (fail-fast
     * 0xc0000409 in cudnn64_9.dll — seen with mediaserver.exe 2026-08-11). */
    printf("\n[3b2] Checking cuDNN 9 component resolvability (missing => cuDNN abort()s the host)\n");
    {
        char found[MAX_PATH];
        if (SearchPathA(NULL, "cudnn_graph64_9.dll", NULL, MAX_PATH, found, NULL) > 0) {
            printf("  OK  cudnn_graph64_9.dll          -> %s\n", found);
        } else {
            fprintf(stderr,
                    "  MISSING cudnn_graph64_9.dll — cudnnCreate() would abort() this\n"
                    "  process. The runtime works around this by prepending its own\n"
                    "  directory to PATH in runtime_init (builds after 2026-08-11); for\n"
                    "  older builds, the full cuDNN package must sit in a PATH directory.\n");
        }
    }

    /* --- 3c. CWD writability — an explicit CWD-relative log_file would need it
     * (the built-in default has been %TEMP%\oaax-runtime-<pid>.log since the
     * 2026-07 audit, so an unwritable CWD is informational only) ----------- */
    printf("\n[3c] Checking CWD writability (matters only for a CWD-relative log_file config)\n");
    char cwd[MAX_PATH] = {0};
    GetCurrentDirectoryA(sizeof(cwd), cwd);
    printf("  CWD: %s\n", cwd);
    HANDLE probe = CreateFileA(".__oaax_write_probe", GENERIC_WRITE, 0, NULL, CREATE_ALWAYS,
                               FILE_ATTRIBUTE_TEMPORARY | FILE_FLAG_DELETE_ON_CLOSE, NULL);
    if (probe == INVALID_HANDLE_VALUE) {
        print_last_error("CWD write probe");
        fprintf(stderr,
                "  NOTE: CWD is NOT writable. The runtime falls back to a console-only\n"
                "  logger in this case and returns normally (it no longer calls exit()),\n"
                "  but you will get no runtime.log file — pass an explicit writable\n"
                "  'log_file' path in the runtime_init Config if you need one.\n");
    } else {
        printf("  CWD is writable — default runtime.log creation will succeed.\n");
        CloseHandle(probe);
    }

    /* --- 4. load the runtime DLL ------------------------------------------ */
    printf("\n[4] Loading %s\n", full);
    HMODULE h = LoadLibraryExA(full, NULL, LOAD_WITH_ALTERED_SEARCH_PATH);
    if (!h) {
        print_last_error("LoadLibraryEx(runtime DLL)");
        fprintf(stderr,
                "\nFor the exact failing import, run in cmd:\n"
                "  dumpbin /dependents \"%s\"\n"
                "or enable loader snaps: gflags /i %s +sls, then check debugger output.\n",
                full, argv[0]);
        return 1;
    }
    printf("  OK — loaded at %p\n", (void *)h);

    /* --- 5. resolve all OAAX symbols --------------------------------------- */
    printf("\n[5] Resolving OAAX v2 exports\n");
    static const char *symbols[] = {
        "runtime_init",    "runtime_load_models", "runtime_enqueue_input", "runtime_retrieve_output",
        "runtime_cleanup", "runtime_get_error",   "runtime_get_version",   "runtime_get_name",
        "runtime_get_info"};
    int missing = 0;
    for (size_t i = 0; i < sizeof(symbols) / sizeof(symbols[0]); i++) {
        FARPROC p = GetProcAddress(h, symbols[i]);
        printf("  %-26s %s\n", symbols[i], p ? "OK" : "MISSING");
        if (!p) missing++;
    }
    if (missing) {
        fprintf(stderr,
                "  %d symbol(s) missing — exports not declared or name-mangled.\n"
                "  Check with: dumpbin /exports \"%s\"\n",
                missing, full);
        return 1;
    }

    /* --- 6. call the safe informational functions -------------------------- */
    printf("\n[6] Calling informational functions\n");
    get_string_fn get_name = GET_FN(get_string_fn, h, "runtime_get_name");
    get_string_fn get_version = GET_FN(get_string_fn, h, "runtime_get_version");
    get_string_fn get_error = GET_FN(get_string_fn, h, "runtime_get_error");
    printf("  runtime_get_name()    = %s\n", get_name() ? get_name() : "(null)");
    printf("  runtime_get_version() = %s\n", get_version() ? get_version() : "(null)");

    /* --- 7. optional init/cleanup round-trip -------------------------------- */
    if (do_init) {
        printf("\n[7] runtime_init (--init)\n");
        runtime_init_fn init = GET_FN(runtime_init_fn, h, "runtime_init");
        runtime_cleanup_fn cleanup = GET_FN(runtime_cleanup_fn, h, "runtime_cleanup");
        const char *keys[] = {"log_stdout"};
        const char *values[] = {"true"};
        Config cfg = {1, keys, values};
        printf("  Calling runtime_init...\n");
        fflush(stdout);
        RuntimeStatus st = init(cfg);
        printf("  runtime_init = %d %s\n", st, st == 0 ? "(SUCCESS)" : "(FAILED)");
        if (st != 0) {
            const char *e = get_error();
            fprintf(stderr, "  runtime_get_error() = %s\n", e ? e : "(null)");
            return 1;
        }
        /* --- 8. optional full model round-trip -------------------------------- */
        if (model_path) {
            printf("\n[8] Model round-trip: %s\n", model_path);
            runtime_load_models_fn load_models = GET_FN(runtime_load_models_fn, h, "runtime_load_models");
            runtime_enqueue_input_fn enqueue = GET_FN(runtime_enqueue_input_fn, h, "runtime_enqueue_input");
            runtime_retrieve_output_fn retrieve = GET_FN(runtime_retrieve_output_fn, h, "runtime_retrieve_output");

            Config empty_cfg = {0, NULL, NULL};
            ModelConfig mc = {model_path, NULL, 0, empty_cfg};
            st = load_models(1, &mc);
            printf("  runtime_load_models = %d %s\n", st, st == 0 ? "(SUCCESS)" : "(FAILED)");
            if (st != 0) {
                fprintf(stderr, "  runtime_get_error() = %s\n", get_error());
                cleanup();
                return 1;
            }

            int shape[8];
            int rank = parse_shape(input_shape_str, shape, 8);
            printf("  Input: name=%s shape=%s\n", input_name, input_shape_str);
            Tensors *input = make_float_input(input_name, shape, rank);
            st = enqueue(0, input);
            printf("  runtime_enqueue_input = %d %s\n", st, st == 0 ? "(SUCCESS)" : "(FAILED)");
            if (st != 0) {
                fprintf(stderr, "  runtime_get_error() = %s\n", get_error());
                cleanup();
                return 1;
            }

            Tensors *output = NULL;
            int out_model_id = -1;
            RuntimeStatus rst = 12; /* RUNTIME_STATUS_NO_OUTPUT_AVAILABLE */
            for (int poll = 0; poll < 200 && rst != 0; poll++) rst = retrieve(&out_model_id, &output, 50);
            if (rst != 0 || !output) {
                fprintf(stderr, "  FAILED: runtime_retrieve_output timed out (status=%d)\n", rst);
                cleanup();
                return 1;
            }
            printf("  runtime_retrieve_output = 0 (SUCCESS) — model_id=%d, %d output tensor(s)\n", out_model_id,
                   output->num_tensors);
            for (int i = 0; i < output->num_tensors; i++)
                printf("    [%d] %s: rank=%d data_size=%zu\n", i, output->tensors[i].name, output->tensors[i].rank,
                       output->tensors[i].data_size);
            free_tensors(output);
        }

        st = cleanup();
        printf("\n[9] runtime_cleanup = %d %s\n", st, st == 0 ? "(SUCCESS)" : "(FAILED)");
    } else {
        printf("\n[7] Skipped runtime_init (pass --init or --model to enable)\n");
    }

    FreeLibrary(h);
    printf("\nAll checks passed.\n");
    return 0;
}
