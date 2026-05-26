/*
 * Minimal test for the ORT-CUDA runtime library (OAAX v2 API).
 * Tests init → load_models → enqueue_input → retrieve_output with SqueezeNet (1,3,224,224).
 *
 * Usage:
 *   gcc test_runtime.c -o test_runtime -L<lib_dir> -lRuntimeLibrary -Wl,-rpath,<lib_dir> -I../include
 *   ./test_runtime [path/to/model.onnx]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "interface.h"

#define DEFAULT_MODEL_PATH "/tmp/ort-output/model.onnx"

/* Build a single-tensor Tensors* for SqueezeNet: [1, 3, 224, 224] float32 */
static Tensors *make_squeezenet_input(void)
{
    size_t N = 1, C = 3, H = 224, W = 224;
    size_t n_elements = N * C * H * W;
    size_t n_bytes    = n_elements * sizeof(float);

    Tensors *ts = (Tensors *)malloc(sizeof(Tensors));
    ts->id          = 0;
    ts->num_tensors = 1;
    ts->tensors     = (TensorDescriptor *)malloc(sizeof(TensorDescriptor));

    ts->tensors[0].name      = strdup("data");
    ts->tensors[0].data_type = DATA_TYPE_FLOAT;
    ts->tensors[0].rank      = 4;
    ts->tensors[0].shape     = (int *)malloc(4 * sizeof(int));
    ts->tensors[0].shape[0]  = (int)N;
    ts->tensors[0].shape[1]  = (int)C;
    ts->tensors[0].shape[2]  = (int)H;
    ts->tensors[0].shape[3]  = (int)W;
    ts->tensors[0].data_size = n_bytes;

    float *buf = (float *)malloc(n_bytes);
    for (size_t i = 0; i < n_elements; ++i)
        buf[i] = 0.5f;
    ts->tensors[0].data = buf;

    return ts;
}

int main(int argc, char **argv)
{
    const char *model_path = (argc > 1) ? argv[1] : DEFAULT_MODEL_PATH;

    printf("=== ORT-CUDA Runtime Test (OAAX v2) ===\n");
    printf("Runtime: %s\n", runtime_get_name());
    printf("Version: %s\n", runtime_get_version());
    printf("Model:   %s\n", model_path);

    /* Initialise — this is where the CUDA EP shared library is loaded */
    Config cfg = {0, NULL, NULL};
    if (runtime_init(cfg) != RUNTIME_STATUS_SUCCESS) {
        fprintf(stderr, "FAIL: runtime_init — %s\n", runtime_get_error());
        return 1;
    }
    printf("[OK] runtime_init\n");

    /* Load model */
    ModelConfig mc = {model_path, NULL, 0, {0, NULL, NULL}};
    if (runtime_load_models(1, &mc) != RUNTIME_STATUS_SUCCESS) {
        fprintf(stderr, "FAIL: runtime_load_models (%s) — %s\n",
                model_path, runtime_get_error());
        runtime_cleanup();
        return 1;
    }
    printf("[OK] runtime_load_models\n");

    /* Enqueue input */
    Tensors *input = make_squeezenet_input();
    if (runtime_enqueue_input(0, input) != RUNTIME_STATUS_SUCCESS) {
        fprintf(stderr, "FAIL: runtime_enqueue_input — %s\n", runtime_get_error());
        runtime_cleanup();
        return 1;
    }
    printf("[OK] runtime_enqueue_input\n");

    /* Retrieve output (block up to 10 seconds) */
    Tensors *output = NULL;
    int model_id = -1;
    RuntimeStatus st = runtime_retrieve_output(&model_id, &output, 10000);
    if (st != RUNTIME_STATUS_SUCCESS || !output) {
        fprintf(stderr, "FAIL: runtime_retrieve_output (status=%d) — %s\n",
                st, runtime_get_error());
        runtime_cleanup();
        return 1;
    }
    printf("[OK] runtime_retrieve_output: model_id=%d, %d output tensor(s)\n",
           model_id, output->num_tensors);

    for (int i = 0; i < output->num_tensors; ++i) {
        TensorDescriptor *td = &output->tensors[i];
        printf("  [%d] name=%s  rank=%d  shape=[", i, td->name, td->rank);
        for (int d = 0; d < td->rank; ++d)
            printf("%s%d", d ? "," : "", td->shape[d]);
        printf("]  dtype=%d  data_size=%zu\n", (int)td->data_type, td->data_size);

        /* Print first 5 float values */
        if (td->data_type == DATA_TYPE_FLOAT && td->data) {
            float *vals = (float *)td->data;
            printf("  first values: ");
            for (int v = 0; v < 5; ++v) printf("%.4f ", vals[v]);
            printf("\n");
        }
    }

    /* Free output (caller owns it) */
    for (int i = 0; i < output->num_tensors; ++i) {
        free(output->tensors[i].name);
        free(output->tensors[i].shape);
        free(output->tensors[i].data);
    }
    free(output->tensors);
    free(output);

    runtime_cleanup();
    printf("[OK] runtime_cleanup\n");
    printf("=== PASS ===\n");
    return 0;
}
