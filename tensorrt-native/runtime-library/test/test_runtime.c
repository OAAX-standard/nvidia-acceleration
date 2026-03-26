/*
 * Minimal test for the TensorRT-native runtime library.
 * Tests load → send_input → receive_output with SqueezeNet (1,3,224,224).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* OAAX tensors_struct (must match the definition in c_utilities) */
typedef enum {
    DATA_TYPE_UNDEFINED = 0,
    DATA_TYPE_FLOAT = 1,
    DATA_TYPE_FLOAT16 = 10,
} tensor_data_type;

typedef struct {
    size_t num_tensors;
    char **names;
    tensor_data_type *data_types;
    size_t *ranks;
    size_t **shapes;
    void **data;
} tensors_struct;

/* OAAX C API */
extern int runtime_initialization(void);
extern int runtime_model_loading(const char *model_path);
extern int send_input(tensors_struct *input);
extern int receive_output(tensors_struct **output);
extern int runtime_destruction(void);
extern const char *runtime_name(void);
extern const char *runtime_version(void);

static tensors_struct *make_squeezenet_input(void)
{
    /* SqueezeNet input: [1, 3, 224, 224] float32 */
    size_t N = 1, C = 3, H = 224, W = 224;
    size_t n_elements = N * C * H * W;
    size_t n_bytes    = n_elements * sizeof(float);

    tensors_struct *ts = (tensors_struct *)malloc(sizeof(tensors_struct));
    ts->num_tensors  = 1;
    ts->names        = (char **)malloc(sizeof(char *));
    ts->data_types   = (tensor_data_type *)malloc(sizeof(tensor_data_type));
    ts->ranks        = (size_t *)malloc(sizeof(size_t));
    ts->shapes       = (size_t **)malloc(sizeof(size_t *));
    ts->data         = (void **)malloc(sizeof(void *));

    ts->names[0]      = strdup("data");
    ts->data_types[0] = DATA_TYPE_FLOAT;
    ts->ranks[0]      = 4;
    ts->shapes[0]     = (size_t *)malloc(4 * sizeof(size_t));
    ts->shapes[0][0]  = N;
    ts->shapes[0][1]  = C;
    ts->shapes[0][2]  = H;
    ts->shapes[0][3]  = W;

    float *buf = (float *)malloc(n_bytes);
    for (size_t i = 0; i < n_elements; ++i)
        buf[i] = 0.5f;
    ts->data[0] = buf;

    return ts;
}

int main(void)
{
    printf("=== TensorRT Runtime Test ===\n");
    printf("Runtime: %s\n", runtime_name());
    printf("Version: %s\n", runtime_version());

    /* Initialise */
    if (runtime_initialization() != 0) {
        fprintf(stderr, "FAIL: runtime_initialization\n");
        return 1;
    }
    printf("[OK] runtime_initialization\n");

    /* Load model */
    const char *engine_path = "/tmp/trt-output/model.trt";
    if (runtime_model_loading(engine_path) != 0) {
        fprintf(stderr, "FAIL: runtime_model_loading (%s)\n", engine_path);
        runtime_destruction();
        return 1;
    }
    printf("[OK] runtime_model_loading\n");

    /* Send input */
    tensors_struct *input = make_squeezenet_input();
    if (send_input(input) != 0) {
        fprintf(stderr, "FAIL: send_input\n");
        runtime_destruction();
        return 1;
    }
    printf("[OK] send_input\n");

    /* Poll for output (up to 10s) */
    tensors_struct *output = NULL;
    int attempts = 0;
    while (receive_output(&output) != 0 && attempts++ < 100)
        /* sleep 100ms (receive_output already sleeps internally) */;

    if (!output) {
        fprintf(stderr, "FAIL: receive_output timed out\n");
        runtime_destruction();
        return 1;
    }
    printf("[OK] receive_output: %zu output tensor(s)\n", output->num_tensors);

    for (size_t i = 0; i < output->num_tensors; ++i) {
        printf("  [%zu] name=%s  rank=%zu  shape=[",
               i, output->names[i], output->ranks[i]);
        for (size_t d = 0; d < output->ranks[i]; ++d)
            printf("%s%zu", d ? "," : "", output->shapes[i][d]);
        printf("]  dtype=%d\n", (int)output->data_types[i]);

        /* Print first 5 float values */
        if (output->data_types[i] == DATA_TYPE_FLOAT && output->data[i]) {
            float *vals = (float *)output->data[i];
            printf("  first values: ");
            for (size_t v = 0; v < 5; ++v) printf("%.4f ", vals[v]);
            printf("\n");
        }
    }

    /* Cleanup output (caller owns it) */
    for (size_t i = 0; i < output->num_tensors; ++i) {
        free(output->names[i]);
        free(output->shapes[i]);
        free(output->data[i]);
    }
    free(output->names);
    free(output->data_types);
    free(output->ranks);
    free(output->shapes);
    free(output->data);
    free(output);

    runtime_destruction();
    printf("[OK] runtime_destruction\n");
    printf("=== PASS ===\n");
    return 0;
}
