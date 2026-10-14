#ifndef OAAX_TEST_UTILS_H
#define OAAX_TEST_UTILS_H

#include <cstdlib>
#include <cstring>

#include "interface.h"

// Allocate a float Tensors with all elements set to 0.5.
// Caller owns the returned pointer; free with free_tensors().
static inline Tensors* make_float_input(const char* name, int* shape, int rank, int request_id) {
    Tensors* ts = (Tensors*)malloc(sizeof(Tensors));
    ts->id = request_id;
    ts->num_tensors = 1;
    ts->tensors = (TensorDescriptor*)malloc(sizeof(TensorDescriptor));

    TensorDescriptor& td = ts->tensors[0];
    td.name = strdup(name);
    td.data_type = DATA_TYPE_FLOAT;
    td.rank = rank;
    td.shape = (int*)malloc(rank * sizeof(int));
    size_t n_elems = 1;
    for (int i = 0; i < rank; i++) {
        td.shape[i] = shape[i];
        n_elems *= (size_t)shape[i];
    }
    td.data_size = n_elems * sizeof(float);
    float* buf = (float*)malloc(td.data_size);
    for (size_t i = 0; i < n_elems; i++) buf[i] = 0.5f;
    td.data = buf;
    return ts;
}

// Free a Tensors allocated by make_float_input or returned by runtime_retrieve_output.
static inline void free_tensors(Tensors* ts) {
    if (!ts) return;
    for (int i = 0; i < ts->num_tensors; i++) {
        free(ts->tensors[i].name);
        free(ts->tensors[i].shape);
        free(ts->tensors[i].data);
    }
    free(ts->tensors);
    free(ts);
}

#endif // OAAX_TEST_UTILS_H
