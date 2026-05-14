// Copyright (c) OAAX. All rights reserved.
// Licensed under the Apache License, Version 2.0.
// Source: https://github.com/OAAX-standard/tools

#ifndef C_UTILITIES_INCLUDE_TENSORS_STRUCT_H_
#define C_UTILITIES_INCLUDE_TENSORS_STRUCT_H_

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum tensor_data_type {
  DATA_TYPE_UNDEFINED = 0,
  DATA_TYPE_FLOAT = 1,
  DATA_TYPE_UINT8 = 2,
  DATA_TYPE_INT8 = 3,
  DATA_TYPE_UINT16 = 4,
  DATA_TYPE_INT16 = 5,
  DATA_TYPE_INT32 = 6,
  DATA_TYPE_INT64 = 7,
  DATA_TYPE_STRING = 8,
  DATA_TYPE_BOOL = 9,
  DATA_TYPE_FLOAT16 = 10,
  DATA_TYPE_DOUBLE = 11,
  DATA_TYPE_UINT32 = 12,
  DATA_TYPE_UINT64 = 13,
  DATA_TYPE_COMPLEX64 = 14,
  DATA_TYPE_COMPLEX128 = 15,
  DATA_TYPE_BFLOAT16 = 16,
} tensor_data_type;

typedef struct tensors_struct {
  size_t num_tensors;
  char** names;
  tensor_data_type* data_types;
  size_t* ranks;
  size_t** shapes;
  void** data;
} tensors_struct;

#ifdef __cplusplus
}
#endif

#endif  // C_UTILITIES_INCLUDE_TENSORS_STRUCT_H_
