#ifndef MLIB_TENSOR_H
#define MLIB_TENSOR_H

#include <stddef.h>
#include <stdbool.h>

typedef struct {
    float *data;
    size_t *shape;
    size_t *strides;
    size_t ndim;
    bool owns_data;
} Tensor;


Tensor tensor_create(size_t ndim, const size_t *shape_input);
void tensor_destroy(Tensor *t);

#endif