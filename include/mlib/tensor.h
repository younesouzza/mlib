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
float tensor_get(const Tensor *t, const size_t *indices);
void tensor_set(Tensor *t, const size_t *indices, float value);
Tensor tensor_add(const Tensor *a, const Tensor *b);
Tensor tensor_scale(const Tensor *a, float scalar);
Tensor tensor_transpose(const Tensor *t);
Tensor tensor_reshape(const Tensor *t, size_t new_ndim, const size_t *new_shape);
Tensor tensor_slice(const Tensor *t, const size_t *start_indices, const size_t *slice_shape);

#endif