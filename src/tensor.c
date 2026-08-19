#include "mlib/tensor.h"
#include <stdlib.h>
#include <string.h>
#include <stdint.h> 

Tensor tensor_create(size_t ndim, const size_t *shape_input)
{
    Tensor t = {0}; 

    size_t total_elements = 1;
    for (size_t i = 0; i < ndim; i++) {
        if (shape_input[i] == 0) {
            total_elements = 0;
            break;
        }
        if (total_elements > SIZE_MAX / shape_input[i]) {
            return t; 
        }
        total_elements *= shape_input[i];
    }

    if (total_elements > 0) {
        t.data = malloc(total_elements * sizeof(float));
        if (t.data == NULL) {
            return t; 
        }
    }

    if (ndim > 0) {
        t.shape = malloc(ndim * sizeof(size_t));
        if (t.shape == NULL) {
            free(t.data);
            return t;
        }
        memcpy(t.shape, shape_input, ndim * sizeof(size_t));
    }

    if (ndim > 0) {
        t.strides = malloc(ndim * sizeof(size_t));
        if (t.strides == NULL) {
            free(t.data);
            free(t.shape);
            return t;
        }
        
        t.strides[ndim - 1] = 1;
        for (size_t i = ndim - 1; i > 0; i--) {
            t.strides[i - 1] = t.strides[i] * t.shape[i];
        }
    }

    t.ndim = ndim;
    t.owns_data = true;

    return t;
}

void tensor_destroy(Tensor *t)
{
    if (t == NULL) {
        return;
    }

    if (t->owns_data) {
        free(t->data);
    }
    
    free(t->shape);
    free(t->strides);

    t->data = NULL;
    t->shape = NULL;
    t->strides = NULL;
    t->ndim = 0;
    t->owns_data = false;
}