#include "mlib/tensor.h"
#include <stdlib.h>
#include <string.h>
#include <stdint.h> 
#include <assert.h>

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

static size_t tensor_calc_offset(const Tensor *t, const size_t *indices)
{
    size_t offset = 0;
    
    for (size_t i = 0; i < t->ndim; i++) {
        assert(indices[i] < t->shape[i]); 
        offset += indices[i] * t->strides[i];
    }
    
    return offset;
}

float tensor_get(const Tensor *t, const size_t *indices)
{
    assert(t != NULL);
    assert(t->data != NULL);
    
    size_t offset = tensor_calc_offset(t, indices);
    return t->data[offset];
}

void tensor_set(Tensor *t, const size_t *indices, float value)
{
    assert(t != NULL);
    assert(t->data != NULL);
    
    size_t offset = tensor_calc_offset(t, indices);
    t->data[offset] = value;
}

Tensor tensor_add(const Tensor *a, const Tensor *b)
{
    if (a == NULL || b == NULL) {
        return (Tensor){0};
    }

    if (a->ndim != b->ndim) {
        return (Tensor){0};
    }
    for (size_t i = 0; i < a->ndim; i++) {
        if (a->shape[i] != b->shape[i]) {
            return (Tensor){0};
        }
    }

    Tensor result = tensor_create(a->ndim, a->shape);


    size_t total_elements = 1;
    for (size_t i = 0; i < a->ndim; i++) {
        total_elements *= a->shape[i];
    }

    if (result.data == NULL && total_elements > 0) {
        return result;  
    }
    if (total_elements == 0) {
        return result;  
    }

    if (a->ndim == 0) {
        result.data[0] = a->data[0] + b->data[0];
        return result;
    }

    size_t *coords = calloc(a->ndim, sizeof(size_t));
    if (coords == NULL) {
        tensor_destroy(&result);
        return (Tensor){0};
    }

    int carry;
    do {
        size_t offset_a = tensor_calc_offset(a, coords);
        size_t offset_b = tensor_calc_offset(b, coords);
        size_t offset_r = tensor_calc_offset(&result, coords);

        result.data[offset_r] = a->data[offset_a] + b->data[offset_b];

        carry = 1;
        for (int i = (int)a->ndim - 1; i >= 0 && carry; i--) {
            coords[i] += (size_t)carry;
            if (coords[i] >= a->shape[i]) {
                coords[i] = 0;
                carry = 1;
            } else {
                carry = 0;
            }
        }
    } while (!carry);

    free(coords);
    return result;
}