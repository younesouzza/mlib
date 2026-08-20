#include <stdio.h>
#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include "mlib/tensor.h"

/* 
 * Helper for floating-point comparison. 
 * Never use == for floats in an ML library!
 */
static void assert_float_eq(float actual, float expected, float epsilon, const char* msg) {
    if (fabsf(actual - expected) > epsilon) {
        printf("FAIL: %s | Expected %f, got %f\n", msg, expected, actual);
        exit(1);
    }
}

int main(void)
{
    float epsilon = 1e-5f;

    /* Test 1: 1D Tensor (behaves like a Vector) */
    {
        size_t shape[] = {5};
        Tensor t = tensor_create(1, shape);
        
        assert(t.ndim == 1);
        assert(t.strides[0] == 1);

        // Write and Read using the new API
        for (size_t i = 0; i < 5; i++) {
            // C99 Compound literal to create the indices array on the fly
            tensor_set(&t, (size_t[]){i}, i * 1.5f); 
        }
        
        for (size_t i = 0; i < 5; i++) {
            float val = tensor_get(&t, (size_t[]){i});
            assert_float_eq(val, i * 1.5f, epsilon, "1D get/set");
        }
        
        tensor_destroy(&t);
    }

    /* Test 2: 2D Tensor (behaves like a Matrix) */
    {
        size_t shape[] = {3, 4};
        Tensor t = tensor_create(2, shape);
        
        assert(t.ndim == 2);
        assert(t.strides[0] == 4);
        assert(t.strides[1] == 1);

        // Write and Read using the new API
        for (size_t i = 0; i < 3; i++) {
            for (size_t j = 0; j < 4; j++) {
                float value = (i * 10.0f) + j; // Unique value for each cell
                tensor_set(&t, (size_t[]){i, j}, value);
            }
        }
        
        for (size_t i = 0; i < 3; i++) {
            for (size_t j = 0; j < 4; j++) {
                float expected = (i * 10.0f) + j;
                float val = tensor_get(&t, (size_t[]){i, j});
                assert_float_eq(val, expected, epsilon, "2D get/set");
            }
        }
        
        tensor_destroy(&t);
    }

    /* Test 3: 3D Tensor */
    {
        size_t shape[] = {2, 3, 4};
        Tensor t = tensor_create(3, shape);
        
        assert(t.ndim == 3);
        assert(t.strides[0] == 12);
        assert(t.strides[1] == 4);
        assert(t.strides[2] == 1);

        // Write and Read using the new API
        for (size_t i = 0; i < 2; i++) {
            for (size_t j = 0; j < 3; j++) {
                for (size_t k = 0; k < 4; k++) {
                    float value = (i * 100.0f) + (j * 10.0f) + k;
                    tensor_set(&t, (size_t[]){i, j, k}, value);
                }
            }
        }
        
        for (size_t i = 0; i < 2; i++) {
            for (size_t j = 0; j < 3; j++) {
                for (size_t k = 0; k < 4; k++) {
                    float expected = (i * 100.0f) + (j * 10.0f) + k;
                    float val = tensor_get(&t, (size_t[]){i, j, k});
                    assert_float_eq(val, expected, epsilon, "3D get/set");
                }
            }
        }
        
        tensor_destroy(&t);
    }

    /* Test 4: Zero-dimension edge case [2, 0, 4] */
    {
        size_t shape[] = {2, 0, 4};
        Tensor t = tensor_create(3, shape);
        
        assert(t.ndim == 3);
        assert(t.data == NULL);
        assert(t.shape[1] == 0);
        
        tensor_destroy(&t);
    }

    /* Test 5: Overflow failure */
    {
        size_t shape[] = {SIZE_MAX, 2};
        Tensor t = tensor_create(2, shape);
        
        assert(t.ndim == 0); 
        assert(t.data == NULL);
        
        tensor_destroy(&t);
    }

    /* Test 6: Idempotent destroy and NULL destroy */
    {
        size_t shape[] = {5};
        Tensor t = tensor_create(1, shape);
        
        tensor_destroy(&t);
        tensor_destroy(&t); 
        tensor_destroy(NULL); 
    }

    return 0;
}