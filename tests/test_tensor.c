#include <stdio.h>
#include <assert.h>
#include <stdint.h>
#include "mlib/tensor.h"

int main(void)
{
    /* Test 1: 1D Tensor (behaves like a Vector) */
    {
        size_t shape[] = {5};
        Tensor t = tensor_create(1, shape);
        
        assert(t.ndim == 1);
        assert(t.shape[0] == 5);
        assert(t.strides[0] == 1);
        assert(t.data != NULL);
        assert(t.owns_data == true);
        
        tensor_destroy(&t);
    }

    /* Test 2: 2D Tensor (behaves like a Matrix) */
    {
        size_t shape[] = {3, 4};
        Tensor t = tensor_create(2, shape);
        
        assert(t.ndim == 2);
        assert(t.shape[0] == 3);
        assert(t.shape[1] == 4);
        assert(t.strides[0] == 4);
        assert(t.strides[1] == 1);
        assert(t.data != NULL);
        
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
        
        tensor_destroy(&t);
    }

    /* Test 4: Zero-dimension edge case [2, 0, 4] */
    {
        size_t shape[] = {2, 0, 4};
        Tensor t = tensor_create(3, shape);
        
        assert(t.ndim == 3);
        assert(t.data == NULL); /* total_elements is 0, so no data allocated */
        assert(t.shape[0] == 2);
        assert(t.shape[1] == 0);
        assert(t.shape[2] == 4);
        
        /* Strides calculation: strides[2]=1, strides[1]=1*4=4, strides[0]=4*0=0 */
        assert(t.strides[0] == 0);
        assert(t.strides[1] == 4);
        assert(t.strides[2] == 1);
        
        tensor_destroy(&t);
    }

    /* Test 5: Overflow failure */
    {
        size_t shape[] = {SIZE_MAX, 2};
        Tensor t = tensor_create(2, shape);
        
        /* Must return the zero-initialized safe state */
        assert(t.ndim == 0); 
        assert(t.data == NULL);
        assert(t.shape == NULL);
        assert(t.strides == NULL);
        
        tensor_destroy(&t); /* Must not crash on a failed tensor */
    }

    /* Test 6: Idempotent destroy and NULL destroy */
    {
        size_t shape[] = {5};
        Tensor t = tensor_create(1, shape);
        
        tensor_destroy(&t);
        tensor_destroy(&t); /* Second destroy must be safe */
        
        tensor_destroy(NULL); /* Passing NULL directly must be safe */
    }

    printf("All tensor tests passed!\n");
    return 0;
}