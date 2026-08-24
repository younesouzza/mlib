#include <stdio.h>
#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include "mlib/tensor.h"

static void assert_float_eq(float actual, float expected, float epsilon, const char* msg) {
    if (fabsf(actual - expected) > epsilon) {
        printf("FAIL: %s | Expected %f, got %f\n", msg, expected, actual);
        exit(1);
    }
}
void test_tensor_scale(void)
{
    float epsilon = 1e-5f;
    printf("Running tensor_scale tests...\n");

    /* Test 1: Contiguous Scaling */
    {
        size_t shape[] = {2, 3};
        Tensor a = tensor_create(2, shape);
        
        for (size_t i = 0; i < 2; i++) {
            for (size_t j = 0; j < 3; j++) {
                tensor_set(&a, (size_t[]){i, j}, 2.0f);
            }
        }

        Tensor scaled = tensor_scale(&a, 3.0f);
        assert(scaled.ndim == 2);
        
        for (size_t i = 0; i < 2; i++) {
            for (size_t j = 0; j < 3; j++) {
                float val = tensor_get(&scaled, (size_t[]){i, j});
                assert_float_eq(val, 6.0f, epsilon, "contiguous scale");
            }
        }
        
        tensor_destroy(&a);
        tensor_destroy(&scaled);
    }

    /* Test 2: Non-Contiguous (Transposed View) Scaling */
    {
        /* Create a contiguous 2x3 matrix filled with 4.0 */
        size_t shape_a[] = {2, 3};
        Tensor a = tensor_create(2, shape_a);
        for(size_t i=0; i<2; i++) 
            for(size_t j=0; j<3; j++) 
                tensor_set(&a, (size_t[]){i, j}, 4.0f);

        /* Manually create a transposed view (shape [3, 2], strides [1, 3]) */
        Tensor a_view = {0};
        a_view.data = a.data;           
        a_view.ndim = 2;
        a_view.shape = (size_t[]){3, 2};
        a_view.strides = (size_t[]){1, 3}; 
        a_view.owns_data = false;       

        /* Scale the view by 2.0. Expected result: 3x2 matrix of 8.0 */
        Tensor scaled_view = tensor_scale(&a_view, 2.0f);
        
        assert(scaled_view.ndim == 2);
        assert(scaled_view.shape[0] == 3);
        assert(scaled_view.shape[1] == 2);

        for(size_t i=0; i<3; i++) {
            for(size_t j=0; j<2; j++) {
                float val = tensor_get(&scaled_view, (size_t[]){i, j});
                assert_float_eq(val, 8.0f, epsilon, "non-contiguous scale");
            }
        }

        tensor_destroy(&a);
        tensor_destroy(&scaled_view);
        /* DO NOT destroy a_view */
    }

    /* Test 3: Zero-element tensor scaling */
    {
        size_t shape[] = {2, 0, 4};
        Tensor a = tensor_create(3, shape);
        Tensor scaled = tensor_scale(&a, 5.0f);
        
        assert(scaled.ndim == 3);
        assert(scaled.data == NULL);
        
        tensor_destroy(&a);
        tensor_destroy(&scaled);
    }

    printf("tensor_scale tests PASSED\n");
}
void test_tensor_add(void)
{
    float epsilon = 1e-5f;
    printf("Running tensor_add tests...\n");

    {
        size_t shape[] = {2, 3};
        Tensor a = tensor_create(2, shape);
        Tensor b = tensor_create(2, shape);

        for (size_t i = 0; i < 2; i++) {
            for (size_t j = 0; j < 3; j++) {
                tensor_set(&a, (size_t[]){i, j}, 1.0f);
                tensor_set(&b, (size_t[]){i, j}, 2.0f);
            }
        }
        Tensor sum = tensor_add(&a, &b);
        assert(sum.ndim == 2);

        for (size_t i = 0; i < 2; i++) {
            for (size_t j = 0; j < 3; j++) {
                float val = tensor_get(&sum, (size_t[]){i, j});
                assert_float_eq(val, 3.0f, epsilon, "contiguous add");
            }
        }

        tensor_destroy(&a);
        tensor_destroy(&b);
        tensor_destroy(&sum);
    }

    {
        size_t shape_a[] = {2, 3};
        Tensor a = tensor_create(2, shape_a);
        for (size_t i = 0; i < 2; i++)
            for (size_t j = 0; j < 3; j++)
                tensor_set(&a, (size_t[]){i, j}, (float)(i * 10 + j));

        Tensor b_view = {0};
        b_view.data = a.data;
        b_view.ndim = 2;
        b_view.shape = (size_t[]){3, 2};
        b_view.strides = (size_t[]){1, 3};
        b_view.owns_data = false;

        size_t shape_c[] = {3, 2};
        Tensor c = tensor_create(2, shape_c);
        for (size_t i = 0; i < 3; i++)
            for (size_t j = 0; j < 2; j++)
                tensor_set(&c, (size_t[]){i, j}, 2.0f);


        Tensor sum = tensor_add(&b_view, &c);
        assert(sum.ndim == 2);
        assert(sum.shape[0] == 3);
        assert(sum.shape[1] == 2);

        float expected[3][2] = {
            {2.0f, 12.0f},
            {3.0f, 13.0f},
            {4.0f, 14.0f}
        };

        for (size_t i = 0; i < 3; i++) {
            for (size_t j = 0; j < 2; j++) {
                float val = tensor_get(&sum, (size_t[]){i, j});
                assert_float_eq(val, expected[i][j], epsilon, "non-contiguous add");
            }
        }

        tensor_destroy(&a);
        tensor_destroy(&c);
        tensor_destroy(&sum);
    }

    printf("tensor_add tests PASSED\n");
}
void test_tensor_transpose(void)
{
    float epsilon = 1e-5f;

    /* Test 1: 2D Transpose & Stride Verification */
    {
        size_t shape[] = {2, 3};
        Tensor t = tensor_create(2, shape);
        
        // Fill with distinct values: [[1, 2, 3], [4, 5, 6]]
        tensor_set(&t, (size_t[]){0, 0}, 1.0f);
        tensor_set(&t, (size_t[]){0, 1}, 2.0f);
        tensor_set(&t, (size_t[]){0, 2}, 3.0f);
        tensor_set(&t, (size_t[]){1, 0}, 4.0f);
        tensor_set(&t, (size_t[]){1, 1}, 5.0f);
        tensor_set(&t, (size_t[]){1, 2}, 6.0f);

        Tensor view = tensor_transpose(&t);
        
        // 1. Verify Shape and Strides swapped
        assert(view.ndim == 2);
        assert(view.shape[0] == 3);
        assert(view.shape[1] == 2);
        assert(view.strides[0] == 1); // Was 3
        assert(view.strides[1] == 3); // Was 1

        // 2. Verify logical data mapping
        assert_float_eq(tensor_get(&view, (size_t[]){0, 0}), 1.0f, epsilon, "view 0,0");
        assert_float_eq(tensor_get(&view, (size_t[]){0, 1}), 4.0f, epsilon, "view 0,1"); // Swapped
        assert_float_eq(tensor_get(&view, (size_t[]){2, 1}), 6.0f, epsilon, "view 2,1"); // Swapped

        tensor_destroy(&t);
        tensor_destroy(&view); 
    }
    /* Test 2: The "View" Proof (Shared Memory) */
    {
        size_t shape[] = {2, 2};
        Tensor t = tensor_create(2, shape);
        
        tensor_set(&t, (size_t[]){0, 0}, 1.0f);
        tensor_set(&t, (size_t[]){0, 1}, 2.0f);
        tensor_set(&t, (size_t[]){1, 0}, 3.0f);
        tensor_set(&t, (size_t[]){1, 1}, 4.0f);

        Tensor view = tensor_transpose(&t);
        tensor_set(&t, (size_t[]){0, 1}, 99.0f);
        float val = tensor_get(&view, (size_t[]){1, 0});
        assert_float_eq(val, 99.0f, epsilon, "shared memory");

        tensor_destroy(&t);
        tensor_destroy(&view);
    }

    /* Test 3: 3D Tensor (Batched Matrix Transpose) */
    {
        // Shape: [Batch=2, Rows=3, Cols=4]
        size_t shape[] = {2, 3, 4};
        Tensor t = tensor_create(3, shape);
        
        // Expected original strides: [12, 4, 1]
        assert(t.strides[0] == 12);
        assert(t.strides[1] == 4);
        assert(t.strides[2] == 1);

        Tensor view = tensor_transpose(&t);

        // Expected new shape: [2, 4, 3]
        assert(view.ndim == 3);
        assert(view.shape[0] == 2);
        assert(view.shape[1] == 4);
        assert(view.shape[2] == 3);

        // Expected new strides: [12, 1, 4] (Only last two swapped)
        assert(view.strides[0] == 12);
        assert(view.strides[1] == 1);
        assert(view.strides[2] == 4);

        // Verify a specific value mapping
        tensor_set(&t, (size_t[]){1, 2, 3}, 42.0f);
        float val = tensor_get(&view, (size_t[]){1, 3, 2});
        assert_float_eq(val, 42.0f, epsilon, "3D mapping");

        tensor_destroy(&t);
        tensor_destroy(&view);
    }

    /* Test 4: Edge Cases (NULL and 1D) */
    {        
        Tensor null_view = tensor_transpose(NULL);
        assert(null_view.ndim == 0);
        assert(null_view.data == NULL);
        tensor_destroy(&null_view);

        size_t shape_1d[] = {5};
        Tensor t_1d = tensor_create(1, shape_1d);
        Tensor bad_view = tensor_transpose(&t_1d);
        assert(bad_view.ndim == 0); // Should fail safely
        assert(bad_view.data == NULL);
        
        tensor_destroy(&t_1d);
        tensor_destroy(&bad_view);
    }
}

int main(void)
{
    float epsilon = 1e-5f;

    {
        size_t shape[] = {5};
        Tensor t = tensor_create(1, shape);
        
        assert(t.ndim == 1);
        assert(t.strides[0] == 1);

        for (size_t i = 0; i < 5; i++) {
            tensor_set(&t, (size_t[]){i}, i * 1.5f); 
        }
        
        for (size_t i = 0; i < 5; i++) {
            float val = tensor_get(&t, (size_t[]){i});
            assert_float_eq(val, i * 1.5f, epsilon, "1D get/set");
        }
        
        tensor_destroy(&t);
    }
    {
        size_t shape[] = {3, 4};
        Tensor t = tensor_create(2, shape);
        
        assert(t.ndim == 2);
        assert(t.strides[0] == 4);
        assert(t.strides[1] == 1);
        for (size_t i = 0; i < 3; i++) {
            for (size_t j = 0; j < 4; j++) {
                float value = (i * 10.0f) + j; 
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
    test_tensor_add();
    test_tensor_scale();
    test_tensor_transpose();
    test_tensor_transpose();

    

    return 0;


}