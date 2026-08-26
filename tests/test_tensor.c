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

}
void test_tensor_add(void)
{
    float epsilon = 1e-5f;

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

void test_tensor_reshape(void)
{
    float epsilon = 1e-5f;

    /* Test 1: The Happy Path (1D to 2D) */
    {
        size_t shape[] = {12};
        Tensor t = tensor_create(1, shape);
        
        // Fill with values 1.0 to 12.0
        for (size_t i = 0; i < 12; i++) {
            tensor_set(&t, (size_t[]){i}, (float)(i + 1));
        }

        size_t new_shape[] = {3, 4};
        Tensor view = tensor_reshape(&t, 2, new_shape);
        
        assert(view.ndim == 2);
        assert(view.shape[0] == 3);
        assert(view.shape[1] == 4);
        assert(view.strides[0] == 4);
        assert(view.strides[1] == 1);
        assert(view.owns_data == false); // Must be a view!

        // Verify data integrity
        assert_float_eq(tensor_get(&view, (size_t[]){0, 0}), 1.0f, epsilon, "r 0,0");
        assert_float_eq(tensor_get(&view, (size_t[]){1, 2}), 7.0f, epsilon, "r 1,2"); // 1*4 + 2 = 6 -> value 7
        assert_float_eq(tensor_get(&view, (size_t[]){2, 3}), 12.0f, epsilon, "r 2,3");

        tensor_destroy(&t);
        tensor_destroy(&view);
    }

    /* Test 2: The Shared Memory Proof (Zero-Copy) */
    {
        size_t shape[] = {6};
        Tensor t = tensor_create(1, shape);
        
        for (size_t i = 0; i < 6; i++) tensor_set(&t, (size_t[]){i}, 0.0f);

        size_t new_shape[] = {2, 3};
        Tensor view = tensor_reshape(&t, 2, new_shape);

        // CRITICAL: Modify the ORIGINAL tensor at index 4
        tensor_set(&t, (size_t[]){4}, 99.0f);

        // Read from the RESHAPED view. 
        // In a [2, 3] matrix, index 4 is row 1, col 1.
        float val = tensor_get(&view, (size_t[]){1, 1});
        assert_float_eq(val, 99.0f, epsilon, "shared memory");

        tensor_destroy(&t);
        tensor_destroy(&view);
    }

    /* Test 3: Failure Path 1 (Mismatched Total Elements) */
    {
        size_t shape[] = {12};
        Tensor t = tensor_create(1, shape);
        
        // Try to reshape 12 elements into a 3x5 (15 elements) matrix
        size_t bad_shape[] = {3, 5};
        Tensor bad_view = tensor_reshape(&t, 2, bad_shape);
        
        // Must return a safe, zero-initialized state
        assert(bad_view.ndim == 0);
        assert(bad_view.data == NULL);
        assert(bad_view.shape == NULL);
        
        tensor_destroy(&t);
        tensor_destroy(&bad_view); // Must be safe to destroy a failed tensor
    }

    /* Test 4: Failure Path 2 (Non-Contiguous Tensor) */
    {
        
        // 1. Create a contiguous 2x3 matrix
        size_t shape_a[] = {2, 3};
        Tensor a = tensor_create(2, shape_a);
        for(size_t i=0; i<2; i++) 
            for(size_t j=0; j<3; j++) 
                tensor_set(&a, (size_t[]){i, j}, 1.0f);

        // 2. Manually create a transposed (non-contiguous) view
        Tensor a_view = {0};
        a_view.data = a.data;           
        a_view.ndim = 2;
        a_view.shape = (size_t[]){3, 2};
        a_view.strides = (size_t[]){1, 3}; // Scrambled strides!
        a_view.owns_data = false;       

        // 3. Try to reshape the non-contiguous view into a 1D array
        size_t new_shape[] = {6};
        Tensor bad_reshape = tensor_reshape(&a_view, 1, new_shape);
        
        // Must fail safely because a_view is not contiguous
        assert(bad_reshape.ndim == 0);
        assert(bad_reshape.data == NULL);
        
        tensor_destroy(&a);
        tensor_destroy(&bad_reshape);
        // DO NOT destroy a_view
    }

}
void test_tensor_slice(void)
{
    float epsilon = 1e-5f;
    /* Test 1: Happy Path (2D Slice) */
    {
        size_t shape[] = {10, 10};
        Tensor t = tensor_create(2, shape);
        
        /* Fill with values: t[i][j] = i * 10 + j */
        for (size_t i = 0; i < 10; i++) {
            for (size_t j = 0; j < 10; j++) {
                tensor_set(&t, (size_t[]){i, j}, (float)(i * 10 + j));
            }
        }

        /* Slice a 3x3 block starting at [2, 2] */
        size_t start[] = {2, 2};
        size_t slice_shape[] = {3, 3};
        Tensor view = tensor_slice(&t, start, slice_shape);
        
        assert(view.ndim == 2);
        assert(view.shape[0] == 3);
        assert(view.shape[1] == 3);
        assert(view.strides[0] == 10); /* Inherited from parent */
        assert(view.strides[1] == 1);  /* Inherited from parent */
        assert(view.owns_data == false);

        assert_float_eq(tensor_get(&view, (size_t[]){0, 0}), 22.0f, epsilon, "slice 0,0");
        assert_float_eq(tensor_get(&view, (size_t[]){0, 1}), 23.0f, epsilon, "slice 0,1");
        assert_float_eq(tensor_get(&view, (size_t[]){0, 2}), 24.0f, epsilon, "slice 0,2");
        assert_float_eq(tensor_get(&view, (size_t[]){1, 0}), 32.0f, epsilon, "slice 1,0");
        assert_float_eq(tensor_get(&view, (size_t[]){2, 2}), 44.0f, epsilon, "slice 2,2");

        tensor_destroy(&t);
        tensor_destroy(&view);
    }

    /* Test 2: The Shared Memory Proof (Zero-Copy) */
    {
        size_t shape[] = {5, 5};
        Tensor t = tensor_create(2, shape);
        
        for (size_t i = 0; i < 5; i++) {
            for (size_t j = 0; j < 5; j++) {
                tensor_set(&t, (size_t[]){i, j}, 0.0f);
            }
        }

        size_t start[] = {1, 1};
        size_t slice_shape[] = {2, 2};
        Tensor view = tensor_slice(&t, start, slice_shape);

        /* CRITICAL: Modify the ORIGINAL tensor at [2, 2] */
        tensor_set(&t, (size_t[]){2, 2}, 99.0f);

      
        float val = tensor_get(&view, (size_t[]){1, 1});
        assert_float_eq(val, 99.0f, epsilon, "shared memory");

        tensor_destroy(&t);
        tensor_destroy(&view);
    }

    /* Test 3: Non-Contiguous Slice (Slice of a Transpose) */
    {
        
        /* 1. Create a contiguous 2x3 matrix */
        size_t shape_a[] = {2, 3};
        Tensor a = tensor_create(2, shape_a);
        
        tensor_set(&a, (size_t[]){0, 0}, 1.0f);
        tensor_set(&a, (size_t[]){0, 1}, 2.0f);
        tensor_set(&a, (size_t[]){0, 2}, 3.0f);
        tensor_set(&a, (size_t[]){1, 0}, 4.0f);
        tensor_set(&a, (size_t[]){1, 1}, 5.0f);
        tensor_set(&a, (size_t[]){1, 2}, 6.0f);

        /* 2. Transpose it (non-contiguous view) */
        Tensor a_t = tensor_transpose(&a);
        
        assert(a_t.strides[0] == 1);
        assert(a_t.strides[1] == 3);

        /* 3. Slice the transposed view: take a 2x1 block starting at [0, 0] */
        size_t start[] = {0, 0};
        size_t slice_shape[] = {2, 1};
        Tensor slice_of_transpose = tensor_slice(&a_t, start, slice_shape);
        
     
        assert(slice_of_transpose.ndim == 2);
        assert(slice_of_transpose.shape[0] == 2);
        assert(slice_of_transpose.shape[1] == 1);
        
        /* The slice inherits the scrambled strides from the transposed parent */
        assert(slice_of_transpose.strides[0] == 1);
        assert(slice_of_transpose.strides[1] == 3);

        assert_float_eq(tensor_get(&slice_of_transpose, (size_t[]){0, 0}), 1.0f, epsilon, "nc slice 0,0");
        assert_float_eq(tensor_get(&slice_of_transpose, (size_t[]){1, 0}), 2.0f, epsilon, "nc slice 1,0");

        tensor_destroy(&a);
        tensor_destroy(&a_t);
        tensor_destroy(&slice_of_transpose);
    }

    /* Test 4: Failure Path (Out of Bounds) */
    {
        size_t shape[] = {10, 10};
        Tensor t = tensor_create(2, shape);
        
     
        size_t start[] = {9, 9};
        size_t slice_shape[] = {3, 3};
        Tensor bad_slice = tensor_slice(&t, start, slice_shape);
        
        /* Must return a safe, zero-initialized state */
        assert(bad_slice.ndim == 0);
        assert(bad_slice.data == NULL);
        assert(bad_slice.shape == NULL);
        
        tensor_destroy(&t);
        tensor_destroy(&bad_slice);
    }

    /* Test 5: Edge Case (Empty Slice with 0 dimension) */
    {
        size_t shape[] = {5, 5};
        Tensor t = tensor_create(2, shape);
        
        /* Slice with shape [0, 3] - valid, but creates an empty tensor */
        size_t start[] = {0, 0};
        size_t slice_shape[] = {0, 3};
        Tensor empty_slice = tensor_slice(&t, start, slice_shape);
        
        assert(empty_slice.ndim == 2);
        assert(empty_slice.shape[0] == 0);
        assert(empty_slice.shape[1] == 3);
        assert(empty_slice.owns_data == false);
        
        tensor_destroy(&t);
        tensor_destroy(&empty_slice);
    }

    /* Test 6: 3D Slice (Batched Matrix) */
    {
        /* Shape: [Batch=4, Rows=5, Cols=6] */
        size_t shape[] = {4, 5, 6};
        Tensor t = tensor_create(3, shape);
        
        /* Fill with values: t[b][i][j] = b*100 + i*10 + j */
        for (size_t b = 0; b < 4; b++) {
            for (size_t i = 0; i < 5; i++) {
                for (size_t j = 0; j < 6; j++) {
                    float val = (float)(b * 100 + i * 10 + j);
                    tensor_set(&t, (size_t[]){b, i, j}, val);
                }
            }
        }

        /* Extract a sub-batch: batches [1, 2], rows [2, 3], cols [3, 4] */
        size_t start[] = {1, 2, 3};
        size_t slice_shape[] = {2, 2, 2};
        Tensor view = tensor_slice(&t, start, slice_shape);
        
        assert(view.ndim == 3);
        assert(view.shape[0] == 2);
        assert(view.shape[1] == 2);
        assert(view.shape[2] == 2);
        
        /* 
         * view[0,0,0] = t[1,2,3] = 1*100 + 2*10 + 3 = 123
         * view[1,1,1] = t[2,3,4] = 2*100 + 3*10 + 4 = 234
         */
        assert_float_eq(tensor_get(&view, (size_t[]){0, 0, 0}), 123.0f, epsilon, "3D 0,0,0");
        assert_float_eq(tensor_get(&view, (size_t[]){1, 1, 1}), 234.0f, epsilon, "3D 1,1,1");

        tensor_destroy(&t);
        tensor_destroy(&view);
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
    test_tensor_reshape();
    test_tensor_slice();

    

    return 0;


}