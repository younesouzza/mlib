#include "mlib/matrix.h"
#include <assert.h>

int main(void){
    Matrix m = matrix_create(2,3);
    Matrix m1 = matrix_create(2,0);
    Matrix m2 = matrix_create(0,0);
    Matrix m3 = matrix_create(10000000000ULL, 10000000000ULL);
    Matrix m4 = matrix_create(2,3);
    matrix_set(&m, 1, 1, 42.0f);
    assert(matrix_get(&m, 1, 1) == 42.0f);
    matrix_set(&m, 1, 1, 0);
  /* fill m and m4 for matrix_add testing */
    float tab1[] = {1.0f, 4.0f, 5.0f, 3.0f, 6.0f, 8.0f};
    float tab2[] = {2.0f, 3.0f, 5.0f, 6.0f, 1.0f, 2.0f};
    size_t c = 0;

    for (size_t i = 0; i < m.rows; i++)
    {
        for (size_t j = 0; j < m.cols; j++)
        {
            matrix_set(&m, i, j, tab1[c]);
            c++;
        }
    }

    c = 0;
    for (size_t i = 0; i < m4.rows; i++)
    {
        for (size_t j = 0; j < m4.cols; j++)
        {
            matrix_set(&m4, i, j, tab2[c]);
            c++;
        }
    }
    assert(matrix_get(&m, 1, 1) == 6.0f);


    assert(m.data != NULL);
    assert(m.cols == 3);
    assert(m.rows == 2);


    assert(m1.data == NULL);
    assert(m1.cols == 0);
    assert(m1.rows == 2);

    assert(m2.data == NULL);
    assert(m2.cols == 0);
    assert(m2.rows == 0);

    assert(m3.data == NULL);
    assert(m3.rows == 0);
    assert(m3.cols == 0);
      /* matrix_add: normal case */
    Matrix sum = matrix_add(&m, &m4);
    assert(sum.data != NULL);
    assert(sum.rows == 2);
    assert(sum.cols == 3);
    assert(matrix_get(&sum, 0, 0) == 3.0f);   /* 1 + 2 */
    assert(matrix_get(&sum, 0, 1) == 7.0f);   /* 4 + 3 */
    assert(matrix_get(&sum, 0, 2) == 10.0f);  /* 5 + 5 */
    assert(matrix_get(&sum, 1, 0) == 9.0f);   /* 3 + 6 */
    assert(matrix_get(&sum, 1, 1) == 7.0f);   /* 6 + 1 */
    assert(matrix_get(&sum, 1, 2) == 10.0f);  /* 8 + 2 */

    /* matrix_add: mismatched dimensions */
    Matrix bad_sum = matrix_add(&m, &m1);
    assert(bad_sum.data == NULL);
    assert(bad_sum.rows == 0);
    assert(bad_sum.cols == 0);

    /* matrix_scale: normal case */
    Matrix scaled = matrix_scale(&m, 2.0f);
    assert(scaled.data != NULL);
    assert(scaled.rows == 2);
    assert(scaled.cols == 3);
    assert(matrix_get(&scaled, 0, 0) == 2.0f);   /* 1 * 2 */
    assert(matrix_get(&scaled, 1, 2) == 16.0f);  /* 8 * 2 */

    /* matrix_multiply: normal case, rectangular shapes */
    Matrix a2 = matrix_create(2, 3);
    matrix_set(&a2, 0, 0, 1.0f);
    matrix_set(&a2, 0, 1, 2.0f);
    matrix_set(&a2, 0, 2, 3.0f);
    matrix_set(&a2, 1, 0, 4.0f);
    matrix_set(&a2, 1, 1, 5.0f);
    matrix_set(&a2, 1, 2, 6.0f);

    Matrix b2 = matrix_create(3, 2);
    matrix_set(&b2, 0, 0, 7.0f);
    matrix_set(&b2, 0, 1, 8.0f);
    matrix_set(&b2, 1, 0, 9.0f);
    matrix_set(&b2, 1, 1, 10.0f);
    matrix_set(&b2, 2, 0, 11.0f);
    matrix_set(&b2, 2, 1, 12.0f);

    Matrix product;
    bool mul_ok = matrix_multiply(&a2, &b2, &product);
    assert(mul_ok == true);
    assert(product.rows == 2);
    assert(product.cols == 2);
    assert(matrix_get(&product, 0, 0) == 58.0f);
    assert(matrix_get(&product, 0, 1) == 64.0f);
    assert(matrix_get(&product, 1, 0) == 139.0f);
    assert(matrix_get(&product, 1, 1) == 154.0f);

    matrix_destroy(&a2);
    matrix_destroy(&b2);
    matrix_destroy(&product);

    /* matrix_multiply: mismatched inner dimensions */
    Matrix a3 = matrix_create(2, 3);
    Matrix b3 = matrix_create(2, 2);  /* a3.cols (3) != b3.rows (2) */

    Matrix bad_product;
    bool mul_bad = matrix_multiply(&a3, &b3, &bad_product);
    assert(mul_bad == false);

    matrix_destroy(&a3);
    matrix_destroy(&b3);
    
    matrix_destroy(&scaled);
    matrix_destroy(&sum);
    matrix_destroy(&bad_sum);
    matrix_destroy(&m4);
    matrix_destroy(&m3);
    matrix_destroy(&m);
    matrix_destroy(&m);
    matrix_destroy(&m1);
    matrix_destroy(&m2);



    return 0;

}