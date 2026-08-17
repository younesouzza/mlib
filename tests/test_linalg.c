#include "mlib/linalg.h"
#include <assert.h>

int main(void)
{
    /* matrix_vector_multiply: normal case */
    Matrix m = matrix_create(2, 3);
    matrix_set(&m, 0, 0, 1.0f);
    matrix_set(&m, 0, 1, 2.0f);
    matrix_set(&m, 0, 2, 3.0f);
    matrix_set(&m, 1, 0, 4.0f);
    matrix_set(&m, 1, 1, 5.0f);
    matrix_set(&m, 1, 2, 6.0f);

    Vector v = vector_create(3);
    v.data[0] = 1.0f;
    v.data[1] = 0.0f;
    v.data[2] = 1.0f;

    /* hand-computed:
       result[0] = 1*1 + 2*0 + 3*1 = 4
       result[1] = 4*1 + 5*0 + 6*1 = 10 */

    Vector result;
    bool ok = matrix_vector_multiply(&m, &v, &result);
    assert(ok == true);
    assert(result.length == 2);
    assert(result.data[0] == 4.0f);
    assert(result.data[1] == 10.0f);

    matrix_destroy(&m);
    vector_destroy(&v);
    vector_destroy(&result);

    /* matrix_vector_multiply: mismatched dimensions */
    Matrix m2 = matrix_create(2, 3);
    Vector v2 = vector_create(4);  /* m2.cols (3) != v2.length (4) */

    Vector bad_result;
    bool bad_ok = matrix_vector_multiply(&m2, &v2, &bad_result);
    assert(bad_ok == false);

    matrix_destroy(&m2);
    vector_destroy(&v2);

    return 0;
}