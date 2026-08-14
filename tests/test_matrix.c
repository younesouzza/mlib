#include "mlib/matrix.h"
#include <assert.h>

int main(void){
    Matrix m = matrix_create(2,3);
    Matrix m1 = matrix_create(2,0);
    Matrix m2 = matrix_create(0,0);
    Matrix m3 = matrix_create(10000000000ULL, 10000000000ULL);

    

    assert(m.data != NULL);
    assert(m.cols == 3);
    assert(m.rows == 2);
    matrix_set(&m, 1, 1, 42.0f);
    assert(matrix_get(&m, 1, 1) == 42.0f);

    assert(m1.data == NULL);
    assert(m1.cols == 0);
    assert(m1.rows == 2);

    assert(m2.data == NULL);
    assert(m2.cols == 0);
    assert(m2.rows == 0);

    assert(m3.data == NULL);
    assert(m3.rows == 0);
    assert(m3.cols == 0);

    matrix_destroy(&m3);
    matrix_destroy(&m);
    matrix_destroy(&m);
    matrix_destroy(&m1);
    matrix_destroy(&m2);



    return 0;

}