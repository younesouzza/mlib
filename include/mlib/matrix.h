#ifndef MLIB_MATRIX_H
#define MLIB_MATRIX_H

#include <stddef.h>
typedef struct 
{
    float *data;
    size_t rows;
    size_t cols;
}Matrix;

Matrix matrix_create(size_t rows , size_t cols);
void matrix_destroy(Matrix *m);



#endif
