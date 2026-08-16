#ifndef MLIB_MATRIX_H
#define MLIB_MATRIX_H

#include <stddef.h>
#include <stdbool.h>
typedef struct 
{
    float *data;
    size_t rows;
    size_t cols;
}Matrix;

Matrix matrix_create(size_t rows , size_t cols);
void matrix_destroy(Matrix *m);
float matrix_get(const Matrix *m, size_t row, size_t col);
void matrix_set(Matrix *m, size_t row, size_t col, float value);
Matrix matrix_add(const Matrix *m1, const Matrix *m2);
Matrix matrix_scale(const Matrix *m , float scalar);
bool matrix_multiply(const Matrix *a, const Matrix *b, Matrix *out_result);

#endif
