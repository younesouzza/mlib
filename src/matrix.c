#include "mlib/matrix.h"
#include <stdlib.h>
#include <stdint.h>
#include <assert.h>
#include <stdbool.h>

Matrix matrix_create(size_t rows , size_t cols){
    Matrix m;
    m.data = NULL;
    m.rows = rows;
    m.cols= cols;

    if (cols!=0 && rows > SIZE_MAX/cols)
    {
        m.data = NULL;
        m.cols = 0;
        m.rows = 0;
        return m ;
    }
    
    if (rows == 0 || cols == 0)
    {
        m.data = NULL;
    }
    else{
        m.data = malloc(rows*cols*sizeof(float));
        
    }
    return m ;
}

void matrix_destroy(Matrix *m){
    free(m->data);
    m->data = NULL;
    m->cols = 0;
    m->rows = 0;

}

float matrix_get(const Matrix *m, size_t row, size_t col){
    assert(row < m->rows && col < m->cols);
    return m->data[row * m->cols + col];

}

void matrix_set(Matrix *m, size_t row, size_t col, float value){
    assert(row < m->rows && col < m->cols);
    m->data[row * m->cols + col] = value;
}

Matrix matrix_add(const Matrix *m1 ,const Matrix *m2){
    if(m1->rows == m2->rows && m1->cols == m2->cols){
        size_t Rrows = m1->rows;
        size_t Rcols = m1->cols;
        Matrix result = matrix_create(Rrows , Rcols);

        for (size_t i = 0; i < m1->rows; i++)
        {
            for (size_t j = 0; j < m1->cols; j++)
            {
                float value;
                value = matrix_get(m1, i, j) + matrix_get(m2 , i, j);
                matrix_set(&result , i, j , value);
            }
            
            
        }
        return result;
        
    }
    else{
        return matrix_create(0,0);
    }

}

Matrix matrix_scale(const Matrix *m , float scalar){
    Matrix result = matrix_create(m->rows , m->cols);
    if (result.data == NULL && m->rows > 0 && m->cols > 0)
    {
        return result;
    }
    for (size_t i = 0; i < m->rows; i++)
    {
        for (size_t j = 0; j < m->cols; j++)
        {
            float value;
            value = matrix_get(m ,i ,j)*scalar;
            matrix_set(&result , i , j , value);
        }
        
    }
    return result;
    

}
bool matrix_multiply(const Matrix *a, const Matrix *b, Matrix *out_result)
{
    if (a->cols != b->rows)
    {
        return false;
    }
    else
    {
        *out_result = matrix_create(a->rows, b->cols);

        for (size_t i = 0; i < a->rows; i++)
        {
            for (size_t j = 0; j < b->cols; j++)
            {
                float sum = 0.0f;
                for (size_t k = 0; k < a->cols; k++)
                {
                    sum += matrix_get(a, i, k) * matrix_get(b, k, j);
                }
                matrix_set(out_result, i, j, sum);
            }
        }
        return true;
    }
}

Matrix matrix_transpose(const Matrix *m)
{
    if (m == NULL) {
        return (Matrix){0}; 
    }
    Matrix t = matrix_create(m->cols, m->rows);
    if (t.data == NULL && (m->rows > 0 && m->cols > 0)) {
        return t; 
    }

    for (size_t i = 0; i < t.rows; i++)
    {
        for (size_t j = 0; j < t.cols; j++) 
        {
            float data = matrix_get(m, j, i);  
            matrix_set(&t, i, j, data); 
        }
    }
    
    return t;
}