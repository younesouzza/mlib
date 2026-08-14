#include "mlib/matrix.h"
#include <stdlib.h>
#include <stdint.h>
#include <assert.h>


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