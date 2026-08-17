#include "mlib/linalg.h"
#include <stdbool.h>
#include <stdlib.h>

bool matrix_vector_multiply(const Matrix *m, const Vector *v, Vector *out_result)
{
    if (m->cols != v->length)
    {
        return false;
    }

    *out_result = vector_create(m->rows);

    for (size_t i = 0; i < m->rows; i++)
    {
        float sum = 0.0f;
        for (size_t k = 0; k < m->cols; k++)
        {
            sum += matrix_get(m, i, k) * v->data[k];
        }
        out_result->data[i] = sum;
    }

    return true;
}