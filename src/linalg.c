#include "mlib/linalg.h"
#include <stdbool.h>

bool matrix_vector_multiply(const Matrix *m, const Vector *v, Vector *out_result)
{
    if (m->cols != v->length)
    {
        out_result->data = NULL;
        out_result->length = 0;
        return false;
    }
    *out_result = vector_create(m->rows);

    if (out_result->data == NULL && m->rows > 0) {
        return false;
    }
    for (size_t i = 0; i < m->rows; i++)
    {
        float sum = 0.0f;
        for (size_t k = 0; k < m->cols; k++)
        {
            sum += matrix_get(m, i, k) * vector_get(v, k);
        }
        vector_set(out_result, i, sum);
    }

    return true;
}