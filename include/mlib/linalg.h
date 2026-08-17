#ifndef MLIB_LINALG_H
#define MLIB_LINALG_H

#include "mlib/vector.h"
#include "mlib/matrix.h"
#include <stdbool.h>

bool matrix_vector_multiply(const Matrix *m, const Vector *v, Vector *out_result);

#endif