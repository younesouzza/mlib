#ifndef MLIB_VECTOR_H
#define MLIB_VECTOR_H

#include <stddef.h>
#include <stdbool.h>

typedef struct {
    float *data;
    size_t length;
} Vector;

Vector vector_create(size_t length);
void vector_destroy(Vector *v);
bool vector_dot(const Vector *a, const Vector *b, float *out_result);
Vector vector_add(const Vector *a, const Vector *b);
Vector vector_scale(const Vector *a, float scalar);

#endif