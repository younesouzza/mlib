#ifndef MLIB_VECTOR_H
#define MLIB_VECTOR_H

#include <stddef.h>

typedef struct {
    float *data;
    size_t length;
} Vector;

Vector vector_create(size_t length);
void vector_destroy(Vector *v);

#endif