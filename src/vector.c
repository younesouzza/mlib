#include "mlib/vector.h"
#include <stdlib.h>
#include <stdbool.h>
#include <assert.h>

Vector vector_create(size_t length)
{
    Vector v;
    v.length = length;

    if (length == 0) {
        v.data = NULL;
    } else {
        v.data = malloc(length * sizeof(float));
    }

    return v;
}

void vector_destroy(Vector *v){
    free(v->data);
    v->data = NULL ; 
    v->length = 0;


}
float vector_get(const Vector *v, size_t index) {
    assert(v->data != NULL && index < v->length);
    return v->data[index];
}

void vector_set(Vector *v, size_t index, float value) {
    assert(v->data != NULL && index < v->length);
    v->data[index] = value;
}

bool vector_dot(const Vector *a, const Vector *b, float *out_result){
    if (a->length == b->length)
    {
        *out_result = 0.0f;

        for (size_t i = 0; i < a->length; i++)
        {
            *out_result += vector_get(a, i) * vector_get(b, i);
        }

        return true;
    }
    else {
        return false;
    }
}

Vector vector_add(const Vector *a, const Vector *b)
{
    if (a->length == b->length)
    {
        size_t result_length = a->length;
        Vector result = vector_create(result_length);
        
        for (size_t i = 0; i < result_length; i++)
        {
            vector_set(&result, i, vector_get(a, i) + vector_get(b, i));
        }
        
        return result;
    }
    else{
        return vector_create(0);
    }
}

Vector vector_scale(const Vector *a, float scalar)
{
    Vector result = vector_create(a->length);

    if (result.data == NULL && a->length > 0) {
        return result;
    }

    for (size_t i = 0; i < a->length; i++)
    {
        vector_set(&result, i, vector_get(a, i) * scalar);
    }

    return result;
}     
    

