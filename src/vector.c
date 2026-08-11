#include "mlib/vector.h"
#include <stdlib.h>


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