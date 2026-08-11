#include <stdio.h>
#include "mlib/vector.h"
#include <assert.h>

int main(void)
{
    Vector v = vector_create(5);
    assert(v.data != NULL);

    v.data[0] = 1.0f;
    v.data[1] = 2.0f;
    v.data[2] = 3.0f;
    v.data[3] = 4.0f;
    v.data[4] = 5.0f;

    assert(v.length == 5);
    assert(v.data[0] == 1.0f);
    assert(v.data[1] == 2.0f);
    assert(v.data[2] == 3.0f);
    assert(v.data[3] == 4.0f);
    assert(v.data[4] == 5.0f);

    Vector v1 = vector_create(0);
    assert(v1.data == NULL);
    assert(v1.length == 0);

    vector_destroy(&v);
    assert(v.data == NULL);
    assert(v.length == 0);

    vector_destroy(&v);
    assert(v.data == NULL);
    assert(v.length == 0);

    vector_destroy(&v1);
    assert(v1.data == NULL);
    assert(v1.length == 0);

    return 0;
}