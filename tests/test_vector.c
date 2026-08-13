#include <stdio.h>
#include "mlib/vector.h"
#include <assert.h>
#include <stdbool.h>

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

    Vector v2 = vector_create(2);
    v2.data[0] = 2.0f;
    v2.data[1] = 5.0f;

    Vector v3 = vector_create(2);
    v3.data[0] = 3.0f;
    v3.data[1] = 4.0f;

    float result;
    bool ok = vector_dot(&v2, &v3, &result);
    assert(ok == true);
    assert(result == 26.0f);

    vector_destroy(&v2);
    vector_destroy(&v3);

    Vector v4 = vector_create(2);
    Vector v5 = vector_create(3);

    bool ok_mismatch = vector_dot(&v4, &v5, &result);
    assert(ok_mismatch == false);

    vector_destroy(&v4);
    vector_destroy(&v5);

    Vector v6 = vector_create(0);
    Vector v7 = vector_create(0);

    float zero_result;
    bool ok_zero = vector_dot(&v6, &v7, &zero_result);
    assert(ok_zero == true);
    assert(zero_result == 0.0f);

    vector_destroy(&v6);
    vector_destroy(&v7);

    /* vector_add: normal case */
    Vector v8 = vector_create(3);
    v8.data[0] = 1.0f;
    v8.data[1] = 2.0f;
    v8.data[2] = 3.0f;

    Vector v9 = vector_create(3);
    v9.data[0] = 10.0f;
    v9.data[1] = 20.0f;
    v9.data[2] = 30.0f;

    Vector sum = vector_add(&v8, &v9);
    assert(sum.data != NULL);
    assert(sum.length == 3);
    assert(sum.data[0] == 11.0f);
    assert(sum.data[1] == 22.0f);
    assert(sum.data[2] == 33.0f);

    vector_destroy(&v8);
    vector_destroy(&v9);
    vector_destroy(&sum);

    /* vector_add: mismatched lengths */
    Vector v10 = vector_create(2);
    Vector v11 = vector_create(4);

    Vector bad_sum = vector_add(&v10, &v11);
    assert(bad_sum.data == NULL);
    assert(bad_sum.length == 0);

    vector_destroy(&v10);
    vector_destroy(&v11);
    vector_destroy(&bad_sum);

    /* vector_scale: normal case */
    Vector v12 = vector_create(3);
    v12.data[0] = 1.0f;
    v12.data[1] = 2.0f;
    v12.data[2] = 3.0f;

    Vector scaled = vector_scale(&v12, 2.0f);
    assert(scaled.data != NULL);
    assert(scaled.length == 3);
    assert(scaled.data[0] == 2.0f);
    assert(scaled.data[1] == 4.0f);
    assert(scaled.data[2] == 6.0f);

    vector_destroy(&v12);
    vector_destroy(&scaled);

    return 0;
}