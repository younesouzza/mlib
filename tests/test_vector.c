#include <stdio.h>
#include "mlib/vector.h"
#include <assert.h>
#include <stdbool.h>

int main(void)
{
    // --- Test: Creation and basic access ---
    Vector v = vector_create(5);
    assert(v.data != NULL); // Testing internal allocation state is fine here

    // Use vector_set to populate
    vector_set(&v, 0, 1.0f);
    vector_set(&v, 1, 2.0f);
    vector_set(&v, 2, 3.0f);
    vector_set(&v, 3, 4.0f);
    vector_set(&v, 4, 5.0f);

    assert(v.length == 5);
    
    // Use vector_get to verify
    assert(vector_get(&v, 0) == 1.0f);
    assert(vector_get(&v, 1) == 2.0f);
    assert(vector_get(&v, 2) == 3.0f);
    assert(vector_get(&v, 3) == 4.0f);
    assert(vector_get(&v, 4) == 5.0f);

    // --- Test: Zero-length vector ---
    Vector v1 = vector_create(0);
    assert(v1.data == NULL);
    assert(v1.length == 0);

    // --- Test: Idempotent destroy ---
    vector_destroy(&v);
    assert(v.data == NULL);
    assert(v.length == 0);

    vector_destroy(&v); // Double destroy check
    assert(v.data == NULL);
    assert(v.length == 0);

    vector_destroy(&v1);
    assert(v1.data == NULL);
    assert(v1.length == 0);

    // --- Test: vector_dot ---
    Vector v2 = vector_create(2);
    vector_set(&v2, 0, 2.0f);
    vector_set(&v2, 1, 5.0f);

    Vector v3 = vector_create(2);
    vector_set(&v3, 0, 3.0f);
    vector_set(&v3, 1, 4.0f);

    float result;
    bool ok = vector_dot(&v2, &v3, &result);
    assert(ok == true);
    assert(result == 26.0f);

    vector_destroy(&v2);
    vector_destroy(&v3);

    // --- Test: vector_dot mismatch ---
    Vector v4 = vector_create(2);
    Vector v5 = vector_create(3);

    bool ok_mismatch = vector_dot(&v4, &v5, &result);
    assert(ok_mismatch == false);

    vector_destroy(&v4);
    vector_destroy(&v5);

    // --- Test: vector_dot empty vectors ---
    Vector v6 = vector_create(0);
    Vector v7 = vector_create(0);

    float zero_result;
    bool ok_zero = vector_dot(&v6, &v7, &zero_result);
    assert(ok_zero == true);
    assert(zero_result == 0.0f);

    vector_destroy(&v6);
    vector_destroy(&v7);

    // --- Test: vector_add normal case ---
    Vector v8 = vector_create(3);
    vector_set(&v8, 0, 1.0f);
    vector_set(&v8, 1, 2.0f);
    vector_set(&v8, 2, 3.0f);

    Vector v9 = vector_create(3);
    vector_set(&v9, 0, 10.0f);
    vector_set(&v9, 1, 20.0f);
    vector_set(&v9, 2, 30.0f);

    Vector sum = vector_add(&v8, &v9);
    assert(sum.data != NULL);
    assert(sum.length == 3);
    assert(vector_get(&sum, 0) == 11.0f);
    assert(vector_get(&sum, 1) == 22.0f);
    assert(vector_get(&sum, 2) == 33.0f);

    vector_destroy(&v8);
    vector_destroy(&v9);
    vector_destroy(&sum);

    // --- Test: vector_add mismatch ---
    Vector v10 = vector_create(2);
    Vector v11 = vector_create(4);

    Vector bad_sum = vector_add(&v10, &v11);
    assert(bad_sum.data == NULL);
    assert(bad_sum.length == 0);

    vector_destroy(&v10);
    vector_destroy(&v11);
    vector_destroy(&bad_sum);

    // --- Test: vector_scale normal case ---
    Vector v12 = vector_create(3);
    vector_set(&v12, 0, 1.0f);
    vector_set(&v12, 1, 2.0f);
    vector_set(&v12, 2, 3.0f);

    Vector scaled = vector_scale(&v12, 2.0f);
    assert(scaled.data != NULL);
    assert(scaled.length == 3);
    assert(vector_get(&scaled, 0) == 2.0f);
    assert(vector_get(&scaled, 1) == 4.0f);
    assert(vector_get(&scaled, 2) == 6.0f);

    vector_destroy(&v12);
    vector_destroy(&scaled);

    return 0;
}