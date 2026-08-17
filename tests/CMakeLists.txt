add_executable(test_vector
    test_vector.c
)

target_link_libraries(test_vector PRIVATE mlib)

add_test(
    NAME test_vector
    COMMAND test_vector
)

add_executable(test_matrix
    test_matrix.c
)

target_link_libraries(test_matrix PRIVATE mlib)

add_test(
    NAME test_matrix
    COMMAND test_matrix
)
add_executable(test_linalg
    test_linalg.c
)

target_link_libraries(test_linalg PRIVATE mlib)

add_test(
    NAME test_linalg
    COMMAND test_linalg
)