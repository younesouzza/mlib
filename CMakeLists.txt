cmake_minimum_required(VERSION 3.20)
project(mlib LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

set(MLIB_WARNING_FLAGS -Wall -Wextra -Wpedantic -Wconversion -Wshadow)

add_library(mlib
    src/vector.c
)

target_include_directories(mlib PUBLIC include)

target_compile_options(mlib PRIVATE ${MLIB_WARNING_FLAGS})

enable_testing()
add_subdirectory(tests)