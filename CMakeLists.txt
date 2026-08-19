cmake_minimum_required(VERSION 3.20)
project(mlib LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

set(MLIB_WARNING_FLAGS -Wall -Wextra -Wpedantic -Wconversion -Wshadow)

add_library(mlib
    src/vector.c
    src/matrix.c
    src/linalg.c
    src/tensor.c


)

target_include_directories(mlib PUBLIC include)

target_compile_options(mlib PRIVATE ${MLIB_WARNING_FLAGS})

enable_testing()
add_subdirectory(tests)

option(MLIB_ENABLE_SANITIZERS "Build with AddressSanitizer and UndefinedBehaviorSanitizer" OFF)

if(MLIB_ENABLE_SANITIZERS)
    target_compile_options(mlib PUBLIC -fsanitize=address,undefined -fno-omit-frame-pointer)
    target_link_options(mlib PUBLIC -fsanitize=address,undefined)
endif()