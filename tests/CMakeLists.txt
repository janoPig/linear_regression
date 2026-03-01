# ---- GTestLite ----
add_library(GTestLite INTERFACE)
target_include_directories(GTestLite INTERFACE
    ${CMAKE_SOURCE_DIR}/GTestLite/include
)

# ---- tests ----
add_executable(tests
    tests/test_linear_regression.cpp
    test_linear_regression_extended.cpp
    tests/main.cpp
)

target_include_directories(tests PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(tests PRIVATE GTestLite)