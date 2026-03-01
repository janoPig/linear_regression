cmake_minimum_required(VERSION 3.10)
project(LinearRegressionTests CXX)

# enable testing
enable_testing()

# ---- GTestLite ----
add_library(GTestLite INTERFACE)
target_include_directories(GTestLite INTERFACE
    ${CMAKE_SOURCE_DIR}/../GTestLite/include
)

# ---- tests executable ----
add_executable(tests
    test_linear_regression.cpp
    test_linear_regression_extended.cpp
    main.cpp
)

# Set C++20 standard
set_target_properties(tests PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

# include your project headers
target_include_directories(tests PRIVATE
    ${CMAKE_SOURCE_DIR}/..
)

# link GTestLite
target_link_libraries(tests PRIVATE GTestLite)

# register with CTest
add_test(NAME linear_tests COMMAND tests)
