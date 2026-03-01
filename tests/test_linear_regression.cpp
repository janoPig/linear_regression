#include <vector>
#include <cmath>
#include "GTestLite.h"
#include "LinearRegression.h"

TEST(LinearRegressionStatic, SimpleNoIntercept)
{
    constexpr int DIM = 1;
    constexpr int SIZE = 4;

    double x0[SIZE] = {1, 2, 3, 4};
    const double* X[DIM] = {x0};
    double y[SIZE] = {2, 4, 6, 8};

    double coeffs[DIM];
    double buffer[4 * DIM * DIM];

    bool ok = linear_regression<double, DIM>(
        X, y, coeffs, SIZE, buffer
    );

    ASSERT_TRUE(ok);
    EXPECT_NEAR(coeffs[0], 2.0, 1e-5);
}

TEST(LinearRegressionStatic, WithIntercept)
{
    constexpr int DIM = 2;
    constexpr int SIZE = 4;

    double x1[SIZE] = {1, 2, 3, 4};

    const double* X[DIM];
    X[0] = nullptr;
    X[1] = x1;

    double y[SIZE] = {5, 7, 9, 11};  // 3 + 2x

    double coeffs[DIM];
    double buffer[4 * DIM * DIM];

    bool ok = linear_regression<double, DIM>(
        X, y, coeffs, SIZE, buffer
    );

    ASSERT_TRUE(ok);
    EXPECT_NEAR(coeffs[0], 3.0, 1e-5);
    EXPECT_NEAR(coeffs[1], 2.0, 1e-5);
}

TEST(LinearRegressionDynamic, WithIntercept)
{
    const int DIM = 2;
    const int SIZE = 4;

    double x1[SIZE] = {1, 2, 3, 4};

    const double* X[DIM];
    X[0] = nullptr;
    X[1] = x1;

    double y[SIZE] = {5, 7, 9, 11};

    double coeffs[DIM];
    std::vector<double> buffer(4 * DIM * DIM);

    bool ok = linear_regression<double>(
        X, y, coeffs, DIM, SIZE, buffer.data()
    );

    ASSERT_TRUE(ok);
    EXPECT_NEAR(coeffs[0], 3.0, 1e-5);
    EXPECT_NEAR(coeffs[1], 2.0, 1e-5);
}

TEST(LinearRegressionStatic, SingularMatrix)
{
    constexpr int DIM = 2;
    constexpr int SIZE = 3;

    double x1[SIZE] = {1, 1, 1};  // All same - linearly dependent with intercept
    const double* X[DIM];
    X[0] = nullptr;
    X[1] = x1;
    
    double y[SIZE] = {2, 2, 2};

    double coeffs[DIM];
    double buffer[4 * DIM * DIM];

    bool ok = linear_regression<double, DIM>(
        X, y, coeffs, SIZE, buffer
    );

    EXPECT_FALSE(ok);
}
