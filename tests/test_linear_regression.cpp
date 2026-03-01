#include <vector>
#include <cmath>
#include "GTestLite.h"
#include "LinearRegression.h"

static constexpr double EPS = 1e-9;

template<typename T>
bool almost_equal(T a, T b, T eps = EPS)
{
    return std::abs(a - b) < eps;
}

/*
 * Test 1:
 * y = 2x
 * bez interceptu
 */
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

    EXPECT_TRUE(ok);
    EXPECT_TRUE(almost_equal(coeffs[0], 2.0));
}

/*
 * Test 2:
 * y = 3 + 2x
 * s interceptom (X[0] == nullptr)
 */
TEST(LinearRegressionStatic, WithIntercept)
{
    constexpr int DIM = 2;   // intercept + 1 feature
    constexpr int SIZE = 4;

    double x1[SIZE] = {1, 2, 3, 4};

    const double* X[DIM];
    X[0] = nullptr;   // intercept
    X[1] = x1;

    double y[SIZE] = {5, 7, 9, 11};  // 3 + 2x

    double coeffs[DIM];
    double buffer[4 * DIM * DIM];

    bool ok = linear_regression<double, DIM>(
        X, y, coeffs, SIZE, buffer
    );

    EXPECT_TRUE(ok);
    EXPECT_TRUE(almost_equal(coeffs[0], 3.0)); // intercept
    EXPECT_TRUE(almost_equal(coeffs[1], 2.0)); // slope
}

/*
 * Test 3:
 * Runtime DIM verzia
 */
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

    EXPECT_TRUE(ok);
    EXPECT_TRUE(almost_equal(coeffs[0], 3.0));
    EXPECT_TRUE(almost_equal(coeffs[1], 2.0));
}

/*
 * Test 4:
 * Singulárna matica → false
 * x konštantné → X^T X nie je invertibilná
 */
TEST(LinearRegressionStatic, SingularMatrix)
{
    constexpr int DIM = 1;
    constexpr int SIZE = 3;

    double x0[SIZE] = {1, 1, 1};  // lineárne závislé
    const double* X[DIM] = {x0};
    double y[SIZE] = {2, 2, 2};

    double coeffs[DIM];
    double buffer[4 * DIM * DIM];

    bool ok = linear_regression<double, DIM>(
        X, y, coeffs, SIZE, buffer
    );

    EXPECT_FALSE(ok);
}
