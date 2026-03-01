#include <vector>
#include <cmath>
#include "GTestLite.h"
#include "LinearRegression.h"

TEST(LinearRegressionStatic, MultiFeatureDIM3)
{
    constexpr int DIM = 3;   // intercept + 2 features
    constexpr int SIZE = 6;

    double x1[SIZE] = {1,2,3,4,5,6};
    double x2[SIZE] = {2,1,0,-1,-2,-3};

    const double* X[DIM];
    X[0] = nullptr;  // intercept
    X[1] = x1;
    X[2] = x2;

    double y[SIZE];

    for (int i = 0; i < SIZE; ++i)
        y[i] = 1.0 + 2.0*x1[i] - 3.0*x2[i];

    double coeffs[DIM];
    double buffer[4 * DIM * DIM];

    bool ok = linear_regression<double, DIM>(
        X, y, coeffs, SIZE, buffer
    );

    EXPECT_TRUE(ok);
    EXPECT_TRUE(almost_equal(coeffs[0], 1.0));
    EXPECT_TRUE(almost_equal(coeffs[1], 2.0));
    EXPECT_TRUE(almost_equal(coeffs[2], -3.0));
}

TEST(LinearRegressionStatic, NumericalStability)
{
    constexpr int DIM = 2;
    constexpr int SIZE = 100;

    double x1[SIZE];
    const double* X[DIM];

    X[0] = nullptr;
    X[1] = x1;

    double y[SIZE];

    for (int i = 0; i < SIZE; ++i)
    {
        double v = 1e8 + i;
        x1[i] = v;
        y[i] = 5.0 + 3.0*v;
    }

    double coeffs[DIM];
    double buffer[4 * DIM * DIM];

    bool ok = linear_regression<double, DIM>(
        X, y, coeffs, SIZE, buffer
    );

    EXPECT_TRUE(ok);

    EXPECT_TRUE(std::abs(coeffs[0] - 5.0) < 1e-3);
    EXPECT_TRUE(std::abs(coeffs[1] - 3.0) < 1e-6);
}

TEST(LinearRegressionStatic, LargeDataset)
{
    constexpr int DIM = 2;
    constexpr int SIZE = 100000;

    std::vector<double> x1(SIZE);
    std::vector<double> y(SIZE);

    const double* X[DIM];
    X[0] = nullptr;
    X[1] = x1.data();

    for (int i = 0; i < SIZE; ++i)
    {
        x1[i] = i * 0.1;
        y[i] = 7.0 - 4.0*x1[i];
    }

    double coeffs[DIM];
    std::vector<double> buffer(4 * DIM * DIM);

    bool ok = linear_regression<double, DIM>(
        X, y.data(), coeffs, SIZE, buffer.data()
    );

    EXPECT_TRUE(ok);
    EXPECT_TRUE(almost_equal(coeffs[0], 7.0, 1e-9));
    EXPECT_TRUE(almost_equal(coeffs[1], -4.0, 1e-9));
}
