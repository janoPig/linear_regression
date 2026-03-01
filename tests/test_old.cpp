// test.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include "../LinearRegression.h"
#include <iostream>
#include <random>
#include <chrono>

using Type = double;
constexpr Type coeffs[] = { 1., -2., 3., -4., 5., -6., 7., -8., 9., -10., 11., -12., 13., -14., 15. };
constexpr Type intercept = -10.;
constexpr int dim = 15;
constexpr int size = 1'000'000;

static void test_intercept()
{
    Type *x[dim + 1];
    Type *y = new Type[size];

    x[0] = nullptr;
    for (int j = 0; j < dim; j++)
    {
        x[j + 1] = new Type[size];
    }

    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < dim; j++)
        {
            x[j + 1][i] = (Type)std::rand() / RAND_MAX;
        }
        y[i] = intercept;
        for (int j = 0; j < dim; j++)
        {
            y[i] += coeffs[j] * x[j + 1][i];
        }
    }

    Type buffer[4 * (dim + 1) * (dim + 1)];

    Type coeffs_1[dim + 1];
    //dynamic dimension
    auto start = std::chrono::high_resolution_clock::now();
    linear_regression<Type>((const Type **)x, y, coeffs_1, dim + 1, size, &buffer[0]);
    std::cout << "test_intercept computing time= " << std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start) << " s" << std::endl;
    for (const auto c : coeffs_1)
        std::cout << c << ",";
    std::cout << std::endl;

    Type coeffs_2[dim + 1];
    // compiler time know dimension
    start = std::chrono::high_resolution_clock::now();
    linear_regression<Type, dim + 1>((const Type **)x, y, coeffs_2, size, buffer);
    std::cout << "test_intercept(static) computing time= " << std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start) << " s" << std::endl;
    for (const auto c : coeffs_2)
        std::cout << c << ",";
    std::cout << std::endl;

    delete[] y;
    for (int j = 0; j < dim; j++)
    {
        delete[] x[j + 1];
    }
}

static void test_no_intercept()
{
    Type *x[dim];
    Type *y = new Type[size];

    for (int j = 0; j < dim; j++)
    {
        x[j] = new Type[size];
    }

    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < dim; j++)
        {
            x[j][i] = (Type)std::rand() / RAND_MAX;
        }
        y[i] = 0.0f;
        for (int j = 0; j < dim; j++)
        {
            y[i] += coeffs[j] * x[j][i];
        }
    }

    Type buffer[4 * dim * dim];

    Type coeffs_1[dim];
    //dynamic dimension
    auto start = std::chrono::high_resolution_clock::now();
    linear_regression<Type>((const Type **)x, y, coeffs_1, dim, size, &buffer[0]);
    std::cout << "test_no_intercept computing time= " << std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start) << " s" << std::endl;
    for (const auto c : coeffs_1)
        std::cout << c << ",";
    std::cout << std::endl;

    Type coeffs_2[dim];
    // compiler time know dimension
    start = std::chrono::high_resolution_clock::now();
    linear_regression<Type, dim>((const Type **)x, y, coeffs_2, size, buffer);
    std::cout << "test_no_intercept(static) computing time= " << std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start) << " s" << std::endl;
    for (const auto c : coeffs_2)
        std::cout << c << ",";
    std::cout << std::endl;

    delete[] y;
    for (int j = 0; j < dim; j++)
    {
        delete[] x[j];
    }
}

int main()
{
    test_intercept();
    test_no_intercept();
}
