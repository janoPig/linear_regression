#pragma once

#include "MatrixInverse.h"

template <typename T, int DIM>
void xt_x(const T** __restrict X, T* __restrict XTX, int size) noexcept
{
    for (auto i = 0; i < DIM*DIM; i++)
        XTX[i] = static_cast<T>(0.0);

    auto start = 0;

    // if compute intercept X[0][k] = 1
    if (!X[0])
    {
        // first row is [size, sum(X[1]) ... sum(X[N])]
        start = 1;
        XTX[0] = static_cast<T>(size);
        for (auto i = 1; i < DIM; i++)
        {
            for (auto k = 0; k < size; k++)
            {
                XTX[i] += X[i][k];
            }
        }
    }

    // compute only right top triangle
    for (auto i = start; i < DIM; i++)
    {
        for (auto j = i; j < DIM; j++)
        {
            for (auto k = 0; k < size; k++)
            {
                XTX[i * DIM + j] += X[i][k] * X[j][k];
            }
        }
    }

    // left down triangle same as rigth top
    for (auto i = 1; i < DIM; i++)
    {
        for (auto j = 0; j < i; j++)
        {
            XTX[i * DIM + j] = XTX[j * DIM + i];
        }
    }
}

template <typename T>
void xt_x(const T **__restrict X, T *__restrict XTX, int DIM, int size) noexcept
{
    for (auto i = 0; i < DIM * DIM; i++)
        XTX[i] = static_cast<T>(0.0);

    auto start = 0;

    // if compute intercept X[0][k] = 1
    if (!X[0])
    {
        // first row is [size, sum(X[1]) ... sum(X[N])]
        start = 1;
        XTX[0] = static_cast<T>(size);
        for (auto i = 1; i < DIM; i++)
        {
            for (auto k = 0; k < size; k++)
            {
                XTX[i] += X[i][k];
            }
        }
    }

    // compute only right top triangle
    for (auto i = start; i < DIM; i++)
    {
        for (auto j = i; j < DIM; j++)
        {
            for (auto k = 0; k < size; k++)
            {
                XTX[i * DIM + j] += X[i][k] * X[j][k];
            }
        }
    }

    // left down triangle same as rigth top
    for (auto i = 1; i < DIM; i++)
    {
        for (auto j = 0; j < i; j++)
        {
            XTX[i * DIM + j] = XTX[j * DIM + i];
        }
    }
}

template <typename T, int DIM>
void xt_y(const T** __restrict X, const T* __restrict y, T* __restrict XTY, int size) noexcept
{
    for (auto i = 0; i < DIM; i++)
        XTY[i] = static_cast<T>(0.0);

    auto start = 0;
    if (!X[0])
    {
        start = 1;
        for (auto k = 0; k < size; k++)
        {
            XTY[0] += y[k];
        }
    }
    for (auto i = start; i < DIM; i++)
    {
        for (auto k = 0; k < size; k++)
        {
            XTY[i] += X[i][k] * y[k];
        }
    }
}

template <typename T>
void xt_y(const T **__restrict X, const T *__restrict y, T *__restrict XTY, int DIM, int size) noexcept
{
    for (auto i = 0; i < DIM; i++)
        XTY[i] = static_cast<T>(0.0);

    auto start = 0;
    if (!X[0])
    {
        start = 1;
        for (auto k = 0; k < size; k++)
        {
            XTY[0] += y[k];
        }
    }
    for (auto i = start; i < DIM; i++)
    {
        for (auto k = 0; k < size; k++)
        {
            XTY[i] += X[i][k] * y[k];
        }
    }
}

template <typename T, int DIM>
void compute_coefficients(const T *__restrict XTX_inv, const T *__restrict XTY, T *__restrict coeffs) noexcept
{
    for (auto i = 0; i < DIM; i++)
    {
        coeffs[i] = static_cast<T>(0.0);
        for (auto j = 0; j < DIM; j++)
        {
            coeffs[i] += XTX_inv[i * DIM + j] * XTY[j];
        }
    }
}

template <typename T>
void compute_coefficients(const T *__restrict XTX_inv, const T *__restrict XTY, T *coeffs, int DIM) noexcept
{
    for (auto i = 0; i < DIM; i++)
    {
        coeffs[i] = static_cast<T>(0.0);
        for (auto j = 0; j < DIM; j++)
        {
            coeffs[i] += XTX_inv[i * DIM + j] * XTY[j];
        }
    }
}

// buffer size must be >= 4*DIM*DIM
// to compute regression with intercept simply pass nullptr in X[0]
template <typename T, int DIM>
bool linear_regression(const T **__restrict X, const T *__restrict y, T *__restrict coeffs, int size, T *__restrict buffer) noexcept
{
    auto XTX = &buffer[0];
    xt_x<T, DIM>(X, XTX, size);
    auto XTX_inv = &buffer[DIM * DIM];
    if (!matrix_inverse<T, DIM>(XTX, XTX_inv, &buffer[2 * DIM * DIM]))
        return false;
    auto XTY = &buffer[2 * DIM * DIM];
    xt_y<T, DIM>(X, y, XTY, size);
    compute_coefficients<T, DIM>((const T *)XTX_inv, XTY, coeffs);
    return true;
}

// buffer size must be >= 4*DIM*DIM
// to compute regression with intercept simply pass nullptr in X[0]
template <typename T>
bool linear_regression(const T **__restrict X, const T *__restrict y, T *__restrict coeffs, int DIM, int size, T *__restrict buffer) noexcept
{
    auto XTX = &buffer[0];
    xt_x<T>(X, XTX, DIM, size);
    auto XTX_inv = &buffer[DIM * DIM];
    if (!matrix_inverse<T>(XTX, XTX_inv, DIM, &buffer[2 * DIM * DIM]))
        return false;
    auto XTY = &buffer[2 * DIM * DIM];
    xt_y<T>(X, y, XTY, DIM, size);
    compute_coefficients<T>((const T *)XTX_inv, XTY, coeffs, DIM);
    return true;
}
