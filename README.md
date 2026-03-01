# linreg

[![CI](https://github.com/janoPig/linear_regression/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/janoPig/linear_regression/actions/workflows/ci.yml)

Pure c++ linear regression from scratch with no dependencies to any library.


```c++
template <typename T, int DIM>
bool linear_regression(const T **__restrict X, const T *__restrict y, T *__restrict coeffs, int size, T *__restrict buffer) noexcept;

template <typename T>
bool linear_regression(const T **__restrict X, const T *__restrict y, T *__restrict coeffs, int DIM, int size, T *__restrict buffer) noexcept;
```
