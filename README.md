# linreg
Pure c++ linear regression from scratch with no dependencies to any library.


```c++
template <typename T, int DIM>
bool linear_regression(const T **__restrict X, const T *__restrict y, T *__restrict coeffs, int size, T *__restrict buffer) noexcept;

template <typename T>
bool linear_regression(const T **__restrict X, const T *__restrict y, T *__restrict coeffs, int DIM, int size, T *__restrict buffer) noexcept;
```
