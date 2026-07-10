#pragma once

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cctype>
#include <stdexcept>
#include <string>
#include <vector>

namespace cuSP::common {

constexpr double kLightSpeed = 3.0e8;
constexpr double kPi = 3.141592653589793238462643383279502884;

using Complex = std::complex<double>;

template <typename T>
struct Matrix {
    std::size_t rows = 0;
    std::size_t cols = 0;
    std::vector<T> data;

    Matrix() = default;

    Matrix(std::size_t r, std::size_t c)
        : rows(r), cols(c), data(r * c) {}

    Matrix(std::size_t r, std::size_t c, const T& value)
        : rows(r), cols(c), data(r * c, value) {}

    void resize(std::size_t r, std::size_t c) {
        rows = r;
        cols = c;
        data.assign(r * c, T{});
    }

    T& operator()(std::size_t r, std::size_t c) {
        return data[r * cols + c];
    }

    const T& operator()(std::size_t r, std::size_t c) const {
        return data[r * cols + c];
    }
};

double db20(double x, double eps = 1e-12);
double db10(double x, double eps = 1e-12);
std::size_t next_pow2(std::size_t n);
std::vector<double> make_window(const std::string& name, std::size_t n);
void fft_inplace(std::vector<Complex>& a, bool inverse);

}  // namespace cuSP::common
