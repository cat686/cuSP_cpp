#include "bsplines/bsplines.h"

#include "filtering/filtering.h"

#include <cmath>
#include <numeric>

namespace cuSP::bsplines {

namespace {

double cubic_bspline_value(double x) {
    const double ax = std::abs(x);
    if (ax < 1.0) {
        return 2.0 / 3.0 - 0.5 * ax * ax * (2.0 - ax);
    }
    if (ax < 2.0) {
        const double t = 2.0 - ax;
        return (1.0 / 6.0) * t * t * t;
    }
    return 0.0;
}

std::vector<double> convolve_same_real(const std::vector<double>& x, const std::vector<double>& kernel) {
    if (x.empty() || kernel.empty()) {
        return {};
    }

    const int n = static_cast<int>(x.size());
    const int m = static_cast<int>(kernel.size());
    const int start = (m - 1) / 2;

    std::vector<double> y(static_cast<std::size_t>(n), 0.0);
    for (int i = 0; i < n; ++i) {
        double acc = 0.0;
        for (int k = 0; k < m; ++k) {
            const int idx = i - start + k;
            if (idx >= 0 && idx < n) {
                acc += x[static_cast<std::size_t>(idx)] * kernel[static_cast<std::size_t>(k)];
            }
        }
        y[static_cast<std::size_t>(i)] = acc;
    }
    return y;
}

}  // namespace

std::vector<double> smooth_with_cubic_bspline(const std::vector<double>& x, int spline_kernel_size) {
    const int kernel_size = cuSP::filtering::ensure_odd(spline_kernel_size);
    std::vector<double> kernel(static_cast<std::size_t>(kernel_size), 0.0);
    if (kernel_size == 1) {
        kernel[0] = 1.0;
        return x;
    }

    for (int i = 0; i < kernel_size; ++i) {
        const double pos = -2.0 + 4.0 * static_cast<double>(i) / static_cast<double>(kernel_size - 1);
        kernel[static_cast<std::size_t>(i)] = cubic_bspline_value(pos);
    }

    const double sum = std::accumulate(kernel.begin(), kernel.end(), 0.0);
    const double norm = (std::abs(sum) < 1e-12) ? 1.0 : sum;
    for (double& v : kernel) {
        v /= norm;
    }
    return convolve_same_real(x, kernel);
}

}  // namespace cuSP::bsplines
