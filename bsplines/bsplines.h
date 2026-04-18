#pragma once

#include <vector>

namespace cuSP::bsplines {

std::vector<double> smooth_with_cubic_bspline(const std::vector<double>& x, int spline_kernel_size);

}  // namespace cuSP::bsplines
