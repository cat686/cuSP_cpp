#pragma once

#include "common/common.h"

#include <vector>

namespace cuSP::common {

std::vector<Complex> fft_complex(const std::vector<Complex>& x, int n = -1);
std::vector<Complex> ifft_complex(const std::vector<Complex>& x, int n = -1);
std::vector<double> absolute_value(const std::vector<Complex>& x);
double safe_max(const std::vector<double>& x);
int argmax_index(const std::vector<double>& x);
std::vector<int> argsort_descending(const std::vector<double>& x);
std::vector<double> unwrap_phase(const std::vector<double>& phase);
std::vector<double> diff_vector(const std::vector<double>& x);
std::vector<double> abs_vector(const std::vector<double>& x);
std::vector<double> reverse_copy(const std::vector<double>& x);
std::vector<double> fft_convolve_real_full(const std::vector<double>& a, const std::vector<double>& b);
std::vector<Complex> hilbert_transform(const std::vector<double>& x);

}  // namespace cuSP::common
