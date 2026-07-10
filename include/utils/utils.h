#pragma once

#include "common/common.h"

#include <cstdint>
#include <vector>

namespace cuSP::utils {

using cuSP::common::Complex;

std::vector<double> build_time_axis(double fs, double duration);
std::vector<Complex> apply_integer_delay(const std::vector<Complex>& x, double fs, double delay_s);
std::vector<double> add_awgn(const std::vector<double>& x, double snr_db, std::uint32_t seed);

}  // namespace cuSP::utils
