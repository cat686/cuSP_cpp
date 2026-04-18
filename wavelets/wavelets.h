#pragma once

#include "../common/common.h"

#include <vector>

namespace cuSP::wavelets {

void extract_time_frequency_domain(const std::vector<double>& x_smooth,
                                   double fs,
                                   double cwt_w,
                                   int cwt_width_min,
                                   int cwt_width_max,
                                   std::vector<int>& widths,
                                   std::vector<double>& pseudo_freqs,
                                   cuSP::common::Matrix<float>& cwt_power,
                                   std::vector<double>& ridge_freq,
                                   std::vector<double>& ridge_energy);

}  // namespace cuSP::wavelets
