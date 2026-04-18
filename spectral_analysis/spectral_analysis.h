#pragma once

#include <vector>

namespace cuSP::spectral_analysis {

void extract_frequency_domain(const std::vector<double>& x_smooth,
                              double fs,
                              int nperseg,
                              int noverlap,
                              std::vector<double>& freqs_pos,
                              std::vector<double>& psd_pos);

}  // namespace cuSP::spectral_analysis
