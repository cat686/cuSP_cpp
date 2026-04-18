#pragma once

#include <vector>

namespace cuSP::peak_finding {

std::vector<int> detect_peaks_1d(const std::vector<double>& x, int order, double threshold_ratio = 0.0);
std::vector<int> fallback_global_peak(const std::vector<double>& x);

}  // namespace cuSP::peak_finding
