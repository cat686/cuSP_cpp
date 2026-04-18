#pragma once

#include <cstdint>
#include <utility>
#include <vector>

namespace cuSP::estimation {

std::pair<double, double> linear_fit(const std::vector<double>& x, const std::vector<double>& y);
std::pair<int, int> longest_true_segment(const std::vector<std::uint8_t>& mask);
void estimate_activity_interval(const std::vector<double>& x_smooth,
                                int spline_kernel_size,
                                double activity_threshold_ratio,
                                double activity_min_duration_s,
                                double fs,
                                std::vector<double>& envelope_smooth,
                                std::vector<std::uint8_t>& activity_mask,
                                int& start_idx,
                                int& end_idx);
std::vector<double> kalman_smooth_1d(const std::vector<double>& measurements,
                                     double dt,
                                     double process_var,
                                     double meas_var);

}  // namespace cuSP::estimation
