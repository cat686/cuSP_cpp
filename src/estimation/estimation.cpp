#include "estimation/estimation.h"

#include "bsplines/bsplines.h"
#include "common/fftw_helpers.h"

#include <algorithm>
#include <limits>
#include <numeric>

namespace cuSP::estimation {

std::pair<double, double> linear_fit(const std::vector<double>& x, const std::vector<double>& y) {
    if (x.size() != y.size() || x.empty()) {
        return {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN()};
    }

    const double x_mean = std::accumulate(x.begin(), x.end(), 0.0) / static_cast<double>(x.size());
    const double y_mean = std::accumulate(y.begin(), y.end(), 0.0) / static_cast<double>(y.size());

    double denom = 0.0;
    double numer = 0.0;
    for (std::size_t i = 0; i < x.size(); ++i) {
        const double dx = x[i] - x_mean;
        const double dy = y[i] - y_mean;
        denom += dx * dx;
        numer += dx * dy;
    }
    denom += 1e-12;
    const double slope = numer / denom;
    const double intercept = y_mean - slope * x_mean;
    return {slope, intercept};
}

std::pair<int, int> longest_true_segment(const std::vector<std::uint8_t>& mask) {
    int best_start = 0;
    int best_end = 0;
    int cur_start = -1;

    for (std::size_t idx = 0; idx < mask.size(); ++idx) {
        if (mask[idx] != 0 && cur_start < 0) {
            cur_start = static_cast<int>(idx);
        }
        if (mask[idx] == 0 && cur_start >= 0) {
            if ((static_cast<int>(idx) - cur_start) > (best_end - best_start)) {
                best_start = cur_start;
                best_end = static_cast<int>(idx);
            }
            cur_start = -1;
        }
    }

    if (cur_start >= 0 && (static_cast<int>(mask.size()) - cur_start) > (best_end - best_start)) {
        best_start = cur_start;
        best_end = static_cast<int>(mask.size());
    }

    if (best_end <= best_start) {
        int peak_idx = 0;
        for (std::size_t i = 0; i < mask.size(); ++i) {
            if (mask[i] != 0U) {
                peak_idx = static_cast<int>(i);
                break;
            }
        }
        return {peak_idx, std::min(peak_idx + 1, static_cast<int>(mask.size()))};
    }
    return {best_start, best_end};
}

void estimate_activity_interval(const std::vector<double>& x_smooth,
                                int spline_kernel_size,
                                double activity_threshold_ratio,
                                double activity_min_duration_s,
                                double fs,
                                std::vector<double>& envelope_smooth,
                                std::vector<std::uint8_t>& activity_mask,
                                int& start_idx,
                                int& end_idx) {
    const std::vector<cuSP::common::Complex> analytic = cuSP::common::hilbert_transform(x_smooth);
    const std::vector<double> envelope = cuSP::common::absolute_value(analytic);
    envelope_smooth = cuSP::bsplines::smooth_with_cubic_bspline(envelope, spline_kernel_size);

    const double threshold = activity_threshold_ratio * cuSP::common::safe_max(envelope_smooth);
    activity_mask.assign(envelope_smooth.size(), 0U);
    for (std::size_t i = 0; i < envelope_smooth.size(); ++i) {
        activity_mask[i] = (envelope_smooth[i] >= threshold) ? 1U : 0U;
    }

    const auto [seg_start, seg_end] = longest_true_segment(activity_mask);
    start_idx = seg_start;
    end_idx = seg_end;

    const int min_len = std::max(3, static_cast<int>(std::llround(activity_min_duration_s * fs)));
    if ((end_idx - start_idx) < min_len && !envelope_smooth.empty()) {
        const int peak_idx = cuSP::common::argmax_index(envelope_smooth);
        const int half_len = min_len / 2;
        start_idx = std::max(0, peak_idx - half_len);
        end_idx = std::min(static_cast<int>(envelope_smooth.size()), peak_idx + half_len);
    }
}

std::vector<double> kalman_smooth_1d(const std::vector<double>& measurements,
                                     double dt,
                                     double process_var,
                                     double meas_var) {
    if (measurements.empty()) {
        return {};
    }
    if (measurements.size() == 1) {
        return measurements;
    }

    double x0 = measurements[0];
    double x1 = 0.0;
    double P00 = 1.0;
    double P01 = 0.0;
    double P10 = 0.0;
    double P11 = 1.0;

    const double Q00 = (dt * dt * dt * dt / 4.0) * process_var;
    const double Q01 = (dt * dt * dt / 2.0) * process_var;
    const double Q10 = Q01;
    const double Q11 = (dt * dt) * process_var;

    std::vector<double> out(measurements.size(), 0.0);
    for (std::size_t i = 0; i < measurements.size(); ++i) {
        const double xp0 = x0 + dt * x1;
        const double xp1 = x1;

        const double PP00 = P00 + dt * (P10 + P01) + dt * dt * P11 + Q00;
        const double PP01 = P01 + dt * P11 + Q01;
        const double PP10 = P10 + dt * P11 + Q10;
        const double PP11 = P11 + Q11;

        const double innovation = measurements[i] - xp0;
        const double S = PP00 + meas_var;
        const double K0 = PP00 / S;
        const double K1 = PP10 / S;

        x0 = xp0 + K0 * innovation;
        x1 = xp1 + K1 * innovation;

        const double IKH00 = 1.0 - K0;
        const double IKH01 = 0.0;
        const double IKH10 = -K1;
        const double IKH11 = 1.0;

        const double A00 = IKH00 * PP00 + IKH01 * PP10;
        const double A01 = IKH00 * PP01 + IKH01 * PP11;
        const double A10 = IKH10 * PP00 + IKH11 * PP10;
        const double A11 = IKH10 * PP01 + IKH11 * PP11;

        P00 = A00 * IKH00 + A01 * IKH01 + K0 * meas_var * K0;
        P01 = A00 * IKH10 + A01 * IKH11 + K0 * meas_var * K1;
        P10 = A10 * IKH00 + A11 * IKH01 + K1 * meas_var * K0;
        P11 = A10 * IKH10 + A11 * IKH11 + K1 * meas_var * K1;

        out[i] = x0;
    }
    return out;
}

}  // namespace cuSP::estimation
