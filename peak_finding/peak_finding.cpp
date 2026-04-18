#include "peak_finding.h"

#include "../common/fftw_helpers.h"

#include <algorithm>

namespace cuSP::peak_finding {

std::vector<int> detect_peaks_1d(const std::vector<double>& x, int order, double threshold_ratio) {
    std::vector<int> peaks;
    if (x.empty()) {
        return peaks;
    }

    const int n = static_cast<int>(x.size());
    const int ord = std::max(1, order);
    for (int i = 0; i < n; ++i) {
        bool is_peak = true;
        for (int delta = 1; delta <= ord; ++delta) {
            const int left = std::clamp(i - delta, 0, n - 1);
            const int right = std::clamp(i + delta, 0, n - 1);
            if (!(x[static_cast<std::size_t>(i)] > x[static_cast<std::size_t>(left)] &&
                  x[static_cast<std::size_t>(i)] > x[static_cast<std::size_t>(right)])) {
                is_peak = false;
                break;
            }
        }
        if (is_peak) {
            peaks.push_back(i);
        }
    }

    if (threshold_ratio > 0.0 && !peaks.empty()) {
        const double threshold = threshold_ratio * cuSP::common::safe_max(x);
        std::vector<int> filtered;
        for (int idx : peaks) {
            if (x[static_cast<std::size_t>(idx)] >= threshold) {
                filtered.push_back(idx);
            }
        }
        peaks = std::move(filtered);
    }
    return peaks;
}

std::vector<int> fallback_global_peak(const std::vector<double>& x) {
    if (x.empty()) {
        return {};
    }
    return {cuSP::common::argmax_index(x)};
}

}  // namespace cuSP::peak_finding
