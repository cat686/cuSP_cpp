#pragma once

#include "common/common.h"

#include <vector>

namespace cuSP::convolution {

using cuSP::common::Complex;

inline std::vector<Complex> correlate_full(const std::vector<Complex>& a, const std::vector<Complex>& b) {
    if (a.empty() || b.empty()) {
        return {};
    }

    const int na = static_cast<int>(a.size());
    const int nb = static_cast<int>(b.size());
    std::vector<Complex> out(static_cast<std::size_t>(na + nb - 1), Complex(0.0, 0.0));

    for (int lag = -(nb - 1); lag <= na - 1; ++lag) {
        Complex sum(0.0, 0.0);
        for (int i = 0; i < na; ++i) {
            const int j = i - lag;
            if (j >= 0 && j < nb) {
                sum += a[static_cast<std::size_t>(i)] *
                       std::conj(b[static_cast<std::size_t>(j)]);
            }
        }
        out[static_cast<std::size_t>(lag + nb - 1)] = sum;
    }

    return out;
}
void extract_time_domain(const std::vector<double>& x_smooth,
                         double fs,
                         std::vector<double>& lags_pos,
                         std::vector<double>& autocorr_pos);

}  // namespace cuSP::convolution
