#include "convolution/convolution.h"

#include "common/fftw_helpers.h"

#include <algorithm>

namespace cuSP::convolution {

void extract_time_domain(const std::vector<double>& x_smooth,
                         double fs,
                         std::vector<double>& lags_pos,
                         std::vector<double>& autocorr_pos) {
    std::vector<double> reversed = cuSP::common::reverse_copy(x_smooth);
    std::vector<double> autocorr = cuSP::common::fft_convolve_real_full(x_smooth, reversed);
    for (double& v : autocorr) {
        v = std::abs(v);
    }
    const double maxv = std::max(cuSP::common::safe_max(autocorr), 1e-12);
    for (double& v : autocorr) {
        v /= maxv;
    }

    const int n = static_cast<int>(x_smooth.size());
    lags_pos.clear();
    autocorr_pos.clear();
    for (int idx = n; idx < 2 * n - 1; ++idx) {
        lags_pos.push_back(static_cast<double>(idx - (n - 1)) / fs);
        autocorr_pos.push_back(autocorr[static_cast<std::size_t>(idx)]);
    }
}

}  // namespace cuSP::convolution
