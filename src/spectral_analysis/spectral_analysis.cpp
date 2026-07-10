#include "spectral_analysis/spectral_analysis.h"

#include "common/common.h"
#include "common/fftw_helpers.h"

#include <algorithm>

namespace cuSP::spectral_analysis {

namespace {

std::vector<double> hann_window_periodic(int n) {
    std::vector<double> w(static_cast<std::size_t>(std::max(0, n)), 0.0);
    if (n <= 1) {
        if (n == 1) {
            w[0] = 1.0;
        }
        return w;
    }
    for (int i = 0; i < n; ++i) {
        w[static_cast<std::size_t>(i)] =
            0.5 - 0.5 * std::cos(2.0 * cuSP::common::kPi * static_cast<double>(i) / static_cast<double>(n));
    }
    return w;
}

std::pair<std::vector<double>, std::vector<double>> welch_psd(const std::vector<double>& x,
                                                              double fs,
                                                              int nperseg,
                                                              int noverlap) {
    if (x.empty()) {
        return {{}, {}};
    }

    const int seg_len = std::min(nperseg, static_cast<int>(x.size()));
    const int overlap = std::min(noverlap, seg_len - 1);
    const int step = std::max(1, seg_len - overlap);
    const int nfft = seg_len;
    const int nfreq = nfft / 2 + 1;

    std::vector<double> freqs(static_cast<std::size_t>(nfreq), 0.0);
    std::vector<double> psd(static_cast<std::size_t>(nfreq), 0.0);
    const std::vector<double> window = hann_window_periodic(seg_len);

    double window_power = 0.0;
    for (double v : window) {
        window_power += v * v;
    }
    const double scale = 1.0 / (fs * std::max(window_power, 1e-12));

    int num_segments = 0;
    for (int start = 0; start + seg_len <= static_cast<int>(x.size()); start += step) {
        std::vector<cuSP::common::Complex> seg(static_cast<std::size_t>(seg_len), cuSP::common::Complex{});
        double mean = 0.0;
        for (int i = 0; i < seg_len; ++i) {
            mean += x[static_cast<std::size_t>(start + i)];
        }
        mean /= static_cast<double>(seg_len);

        for (int i = 0; i < seg_len; ++i) {
            const double v = (x[static_cast<std::size_t>(start + i)] - mean) *
                             window[static_cast<std::size_t>(i)];
            seg[static_cast<std::size_t>(i)] = cuSP::common::Complex(v, 0.0);
        }

        const std::vector<cuSP::common::Complex> spectrum = cuSP::common::fft_complex(seg, nfft);
        for (int k = 0; k < nfreq; ++k) {
            double power = std::norm(spectrum[static_cast<std::size_t>(k)]) * scale;
            if (k != 0 && !(nfft % 2 == 0 && k == nfft / 2)) {
                power *= 2.0;
            }
            psd[static_cast<std::size_t>(k)] += power;
        }
        ++num_segments;
    }

    const double denom = std::max(1, num_segments);
    for (int k = 0; k < nfreq; ++k) {
        freqs[static_cast<std::size_t>(k)] = static_cast<double>(k) * fs / static_cast<double>(nfft);
        psd[static_cast<std::size_t>(k)] /= static_cast<double>(denom);
    }
    return {freqs, psd};
}

}  // namespace

void extract_frequency_domain(const std::vector<double>& x_smooth,
                              double fs,
                              int nperseg,
                              int noverlap,
                              std::vector<double>& freqs_pos,
                              std::vector<double>& psd_pos) {
    auto result = welch_psd(x_smooth, fs, nperseg, noverlap);
    freqs_pos = std::move(result.first);
    psd_pos = std::move(result.second);
}

}  // namespace cuSP::spectral_analysis
