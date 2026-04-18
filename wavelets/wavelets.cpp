#include "wavelets.h"

#include "../common/common.h"

#include <algorithm>

namespace cuSP::wavelets {

namespace {

std::vector<cuSP::common::Complex> morlet2_wavelet(int M, double s, double w) {
    std::vector<cuSP::common::Complex> wavelet(static_cast<std::size_t>(M), cuSP::common::Complex{});
    const double scale = std::pow(cuSP::common::kPi, -0.25) * std::sqrt(1.0 / s);
    for (int i = 0; i < M; ++i) {
        const double x = (static_cast<double>(i) - 0.5 * static_cast<double>(M - 1)) / s;
        const double envelope = std::exp(-0.5 * x * x);
        wavelet[static_cast<std::size_t>(i)] =
            scale * envelope * cuSP::common::Complex(std::cos(w * x), std::sin(w * x));
    }
    return wavelet;
}

std::vector<cuSP::common::Complex> convolve_same_complex(const std::vector<double>& x,
                                                         const std::vector<cuSP::common::Complex>& kernel) {
    if (x.empty() || kernel.empty()) {
        return {};
    }

    const int n = static_cast<int>(x.size());
    const int m = static_cast<int>(kernel.size());
    const int start = (m - 1) / 2;
    std::vector<cuSP::common::Complex> y(static_cast<std::size_t>(n), cuSP::common::Complex{});

    for (int i = 0; i < n; ++i) {
        cuSP::common::Complex acc{};
        for (int k = 0; k < m; ++k) {
            const int idx = i - start + k;
            if (idx >= 0 && idx < n) {
                acc += x[static_cast<std::size_t>(idx)] * kernel[static_cast<std::size_t>(k)];
            }
        }
        y[static_cast<std::size_t>(i)] = acc;
    }
    return y;
}

}  // namespace

void extract_time_frequency_domain(const std::vector<double>& x_smooth,
                                   double fs,
                                   double cwt_w,
                                   int cwt_width_min,
                                   int cwt_width_max,
                                   std::vector<int>& widths,
                                   std::vector<double>& pseudo_freqs,
                                   cuSP::common::Matrix<float>& cwt_power,
                                   std::vector<double>& ridge_freq,
                                   std::vector<double>& ridge_energy) {
    widths.clear();
    pseudo_freqs.clear();
    for (int width = cwt_width_min; width <= cwt_width_max; ++width) {
        widths.push_back(width);
        pseudo_freqs.push_back(cwt_w * fs / (2.0 * cuSP::common::kPi * static_cast<double>(width)));
    }

    cwt_power.resize(widths.size(), x_smooth.size());
    for (std::size_t r = 0; r < widths.size(); ++r) {
        const int width = widths[r];
        const int M = std::min(10 * width, static_cast<int>(x_smooth.size()));
        std::vector<cuSP::common::Complex> wavelet = morlet2_wavelet(M, static_cast<double>(width), cwt_w);
        std::vector<cuSP::common::Complex> kernel(wavelet.size(), cuSP::common::Complex{});
        for (std::size_t i = 0; i < wavelet.size(); ++i) {
            kernel[i] = std::conj(wavelet[wavelet.size() - 1U - i]);
        }

        const std::vector<cuSP::common::Complex> conv = convolve_same_complex(x_smooth, kernel);
        for (std::size_t c = 0; c < conv.size(); ++c) {
            cwt_power(r, c) = static_cast<float>(std::norm(conv[c]));
        }
    }

    ridge_freq.assign(x_smooth.size(), 0.0);
    ridge_energy.assign(x_smooth.size(), 0.0);
    for (std::size_t c = 0; c < x_smooth.size(); ++c) {
        std::size_t best_r = 0;
        float best_val = -1.0f;
        for (std::size_t r = 0; r < widths.size(); ++r) {
            const float v = cwt_power(r, c);
            if (v > best_val) {
                best_val = v;
                best_r = r;
            }
        }
        ridge_freq[c] = pseudo_freqs[best_r];
        ridge_energy[c] = static_cast<double>(best_val);
    }
}

}  // namespace cuSP::wavelets
