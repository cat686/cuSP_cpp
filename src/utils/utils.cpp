#include "utils/utils.h"

#include <algorithm>
#include <cmath>
#include <random>

namespace cuSP::utils {

std::vector<double> build_time_axis(double fs, double duration) {
    const int n = static_cast<int>(std::llround(fs * duration));
    std::vector<double> t(static_cast<std::size_t>(std::max(0, n)), 0.0);
    for (int i = 0; i < n; ++i) {
        t[static_cast<std::size_t>(i)] = static_cast<double>(i) / fs;
    }
    return t;
}

std::vector<Complex> apply_integer_delay(const std::vector<Complex>& x, double fs, double delay_s) {
    const int delay_samples = static_cast<int>(std::llround(delay_s * fs));
    if (delay_samples <= 0) {
        return x;
    }

    std::vector<Complex> out(x.size(), Complex{});
    for (std::size_t i = static_cast<std::size_t>(delay_samples); i < x.size(); ++i) {
        out[i] = x[i - static_cast<std::size_t>(delay_samples)];
    }
    return out;
}

std::vector<double> add_awgn(const std::vector<double>& x, double snr_db, std::uint32_t seed) {
    if (x.empty()) {
        return {};
    }

    double signal_power = 0.0;
    for (double v : x) {
        signal_power += v * v;
    }
    signal_power /= static_cast<double>(x.size());

    const double snr_linear = std::pow(10.0, snr_db / 10.0);
    const double noise_power = signal_power / std::max(snr_linear, 1e-12);
    const double sigma = std::sqrt(std::max(noise_power, 0.0));

    std::mt19937 rng(seed);
    std::normal_distribution<double> normal(0.0, sigma);

    std::vector<double> y(x.size(), 0.0);
    for (std::size_t i = 0; i < x.size(); ++i) {
        y[i] = x[i] + normal(rng);
    }
    return y;
}

}  // namespace cuSP::utils
