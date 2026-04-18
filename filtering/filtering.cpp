#include "filtering.h"

#include "../common/common.h"
#include "../common/fftw_helpers.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace cuSP::filtering {

namespace {

double sinc_pi(double x) {
    if (std::abs(x) < 1e-12) {
        return 1.0;
    }
    return std::sin(cuSP::common::kPi * x) / (cuSP::common::kPi * x);
}

std::vector<double> hamming_window(int n, bool fftbins) {
    std::vector<double> w(static_cast<std::size_t>(std::max(0, n)), 0.0);
    if (n <= 1) {
        if (n == 1) {
            w[0] = 1.0;
        }
        return w;
    }

    const double denom = fftbins ? static_cast<double>(n) : static_cast<double>(n - 1);
    for (int i = 0; i < n; ++i) {
        w[static_cast<std::size_t>(i)] =
            0.54 - 0.46 * std::cos(2.0 * cuSP::common::kPi * static_cast<double>(i) / denom);
    }
    return w;
}

std::vector<double> firwin_bandpass(int numtaps, double low_hz, double high_hz, double fs) {
    const int taps = ensure_odd(numtaps);
    const double nyq = 0.5 * fs;
    const double left = low_hz / nyq;
    const double right = high_hz / nyq;
    const double alpha = 0.5 * static_cast<double>(taps - 1);
    const std::vector<double> win = hamming_window(taps, false);

    std::vector<double> h(static_cast<std::size_t>(taps), 0.0);
    for (int i = 0; i < taps; ++i) {
        const double m = static_cast<double>(i) - alpha;
        h[static_cast<std::size_t>(i)] =
            (right * sinc_pi(right * m) - left * sinc_pi(left * m)) * win[static_cast<std::size_t>(i)];
    }

    const double scale_frequency = 0.5 * (left + right);
    double scale = 0.0;
    for (int i = 0; i < taps; ++i) {
        const double m = static_cast<double>(i) - alpha;
        scale += h[static_cast<std::size_t>(i)] * std::cos(cuSP::common::kPi * m * scale_frequency);
    }
    scale = (std::abs(scale) < 1e-12) ? 1.0 : scale;
    for (double& v : h) {
        v /= scale;
    }
    return h;
}

std::vector<double> odd_extension(const std::vector<double>& x, int edge) {
    if (edge <= 0) {
        return x;
    }

    const int n = static_cast<int>(x.size());
    std::vector<double> ext(static_cast<std::size_t>(n + 2 * edge), 0.0);
    for (int i = 0; i < edge; ++i) {
        ext[static_cast<std::size_t>(i)] = 2.0 * x.front() - x[static_cast<std::size_t>(edge - i)];
    }
    for (int i = 0; i < n; ++i) {
        ext[static_cast<std::size_t>(edge + i)] = x[static_cast<std::size_t>(i)];
    }
    for (int i = 0; i < edge; ++i) {
        ext[static_cast<std::size_t>(edge + n + i)] = 2.0 * x.back() - x[static_cast<std::size_t>(n - 2 - i)];
    }
    return ext;
}

std::vector<double> constant_extension(const std::vector<double>& x, int edge) {
    if (edge <= 0) {
        return x;
    }

    const int n = static_cast<int>(x.size());
    std::vector<double> ext(static_cast<std::size_t>(n + 2 * edge), 0.0);
    for (int i = 0; i < edge; ++i) {
        ext[static_cast<std::size_t>(i)] = x.front();
    }
    for (int i = 0; i < n; ++i) {
        ext[static_cast<std::size_t>(edge + i)] = x[static_cast<std::size_t>(i)];
    }
    for (int i = 0; i < edge; ++i) {
        ext[static_cast<std::size_t>(edge + n + i)] = x.back();
    }
    return ext;
}

std::vector<double> even_extension(const std::vector<double>& x, int edge) {
    if (edge <= 0) {
        return x;
    }

    const int n = static_cast<int>(x.size());
    std::vector<double> ext(static_cast<std::size_t>(n + 2 * edge), 0.0);
    for (int i = 0; i < edge; ++i) {
        ext[static_cast<std::size_t>(i)] = x[static_cast<std::size_t>(edge - i)];
    }
    for (int i = 0; i < n; ++i) {
        ext[static_cast<std::size_t>(edge + i)] = x[static_cast<std::size_t>(i)];
    }
    for (int i = 0; i < edge; ++i) {
        ext[static_cast<std::size_t>(edge + n + i)] = x[static_cast<std::size_t>(n - 2 - i)];
    }
    return ext;
}

std::vector<double> firfilter_zi_impl(const std::vector<double>& b) {
    if (b.size() <= 1) {
        return {};
    }

    std::vector<double> zi(b.size() - 1, 0.0);
    double acc = 0.0;
    for (int i = static_cast<int>(b.size()) - 1; i >= 1; --i) {
        acc += b[static_cast<std::size_t>(i)];
        zi[static_cast<std::size_t>(i - 1)] = acc;
    }
    return zi;
}

std::pair<std::vector<double>, std::vector<double>> firfilter_impl(
    const std::vector<double>& b,
    const std::vector<double>& x,
    const std::vector<double>& zi) {
    const std::size_t full_size = x.size() + b.size() - 1;
    std::vector<double> out_full(full_size, 0.0);

    for (std::size_t i = 0; i < x.size(); ++i) {
        for (std::size_t j = 0; j < b.size(); ++j) {
            out_full[i + j] += b[j] * x[i];
        }
    }

    for (std::size_t i = 0; i < zi.size(); ++i) {
        out_full[i] += zi[i];
    }

    std::vector<double> y(x.size(), 0.0);
    std::copy(out_full.begin(), out_full.begin() + static_cast<std::ptrdiff_t>(x.size()), y.begin());

    std::vector<double> zf(b.size() > 0 ? (b.size() - 1) : 0, 0.0);
    if (!zf.empty()) {
        std::copy(out_full.begin() + static_cast<std::ptrdiff_t>(x.size()),
                  out_full.begin() + static_cast<std::ptrdiff_t>(x.size() + zf.size()),
                  zf.begin());
    }
    return {y, zf};
}

std::vector<double> filtfilt_fir(const std::vector<double>& b,
                                 const std::vector<double>& x,
                                 const std::string& padtype) {
    if (x.empty()) {
        return {};
    }

    const int edge = static_cast<int>(b.size()) * 3;
    if (static_cast<int>(x.size()) <= edge) {
        throw std::runtime_error("Input length must be larger than filtfilt padlen.");
    }

    std::vector<double> ext;
    if (padtype == "odd") {
        ext = odd_extension(x, edge);
    } else if (padtype == "even") {
        ext = even_extension(x, edge);
    } else if (padtype == "constant") {
        ext = constant_extension(x, edge);
    } else {
        ext = x;
    }

    const std::vector<double> zi = firfilter_zi_impl(b);
    std::vector<double> zi_forward = zi;
    for (double& v : zi_forward) {
        v *= ext.front();
    }
    auto forward = firfilter_impl(b, ext, zi_forward).first;

    std::vector<double> rev = cuSP::common::reverse_copy(forward);
    std::vector<double> zi_backward = zi;
    for (double& v : zi_backward) {
        v *= forward.back();
    }
    auto backward = firfilter_impl(b, rev, zi_backward).first;
    backward = cuSP::common::reverse_copy(backward);

    std::vector<double> y(x.size(), 0.0);
    std::copy(backward.begin() + edge,
              backward.begin() + edge + static_cast<std::ptrdiff_t>(x.size()),
              y.begin());
    return y;
}

}  // namespace

int ensure_odd(int n) {
    return (n % 2 == 0) ? (n + 1) : n;
}

void preprocess_signal(const std::vector<double>& rx,
                       int fir_numtaps,
                       double fir_band_low_hz,
                       double fir_band_high_hz,
                       double fs,
                       const std::string& filtfilt_padtype,
                       std::vector<double>& detrended,
                       std::vector<double>& filtered) {
    detrended = rx;
    if (!detrended.empty()) {
        const double mean =
            std::accumulate(detrended.begin(), detrended.end(), 0.0) / static_cast<double>(detrended.size());
        for (double& v : detrended) {
            v -= mean;
        }
    }

    const std::vector<double> b = firwin_bandpass(fir_numtaps, fir_band_low_hz, fir_band_high_hz, fs);
    filtered = filtfilt_fir(b, detrended, filtfilt_padtype);
}

}  // namespace cuSP::filtering
