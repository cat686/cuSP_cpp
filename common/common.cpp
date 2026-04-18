#include "common.h"

namespace cuSP::common {

double db20(double x, double eps) {
    return 20.0 * std::log10(std::max(std::abs(x), eps));
}

double db10(double x, double eps) {
    return 10.0 * std::log10(std::max(x, eps));
}

std::size_t next_pow2(std::size_t n) {
    if (n <= 1) {
        return 1;
    }

    std::size_t value = 1;
    while (value < n) {
        value <<= 1;
    }
    return value;
}

std::vector<double> make_window(const std::string& name, std::size_t n) {
    std::vector<double> window(n, 1.0);
    if (name.empty()) {
        return window;
    }

    std::string lower = name;
    std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });

    if (n == 0) {
        return window;
    }
    if (n == 1) {
        window[0] = 1.0;
        return window;
    }

    if (lower == "hamming") {
        for (std::size_t i = 0; i < n; ++i) {
            window[i] = 0.54 - 0.46 * std::cos(2.0 * kPi * static_cast<double>(i) /
                                               static_cast<double>(n - 1));
        }
        return window;
    }

    if (lower == "hann" || lower == "hanning") {
        for (std::size_t i = 0; i < n; ++i) {
            window[i] = 0.5 - 0.5 * std::cos(2.0 * kPi * static_cast<double>(i) /
                                             static_cast<double>(n - 1));
        }
        return window;
    }

    throw std::invalid_argument("Unsupported window type: " + name);
}

void fft_inplace(std::vector<Complex>& a, bool inverse) {
    const std::size_t n = a.size();
    if (n == 0) {
        return;
    }
    if ((n & (n - 1)) != 0) {
        throw std::invalid_argument("fft_inplace requires power-of-two length");
    }

    for (std::size_t i = 1, j = 0; i < n; ++i) {
        std::size_t bit = n >> 1;
        for (; j & bit; bit >>= 1) {
            j ^= bit;
        }
        j ^= bit;
        if (i < j) {
            std::swap(a[i], a[j]);
        }
    }

    for (std::size_t len = 2; len <= n; len <<= 1) {
        const double angle = 2.0 * kPi / static_cast<double>(len) * (inverse ? 1.0 : -1.0);
        const Complex wlen(std::cos(angle), std::sin(angle));
        for (std::size_t i = 0; i < n; i += len) {
            Complex w(1.0, 0.0);
            for (std::size_t j = 0; j < len / 2; ++j) {
                const Complex u = a[i + j];
                const Complex v = a[i + j + len / 2] * w;
                a[i + j] = u + v;
                a[i + j + len / 2] = u - v;
                w *= wlen;
            }
        }
    }

    if (inverse) {
        const double scale = 1.0 / static_cast<double>(n);
        for (auto& value : a) {
            value *= scale;
        }
    }
}

}  // namespace cuSP::common
