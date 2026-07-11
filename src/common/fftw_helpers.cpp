#include "common/fftw_helpers.h"
#include "common/common.h"

#include <algorithm>
#include <stdexcept>

namespace cuSP::common {

std::vector<Complex> fft_complex(const std::vector<Complex>& x, int n) {
    const int fft_size = (n > 0) ? n : static_cast<int>(x.size());
    if (fft_size <= 0) {
        return {};
    }

    const std::size_t padded_size = next_pow2(static_cast<std::size_t>(fft_size));
    std::vector<Complex> padded(padded_size, Complex{});
    const std::size_t copy_len = std::min(static_cast<std::size_t>(fft_size), x.size());
    std::copy_n(x.begin(), copy_len, padded.begin());

    fft_inplace(padded, false);

    padded.resize(static_cast<std::size_t>(fft_size));
    return padded;
}

std::vector<Complex> ifft_complex(const std::vector<Complex>& x, int n) {
    const int fft_size = (n > 0) ? n : static_cast<int>(x.size());
    if (fft_size <= 0) {
        return {};
    }

    const std::size_t padded_size = next_pow2(static_cast<std::size_t>(fft_size));
    std::vector<Complex> padded(padded_size, Complex{});
    const std::size_t copy_len = std::min(static_cast<std::size_t>(fft_size), x.size());
    std::copy_n(x.begin(), copy_len, padded.begin());

    fft_inplace(padded, true);

    padded.resize(static_cast<std::size_t>(fft_size));
    const double scale_correction =
        static_cast<double>(padded_size) / static_cast<double>(fft_size);
    for (auto& v : padded) {
        v *= scale_correction;
    }
    return padded;
}

std::vector<double> absolute_value(const std::vector<Complex>& x) {
    std::vector<double> y(x.size(), 0.0);
    for (std::size_t i = 0; i < x.size(); ++i) {
        y[i] = std::abs(x[i]);
    }
    return y;
}

double safe_max(const std::vector<double>& x) {
    if (x.empty()) {
        return 0.0;
    }
    return *std::max_element(x.begin(), x.end());
}

int argmax_index(const std::vector<double>& x) {
    if (x.empty()) {
        return 0;
    }
    return static_cast<int>(std::distance(x.begin(), std::max_element(x.begin(), x.end())));
}

std::vector<int> argsort_descending(const std::vector<double>& x) {
    std::vector<int> idx(x.size(), 0);
    for (std::size_t i = 0; i < x.size(); ++i) {
        idx[i] = static_cast<int>(i);
    }
    std::sort(idx.begin(), idx.end(), [&](int a, int b) {
        return x[static_cast<std::size_t>(a)] > x[static_cast<std::size_t>(b)];
    });
    return idx;
}

std::vector<double> unwrap_phase(const std::vector<double>& phase) {
    if (phase.empty()) {
        return {};
    }

    std::vector<double> out(phase.size(), 0.0);
    out[0] = phase[0];
    double offset = 0.0;
    for (std::size_t i = 1; i < phase.size(); ++i) {
        double delta = phase[i] - phase[i - 1];
        while (delta > kPi) {
            offset -= 2.0 * kPi;
            delta -= 2.0 * kPi;
        }
        while (delta < -kPi) {
            offset += 2.0 * kPi;
            delta += 2.0 * kPi;
        }
        out[i] = phase[i] + offset;
    }
    return out;
}

std::vector<double> diff_vector(const std::vector<double>& x) {
    if (x.size() <= 1) {
        return {};
    }

    std::vector<double> y(x.size() - 1, 0.0);
    for (std::size_t i = 0; i + 1 < x.size(); ++i) {
        y[i] = x[i + 1] - x[i];
    }
    return y;
}

std::vector<double> abs_vector(const std::vector<double>& x) {
    std::vector<double> y(x.size(), 0.0);
    for (std::size_t i = 0; i < x.size(); ++i) {
        y[i] = std::abs(x[i]);
    }
    return y;
}

std::vector<double> reverse_copy(const std::vector<double>& x) {
    return std::vector<double>(x.rbegin(), x.rend());
}

std::vector<double> fft_convolve_real_full(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.empty() || b.empty()) {
        return {};
    }

    const std::size_t full_size = a.size() + b.size() - 1;
    const std::size_t nfft = next_pow2(full_size);

    std::vector<Complex> ac(nfft, Complex{});
    std::vector<Complex> bc(nfft, Complex{});
    for (std::size_t i = 0; i < a.size(); ++i) {
        ac[i] = Complex(a[i], 0.0);
    }
    for (std::size_t i = 0; i < b.size(); ++i) {
        bc[i] = Complex(b[i], 0.0);
    }

    std::vector<Complex> A = fft_complex(ac, static_cast<int>(nfft));
    std::vector<Complex> B = fft_complex(bc, static_cast<int>(nfft));
    for (std::size_t i = 0; i < nfft; ++i) {
        A[i] *= B[i];
    }
    const std::vector<Complex> c = ifft_complex(A, static_cast<int>(nfft));

    std::vector<double> out(full_size, 0.0);
    for (std::size_t i = 0; i < full_size; ++i) {
        out[i] = c[i].real();
    }
    return out;
}

std::vector<Complex> hilbert_transform(const std::vector<double>& x) {
    std::vector<Complex> xc(x.size(), Complex{});
    for (std::size_t i = 0; i < x.size(); ++i) {
        xc[i] = Complex(x[i], 0.0);
    }

    std::vector<Complex> X = fft_complex(xc);
    const std::size_t n = X.size();
    if (n == 0) {
        return {};
    }

    std::vector<double> h(n, 0.0);
    if (n % 2 == 0) {
        h[0] = 1.0;
        h[n / 2] = 1.0;
        for (std::size_t i = 1; i < n / 2; ++i) {
            h[i] = 2.0;
        }
    } else {
        h[0] = 1.0;
        for (std::size_t i = 1; i <= n / 2; ++i) {
            h[i] = 2.0;
        }
    }

    for (std::size_t i = 0; i < n; ++i) {
        X[i] *= h[i];
    }
    return ifft_complex(X);
}

}  // namespace cuSP::common