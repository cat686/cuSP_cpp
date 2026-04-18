#include "radartools.h"

namespace cuSP::radartools {

std::vector<Complex> make_lfm_pulse(double fs, double Tp, double B) {
    const std::size_t N = static_cast<std::size_t>(std::llround(Tp * fs));
    std::vector<Complex> pulse(N);

    const double k = B / Tp;
    for (std::size_t i = 0; i < N; ++i) {
        const double t = (static_cast<double>(i) - static_cast<double>(N) / 2.0) / fs;
        const double phase = cuSP::common::kPi * k * t * t;
        pulse[i] = Complex(std::cos(phase), std::sin(phase));
    }
    return pulse;
}

Matrix<Complex> simulate_echo_matrix(const std::vector<Complex>& templ,
                                     double fs,
                                     double PRI,
                                     std::size_t num_pulses,
                                     double fc,
                                     const std::vector<Target>& targets,
                                     double snr_db,
                                     std::mt19937& rng) {
    const double lambda = cuSP::common::kLightSpeed / fc;
    const std::size_t samples_per_pulse = templ.size();

    Matrix<Complex> x(num_pulses, samples_per_pulse, Complex(0.0, 0.0));

    for (const auto& target : targets) {
        const double tau = 2.0 * target.R / cuSP::common::kLightSpeed;
        const double fd = 2.0 * target.v / lambda;
        const std::size_t delay_samp = static_cast<std::size_t>(std::llround(tau * fs));

        if (delay_samp >= samples_per_pulse) {
            continue;
        }

        std::vector<Complex> delayed(samples_per_pulse, Complex(0.0, 0.0));
        const std::size_t valid_len = samples_per_pulse - delay_samp;
        for (std::size_t i = 0; i < valid_len; ++i) {
            delayed[delay_samp + i] = templ[i];
        }

        for (std::size_t p = 0; p < num_pulses; ++p) {
            const double phase = 2.0 * cuSP::common::kPi * fd * static_cast<double>(p) * PRI;
            const Complex slow_phase(std::cos(phase), std::sin(phase));
            for (std::size_t n = 0; n < samples_per_pulse; ++n) {
                x(p, n) += target.amp * slow_phase * delayed[n];
            }
        }
    }

    double signal_power = 0.0;
    for (const auto& value : x.data) {
        signal_power += std::norm(value);
    }
    signal_power /= static_cast<double>(x.data.size());

    const double snr_linear = std::pow(10.0, snr_db / 10.0);
    const double noise_power = (snr_linear > 0.0) ? (signal_power / snr_linear) : 0.0;
    const double noise_std = std::sqrt(noise_power / 2.0);

    std::normal_distribution<double> gaussian(0.0, 1.0);
    for (auto& value : x.data) {
        value += noise_std * Complex(gaussian(rng), gaussian(rng));
    }

    return x;
}

std::vector<double> build_range_axis(double fs, std::size_t nfft) {
    std::vector<double> range_axis(nfft, 0.0);
    for (std::size_t i = 0; i < nfft; ++i) {
        const double tau = static_cast<double>(i) / fs;
        range_axis[i] = cuSP::common::kLightSpeed * tau / 2.0;
    }
    return range_axis;
}

std::vector<double> build_velocity_axis(double fc, double PRI, std::size_t nfft_doppler) {
    const double lambda = cuSP::common::kLightSpeed / fc;
    std::vector<double> velocity_axis(nfft_doppler, 0.0);

    const long long half = static_cast<long long>(nfft_doppler) / 2;
    for (std::size_t i = 0; i < nfft_doppler; ++i) {
        const long long shifted_bin = static_cast<long long>(i) - half;
        const double fd = static_cast<double>(shifted_bin) /
                          (static_cast<double>(nfft_doppler) * PRI);
        velocity_axis[i] = lambda * fd / 2.0;
    }
    return velocity_axis;
}

Matrix<Complex> pulse_compression_cpu(const Matrix<Complex>& x,
                                      const std::vector<Complex>& templ,
                                      bool normalize,
                                      const std::string& window,
                                      std::size_t nfft) {
    const std::size_t num_pulses = x.rows;
    const std::size_t samples_per_pulse = x.cols;
    if (nfft == 0) {
        nfft = samples_per_pulse;
    }

    std::vector<Complex> tpl = templ;
    if (!window.empty()) {
        const auto weights = cuSP::common::make_window(window, tpl.size());
        for (std::size_t i = 0; i < tpl.size(); ++i) {
            tpl[i] *= weights[i];
        }
    }

    if (normalize) {
        double norm = 0.0;
        for (const auto& value : tpl) {
            norm += std::norm(value);
        }
        norm = std::sqrt(norm);
        if (norm > 0.0) {
            for (auto& value : tpl) {
                value /= norm;
            }
        }
    }

    std::vector<Complex> fft_template(nfft, Complex(0.0, 0.0));
    for (std::size_t i = 0; i < tpl.size() && i < nfft; ++i) {
        fft_template[i] = tpl[i];
    }
    cuSP::common::fft_inplace(fft_template, false);
    for (auto& value : fft_template) {
        value = std::conj(value);
    }

    Matrix<Complex> compressed(num_pulses, nfft, Complex(0.0, 0.0));
    std::vector<Complex> fft_row(nfft, Complex(0.0, 0.0));

    for (std::size_t p = 0; p < num_pulses; ++p) {
        std::fill(fft_row.begin(), fft_row.end(), Complex(0.0, 0.0));
        for (std::size_t n = 0; n < samples_per_pulse && n < nfft; ++n) {
            fft_row[n] = x(p, n);
        }

        cuSP::common::fft_inplace(fft_row, false);
        for (std::size_t k = 0; k < nfft; ++k) {
            fft_row[k] *= fft_template[k];
        }
        cuSP::common::fft_inplace(fft_row, true);

        for (std::size_t k = 0; k < nfft; ++k) {
            compressed(p, k) = fft_row[k];
        }
    }

    return compressed;
}

Matrix<Complex> pulse_doppler_cpu(const Matrix<Complex>& x,
                                  const std::string& window,
                                  std::size_t nfft) {
    const std::size_t num_pulses = x.rows;
    const std::size_t samples_per_pulse = x.cols;
    if (nfft == 0) {
        nfft = num_pulses;
    }

    std::vector<double> weights;
    if (!window.empty()) {
        weights = cuSP::common::make_window(window, num_pulses);
    }

    Matrix<Complex> pd(nfft, samples_per_pulse, Complex(0.0, 0.0));
    std::vector<Complex> fft_col(nfft, Complex(0.0, 0.0));

    for (std::size_t c = 0; c < samples_per_pulse; ++c) {
        std::fill(fft_col.begin(), fft_col.end(), Complex(0.0, 0.0));
        for (std::size_t r = 0; r < num_pulses; ++r) {
            fft_col[r] = x(r, c);
            if (!weights.empty()) {
                fft_col[r] *= weights[r];
            }
        }

        cuSP::common::fft_inplace(fft_col, false);
        for (std::size_t r = 0; r < nfft; ++r) {
            pd(r, c) = fft_col[r];
        }
    }

    return pd;
}

double cfar_alpha_cpu(double pfa, int N) {
    return static_cast<double>(N) * (std::pow(pfa, -1.0 / static_cast<double>(N)) - 1.0);
}

Matrix<double> integral_image_2d(const Matrix<float>& array) {
    Matrix<double> integral(array.rows, array.cols, 0.0);
    for (std::size_t i = 0; i < array.rows; ++i) {
        double row_sum = 0.0;
        for (std::size_t j = 0; j < array.cols; ++j) {
            row_sum += static_cast<double>(array(i, j));
            integral(i, j) = row_sum + (i > 0 ? integral(i - 1, j) : 0.0);
        }
    }
    return integral;
}

double rect_sum_2d(const Matrix<double>& integral, int x1, int y1, int x2, int y2) {
    double total = integral(static_cast<std::size_t>(x2), static_cast<std::size_t>(y2));
    if (x1 > 0) {
        total -= integral(static_cast<std::size_t>(x1 - 1), static_cast<std::size_t>(y2));
    }
    if (y1 > 0) {
        total -= integral(static_cast<std::size_t>(x2), static_cast<std::size_t>(y1 - 1));
    }
    if (x1 > 0 && y1 > 0) {
        total += integral(static_cast<std::size_t>(x1 - 1), static_cast<std::size_t>(y1 - 1));
    }
    return total;
}

std::pair<Matrix<float>, Matrix<std::uint8_t>> ca_cfar_cpu(const Matrix<float>& array,
                                                           std::pair<int, int> guard_cells,
                                                           std::pair<int, int> reference_cells,
                                                           double pfa) {
    if (array.rows == 0 || array.cols == 0) {
        throw std::invalid_argument("ca_cfar_cpu requires a non-empty 2D array");
    }

    const int gx = guard_cells.first;
    const int gy = guard_cells.second;
    const int rx = reference_cells.first;
    const int ry = reference_cells.second;

    if (static_cast<int>(array.rows) - 2 * gx - 2 * rx <= 0) {
        throw std::invalid_argument("Array first dimension too small for given parameters.");
    }
    if (static_cast<int>(array.cols) - 2 * gy - 2 * ry <= 0) {
        throw std::invalid_argument("Array second dimension too small for given parameters.");
    }

    Matrix<float> threshold(array.rows, array.cols, 0.0f);
    Matrix<std::uint8_t> detections(array.rows, array.cols, 0U);
    const auto integral = integral_image_2d(array);

    const int N = 2 * rx * (2 * ry + 2 * gy + 1) + 2 * (2 * gx + 1) * ry;
    const double alpha = cfar_alpha_cpu(pfa, N);

    for (int i = 0; i < static_cast<int>(array.rows) - 2 * (rx + gx); ++i) {
        for (int j = 0; j < static_cast<int>(array.cols) - 2 * (ry + gy); ++j) {
            const int x = i + gx + rx;
            const int y = j + gy + ry;

            const int x1o = x - gx - rx;
            const int y1o = y - gy - ry;
            const int x2o = x + gx + rx;
            const int y2o = y + gy + ry;

            const int x1i = x - gx;
            const int y1i = y - gy;
            const int x2i = x + gx;
            const int y2i = y + gy;

            const double outer_area = rect_sum_2d(integral, x1o, y1o, x2o, y2o);
            const double inner_area = rect_sum_2d(integral, x1i, y1i, x2i, y2i);
            const float T = static_cast<float>((outer_area - inner_area) * alpha /
                                               static_cast<double>(N));
            threshold(static_cast<std::size_t>(x), static_cast<std::size_t>(y)) = T;
        }
    }

    for (std::size_t i = 0; i < array.rows; ++i) {
        for (std::size_t j = 0; j < array.cols; ++j) {
            detections(i, j) = (array(i, j) - threshold(i, j) > 0.0f) ? 1U : 0U;
        }
    }

    return {threshold, detections};
}

Matrix<float> ambgfun_2d(const std::vector<Complex>& x,
                         double fs,
                         double prf,
                         const std::vector<Complex>* y,
                         const std::string& cut,
                         double cut_value) {
    (void)fs;
    (void)prf;
    (void)cut_value;

    std::string cut_lower = cut;
    std::transform(cut_lower.begin(), cut_lower.end(), cut_lower.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });

    if (cut_lower != "2d") {
        throw std::invalid_argument("task1.cpp only needs ambgfun_cpu(..., cut=\"2d\")");
    }
    if (x.empty()) {
        throw std::invalid_argument("ambgfun_cpu requires a non-empty input waveform");
    }

    std::vector<Complex> xnorm = x;
    double xnorm_scale = 0.0;
    for (const auto& value : xnorm) {
        xnorm_scale += std::norm(value);
    }
    xnorm_scale = std::sqrt(xnorm_scale);
    if (xnorm_scale > 0.0) {
        for (auto& value : xnorm) {
            value /= xnorm_scale;
        }
    }

    std::vector<Complex> ynorm = y ? *y : x;
    double ynorm_scale = 0.0;
    for (const auto& value : ynorm) {
        ynorm_scale += std::norm(value);
    }
    ynorm_scale = std::sqrt(ynorm_scale);
    if (ynorm_scale > 0.0) {
        for (auto& value : ynorm) {
            value /= ynorm_scale;
        }
    }

    const std::size_t len_seq = xnorm.size() + ynorm.size();
    const std::size_t full_rows = len_seq - 1;
    const std::size_t nfreq = cuSP::common::next_pow2(len_seq - 1);
    const std::size_t xlen = xnorm.size();

    Matrix<float> data(full_rows, nfreq, 0.0f);
    std::vector<Complex> row_freq(nfreq, Complex(0.0, 0.0));

    for (std::size_t row = 0; row < full_rows; ++row) {
        std::fill(row_freq.begin(), row_freq.end(), Complex(0.0, 0.0));

        for (std::size_t col = 0; col < xlen; ++col) {
            const long long x_col = static_cast<long long>(col) -
                                    static_cast<long long>(xlen - 1) +
                                    static_cast<long long>(row);
            if (x_col >= 0 && static_cast<std::size_t>(x_col) < xlen) {
                row_freq[col] = ynorm[col] * std::conj(xnorm[static_cast<std::size_t>(x_col)]);
            }
        }

        cuSP::common::fft_inplace(row_freq, true);

        for (std::size_t k = 0; k < nfreq; ++k) {
            const std::size_t shifted_k = (k + nfreq / 2) % nfreq;
            data(row, shifted_k) = static_cast<float>(nfreq * std::abs(row_freq[k]));
        }
    }

    return data;
}

Matrix<Complex> fftshift_axis0(const Matrix<Complex>& x) {
    Matrix<Complex> shifted(x.rows, x.cols, Complex(0.0, 0.0));
    const std::size_t shift = x.rows / 2;
    for (std::size_t i = 0; i < x.rows; ++i) {
        const std::size_t src = (i + shift) % x.rows;
        for (std::size_t j = 0; j < x.cols; ++j) {
            shifted(i, j) = x(src, j);
        }
    }
    return shifted;
}

Matrix<float> abs_square(const Matrix<Complex>& x) {
    Matrix<float> out(x.rows, x.cols, 0.0f);
    for (std::size_t i = 0; i < x.rows; ++i) {
        for (std::size_t j = 0; j < x.cols; ++j) {
            out(i, j) = static_cast<float>(std::norm(x(i, j)));
        }
    }
    return out;
}

}  // namespace cuSP::radartools
