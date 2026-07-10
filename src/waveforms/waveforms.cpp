#include "waveforms/waveforms.h"

#include "utils/utils.h"

#include <cmath>

namespace cuSP::waveforms {

namespace {

Complex gaussian_pulse_sample(double tau, double fc, double bw) {
    const double ref = std::pow(10.0, -6.0 / 20.0);
    const double a = -std::pow(cuSP::common::kPi * fc * bw, 2.0) / (4.0 * std::log(ref));
    const double env = std::exp(-a * tau * tau);
    const double phase = 2.0 * cuSP::common::kPi * fc * tau;
    return env * Complex(std::cos(phase), std::sin(phase));
}

}  // namespace

std::vector<Complex> generate_lfm_component(const std::vector<double>& t,
                                            double lfm_start,
                                            double lfm_duration,
                                            double lfm_f0,
                                            double lfm_f1,
                                            double lfm_amp) {
    const double stop = lfm_start + lfm_duration;
    const double beta = (lfm_f1 - lfm_f0) / lfm_duration;

    std::vector<Complex> out(t.size(), Complex{});
    for (std::size_t i = 0; i < t.size(); ++i) {
        if (t[i] >= lfm_start && t[i] < stop) {
            const double tau = t[i] - lfm_start;
            const double phase =
                2.0 * cuSP::common::kPi * (lfm_f0 * tau + 0.5 * beta * tau * tau);
            out[i] = lfm_amp * Complex(std::cos(phase), std::sin(phase));
        }
    }
    return out;
}

std::vector<Complex> generate_gaussian_component(const std::vector<double>& t,
                                                 double gp_early_center_s,
                                                 double lfm_start,
                                                 double gp_offset_from_lfm_start_s,
                                                 double gp_fc,
                                                 double gp_bw,
                                                 double gp_amp) {
    const double gp_center_lfm = lfm_start + gp_offset_from_lfm_start_s;

    std::vector<Complex> gp(t.size(), Complex{});
    for (std::size_t i = 0; i < t.size(); ++i) {
        const Complex early = gaussian_pulse_sample(t[i] - gp_early_center_s, gp_fc, gp_bw);
        const Complex lfm = gaussian_pulse_sample(t[i] - gp_center_lfm, gp_fc, gp_bw);
        gp[i] = gp_amp * (early + lfm);
    }
    return gp;
}

std::vector<double> generate_square_interference(const std::vector<double>& t,
                                                 double interf_square_freq,
                                                 double interf_square_duty,
                                                 double interf_square_amp) {
    std::vector<double> out(t.size(), 0.0);
    const double threshold = 2.0 * cuSP::common::kPi * interf_square_duty;
    const double two_pi = 2.0 * cuSP::common::kPi;
    for (std::size_t i = 0; i < t.size(); ++i) {
        const double phase = std::fmod(two_pi * interf_square_freq * t[i], two_pi);
        const double wrapped = (phase < 0.0) ? (phase + two_pi) : phase;
        out[i] = interf_square_amp * ((wrapped < threshold) ? 1.0 : -1.0);
    }
    return out;
}

void synthesize_received_signal(const std::vector<double>& t,
                                double fs,
                                double lfm_start,
                                double lfm_duration,
                                double lfm_f0,
                                double lfm_f1,
                                double lfm_amp,
                                double gp_early_center_s,
                                double gp_offset_from_lfm_start_s,
                                double gp_fc,
                                double gp_bw,
                                double gp_amp,
                                double echo_delay_s,
                                double echo_freq_offset_hz,
                                double echo_atten,
                                double interf_square_freq,
                                double interf_square_duty,
                                double interf_square_amp,
                                double dc_offset,
                                double snr_db,
                                std::uint32_t seed,
                                std::vector<Complex>& target_clean,
                                std::vector<Complex>& echo_complex,
                                std::vector<double>& rx_noiseless,
                                std::vector<double>& rx) {
    const std::vector<Complex> lfm =
        generate_lfm_component(t, lfm_start, lfm_duration, lfm_f0, lfm_f1, lfm_amp);
    const std::vector<Complex> gp =
        generate_gaussian_component(t, gp_early_center_s, lfm_start, gp_offset_from_lfm_start_s,
                                    gp_fc, gp_bw, gp_amp);

    target_clean.assign(t.size(), Complex{});
    for (std::size_t i = 0; i < t.size(); ++i) {
        target_clean[i] = lfm[i] + gp[i];
    }

    const std::vector<Complex> delayed = cuSP::utils::apply_integer_delay(target_clean, fs, echo_delay_s);
    echo_complex.assign(t.size(), Complex{});
    for (std::size_t i = 0; i < t.size(); ++i) {
        const double phase = 2.0 * cuSP::common::kPi * echo_freq_offset_hz * t[i];
        const Complex freq_shift(std::cos(phase), std::sin(phase));
        echo_complex[i] = echo_atten * delayed[i] * freq_shift;
    }

    const std::vector<double> interference =
        generate_square_interference(t, interf_square_freq, interf_square_duty, interf_square_amp);
    rx_noiseless.assign(t.size(), 0.0);
    for (std::size_t i = 0; i < t.size(); ++i) {
        rx_noiseless[i] = echo_complex[i].real() + interference[i] + dc_offset;
    }
    rx = cuSP::utils::add_awgn(rx_noiseless, snr_db, seed);
}

}  // namespace cuSP::waveforms
