#pragma once

#include "cuSP_common.hpp"
#include "bsplines/bsplines.h"
#include "convolution/convolution.h"
#include "demod/demod.h"
#include "estimation/estimation.h"
#include "filtering/filtering.h"
#include "peak_finding/peak_finding.h"
#include "spectral_analysis/spectral_analysis.h"
#include "utils/utils.h"
#include "waveforms/waveforms.h"
#include "wavelets/wavelets.h"

#include <cstdint>
#include <complex>
#include <utility>
#include <vector>

namespace cuSP::task2_ops {

using ComplexD = cuSP::common::Complex;
using ComplexF = std::complex<float>;

inline float db10(float x, float eps = 1e-12f) {
    return static_cast<float>(cuSP::common::db10(x, eps));
}

inline int ensure_odd(int n) {
    return cuSP::filtering::ensure_odd(n);
}

inline std::vector<float> build_time_axis(float fs, float duration) {
    return cuSP::common::cast_vector<float>(
        cuSP::utils::build_time_axis(static_cast<double>(fs), static_cast<double>(duration)));
}

inline std::vector<ComplexF> apply_integer_delay(const std::vector<ComplexF>& x,
                                                 float fs,
                                                 float delay_s) {
    const auto out = cuSP::utils::apply_integer_delay(
        cuSP::common::cast_vector<ComplexD>(x),
        static_cast<double>(fs),
        static_cast<double>(delay_s));
    return cuSP::common::cast_vector<ComplexF>(out);
}

inline std::vector<float> add_awgn(const std::vector<float>& x,
                                   float snr_db,
                                   std::uint32_t seed) {
    const auto out = cuSP::utils::add_awgn(
        cuSP::common::cast_vector<double>(x),
        static_cast<double>(snr_db),
        seed);
    return cuSP::common::cast_vector<float>(out);
}

inline std::pair<float, float> linear_fit(const std::vector<float>& x,
                                          const std::vector<float>& y) {
    const auto result = cuSP::estimation::linear_fit(
        cuSP::common::cast_vector<double>(x),
        cuSP::common::cast_vector<double>(y));
    return {static_cast<float>(result.first), static_cast<float>(result.second)};
}

inline std::pair<int, int> longest_true_segment(const std::vector<std::uint8_t>& mask) {
    return cuSP::estimation::longest_true_segment(mask);
}

template <typename ParamsLike>
inline std::vector<ComplexF> generate_lfm_component(const std::vector<float>& t,
                                                    const ParamsLike& params) {
    const auto out = cuSP::waveforms::generate_lfm_component(
        cuSP::common::cast_vector<double>(t),
        static_cast<double>(params.lfm_start),
        static_cast<double>(params.lfm_duration),
        static_cast<double>(params.lfm_f0),
        static_cast<double>(params.lfm_f1),
        static_cast<double>(params.lfm_amp));
    return cuSP::common::cast_vector<ComplexF>(out);
}

template <typename ParamsLike>
inline std::vector<ComplexF> generate_gaussian_component(const std::vector<float>& t,
                                                         const ParamsLike& params) {
    const auto out = cuSP::waveforms::generate_gaussian_component(
        cuSP::common::cast_vector<double>(t),
        static_cast<double>(params.gp_early_center_s),
        static_cast<double>(params.lfm_start),
        static_cast<double>(params.gp_offset_from_lfm_start_s),
        static_cast<double>(params.gp_fc),
        static_cast<double>(params.gp_bw),
        static_cast<double>(params.gp_amp));
    return cuSP::common::cast_vector<ComplexF>(out);
}

template <typename ParamsLike>
inline std::vector<float> generate_square_interference(const std::vector<float>& t,
                                                       const ParamsLike& params) {
    const auto out = cuSP::waveforms::generate_square_interference(
        cuSP::common::cast_vector<double>(t),
        static_cast<double>(params.interf_square_freq),
        static_cast<double>(params.interf_square_duty),
        static_cast<double>(params.interf_square_amp));
    return cuSP::common::cast_vector<float>(out);
}

template <typename ParamsLike>
inline void synthesize_received_signal(const std::vector<float>& t,
                                       const ParamsLike& params,
                                       std::vector<ComplexF>& target_clean,
                                       std::vector<ComplexF>& echo_complex,
                                       std::vector<float>& rx_noiseless,
                                       std::vector<float>& rx) {
    std::vector<ComplexD> target_clean_d;
    std::vector<ComplexD> echo_complex_d;
    std::vector<double> rx_noiseless_d;
    std::vector<double> rx_d;

    cuSP::waveforms::synthesize_received_signal(
        cuSP::common::cast_vector<double>(t),
        static_cast<double>(params.fs),
        static_cast<double>(params.lfm_start),
        static_cast<double>(params.lfm_duration),
        static_cast<double>(params.lfm_f0),
        static_cast<double>(params.lfm_f1),
        static_cast<double>(params.lfm_amp),
        static_cast<double>(params.gp_early_center_s),
        static_cast<double>(params.gp_offset_from_lfm_start_s),
        static_cast<double>(params.gp_fc),
        static_cast<double>(params.gp_bw),
        static_cast<double>(params.gp_amp),
        static_cast<double>(params.echo_delay_s),
        static_cast<double>(params.echo_freq_offset_hz),
        static_cast<double>(params.echo_atten),
        static_cast<double>(params.interf_square_freq),
        static_cast<double>(params.interf_square_duty),
        static_cast<double>(params.interf_square_amp),
        static_cast<double>(params.dc_offset),
        static_cast<double>(params.snr_db),
        params.seed,
        target_clean_d,
        echo_complex_d,
        rx_noiseless_d,
        rx_d);

    target_clean = cuSP::common::cast_vector<ComplexF>(target_clean_d);
    echo_complex = cuSP::common::cast_vector<ComplexF>(echo_complex_d);
    rx_noiseless = cuSP::common::cast_vector<float>(rx_noiseless_d);
    rx = cuSP::common::cast_vector<float>(rx_d);
}

template <typename ParamsLike>
inline void preprocess_signal(const std::vector<float>& rx,
                              const ParamsLike& params,
                              std::vector<float>& detrended,
                              std::vector<float>& filtered) {
    std::vector<double> detrended_d;
    std::vector<double> filtered_d;
    cuSP::filtering::preprocess_signal(
        cuSP::common::cast_vector<double>(rx),
        params.fir_numtaps,
        static_cast<double>(params.fir_band_low_hz),
        static_cast<double>(params.fir_band_high_hz),
        static_cast<double>(params.fs),
        params.filtfilt_padtype,
        detrended_d,
        filtered_d);
    detrended = cuSP::common::cast_vector<float>(detrended_d);
    filtered = cuSP::common::cast_vector<float>(filtered_d);
}

template <typename ParamsLike>
inline std::vector<float> smooth_with_cubic_bspline(const std::vector<float>& x,
                                                    const ParamsLike& params) {
    const auto out = cuSP::bsplines::smooth_with_cubic_bspline(
        cuSP::common::cast_vector<double>(x),
        params.spline_kernel_size);
    return cuSP::common::cast_vector<float>(out);
}

template <typename ParamsLike>
inline void estimate_activity_interval(const std::vector<float>& x_smooth,
                                       const ParamsLike& params,
                                       std::vector<float>& envelope_smooth,
                                       std::vector<std::uint8_t>& activity_mask,
                                       int& start_idx,
                                       int& end_idx) {
    std::vector<double> envelope_d;
    cuSP::estimation::estimate_activity_interval(
        cuSP::common::cast_vector<double>(x_smooth),
        params.spline_kernel_size,
        static_cast<double>(params.activity_threshold_ratio),
        static_cast<double>(params.activity_min_duration_s),
        static_cast<double>(params.fs),
        envelope_d,
        activity_mask,
        start_idx,
        end_idx);
    envelope_smooth = cuSP::common::cast_vector<float>(envelope_d);
}

template <typename ParamsLike>
inline void extract_modulation_domain(const std::vector<float>& x_smooth,
                                      const std::vector<float>& t,
                                      const ParamsLike& params,
                                      std::vector<ComplexF>& analytic,
                                      std::vector<float>& t_inst,
                                      std::vector<float>& inst_freq) {
    std::vector<ComplexD> analytic_d;
    std::vector<double> t_inst_d;
    std::vector<double> inst_freq_d;
    cuSP::demod::extract_modulation_domain(
        cuSP::common::cast_vector<double>(x_smooth),
        cuSP::common::cast_vector<double>(t),
        static_cast<double>(params.fs),
        analytic_d,
        t_inst_d,
        inst_freq_d);
    analytic = cuSP::common::cast_vector<ComplexF>(analytic_d);
    t_inst = cuSP::common::cast_vector<float>(t_inst_d);
    inst_freq = cuSP::common::cast_vector<float>(inst_freq_d);
}

template <typename ParamsLike>
inline void extract_time_domain(const std::vector<float>& x_smooth,
                                const ParamsLike& params,
                                std::vector<float>& lags_pos,
                                std::vector<float>& autocorr_pos) {
    std::vector<double> lags_d;
    std::vector<double> autocorr_d;
    cuSP::convolution::extract_time_domain(
        cuSP::common::cast_vector<double>(x_smooth),
        static_cast<double>(params.fs),
        lags_d,
        autocorr_d);
    lags_pos = cuSP::common::cast_vector<float>(lags_d);
    autocorr_pos = cuSP::common::cast_vector<float>(autocorr_d);
}

template <typename ParamsLike>
inline void extract_frequency_domain(const std::vector<float>& x_smooth,
                                     const ParamsLike& params,
                                     std::vector<float>& freqs_pos,
                                     std::vector<float>& psd_pos) {
    std::vector<double> freqs_d;
    std::vector<double> psd_d;
    cuSP::spectral_analysis::extract_frequency_domain(
        cuSP::common::cast_vector<double>(x_smooth),
        static_cast<double>(params.fs),
        params.csd_nperseg,
        params.csd_noverlap,
        freqs_d,
        psd_d);
    freqs_pos = cuSP::common::cast_vector<float>(freqs_d);
    psd_pos = cuSP::common::cast_vector<float>(psd_d);
}

template <template <typename> class MatrixLike, typename ParamsLike>
inline void extract_time_frequency_domain(const std::vector<float>& x_smooth,
                                          const ParamsLike& params,
                                          std::vector<int>& widths,
                                          std::vector<float>& pseudo_freqs,
                                          MatrixLike<float>& cwt_power,
                                          std::vector<float>& ridge_freq,
                                          std::vector<float>& ridge_energy) {
    cuSP::common::Matrix<float> lib_cwt_power;
    std::vector<double> pseudo_freqs_d;
    std::vector<double> ridge_freq_d;
    std::vector<double> ridge_energy_d;

    cuSP::wavelets::extract_time_frequency_domain(
        cuSP::common::cast_vector<double>(x_smooth),
        static_cast<double>(params.fs),
        static_cast<double>(params.cwt_w),
        params.cwt_width_min,
        params.cwt_width_max,
        widths,
        pseudo_freqs_d,
        lib_cwt_power,
        ridge_freq_d,
        ridge_energy_d);

    pseudo_freqs = cuSP::common::cast_vector<float>(pseudo_freqs_d);
    cwt_power = cuSP::common::from_common_matrix<MatrixLike>(lib_cwt_power);
    ridge_freq = cuSP::common::cast_vector<float>(ridge_freq_d);
    ridge_energy = cuSP::common::cast_vector<float>(ridge_energy_d);
}

inline std::vector<int> detect_peaks_1d(const std::vector<float>& x,
                                        int order,
                                        float threshold_ratio = 0.0f) {
    return cuSP::peak_finding::detect_peaks_1d(
        cuSP::common::cast_vector<double>(x),
        order,
        static_cast<double>(threshold_ratio));
}

inline std::vector<int> fallback_global_peak(const std::vector<float>& x) {
    return cuSP::peak_finding::fallback_global_peak(
        cuSP::common::cast_vector<double>(x));
}

template <typename ParamsLike>
inline std::vector<float> kalman_smooth_1d(const std::vector<float>& measurements,
                                           const ParamsLike& params) {
    const auto out = cuSP::estimation::kalman_smooth_1d(
        cuSP::common::cast_vector<double>(measurements),
        static_cast<double>(params.kalman_dt),
        static_cast<double>(params.kalman_process_var),
        static_cast<double>(params.kalman_meas_var));
    return cuSP::common::cast_vector<float>(out);
}

}  // namespace cuSP::task2_ops
