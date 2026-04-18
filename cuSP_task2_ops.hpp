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
#include <utility>
#include <vector>

namespace cuSP::task2_ops {

using cuSP::common::Complex;

inline double db10(double x, double eps = 1e-12) {
    return cuSP::common::db10(x, eps);
}

inline int ensure_odd(int n) {
    return cuSP::filtering::ensure_odd(n);
}

inline std::vector<double> build_time_axis(double fs, double duration) {
    return cuSP::utils::build_time_axis(fs, duration);
}

inline std::vector<Complex> apply_integer_delay(const std::vector<Complex>& x, double fs, double delay_s) {
    return cuSP::utils::apply_integer_delay(x, fs, delay_s);
}

inline std::vector<double> add_awgn(const std::vector<double>& x, double snr_db, std::uint32_t seed) {
    return cuSP::utils::add_awgn(x, snr_db, seed);
}

inline std::pair<double, double> linear_fit(const std::vector<double>& x, const std::vector<double>& y) {
    return cuSP::estimation::linear_fit(x, y);
}

inline std::pair<int, int> longest_true_segment(const std::vector<std::uint8_t>& mask) {
    return cuSP::estimation::longest_true_segment(mask);
}

template <typename ParamsLike>
inline std::vector<Complex> generate_lfm_component(const std::vector<double>& t, const ParamsLike& params) {
    return cuSP::waveforms::generate_lfm_component(
        t, params.lfm_start, params.lfm_duration, params.lfm_f0, params.lfm_f1, params.lfm_amp);
}

template <typename ParamsLike>
inline std::vector<Complex> generate_gaussian_component(const std::vector<double>& t, const ParamsLike& params) {
    return cuSP::waveforms::generate_gaussian_component(
        t, params.gp_early_center_s, params.lfm_start, params.gp_offset_from_lfm_start_s,
        params.gp_fc, params.gp_bw, params.gp_amp);
}

template <typename ParamsLike>
inline std::vector<double> generate_square_interference(const std::vector<double>& t, const ParamsLike& params) {
    return cuSP::waveforms::generate_square_interference(
        t, params.interf_square_freq, params.interf_square_duty, params.interf_square_amp);
}

template <typename ParamsLike>
inline void synthesize_received_signal(const std::vector<double>& t,
                                       const ParamsLike& params,
                                       std::vector<Complex>& target_clean,
                                       std::vector<Complex>& echo_complex,
                                       std::vector<double>& rx_noiseless,
                                       std::vector<double>& rx) {
    cuSP::waveforms::synthesize_received_signal(
        t,
        params.fs,
        params.lfm_start,
        params.lfm_duration,
        params.lfm_f0,
        params.lfm_f1,
        params.lfm_amp,
        params.gp_early_center_s,
        params.gp_offset_from_lfm_start_s,
        params.gp_fc,
        params.gp_bw,
        params.gp_amp,
        params.echo_delay_s,
        params.echo_freq_offset_hz,
        params.echo_atten,
        params.interf_square_freq,
        params.interf_square_duty,
        params.interf_square_amp,
        params.dc_offset,
        params.snr_db,
        params.seed,
        target_clean,
        echo_complex,
        rx_noiseless,
        rx);
}

template <typename ParamsLike>
inline void preprocess_signal(const std::vector<double>& rx,
                              const ParamsLike& params,
                              std::vector<double>& detrended,
                              std::vector<double>& filtered) {
    cuSP::filtering::preprocess_signal(
        rx,
        params.fir_numtaps,
        params.fir_band_low_hz,
        params.fir_band_high_hz,
        params.fs,
        params.filtfilt_padtype,
        detrended,
        filtered);
}

template <typename ParamsLike>
inline std::vector<double> smooth_with_cubic_bspline(const std::vector<double>& x, const ParamsLike& params) {
    return cuSP::bsplines::smooth_with_cubic_bspline(x, params.spline_kernel_size);
}

template <typename ParamsLike>
inline void estimate_activity_interval(const std::vector<double>& x_smooth,
                                       const ParamsLike& params,
                                       std::vector<double>& envelope_smooth,
                                       std::vector<std::uint8_t>& activity_mask,
                                       int& start_idx,
                                       int& end_idx) {
    cuSP::estimation::estimate_activity_interval(
        x_smooth,
        params.spline_kernel_size,
        params.activity_threshold_ratio,
        params.activity_min_duration_s,
        params.fs,
        envelope_smooth,
        activity_mask,
        start_idx,
        end_idx);
}

template <typename ParamsLike>
inline void extract_modulation_domain(const std::vector<double>& x_smooth,
                                      const std::vector<double>& t,
                                      const ParamsLike& params,
                                      std::vector<Complex>& analytic,
                                      std::vector<double>& t_inst,
                                      std::vector<double>& inst_freq) {
    cuSP::demod::extract_modulation_domain(x_smooth, t, params.fs, analytic, t_inst, inst_freq);
}

template <typename ParamsLike>
inline void extract_time_domain(const std::vector<double>& x_smooth,
                                const ParamsLike& params,
                                std::vector<double>& lags_pos,
                                std::vector<double>& autocorr_pos) {
    cuSP::convolution::extract_time_domain(x_smooth, params.fs, lags_pos, autocorr_pos);
}

template <typename ParamsLike>
inline void extract_frequency_domain(const std::vector<double>& x_smooth,
                                     const ParamsLike& params,
                                     std::vector<double>& freqs_pos,
                                     std::vector<double>& psd_pos) {
    cuSP::spectral_analysis::extract_frequency_domain(
        x_smooth, params.fs, params.csd_nperseg, params.csd_noverlap, freqs_pos, psd_pos);
}

template <template <typename> class MatrixLike, typename ParamsLike>
inline void extract_time_frequency_domain(const std::vector<double>& x_smooth,
                                          const ParamsLike& params,
                                          std::vector<int>& widths,
                                          std::vector<double>& pseudo_freqs,
                                          MatrixLike<float>& cwt_power,
                                          std::vector<double>& ridge_freq,
                                          std::vector<double>& ridge_energy) {
    cuSP::common::Matrix<float> lib_cwt_power;
    cuSP::wavelets::extract_time_frequency_domain(
        x_smooth,
        params.fs,
        params.cwt_w,
        params.cwt_width_min,
        params.cwt_width_max,
        widths,
        pseudo_freqs,
        lib_cwt_power,
        ridge_freq,
        ridge_energy);
    cwt_power = cuSP::common::from_common_matrix<MatrixLike>(lib_cwt_power);
}

inline std::vector<int> detect_peaks_1d(const std::vector<double>& x, int order, double threshold_ratio = 0.0) {
    return cuSP::peak_finding::detect_peaks_1d(x, order, threshold_ratio);
}

inline std::vector<int> fallback_global_peak(const std::vector<double>& x) {
    return cuSP::peak_finding::fallback_global_peak(x);
}

template <typename ParamsLike>
inline std::vector<double> kalman_smooth_1d(const std::vector<double>& measurements, const ParamsLike& params) {
    return cuSP::estimation::kalman_smooth_1d(
        measurements, params.kalman_dt, params.kalman_process_var, params.kalman_meas_var);
}

}  // namespace cuSP::task2_ops
