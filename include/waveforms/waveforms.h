#pragma once

#include "common/common.h"

#include <cstdint>
#include <vector>

namespace cuSP::waveforms {

using cuSP::common::Complex;

std::vector<Complex> generate_lfm_component(const std::vector<double>& t,
                                            double lfm_start,
                                            double lfm_duration,
                                            double lfm_f0,
                                            double lfm_f1,
                                            double lfm_amp);
std::vector<Complex> generate_gaussian_component(const std::vector<double>& t,
                                                 double gp_early_center_s,
                                                 double lfm_start,
                                                 double gp_offset_from_lfm_start_s,
                                                 double gp_fc,
                                                 double gp_bw,
                                                 double gp_amp);
std::vector<double> generate_square_interference(const std::vector<double>& t,
                                                 double interf_square_freq,
                                                 double interf_square_duty,
                                                 double interf_square_amp);
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
                                std::vector<double>& rx);

}  // namespace cuSP::waveforms
