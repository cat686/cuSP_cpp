#pragma once

#include <string>
#include <vector>

namespace cuSP::filtering {

int ensure_odd(int n);
void preprocess_signal(const std::vector<double>& rx,
                       int fir_numtaps,
                       double fir_band_low_hz,
                       double fir_band_high_hz,
                       double fs,
                       const std::string& filtfilt_padtype,
                       std::vector<double>& detrended,
                       std::vector<double>& filtered);

}  // namespace cuSP::filtering
