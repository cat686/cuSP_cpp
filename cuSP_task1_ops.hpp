#pragma once

#include "cuSP_common.hpp"
#include "convolution/convolution.h"
#include "radartools/radartools.h"

#include <cstdint>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace cuSP::task1_ops {

using cuSP::common::Complex;

inline std::vector<Complex> make_lfm_pulse(double fs, double Tp, double B) {
    return cuSP::radartools::make_lfm_pulse(fs, Tp, B);
}

template <template <typename> class MatrixLike, typename TargetLike>
inline MatrixLike<Complex> simulate_echo_matrix(const std::vector<Complex>& templ,
                                                double fs,
                                                double PRI,
                                                std::size_t num_pulses,
                                                double fc,
                                                const std::vector<TargetLike>& targets,
                                                double snr_db,
                                                std::mt19937& rng) {
    std::vector<cuSP::radartools::Target> lib_targets;
    lib_targets.reserve(targets.size());
    for (const auto& target : targets) {
        lib_targets.push_back(cuSP::radartools::Target{target.R, target.v, target.amp});
    }
    const auto lib_matrix = cuSP::radartools::simulate_echo_matrix(
        templ, fs, PRI, num_pulses, fc, lib_targets, snr_db, rng);
    return cuSP::common::from_common_matrix<MatrixLike>(lib_matrix);
}

inline std::vector<double> build_range_axis(double fs, std::size_t nfft) {
    return cuSP::radartools::build_range_axis(fs, nfft);
}

inline std::vector<double> build_velocity_axis(double fc, double PRI, std::size_t nfft_doppler) {
    return cuSP::radartools::build_velocity_axis(fc, PRI, nfft_doppler);
}

template <template <typename> class MatrixLike>
inline MatrixLike<Complex> pulse_compression_cpu(const MatrixLike<Complex>& x,
                                                 const std::vector<Complex>& templ,
                                                 bool normalize = false,
                                                 const std::string& window = "",
                                                 std::size_t nfft = 0) {
    const auto lib_x = cuSP::common::to_common_matrix(x);
    const auto lib_out = cuSP::radartools::pulse_compression_cpu(lib_x, templ, normalize, window, nfft);
    return cuSP::common::from_common_matrix<MatrixLike>(lib_out);
}

template <template <typename> class MatrixLike>
inline MatrixLike<Complex> pulse_doppler_cpu(const MatrixLike<Complex>& x,
                                             const std::string& window = "",
                                             std::size_t nfft = 0) {
    const auto lib_x = cuSP::common::to_common_matrix(x);
    const auto lib_out = cuSP::radartools::pulse_doppler_cpu(lib_x, window, nfft);
    return cuSP::common::from_common_matrix<MatrixLike>(lib_out);
}

inline double cfar_alpha_cpu(double pfa, int N) {
    return cuSP::radartools::cfar_alpha_cpu(pfa, N);
}

template <template <typename> class MatrixLike>
inline MatrixLike<double> integral_image_2d(const MatrixLike<float>& array) {
    const auto lib_array = cuSP::common::to_common_matrix(array);
    const auto lib_out = cuSP::radartools::integral_image_2d(lib_array);
    return cuSP::common::from_common_matrix<MatrixLike>(lib_out);
}

template <template <typename> class MatrixLike>
inline double rect_sum_2d(const MatrixLike<double>& integral,
                          int x1,
                          int y1,
                          int x2,
                          int y2) {
    const auto lib_integral = cuSP::common::to_common_matrix(integral);
    return cuSP::radartools::rect_sum_2d(lib_integral, x1, y1, x2, y2);
}

template <template <typename> class MatrixLike>
inline std::pair<MatrixLike<float>, MatrixLike<std::uint8_t>> ca_cfar_cpu(
    const MatrixLike<float>& array,
    std::pair<int, int> guard_cells,
    std::pair<int, int> reference_cells,
    double pfa = 1e-3) {
    const auto lib_array = cuSP::common::to_common_matrix(array);
    const auto lib_out = cuSP::radartools::ca_cfar_cpu(lib_array, guard_cells, reference_cells, pfa);
    return {
        cuSP::common::from_common_matrix<MatrixLike>(lib_out.first),
        cuSP::common::from_common_matrix<MatrixLike>(lib_out.second),
    };
}

template <template <typename> class MatrixLike>
inline MatrixLike<float> ambgfun_2d(const std::vector<Complex>& x,
                                    double fs,
                                    double prf,
                                    const std::vector<Complex>* y = nullptr,
                                    const std::string& cut = "2d",
                                    double cut_value = 0.0) {
    const auto lib_out = cuSP::radartools::ambgfun_2d(x, fs, prf, y, cut, cut_value);
    return cuSP::common::from_common_matrix<MatrixLike>(lib_out);
}

template <template <typename> class MatrixLike>
inline MatrixLike<Complex> fftshift_axis0(const MatrixLike<Complex>& x) {
    const auto lib_x = cuSP::common::to_common_matrix(x);
    const auto lib_out = cuSP::radartools::fftshift_axis0(lib_x);
    return cuSP::common::from_common_matrix<MatrixLike>(lib_out);
}

template <template <typename> class MatrixLike>
inline MatrixLike<float> abs_square(const MatrixLike<Complex>& x) {
    const auto lib_x = cuSP::common::to_common_matrix(x);
    const auto lib_out = cuSP::radartools::abs_square(lib_x);
    return cuSP::common::from_common_matrix<MatrixLike>(lib_out);
}

inline std::vector<Complex> correlate_full(const std::vector<Complex>& a, const std::vector<Complex>& b) {
    return cuSP::convolution::correlate_full(a, b);
}

}  // namespace cuSP::task1_ops
