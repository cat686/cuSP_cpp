#pragma once

#include "cuSP_common.hpp"
#include "convolution/convolution.h"
#include "radartools/radartools.h"

#include <cstdint>
#include <complex>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace cuSP::task1_ops {

using ComplexD = cuSP::common::Complex;

template <typename Real>
using ComplexOf = std::complex<Real>;

template <typename Real>
inline std::vector<ComplexOf<Real>> make_lfm_pulse(Real fs, Real Tp, Real B) {
    const auto out = cuSP::radartools::make_lfm_pulse(
        static_cast<double>(fs),
        static_cast<double>(Tp),
        static_cast<double>(B));
    return cuSP::common::cast_vector<ComplexOf<Real>>(out);
}

template <template <typename> class MatrixLike, typename TargetLike, typename ComplexT>
inline MatrixLike<ComplexT> simulate_echo_matrix(const std::vector<ComplexT>& templ,
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
        cuSP::common::cast_vector<ComplexD>(templ),
        fs,
        PRI,
        num_pulses,
        fc,
        lib_targets,
        snr_db,
        rng);
    return cuSP::common::from_common_matrix_cast<MatrixLike, ComplexT>(lib_matrix);
}

template <typename Real>
inline std::vector<Real> build_range_axis(Real fs, std::size_t nfft) {
    return cuSP::common::cast_vector<Real>(
        cuSP::radartools::build_range_axis(static_cast<double>(fs), nfft));
}

template <typename Real>
inline std::vector<Real> build_velocity_axis(Real fc, Real PRI, std::size_t nfft_doppler) {
    return cuSP::common::cast_vector<Real>(
        cuSP::radartools::build_velocity_axis(
            static_cast<double>(fc),
            static_cast<double>(PRI),
            nfft_doppler));
}

template <template <typename> class MatrixLike, typename ComplexT>
inline MatrixLike<ComplexT> pulse_compression_cpu(const MatrixLike<ComplexT>& x,
                                                  const std::vector<ComplexT>& templ,
                                                  bool normalize = false,
                                                  const std::string& window = "",
                                                  std::size_t nfft = 0) {
    const auto lib_x = cuSP::common::to_common_matrix_cast<ComplexD>(x);
    const auto lib_templ = cuSP::common::cast_vector<ComplexD>(templ);
    const auto lib_out = cuSP::radartools::pulse_compression_cpu(
        lib_x, lib_templ, normalize, window, nfft);
    return cuSP::common::from_common_matrix_cast<MatrixLike, ComplexT>(lib_out);
}

template <template <typename> class MatrixLike, typename ComplexT>
inline MatrixLike<ComplexT> pulse_doppler_cpu(const MatrixLike<ComplexT>& x,
                                              const std::string& window = "",
                                              std::size_t nfft = 0) {
    const auto lib_x = cuSP::common::to_common_matrix_cast<ComplexD>(x);
    const auto lib_out = cuSP::radartools::pulse_doppler_cpu(lib_x, window, nfft);
    return cuSP::common::from_common_matrix_cast<MatrixLike, ComplexT>(lib_out);
}

template <typename Real>
inline Real cfar_alpha_cpu(Real pfa, int N) {
    return static_cast<Real>(cuSP::radartools::cfar_alpha_cpu(static_cast<double>(pfa), N));
}

template <template <typename> class MatrixLike>
inline MatrixLike<float> integral_image_2d(const MatrixLike<float>& array) {
    const auto lib_array = cuSP::common::to_common_matrix(array);
    const auto lib_out = cuSP::radartools::integral_image_2d(lib_array);
    return cuSP::common::from_common_matrix_cast<MatrixLike, float>(lib_out);
}

template <template <typename> class MatrixLike, typename T>
inline float rect_sum_2d(const MatrixLike<T>& integral,
                         int x1,
                         int y1,
                         int x2,
                         int y2) {
    const auto lib_integral = cuSP::common::to_common_matrix_cast<double>(integral);
    return static_cast<float>(cuSP::radartools::rect_sum_2d(lib_integral, x1, y1, x2, y2));
}

template <template <typename> class MatrixLike>
inline std::pair<MatrixLike<float>, MatrixLike<std::uint8_t>> ca_cfar_cpu(
    const MatrixLike<float>& array,
    std::pair<int, int> guard_cells,
    std::pair<int, int> reference_cells,
    float pfa = 1e-3f) {
    const auto lib_array = cuSP::common::to_common_matrix(array);
    const auto lib_out = cuSP::radartools::ca_cfar_cpu(
        lib_array, guard_cells, reference_cells, static_cast<double>(pfa));
    return {
        cuSP::common::from_common_matrix<MatrixLike>(lib_out.first),
        cuSP::common::from_common_matrix<MatrixLike>(lib_out.second),
    };
}

template <template <typename> class MatrixLike, typename ComplexT>
inline MatrixLike<float> ambgfun_2d(const std::vector<ComplexT>& x,
                                    double fs,
                                    double prf,
                                    const std::vector<ComplexT>* y = nullptr,
                                    const std::string& cut = "2d",
                                    double cut_value = 0.0) {
    const auto lib_x = cuSP::common::cast_vector<ComplexD>(x);
    std::vector<ComplexD> lib_y_storage;
    const std::vector<ComplexD>* lib_y = nullptr;
    if (y != nullptr) {
        lib_y_storage = cuSP::common::cast_vector<ComplexD>(*y);
        lib_y = &lib_y_storage;
    }
    const auto lib_out = cuSP::radartools::ambgfun_2d(
        lib_x, fs, prf, lib_y, cut, cut_value);
    return cuSP::common::from_common_matrix<MatrixLike>(lib_out);
}

template <template <typename> class MatrixLike, typename ComplexT>
inline MatrixLike<ComplexT> fftshift_axis0(const MatrixLike<ComplexT>& x) {
    const auto lib_x = cuSP::common::to_common_matrix_cast<ComplexD>(x);
    const auto lib_out = cuSP::radartools::fftshift_axis0(lib_x);
    return cuSP::common::from_common_matrix_cast<MatrixLike, ComplexT>(lib_out);
}

template <template <typename> class MatrixLike, typename ComplexT>
inline MatrixLike<float> abs_square(const MatrixLike<ComplexT>& x) {
    const auto lib_x = cuSP::common::to_common_matrix_cast<ComplexD>(x);
    const auto lib_out = cuSP::radartools::abs_square(lib_x);
    return cuSP::common::from_common_matrix<MatrixLike>(lib_out);
}

template <typename ComplexT>
inline std::vector<ComplexT> correlate_full(const std::vector<ComplexT>& a,
                                            const std::vector<ComplexT>& b) {
    const auto out = cuSP::convolution::correlate_full(
        cuSP::common::cast_vector<ComplexD>(a),
        cuSP::common::cast_vector<ComplexD>(b));
    return cuSP::common::cast_vector<ComplexT>(out);
}

}  // namespace cuSP::task1_ops
