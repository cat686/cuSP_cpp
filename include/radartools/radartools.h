#pragma once

#include "common/common.h"

#include <cstdint>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace cuSP::radartools {

using cuSP::common::Complex;
using cuSP::common::Matrix;

struct Target {
    double R = 0.0;
    double v = 0.0;
    double amp = 0.0;
};

struct AmbiguityImage {
    Matrix<float> data;
    std::size_t rows = 0;
    std::size_t cols = 0;
};

std::vector<Complex> make_lfm_pulse(double fs, double Tp, double B);
Matrix<Complex> simulate_echo_matrix(const std::vector<Complex>& templ,
                                     double fs,
                                     double PRI,
                                     std::size_t num_pulses,
                                     double fc,
                                     const std::vector<Target>& targets,
                                     double snr_db,
                                     std::mt19937& rng);
std::vector<double> build_range_axis(double fs, std::size_t nfft);
std::vector<double> build_velocity_axis(double fc, double PRI, std::size_t nfft_doppler);
Matrix<Complex> pulse_compression_cpu(const Matrix<Complex>& x,
                                      const std::vector<Complex>& templ,
                                      bool normalize = false,
                                      const std::string& window = "",
                                      std::size_t nfft = 0);
Matrix<Complex> pulse_doppler_cpu(const Matrix<Complex>& x,
                                  const std::string& window = "",
                                  std::size_t nfft = 0);
double cfar_alpha_cpu(double pfa, int N);
Matrix<double> integral_image_2d(const Matrix<float>& array);
double rect_sum_2d(const Matrix<double>& integral, int x1, int y1, int x2, int y2);
std::pair<Matrix<float>, Matrix<std::uint8_t>> ca_cfar_cpu(const Matrix<float>& array,
                                                           std::pair<int, int> guard_cells,
                                                           std::pair<int, int> reference_cells,
                                                           double pfa = 1e-3);
Matrix<float> ambgfun_2d(const std::vector<Complex>& x,
                         double fs,
                         double prf,
                         const std::vector<Complex>* y = nullptr,
                         const std::string& cut = "2d",
                         double cut_value = 0.0);
Matrix<Complex> fftshift_axis0(const Matrix<Complex>& x);
Matrix<float> abs_square(const Matrix<Complex>& x);

}  // namespace cuSP::radartools
