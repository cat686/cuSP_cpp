// task1.cpp
//
// CPU-only C++ translation of task1.cu.
//
// Pipeline:
// 1) LFM waveform design
// 2) Multi-target echo simulation (delay + Doppler + noise)
// 3) pulse_compression_cpu
// 4) pulse_doppler_cpu
// 5) 2D ca_cfar_cpu
// 6) ambgfun_cpu
// 7) Range / velocity estimation
// 8) Visualization-data export
//
// Notes:
// - Pure CPU implementation, no GPU / CUDA / CuPy / cuSignal usage.
// - The file keeps the same high-level structure and processing order as task1.cu.
// - Plotting is replaced with CSV / image export so the file stays self-contained.
//
#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <type_traits>
#include <utility>
#include <vector>

#include "cuSP_task1_ops.hpp"

namespace {

constexpr float kLightSpeed = 3.0e8;
constexpr float kPi = 3.141592653589793238462643383279502884;

using Complex = std::complex<float>;

template <typename T>
struct Matrix {
    std::size_t rows = 0;
    std::size_t cols = 0;
    std::vector<T> data;

    Matrix() = default;

    Matrix(std::size_t r, std::size_t c, const T& value = T{})
        : rows(r), cols(c), data(r * c, value) {}

    T& operator()(std::size_t r, std::size_t c) {
        return data[r * cols + c];
    }

    const T& operator()(std::size_t r, std::size_t c) const {
        return data[r * cols + c];
    }
};

struct Target {
    float R = 0.0;
    float v = 0.0;
    float amp = 0.0;
};

// =========================
// 1. Parameter config
// =========================
struct Params {
    float fc = 70e9;
    float B = 200e6;
    float Tp = 5e-6;
    float PRI = 50e-6;
    float fs = 500e6;
    std::size_t num_pulses = 64;

    std::vector<Target> targets = {
        {30.0, 10.0, 1.0},
        {300.0, 2.0, 0.8},
    };

    float snr_db = 10.0;

    std::string pc_window = "hamming";
    bool pc_normalize = true;

    std::string doppler_window = "hann";

    std::pair<int, int> guard_cells = {3, 3};
    std::pair<int, int> reference_cells = {16, 16};
    float pfa = 1e-4;

    std::size_t top_k = 2;
};

Params params_for_level(const std::string& level) {
    Params params;
    if (level == "L1") {
        params.Tp = 2.5e-6;
        params.num_pulses = 64;
    } else if (level == "L2") {
        params.Tp = 5e-6;
        params.num_pulses = 64;
    } else if (level == "L3") {
        params.Tp = 15e-6;
        params.num_pulses = 256;
    } else {
        throw std::invalid_argument("Unsupported data level: " + level);
    }
    return params;
}

struct RuntimeOptions {
    bool save_output = false;
    bool enable_visualization = true;
    bool headless = false;
    std::string plot_python;
    std::string data_level = "L2";
};

struct TimingRecord {
    std::string stream_name = "main-serial";
    std::string name;
    std::string category = "kernel";
    float cpu_ms = 0.0;
    float gpu_ms = 0.0;
};

struct Detection {
    int doppler_bin = 0;
    int range_bin = 0;
    float range_m = 0.0;
    float velocity_mps = 0.0;
    float power = 0.0;
};

struct AmbiguityImage {
    Matrix<float> data;
    std::size_t rows = 0;
    std::size_t cols = 0;
};

std::size_t ceil_div(std::size_t a, std::size_t b) {
    return (a + b - 1) / b;
}

template <typename Fn>
auto time_cpu_op(const std::string& name,
                 const std::string& category,
                 std::vector<TimingRecord>& timings,
                 Fn&& fn) {
    using Ret = std::invoke_result_t<Fn&>;
    const auto start = std::chrono::steady_clock::now();

    if constexpr (std::is_void_v<Ret>) {
        fn();
        const auto stop = std::chrono::steady_clock::now();
        timings.push_back(TimingRecord{
            "main-serial",
            name,
            category,
            std::chrono::duration<float, std::milli>(stop - start).count(),
            0.0,
        });
    } else {
        Ret ret = fn();
        const auto stop = std::chrono::steady_clock::now();
        timings.push_back(TimingRecord{
            "main-serial",
            name,
            category,
            std::chrono::duration<float, std::milli>(stop - start).count(),
            0.0,
        });
        return ret;
    }
}

bool is_operator_timing_record(const TimingRecord& timing) {
    return timing.category == "h2d" ||
           timing.category == "kernel" ||
           timing.category == "d2h";
}

struct TimingTotals {
    float gpu_operator_ms = 0.0f;
    float cpu_operator_ms = 0.0f;
    float total_operator_ms = 0.0f;
};

struct StreamTimingTotals {
    std::string stream_name;
    float kernel_gpu_ms = 0.0f;
    float kernel_total_ms = 0.0f;
    float transfer_ms = 0.0f;
    float total_ms = 0.0f;
};

float kernel_total_ms(const TimingRecord& timing) {
    return timing.gpu_ms > 0.0f ? timing.gpu_ms : timing.cpu_ms;
}

TimingTotals compute_timing_totals(const std::vector<TimingRecord>& timings) {
    std::vector<StreamTimingTotals> stream_totals;

    for (const auto& t : timings) {
        if (!is_operator_timing_record(t)) {
            continue;
        }

        auto it = std::find_if(stream_totals.begin(), stream_totals.end(), [&](const auto& entry) {
            return entry.stream_name == t.stream_name;
        });
        if (it == stream_totals.end()) {
            stream_totals.push_back(StreamTimingTotals{t.stream_name});
            it = std::prev(stream_totals.end());
        }

        if (t.category == "kernel") {
            it->kernel_gpu_ms += t.gpu_ms;
            it->kernel_total_ms += kernel_total_ms(t);
            it->total_ms += kernel_total_ms(t);
        } else {
            it->transfer_ms += t.cpu_ms;
            it->total_ms += t.cpu_ms;
        }
    }

    TimingTotals totals;
    for (const auto& entry : stream_totals) {
        totals.gpu_operator_ms = std::max(totals.gpu_operator_ms, entry.kernel_gpu_ms);
        totals.cpu_operator_ms = std::max(totals.cpu_operator_ms, entry.transfer_ms);
        totals.total_operator_ms = std::max(totals.total_operator_ms, entry.total_ms);
    }
    return totals;
}

void print_timing_summary(const std::vector<TimingRecord>& timings) {
    std::cout << "========== 绠楀瓙璁℃椂 ==========\n";
    for (const auto& t : timings) {
        if (!is_operator_timing_record(t)) {
            continue;
        }
        std::cout << std::fixed << std::setprecision(3)
                  << "[" << t.stream_name << "] "
                  << "[" << t.category << "] "
                  << t.name
                  << ": cpu=" << t.cpu_ms << " ms"
                  << ", cudaEvent=" << t.gpu_ms << " ms\n";
    }
    std::cout << "---------- Timing totals ----------\n";
    const TimingTotals totals = compute_timing_totals(timings);
    std::cout << "GPU operator time: " << totals.gpu_operator_ms << " ms\n"
              << "Cpu operator time: " << totals.cpu_operator_ms << " ms\n"
              << "Total operator time: " << totals.total_operator_ms << " ms\n";
    std::cout << '\n';
}

// =========================
// 2. Helper functions
// =========================
float db20(float x, float eps = 1e-12) {
    return cuSP::common::db20(x, eps);
}

float db10(float x, float eps = 1e-12) {
    return cuSP::common::db10(x, eps);
}

std::size_t next_pow2(std::size_t n) {
    return cuSP::common::next_pow2(n);
}

std::vector<float> make_window(const std::string& name, std::size_t n) {
    return cuSP::common::make_window_f32(name, n);
}

void fft_inplace(std::vector<Complex>& a, bool inverse) {
    cuSP::common::fft_inplace(a, inverse);
}

std::vector<Complex> make_lfm_pulse(float fs, float Tp, float B) {
    return cuSP::task1_ops::make_lfm_pulse(fs, Tp, B);
}

Matrix<Complex> simulate_echo_matrix(const std::vector<Complex>& templ,
                                     float fs,
                                     float PRI,
                                     std::size_t num_pulses,
                                     float fc,
                                     const std::vector<Target>& targets,
                                     float snr_db,
                                     std::mt19937& rng) {
    const float lambda = kLightSpeed / fc;
    const std::size_t samples_per_pulse = templ.size();

    Matrix<Complex> x(num_pulses, samples_per_pulse, Complex(0.0, 0.0));

    for (const auto& target : targets) {
        const float tau = 2.0 * target.R / kLightSpeed;
        const float fd = 2.0 * target.v / lambda;
        const int delay_samp = static_cast<int>(std::llround(tau * fs));

        if (delay_samp < 0 || static_cast<std::size_t>(delay_samp) >= samples_per_pulse) {
            continue;
        }

        std::vector<Complex> delayed(samples_per_pulse, Complex(0.0, 0.0));
        const std::size_t valid_len = samples_per_pulse - static_cast<std::size_t>(delay_samp);
        for (std::size_t n = 0; n < valid_len; ++n) {
            delayed[static_cast<std::size_t>(delay_samp) + n] = templ[n];
        }

        for (std::size_t p = 0; p < num_pulses; ++p) {
            const float phase = 2.0 * kPi * fd * static_cast<float>(p) * PRI;
            const Complex slow_phase(std::cos(phase), std::sin(phase));
            for (std::size_t n = 0; n < samples_per_pulse; ++n) {
                x(p, n) += (slow_phase * delayed[n]) * target.amp;
            }
        }
    }

    float signal_power = 0.0;
    for (const auto& value : x.data) {
        signal_power += std::norm(value);
    }
    signal_power /= static_cast<float>(x.data.size());

    const float snr_linear = std::pow(10.0, snr_db / 10.0);
    const float noise_power = signal_power / snr_linear;
    const float noise_sigma = std::sqrt(noise_power / 2.0);

    std::normal_distribution<float> normal(0.0f, 1.0f);
    for (auto& value : x.data) {
        value += Complex(static_cast<float>(noise_sigma) * normal(rng),
                         static_cast<float>(noise_sigma) * normal(rng));
    }

    return x;
}

std::vector<float> build_range_axis(float fs, std::size_t nfft) {
    return cuSP::task1_ops::build_range_axis(fs, nfft);
}

std::vector<float> build_velocity_axis(float fc, float PRI, std::size_t nfft_doppler) {
    return cuSP::task1_ops::build_velocity_axis(fc, PRI, nfft_doppler);
}

std::vector<std::pair<int, int>> nms_2d(const std::vector<std::pair<int, int>>& indices,
                                        const Matrix<float>& metric,
                                        int min_doppler_gap = 1,
                                        int min_range_gap = 2) {
    if (indices.empty()) {
        return {};
    }

    std::vector<std::size_t> order(indices.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](std::size_t lhs, std::size_t rhs) {
        const auto [li, lj] = indices[lhs];
        const auto [ri, rj] = indices[rhs];
        return metric(static_cast<std::size_t>(li), static_cast<std::size_t>(lj)) >
               metric(static_cast<std::size_t>(ri), static_cast<std::size_t>(rj));
    });

    std::vector<std::uint8_t> occupied(metric.rows * metric.cols, 0);
    auto occupied_at = [&](int i, int j) -> std::uint8_t& {
        return occupied[static_cast<std::size_t>(i) * metric.cols + static_cast<std::size_t>(j)];
    };

    std::vector<std::pair<int, int>> selected;
    selected.reserve(indices.size());

    for (const std::size_t idx : order) {
        const auto [i, j] = indices[idx];
        if (occupied_at(i, j)) {
            continue;
        }

        selected.emplace_back(i, j);

        const int i_min = std::max(0, i - min_doppler_gap);
        const int i_max = std::min(static_cast<int>(metric.rows), i + min_doppler_gap + 1);
        const int j_min = std::max(0, j - min_range_gap);
        const int j_max = std::min(static_cast<int>(metric.cols), j + min_range_gap + 1);

        for (int ii = i_min; ii < i_max; ++ii) {
            for (int jj = j_min; jj < j_max; ++jj) {
                occupied_at(ii, jj) = 1;
            }
        }
    }

    return selected;
}

std::vector<Detection> extract_detections(const Matrix<float>& rd_map_power,
                                          const Matrix<std::uint8_t>& det_mask,
                                          const std::vector<float>& range_axis,
                                          const std::vector<float>& velocity_axis,
                                          std::size_t top_k = 10) {
    std::vector<std::pair<int, int>> det_idx;
    for (std::size_t i = 0; i < det_mask.rows; ++i) {
        for (std::size_t j = 0; j < det_mask.cols; ++j) {
            if (det_mask(i, j) != 0) {
                det_idx.emplace_back(static_cast<int>(i), static_cast<int>(j));
            }
        }
    }

    if (det_idx.empty()) {
        return {};
    }

    auto pairs = nms_2d(det_idx, rd_map_power);
    std::sort(pairs.begin(), pairs.end(), [&](const auto& lhs, const auto& rhs) {
        return rd_map_power(static_cast<std::size_t>(lhs.first),
                            static_cast<std::size_t>(lhs.second)) >
               rd_map_power(static_cast<std::size_t>(rhs.first),
                            static_cast<std::size_t>(rhs.second));
    });

    if (pairs.size() > top_k) {
        pairs.resize(top_k);
    }

    std::vector<Detection> results;
    results.reserve(pairs.size());
    for (const auto& [i, j] : pairs) {
        results.push_back(Detection{
            i,
            j,
            range_axis[static_cast<std::size_t>(j)],
            velocity_axis[static_cast<std::size_t>(i)],
            rd_map_power(static_cast<std::size_t>(i), static_cast<std::size_t>(j)),
        });
    }
    return results;
}

void print_target_summary(const Params& params) {
    const float lambda = kLightSpeed / params.fc;
    std::cout << "========== 理论目标参数 ==========\n";
    for (std::size_t k = 0; k < params.targets.size(); ++k) {
        const auto& target = params.targets[k];
        const float tau = 2.0 * target.R / kLightSpeed;
        const float fd = 2.0 * target.v / lambda;
        std::cout << "目标" << (k + 1)
                  << ": R=" << std::fixed << std::setprecision(2) << target.R << " m"
                  << ", v=" << target.v << " m/s"
                  << ", tau=" << std::setprecision(3) << tau * 1e6 << " us"
                  << ", fd=" << std::setprecision(2) << fd << " Hz\n";
    }
    std::cout << '\n';
}

Matrix<Complex> pulse_compression_cpu(const Matrix<Complex>& x,
                                      const std::vector<Complex>& templ,
                                      bool normalize = false,
                                      const std::string& window = "",
                                      std::size_t nfft = 0) {
    return cuSP::task1_ops::pulse_compression_cpu<Matrix>(x, templ, normalize, window, nfft);
}

Matrix<Complex> pulse_doppler_cpu(const Matrix<Complex>& x,
                                  const std::string& window = "",
                                  std::size_t nfft = 0) {
    return cuSP::task1_ops::pulse_doppler_cpu<Matrix>(x, window, nfft);
}

float cfar_alpha_cpu(float pfa, int N) {
    return cuSP::task1_ops::cfar_alpha_cpu(pfa, N);
}

Matrix<float> integral_image_2d(const Matrix<float>& array) {
    return cuSP::task1_ops::integral_image_2d<Matrix>(array);
}

float rect_sum_2d(const Matrix<float>& integral,
                   int x1,
                   int y1,
                   int x2,
                   int y2) {
    return cuSP::task1_ops::rect_sum_2d<Matrix>(integral, x1, y1, x2, y2);
}

std::pair<Matrix<float>, Matrix<std::uint8_t>> ca_cfar_cpu(const Matrix<float>& array,
                                                           std::pair<int, int> guard_cells,
                                                           std::pair<int, int> reference_cells,
                                                           float pfa = 1e-3) {
    return cuSP::task1_ops::ca_cfar_cpu<Matrix>(array, guard_cells, reference_cells, pfa);
}

AmbiguityImage ambgfun_cpu(const std::vector<Complex>& x,
                           float fs,
                           float prf,
                           const std::vector<Complex>* y = nullptr,
                           const std::string& cut = "2d",
                           float cut_value = 0.0) {
    AmbiguityImage result;
    result.data = cuSP::task1_ops::ambgfun_2d<Matrix>(x, fs, prf, y, cut, cut_value);
    result.rows = result.data.rows;
    result.cols = result.data.cols;
    return result;
}

Matrix<Complex> fftshift_axis0(const Matrix<Complex>& x) {
    return cuSP::task1_ops::fftshift_axis0<Matrix>(x);
}

Matrix<float> abs_square(const Matrix<Complex>& x) {
    return cuSP::task1_ops::abs_square<Matrix>(x);
}

std::vector<Complex> correlate_full(const std::vector<Complex>& a,
                                    const std::vector<Complex>& b) {
    return cuSP::task1_ops::correlate_full(a, b);
}

std::filesystem::path resolve_output_dir() {
    const auto cwd = std::filesystem::current_path();
    if (std::filesystem::exists(cwd / "cuSP_cpp" / "CMakeLists.txt")) {
        return cwd / "cuSP_cpp" / "task1_cpp_outputs";
    }
    if (std::filesystem::exists(cwd / "CMakeLists.txt") &&
        std::filesystem::exists(cwd / "include" / "cuSP_task1_ops.hpp")) {
        return cwd / "task1_cpp_outputs";
    }
    return cwd / "task1_cpp_outputs";
}

std::filesystem::path make_transient_output_dir(const std::string& prefix, const std::string& data_level) {
    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    return std::filesystem::temp_directory_path() /
           (prefix + "_" + data_level + "_" + std::to_string(stamp));
}

std::filesystem::path resolve_runtime_output_dir(const RuntimeOptions& runtime, const std::string& prefix) {
    if (runtime.save_output) {
        return resolve_output_dir() / runtime.data_level;
    }
    return make_transient_output_dir(prefix, runtime.data_level);
}

void cleanup_runtime_output_dir(const RuntimeOptions& runtime, const std::filesystem::path& out_dir) {
    if (!runtime.save_output) {
        std::error_code ec;
        std::filesystem::remove_all(out_dir, ec);
    }
}

void ensure_directory(const std::filesystem::path& dir) {
    std::filesystem::create_directories(dir);
}

template <typename T>
float to_float_value(const T& value) {
    return static_cast<float>(value);
}

float to_float_value(const std::uint8_t& value) {
    return static_cast<float>(value);
}

std::string shell_quote(const std::string& input) {
    std::string quoted = "'";
    for (const char ch : input) {
        if (ch == '\'') {
            quoted += "'\"'\"'";
        } else {
            quoted.push_back(ch);
        }
    }
    quoted.push_back('\'');
    return quoted;
}

std::string format_float_arg(float value) {
    std::ostringstream oss;
    oss << std::setprecision(17) << value;
    return oss.str();
}

void save_series_csv(const std::filesystem::path& path,
                     const std::string& x_label,
                     const std::string& y_label,
                     const std::vector<float>& xs,
                     const std::vector<float>& ys) {
    if (xs.size() != ys.size()) {
        throw std::invalid_argument("save_series_csv requires matching x/y lengths");
    }

    std::ofstream ofs(path);
    ofs << std::setprecision(10);
    ofs << x_label << ',' << y_label << '\n';
    for (std::size_t i = 0; i < xs.size(); ++i) {
        ofs << xs[i] << ',' << ys[i] << '\n';
    }
}

template <typename T>
void save_matrix_csv(const std::filesystem::path& path,
                     const std::string& row_label,
                     const std::vector<float>& row_axis,
                     const std::string& col_label,
                     const std::vector<float>& col_axis,
                     const Matrix<T>& matrix) {
    if (row_axis.size() != matrix.rows || col_axis.size() != matrix.cols) {
        throw std::invalid_argument("save_matrix_csv axis lengths do not match matrix shape");
    }

    std::ofstream ofs(path);
    ofs << std::setprecision(10);
    ofs << row_label << "\\" << col_label;
    for (const float value : col_axis) {
        ofs << ',' << value;
    }
    ofs << '\n';

    for (std::size_t r = 0; r < matrix.rows; ++r) {
        ofs << row_axis[r];
        for (std::size_t c = 0; c < matrix.cols; ++c) {
            ofs << ',' << to_float_value(matrix(r, c));
        }
        ofs << '\n';
    }
}

void save_raw_matrix_f32(const std::filesystem::path& path, const Matrix<float>& matrix) {
    std::ofstream ofs(path, std::ios::binary);
    ofs.write(reinterpret_cast<const char*>(matrix.data.data()),
              static_cast<std::streamsize>(matrix.data.size() * sizeof(float)));
}

void save_raw_meta(const std::filesystem::path& path,
                   const std::string& dtype,
                   const std::vector<std::size_t>& shape) {
    std::ofstream ofs(path.string() + ".meta");
    ofs << "dtype=" << dtype << "\n";
    ofs << "shape=";
    for (std::size_t i = 0; i < shape.size(); ++i) {
        if (i != 0) {
            ofs << ",";
        }
        ofs << shape[i];
    }
    ofs << "\n";
}

template <typename T>
void save_raw_vector(const std::filesystem::path& path,
                     const std::vector<T>& values,
                     const std::string& dtype,
                     const std::vector<std::size_t>& shape) {
    std::ofstream ofs(path, std::ios::binary);
    if (!values.empty()) {
        ofs.write(reinterpret_cast<const char*>(values.data()),
                  static_cast<std::streamsize>(values.size() * sizeof(T)));
    }
    save_raw_meta(path, dtype, shape);
}

std::vector<float> to_float_vector(const std::vector<float>& values) {
    return std::vector<float>(values.begin(), values.end());
}

std::vector<std::complex<float>> to_complex64_vector(const std::vector<Complex>& values) {
    std::vector<std::complex<float>> out;
    out.reserve(values.size());
    for (const auto& value : values) {
        out.emplace_back(static_cast<float>(value.real()), static_cast<float>(value.imag()));
    }
    return out;
}

std::vector<std::complex<float>> to_complex64_vector(const Matrix<Complex>& matrix) {
    return to_complex64_vector(matrix.data);
}

void save_vector_f32_raw(const std::filesystem::path& dir,
                         const std::string& name,
                         const std::vector<float>& values) {
    save_raw_vector(dir / (name + ".bin"), to_float_vector(values), "float32", {values.size()});
}

void save_vector_complex64_raw(const std::filesystem::path& dir,
                               const std::string& name,
                               const std::vector<Complex>& values) {
    save_raw_vector(dir / (name + ".bin"), to_complex64_vector(values), "complex64", {values.size()});
}

void save_matrix_complex64_raw(const std::filesystem::path& dir,
                               const std::string& name,
                               const Matrix<Complex>& matrix) {
    save_raw_vector(dir / (name + ".bin"),
                    to_complex64_vector(matrix),
                    "complex64",
                    {matrix.rows, matrix.cols});
}

void save_matrix_f32_raw(const std::filesystem::path& dir,
                         const std::string& name,
                         const Matrix<float>& matrix) {
    const auto path = dir / (name + ".bin");
    save_raw_matrix_f32(path, matrix);
    save_raw_meta(path, "float32", {matrix.rows, matrix.cols});
}

void save_matrix_u8_raw(const std::filesystem::path& dir,
                        const std::string& name,
                        const Matrix<std::uint8_t>& matrix) {
    save_raw_vector(dir / (name + ".bin"), matrix.data, "uint8", {matrix.rows, matrix.cols});
}

std::vector<Complex> complex_window_vector(const std::string& name, std::size_t n) {
    const auto weights = make_window(name, n);
    std::vector<Complex> out;
    out.reserve(weights.size());
    for (float value : weights) {
        out.emplace_back(value, 0.0);
    }
    return out;
}

Matrix<float> to_float_matrix(const Matrix<std::uint8_t>& matrix) {
    Matrix<float> out(matrix.rows, matrix.cols, 0.0f);
    for (std::size_t i = 0; i < matrix.rows; ++i) {
        for (std::size_t j = 0; j < matrix.cols; ++j) {
            out(i, j) = static_cast<float>(matrix(i, j));
        }
    }
    return out;
}

void save_task1_intermediate_data(const std::filesystem::path& out_dir,
                                  const std::vector<Complex>& tx,
                                  const Matrix<Complex>& rx,
                                  const std::vector<Complex>& pc_window,
                                  const std::vector<Complex>& doppler_window,
                                  const Matrix<Complex>& pc,
                                  const Matrix<Complex>& pd,
                                  const Matrix<Complex>& rd,
                                  const Matrix<float>& rd_power,
                                  const Matrix<float>& threshold,
                                  const Matrix<std::uint8_t>& det_mask,
                                  const Matrix<float>& amf,
                                  const std::vector<float>& range_axis,
                                  const std::vector<float>& velocity_axis) {
    const auto data_dir = out_dir / "data";
    ensure_directory(data_dir);

    save_vector_complex64_raw(data_dir, "tx", tx);
    save_matrix_complex64_raw(data_dir, "rx", rx);
    save_vector_complex64_raw(data_dir, "pc_window", pc_window);
    save_vector_complex64_raw(data_dir, "doppler_window", doppler_window);
    save_matrix_complex64_raw(data_dir, "pc", pc);
    save_matrix_complex64_raw(data_dir, "pd", pd);
    save_matrix_complex64_raw(data_dir, "rd", rd);
    save_matrix_f32_raw(data_dir, "rd_power", rd_power);
    save_matrix_f32_raw(data_dir, "cfar_threshold", threshold);
    save_matrix_u8_raw(data_dir, "cfar_mask", det_mask);
    save_matrix_f32_raw(data_dir, "ambiguity_function", amf);
    save_vector_f32_raw(data_dir, "range_axis", range_axis);
    save_vector_f32_raw(data_dir, "velocity_axis", velocity_axis);

    std::cout << "Intermediate data saved to: " << data_dir.string() << "\n";
}

void save_timing_csv(const std::filesystem::path& out_dir,
                     const std::vector<TimingRecord>& timings,
                     const TimingTotals& totals) {
    ensure_directory(out_dir);
    std::ofstream ofs(out_dir / "timing.csv");
    ofs << "task,implementation,component,stream,category,cpu_ms,gpu_ms,composition_ms,notes\n";
    ofs << std::setprecision(10);
    for (const auto& t : timings) {
        if (!is_operator_timing_record(t)) {
            continue;
        }
        const float composition_ms = (t.category == "kernel") ? kernel_total_ms(t) : t.cpu_ms;
        ofs << "task1,cpp," << t.name << ','
            << t.stream_name << ','
            << t.category << ','
            << t.cpu_ms << ','
            << t.gpu_ms << ','
            << composition_ms << ",\n";
    }
    ofs << "task1,cpp,total_gpu_operator,main-serial,kernel,0,"
        << totals.gpu_operator_ms << ','
        << totals.gpu_operator_ms
        << ",max stream sum(kernel cudaEvent)\n";
    ofs << "task1,cpp,total_cpu_operator,main-serial,h2d+d2h,"
        << totals.cpu_operator_ms
        << ",0,"
        << totals.cpu_operator_ms
        << ",max stream sum(h2d+d2h cpu_ms)\n";
    ofs << "task1,cpp,total_operator,main-serial,h2d+kernel+d2h,"
        << totals.total_operator_ms << ','
        << totals.gpu_operator_ms << ','
        << totals.total_operator_ms
        << ",CPU-only kernel uses cpu_ms in composition_ms\n";
}

std::string resolve_plot_python(const RuntimeOptions& options) {
    if (!options.plot_python.empty()) {
        return options.plot_python;
    }

    if (const char* env_python = std::getenv("TASK1_CPP_PYTHON")) {
        if (*env_python != '\0') {
            return env_python;
        }
    }

    if (const char* conda_prefix = std::getenv("CONDA_PREFIX")) {
        if (*conda_prefix != '\0') {
            const auto candidate = std::filesystem::path(conda_prefix) / "bin" / "python";
            if (std::filesystem::exists(candidate)) {
                return candidate.string();
            }
        }
    }

    const std::vector<std::string> candidates = {
        "/home/zhu/miniforge3/bin/python",
        "/home/zhu/miniforge3/envs/cusignal-dev/bin/python",
        "/usr/bin/python3",
        "python3",
    };

    for (const auto& candidate : candidates) {
        if (candidate == "python3" || std::filesystem::exists(candidate)) {
            return candidate;
        }
    }

    throw std::runtime_error("No usable Python executable found for matplotlib plotting.");
}

void launch_matplotlib_visualization(const RuntimeOptions& runtime,
                                     const std::filesystem::path& out_dir,
                                     const std::filesystem::path& tx_csv,
                                     const std::filesystem::path& rx_csv,
                                     const std::filesystem::path& pc_csv,
                                     const std::filesystem::path& acf_csv,
                                     const std::filesystem::path& rd_bin,
                                     const std::filesystem::path& th_bin,
                                     const std::filesystem::path& det_bin,
                                     const std::filesystem::path& amf_bin,
                                     std::size_t rd_rows,
                                     std::size_t rd_cols,
                                     std::size_t amf_rows,
                                     std::size_t amf_cols,
                                     float range_min,
                                     float range_max,
                                     float vel_min,
                                     float vel_max) {
    const auto script_path = out_dir / "task1_cpp_plot_driver.py";

    std::ofstream script(script_path);
    script << R"PY(import os
import sys

out_dir = sys.argv[1]
tx_csv = sys.argv[2]
rx_csv = sys.argv[3]
pc_csv = sys.argv[4]
acf_csv = sys.argv[5]
rd_bin = sys.argv[6]
th_bin = sys.argv[7]
det_bin = sys.argv[8]
amf_bin = sys.argv[9]
rd_rows = int(sys.argv[10])
rd_cols = int(sys.argv[11])
amf_rows = int(sys.argv[12])
amf_cols = int(sys.argv[13])
range_min = float(sys.argv[14])
range_max = float(sys.argv[15])
vel_min = float(sys.argv[16])
vel_max = float(sys.argv[17])
headless = sys.argv[18] == "1"

mpl_dir = os.path.join(out_dir, ".matplotlib")
os.makedirs(mpl_dir, exist_ok=True)
os.environ["MPLCONFIGDIR"] = mpl_dir

import matplotlib
if headless:
    matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

def read_series(path):
    arr = np.loadtxt(path, delimiter=",", skiprows=1)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr[:, 0], arr[:, 1]

tx_t, tx_y = read_series(tx_csv)
rx_x, rx_y = read_series(rx_csv)
pc_x, pc_y = read_series(pc_csv)
acf_x, acf_y = read_series(acf_csv)

rd = np.fromfile(rd_bin, dtype=np.float32).reshape(rd_rows, rd_cols)
th = np.fromfile(th_bin, dtype=np.float32).reshape(rd_rows, rd_cols)
det = np.fromfile(det_bin, dtype=np.float32).reshape(rd_rows, rd_cols)
amf = None
if amf_rows > 0 and amf_cols > 0:
    amf = np.memmap(amf_bin, dtype=np.float32, mode="r", shape=(amf_rows, amf_cols))

extent_cut = [range_min, range_max, vel_min, vel_max]

fig1 = plt.figure(figsize=(12, 4))
plt.plot(tx_t, tx_y)
plt.title("Transmitted LFM Waveform (Full Duration)")
plt.xlabel("Time (us)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.tight_layout()

fig2 = plt.figure(figsize=(12, 4))
plt.plot(rx_x, rx_y)
plt.title("Received Echo Magnitude (First Pulse)")
plt.xlabel("Fast-Time Sample Index")
plt.ylabel("Magnitude")
plt.grid(True)
plt.tight_layout()

fig3 = plt.figure(figsize=(12, 4))
plt.plot(pc_x, pc_y)
plt.title("Pulse Compression Output (0~500 m)")
plt.xlabel("Range (m)")
plt.ylabel("Magnitude")
plt.grid(True)
plt.tight_layout()

fig4 = plt.figure(figsize=(12, 6))
plt.imshow(rd, aspect="auto", origin="lower", extent=extent_cut)
plt.colorbar(label="Power (dB)")
plt.title("Range-Doppler Map (0~500 m)")
plt.xlabel("Range (m)")
plt.ylabel("Velocity (m/s)")
plt.tight_layout()

fig5 = plt.figure(figsize=(12, 6))
plt.imshow(th, aspect="auto", origin="lower", extent=extent_cut)
plt.colorbar(label="Threshold (dB)")
plt.title("CFAR Threshold (0~500 m)")
plt.xlabel("Range (m)")
plt.ylabel("Velocity (m/s)")
plt.tight_layout()

fig6 = plt.figure(figsize=(12, 6))
plt.imshow(det, aspect="auto", origin="lower", extent=extent_cut)
plt.colorbar(label="Detection Flag")
plt.title("CFAR Detection (0~500 m)")
plt.xlabel("Range (m)")
plt.ylabel("Velocity (m/s)")
plt.tight_layout()

fig7 = plt.figure(figsize=(10, 5))
plt.plot(acf_x, acf_y)
plt.title("LFM Autocorrelation (Mainlobe & Sidelobes)")
plt.xlabel("Lag")
plt.ylabel("Amplitude (dB)")
plt.ylim([-60, 0])
plt.grid(True)
plt.tight_layout()

figures = [
    (fig1, "01_transmitted_lfm_waveform_full_duration.png"),
    (fig2, "02_received_echo_magnitude_first_pulse.png"),
    (fig3, "03_pulse_compression_output_0_500m.png"),
    (fig4, "04_range_doppler_map_0_500m.png"),
    (fig5, "05_cfar_threshold_0_500m.png"),
    (fig6, "06_cfar_detection_0_500m.png"),
    (fig7, "07_lfm_autocorrelation_mainlobe_sidelobes.png"),
]

if amf is not None:
    fig8 = plt.figure(figsize=(8, 6))
    plt.imshow(amf, aspect="auto", origin="lower")
    plt.colorbar(label="Normalized Magnitude")
    plt.title("2D Ambiguity Function of LFM Waveform")
    plt.xlabel("Doppler Bin")
    plt.ylabel("Delay Bin")
    plt.tight_layout()
    figures.append((fig8, "08_ambiguity_function_2d.png"))

if headless:
    for fig, name in figures:
        fig.savefig(os.path.join(out_dir, name), dpi=100)
else:
    plt.show()
)PY";
    script.close();

    const std::string python = resolve_plot_python(runtime);
    const std::string command =
        shell_quote(python) + " " +
        shell_quote(script_path.string()) + " " +
        shell_quote(out_dir.string()) + " " +
        shell_quote(tx_csv.string()) + " " +
        shell_quote(rx_csv.string()) + " " +
        shell_quote(pc_csv.string()) + " " +
        shell_quote(acf_csv.string()) + " " +
        shell_quote(rd_bin.string()) + " " +
        shell_quote(th_bin.string()) + " " +
        shell_quote(det_bin.string()) + " " +
        shell_quote(amf_bin.string()) + " " +
        std::to_string(rd_rows) + " " +
        std::to_string(rd_cols) + " " +
        std::to_string(amf_rows) + " " +
        std::to_string(amf_cols) + " " +
        format_float_arg(range_min) + " " +
        format_float_arg(range_max) + " " +
        format_float_arg(vel_min) + " " +
        format_float_arg(vel_max) + " " +
        (runtime.headless ? "1" : "0");

    const int status = std::system(command.c_str());
    if (status != 0) {
        throw std::runtime_error("Matplotlib plotting failed. Check Python/matplotlib in the selected environment.");
    }
}

RuntimeOptions parse_runtime_options(int argc, char** argv) {
    RuntimeOptions options;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--save-output" || arg == "--output") {
            options.save_output = true;
        } else if (arg == "--no-viz") {
            options.enable_visualization = false;
        } else if (arg == "--headless") {
            options.headless = true;
        } else if (arg == "--plot-python") {
            if (i + 1 >= argc) {
                throw std::invalid_argument("--plot-python requires a following path");
            }
            options.plot_python = argv[++i];
        } else if (arg == "--level") {
            if (i + 1 >= argc) {
                throw std::invalid_argument("--level requires L1, L2, or L3");
            }
            options.data_level = argv[++i];
            if (options.data_level != "L1" && options.data_level != "L2" && options.data_level != "L3") {
                throw std::invalid_argument("--level must be L1, L2, or L3");
            }
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: task1_cpp [--level L1|L2|L3] [--save-output] [--no-viz] [--headless] [--plot-python PATH]\n";
            std::exit(0);
        } else {
            throw std::invalid_argument("Unknown argument: " + arg);
        }
    }

    return options;
}

}  // namespace

// =========================
// 3. Main flow
// =========================
int main(int argc, char** argv) {
    try {
        const RuntimeOptions runtime = parse_runtime_options(argc, argv);
        const Params params = params_for_level(runtime.data_level);

        std::mt19937 rng(0);
        std::vector<TimingRecord> timings;

        const float fc = params.fc;
        const float B = params.B;
        const float Tp = params.Tp;
        const float PRI = params.PRI;
        const float fs = params.fs;
        const std::size_t num_pulses = params.num_pulses;
        const auto& targets = params.targets;
        const float snr_db = params.snr_db;

        std::cout << "Data level = " << runtime.data_level << '\n';
        print_target_summary(params);

        const float lambda = kLightSpeed / fc;
        const float range_resolution = kLightSpeed / (2.0 * B);
        const float velocity_resolution = lambda / (2.0 * static_cast<float>(num_pulses) * PRI);
        const float vmax_unamb = lambda / (4.0 * PRI);
        const float rmax_unamb = kLightSpeed * PRI / 2.0;

        std::cout << "========== 雷达参数 ==========\n";
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "载频 fc = " << fc / 1e9 << " GHz\n";
        std::cout << std::setprecision(3)
                  << "波长 lambda = " << lambda * 1e3 << " mm\n";
        std::cout << std::setprecision(2)
                  << "带宽 B = " << B / 1e6 << " MHz\n"
                  << "脉冲宽度 Tp = " << Tp * 1e6 << " us\n"
                  << "PRI = " << PRI * 1e6 << " us\n"
                  << "采样率 fs = " << fs / 1e6 << " MHz\n"
                  << "脉冲数 = " << num_pulses << '\n';
        std::cout << std::setprecision(3)
                  << "理论距离分辨率 = " << range_resolution << " m\n"
                  << "理论速度分辨率 = " << velocity_resolution << " m/s\n"
                  << "无模糊距离 = " << rmax_unamb / 1e3 << " km\n"
                  << "近似双边无模糊速度 = ±" << vmax_unamb << " m/s\n\n";

        // =========================
        // 1) Transmit waveform
        // =========================
        const auto tx = time_cpu_op("host::make_lfm_pulse", "host", timings, [&]() {
            return make_lfm_pulse(fs, Tp, B);
        });
        const std::size_t samples_per_pulse = tx.size();
        std::cout << "每脉冲采样点数 = " << samples_per_pulse << '\n';

        const std::size_t nfft_range = next_pow2(samples_per_pulse);
        const std::size_t nfft_doppler = next_pow2(num_pulses);

        // =========================
        // 2) Echo simulation
        // =========================
        const auto rx = time_cpu_op("host::simulate_echo_matrix", "host", timings, [&]() {
            return simulate_echo_matrix(
                tx,
                fs,
                PRI,
                num_pulses,
                fc,
                targets,
                snr_db,
                rng);
        });

        const auto pc_window = time_cpu_op("windows::get_window(hamming)", "kernel", timings, [&]() {
            return complex_window_vector(params.pc_window, samples_per_pulse);
        });
        const auto doppler_window = time_cpu_op("windows::get_window(hann)", "kernel", timings, [&]() {
            return complex_window_vector(params.doppler_window, num_pulses);
        });

        // =========================
        // 3) Pulse compression
        // =========================
        const auto pc = time_cpu_op("radartools::pulse_compression", "kernel", timings, [&]() {
            return pulse_compression_cpu(
                rx,
                tx,
                params.pc_normalize,
                params.pc_window,
                nfft_range);
        });

        // =========================
        // 4) Pulse Doppler
        // =========================
        const auto pd = time_cpu_op("radartools::pulse_doppler", "kernel", timings, [&]() {
            return pulse_doppler_cpu(
                pc,
                params.doppler_window,
                nfft_doppler);
        });

        const auto rd = time_cpu_op("fft::fftshift(axis=0)", "kernel", timings, [&]() {
            return fftshift_axis0(pd);
        });
        const auto rd_power = time_cpu_op("kernel::abs_square", "kernel", timings, [&]() {
            return abs_square(rd);
        });

        // =========================
        // 5) CFAR detection
        // =========================
        const auto cfar = time_cpu_op("radartools::ca_cfar", "kernel", timings, [&]() {
            return ca_cfar_cpu(
                rd_power,
                params.guard_cells,
                params.reference_cells,
                params.pfa);
        });
        const auto& threshold = cfar.first;
        const auto& det_mask = cfar.second;

        // =========================
        // 6) Ambiguity function
        // =========================
        const AmbiguityImage amf = time_cpu_op("radartools::ambgfun", "kernel", timings, [&]() {
            return ambgfun_cpu(tx, fs, 1.0 / PRI, nullptr, "2d");
        });

        // =========================
        // 7) Range / velocity axes
        // =========================
        const auto range_axis = time_cpu_op("host::build_range_axis", "host", timings, [&]() {
            return build_range_axis(fs, nfft_range);
        });
        const auto velocity_axis = time_cpu_op("host::build_velocity_axis", "host", timings, [&]() {
            return build_velocity_axis(fc, PRI, nfft_doppler);
        });

        // =========================
        // 8) Detection extraction
        // =========================
        const auto detections = time_cpu_op("host::extract_detections", "host", timings, [&]() {
            return extract_detections(
                rd_power,
                det_mask,
                range_axis,
                velocity_axis,
                params.top_k);
        });

        std::cout << "========== 检测结果 ==========\n";
        if (detections.empty()) {
            std::cout << "未检测到目标\n";
        } else {
            for (std::size_t idx = 0; idx < detections.size(); ++idx) {
                const auto& d = detections[idx];
                std::cout << "[检测" << (idx + 1) << "] "
                          << "距离=" << std::fixed << std::setprecision(3) << d.range_m << " m, "
                          << "速度=" << d.velocity_mps << " m/s, \n";
            }
        }
        std::cout << '\n';

        if (!runtime.save_output && !runtime.enable_visualization) {
            print_timing_summary(timings);
            return 0;
        }

        const auto out_dir = resolve_runtime_output_dir(runtime, "task1_cpp_plot");
        ensure_directory(out_dir);
        if (runtime.save_output) {
            time_cpu_op("save_task1_intermediate_data", "host", timings, [&]() {
                save_task1_intermediate_data(
                    out_dir,
                    tx,
                    rx,
                    pc_window,
                    doppler_window,
                    pc,
                    pd,
                    rd,
                    rd_power,
                    threshold,
                    det_mask,
                    amf.data,
                    range_axis,
                    velocity_axis);
            });
        }
        print_timing_summary(timings);
        if (runtime.save_output) {
            save_timing_csv(out_dir, timings, compute_timing_totals(timings));
        }

        if (!runtime.enable_visualization) {
            return 0;
        }

        // =========================
        // 4. Visualization
        // =========================
        const float max_range = 500.0;
        std::vector<std::size_t> range_idx;
        range_idx.reserve(range_axis.size());
        for (std::size_t i = 0; i < range_axis.size(); ++i) {
            if (range_axis[i] <= max_range) {
                range_idx.push_back(i);
            }
        }

        std::vector<float> time_axis_us(tx.size(), 0.0);
        std::vector<float> tx_real(tx.size(), 0.0);
        for (std::size_t i = 0; i < tx.size(); ++i) {
            time_axis_us[i] = static_cast<float>(i) / fs * 1e6;
            tx_real[i] = tx[i].real();
        }
        save_series_csv(out_dir / "transmitted_lfm_waveform_full_duration.csv",
                        "time_us",
                        "real_amplitude",
                        time_axis_us,
                        tx_real);

        std::vector<float> rx_index(rx.cols, 0.0);
        std::vector<float> rx_mag(rx.cols, 0.0);
        for (std::size_t n = 0; n < rx.cols; ++n) {
            rx_index[n] = static_cast<float>(n);
            rx_mag[n] = std::abs(rx(0, n));
        }
        save_series_csv(out_dir / "received_echo_magnitude_first_pulse.csv",
                        "sample_index",
                        "magnitude",
                        rx_index,
                        rx_mag);

        std::vector<float> range_cut(range_idx.size(), 0.0);
        std::vector<float> pc_mag_cut(range_idx.size(), 0.0);
        for (std::size_t k = 0; k < range_idx.size(); ++k) {
            const std::size_t idx = range_idx[k];
            range_cut[k] = range_axis[idx];
            pc_mag_cut[k] = std::abs(pc(0, idx));
        }
        save_series_csv(out_dir / "pulse_compression_output_0_500m.csv",
                        "range_m",
                        "magnitude",
                        range_cut,
                        pc_mag_cut);

        Matrix<float> rd_db_cut(rd_power.rows, range_idx.size(), 0.0f);
        Matrix<float> th_db_cut(threshold.rows, range_idx.size(), 0.0f);
        Matrix<std::uint8_t> det_cut(det_mask.rows, range_idx.size(), 0);

        for (std::size_t i = 0; i < rd_power.rows; ++i) {
            for (std::size_t k = 0; k < range_idx.size(); ++k) {
                const std::size_t idx = range_idx[k];
                rd_db_cut(i, k) = static_cast<float>(db10(rd_power(i, idx) + 1e-12));
                th_db_cut(i, k) = static_cast<float>(db10(threshold(i, idx) + 1e-12));
                det_cut(i, k) = det_mask(i, idx);
            }
        }

        save_matrix_csv(out_dir / "range_doppler_map_0_500m.csv",
                        "velocity_mps",
                        velocity_axis,
                        "range_m",
                        range_cut,
                        rd_db_cut);

        save_matrix_csv(out_dir / "cfar_threshold_0_500m.csv",
                        "velocity_mps",
                        velocity_axis,
                        "range_m",
                        range_cut,
                        th_db_cut);

        save_matrix_csv(out_dir / "cfar_detection_0_500m.csv",
                        "velocity_mps",
                        velocity_axis,
                        "range_m",
                        range_cut,
                        det_cut);

        const auto det_float_cut = to_float_matrix(det_cut);
        save_raw_matrix_f32(out_dir / "range_doppler_map_0_500m.bin", rd_db_cut);
        save_raw_matrix_f32(out_dir / "cfar_threshold_0_500m.bin", th_db_cut);
        save_raw_matrix_f32(out_dir / "cfar_detection_0_500m.bin", det_float_cut);

        const auto acf = correlate_full(tx, tx);
        std::vector<float> lags;
        std::vector<float> acf_db;
        lags.reserve(acf.size());
        acf_db.reserve(acf.size());

        float acf_peak = 0.0;
        for (const auto& value : acf) {
            acf_peak = std::max(acf_peak, std::abs(value));
        }
        if (acf_peak <= 0.0) {
            acf_peak = 1.0;
        }

        const int lag_start = -static_cast<int>(tx.size()) + 1;
        for (std::size_t i = 0; i < acf.size(); ++i) {
            const int lag = lag_start + static_cast<int>(i);
            if (lag >= -80 && lag <= 80) {
                lags.push_back(static_cast<float>(lag));
                acf_db.push_back(db20(std::abs(acf[i]) / acf_peak + 1e-12));
            }
        }
        save_series_csv(out_dir / "lfm_autocorrelation_mainlobe_sidelobes.csv",
                        "lag",
                        "amplitude_db",
                        lags,
                        acf_db);

        const auto amf_bin_path = out_dir / "ambiguity_function_2d.bin";
        save_raw_matrix_f32(amf_bin_path, amf.data);
        const std::size_t amf_rows = amf.rows;
        const std::size_t amf_cols = amf.cols;

        try {
            launch_matplotlib_visualization(
                runtime,
                out_dir,
                out_dir / "transmitted_lfm_waveform_full_duration.csv",
                out_dir / "received_echo_magnitude_first_pulse.csv",
                out_dir / "pulse_compression_output_0_500m.csv",
                out_dir / "lfm_autocorrelation_mainlobe_sidelobes.csv",
                out_dir / "range_doppler_map_0_500m.bin",
                out_dir / "cfar_threshold_0_500m.bin",
                out_dir / "cfar_detection_0_500m.bin",
                amf_bin_path,
                rd_db_cut.rows,
                rd_db_cut.cols,
                amf_rows,
                amf_cols,
                range_cut.front(),
                range_cut.back(),
                velocity_axis.front(),
                velocity_axis.back());
        } catch (...) {
            cleanup_runtime_output_dir(runtime, out_dir);
            throw;
        }

        if (runtime.save_output && runtime.headless) {
            std::cout << "Plots saved to " << out_dir << '\n';
        }
        cleanup_runtime_output_dir(runtime, out_dir);

        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "task1.cpp failed: " << ex.what() << '\n';
        return 1;
    }
}
