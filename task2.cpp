// task2.cpp
//
// CPU-only C++ translation of cusignal_task/task2_final.py.
//
// Notes:
// - Pure CPU implementation, no GPU / CUDA / CuPy / cuSignal runtime usage.
// - The four feature domains are processed serially, with no multi-threading.
// - Visualization is driven by matplotlib from exported CPU-side results.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "cuSP_task2_ops.hpp"

namespace task2_cpu {

using Complex = std::complex<double>;

template <typename T>
struct Matrix {
    std::size_t rows = 0;
    std::size_t cols = 0;
    std::vector<T> data;

    void resize(std::size_t r, std::size_t c) {
        rows = r;
        cols = c;
        data.assign(r * c, T{});
    }

    T& operator()(std::size_t r, std::size_t c) {
        return data[r * cols + c];
    }

    const T& operator()(std::size_t r, std::size_t c) const {
        return data[r * cols + c];
    }
};

// =========================
// 1. 参数表
// =========================
struct ParamMeta {
    std::string name;
    std::string value_repr;
    std::string unit;
    std::string description;
    bool uncertain_high_impact = false;
    std::string note;
};

struct Params {
    // ---------- 仿真与采样环境 ----------
    double fs = 10e6;
    double duration = 2.0e-3;
    std::uint32_t seed = 2026;

    // ---------- 目标信号：LFM ----------
    double lfm_f0 = 0.20e6;
    double lfm_f1 = 0.60e6;
    double lfm_start = 0.70e-3;
    double lfm_duration = 0.80e-3;
    double lfm_amp = 1.00;

    // ---------- 目标信号：Gaussian Pulse ----------
    double gp_fc = 0.30e6;
    double gp_bw = 0.35;
    double gp_early_center_s = 0.22e-3;
    double gp_offset_from_lfm_start_s = 0.0;
    double gp_amp = 0.80;

    // ---------- 回波 ----------
    double echo_delay_s = 0.15e-3;
    double echo_freq_offset_hz = 20e3;
    double echo_atten = 0.90;

    // ---------- 干扰与噪声 ----------
    double interf_square_freq = 50e3;
    double interf_square_duty = 0.50;
    double interf_square_amp = 0.30;
    double dc_offset = 0.20;
    double snr_db = 5.0;

    // ---------- 预处理 ----------
    std::string detrend_type = "constant";
    int fir_numtaps = 129;
    double fir_band_low_hz = 0.15e6;
    double fir_band_high_hz = 0.70e6;
    std::string fir_window = "hamming";
    std::string filtfilt_padtype = "odd";

    // ---------- 样条平滑 ----------
    int spline_kernel_size = 21;

    // ---------- 多域特征提取 ----------
    std::string correlate_mode = "full";
    std::string correlate_method = "auto";
    std::string csd_window = "hann";
    int csd_nperseg = 512;
    int csd_noverlap = 256;
    double cwt_w = 6.0;
    int cwt_width_min = 4;
    int cwt_width_max = 48;

    // ---------- 峰值检测 ----------
    int mod_peak_order = 12;
    int time_peak_order = 20;
    int freq_peak_order = 6;
    double min_period_s = 8e-6;
    int freq_peak_topk = 5;

    // ---------- Kalman 估计 ----------
    double kalman_dt = 1.0;
    double kalman_process_var = 5e-4;
    double kalman_meas_var = 5e-3;

    // ---------- 活动区检测与可视化 ----------
    double activity_threshold_ratio = 0.20;
    double activity_min_duration_s = 0.20e-3;
};

struct RuntimeOptions {
    bool enable_visualization = true;
    bool headless = false;
    std::string plot_python;
};

const std::vector<ParamMeta> PARAM_TABLE = {
    {"fs", "10000000.0", "Hz", "采样率，必须覆盖目标与干扰的最高频率成分", true, "过低会直接导致混叠，并破坏后续所有域的特征估计"},
    {"duration", "0.002", "s", "总观测时长，需要完整覆盖高斯脉冲、LFM 段和周期干扰", false, ""},
    {"seed", "2026", "-", "随机种子，用于保证噪声与仿真可复现", false, ""},
    {"lfm_f0", "200000.0", "Hz", "LFM 起始频率，放在中频区域，避免靠近 0 和 Nyquist 边缘", true, "任务文档只给了选取原则，没有给出唯一数值"},
    {"lfm_f1", "600000.0", "Hz", "LFM 终止频率，与起始频率共同决定带宽和斜率", true, "直接决定 STFT/CWT 中斜率是否明显"},
    {"lfm_start", "0.0007", "s", "LFM 在总观测窗中的起始时刻", false, ""},
    {"lfm_duration", "0.0008", "s", "LFM 持续时间", true, "与斜率共同决定时频图中的轨迹长度和可辨识度"},
    {"lfm_amp", "1.0", "-", "LFM 主分量幅度", false, ""},
    {"gp_fc", "300000.0", "Hz", "高斯脉冲中心频率", false, ""},
    {"gp_bw", "0.35", "-", "高斯脉冲分数带宽", true, "过窄会接近单频，过宽会变成尖刺，不利于峰值定位"},
    {"gp_early_center_s", "0.00022", "s", "第一个高斯脉冲中心时刻，放在 LFM 很早之前", false, ""},
    {"gp_offset_from_lfm_start_s", "0.0", "s", "第二个高斯脉冲中心相对 LFM 起始时刻的偏移，0 表示放在 LFM 开头", false, ""},
    {"gp_amp", "0.8", "-", "高斯脉冲幅度，略低于 LFM 主分量", false, ""},
    {"echo_delay_s", "0.00015", "s", "目标回波时延", false, ""},
    {"echo_freq_offset_hz", "20000.0", "Hz", "目标回波频偏", false, ""},
    {"echo_atten", "0.9", "-", "目标回波衰减", false, ""},
    {"interf_square_freq", "50000.0", "Hz", "square 周期干扰频率", true, "决定时域周期和频域谐波分布"},
    {"interf_square_duty", "0.5", "-", "square 干扰占空比", false, ""},
    {"interf_square_amp", "0.3", "-", "square 干扰幅度，应略低于主目标成分", true, "过大会导致时域周期和频域谐波主导结果"},
    {"dc_offset", "0.2", "-", "仿真中显式加入的 DC 偏置大小；输入多少就加入多少，后续再用 detrend 去除", false, ""},
    {"snr_db", "5.0", "dB", "高斯白噪声信噪比，按文档要求取 5 dB 左右低噪", true, "直接影响峰值检测和 Kalman 稳定性"},
    {"detrend_type", "\"constant\"", "-", "去趋势类型，按文档主线用 constant 去掉 DC 偏置", false, ""},
    {"fir_numtaps", "129", "samples", "firwin 带通 FIR 的抽头数，取奇数以保证线性相位", true, "过小会导致过渡带过宽，过大则增加边缘效应和计算量"},
    {"fir_band_low_hz", "150000.0", "Hz", "带通 FIR 下截止频率，用于压制低频 square 干扰与 DC", true, "过高会削弱 Gaussian Pulse 和 Chirp 起始频率成分"},
    {"fir_band_high_hz", "700000.0", "Hz", "带通 FIR 上截止频率，用于保留目标主频带", true, "过低会截断 Chirp 高频端，过高会放进更多带外噪声"},
    {"fir_window", "\"hamming\"", "-", "firwin 所用窗口类型", false, ""},
    {"filtfilt_padtype", "\"odd\"", "-", "filtfilt 边界延拓方式", false, ""},
    {"spline_kernel_size", "21", "samples", "三次 B-spline 平滑核长度", true, "任务文档要求“不用过度平滑”，该参数直接控制平滑强度"},
    {"correlate_mode", "\"full\"", "-", "自相关输出模式", false, ""},
    {"correlate_method", "\"auto\"", "-", "自相关计算方法", false, ""},
    {"csd_window", "\"hann\"", "-", "PSD/CSD 使用的窗函数", false, ""},
    {"csd_nperseg", "512", "samples", "CSD 分段长度", true, "频率分辨率与时域稳健性之间存在直接折中"},
    {"csd_noverlap", "256", "samples", "CSD 分段重叠长度", false, ""},
    {"cwt_w", "6.0", "-", "Morlet2 小波参数 w", false, ""},
    {"cwt_width_min", "4", "samples", "CWT 最小尺度", true, "与最大尺度一起决定时频域扫描频段"},
    {"cwt_width_max", "48", "samples", "CWT 最大尺度", true, "尺度过大时会过分强调低频周期分量，削弱 chirp ridge"},
    {"mod_peak_order", "12", "samples", "调制域 argrelextrema 阶数", true, "直接影响瞬时频率峰值的稳定性"},
    {"time_peak_order", "20", "samples", "时域 argrelextrema 阶数", true, "过小会产生大量伪周期峰，过大则会漏检"},
    {"freq_peak_order", "6", "bins", "频域 argrelextrema 阶数", true, "决定是否把邻近谱峰误分成多个峰"},
    {"min_period_s", "8e-06", "s", "时域周期估计时排除零延迟附近伪峰的最小周期约束", true, "过小会把主瓣旁瓣误当成周期峰"},
    {"freq_peak_topk", "5", "-", "参与频域 Kalman 估计的前 K 个峰", true, "K 过大时容易把干扰峰一并纳入估计"},
    {"kalman_dt", "1.0", "-", "Kalman 平滑的归一化时间步长", false, ""},
    {"kalman_process_var", "0.0005", "-", "Kalman 过程噪声方差", true, "决定估计结果跟踪变化趋势还是过度平滑"},
    {"kalman_meas_var", "0.005", "-", "Kalman 观测噪声方差", true, "决定对峰值检测结果的信任程度"},
    {"activity_threshold_ratio", "0.2", "-", "根据平滑后包络检测主活动区间的阈值比例", true, "任务文档给了开始/结束时间目标，但没有给出唯一门限选取规则"},
    {"activity_min_duration_s", "0.0002", "s", "活动区最短持续时间约束，避免只截到单个振荡峰", true, "过小会把局部峰误判为活动区，过大会把相邻成分连在一起"},
};

const Params params{};

struct TruthSummary {
    double lfm_start_s = std::numeric_limits<double>::quiet_NaN();
    double lfm_end_s = std::numeric_limits<double>::quiet_NaN();
    double lfm_f0_hz = std::numeric_limits<double>::quiet_NaN();
    double lfm_f1_hz = std::numeric_limits<double>::quiet_NaN();
    double lfm_bw_hz = std::numeric_limits<double>::quiet_NaN();
    double lfm_slope_hz_per_s = std::numeric_limits<double>::quiet_NaN();
    double gp_early_center_s = std::numeric_limits<double>::quiet_NaN();
    double gp_lfm_center_s = std::numeric_limits<double>::quiet_NaN();
    double interference_period_s = std::numeric_limits<double>::quiet_NaN();
};

struct Results {
    // 原始/预处理链路
    std::vector<double> t;
    std::vector<Complex> target_clean;
    std::vector<Complex> echo_complex;
    std::vector<double> rx_noiseless;
    std::vector<double> rx;
    std::vector<double> rx_detrended;
    std::vector<double> rx_denoised;
    std::vector<double> rx_smooth;
    std::vector<double> activity_envelope;

    // 调制域
    std::vector<Complex> analytic;
    std::vector<double> t_inst;
    std::vector<double> inst_freq;
    std::vector<double> inst_freq_kf;

    // 时域
    std::vector<double> lags_pos;
    std::vector<double> autocorr_pos;

    // 频域
    std::vector<double> freqs_pos;
    std::vector<double> psd_pos;

    // 时频域
    std::vector<int> widths;
    std::vector<double> pseudo_freqs;
    Matrix<float> cwt_power;
    std::vector<double> ridge_freq;
    std::vector<double> ridge_energy;
    std::vector<double> ridge_freq_kf;

    // 峰值
    std::vector<int> mod_peaks;
    std::vector<int> time_peaks;
    std::vector<int> freq_peaks;

    // 活动区与估计结果
    int activity_start_idx = 0;
    int activity_end_idx = 0;
    int activity_inst_start_idx = 0;
    int activity_inst_end_idx = 0;

    double est_start_s = std::numeric_limits<double>::quiet_NaN();
    double est_end_s = std::numeric_limits<double>::quiet_NaN();
    double est_main_freq_hz = std::numeric_limits<double>::quiet_NaN();
    double est_period_s = std::numeric_limits<double>::quiet_NaN();
    double est_center_freq_hz = std::numeric_limits<double>::quiet_NaN();
    double est_bandwidth_hz = std::numeric_limits<double>::quiet_NaN();
    double mod_slope_hz_per_s = std::numeric_limits<double>::quiet_NaN();
    double mod_intercept_hz = std::numeric_limits<double>::quiet_NaN();
    double ridge_slope_hz_per_s = std::numeric_limits<double>::quiet_NaN();
    double ridge_intercept_hz = std::numeric_limits<double>::quiet_NaN();

    std::string modulation_label = "Unknown";
    TruthSummary truth;
};

double db10(double x, double eps) {
    return cuSP::task2_ops::db10(x, eps);
}

int ensure_odd(int n) {
    return cuSP::task2_ops::ensure_odd(n);
}

std::vector<double> build_time_axis(double fs, double duration) {
    return cuSP::task2_ops::build_time_axis(fs, duration);
}

std::vector<Complex> apply_integer_delay(const std::vector<Complex>& x, double fs, double delay_s) {
    return cuSP::task2_ops::apply_integer_delay(x, fs, delay_s);
}

std::vector<double> add_awgn(const std::vector<double>& x, double snr_db, std::uint32_t seed) {
    return cuSP::task2_ops::add_awgn(x, snr_db, seed);
}

std::pair<double, double> linear_fit(const std::vector<double>& x, const std::vector<double>& y) {
    return cuSP::task2_ops::linear_fit(x, y);
}

std::pair<int, int> longest_true_segment(const std::vector<std::uint8_t>& mask) {
    return cuSP::task2_ops::longest_true_segment(mask);
}

void print_parameter_summary() {
    std::cout << "========== 参数表摘要 ==========\n";
    std::cout << "采样率 fs = " << std::fixed << std::setprecision(2) << (params.fs / 1e6) << " MHz\n";
    std::cout << "观测时长 duration = " << std::fixed << std::setprecision(3) << (params.duration * 1e3) << " ms\n";
    std::cout << "噪声 SNR = " << std::fixed << std::setprecision(2) << params.snr_db << " dB\n";
    std::cout << "高影响且不确定的参数：\n";
    for (const auto& meta : PARAM_TABLE) {
        if (meta.uncertain_high_impact) {
            std::cout << "  - " << meta.name << " = " << meta.value_repr;
            if (!meta.note.empty()) {
                std::cout << " (" << meta.note << ")";
            }
            std::cout << "\n";
        }
    }
    std::cout << "\n";
}

std::filesystem::path default_output_dir() {
    const std::filesystem::path cwd = std::filesystem::current_path();
    const std::filesystem::path repo_style = cwd / "cusignal_task";
    if (std::filesystem::exists(repo_style) && std::filesystem::is_directory(repo_style)) {
        return repo_style / "task2_cpp_outputs";
    }
    if (cwd.filename() == "cusignal_task") {
        return cwd / "task2_cpp_outputs";
    }
    return cwd / "task2_cpp_outputs";
}

void ensure_directory(const std::filesystem::path& dir) {
    std::filesystem::create_directories(dir);
}

std::string shell_quote(const std::string& input) {
    std::string quoted = "'";
    for (char ch : input) {
        if (ch == '\'') {
            quoted += "'\"'\"'";
        } else {
            quoted.push_back(ch);
        }
    }
    quoted.push_back('\'');
    return quoted;
}

std::string format_double_arg(double value) {
    std::ostringstream oss;
    oss << std::setprecision(17) << value;
    return oss.str();
}

void save_series_csv(const std::filesystem::path& path,
                     const std::string& x_label,
                     const std::string& y_label,
                     const std::vector<double>& xs,
                     const std::vector<double>& ys) {
    if (xs.size() != ys.size()) {
        throw std::invalid_argument("save_series_csv requires matching x/y lengths");
    }

    std::ofstream ofs(path);
    ofs << std::setprecision(10);
    ofs << x_label << "," << y_label << "\n";
    for (std::size_t i = 0; i < xs.size(); ++i) {
        ofs << xs[i] << "," << ys[i] << "\n";
    }
}

void save_index_csv(const std::filesystem::path& path, const std::string& label, const std::vector<int>& values) {
    std::ofstream ofs(path);
    ofs << label << "\n";
    for (int value : values) {
        ofs << value << "\n";
    }
}

void save_vector_csv(const std::filesystem::path& path, const std::string& label, const std::vector<double>& values) {
    std::ofstream ofs(path);
    ofs << std::setprecision(10);
    ofs << label << "\n";
    for (double value : values) {
        ofs << value << "\n";
    }
}

void save_raw_matrix_f32(const std::filesystem::path& path, const Matrix<float>& matrix) {
    std::ofstream ofs(path, std::ios::binary);
    ofs.write(reinterpret_cast<const char*>(matrix.data.data()),
              static_cast<std::streamsize>(matrix.data.size() * sizeof(float)));
}

std::string resolve_plot_python(const RuntimeOptions& runtime) {
    if (!runtime.plot_python.empty()) {
        return runtime.plot_python;
    }

    if (const char* env_python = std::getenv("TASK2_CPP_PYTHON")) {
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

// =========================
// 3. 建模模块
// =========================
std::vector<Complex> generate_lfm_component(const std::vector<double>& t) {
    return cuSP::task2_ops::generate_lfm_component(t, params);
}

std::vector<Complex> generate_gaussian_component(const std::vector<double>& t) {
    return cuSP::task2_ops::generate_gaussian_component(t, params);
}

std::vector<double> generate_square_interference(const std::vector<double>& t) {
    return cuSP::task2_ops::generate_square_interference(t, params);
}

void synthesize_received_signal(
    const std::vector<double>& t,
    std::vector<Complex>& target_clean,
    std::vector<Complex>& echo_complex,
    std::vector<double>& rx_noiseless,
    std::vector<double>& rx) {
    cuSP::task2_ops::synthesize_received_signal(t, params, target_clean, echo_complex, rx_noiseless, rx);
}

void preprocess_signal(
    const std::vector<double>& rx,
    std::vector<double>& detrended,
    std::vector<double>& filtered) {
    cuSP::task2_ops::preprocess_signal(rx, params, detrended, filtered);
}

std::vector<double> smooth_with_cubic_bspline(const std::vector<double>& x) {
    return cuSP::task2_ops::smooth_with_cubic_bspline(x, params);
}

void estimate_activity_interval(
    const std::vector<double>& x_smooth,
    std::vector<double>& envelope_smooth,
    std::vector<std::uint8_t>& activity_mask,
    int& start_idx,
    int& end_idx) {
    cuSP::task2_ops::estimate_activity_interval(
        x_smooth, params, envelope_smooth, activity_mask, start_idx, end_idx);
}

// =========================
// 5. 多域特征提取模块
// =========================
namespace {

std::vector<double> slice_vector(const std::vector<double>& x, int start, int end) {
    const int n = static_cast<int>(x.size());
    const int s = std::clamp(start, 0, n);
    const int e = std::clamp(end, s, n);
    return std::vector<double>(x.begin() + s, x.begin() + e);
}

std::vector<int> argsort_descending(const std::vector<double>& x) {
    std::vector<int> idx(x.size(), 0);
    for (std::size_t i = 0; i < x.size(); ++i) {
        idx[i] = static_cast<int>(i);
    }
    std::sort(idx.begin(), idx.end(), [&](int a, int b) { return x[static_cast<std::size_t>(a)] > x[static_cast<std::size_t>(b)]; });
    return idx;
}

std::vector<double> diff_vector(const std::vector<double>& x) {
    if (x.size() <= 1) {
        return {};
    }

    std::vector<double> y(x.size() - 1, 0.0);
    for (std::size_t i = 0; i + 1 < x.size(); ++i) {
        y[i] = x[i + 1] - x[i];
    }
    return y;
}

std::vector<double> abs_vector(const std::vector<double>& x) {
    std::vector<double> y(x.size(), 0.0);
    for (std::size_t i = 0; i < x.size(); ++i) {
        y[i] = std::abs(x[i]);
    }
    return y;
}

}  // namespace

void extract_modulation_domain(
    const std::vector<double>& x_smooth,
    const std::vector<double>& t,
    std::vector<Complex>& analytic,
    std::vector<double>& t_inst,
    std::vector<double>& inst_freq) {
    cuSP::task2_ops::extract_modulation_domain(x_smooth, t, params, analytic, t_inst, inst_freq);
}

void extract_time_domain(
    const std::vector<double>& x_smooth,
    std::vector<double>& lags_pos,
    std::vector<double>& autocorr_pos) {
    cuSP::task2_ops::extract_time_domain(x_smooth, params, lags_pos, autocorr_pos);
}

void extract_frequency_domain(
    const std::vector<double>& x_smooth,
    std::vector<double>& freqs_pos,
    std::vector<double>& psd_pos) {
    cuSP::task2_ops::extract_frequency_domain(x_smooth, params, freqs_pos, psd_pos);
}

void extract_time_frequency_domain(
    const std::vector<double>& x_smooth,
    std::vector<int>& widths,
    std::vector<double>& pseudo_freqs,
    Matrix<float>& cwt_power,
    std::vector<double>& ridge_freq,
    std::vector<double>& ridge_energy) {
    cuSP::task2_ops::extract_time_frequency_domain<Matrix>(
        x_smooth, params, widths, pseudo_freqs, cwt_power, ridge_freq, ridge_energy);
}

// =========================
// 6. 峰值检测模块
// =========================
std::vector<int> detect_peaks_1d(const std::vector<double>& x, int order, double threshold_ratio) {
    return cuSP::task2_ops::detect_peaks_1d(x, order, threshold_ratio);
}

std::vector<int> fallback_global_peak(const std::vector<double>& x) {
    return cuSP::task2_ops::fallback_global_peak(x);
}

// =========================
// 7. Kalman 估计模块
// =========================
std::vector<double> kalman_smooth_1d(const std::vector<double>& measurements) {
    return cuSP::task2_ops::kalman_smooth_1d(measurements, params);
}

// =========================
// 8. 结果汇总与可视化模块
// =========================
TruthSummary build_truth_summary() {
    TruthSummary truth;
    truth.lfm_start_s = params.lfm_start + params.echo_delay_s;
    truth.lfm_end_s = truth.lfm_start_s + params.lfm_duration;
    truth.lfm_f0_hz = params.lfm_f0 + params.echo_freq_offset_hz;
    truth.lfm_f1_hz = params.lfm_f1 + params.echo_freq_offset_hz;
    truth.lfm_bw_hz = truth.lfm_f1_hz - truth.lfm_f0_hz;
    truth.lfm_slope_hz_per_s = (params.lfm_f1 - params.lfm_f0) / params.lfm_duration;
    truth.gp_early_center_s = params.gp_early_center_s + params.echo_delay_s;
    truth.gp_lfm_center_s = params.lfm_start + params.gp_offset_from_lfm_start_s + params.echo_delay_s;
    truth.interference_period_s = 1.0 / params.interf_square_freq;
    return truth;
}

void add_text_box_placeholder() {}

namespace {

std::vector<double> real_part_vector(const std::vector<Complex>& x) {
    std::vector<double> out(x.size(), 0.0);
    for (std::size_t i = 0; i < x.size(); ++i) {
        out[i] = x[i].real();
    }
    return out;
}

void save_plot_metadata(const std::filesystem::path& path, const Results& results) {
    std::ofstream ofs(path);
    ofs << std::setprecision(17);
    ofs << "activity_start_idx=" << results.activity_start_idx << "\n";
    ofs << "activity_end_idx=" << results.activity_end_idx << "\n";
    ofs << "activity_inst_start_idx=" << results.activity_inst_start_idx << "\n";
    ofs << "activity_inst_end_idx=" << results.activity_inst_end_idx << "\n";
    ofs << "est_start_s=" << results.est_start_s << "\n";
    ofs << "est_end_s=" << results.est_end_s << "\n";
    ofs << "est_main_freq_hz=" << results.est_main_freq_hz << "\n";
    ofs << "est_period_s=" << results.est_period_s << "\n";
    ofs << "est_center_freq_hz=" << results.est_center_freq_hz << "\n";
    ofs << "est_bandwidth_hz=" << results.est_bandwidth_hz << "\n";
    ofs << "mod_slope_hz_per_s=" << results.mod_slope_hz_per_s << "\n";
    ofs << "mod_intercept_hz=" << results.mod_intercept_hz << "\n";
    ofs << "ridge_slope_hz_per_s=" << results.ridge_slope_hz_per_s << "\n";
    ofs << "ridge_intercept_hz=" << results.ridge_intercept_hz << "\n";
    ofs << "modulation_label=" << results.modulation_label << "\n";
}

void launch_matplotlib_visualization(const RuntimeOptions& runtime, const std::filesystem::path& out_dir) {
    const auto script_path = out_dir / "task2_cpp_plot_driver.py";

    std::ofstream script(script_path);
    script << R"PY(import math
import os
import sys

out_dir = sys.argv[1]
headless = sys.argv[2] == "1"

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
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        next(f, None)
        for line in f:
            line = line.strip()
            if not line:
                continue
            x_str, y_str = line.split(",", 1)
            rows.append((float(x_str), float(y_str)))
    if not rows:
        return np.array([], dtype=float), np.array([], dtype=float)
    arr = np.asarray(rows, dtype=float)
    return arr[:, 0], arr[:, 1]

def read_vector(path):
    values = []
    with open(path, "r", encoding="utf-8") as f:
        next(f, None)
        for line in f:
            line = line.strip()
            if line:
                values.append(float(line))
    return np.asarray(values, dtype=float)

def read_indices(path):
    values = []
    with open(path, "r", encoding="utf-8") as f:
        next(f, None)
        for line in f:
            line = line.strip()
            if line:
                values.append(int(float(line)))
    return np.asarray(values, dtype=int)

def read_meta(path):
    meta = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or "=" not in line:
                continue
            key, value = line.split("=", 1)
            meta[key] = value

    int_keys = {
        "activity_start_idx",
        "activity_end_idx",
        "activity_inst_start_idx",
        "activity_inst_end_idx",
    }
    float_keys = {
        "est_start_s",
        "est_end_s",
        "est_main_freq_hz",
        "est_period_s",
        "est_center_freq_hz",
        "est_bandwidth_hz",
        "mod_slope_hz_per_s",
        "mod_intercept_hz",
        "ridge_slope_hz_per_s",
        "ridge_intercept_hz",
    }

    for key in int_keys:
        meta[key] = int(float(meta.get(key, "0")))
    for key in float_keys:
        meta[key] = float(meta.get(key, "nan"))
    meta["modulation_label"] = meta.get("modulation_label", "Unknown")
    return meta

def add_text_box(ax, lines, x=0.02, y=0.98):
    ax.text(
        x,
        y,
        "\n".join(lines),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.80, "edgecolor": "gray"},
    )

meta = read_meta(os.path.join(out_dir, "plot_metadata.txt"))
t_np, target_np = read_series(os.path.join(out_dir, "target_clean_real.csv"))
_, rx_np = read_series(os.path.join(out_dir, "received_signal.csv"))
_, filtered_np = read_series(os.path.join(out_dir, "filtered_signal.csv"))
_, smooth_np = read_series(os.path.join(out_dir, "spline_smoothed_signal.csv"))
t_inst_np, inst_freq_np = read_series(os.path.join(out_dir, "inst_freq.csv"))
t_inst_active_np, inst_freq_kf_np = read_series(os.path.join(out_dir, "inst_freq_kf.csv"))
lags_np, autocorr_np = read_series(os.path.join(out_dir, "autocorr_pos.csv"))
freqs_np, psd_np = read_series(os.path.join(out_dir, "psd_pos.csv"))
t_ridge_np, ridge_freq_np = read_series(os.path.join(out_dir, "ridge_freq.csv"))
t_ridge_kf_np, ridge_kf_np = read_series(os.path.join(out_dir, "ridge_freq_kf.csv"))
pseudo_freqs_np = read_vector(os.path.join(out_dir, "pseudo_freqs.csv"))
mod_peak_idx_np = read_indices(os.path.join(out_dir, "mod_peaks.csv"))
time_peak_idx_np = read_indices(os.path.join(out_dir, "time_peaks.csv"))
freq_peak_idx_np = read_indices(os.path.join(out_dir, "freq_peaks.csv"))

cwt_power_np = np.fromfile(os.path.join(out_dir, "cwt_power.bin"), dtype=np.float32)
if pseudo_freqs_np.size > 0 and t_np.size > 0:
    cwt_power_np = cwt_power_np.reshape(pseudo_freqs_np.size, t_np.size)
else:
    cwt_power_np = np.zeros((0, 0), dtype=np.float32)

act_start = meta["activity_start_idx"]
act_end = meta["activity_end_idx"]
activity_inst_start = meta["activity_inst_start_idx"]
activity_inst_end = meta["activity_inst_end_idx"]

mod_fit_line_np = None
ridge_fit_line_np = None
if math.isfinite(meta["mod_slope_hz_per_s"]) and math.isfinite(meta["mod_intercept_hz"]):
    mod_fit_line_np = meta["mod_slope_hz_per_s"] * t_inst_np[activity_inst_start:activity_inst_end] + meta["mod_intercept_hz"]
if math.isfinite(meta["ridge_slope_hz_per_s"]) and math.isfinite(meta["ridge_intercept_hz"]):
    ridge_fit_line_np = meta["ridge_slope_hz_per_s"] * t_np[act_start:act_end] + meta["ridge_intercept_hz"]

fig1, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
ax = axes[0]
ax.plot(t_np * 1e3, target_np, label="Transmit Target", linewidth=1.2)
ax.plot(t_np * 1e3, rx_np, label="Received Signal", linewidth=1.0, alpha=0.70)
ax.set_title("Figure 1. Transmit Target vs Received Signal")
ax.set_ylabel("Amplitude")
ax.grid(True)
ax.legend()

ax = axes[1]
ax.plot(t_np * 1e3, filtered_np, color="tab:green", linewidth=1.2, label="Filtered Signal")
ax.set_title("Figure 1.2 Filtered Signal (detrend + FIR bandpass + filtfilt)")
ax.set_ylabel("Amplitude")
ax.grid(True)
ax.legend()

ax = axes[2]
ax.plot(t_np * 1e3, smooth_np, color="tab:orange", linewidth=1.2, label="Spline Smoothed")
if t_np.size > 0:
    ax.axvspan(
        t_np[act_start] * 1e3,
        t_np[max(act_start, act_end - 1)] * 1e3,
        color="orange",
        alpha=0.15,
        label="Activity Interval",
    )
ax.set_title("Figure 1.3 Cubic B-Spline Smoothed Signal")
ax.set_xlabel("Time (ms)")
ax.set_ylabel("Amplitude")
ax.grid(True)
ax.legend()
fig1.tight_layout()

fig2, ax2 = plt.subplots(figsize=(16, 8))
ax2.plot(t_inst_np * 1e3, inst_freq_np / 1e6, label="Instantaneous Frequency", alpha=0.40)
if inst_freq_kf_np.size > 0:
    ax2.plot(
        t_inst_active_np * 1e3,
        inst_freq_kf_np / 1e6,
        linewidth=1.8,
        color="tab:red",
        label="Kalman Estimate",
    )
if mod_fit_line_np is not None:
    ax2.plot(
        t_inst_np[activity_inst_start:activity_inst_end] * 1e3,
        mod_fit_line_np / 1e6,
        linestyle="--",
        color="black",
        linewidth=1.3,
        label="Linear Fit",
    )
if mod_peak_idx_np.size > 0:
    ax2.scatter(
        t_inst_np[mod_peak_idx_np] * 1e3,
        inst_freq_np[mod_peak_idx_np] / 1e6,
        color="tab:purple",
        s=22,
        label="Detected Peaks",
    )
ax2.set_title("Figure 2. Modulation Domain")
ax2.set_xlabel("Time (ms)")
ax2.set_ylabel("Frequency (MHz)")
ax2.grid(True)
ax2.legend()
add_text_box(
    ax2,
    [
        f"Modulation: {meta['modulation_label']}",
        f"Slope = {meta['mod_slope_hz_per_s']:.3e} Hz/s" if math.isfinite(meta["mod_slope_hz_per_s"]) else "Slope = NaN",
        f"Peak count = {mod_peak_idx_np.size}",
    ],
)
fig2.tight_layout()

fig3, ax3 = plt.subplots(figsize=(16, 8))
ax3.plot(lags_np * 1e6, autocorr_np, linewidth=1.2, label="Autocorrelation")
if time_peak_idx_np.size > 0:
    ax3.scatter(
        lags_np[time_peak_idx_np] * 1e6,
        autocorr_np[time_peak_idx_np],
        color="tab:red",
        s=26,
        label="Detected Peaks",
    )
if math.isfinite(meta["est_period_s"]):
    ax3.axvline(
        meta["est_period_s"] * 1e6,
        color="tab:green",
        linestyle="--",
        linewidth=1.5,
        label="Estimated Period",
    )
ax3.set_title("Figure 3. Time Domain Feature")
ax3.set_xlabel("Lag (us)")
ax3.set_ylabel("Normalized Correlation")
ax3.grid(True)
ax3.legend()
add_text_box(
    ax3,
    [
        f"Detected peaks = {time_peak_idx_np.size}",
        f"Estimated period = {meta['est_period_s'] * 1e6:.3f} us" if math.isfinite(meta["est_period_s"]) else "Estimated period = NaN",
    ],
)
fig3.tight_layout()

fig4, ax4 = plt.subplots(figsize=(16, 8))
psd_db_np = 10.0 * np.log10(psd_np + 1e-12)
ax4.plot(freqs_np / 1e6, psd_db_np, linewidth=1.2, label="PSD via CSD(y=x)")
if freq_peak_idx_np.size > 0:
    ax4.scatter(
        freqs_np[freq_peak_idx_np] / 1e6,
        psd_db_np[freq_peak_idx_np],
        color="tab:red",
        s=26,
        label="Detected Peaks",
    )
if math.isfinite(meta["est_main_freq_hz"]):
    ax4.axvline(
        meta["est_main_freq_hz"] / 1e6,
        color="tab:green",
        linestyle="--",
        linewidth=1.5,
        label="Kalman Main Frequency",
    )
ax4.set_title("Figure 4. Frequency Domain Feature")
ax4.set_xlabel("Frequency (MHz)")
ax4.set_ylabel("Power (dB)")
ax4.grid(True)
ax4.legend()
add_text_box(
    ax4,
    [
        f"Detected peaks = {freq_peak_idx_np.size}",
        f"Estimated main freq = {meta['est_main_freq_hz'] / 1e6:.3f} MHz" if math.isfinite(meta["est_main_freq_hz"]) else "Estimated main freq = NaN",
    ],
)
fig4.tight_layout()

fig5, ax5 = plt.subplots(figsize=(16, 8))
cwt_plot = cwt_power_np[::-1]
freq_plot = pseudo_freqs_np[::-1] / 1e6
extent = [t_np[0] * 1e3, t_np[-1] * 1e3, freq_plot[0], freq_plot[-1]]
im = ax5.imshow(
    10.0 * np.log10(cwt_plot + 1e-12),
    aspect="auto",
    origin="lower",
    extent=extent,
    cmap="viridis",
)
ax5.plot(t_ridge_np * 1e3, ridge_freq_np / 1e6, color="white", linewidth=1.0, label="Raw Ridge")
if ridge_kf_np.size > 0:
    ax5.plot(
        t_ridge_kf_np * 1e3,
        ridge_kf_np / 1e6,
        color="tab:red",
        linewidth=1.8,
        label="Kalman Ridge",
    )
if ridge_fit_line_np is not None:
    ax5.plot(
        t_np[act_start:act_end] * 1e3,
        ridge_fit_line_np / 1e6,
        color="black",
        linestyle="--",
        linewidth=1.3,
        label="Linear Fit",
    )
ax5.scatter(
    [meta["est_start_s"] * 1e3, meta["est_end_s"] * 1e3],
    [ridge_kf_np[0] / 1e6 if ridge_kf_np.size > 0 else np.nan, ridge_kf_np[-1] / 1e6 if ridge_kf_np.size > 0 else np.nan],
    color="cyan",
    s=36,
    label="Estimated Start/End",
)
ax5.set_title("Figure 5. Time-Frequency Domain Feature")
ax5.set_xlabel("Time (ms)")
ax5.set_ylabel("Pseudo Frequency (MHz)")
ax5.legend(loc="upper right")
fig5.colorbar(im, ax=ax5, label="Energy (dB)")
add_text_box(
    ax5,
    [
        f"Start / End = {meta['est_start_s'] * 1e3:.3f} / {meta['est_end_s'] * 1e3:.3f} ms",
        f"Center freq = {meta['est_center_freq_hz'] / 1e6:.3f} MHz" if math.isfinite(meta["est_center_freq_hz"]) else "Center freq = NaN",
        f"Bandwidth = {meta['est_bandwidth_hz'] / 1e6:.3f} MHz" if math.isfinite(meta["est_bandwidth_hz"]) else "Bandwidth = NaN",
        f"Slope = {meta['ridge_slope_hz_per_s']:.3e} Hz/s" if math.isfinite(meta["ridge_slope_hz_per_s"]) else "Slope = NaN",
    ],
)
fig5.tight_layout()

figures = [
    (fig1, "01_processing_chain.png"),
    (fig2, "02_modulation_domain.png"),
    (fig3, "03_time_domain_feature.png"),
    (fig4, "04_frequency_domain_feature.png"),
    (fig5, "05_time_frequency_domain_feature.png"),
]

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
        (runtime.headless ? "1" : "0");

    const int status = std::system(command.c_str());
    if (status != 0) {
        throw std::runtime_error("Matplotlib plotting failed. Check Python/matplotlib in the selected environment.");
    }
}

}  // namespace

void visualize_results(const Results& results, const RuntimeOptions& runtime) {
    if (!runtime.enable_visualization) {
        return;
    }

    const auto out_dir = default_output_dir();
    ensure_directory(out_dir);

    const std::vector<double> target_real = real_part_vector(results.target_clean);
    const std::vector<double> t_active = slice_vector(results.t, results.activity_start_idx, results.activity_end_idx);
    const std::vector<double> t_inst_active =
        slice_vector(results.t_inst, results.activity_inst_start_idx, results.activity_inst_end_idx);

    save_series_csv(out_dir / "target_clean_real.csv", "t_s", "target_real", results.t, target_real);
    save_series_csv(out_dir / "received_signal.csv", "t_s", "rx", results.t, results.rx);
    save_series_csv(out_dir / "filtered_signal.csv", "t_s", "filtered", results.t, results.rx_denoised);
    save_series_csv(out_dir / "spline_smoothed_signal.csv", "t_s", "smooth", results.t, results.rx_smooth);
    save_series_csv(out_dir / "inst_freq.csv", "t_s", "inst_freq_hz", results.t_inst, results.inst_freq);
    save_series_csv(out_dir / "inst_freq_kf.csv", "t_s", "inst_freq_kf_hz", t_inst_active, results.inst_freq_kf);
    save_series_csv(out_dir / "autocorr_pos.csv", "lag_s", "autocorr", results.lags_pos, results.autocorr_pos);
    save_series_csv(out_dir / "psd_pos.csv", "freq_hz", "psd", results.freqs_pos, results.psd_pos);
    save_series_csv(out_dir / "ridge_freq.csv", "t_s", "ridge_freq_hz", results.t, results.ridge_freq);
    save_series_csv(out_dir / "ridge_freq_kf.csv", "t_s", "ridge_freq_kf_hz", t_active, results.ridge_freq_kf);
    save_vector_csv(out_dir / "pseudo_freqs.csv", "pseudo_freq_hz", results.pseudo_freqs);
    save_index_csv(out_dir / "mod_peaks.csv", "index", results.mod_peaks);
    save_index_csv(out_dir / "time_peaks.csv", "index", results.time_peaks);
    save_index_csv(out_dir / "freq_peaks.csv", "index", results.freq_peaks);
    save_raw_matrix_f32(out_dir / "cwt_power.bin", results.cwt_power);
    save_plot_metadata(out_dir / "plot_metadata.txt", results);

    launch_matplotlib_visualization(runtime, out_dir);
}

RuntimeOptions parse_runtime_options(int argc, char** argv) {
    RuntimeOptions runtime;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--no-viz") {
            runtime.enable_visualization = false;
        } else if (arg == "--headless") {
            runtime.headless = true;
        } else if (arg == "--plot-python") {
            if (i + 1 >= argc) {
                throw std::invalid_argument("--plot-python requires a following path");
            }
            runtime.plot_python = argv[++i];
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: task2 [--no-viz] [--headless] [--plot-python <path>]\n";
            std::exit(0);
        } else {
            throw std::invalid_argument("Unknown argument: " + arg);
        }
    }
    return runtime;
}

}  // namespace task2_cpu

using namespace task2_cpu;

// =========================
// 9. 主流程
// =========================
int main(int argc, char** argv) {
    const RuntimeOptions runtime = parse_runtime_options(argc, argv);
    const auto t0 = std::chrono::steady_clock::now();
    print_parameter_summary();

    // ---- 1) 建模 ----
    const std::vector<double> t = build_time_axis(params.fs, params.duration);
    std::vector<Complex> target_clean;
    std::vector<Complex> echo_complex;
    std::vector<double> rx_noiseless;
    std::vector<double> rx;
    synthesize_received_signal(t, target_clean, echo_complex, rx_noiseless, rx);

    // ---- 2) 预处理：detrend + firwin bandpass + filtfilt ----
    std::vector<double> rx_detrended;
    std::vector<double> rx_denoised;
    preprocess_signal(rx, rx_detrended, rx_denoised);

    // ---- 3) 样条平滑 ----
    const std::vector<double> rx_smooth = smooth_with_cubic_bspline(rx_denoised);

    // ---- 4) 从平滑结果中估计主活动区 ----
    std::vector<double> activity_envelope;
    std::vector<std::uint8_t> activity_mask;
    int activity_start_idx = 0;
    int activity_end_idx = 0;
    estimate_activity_interval(rx_smooth, activity_envelope, activity_mask, activity_start_idx, activity_end_idx);
    const double est_start_s = t.empty() ? std::numeric_limits<double>::quiet_NaN() : t[static_cast<std::size_t>(activity_start_idx)];
    const double est_end_s = t.empty()
                                 ? std::numeric_limits<double>::quiet_NaN()
                                 : t[static_cast<std::size_t>(std::max(activity_start_idx, activity_end_idx - 1))];

    // ---- 5) 四域特征提取 ----
    std::vector<Complex> analytic;
    std::vector<double> t_inst;
    std::vector<double> inst_freq;
    extract_modulation_domain(rx_smooth, t, analytic, t_inst, inst_freq);

    std::vector<double> lags_pos;
    std::vector<double> autocorr_pos;
    extract_time_domain(rx_smooth, lags_pos, autocorr_pos);

    std::vector<double> freqs_pos;
    std::vector<double> psd_pos;
    extract_frequency_domain(rx_smooth, freqs_pos, psd_pos);

    std::vector<int> widths;
    std::vector<double> pseudo_freqs;
    Matrix<float> cwt_power;
    std::vector<double> ridge_freq;
    std::vector<double> ridge_energy;
    extract_time_frequency_domain(rx_smooth, widths, pseudo_freqs, cwt_power, ridge_freq, ridge_energy);

    // ---- 6) 峰值检测 ----
    std::vector<int> mod_peaks = detect_peaks_1d(abs_vector(inst_freq), params.mod_peak_order, 0.35);
    if (mod_peaks.empty()) {
        mod_peaks = fallback_global_peak(abs_vector(inst_freq));
    }

    std::vector<int> time_peaks = detect_peaks_1d(autocorr_pos, params.time_peak_order, 0.20);
    if (!time_peaks.empty()) {
        std::vector<int> filtered;
        for (int idx : time_peaks) {
            if (lags_pos[static_cast<std::size_t>(idx)] >= params.min_period_s) {
                filtered.push_back(idx);
            }
        }
        time_peaks = std::move(filtered);
    }

    std::vector<int> freq_peaks = detect_peaks_1d(psd_pos, params.freq_peak_order, 0.15);
    if (freq_peaks.empty()) {
        freq_peaks = fallback_global_peak(psd_pos);
    }

    // ---- 7) Kalman 估计 ----
    const int activity_inst_start_idx = std::max(0, activity_start_idx - 1);
    const int activity_inst_end_idx = std::min(static_cast<int>(inst_freq.size()), std::max(0, activity_end_idx - 1));
    const std::vector<double> inst_freq_active = slice_vector(inst_freq, activity_inst_start_idx, activity_inst_end_idx);
    const std::vector<double> t_inst_active = slice_vector(t_inst, activity_inst_start_idx, activity_inst_end_idx);
    const std::vector<double> inst_freq_kf = kalman_smooth_1d(inst_freq_active);

    double mod_slope = std::numeric_limits<double>::quiet_NaN();
    double mod_intercept = std::numeric_limits<double>::quiet_NaN();
    std::string modulation_label = "Unknown";
    if (inst_freq_kf.size() > 1) {
        const auto fit = linear_fit(t_inst_active, inst_freq_kf);
        mod_slope = fit.first;
        mod_intercept = fit.second;
        modulation_label = (std::abs(mod_slope) > 1e6) ? "LFM-like" : "Non-LFM-like";
    }

    std::vector<double> period_candidates;
    if (time_peaks.size() >= 2) {
        std::vector<double> lag_candidates;
        lag_candidates.reserve(time_peaks.size());
        for (int idx : time_peaks) {
            lag_candidates.push_back(lags_pos[static_cast<std::size_t>(idx)]);
        }
        period_candidates = diff_vector(lag_candidates);
    } else if (time_peaks.size() == 1) {
        period_candidates = {lags_pos[static_cast<std::size_t>(time_peaks[0])]};
    }

    const std::vector<double> period_kf = kalman_smooth_1d(period_candidates);
    const double est_period_s = period_kf.empty() ? std::numeric_limits<double>::quiet_NaN() : period_kf.back();

    std::vector<double> peak_power;
    peak_power.reserve(freq_peaks.size());
    for (int idx : freq_peaks) {
        peak_power.push_back(psd_pos[static_cast<std::size_t>(idx)]);
    }
    const std::vector<int> sort_idx = argsort_descending(peak_power);
    std::vector<double> freq_candidates;
    for (int order_idx : sort_idx) {
        if (static_cast<int>(freq_candidates.size()) >= params.freq_peak_topk) {
            break;
        }
        const int peak_idx = freq_peaks[static_cast<std::size_t>(order_idx)];
        freq_candidates.push_back(freqs_pos[static_cast<std::size_t>(peak_idx)]);
    }
    const std::vector<double> freq_kf = kalman_smooth_1d(freq_candidates);
    const double est_main_freq_hz = freq_kf.empty() ? std::numeric_limits<double>::quiet_NaN() : freq_kf.back();

    const std::vector<double> ridge_active = slice_vector(ridge_freq, activity_start_idx, activity_end_idx);
    const std::vector<double> t_active = slice_vector(t, activity_start_idx, activity_end_idx);
    const std::vector<double> ridge_freq_kf = kalman_smooth_1d(ridge_active);

    double ridge_slope = std::numeric_limits<double>::quiet_NaN();
    double ridge_intercept = std::numeric_limits<double>::quiet_NaN();
    double est_bandwidth_hz = std::numeric_limits<double>::quiet_NaN();
    double est_center_freq_hz = std::numeric_limits<double>::quiet_NaN();
    if (ridge_freq_kf.size() > 1) {
        const auto fit = linear_fit(t_active, ridge_freq_kf);
        ridge_slope = fit.first;
        ridge_intercept = fit.second;
        const auto [ridge_min_it, ridge_max_it] = std::minmax_element(ridge_freq_kf.begin(), ridge_freq_kf.end());
        est_bandwidth_hz = *ridge_max_it - *ridge_min_it;
        est_center_freq_hz = 0.5 * (*ridge_max_it + *ridge_min_it);
    }

    const TruthSummary truth = build_truth_summary();
    const auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    // ---- 8) 控制台输出 ----
    std::cout << "========== 理论真值 ==========\n";
    std::cout << "LFM start/end = " << std::fixed << std::setprecision(3) << (truth.lfm_start_s * 1e3) << " / "
              << (truth.lfm_end_s * 1e3) << " ms\n";
    std::cout << "LFM f0/f1 = " << std::fixed << std::setprecision(3) << (truth.lfm_f0_hz / 1e6) << " / "
              << (truth.lfm_f1_hz / 1e6) << " MHz\n";
    std::cout << "LFM bandwidth = " << std::fixed << std::setprecision(3) << (truth.lfm_bw_hz / 1e6) << " MHz\n";
    std::cout << "Gaussian centers = " << std::fixed << std::setprecision(3) << (truth.gp_early_center_s * 1e3) << " / "
              << (truth.gp_lfm_center_s * 1e3) << " ms\n";
    std::cout << "Square interference period = " << std::fixed << std::setprecision(3) << (truth.interference_period_s * 1e6)
              << " us\n\n";

    std::cout << "========== 估计结果 ==========\n";
    std::cout << "调制类型判定 = " << modulation_label << "\n";
    std::cout << "主活动区开始/结束 = " << std::fixed << std::setprecision(3) << (est_start_s * 1e3) << " / "
              << (est_end_s * 1e3) << " ms\n";
    std::cout << "活动区脉冲宽度 = " << std::fixed << std::setprecision(3) << ((est_end_s - est_start_s) * 1e3) << " ms\n";
    if (std::isfinite(est_main_freq_hz)) {
        std::cout << "频域主峰估计 = " << std::fixed << std::setprecision(3) << (est_main_freq_hz / 1e6) << " MHz\n";
    } else {
        std::cout << "频域主峰估计 = NaN\n";
    }
    if (std::isfinite(est_period_s)) {
        std::cout << "稳定周期估计 = " << std::fixed << std::setprecision(3) << (est_period_s * 1e6) << " us\n";
    } else {
        std::cout << "稳定周期估计 = NaN\n";
    }
    if (std::isfinite(est_center_freq_hz)) {
        std::cout << "时频中心频率估计 = " << std::fixed << std::setprecision(3) << (est_center_freq_hz / 1e6) << " MHz\n";
        std::cout << "时频带宽估计 = " << std::fixed << std::setprecision(3) << (est_bandwidth_hz / 1e6) << " MHz\n";
    } else {
        std::cout << "时频中心频率估计 = NaN\n";
        std::cout << "时频带宽估计 = NaN\n";
    }
    if (std::isfinite(ridge_slope)) {
        std::cout << "时频 ridge 斜率估计 = " << std::scientific << std::setprecision(3) << ridge_slope << " Hz/s\n";
    } else {
        std::cout << "时频 ridge 斜率估计 = NaN\n";
    }
    if (std::isfinite(mod_slope)) {
        std::cout << "调制域瞬时频率斜率估计 = " << std::scientific << std::setprecision(3) << mod_slope << " Hz/s\n";
    } else {
        std::cout << "调制域瞬时频率斜率估计 = NaN\n";
    }
    std::cout << "\n";
    std::cout << std::fixed << std::setprecision(6) << "Processing time before visualization: " << elapsed << " s\n\n";

    Results results;
    results.t = t;
    results.target_clean = target_clean;
    results.echo_complex = echo_complex;
    results.rx_noiseless = rx_noiseless;
    results.rx = rx;
    results.rx_detrended = rx_detrended;
    results.rx_denoised = rx_denoised;
    results.rx_smooth = rx_smooth;
    results.activity_envelope = activity_envelope;
    results.analytic = analytic;
    results.t_inst = t_inst;
    results.inst_freq = inst_freq;
    results.inst_freq_kf = inst_freq_kf;
    results.lags_pos = lags_pos;
    results.autocorr_pos = autocorr_pos;
    results.freqs_pos = freqs_pos;
    results.psd_pos = psd_pos;
    results.widths = widths;
    results.pseudo_freqs = pseudo_freqs;
    results.cwt_power = cwt_power;
    results.ridge_freq = ridge_freq;
    results.ridge_energy = ridge_energy;
    results.ridge_freq_kf = ridge_freq_kf;
    results.mod_peaks = mod_peaks;
    results.time_peaks = time_peaks;
    results.freq_peaks = freq_peaks;
    results.activity_start_idx = activity_start_idx;
    results.activity_end_idx = activity_end_idx;
    results.activity_inst_start_idx = activity_inst_start_idx;
    results.activity_inst_end_idx = activity_inst_end_idx;
    results.est_start_s = est_start_s;
    results.est_end_s = est_end_s;
    results.est_main_freq_hz = est_main_freq_hz;
    results.est_period_s = est_period_s;
    results.est_center_freq_hz = est_center_freq_hz;
    results.est_bandwidth_hz = est_bandwidth_hz;
    results.mod_slope_hz_per_s = mod_slope;
    results.mod_intercept_hz = mod_intercept;
    results.ridge_slope_hz_per_s = ridge_slope;
    results.ridge_intercept_hz = ridge_intercept;
    results.modulation_label = modulation_label;
    results.truth = truth;

    // ---- 9) 可视化与结果输出 ----
    visualize_results(results, runtime);
    return 0;
}
