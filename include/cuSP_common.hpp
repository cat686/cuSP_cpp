#pragma once

#include "common/common.h"

#include <complex>
#include <type_traits>
#include <vector>

namespace cuSP::common {

template <typename T>
struct is_std_complex : std::false_type {};

template <typename T>
struct is_std_complex<std::complex<T>> : std::true_type {};

template <typename Out, typename In>
inline Out value_cast(const In& value) {
    if constexpr (std::is_same_v<Out, In>) {
        return value;
    } else if constexpr (is_std_complex<Out>::value && is_std_complex<In>::value) {
        using OutScalar = typename Out::value_type;
        return Out(static_cast<OutScalar>(value.real()), static_cast<OutScalar>(value.imag()));
    } else {
        return static_cast<Out>(value);
    }
}

template <typename Out, typename In>
inline std::vector<Out> cast_vector(const std::vector<In>& src) {
    std::vector<Out> out;
    out.reserve(src.size());
    for (const auto& value : src) {
        out.push_back(value_cast<Out>(value));
    }
    return out;
}

template <template <typename> class MatrixLike, typename T>
inline Matrix<T> to_common_matrix(const MatrixLike<T>& src) {
    Matrix<T> out(src.rows, src.cols);
    out.data = src.data;
    return out;
}

template <typename OutT, template <typename> class MatrixLike, typename InT>
inline Matrix<OutT> to_common_matrix_cast(const MatrixLike<InT>& src) {
    Matrix<OutT> out(src.rows, src.cols);
    out.data = cast_vector<OutT>(src.data);
    return out;
}

template <template <typename> class MatrixLike, typename T>
inline MatrixLike<T> from_common_matrix(const Matrix<T>& src) {
    MatrixLike<T> out;
    out.rows = src.rows;
    out.cols = src.cols;
    out.data = src.data;
    return out;
}

template <template <typename> class MatrixLike, typename OutT, typename InT>
inline MatrixLike<OutT> from_common_matrix_cast(const Matrix<InT>& src) {
    MatrixLike<OutT> out;
    out.rows = src.rows;
    out.cols = src.cols;
    out.data = cast_vector<OutT>(src.data);
    return out;
}

inline std::vector<float> make_window_f32(const std::string& name, std::size_t n) {
    return cast_vector<float>(make_window(name, n));
}

inline void fft_inplace(std::vector<std::complex<float>>& a, bool inverse) {
    std::vector<Complex> tmp = cast_vector<Complex>(a);
    fft_inplace(tmp, inverse);
    a = cast_vector<std::complex<float>>(tmp);
}

}  // namespace cuSP::common
