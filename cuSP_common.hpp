#pragma once

#include "common/common.h"

namespace cuSP::common {

template <template <typename> class MatrixLike, typename T>
inline Matrix<T> to_common_matrix(const MatrixLike<T>& src) {
    Matrix<T> out(src.rows, src.cols);
    out.data = src.data;
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

}  // namespace cuSP::common
