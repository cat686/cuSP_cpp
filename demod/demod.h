#pragma once

#include "../common/common.h"

#include <vector>

namespace cuSP::demod {

using cuSP::common::Complex;

void extract_modulation_domain(const std::vector<double>& x_smooth,
                               const std::vector<double>& t,
                               double fs,
                               std::vector<Complex>& analytic,
                               std::vector<double>& t_inst,
                               std::vector<double>& inst_freq);

}  // namespace cuSP::demod
