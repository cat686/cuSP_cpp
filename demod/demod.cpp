#include "demod.h"

#include "../common/fftw_helpers.h"

namespace cuSP::demod {

void extract_modulation_domain(const std::vector<double>& x_smooth,
                               const std::vector<double>& t,
                               double fs,
                               std::vector<Complex>& analytic,
                               std::vector<double>& t_inst,
                               std::vector<double>& inst_freq) {
    analytic = cuSP::common::hilbert_transform(x_smooth);
    std::vector<double> phase(analytic.size(), 0.0);
    for (std::size_t i = 0; i < analytic.size(); ++i) {
        phase[i] = std::arg(analytic[i]);
    }
    const std::vector<double> unwrapped = cuSP::common::unwrap_phase(phase);
    const std::vector<double> phase_diff = cuSP::common::diff_vector(unwrapped);

    inst_freq.resize(phase_diff.size(), 0.0);
    for (std::size_t i = 0; i < phase_diff.size(); ++i) {
        inst_freq[i] = phase_diff[i] * fs / (2.0 * cuSP::common::kPi);
    }

    t_inst.clear();
    if (t.size() > 1) {
        t_inst.insert(t_inst.end(), t.begin() + 1, t.end());
    }
}

}  // namespace cuSP::demod
