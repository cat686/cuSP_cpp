#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
build_dir="${1:-${script_dir}/build}"
build_type="${CMAKE_BUILD_TYPE:-Release}"
generator="${CMAKE_GENERATOR:-Unix Makefiles}"

cmake -S "${script_dir}" \
      -B "${build_dir}" \
      -G "${generator}" \
      -DCMAKE_BUILD_TYPE="${build_type}"

cmake --build "${build_dir}" --parallel
