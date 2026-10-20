# ===- run_pytest.sh --------------------------------------------------------===
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause.
# For more license information:
#   https://github.com/qualcomm/cpu-mlir/LICENSE.txt
#
# ===------------------------------------------------------------------------===

#!/bin/bash 
SCRIPT_DIR="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
ROOT_DIR="$SCRIPT_DIR/.."

set -x
#Include dirs
export ARM64_BACKEND_PATH=$SCRIPT_DIR/../..
export TRITON_SHARED_ROOT=$ARM64_BACKEND_PATH/../triton-shared/
#Library dirs
CLANG_VERSION=$($TARGET_COMPILER/bin/clang --version | grep "clang version" | sed "s/.*version //" | sed "s/\.[0-9].*//")
CLANG_RT_SUBDIR=$(ls "$TARGET_COMPILER/lib/clang/$CLANG_VERSION/lib/" | grep aarch64 | tail -1)
export LAUNCHER_LIBPATHS="$TARGET_COMPILER/lib $TARGET_COMPILER/$CLANG_RT_SUBDIR/lib \
	                  $TARGET_COMPILER/lib/clang/$CLANG_VERSION/lib/$CLANG_RT_SUBDIR"

#Device mode env vars
# export RUN_ON_SIM=0
# export ADB_PATH=<HOST_ADB_PATH>
# export DEV_HOST=<DEVICE_HOST_NAME>
# export DEV_ID=<DEVICE_ID>

#Benchmarking mode
#export TRITON_ARM64_BENCHMARK=1
#export TRITON_ARM64_BENCHMARK_ITERATIONS=10

#Simulator mode env vars
# export RUN_ON_SIM=1
# export QEMU_PATH=<QEMU_AARCH64_PATH>

#Debugging options
#export TRITON_ARM64_VERBOSE=1
#export TRITON_ARM64_SAVE_TEMPS=0
#export TRITON_ARM64_DEBUG=1

if [[ "$TRITON_ARM64_DEBUG" == 1 ]]; then
    export TRITON_DISABLE_LINE_INFO=0
fi

# export MLIR_ENABLE_DUMP=1   # 1 = dumps MLIR for all high level TTIR passes
# export MLIR_ENABLE_DUMP_TRITON_SHARED=1   # 1 = dumps MLIR for Triton-shared passes
# export MLIR_DUMP_TRITON_SHARED_SPECIFIC_PASS="ConvertVectorToLLVMPass" # Specific single pass MLIR dump
# export MLIR_DUMP_ALL_IR_DATA=1 # Dumps all constant data (default omits constant data from dump)

# Keep benchmark output visible without requiring callers to remember -s.
pytest -s --tb=short "$@"
