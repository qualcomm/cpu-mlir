//===- Passes.td ----------------------------------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/cpu-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_ARM64_LOWER_TRITON_LINALG_TO_LLVM_PASSES
#define TRITON_ARM64_LOWER_TRITON_LINALG_TO_LLVM_PASSES

include "mlir/Pass/PassBase.td"

def TritonARM64LowerTmTensorPass :
    InterfacePass<"triton-arm64-lower-tm-tensor", "mlir::FunctionOpInterface"> {
  let summary = "Lower tm-tensor ops to mlir dialect ops";
  let constructor = "triton::createTritonARM64LowerTmTensorPass()";
}

def TritonARM64ProfPass :
    Pass<"triton-arm64-prof-instrumentation", "func::FuncOp"> {
  let summary = "Pass to instrument and profile with arm64 instrinsic";

  let options = [
  Option<"prof_disable_loop", "disable-loop", "bool",
          /*default=*/"false",
          "Turn on or off loop level instrumentation">,
  Option<"prof_loop_depth", "loop-depth", "int",
          /*default=*/"1",
          "Depth level instrumentation">,
  ];

  let constructor = "triton::createTritonARM64ProfPass()";
}

def TritonARM64LowerConstantsSeparately :
    Pass<"triton-arm64-lower-constants-separately", "mlir::ModuleOp"> {
  let summary = "Lowers constants separately from the code, possibly in multiple modules if above a certain threshold";
  let constructor = "mlir::triton::createLowerConstantsSeparatelyPass()";
}

def TritonARM64MathApproximation :
    Pass<"triton-arm64-math-approximation", "mlir::ModuleOp"> {
  let summary = "Expand selected math ops into polynomial approximations";
  let description = [{
    Rewrites math ops into arith/vector ops (upstream MLIR polynomial
    approximations), so they stay vectorized instead of becoming per-lane
    libm calls. f16/bf16 ops are computed in f32. Must run before
    convert-math-to-llvm; math.erf has no LLVM lowering otherwise.
  }];

  let options = [
  ListOption<"ops", "ops", "std::string",
             "Math op names to approximate (default: math.erf,math.exp,math.tanh)">,
  ];

  let constructor = "mlir::triton::createTritonARM64MathApproximationPass()";
}

#endif
