//===- ARM64MathApproximation.cpp -----------------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/cpu-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// Expands selected math ops into upstream MLIR polynomial approximations.
// Without this, math.erf cannot be lowered to LLVM at all, and math.exp on
// scalable vectors becomes llvm.exp.nxv*, which the AArch64 backend
// scalarizes into one libm call per lane.
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Math/Transforms/Passes.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/StringSet.h"

#include "LowerTritonLinalgToLLVM/LowerTritonLinalgToLLVM.h"

using namespace mlir;
using namespace triton;

#define GEN_PASS_DEF_TRITONARM64MATHAPPROXIMATION
#include "LowerTritonLinalgToLLVM/Passes.h.inc"

namespace {

// Ops used by activation functions. Other ops keep their libm lowering until
// their approximation accuracy has been validated on this backend.
constexpr StringLiteral kDefaultOps[] = {"math.erf", "math.exp", "math.tanh"};

struct TritonARM64MathApproximationPass
    : public ::impl::TritonARM64MathApproximationBase<
          TritonARM64MathApproximationPass> {
  using TritonARM64MathApproximationBase::TritonARM64MathApproximationBase;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, math::MathDialect,
                    vector::VectorDialect>();
  }

  void runOnOperation() override {
    llvm::StringSet<> selected;
    if (ops.empty())
      selected.insert(std::begin(kDefaultOps), std::end(kDefaultOps));
    else
      selected.insert(ops.begin(), ops.end());
    auto predicate = [&](StringRef name) { return selected.contains(name); };

    RewritePatternSet patterns(&getContext());
    // f16/bf16 -> f32, then the f32 polynomial.
    populateMathF32ExpansionPatterns(patterns, predicate);
    populateMathPolynomialApproximationPatterns(patterns, predicate);
    // Not an approximation, so always on: math.fpowi on vectors has no LLVM
    // lowering (llvm.intr.powi takes a scalar exponent). A constant power
    // becomes multiplies (torch-mlir emits x^3 for GELU tanh), any other powf.
    math::populateExpansionPatterns(patterns, {"fpowi"});
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
triton::createTritonARM64MathApproximationPass(
    const TritonARM64MathApproximationOptions &options) {
  return std::make_unique<TritonARM64MathApproximationPass>(options);
}
