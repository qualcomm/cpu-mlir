//===- arm64-mlir-opt.cpp -------------------------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/cpu-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/Debug.h"

#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/ArmSME/IR/ArmSME.h"
#include "mlir/Dialect/ArmSVE/IR/ArmSVEDialect.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "LowerTritonLinalgToLLVM/LowerTritonLinalgToLLVM.h"
#include "LowerTritonLinalgToLLVM/Passes.h"
#include "OptimizeTritonShared/OptimizeTritonShared.h"
#include "OptimizeTritonShared/Passes.h"

#include <iostream>
#include <string>
#include <unordered_map>

#define DEBUG_TYPE "arm64-milr-opt"

using namespace mlir;

namespace {
struct ConditionalCWrappersPass
    : public PassWrapper<ConditionalCWrappersPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConditionalCWrappersPass)
  void runOnOperation() override {
    getOperation().walk([](func::FuncOp func) {
      if (!func.getResultTypes().empty())
        func->setAttr(LLVM::LLVMDialect::getEmitCWrapperAttrName(),
                      UnitAttr::get(func->getContext()));
    });
  }
};
} // namespace

static llvm::cl::opt<arm64VectorExtension> vectorExt{
    "vector-ext",
    llvm::cl::desc("ARM64 vector extension to generate code for"),
    llvm::cl::init(arm64VectorExtension::SME),
    llvm::cl::values(
      clEnumValN(arm64VectorExtension::NEON, "neon", "Generate code for NEON only"),
      clEnumValN(arm64VectorExtension::SVE, "sve", "Generate code for SVE and NEON"),
      clEnumValN(arm64VectorExtension::SME, "sme", "Generate code for SME, SVE and NEON")
    )
  };

static llvm::cl::opt<bool> profDisableLoop{
    "prof-disable-loop",
    llvm::cl::desc("Disable loop-level profiling instrumentation"),
    llvm::cl::init(false)};

static llvm::cl::opt<int> profLoopDepth{
    "prof-loop-depth",
    llvm::cl::desc("Loop depth at which to insert profiling instrumentation"),
    llvm::cl::init(1)};

int main(int argc, char **argv) {
  DialectRegistry registry;
  registerAllDialects(registry);
  registerAllPasses();
  registerAllExtensions(registry);

  // register non-standard dialects
  // registry.insert<tm_tensor::TmTensorDialect>();
  registry.insert<triton::TritonDialect>();

  // register our passes
  triton::registerTritonARM64MatmulToMmt4D();
  triton::registerTritonARM64LinalgTiling();
  triton::registerTritonARM64LinalgVectorization();
  triton::registerTritonARM64VectorLowering();
  triton::registerTritonARM64ProfPass();
  triton::registerTritonARM64MathApproximation();
  triton::registerTritonARM64RVO();
  triton::registerTritonARM64PuntBuffer();
  triton::registerTritonARM64BubbleUpExtractSlice();

  // register our main pass pipelines directly into the outer PassManager,
  // avoiding nested PassManager creation inside runOnOperation()
  PassPipelineRegistration<> optimizePipeline(
      "optimize-triton-shared", "Runs optimize triton shared pipeline",
      [](OpPassManager &pm) {
        triton::addOptimizeTritonSharedPasses(pm, vectorExt);
      });

  PassPipelineRegistration<> lowerPipeline(
      "triton-linalg-to-llvm", "Runs triton linalg to llvm mlir pipeline",
      [](OpPassManager &pm) {
        triton::TritonARM64ProfPassOptions profOpts;
        profOpts.prof_disable_loop = profDisableLoop;
        profOpts.prof_loop_depth = profLoopDepth;
        pm.addPass(std::make_unique<ConditionalCWrappersPass>());
        triton::addTritonLinalgToLLVMPasses(pm, vectorExt, profOpts,
                                            /*debug_mode=*/0);
      });

  return asMainReturnCode(MlirOptMain(
      argc, argv, "ARM64 MLIR opt\n", registry));
}
