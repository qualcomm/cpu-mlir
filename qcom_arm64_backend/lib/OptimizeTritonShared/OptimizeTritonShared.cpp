//===- OptimizeTritonShared.cpp -------------------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/cpu-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

#include "OptimizeTritonShared/OptimizeTritonShared.h"

#include "mlir/Conversion/Passes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Dialect/ArmSME/Transforms/Passes.h"
#include "mlir/Dialect/ArmSVE/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/Pipelines/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Affine/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include "mlir/InitAllPasses.h"
#include "llvm/Support/Debug.h"

#include "OptimizeTritonShared/Passes.h"

namespace mlir {
namespace triton {

#define GEN_PASS_DEF_ARM64OPTIMIZETRITONSHARED
#include "OptimizeTritonShared/Passes.h.inc"

static void addOptimizationPassesForSME(mlir::OpPassManager &pm) {
  pm.addPass(createCanonicalizerPass());

  pm.addPass(arm_sme::createVectorLegalizationPass());
  pm.addPass(createConvertVectorToArmSMEPass());
  pm.addNestedPass<func::FuncOp>(arm_sme::createOuterProductFusionPass());
  pm.addNestedPass<func::FuncOp>(arm_sve::createLegalizeVectorStoragePass());
  pm.addPass(createArithToArmSMEConversionPass());
  pm.addPass(createConvertArmSMEToSCFPass());
  pm.addPass(createConvertVectorToSCFPass(
    VectorTransferToSCFOptions().enableFullUnroll()));
  pm.addPass(createArithToArmSMEConversionPass());

  // Enable streaming-mode and ZA.
  pm.addNestedPass<func::FuncOp>(arm_sme::createEnableArmStreamingPass(
    arm_sme::ArmStreamingMode::Streaming, arm_sme::ArmZaMode::NewZA,
    /*ifRequiredByOps=*/true));

  // Lower any remaining linalg.generic ops (e.g. masked gathers that couldn't
  // be vectorized) to SCF loops. linalg.generic does not support multiple
  // blocks in its body region, but createSCFToControlFlowPass would produce
  // them if scf.if ops remain inside linalg.generic bodies.
  pm.addPass(createConvertLinalgToLoopsPass());

  // Convert SCF to CF (required for ArmSME tile allocation).
  pm.addPass(createSCFToControlFlowPass());
  pm.addNestedPass<func::FuncOp>(createConvertArmSMEToLLVMPass());

  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
}

ModuleOp arm64OptimizeTritonShared(ModuleOp mod, arm64VectorExtension vec_ext,
    int verbose_level, int debug_mode) {
  MLIRContext* context = mod->getContext();
  mlir::PassManager pm(context);

  std::string targetTripleStr = "aarch64";
  mod->setAttr(LLVM::LLVMDialect::getTargetTripleAttrName(),
               StringAttr::get(mod->getContext(), targetTripleStr));

  if (verbose_level > 0) {
    llvm::dbgs() << "\nOptimizing Triton-Shared for "
                 << ARM64VectorExtensionNames[vec_ext] << "\n";
  }

  addOptimizeTritonSharedPasses(pm, vec_ext);
  enablePrintIR(pm, debug_mode);

  if (failed(pm.run(mod))) {
    llvm::errs() << "Triton Shared Optimization Pass execution failed.\n";
    return nullptr;
  }
  return mod;
}

void addOptimizeTritonSharedPasses(mlir::OpPassManager &pm,
    arm64VectorExtension vec_ext) {
  // triton-shared emits one linalg.generic per elementwise op. Fusing them
  // avoids temporaries and keeps long chains (e.g. GELU) within the greedy
  // rewrite limit of bubble-up-extract-slice.
  pm.addPass(createLinalgElementwiseOpFusionPass());
  pm.addPass(createCanonicalizerPass());
  pm.addNestedPass<func::FuncOp>(createTritonARM64PuntBufferPass());
  pm.addNestedPass<func::FuncOp>(createTritonARM64RVOPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  pm.addNestedPass<func::FuncOp>(createTritonARM64BubbleUpExtractSlicePass());
  pm.addPass(createTritonARM64MatmulToMmt4DPass());
  pm.addPass(createTritonARM64LinalgTilingPass({vec_ext}));
  pm.addNestedPass<func::FuncOp>(affine::createLoopCoalescingPass());
  pm.addPass(createTritonARM64LinalgVectorizationPass({vec_ext}));
  pm.addPass(bufferization::createEmptyTensorEliminationPass());
  pm.addPass(createTritonARM64VectorLoweringPass({vec_ext}));
  // The vector lowering pass bufferizes, but bufferization never frees what it
  // allocates, so every program instance leaked its temporaries. This must run
  // while loops are still structured (SME lowers them to cf below).
  bufferization::buildBufferDeallocationPipeline(pm);
  // Deallocs whose aliasing is only known at runtime (e.g. buffers returned
  // from loops) call a generated @dealloc_helper. Inline it: the driver
  // expects a single function per kernel module.
  pm.addPass(createInlinerPass());

  if (vec_ext == SME)
    addOptimizationPassesForSME(pm);

  // Add `fastmath` attribute to ops. E.g.: summation can use pair-wise `faddp`
  // (O(log n)) instead of sequential `fadda` (O(n)).
  // pm.addPass(createTritonARM64AddFastMathPass());
}

} // namespace triton
} // namespace mlir
