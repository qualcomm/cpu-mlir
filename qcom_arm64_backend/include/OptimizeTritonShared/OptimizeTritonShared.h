//===- OptimizeTritonShared.h ---------------------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/cpu-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_ARM64_OPTIMIZE_TRITON_SHARED_H
#define TRITON_ARM64_OPTIMIZE_TRITON_SHARED_H
#include "common.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

namespace mlir {
namespace triton {
#define GEN_PASS_DECL
#include "OptimizeTritonShared/Passes.h.inc"

std::unique_ptr<OperationPass<ModuleOp>> createTritonARM64LinalgTilingPass(
    const TritonARM64LinalgTilingOptions &options = TritonARM64LinalgTilingOptions()
);
std::unique_ptr<OperationPass<ModuleOp>> createTritonARM64LinalgVectorizationPass(
    const TritonARM64LinalgVectorizationOptions &options = TritonARM64LinalgVectorizationOptions()
);
std::unique_ptr<OperationPass<ModuleOp>> createTritonARM64VectorLoweringPass(
    const TritonARM64VectorLoweringOptions &options = TritonARM64VectorLoweringOptions()
);
std::unique_ptr<OperationPass<ModuleOp>> createTritonARM64MatmulToMmt4DPass();
std::unique_ptr<OperationPass<ModuleOp>> createTritonARM64SMEFuncOutlinePass();

std::unique_ptr<OperationPass<func::FuncOp>> createTritonARM64RVOPass();

std::unique_ptr<OperationPass<func::FuncOp>> createTritonARM64PuntBufferPass();

std::unique_ptr<OperationPass<func::FuncOp>> createTritonARM64BubbleUpExtractSlicePass();

// Transposes with fewer elements than this keep the regular SVE path: below
// it the per-call cost of streaming mode outweighs the ZA tile transpose
// (measured on device: about even at 262K elements, 1.6x faster at 786K).
constexpr int64_t kSMETransposeMinElements = 1 << 19;

// Set by the tiling pass on each tile of a transpose it tiled for SME, holding
// the tile size per vscale. The vectorization pass, which only sees
// dynamically shaped tiles, vectorizes a marked tile as one ZA tile.
constexpr llvm::StringLiteral kSMETransposeTileAttr = "arm64.sme_transpose_tile";

} // namespace triton
} // namespace mlir

namespace mlir {
class ModuleOp;
} // namespace mlir

namespace mlir {
namespace triton {

ModuleOp arm64OptimizeTritonShared(ModuleOp module, arm64VectorExtension vec_ext,
    int verbose_level, int debug_mode);

void addOptimizeTritonSharedPasses(mlir::OpPassManager &pm,
    arm64VectorExtension vec_ext);

} // namespace triton
} // namespace mlir

#endif // TRITON_ARM64_OPTIMIZE_TRITON_SHARED_H
