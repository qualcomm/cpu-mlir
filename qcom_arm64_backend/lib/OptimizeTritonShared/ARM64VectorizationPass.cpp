//===- ARM64Vectorization.cpp - Implementation of Vectorization Pass  -----===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/cpu-mlir/LICENSE.txt
//
// This file implements lowering of inner loop to vector form so that vector
// to llvm pass can convert to vectorized llvm-ir.
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/Support/Debug.h"

#include "OptimizeTritonShared/OptimizeTritonShared.h"

#define DEBUG_TYPE "triton-arm64-linalg-vectorize"

#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define DBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace triton;

#define GEN_PASS_DEF_TRITONARM64LINALGVECTORIZATION
#include "OptimizeTritonShared/Passes.h.inc"

namespace {

static LogicalResult vectorizeLinalgOp(linalg::LinalgOp op, arm64VectorExtension vectorExt) {
  auto fnName = op->getAttrOfType<StringAttr>("library_call");
  if (fnName) {
    DBG("-> skipping vectorization. Op will be replaced with a library call.");
    return failure();
  }
  IRRewriter rewriter(op.getContext());

  auto numLoops = op.getNumLoops();
  if (numLoops < 1) {
    return failure();
  }
  DBG("numLoops: " << numLoops << '\n');

  auto innerLoopDim = op.getStaticLoopRanges()[numLoops - 1];
  DBG("innerLoopDim: " << innerLoopDim << '\n');

  bool useScalableVector = ShapedType::isDynamic(innerLoopDim) &&
      (vectorExt != NEON) &&
      (linalg::isElementwise(op) || isa<linalg::ReduceOp>(op) ||
       isa<linalg::MatvecOp>(op) || isa<linalg::MatmulOp>(op));

  // masked.load/store of i1 vectors is not legal on aarch64; skip
  // vectorization and let the op lower to scalar loops instead.
  auto hasI1Tensor = [](ValueRange vals) {
    return llvm::any_of(vals, [](Value v) {
      auto t = dyn_cast<RankedTensorType>(v.getType());
      return t && t.getElementType().isInteger(1);
    });
  };
  if (hasI1Tensor(op->getResults()) || hasI1Tensor(op->getOperands())) {
    DBG("-> skipping vectorization. i1 tensors cannot use masked.load/store "
        "on aarch64.");
    return failure();
  }

  if (useScalableVector) {
    // Scalable vector.gather is not selectable by the AArch64 backend
    // (masked_gather with signed nxv4i64 byte offsets has no SVE instruction).
    // If the body contains tensor.extract, vectorization would create a
    // scalable vector.gather; skip so createConvertLinalgToLoopsPass can lower
    // it to scalar loops instead.
    bool hasTensorExtract = false;
    op.getBlock()->walk([&hasTensorExtract](tensor::ExtractOp) {
      hasTensorExtract = true;
      return WalkResult::interrupt();
    });
    if (hasTensorExtract) {
      DBG("-> skipping scalable vectorization. tensor.extract would produce "
          "an unselectable scalable vector.gather on AArch64.");
      return failure();
    }
  }

  auto innermostDim = numLoops - 1;
  SmallVector<int64_t> vecSizes(numLoops, 1);
  SmallVector<bool> scalableDims(numLoops, false);
  bool assumeDynamicDimsMatchVecSizes = false;
  if (isa<linalg::Conv1DNcwFcwOp>(op)) {
    // The preceding Conv2D schedule decomposes one output/filter-height tile
    // to NCW/FCW. Vectorize F and OW as a 4x4 outer product while retaining
    // C and KW as scalar reduction tiles.
    if (numLoops != 5)
      return failure();
    vecSizes = {1, 4, 4, 1, 1};
  } else if (isa<linalg::MatmulOp>(op) && vectorExt == SME) {
    scalableDims = {true, true, false};
    vecSizes = {8, 8, 2};
  } else if (isa<linalg::Mmt4DOp>(op) && vectorExt == SME) {
    scalableDims = {false, false, false, true, true, false};
    vecSizes = {1, 1, 1, 8, 8, 1};
    // linalg.mmt4d is used in conjunction with linalg.pack which has padding
    // semantics, so we can assume the dims will match vec sizes.
    assumeDynamicDimsMatchVecSizes = true;
  } else {
    vecSizes[innermostDim] = useScalableVector ? 4 : innerLoopDim;
    scalableDims[innermostDim] = useScalableVector ? true : false;
    DBG("Vector size: " << vecSizes[innermostDim] << '\n');
    DBG("Scalable: " << scalableDims[innermostDim] << '\n');
  }

  auto vectorResults =
      linalg::vectorize(rewriter, op, vecSizes, scalableDims,
                        /*vectorizeNDExtract*/false,
                        /*flatten1DDepthwiseConv*/false,
                        assumeDynamicDimsMatchVecSizes);
  if (failed(vectorResults))
    return failure();
  rewriter.replaceOp(op, vectorResults->replacements);
  return success();
}

struct TritonARM64LinalgVectorizationPass
    : public ::impl::TritonARM64LinalgVectorizationBase<TritonARM64LinalgVectorizationPass> {
public:
  explicit TritonARM64LinalgVectorizationPass(const TritonARM64LinalgVectorizationOptions &options)
      : TritonARM64LinalgVectorizationBase(options) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<func::FuncDialect, arith::ArithDialect, math::MathDialect,
                linalg::LinalgDialect, affine::AffineDialect, scf::SCFDialect,
                tensor::TensorDialect, bufferization::BufferizationDialect,
                vector::VectorDialect>();
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();
    MLIRContext *context = moduleOp.getContext();
    moduleOp.walk([&](linalg::LinalgOp op) {
      DBG("vectorization candidate: " << op << "\n");
      if (succeeded(vectorizeLinalgOp(op, vectorExt)))
        DBG(" -> vectorization succeeded.\n");
      else
        DBG(" -> vectorization failed.\n");
      return WalkResult::advance();
    });
   DBG("Module after vectorizing:\n" << moduleOp << "\n");
  }
};
} // namespace
std::unique_ptr<OperationPass<ModuleOp>>
triton::createTritonARM64LinalgVectorizationPass(const TritonARM64LinalgVectorizationOptions &options) {
  return std::make_unique<TritonARM64LinalgVectorizationPass>(options);
}
