//===- ARM64TilingPass.cpp - tiling to enable vectorization ---------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/cpu-mlir/LICENSE.txt
//
// This file implements linalg op tiling for ARM64 targets with specfic
// intention to use scalable vectors when appropriate. So scalable vectorization
// can be done by subsequent vectorization pass and vectorized ops can lowered
// to SVE or SME instructions.
//===----------------------------------------------------------------------===//
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Interfaces/TilingInterface.h"

#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/Support/Debug.h"
#include <algorithm>
#include <numeric>
#include <vector>

#include "OptimizeTritonShared/OptimizeTritonShared.h"

#define DEBUG_TYPE "triton-arm64-linalg-tiling"

#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define DBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace triton;

#define GEN_PASS_DEF_TRITONARM64LINALGTILING
#include "OptimizeTritonShared/Passes.h.inc"

namespace {

static LogicalResult tileMmt4DOp(linalg::Mmt4DOp op, arm64VectorExtension vectorExt,
    unsigned numLoops) {
  IRRewriter rewriter(op.getContext());
  rewriter.setInsertionPoint(op);
  Location loc = op.getLoc();

  // Constant OpFoldResults for zero and one tile sizes.
  OpFoldResult zero = rewriter.getIndexAttr(0);
  OpFoldResult one  = rewriter.getIndexAttr(1);

  // Compute scalable tile size once; used as OpFoldResult for both steps so
  // that a single vscale/muli pair is shared across the two tiling calls.
  OpFoldResult mTile, nTile;
  if (vectorExt == SME || vectorExt == SVE) {
    Value vscale = vector::VectorScaleOp::create(rewriter, loc, rewriter.getIndexType());
    Value c8     = arith::ConstantIndexOp::create(rewriter, loc, 8);
    Value scale8 = arith::MulIOp::create(rewriter, loc, c8, vscale).getResult();
    if (vectorExt == SME) {
      mTile = nTile = scale8;          // {[8], [8], 0}
    } else {
      mTile = rewriter.getIndexAttr(8); // {8, [8], 0}
      nTile = scale8;
    }
  } else { // NEON
    mTile = nTile = rewriter.getIndexAttr(8); // {8, 8, 0}
  }

  // Step 1: tile [1, 1, 0, mTile, nTile, 0] — parallel M/N dims.
  scf::SCFTilingOptions parallelOpts;
  parallelOpts.setTileSizes({one, one, zero, mTile, nTile, zero});

  auto tiledParallel = scf::tileUsingSCF(
      rewriter, cast<TilingInterface>(op.getOperation()), parallelOpts);
  if (failed(tiledParallel))
    return failure();
  rewriter.replaceOp(op, tiledParallel->replacements);

  if (vectorExt != SME)
    return success();

  // Step 2: tile [0, 0, 1, 0, 0, 0] — K-outer reduction dim for SME.
  scf::SCFTilingOptions reductionOpts;
  reductionOpts.setTileSizes({zero, zero, one, zero, zero, zero});

  auto innerOp = cast<TilingInterface>(tiledParallel->tiledOps[0]);
  auto tiledReduction = scf::tileUsingSCF(rewriter, innerOp, reductionOpts);
  if (failed(tiledReduction))
    return failure();
  rewriter.replaceOp(tiledParallel->tiledOps[0], tiledReduction->replacements);

  return success();
}

// Tile height dimensions, OH and KH, then decompose NCHW/FCHW convolution to
// NCW/FCW.
static LogicalResult tileAndDecomposeConv2D(linalg::Conv2DNchwFchwOp op) {
  IRRewriter rewriter(op.getContext());
  rewriter.setInsertionPoint(op);

  OpFoldResult zero = rewriter.getIndexAttr(0);
  OpFoldResult one = rewriter.getIndexAttr(1);
  scf::SCFTilingOptions options;
  // Loop order: N, F, OH, OW, C, KH, KW.
  options.setTileSizes({zero, zero, one, zero, zero, one, zero});

  auto tiled = scf::tileUsingSCF(
      rewriter, cast<TilingInterface>(op.getOperation()), options);
  if (failed(tiled) || tiled->tiledOps.size() != 1)
    return failure();
  rewriter.replaceOp(op, tiled->replacements);

  Operation *tiledOp = tiled->tiledOps.front();
  rewriter.setInsertionPoint(tiledOp);
  // The tiled convolution has unit OH and KH dimensions.
  FailureOr<linalg::LinalgOp> decomposed =
      linalg::downscaleSizeOneWindowedConvolution(
          rewriter, cast<linalg::LinalgOp>(tiledOp));
  if (failed(decomposed))
    return failure();
  return success();
}

// Tile F and OW, and split C and KW, before vectorization.
static LogicalResult tileConv1D(linalg::Conv1DNcwFcwOp op) {
  IRRewriter rewriter(op.getContext());
  rewriter.setInsertionPoint(op);

  OpFoldResult zero = rewriter.getIndexAttr(0);
  OpFoldResult one = rewriter.getIndexAttr(1);
  OpFoldResult four = rewriter.getIndexAttr(4);
  scf::SCFTilingOptions options;
  // Loop order: N, F, OW, C, KW.
  options.setTileSizes({zero, four, four, one, one});

  auto tiled = scf::tileUsingSCF(
      rewriter, cast<TilingInterface>(op.getOperation()), options);
  if (failed(tiled))
    return failure();
  rewriter.replaceOp(op, tiled->replacements);
  return success();
}

static LogicalResult tileLinalgOp(linalg::LinalgOp op, arm64VectorExtension vectorExt) {
  // Early exit.
  unsigned numLoops = op.getNumLoops();
  if (!op.hasPureTensorSemantics() || !op->getNumResults() ||
      !isa<DestinationStyleOpInterface>(op.getOperation()) || numLoops < 1) {
    DBG("-> tiling aborted. constraints not satisfied");
    return failure();
  }

  // Specialized tiling for certain ops
  if (isa<linalg::Conv2DNchwFchwOp>(op))
    return tileAndDecomposeConv2D(cast<linalg::Conv2DNchwFchwOp>(op));
  if (isa<linalg::Conv1DNcwFcwOp>(op))
    return tileConv1D(cast<linalg::Conv1DNcwFcwOp>(op));
  if (isa<linalg::Mmt4DOp>(op))
    return tileMmt4DOp(cast<linalg::Mmt4DOp>(op), vectorExt, numLoops);

  linalg::LinalgOp tilingTarget = op;
  linalg::LinalgTilingOptions tileOption;
  SmallVector<int64_t, 10> tileSizes(numLoops, 1);

  // TODO: set tile size according to element type size
  tileSizes[numLoops - 1] = 8;

  // For now only allow scalable vector on the last tile (inner most loop)
  bool lastTileSizeScalable = vectorExt != NEON &&
      (linalg::isElementwise(op) || isa<linalg::ReduceOp>(op) ||
      isa<linalg::MatvecOp>(op) || isa<linalg::MatmulOp>(op) ||
      isa<linalg::Mmt4DOp>(op));

  if (lastTileSizeScalable) {
    tileOption.setTileSizeComputationFunction([&, tileSizes](OpBuilder &b,
                                                             Operation *op) {
      SmallVector<Value> sizes;
      sizes.reserve(tileSizes.size());

      auto Loc = op->getLoc();
      Value vscale =
          b.create<vector::VectorScaleOp>(Loc, b.getIndexType());

      if (isa<linalg::MatmulOp>(op) && vectorExt == SME) {
        auto valX = b.create<arith::ConstantIndexOp>(Loc, 8);
        sizes.push_back(b.create<arith::MulIOp>(Loc, valX, vscale).getResult());
        auto valY = b.create<arith::ConstantIndexOp>(Loc, 8);
        sizes.push_back(b.create<arith::MulIOp>(Loc, valY, vscale).getResult());
        sizes.push_back(b.create<arith::ConstantIndexOp>(Loc, 2));
      } else {
        for (int i = 0; i < numLoops - 1; ++i)
          sizes.push_back(b.create<arith::ConstantIndexOp>(Loc, tileSizes[i]));
        auto val = b.create<arith::ConstantIndexOp>(Loc, 4);
        sizes.push_back(b.create<arith::MulIOp>(Loc, val, vscale).getResult());
      }
      return sizes;
    });
  } else {
    // NOTE: NEON vector registers are 128-bit wide.
    // TODO: set tile size based on element size
    tileOption.setTileSizes(tileSizes);
  }

  IRRewriter rewriter(op.getContext());
  rewriter.setInsertionPoint(tilingTarget);
  auto tiledParallel = linalg::tileLinalgOp(rewriter, tilingTarget, tileOption);
  if (failed(tiledParallel)) {
    return failure();
  }
  rewriter.replaceOp(tilingTarget, tiledParallel->tensorResults);

  return success();
}

struct TritonARM64LinalgTilingPass
    : public ::impl::TritonARM64LinalgTilingBase<TritonARM64LinalgTilingPass> {
  TritonARM64LinalgTilingPass() = default;

  explicit TritonARM64LinalgTilingPass(const TritonARM64LinalgTilingOptions &options)
      : TritonARM64LinalgTilingBase(options) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<func::FuncDialect, vector::VectorDialect>();
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();
    moduleOp.walk([&](linalg::LinalgOp op) {
      DBG(" tiling candidate: " << op << "\n");
      if (succeeded(tileLinalgOp(op, vectorExt)))
        DBG(" -> tiling succeeded\n");
      else
        DBG(" -> tiling failed\n");
      return WalkResult::advance();
    });
    DBG("Module after tiliing:\n" << moduleOp << "\n");
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
triton::createTritonARM64LinalgTilingPass(const TritonARM64LinalgTilingOptions &options) {
  return std::make_unique<TritonARM64LinalgTilingPass>(options);
}
