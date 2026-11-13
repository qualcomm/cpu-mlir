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
#include "mlir/Dialect/MemRef/Utils/MemRefUtils.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"

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

// If `op` swaps its two innermost dimensions and its element type fits an SME
// ZA tile, returns the tile size per vscale (128 / element bits, e.g. 4 for f32
// and 8 for f16). Otherwise returns 0.
static int64_t getSMETransposeTileSize(linalg::TransposeOp op) {
  // The permutation must be [0, 1, ..., rank-3, rank-1, rank-2]: the outer
  // dimensions stay in place and only the last two are swapped. Then every
  // [s]x[s] tile is a plain 2-D transpose of rows that are contiguous in
  // memory. A permutation that keeps the innermost dimension in place (e.g.
  // [0, 2, 1, 3]) only moves whole rows and needs no ZA tile.
  ArrayRef<int64_t> perm = op.getPermutation();
  int64_t rank = perm.size();
  if (rank < 2 || perm[rank - 1] != rank - 2 || perm[rank - 2] != rank - 1)
    return 0;
  for (int64_t i = 0; i < rank - 2; ++i)
    if (perm[i] != i)
      return 0;
  Type elemType = op.getInit().getType().getElementType();
  if (!elemType.isIntOrFloat())
    return 0;
  unsigned bits = elemType.getIntOrFloatBitWidth();
  if (bits != 8 && bits != 16 && bits != 32 && bits != 64)
    return 0;
  return 128 / bits;
}

// Tile the two innermost (swapped) dimensions of a transpose to one SME ZA
// tile, [s]x[s], and all outer dimensions to 1. Vectorization then reads each
// tile with a transposing transfer_read, which ArmSME lowers to rows written
// into ZA as vertical slices and stored back horizontally.
static LogicalResult tileTransposeForSME(linalg::TransposeOp op,
                                         int64_t tileSize) {
  IRRewriter rewriter(op.getContext());
  rewriter.setInsertionPoint(op);
  Location loc = op.getLoc();

  Value vscale =
      vector::VectorScaleOp::create(rewriter, loc, rewriter.getIndexType());
  Value size = arith::ConstantIndexOp::create(rewriter, loc, tileSize);
  OpFoldResult scaled = arith::MulIOp::create(rewriter, loc, size, vscale).getResult();

  unsigned numLoops = op.getNumLoops();
  SmallVector<OpFoldResult> tileSizes(numLoops, rewriter.getIndexAttr(1));
  tileSizes[numLoops - 2] = scaled;
  tileSizes[numLoops - 1] = scaled;

  scf::SCFTilingOptions options;
  options.setTileSizes(tileSizes);
  auto tiled = scf::tileUsingSCF(
      rewriter, cast<TilingInterface>(op.getOperation()), options);
  if (failed(tiled))
    return failure();
  for (Operation *tiledOp : tiled->tiledOps)
    tiledOp->setAttr(kSMETransposeTileAttr,
                     rewriter.getI64IntegerAttr(tileSize));
  rewriter.replaceOp(op, tiled->replacements);
  return success();
}

// Whether a transpose is large enough for the ZA tile path to pay off. A
// dynamic dimension counts with its constant upper bound, e.g. the masked block
// of a Triton kernel: tl.trans of a 1024x512 block whose edge is masked becomes
// a transpose of an extract_slice with sizes min(.., 1024) x min(.., 512).
// Every program instance runs the same code, so the block size decides. A
// dimension without a known bound keeps the regular path.
static bool isLargeEnoughForSMETranspose(linalg::TransposeOp op) {
  Value init = op.getInit();
  auto type = cast<ShapedType>(init.getType());
  int64_t numElements = 1;
  for (int64_t dim = 0; dim < type.getRank(); ++dim) {
    int64_t size = type.getDimSize(dim);
    if (ShapedType::isDynamic(size)) {
      ValueBoundsOptions options;
      options.closedUB = true;
      FailureOr<int64_t> bound = ValueBoundsConstraintSet::computeConstantBound(
          presburger::BoundType::UB, {init, dim}, /*stopCondition=*/nullptr,
          options);
      if (failed(bound))
        return false;
      size = *bound;
    }
    numElements *= size;
  }
  return numElements >= kSMETransposeMinElements;
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
  // Early exit. Besides tensor ops, copies between buffers are tiled: they are
  // the memref.copy ops rewritten by convertStridedCopiesToLinalg.
  unsigned numLoops = op.getNumLoops();
  bool isBufferCopy = isa<linalg::CopyOp>(op) && op.hasPureBufferSemantics();
  if ((!isBufferCopy && (!op.hasPureTensorSemantics() || !op->getNumResults())) ||
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
  if (vectorExt == SME && isa<linalg::TransposeOp>(op)) {
    auto transposeOp = cast<linalg::TransposeOp>(op);
    if (isLargeEnoughForSMETranspose(transposeOp))
      if (int64_t tileSize = getSMETransposeTileSize(transposeOp))
        return tileTransposeForSME(transposeOp, tileSize);
  }

  linalg::LinalgOp tilingTarget = op;
  linalg::LinalgTilingOptions tileOption;
  SmallVector<int64_t, 10> tileSizes(numLoops, 1);

  // TODO: set tile size according to element type size
  tileSizes[numLoops - 1] = 8;

  // For now only allow scalable vector on the last tile (inner most loop)
  bool lastTileSizeScalable = vectorExt != NEON &&
      (linalg::isElementwise(op) || isa<linalg::ReduceOp>(op) ||
      isa<linalg::MatvecOp>(op) || isa<linalg::MatmulOp>(op) ||
      isa<linalg::Mmt4DOp>(op) || isInnermostReduction(op));

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
        auto val = b.create<arith::ConstantIndexOp>(
            Loc, getScalableVectorSize(op));
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

// memref.copy between dynamically shaped, strided buffers, such as the masked
// tl.load of triton-shared (a subview of the input copied into a subview of a
// local buffer), is lowered to a call to the generic memrefCopy runtime
// function, which walks the elements one by one. As a linalg.copy it is tiled
// and vectorized like any elementwise op. Copies that lower to a plain memcpy
// (identity layout or static contiguous shape) are left alone.
static void convertStridedCopiesToLinalg(ModuleOp module) {
  auto lowersToMemcpy = [](MemRefType type) {
    return type.getLayout().isIdentity() ||
           (type.hasStaticShape() && type.getNumElements() > 0 &&
            memref::isStaticShapeAndContiguousRowMajor(type));
  };
  SmallVector<memref::CopyOp> copies;
  module.walk([&](memref::CopyOp copy) {
    auto source = dyn_cast<MemRefType>(copy.getSource().getType());
    auto target = dyn_cast<MemRefType>(copy.getTarget().getType());
    if (source && target && source.getRank() > 0 &&
        source.getElementType() == target.getElementType() &&
        !(lowersToMemcpy(source) && lowersToMemcpy(target)))
      copies.push_back(copy);
  });
  IRRewriter rewriter(module.getContext());
  for (memref::CopyOp copy : copies) {
    rewriter.setInsertionPoint(copy);
    rewriter.replaceOpWithNewOp<linalg::CopyOp>(copy, copy.getSource(),
                                                copy.getTarget());
  }
}

struct TritonARM64LinalgTilingPass
    : public ::impl::TritonARM64LinalgTilingBase<TritonARM64LinalgTilingPass> {
  TritonARM64LinalgTilingPass() = default;

  explicit TritonARM64LinalgTilingPass(const TritonARM64LinalgTilingOptions &options)
      : TritonARM64LinalgTilingBase(options) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<func::FuncDialect, vector::VectorDialect,
                    linalg::LinalgDialect, scf::SCFDialect,
                    arith::ArithDialect>();
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();
    convertStridedCopiesToLinalg(moduleOp);
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

bool triton::isInnermostReduction(Operation *op) {
  auto genericOp = dyn_cast<linalg::GenericOp>(op);
  if (!genericOp || genericOp.getNumReductionLoops() != 1)
    return false;
  return genericOp.getIteratorTypesArray().back() ==
         utils::IteratorType::reduction;
}

std::unique_ptr<OperationPass<ModuleOp>>
triton::createTritonARM64LinalgTilingPass(const TritonARM64LinalgTilingOptions &options) {
  return std::make_unique<TritonARM64LinalgTilingPass>(options);
}
