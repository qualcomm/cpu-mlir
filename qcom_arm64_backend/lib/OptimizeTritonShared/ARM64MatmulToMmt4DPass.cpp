//===- ARM64MatmulToMmt4DPass.cpp -----------------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/cpu-mlir/LICENSE.txt
//
// This file implements the conversion of linalg.matmul to a sequence of
// linalg.pack, linalg.mmt4d, and linalg.unpack operations.
//===----------------------------------------------------------------------===//
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "OptimizeTritonShared/OptimizeTritonShared.h"

#define DEBUG_TYPE "triton-arm64-matmul-to-mmt4d"

#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define DBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace triton;
using namespace mlir::linalg;

#define GEN_PASS_DEF_TRITONARM64MATMULTOMMT4D
#include "OptimizeTritonShared/Passes.h.inc"

namespace {

struct LowerPackPattern : public OpRewritePattern<PackOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(PackOp packOp,
                                PatternRewriter &rewriter) const override {
    return linalg::lowerPack(rewriter, packOp,
                             /*lowerPadLikeWithInsertSlice*/false);
  }
};

struct LowerUnPackPattern : public OpRewritePattern<UnPackOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(UnPackOp unPackOp,
                                PatternRewriter &rewriter) const override {
    return linalg::lowerUnPack(rewriter, unPackOp,
                             /*lowerUnpadLikeWithInsertSlice*/false);
  }
};

// In a loop that accumulates a matmul into a loop-carried tensor, such as the
// k-loop of a Triton matmul, block packing packs the accumulator at the start
// of every iteration and unpacks it at the end:
//
//   scf.for ... iter_args(%acc = %init) {
//     %p = linalg.pack %acc ...
//     %r = linalg.mmt4d ins(...) outs(%p)
//     %u = linalg.unpack %r into %acc
//     scf.yield %u
//   }
//
// Carry the packed tensor instead, so that the accumulator is packed once
// before the loop and unpacked once after it.
static LogicalResult hoistPackedAccumulator(RewriterBase &rewriter,
                                            scf::ForOp loop, unsigned idx) {
  BlockArgument iterArg = loop.getRegionIterArg(idx);
  auto yield = cast<scf::YieldOp>(loop.getBody()->getTerminator());
  auto unpack = yield.getOperand(idx).getDefiningOp<linalg::UnPackOp>();
  if (!unpack || !unpack->hasOneUse() || unpack->getBlock() != loop.getBody())
    return failure();

  // The accumulator may only be packed (once) and be the unpack destination.
  linalg::PackOp pack;
  for (OpOperand &use : iterArg.getUses()) {
    auto user = dyn_cast<linalg::PackOp>(use.getOwner());
    if (user && !pack && use.get() == user.getSource()) {
      pack = user;
      continue;
    }
    if (use.getOwner() == unpack.getOperation() &&
        &use == &unpack.getDestMutable())
      continue;
    return failure();
  }
  // With padding, the unpack drops the padded rows and the next pack refills
  // them with the padding value, which the packed tensor would not do.
  if (!pack || pack.getPaddingValue() ||
      pack.getDestType() != unpack.getSourceType() ||
      !pack.getDestType().hasStaticShape() ||
      !unpack.getDestType().hasStaticShape() ||
      pack.getInnerDimsPos() != unpack.getInnerDimsPos() ||
      pack.getOuterDimsPerm() != unpack.getOuterDimsPerm() ||
      pack.getStaticInnerTiles() != unpack.getStaticInnerTiles() ||
      !pack.getInnerTiles().empty() || !unpack.getInnerTiles().empty())
    return failure();

  Location loc = loop.getLoc();
  ShapedType packedType = pack.getDestType();
  ShapedType unpackedType = unpack.getDestType();

  // Pack the initial value before the loop.
  rewriter.setInsertionPoint(loop);
  Value packDest = tensor::EmptyOp::create(rewriter, loc, packedType.getShape(),
                                           packedType.getElementType());
  IRMapping packMapping;
  packMapping.map(pack.getSource(), loop.getInitArgs()[idx]);
  packMapping.map(pack.getDest(), packDest);
  Value packedInit = rewriter.clone(*pack, packMapping)->getResult(0);

  SmallVector<Value> inits(loop.getInitArgs());
  inits[idx] = packedInit;
  auto newLoop = scf::ForOp::create(
      rewriter, loc, loop.getLowerBound(), loop.getUpperBound(),
      loop.getStep(), inits, [](OpBuilder &, Location, Value, ValueRange) {},
      loop.getUnsignedCmp());
  newLoop->setDiscardableAttrs(loop->getDiscardableAttrDictionary());

  // Unpack the final value after the loop.
  rewriter.setInsertionPointAfter(newLoop);
  Value unpackDest = tensor::EmptyOp::create(
      rewriter, loc, unpackedType.getShape(), unpackedType.getElementType());
  IRMapping unpackMapping;
  unpackMapping.map(unpack.getSource(), newLoop.getResult(idx));
  unpackMapping.map(unpack.getDest(), unpackDest);
  Value result = rewriter.clone(*unpack, unpackMapping)->getResult(0);

  // Move the body over: yield the packed result, use the packed iter_arg.
  rewriter.modifyOpInPlace(
      yield, [&] { yield.setOperand(idx, unpack.getSource()); });
  rewriter.eraseOp(unpack);
  Block *newBody = newLoop.getBody();
  rewriter.mergeBlocks(loop.getBody(), newBody, newBody->getArguments());
  rewriter.replaceAllUsesWith(pack.getResult(), newLoop.getRegionIterArg(idx));
  rewriter.eraseOp(pack);

  SmallVector<Value> results(newLoop.getResults());
  results[idx] = result;
  rewriter.replaceOp(loop, results);
  return success();
}

static void hoistPackedAccumulators(Operation *root) {
  IRRewriter rewriter(root->getContext());
  SmallVector<scf::ForOp> loops;
  root->walk([&](scf::ForOp loop) { loops.push_back(loop); });
  for (scf::ForOp loop : loops) {
    for (unsigned idx = 0; idx < loop.getNumRegionIterArgs(); ++idx) {
      if (succeeded(hoistPackedAccumulator(rewriter, loop, idx))) {
        DBG("hoisted the packing of accumulator #" << idx);
        break; // `loop` was replaced
      }
    }
  }
}

struct TritonARM64MatmulToMmt4DPass
    : public ::impl::TritonARM64MatmulToMmt4DBase<TritonARM64MatmulToMmt4DPass> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<func::FuncDialect, scf::SCFDialect,
                    tensor::TensorDialect>();
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();
    auto ctx = moduleOp.getContext();
    RewritePatternSet patterns(ctx);

    ControlBlockPackMatmulFn controlFn =
        [&](linalg::LinalgOp op) -> BlockPackMatmulOptions {
      BlockPackMatmulOptions options;
      // TODO: figure out optimal block sizes
      options.blockFactors = SmallVector<int64_t>{32, 32, 1};
      return options;
    };

    linalg::populateBlockPackMatmulPatterns(patterns, controlFn);
    linalg::populateLinalgGenericOpsSpecializationPatterns(patterns);
    linalg::populateFoldAddIntoDestPatterns(patterns);
    if (failed(applyPatternsGreedily(getOperation(),
                                     std::move(patterns)))) {
      return signalPassFailure();
    }

    // Before the packs are lowered: the hoisting matches pack/unpack pairs.
    hoistPackedAccumulators(getOperation());

    RewritePatternSet lowerPatterns(ctx);
    lowerPatterns.add<LowerPackPattern>(ctx);
    lowerPatterns.add<LowerUnPackPattern>(ctx);
    if (failed(applyPatternsGreedily(getOperation(),
                                     std::move(lowerPatterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
triton::createTritonARM64MatmulToMmt4DPass() {
  return std::make_unique<TritonARM64MatmulToMmt4DPass>();
}
