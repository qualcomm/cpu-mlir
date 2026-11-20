//===- ARM64ReductionSchedulingPass.cpp - Reduction scheduling ------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/cpu-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// A vector reduction in a loop is normally lowered to a loop-carried scalar
// accumulator. On AArch64 that becomes a dependency chain of `fadda`
// instructions, so its latency, rather than its throughput, limits a long
// reduction. This pass unrolls the narrowly-defined reduction loop by four
// and gives each unrolled copy an independent accumulator. The four partial
// sums are combined after the loop.
//
// Reassociating floating-point additions changes their rounding. Triton's
// `tl.sum` does not prescribe an execution order, but this is nevertheless an
// explicitly experimental scheduling choice: every generated reduction and
// final combine is marked `reassoc`.
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "OptimizeTritonShared/OptimizeTritonShared.h"

using namespace mlir;
using namespace triton;

#define GEN_PASS_DEF_TRITONARM64REDUCTIONSCHEDULING
#include "OptimizeTritonShared/Passes.h.inc"

namespace {

constexpr unsigned kAccumulatorCount = 4;

static bool isFloatingPointValue(Type type) {
  return isa<FloatType>(getElementTypeOrSelf(type));
}

static bool isEligibleReductionLoop(scf::ForOp loop,
                                    vector::ReductionOp &reduction) {
  // The output of this loop is the scalar (represented as vector<1xf32> in
  // this pipeline) accumulator. Multiple carried values are either a
  // different reduction form or a loop we already transformed.
  if (loop.getNumRegionIterArgs() != 1 || loop.getNumResults() != 1 ||
      !isFloatingPointValue(loop.getResult(0).getType()))
    return false;

  Block *body = loop.getBody();
  for (Operation &op : body->without_terminator()) {
    // The four cloned iterations may execute for a masked tail. Do not clone
    // writes, calls, or nested control flow into those iterations. A
    // vector.mask is the expected masked-tail representation: cloning it is
    // safe and preserves the inactive lanes of the final partial vectors.
    if ((op.getNumRegions() != 0 && !isa<vector::MaskOp>(op)) ||
        hasEffect<MemoryEffects::Write>(&op) ||
        hasEffect<MemoryEffects::Allocate>(&op) ||
        hasEffect<MemoryEffects::Free>(&op))
      return false;
  }

  SmallVector<vector::ReductionOp> reductions;
  body->walk([&](vector::ReductionOp op) { reductions.push_back(op); });
  if (reductions.size() != 1)
    return false;

  // Only sum reductions have been investigated, a similar write-after-read 
  // pattern likely appears in other reductions such as max, but this requires
  // further testing and subsequent tuning to guage profitability
  reduction = reductions.front();
  if (reduction.getKind() != vector::CombiningKind::ADD ||
      !isFloatingPointValue(reduction.getDest().getType()))
    return false;

  auto vectorType = dyn_cast<VectorType>(reduction.getVector().getType());
  if (!vectorType || !vectorType.isScalable())
    return false;

  // Recognize exactly the loop-carried reduction form emitted by vectorization:
  //   extract(iter_arg) -> reduction -> [mask] -> broadcast -> scf.yield.
  // Merely consuming the loop-carried accumulator is insufficient: a
  // transformed reduction result (e.g. scaled before yielding) is a different
  // recurrence and cannot be decomposed into independent partial sums.
  //
  // This is a very tight and conservative check, there are other reductions 
  // that we could potentially transform, with proper investigation
  auto extract = reduction.getAcc().getDefiningOp<vector::ExtractOp>();
  if (!extract || extract->getOperand(0) != loop.getRegionIterArgs().front())
    return false;

  auto yield = cast<scf::YieldOp>(body->getTerminator());
  auto broadcast =
      yield.getOperand(0).getDefiningOp<vector::BroadcastOp>();
  if (!broadcast || broadcast.getResult().getType() != loop.getResult(0).getType())
    return false;

  Value reductionResult = reduction.getResult();
  Value broadcastInput = broadcast->getOperand(0);
  if (broadcastInput == reductionResult)
    return reduction->getBlock() == body;

  auto mask = broadcastInput.getDefiningOp<vector::MaskOp>();
  return mask && mask->getNumResults() == 1 &&
         reduction->getParentOp() == mask;
}

static void setReassoc(vector::ReductionOp reduction) {
  arith::FastMathFlags flags = reduction.getFastmath();
  flags |= arith::FastMathFlags::reassoc;
  reduction->setAttr(reduction.getFastmathAttrName(),
                      arith::FastMathFlagsAttr::get(reduction.getContext(),
                                                    flags));
}

static LogicalResult rescheduleReductionLoop(IRRewriter &rewriter,
                                             scf::ForOp loop) {
  vector::ReductionOp reduction;
  if (!isEligibleReductionLoop(loop, reduction))
    return failure();

  // bail out (for testing)
  if (kAccumulatorCount < 2)
    return failure();

  // Ensure that both cloned reductions and the final accumulation carry the
  // contract which permits their changed association.
  setReassoc(reduction);

  Location loc = loop.getLoc();
  rewriter.setInsertionPoint(loop);
  Value unrolledStep = rewriter.create<arith::MulIOp>(
      loc, loop.getStep(),
      rewriter.create<arith::ConstantIndexOp>(loc, kAccumulatorCount));

  // Only one partial reduction may inherit the original accumulator
  Value initialAccumulator = loop.getInitArgs().front();
  Value zero = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getZeroAttr(initialAccumulator.getType()));
  SmallVector<Value> initialAccumulators(kAccumulatorCount, zero);
  initialAccumulators.front() = initialAccumulator;
  scf::ForOp scheduledLoop = scf::ForOp::create(
      rewriter, loc, loop.getLowerBound(), loop.getUpperBound(), unrolledStep,
      initialAccumulators, [](OpBuilder &, Location, Value, ValueRange) {},
      loop.getUnsignedCmp());

  Block *oldBody = loop.getBody();
  Block *newBody = scheduledLoop.getBody();
  rewriter.setInsertionPointToStart(newBody);
  SmallVector<Value> yieldedAccumulators;
  yieldedAccumulators.reserve(kAccumulatorCount);

  auto oldYield = cast<scf::YieldOp>(oldBody->getTerminator());
  for (unsigned accumulatorIndex = 0;
       accumulatorIndex < kAccumulatorCount; ++accumulatorIndex) {
    IRMapping mapping;
    Value inductionVar = scheduledLoop.getInductionVar();
    if (accumulatorIndex != 0) {
      Value offset = rewriter.create<arith::MulIOp>(
          loc, loop.getStep(),
          rewriter.create<arith::ConstantIndexOp>(loc, accumulatorIndex));
      inductionVar = rewriter.create<arith::AddIOp>(loc, inductionVar, offset);
    }
    mapping.map(loop.getInductionVar(), inductionVar);
    Value accumulator = scheduledLoop.getRegionIterArgs()[accumulatorIndex];
    mapping.map(loop.getRegionIterArgs().front(), accumulator);

    // The scheduled loop advances by four original iterations. Its final
    // iteration can therefore have fewer than four valid original iteration
    // positions. A vector mask only protects the eventual transfer; it does
    // not make it valid to form an out-of-bounds subview before that transfer.
    // Keep the first copy unguarded (the scf.for condition proves it is in
    // bounds) and guard every additional copy as a whole.
    if (accumulatorIndex == 0) {
      for (Operation &op : oldBody->without_terminator())
        rewriter.clone(op, mapping);
      yieldedAccumulators.push_back(
          mapping.lookupOrDefault(oldYield.getOperand(0)));
      continue;
    }

    arith::CmpIPredicate predicate = loop.getUnsignedCmp()
                                        ? arith::CmpIPredicate::ult
                                        : arith::CmpIPredicate::slt;
    Value inBounds = rewriter.create<arith::CmpIOp>(
        loc, predicate, inductionVar, loop.getUpperBound());
    auto guard = rewriter.create<scf::IfOp>(
        loc, TypeRange{accumulator.getType()}, inBounds,
        /*withElseRegion=*/true);

    rewriter.setInsertionPointToStart(&guard.getThenRegion().front());
    for (Operation &op : oldBody->without_terminator())
      rewriter.clone(op, mapping);
    Value updatedAccumulator =
        mapping.lookupOrDefault(oldYield.getOperand(0));
    rewriter.create<scf::YieldOp>(loc, updatedAccumulator);

    rewriter.setInsertionPointToStart(&guard.getElseRegion().front());
    rewriter.create<scf::YieldOp>(loc, accumulator);
    rewriter.setInsertionPointAfter(guard);
    yieldedAccumulators.push_back(guard.getResult(0));
  }
  rewriter.create<scf::YieldOp>(loc, yieldedAccumulators);

  rewriter.setInsertionPointAfter(scheduledLoop);
  Value combined = scheduledLoop.getResult(0);
  for (Value partial : llvm::drop_begin(scheduledLoop.getResults())) {
    auto add = rewriter.create<arith::AddFOp>(loc, combined, partial);
    add.setFastmath(arith::FastMathFlags::reassoc);
    combined = add.getResult();
  }
  rewriter.replaceOp(loop, combined);
  return success();
}

struct TritonARM64ReductionSchedulingPass
    : public ::impl::TritonARM64ReductionSchedulingBase<
          TritonARM64ReductionSchedulingPass> {
  explicit TritonARM64ReductionSchedulingPass(
      const TritonARM64ReductionSchedulingOptions &options)
      : TritonARM64ReductionSchedulingBase(options) {}

  void runOnOperation() override {
    // Currently, only scalable vectors have been investigated, a similar 
    // pattern would likely be beneficial in NEON, but that requires it's own
    // tuning to gauge profitability
    if (vectorExt == NEON)
      return;

    SmallVector<scf::ForOp> loops;
    getOperation()->walk<WalkOrder::PostOrder>(
        [&](scf::ForOp loop) { loops.push_back(loop); });

    IRRewriter rewriter(getOperation()->getContext());
    for (scf::ForOp loop : loops)
      (void)rescheduleReductionLoop(rewriter, loop);
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
triton::createTritonARM64ReductionSchedulingPass(
    const TritonARM64ReductionSchedulingOptions &options) {
  return std::make_unique<TritonARM64ReductionSchedulingPass>(options);
}
