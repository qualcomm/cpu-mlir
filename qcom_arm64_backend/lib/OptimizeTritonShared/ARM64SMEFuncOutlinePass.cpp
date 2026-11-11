//===- ARM64SMEFuncOutlinePass.cpp ----------------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/cpu-mlir/LICENSE.txt
//
// This file moves each loop nest that works on SME ZA tiles into its own
// function, so that only that function runs in streaming mode.
//===----------------------------------------------------------------------===//
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/RegionUtils.h"

#include "llvm/Support/Debug.h"

#include "OptimizeTritonShared/OptimizeTritonShared.h"

#define DEBUG_TYPE "triton-arm64-sme-func-outline"

#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define DBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace triton;

#define GEN_PASS_DEF_TRITONARM64SMEFUNCOUTLINE
#include "OptimizeTritonShared/Passes.h.inc"

namespace {

// A ZA tile value: a vector whose two innermost dimensions are scalable, such
// as vector<[4]x[4]xf32> or vector<1x1x[4]x[4]xf32>. Only code that the ArmSME
// passes turn into ZA operations works on such vectors.
static bool isZATileType(Type type) {
  auto vecType = dyn_cast<VectorType>(type);
  if (!vecType || vecType.getRank() < 2)
    return false;
  ArrayRef<bool> scalable = vecType.getScalableDims();
  return scalable.back() && scalable[scalable.size() - 2];
}

static bool usesZATiles(Operation *root) {
  WalkResult result = root->walk([](Operation *op) {
    if (llvm::any_of(op->getResultTypes(), isZATileType) ||
        llvm::any_of(op->getOperandTypes(), isZATileType))
      return WalkResult::interrupt();
    return WalkResult::advance();
  });
  return result.wasInterrupted();
}

// Whether `value` is computed from vector.vscale. Inside the outlined
// function, which runs in streaming mode, vscale is derived from the streaming
// vector length, which may differ from the SVE vector length the caller sees.
static bool dependsOnVScale(Value value, DenseSet<Operation *> &visited) {
  Operation *def = value.getDefiningOp();
  if (!def || !visited.insert(def).second)
    return false;
  if (isa<vector::VectorScaleOp>(def))
    return true;
  return llvm::any_of(def->getOperands(), [&](Value operand) {
    return dependsOnVScale(operand, visited);
  });
}

// Splits the values `loop` uses from outside into arguments of the outlined
// function and pure ops to recompute inside it. Pure computations (constants,
// vscale and the index arithmetic on them) are recomputed so that, in
// particular, every vscale-based size is evaluated in streaming mode.
// `toClone` is in an order where operands come before their users.
static LogicalResult collectOutlinedInputs(scf::ForOp loop,
                                           SmallVectorImpl<Value> &args,
                                           SetVector<Operation *> &toClone) {
  SetVector<Value> used;
  getUsedValuesDefinedAbove(loop->getRegions(), used);
  used.insert(loop->operand_begin(), loop->operand_end());

  SetVector<Value> argSet;
  std::function<void(Value)> visit = [&](Value value) {
    Operation *def = value.getDefiningOp();
    if (def && toClone.contains(def))
      return;
    if (def && isPure(def) && def->getNumRegions() == 0) {
      for (Value operand : def->getOperands())
        visit(operand);
      toClone.insert(def); // after its operands
      return;
    }
    argSet.insert(value);
  };
  for (Value value : used)
    visit(value);

  for (Value arg : argSet) {
    DenseSet<Operation *> visited;
    // A vector would cross the streaming boundary, and a value computed from
    // vscale outside would use the wrong vector length.
    if (isa<VectorType>(arg.getType()) || dependsOnVScale(arg, visited))
      return failure();
  }
  args.assign(argSet.begin(), argSet.end());
  return success();
}

// Moves `loop` into a new private function after `parent` and replaces it with
// a call.
static LogicalResult outlineLoop(scf::ForOp loop, func::FuncOp parent,
                                 unsigned index) {
  // After bufferization the loop writes to memrefs and has no results.
  if (loop->getNumResults() != 0)
    return failure();

  SmallVector<Value> args;
  SetVector<Operation *> toClone;
  if (failed(collectOutlinedInputs(loop, args, toClone)))
    return failure();

  OpBuilder builder(parent);
  builder.setInsertionPointAfter(parent);
  std::string name = (parent.getSymName() + "_sme_" + Twine(index)).str();
  auto funcType = builder.getFunctionType(ValueRange(args).getTypes(), {});
  auto func = func::FuncOp::create(builder, loop.getLoc(), name, funcType);
  // Private and internal: only `parent` calls it, so it needs neither a C
  // interface nor an exported symbol.
  func.setPrivate();
  func->setAttr("llvm.linkage", LLVM::LinkageAttr::get(builder.getContext(),
                                                       LLVM::Linkage::Internal));
  Block *entry = func.addEntryBlock();

  IRMapping mapping;
  mapping.map(args, entry->getArguments());
  builder.setInsertionPointToStart(entry);
  for (Operation *op : toClone)
    builder.clone(*op, mapping);
  builder.clone(*loop, mapping);
  func::ReturnOp::create(builder, loop.getLoc());

  builder.setInsertionPoint(loop);
  func::CallOp::create(builder, loop.getLoc(), func, args);
  loop.erase();
  return success();
}

struct TritonARM64SMEFuncOutlinePass
    : public ::impl::TritonARM64SMEFuncOutlineBase<TritonARM64SMEFuncOutlinePass> {
  void runOnOperation() override {
    // Copy the list first: outlining adds functions to the module.
    SmallVector<func::FuncOp> funcs(getOperation().getOps<func::FuncOp>());
    for (func::FuncOp func : funcs) {
      if (func.isExternal())
        continue;
      // Outermost loops only: skip() keeps the walk out of a loop that is
      // already being outlined.
      SmallVector<scf::ForOp> loops;
      func.walk<WalkOrder::PreOrder>([&](scf::ForOp loop) {
        if (!usesZATiles(loop))
          return WalkResult::advance();
        loops.push_back(loop);
        return WalkResult::skip();
      });
      for (auto [index, loop] : llvm::enumerate(loops))
        if (failed(outlineLoop(loop, func, index)))
          DBG("could not outline: " << loop);
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
triton::createTritonARM64SMEFuncOutlinePass() {
  return std::make_unique<TritonARM64SMEFuncOutlinePass>();
}
