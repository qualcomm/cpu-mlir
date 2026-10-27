//===- ARM64ProfInstrumentation.cpp ---------------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/cpu-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

//===- ARM64ProfInstrumentation.cpp - profiling instrumentation pass ------===//
//
//===----------------------------------------------------------------------===//
// This file implements a profiler pass for ARM64 targets.
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include <string>

#include "LowerTritonLinalgToLLVM/LowerTritonLinalgToLLVM.h"

#define DEBUG_TYPE "trition-arm64-prof-instrumentation"

#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define DBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace std::literals;
using namespace mlir;
using namespace mlir::scf;
using namespace mlir::LLVM;
using namespace triton;

#define GEN_PASS_DEF_TRITONARM64PROFPASS
#include "LowerTritonLinalgToLLVM/Passes.h.inc"

namespace {
struct TritonARM64ProfPass : public ::impl::TritonARM64ProfPassBase<TritonARM64ProfPass> {
  TritonARM64ProfPass() = default;
  explicit TritonARM64ProfPass(const TritonARM64ProfPassOptions &options)
      : TritonARM64ProfPassBase(options) {}

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::LLVM::LLVMDialect>();
  }

  // Compute loop depth (distance to outermost loop)
  static int getLoopDepth(Operation *op) {
    int depth = 0;
    while (auto parent = dyn_cast_or_null<scf::ForOp>(op->getParentOp())) {
      ++depth;
      op = parent;
    }
    return depth;
  }

  // Check if a loop has at least one sibling scf.for loop
  static bool hasSiblingLoop(scf::ForOp forOp) {
    Operation *parentOp = forOp->getParentOp();
    if (!parentOp || parentOp->getNumRegions() == 0)
      return false;

    auto &block = parentOp->getRegion(0).front();
    for (auto &op : block) {
      if (&op != forOp.getOperation() && isa<scf::ForOp>(op))
        return true;
    }
    return false;
  }

  // Dump the location and ops info into a file so they can be postprocessed.
  void printLines(const std::vector<int> &lines, int loopId,
                  const std::vector<std::string> &opNames) const {
    static bool fileInitialized = false;
    // TODO: perhaps save to staging dir instead of /tmp, not currently trivial
    //       to determine staging dir while in this pass
    const std::string filePath = "/tmp/prof_infodump.txt";

    // Remove the stale file for fresh start.
    if (!fileInitialized) {
      if (llvm::sys::fs::exists(filePath)) {
        llvm::sys::fs::remove(filePath);
      }
      fileInitialized = true;
    }

    std::error_code EC;
    llvm::raw_fd_ostream fileStream(filePath, EC, llvm::sys::fs::OF_Append);

    if (EC) {
      llvm::errs() << "Error opening file: " << EC.message() << "\n";
      return;
    }

    fileStream << "Location ";
    for (size_t i = 0; i < lines.size(); ++i) {
      fileStream << lines[i];
      if (i != lines.size() - 1)
        fileStream << ", ";
    }

    fileStream << " corresponds to ID " << loopId << " | Collected ops: ";

    if (opNames.empty()) {
      fileStream << "\n";
    } else {
      for (size_t i = 0; i < opNames.size(); ++i) {
        fileStream << opNames[i];
        if (i != opNames.size() - 1)
          fileStream << ", ";
      }
      fileStream << "\n";
    }

    fileStream.flush();
  }

  // Get location info in a list.
  static std::vector<int> collectLoc(Location loc) {
    std::vector<int> lines;
    // FusedLoc can have nested FusedLoc.
    // Recursively get all the relevant locations in lines list.
    std::function<void(Location)> recurse = [&](Location loc) {
      if (auto fusedLoc = dyn_cast<FusedLoc>(loc)) {
        for (Location subLoc : fusedLoc.getLocations())
          recurse(subLoc);
      } else if (auto fileLoc = dyn_cast<FileLineColLoc>(loc)) {
        lines.push_back(fileLoc.getLine());
      }
    };

    recurse(loc);
    return lines;
  }

  // Collect ops of interest.
  std::vector<std::string> collectOps(scf::ForOp forOp) {
    std::vector<std::string> opNames;
    forOp.getBody()->walk([&](Operation *op) -> void {
      // Filter out non-trivial operations
      auto opName = op->getName().getStringRef();

      if (opName == "memref.subview" || opName == "memref.load" ||
          opName == "memref.store" || opName == "vector.insertelement" ||
          opName == "vector.transfer_read" ||
          opName == "vector.transfer_write" || opName == "vector.broadcast" ||
          opName == "vector.extractelement" || opName == "affine.apply" ||
          opName == "cf.assert" || opName == "scf.yield" ||
          opName == "scf.for") {
        return;
      }

      if (auto reductionOp = dyn_cast<mlir::vector::ReductionOp>(op)) {
        auto kind =
            reductionOp.getKind(); // This returns a vector::CombiningKind enum
        std::string formattedName = "vector.reduction<" +
                                    vector::stringifyCombiningKind(kind).str() +
                                    ">";
        opNames.push_back(formattedName);
      } else {
        opNames.push_back(opName.str());
      }
    });

    return opNames;
  }

  void runOnOperation() override {
    auto func = getOperation();
    ModuleOp module = func->getParentOfType<ModuleOp>();
    MLIRContext *ctx = &getContext();
    OpBuilder builder(module.getContext());

    auto i32Ty = IntegerType::get(ctx, 32);

    static int increment_loopId = 1;

    // Retrieve or create the LLVM function 'arm64_prof_func'.
    auto getOrCreateInstrFunc = [&]() {
      if (!module.lookupSymbol<LLVM::LLVMFuncOp>(
              "arm64_prof_func")) {
        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToStart(module.getBody());
        auto voidTy = LLVM::LLVMVoidType::get(ctx);
        auto funcTy = LLVM::LLVMFunctionType::get(
            voidTy, ArrayRef<Type>{i32Ty}, false);
        auto func = builder.create<LLVM::LLVMFuncOp>(
            module.getLoc(), "arm64_prof_func", funcTy);
      }
    };

    // Create the call to the profiling function
    auto emitInstrumentationCall = [&](OpBuilder builder, Location loc,
                                       Value id) {
      auto symbolRef =
          FlatSymbolRefAttr::get(ctx, "arm64_prof_func");
      auto instr = builder.create<LLVM::CallOp>(loc, TypeRange{}, symbolRef,
                                                ValueRange{id});
    };

    getOrCreateInstrFunc();

    // === Instrument Functions ===
    {
      Location loc = func.getLoc();
      std::vector<int> lines = collectLoc(loc);
      printLines(lines, increment_loopId, {});

      OpBuilder builder(func.getBody());

      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(&func.getBody().front());

      auto id = builder.create<LLVM::ConstantOp>(
          loc, i32Ty, builder.getI32IntegerAttr(increment_loopId++));

      emitInstrumentationCall(builder, loc, id);

      func.walk([&](func::ReturnOp retOp) {
        builder.setInsertionPoint(retOp);
        emitInstrumentationCall(builder, loc, id);
      });
    }

    if (!prof_disable_loop) {
      // === Instrument Loops ===
      func.walk([&](scf::ForOp forOp) {
        int depth = getLoopDepth(forOp);

        bool shouldInstrument =
            (depth == 0) || (depth <= prof_loop_depth && hasSiblingLoop(forOp));

        if (!shouldInstrument)
          return;

        Location loc = forOp.getLoc();
        OpBuilder builder(forOp);

        std::vector<std::string> opNames = collectOps(forOp);
        std::vector<int> lines = collectLoc(loc);

        printLines(lines, increment_loopId, opNames);

        auto id = builder.create<LLVM::ConstantOp>(
            loc, i32Ty, builder.getI32IntegerAttr(increment_loopId++));

        {
          OpBuilder::InsertionGuard guard(builder);
          builder.setInsertionPoint(forOp);
          emitInstrumentationCall(builder, loc, id);
        }

        {
          OpBuilder::InsertionGuard guard(builder);
          builder.setInsertionPointAfter(forOp);
          emitInstrumentationCall(builder, loc, id);
        }
      });
    }
  }
};
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
triton::createTritonARM64ProfPass(const TritonARM64ProfPassOptions &options) {
  return std::make_unique<TritonARM64ProfPass>(options);
}

