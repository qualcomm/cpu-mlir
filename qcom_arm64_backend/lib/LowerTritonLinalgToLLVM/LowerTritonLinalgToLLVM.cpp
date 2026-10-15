//===- LowerTritonLinalgToLLVM.cpp ----------------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/cpu-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

#include "LowerTritonLinalgToLLVM/LowerTritonLinalgToLLVM.h"
#include "LowerTritonLinalgToLLVM/LowerConstantsSeparately.h"

#include "triton/Target/LLVMIR/Passes.h"

#include "mlir/Conversion/Passes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/Transforms/RequestCWrappers.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SCF/Transforms/Passes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include "mlir/InitAllPasses.h"
#include "llvm/Support/Debug.h"

#include "LowerTritonLinalgToLLVM/Passes.h"

#include <cstdlib>
#include <string>
#include <unordered_map>

namespace mlir {
namespace triton {

#define GEN_PASS_DEF_TRITONARM64PROFPASS
#include "LowerTritonLinalgToLLVM/Passes.h.inc"

bool doesFuncReturnValue(ModuleOp moduleOp) {

  return moduleOp
      .walk([&](func::FuncOp op) {
        if (!op.getResultTypes().empty())
          return WalkResult::interrupt();
        return WalkResult::advance();
      }).wasInterrupted();
}

void addTritonLinalgToLLVMPasses(
    mlir::OpPassManager &pm, arm64VectorExtension vec_ext, TritonARM64ProfPassOptions prof_options, int debug_mode) {
  // TODO: LowerTmTensor pass produces linalg operations, so it should be moved
  //       before our linalg optimization passes
  pm.addNestedPass<func::FuncOp>(createTritonARM64LowerTmTensorPass());
  pm.addPass(createCanonicalizerPass());

  mlir::bufferization::OneShotBufferizePassOptions passOpts;
  passOpts.allowReturnAllocsFromLoops = true;
  passOpts.bufferizeFunctionBoundaries = true; // to enforce lowering of bufferization.to_buffer
  passOpts.allowUnknownOps = true; // torch-mlir emits to_tensor ops without `restrict` 
  pm.addPass(bufferization::createOneShotBufferizePass(passOpts));
  pm.addPass(bufferization::createEmptyTensorEliminationPass());
  pm.addPass(bufferization::createEmptyTensorToAllocTensorPass());

  pm.addPass(createConvertLinalgToLoopsPass());
  pm.addPass(createLowerAffinePass());
  pm.addNestedPass<func::FuncOp>(createConvertVectorToSCFPass());
  const char *envProfVal = std::getenv("TRITON_ARM64_PROFILE");
  if (envProfVal && std::string(envProfVal) == "1")
    pm.addNestedPass<func::FuncOp>(createTritonARM64ProfPass(prof_options));
  pm.addPass(createSCFToControlFlowPass());
  pm.addPass(createConvertControlFlowToLLVMPass());

  // Must run before ConvertMathToLLVM: math.erf has no LLVM lowering, and
  // vector math.exp would otherwise be scalarized into libm calls.
  pm.addPass(createTritonARM64MathApproximationPass());
  ConvertVectorToLLVMPassOptions options{};
  options.armSVE = (vec_ext == SVE || vec_ext == SME);
  pm.addPass(createConvertVectorToLLVMPass(options));
  pm.addPass(createConvertMathToLLVMPass());
  pm.addPass(createConvertComplexToLLVMPass());
  pm.addPass(createConvertIndexToLLVMPass());

  pm.addPass(memref::createExpandOpsPass());
  pm.addPass(memref::createExpandStridedMetadataPass());
  pm.addPass(createFinalizeMemRefToLLVMConversionPass());
  pm.addPass(createConvertFuncToLLVMPass());
  pm.addPass(createLowerAffinePass());
  pm.addPass(createArithToLLVMConversionPass());

  pm.addPass(createUBToLLVMConversionPass());
  pm.addPass(createReconcileUnrealizedCastsPass());

  if(debug_mode) {
    // Attach DISubprogramAttr to every LLVMFuncOp so that mlir-translate
    // --mlir-to-llvmir emits !dbg metadata for all ops in the function.
    pm.addPass(mlir::createLLVMDIScope());
  }
}

ModuleOp arm64TritonLinalgToLLVMMLIR(ModuleOp mod, arm64VectorExtension vec_ext,
    std::unordered_map<std::string, std::string> prof_options_map, int verbose_level, int debug_mode) {
  MLIRContext* context = mod->getContext();
  mlir::PassManager pm(context);

  std::string targetTripleStr = "aarch64";
  mod->setAttr(LLVM::LLVMDialect::getTargetTripleAttrName(),
               StringAttr::get(mod->getContext(), targetTripleStr));

  if (verbose_level > 1) {
    llvm::dbgs() << "Lowering optimized MLIR with "
                 << ARM64VectorExtensionNames[vec_ext] << "\n";
  }

  TritonARM64ProfPassOptions prof_options;
  prof_options.prof_disable_loop = !prof_options_map.at("prof_disable_loop").compare("True");
  prof_options.prof_loop_depth = std::stoi(prof_options_map.at("prof_loop_depth"));

  if (doesFuncReturnValue(mod))
    pm.addNestedPass<func::FuncOp>(LLVM::createLLVMRequestCWrappersPass());
  addTritonLinalgToLLVMPasses(pm, vec_ext, prof_options, debug_mode);

  enablePrintIR(pm, debug_mode);

  // ARM64VectorLoweringPass runs its own bufferization which may leave behind
  // to_tensor ops without `restrict`. Mark them all restrict before the
  // OneShotBufferize pass runs, otherwise it will reject them.
  mod.walk([&](bufferization::ToTensorOp op) {
    if (!op.getRestrict()) {
      op.setRestrict(true);
    }
  });

  if (failed(pm.run(mod))) {
    llvm::errs() << "Triton Linalg To LLVM MLIR Pass execution failed.\n";
    return nullptr;
  }
  return mod;
}

// -------------------------------------------------
// --- Translating to multiple LLVM/MLIR modules ---
// -------------------------------------------------

// A PassManager subclass that runs the standard lowering passes followed by
// LowerConstantsSeparatelyPass, and exposes the list of constant-only modules
// produced by that pass.
class CustomPassManager : public mlir::PassManager {
public:
  CustomPassManager(mlir::MLIRContext *context,
                    arm64VectorExtension vec_ext,
                    std::unordered_map<std::string, std::string> prof_options_map,
                    bool doesReturnValue, int debug_mode)
      : mlir::PassManager(context),
        moduleSink(std::make_shared<std::vector<ModuleOp>>()) {
    TritonARM64ProfPassOptions prof_options;
    prof_options.prof_disable_loop =
        !prof_options_map.at("prof_disable_loop").compare("True");
    prof_options.prof_loop_depth =
        std::stoi(prof_options_map.at("prof_loop_depth"));

    if (doesReturnValue)
      addNestedPass<func::FuncOp>(LLVM::createLLVMRequestCWrappersPass());
    addTritonLinalgToLLVMPasses(*this, vec_ext, prof_options, debug_mode);

    // Pass ownership to the PassManager; results are written into moduleSink.
    addPass(createLowerConstantsSeparatelyPass(moduleSink));
  }

  std::vector<ModuleOp> getProducedModules() const {
    return *moduleSink;
  }

private:
  std::shared_ptr<std::vector<ModuleOp>> moduleSink;
};

std::vector<ModuleOp> arm64TritonLinalgToMultipleLLVMMLIRModules(
    ModuleOp mod,
    arm64VectorExtension vec_ext,
    std::unordered_map<std::string, std::string> prof_options_map,
    int verbose_level, int debug_mode) {

  MLIRContext *context = mod->getContext();

  std::string targetTripleStr = "aarch64";
  mod->setAttr(LLVM::LLVMDialect::getTargetTripleAttrName(),
               StringAttr::get(context, targetTripleStr));

  if (verbose_level > 1) {
    llvm::dbgs() << "Lowering optimized MLIR (separate constants) with "
                 << ARM64VectorExtensionNames[vec_ext] << "\n";
  }

  // ARM64VectorLoweringPass runs its own bufferization which may leave behind
  // to_tensor ops without `restrict`. Mark them all restrict before the
  // OneShotBufferize pass runs, otherwise it will reject them.
  mod.walk([&](bufferization::ToTensorOp op) {
    if (!op.getRestrict()) {
      OpBuilder builder(op);
      Value tensor = builder.create<bufferization::ToTensorOp>(
          op.getLoc(), op.getType(), op.getBuffer(),
          /*restrict=*/true, /*writable=*/op.getWritable());
      op.replaceAllUsesWith(tensor);
      op.erase();
    }
  });

  CustomPassManager pm(context, vec_ext, prof_options_map,
                       doesFuncReturnValue(mod), debug_mode);
  if (failed(pm.run(mod))) {
    llvm::errs() << "Triton Linalg To LLVM MLIR (separate constants) Pass execution failed.\n";
    return {};
  }

  // The constant-only modules produced by LowerConstantsSeparatelyPass
  std::vector<ModuleOp> all_modules = pm.getProducedModules();
  // Insert the main module (with constants replaced by externs) at the front
  all_modules.insert(all_modules.begin(), mod);
  return all_modules;
}

} // namespace triton
} // namespace mlir
