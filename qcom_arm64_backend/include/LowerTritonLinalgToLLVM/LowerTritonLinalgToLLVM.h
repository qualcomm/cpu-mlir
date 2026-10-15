//===- LowerTritonLinalgToLLVM.h ------------------------------------   ---===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/cpu-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_ARM64_LOWER_TRITON_LINALG_TO_LLVM_H
#define TRITON_ARM64_LOWER_TRITON_LINALG_TO_LLVM_H
#include "common.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include <memory>
#include <unordered_map>
#include <vector>

namespace mlir {
namespace triton {
#define GEN_PASS_DECL_TRITONARM64PROFPASS
#define GEN_PASS_DECL_TRITONARM64LOWERCONSTANTSSEPARATELY
#define GEN_PASS_DECL_TRITONARM64MATHAPPROXIMATION
#include "LowerTritonLinalgToLLVM/Passes.h.inc"

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createTritonARM64LowerTmTensorPass();

std::unique_ptr<OperationPass<func::FuncOp>> createTritonARM64ProfPass(
    const TritonARM64ProfPassOptions &options = TritonARM64ProfPassOptions()
);

std::unique_ptr<OperationPass<ModuleOp>> createLowerConstantsSeparatelyPass(
    std::shared_ptr<std::vector<ModuleOp>> sink = nullptr);

std::unique_ptr<OperationPass<ModuleOp>> createTritonARM64MathApproximationPass(
    const TritonARM64MathApproximationOptions &options =
        TritonARM64MathApproximationOptions());

} // namespace triton
} // namespace mlir

namespace mlir {
class ModuleOp;
} // namespace mlir

namespace mlir {

namespace triton {

ModuleOp arm64TritonLinalgToLLVMMLIR(ModuleOp module, arm64VectorExtension vec_ext,
    std::unordered_map<std::string, std::string> prof_options_map, int verbose_level, int debug_mode);

void addTritonLinalgToLLVMPasses(mlir::OpPassManager &pm,
    arm64VectorExtension vec_ext, TritonARM64ProfPassOptions prof_options,
    int debug_mode);

std::vector<ModuleOp> arm64TritonLinalgToMultipleLLVMMLIRModules(
    ModuleOp module, arm64VectorExtension vec_ext,
    std::unordered_map<std::string, std::string> prof_options_map, int verbose_level, int debug_mode);

} // namespace triton
} // namespace mlir

#endif // TRITON_ARM64_LOWER_TRITON_LINALG_TO_LLVM_H
