# ===- torch_mlir_launcher.py -----------------------------------------------===
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause.
# For more license information:
#   https://github.com/qualcomm/cpu-mlir/LICENSE.txt
#
# ===------------------------------------------------------------------------===

import os, tempfile
import torch
import shutil
from pathlib import Path
from typing import Optional
import subprocess
# import time
from torch import Tensor  # For type annotations
from triton._C.libtriton import qcom_arm64_backend, ir

from triton.backends.qcom_arm64_backend.utils import get_work_dir, profile_torch_mlir_inputs, parse_return_types, is_function_streaming, get_verbose_level, get_debug_mode, get_codegen_options, ctype_to_torch_type
from triton.backends.qcom_arm64_backend.driver import ARM64TorchLauncher
from triton.backends.qcom_arm64_backend.compiler import ARM64CPUOptions
from triton.backends.qcom_arm64_backend.launcher_host import generate_torch_mlir_host_launcher

import sys
sys.path.insert(0, os.path.join(os.getenv("TRITON_SHARED_ROOT"), "backend"))
from driver import compile_module

def _get_llvm_bin_path(bin_name: str) -> str:
    path = os.getenv("LLVM_BINARY_DIR", "")
    if path == "":
        raise Exception("LLVM_BINARY_DIR is not set.")
    return os.path.join(path, bin_name)

def _llvmmlir_to_llir(mlir_text: str, out_path: str) -> None:
    """Write MLIR text to a temp file and translate it to LLVM IR at out_path."""
    mlir_path = out_path.replace(".ll", ".mlir")
    Path(mlir_path).write_text(mlir_text)
    mlir_translate_path = _get_llvm_bin_path("mlir-translate")
    subprocess.check_call([
        mlir_translate_path, mlir_path,
        "--mlir-to-llvmir",
        "-o", out_path,
    ])
class ARM64HostTorchLauncher(ARM64TorchLauncher):
    def invoke(self, llir_path, inputs):
        llc_path = _get_llvm_bin_path("llc")
        with tempfile.TemporaryDirectory() as tmpdir:
            obj_path = os.path.join(tmpdir, "kernel.o")
            subprocess.check_call([llc_path, llir_path, "-filetype=obj",
                                    "-relocation-model=pic", "-o", obj_path])
            kernel_obj = Path(obj_path).read_bytes()

        output_tensors = [
            torch.empty(r.sizes, dtype=ctype_to_torch_type(r.dtype))
            for r in self.return_types
        ]

        with open(llir_path, 'r') as f:
            llir = f.read()
        is_streaming = is_function_streaming(llir, f"_mlir_ciface_{self.kernel_name}")

        launcher_src = generate_torch_mlir_host_launcher(self.input_profs, self.return_types, self.kernel_name, is_streaming)
        launch = compile_module(launcher_src, self.kernel_name)
        kernel_metadata = (None, None, None, None, None, None, self.kernel_name)

        launch(1, 1, 1, None, kernel_obj, kernel_metadata, None, None, None, *output_tensors, *inputs)
        return output_tensors

class TorchMLIRLauncher():
    def compile_torch_mlir(
        self,
        mlir_bytecode_path: str,
        func_name: str,
        user_options: dict,
        separate_constants: bool = False,
    ) -> tuple[str, list[str]]:
        """Compile an MLIR bytecode file to LLVM IR.

        When separate_constants is False (default), constants are lowered
        together with the code into a single LLVM IR file, and the path to
        that file is returned.

        When separate_constants is True, constants are separated into one or
        more additional LLVM IR files. The returned tuple is (main_llir_path, result_types)
        """

        # Context is being initialized through triton source code - python/src/ir.cc
        # This creates a dependency on triton code and
        # will cause issue when we separate torch-mlir workflow from triton.
        # TODO: We need to define the context in arm64 backend.
        context = ir.context()
        qcom_arm64_backend.load_dialects(context)
        mlir_mod = qcom_arm64_backend.parse_mlir_module_from_file(mlir_bytecode_path, context)
        result_types = qcom_arm64_backend.get_return_list(context, mlir_mod, func_name)
        ir_dir = get_work_dir()

        # We should consider the ability to dump the original Torch MLIR here
        # (but avoid dumping constant data)
        # print(str(mlir_mod))

        options = ARM64CPUOptions(**{k: v for k, v in user_options.items() if k in ARM64CPUOptions.__dataclass_fields__})

        optimized = qcom_arm64_backend.optimize_triton_shared_mlir(str(mlir_mod), context, options.vector_ext, get_verbose_level(), get_debug_mode())

        const_llir_paths = []
        if not separate_constants:
            llmlir = qcom_arm64_backend.triton_linalg_to_llvm_mlir(str(optimized), context, get_codegen_options(options), get_verbose_level(), get_debug_mode())

            llmlir_path = os.path.join(ir_dir, f"{func_name}-llvm-dialect.mlir")
            Path(llmlir_path).write_text(llmlir)

            llir_path = os.path.join(ir_dir, f"{func_name}-llvm-ir.ll")
            _llvmmlir_to_llir(llmlir, llir_path)

        else:
            # Returns a list of MLIR strings: index 0 is the main module
            # (constants replaced by externs), indices 1..N are constant-only modules.
            all_mlir_strings = qcom_arm64_backend.triton_linalg_to_llvm_mlir_separate_constants(
                str(optimized), context, get_codegen_options(options), get_verbose_level(), get_debug_mode()
            )

            nb_modules = len(all_mlir_strings)
            print(f"Separate-constants lowering produced {nb_modules} module(s) "
                  f"({nb_modules - 1} constant-only).")

            # Translate main module (index 0) to LLVM IR
            llir_path = os.path.join(ir_dir, f"{func_name}-llvm-ir.ll")
            _llvmmlir_to_llir(all_mlir_strings[0], llir_path)

            # Translate each constant-only module (indices 1..N) to LLVM IR
            for i, const_mlir in enumerate(all_mlir_strings[1:], start=1):
                const_llir_path = os.path.join(ir_dir, f"{func_name}-consts-{i}-llvm-ir.ll")
                _llvmmlir_to_llir(const_mlir, const_llir_path)
                const_llir_paths.append(const_llir_path)
                print(f"==> Constant module {i} saved as LLVM IR: {const_llir_path}")

        return llir_path, result_types, const_llir_paths

    # Toplevel function that compiles the mlir and execute the result.
    # Executes either on device or on simulator depending on the env var RUN_ON_SIM.
    def run_torch_mlir(
        self,
        mlir_bytecode_path: str,
        inputs: list[Tensor],
        func_name: str,
        options: dict,
        separate_constants: bool = False,
    ) -> list[Tensor]:

        main_llir_path, result_types, const_llir_paths = self.compile_torch_mlir(
            mlir_bytecode_path, func_name, options, separate_constants)

        return_types = parse_return_types(result_types)
        input_profs = profile_torch_mlir_inputs(inputs)

        if os.getenv("TRITON_ARM64_HOST_BUILD", '0') == '1':
            torchlauncher = ARM64HostTorchLauncher(func_name, input_profs, return_types)
        else:
            torchlauncher = ARM64TorchLauncher(func_name, input_profs, return_types)
        results = torchlauncher.invoke(
            main_llir_path, inputs,
            const_llir_paths=const_llir_paths
        )

        return results
