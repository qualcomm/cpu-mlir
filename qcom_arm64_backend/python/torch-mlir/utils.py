# ===- utils.py -------------------------------------------------------------===
# 
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  SPDX-License-Identifier: BSD-3-Clause.
#  For more license information:
#    https://github.com/qualcomm/cpu-mlir/LICENSE.txt
# 
# ===------------------------------------------------------------------------===

import sys
import torch
import torch_mlir.fx as fx_mlir
from pathlib import Path
import subprocess
import os

VECTOR_EXT_OPTIONS = {
    "neon": {"vector_ext": "neon", "arch_features": "+neon"},
    "sve":  {"vector_ext": "sve",  "arch_features": "+sve,+sve2"},
    "sme":  {"vector_ext": "sme",  "arch_features": "+sve,+sve2,+sme"},
}

from triton.backends.qcom_arm64_backend.torch_mlir_launcher import TorchMLIRLauncher
from triton.backends.qcom_arm64_backend.utils import get_work_dir

def get_vector_ext_options_from_args(default="sme"):
    mode = sys.argv[1] if len(sys.argv) > 1 else default
    if mode not in VECTOR_EXT_OPTIONS:
        print(f"Usage: {sys.argv[0]} [neon|sve|sme]")
        sys.exit(1)
    return VECTOR_EXT_OPTIONS[mode]


class ModelManager:
    def __init__(self, model, *inputs, dump=False):
        self.model = model
        self.func_name = model.__class__.__name__
        self.linalg_filename = os.path.join(get_work_dir(), f"{self.func_name}.mlirbc")
        self.inputs = inputs
        self.dump_outputs = dump

    # Convert the model to MLIR
    def create_linalg_module(self):
        self.mlir_module = fx_mlir.export_and_import(
            self.model,
            *self.inputs,
            output_type="linalg-on-tensors",
            func_name=self.func_name,
        )

    # Save the linalg module to a bytecode file
    def write_bytecode_to_file(self):
        bytecode = self.mlir_module.operation.get_asm(binary=True)
        # Save the linalg module as a bytecode in a file
        with open(self.linalg_filename, "wb") as f:
            f.write(bytecode)

    def execute_and_compare(self, rtol=1e-05, atol=1e-08, options={}, separate_constants=False, reference=None):
        # Generate reference output, unless the caller gives one
        if reference is None:
            reference = self.model(*self.inputs)

        # ARM64 execution
        output = TorchMLIRLauncher().run_torch_mlir(
            str(self.linalg_filename), self.inputs, self.func_name, options, separate_constants
        )

        # Compare results
        if len(output) == 1:
            if self.dump_outputs:
                print("\nReference output:\n", reference.shape, reference)
                print("\narm64 output:\n", output[0].shape, output[0])

            match = torch.allclose(output[0], reference, rtol, atol)
            print("Match: ", match)
            return match

        elif len(output) > 1:
            matches = []
            for index in range(len(output)):
                if self.dump_outputs:
                    print("\nReference output:\n", reference[index])
                    print("\narm64 output:\n", output[index])

                match = torch.allclose(output[index], reference[index], rtol, atol)
                print("Match: ", match)
                matches.append(match)
            return all(matches)
