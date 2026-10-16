# ===- test_activations_torch.py --------------------------------------------===
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  SPDX-License-Identifier: BSD-3-Clause.
#  For more license information:
#    https://github.com/qualcomm/cpu-mlir/LICENSE.txt
#
# ===------------------------------------------------------------------------===

# The activations of examples/test_activations.py, compiled from PyTorch through
# torch-mlir instead of Triton.

import torch
import torch.nn as nn
import utils


def gelu_tanh_as_imported(x):
    # torch-mlir lowers GELU(approximate="tanh") with sqrt(2/pi) = 0.797724
    # instead of 0.7978846, which alone moves outputs by up to 5e-5. Check the
    # backend against the formula it is given, not against PyTorch.
    return 0.5 * x * (1.0 + torch.tanh(0.797724 * (x + 0.044715 * x * x * x)))


# name: (module, atol, reference; None = the module itself)
ACTIVATIONS = {
    "relu": (nn.ReLU, 0.0, None),
    "leaky_relu": (lambda: nn.LeakyReLU(0.01), 1e-7, None),
    "sigmoid": (nn.Sigmoid, 1e-6, None),
    "silu": (nn.SiLU, 1e-5, None),
    "tanh": (nn.Tanh, 1e-6, None),
    "gelu_erf": (nn.GELU, 1e-5, None),
    "gelu_tanh": (lambda: nn.GELU(approximate="tanh"), 1e-5, gelu_tanh_as_imported),
}


def make_input():
    torch.manual_seed(0)
    # A size that is not a multiple of any vector length, plus the saturation
    # regions and signed zeros, where approximations usually break.
    edges = torch.tensor([0.0, -0.0, 1e-7, -1e-7, 5.0, -5.0, 10.0, -10.0, 30.0, -30.0,
                          88.0, -88.0, 1e4, -1e4])
    return torch.cat([torch.randn(3 * 1024 + 7) * 4, edges])


def test_activations_torch(options={}):
    failed = []
    for name, (make_module, atol, reference) in ACTIVATIONS.items():
        model = make_module()
        # Each activation needs its own function name (and so its own files).
        model.__class__ = type(f"Activation_{name}", (model.__class__,), {})
        print(f"== {name}")
        x = make_input()
        manager = utils.ModelManager(model, x)
        manager.create_linalg_module()
        manager.write_bytecode_to_file()
        if not manager.execute_and_compare(rtol=1e-5, atol=atol, options=options,
                                           reference=reference(x) if reference else None):
            failed.append(name)
    assert not failed, f"mismatch: {failed}"


if __name__ == "__main__":
    test_activations_torch(utils.get_vector_ext_options_from_args())
