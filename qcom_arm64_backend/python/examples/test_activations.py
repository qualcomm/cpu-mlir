# ===- test_activations.py --------------------------------------------------===
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  SPDX-License-Identifier: BSD-3-Clause.
#  For more license information:
#    https://github.com/qualcomm/cpu-mlir/LICENSE.txt
#
# ===------------------------------------------------------------------------===

import pytest

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

BLOCK = 1024
SQRT_2_OVER_PI = tl.constexpr(0.7978845608028654)
INV_SQRT_2 = tl.constexpr(0.7071067811865476)


@triton.jit
def _relu(x):
    return tl.maximum(x, 0.0)


@triton.jit
def _leaky_relu(x):
    return tl.where(x >= 0.0, x, 0.01 * x)


@triton.jit
def _sigmoid(x):
    return tl.sigmoid(x)


@triton.jit
def _silu(x):
    return x * tl.sigmoid(x)


@triton.jit
def _tanh(x):
    # No tl tanh on CPU; this form does not overflow for large |x|.
    return 2.0 * tl.sigmoid(2.0 * x) - 1.0


@triton.jit
def _gelu_erf(x):
    return 0.5 * x * (1.0 + tl.math.erf(x * INV_SQRT_2))


@triton.jit
def _gelu_tanh(x):
    inner = SQRT_2_OVER_PI * (x + 0.044715 * x * x * x)
    return 0.5 * x * (1.0 + (2.0 * tl.sigmoid(2.0 * inner) - 1.0))


def _make_kernel(body):
    @triton.jit
    def kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask)
        tl.store(out_ptr + offs, body(x), mask=mask)
    return kernel


# name: (triton body, torch reference, atol)
ACTIVATIONS = {
    "relu": (_relu, torch.relu, 0.0),
    "leaky_relu": (_leaky_relu, lambda x: F.leaky_relu(x, 0.01), 1e-7),
    "sigmoid": (_sigmoid, torch.sigmoid, 1e-6),
    "silu": (_silu, F.silu, 1e-5),
    "tanh": (_tanh, torch.tanh, 1e-6),
    "gelu_erf": (_gelu_erf, F.gelu, 1e-5),
    "gelu_tanh": (_gelu_tanh, lambda x: F.gelu(x, approximate="tanh"), 1e-5),
}
KERNELS = {name: _make_kernel(body) for name, (body, _, _) in ACTIVATIONS.items()}


def run_activation(name, x):
    out = torch.empty_like(x)
    n = x.numel()
    KERNELS[name][(triton.cdiv(n, BLOCK),)](x, out, n, BLOCK=BLOCK)
    return out


@pytest.mark.parametrize("name", ACTIVATIONS.keys())
@pytest.mark.parametrize("n", [1, BLOCK, 3 * BLOCK + 7])
def test_activation(name, n, device):
    torch.manual_seed(0)
    x = torch.randn(n, device=device) * 4
    _, ref, atol = ACTIVATIONS[name]
    torch.testing.assert_close(run_activation(name, x), ref(x), atol=atol, rtol=1e-5)


@pytest.mark.parametrize("name", ACTIVATIONS.keys())
def test_activation_edge_values(name, device):
    # Saturation regions and signed zeros, where approximations usually break.
    x = torch.tensor([0.0, -0.0, 1e-7, -1e-7, 5.0, -5.0, 10.0, -10.0, 30.0, -30.0,
                      88.0, -88.0, 1e4, -1e4], device=device)
    if name == "gelu_tanh":
        x = x[x.abs() < 1e3]  # the cubic term overflows to inf in fp32 for huge |x|
    _, ref, atol = ACTIVATIONS[name]
    torch.testing.assert_close(run_activation(name, x), ref(x), atol=atol, rtol=1e-5)
