# ===- test_transpose.py ----------------------------------------------------===
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  SPDX-License-Identifier: BSD-3-Clause.
#  For more license information:
#    https://github.com/qualcomm/cpu-mlir/LICENSE.txt
#
# ===------------------------------------------------------------------------===
#
# tl.trans on a block. On SME, a block that swaps its two innermost dimensions
# and has at least 2^19 elements is transposed through an SME ZA tile; smaller
# blocks, and transposes that keep the innermost dimension, take the regular
# path. The matrix sizes are not multiples of the block or tile sizes, so the
# masked edges are covered.

import pytest

import torch
import triton
import triton.language as tl


@triton.jit
def transpose_2d_kernel(x_ptr, out_ptr, M, N,
                        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    rm = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    x = tl.load(x_ptr + rm[:, None] * N + rn[None, :],
                mask=(rm[:, None] < M) & (rn[None, :] < N), other=0)
    # out[n, m] = x[m, n]
    tl.store(out_ptr + rn[:, None] * M + rm[None, :], tl.trans(x),
             mask=(rn[:, None] < N) & (rm[None, :] < M))


@triton.jit
def transpose_3d_kernel(x_ptr, out_ptr, B: tl.constexpr, M, N,
                        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                        INNER: tl.constexpr):
    rb = tl.arange(0, B)
    rm = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    x = tl.load(x_ptr + rb[:, None, None] * M * N + rm[None, :, None] * N + rn[None, None, :],
                mask=(rm[None, :, None] < M) & (rn[None, None, :] < N), other=0)
    if INNER:
        # [B, M, N] -> [B, N, M]: swaps the two innermost dimensions.
        tl.store(out_ptr + rb[:, None, None] * N * M + rn[None, :, None] * M + rm[None, None, :],
                 tl.trans(x, 0, 2, 1),
                 mask=(rn[None, :, None] < N) & (rm[None, None, :] < M))
    else:
        # [B, M, N] -> [M, B, N]: keeps the innermost dimension.
        tl.store(out_ptr + rm[:, None, None] * B * N + rb[None, :, None] * N + rn[None, None, :],
                 tl.trans(x, 1, 0, 2),
                 mask=(rm[:, None, None] < M) & (rn[None, None, :] < N))


def _random(shape, dtype, device):
    if dtype.is_floating_point:
        return torch.randn(shape, device=device).to(dtype)
    return torch.randint(-128, 128, shape, dtype=dtype, device=device)


DTYPES = [torch.float32, torch.float16, torch.int8]


# (M, N, BLOCK_M, BLOCK_N): 32x64 blocks take the regular path, 1024x512
# blocks (2^19 elements) the ZA tile path.
@pytest.mark.parametrize("M, N, BLOCK_M, BLOCK_N",
                         [(100, 70, 32, 64), (1031, 1029, 1024, 512)])
@pytest.mark.parametrize("dtype", DTYPES)
def test_transpose_2d(M, N, BLOCK_M, BLOCK_N, dtype, device):
    torch.manual_seed(0)
    x = _random((M, N), dtype, device)
    out = torch.empty((N, M), dtype=dtype, device=device)
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    transpose_2d_kernel[grid](x, out, M, N, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)
    # A transpose only moves values, so the result must be exact.
    torch.testing.assert_close(out, x.t(), atol=0, rtol=0)


# (B, M, N, BLOCK_M, BLOCK_N): 2x512x512 blocks (2^19 elements) take the ZA
# tile path for the inner swap.
@pytest.mark.parametrize("B, M, N, BLOCK_M, BLOCK_N",
                         [(2, 40, 70, 32, 64), (2, 515, 517, 512, 512)])
@pytest.mark.parametrize("inner", [True, False])
def test_transpose_3d(B, M, N, BLOCK_M, BLOCK_N, inner, device):
    torch.manual_seed(0)
    x = _random((B, M, N), torch.float32, device)
    ref = x.transpose(1, 2) if inner else x.transpose(0, 1)
    out = torch.empty(ref.shape, dtype=x.dtype, device=device)
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    transpose_3d_kernel[grid](x, out, B, M, N, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
                              INNER=inner)
    torch.testing.assert_close(out, ref.contiguous(), atol=0, rtol=0)
