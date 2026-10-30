# ===- test_sum_reduce_triton.py --------------------------------------------===
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  SPDX-License-Identifier: BSD-3-Clause.
#  For more license information:
#    https://github.com/qualcomm/cpu-mlir/LICENSE.txt
#
# ===------------------------------------------------------------------------===

import math

import pytest
import torch

import triton
import triton.language as tl

SUM_REDUCTION_CASES = (
    ((128,), (0,), "r1_p2"),
    ((127,), (0,), "r1_ragged"),
    ((8, 128), (0,), "r2_outer"),
    ((8, 128), (1,), "r2_inner"),
    ((7, 127), (1,), "r2_ragged_inner"),
    ((4, 8, 128), (1,), "r3_middle"),
    ((4, 8, 128), (2,), "r3_inner"),
    ((3, 7, 127), (1, 2), "r3_tail_multi_ragged"),
    ((2, 4, 8, 128), (0, 2), "r4_noncontiguous_multi"),
    ((2, 3, 7, 127), (0, 1, 2, 3), "r4_full_ragged"),
)

TRITON_CASES = [
    pytest.param(shape, axes, case_id, id=case_id)
    for shape, axes, case_id in SUM_REDUCTION_CASES
]


@triton.jit
def sum_reduce_rank1_kernel(input_ptr, output_ptr, d0: tl.constexpr,
                            block_size: tl.constexpr):
    offsets = tl.arange(0, block_size)
    value = tl.sum(tl.load(input_ptr + offsets, mask=offsets < d0, other=0.0), axis=0)
    tl.store(output_ptr, value)


@triton.jit
def sum_reduce_rank2_ragged_inner_kernel(input_ptr, output_ptr,
                                         d0: tl.constexpr, d1: tl.constexpr,
                                         p0: tl.constexpr, p1: tl.constexpr):
    offsets = tl.arange(0, p0 * p1)
    row = offsets // p1
    col = offsets % p1
    mask = (row < d0) & (col < d1)
    value = tl.load(input_ptr + row * d1 + col, mask=mask, other=0.0)
    result = tl.sum(value.reshape(p0, p1), axis=1)
    output_offsets = tl.arange(0, p0)
    tl.store(output_ptr + output_offsets, result, mask=output_offsets < d0)


@triton.jit
def sum_reduce_rank3_ragged_tail_kernel(input_ptr, output_ptr,
                                        d0: tl.constexpr, d1: tl.constexpr,
                                        d2: tl.constexpr, p0: tl.constexpr,
                                        p1: tl.constexpr, p2: tl.constexpr):
    offsets = tl.arange(0, p0 * p1 * p2)
    i0 = offsets // (p1 * p2)
    i1 = (offsets // p2) % p1
    i2 = offsets % p2
    mask = (i0 < d0) & (i1 < d1) & (i2 < d2)
    value = tl.load(
        input_ptr + (i0 * d1 + i1) * d2 + i2, mask=mask, other=0.0
    ).reshape(p0, p1, p2)
    result = tl.sum(tl.sum(value, axis=2), axis=1)
    output_offsets = tl.arange(0, p0)
    tl.store(output_ptr + output_offsets, result, mask=output_offsets < d0)


@triton.jit
def sum_reduce_rank4_ragged_full_kernel(input_ptr, output_ptr,
                                        d0: tl.constexpr, d1: tl.constexpr,
                                        d2: tl.constexpr, d3: tl.constexpr,
                                        p0: tl.constexpr, p1: tl.constexpr,
                                        p2: tl.constexpr, p3: tl.constexpr):
    offsets = tl.arange(0, p0 * p1 * p2 * p3)
    i0 = offsets // (p1 * p2 * p3)
    i1 = (offsets // (p2 * p3)) % p1
    i2 = (offsets // p3) % p2
    i3 = offsets % p3
    mask = (i0 < d0) & (i1 < d1) & (i2 < d2) & (i3 < d3)
    value = tl.load(
        input_ptr + ((i0 * d1 + i1) * d2 + i2) * d3 + i3,
        mask=mask,
        other=0.0,
    ).reshape(p0, p1, p2, p3)
    value = tl.sum(tl.sum(tl.sum(tl.sum(value, axis=3), axis=2), axis=1), axis=0)
    tl.store(output_ptr, value)


@triton.jit
def sum_reduce_rank2_kernel(input_ptr, output_ptr, d0: tl.constexpr,
                            d1: tl.constexpr, reduction_mode: tl.constexpr,
                            output_numel: tl.constexpr):
    input_desc = tl.make_tensor_descriptor(
        base=input_ptr, shape=[d0 * d1], strides=[1], block_shape=[d0 * d1]
    )
    output_desc = tl.make_tensor_descriptor(
        base=output_ptr, shape=[output_numel], strides=[1],
        block_shape=[output_numel]
    )
    value = input_desc.load([0]).reshape(d0, d1)
    if reduction_mode == 0:
        value = tl.sum(value, axis=0)
    else:
        value = tl.sum(value, axis=1)
    output_desc.store([0], value.reshape(output_numel))


@triton.jit
def sum_reduce_rank3_kernel(input_ptr, output_ptr, d0: tl.constexpr,
                            d1: tl.constexpr, d2: tl.constexpr,
                            reduction_mode: tl.constexpr,
                            output_numel: tl.constexpr):
    input_desc = tl.make_tensor_descriptor(
        base=input_ptr, shape=[d0 * d1 * d2], strides=[1],
        block_shape=[d0 * d1 * d2]
    )
    output_desc = tl.make_tensor_descriptor(
        base=output_ptr, shape=[output_numel], strides=[1],
        block_shape=[output_numel]
    )
    value = input_desc.load([0]).reshape(d0, d1, d2)
    if reduction_mode == 0:
        value = tl.sum(value, axis=1)
    elif reduction_mode == 1:
        value = tl.sum(value, axis=2)
    else:
        value = tl.sum(tl.sum(value, axis=2), axis=1)
    output_desc.store([0], value.reshape(output_numel))


@triton.jit
def sum_reduce_rank4_kernel(input_ptr, output_ptr, d0: tl.constexpr,
                            d1: tl.constexpr, d2: tl.constexpr,
                            d3: tl.constexpr, reduction_mode: tl.constexpr,
                            output_numel: tl.constexpr):
    input_desc = tl.make_tensor_descriptor(
        base=input_ptr, shape=[d0 * d1 * d2 * d3], strides=[1],
        block_shape=[d0 * d1 * d2 * d3]
    )
    output_desc = tl.make_tensor_descriptor(
        base=output_ptr, shape=[output_numel], strides=[1],
        block_shape=[output_numel]
    )
    value = input_desc.load([0]).reshape(d0, d1, d2, d3)
    if reduction_mode == 0:
        value = tl.sum(tl.sum(value, axis=2), axis=0)
    else:
        value = tl.sum(tl.sum(tl.sum(tl.sum(value, axis=3), axis=2), axis=1), axis=0)
    output_desc.store([0], value.reshape(output_numel))


def torch_sum(value, axes):
    for axis in sorted(axes, reverse=True):
        value = torch.sum(value, dim=axis)
    return value


def next_power_of_2(value):
    return 1 << (value - 1).bit_length()


def is_power_of_2(value):
    return value > 0 and value & (value - 1) == 0


def run_sum_reduce(value, axes):
    shape = tuple(value.shape)
    expected = torch_sum(value, axes)
    output = torch.empty_like(expected)
    output_numel = math.prod(expected.shape) if expected.ndim else 1

    if len(shape) == 1:
        sum_reduce_rank1_kernel[(1,)](
            value, output, shape[0], next_power_of_2(shape[0])
        )
    elif len(shape) == 2:
        if axes == (1,) and not is_power_of_2(math.prod(shape)):
            p0, p1 = (next_power_of_2(dim) for dim in shape)
            sum_reduce_rank2_ragged_inner_kernel[(1,)](
                value, output, *shape, p0, p1
            )
        else:
            reduction_mode = 0 if axes == (0,) else 1
            sum_reduce_rank2_kernel[(1,)](
                value, output, *shape, reduction_mode, output_numel
            )
    elif len(shape) == 3:
        if axes == (1, 2) and not is_power_of_2(math.prod(shape)):
            p0, p1, p2 = (next_power_of_2(dim) for dim in shape)
            sum_reduce_rank3_ragged_tail_kernel[(1,)](
                value, output, *shape, p0, p1, p2
            )
        else:
            reduction_mode = {(1,): 0, (2,): 1, (1, 2): 2}[axes]
            sum_reduce_rank3_kernel[(1,)](
                value, output, *shape, reduction_mode, output_numel
            )
    else:
        if axes == (0, 1, 2, 3) and not is_power_of_2(math.prod(shape)):
            p0, p1, p2, p3 = (next_power_of_2(dim) for dim in shape)
            sum_reduce_rank4_ragged_full_kernel[(1,)](
                value, output, *shape, p0, p1, p2, p3
            )
        else:
            reduction_mode = 0 if axes == (0, 2) else 1
            sum_reduce_rank4_kernel[(1,)](
                value, output, *shape, reduction_mode, output_numel
            )
    return output, expected


@pytest.mark.parametrize(
    "shape, axes, _",
    TRITON_CASES,
)
def test_sum_reduce_triton(shape, axes, _, device):
    torch.manual_seed(0)
    value = torch.rand(shape, device=device, dtype=torch.float32)
    output, expected = run_sum_reduce(value, axes)
    torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-6)
