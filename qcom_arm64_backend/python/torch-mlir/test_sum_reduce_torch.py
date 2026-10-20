# ===- test_sum_reduce_torch.py ---------------------------------------------===
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  SPDX-License-Identifier: BSD-3-Clause.
#  For more license information:
#    https://github.com/qualcomm/cpu-mlir/LICENSE.txt
#
# ===------------------------------------------------------------------------===

import pytest
import torch
import torch.nn as nn

import utils

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


class SumReduce(nn.Module):
    def __init__(self, axes):
        super().__init__()
        self.axes = axes

    def forward(self, value):
        # Reduce in descending order so removing a dimension cannot change the
        # number of an axis that remains to be reduced. The Triton benchmark
        # uses this same reduction order.
        for axis in sorted(self.axes, reverse=True):
            value = torch.sum(value, dim=axis)
        return value


@pytest.mark.parametrize(
    "shape, axes, _",
    SUM_REDUCTION_CASES,
    ids=[case_id for _, _, case_id in SUM_REDUCTION_CASES],
)
def test_sum_reduce_torch(shape, axes, _, options):
    torch.manual_seed(0)
    model = SumReduce(axes)
    value = torch.rand(shape, device="cpu", dtype=torch.float32)

    manager = utils.ModelManager(model, value)
    manager.create_linalg_module()
    manager.write_bytecode_to_file()
    manager.execute_and_compare(options=options)
