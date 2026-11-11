# ===- test_transpose_torch.py ----------------------------------------------===
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  SPDX-License-Identifier: BSD-3-Clause.
#  For more license information:
#    https://github.com/qualcomm/cpu-mlir/LICENSE.txt
#
# ===------------------------------------------------------------------------===
#
# Usage: python test_transpose_torch.py [neon|sve|sme]

import os
import torch
import utils

# On qemu, emulate a CPU without FEAT_SME_FA64, like the phones we run on: with
# it, NEON code that wrongly ends up in streaming mode would run fine on qemu
# but fault on the device.
os.environ.setdefault("QEMU_CPU", "max,sme_fa64=off")


class Transpose(torch.nn.Module):
    def __init__(self, dim0, dim1, then_add=False):
        super(Transpose, self).__init__()
        self.dim0 = dim0
        self.dim1 = dim1
        self.then_add = then_add

    def forward(self, x):
        y = x.transpose(self.dim0, self.dim1).contiguous()
        # Other work in the same function, as in a whole model: only the ZA
        # transpose may run in streaming mode.
        return y + y if self.then_add else y


# On SME, transposes of the two innermost dimensions with at least 2^19
# elements go through ZA tiles; smaller ones and the heads transpose (dims 1
# and 2, which keeps the innermost dimension) take the regular path.
VARIANTS = [
    # Small, regular path.
    ("small_2d_f32", (129, 77), torch.float32, 0, 1),
    ("small_2d_f16", (50, 768), torch.float16, 0, 1),
    ("small_2d_i8", (33, 70), torch.int8, 0, 1),
    ("small_attention_k_f32", (1, 12, 7, 64), torch.float32, 2, 3),
    # Just below the threshold, regular path.
    ("mid_2d_f32_512", (512, 512), torch.float32, 0, 1),
    ("mid_2d_f16_512", (512, 512), torch.float16, 0, 1),
    ("mid_2d_i8_512", (512, 512), torch.int8, 0, 1),
    ("edge_2d_f32_below", (1024, 511), torch.float32, 0, 1),
    # At and above the threshold, ZA tiles. The odd shapes are not multiples
    # of the tile size, so the masked tile edges are covered.
    ("edge_2d_f32_at", (1024, 512), torch.float32, 0, 1),
    ("za_2d_f32_odd", (1031, 1029), torch.float32, 0, 1),
    ("za_2d_f16_odd", (1025, 770), torch.float16, 0, 1),
    ("za_2d_i8_odd", (1023, 1025), torch.int8, 0, 1),
    ("za_2d_f32_1024", (1024, 1024), torch.float32, 0, 1),
    ("za_2d_f32_2048", (2048, 2048), torch.float32, 0, 1),
    ("za_2d_f16_1024", (1024, 1024), torch.float16, 0, 1),
    ("za_2d_i8_1024", (1024, 1024), torch.int8, 0, 1),
    ("za_attention_k_f32", (1, 12, 1024, 64), torch.float32, 2, 3),
    ("za_gpt2_lm_head_f16", (50257, 768), torch.float16, 0, 1),
    # Heads transpose, regular path at any size.
    ("attention_heads_f16", (1, 1024, 12, 64), torch.float16, 1, 2),
    # A ZA transpose followed by SVE code in the same function.
    ("za_2d_f32_then_add", (1031, 1029), torch.float32, 0, 1, True),
    ("za_attention_k_f16_then_add", (1, 12, 1024, 64), torch.float16, 2, 3, True),
]


def test_transpose_torch(options={}):
    all_match = True
    for name, shape, dtype, dim0, dim1, *then_add in VARIANTS:
        numel = 1
        for d in shape:
            numel *= d
        print(f"\n==> Testing {name} shape={shape} dtype={dtype} elements={numel}")
        model = Transpose(dim0, dim1, *then_add)
        if dtype.is_floating_point:
            x = torch.randn(shape).to(dtype)
        else:
            x = torch.randint(-128, 128, shape, dtype=dtype)

        manager = utils.ModelManager(model, x)
        manager.create_linalg_module()
        manager.write_bytecode_to_file()
        # A transpose only moves values (and y + y is exact), so the result
        # must match exactly.
        all_match &= manager.execute_and_compare(rtol=0, atol=0, options=options)
    assert all_match


if __name__ == "__main__":
    test_transpose_torch(utils.get_vector_ext_options_from_args())
