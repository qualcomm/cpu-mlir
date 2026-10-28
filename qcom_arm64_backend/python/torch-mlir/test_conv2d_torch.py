# ===- test_conv2d_torch.py -------------------------------------------------===
# 
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  SPDX-License-Identifier: BSD-3-Clause.
#  For more license information:
#    https://github.com/qualcomm/cpu-mlir/LICENSE.txt
# 
# ===------------------------------------------------------------------------===

import torch
import utils
import pytest
import subprocess
import os


# Define the computation graph
class Conv2d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(Conv2d, self).__init__()
        # torchvision's ResNet18 convolutions do not use a bias term.
        self.conv = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )

    def forward(self, x):
        x = self.conv(x)
        return x


def test_conv2d_torch(options={}):
    # Unique Conv2d shapes used by torchvision.models.resnet18 for a
    # 1x3x224x224 input. Each entry is compiled and executed separately so
    # benchmarking reports one timing per convolution variant.
    conv_variants = [
        ("stem_7x7_s2", 3, 64, 7, 2, 3, 224),
        ("residual_64_3x3_s1", 64, 64, 3, 1, 1, 56),
        ("transition_64_to_128_3x3_s2", 64, 128, 3, 2, 1, 56),
        ("residual_128_3x3_s1", 128, 128, 3, 1, 1, 28),
        ("transition_128_to_256_3x3_s2", 128, 256, 3, 2, 1, 28),
        ("residual_256_3x3_s1", 256, 256, 3, 1, 1, 14),
        ("transition_256_to_512_3x3_s2", 256, 512, 3, 2, 1, 14),
        ("residual_512_3x3_s1", 512, 512, 3, 1, 1, 7),
        ("downsample_64_to_128_1x1_s2", 64, 128, 1, 2, 0, 56),
        ("downsample_128_to_256_1x1_s2", 128, 256, 1, 2, 0, 28),
        ("downsample_256_to_512_1x1_s2", 256, 512, 1, 2, 0, 14),
    ]

    for name, in_channels, out_channels, kernel_size, stride, padding, input_size in conv_variants:
        print(f"\n==> Testing {name}")
        model = Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        x = torch.randn(1, in_channels, input_size, input_size)

        manager = utils.ModelManager(model, x)
        manager.create_linalg_module()
        manager.write_bytecode_to_file()
        # Allow for floating-point reduction-order differences while keeping a
        # tighter operator-level accuracy threshold than the end-to-end model.
        manager.execute_and_compare(rtol=1e-05, atol=1e-03, options=options)



if __name__ == "__main__":
    test_conv2d_torch(utils.get_vector_ext_options_from_args())
