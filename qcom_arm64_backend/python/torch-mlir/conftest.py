# ===- conftest.py ----------------------------------------------------------===
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  SPDX-License-Identifier: BSD-3-Clause.
#  For more license information:
#    https://github.com/qualcomm/cpu-mlir/LICENSE.txt
#
# ===------------------------------------------------------------------------===

import pytest

import utils


def pytest_addoption(parser):
    parser.addoption(
        "--target",
        action="store",
        choices=utils.VECTOR_EXT_OPTIONS.keys(),
        default="sme",
        help="ARM64 vector extension to generate code for",
    )


@pytest.fixture
def options(request):
    return utils.VECTOR_EXT_OPTIONS[request.config.getoption("--target")]
