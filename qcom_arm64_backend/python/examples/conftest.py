# ===- conftest.py ----------------------------------------------------------===
# 
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  SPDX-License-Identifier: BSD-3-Clause.
#  For more license information:
#    https://github.com/qualcomm/cpu-mlir/LICENSE.txt
# 
# ===------------------------------------------------------------------------===

import os
import pytest
import sys
import triton
from triton.backends.triton_shared.driver import CPUDriver
from triton.backends.qcom_arm64_backend.driver import ARM64CPUDriver
from triton.backends.qcom_arm64_backend.utils import get_verbose_level
import triton.backends.triton_shared.driver as _ts_driver

# Track the active ARM64 target so CPUDriver redirects use the right extension.
_arm64_target = "sme"

class _CPUDriverRedirect(ARM64CPUDriver):
    """Redirect CPUDriver() -> ARM64CPUDriver so triton_shared tests run via the ARM64 backend."""
    def __init__(self):
        super().__init__(_arm64_target)

# Patch before inner conftest and test files are imported so their
# `from triton.backends.triton_shared.driver import CPUDriver` picks up the redirect.
_ts_driver.CPUDriver = _CPUDriverRedirect

# TODO: for some reason we need to active a driver before pytest_collection_finish(),
#       otherwise 'vector_ext' is missing when creating ARM64CPUDriver.
triton.runtime.driver.set_active(CPUDriver())

def empty_decorator(func):
    return func

pytest.mark.interpreter = empty_decorator

@pytest.fixture
def device(request):
    return "cpu"

def pytest_addoption(parser):
    parser.addoption(
        "--target",
        action="store",
        default="sme",
        help="ARM64 vector extension to generate code for, valid values are \"none\", \"sve\", \"sme\""
    )

arm64_vect=["neon", "sve", "sme"]
def set_target(config: pytest.Config):
    global _arm64_target
    target = config.getoption("--target")
    verbose = get_verbose_level()
    if target not in arm64_vect:
        if verbose > 0: print(f"\nPyTest: Invalid vector extension: {target}, default to sme" )
        _arm64_target = "sme"
    else:
        if verbose > 0: print(f"\nPyTest: Activating backend with vector extension: {target}" )
        _arm64_target = target
    triton.runtime.driver.set_active(ARM64CPUDriver(_arm64_target))


supported = {
    # from Triton-Shared
    "test_addptr.py",
    "test_blockptr_complex_offset.py",
    "test_early_return.py",
    "test_einsum_qhmd_hmpd_to_qhmp.py",
    "test_gather_scatter.py",
    "test_gather_scatter_continuous.py",
    "test_index_select.py",
    "test_load_2d_tensor_block.py",
    "test_load_2d_tensor_col.py",
    "test_mask.py",
    "test_mask_loop_iter_arg.py",
    "test_matmul.py",
    "test_modulo.py",
    "test_nested_loops.py",
    "test_scalar_store.py",
    "test_sign_extend.py",
    "test_softmax.py",
    "test_splat.py",
    "test_swap.py",
    "test_tensor_index_iterargs.py",
    "test_unstructured_mask.py",
    "test_vec_add.py",
    "test_mm.py",
    "test_reduce.py",
    "test_layernorm.py",

    # QC-internal tests
    "test_activations.py",
    "test_flash_attention.py"
}

def pytest_collection_modifyitems(config, items):
    skip_marker = pytest.mark.skip(reason="ARM64 backend does not support this test yet")
    for item in items:
        test_path = str(item.fspath)
        test_file = os.path.basename(test_path)
        if (test_file) not in supported:
            item.add_marker(skip_marker)
            continue

def pytest_collection_finish(session: pytest.Session):
    set_target(session.config)
    pass
