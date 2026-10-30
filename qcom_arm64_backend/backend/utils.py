# ===- utils.py -------------------------------------------------------------===
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause.
# For more license information:
#   https://github.com/qualcomm/cpu-mlir/LICENSE.txt
#
# ===------------------------------------------------------------------------===

import os, re, tempfile, subprocess, sys
import torch
from collections import namedtuple

def _gxx_supports_float16_cxx():
    if sys.platform == "darwin":
        return True
    try:
        result = subprocess.run(["g++", "--version"], capture_output=True, text=True)
        m = re.search(r'(\d+)\.\d+\.\d+', result.stdout)
        return m is not None and int(m.group(1)) >= 12
    except Exception:
        return False

# Only GCC 12+ has support for _Float16 on ARM64, older versions use __fp16
_FLOAT16_CTYPE = "_Float16" if _gxx_supports_float16_cxx() else "__fp16"

_work_dir = ""
def get_work_dir() -> str:
    global _work_dir
    if not _work_dir:
        _work_dir = os.getcwd() if os.getenv("TRITON_ARM64_SAVE_TEMPS", "0") == "1" else tempfile.mkdtemp()
    return _work_dir

def get_kernel_name(ir:str) -> str:
    # find kernel name in mlir
    mlir_pattern = r".+\.func .*@(\w+)\(.+"
    mlir_matches = re.findall(mlir_pattern, ir)
    if len(mlir_matches) == 1:
        return mlir_matches[0]
    else:
        assert len(mlir_matches) == 0

    # find kernel name in llvm-ir
    llvm_pattern = r"define void @(\w+)\(.+"
    llvm_matches = re.findall(llvm_pattern, ir)
    assert len(llvm_matches) == 1
    return llvm_matches[0]

def get_verbose_level() -> int:
    return int(os.getenv("TRITON_ARM64_VERBOSE", 0))

def get_debug_mode() -> int:
    return int(os.getenv("TRITON_ARM64_DEBUG", 0))

def get_codegen_options(options):
    return {"arm64_vector_ext"  : str(options.vector_ext),
            "prof_disable_loop" : str(options.prof_disable_loop),
            "prof_loop_depth"   : str(options.prof_loop_depth)}

class MissingEnvironmentVariable(Exception):
    pass

def get_env_var(name, default=None):
    if default is not None:
        return os.environ.get(name, default)
    try:
        return os.environ[name]
    except KeyError:
        raise MissingEnvironmentVariable(f"{name} variable needed in the environment!")

def split_path(path):
    filename = os.path.basename(path)
    directory = os.path.dirname(path)
    filename_without_ext, extension = os.path.splitext(filename)
    return (directory, filename_without_ext, extension)

def to_torch_type(type_id):
    cpp_to_torch_map = {
        1: torch.float16,
        2: torch.float32,
        3: torch.float64,
        4: torch.int8,
        5: torch.int16,
        6: torch.int32,
        7: torch.int64,
        8: torch.uint8,
    }
    # Old torch versions do not support torch.uint16, torch.uint32, torch.uint64
    try:
        cpp_to_torch_map[9] = torch.uint16
        cpp_to_torch_map[10] = torch.uint32
    except:
        pass
    return cpp_to_torch_map[type_id]

def replace_list_brackets(dimension_list):
    """Replaces a list surrounded by square brackets with curly braces"""
    dimension_list = str(dimension_list)
    if "[" in str(dimension_list) and "]" in dimension_list:
        return dimension_list.translate(str.maketrans({"[": "{", "]": "}"}))

def get_shape(shape):
    """Converts torch sizes to conventional list of sizes/strides"""
    sizes = [
        dimension for dimension in shape
    ]  # shape here will be of form torch.Size[...]
    # Now, we want the strides. NOTE: We can skip the first dimension and keep inserting and multiplying with val at first index
    strides = [] if len(sizes) == 0 else [1]
    for dim_shape in reversed(sizes[1:]):
        strides.insert(0, strides[0] * dim_shape)
    sizes = replace_list_brackets(sizes)
    strides = replace_list_brackets(strides)
    return sizes, strides

def get_ctype(dtype):
    """Maps a torch data type to llvm/clang compatible Dtypes"""
    mapping = {
        torch.double: "double",
        torch.float: "float",
        torch.float64: "double",
        torch.float32: "float",
        torch.float16: _FLOAT16_CTYPE,
        torch.long: "int32_t",
        torch.int: "int32_t",
        torch.int64: "int64_t",
        torch.int32: "int32_t",
        torch.int16: "int16_t",
        torch.int8: "int8_t",
        torch.uint8: "uint8_t",
        torch.bool: "bool",
    }
    # Old torch versions do not support torch.uint16, torch.uint32 and
    # torch.uint64. Add entry into the map only if there are no errors
    try:
        mapping[torch.uint64] = "uint64_t"
        mapping[torch.uint32] = "uint32_t"
        mapping[torch.uint16] = "uint16_t"
    except:
        pass
    return mapping.get(
        dtype, "unknown_type"
    )  # returns value of key-value pair if found in mapping, else returns "unknown_type"

def ctype_to_torch_type(ctype):
    cpp_to_torch_map = {
        _FLOAT16_CTYPE: torch.float16,
        "float": torch.float32,
        "double": torch.float64,
        "int8_t": torch.int8,
        "int16_t": torch.int16,
        "int32_t": torch.int32,
        "int64_t": torch.int64,
        "uint8_t": torch.uint8,
    }
    # Old torch versions do not support torch.uint16, torch.uint32
    try:
        cpp_to_torch_map["uint16_t"] = torch.uint16
        cpp_to_torch_map["uint32_t"] = torch.uint32
    except:
        pass
    return cpp_to_torch_map[ctype]

def profile_torch_mlir_inputs(inputs):
    return profile_inputs(inputs, True)

def profile_inputs(inputs, lowered_by_JIT):
    """ Generate a profile for each input """
    Profiled_input = namedtuple('InputProfile', (
        'idx',        # Index for tensor or scalar input
        'input_id',   # Corresponding index in the input list
        'input_type', # Type of the input (tensor or scalar)
        'value',      # Actual value of the input
        'dtype',      # CPP data type of the input value
        'rank',       # Rank of the input (None for scalars)
        'shape'       # Shape of the input (None for scalars)
    ))
    profiled = []
    tensor_index = 0
    scalar_index = 0
    for index, inp in enumerate(inputs):
        if isinstance(inp, torch.Tensor):
            profiled.append(
                Profiled_input(input_type = 'tensor',
                               idx = tensor_index,
                               input_id = index,
                               value = inp.data,
                               dtype = get_ctype(inp.dtype),
                               rank  = inp.dim(),
                               shape = get_shape(inp.shape)
            ))
            tensor_index += 1
        elif isinstance(inp, int):
            # If we pass Triton kernels to our Triton driver, the kernel is lowered by Triton's JIT compiler.
            # In this case, we don't need to keep integer inputs with a value of 1, as they are removed
            # from the kernel arguments by the JIT and used as constants instead.
            if lowered_by_JIT and inp != 1:
                profiled.append(
                    Profiled_input(input_type = 'scalar',
                                idx = scalar_index,
                                input_id = index,
                                value = inp,
                                dtype = type(inp).__name__,
                                rank = None,
                                shape = None
                ))
                scalar_index += 1
        elif isinstance(inp, float):
            profiled.append(
                Profiled_input(input_type = 'scalar',
                               idx = scalar_index,
                               input_id = index,
                               value = inp,
                               dtype = type(inp).__name__,
                               rank = None,
                               shape = None
            ))
            scalar_index += 1
        else:
            raise ValueError(f"Unsupported input type {type(inp.__name__)}")
    return profiled, tensor_index

def parse_triton_llvm_kernel_signature(llir, num_input_tensors, num_inputs, lowered_by_JIT):
    pattern = r"define (.+) @(\w+)\((.+)\)"
    matches = re.findall(pattern, llir)

    # All IR lowered by torch-mlir will have an additional c interfance function
    # in addition to the kernel itself. In such a case, we want to call the c
    # interface function instead of the kernel
    if len(matches) > 1:
        assert len(matches) == 2
        for match in matches:
            if "ciface" in match[1]:
                return_type = match[0]
                kernel_name = match[1]
                is_func_ciface = True
                # Sanity check to make sure the ciface function has matching number of arguments
                # The ciface function takes (result_out_ptr, input_ptr_0, ..., input_ptr_N).
                # Count how many ptr args it has; subtract 1 for the result out-parameter
                # to get the expected number of input tensors.
                ciface_args = match[2].split(',')
                ciface_ptr_count = sum(1 for arg in ciface_args if "ptr" in arg)
                expected_inputs = ciface_ptr_count - 1  # subtract result out-param
                if expected_inputs != num_input_tensors:
                    raise ValueError(
                        f"Input count mismatch for '{kernel_name}': "
                        f"the compiled function expects {expected_inputs} input tensor(s) "
                        f"but only {num_input_tensors} were provided. "
                    )
                return (return_type, kernel_name, is_func_ciface)

    # Otherwise, there should only be one function: the kernel
    assert len(matches) == 1
    # Parse return type
    return_type = matches[0][0]
    if return_type != "void":
       return_type_rank_matches = re.findall(r"{ ptr, ptr, i64, \[(\w+) x i64\], \[(\w+) x i64\] }", return_type)
       assert len(return_type_rank_matches) == 1 and (return_type_rank_matches[0][0] == return_type_rank_matches[0][1])
       return_type = f"MemRefDescriptor<void, {return_type_rank_matches[0][0]}>"
    # Get kernel name
    kernel_name = matches[0][1]
    is_func_ciface = False

    # Verify that kernel tensor arguments are unranked.
    #   Note: Qualcomm's triton backend should only generate unranked tensors,
    #         which use one pointer argument per tensor (ranked tensors each use
    #         2 pointer arguments and are not supported)
    kernel_args = matches[0][2].split(',')
    ptr_cnt = sum(1 for arg in kernel_args if "ptr" in arg)
    assert ptr_cnt == num_input_tensors, "parse_triton_llvm_kernel_signature: Wrong number or combination of inputs! Ensure correct number of inputs are passed and input tensors are unranked."
    return (return_type, kernel_name, is_func_ciface)

def get_exec_mode():
    """
    Determines the expected execution target (device or simulator)
    which is required for generating the cpp wrapper and
    configuring the HexagonExecutor.
    Defaults to runnig the kernel on the device.
    """
    RUN_ON_SIM = get_env_var("RUN_ON_SIM", default = '0')
    if RUN_ON_SIM == '0':
        return "device"
    elif RUN_ON_SIM == '1':
        return "simulator"
    else:
        raise ValueError (
            "Invalid value for RUN_ON_SIM. Set RUN_ON_SIM to 1 "
            "to run the kernel on the simulator and 0 to run it on the device.")

def get_benchmark_mode(exec_mode):
    """
    Determines use of benchmarking mode
    """
    if exec_mode != "device":
        return (False, 1)

    TRITON_ARM64_BENCHMARK = get_env_var("TRITON_ARM64_BENCHMARK", default = '0')
    iterations = int(get_env_var("TRITON_ARM64_BENCHMARK_ITERATIONS", default = '1'))

    if TRITON_ARM64_BENCHMARK != '0' and TRITON_ARM64_BENCHMARK != '1':
        raise ValueError ("Invalid value for TRITON_ARM64_BENCHMARK.")

    return (TRITON_ARM64_BENCHMARK == '1', iterations)

def get_profile_mode(exec_mode):
    """
    Determines if profiling mode is on or off

    exec_mode : string
        current expected execution target (device or simulator)
    """
    if exec_mode != "device":
        return False
    return get_env_var("TRITON_ARM64_PROFILE", default = '0') == '1'

def get_mlir_to_ctype(dtype):
    """Maps a MLIR data type to llvm/clang compatible data types"""
    mapping = {
        "f64": "double",
        "f32": "float",
        "f16": _FLOAT16_CTYPE,
        "i32": "int32_t",
        "i64": "int64_t",
        "i16": "int16_t",
        "i8": "int8_t",
        "b8": "bool",
    }

    return mapping.get(
        dtype, "unknown_type"
    )  # returns value of key-value pair if found in mapping, else returns "unknown_type"

def make_profiled_return(iter_in):
    """Generate a profile for each return type"""
    profiled_return = namedtuple(
        "ReturnProfile",
        (
            "idx",   # Index of return type
            "dtype", # CPP data type of the return type
            "rank",  # Rank of the return type (None for scalars)
            "shape", # (sizes_str, strides_str) for C++ codegen (None for scalars)
            "sizes", # Raw list of ints for torch.empty (None for scalars)
        ),
    )
    return profiled_return(*iter_in)


def parse_return_types(return_type_list):
    return_types = []
    for idx, r in enumerate(return_type_list):
        rank = 0 if r[0] == 0 else r[0]
        dtype = r[1]
        raw_sizes = list(r[2]) if r[0] > 0 else None
        shape = get_shape(r[2]) if r[0] > 0 else None
        return_types.append(make_profiled_return([idx, get_mlir_to_ctype(dtype), rank, shape, raw_sizes]))

    return return_types
