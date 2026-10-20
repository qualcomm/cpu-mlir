# ===- launcher.py ----------------------------------------------------------===
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause.
# For more license information:
#   https://github.com/qualcomm/cpu-mlir/LICENSE.txt
#
# ===------------------------------------------------------------------------===

from pathlib import Path
from math import prod
import os, shutil, subprocess, tempfile, struct, sys
import importlib.util
import torch
import numpy as np
from collections import namedtuple

from triton.backends.qcom_arm64_backend.utils import get_env_var, get_exec_mode, get_kernel_name, get_verbose_level, get_work_dir, parse_triton_llvm_kernel_signature, profile_inputs, to_torch_type, split_path

class WrapperStrings:
   def __init__(self):
        self.input_pref = """t"""
        self.output_pref = """o"""
        self.input_tensor_init_template = ("""
    Tensor<{tensor_ctype}, {tensor_rank}> t{i}({{{sizes}, {strides}, alignment}}, MemType::HEAP);\nMemRefDescriptor<{tensor_ctype}, {tensor_rank}> *dt{i} = t{i}.toMemRefDesc(); \n""")
        self.output_tensor_init_template = (
            """\nTensor<{tensor_ctype}, {tensor_rank}> o{i}(r->do{i}, MemType::HEAP, alignment);\n""")
        self.result_struct_init = """FuncResult *r = new FuncResult;\n"""
        self.result_struct_def = """
struct FuncResult
{{{result_fields}
}};
"""
        self.scalar_init_template = ("""{scalar_ctype} S{i} = {scalar_val}; \n""")
        self.extern_llvm_func_arg_tensor = (
            """MemRefDescriptor<{tensor_ctype}, {tensor_rank}> *, """
        )
        self.extern_llvm_func_args_scalar = ("""{scalar_ctype}, """)
        self.extern_llvm_func_defn = """extern "C" void {kernel_name}({function_arg_string}){streaming_attr};"""
        self.load_from_file_string = (
            """t{i}.load_from_file("{path}/{input_name}_t{i}.raw");\n"""
        )
        self.dump_to_file_string = (
            """{pref}{i}.dump_to_file("{path}/{input_name}_o{i}.raw");\n"""
        )
        self.alignment_string = """const int alignment = {alignment};"""
        self.perf_counter_enable_string = """enable_perf_counter_reads();\n"""
        self.write_prof_to_file_string = """write_prof_output("{path}/prof_report.json");\n"""

        self.extern_llvm_func_call = ("""
  int gridX = {grid[0]};
  int gridY = {grid[1]};
  int gridZ = {grid[2]};
  if (gridX * gridY * gridZ > 0) {{
    {benchmark_pre}
    {streaming_open}for(int x = 0; x < gridX; x++) {{
      for(int y = 0; y < gridY; y++) {{
        for(int z = 0; z < gridZ; z++) {{
          {func_name}({descriptor_string});
        }}
      }}
    }}{streaming_close}
    {benchmark_post}
  }}
  """
        )

        self.code_string = """
#include <assert.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
// From triton_test/utils/dsp/include
#include "common.h"
#include "prof.h"
#include "tensor.h"
// From triton_shared/backend/include
#include "ExecutionEngine/CRunnerUtils.h"
#include "ExecutionEngine/CRunnerUtils.cpp"

{llvm_func_sign}
{alignment_str}

{result_struct_def}

int main() {{
{enable_perf_counter_reads}
{tensor_definition_str}
{result_struct_init}
{read_from_file_calls}
{call_llvm_extern_func}
{write_prof_to_file}
{write_to_file_calls}
return 0;
}}
"""
        self.benchmarking_pre_string = """
    auto start = std::chrono::high_resolution_clock::now();
    for(int b=0;b<{iterations};++b) {{
"""
        self.benchmarking_post_string = """
    }}
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    FILE *bfile = fopen("{exec_dir}/benchmark.log", "w"); // add error check here!
    fprintf(bfile, "Elapsed time: %.3f ms", duration.count());
    fclose(bfile);
"""

class WrapperGenerator:
    def __init__(self, inputs, output_profs, kernel_llir, filename, launch_grid, exec_mode, benchmark_mode, profile_mode, exec_dir, lowered_by_JIT):
        """
        Initialize the WrapperGenerator instance.

        Parameters:
            inputs (list): A list of input tensors and scalars for the kernel.
            output_profs (list): A list of output profiles for the kernel
            kernel_llir (str): The LLIR representation of the kernel.
            filename (str): The name of the file for which the cpp wrapper will be genrated.
            launch_grid (tuple): A 3-tuple representing the launch grid configuration as (grid_x, grid_y, grid_z).
            exec_mode (str): Execution mode where "simulator" is for running on the simulator, and "device" is for running on the device.
            benchmark_mode (tuple): A 2-tuple of (bool, int) indicating if benchmarking is enabled and the number of iterations used for benchmarking
            profile_mode (bool): Indicates if profiling mode is enabled
            exec_dir (str): The directory designated for storing the kernel's input and output tensor files, as well as
            the shared object file necessary for execution.
            lowered_by_JIT (bool): Indicates if we are creating a wrapper for an LLVM IR generated by a JIT compiler from Triton kernel
            versus a user-defined LLIR or one created from user-defined MLIR.
        """
        self.inputs = inputs
        self.output_profs = output_profs
        self.name = filename
        self.exec_mode = exec_mode
        self.benchmark_mode = benchmark_mode
        self.profile_mode = profile_mode
        self.exec_dir = exec_dir
        self.common_strings = WrapperStrings()
        self.grid = launch_grid
        self.grid_strides = (
            launch_grid[1] * launch_grid[2],
            launch_grid[2],
            1
        )
        self.lowered_by_JIT = lowered_by_JIT
        self.is_streaming = "aarch64_pstate_sm_enabled" in kernel_llir
        self.input_profs, self.num_input_tensors = profile_inputs(self.inputs, self.lowered_by_JIT)
        self.return_type, self.func_name, self.is_func_ciface = parse_triton_llvm_kernel_signature(kernel_llir, self.num_input_tensors, len(self.inputs), self.lowered_by_JIT)



    def generate_input_declarations(self):
        """
        Generate declaration for each input
        for torch tensors, write them in Tensor class form,
        for scalars, declare and intialize a variable with scalar input's value.
        """
        formatted_tensor_string = ""
        for inp in self.input_profs:
            if inp.input_type == 'tensor':
                tensor_declaration = self.common_strings.input_tensor_init_template.format(
                    tensor_ctype=inp.dtype,
                    tensor_rank=inp.rank,
                    i= inp.idx,
                    sizes=inp.shape[0],
                    strides=inp.shape[1],
                ).lstrip()
                formatted_tensor_string += tensor_declaration
            else:
                scalar_declaration = self.common_strings.scalar_init_template.format(
                    i= inp.idx,
                    scalar_ctype=inp.dtype,
                    scalar_val=inp.value,
                ).lstrip()
                formatted_tensor_string += scalar_declaration
        return formatted_tensor_string

    def generate_llvm_function_signature(self):
        """Generate lowered LLVM function definition to be called from CPP launcher"""
        function_arg_string = ""
        if len(self.output_profs) > 0:
            function_arg_string += "void *, "
        for inp in self.input_profs:
            if inp.input_type == 'tensor':
                # non c interface functions require an additional integer
                # argument for each tensor
                if not self.is_func_ciface:
                    function_arg_string += "int64_t, "
                extern_args = self.common_strings.extern_llvm_func_arg_tensor.format(
                    tensor_ctype=inp.dtype, tensor_rank=inp.rank
                )
                function_arg_string += extern_args
            else:
                extern_args = self.common_strings.extern_llvm_func_args_scalar.format(
                    scalar_ctype=inp.dtype
                )
                function_arg_string += extern_args
        if self.is_func_ciface:
            function_arg_string = function_arg_string[:-2]
        else:
            function_arg_string += " int, int, int, int, int, int"
        return self.common_strings.extern_llvm_func_defn.format(
            kernel_name=self.func_name,
            function_arg_string=function_arg_string,
            return_type=self.return_type,
            streaming_attr=" __attribute__((arm_streaming))" if self.is_streaming else "",
        )

    def generate_llvm_function_call(self):
        """Generates actual function call to lowered LLVM function call"""
        function_call_descriptor_string = ""
        if len(self.output_profs) > 0:
            function_call_descriptor_string += (f"r, ")
        for inp in self.input_profs:
            if inp.input_type == 'tensor':
                tensor_idx = inp.idx
                if self.is_func_ciface:
                    function_call_descriptor_string += f"dt{tensor_idx}, "
                else:
                    tensor_rank = inp.rank
                    function_call_descriptor_string += f"{tensor_rank}, dt{tensor_idx}, "
            else:
                scalar_index = inp.idx
                function_call_descriptor_string += f"S{scalar_index}, "
        if self.is_func_ciface:
            function_call_descriptor_string = function_call_descriptor_string[:-2]
        else:
            function_call_descriptor_string += "gridX, gridY, gridZ, x, y, z"

        benchmark_pre = ""
        benchmark_post = ""
        if(self.benchmark_mode[0]):
            benchmark_pre = self.common_strings.benchmarking_pre_string.format(iterations=self.benchmark_mode[1])
            benchmark_post = self.common_strings.benchmarking_post_string.format(exec_dir=self.exec_dir)

        return self.common_strings.extern_llvm_func_call.format(
            func_name=self.func_name,
            grid=self.grid,
            descriptor_string=function_call_descriptor_string,
            benchmark_pre=benchmark_pre,
            benchmark_post=benchmark_post,
            streaming_open='__asm__ volatile("smstart sm" ::: "memory");\n    ' if self.is_streaming else "",
            streaming_close='\n    __asm__ volatile("smstop sm" ::: "memory");' if self.is_streaming else "",
        )


    def generate_tensor_read_from_file_calls(self):
        """Generates calls to read tensors from file"""
        read_from_file_string = ""
        for inp in self.input_profs:
            if inp.input_type == 'tensor':
                tensor_idx = inp.idx
                write_call = self.common_strings.load_from_file_string.format(
                    i=tensor_idx, path=self.exec_dir, input_name=self.name
                )
                read_from_file_string += write_call
        return read_from_file_string

    def generate_tensor_write_to_file_calls(self):
        """Generates calls to dump tensors to file"""
        write_to_file_string = ""
        # If there is no output, that means the kernel result is stored in one
        # (or more) of the inputs tensors, and thus all input tensors are dumped
        if len(self.output_profs) == 0:
            for inp in self.input_profs:
                if inp.input_type == 'tensor':
                    tensor_idx = inp.idx
                    write_call = self.common_strings.dump_to_file_string.format(
                        pref=self.common_strings.input_pref,
                        i=tensor_idx,
                        path=self.exec_dir,
                        input_name=self.name
                    )
                    write_to_file_string += write_call
        # Otherwise, there is at least one output tensor to dump
        else:
            for idx, res in enumerate(self.output_profs):
                write_to_file_string += self.common_strings.output_tensor_init_template.format(
                    tensor_ctype=res.dtype,
                    tensor_rank=res.rank,
                    i=idx
                )
                write_to_file_string += self.common_strings.dump_to_file_string.format(
                    pref=self.common_strings.output_pref,
                    i=idx,
                    path=self.exec_dir,
                    input_name=self.name
                )
        return write_to_file_string

    def generate_alignment_info(self):
        """Generates an initialization for alignment"""
        alignment = 128
        return self.common_strings.alignment_string.format(alignment=alignment)

    def generate_perf_counter_enable(self):
        """Generates call that enables user to read arm64 PMCCNTR_EL0 register"""
        return "" if not self.profile_mode else self.common_strings.perf_counter_enable_string

    def generate_write_prof_to_file(self):
        """Generates call to write profiling buffer to file"""
        return "" if not self.profile_mode else self.common_strings.write_prof_to_file_string.format(path=self.exec_dir)

    def generate_result_struct(self):
        """Generates the FuncResult struct"""
        if len(self.output_profs) > 0:
            result_fields_string = ""
            for idx, res in enumerate(self.output_profs):
                result_fields_string += (
                    f"\n\tMemRefDescriptor<{res.dtype},{res.rank}> do{idx};"
                )
            return self.common_strings.result_struct_def.format(result_fields=result_fields_string)
        else:
            return ""

    def generate_result_struct_init(self):
        """Generates initialization of FuncResult"""
        if len(self.output_profs) > 0:
            return self.common_strings.result_struct_init
        else:
            return ""

    def generate_standalone_launcher(self):
        """
        Generates skeleton cpp file which launches triton kernel.
        """
        #TODO: multi-threading support
        return self.common_strings.code_string.format(
            llvm_func_sign=self.generate_llvm_function_signature(),
            alignment_str=self.generate_alignment_info(),
            result_struct_def=self.generate_result_struct(),
            enable_perf_counter_reads=self.generate_perf_counter_enable(),
            tensor_definition_str=self.generate_input_declarations(),
            result_struct_init=self.generate_result_struct_init(),
            read_from_file_calls=self.generate_tensor_read_from_file_calls(),
            call_llvm_extern_func=self.generate_llvm_function_call(),
            write_prof_to_file=self.generate_write_prof_to_file(),
            write_to_file_calls=self.generate_tensor_write_to_file_calls(),
        )

    def dump_wrapper(self, directory):
        """Dumps CPP file"""
        cpp_wrapper_path = os.path.join(
                directory, self.name + "_wrapper.cpp"
            )
        with open(cpp_wrapper_path, "w") as file:
            wrapper = self.generate_standalone_(directory)
            file.write(wrapper)
        return cpp_wrapper_path

    def dump_tensor_inputs(self, directory):
        """Dump input tensors to files"""
        input_paths = []
        for inp in self.input_profs:
            if inp.input_type == 'tensor':
                i = inp.idx
                data = inp.value
                input_path = os.path.join(directory, f"{self.name}_t{i}.raw")
                with open(input_path, "wb") as file:
                    file.write(data.numpy().tobytes())
                input_paths.append(input_path)
        return input_paths

