// RUN: arm64-mlir-opt %s -triton-arm64-math-approximation | FileCheck %s
// RUN: arm64-mlir-opt %s -triton-arm64-math-approximation="ops=math.log" | FileCheck %s --check-prefix=OPS

// Default set (erf, exp, tanh): expanded to arith ops, others untouched.

// CHECK-LABEL: func.func @erf_scalable
// CHECK-NOT:     math.erf
// CHECK:         arith.select
// CHECK:         return {{.*}} : vector<[4]xf32>
// OPS-LABEL:   func.func @erf_scalable
// OPS:           math.erf
func.func @erf_scalable(%x: vector<[4]xf32>) -> vector<[4]xf32> {
  %0 = math.erf %x : vector<[4]xf32>
  return %0 : vector<[4]xf32>
}

// CHECK-LABEL: func.func @exp_scalable
// CHECK-NOT:     math.exp
// CHECK:         arith.bitcast {{.*}} : vector<[4]xi32> to vector<[4]xf32>
// CHECK:         return {{.*}} : vector<[4]xf32>
func.func @exp_scalable(%x: vector<[4]xf32>) -> vector<[4]xf32> {
  %0 = math.exp %x : vector<[4]xf32>
  return %0 : vector<[4]xf32>
}

// CHECK-LABEL: func.func @tanh_f32
// CHECK-NOT:     math.tanh
// CHECK:         return {{.*}} : f32
func.func @tanh_f32(%x: f32) -> f32 {
  %0 = math.tanh %x : f32
  return %0 : f32
}

// f16 is computed in f32.
// CHECK-LABEL: func.func @exp_f16
// CHECK:         arith.extf {{.*}} : vector<8xf16> to vector<8xf32>
// CHECK-NOT:     math.exp
// CHECK:         arith.truncf {{.*}} : vector<8xf32> to vector<8xf16>
func.func @exp_f16(%x: vector<8xf16>) -> vector<8xf16> {
  %0 = math.exp %x : vector<8xf16>
  return %0 : vector<8xf16>
}

// Not in the default set: keeps its libm lowering.
// CHECK-LABEL: func.func @log_f32
// CHECK:         math.log
// OPS-LABEL:   func.func @log_f32
// OPS-NOT:       math.log
func.func @log_f32(%x: f32) -> f32 {
  %0 = math.log %x : f32
  return %0 : f32
}

// fpowi is expanded whatever ops= says: llvm.intr.powi needs a scalar exponent,
// so a vector fpowi cannot be lowered to LLVM. x^3 (torch-mlir's GELU tanh)
// becomes multiplies.
// CHECK-LABEL: func.func @fpowi_const_scalable
// CHECK-NOT:     math.fpowi
// CHECK:         arith.mulf
// CHECK:         return {{.*}} : vector<[4]xf32>
// OPS-LABEL:   func.func @fpowi_const_scalable
// OPS-NOT:       math.fpowi
// OPS:           return
func.func @fpowi_const_scalable(%x: vector<[4]xf32>) -> vector<[4]xf32> {
  %c3 = arith.constant dense<3> : vector<[4]xi64>
  %0 = math.fpowi %x, %c3 : vector<[4]xf32>, vector<[4]xi64>
  return %0 : vector<[4]xf32>
}

// A non-constant power becomes powf, which has a vector lowering.
// CHECK-LABEL: func.func @fpowi_var
// CHECK-NOT:     math.fpowi
// CHECK:         arith.sitofp
// CHECK:         math.powf
func.func @fpowi_var(%x: vector<[4]xf32>, %n: vector<[4]xi64>) -> vector<[4]xf32> {
  %0 = math.fpowi %x, %n : vector<[4]xf32>, vector<[4]xi64>
  return %0 : vector<[4]xf32>
}
