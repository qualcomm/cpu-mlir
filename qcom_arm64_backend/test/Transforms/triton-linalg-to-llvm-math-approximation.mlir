// RUN: arm64-mlir-opt %s --vector-ext=sve -triton-linalg-to-llvm | FileCheck %s

// GELU (erf form) after -optimize-triton-shared. math.erf has no LLVM lowering,
// so it must be approximated inside the pipeline; the result stays an SVE
// vector computation with no libm calls.

// CHECK-LABEL: llvm.func @gelu_erf
// CHECK-NOT:     math.
// CHECK-NOT:     llvm.call
// CHECK:         llvm.intr.fma{{.*}} : (vector<[4]xf32>, vector<[4]xf32>, vector<[4]xf32>) -> vector<[4]xf32>
// CHECK-NOT:     math.
// CHECK-NOT:     llvm.call

#map = affine_map<(d0)[s0, s1] -> (-d0 + s0, s1)>
module {
  func.func @gelu_erf(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32) {
    %cst = arith.constant dense<1.000000e+00> : vector<[4]xf32>
    %cst_0 = arith.constant dense<0.707106769> : vector<[4]xf32>
    %cst_1 = arith.constant dense<5.000000e-01> : vector<[4]xf32>
    %0 = ub.poison : f32
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1024 = arith.constant 1024 : index
    %c1024_i32 = arith.constant 1024 : i32
    %1 = arith.muli %arg6, %c1024_i32 : i32
    %2 = arith.index_cast %1 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%2], sizes: [1024], strides: [1] : memref<*xf32> to memref<1024xf32, strided<[1], offset: ?>>
    %3 = arith.addi %2, %c1024 : index
    %4 = arith.index_cast %arg2 : i32 to index
    %5 = arith.minsi %3, %4 : index
    %6 = arith.maxsi %5, %2 : index
    %7 = arith.subi %6, %2 : index
    %reinterpret_cast_2 = memref.reinterpret_cast %arg1 to offset: [%2], sizes: [1024], strides: [1] : memref<*xf32> to memref<1024xf32, strided<[1], offset: ?>>
    %subview = memref.subview %reinterpret_cast_2[0] [%7] [1] : memref<1024xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
    %subview_3 = memref.subview %reinterpret_cast[0] [%7] [1] : memref<1024xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
    %vscale = vector.vscale
    %c4_vscale = arith.muli %vscale, %c4 : index
    scf.for %arg9 = %c0 to %7 step %c4_vscale {
      %8 = affine.min #map(%arg9)[%7, %c4_vscale]
      %subview_4 = memref.subview %subview_3[%arg9] [%8] [1] : memref<?xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
      %subview_5 = memref.subview %subview[%arg9] [%8] [1] : memref<?xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
      %9 = vector.create_mask %8 : vector<[4]xi1>
      %10 = vector.transfer_read %subview_4[%c0], %0, %9 {in_bounds = [true]} : memref<?xf32, strided<[1], offset: ?>>, vector<[4]xf32>
      %11 = arith.mulf %10, %cst_1 : vector<[4]xf32>
      %12 = arith.mulf %10, %cst_0 : vector<[4]xf32>
      %13 = math.erf %12 : vector<[4]xf32>
      %14 = arith.addf %13, %cst : vector<[4]xf32>
      %15 = arith.mulf %11, %14 : vector<[4]xf32>
      vector.transfer_write %15, %subview_5[%c0], %9 {in_bounds = [true]} : vector<[4]xf32>, memref<?xf32, strided<[1], offset: ?>>
      memref.copy %subview_5, %subview_5 : memref<?xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
    }
    return
  }
}

