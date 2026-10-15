// RUN: arm64-mlir-opt %s --vector-ext=sve -optimize-triton-shared | FileCheck %s

// GELU (tanh form) as emitted by triton-shared: one linalg.generic per op.
// Without elementwise fusion first, bubble-up-extract-slice hits the greedy
// rewrite limit on this chain and the pipeline fails without a diagnostic.
// With fusion: one vectorized loop, no temporary buffers. The checks follow
// the chain x -> 0.5x and x + c*x^3 -> exp(-2u) -> 1 / (1 + exp) -> 0.5x * (...) -> store
// inside the loop body.

// CHECK-LABEL: func.func @gelu_tanh
// CHECK-NOT:     memref.alloc
// CHECK:         scf.for
// CHECK:           %[[X:.+]] = vector.transfer_read {{.*}} vector<[4]xf32>
// CHECK:           %[[HALF_X:.+]] = arith.mulf %[[X]], %{{.+}} : vector<[4]xf32>
// CHECK:           arith.addf %[[X]], %{{.+}} : vector<[4]xf32>
// CHECK:           %[[NEG:.+]] = arith.negf %{{.+}} : vector<[4]xf32>
// CHECK-NEXT:      %[[EXP:.+]] = math.exp %[[NEG]] : vector<[4]xf32>
// CHECK-NEXT:      %[[DEN:.+]] = arith.addf %[[EXP]], %{{.+}} : vector<[4]xf32>
// CHECK-NEXT:      arith.divf %{{.+}}, %[[DEN]] : vector<[4]xf32>
// CHECK:           %[[RES:.+]] = arith.mulf %[[HALF_X]], %{{.+}} : vector<[4]xf32>
// CHECK-NEXT:      vector.transfer_write %[[RES]], {{.*}} vector<[4]xf32>
// CHECK-NEXT:    }
// CHECK-NOT:     scf.for
// CHECK-NOT:     memref.alloc
// CHECK:         return

#map = affine_map<(d0) -> (d0)>
module {
  func.func @gelu_tanh(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32) {
    %cst = arith.constant 4.471500e-02 : f32
    %c1024 = arith.constant 1024 : index
    %cst_0 = arith.constant 0.797884583 : f32
    %cst_1 = arith.constant 5.000000e-01 : f32
    %cst_2 = arith.constant 2.000000e+00 : f32
    %cst_3 = arith.constant 1.000000e+00 : f32
    %c1024_i32 = arith.constant 1024 : i32
    %0 = tensor.empty() : tensor<1024xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<1024xf32>) -> tensor<1024xf32>
    %2 = linalg.fill ins(%cst_0 : f32) outs(%0 : tensor<1024xf32>) -> tensor<1024xf32>
    %3 = linalg.fill ins(%cst_1 : f32) outs(%0 : tensor<1024xf32>) -> tensor<1024xf32>
    %4 = linalg.fill ins(%cst_2 : f32) outs(%0 : tensor<1024xf32>) -> tensor<1024xf32>
    %5 = linalg.fill ins(%cst_3 : f32) outs(%0 : tensor<1024xf32>) -> tensor<1024xf32>
    %6 = arith.muli %arg6, %c1024_i32 : i32
    %7 = arith.index_cast %6 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%7], sizes: [1024], strides: [1] : memref<*xf32> to memref<1024xf32, strided<[1], offset: ?>>
    %8 = arith.addi %7, %c1024 : index
    %9 = arith.index_cast %arg2 : i32 to index
    %10 = arith.minsi %8, %9 : index
    %11 = arith.maxsi %10, %7 : index
    %12 = arith.subi %11, %7 : index
    %alloc = memref.alloc() : memref<1024xf32>
    %subview = memref.subview %reinterpret_cast[0] [%12] [1] : memref<1024xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
    %subview_4 = memref.subview %alloc[0] [%12] [1] : memref<1024xf32> to memref<?xf32, strided<[1]>>
    memref.copy %subview, %subview_4 : memref<?xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1]>>
    %13 = bufferization.to_tensor %alloc restrict writable : memref<1024xf32> to tensor<1024xf32>
    %reinterpret_cast_5 = memref.reinterpret_cast %arg1 to offset: [%7], sizes: [1024], strides: [1] : memref<*xf32> to memref<1024xf32, strided<[1], offset: ?>>
    %14 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%13, %1 : tensor<1024xf32>, tensor<1024xf32>) outs(%13 : tensor<1024xf32>) {
    ^bb0(%in: f32, %in_7: f32, %out: f32):
      %29 = arith.mulf %in, %in_7 : f32
      linalg.yield %29 : f32
    } -> tensor<1024xf32>
    %15 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%14, %13 : tensor<1024xf32>, tensor<1024xf32>) outs(%14 : tensor<1024xf32>) {
    ^bb0(%in: f32, %in_7: f32, %out: f32):
      %29 = arith.mulf %in, %in_7 : f32
      linalg.yield %29 : f32
    } -> tensor<1024xf32>
    %16 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%15, %13 : tensor<1024xf32>, tensor<1024xf32>) outs(%15 : tensor<1024xf32>) {
    ^bb0(%in: f32, %in_7: f32, %out: f32):
      %29 = arith.mulf %in, %in_7 : f32
      linalg.yield %29 : f32
    } -> tensor<1024xf32>
    %17 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%13, %16 : tensor<1024xf32>, tensor<1024xf32>) outs(%13 : tensor<1024xf32>) {
    ^bb0(%in: f32, %in_7: f32, %out: f32):
      %29 = arith.addf %in, %in_7 : f32
      linalg.yield %29 : f32
    } -> tensor<1024xf32>
    %18 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%17, %2 : tensor<1024xf32>, tensor<1024xf32>) outs(%17 : tensor<1024xf32>) {
    ^bb0(%in: f32, %in_7: f32, %out: f32):
      %29 = arith.mulf %in, %in_7 : f32
      linalg.yield %29 : f32
    } -> tensor<1024xf32>
    %19 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%13, %3 : tensor<1024xf32>, tensor<1024xf32>) outs(%13 : tensor<1024xf32>) {
    ^bb0(%in: f32, %in_7: f32, %out: f32):
      %29 = arith.mulf %in, %in_7 : f32
      linalg.yield %29 : f32
    } -> tensor<1024xf32>
    %20 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%18, %4 : tensor<1024xf32>, tensor<1024xf32>) outs(%18 : tensor<1024xf32>) {
    ^bb0(%in: f32, %in_7: f32, %out: f32):
      %29 = arith.mulf %in, %in_7 : f32
      linalg.yield %29 : f32
    } -> tensor<1024xf32>
    %21 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%20 : tensor<1024xf32>) outs(%20 : tensor<1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %29 = arith.negf %in : f32
      linalg.yield %29 : f32
    } -> tensor<1024xf32>
    %22 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%21 : tensor<1024xf32>) outs(%21 : tensor<1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %29 = math.exp %in : f32
      linalg.yield %29 : f32
    } -> tensor<1024xf32>
    %23 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%22, %5 : tensor<1024xf32>, tensor<1024xf32>) outs(%22 : tensor<1024xf32>) {
    ^bb0(%in: f32, %in_7: f32, %out: f32):
      %29 = arith.addf %in, %in_7 : f32
      linalg.yield %29 : f32
    } -> tensor<1024xf32>
    %24 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%5, %23 : tensor<1024xf32>, tensor<1024xf32>) outs(%5 : tensor<1024xf32>) {
    ^bb0(%in: f32, %in_7: f32, %out: f32):
      %29 = arith.divf %in, %in_7 : f32
      linalg.yield %29 : f32
    } -> tensor<1024xf32>
    %25 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%24, %4 : tensor<1024xf32>, tensor<1024xf32>) outs(%24 : tensor<1024xf32>) {
    ^bb0(%in: f32, %in_7: f32, %out: f32):
      %29 = arith.mulf %in, %in_7 : f32
      linalg.yield %29 : f32
    } -> tensor<1024xf32>
    %26 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%25, %5 : tensor<1024xf32>, tensor<1024xf32>) outs(%25 : tensor<1024xf32>) {
    ^bb0(%in: f32, %in_7: f32, %out: f32):
      %29 = arith.subf %in, %in_7 : f32
      linalg.yield %29 : f32
    } -> tensor<1024xf32>
    %27 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%26, %5 : tensor<1024xf32>, tensor<1024xf32>) outs(%26 : tensor<1024xf32>) {
    ^bb0(%in: f32, %in_7: f32, %out: f32):
      %29 = arith.addf %in, %in_7 : f32
      linalg.yield %29 : f32
    } -> tensor<1024xf32>
    %28 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%19, %27 : tensor<1024xf32>, tensor<1024xf32>) outs(%19 : tensor<1024xf32>) {
    ^bb0(%in: f32, %in_7: f32, %out: f32):
      %29 = arith.mulf %in, %in_7 : f32
      linalg.yield %29 : f32
    } -> tensor<1024xf32>
    %extracted_slice = tensor.extract_slice %28[0] [%12] [1] : tensor<1024xf32> to tensor<?xf32>
    %subview_6 = memref.subview %reinterpret_cast_5[0] [%12] [1] : memref<1024xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
    bufferization.materialize_in_destination %extracted_slice in writable %subview_6 : (tensor<?xf32>, memref<?xf32, strided<[1], offset: ?>>) -> ()
    return
  }
}

