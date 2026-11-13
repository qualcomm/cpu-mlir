// RUN: arm64-mlir-opt %s -split-input-file --vector-ext=sme -optimize-triton-shared | FileCheck %s

// Softmax over rows of 781 columns, as torch-mlir emits it (whole matrix, max
// and sum as linalg.generic reductions) and as Triton emits it (one row per
// program, masked load). Both must vectorize the same way:
//  - the max and sum reductions use masked scalable vectors, two registers'
//    worth per iteration ([8]), with the accumulator carried in a register
//    (a loop block argument), not loaded and stored per iteration;
//  - no element-at-a-time loads (memref.load) and no runtime memrefCopy call
//    (memref.copy of the masked load).

// CHECK-LABEL: func.func @Softmax(
// CHECK-NOT:     memref.load
// CHECK:         cf.br ^bb{{[0-9]+}}(%c0{{.*}}, %{{.+}} : index, vector<1xf32>)
// CHECK:         vector.mask %{{.+}} { vector.reduction <maximumf>, %{{.+}}, %{{.+}} : vector<[8]xf32> into f32 } : vector<[8]xi1> -> f32
// CHECK:         math.exp %{{.+}} : vector<[4]xf32>
// CHECK:         cf.br ^bb{{[0-9]+}}(%c0{{.*}}, %{{.+}} : index, vector<1xf32>)
// CHECK:         vector.mask %{{.+}} { vector.reduction <add>, %{{.+}}, %{{.+}} : vector<[8]xf32> into f32 } : vector<[8]xi1> -> f32
// CHECK:         arith.divf %{{.+}}, %{{.+}} : vector<[4]xf32>
// CHECK-NOT:     memref.load
// CHECK:         return
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>
#map2 = affine_map<(d0, d1) -> (d0, 0)>
module {
  func.func @Softmax(%arg0: tensor<256x781xf32>) -> tensor<256x781xf32> {
    %c0_i64 = arith.constant 0 : i64
    %cst = arith.constant 0xFF800000 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<256xi64>
    %1 = linalg.fill ins(%c0_i64 : i64) outs(%0 : tensor<256xi64>) -> tensor<256xi64>
    %2 = tensor.empty() : tensor<256xf32>
    %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<256xf32>) -> tensor<256xf32>
    %4:2 = linalg.generic {indexing_maps = [#map, #map1, #map1], iterator_types = ["parallel", "reduction"]} ins(%arg0 : tensor<256x781xf32>) outs(%3, %1 : tensor<256xf32>, tensor<256xi64>) {
    ^bb0(%in: f32, %out: f32, %out_1: i64):
      %12 = linalg.index 1 : index
      %13 = arith.index_cast %12 : index to i64
      %14 = arith.maximumf %in, %out : f32
      %15 = arith.cmpf ogt, %in, %out : f32
      %16 = arith.select %15, %13, %out_1 : i64
      linalg.yield %14, %16 : f32, i64
    } -> (tensor<256xf32>, tensor<256xi64>)
    %expanded = tensor.expand_shape %4#0 [[0, 1]] output_shape [256, 1] : tensor<256xf32> into tensor<256x1xf32>
    %5 = tensor.empty() : tensor<256x781xf32>
    %6 = linalg.generic {indexing_maps = [#map, #map2, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %expanded : tensor<256x781xf32>, tensor<256x1xf32>) outs(%5 : tensor<256x781xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %12 = arith.subf %in, %in_1 : f32
      linalg.yield %12 : f32
    } -> tensor<256x781xf32>
    %7 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%6 : tensor<256x781xf32>) outs(%5 : tensor<256x781xf32>) {
    ^bb0(%in: f32, %out: f32):
      %12 = math.exp %in : f32
      linalg.yield %12 : f32
    } -> tensor<256x781xf32>
    %8 = tensor.empty() : tensor<256x1xf32>
    %9 = linalg.fill ins(%cst_0 : f32) outs(%8 : tensor<256x1xf32>) -> tensor<256x1xf32>
    %10 = linalg.generic {indexing_maps = [#map, #map2], iterator_types = ["parallel", "reduction"]} ins(%7 : tensor<256x781xf32>) outs(%9 : tensor<256x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %12 = arith.addf %in, %out : f32
      linalg.yield %12 : f32
    } -> tensor<256x1xf32>
    %11 = linalg.generic {indexing_maps = [#map, #map2, #map], iterator_types = ["parallel", "parallel"]} ins(%7, %10 : tensor<256x781xf32>, tensor<256x1xf32>) outs(%5 : tensor<256x781xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %12 = arith.divf %in, %in_1 : f32
      linalg.yield %12 : f32
    } -> tensor<256x781xf32>
    return %11 : tensor<256x781xf32>
  }
}

// -----

// CHECK-LABEL: func.func @softmax_kernel(
// CHECK-NOT:     memref.copy
// CHECK:         vector.transfer_read {{.+}} : memref<?xf32, strided<[1], offset: ?>>, vector<[4]xf32>
// CHECK:         cf.br ^bb{{[0-9]+}}(%c0{{.*}}, %{{.+}} : index, vector<1xf32>)
// CHECK:         vector.mask %{{.+}} { vector.reduction <maxnumf>, %{{.+}}, %{{.+}} : vector<[8]xf32> into f32 } : vector<[8]xi1> -> f32
// CHECK:         cf.br ^bb{{[0-9]+}}(%c0{{.*}}, %{{.+}} : index, vector<1xf32>)
// CHECK:         vector.mask %{{.+}} { vector.reduction <add>, %{{.+}}, %{{.+}} : vector<[8]xf32> into f32 } : vector<[8]xi1> -> f32
// CHECK-NOT:     memref.copy
// CHECK:         return
#map = affine_map<(d0) -> (d0)>
module {
  func.func @softmax_kernel(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32) {
    %c1024 = arith.constant 1024 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0xFF800000 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %0 = arith.muli %arg8, %arg2 : i32
    %1 = arith.index_cast %0 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%1], sizes: [1024], strides: [1] : memref<*xf32> to memref<1024xf32, strided<[1], offset: ?>>
    %2 = arith.index_cast %arg4 : i32 to index
    %3 = arith.minsi %2, %c1024 : index
    %4 = arith.maxsi %3, %c0 : index
    %alloc = memref.alloc() : memref<1024xf32>
    %5 = arith.cmpi slt, %4, %c1024 : index
    scf.if %5 {
      linalg.fill ins(%cst : f32) outs(%alloc : memref<1024xf32>)
    }
    %subview = memref.subview %reinterpret_cast[0] [%4] [1] : memref<1024xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
    %subview_1 = memref.subview %alloc[0] [%4] [1] : memref<1024xf32> to memref<?xf32, strided<[1]>>
    memref.copy %subview, %subview_1 : memref<?xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1]>>
    %6 = bufferization.to_tensor %alloc restrict writable : memref<1024xf32> to tensor<1024xf32>
    %7 = bufferization.alloc_tensor() : tensor<f32>
    %inserted = tensor.insert %cst into %7[] : tensor<f32>
    %reduced = linalg.reduce ins(%6 : tensor<1024xf32>) outs(%inserted : tensor<f32>) dimensions = [0] 
      (%in: f32, %init: f32) {
        %17 = arith.maxnumf %in, %init : f32
        linalg.yield %17 : f32
      }
    %extracted = tensor.extract %reduced[] : tensor<f32>
    %8 = tensor.empty() : tensor<1024xf32>
    %9 = linalg.fill ins(%extracted : f32) outs(%8 : tensor<1024xf32>) -> tensor<1024xf32>
    %10 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%6, %9 : tensor<1024xf32>, tensor<1024xf32>) outs(%6 : tensor<1024xf32>) {
    ^bb0(%in: f32, %in_7: f32, %out: f32):
      %17 = arith.subf %in, %in_7 : f32
      linalg.yield %17 : f32
    } -> tensor<1024xf32>
    %11 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%10 : tensor<1024xf32>) outs(%10 : tensor<1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %17 = math.exp %in : f32
      linalg.yield %17 : f32
    } -> tensor<1024xf32>
    %12 = bufferization.alloc_tensor() : tensor<f32>
    %inserted_2 = tensor.insert %cst_0 into %12[] : tensor<f32>
    %reduced_3 = linalg.reduce ins(%11 : tensor<1024xf32>) outs(%inserted_2 : tensor<f32>) dimensions = [0] 
      (%in: f32, %init: f32) {
        %17 = arith.addf %in, %init : f32
        linalg.yield %17 : f32
      }
    %extracted_4 = tensor.extract %reduced_3[] : tensor<f32>
    %13 = linalg.fill ins(%extracted_4 : f32) outs(%8 : tensor<1024xf32>) -> tensor<1024xf32>
    %14 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%11, %13 : tensor<1024xf32>, tensor<1024xf32>) outs(%11 : tensor<1024xf32>) {
    ^bb0(%in: f32, %in_7: f32, %out: f32):
      %17 = arith.divf %in, %in_7 : f32
      linalg.yield %17 : f32
    } -> tensor<1024xf32>
    %15 = arith.muli %arg8, %arg3 : i32
    %16 = arith.index_cast %15 : i32 to index
    %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%16], sizes: [1024], strides: [1] : memref<*xf32> to memref<1024xf32, strided<[1], offset: ?>>
    %extracted_slice = tensor.extract_slice %14[0] [%4] [1] : tensor<1024xf32> to tensor<?xf32>
    %subview_6 = memref.subview %reinterpret_cast_5[0] [%4] [1] : memref<1024xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
    bufferization.materialize_in_destination %extracted_slice in writable %subview_6 : (tensor<?xf32>, memref<?xf32, strided<[1], offset: ?>>) -> ()
    return
  }
}
