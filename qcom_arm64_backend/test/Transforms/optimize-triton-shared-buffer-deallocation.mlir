// RUN: arm64-mlir-opt %s -split-input-file --vector-ext=sve -optimize-triton-shared | FileCheck %s
// RUN: arm64-mlir-opt %s -split-input-file --vector-ext=sme -optimize-triton-shared | FileCheck %s

// A temporary used by two consumers: elementwise fusion keeps it, so it is
// bufferized into a buffer that must be freed before returning, along with the
// two results. With SME the loops are lowered to cf, so deallocation has to run
// before that.

// CHECK-LABEL: func.func @two_uses
// CHECK-COUNT-3: memref.alloc
// CHECK-COUNT-3: memref.dealloc
// CHECK-NOT:     memref.alloc
// CHECK:         return

#map = affine_map<(d0) -> (d0)>
func.func @two_uses(%arg0: memref<1024xf32>, %arg1: memref<1024xf32>, %arg2: memref<1024xf32>) {
  %x = bufferization.to_tensor %arg0 restrict : memref<1024xf32> to tensor<1024xf32>
  %e0 = tensor.empty() : tensor<1024xf32>
  %e = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%x : tensor<1024xf32>) outs(%e0 : tensor<1024xf32>) {
  ^bb0(%in: f32, %out: f32):
    %0 = math.exp %in : f32
    linalg.yield %0 : f32
  } -> tensor<1024xf32>
  %a0 = tensor.empty() : tensor<1024xf32>
  %a = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%e : tensor<1024xf32>) outs(%a0 : tensor<1024xf32>) {
  ^bb0(%in: f32, %out: f32):
    %0 = arith.negf %in : f32
    linalg.yield %0 : f32
  } -> tensor<1024xf32>
  %b0 = tensor.empty() : tensor<1024xf32>
  %b = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%e : tensor<1024xf32>) outs(%b0 : tensor<1024xf32>) {
  ^bb0(%in: f32, %out: f32):
    %0 = arith.mulf %in, %in : f32
    linalg.yield %0 : f32
  } -> tensor<1024xf32>
  bufferization.materialize_in_destination %a in writable %arg1 : (tensor<1024xf32>, memref<1024xf32>) -> ()
  bufferization.materialize_in_destination %b in writable %arg2 : (tensor<1024xf32>, memref<1024xf32>) -> ()
  return
}

// -----

// A buffer returned from a loop: whether to free it is decided at runtime by a
// generated helper, which must be inlined into the kernel. After the loop both
// %init and the loop result %r must be freed; the checks follow %r from its use
// in the copy to %out to its own dealloc.

// CHECK-LABEL: func.func @loop_buffer
// CHECK-NOT:     dealloc_helper
// CHECK:         %[[INIT:.+]] = memref.alloc() : memref<16xf32>
// CHECK:         memref.copy %[[R:[^,]+]], %arg1 :
// CHECK:         %[[R_BASE:[^,]+]], {{.*}} = memref.extract_strided_metadata %[[R]] :
// CHECK-DAG:     memref.dealloc %[[INIT]] :
// CHECK-DAG:     memref.dealloc %[[R_BASE]] :
// CHECK-NOT:     dealloc_helper
func.func @loop_buffer(%n: index, %out: memref<16xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %init = memref.alloc() : memref<16xf32>
  %r = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %init) -> (memref<16xf32>) {
    %new = memref.alloc() : memref<16xf32>
    memref.copy %acc, %new : memref<16xf32> to memref<16xf32>
    scf.yield %new : memref<16xf32>
  }
  memref.copy %r, %out : memref<16xf32> to memref<16xf32>
  return
}
