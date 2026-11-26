// RUN: arm64-mlir-opt %s -split-input-file --vector-ext=sve -optimize-triton-shared | FileCheck %s
// RUN: arm64-mlir-opt %s -split-input-file --vector-ext=sme -optimize-triton-shared | FileCheck %s

// A masked load in a loop, as triton-shared emits it for a Triton k-loop: a
// static buffer is allocated, filled with the padding value when the mask is
// partial, and the valid part is copied in. The buffer is allocated once before
// the loop and freed once after it, not in every iteration. With SME the loop
// is lowered to cf, so it is matched as either scf.for or cf.cond_br.

// CHECK-LABEL: func.func @masked_load_in_loop
// CHECK-NOT:     {{scf.for|cf.cond_br}}
// CHECK:         %[[BUF:.+]] = memref.alloc() : memref<16xf32>
// CHECK:         {{scf.for|cf.cond_br}}
// CHECK-NOT:     memref.alloc() : memref<16xf32>
// CHECK:         memref.dealloc %[[BUF]]
// CHECK-NEXT:    return

#map = affine_map<(d0) -> (d0)>
func.func @masked_load_in_loop(%in: memref<?xf32>, %out: memref<?xf32>, %n: index) {
  %c0 = arith.constant 0 : index
  %c16 = arith.constant 16 : index
  %zero = arith.constant 0.0 : f32
  scf.for %k = %c0 to %n step %c16 {
    %left = arith.subi %n, %k : index
    %len = arith.minsi %left, %c16 : index
    %buf = memref.alloc() : memref<16xf32>
    %partial = arith.cmpi slt, %len, %c16 : index
    scf.if %partial {
      linalg.fill ins(%zero : f32) outs(%buf : memref<16xf32>)
    }
    %src = memref.subview %in[%k] [%len] [1] : memref<?xf32> to memref<?xf32, strided<[1], offset: ?>>
    %dst = memref.subview %buf[0] [%len] [1] : memref<16xf32> to memref<?xf32, strided<[1]>>
    memref.copy %src, %dst : memref<?xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1]>>
    %t = bufferization.to_tensor %buf restrict writable : memref<16xf32> to tensor<16xf32>
    %e = tensor.empty() : tensor<16xf32>
    %neg = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%t : tensor<16xf32>) outs(%e : tensor<16xf32>) {
    ^bb0(%a: f32, %o: f32):
      %v = arith.negf %a : f32
      linalg.yield %v : f32
    } -> tensor<16xf32>
    %o = memref.subview %out[%k] [16] [1] : memref<?xf32> to memref<16xf32, strided<[1], offset: ?>>
    bufferization.materialize_in_destination %neg in writable %o : (tensor<16xf32>, memref<16xf32, strided<[1], offset: ?>>) -> ()
  }
  return
}
