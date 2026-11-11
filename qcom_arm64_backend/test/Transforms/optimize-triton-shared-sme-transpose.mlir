// RUN: arm64-mlir-opt %s -split-input-file --vector-ext=sme -optimize-triton-shared | FileCheck %s

// On SME, a transpose of the two innermost dimensions with at least 2^19
// elements goes through a ZA tile: each input row is written into the tile as
// a vertical slice (MOVA) and the tile is stored back horizontally. Edges that
// are not a multiple of the tile size are masked. Smaller transposes keep the
// regular path. The ZA loop nest is outlined into a private function, so that
// only that function runs in streaming mode.

// CHECK-LABEL: func.func @transpose_2d_f32(
// CHECK-NOT:     llvm.arm_streaming
// CHECK:         call @transpose_2d_f32_sme_0(
// CHECK-LABEL: func.func private @transpose_2d_f32_sme_0(
// CHECK-SAME:    llvm.arm_new_za
// CHECK-SAME:    llvm.arm_streaming
// CHECK:         "arm_sme.intr.write.vert"({{.+}}) <{tile_id = 0 : i32}> : (i32, vector<[4]xi1>, vector<[4]xf32>) -> ()
// CHECK:         "arm_sme.intr.st1w.horiz"({{.+}}) <{tile_id = 0 : i32}> : (vector<[4]xi1>, !llvm.ptr, i32) -> ()
func.func @transpose_2d_f32(%arg0: memref<1031x1029xf32>, %arg1: memref<1029x1031xf32>) {
  %x = bufferization.to_tensor %arg0 restrict : memref<1031x1029xf32> to tensor<1031x1029xf32>
  %e = tensor.empty() : tensor<1029x1031xf32>
  %t = linalg.transpose ins(%x : tensor<1031x1029xf32>) outs(%e : tensor<1029x1031xf32>) permutation = [1, 0]
  bufferization.materialize_in_destination %t in writable %arg1 : (tensor<1029x1031xf32>, memref<1029x1031xf32>) -> ()
  return
}

// -----

// CHECK-LABEL: func.func private @transpose_2d_f16_sme_0(
// CHECK:         "arm_sme.intr.write.vert"({{.+}}) <{tile_id = 0 : i32}> : (i32, vector<[8]xi1>, vector<[8]xf16>) -> ()
// CHECK:         "arm_sme.intr.st1h.horiz"({{.+}}) <{tile_id = 0 : i32}> : (vector<[8]xi1>, !llvm.ptr, i32) -> ()
func.func @transpose_2d_f16(%arg0: memref<1025x770xf16>, %arg1: memref<770x1025xf16>) {
  %x = bufferization.to_tensor %arg0 restrict : memref<1025x770xf16> to tensor<1025x770xf16>
  %e = tensor.empty() : tensor<770x1025xf16>
  %t = linalg.transpose ins(%x : tensor<1025x770xf16>) outs(%e : tensor<770x1025xf16>) permutation = [1, 0]
  bufferization.materialize_in_destination %t in writable %arg1 : (tensor<770x1025xf16>, memref<770x1025xf16>) -> ()
  return
}

// -----

// The attention K transpose (B x H x S x D -> B x H x D x S).
// CHECK-LABEL: func.func private @transpose_attention_k_sme_0(
// CHECK:         "arm_sme.intr.write.vert"
// CHECK:         "arm_sme.intr.st1w.horiz"
func.func @transpose_attention_k(%arg0: memref<1x12x1024x64xf32>, %arg1: memref<1x12x64x1024xf32>) {
  %x = bufferization.to_tensor %arg0 restrict : memref<1x12x1024x64xf32> to tensor<1x12x1024x64xf32>
  %e = tensor.empty() : tensor<1x12x64x1024xf32>
  %t = linalg.transpose ins(%x : tensor<1x12x1024x64xf32>) outs(%e : tensor<1x12x64x1024xf32>) permutation = [0, 1, 3, 2]
  bufferization.materialize_in_destination %t in writable %arg1 : (tensor<1x12x64x1024xf32>, memref<1x12x64x1024xf32>) -> ()
  return
}

// -----

// A small transpose: the per-call cost of streaming mode outweighs the ZA tile.
// CHECK-LABEL: func.func @transpose_2d_small
// CHECK-NOT:     llvm.arm_streaming
// CHECK-NOT:     arm_sme
// CHECK:         return
func.func @transpose_2d_small(%arg0: memref<129x77xf32>, %arg1: memref<77x129xf32>) {
  %x = bufferization.to_tensor %arg0 restrict : memref<129x77xf32> to tensor<129x77xf32>
  %e = tensor.empty() : tensor<77x129xf32>
  %t = linalg.transpose ins(%x : tensor<129x77xf32>) outs(%e : tensor<77x129xf32>) permutation = [1, 0]
  bufferization.materialize_in_destination %t in writable %arg1 : (tensor<77x129xf32>, memref<77x129xf32>) -> ()
  return
}

// -----

// Splitting attention heads keeps the innermost dimension: a row copy, no ZA.
// CHECK-LABEL: func.func @transpose_heads
// CHECK-NOT:     llvm.arm_streaming
// CHECK-NOT:     arm_sme
// CHECK:         return
func.func @transpose_heads(%arg0: memref<1x7x12x64xf16>, %arg1: memref<1x12x7x64xf16>) {
  %x = bufferization.to_tensor %arg0 restrict : memref<1x7x12x64xf16> to tensor<1x7x12x64xf16>
  %e = tensor.empty() : tensor<1x12x7x64xf16>
  %t = linalg.transpose ins(%x : tensor<1x7x12x64xf16>) outs(%e : tensor<1x12x7x64xf16>) permutation = [0, 2, 1, 3]
  bufferization.materialize_in_destination %t in writable %arg1 : (tensor<1x12x7x64xf16>, memref<1x12x7x64xf16>) -> ()
  return
}

// -----

// A transpose followed by other work, like a whole torch-mlir model: only the
// outlined ZA loop nest is streaming. The caller keeps its SVE code and must
// not become streaming, where NEON is illegal without FEAT_SME_FA64. Only the
// buffers are passed in: the tile size (vscale) is recomputed inside, where it
// follows the streaming vector length, which can differ from the caller's.
// CHECK-LABEL: func.func @transpose_then_add(
// CHECK-NOT:     llvm.arm_streaming
// CHECK-NOT:     arm_sme
// CHECK:         call @transpose_then_add_sme_0(%arg0, %{{.+}}) : (memref<1031x1029xf32>, memref<1029x1031xf32>) -> ()
// CHECK-NOT:     arm_sme
// CHECK:         arith.addf {{.+}} : vector<[4]xf32>
// CHECK:         return
// CHECK-LABEL: func.func private @transpose_then_add_sme_0(
// CHECK-SAME:    llvm.arm_new_za
// CHECK-SAME:    llvm.arm_streaming
// CHECK-SAME:    llvm.linkage = #llvm.linkage<internal>
// CHECK:         vector.vscale
// CHECK:         "arm_sme.intr.write.vert"
// CHECK:         "arm_sme.intr.st1w.horiz"
func.func @transpose_then_add(%arg0: memref<1031x1029xf32>, %arg1: memref<1029x1031xf32>) {
  %x = bufferization.to_tensor %arg0 restrict : memref<1031x1029xf32> to tensor<1031x1029xf32>
  %e = tensor.empty() : tensor<1029x1031xf32>
  %t = linalg.transpose ins(%x : tensor<1031x1029xf32>) outs(%e : tensor<1029x1031xf32>) permutation = [1, 0]
  %e2 = tensor.empty() : tensor<1029x1031xf32>
  %s = linalg.add ins(%t, %t : tensor<1029x1031xf32>, tensor<1029x1031xf32>) outs(%e2 : tensor<1029x1031xf32>) -> tensor<1029x1031xf32>
  bufferization.materialize_in_destination %s in writable %arg1 : (tensor<1029x1031xf32>, memref<1029x1031xf32>) -> ()
  return
}
