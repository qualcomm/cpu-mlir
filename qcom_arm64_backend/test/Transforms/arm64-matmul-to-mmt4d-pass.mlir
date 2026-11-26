// RUN: arm64-mlir-opt %s -split-input-file -triton-arm64-matmul-to-mmt4d | FileCheck %s

// CHECK-LABEL: func.func @test_matmul
// CHECK:       linalg.mmt4d
func.func @test_matmul(%A: tensor<?x?xf32>,
                       %B: tensor<?x?xf32>,
                       %C: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = linalg.matmul
    ins(%A, %B: tensor<?x?xf32>, tensor<?x?xf32>)
    outs(%C: tensor<?x?xf32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// -----

// A matmul that accumulates into a loop-carried tensor (a Triton k-loop): the
// accumulator is packed once before the loop and unpacked once after it, and
// the loop carries the packed tensor. Only A and B are packed per iteration.
// CHECK-LABEL: func.func @hoist_packed_accumulator
// CHECK:         %[[ACC0:.+]] = linalg.transpose ins(%{{.+}} : tensor<1x32x2x32xf32>) outs(%{{.+}} : tensor<1x2x32x32xf32>)
// CHECK:         %[[R:.+]] = scf.for {{.+}} iter_args(%[[ACC:.+]] = %[[ACC0]]) -> (tensor<1x2x32x32xf32>)
// CHECK-NOT:       tensor<32x64xf32>
// CHECK:           %[[M:.+]] = linalg.mmt4d {{.+}} outs(%[[ACC]] : tensor<1x2x32x32xf32>)
// CHECK:           scf.yield %[[M]]
// CHECK:         linalg.transpose ins(%[[R]] : tensor<1x2x32x32xf32>)
// CHECK:         tensor.collapse_shape {{.+}} into tensor<32x64xf32>
func.func @hoist_packed_accumulator(%a: tensor<8x32x16xf32>, %b: tensor<8x16x64xf32>) -> tensor<32x64xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<32x64xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<32x64xf32>) -> tensor<32x64xf32>
  %r = scf.for %k = %c0 to %c8 step %c1 iter_args(%acc = %init) -> (tensor<32x64xf32>) {
    %ak = tensor.extract_slice %a[%k, 0, 0] [1, 32, 16] [1, 1, 1] : tensor<8x32x16xf32> to tensor<32x16xf32>
    %bk = tensor.extract_slice %b[%k, 0, 0] [1, 16, 64] [1, 1, 1] : tensor<8x16x64xf32> to tensor<16x64xf32>
    %m = linalg.matmul ins(%ak, %bk : tensor<32x16xf32>, tensor<16x64xf32>) outs(%acc : tensor<32x64xf32>) -> tensor<32x64xf32>
    scf.yield %m : tensor<32x64xf32>
  }
  return %r : tensor<32x64xf32>
}

// -----

// The accumulator has another use in the loop, which needs it unpacked:
// nothing is hoisted.
// CHECK-LABEL: func.func @no_hoist_other_use
// CHECK:         scf.for {{.+}} -> (tensor<32x64xf32>, tensor<32x64xf32>)
// CHECK:           linalg.mmt4d
func.func @no_hoist_other_use(%a: tensor<32x16xf32>, %b: tensor<16x64xf32>, %init: tensor<32x64xf32>) -> (tensor<32x64xf32>, tensor<32x64xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %r:2 = scf.for %k = %c0 to %c8 step %c1 iter_args(%acc = %init, %prev = %init) -> (tensor<32x64xf32>, tensor<32x64xf32>) {
    %m = linalg.matmul ins(%a, %b : tensor<32x16xf32>, tensor<16x64xf32>) outs(%acc : tensor<32x64xf32>) -> tensor<32x64xf32>
    scf.yield %m, %acc : tensor<32x64xf32>, tensor<32x64xf32>
  }
  return %r#0, %r#1 : tensor<32x64xf32>, tensor<32x64xf32>
}
