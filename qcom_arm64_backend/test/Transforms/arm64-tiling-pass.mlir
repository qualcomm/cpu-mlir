// RUN: arm64-mlir-opt %s -triton-arm64-linalg-tiling="vector-ext=sve" | FileCheck %s
// RUN: arm64-mlir-opt %s -triton-arm64-linalg-tiling="vector-ext=sme" | FileCheck %s --check-prefix=CHECK-SME

// CHECK-LABEL: func.func @test_elementwise_add
// CHECK: [[VSCALE:%.+]] = vector.vscale
// CHECK: [[CONST4:%.+]] = arith.constant 4 : index
// CHECK: [[STEP_S:%.+]] = arith.muli [[CONST4]], [[VSCALE]]
// CHECK: scf.for {{.+}} step [[STEP_S]]
// Check that SME flat is ignored for non-matmul ops
// CHECK-SME-LABEL: func.func @test_elementwise_add
// CHECK-SME: [[VSCALE:%.+]] = vector.vscale
// CHECK-SME: [[CONST4:%.+]] = arith.constant 4 : index
// CHECK-SME: [[STEP_S:%.+]] = arith.muli [[CONST4]], [[VSCALE]]
// CHECK-SME-NOT:   arith.constant 2
// CHECK-SME: scf.for {{.+}} step [[STEP_S]]

func.func @test_elementwise_add(%arg0: tensor<128xf32>,
                                 %arg1: tensor<128xf32>,
                                 %arg2: tensor<128xf32>) -> tensor<128xf32> {
  %0 = linalg.map { arith.addf }
    ins(%arg0, %arg1 : tensor<128xf32>, tensor<128xf32>)
    outs(%arg2 : tensor<128xf32>)
  return %0 : tensor<128xf32>
}

// Reductions get two SVE registers' worth per iteration: 8 x vscale.
// CHECK-LABEL: func.func @test_reduce
// CHECK: [[VSCALE:%.+]] = vector.vscale
// CHECK: [[CONST8:%.+]] = arith.constant 8 : index
// CHECK: [[STEP_S:%.+]] = arith.muli [[CONST8]], [[VSCALE]]
// CHECK: scf.for {{.+}} step [[STEP_S]]
func.func @test_reduce(%arg0: tensor<128xf32>,
                        %arg1: tensor<f32>) -> tensor<f32> {
  %0 = linalg.reduce { arith.addf }
    ins(%arg0 : tensor<128xf32>)
    outs(%arg1 : tensor<f32>)
    dimensions = [0]
  return %0 : tensor<f32>
}

// CHECK-LABEL: func.func @test_matvec
// CHECK: [[VSCALE:%.+]] = vector.vscale
// CHECK: [[CONST4:%.+]] = arith.constant 4 : index
// CHECK: [[STEP_S:%.+]] = arith.muli [[CONST4]], [[VSCALE]]
// CHECK: scf.for {{.+}} step [[STEP_S]]
// CHECK-SME-LABEL: func.func @test_matvec
// CHECK-SME: [[VSCALE:%.+]] = vector.vscale
// CHECK-SME: [[CONST4:%.+]] = arith.constant 4 : index
// CHECK-SME: [[STEP_S:%.+]] = arith.muli [[CONST4]], [[VSCALE]]
// CHECK-SME: scf.for {{.+}} step [[STEP_S]]
// CHECK-SME-NOT: arith.constant 2
// CHECK-SME:     return
func.func @test_matvec(%arg0: tensor<64x128xf32>,
                        %arg1: tensor<128xf32>,
                        %arg2: tensor<64xf32>) -> tensor<64xf32> {
  %0 = linalg.matvec
    ins(%arg0, %arg1 : tensor<64x128xf32>, tensor<128xf32>)
    outs(%arg2 : tensor<64xf32>) -> tensor<64xf32>
  return %0 : tensor<64xf32>
}

// CHECK-LABEL: func.func @test_matmul
// CHECK: [[VSCALE:%.+]] = vector.vscale
// CHECK: [[CONST4:%.+]] = arith.constant 4
// CHECK: [[STEP_S:%.+]] = arith.muli [[CONST4]], [[VSCALE]]
// CHECK: scf.for {{.+}} step [[STEP_S]]
// CHECK-SME-LABEL: func.func @test_matmul
// CHECK-SME: [[VSCALE:%.+]] = vector.vscale
// CHECK-SME: [[C8_D0:%.+]] = arith.constant 8
// CHECK-SME: [[STEP_D0:%.+]] = arith.muli [[C8_D0]], [[VSCALE]]
// CHECK-SME: [[C8_D1:%.+]] = arith.constant 8
// CHECK-SME: [[STEP_D1:%.+]] = arith.muli [[C8_D1]], [[VSCALE]]
// CHECK-SME: [[CONST2:%.+]] = arith.constant 2
// CHECK-SME-NOT: arith.muli
// CHECK-SME: scf.for {{.+}} step [[STEP_D0]]
// CHECK-SME: scf.for {{.+}} step [[STEP_D1]]
// CHECK-SME: scf.for {{.+}} step [[CONST2]]
func.func @test_matmul(%arg0: tensor<64x128xf32>,
                        %arg1: tensor<128x64xf32>,
                        %arg2: tensor<64x64xf32>) -> tensor<64x64xf32> {
  %0 = linalg.matmul
    ins(%arg0, %arg1 : tensor<64x128xf32>, tensor<128x64xf32>)
    outs(%arg2 : tensor<64x64xf32>) -> tensor<64x64xf32>
  return %0 : tensor<64x64xf32>
}

// CHECK-LABEL: func.func @test_no_tiling_memref
// CHECK-NOT: scf.for
func.func @test_no_tiling_memref(%arg0: memref<128xf32>,
                                  %arg1: memref<128xf32>,
                                  %arg2: memref<128xf32>) {
  linalg.map { arith.addf }
    ins(%arg0, %arg1 : memref<128xf32>, memref<128xf32>)
    outs(%arg2 : memref<128xf32>)
  return
}

// CHECK-LABEL: func.func @test_elementwise_2d
// CHECK: [[VSCALE:%.+]] = vector.vscale
// CHECK: [[STEP_1:%.+]] = arith.constant 1
// CHECK: [[CONST4:%.+]] = arith.constant 4 : index
// CHECK: [[STEP_S:%.+]] = arith.muli [[CONST4]], [[VSCALE]]
// CHECK: scf.for {{.+}} step [[STEP_1]]
// CHECK: scf.for {{.+}} step [[STEP_S]]
func.func @test_elementwise_2d(%arg0: tensor<32x128xf32>,
                                %arg1: tensor<32x128xf32>,
                                %arg2: tensor<32x128xf32>) -> tensor<32x128xf32> {
  %0 = linalg.map { arith.addf }
    ins(%arg0, %arg1 : tensor<32x128xf32>, tensor<32x128xf32>)
    outs(%arg2 : tensor<32x128xf32>)
  return %0 : tensor<32x128xf32>
}

// A linalg.generic whose innermost loop is its reduction (torch-mlir's form
// of a row sum or max) is tiled like linalg.reduce: rows by 1, the reduction
// by 8 x vscale.
// CHECK-LABEL: func.func @test_generic_innermost_reduction
// CHECK:       [[VSCALE:%.+]] = vector.vscale
// CHECK:       [[CONST8:%.+]] = arith.constant 8 : index
// CHECK:       [[STEP_S:%.+]] = arith.muli [[CONST8]], [[VSCALE]]
// CHECK:       scf.for {{.+}} to %c64{{.*}} step %c1
// CHECK:         scf.for {{.+}} to %c64{{.*}} step [[STEP_S]]
// CHECK:           linalg.generic {{.+}} iterator_types = ["parallel", "reduction"]{{.+}} ins({{.+}} : tensor<1x?xf32>)
func.func @test_generic_innermost_reduction(%arg0: tensor<64x64xf32>,
                                     %arg1: tensor<64xf32>) -> tensor<64xf32> {
  %0 = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0)>
    ],
    iterator_types = ["parallel", "reduction"]
  }
  ins(%arg0 : tensor<64x64xf32>)
  outs(%arg1 : tensor<64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %sum = arith.addf %in, %out : f32
      linalg.yield %sum : f32
  } -> tensor<64xf32>
  return %0 : tensor<64xf32>
}

// CHECK-LABEL: func.func @test_mixed_tileable_and_memref
// CHECK: [[VSCALE:%.+]] = vector.vscale
// CHECK: [[CONST4:%.+]] = arith.constant 4 : index
// CHECK: [[STEP_S:%.+]] = arith.muli [[CONST4]], [[VSCALE]]
// CHECK: scf.for {{.+}} step [[STEP_S]]
func.func @test_mixed_tileable_and_memref(%arg0: tensor<128xf32>,
                                           %arg1: tensor<128xf32>,
                                           %arg2: tensor<128xf32>,
                                           %arg3: memref<128xf32>,
                                           %arg4: memref<128xf32>,
                                           %arg5: memref<128xf32>) -> tensor<128xf32> {
  %0 = linalg.map { arith.addf }
    ins(%arg0, %arg1 : tensor<128xf32>, tensor<128xf32>)
    outs(%arg2 : tensor<128xf32>)
  linalg.map { arith.addf }
    ins(%arg3, %arg4 : memref<128xf32>, memref<128xf32>)
    outs(%arg5 : memref<128xf32>)
  return %0 : tensor<128xf32>
}

// CHECK-LABEL: func.func @test_no_tiling_zero_loops
// CHECK:       linalg.generic
// CHECK-NOT:   scf.for
func.func @test_no_tiling_zero_loops(%arg0: tensor<f32>,
                                      %arg1: tensor<f32>) -> tensor<f32> {
  %0 = linalg.generic {
    indexing_maps = [
      affine_map<() -> ()>,
      affine_map<() -> ()>
    ],
    iterator_types = []
  }
  ins(%arg0 : tensor<f32>)
  outs(%arg1 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %sum = arith.addf %in, %out : f32
      linalg.yield %sum : f32
  } -> tensor<f32>
  return %0 : tensor<f32>
}

// CHECK-LABEL: func.func @test_conv1d_outer_product_tiling
// CHECK:       scf.for {{.+}} step {{.+}}4
// CHECK:       scf.for {{.+}} step {{.+}}4
// CHECK:       scf.for {{.+}} step {{.+}}1
// CHECK:       scf.for {{.+}} step {{.+}}1
// CHECK:       linalg.conv_1d_ncw_fcw
func.func @test_conv1d_outer_product_tiling(
    %input: tensor<1x3x6xf32>, %filter: tensor<4x3x3xf32>,
    %output: tensor<1x4x4xf32>) -> tensor<1x4x4xf32> {
  %0 = linalg.conv_1d_ncw_fcw
    ins(%input, %filter : tensor<1x3x6xf32>, tensor<4x3x3xf32>)
    outs(%output : tensor<1x4x4xf32>) -> tensor<1x4x4xf32>
  return %0 : tensor<1x4x4xf32>
}

// CHECK-LABEL: func.func @test_conv2d_decomposes_to_conv1d
// CHECK:       scf.for
// CHECK:       scf.for
// CHECK-NOT:   linalg.conv_2d_nchw_fchw
// CHECK:       linalg.conv_1d_ncw_fcw
// CHECK:       return
func.func @test_conv2d_decomposes_to_conv1d(
    %input: tensor<1x3x5x6xf32>,
    %filter: tensor<4x3x3x3xf32>,
    %output: tensor<1x4x3x4xf32>) -> tensor<1x4x3x4xf32> {
  %0 = linalg.conv_2d_nchw_fchw
    ins(%input, %filter : tensor<1x3x5x6xf32>, tensor<4x3x3x3xf32>)
    outs(%output : tensor<1x4x3x4xf32>) -> tensor<1x4x3x4xf32>
  return %0 : tensor<1x4x3x4xf32>
}

// A transpose of the two innermost dimensions with at least 2^19 elements is
// tiled to one [4]x[4] SME tile on SME, and the tile is marked for the
// vectorizer; on SVE only the innermost dimension is scalable.
// CHECK-LABEL:     func.func @test_transpose_inner_dims
// CHECK:           linalg.transpose ins({{.+}} : tensor<1x1x?x1xf32>) outs({{.+}} : tensor<1x1x1x?xf32>)
// CHECK-NOT:       arm64.sme_transpose_tile
// CHECK-SME-LABEL: func.func @test_transpose_inner_dims
// CHECK-SME:       [[VSCALE:%.+]] = vector.vscale
// CHECK-SME:       [[TILE:%.+]] = arith.muli %c4{{.*}}, [[VSCALE]] : index
// CHECK-SME:       scf.for {{.+}} to %c64{{.*}} step [[TILE]]
// CHECK-SME:       scf.for {{.+}} to %c1024{{.*}} step [[TILE]]
// CHECK-SME:       linalg.transpose ins({{.+}} : tensor<1x1x?x?xf32>) outs({{.+}} : tensor<1x1x?x?xf32>) permutation = [0, 1, 3, 2] {arm64.sme_transpose_tile = 4 : i64}
func.func @test_transpose_inner_dims(%arg0: tensor<1x12x1024x64xf32>, %arg1: tensor<1x12x64x1024xf32>) -> tensor<1x12x64x1024xf32> {
  %0 = linalg.transpose ins(%arg0 : tensor<1x12x1024x64xf32>) outs(%arg1 : tensor<1x12x64x1024xf32>) permutation = [0, 1, 3, 2]
  return %0 : tensor<1x12x64x1024xf32>
}

// A small transpose keeps the regular path on SME: the per-call cost of
// streaming mode outweighs the ZA tile transpose.
// CHECK-SME-LABEL: func.func @test_transpose_inner_dims_small
// CHECK-SME:       linalg.transpose ins({{.+}} : tensor<1x1x?x1xf32>) outs({{.+}} : tensor<1x1x1x?xf32>)
// CHECK-SME-NOT:   arm64.sme_transpose_tile
func.func @test_transpose_inner_dims_small(%arg0: tensor<1x12x7x64xf32>, %arg1: tensor<1x12x64x7xf32>) -> tensor<1x12x64x7xf32> {
  %0 = linalg.transpose ins(%arg0 : tensor<1x12x7x64xf32>) outs(%arg1 : tensor<1x12x64x7xf32>) permutation = [0, 1, 3, 2]
  return %0 : tensor<1x12x64x7xf32>
}

// A dynamic shape counts with its proven upper bound: a Triton block of
// 1024x512 whose edges are masked (sizes min(.., 1024) x min(.., 512)) takes
// the ZA tile path.
// CHECK-SME-LABEL: func.func @test_transpose_bounded_dynamic
// CHECK-SME:       linalg.transpose {{.+}} permutation = [1, 0] {arm64.sme_transpose_tile = 4 : i64}
func.func @test_transpose_bounded_dynamic(%arg0: tensor<1024x512xf32>, %m: index, %n: index) -> tensor<?x?xf32> {
  %c1024 = arith.constant 1024 : index
  %c512 = arith.constant 512 : index
  %rows = arith.minsi %m, %c1024 : index
  %cols = arith.minsi %n, %c512 : index
  %slice = tensor.extract_slice %arg0[0, 0] [%rows, %cols] [1, 1] : tensor<1024x512xf32> to tensor<?x?xf32>
  %init = tensor.empty(%cols, %rows) : tensor<?x?xf32>
  %0 = linalg.transpose ins(%slice : tensor<?x?xf32>) outs(%init : tensor<?x?xf32>) permutation = [1, 0]
  return %0 : tensor<?x?xf32>
}

// Without a known bound the size is unknown: regular path.
// CHECK-SME-LABEL: func.func @test_transpose_unbounded_dynamic
// CHECK-SME-NOT:   arm64.sme_transpose_tile
// CHECK-SME:       return
func.func @test_transpose_unbounded_dynamic(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = linalg.transpose ins(%arg0 : tensor<?x?xf32>) outs(%arg1 : tensor<?x?xf32>) permutation = [1, 0]
  return %0 : tensor<?x?xf32>
}

// A transpose that keeps the innermost dimension is not an SME tile transpose,
// even when it is large enough (786K elements): it only moves whole rows, so
// it keeps the regular path with only the innermost dimension scalable.
// CHECK-SME-LABEL: func.func @test_transpose_outer_dims
// CHECK-SME-NOT:   arm64.sme_transpose_tile
// CHECK-SME:       linalg.transpose ins({{.+}} : tensor<1x1x1x?xf16>) outs({{.+}} : tensor<1x1x1x?xf16>) permutation = [0, 2, 1, 3]
// CHECK-SME-NOT:   arm64.sme_transpose_tile
// CHECK-SME:       return
func.func @test_transpose_outer_dims(%arg0: tensor<1x1024x12x64xf16>, %arg1: tensor<1x12x1024x64xf16>) -> tensor<1x12x1024x64xf16> {
  %0 = linalg.transpose ins(%arg0 : tensor<1x1024x12x64xf16>) outs(%arg1 : tensor<1x12x1024x64xf16>) permutation = [0, 2, 1, 3]
  return %0 : tensor<1x12x1024x64xf16>
}

// A copy between dynamically shaped, strided buffers (triton-shared's masked
// tl.load) would be lowered to a call to the element-by-element memrefCopy
// runtime function: it becomes a linalg.copy tiled to 4 x vscale, which the
// vectorizer turns into masked vector loads and stores.
// CHECK-SME-LABEL: func.func @test_strided_copy
// CHECK-SME-NOT:   memref.copy
// CHECK-SME:       [[VSCALE:%.+]] = vector.vscale
// CHECK-SME:       [[STEP:%.+]] = arith.muli %c4{{.*}}, [[VSCALE]] : index
// CHECK-SME:       scf.for {{.+}} step [[STEP]]
// CHECK-SME:         linalg.copy ins({{.+}} : memref<?xf32, strided<[1], offset: ?>>) outs({{.+}} : memref<?xf32, strided<[1], offset: ?>>)
func.func @test_strided_copy(%src: memref<1024xf32, strided<[1], offset: ?>>, %n: index) {
  %buf = memref.alloc() : memref<1024xf32>
  %s = memref.subview %src[0] [%n] [1] : memref<1024xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
  %d = memref.subview %buf[0] [%n] [1] : memref<1024xf32> to memref<?xf32, strided<[1]>>
  memref.copy %s, %d : memref<?xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1]>>
  memref.dealloc %buf : memref<1024xf32>
  return
}

// A copy that lowers to a plain memcpy (identity layout) is left alone.
// CHECK-SME-LABEL: func.func @test_contiguous_copy
// CHECK-SME:       memref.copy %{{.+}}, %{{.+}} : memref<1024xf32> to memref<1024xf32>
// CHECK-SME-NOT:   linalg.copy
func.func @test_contiguous_copy(%src: memref<1024xf32>, %dst: memref<1024xf32>) {
  memref.copy %src, %dst : memref<1024xf32> to memref<1024xf32>
  return
}
